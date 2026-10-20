from typing import Optional
from blessed import Terminal
import ctypes
import functools
import logging
import platform
import re
import sys
import os
import subprocess
import getpass
import json
import socket
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import deque
from queue import Empty, Queue
import threading

import paramiko
from paramiko import AuthenticationException
from paramiko.ssh_exception import PasswordRequiredException
import pandas as pd
from termcolor import colored as _term_colored
from . import config as nvidb_config
from .data_modules import ServerInfo, ServerListInfo
from .dcgm import make_dcgm_snapshot_command
from .metrics import (
    ADVANCED_METRIC_GROUPS,
    HIDDEN_COLUMN_PREFIX,
    present_columns,
    visible_columns,
)
from .mouse import (
    MouseSequenceParser,
    set_mouse_reporting,
)
from .nvml import PynvmlCollector, make_nvml_agent_command
from .process_view import ProcessPanel
from .tui_controls import ActionMenu, HoverState, action_bar
from .ssh_proxy import open_proxyjump_socket
from .tui_theme import (
    ANSI_THEMES,
    DiffScreen,
    display_width,
    fit,
    fit_display,
    frame_bottom,
    frame_separator,
    frame_top,
    meter,
    pad_display,
    wrap_display,
)
from .utils import (
    extract_numbers,
    extract_value_and_unit,
    format_bandwidth,
    format_pcie_link,
    get_pcie_load_color,
    get_utilization_color,
    parse_link_number,
    pcie_link_capacity_kib_per_second,
)


_monitor_theme = "classic"


def _set_monitor_theme(theme):
    global _monitor_theme
    _monitor_theme = theme if theme in ANSI_THEMES else "classic"


def colored(text, color=None, on_color=None, attrs=None):
    """Apply the active monitor palette through termcolor."""
    mapped = ANSI_THEMES[_monitor_theme].get(color, color)
    return _term_colored(text, mapped, on_color, attrs)


def _as_gpu_index(value):
    """Read a GPU index out of a table cell, or None when it is not one."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_leading_float(value):
    """Return the first numeric component of a text field, or None."""
    numbers = extract_numbers(str(value))
    return float(numbers[0]) if numbers else None


def _terminal_size(columns=80, lines=24):
    """The terminal's size, or the given fallback when stdout is no TTY."""
    try:
        return os.get_terminal_size()
    except OSError:
        return os.terminal_size((columns, lines))


def _clip(text, width, ellipsis=".."):
    """Fit text to `width` columns, marking a cut only where the mark fits."""
    return fit_display(text, width, ellipsis=ellipsis if width > display_width(ellipsis) else "")


def _xml_text(element, path, default="N/A"):
    """Text of `path` under an `nvidia-smi -q -x` element, or `default`."""
    if element is None:
        return default
    found = element.find(path)
    return found.text if found is not None else default


def _format_rounded(value, unit, divisor=1):
    """An NVML reading as a rounded integer with its unit, or "N/A"."""
    try:
        return f"{int(round(float(value) / divisor))} {unit}"
    except (TypeError, ValueError):
        return "N/A"


def _format_milliwatts(value):
    try:
        watts = float(value) / 1000.0
    except (TypeError, ValueError):
        return "N/A"
    return f"{watts:.2f} W"


def _dcgm_percent(value):
    """A DCGM profiling reading as 0-100, or None when it is not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    # Profiling fields may be 0-1 ratios; normalize to 0-100%.
    if 0.0 <= number <= 1.0:
        number *= 100.0
    return min(max(number, 0.0), 100.0)


def _format_dcgm_percent(value) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    if not text or text in {"N/A", "-"}:
        return "N/A"
    number = _dcgm_percent(text)
    # Text that is not a number is shown as it came.
    return text if number is None else f"{number:.0f}%"


def _dcgm_metric_row(gpu):
    """One GPU's DCGM row, plus the two columns derived from its readings."""
    pipes = [
        _dcgm_percent(gpu.get(field))
        for field in ("tensor_active", "fp64_active", "fp32_active", "fp16_active", "int_active")
    ]
    pipes = [value for value in pipes if value is not None]
    dram = _dcgm_percent(gpu.get("dram_active"))
    # Speed of light: the busiest of the pipes and DRAM.
    bounds = pipes + ([dram] if dram is not None else [])
    instruction_throughput = max(pipes) if pipes else None
    sol = max(bounds) if bounds else None
    # The derived values are already 0-100, so they skip the ratio scaling.
    return {
        "GPU": gpu["gpu_index"],
        "sm_active": _format_dcgm_percent(gpu.get("sm_active")),
        "sm_occupancy": _format_dcgm_percent(gpu.get("sm_occupancy")),
        "instruction_throughput": (
            "N/A" if instruction_throughput is None else f"{instruction_throughput:.0f}%"
        ),
        "sol": "N/A" if sol is None else f"{sol:.0f}%",
        "tensor_active": _format_dcgm_percent(gpu.get("tensor_active")),
        "fp64_active": _format_dcgm_percent(gpu.get("fp64_active")),
        "fp32_active": _format_dcgm_percent(gpu.get("fp32_active")),
        "fp16_active": _format_dcgm_percent(gpu.get("fp16_active")),
    }


def _last_json_object(text):
    """The last line of `text` that parses as a JSON object, or None.

    Scanning from the end skips whatever the remote shell printed first.
    """
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def _parse_meminfo_kb(text):
    """`/proc/meminfo` lines as {field: kB}."""
    mem_info = {}
    for line in text.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            # Extract numeric value (in kB)
            value = value.strip().split()[0]
            try:
                mem_info[key.strip()] = int(value)
            except ValueError:
                pass
    return mem_info


def _vm_stat_available_bytes(text):
    """Free + inactive + speculative memory from macOS `vm_stat`, in bytes."""
    page_size = 4096  # Default page size
    pages = {"Pages free:": 0, "Pages inactive:": 0, "Pages speculative:": 0}
    for line in text.split("\n"):
        if "page size of" in line:
            try:
                page_size = int(line.split()[-2])
            except ValueError:
                pass
            continue
        for label in pages:
            if label in line:
                try:
                    pages[label] = int(line.split()[-1].rstrip("."))
                except ValueError:
                    pass
                break
    return sum(pages.values()) * page_size


def _parse_swapusage_gb(text):
    """`sysctl -n vm.swapusage` ("total = 2048.00M  used = ...") in GB."""
    keys = {"total": "swap_total_gb", "used": "swap_used_gb"}
    swap = {}
    parts = text.split()
    for i, part in enumerate(parts):
        if part in keys and i + 2 < len(parts):
            try:
                swap[keys[part]] = float(parts[i + 2].rstrip("M")) / 1024
            except ValueError:
                pass
    return swap


@functools.lru_cache(maxsize=1)
def _mach_host():
    # One host port for the process: every mach_host_self() call adds a
    # reference to it that would otherwise never be released.
    libc = ctypes.CDLL(None)
    libc.mach_host_self.restype = ctypes.c_uint
    return libc, libc.mach_host_self()


def _darwin_cpu_ticks():
    """(busy, total) CPU ticks since boot, from Mach's HOST_CPU_LOAD_INFO."""
    libc, host = _mach_host()
    ticks = (ctypes.c_uint * 4)()  # user, system, idle, nice
    count = ctypes.c_uint(len(ticks))  # HOST_CPU_LOAD_INFO_COUNT
    if libc.host_statistics(host, 3, ticks, ctypes.byref(count)) != 0:  # HOST_CPU_LOAD_INFO
        raise OSError("host_statistics(HOST_CPU_LOAD_INFO) failed")
    user, system, idle, nice = ticks
    busy = user + system + nice
    return busy, busy + idle


class _ConnectProgress:
    """A single self-rewriting line reporting the startup connect sweep.

    Disabled unless stdout is a terminal, so piped and redirected runs
    keep the plain log output a script would parse. While it is enabled
    it owns its line: anything that needs to prompt calls
    `pause_connect_progress()` first so the question is not drawn over.
    """

    _BAR_WIDTH = 18

    def __init__(self, total, stream=None):
        self.total = max(0, int(total))
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = bool(self.total) and bool(
            getattr(self.stream, "isatty", lambda: False)()
        )
        self.done = 0
        self.failed = 0
        self._painted = False

    def _write(self, text):
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:
            # A closed or broken stdout must not take the whole run down.
            self.enabled = False

    def clear(self):
        """Wipe the meter's line, leaving the cursor at its start."""
        if self.enabled and self._painted:
            self._write("\r\x1b[K")
            self._painted = False

    def show(self, label):
        """Redraw the meter while `label` is being connected."""
        if not self.enabled:
            return
        bar = meter(self.done, self.total, self._BAR_WIDTH)
        line = (
            colored("  nvidb ", "green", attrs=["bold"])
            + colored(bar, "cyan")
            + colored(f"  {self.done}/{self.total}  ", "dark_grey")
            + fit(str(label), 40)
        )
        self._write("\r\x1b[K" + line)
        self._painted = True

    def mark(self, ok):
        self.done += 1
        if not ok:
            self.failed += 1

    def finish(self):
        """Clear the line: the UI is about to paint over this row."""
        self.clear()


# Set while connect_all is drawing, so an interactive prompt raised from
# deep inside a client can take the line back before asking.
_active_connect_progress = None


def pause_connect_progress():
    """Wipe the startup meter so a prompt owns the line it was using."""
    if _active_connect_progress is not None:
        _active_connect_progress.clear()


@functools.lru_cache(maxsize=1)
def local_platform_summary() -> str:
    """One-line description of the machine nvidb itself runs on."""
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        release = platform.mac_ver()[0]
        label = f"macOS {release}".strip()
    elif system == "Linux":
        label = f"Linux {platform.release()}"
    elif system == "Windows":
        label = f"Windows {platform.release()}"
    else:
        label = system or "Unknown OS"
    return " ".join(part for part in (label, machine) if part)


def _prompt_secret(prompt):
    """Ask for a password or passphrase without the meter overwriting it."""
    pause_connect_progress()
    return getpass.getpass(prompt=prompt)


class BaseClient(ABC):
    """Base class for both RemoteClient and LocalClient with common functionality"""
    
    def __init__(self):
        self.host = None
        self.port = None
        self.username = None
        self.description = None
        # GPU indices nvidb was handed on this host; None means all of them.
        self.gpu_ids = None
        self.connected = False
        self.last_connect_error = None
        self.last_error_type = None
        # A host that answered "no DCGM here" stays that way, and core count
        # never changes; remembering both saves one remote round trip each.
        self._dcgm_probe_failed = False
        self._cpu_cores = None
        # (busy, total) CPU ticks of the previous sample, for a per-interval CPU%.
        self._cpu_ticks = None

    def _set_connect_error(self, message: str, error_type: str = "error"):
        self.connected = False
        self.last_connect_error = message
        self.last_error_type = error_type

    def _set_connected(self):
        self.connected = True
        self.last_connect_error = None
        self.last_error_type = None

    def ensure_connected(self) -> bool:
        """Return whether this client can serve a query right now.

        Transports that can break mid-session (SSH) override this to revive
        themselves; a local client is either usable or was never usable.
        """
        return self.connected is not False

    @abstractmethod
    def connect(self, *, allow_prompt: bool = True, announce: bool = True) -> bool:
        """Connect to the client (remote or local).

        `announce` is cleared by callers that own the terminal line - the
        startup progress meter, the background reconnect - so nothing is
        printed over what they are drawing.
        """
        pass
    
    @abstractmethod
    def execute_command(self, command: str) -> str:
        """Execute a command and return the output"""
        pass

    def _chunked(self, items, chunk_size: int):
        for i in range(0, len(items), chunk_size):
            yield items[i : i + chunk_size]

    def get_pid_user_map(self, pids, chunk_size: int = 128):
        """Batch query pid -> username mapping via a single ps call (or a few chunked calls)."""
        details = self._query_pid_details(list(dict.fromkeys(pids)), ("pid", "user"), chunk_size)
        return {pid: info["username"] for pid, info in details.items()}

    # `ps` keyword sets tried in order: Linux (with thread count), BSD/macOS,
    # then the minimal set every ps implementation supports. `args` stays last
    # because it is the only field that may contain spaces.
    PROCESS_DETAIL_FIELD_SETS = (
        ("pid", "user", "pcpu", "pmem", "rss", "etime", "stat", "nlwp", "args"),
        ("pid", "user", "pcpu", "pmem", "rss", "etime", "stat", "args"),
        ("pid", "user", "args"),
    )

    def _query_pid_details(self, pids, fields, chunk_size: int):
        selector = ",".join(f"{field}=" for field in fields)
        details = {}
        for pid_chunk in self._chunked(pids, chunk_size):
            pid_list = ",".join(pid_chunk)
            output = self.execute_command(f"ps -o {selector} -p {pid_list}")
            for line in output.splitlines():
                # strip() first: the maxsplit tail keeps trailing whitespace.
                parts = line.strip().split(None, len(fields) - 1)
                if len(parts) < len(fields):
                    continue
                values = dict(zip(fields, parts))
                details[values["pid"]] = {
                    "username": values.get("user"),
                    "cpu_percent": parse_leading_float(values.get("pcpu")),
                    "mem_percent": parse_leading_float(values.get("pmem")),
                    "rss_kb": parse_leading_float(values.get("rss")),
                    "elapsed": values.get("etime"),
                    "state": values.get("stat"),
                    "threads": parse_leading_float(values.get("nlwp")),
                    "command": values.get("args"),
                }
        return details

    def get_pid_process_info(self, pids, chunk_size: int = 128):
        """Batch query htop-style details (CPU, RSS, state, full command) per pid."""
        unique_pids = list(dict.fromkeys(pids))
        for fields in self.PROCESS_DETAIL_FIELD_SETS:
            details = self._query_pid_details(unique_pids, fields, chunk_size)
            if details:
                return details
        return {}

    def _connection_error_info(self):
        return {
            "error": self.last_connect_error or "Not connected",
            "error_type": self.last_error_type or "error",
        }

    def get_full_gpu_info(self, *, include_advanced=True):
        """Get GPU information (base + optional advanced profiling).

        - Base table: NVML, with `nvidia-smi -q -x` used only when NVML is unavailable.
        - Advanced table (DCGM profiling): fetched via DCGM python bindings when available.
        """
        # A dropped link is revived here rather than reported as "no GPUs":
        # every refresh tick gets one (rate-limited) chance to reconnect.
        if not self.ensure_connected():
            return pd.DataFrame(), self._connection_error_info()

        stats, system_info, nvml_error = self._nvml_base_info()
        if nvml_error is not None:
            stats, system_info = self._nvidia_smi_base_info()
            system_info["fallback_reason"] = nvml_error
        if stats.empty:
            # The link can die mid-fetch. Say so, instead of letting one tick
            # look like a machine that never had an NVIDIA GPU.
            if self.connected is False:
                return pd.DataFrame(), self._connection_error_info()
            return stats, system_info

        if not include_advanced:
            return stats, system_info
        return self._add_advanced_gpu_info(stats, system_info)

    def _nvml_base_info(self):
        """The base table from NVML, or (None, None, why NVML failed)."""
        try:
            payload = self.query_nvml_snapshot()
        except Exception as error:
            logging.debug(msg=f"Failed to query NVML snapshot: {error}")
            return None, None, f"{type(error).__name__}: {error}"
        if payload.get("ok") is not True:
            return None, None, str(payload.get("error", "NVML snapshot failed"))

        rows = []
        for gpu in payload["gpus"]:
            processes = [
                {
                    "gpu_instance_id": process.get("gpu_instance_id"),
                    "compute_instance_id": process.get("compute_instance_id"),
                    "pid": process.get("pid"),
                    "type": process.get("type") or "N/A",
                    "process_name": process.get("process_name") or "N/A",
                    "used_memory": _format_rounded(
                        process.get("used_gpu_memory_bytes"), "MiB", 1024 * 1024
                    ),
                    "username": process.get("username"),
                }
                for process in gpu.get("processes") or []
            ]
            try:
                power_state = f"P{int(gpu.get('performance_state'))}"
            except (TypeError, ValueError):
                power_state = "N/A"

            rows.append(
                {
                    "gpu_index": gpu["gpu_index"],
                    "product_name": gpu.get("name") or "Unknown GPU",
                    "product_architecture": gpu.get("architecture") or "N/A",
                    "tx_util": _format_rounded(gpu.get("pcie_tx_kib_per_s"), "KB/s"),
                    "rx_util": _format_rounded(gpu.get("pcie_rx_kib_per_s"), "KB/s"),
                    "fan_speed": _format_rounded(gpu.get("fan_percent"), "%"),
                    "total": _format_rounded(gpu.get("memory_total_bytes"), "MiB", 1024 * 1024),
                    "used": _format_rounded(gpu.get("memory_used_bytes"), "MiB", 1024 * 1024),
                    "free": _format_rounded(gpu.get("memory_free_bytes"), "MiB", 1024 * 1024),
                    "gpu_util": _format_rounded(gpu.get("gpu_util_percent"), "%"),
                    "memory_util": _format_rounded(gpu.get("memory_util_percent"), "%"),
                    "gpu_temp": _format_rounded(gpu.get("temperature_c"), "C"),
                    "power_state": power_state,
                    "power_draw": _format_milliwatts(gpu.get("power_usage_mw")),
                    "current_power_limit": _format_milliwatts(gpu.get("power_limit_mw")),
                    "pcie_link_gen_current": gpu.get("pcie_link_gen_current"),
                    "pcie_link_width_current": gpu.get("pcie_link_width_current"),
                    "pcie_link_gen_max": gpu.get("pcie_link_gen_max"),
                    "pcie_link_width_max": gpu.get("pcie_link_width_max"),
                    "processes": processes,
                }
            )

        backend = payload.get("backend") or "unknown"
        system_info = {
            "driver_version": payload.get("driver_version") or "N/A",
            "cuda_version": payload.get("cuda_version") or "N/A",
            "attached_gpus": str(len(rows)),
            "data_source": "nvml",
            "data_source_detail": "nvidia-ml-py" if backend == "pynvml" else "libnvidia-ml.so.1",
        }
        return pd.DataFrame(rows), system_info, None

    def _nvidia_smi_base_info(self):
        """The base table from `nvidia-smi -q -x`, the fallback when NVML fails."""
        try:
            result = self.execute_command("nvidia-smi -q -x")
            if not result.lstrip().startswith("<?xml"):
                return pd.DataFrame(), {"data_source": "unsupported"}

            root = ET.fromstring(result)
            rows = []
            for gpu_index, gpu in enumerate(root.findall("gpu")):
                pci = gpu.find("pci")
                # nvidia-smi spells widths as "16x"; parse_link_number copes.
                link_info = pci.find("pci_gpu_link_info") if pci is not None else None
                fb_memory_usage = gpu.find("fb_memory_usage")
                utilization = gpu.find("utilization")
                gpu_power_readings = gpu.find("gpu_power_readings")
                # On newer drivers (e.g. WSL/591.x) the P-state under
                # <gpu_power_readings><power_state> is deprecated and returns
                # "Requested functionality has been deprecated"; the real perf
                # state now lives in <gpu><performance_state>.
                power_state = _xml_text(gpu_power_readings, "power_state")
                if not power_state or power_state == "N/A" or "deprecated" in power_state.lower():
                    power_state = _xml_text(gpu, "performance_state")

                power_draw = _xml_text(gpu_power_readings, "power_draw", None)
                if power_draw is None or power_draw == "N/A":
                    power_draw = _xml_text(gpu_power_readings, "instant_power_draw")

                # Same shape as the NVML process entries.
                processes_element = gpu.find("processes")
                processes = [
                    {
                        "pid": _xml_text(process_info, "pid"),
                        "process_name": _xml_text(process_info, "process_name"),
                        "used_memory": _xml_text(process_info, "used_memory", "0 MiB"),
                        "gpu_instance_id": _xml_text(process_info, "gpu_instance_id"),
                        "compute_instance_id": _xml_text(process_info, "compute_instance_id"),
                        "type": _xml_text(process_info, "type"),
                        "username": _xml_text(process_info, "username", None),
                    }
                    for process_info in (
                        processes_element.findall("process_info")
                        if processes_element is not None
                        else ()
                    )
                ]

                rows.append(
                    {
                        "gpu_index": gpu_index,
                        "product_name": _xml_text(gpu, "product_name", "Unknown GPU"),
                        "product_architecture": _xml_text(gpu, "product_architecture"),
                        "tx_util": _xml_text(pci, "tx_util"),
                        "rx_util": _xml_text(pci, "rx_util"),
                        "fan_speed": _xml_text(gpu, "fan_speed"),
                        "total": _xml_text(fb_memory_usage, "total"),
                        "used": _xml_text(fb_memory_usage, "used"),
                        "free": _xml_text(fb_memory_usage, "free"),
                        "gpu_util": _xml_text(utilization, "gpu_util"),
                        "memory_util": _xml_text(utilization, "memory_util"),
                        "gpu_temp": _xml_text(gpu.find("temperature"), "gpu_temp"),
                        "power_state": power_state,
                        "power_draw": power_draw,
                        "current_power_limit": _xml_text(gpu_power_readings, "current_power_limit"),
                        "pcie_link_gen_current": _xml_text(link_info, "pcie_gen/current_link_gen", None),
                        "pcie_link_width_current": _xml_text(
                            link_info, "link_widths/current_link_width", None
                        ),
                        "pcie_link_gen_max": _xml_text(link_info, "pcie_gen/max_link_gen", None),
                        "pcie_link_width_max": _xml_text(link_info, "link_widths/max_link_width", None),
                        "processes": processes,
                    }
                )

            system_info = {
                "driver_version": _xml_text(root, "driver_version"),
                "cuda_version": _xml_text(root, "cuda_version"),
                "attached_gpus": _xml_text(root, "attached_gpus", "0"),
                "data_source": "nvidia-smi",
                "data_source_detail": "nvidia-smi -q -x",
            }
            return pd.DataFrame(rows), system_info

        except Exception as e:
            logging.debug(msg=f"Failed to get base GPU info via nvidia-smi: {e}")
            return pd.DataFrame(), {"data_source": "unsupported"}

    def _add_advanced_gpu_info(self, stats, system_info):
        """Enrich a base snapshot after its GPU rows can already be displayed."""
        # Fetch advanced profiling metrics via DCGM (optional). A host without
        # DCGM answers the same way on every refresh, so the failure is cached
        # instead of paying the remote python startup per tick forever.
        if stats.empty or self._dcgm_probe_failed:
            return stats, system_info
        try:
            payload = _last_json_object(self.execute_command(make_dcgm_snapshot_command()))
            if payload is None or payload.get("ok") is not True:
                self._dcgm_probe_failed = True
                return stats, system_info
            rows = [_dcgm_metric_row(gpu) for gpu in payload.get("gpus") or []]
        except Exception:
            # DCGM is optional, and a dropped link is only temporary, so this
            # does not mark the host as lacking DCGM.
            return stats, system_info

        if rows:
            system_info["advanced_source"] = "dcgm"
            system_info["advanced_source_detail"] = payload.get("connection") or "N/A"
            system_info["advanced_metrics"] = pd.DataFrame(rows)
            system_info["advanced_supported"] = any(
                value not in {"N/A", "-", ""}
                for row in rows
                for column, value in row.items()
                if column != "GPU"
            )
        return stats, system_info

    def get_system_stats(self) -> dict:
        """Get system statistics: CPU cores, CPU usage, memory usage, and swap usage.

        Returns:
            dict: System statistics with keys:
                - cpu_cores: Number of CPU cores
                - cpu_percent: Total CPU utilization percentage
                - mem_used_gb: Used memory in GB
                - mem_total_gb: Total memory in GB
                - swap_used_gb: Used swap in GB
                - swap_total_gb: Total swap in GB
        """
        result = {
            "cpu_cores": 0,
            "cpu_percent": 0.0,
            "mem_used_gb": 0.0,
            "mem_total_gb": 0.0,
            "swap_used_gb": 0.0,
            "swap_total_gb": 0.0,
        }

        try:
            # Core count never changes, so ask the host once and remember.
            if self._cpu_cores is None:
                # Linux first, then macOS.
                for command in ("nproc", "sysctl -n hw.ncpu"):
                    cores = self.execute_command(command).strip()
                    if cores.isdigit():
                        self._cpu_cores = int(cores)
                        break
            if self._cpu_cores is not None:
                result["cpu_cores"] = self._cpu_cores

            cpu_percent = self._cpu_percent()
            if cpu_percent is not None:
                result["cpu_percent"] = cpu_percent

            # Get memory info - try Linux first
            mem_output = self.execute_command(
                "grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo"
            )
            if mem_output.strip():
                mem_info = _parse_meminfo_kb(mem_output)
                mem_total_kb = mem_info.get("MemTotal", 0)
                mem_available_kb = mem_info.get("MemAvailable", 0)
                swap_total_kb = mem_info.get("SwapTotal", 0)
                swap_free_kb = mem_info.get("SwapFree", 0)

                result["mem_total_gb"] = mem_total_kb / (1024 * 1024)
                result["mem_used_gb"] = (mem_total_kb - mem_available_kb) / (1024 * 1024)
                result["swap_total_gb"] = swap_total_kb / (1024 * 1024)
                result["swap_used_gb"] = (swap_total_kb - swap_free_kb) / (1024 * 1024)
            else:
                # Try macOS - use sysctl for memory
                mem_total_output = self.execute_command("sysctl -n hw.memsize")
                if mem_total_output.strip().isdigit():
                    mem_total_bytes = int(mem_total_output.strip())
                    result["mem_total_gb"] = mem_total_bytes / (1024 * 1024 * 1024)

                    vm_stat_output = self.execute_command("vm_stat")
                    if vm_stat_output:
                        available_bytes = _vm_stat_available_bytes(vm_stat_output)
                        result["mem_used_gb"] = (mem_total_bytes - available_bytes) / (1024 * 1024 * 1024)

                result.update(_parse_swapusage_gb(self.execute_command("sysctl -n vm.swapusage")))

        except Exception as e:
            logging.debug(msg=f"Failed to get system stats: {e}")

        return result

    def _cpu_percent(self):
        """Total CPU utilization in percent, or None when it cannot be read."""
        # Linux: "cpu  user nice system idle ..." ticks since boot.
        fields = self.execute_command("grep 'cpu ' /proc/stat").split()
        if fields:
            try:
                user, system, idle = int(fields[1]), int(fields[3]), int(fields[4])
            except ValueError:
                return None
            return self._cpu_percent_since_last(user + system, user + system + idle)

        # Try macOS - use top for CPU usage (user + sys)
        parts = self.execute_command(
            "top -l 1 -n 0 | grep 'CPU usage' | awk '{print $3, $5}' | tr -d '%,'"
        ).split()
        if len(parts) >= 2:
            try:
                return float(parts[0]) + float(parts[1])
            except ValueError:
                pass
        return None

    def _cpu_percent_since_last(self, busy, total):
        """Busy share of the CPU ticks counted since the previous sample.

        The counters run from boot, so the first sample - or one where the
        counters did not advance or wrapped - can only give the since-boot
        average.
        """
        previous, self._cpu_ticks = self._cpu_ticks, (busy, total)
        if previous is not None and total > previous[1]:
            busy, total = busy - previous[0], total - previous[1]
        return busy * 100 / total

    def get_process_summary(self, gpu_stats, detailed: bool = False, *, resolve_users=True):
        """Get detailed GPU processes information with user summary.

        With `detailed`, one extra batched `ps` call per host enriches every GPU
        process with htop-style fields (CPU/MEM share, RSS, elapsed time, state,
        thread count) and the full command line.
        With `resolve_users=False`, only the supplied GPU snapshot is used;
        PIDs remain visible before any additional SSH round trips complete.
        """
        # (entry, pid as text, used MiB) per GPU process.
        process_entries = []
        pids = []
        all_pids = []

        # First pass: extract processes and collect all pids
        for idx, row in gpu_stats.iterrows():
            try:
                gpu_index = int(row.get("gpu_index", idx))
            except (TypeError, ValueError):
                gpu_index = idx

            for process in row.get("processes") or []:
                pid = process.get("pid", "N/A")
                used_memory = process.get("used_memory", "0 MiB")
                username = process.get("username")
                gpu_instance_id = process.get("gpu_instance_id")
                compute_instance_id = process.get("compute_instance_id")

                memory_value = 0
                if used_memory != "N/A":
                    try:
                        memory_value = int(
                            float(str(used_memory).replace("MiB", "").strip())
                        )
                    except (TypeError, ValueError):
                        memory_value = 0

                pid_str = str(pid).strip()
                if pid_str and pid_str != "N/A":
                    all_pids.append(pid_str)
                    if not username or username == "N/A":
                        pids.append(pid_str)

                entry = {
                    "gpu_instance_id": (
                        gpu_instance_id if gpu_instance_id is not None else "N/A"
                    ),
                    "compute_instance_id": (
                        compute_instance_id if compute_instance_id is not None else "N/A"
                    ),
                    "pid": pid,
                    "type": process.get("type") or "N/A",
                    "process_name": process.get("process_name") or "N/A",
                    "used_memory": used_memory,
                    "username": username,
                    "gpu_index": gpu_index,
                }
                process_entries.append((entry, pid_str, memory_value))

        pid_details = self.get_pid_process_info(all_pids) if detailed and resolve_users else {}
        missing_user_pids = [
            pid_str for pid_str in pids if not pid_details.get(pid_str, {}).get("username")
        ]
        pid_to_user = self.get_pid_user_map(missing_user_pids) if resolve_users else {}

        all_processes = []
        user_memory_summary = {}
        for entry, pid_str, memory_value in process_entries:
            details = pid_details.get(pid_str, {})
            username = entry["username"]
            if not username or username == "N/A":
                username = details.get("username")
            if not username or username == "N/A":
                username = pid_to_user.get(pid_str, "N/A")

            all_processes.append(
                {
                    **entry,
                    "username": username,
                    "command": details.get("command") or entry["process_name"],
                    "cpu_percent": details.get("cpu_percent"),
                    "mem_percent": details.get("mem_percent"),
                    "rss_kb": details.get("rss_kb"),
                    "elapsed": details.get("elapsed"),
                    "state": details.get("state"),
                    "threads": details.get("threads"),
                }
            )

            if username != "N/A" and memory_value > 0:
                user_memory_summary[username] = user_memory_summary.get(username, 0) + memory_value

        return all_processes, user_memory_summary

    def format_user_memory_compact(self, user_summary):
        """Format user memory summary in compact format like 'qbs(23082M) gdm(4M)' sorted by memory usage desc"""
        return " ".join(
            f"{username}({total_memory}M)"
            for username, total_memory in sorted(user_summary.items(), key=lambda x: x[1], reverse=True)
        )


class RemoteClient(BaseClient):
    # Seconds to wait after the Nth failed reconnect, so an unreachable node
    # costs one handshake every 15s instead of one on every refresh tick. The
    # cap doubles as the worst-case delay before a recovered node reappears.
    RECONNECT_BACKOFF_SECONDS = (2.0, 5.0, 10.0, 15.0)
    CONNECT_TIMEOUT_SECONDS = 8.0
    COMMAND_TIMEOUT_SECONDS = 30.0
    # SSH keepalives turn a silently dropped link into an inactive transport,
    # so a dead node is detected in seconds instead of blocking on every read.
    KEEPALIVE_SECONDS = 10

    def __init__(self, server: ServerInfo):
        super().__init__()
        self.host = server.host
        self.port = server.port
        self.username = server.username
        self.auth = server.auth
        self.description = server.description
        self.password = server.password
        self.identityfile = server.identityfile
        self.proxyjump = server.proxyjump
        self.gpu_ids = server.gpu_ids
        self._proxy = None
        self.client = self._make_ssh_client()
        # Lock order is always _connect_lock -> _nvml_agent_lock; never take the
        # connect lock while holding the NVML one.
        self._connect_lock = threading.RLock()
        self._reconnect_attempts = 0
        self._reconnect_after = 0.0
        self._nvml_agent_lock = threading.RLock()
        self._nvml_agent_stdin = None
        self._nvml_agent_stdout = None
        self._nvml_agent_stderr = None
        self._nvml_agent_channel = None
        self.secret_prompt = _prompt_secret

    def _make_ssh_client(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if self.auth in ["auto", "key"]:
            client.load_system_host_keys()
        return client

    def _close_proxy(self):
        if self._proxy is not None:
            try:
                self._proxy.close()
            except Exception:
                pass
        self._proxy = None

    def _close_ssh(self):
        try:
            self.client.close()
        except Exception:
            pass
        self._close_proxy()

    def _connect_client(self, *, batch_mode: bool = False, **kwargs):
        """Connect a fresh Paramiko client, optionally through ProxyJump.

        `batch_mode` forbids the ProxyJump helper from prompting, which matters
        on the background reconnect path where nothing can answer a prompt.
        """
        self._close_ssh()
        self.client = self._make_ssh_client()

        kwargs.setdefault("timeout", self.CONNECT_TIMEOUT_SECONDS)
        kwargs.setdefault("banner_timeout", self.CONNECT_TIMEOUT_SECONDS)
        kwargs.setdefault("auth_timeout", self.CONNECT_TIMEOUT_SECONDS)

        # A native ssh jump helper cannot use our UI-owned secret prompt.
        # Never let it read /dev/tty concurrently with the live monitor.
        batch_mode = batch_mode or self.secret_prompt is not _prompt_secret
        try:
            self._proxy = open_proxyjump_socket(
                self.proxyjump,
                self.host,
                self.port,
                connect_timeout=self.CONNECT_TIMEOUT_SECONDS,
                batch_mode=batch_mode,
            )
            if self._proxy is not None:
                kwargs["sock"] = self._proxy
            self.client.connect(**kwargs)
        except Exception as error:
            # Paramiko reports a link that dropped or stalled mid-auth as an
            # AuthenticationException too. That is a network failure worth
            # retrying, not credentials the server rejected.
            link_lost = isinstance(error, AuthenticationException) and (
                not self._transport_alive() or str(error) == "Authentication timeout."
            )
            self._close_ssh()
            if link_lost:
                raise ConnectionError(f"Connection lost during authentication: {error}") from error
            raise
        transport = self.client.get_transport()
        if transport is not None:
            transport.set_keepalive(self.KEEPALIVE_SECONDS)

    def _transport_alive(self) -> bool:
        transport = self.client.get_transport()
        return bool(transport is not None and transport.is_active())

    def _close_link_locked(self):
        """Drop the SSH session (and the NVML agent riding on it)."""
        with self._nvml_agent_lock:
            self._close_nvml_agent_locked()
        self._close_ssh()

    def _mark_link_lost(self, error=None):
        """Note that the SSH session died so the next tick reconnects.

        A command can also fail while the link is healthy (a missing binary, a
        non-zero exit); that must not tear down a working connection, so the
        transport gets the final say.
        """
        with self._connect_lock:
            if self._transport_alive():
                return
            self._close_link_locked()
            reason = (
                f"{type(error).__name__}: {error}" if error is not None else "connection lost"
            )
            logging.warning(msg=f"Lost connection to {self.description}: {reason}")
            self._set_connect_error(f"Connection lost ({reason}); reconnecting", error_type="connect")
            # Retry on the very next refresh: the common case - a laptop that
            # slept, a Wi-Fi hiccup - is over before the user looks back.
            self._reconnect_attempts = 0
            self._reconnect_after = 0.0

    def ensure_connected(self) -> bool:
        """Reconnect a dropped session, at most once per backoff window."""
        with self._connect_lock:
            if self._transport_alive():
                if not self.connected:
                    self._set_connected()
                return True

            if self.connected:
                # Nothing raised yet, but the transport is gone.
                self._mark_link_lost()

            # Credentials the background thread cannot supply will not become
            # valid by retrying; leave the message on screen instead of looping.
            if self.last_error_type == "auth":
                return False

            if time.monotonic() < self._reconnect_after:
                return False
            return self._reconnect_locked()

    def _reconnect_locked(self) -> bool:
        self._close_link_locked()
        self._reconnect_attempts += 1
        try:
            reconnected = self.connect(allow_prompt=False, announce=False)
        except Exception as error:
            logging.debug(msg=f"Reconnect to {self.description} failed: {error}")
            self._set_connect_error(f"{type(error).__name__}: {error}", error_type="connect")
            reconnected = False

        if reconnected:
            logging.info(msg=f"Reconnected to {self.host}:{self.port} as {self.username}")
            self._reconnect_attempts = 0
            self._reconnect_after = 0.0
            return True

        index = min(self._reconnect_attempts - 1, len(self.RECONNECT_BACKOFF_SECONDS) - 1)
        delay = self.RECONNECT_BACKOFF_SECONDS[index]
        self._reconnect_after = time.monotonic() + delay
        reason = self.last_connect_error or "Connection failed"
        if self.last_error_type != "auth":
            suffix = f"; retrying in {int(delay)}s" if delay else "; retrying"
            self._set_connect_error(f"{reason}{suffix}", error_type=self.last_error_type or "connect")
        return False

    def __del__(self):
        try:
            self._close_link_locked()
            # Debug, not info: `__del__` runs at interpreter shutdown, so at
            # info level every node dropped a line onto the user's shell
            # prompt after the UI had already gone.
            logging.debug(msg=f"Connection to {self.host}:{self.port} closed.")
        except Exception:
            pass

    def _close_nvml_agent_locked(self):
        channel = self._nvml_agent_channel
        if channel is not None and not channel.closed:
            try:
                channel.sendall(b"quit\n")
            except Exception:
                pass

        for stream in (self._nvml_agent_stdin, self._nvml_agent_stdout, self._nvml_agent_stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        self._nvml_agent_stdin = self._nvml_agent_stdout = self._nvml_agent_stderr = None

        if channel is not None:
            try:
                channel.close()
            except Exception:
                pass
        self._nvml_agent_channel = None

    def _start_nvml_agent_locked(self):
        # exec_command's timeout is also set on the channel.
        stdin, stdout, stderr = self.client.exec_command(
            command=make_nvml_agent_command(), timeout=10.0,
        )
        self._nvml_agent_stdin = stdin
        self._nvml_agent_stdout = stdout
        self._nvml_agent_stderr = stderr
        self._nvml_agent_channel = stdout.channel

    def query_nvml_snapshot(self) -> dict:
        """Query the persistent remote NVML agent, restarting it once on failure."""
        if not self.ensure_connected():
            return {
                "ok": False,
                "backend": "ctypes",
                "error": self.last_connect_error or "Not connected",
            }

        last_error = None
        with self._nvml_agent_lock:
            for _ in range(2):
                try:
                    channel = self._nvml_agent_channel
                    if channel is None or channel.closed or channel.exit_status_ready():
                        self._close_nvml_agent_locked()
                        self._start_nvml_agent_locked()
                        channel = self._nvml_agent_channel

                    channel.sendall(b"snapshot\n")
                    line = self._nvml_agent_stdout.readline()
                    if not line:
                        raise RuntimeError("Remote NVML agent closed its output")
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("Remote NVML agent returned a non-object payload")
                    return payload
                except Exception as error:
                    last_error = error
                    self._close_nvml_agent_locked()

        # Outside the NVML lock: marking the link lost takes the connect lock,
        # and that order must never be reversed.
        self._mark_link_lost(last_error)
        return {
            "ok": False,
            "backend": "ctypes",
            "error": f"{type(last_error).__name__}: {last_error}",
        }

    def _authenticate_with_password(self, max_attempts=3, *, allow_prompt: bool = True) -> bool:
        """Attempt password authentication with retry limit.

        Args:
            max_attempts: Maximum number of password attempts (default: 3)
            allow_prompt: Allow asking the user; background reconnects cannot
                (nothing is reading the terminal) and fail instead.

        Returns:
            bool: True if authentication succeeds

        """
        for attempt in range(max_attempts):
            try:
                if self.password is not None and attempt == 0:
                    password = self.password
                elif not allow_prompt:
                    self._set_connect_error(
                        "Password required; restart nvidb to enter it",
                        error_type="auth",
                    )
                    return False
                else:
                    remaining = max_attempts - attempt
                    if attempt > 0 and self.secret_prompt is _prompt_secret:
                        logging.warning(msg=f"Authentication failed. {remaining} attempt(s) remaining.")
                        pause_connect_progress()
                        print(f"  ⚠ Authentication failed. {remaining} attempt(s) remaining.")
                    password = self.secret_prompt(f"Enter password for {self.username}@{self.host}:{self.port} (attempt {attempt + 1}/{max_attempts}) -> ")

                self._connect_client(
                    batch_mode=not allow_prompt,
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    password=password,
                    allow_agent=False,
                    look_for_keys=False,
                )
                self._set_connected()
                logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username}")
                return True

            except AuthenticationException:
                continue
            except Exception as e:
                logging.error(msg=f"Password authentication error on {self.description}: {e}")
                self._set_connect_error(str(e), error_type="connect")
                return False

        logging.error(
            msg=f"Password authentication failed after {max_attempts} attempts on {self.description}"
        )
        self._set_connect_error("Password incorrect", error_type="auth")
        return False

    def _connect_with_key(self, *, allow_prompt: bool) -> bool:
        """Key (or agent) authentication; `auth: auto` falls back to a password."""
        password_fallback = self.auth == "auto"
        identityfile = os.path.expanduser(str(self.identityfile)) if self.identityfile else None
        target = dict(
            batch_mode=not allow_prompt,
            hostname=self.host,
            port=self.port,
            username=self.username,
            key_filename=identityfile,
        )
        try:
            self._connect_client(
                allow_agent=not identityfile,
                look_for_keys=not identityfile,
                **target,
            )
        except PasswordRequiredException:
            if not identityfile:
                if password_fallback:
                    logging.error(msg=f"Key requires passphrase on {self.description}, trying password...")
                    return self._authenticate_with_password(allow_prompt=allow_prompt)
                self._set_connect_error("Key requires passphrase; set `identityfile` or use `auth: password`", error_type="auth")
                return False
            if not allow_prompt:
                self._set_connect_error(
                    "Key passphrase required; restart nvidb to enter it",
                    error_type="auth",
                )
                return False
            passphrase = self.secret_prompt(f"Enter passphrase for key {identityfile} -> ")
            # Raised from inside this handler, a wrong passphrase skips the
            # sibling handler below and fails as a connection error.
            self._connect_client(
                allow_agent=False,
                look_for_keys=False,
                passphrase=passphrase,
                **target,
            )
        except AuthenticationException as e:
            if password_fallback:
                logging.error(msg=f"Key-based authentication failed on {self.description}, trying password...")
                return self._authenticate_with_password(allow_prompt=allow_prompt)
            logging.error(msg=f"Key-based authentication failed on {self.description}: {e}")
            self._set_connect_error("Key authentication failed", error_type="auth")
            return False
        self._set_connected()
        suffix = "" if password_fallback else " (key auth)"
        logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username}{suffix}")
        return True

    def connect(self, *, allow_prompt: bool = True, announce: bool = True) -> bool:
        with self._connect_lock:
            connected = self._connect_once(allow_prompt=allow_prompt, announce=announce)
            if connected:
                self._reconnect_attempts = 0
                self._reconnect_after = 0.0
            elif self._reconnect_attempts == 0:
                # Initial failure must also enter backoff: the first sample
                # must not repeat the handshake that just timed out.
                self._reconnect_attempts = 1
                self._reconnect_after = time.monotonic() + self.RECONNECT_BACKOFF_SECONDS[0]
            return connected

    def _connect_once(self, *, allow_prompt: bool = True, announce: bool = True) -> bool:
        """Open the SSH session.

        `allow_prompt` and `announce` are cleared by the background reconnect,
        which runs under a live TUI: nothing can answer a prompt and any print
        would land in the middle of the rendered screen.
        """
        if announce:
            # A log line, not a print: the startup meter draws its own
            # status, and a bare print would land in the middle of it.
            logging.info(
                msg=f"Connecting to {self.host}:{self.port} as {self.username}"
            )
        # catch the OSError exception when the host is not reachable
        try:
            if self.auth in ("auto", "key"):
                return self._connect_with_key(allow_prompt=allow_prompt)
            if self.auth == "password":
                # Password mode: try configured password first, then prompt if needed.
                return self._authenticate_with_password(allow_prompt=allow_prompt)
            logging.error(msg=f"Unsupported authentication method: {self.auth}, please use 'auto', 'key', or 'password'.")
            self._set_connect_error(f"Unsupported auth method: {self.auth}", error_type="error")
            return False
        except Exception as e:
            logging.error(msg=f"Connection failed: {e}")
            self._set_connect_error(str(e), error_type="connect")
            return False

    def execute_command(self, command: str) -> str:
        """Execute command on remote server.

        Raises when the session is down so callers surface a real error instead
        of an empty reading, and so a dead node costs one bounded reconnect
        attempt per tick rather than a timeout per command.
        """
        if not self.ensure_connected():
            raise ConnectionError(self.last_connect_error or "Not connected")
        try:
            stdin, stdout, stderr = self.client.exec_command(
                command=command,
                timeout=self.COMMAND_TIMEOUT_SECONDS,
            )
            result = stdout.read().decode()
        except Exception as error:
            self._mark_link_lost(error)
            raise
        return result



class LocalClient(BaseClient):
    def __init__(self):
        super().__init__()
        self.host = "localhost"
        self.port = "local"
        self.username = getpass.getuser()
        self.description = f"Local Machine ({self.username}@{self.host})"
        self._nvml_collector = PynvmlCollector()

    def __del__(self):
        try:
            self._nvml_collector.close()
        except Exception:
            pass

    def query_nvml_snapshot(self) -> dict:
        return self._nvml_collector.collect()

    def _cpu_percent(self):
        if sys.platform != "darwin":
            return super()._cpu_percent()
        # `top -l 1` costs about two seconds of CPU per sample on macOS; the
        # kernel tick counters it reports from are one Mach call away.
        return self._cpu_percent_since_last(*_darwin_cpu_ticks())

    def get_system_stats(self) -> dict:
        stats = super().get_system_stats()
        # The load average feeds the local header strip only, so it is read
        # straight from the OS here instead of adding one more command to
        # the generic path every remote host pays for.
        try:
            stats["load_avg"] = os.getloadavg()
        except (AttributeError, OSError):
            pass
        return stats

    def connect(self, *, allow_prompt: bool = True, announce: bool = True) -> bool:
        """Local connection is always successful"""
        logging.info(msg=f"Connected to local machine as {self.username}")
        self._set_connected()
        return True
    
    def execute_command(self, command: str) -> str:
        """Execute local command"""
        try:
            # Use shell=True to support pipes and complex commands
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False, timeout=10.0)
            if result.returncode != 0:
                # Only log at debug level for expected failures on systems without NVIDIA GPUs
                logging.debug(msg=f"Command '{command}' execution failed with return code {result.returncode}")
                return ""
            return result.stdout
        except Exception as e:
            logging.debug(msg=f"Unexpected error executing command: {e}")
            return ""


class NVClientPool:
    DISPLAY_MODE_NODES = "nodes"
    DISPLAY_MODE_UNIFIED = "unified"
    # "│ " + " │": the outer panel border's own side padding, reserved out of
    # the content width so nothing sits flush against the frame.
    _PANEL_BORDER_MARGIN = 4
    # A content line equal to this exact string becomes a full-width tee
    # separator ("├───┤") when the panel border is drawn, instead of being
    # padded and sided like ordinary text.
    _PANEL_SEPARATOR = "\x00__nvidb_panel_separator__\x00"
    # Click-target kinds whose whole row lights up under the pointer.
    _HOVER_ROW_KINDS = {"gpu", "server", "process", "command", "close_help"}
    UNIFIED_TABLE_COLUMNS = (
        "Node",
        "Hostname",
        "GPU",
        "name",
        "util",
        "memory[used/total]",
        "temp",
        "fan",
        "power",
        "link",
        "rx",
        "tx",
        "processes",
    )
    UNIFIED_TABLE_LABELS = {
        "sel": "",
        "Hostname": "Hostname/IP",
        "name": "Model",
        "util": "Util",
        "memory[used/total]": "VRAM U/T MiB",
        "temp": "Temp",
        "fan": "Fan",
        "power": "Power",
        "link": "PCIe",
        "rx": "RX",
        "tx": "TX",
        "processes": "Processes",
    }
    UNIFIED_SORT_MODES = ("node", "available", "utilization")
    UNIFIED_SORT_LABELS = {
        "node": "Node order",
        "available": "Available GPUs first",
        "utilization": "Highest utilization first",
    }
    UNIFIED_FILTER_MODES = ("all", "available", "busy", "errors")
    UNIFIED_FILTER_LABELS = {
        "all": "All",
        "available": "Available",
        "busy": "Busy",
        "errors": "Errors",
    }
    UNIFIED_PROCESS_SORT_MODES = (
        "vram",
        "cpu",
        "mem",
        "rss",
        "time",
        "pid",
        "command",
    )
    UNIFIED_PROCESS_SORT_LABELS = {
        "vram": "VRAM",
        "cpu": "CPU",
        "mem": "MEM",
        "rss": "RSS",
        "time": "TIME",
        "pid": "PID",
        "command": "COMMAND",
    }

    def __init__(
        self,
        server_list: ServerListInfo,
        *,
        compact: bool = False,
        view_settings=None,
        defer_connect: bool = False,
        debug: bool = False,
        dcgm: bool = False,
    ):
        """Build the client pool.

        `view_settings` restores the layout persisted in config.yml; passing it
        also enables writing later view changes back to that file.
        `defer_connect` lets the interactive monitor paint its first frame
        before the per-node workers open their SSH sessions.
        """
        self.pool = [LocalClient()]
        if server_list is not None:
            self.pool += [RemoteClient(server) for server in server_list]
        # Debug, not info: the connect meter counts the same nodes a moment
        # later, so at info level this only put a bare log line on screen
        # ahead of it.
        logging.debug(msg=f"Initialized pool with {len(self.pool)} clients.")
        if not defer_connect:
            self.connect_all()
        self.term = Terminal()
        self.compact = bool(compact)
        # Diagnostic chrome is opt-in per invocation, never a persisted view.
        self.debug = bool(debug)
        # Interactive profiling is explicit; --debug controls diagnostics only.
        # Synchronous logger/web/--once snapshots keep their existing policy.
        self.dcgm = bool(dcgm)
        self.quit_flag = threading.Event()  # Exit flag for inter-thread communication
        # Collapsible display state - every server starts expanded so the
        # dashboard opens showing the whole cluster; the stacked tables
        # then scale to whatever rows the terminal has (see
        # _scale_server_blocks).
        self.expanded_servers = set(range(len(self.pool)))
        self.selected_server = 0  # Currently selected server for navigation
        # First server section shown by the nodes-view viewport when the
        # list is taller than the terminal (see _nodes_viewport).
        self._nodes_view_start = 0
        self._persist_view_enabled = view_settings is not None
        self._default_expansion_applied = False
        self._expansion_touched = False
        settings = nvidb_config.normalize_view_settings(view_settings)
        self.display_mode = settings["mode"]
        self.unified_detailed = settings["detailed"]
        self.unified_sort_mode = settings["sort"]
        self.unified_filter_mode = settings["filter"]
        self.unified_selected_gpu = 0
        self.unified_selected_gpu_key = None
        self.unified_show_processes = settings["processes"]
        self.unified_show_trends = settings["trends"]
        self.unified_group_by_node = settings["group_by_node"]
        self.hide_unsupported = settings["hide_unsupported"]
        self.mouse_enabled = settings["mouse"]
        self.theme = settings["theme"]
        _set_monitor_theme(self.theme)
        # Detailed unified mode has two selectable panes.  The GPU cards stay
        # on screen while the lower pane shows and controls their processes.
        self.unified_active_pane = "gpu"
        # Detailed mode shows the process pane by default. `p` can hide it for
        # the current session without changing the persisted single-line
        # process-pane preference.
        self.unified_process_panel_hidden = False
        self.unified_selected_process = 0
        self.unified_selected_process_pid = None
        self.unified_process_filter = ""
        self.unified_process_filter_editing = False
        self.unified_process_sort_mode = "vram"
        self.unified_process_sort_descending = True
        self.unified_process_rows = 0
        self._unified_process_visible_rows = 1
        self._unified_process_count = 0
        self.unified_command_scroll = 0
        self._unified_command_line_count = 0
        self._unified_command_page_size = 0
        self._pending_process_signal = None
        self._process_action_notice = None
        self.tui_help_visible = False
        self._click_targets = {}
        self._click_regions = []
        self._body_click_targets = {}
        self._body_click_regions = []
        self._tui_menu = None
        self._tui_hover = HoverState()
        self._tui_hover_dirty = False
        self._tui_base_rows = None
        # Created on the first paint and dropped whenever the screen must be
        # repainted from scratch (see _answer_secret_request).
        self._tui_diff_screen = None
        self._toolbar_overflow = []
        self._unified_page_start = 0
        self._unified_gpu_count = 0
        self._unified_page_size = 1
        self._unified_gpu_history = {}
        self._unified_process_history = {}
        self._unified_history_lock = threading.Lock()
        self.refresh_needed = threading.Event()  # Flag to trigger immediate refresh after key press
        self.cached_stats = None  # Cached GPU stats data
        self.cached_raw_stats = {}  # Cached raw stats per client index
        self._cache_lock = threading.Lock()
        # Rendered node blocks reused while their sample and layout hold
        # (see _format_client_blocks).
        self._formatted_node_blocks = {}
        # Terminal width the cached blocks were laid out for.
        self._blocks_formatted_width = None
        self._last_update_time = None
        self._last_fetch_duration = None
        self._last_fetch_error = None
        self._toggle_disabled_servers = set()
        # Pool indices currently listed as nodes; the promoted local
        # machine lives in the top strip instead (see _compute_node_indices).
        self._node_listed_indices = list(range(len(self.pool)))
        self._node_samples = {}
        self._node_workers = []
        self._signal_worker = None
        self._secret_requests = Queue()
    
    def connect_all(self):
        """Open every client's session behind one self-rewriting line.

        Paramiko logs a banner and an auth line per host and each client
        announced itself as well, so starting up scrolled a dozen lines
        of transport chatter past before the first frame drew. A meter
        says the same thing in one line and leaves the screen clean.
        Only when stdout is a terminal: piped or redirected runs keep
        their plain log output, which is what a script would parse.
        """
        global _active_connect_progress

        progress = _ConnectProgress(len(self.pool))
        # Paramiko's INFO lines would land on top of the meter. Whether a
        # node came up is reported by the meter and, in full, by the node's
        # own error panel once the UI opens.
        if progress.enabled:
            logging.disable(logging.CRITICAL)
            _active_connect_progress = progress
        try:
            for client in self.pool:
                label = str(client.description or "node")
                progress.show(label)
                try:
                    connected = client.connect(announce=False)
                except Exception as e:
                    logging.error(msg=f"Failed to connect to {label}: {e}")
                    client._set_connect_error(str(e), error_type="connect")
                    connected = False
                progress.mark(bool(connected))
        finally:
            if progress.enabled:
                logging.disable(logging.NOTSET)
            _active_connect_progress = None
            progress.finish()

    @staticmethod
    def _throughput_kib_per_second(value):
        """Read a formatted PCIe reading back into KiB/s, or None if unknown."""
        text = str(value if value is not None else "").strip()
        if not text or text in {"N/A", "-"}:
            return None
        amount_text, unit = extract_value_and_unit(text)
        try:
            amount = float(amount_text)
        except (TypeError, ValueError):
            return None
        unit = unit.lower()
        if unit.startswith("mb"):
            return amount * 1024
        if unit.startswith("gb"):
            return amount * 1024 * 1024
        # Both NVML and `nvidia-smi -q -x` report PCIe throughput in KB/s.
        return amount

    def _add_pcie_link_columns(self, stats):
        """Describe each GPU's PCIe link and how hard RX/TX push against it.

        The ceiling is the link the card is running on right now, not the width
        and generation it could reach in a different slot. NVML's "max" figures
        describe the card alone: a 3090 Ti wired through an x4 riser still
        reports a maximum of x16, and dividing by that would hide a saturated
        link behind a comfortable-looking percentage. The live state is also
        the honest one at idle, where the link drops to gen1 to save power but
        the traffic being divided by it is near zero anyway.
        """
        links = []
        maximum_links = []
        capacities = []
        rx_percent = []
        tx_percent = []

        for _, row in stats.iterrows():
            generation = row.get("pcie_link_gen_current")
            width = row.get("pcie_link_width_current")
            gen_max = row.get("pcie_link_gen_max")
            width_max = row.get("pcie_link_width_max")
            # A driver that only answers for the ceiling still says something.
            if parse_link_number(generation) is None or parse_link_number(width) is None:
                generation, width = gen_max, width_max

            link = format_pcie_link(generation, width)
            maximum_link = format_pcie_link(gen_max, width_max)
            links.append(link)
            maximum_links.append("" if maximum_link in {link, "N/A"} else maximum_link)

            capacity = pcie_link_capacity_kib_per_second(generation, width)
            capacities.append(
                format_bandwidth(f"{capacity:.0f}", "KB/s") if capacity else "N/A"
            )
            for column, collected in (("rx", rx_percent), ("tx", tx_percent)):
                measured = self._throughput_kib_per_second(row.get(column))
                if capacity and measured is not None:
                    collected.append(max(0.0, min(100.0, measured / capacity * 100)))
                else:
                    collected.append(None)

        stats["link"] = links
        stats[f"{HIDDEN_COLUMN_PREFIX}link_max"] = maximum_links
        stats[f"{HIDDEN_COLUMN_PREFIX}link_capacity"] = capacities
        stats[f"{HIDDEN_COLUMN_PREFIX}rx_percent"] = rx_percent
        stats[f"{HIDDEN_COLUMN_PREFIX}tx_percent"] = tx_percent

    @staticmethod
    def _apply_gpu_allowlist(client, stats, system_info):
        """Drop the GPUs a server did not hand to nvidb.

        Filtering here - before any column is derived - keeps every later
        per-row list the same length as the table. The GPU index column
        is left alone on purpose: it carries the host's own GPU index,
        which is what the process lookups join on and what the UI must
        keep showing, so a host that lends out only GPU 2 still calls it 2.
        """
        allowed = getattr(client, "gpu_ids", None)
        if allowed is None:
            return stats, system_info

        allowed = set(allowed)

        def keep(frame):
            if frame.empty:
                return frame
            for column in ("gpu_index", "GPU"):
                if column in frame.columns:
                    return frame[
                        frame[column].map(
                            lambda value: _as_gpu_index(value) in allowed
                        )
                    ]
            return frame

        stats = keep(stats)
        system_info = dict(system_info)
        advanced = system_info.get("advanced_metrics")
        if isinstance(advanced, pd.DataFrame):
            system_info["advanced_metrics"] = keep(advanced)
        # The header count must describe what is on screen, not what
        # the driver reported.
        system_info["attached_gpus"] = str(len(stats))
        return stats, system_info

    def _collect_client_gpu_info(
        self, client, *, want_process_details=False, snapshot=None, enrich=True,
    ):
        """Collect and normalize one node without touching shared UI state."""
        stats, system_info = snapshot if snapshot is not None else client.get_full_gpu_info()
        stats, system_info = stats.copy(), dict(system_info)
        stats, system_info = self._apply_gpu_allowlist(client, stats, system_info)

        # Fetch system stats (CPU, memory, swap) and merge into system_info
        if enrich and not system_info.get("error"):
            try:
                sys_stats = client.get_system_stats()
                system_info["system_stats"] = sys_stats
            except Exception as e:
                logging.warning(f"Failed to get system stats for {client.description}: {e}")

        # Empty/error samples still retain their node identity.
        if stats.empty:
            return stats, system_info, {}, {}

        # Compact rx/tx display, cut to the column width limit.
        stats['rx'] = [format_bandwidth(*extract_value_and_unit(value))[:11] for value in stats['rx_util']]
        stats['tx'] = [format_bandwidth(*extract_value_and_unit(value))[:11] for value in stats['tx_util']]
        self._add_pcie_link_columns(stats)
        stats['power'] = [f"{row['power_state']} {'/'.join(extract_numbers(row['power_draw']))}/{'/'.join(extract_numbers(row['current_power_limit']))}" for _, row in stats.iterrows()]
        stats['memory[used/total]'] = [f"{'/'.join(extract_numbers(row['used']))}/{'/'.join(extract_numbers(row['total']))}" for _, row in stats.iterrows()]

        # Add process information as a column (batch pid->user lookup to avoid per-process SSH calls)
        process_list = []
        try:
            all_processes, user_summary_for_client = client.get_process_summary(
                stats,
                detailed=want_process_details,
            ) if enrich else client.get_process_summary(stats, resolve_users=False)

            per_gpu_user_summary = {}
            per_gpu_process_details = {}
            for proc in all_processes:
                gpu_idx = proc.get("gpu_index")
                gpu_key = str(gpu_idx)
                username = proc.get("username")
                used_memory = proc.get("used_memory", "0 MiB")
                per_gpu_process_details.setdefault(gpu_key, []).append(
                    {
                        "pid": proc.get("pid", "N/A"),
                        "username": username or "N/A",
                        "used_memory": used_memory,
                        "type": proc.get("type", "N/A"),
                        "process_name": proc.get("process_name", "N/A"),
                        "command": proc.get("command") or proc.get("process_name", "N/A"),
                        "cpu_percent": proc.get("cpu_percent"),
                        "mem_percent": proc.get("mem_percent"),
                        "rss_kb": proc.get("rss_kb"),
                        "elapsed": proc.get("elapsed"),
                        "state": proc.get("state"),
                        "threads": proc.get("threads"),
                    }
                )

                if not username or username == "N/A":
                    continue
                try:
                    memory_value = int(str(used_memory).replace("MiB", "").strip())
                except Exception:
                    memory_value = 0
                if memory_value <= 0:
                    continue

                gpu_summary = per_gpu_user_summary.setdefault(gpu_idx, {})
                gpu_summary[username] = gpu_summary.get(username, 0) + memory_value

            for label, row in stats.iterrows():
                # Join on the same key get_process_summary gave each process:
                # the host's GPU index, which stops matching the row label
                # once the driver skips a GPU it cannot open.
                try:
                    gpu_idx = int(row.get("gpu_index", label))
                except (TypeError, ValueError):
                    gpu_idx = label
                user_summary = per_gpu_user_summary.get(gpu_idx, {})
                if user_summary:
                    process_list.append(client.format_user_memory_compact(user_summary))
                else:
                    process_list.append("-")
        except Exception as e:
            logging.warning(f"Failed to get process info: {e}")
            process_list = ["-" for _ in range(len(stats))]
            user_summary_for_client = {}
            per_gpu_process_details = {}

        stats['processes'] = process_list

        # rename columns: product_name -> name, gpu_temp -> temp, fan_speed -> fan, memory_util -> mem_util, gpu_util -> util, gpu_index -> GPU
        stats = stats.rename(columns={'product_name': 'name', 'gpu_temp': 'temp', 'fan_speed': 'fan', 'memory_util': 'mem_util', 'gpu_util': 'util', 'gpu_index': 'GPU'})

        # replace the NVIDIA/GeForce with "" in name column
        stats['name'] = stats['name'].str.replace('NVIDIA', '').str.replace('GeForce', '').str.strip()

        # remove rows: product_architecture, rx_util, tx_util, power_state, power_draw, current_power_limit, used, total, free
        # but keep the new processes column
        columns_to_drop = [
            'product_architecture', 'rx_util', 'tx_util', 'power_state',
            'power_draw', 'current_power_limit', 'used', 'total', 'free',
            'pcie_link_gen_current', 'pcie_link_width_current',
            'pcie_link_gen_max', 'pcie_link_width_max',
        ]
        stats = stats.drop(columns=columns_to_drop, errors="ignore")

        # Reorder columns: move mem_util before memory[used/total] and processes at the end
        desired_order = ['GPU', 'name', 'fan', 'util', 'temp', 'link', 'rx', 'tx', 'power', 'mem_util', 'memory[used/total]', 'processes']
        # Only keep columns that exist in stats
        ordered_columns = [col for col in desired_order if col in stats.columns]
        # Add any remaining columns that weren't in desired_order
        remaining_columns = [col for col in stats.columns if col not in ordered_columns]
        stats = stats[ordered_columns + remaining_columns]

        return stats, system_info, user_summary_for_client, per_gpu_process_details

    @staticmethod
    def _sample_metadata(samples):
        users = {idx: sample[2] for idx, sample in samples.items()}
        processes = {idx: sample[3] for idx, sample in samples.items()}
        totals = {}
        for summary in users.values():
            for username, mib in summary.items():
                totals[username] = totals.get(username, 0) + mib
        return {
            "user_memory_by_client": users,
            "user_memory_global": totals,
            "process_details_by_client": processes,
        }

    def get_client_gpus_info(self, return_raw: bool = False):
        # Synchronous consumers (--once, logger, web) keep a complete snapshot.
        want_details = self._process_panel_visible()
        samples = {
            idx: self._collect_client_gpu_info(client, want_process_details=want_details)
            for idx, client in enumerate(self.pool)
        }
        raw = {idx: (sample[0], sample[1]) for idx, sample in samples.items()}
        raw["_nvidb"] = self._sample_metadata(samples)
        formatted = self._format_client_blocks([raw[idx] for idx in range(len(self.pool))])
        return (formatted, raw) if return_raw else formatted

    def _format_client_blocks(self, blocks):
        """Render each client's (stats, system_info) pair into its TUI block.

        Kept separate from the fetch so print_stats can re-run it from the
        cached raw stats when the terminal is resized between refreshes -
        the blocks are laid out for one specific width.
        """
        # These blocks only ever appear inside the per-node view's panel
        # border, so they reserve its margin up front - narrow terminals
        # drop the border (see `compact_layout`) and so reserve nothing.
        width = _terminal_size().columns
        self._blocks_formatted_width = width
        panel_margin = 0 if width < 72 else self._PANEL_BORDER_MARGIN
        formatted_stats = []
        cache = self._formatted_node_blocks
        render_key = (width, _monitor_theme, self.compact)

        def with_detail(name, detail):
            if detail and detail not in {"N/A", "-", ""}:
                return f"{name} ({detail})"
            return str(name)

        for idx, (client, (stats, system_info)) in enumerate(zip(self.pool, blocks)):
            monitor_text = self._node_monitor_text(system_info) if system_info.get("_monitor") else ""
            monitor_line = "\n" + monitor_text if monitor_text else ""
            cached = cache.get(idx)
            if cached and cached[0] is stats and cached[1] is system_info and cached[2] == render_key:
                formatted_stats.append(cached[3] + monitor_line)
                continue
            if system_info.get("loading"):
                formatted_stats.append("")
                continue
            # Create formatted table display (or error panel)
            if system_info.get("error"):
                error_panel = self._format_error_panel(
                    system_info["error"],
                    error_type=system_info.get("error_type"),
                    outer_margin=panel_margin,
                )
                formatted_stats.append(error_panel + monitor_line)
                continue

            formatted_table = self._format_fixed_width_table(
                stats, outer_margin=panel_margin
            )

            system_info_header = ""
            if system_info:
                source_display = with_detail(
                    system_info.get("data_source", "N/A"),
                    system_info.get("data_source_detail"),
                )
                # Surface DCGM status: it is easy to mistake a hidden advanced table
                # for "DCGM not connected" when profiling fields are just unsupported
                # (DCP metrics are datacenter-GPU only).
                advanced_display = ""
                adv_source = system_info.get("advanced_source")
                if adv_source:
                    if system_info.get("advanced_supported") is True:
                        advanced_display = " · Advanced " + with_detail(
                            adv_source, system_info.get("advanced_source_detail")
                        )
                    else:
                        advanced_display = f" · Advanced {adv_source} (profiling unavailable)"
                # Static facts, dimmed whole: the eye should land on the
                # GPU rows below, not on the driver version.
                system_info_header = (
                    colored(
                        f"Driver {system_info.get('driver_version', 'N/A')} · "
                        f"CUDA {system_info.get('cuda_version', 'N/A')} · "
                        f"GPUs {system_info.get('attached_gpus', '0')} · "
                        f"Source {source_display}{advanced_display}",
                        "dark_grey",
                    )
                    + "\n"
                )

            advanced_block = ""
            advanced_supported = system_info.get("advanced_supported") is True
            advanced_df = system_info.get("advanced_metrics")
            if advanced_supported and isinstance(advanced_df, pd.DataFrame) and not advanced_df.empty:
                try:
                    sub_blocks = ["Advanced (DCGM)"]
                    for title, cols in ADVANCED_METRIC_GROUPS:
                        cols_present = present_columns(advanced_df.columns, list(cols))
                        if len(cols_present) < 2:
                            continue
                        sub_df = advanced_df[cols_present].copy()
                        sub_table = self._format_fixed_width_table(
                            sub_df, outer_margin=panel_margin
                        )
                        sub_blocks.append(f"{title}\n{sub_table}")

                    advanced_block = "\n" + "\n".join(sub_blocks)
                except Exception:
                    advanced_block = ""

            block = f"{system_info_header}{formatted_table}{advanced_block}"
            cache[idx] = (stats, system_info, render_key, block)
            formatted_stats.append(block + monitor_line)
        return formatted_stats

    @staticmethod
    def _with_version_header(line, width):
        """Reserve the right edge for the same version reported by --version."""
        version = f"v{nvidb_config.VERSION}"
        remaining = width - display_width(version) - 1
        if remaining <= 0:
            return fit_display(version, max(0, width))
        # The shared fit/pad helpers accept plain text, not ANSI styling.
        # Counting escape bytes here clips visible metrics (e.g. "Load") and
        # leaves the version well short of the real right edge in a colour TTY.
        plain = NVClientPool._ANSI_ESCAPE_RE.sub("", str(line))
        left = str(line) if display_width(plain) <= remaining else fit_display(plain, remaining)
        visible_width = display_width(NVClientPool._ANSI_ESCAPE_RE.sub("", left))
        return left + " " * (remaining - visible_width + 1) + colored(version, "dark_grey")

    def _node_monitor_text(self, system_info):
        if not self.debug:
            return ""
        monitor = system_info.get("_monitor", {})
        if not monitor:
            return "Loading..."
        phase = monitor.get("phase", "connecting")
        labels = {"connecting": "Connecting SSH", "collecting": "Collecting GPU", "details": "Loading details"}
        if phase in labels:
            elapsed = max(0, time.time() - monitor.get("started", time.time()))
            text = f"{labels[phase]} ({elapsed:.0f}s)"
        else:
            text = "Telemetry"
        if monitor.get("updated"):
            age = max(0, time.time() - monitor["updated"])
            text += f" · updated {age:.0f}s ago"
        timings = monitor.get("timings", {})
        if timings:
            text += " · " + " / ".join(f"{key.upper()} {value:.1f}s" for key, value in timings.items())
        return text

    def _local_client_index(self):
        """Pool index of the local machine, or None when it is absent."""
        for idx, client in enumerate(self.pool):
            if client.port == "local":
                return idx
        return None

    def _compute_node_indices(self, raw_stats_by_client):
        """Pool indices listed as nodes in the TUI.

        The local machine reports through the header strip at the top of
        the page instead of the node list; it keeps a node section only
        while it has GPU rows of its own (a CUDA workstation must not
        lose its table) or when it is the only machine in the pool.
        """
        indices = list(range(len(self.pool)))
        local_idx = self._local_client_index()
        if local_idx is None or len(indices) <= 1:
            return indices
        entry = raw_stats_by_client.get(local_idx)
        if entry is not None and not entry[0].empty:
            return indices
        indices.remove(local_idx)
        return indices

    def _format_local_strip(self, terminal_width, raw_stats_by_client):
        """The local machine's one-line strip at the very top of the page.

        The machine nvidb runs on is not one more node to manage: by
        default it only reports the ground the user stands on - what
        system it is, CPU utilisation, load average, and memory - and
        stays out of the node list unless it has GPUs of its own.
        """
        local_idx = self._local_client_index()
        if local_idx is None:
            return []
        client = self.pool[local_idx]
        username = str(getattr(client, "username", "") or "") or getpass.getuser()
        hostname = platform.node().split(".")[0] or "localhost"
        identity_plain = f"{username}@{hostname}"
        os_plain = local_platform_summary()

        sep_plain = " · "
        sep = colored(sep_plain, "dark_grey")
        head_plain = f"⌂ {identity_plain}{sep_plain}{os_plain}"
        identity_styled = colored("⌂ ", "cyan", attrs=["bold"]) + colored(identity_plain, attrs=["bold"])
        head = identity_styled + sep + colored(os_plain, "dark_grey")

        _stats, system_info = raw_stats_by_client.get(local_idx, (None, {}))
        sys_stats = system_info.get("system_stats") or {}
        if not sys_stats:
            if not self.debug:
                return [head if display_width(head_plain) <= terminal_width else fit_display(head_plain, terminal_width)]
            tail_plain = "collecting system stats…"
            plain = f"{head_plain}{sep_plain}{tail_plain}"
            if display_width(plain) > terminal_width:
                return [fit(plain, terminal_width)]
            return [head + sep + colored(tail_plain, "dark_grey")]

        cores = int(sys_stats.get("cpu_cores") or 0)
        cpu_percent = int(round(float(sys_stats.get("cpu_percent") or 0.0)))
        cpu_color = self._threshold_color(cpu_percent, 50, 80, "green")

        load_avg = sys_stats.get("load_avg")
        load_values = tuple(float(value) for value in load_avg) if load_avg else None
        load_color = None
        if load_values is not None and cores > 0:
            load_color = self._threshold_color(load_values[0] / cores, 0.7, 1.0, "green")

        mem_used = float(sys_stats.get("mem_used_gb") or 0.0)
        mem_total = float(sys_stats.get("mem_total_gb") or 0.0)
        mem_percent = (
            int(round(mem_used / mem_total * 100)) if mem_total > 0 else None
        )
        mem_color = self._threshold_color(mem_percent, 75, 90, "green")

        swap_used = float(sys_stats.get("swap_used_gb") or 0.0)
        swap_total = float(sys_stats.get("swap_total_gb") or 0.0)

        def _metric_parts(*, bars, cores_suffix, full_load, swap):
            """(plain, styled) metric pieces for one strip variant."""
            parts = []
            cores_plain = f" ({cores}C)" if cores_suffix and cores > 0 else ""
            bar_plain = " " + "─" * 6 if bars else ""
            cpu_plain = f"CPU {cpu_percent}%{bar_plain}{cores_plain}"
            cpu_styled = (
                colored("CPU ", "dark_grey")
                + colored(f"{cpu_percent}%", cpu_color)
                + ((" " + self._mini_bar(cpu_percent, cpu_color)) if bars else "")
                + cores_plain
            )
            parts.append((cpu_plain, cpu_styled))
            if load_values is not None:
                if full_load:
                    load_plain_values = " ".join(
                        f"{value:.2f}" for value in load_values
                    )
                else:
                    load_plain_values = f"{load_values[0]:.2f}"
                load_styled_values = (
                    colored(load_plain_values, load_color)
                    if load_color
                    else load_plain_values
                )
                parts.append(
                    (
                        f"Load {load_plain_values}",
                        colored("Load ", "dark_grey") + load_styled_values,
                    )
                )
            if mem_percent is not None:
                mem_detail = f" {mem_used:.1f}/{mem_total:.0f}G" if bars or full_load else ""
                mem_plain = f"Mem {mem_percent}%{bar_plain}{mem_detail}"
                mem_styled = (
                    colored("Mem ", "dark_grey")
                    + colored(f"{mem_percent}%", mem_color)
                    + ((" " + self._mini_bar(mem_percent, mem_color)) if bars else "")
                    + mem_detail
                )
                parts.append((mem_plain, mem_styled))
            if swap and swap_total > 0 and swap_used >= 0.1:
                swap_plain = f"Swap {swap_used:.1f}/{swap_total:.0f}G"
                parts.append(
                    (swap_plain, colored("Swap ", "dark_grey") + swap_plain[5:])
                )
            return parts

        variants = (
            _metric_parts(bars=True, cores_suffix=True, full_load=True, swap=True),
            _metric_parts(bars=False, cores_suffix=True, full_load=True, swap=False),
            _metric_parts(bars=False, cores_suffix=False, full_load=False, swap=False),
        )
        for variant_index, parts in enumerate(variants):
            drop_os = variant_index == len(variants) - 1
            plain_pieces = [head_plain if not drop_os else f"⌂ {identity_plain}"]
            styled_pieces = [identity_styled if drop_os else head]
            plain_pieces.extend(plain for plain, _styled in parts)
            styled_pieces.extend(styled for _plain, styled in parts)
            plain = sep_plain.join(plain_pieces)
            if display_width(plain) <= terminal_width:
                return [sep.join(styled_pieces)]
        return [fit(plain, terminal_width)]

    def _client_table_identity(self, client_index):
        """Return compact node and hostname labels for the unified TUI table."""
        client = self.pool[client_index]
        description = str(client.description or "").strip()
        configured_host = str(client.host or "").strip()

        if client.port == "local":
            try:
                hostname = socket.gethostname().strip()
            except Exception:
                hostname = ""
            return "Local", hostname or configured_host or "localhost"

        node = description or configured_host or f"Node {client_index + 1}"
        return node, configured_host or "N/A"

    def _build_unified_gpu_table(self, raw_stats_by_client):
        """Combine all available per-node GPU rows into one display table."""
        frames = []
        for idx in range(len(self.pool)):
            entry = raw_stats_by_client.get(idx)
            if entry is None or entry[0].empty:
                continue
            stats = entry[0]

            node, hostname = self._client_table_identity(idx)
            table = stats.copy()
            table["Node"] = node
            table["Hostname"] = hostname
            columns = [column for column in self.UNIFIED_TABLE_COLUMNS if column in table.columns]
            columns += [
                column
                for column in table.columns
                if str(column).startswith(HIDDEN_COLUMN_PREFIX)
            ]
            frames.append(table[columns])

        if not frames:
            return pd.DataFrame(columns=list(self.UNIFIED_TABLE_COLUMNS))
        return pd.concat(frames, ignore_index=True, sort=False).fillna("N/A")

    def _unified_gpu_capacity(self, row):
        """Return utilization and VRAM values used by summaries and sorting."""
        utilization = parse_leading_float(row.get("util"))
        memory_value = str(row.get("memory[used/total]", "") or "")
        used_mib = total_mib = None
        if "/" in memory_value:
            used_value, total_value = memory_value.split("/", 1)
            used_mib = parse_leading_float(used_value)
            total_mib = parse_leading_float(total_value)

        memory_percent = None
        free_mib = None
        if used_mib is not None and total_mib is not None and total_mib > 0:
            memory_percent = max(0.0, min(100.0, used_mib / total_mib * 100))
            free_mib = max(0.0, total_mib - used_mib)

        available = (
            utilization is not None
            and utilization < 5
            and memory_percent is not None
            and memory_percent < 10
        )
        return {
            "utilization": utilization,
            "used_mib": used_mib,
            "total_mib": total_mib,
            "free_mib": free_mib,
            "memory_percent": memory_percent,
            "available": available,
            "busy": utilization is not None and utilization >= 50,
        }

    def _get_unified_sort_mode(self):
        mode = self.unified_sort_mode
        return mode if mode in self.UNIFIED_SORT_MODES else "node"

    def _get_unified_filter_mode(self):
        mode = self.unified_filter_mode
        return mode if mode in self.UNIFIED_FILTER_MODES else "all"

    def _filter_unified_gpu_table(self, table):
        """Filter unified GPU rows by availability or load state."""
        mode = self._get_unified_filter_mode()
        if table.empty or mode == "all":
            return table.copy()
        if mode == "errors":
            return table.iloc[0:0].copy()

        matches = []
        for _, row in table.iterrows():
            capacity = self._unified_gpu_capacity(row)
            matches.append(
                capacity["available"]
                if mode == "available"
                else capacity["busy"]
            )
        return table.loc[matches].reset_index(drop=True)

    def _sort_unified_gpu_table(self, table):
        """Sort unified rows without changing the cached source table."""
        mode = self._get_unified_sort_mode()
        if table.empty or mode == "node":
            return table.copy()

        sorted_table = table.copy()
        capacity = [
            self._unified_gpu_capacity(row)
            for _, row in sorted_table.iterrows()
        ]
        sorted_table["_nvidb_order"] = range(len(sorted_table))
        sorted_table["_nvidb_available"] = [
            1 if values["available"] else 0 for values in capacity
        ]
        sorted_table["_nvidb_util"] = [
            values["utilization"]
            if values["utilization"] is not None
            else (-1 if mode == "utilization" else 101)
            for values in capacity
        ]
        sorted_table["_nvidb_memory"] = [
            values["memory_percent"]
            if values["memory_percent"] is not None
            else (-1 if mode == "utilization" else 101)
            for values in capacity
        ]
        sorted_table["_nvidb_free"] = [
            values["free_mib"] if values["free_mib"] is not None else -1
            for values in capacity
        ]

        if mode == "available":
            sort_columns = [
                "_nvidb_available",
                "_nvidb_util",
                "_nvidb_free",
                "_nvidb_memory",
                "_nvidb_order",
            ]
            ascending = [False, True, False, True, True]
        else:
            sort_columns = [
                "_nvidb_util",
                "_nvidb_memory",
                "_nvidb_order",
            ]
            ascending = [False, False, True]

        sorted_table = sorted_table.sort_values(
            sort_columns,
            ascending=ascending,
            kind="stable",
        )
        return sorted_table.loc[:, table.columns].reset_index(drop=True)

    def _format_unified_capacity_summary(self, table):
        """Format cluster-wide availability and memory capacity."""
        if table.empty:
            return "Capacity: no GPU data"

        capacity = [
            self._unified_gpu_capacity(row)
            for _, row in table.iterrows()
        ]
        available_count = sum(values["available"] for values in capacity)
        busy_count = sum(values["busy"] for values in capacity)
        known_utilization = [
            values["utilization"]
            for values in capacity
            if values["utilization"] is not None
        ]
        average_utilization = (
            sum(known_utilization) / len(known_utilization)
            if known_utilization
            else None
        )
        known_memory = [
            values
            for values in capacity
            if values["used_mib"] is not None and values["total_mib"] is not None
        ]
        used_mib = sum(values["used_mib"] for values in known_memory)
        total_mib = sum(values["total_mib"] for values in known_memory)
        free_mib = max(0.0, total_mib - used_mib)

        terminal_width = _terminal_size().columns

        average_display = (
            f"{average_utilization:.0f}%"
            if average_utilization is not None
            else "N/A"
        )

        wide = total_mib > 0 and terminal_width >= 100
        # (label, value, value colour) per segment; the plain and styled
        # lines below are built from the same list.
        segments = [
            ("available " if wide else "avail ", f"{available_count}/{len(table)}", "green"),
            ("busy ", str(busy_count), "yellow"),
            (
                "avg util " if wide else "avg ",
                average_display,
                self._threshold_color(average_utilization, 50, 80, "green"),
            ),
        ]
        if total_mib > 0:
            memory_percent = used_mib / total_mib * 100
            if wide:
                vram_text = (
                    f"{used_mib / 1024:.1f}/{total_mib / 1024:.1f} GiB "
                    f"({memory_percent:.0f}%)"
                )
                free_text = f"{free_mib / 1024:.1f} GiB"
            else:
                vram_text = f"{used_mib / 1024:.0f}/{total_mib / 1024:.0f}G"
                free_text = f"{free_mib / 1024:.0f}G"
            segments += [
                ("VRAM ", vram_text, self._threshold_color(memory_percent, 75, 90, "green")),
                ("free ", free_text, "green"),
            ]

        plain_summary = (
            "Capacity: "
            + " | ".join(label + value for label, value, _color in segments)
            + ("" if total_mib > 0 else " | VRAM N/A")
        )
        if len(plain_summary) > terminal_width:
            return plain_summary[: max(0, terminal_width - 3)] + "..."
        sep = colored(" · ", "dark_grey")
        return (
            colored("Capacity:", "cyan", attrs=["bold"])
            + " "
            + sep.join(label + colored(value, color) for label, value, color in segments)
            + ("" if total_mib > 0 else sep + "VRAM N/A")
        )

    def _unified_grouping_active(self):
        """Group rows into node blocks only while the table follows node order."""
        return (
            bool(self.unified_group_by_node)
            and self._get_unified_sort_mode() == "node"
        )

    def _format_unified_node_band(self, node, hostname, rows):
        """Return (label, stats) describing one node's block of GPU rows."""
        capacity = [self._unified_gpu_capacity(row) for row in rows]
        count = len(capacity)
        available = sum(1 for values in capacity if values["available"])
        utilizations = [
            values["utilization"]
            for values in capacity
            if values["utilization"] is not None
        ]
        used_mib = sum(
            values["used_mib"] for values in capacity if values["used_mib"] is not None
        )
        total_mib = sum(
            values["total_mib"] for values in capacity if values["total_mib"] is not None
        )

        parts = [
            f"{count} GPU" if count == 1 else f"{count} GPUs",
            f"free {available}",
            (
                f"avg {sum(utilizations) / len(utilizations):.0f}%"
                if utilizations
                else "avg N/A"
            ),
        ]
        if total_mib > 0:
            parts.append(f"VRAM {used_mib / 1024:.0f}/{total_mib / 1024:.0f}G")
        return f"{node} ({hostname})", " · ".join(parts)

    def _unified_section_headers(self, visible_table, scope_table):
        """Map visible row index -> node band text for the grouped views."""
        headers = {}
        if not self._unified_grouping_active() or visible_table.empty:
            return headers

        def identity(row):
            return str(row.get("Node", "")), str(row.get("Hostname", ""))

        # One pass over the whole scope: a band summarises every row of its
        # node, including the ones on other pages.
        rows_by_node = {}
        for _, node_row in scope_table.iterrows():
            rows_by_node.setdefault(identity(node_row), []).append(node_row)

        previous_key = None
        for position, (_, row) in enumerate(visible_table.iterrows()):
            key = identity(row)
            if key == previous_key:
                continue
            previous_key = key
            headers[position] = self._format_unified_node_band(
                key[0], key[1], rows_by_node[key]
            )
        return headers

    def _paginate_unified_gpu_table(self, table, *, detailed):
        """Return the visible GPU page and selected row within that page."""
        terminal_height = _terminal_size().lines

        show_processes = self._process_panel_visible()
        show_trends = bool(self.unified_show_trends)
        extra_lines = (
            (18 if show_processes else 0)
            + (6 if show_trends else 0)
            # Room for the "filter is hiding GPUs" warning line.
            + (1 if self._get_unified_filter_mode() != "all" else 0)
            # The local machine's strip at the top of the page.
            + (1 if self._local_client_index() is not None else 0)
        )
        if detailed:
            reserved_lines = 10 + extra_lines
            page_size = max(1, (terminal_height - reserved_lines) // 5)
        else:
            # Node bands exist only in the compact table, one line each.
            group_count = (
                len(table[["Node", "Hostname"]].drop_duplicates())
                if self._unified_grouping_active() and not table.empty
                else 0
            )
            reserved_lines = 12 + extra_lines + group_count
            page_size = max(1, terminal_height - reserved_lines)

        total = len(table)
        selected_key = self.unified_selected_gpu_key
        selected = self._sync_selected_gpu(table)
        if (
            selected_key is not None
            and self.unified_selected_gpu_key != selected_key
        ):
            self.unified_selected_process = 0
            self.unified_selected_process_pid = None
            self.unified_command_scroll = 0
            if self._pending_process_signal:
                self._pending_process_signal = None
                self._set_process_action_notice(
                    "Selected GPU changed; signal cancelled",
                    "yellow",
                )
        self._unified_gpu_count = total
        self._unified_page_size = page_size

        if total == 0:
            self._unified_page_start = 0
            return table.copy(), None, ""

        page_start = (selected // page_size) * page_size
        self._unified_page_start = page_start
        page_end = min(total, page_start + page_size)
        visible_table = table.iloc[page_start:page_end].reset_index(drop=True)
        selected_on_page = selected - page_start
        page_status = f"Rows {page_start + 1}-{page_end}/{total}"
        return visible_table, selected_on_page, page_status

    def _sync_selected_gpu(self, table):
        """Re-find the selected GPU in `table` by its key, clamp, and store it.

        Returns the selected position; an empty table selects 0 and no key.
        """
        selected = int(self.unified_selected_gpu or 0)
        selected_key = self.unified_selected_gpu_key
        if selected_key is not None:
            for position, (_, row) in enumerate(table.iterrows()):
                if self._unified_gpu_history_key(row) == selected_key:
                    selected = position
                    break
        selected = max(0, min(selected, max(0, len(table) - 1)))
        self.unified_selected_gpu = selected
        self.unified_selected_gpu_key = (
            self._unified_gpu_history_key(table.iloc[selected])
            if len(table)
            else None
        )
        return selected

    def _sync_selected_process(self, processes):
        """Re-find the selected process by PID, clamp, and store it.

        Returns the selected index; an empty list selects 0 and no PID.
        """
        index = int(self.unified_selected_process or 0)
        selected_pid = self.unified_selected_process_pid
        if selected_pid is not None:
            for process_index, process in enumerate(processes):
                if str(process.get("pid", "")) == str(selected_pid):
                    index = process_index
                    break
        index = max(0, min(index, len(processes) - 1))
        self.unified_selected_process = index
        self.unified_selected_process_pid = (
            str(processes[index].get("pid", "")) if processes else None
        )
        return index

    def _unified_client_index_for_row(self, selected_row):
        """Resolve a unified-table row back to its local/remote client."""
        if selected_row is None:
            return None
        node = str(selected_row.get("Node", ""))
        hostname = str(selected_row.get("Hostname", ""))
        for index in range(len(self.pool)):
            client_node, client_hostname = self._client_table_identity(index)
            if client_node == node and client_hostname == hostname:
                return index
        return None

    def _get_unified_process_details(self, raw_stats_by_client, selected_row):
        """Return cached process rows for one selected unified GPU."""
        client_index = self._unified_client_index_for_row(selected_row)
        if client_index is None:
            return []

        # No "_nvidb" block before the first publish, and no entry for a
        # node that has not been sampled yet.
        meta = raw_stats_by_client.get("_nvidb", {})
        client_details = meta.get("process_details_by_client", {}).get(client_index, {})
        return list(client_details.get(str(selected_row.get("GPU", "")), []))

    def _get_sorted_unified_processes(
        self,
        raw_stats_by_client,
        selected_row,
    ):
        processes = self._get_unified_process_details(
            raw_stats_by_client,
            selected_row,
        )
        return self._filter_and_sort_processes(processes)

    def _filter_and_sort_processes(self, processes):
        """Apply the active view to cached rows without mutating the source."""
        query = str(
            self.unified_process_filter or ""
        ).strip().casefold()
        if query:
            processes = [
                process
                for process in processes
                if query
                in " ".join(
                    str(process.get(field) or "")
                    for field in (
                        "pid",
                        "username",
                        "type",
                        "state",
                        "process_name",
                        "command",
                    )
                ).casefold()
            ]

        mode = self._get_unified_process_sort_mode()
        descending = self.unified_process_sort_descending
        present = []
        missing = []
        for process in processes:
            value = self._unified_process_sort_value(process, mode)
            if value is None:
                missing.append(process)
            else:
                present.append((value, process))
        present.sort(key=lambda item: item[0], reverse=descending)
        return [process for _value, process in present] + missing

    def _get_unified_process_sort_mode(self):
        mode = self.unified_process_sort_mode
        if mode in self.UNIFIED_PROCESS_SORT_MODES:
            return mode
        return "vram"

    @staticmethod
    def _process_elapsed_seconds(value):
        """Convert ps etime text ([[dd-]hh:]mm:ss) into seconds."""
        text = str(value or "").strip()
        if not text:
            return None
        days = 0
        if "-" in text:
            days_text, text = text.split("-", 1)
            try:
                days = int(days_text)
            except ValueError:
                return None
        try:
            parts = [int(part) for part in text.split(":")]
        except ValueError:
            return None
        if not 1 <= len(parts) <= 3:
            return None
        seconds = 0
        for part in parts:
            seconds = seconds * 60 + part
        return days * 86400 + seconds

    # Process field read as a number for each numeric process sort mode.
    _PROCESS_SORT_FIELDS = {
        "vram": "used_memory",
        "cpu": "cpu_percent",
        "mem": "mem_percent",
        "rss": "rss_kb",
        "pid": "pid",
    }

    def _unified_process_sort_value(self, process, mode):
        if mode == "time":
            return self._process_elapsed_seconds(process.get("elapsed"))
        if mode == "command":
            command = (
                process.get("command")
                or process.get("process_name")
                or ""
            )
            return str(command).casefold() or None
        field = self._PROCESS_SORT_FIELDS.get(mode)
        return parse_leading_float(process.get(field)) if field else None

    @staticmethod
    def _format_memory_kb(value):
        """Render a KiB reading from `ps` as a compact K/M/G string."""
        if value is None:
            return "-"
        try:
            kib = float(value)
        except (TypeError, ValueError):
            return "-"
        if kib >= 1024 * 1024:
            return f"{kib / (1024 * 1024):.1f}G"
        if kib >= 1024:
            return f"{kib / 1024:.0f}M"
        return f"{kib:.0f}K"

    @staticmethod
    def _threshold_color(value, warning, critical, default="cyan"):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number >= critical:
            return "red"
        if number >= warning:
            return "yellow"
        return default

    @staticmethod
    def _process_state_color(state):
        """Running is green, uninterruptible/zombie/dead red, the rest dim."""
        if state.startswith("R"):
            return "green"
        if state.startswith(("D", "Z", "X")):
            return "red"
        return "dark_grey"

    @staticmethod
    def _process_vram_share(process, total_mib):
        """(VRAM MiB, % of `total_mib`) for one process; either may be None."""
        vram_mib = parse_leading_float(process.get("used_memory"))
        percent = vram_mib / total_mib * 100 if vram_mib is not None and total_mib else None
        return vram_mib, percent

    # (key, label, minimum width, alignment) of the process list columns.
    _PROCESS_TABLE_COLUMNS = (
        ("sel", "", 1, "left"),
        ("pid", "PID", 7, "right"),
        ("user", "USER", 9, "left"),
        ("vram", "VRAM", 10, "right"),
        ("gpu", "VRAM%", 6, "right"),
        ("cpu", "CPU%", 6, "right"),
        ("mem", "MEM%", 6, "right"),
        ("rss", "RSS", 7, "right"),
        ("time", "TIME", 9, "right"),
        ("state", "S", 3, "left"),
        ("command", "COMMAND", 12, "left"),
    )

    @classmethod
    def _process_table_columns(cls, inner_width, sort_column, arrow):
        """The process list columns that fit `inner_width`, with their widths.

        The sorted column's label carries `arrow`. Optional columns drop in a
        fixed order until the rest fit; COMMAND takes any width left over.
        """
        columns = [
            {
                "key": key,
                "label": f"{label}{arrow}" if key == sort_column else label,
                "minimum": minimum,
                "align": align,
            }
            for key, label, minimum, align in cls._PROCESS_TABLE_COLUMNS
        ]

        def columns_width(columns):
            return sum(column["minimum"] for column in columns) + max(
                0, len(columns) - 1
            )

        for optional_key in (
            "state",
            "time",
            "rss",
            "mem",
            "user",
            "gpu",
            "cpu",
            "vram",
        ):
            if columns_width(columns) <= inner_width:
                break
            columns = [
                column for column in columns if column["key"] != optional_key
            ]
        if columns_width(columns) > inner_width:
            for column in columns:
                if column["key"] == "pid":
                    column["minimum"] = 5
                elif column["key"] == "command":
                    column["minimum"] = max(4, inner_width - 8)

        extra_width = max(0, inner_width - columns_width(columns))
        for column in columns:
            column["width"] = column["minimum"]
            if column["key"] == "command":
                column["width"] += extra_width
        return columns

    def _process_notice_lines(self, panel):
        """The process-action notice as panel lines while it is live.

        An expired notice is cleared and yields no line.
        """
        notice = self._process_action_notice
        if notice and notice.get("expires_at", 0) > time.monotonic():
            message = str(notice.get("message", ""))
            return [
                panel.line(
                    message,
                    colored(message, notice.get("color") or "white", attrs=["bold"]),
                )
            ]
        if notice:
            self._process_action_notice = None
        return []

    def _format_unified_process_history(
        self,
        selected_row,
        process,
        width,
        *,
        include_gpu=False,
    ):
        """Return plain/styled rows for the selected process history."""
        history = self._get_unified_process_history(selected_row, process)
        gpu_history = (
            self._get_unified_gpu_history(selected_row)
            if include_gpu
            else []
        )
        title = (
            f"History  GPU {len(gpu_history)}/{self._HISTORY_SAMPLES}"
            f" | process {len(history)}/{self._HISTORY_SAMPLES}"
            if include_gpu
            else f"History  {len(history)}/{self._HISTORY_SAMPLES} samples"
        )
        if not history and not gpu_history:
            plain = fit(f"{title} | waiting for samples", width)
            return [(plain, colored(plain, "cyan"))]

        rows = [(title, colored(title, "cyan", attrs=["bold"]))]

        def metric_row(
            label,
            key,
            formatter,
            maximum,
            color,
            samples,
            *,
            minimum=0,
        ):
            values = [sample.get(key) for sample in samples]
            known = [float(value) for value in values if value is not None]
            if not known:
                plain = f"{label:<7}N/A"
                return plain, colored(plain, color)

            current = known[-1]
            average = sum(known) / len(known)
            peak = max(known)
            suffix = (
                f"now {formatter(current)}  avg {formatter(average)}  "
                f"max {formatter(peak)}"
            )
            spark_width = max(6, width - 8 - display_width(suffix) - 1)
            spark = self._sparkline(
                values,
                minimum=minimum,
                maximum=max(maximum, peak),
                width=spark_width,
            )
            plain = f"{label:<7}{spark} {suffix}"
            if display_width(plain) > width:
                plain = fit_display(plain, width, ellipsis="..")
                return plain, colored(plain, color)
            styled = (
                colored(f"{label:<7}", color, attrs=["bold"])
                + colored(spark, color)
                + f" {suffix}"
            )
            return plain, styled

        if include_gpu:
            rows.extend(
                [
                    metric_row(
                        "GPU U",
                        "utilization",
                        lambda value: f"{value:.0f}%",
                        100,
                        "green",
                        gpu_history,
                    ),
                    metric_row(
                        "GPU M",
                        "memory",
                        lambda value: f"{value:.0f}%",
                        100,
                        "cyan",
                        gpu_history,
                    ),
                    metric_row(
                        "GPU T",
                        "temperature",
                        lambda value: f"{value:.0f}C",
                        100,
                        "red",
                        gpu_history,
                        minimum=20,
                    ),
                ]
            )

        rows.extend(
            [
                metric_row(
                    "CPU",
                    "cpu_percent",
                    lambda value: f"{value:.1f}%",
                    100,
                    "magenta",
                    history,
                ),
                metric_row(
                    "VRAM",
                    "gpu_vram_percent",
                    lambda value: f"{value:.1f}%",
                    100,
                    "blue",
                    history,
                ),
                metric_row(
                    "RAM",
                    "mem_percent",
                    lambda value: f"{value:.1f}%",
                    100,
                    "yellow",
                    history,
                ),
                metric_row(
                    "RSS",
                    "rss_kb",
                    self._format_memory_kb,
                    1,
                    "light_grey",
                    history,
                ),
            ]
        )
        return rows

    def _format_unified_process_details(
        self,
        raw_stats_by_client,
        selected_row,
        *,
        height_budget=None,
        process_line_map=None,
        command_line_map=None,
        action_regions=None,
    ):
        """Format the selectable htop-style process pane below GPU cards."""
        if selected_row is None:
            self._unified_process_count = 0
            self._unified_process_visible_rows = 0
            self._unified_command_line_count = 0
            self._unified_command_page_size = 0
            return "Process details: no GPU selected"

        terminal_size = _terminal_size()
        terminal_width, terminal_height = terminal_size.columns, terminal_size.lines
        width = max(20, terminal_width)
        inner_width = max(1, width - 4)
        height_budget = max(
            8,
            int(height_budget if height_budget is not None else terminal_height),
        )
        compact_layout = (
            terminal_height < 28
            or height_budget <= 16
            or (
                width < 70
                and height_budget < 24
                and bool(self.unified_show_trends)
            )
        )
        compact_process_list = terminal_height < 36 or height_budget < 20
        ultra_compact = height_budget < 12
        compact_history = (
            terminal_height < 36
            and bool(self.unified_show_trends)
        )
        pane_focused = (
            self.unified_active_pane == "process"
        )
        label_color = "cyan" if pane_focused else "dark_grey"
        process_line_map = process_line_map if process_line_map is not None else {}
        command_line_map = command_line_map if command_line_map is not None else {}
        action_regions = action_regions if action_regions is not None else []

        panel = ProcessPanel(width, focused=pane_focused, colorize=colored)

        all_processes = self._get_unified_process_details(
            raw_stats_by_client,
            selected_row,
        )
        processes = self._filter_and_sort_processes(all_processes)
        self._unified_process_count = len(processes)
        selected_pid = self.unified_selected_process_pid
        selected_index = self._sync_selected_process(processes)
        if (
            selected_pid is not None
            and self.unified_selected_process_pid != str(selected_pid)
        ):
            self.unified_command_scroll = 0
            pending = self._pending_process_signal
            if pending:
                self._pending_process_signal = None
                self._set_process_action_notice(
                    "Selected process changed; signal cancelled",
                    "yellow",
                )

        capacity = self._unified_gpu_capacity(selected_row)
        node = selected_row.get("Node", "N/A")
        hostname = selected_row.get("Hostname", "N/A")
        gpu_index = selected_row.get("GPU", "N/A")
        filter_query = str(
            self.unified_process_filter or ""
        )
        filter_editing = bool(
            self.unified_process_filter_editing
        )
        if filter_query or filter_editing:
            count_label = f"{len(processes)}/{len(all_processes)} match"
            filter_label = (
                f"/{filter_query}{'_' if filter_editing else ''}"
            )
        else:
            count_label = f"{len(processes)} active"
            filter_label = ""
        process_sort_mode = self._get_unified_process_sort_mode()
        process_sort_label = self.UNIFIED_PROCESS_SORT_LABELS[
            process_sort_mode
        ]
        process_sort_arrow = "▼" if self.unified_process_sort_descending else "▲"
        focus_label = "ACTIVE" if pane_focused else "inactive · Enter/→"
        filter_title = f" | {filter_label}" if filter_label else ""
        title = (
            f"Processes | {count_label}{filter_title} | "
            f"{node} ({hostname}) | GPU {gpu_index} | "
            f"sort {process_sort_label}{process_sort_arrow} | {focus_label}"
        )
        lines = [panel.heading(title)]
        # Keep search and close available even when no process matches.
        # Short screens reuse the heading for these (see below), so they get
        # the short labels too.
        short = compact_layout or inner_width < 28
        if filter_editing:
            find_control = ("OK" if short else "Apply", "key", "\n", "cyan")
        else:
            find_control = ("/" if short else "/ Find", "key", "/", "cyan")
        process_controls = [
            find_control,
            ("x" if short else "Clear", "clear_search", None, "cyan"),
            ("p" if short else "Hide", "key", "p", "cyan"),
        ]
        button_lines, button_regions, _ = action_bar(process_controls, inner_width, self._control_style)
        if compact_layout:
            # Reuse the heading on short screens so controls cannot push the
            # signal buttons or command pager below the terminal edge.
            buttons_width = display_width(self._ANSI_ESCAPE_RE.sub("", button_lines[0]))
            button_start = width - 2 - buttons_width
            prefix = pad_display(fit_display(title, max(0, button_start - 3)), button_start - 2)
            lines[0] = colored("╭ " + prefix, label_color) + button_lines[0] + colored(" ╮", label_color)
            action_regions.extend((0, button_start + start, button_start + end, target)
                                  for _, start, end, target in button_regions)
        else:
            action_regions.extend((len(lines), 2 + start, 2 + end, target)
                                  for _, start, end, target in button_regions)
            lines.append(panel.line(self._ANSI_ESCAPE_RE.sub("", button_lines[0]), button_lines[0]))

        total_process_vram = sum(
            parse_leading_float(process.get("used_memory")) or 0
            for process in processes
        )
        total_gpu_mib = capacity.get("total_mib")
        vram_summary = f"{total_process_vram / 1024:.1f} GiB"
        if total_gpu_mib:
            vram_summary += (
                f" ({total_process_vram / total_gpu_mib * 100:.1f}% of GPU VRAM)"
            )
        visible_summary = (
            count_label
            + f" | {'matched' if filter_query else 'process'} VRAM "
            f"{vram_summary}"
            + (
                f" | {'selected' if pane_focused else 'details'} "
                f"{selected_index + 1}/{len(processes)}"
                if processes
                else ""
            )
        )
        if terminal_height >= 28:
            lines.append(
                panel.line(
                    visible_summary,
                    colored(visible_summary, "light_grey"),
                )
            )

        if not processes:
            self._unified_command_line_count = 0
            self._unified_command_page_size = 0
            self.unified_command_scroll = 0
            lines.append(panel.rule("PROCESS LIST"))
            empty = (
                f"No processes match /{filter_query}"
                if filter_query
                else "No active GPU processes"
            )
            lines.append(panel.line(empty, colored(empty, "dark_grey")))
            lines.append(panel.rule("ACTIONS"))
            history_label = (
                "[t] History ON"
                if self.unified_show_trends
                else "[t] History OFF"
            )
            action_line_index = len(lines)
            action_plain = f"Actions  {history_label}"
            action_start = action_plain.index(history_label)
            action_regions.append(
                (
                    action_line_index,
                    2 + action_start,
                    2 + action_start + len(history_label) - 1,
                    ("toggle_trends", None),
                )
            )
            lines.append(
                panel.line(
                    action_plain,
                    "Actions  "
                    + colored(
                        history_label,
                        label_color,
                        attrs=["bold"] if pane_focused else None,
                    ),
                )
            )
            lines.extend(self._process_notice_lines(panel))
            lines.append(panel.rule("", bottom=True))
            return "\n".join(lines)

        columns = self._process_table_columns(
            inner_width, process_sort_mode, process_sort_arrow
        )

        def padded_cell(value, column):
            value = panel.trim(value, column["width"])
            if column["align"] == "right":
                return " " * max(0, column["width"] - display_width(value)) + value
            return pad_display(value, column["width"])

        def process_cells(process, index):
            vram_mib, gpu_percent = self._process_vram_share(process, total_gpu_mib)
            cpu_percent = parse_leading_float(process.get("cpu_percent"))
            mem_percent = parse_leading_float(process.get("mem_percent"))
            rss_kb = parse_leading_float(process.get("rss_kb"))
            state = str(process.get("state") or "-")
            command = str(
                process.get("command")
                or process.get("process_name")
                or "N/A"
            )
            values = {
                "sel": (
                    "›"
                    if pane_focused and index == selected_index
                    else " "
                ),
                "pid": str(process.get("pid", "N/A")),
                "user": str(process.get("username", "N/A")),
                "vram": (
                    f"{vram_mib:.0f} MiB" if vram_mib is not None else "-"
                ),
                "gpu": (
                    f"{gpu_percent:.1f}" if gpu_percent is not None else "-"
                ),
                "cpu": (
                    f"{cpu_percent:.1f}" if cpu_percent is not None else "-"
                ),
                "mem": (
                    f"{mem_percent:.1f}" if mem_percent is not None else "-"
                ),
                "rss": self._format_memory_kb(rss_kb),
                "time": str(process.get("elapsed") or "-"),
                "state": state,
                "command": command,
            }
            colors = {
                "sel": "cyan",
                "pid": "light_grey",
                "user": "light_grey",
                "vram": self._threshold_color(
                    gpu_percent, 50, 80
                ),
                "gpu": self._threshold_color(
                    gpu_percent, 50, 80
                ),
                "cpu": self._threshold_color(
                    cpu_percent, 70, 100
                ),
                "mem": self._threshold_color(
                    mem_percent, 20, 40
                ),
                "rss": "dark_grey",
                "time": "dark_grey",
                "state": self._process_state_color(state),
                "command": "light_grey",
            }
            return values, colors

        header_cells = [
            padded_cell(column["label"], column) for column in columns
        ]
        header_plain = " ".join(header_cells)
        header_styled = " ".join(
            colored(
                cell,
                "cyan" if column["key"] == process_sort_mode else "dark_grey",
                attrs=(
                    ["bold"]
                    if column["key"] == process_sort_mode
                    else None
                ),
            )
            for cell, column in zip(header_cells, columns)
        )
        if not ultra_compact:
            if not compact_layout:
                lines.append(panel.rule("PROCESS LIST"))
            header_line_index = len(lines)
            lines.append(panel.line(header_plain, header_styled))
            column_start = 0
            for column in columns:
                # Clicking VRAM% sorts by VRAM, like the VRAM column itself.
                sort_target = "vram" if column["key"] == "gpu" else column["key"]
                if sort_target in self.UNIFIED_PROCESS_SORT_MODES:
                    action_regions.append(
                        (
                            header_line_index,
                            2 + column_start,
                            2 + column_start + column["width"] - 1,
                            ("process_sort", sort_target),
                        )
                    )
                column_start += column["width"] + 1

        automatic_visible = (
            1
            if compact_process_list
            else max(
                1,
                min(6, terminal_height // 8, max(1, height_budget // 5)),
            )
        )
        tail_reserve = (
            14
            + (3 if inner_width < 70 else 0)
            + (
                6
                if bool(self.unified_show_trends)
                else 0
            )
        )
        safe_visible = max(
            1,
            min(
                12,
                height_budget - len(lines) - tail_reserve,
            ),
        )
        requested_visible = max(
            0,
            int(self.unified_process_rows or 0),
        )
        max_visible = min(
            safe_visible,
            requested_visible or automatic_visible,
        )
        self._unified_process_visible_rows = max_visible
        start = max(
            0,
            min(
                selected_index - max_visible // 2,
                max(0, len(processes) - max_visible),
            ),
        )
        end = min(len(processes), start + max_visible)
        for index in range(start, end):
            process = processes[index]
            values, colors = process_cells(process, index)
            cells = [padded_cell(values[column["key"]], column) for column in columns]
            plain = " ".join(cells)
            if pane_focused and index == selected_index:
                styled = colored(
                    plain,
                    "light_grey",
                    "on_dark_grey",
                )
            else:
                styled = " ".join(
                    colored(
                        cell,
                        colors.get(column["key"]),
                    )
                    if colors.get(column["key"])
                    else cell
                    for cell, column in zip(cells, columns)
                )
            process_line_map[len(lines)] = index
            lines.append(panel.line(plain, styled))

        if terminal_height >= 28 and (start > 0 or end < len(processes)):
            scroll_status = f"Rows {start + 1}-{end}/{len(processes)} | wheel or j/k"
            lines.append(
                panel.line(
                    scroll_status,
                    colored(scroll_status, "dark_grey"),
                )
            )

        process = processes[selected_index]
        if not compact_history:
            selected_label = (
                f"{'SELECTED PROCESS' if pane_focused else 'PROCESS DETAILS'}"
                f" | PID {process.get('pid', 'N/A')}"
            )
            if compact_layout:
                selected_label += (
                    f" | {process.get('username', 'N/A')}"
                    f" | {process.get('type', 'N/A')}"
                    f" | {process.get('state') or '-'}"
                )
            lines.append(panel.rule(selected_label))
        vram_mib, gpu_percent = self._process_vram_share(process, total_gpu_mib)
        gpu_used_mib = capacity.get("used_mib")
        used_share = (
            vram_mib / gpu_used_mib * 100
            if vram_mib is not None and gpu_used_mib
            else None
        )
        cpu_percent = parse_leading_float(process.get("cpu_percent"))
        mem_percent = parse_leading_float(process.get("mem_percent"))
        vram_color = self._threshold_color(gpu_percent, 50, 80)
        if compact_layout:
            usage_fields = [
                (f"VRAM {vram_mib:,.0f}M" if vram_mib is not None else "VRAM N/A", vram_color),
            ]
            if gpu_percent is not None:
                usage_fields.append((f"{gpu_percent:.1f}% total", "cyan"))
            if used_share is not None:
                usage_fields.append((f"{used_share:.1f}% used", "cyan"))
        else:
            usage_fields = [
                (f"VRAM {vram_mib:,.0f} MiB" if vram_mib is not None else "VRAM N/A", vram_color),
                (
                    f"{gpu_percent:.1f}% of GPU VRAM"
                    if gpu_percent is not None
                    else "% of GPU VRAM N/A",
                    "cyan",
                ),
                (
                    f"{used_share:.1f}% of used VRAM"
                    if used_share is not None
                    else "% of used VRAM N/A",
                    "cyan",
                ),
            ]
        mem_label = "MEM" if compact_layout else "Host MEM"
        usage_fields += [
            (
                f"CPU {cpu_percent:.1f}%" if cpu_percent is not None else "CPU N/A",
                self._threshold_color(cpu_percent, 70, 100),
            ),
            (
                f"{mem_label} {mem_percent:.1f}%"
                if mem_percent is not None
                else f"{mem_label} N/A",
                self._threshold_color(mem_percent, 20, 40),
            ),
            (f"RSS {self._format_memory_kb(process.get('rss_kb'))}", "dark_grey"),
        ]
        threads = process.get("threads")
        try:
            threads_display = (
                "-"
                if threads is None
                else str(int(float(threads)))
            )
        except (TypeError, ValueError):
            threads_display = str(threads or "-")
        state = str(process.get("state") or "-")
        detail_fields = [
            (f"User {process.get('username', 'N/A')}", "light_grey"),
            (f"Type {process.get('type', 'N/A')}", "dark_grey"),
            (f"State {state}", self._process_state_color(state)),
            (f"Threads {threads_display}", "dark_grey"),
            (f"Elapsed {process.get('elapsed') or '-'}", "dark_grey"),
        ]
        if not compact_history:
            usage_rows = panel.fields("Usage", usage_fields)
            if ultra_compact:
                usage_rows = usage_rows[:1]
            for plain, styled in usage_rows:
                process_line_map[len(lines)] = selected_index
                lines.append(panel.line(plain, styled))
            if not compact_layout:
                for plain, styled in panel.fields("Details", detail_fields):
                    process_line_map[len(lines)] = selected_index
                    lines.append(panel.line(plain, styled))

        command = str(
            process.get("command") or process.get("process_name") or "N/A"
        ).strip()
        command_prefix = "Command  "
        command_width = max(1, inner_width - display_width(command_prefix))
        all_command_parts = wrap_display(command, command_width) or ["N/A"]
        history_rows = (
            self._format_unified_process_history(
                selected_row,
                process,
                inner_width,
                include_gpu=compact_history,
            )
            if self.unified_show_trends
            else []
        )
        if compact_history and ultra_compact:
            compact_labels = ("History", "GPU U", "CPU", "VRAM", "RSS")
            history_rows = [
                row
                for row in history_rows
                if row[0].lstrip().startswith(compact_labels)
            ]
            history_rows = history_rows[:max(0, height_budget - len(lines) - 3)]
        history_rule_lines = (
            0
            if compact_history and ultra_compact
            else (1 if history_rows else 0)
        )
        reserved_after_command = (
            history_rule_lines
            + len(history_rows)
            + (0 if compact_layout else 1)
            + 1  # action row
            + 1  # confirmation or action notice
            + 1  # bottom border
        )
        available_command_lines = max(
            1,
            height_budget - len(lines) - reserved_after_command,
        )
        command_page_cap = (
            5
            if compact_layout
            else max(5, min(12, height_budget // 2))
        )
        command_page_size = (
            0
            if compact_history
            else min(
                len(all_command_parts),
                command_page_cap,
                available_command_lines,
            )
        )
        command_start = (
            int(self.unified_command_scroll or 0)
            if 0 < command_page_size < len(all_command_parts)
            else 0
        )
        command_start = max(
            0,
            min(
                command_start,
                max(0, len(all_command_parts) - command_page_size),
            ),
        )
        command_end = min(
            len(all_command_parts),
            command_start + command_page_size,
        )
        command_parts = all_command_parts[command_start:command_end]
        self.unified_command_scroll = command_start
        self._unified_command_line_count = len(all_command_parts)
        self._unified_command_page_size = command_page_size
        if not compact_history:
            for command_index, command_part in enumerate(command_parts):
                prefix = (
                    command_prefix
                    if command_index == 0
                    else " " * display_width(command_prefix)
                )
                plain = prefix + command_part
                styled = (
                    colored(
                        prefix,
                        label_color,
                        attrs=["bold"] if command_index == 0 else None,
                    )
                    + colored(
                        command_part,
                        "light_grey",
                    )
                )
                process_line_map[len(lines)] = selected_index
                command_line_map[len(lines)] = selected_index
                lines.append(panel.line(plain, styled))

        if history_rows:
            if history_rule_lines:
                lines.append(panel.rule("SELECTED PROCESS HISTORY"))
            for plain, styled in history_rows:
                lines.append(panel.line(plain, styled))

        if not compact_layout:
            lines.append(panel.rule("ACTIONS"))
        command_paged = (
            not compact_history
            and command_page_size < len(all_command_parts)
        )
        history_enabled = bool(self.unified_show_trends)
        compact_actions = inner_width < 70
        tiny_actions = inner_width < 56
        action_separator = " " if compact_actions else "  "

        def action_color(color):
            return color if pane_focused else "dark_grey"

        def action_label(tiny, compact, full):
            if tiny_actions:
                return tiny
            return compact if compact_actions else full

        history_state = "ON" if history_enabled else "OFF"
        history_label = "[t]Hist" if command_paged else "[t] History"
        action_segments = []
        if not compact_actions:
            action_segments.append(("Actions", None, None))
        for key, signal, color in (
            ("i", "INT", "cyan"),
            ("T", "TERM", "yellow"),
            ("K", "KILL", "red"),
        ):
            label = action_label(
                f"{key}:{signal}", f"[{key}]{signal}", f"[{key}] {signal}",
            )
            action_segments.append((label, action_color(color), ("signal", signal)))
        action_segments.append((
            action_label(
                f"t:{int(history_enabled)}",
                f"[t]H:{history_state}",
                f"{history_label} {history_state}",
            ),
            action_color("cyan"),
            ("toggle_trends", None),
        ))
        if command_paged:
            page_label = (
                f"{command_start + 1}/{len(all_command_parts)}"
                if tiny_actions
                else f"{command_start + 1}-{command_end}/{len(all_command_parts)}"
            )
            action_segments.extend([
                ("<" if tiny_actions else "[<]Cmd",
                 action_color("cyan"), ("command_scroll", -1)),
                (page_label, "dark_grey", None),
                (">" if tiny_actions else "[>]Cmd",
                 action_color("cyan"), ("command_scroll", 1)),
            ])
        for symbol, delta in (("-", -1), ("+", 1)):
            action_segments.append((
                action_label(symbol, f"{symbol}r", f"[{symbol}] Rows"),
                action_color("cyan"),
                ("process_rows", delta),
            ))
        action_plain = action_separator.join(
            text for text, _color, _target in action_segments
        )
        action_styled = action_separator.join(
            colored(text, color) if color else text
            for text, color, _target in action_segments
        )
        action_line_index = len(lines)
        search_from = 0
        for text, _color, target in action_segments:
            character_start = action_plain.find(text, search_from)
            search_from = character_start + len(text)
            start_column = display_width(action_plain[:character_start])
            if target and start_column < inner_width:
                action_regions.append(
                    (
                        action_line_index,
                        2 + start_column,
                        2 + min(
                            inner_width,
                            start_column + display_width(text),
                        )
                        - 1,
                        target,
                    )
                )
        lines.append(panel.line(action_plain, action_styled))

        pending = self._pending_process_signal
        if pending and pending.get("expires_at", 0) <= time.monotonic():
            self._pending_process_signal = None
            pending = None
        if pending:
            # Confirm and cancel are separate buttons, so neither the signal
            # label nor a changing process row has to be clicked a second time.
            confirm_controls = [
                ("Confirm", "key", "\n", "yellow"),
                ("Cancel", "key", "\x1b", "cyan"),
            ]
            confirm_lines, confirm_regions, _ = action_bar(confirm_controls, inner_width, self._control_style)
            action_regions.extend((len(lines), 2 + start, 2 + end, target)
                                  for _, start, end, target in confirm_regions)
            buttons_plain = self._ANSI_ESCAPE_RE.sub("", confirm_lines[0])
            message = f" SIG{pending['signal']} → PID {pending['pid']}"
            plain = buttons_plain + message
            lines.append(panel.line(plain, confirm_lines[0] + colored(message, "yellow")))
        else:
            lines.extend(self._process_notice_lines(panel))

        lines.append(panel.rule("", bottom=True))
        return "\n".join(lines)

    # Telemetry samples kept per GPU and per process for the trend views.
    _HISTORY_SAMPLES = 60

    @staticmethod
    def _unified_gpu_history_key(row):
        return (
            str(row.get("Node", "")),
            str(row.get("Hostname", "")),
            str(row.get("GPU", "")),
        )

    @classmethod
    def _unified_process_history_key(cls, row, process):
        return cls._unified_gpu_history_key(row) + (str(process.get("pid", "")),)

    def _record_unified_gpu_history(
        self,
        raw_stats_by_client,
        *,
        timestamp=None,
    ):
        """Record one cached telemetry sample per GPU."""
        table = self._build_unified_gpu_table(raw_stats_by_client)
        if table.empty:
            return
        timestamp = time.time() if timestamp is None else timestamp
        gpu_samples = []
        process_samples = []
        for _, row in table.iterrows():
            capacity = self._unified_gpu_capacity(row)
            gpu_samples.append(
                (
                    self._unified_gpu_history_key(row),
                    {
                        "timestamp": timestamp,
                        "utilization": capacity["utilization"],
                        "memory": capacity["memory_percent"],
                        "temperature": parse_leading_float(
                            row.get("temp")
                        ),
                    },
                )
            )
            total_mib = capacity.get("total_mib")
            for process in self._get_unified_process_details(
                raw_stats_by_client,
                row,
            ):
                pid = str(process.get("pid", "")).strip()
                if not pid.isdigit():
                    continue
                vram_mib, gpu_vram_percent = self._process_vram_share(process, total_mib)
                process_samples.append(
                    (
                        self._unified_process_history_key(row, process),
                        {
                            "timestamp": timestamp,
                            "cpu_percent": parse_leading_float(
                                process.get("cpu_percent")
                            ),
                            "mem_percent": parse_leading_float(
                                process.get("mem_percent")
                            ),
                            "rss_kb": parse_leading_float(
                                process.get("rss_kb")
                            ),
                            "vram_mib": vram_mib,
                            "gpu_vram_percent": gpu_vram_percent,
                        },
                    )
                )

        with self._unified_history_lock:
            for key, sample in gpu_samples:
                history = self._unified_gpu_history.setdefault(
                    key,
                    deque(maxlen=self._HISTORY_SAMPLES),
                )
                history.append(sample)
            for key, sample in process_samples:
                history = self._unified_process_history.setdefault(
                    key,
                    deque(maxlen=self._HISTORY_SAMPLES),
                )
                history.append(sample)

            # PID churn should not grow a long-running TUI without bound. Keep
            # exited-process history around for an hour so a just-finished job
            # can still be inspected.
            stale_before = timestamp - 3600
            stale_keys = [
                key
                for key, history in self._unified_process_history.items()
                if history[-1]["timestamp"] < stale_before
            ]
            for key in stale_keys:
                del self._unified_process_history[key]

    def _get_unified_gpu_history(self, selected_row):
        key = self._unified_gpu_history_key(selected_row)
        with self._unified_history_lock:
            return list(self._unified_gpu_history.get(key, ()))

    def _get_unified_process_history(self, selected_row, process):
        key = self._unified_process_history_key(selected_row, process)
        with self._unified_history_lock:
            return list(
                self._unified_process_history.get(key, ())
            )

    @staticmethod
    def _sparkline(values, *, minimum, maximum, width):
        blocks = "▁▂▃▄▅▆▇█"
        width = max(1, width)
        values = list(values)[-width:]
        if not values:
            return ""
        span = max(1.0, maximum - minimum)
        rendered = []
        for value in values:
            if value is None:
                rendered.append(".")
                continue
            normalized = max(0.0, min(1.0, (float(value) - minimum) / span))
            index = min(len(blocks) - 1, int(round(normalized * (len(blocks) - 1))))
            rendered.append(blocks[index])
        return "".join(rendered)

    def _format_unified_gpu_trends(self, selected_row):
        """Format utilization, VRAM, and temperature history sparklines."""
        if selected_row is None:
            return "Trends: no GPU selected"
        history = self._get_unified_gpu_history(selected_row)
        node = selected_row.get("Node", "N/A")
        gpu_index = selected_row.get("GPU", "N/A")
        title = f"Trends: {node} GPU {gpu_index} | {len(history)}/{self._HISTORY_SAMPLES} samples"
        if not history:
            return f"{title}\nWaiting for telemetry samples"

        terminal_width = _terminal_size().columns

        def metric_line(label, key, minimum, maximum, unit, color):
            values = [sample.get(key) for sample in history]
            known = [float(value) for value in values if value is not None]
            if not known:
                return colored(f"{label:<5} N/A", color)
            current = known[-1]
            average = sum(known) / len(known)
            suffix = f"now {current:.0f}{unit} avg {average:.0f}{unit}"
            spark_width = max(
                8,
                terminal_width
                - display_width(label)
                - display_width(suffix)
                - 4,
            )
            spark = self._sparkline(
                values,
                minimum=minimum,
                maximum=maximum,
                width=spark_width,
            )
            line = f"{label:<5} {spark} {suffix}"
            if display_width(line) > terminal_width:
                return colored(fit(line, terminal_width), color)
            return (
                colored(f"{label:<5}", color, attrs=["bold"])
                + " "
                + colored(spark, color)
                + f" {suffix}"
            )

        return "\n".join(
            [
                colored(fit(title, terminal_width), "cyan", attrs=["bold"]),
                metric_line("Util", "utilization", 0, 100, "%", "green"),
                metric_line("VRAM", "memory", 0, 100, "%", "yellow"),
                metric_line("Temp", "temperature", 20, 100, "C", "red"),
            ]
        )

    def _format_unified_detailed_table(
        self,
        table,
        *,
        selected_row=None,
        row_line_map=None,
    ):
        """Format each GPU as a readable, color-aware four-line card.

        `row_line_map` is filled in with line offset -> row index so a click
        anywhere on a card can be traced back to its GPU.
        """
        if table.empty:
            return "No GPU data available"

        terminal_width = _terminal_size().columns
        table_width = max(20, terminal_width)
        content_width = max(1, table_width - 4)
        pane_focused = self.unified_active_pane == "gpu"
        border_horizontal = "─" if pane_focused else "╌"
        border_vertical = "│" if pane_focused else "┊"

        def clean(value):
            if value is None:
                return "N/A"
            try:
                if pd.isna(value):
                    return "N/A"
            except (TypeError, ValueError):
                pass
            value = str(value).strip()
            return value if value else "N/A"

        def add_unit(value, unit):
            value = clean(value)
            if value in {"N/A", "-"} or unit.lower() in value.lower():
                return value
            return f"{value} {unit}"

        def utilization_status(percent):
            if percent is None:
                return "UNKNOWN", None
            if percent >= 80:
                return "HIGH", "red"
            if percent >= 50:
                return "BUSY", "yellow"
            if percent >= 5:
                return "ACTIVE", "cyan"
            return "IDLE", "dark_grey"

        def link_load_percent(row, direction):
            value = row.get(f"{HIDDEN_COLUMN_PREFIX}{direction}_percent")
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def link_load(row, direction):
            percent = link_load_percent(row, direction)
            # An idle direction already reads as 0 KB/s; a percentage of the
            # link would only repeat it.
            if percent is None or percent <= 0:
                return ""
            # Rounding a trickle to "0%" would read as "no traffic at all".
            return " <1%" if percent < 1 else f" {percent:.0f}%"

        def link_load_color(row, direction):
            return get_pcie_load_color(link_load_percent(row, direction)) or "dark_grey"

        def make_field(key, text, minimum, color=None, bold=False):
            return {
                "key": key,
                "text": clean(text),
                "minimum": minimum,
                "color": color,
                "bold": bold,
            }

        def fit_fields(
            fields,
            *,
            drop_order=(),
            shrink_order=(),
            highlight=False,
        ):
            active = [dict(field) for field in fields]
            separator = "  "

            def minimum_total():
                return (
                    sum(
                        min(display_width(field["text"]), field["minimum"])
                        for field in active
                    )
                    + display_width(separator) * (len(active) - 1)
                )

            for key in drop_order:
                if minimum_total() <= content_width:
                    break
                active = [field for field in active if field["key"] != key]

            widths = [display_width(field["text"]) for field in active]
            excess = (
                sum(widths)
                + display_width(separator) * (len(active) - 1)
                - content_width
            )
            for key in shrink_order:
                if excess <= 0:
                    break
                for index, field in enumerate(active):
                    if field["key"] != key:
                        continue
                    minimum = min(display_width(field["text"]), field["minimum"])
                    reducible = max(0, widths[index] - minimum)
                    reduction = min(excess, reducible)
                    widths[index] -= reduction
                    excess -= reduction
                    break

            if excess > 0:
                for index in reversed(range(len(active))):
                    if excess <= 0:
                        break
                    reducible = max(0, widths[index] - 1)
                    reduction = min(excess, reducible)
                    widths[index] -= reduction
                    excess -= reduction

            plain_parts = [
                _clip(field["text"], width)
                for field, width in zip(active, widths)
            ]
            plain_length = (
                sum(display_width(part) for part in plain_parts)
                + display_width(separator) * (len(plain_parts) - 1)
            )
            padding = " " * max(0, content_width - plain_length)
            plain_line = separator.join(plain_parts) + padding
            if highlight:
                return colored(
                    plain_line,
                    "light_grey",
                    "on_dark_grey",
                )
            styled_parts = []
            for field, part in zip(active, plain_parts):
                color = field.get("color")
                attrs = ["bold"] if field.get("bold") else None
                styled_parts.append(
                    colored(part, color, attrs=attrs) if color else part
                )

            line = separator.join(styled_parts)
            return line + padding

        def progress_bar(percent, width=10):
            """Render a smooth line bar; empty when the metric is unknown."""
            if percent is None or table_width < 100:
                return ""
            value = max(0.0, min(100.0, percent))
            filled = int(round(value / 100.0 * width))
            if value > 0 and filled == 0:
                filled = 1
            return " [" + "━" * filled + "─" * (width - filled) + "]"

        def card_rule(left, right):
            return colored(
                left
                + border_horizontal * (table_width - 2)
                + right,
                "dark_grey",
            )

        top_border = card_rule("╭", "╮")
        middle_border = card_rule("├", "┤")
        bottom_border = card_rule("╰", "╯")

        def card_line(content):
            return (
                colored(f"{border_vertical} ", "dark_grey")
                + content
                + colored(f" {border_vertical}", "dark_grey")
            )
        lines = [top_border]
        for row_index, (_, row) in enumerate(table.iterrows()):
            capacity = self._unified_gpu_capacity(row)
            utilization = clean(row.get("util")).replace(" ", "")
            status, utilization_color = utilization_status(capacity["utilization"])
            marker = (
                "› "
                if pane_focused and row_index == selected_row
                else "  "
            )
            # Node name first: the node is what users scan for before the GPU index.
            gpu_badge = (
                f"{marker}{clean(row.get('Node'))} "
                f"GPU {clean(row.get('GPU'))} [{status}]"
            )
            identity_fields = [
                make_field("section", "GPU", 3, color="cyan", bold=True),
                make_field(
                    "gpu",
                    gpu_badge,
                    20,
                    color=utilization_color,
                    bold=True,
                ),
                make_field(
                    "model",
                    f"Model {clean(row.get('name'))}",
                    18,
                    color="light_grey",
                ),
                make_field(
                    "host",
                    clean(row.get("Hostname")),
                    13,
                    color="dark_grey",
                ),
            ]

            memory_percent = capacity["memory_percent"]
            load_fields = [
                make_field("section", "LOAD", 4, color="green", bold=True),
                make_field(
                    "util",
                    f"Util {utilization}{progress_bar(capacity['utilization'])}",
                    8,
                    color=utilization_color,
                    bold=True,
                ),
                make_field(
                    "power",
                    f"Power {add_unit(row.get('power'), 'W')}",
                    18,
                    color="light_grey",
                ),
            ]
            memory_fields = [
                make_field("section", "MEM/TEMP", 8, color="yellow", bold=True),
                make_field(
                    "vram",
                    f"VRAM {add_unit(row.get('memory[used/total]'), 'MiB')}"
                    f"{progress_bar(memory_percent, 8)}",
                    18,
                    color=self._threshold_color(memory_percent, 75, 90),
                    bold=True,
                ),
                make_field(
                    "temp",
                    f"Temp {clean(row.get('temp'))}",
                    9,
                    color=self._threshold_color(parse_leading_float(row.get("temp")), 70, 85),
                ),
                make_field(
                    "fan",
                    f"Fan {clean(row.get('fan'))}",
                    8,
                    color="dark_grey",
                ),
            ]
            # RX/TX read as a bare number unless the link they run over is on
            # screen too, so the card names the live mode and its per-direction
            # ceiling, and turns each direction's share of it into a percentage.
            # The card's own maximum is a separate field so a narrow card drops
            # it whole instead of slicing it mid-word.
            link_mode = clean(row.get("link"))
            link_max = clean(row.get(f"{HIDDEN_COLUMN_PREFIX}link_max"))
            link_capacity = clean(row.get(f"{HIDDEN_COLUMN_PREFIX}link_capacity"))
            link_text = f"Link {link_mode}"
            if link_capacity != "N/A":
                link_text += f" {link_capacity}"
            auxiliary_fields = [
                make_field("section", "I/O", 3, color="magenta", bold=True),
                make_field(
                    "link",
                    link_text,
                    10,
                    color="dark_grey",
                ),
            ]
            if link_max != "N/A":
                auxiliary_fields.append(
                    make_field("linkmax", f"of {link_max}", 9, color="dark_grey")
                )
            auxiliary_fields += [
                make_field(
                    "rx",
                    f"RX {clean(row.get('rx'))}{link_load(row, 'rx')}",
                    11,
                    color=link_load_color(row, "rx"),
                ),
                make_field(
                    "tx",
                    f"TX {clean(row.get('tx'))}{link_load(row, 'tx')}",
                    11,
                    color=link_load_color(row, "tx"),
                ),
                make_field(
                    "proc",
                    f"Proc {clean(row.get('processes'))}",
                    14,
                    color="light_grey",
                ),
            ]

            identity_line = fit_fields(
                identity_fields,
                drop_order=("host",),
                shrink_order=("model", "host", "gpu"),
                highlight=(pane_focused and row_index == selected_row),
            )
            load_line = fit_fields(
                load_fields,
                drop_order=("power",),
                shrink_order=("power", "util"),
            )
            memory_line = fit_fields(
                memory_fields,
                drop_order=("fan", "temp"),
                shrink_order=("vram", "temp", "fan"),
            )
            auxiliary_line = fit_fields(
                auxiliary_fields,
                drop_order=("linkmax", "link", "tx", "rx"),
                shrink_order=("proc", "link", "rx", "tx"),
            )
            if row_line_map is not None:
                for offset in range(4):
                    row_line_map[len(lines) + offset] = row_index
            lines.extend(
                [
                    card_line(identity_line),
                    card_line(load_line),
                    card_line(memory_line),
                    card_line(auxiliary_line),
                    (
                        bottom_border
                        if row_index == len(table) - 1
                        else middle_border
                    ),
                ]
            )
        return "\n".join(lines)

    def _get_unified_node_status_lines(self, raw_stats_by_client, *, errors_only=False):
        """Describe nodes that cannot contribute rows to the unified table."""
        lines = []
        hidden_count = 0
        # The promoted local machine reports through the top strip, so it
        # is neither listed nor counted among the hidden nodes here.
        for idx in self._compute_node_indices(raw_stats_by_client):
            node, hostname = self._client_table_identity(idx)
            stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))
            if system_info.get("error"):
                message = system_info.get("error", "Error")
                # A node that dropped off must not read like a node that simply
                # has no NVIDIA GPU, so it keeps the red marker of a real error.
                lines.append(colored(f"! {node} ({hostname}): {message}", "red"))
            elif idx not in raw_stats_by_client or system_info.get("loading"):
                if not errors_only and self.debug:
                    lines.append(f"- {node} ({hostname}): {self._node_monitor_text(system_info)}")
            elif not errors_only and stats.empty:
                # Machines without NVIDIA GPUs (e.g. a macOS laptop) would
                # otherwise repeat the same "no data" line on every refresh.
                if self.hide_unsupported:
                    hidden_count += 1
                    continue
                lines.append(f"- {node} ({hostname}): No GPU data available")
        if hidden_count:
            label = "node" if hidden_count == 1 else "nodes"
            lines.append(
                f"- {hidden_count} {label} without GPU support hidden ([u] to show)"
            )
        return lines

    def _apply_default_expansion(self, raw_stats_by_client, last_update_time):
        """Expand every node that actually reports GPUs, once data arrives.

        Without this the local machine stays expanded even when it has no
        NVIDIA GPU (macOS, CPU-only hosts), pushing real nodes off screen.
        Nodes without GPUs collapse instead, keeping the opening view full
        of live tables rather than "no data" panels.
        """
        if last_update_time is None or self._default_expansion_applied:
            return
        raw = raw_stats_by_client
        self._default_expansion_applied = all(
            idx in raw and not raw[idx][1].get("loading")
            for idx in range(len(self.pool))
        )
        if self._expansion_touched or not self.hide_unsupported:
            return
        self.expanded_servers = {
            idx
            for idx in range(len(self.pool))
            if idx not in raw or not raw[idx][0].empty or raw[idx][1].get("loading")
        }

    @staticmethod
    def _screen_line_count(elements):
        """Count the terminal rows a list of frame elements occupies."""
        return sum(len(str(element).split("\n")) for element in elements)

    # Also matches the ``ESC ( B`` charset reset blessed's term.normal
    # emits; missing it made every styled selected row measure two
    # columns wide of its visible width.
    _ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\([0-9A-Za-z]")

    def _build_server_blocks(self, server_rows, user_memory_by_client, width):
        """Render each expanded server's block at its natural height.

        Nothing is dropped here to make room for another server: the
        caller scrolls its viewport instead. That keeps every GPU a node
        reports on screen, and means what one server shows never depends
        on how many others are expanded or on where the cursor sits.
        `width` is the content width the blocks are drawn into (inside the
        panel border when there is one). Returns {idx: [block lines]} for
        the expanded servers.
        """
        blocks = {}
        for idx, _selected, is_expanded, _header, _summary, stats_info in server_rows:
            if not is_expanded:
                continue
            block = list(str(stats_info).splitlines())
            users = user_memory_by_client.get(idx) or {}
            if users:
                user_line = fit_display(
                    f"Users: {self._format_user_memory_totals(users)}", width, ellipsis="..."
                )
                block.append(colored(user_line, "dark_grey"))
            block.append("")
            blocks[idx] = block
        return blocks

    def _trim_server_block(self, lines, keep, width):
        """Shorten one expanded block to `keep` lines by dropping GPU rows.

        The table keeps its frame: everything up to the header separator
        stays, as many data rows as fit follow, and the original bottom
        border closes it. The marker naming the dropped lines is fitted to
        `width`. Returns None when there is no table to shrink or `keep`
        cannot hold the frame, meaning the caller should collapse the
        server instead.
        """
        plain = [self._ANSI_ESCAPE_RE.sub("", line) for line in lines]
        top_index = next(
            (index for index, text in enumerate(plain) if text.startswith("╭")),
            None,
        )
        if top_index is None:
            return None
        separator_index = next(
            (
                index
                for index in range(top_index, len(plain))
                if plain[index].startswith("├")
            ),
            None,
        )
        if separator_index is None:
            return None
        bottom_index = next(
            (
                index
                for index in range(separator_index + 1, len(plain))
                if plain[index].startswith("╰")
            ),
            None,
        )
        if bottom_index is None:
            return None

        head = lines[: separator_index + 1]
        bottom = lines[bottom_index]
        marker_text = fit(
            f"  … {len(lines) - keep} line(s) hidden — enlarge the terminal or press [c]",
            width,
        )
        marker = colored(marker_text, "dark_grey")
        # A frame with no data rows left says nothing a collapsed header
        # cannot, so one row is the minimum an expansion keeps.
        overhead = len(head) + 3  # bottom border, marker, one data row
        if keep < overhead:
            return None
        rows = lines[separator_index + 1 : bottom_index]
        kept_rows = keep - overhead + 1
        return head + rows[:kept_rows] + [bottom, marker]

    def _selected_unified_row(self, raw_stats_by_client):
        """Return the currently selected GPU row across filtering and sorting."""
        table = self._sort_unified_gpu_table(
            self._filter_unified_gpu_table(
                self._build_unified_gpu_table(raw_stats_by_client)
            )
        )
        if table.empty:
            return None
        selected = self._sync_selected_gpu(table)
        self._unified_gpu_count = len(table)
        return table.iloc[selected]

    def _format_tui_help_lines(self, width, height):
        """Render a compact, context-sensitive keyboard and mouse reference."""
        width = max(20, int(width))
        height = max(4, int(height))
        inner_width = max(1, width - 4)
        unified = self.display_mode == self.DISPLAY_MODE_UNIFIED
        pane = self.unified_active_pane

        if not unified:
            context = "PER-NODE"
            entries = [
                ("section", "NAVIGATION", ""),
                ("key", "j/k · ↑/↓", "Select a node"),
                ("key", "Enter/Space", "Expand or collapse node details"),
                ("key", "a / c", "Expand all / collapse all"),
                ("section", "VIEWS", ""),
                ("key", "v", "Open the unified GPU view"),
                ("key", "C", "Switch the colour theme"),
                ("key", "?", "Close this help"),
                ("key", "q", "Close help; q again quits"),
            ]
        elif pane == "process" and self._process_panel_visible():
            context = "UNIFIED · PROCESS"
            mode = self._get_unified_process_sort_mode()
            arrow = "▼" if self.unified_process_sort_descending else "▲"
            state = (
                f"sort {self.UNIFIED_PROCESS_SORT_LABELS[mode]}{arrow}"
                f" · filter /{self.unified_process_filter or '-'}"
                f" · rows {self._unified_process_visible_rows or 1}"
            )
            entries = [
                ("state", "Current", state),
                ("section", "NAVIGATION", ""),
                ("key", "j/k · ↑/↓", "Select a process"),
                ("key", "PgUp/PgDn", "Move by five processes"),
                ("key", "← / h", "Return to GPU/node selection"),
                ("key", "Tab", "Switch the active pane"),
                ("key", "p", "Show or hide the process pane"),
                ("section", "FIND AND ORDER", ""),
                ("key", "/", "Edit a live PID/user/command filter"),
                ("key", "Enter/Esc", "Apply / clear the filter"),
                ("key", "o / F6", "Cycle the process sort field"),
                ("key", "O", "Reverse the current sort"),
                ("key", "Mouse header", "Sort or reverse that column"),
                ("key", "+ / -", "Show more / fewer process rows"),
                ("section", "DETAILS AND ACTIONS", ""),
                ("key", "[ / ]", "Page through the full command"),
                ("key", "t", "Toggle GPU and process history"),
                ("key", "C", "Switch the colour theme"),
                ("key", "i / T / K", "Arm INT / TERM / KILL"),
                ("key", "? / q", "Close this help"),
            ]
        else:
            context = "UNIFIED · GPU/NODE"
            entries = [
                ("section", "NAVIGATION", ""),
                ("key", "j/k · ↑/↓", "Select a GPU"),
                ("key", "PgUp/PgDn", "Move by a GPU page"),
                ("key", "Enter/→/l", "Enter the selected GPU processes"),
                ("key", "Tab", "Switch to a visible process pane"),
                ("key", "p", "Show or hide the process pane"),
                ("section", "VIEW OPTIONS", ""),
                ("key", "d", "Switch Detailed / Single-line"),
                ("key", "s", "Cycle GPU sorting"),
                ("key", "f", "Cycle GPU filters"),
                ("key", "g", "Toggle per-node grouping"),
                ("key", "u", "Show or hide unsupported nodes"),
                ("key", "t", "Toggle GPU history"),
                ("key", "C", "Switch the colour theme"),
                ("key", "v", "Return to the per-node view"),
                ("key", "? / q", "Close this help"),
            ]

        entries.insert(0, ("key", "Mouse / m", "Click controls; More opens all hidden actions"))

        title = _clip(f"─ Help · {context} · ?/Esc/q close ", width - 2, "…")
        lines = [
            colored("╭", "dark_grey")
            + colored(title, "cyan", attrs=["bold"])
            + colored(
                "─" * max(0, width - 2 - display_width(title)) + "╮",
                "dark_grey",
            )
        ]

        max_entries = max(1, height - 2)
        clipped = len(entries) > max_entries
        if clipped:
            entries = entries[: max(1, max_entries - 1)]
            entries.append(("state", "More", "Enlarge the terminal for all keys"))

        key_width = min(16, max(7, inner_width // 3))
        description_width = max(0, inner_width - key_width - 1)
        for kind, key, description in entries:
            if kind == "section":
                plain = _clip(f"── {key} ", inner_width, "…")
                plain += "─" * max(0, inner_width - display_width(plain))
                styled = colored(plain, "dark_grey")
            else:
                key_text = pad_display(_clip(key, key_width, "…"), key_width)
                description_text = pad_display(
                    _clip(description, description_width, "…"),
                    description_width,
                )
                plain = f"{key_text} {description_text}"
                styled = colored(
                    key_text,
                    "cyan" if kind == "key" else "yellow",
                    attrs=["bold"] if kind == "key" else None,
                ) + " " + colored(description_text, "light_grey")
            lines.append(
                colored("│ ", "dark_grey")
                + styled
                + colored(" │", "dark_grey")
            )

        lines.append(colored("╰" + "─" * (width - 2) + "╯", "dark_grey"))
        return lines

    def _render_unified_gpu_lines(
        self, raw_stats_by_client, last_update_time, *, height_budget=None
    ):
        """Render the unified TUI body from cached per-node GPU data."""
        self._body_click_targets = {}
        self._body_click_regions = []
        if last_update_time is None and not raw_stats_by_client:
            return ["Unified GPU table"] + (["Loading GPU data..."] if self.debug else [])

        source_table = self._build_unified_gpu_table(raw_stats_by_client)
        if source_table.empty:
            node_count = 0
        else:
            node_count = len(
                source_table[["Node", "Hostname"]].drop_duplicates()
            )
        node_total = len(self._compute_node_indices(raw_stats_by_client))
        detailed = self.unified_detailed
        terminal_size = _terminal_size()
        terminal_width = terminal_size.columns
        terminal_height = terminal_size.lines
        filter_mode = self._get_unified_filter_mode()
        errors = self._get_unified_node_status_lines(raw_stats_by_client, errors_only=True)
        if filter_mode == "errors":
            # Errors have their own scrollable page; a long process panel
            # must never hide the machines that stopped responding.
            budget = max(3, height_budget if height_budget is not None else terminal_height - 4)
            page_size = max(1, budget - 2)
            selected = max(0, min(self.unified_selected_gpu, len(errors) - 1))
            start = selected // page_size * page_size
            self.unified_active_pane = "gpu"
            self.unified_selected_gpu = selected
            self.unified_selected_gpu_key = None
            self._unified_gpu_count = len(errors)
            self._unified_page_size = page_size
            self._unified_page_start = start
            self._body_click_targets = {
                row + 2: ("error", start + row)
                for row in range(len(errors[start:start + page_size]))
            }
            title = (
                f"Node errors {len(errors)} · GPUs 0/{len(source_table)} · "
                f"Rows {start + 1 if errors else 0}-{min(start + page_size, len(errors))}"
                f"/{len(errors)} · j/k / PgUp/PgDn"
            )
            return [
                fit_display(title, terminal_width),
                "Error filter: GPU rows hidden",
                *[
                    colored(fit_display(self._ANSI_ESCAPE_RE.sub('', line), terminal_width), "red")
                    for line in (errors[start:start + page_size] or ["No node errors"])
                ],
            ]
        show_process_panel = self._process_panel_visible()
        compact_history = (
            show_process_panel
            and terminal_height < 36
            and self.unified_show_trends
        )
        filtered_table = self._filter_unified_gpu_table(source_table)
        sorted_table = self._sort_unified_gpu_table(filtered_table)
        table, selected_row, page_status = self._paginate_unified_gpu_table(
            sorted_table,
            detailed=detailed,
        )
        title = "Unified GPU table (Detailed)" if detailed else "Unified GPU table"
        row_line_map = {}
        header_regions = []
        if table.empty and not source_table.empty:
            filter_label = self.UNIFIED_FILTER_LABELS[filter_mode]
            rendered_table = f"No GPUs match the {filter_label} filter"
        elif detailed:
            rendered_table = self._format_unified_detailed_table(
                table,
                selected_row=selected_row,
                row_line_map=row_line_map,
            )
        else:
            # Detailed cards already repeat node identity, so node bands only
            # spend scarce vertical space there; only this table gets them.
            section_headers = self._unified_section_headers(table, sorted_table)
            display_table = table.copy()
            if not display_table.empty:
                display_table.insert(
                    0,
                    "sel",
                    [
                        "❯" if index == selected_row else " "
                        for index in range(len(display_table))
                    ],
                )
                if section_headers:
                    # Node identity already lives in the band above each block.
                    display_table = display_table.drop(
                        columns=["Node", "Hostname"],
                        errors="ignore",
                    )
            rendered_table = self._format_fixed_width_table(
                display_table,
                column_labels=self.UNIFIED_TABLE_LABELS,
                fixed_width_columns=("sel",),
                section_headers=section_headers,
                row_line_map=row_line_map,
                header_regions=header_regions,
            )
        selected_gpu_row = (
            table.iloc[selected_row]
            if selected_row is not None and not table.empty
            else None
        )
        page_suffix = f" · {page_status}" if page_status else ""
        gpu_count_display = str(len(source_table))
        if filter_mode != "all":
            gpu_count_display = f"{len(filtered_table)}/{len(source_table)}"
        gpu_pane_focused = (
            self.unified_active_pane == "gpu"
        )
        focus_prefix = (
            "Focus GPU/node"
            if gpu_pane_focused
            else "Focus process"
        )
        if detailed and terminal_width < 60:
            focus_prefix = "F:GPU" if gpu_pane_focused else "F:PROC"
            compact_page = page_status.removeprefix("Rows ")
            title_line = (
                f"{focus_prefix} · D · G:{gpu_count_display} · "
                f"N:{node_count}/{node_total}"
                + (f" · {compact_page}" if compact_page else "")
            )
        elif detailed and terminal_width < 100:
            title_line = (
                f"{focus_prefix} · Detailed · GPUs {gpu_count_display} · "
                f"Nodes {node_count}/{node_total}{page_suffix}"
            )
        else:
            title_with_focus = (
                f"{focus_prefix} · {title}"
                if detailed
                else title
            )
            title_line = (
                f"{title_with_focus} · GPUs {gpu_count_display} · "
                f"Nodes with GPU {node_count}/{node_total}{page_suffix}"
            )
        if display_width(title_line) > terminal_width:
            title_line = fit_display(title_line, terminal_width, ellipsis="..")
        if detailed and title_line.startswith(focus_prefix):
            title_line = (
                colored(focus_prefix, "cyan", attrs=["bold"])
                + title_line[len(focus_prefix):]
            )
        lines = [title_line]
        if errors:
            # Keep the count and navigation hint ahead of the truncated text.
            banner = fit_display(
                f"! {len(errors)} node error(s) [click: errors] · "
                + self._ANSI_ESCAPE_RE.sub("", errors[0]).lstrip("! "),
                terminal_width,
            )
            lines.insert(0, colored(banner, "red", attrs=["bold"]))
            self._body_click_regions.append(
                (0, 0, terminal_width - 1, ("error_filter", None))
            )
        if not compact_history:
            lines.append(self._format_unified_capacity_summary(source_table))
        # The filter is restored from config.yml, so say it out loud whenever it
        # is the reason a GPU is missing from the table.
        hidden_by_filter = len(source_table) - len(filtered_table)
        if filter_mode != "all" and hidden_by_filter > 0:
            filter_label = self.UNIFIED_FILTER_LABELS[filter_mode]
            lines.append(
                colored(
                    f"Filter {filter_label}: {hidden_by_filter} of "
                    f"{len(source_table)} GPUs hidden ([f] to change)",
                    "yellow",
                    attrs=["bold"],
                )
            )
        # Remember where each GPU row landed so a click can select it.
        table_offset = self._screen_line_count(lines)
        self._body_click_targets = {
            table_offset + line_index: ("gpu", self._unified_page_start + row_index)
            for line_index, row_index in row_line_map.items()
        }
        self._body_click_regions.extend(
            (table_offset + line, start, end, ("gpu_sort", mode))
            for line, start, end, mode in header_regions
        )
        lines.append(rendered_table)

        if show_process_panel:
            process_line_map = {}
            command_line_map = {}
            action_regions = []
            process_offset = self._screen_line_count(lines)
            frame_reserve = 3 if self._compact_detailed_layout(terminal_height) else 4
            selected_processes = self._get_sorted_unified_processes(
                raw_stats_by_client,
                selected_gpu_row,
            )
            future_trend_lines = (
                4
                if self.unified_show_trends
                and (not compact_history or not selected_processes)
                else 0
            )
            process_height_budget = max(
                8,
                (height_budget if height_budget is not None else terminal_height - frame_reserve)
                - process_offset
                - future_trend_lines,
            )
            process_panel = self._format_unified_process_details(
                raw_stats_by_client,
                selected_gpu_row,
                height_budget=process_height_budget,
                process_line_map=process_line_map,
                command_line_map=command_line_map,
                action_regions=action_regions,
            )
            self._body_click_targets.update(
                {
                    process_offset + line_index: ("process", process_index)
                    for line_index, process_index in process_line_map.items()
                }
            )
            self._body_click_targets.update(
                {
                    process_offset + line_index: ("command", process_index)
                    for line_index, process_index in command_line_map.items()
                }
            )
            self._body_click_regions.extend(
                (
                    process_offset + line_index,
                    start_column,
                    end_column,
                    target,
                )
                for line_index, start_column, end_column, target in action_regions
            )
            lines.append(process_panel)
        else:
            self._unified_process_count = 0
            self._unified_process_visible_rows = 0
            self._unified_command_line_count = 0
            self._unified_command_page_size = 0
            self.unified_command_scroll = 0
            if self.unified_active_pane == "process":
                self.unified_active_pane = "gpu"

        if self.unified_show_trends and (
            not compact_history or self._unified_process_count == 0
        ):
            lines.append(
                self._format_unified_gpu_trends(selected_gpu_row)
            )

        # Detailed mode reserves the lower half for process interaction, so
        # it omits the routine "unsupported node hidden" reminder that is
        # already reflected in the title. Real errors are always visible in
        # the top banner and its error page instead of here.
        if not detailed:
            status_lines = [
                line
                for line in self._get_unified_node_status_lines(raw_stats_by_client)
                if line not in errors
            ]
            if status_lines:
                lines.append("Node status:")
                lines.extend(status_lines)
        return lines

    def _compact_detailed_layout(self, terminal_height):
        """Whether detailed mode must save rows: a short window, or history on."""
        return self.unified_detailed and (
            terminal_height < 28
            or (terminal_height < 36 and self.unified_show_trends)
        )

    @staticmethod
    def _control_style(text, style):
        return colored(text, "dark_grey" if style == "bright_black" else style)

    def _monitor_controls(self):
        """Visible controls and overflow share the same action definitions."""
        if self.tui_help_visible:
            return [("Close help", "key", "?", "cyan")]
        unified = self.display_mode == self.DISPLAY_MODE_UNIFIED
        controls = [
            (("● " if unified else "") + "GPUs", "view", self.DISPLAY_MODE_UNIFIED, "cyan"),
            (("● " if not unified else "") + "Nodes", "view", self.DISPLAY_MODE_NODES, "cyan"),
        ]
        pending = self._pending_process_signal
        if pending and pending.get("expires_at", 0) > time.monotonic():
            controls[:0] = [
                (f"Confirm {pending['signal']}:{pending['pid']}", "key", "\n", "yellow"),
                ("Cancel", "key", "\x1b", "cyan"),
            ]
        if unified:
            process_focused = self._process_panel_visible() and self.unified_active_pane == "process"
            if self.unified_process_filter_editing:
                controls += [
                    ("Apply search", "key", "\n", "cyan"),
                    ("Clear search", "clear_search", None, "yellow"),
                ]
            if process_focused:
                controls += [
                    ("/ Find", "key", "/", "cyan"),
                    ("Sort processes ▾", "menu", "process_sort", "cyan"),
                    ("← GPUs", "key", "h", "cyan"),
                ]
            controls += [
                (f"Filter:{self.UNIFIED_FILTER_LABELS[self._get_unified_filter_mode()]} ▾", "menu", "filter", "cyan"),
                (f"Sort:{self._get_unified_sort_mode()} ▾", "menu", "sort", "cyan"),
                ("Hide processes" if self._process_panel_visible() else "Processes", "key", "p", "cyan"),
                ("Single-line" if self.unified_detailed else "Detailed", "key", "d", "cyan"),
                ("Find process /", "key", "/", "cyan"),
                ("History " + ("on" if self.unified_show_trends else "off"), "key", "t", "cyan"),
                ("Flat view" if self.unified_group_by_node else "Group by node", "key", "g", "cyan"),
                ("Show unsupported" if self.hide_unsupported else "Hide unsupported", "key", "u", "cyan"),
                ("Previous GPUs", "gpu_page", -1, "cyan"),
                ("Next GPUs", "gpu_page", 1, "cyan"),
            ]
            if self._process_panel_visible():
                if not process_focused:
                    controls.append(("Sort processes ▾", "menu", "process_sort", "cyan"))
                controls += [
                    ("More process rows", "key", "+", "cyan"),
                    ("Fewer process rows", "key", "-", "cyan"),
                    ("Previous command page", "key", "[", "cyan"),
                    ("Next command page", "key", "]", "cyan"),
                ]
        else:
            controls += [
                ("Expand all", "key", "a", "cyan"),
                ("Collapse all", "key", "c", "cyan"),
            ]
        controls += [
            (f"Theme:{self.theme}", "key", "C", "cyan"),
            ("? Help", "key", "?", "cyan"),
            ("q Quit", "key", "q", "cyan"),
        ]
        return controls

    def _open_tui_menu(self, name):
        self._tui_hover.clear()
        selected = 0
        if name == "filter":
            title = "Filter GPUs"
            entries = [(self.UNIFIED_FILTER_LABELS[mode], ("gpu_filter", mode))
                       for mode in self.UNIFIED_FILTER_MODES]
            selected = self.UNIFIED_FILTER_MODES.index(self._get_unified_filter_mode())
        elif name == "sort":
            title = "Sort GPUs"
            entries = [(self.UNIFIED_SORT_LABELS[mode], ("gpu_sort", mode)) for mode in self.UNIFIED_SORT_MODES]
            selected = self.UNIFIED_SORT_MODES.index(self._get_unified_sort_mode())
        elif name == "process_sort":
            title = "Sort processes · choose current field to reverse"
            entries = [(self.UNIFIED_PROCESS_SORT_LABELS[mode], ("process_sort", mode))
                       for mode in self.UNIFIED_PROCESS_SORT_MODES]
            selected = self.UNIFIED_PROCESS_SORT_MODES.index(self._get_unified_process_sort_mode())
        else:
            title = "More actions"
            controls = self._toolbar_overflow or self._monitor_controls()
            entries = [(label, (action, value)) for label, action, value, _ in controls]
        self._tui_menu = ActionMenu(title, entries, selected=selected)
        self.unified_process_filter_editing = False
        self._pending_process_signal = None
        self._request_ui_refresh()
        return True

    def _finish_tui_menu_input(self, result):
        closed, target = result
        if closed:
            self._tui_menu = None
        self._request_ui_refresh()
        if target:
            return self._activate_tui_control(*target)
        return True

    def _activate_tui_control(self, action, value):
        # A click leaves text entry before dispatch, so Quit is never typed
        # into the process filter and keyboard focus matches the new view.
        self.unified_process_filter_editing = False
        if action == "menu":
            return self._open_tui_menu(value)
        if action == "view":
            if self.display_mode != value:
                self._toggle_display_mode()
            self._request_ui_refresh()
            return True
        if action == "key":
            self._handle_keypress(value)
            self._request_ui_refresh()
            return True
        if action == "gpu_sort":
            self._set_unified_sort_mode(value)
        elif action == "gpu_filter":
            self._set_unified_filter_mode(value)
        elif action == "gpu_page":
            self.unified_active_pane = "gpu"
            self._move_unified_selection(value * max(1, self._unified_page_size))
        elif action == "clear_search":
            self._clear_unified_process_filter()
        elif action == "process_sort":
            self._enter_unified_process_pane()
            self._set_unified_process_sort(value, toggle=True)
        self._request_ui_refresh()
        return True

    def _write_tui_lines(self, output_lines):
        """Paint one TUI frame, rewriting only the rows that changed.

        Whole-screen clears are what made refreshes flicker, especially
        over SSH; diffing against the previous frame keeps a refresh
        invisible when nothing moved.
        """
        rows = []
        for line in output_lines:
            rows.extend(str(line).split("\n"))
        width, height = self.term.width or 80, self.term.height or 24
        menu = self._tui_menu
        if menu is not None:
            rows = menu.render(
                width, max(1, height - 1),
                self._control_style,
                hover_enabled=False,
            )
            # The menu owns input; no stale targets survive underneath it.
            self._click_regions = []
            self._click_targets = {}
        else:
            self._click_regions = [region for region in self._click_regions if region[0] < height - 1]
            self._click_targets = {row: target for row, target in self._click_targets.items() if row < height - 1}
        # Pointer feedback reuses this layout. Rebuilding it on every motion
        # used to reformat every node's dataframes before changing two rows.
        self._tui_base_rows = rows[:max(0, height - 1)]
        self._tui_frame_size = (width, height)
        self._tui_frame_menu = menu
        self._paint_tui_hover()

    def _paint_tui_hover(self):
        """Repaint pointer feedback without collecting or formatting data."""
        rows = self._tui_base_rows
        menu = self._tui_menu
        size = (self.term.width or 80, self.term.height or 24)
        if (rows is None or size != self._tui_frame_size
                or menu is not self._tui_frame_menu):
            return False
        hover = menu.hover if menu is not None else self._tui_hover
        rows = hover.paint(
            rows,
            menu.regions if menu is not None else self._click_regions,
            {} if menu is not None else self._click_targets,
            size[0], enabled=self.mouse_enabled and bool(self.term.does_styling),
            row_kinds=self._HOVER_ROW_KINDS,
        )
        if self._tui_diff_screen is None:
            self._tui_diff_screen = DiffScreen(self.term)
        self._tui_diff_screen.paint(rows)
        self._tui_hover_dirty = False
        return True

    def _wrap_panel_border(self, lines, content_width):
        """Enclose already-built content lines in a rounded panel border.

        Every line is assumed to already fit `content_width` (the render
        loop narrowed itself to that width up front); this only pads and
        sides them. A line equal to `_PANEL_SEPARATOR` becomes a full-width
        tee rule instead, so an inter-server divider connects cleanly to
        the frame's own corners rather than nesting inside a second pair
        of side bars.
        """
        frame_width = content_width + self._PANEL_BORDER_MARGIN
        side_l = colored("│ ", "dark_grey")
        side_r = colored(" │", "dark_grey")
        wrapped = [colored(frame_top(frame_width), "dark_grey")]
        for line in lines:
            if line == self._PANEL_SEPARATOR:
                wrapped.append(colored(frame_separator(frame_width), "dark_grey"))
                continue
            visible = display_width(self._ANSI_ESCAPE_RE.sub("", line))
            pad = max(0, content_width - visible)
            wrapped.append(side_l + line + (" " * pad) + side_r)
        wrapped.append(colored(frame_bottom(frame_width), "dark_grey"))
        return wrapped

    def _format_error_panel(
        self,
        message: str,
        error_type: Optional[str] = None,
        *,
        outer_margin: int = 0,
    ) -> str:
        width = _terminal_size().columns
        if outer_margin:
            width = max(20, width - outer_margin)

        if error_type == "auth" and str(message).strip() == "Password incorrect":
            title = "Password incorrect"
        elif error_type == "auth":
            title = "Authentication failed"
        else:
            title = "Error"
        def centered(text):
            text = fit(f" {text} ", width)
            remaining = max(0, width - display_width(text))
            left = remaining // 2
            return " " * left + text + " " * (remaining - left)

        return "\n".join(
            colored(line, "white", "on_red", attrs=["bold"])
            for line in (centered(title), centered(message), " " * width)
        )
    
    @staticmethod
    def _format_section_band(label, detail, width):
        """Render a node band: bold label, plain stats, dim rule to full width.

        Deliberately background-free; a full-width color block reads as an alert
        rather than a section divider.
        """
        if width <= 0:
            return ""
        label = str(label)
        detail = str(detail)
        separator = " · "

        if display_width(label) >= width:
            label = _clip(label, width)
            separator = ""
            detail = ""
        elif (
            display_width(label) + display_width(separator) + display_width(detail)
            > width
        ):
            room = width - display_width(label) - display_width(separator)
            if room <= 3:
                separator = ""
                detail = ""
            else:
                detail = fit_display(detail, room, ellipsis="..")

        pieces = [colored(label, "cyan", attrs=["bold"])]
        if separator:
            pieces.append(colored(separator, attrs=["dark"]))
        if detail:
            pieces.append(detail)
        tail = (
            width
            - display_width(label)
            - display_width(separator)
            - display_width(detail)
        )
        if tail == 1:
            pieces.append(" ")
        elif tail > 1:
            pieces.append(" " + colored("─" * (tail - 1), attrs=["dark"]))
        return "".join(pieces)

    def _format_fixed_width_table(
        self,
        df,
        column_labels=None,
        *,
        fixed_width_columns=(),
        section_headers=None,
        row_line_map=None,
        header_regions=None,
        outer_margin: int = 0,
    ):
        """Format a GPU table inside a rounded frame, fitted to the terminal.

        `fixed_width_columns` never absorb spare width, and
        `section_headers` (row index -> banner text) adds node bands for
        grouped views. `row_line_map` is filled in with line offset -> data
        row index so callers can turn a mouse click into a row. `outer_margin`
        shrinks the table to leave room for a caller-drawn frame around it
        (e.g. the per-node view's panel border) instead of using the
        terminal's full width.
        """
        if df.empty:
            return "No GPU data available"

        column_labels = column_labels or {}
        section_headers = section_headers or {}
        left_align_columns = {"Node", "Hostname"}
        fixed_width_columns = set(fixed_width_columns or ())

        # Get terminal width
        terminal_width = _terminal_size(columns=120).columns
        if outer_margin:
            terminal_width = max(20, terminal_width - outer_margin)
        
        # Define minimum width for each column
        min_widths = {
            'Node': 12,
            'Hostname': 15,
            'GPU': 4,
            'name': 12,
            'temp': 8,
            'fan': 8,
            'util': 8,
            'mem_util': 8,
            'link': 9,
            'rx': 10,
            'tx': 10,
            'power': 18,
            'memory[used/total]': 16,
            'processes': 20  # Add width for processes column
        }

        all_columns = visible_columns(df.columns)
        # "sel" is the selection marker column; it only exists in the unified table,
        # which may drop Node/Hostname when rows are grouped into node blocks.
        is_unified_table = any(
            column in all_columns for column in ("Node", "Hostname", "sel")
        )
        if is_unified_table:
            min_widths.update(
                {
                    "sel": 1,
                    "Node": 10,
                    "Hostname": 13,
                    "GPU": 3,
                    "name": 10,
                    "util": 7,
                    "memory[used/total]": 13,
                    "temp": 6,
                    "power": 12,
                    "link": 7,
                    "processes": 14,
                }
            )

        # Calculate widths based on terminal space
        separator_width = 3
        outer_width = 4  # "│ " + " │"
        max_content_width = max(0, terminal_width - outer_width)

        # Column importance: larger => dropped earlier when narrow
        importance = {
            "sel": -1,
            "Node": 0,
            "Hostname": 1,
            "GPU": 2,
            "util": 3,
            "memory[used/total]": 4,
            "name": 5,
            "processes": 6,
            "mem_util": 7,
            "temp": 8,
            "power": 9,
            "rx": 10,
            "tx": 11,
            # RX/TX outlive the link label: their bars already show the load,
            # and the label is worth more than fan speed on a narrow terminal.
            "link": 12,
            "fan": 13,
        }
        if is_unified_table:
            must_keep_names = (
                "sel",
                "Node",
                "Hostname",
                "GPU",
                "name",
                "util",
                "memory[used/total]",
            )
        else:
            must_keep_names = ("GPU", "util", "mem_util")
        must_keep = [column for column in must_keep_names if column in all_columns]

        min_widths_for_cols = {}
        for col in all_columns:
            min_width = min_widths.get(col, 12)
            label = column_labels.get(col, col)
            min_widths_for_cols[col] = max(min_width, display_width(str(label)))

        def min_table_width(cols) -> int:
            if not cols:
                return 0
            return sum(min_widths_for_cols[c] for c in cols) + separator_width * (len(cols) - 1)

        # Drop least important columns until it fits the terminal width
        selected_columns = list(all_columns)
        while selected_columns and min_table_width(selected_columns) > max_content_width:
            droppable = [c for c in selected_columns if c not in must_keep]
            if not droppable:
                droppable = list(selected_columns)
            drop_col = max(
                droppable,
                key=lambda c: (importance.get(c, 100), selected_columns.index(c)),
            )
            selected_columns.remove(drop_col)

        if not selected_columns:
            selected_columns = all_columns[:1]

        df_display = df[selected_columns]

        # Compute content width for each column
        content_widths = {}
        for col in selected_columns:
            max_len = display_width(str(col))
            for value in df_display[col]:
                value_len = display_width(str(value))
                if value_len > max_len:
                    max_len = value_len
            content_widths[col] = max_len

        desired_widths = {}
        min_widths_selected = {}
        for col in selected_columns:
            min_width = min_widths_for_cols.get(col, 12)
            min_widths_selected[col] = min_width
            desired_widths[col] = max(min_width, content_widths.get(col, min_width))

        total_separator_width = separator_width * (len(selected_columns) - 1)
        available_width = max_content_width - total_separator_width  # available for column contents
        if available_width < 1:
            available_width = 1

        min_total = sum(min_widths_selected.values())
        desired_total = sum(desired_widths.values())

        fill_priority = [
            "processes",
            "name",
            "Hostname",
            "Node",
            "power",
            "memory[used/total]",
            "rx",
            "tx",
            "temp",
            "fan",
            "util",
            "mem_util",
            "GPU",
        ]

        if available_width >= desired_total:
            column_widths = dict(desired_widths)
            extra = available_width - desired_total
            if extra > 0 and not self.compact:
                spread_columns = [
                    c for c in selected_columns if c not in fixed_width_columns
                ]
                if not spread_columns:
                    spread_columns = list(selected_columns)
                share, remainder = divmod(extra, len(spread_columns))
                if share:
                    for col in spread_columns:
                        column_widths[col] += share
                if remainder:
                    offset = terminal_width % len(spread_columns)
                    for i in range(remainder):
                        column_widths[spread_columns[(offset + i) % len(spread_columns)]] += 1
        elif available_width >= min_total:
            column_widths = dict(min_widths_selected)
            extra = available_width - min_total
            for col in (c for c in fill_priority if c in selected_columns):
                if extra <= 0:
                    break
                need = desired_widths[col] - column_widths[col]
                if need > 0:
                    add = min(extra, need)
                    column_widths[col] += add
                    extra -= add
        else:
            # Extremely narrow terminal: shrink columns below minimums, but keep at least 1 char
            column_widths = dict(min_widths_selected)
            excess = sum(column_widths.values()) - available_width
            if excess > 0:
                shrink_order = sorted(
                    selected_columns,
                    key=lambda c: (importance.get(c, 100), selected_columns.index(c)),
                    reverse=True,
                )
                for col in shrink_order:
                    if excess <= 0:
                        break
                    reducible = column_widths[col] - 1
                    if reducible <= 0:
                        continue
                    reduce_by = min(excess, reducible)
                    column_widths[col] -= reduce_by
                    excess -= reduce_by

        def truncate_text(text: str, width: int, tail_preserve: bool = False) -> str:
            text = "" if text is None else str(text)
            if tail_preserve and width > 2 and display_width(text) > width:
                budget = width - 2
                kept = []
                used = 0
                for char in reversed(text):
                    char_width = display_width(char)
                    if used + char_width > budget:
                        break
                    kept.append(char)
                    used += char_width
                return ".." + "".join(reversed(kept))
            return _clip(text, width)

        def align_text(text: str, width: int, alignment: str = "center") -> str:
            remaining = max(0, width - display_width(text))
            if alignment == "left":
                return text + " " * remaining
            if alignment == "right":
                return " " * remaining + text
            left = remaining // 2
            return " " * left + text + " " * (remaining - left)

        def parse_ratio_percent(value: str) -> Optional[float]:
            if "/" not in str(value):
                return None
            used_str, total_str = str(value).split("/", 1)
            used_val = parse_leading_float(used_str)
            total_val = parse_leading_float(total_str)
            if used_val is None or total_val is None or total_val <= 0:
                return None
            return used_val / total_val * 100

        def mem_bar_bg(percent: Optional[float]) -> Optional[str]:
            if percent is None:
                return None
            if percent >= 60:
                return "on_red"
            if percent >= 10:
                return "on_yellow"
            if percent > 0:
                return "on_green"
            return None

        def link_bar_bg(percent: Optional[float]) -> Optional[str]:
            """Colour a PCIe direction against its own saturation thresholds."""
            color = get_pcie_load_color(percent)
            return f"on_{color}" if color else None

        def hidden_percent(row_index: int, column: str) -> Optional[float]:
            """Read a load percentage the collector attached to the row."""
            name = f"{HIDDEN_COLUMN_PREFIX}{column}"
            if name not in df.columns:
                return None
            try:
                value = float(df.iloc[row_index][name])
            except (TypeError, ValueError):
                return None
            # A GPU with an unknown link shares a float64 column with ones
            # that report theirs, so its missing load arrives as NaN; NaN
            # would clamp to a full bar and read as a saturated link.
            return None if pd.isna(value) else value

        def format_bar_cell(text: str, percent: Optional[float], width: int, bar_bg=None) -> str:
            if width <= 0:
                return ""
            text = truncate_text(text, width)

            chars = [" "] * width
            start = max(0, (width - display_width(text)) // 2)
            for i, ch in enumerate(text):
                pos = start + i
                if 0 <= pos < width:
                    chars[pos] = ch

            if percent is None:
                return "".join(chars)

            p = max(0.0, min(100.0, percent))
            fill_len = int(round((p / 100.0) * width))
            if p > 0 and fill_len == 0:
                fill_len = 1

            # Keep the unfilled portion on the terminal default background.
            # `termcolor`'s "on_grey" maps to ANSI 40 (black), which looks wrong
            # on light/gray terminal themes.
            fill_bg = (bar_bg or mem_bar_bg)(p)

            left = "".join(chars[:fill_len])
            right = "".join(chars[fill_len:])
            if fill_len > 0 and fill_bg:
                # Pair the bar background with a contrasting foreground; without a
                # foreground, the terminal's default (usually white/light) text
                # sits on a bright yellow/green bar and becomes unreadable. Only the
                # filled portion is recolored, so text over the unfilled region keeps
                # the terminal default color.
                bar_fg = {"on_red": "white", "on_yellow": "black", "on_green": "black"}.get(fill_bg)
                left = colored(left, bar_fg, on_color=fill_bg)
            return left + right

        # Format table header. Interior separators are the same hairlines
        # as the frame, so the table reads as one continuous surface
        # instead of ASCII art.
        cell_separator = colored(" │ ", "dark_grey")
        header_parts = []
        header_styled_parts = []
        header_column = 2
        for col in selected_columns:
            width = column_widths.get(col, 12)
            col_name = truncate_text(str(column_labels.get(col, col)), width)
            cell = align_text(col_name, width)
            header_parts.append(cell)
            header_styled_parts.append(colored(cell, "cyan", attrs=["bold"]))
            mode = {"Node": "node", "Hostname": "node", "GPU": "node", "util": "utilization"}.get(col)
            if header_regions is not None and mode:
                header_regions.append((1, header_column, header_column + width - 1, mode))
            header_column += width + 3
        header_plain = " │ ".join(header_parts)
        header = cell_separator.join(header_styled_parts)

        # Format data rows
        inner_width = display_width(header_plain)
        data_lines = []
        row_lines = {}
        for row_index, (_, row) in enumerate(df_display.iterrows()):
            band = section_headers.get(row_index)
            if band:
                data_lines.append(self._format_section_band(*band, inner_width))
            row_lines[len(data_lines)] = row_index
            row_parts = []
            row_util = str(row.get("util", "0"))
            row_util_color = get_utilization_color(row_util)
            for col in selected_columns:
                width = column_widths.get(col, 12)
                raw_value = row.get(col, "")
                value = truncate_text(
                    raw_value,
                    width,
                    tail_preserve=(col == "name" and not is_unified_table),
                )

                if col in left_align_columns:
                    row_parts.append(align_text(value, width, "left"))
                    continue

                if col == "GPU":
                    cell = align_text(value, width)
                    if row_util_color:
                        cell = colored(cell, row_util_color, attrs=["bold"])
                    row_parts.append(cell)
                    continue

                if col in {"util", "mem_util"} and value != "N/A":
                    percent = parse_leading_float(raw_value)
                    display_text = str(raw_value).replace(" ", "")
                    if percent is not None:
                        display_text = f"{int(round(percent))}%"
                    row_parts.append(format_bar_cell(display_text, percent, width))
                    continue

                if col == "memory[used/total]" and value != "N/A":
                    percent = parse_ratio_percent(raw_value)
                    row_parts.append(format_bar_cell(value, percent, width))
                    continue

                # RX/TX only mean something next to the link they run over, so
                # the cell fills up as the traffic approaches the PCIe ceiling.
                if col in {"rx", "tx"} and value != "N/A":
                    percent = hidden_percent(row_index, f"{col}_percent")
                    row_parts.append(
                        format_bar_cell(value, percent, width, bar_bg=link_bar_bg)
                    )
                    continue

                if col == "temp":
                    degrees = parse_leading_float(raw_value)
                    cell = align_text(value, width)
                    if degrees is not None:
                        if degrees >= 80:
                            cell = colored(cell, "red", attrs=["bold"])
                        elif degrees >= 65:
                            cell = colored(cell, "yellow")
                        else:
                            # A cool GPU needs no attention; keep it quiet.
                            cell = colored(cell, "dark_grey")
                    row_parts.append(cell)
                    continue

                # Absent readings recede to a dim dash instead of shouting
                # N/A across every unsupported column.
                if value in {"N/A", "-"}:
                    row_parts.append(colored(f"{'–':^{width}}", "dark_grey"))
                    continue

                # Center-align all other columns
                row_parts.append(align_text(value, width))

            data_lines.append(cell_separator.join(row_parts))

        # A rounded hairline frame instead of ASCII `+--+` art: the corners
        # flow into the rules and the whole border stays one dim grey, so
        # the frame reads as a hairline, not a cage.
        frame_width = inner_width + 4
        side_l = colored("│ ", "dark_grey")
        side_r = colored(" │", "dark_grey")
        result_lines = (
            [
                colored(frame_top(frame_width), "dark_grey"),
                side_l + header + side_r,
                colored(frame_separator(frame_width), "dark_grey"),
            ]
            + [side_l + line + side_r for line in data_lines]
            + [colored(frame_bottom(frame_width), "dark_grey")]
        )

        if row_line_map is not None:
            for line_index, row_index in row_lines.items():
                row_line_map[3 + line_index] = row_index

        return "\n".join(result_lines)
    
    def _get_server_summary_data(self, stats, system_info):
        """Compute summary data for formatting/alignment.

        `stats` is a processed node frame (see _collect_client_gpu_info), so
        VRAM comes from its `memory[used/total]` MiB column.
        """
        sys_stats = system_info.get("system_stats", {})
        system = {
            "cpu_cores": sys_stats.get("cpu_cores", 0),
            "cpu_percent": sys_stats.get("cpu_percent", 0.0),
            "mem_used_gb": sys_stats.get("mem_used_gb", 0.0),
            "mem_total_gb": sys_stats.get("mem_total_gb", 0.0),
            "swap_used_gb": sys_stats.get("swap_used_gb", 0.0),
            "swap_total_gb": sys_stats.get("swap_total_gb", 0.0),
            "data_source": system_info.get("data_source"),
        }

        if stats.empty:
            # Return system stats even when no GPU data
            return {
                "gpu_count": 0,
                "idle_count": 0,
                "avg_util": 0,
                "util_color": None,
                "mem_display": "N/A",
                "mem_percent": None,
                "no_gpu": True,
                **system,
            }

        gpu_count = len(stats)
        idle_count = 0
        total_util = 0
        total_mem_used_mib = 0
        total_mem_total_mib = 0

        for _, row in stats.iterrows():
            util_str = str(row.get("util", "0"))
            try:
                util_val = int(util_str.replace("%", "").strip())
                total_util += util_val
                if util_val < 5:
                    idle_count += 1
            except (ValueError, AttributeError):
                pass

            mem_col = row.get("memory[used/total]", "")
            if mem_col and "/" in str(mem_col):
                used_str, total_str = str(mem_col).split("/", 1)
                total_mem_used_mib += int("".join(filter(str.isdigit, used_str)) or 0)
                total_mem_total_mib += int("".join(filter(str.isdigit, total_str)) or 0)

        avg_util = total_util // gpu_count if gpu_count > 0 else 0

        if total_mem_total_mib > 0:
            if total_mem_total_mib >= 1024:
                mem_display = f"{round(total_mem_used_mib/1024)}GB/{round(total_mem_total_mib/1024)}GB"
            else:
                mem_display = f"{total_mem_used_mib}MB/{total_mem_total_mib}MB"
        else:
            mem_display = "N/A"

        util_color = self._threshold_color(avg_util, 50, 80, "green")

        return {
            "gpu_count": gpu_count,
            "idle_count": idle_count,
            "avg_util": avg_util,
            "util_color": util_color,
            "mem_display": mem_display,
            "mem_percent": (
                total_mem_used_mib / total_mem_total_mib * 100
                if total_mem_total_mib > 0
                else None
            ),
            **system,
        }

    @staticmethod
    def _server_icon_color(expand_icon, toggle_disabled):
        """Expand icon colour: green = open, red = blocked, dim = closed."""
        if toggle_disabled:
            return "red"
        if expand_icon == "▾":
            return "green"
        return "dark_grey"

    @staticmethod
    def _server_health_dot(summary_data):
        """One glanceable dot per server: is there a free GPU here or not?

        Green = at least one idle GPU, yellow = all GPUs busy, red = the
        node errored, hollow/dim = still loading or nothing to report.
        """
        status = summary_data.get("status")
        if status == "error":
            return ("●", "red")
        if status == "loading":
            return ("◌", "dark_grey")
        if summary_data.get("no_gpu"):
            return ("●", "dark_grey")
        if summary_data.get("idle_count", 0) > 0:
            return ("●", "green")
        return ("●", "yellow")

    @staticmethod
    def _mini_bar(percent, color, width=6):
        """A small inline load bar for the server summary row.

        The same smooth one-line style the tables use, shrunk to fit a
        header: filled part in the metric's own threshold colour, the
        rest a dim rule. Unknown readings render as an all-dim rule so
        columns still line up across servers.
        """
        try:
            value = max(0.0, min(100.0, float(percent)))
        except (TypeError, ValueError):
            return colored("─" * width, "dark_grey")
        filled = int(round(value / 100.0 * width))
        if value > 0 and filled == 0:
            filled = 1
        bar = ""
        if filled > 0:
            bar += colored("━" * filled, color or "green")
        if width - filled > 0:
            bar += colored("─" * (width - filled), "dark_grey")
        return bar

    def _format_server_summary(self, summary_data, widths=None, *, compact=False) -> str:
        """Format summary string; if widths provided, columns are aligned.

        `compact` renders the plain short form for narrow terminals, where
        the aligned form would push a server header past the terminal edge.
        """
        status = summary_data.get("status")
        if compact:
            if status == "loading":
                return colored(summary_data.get("message", "loading…"), "dark_grey")
            if status == "error":
                message = " ".join(str(summary_data.get("message", "Error")).split())
                return colored(f"err: {message or 'error'}", "red")
            if summary_data.get("no_gpu"):
                source = summary_data.get("data_source")
                return colored(
                    "unsupported" if source == "unsupported" else "no GPU",
                    "dark_grey",
                )
            return (
                f"{summary_data['gpu_count']}G · {summary_data['avg_util']}% · "
                f"{summary_data['mem_display']}"
            )
        if status == "loading":
            return colored(summary_data.get("message", "Loading..."), "dark_grey")
        if status == "error":
            return colored(summary_data.get("message", "Error"), "red", attrs=["bold"])

        gpu_count = summary_data["gpu_count"]
        idle_count = summary_data["idle_count"]
        avg_util = summary_data["avg_util"]
        util_color = summary_data["util_color"]
        mem_display = summary_data["mem_display"]
        no_gpu = summary_data.get("no_gpu", False)

        # System stats
        cpu_cores = summary_data.get("cpu_cores", 0)
        cpu_percent = summary_data.get("cpu_percent", 0.0)
        mem_used_gb = summary_data.get("mem_used_gb", 0.0)
        mem_total_gb = summary_data.get("mem_total_gb", 0.0)
        swap_used_gb = summary_data.get("swap_used_gb", 0.0)
        swap_total_gb = summary_data.get("swap_total_gb", 0.0)

        # Format system stats parts
        # CPU: utilization first, cores in brackets
        if cpu_cores > 0:
            cpu_percent_int = int(round(cpu_percent))
            cpu_color = self._threshold_color(cpu_percent_int, 50, 80, "green")
            cpu_util_str = colored(f"{cpu_percent_int:>3}%", cpu_color)
            cpu_part = colored("CPU ", "dark_grey") + f"{cpu_util_str}({cpu_cores}C)"
        else:
            cpu_part = ""

        # Memory: used/total with swap in brackets
        if mem_total_gb > 0:
            if swap_total_gb > 0:
                mem_sys_part = (
                    colored("Mem ", "dark_grey")
                    + f"{mem_used_gb:.0f}/{mem_total_gb:.0f}G"
                    + f"(Swap {swap_used_gb:.1f}/{swap_total_gb:.0f}G)"
                )
            else:
                mem_sys_part = (
                    colored("Mem ", "dark_grey")
                    + f"{mem_used_gb:.0f}/{mem_total_gb:.0f}G"
                )
        else:
            mem_sys_part = ""

        sep = colored(" · ", "dark_grey")

        # Handle no GPU case - only show system stats
        if no_gpu:
            source = summary_data.get("data_source") or "unknown"
            label = "Unsupported" if source == "unsupported" else "No GPU"
            sys_parts = [p for p in [cpu_part, mem_sys_part] if p]
            pieces = [colored(label, "dark_grey"), colored(f"src:{source}", "dark_grey")]
            return sep.join(pieces + sys_parts)

        widths = widths or {}
        gpu_digits = widths.get("gpu_digits", len(str(gpu_count)))
        idle_digits = widths.get("idle_digits", len(str(idle_count)))
        util_digits = widths.get("util_digits", len(str(avg_util)))
        mem_width = widths.get("mem_width", len(str(mem_display)))

        gpu_part = f"{gpu_count:>{gpu_digits}} GPUs"
        # The idle count is what someone scanning for a free card reads
        # first: green while there is one, dim once the node is full.
        idle_part = colored(
            f"{idle_count:>{idle_digits}} idle",
            "green" if idle_count > 0 else "dark_grey",
        )
        util_plain = f"{avg_util:>{util_digits}}%"
        util_part = (
            (colored(util_plain, util_color) if util_color else util_plain)
            + " "
            + self._mini_bar(avg_util, util_color)
        )
        mem_percent = summary_data.get("mem_percent")
        mem_color = self._threshold_color(mem_percent, 75, 90, "green")
        mem_part = (
            f"{str(mem_display):>{mem_width}} "
            + self._mini_bar(mem_percent, mem_color)
        )

        # Build the summary string
        parts = [sep.join([gpu_part, idle_part, util_part, mem_part])]
        sys_parts = [p for p in [cpu_part, mem_sys_part] if p]
        if sys_parts:
            parts.append(sep.join(sys_parts))

        source = summary_data.get("data_source")
        if source:
            parts.append(colored(f"src:{source}", "dark_grey"))

        return sep.join(parts)

    def _format_user_memory_totals(self, user_summary: dict, *, max_users: int = 12) -> str:
        """Format per-user total VRAM usage (MiB) as a compact single-line string."""
        # The MiB values are int sums (see get_process_summary / _sample_metadata).
        items = [(str(user), mib) for user, mib in user_summary.items() if mib > 0]
        if not items:
            return ""

        items.sort(key=lambda x: x[1], reverse=True)
        parts = []
        for user, mib_int in items[:max_users]:
            if mib_int >= 1024:
                parts.append(f"{user}({mib_int/1024:.1f}G)")
            else:
                parts.append(f"{user}({mib_int}M)")

        remaining = len(items) - max_users
        if remaining > 0:
            parts.append(f"+{remaining} more")
        return " ".join(parts)
    
    def print_stats(self, use_cache=False):
        """Print GPU stats with collapsible server view."""
        _set_monitor_theme(self.theme)
        # Fetch new data or use cache
        if use_cache:
            with self._cache_lock:
                stats_list = self.cached_stats
                raw_stats_by_client = self.cached_raw_stats
                last_update_time = self._last_update_time
                last_fetch_duration = self._last_fetch_duration
                last_fetch_error = self._last_fetch_error
        else:
            fetch_started_at = time.time()
            try:
                stats_list, raw_stats_by_client = self.get_client_gpus_info(return_raw=True)
                fetch_error = None
            except Exception as e:
                stats_list, raw_stats_by_client = None, {}
                fetch_error = e
            fetch_duration = time.time() - fetch_started_at

            if stats_list is not None:
                self._record_unified_gpu_history(raw_stats_by_client)
            with self._cache_lock:
                if stats_list is not None:
                    self.cached_stats = stats_list
                    self.cached_raw_stats = raw_stats_by_client
                    self._last_update_time = time.time()
                    self._last_fetch_error = None
                else:
                    self._last_fetch_error = fetch_error
                self._last_fetch_duration = fetch_duration

            last_update_time = self._last_update_time
            last_fetch_duration = self._last_fetch_duration
            last_fetch_error = self._last_fetch_error

        if stats_list is None:
            # The header summary already says "Loading..."; a block under it
            # would only repeat the same word twice per server.
            stats_list = [""] * len(self.pool)
        terminal_size = _terminal_size()
        terminal_width = terminal_size.columns
        terminal_height = terminal_size.lines
        if stats_list and raw_stats_by_client:
            # Responsive relayout: the cached blocks were formatted for the
            # width the terminal had at fetch time. If the window has been
            # resized since, re-render them from the raw stats now instead
            # of painting tables that no longer fit (or waste) the width.
            if self._node_workers or self._blocks_formatted_width != terminal_width:
                try:
                    stats_list = self._format_client_blocks(
                        [
                            raw_stats_by_client.get(idx, (pd.DataFrame(), {"loading": True}))
                            for idx in range(len(self.pool))
                        ]
                    )
                except Exception as error:
                    logging.debug(msg=f"Block relayout after resize failed: {error}")

        self._apply_default_expansion(raw_stats_by_client, last_update_time)

        # The local machine reports through the top strip; the node list
        # holds it only while it has GPU rows (or is the only machine).
        node_indices = self._compute_node_indices(raw_stats_by_client)
        self._node_listed_indices = node_indices

        # Disable expand/collapse for servers with auth errors (e.g., password incorrect)
        self._toggle_disabled_servers = {
            idx
            for idx in range(len(self.pool))
            if raw_stats_by_client.get(idx, (None, {}))[1].get("error_type") == "auth"
        }

        current_time = time.strftime("%H:%M:%S")

        # Ensure selected_server is within bounds
        self.selected_server = max(0, min(self.selected_server, len(self.pool) - 1))

        update_display = "--:--:--"
        if last_update_time:
            update_display = time.strftime("%H:%M:%S", time.localtime(last_update_time))
        fetch_display = ""
        debug = self.debug
        if debug and isinstance(last_fetch_duration, (int, float)):
            fetch_display = f" ({last_fetch_duration:.1f}s)"
        warn_display = " · WARN: refresh failed" if last_fetch_error else ""

        output_lines = []
        server_count = len(node_indices)
        server_label = "Server" if server_count == 1 else "Servers"

        display_mode = self.display_mode
        # Narrow windows get their own chrome. Every line the frame builds
        # itself is sized to fit one terminal row, so the terminal never
        # soft-wraps a server header into the next line.
        compact_layout = terminal_width < 72
        # The per-node view gets an outer panel border when there is room
        # for one; the unified table has its own border already and stays
        # as-is. Reserving the margin here - before any chrome line below
        # is built - means every width computation downstream (title,
        # controls, the server rows) narrows automatically instead of
        # needing its own adjustment.
        show_panel_border = (
            display_mode != self.DISPLAY_MODE_UNIFIED
            and not compact_layout
            # A very short window cannot spare two rows for decoration:
            # data completeness beats the frame.
            and terminal_height >= 14
            and not self.tui_help_visible
        )
        # The local machine's own strip opens the page in both views. It
        # sits above the per-node panel frame, so it is laid out for the
        # full window width before the border margin is reserved.
        version_space = display_width(f"v{nvidb_config.VERSION}") + 1
        strip_lines = self._format_local_strip(max(1, terminal_width - version_space), raw_stats_by_client)
        if strip_lines:
            strip_lines[0] = self._with_version_header(strip_lines[0], terminal_width)
        output_lines.extend(strip_lines)
        if show_panel_border:
            terminal_width -= self._PANEL_BORDER_MARGIN
        if display_mode == self.DISPLAY_MODE_UNIFIED:
            detailed = self.unified_detailed
            view_label = "Unified/Detailed" if detailed else "Unified/Single-line"
        else:
            view_label = "Per-node"
        if compact_layout:
            plain_title = (
                f"Updated {update_display} · {server_count} {server_label.lower()}"
                + (" · refresh failed" if last_fetch_error else "")
            )
            title_line = (
                fit(plain_title, terminal_width)
                if display_width(plain_title) > terminal_width
                else colored("Updated ", "dark_grey") + colored(update_display, "cyan")
                + colored(" · ", "dark_grey")
                + colored(f"{server_count}", "magenta")
                + f" {server_label.lower()}"
                + (colored(" · refresh failed", "red") if last_fetch_error else "")
            )
        else:
            title_sep = colored(" · ", "dark_grey")
            plain_title = (
                (f"Time {current_time} · " if debug else "")
                + f"Updated {update_display}{fetch_display} · "
                f"{server_label} {server_count} · View {view_label}{warn_display}"
            )
            title_line = (
                fit(plain_title, terminal_width)
                if display_width(plain_title) > terminal_width
                else (colored("Time ", "dark_grey") + colored(current_time, "cyan") + title_sep if debug else "")
                + colored("Updated ", "dark_grey")
                + colored(f"{update_display}{fetch_display}", "cyan")
                + title_sep
                + colored(f"{server_label} ", "dark_grey")
                + colored(str(server_count), "magenta")
                + title_sep
                + colored("View ", "dark_grey") + colored(view_label, "green")
                + (colored(warn_display, "red") if warn_display else "")
            )
        output_lines.append(title_line if strip_lines else self._with_version_header(title_line, terminal_width))

        toolbar_row = self._screen_line_count(output_lines)
        toolbar_lines, toolbar_regions, self._toolbar_overflow = action_bar(
            self._monitor_controls(), terminal_width, self._control_style,
        )
        output_lines.extend(toolbar_lines)
        toolbar_regions = [
            (toolbar_row + row, start, end, target)
            for row, start, end, target in toolbar_regions
        ]

        if debug:
            pending_nodes = []
            slow_nodes = []
            for idx in node_indices:
                _stats, info = raw_stats_by_client.get(idx, (None, {"loading": True}))
                monitor = info.get("_monitor", {})
                if info.get("loading"):
                    pending_nodes.append(self._client_table_identity(idx)[0])
                elif monitor.get("updated") and not info.get("error") and time.time() - monitor["updated"] > 5:
                    slow_nodes.append(self._client_table_identity(idx)[0])
            if pending_nodes or slow_nodes:
                notices = []
                for label, nodes in (("Loading", pending_nodes), ("Stale >5s", slow_nodes)):
                    if nodes:
                        notices.append(f"{label} {len(nodes)}: {', '.join(nodes[:3])}" + (" …" if len(nodes) > 3 else ""))
                output_lines.append(colored(fit_display(" · ".join(notices), terminal_width), "yellow"))

        # The rule under the chrome is decoration; a very short window
        # spends the row on data instead.
        if terminal_height >= 12:
            if self.compact:
                separator_width = min(80, terminal_width)
            else:
                separator_width = max(20, terminal_width)
            output_lines.append(colored("─" * separator_width, "dark_grey"))

        if self.tui_help_visible:
            help_offset = self._screen_line_count(output_lines)
            help_lines = self._format_tui_help_lines(
                terminal_width,
                max(4, terminal_height - help_offset),
            )
            output_lines.extend(help_lines)
            self._click_targets = {
                help_offset + index: ("close_help", None)
                for index in range(len(help_lines))
            }
            self._click_regions = toolbar_regions
            self._body_click_targets = {}
            self._body_click_regions = []
            self._write_tui_lines(output_lines)
            return

        meta = raw_stats_by_client.get("_nvidb") or {}
        user_memory_by_client = meta.get("user_memory_by_client", {}) or {}
        global_user_memory = meta.get("user_memory_global", {}) or {}

        compact_detailed_screen = (
            display_mode == self.DISPLAY_MODE_UNIFIED
            and self._compact_detailed_layout(terminal_height)
        )
        if global_user_memory and not compact_detailed_screen:
            user_totals = self._format_user_memory_totals(global_user_memory)
            plain_line = f"Users (all nodes): {user_totals}"
            if self.term.length(plain_line) > terminal_width:
                output_lines.append(fit(plain_line, terminal_width))
            else:
                global_line = (
                    colored("Users (all nodes):", "magenta", attrs=["bold"])
                    + f" {user_totals}"
                )
                output_lines.append(global_line)

        if display_mode == self.DISPLAY_MODE_UNIFIED:
            body_offset = self._screen_line_count(output_lines)
            body = self._render_unified_gpu_lines(
                raw_stats_by_client,
                last_update_time,
                height_budget=max(0, terminal_height - 1 - body_offset),
            )
            output_lines.extend(body)
            # Translate the body-relative hit boxes into absolute screen rows.
            self._click_targets = {
                body_offset + line: target
                for line, target in self._body_click_targets.items()
            }
            self._click_regions = toolbar_regions + [
                (body_offset + line, start, end, target)
                for line, start, end, target in self._body_click_regions
            ]
            self._write_tui_lines(output_lines)
            return

        self._click_targets = {}
        self._click_regions = toolbar_regions
        self._body_click_targets = {}
        self._body_click_regions = []

        # The cursor may sit on the local machine from an earlier frame in
        # which it was still listed; land it on a row that is on screen.
        if node_indices and self.selected_server not in node_indices:
            self.selected_server = node_indices[0]

        # Align the collapsed server list by padding headers to the same display width
        index_width = len(str(max(1, len(node_indices))))
        server_rows = []
        max_header_width = 0

        for position, idx in enumerate(node_indices):
            client = self.pool[idx]
            stats_info = stats_list[idx] if idx < len(stats_list) else ""
            is_selected = idx == self.selected_server
            is_expanded = idx in self.expanded_servers
            toggle_disabled = idx in self._toggle_disabled_servers

            # Use cached raw stats for summary
            stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))

            # Build summary
            if system_info.get("error"):
                summary_data = {
                    "status": "error",
                    "message": system_info.get("error", "Error"),
                    "error_type": system_info.get("error_type"),
                }
            elif idx not in raw_stats_by_client or system_info.get("loading") or (last_update_time is None and stats.empty):
                summary_data = {"status": "loading", "message": self._node_monitor_text(system_info)}
            else:
                summary_data = self._get_server_summary_data(stats, system_info)

            # Format the header line (index padded for consistent width)
            if toggle_disabled:
                expand_icon = "!"
            else:
                expand_icon = "▾" if is_expanded else "▸"
            selector = "❯" if is_selected else " "
            dot, _dot_color = self._server_health_dot(summary_data)
            header_plain = (
                f"{selector} {expand_icon} [{position + 1:{index_width}d}] "
                f"{dot} {client.description}"
            )

            max_header_width = max(max_header_width, display_width(header_plain))
            server_rows.append((idx, is_selected, is_expanded, header_plain, summary_data, stats_info))

        non_empty_summaries = [row[4] for row in server_rows if "gpu_count" in row[4]]
        if non_empty_summaries:
            widths = {
                "gpu_digits": max(len(str(s["gpu_count"])) for s in non_empty_summaries),
                "idle_digits": max(len(str(s["idle_count"])) for s in non_empty_summaries),
                "util_digits": max(len(str(s["avg_util"])) for s in non_empty_summaries),
                "mem_width": max(len(str(s["mem_display"])) for s in non_empty_summaries),
            }
        else:
            widths = None

        # Every expanded server keeps its whole table. Making the stacked
        # blocks share the window used to mean dropping GPU rows from the
        # tallest one, which hid real cards and - because the selection was
        # exempt from the trimming - made the layout shift as the cursor
        # moved. The viewport below scrolls instead.
        border_rows = 2 if show_panel_border else 0
        # Dividers between servers are decoration; a window too short to
        # hold them spends the rows on GPU data instead.
        rows_for_body = (
            (terminal_height - 1)
            - self._screen_line_count(output_lines)
            - border_rows
        )
        show_separators = (
            len(server_rows) > 1
            and show_panel_border
            and rows_for_body - (len(server_rows) - 1) >= len(server_rows)
        )
        visible_blocks = self._build_server_blocks(
            server_rows, user_memory_by_client, terminal_width
        )

        sections = []
        for position, (
            idx,
            is_selected,
            is_expanded,
            header_plain,
            summary_data,
            _stats_info,
        ) in enumerate(server_rows):
            toggle_disabled = idx in self._toggle_disabled_servers
            # The icon reports what the user chose, nothing else: the
            # layout no longer collapses servers on its own.
            expand_icon = (
                "!"
                if toggle_disabled
                else ("▾" if is_expanded else "▸")
            )
            selector = "❯" if is_selected else " "
            dot, dot_color = self._server_health_dot(summary_data)
            summary = self._format_server_summary(
                summary_data, widths=widths, compact=compact_layout
            )
            # A server header is exactly one terminal line. The description
            # gives ground to the summary when both cannot fit, so the
            # terminal never soft-wraps the row and shears the click map.
            summary_room = (
                terminal_width
                - 2
                - display_width(self._ANSI_ESCAPE_RE.sub("", summary))
            )
            # Colour is skipped on the selected row: it is already picked
            # out with reverse video, and termcolor's per-segment reset
            # codes would cut that reverse video short partway through.
            icon_color = None if is_selected else self._server_icon_color(
                expand_icon, toggle_disabled
            )
            if compact_layout:
                prefix = f"{selector} {expand_icon} {position + 1} {dot} "
                description = fit(
                    self.pool[idx].description,
                    max(1, summary_room - display_width(prefix)),
                )
                if is_selected:
                    header_display = (
                        self.term.reverse + prefix + description + self.term.normal
                    )
                else:
                    icon_display = (
                        colored(expand_icon, icon_color) if icon_color else expand_icon
                    )
                    header_display = (
                        f"{selector} {icon_display} {position + 1} "
                        f"{colored(dot, dot_color)} "
                        + colored(description, attrs=["bold"])
                    )
            else:
                pad = max_header_width - display_width(header_plain)
                header_padded = header_plain + " " * pad
                header_padded = fit(header_padded, max(1, summary_room))
                if is_selected:
                    header_display = (
                        self.term.reverse + header_padded + self.term.normal
                    )
                else:
                    # The icon, index, and dot prefix are a handful of
                    # characters, always shorter than summary_room in
                    # practice (this branch only runs at width >= 72); the
                    # rest (description + trailing pad) is bolded as one
                    # block so a mid-word cut never splits an ANSI code.
                    prefix_plain = (
                        f"{selector} {expand_icon} [{position + 1:{index_width}d}] {dot} "
                    )
                    if len(header_padded) >= len(prefix_plain):
                        rest = header_padded[len(prefix_plain):]
                        icon_display = (
                            colored(expand_icon, icon_color)
                            if icon_color
                            else expand_icon
                        )
                        header_display = (
                            f"{selector} {icon_display} "
                            f"[{position + 1:{index_width}d}] "
                            f"{colored(dot, dot_color)} "
                            + colored(rest, attrs=["bold"])
                        )
                    else:
                        header_display = header_padded
            row = f"{header_display}  {summary}"
            if display_width(self._ANSI_ESCAPE_RE.sub("", row)) > terminal_width:
                # A pathologically long error message is the one input that
                # can still overflow; dropping its colour beats wrapping.
                row = fit(self._ANSI_ESCAPE_RE.sub("", row), terminal_width)

            # One section per server: its header plus its whole table.
            section_lines = [row]
            block = visible_blocks.get(idx)
            if block is not None:
                section_lines.extend(block)
            sections.append((idx, section_lines))

        # The viewport: when the sections outgrow the window (many
        # servers, a short terminal), show a window of servers that
        # follows the selection. An indicator line at each edge counts what
        # is scrolled out, so nothing can fall off the screen unnoticed.
        region_height = max(1, rows_for_body)
        # One server taller than the entire region is the only case where
        # rows still have to go: there is nowhere to scroll them to. Only
        # that server's own table is shortened, and it always says by how
        # much - a silently clipped table is what makes a card look like
        # it vanished. Trimming always starts from the untouched section
        # so the dropped-line count stays honest across both passes.
        natural_sections = [(idx, list(lines)) for idx, lines in sections]
        # Markers sit inside the panel frame, so they fit its content width.
        marker_width = max(10, terminal_width)

        def fit_section(position, budget):
            idx, lines = natural_sections[position]
            budget = max(1, budget)
            if len(lines) <= budget:
                sections[position] = (idx, lines)
                return
            if budget == 1:
                # Only the server's own row fits; its summary still says
                # how many GPUs are down there.
                sections[position] = (idx, [lines[0]])
                return
            trimmed = self._trim_server_block(lines[1:], budget - 1, marker_width)
            if trimmed is None:
                # Not even one data row survives under the header; saying
                # so beats painting a headless slice of the table.
                trimmed = [
                    colored(
                        fit(
                            "  … table hidden — enlarge the terminal or press [c]",
                            marker_width,
                        ),
                        "dark_grey",
                    )
                ]
            sections[position] = (idx, [lines[0]] + trimmed)

        for position in range(len(sections)):
            fit_section(position, region_height)

        selected_pos = 0
        for position, (idx, _lines) in enumerate(sections):
            if idx == self.selected_server:
                selected_pos = position
                break
        view_start, view_end = self._nodes_viewport(
            [len(lines) for _idx, lines in sections],
            region_height,
            1 if show_separators else 0,
            selected_pos,
        )
        # How many rows the edge indicators claim is only known once the
        # window has landed. When a single oversized section fills it, that
        # section is re-trimmed against what is actually left.
        indicators = (1 if view_start > 0 else 0) + (
            1 if view_end < len(sections) else 0
        )
        if indicators and view_end - view_start == 1:
            fit_section(view_start, region_height - indicators)
        self._nodes_view_start = view_start

        if view_start > 0:
            # Clicking an indicator selects the nearest hidden server,
            # which scrolls the window toward it.
            self._click_targets[self._screen_line_count(output_lines)] = (
                "server",
                sections[view_start - 1][0],
            )
            output_lines.append(
                colored(f"  … {view_start} more server(s) above", "dark_grey")
            )
        for position in range(view_start, view_end):
            idx, section_lines = sections[position]
            if position > view_start and show_separators:
                output_lines.append(self._PANEL_SEPARATOR)
            self._click_targets[self._screen_line_count(output_lines)] = ("server", idx)
            output_lines.extend(section_lines)
        if view_end < len(sections):
            self._click_targets[self._screen_line_count(output_lines)] = (
                "server",
                sections[view_end][0],
            )
            output_lines.append(
                colored(
                    f"  … {len(sections) - view_end} more server(s) below",
                    "dark_grey",
                )
            )

        if show_panel_border:
            # The frame wraps the node list only; the local strip stays
            # above it as the first line of the page.
            strip_count = len(strip_lines)
            output_lines = output_lines[:strip_count] + self._wrap_panel_border(
                output_lines[strip_count:], terminal_width
            )
            self._click_targets = {
                row + 1: target for row, target in self._click_targets.items()
            }

            self._click_regions = [
                (row + 1, start + 2, end + 2, target)
                for row, start, end, target in self._click_regions
            ]

        self._write_tui_lines(output_lines)

    def _nodes_viewport(self, heights, region_height, sep_cost, selected_pos):
        """Pick the window of server sections that fits the terminal.

        Every server must stay reachable no matter how short the window
        is: the viewport follows the selection, and the caller draws one
        indicator line per edge naming how many servers are scrolled out.
        Returns (start, end) section positions; end is exclusive.
        """
        count = len(heights)
        if count == 0:
            return 0, 0

        def total_from(start):
            return sum(heights[start:]) + sep_cost * max(0, count - start - 1)

        if total_from(0) <= region_height:
            return 0, count

        def end_for(start):
            rows = region_height - (1 if start > 0 else 0)
            if total_from(start) <= rows:
                return count
            rows -= 1  # the bottom indicator line
            used = 0
            end = start
            while end < count:
                need = heights[end] + (sep_cost if end > start else 0)
                if used + need > rows:
                    break
                used += need
                end += 1
            # A section taller than the whole window still gets shown (the
            # painter clips its tail) rather than vanishing entirely.
            return max(end, start + 1)

        start = max(
            0,
            min(self._nodes_view_start, count - 1, selected_pos),
        )
        end = end_for(start)
        while selected_pos >= end and start < selected_pos:
            start += 1
            end = end_for(start)
        return start, end

    def _request_ui_refresh(self):
        self.refresh_needed.set()

    def _current_view_settings(self):
        """Snapshot the layout state persisted between runs."""
        return {
            "mode": (
                "unified"
                if self.display_mode
                == self.DISPLAY_MODE_UNIFIED
                else "nodes"
            ),
            "detailed": bool(self.unified_detailed),
            "sort": self._get_unified_sort_mode(),
            "filter": self._get_unified_filter_mode(),
            "processes": bool(self.unified_show_processes),
            "trends": bool(self.unified_show_trends),
            "group_by_node": bool(self.unified_group_by_node),
            "hide_unsupported": bool(self.hide_unsupported),
            "mouse": bool(self.mouse_enabled),
            "theme": self.theme,
        }

    def _persist_view_settings(self):
        """Write the current layout to config.yml; never break the TUI on error."""
        if not self._persist_view_enabled:
            return
        try:
            # Merge over what is stored so future view settings this version
            # does not know about survive the write.
            settings = nvidb_config.load_view_settings()
            settings.update(self._current_view_settings())
            nvidb_config.save_view_settings(settings)
        except Exception as error:
            logging.debug(msg=f"Failed to persist view settings: {error}")

    def _apply_view_change(self):
        self._request_ui_refresh()
        self._persist_view_settings()

    def _toggle_theme(self):
        self.theme = "muted" if self.theme == "classic" else "classic"
        _set_monitor_theme(self.theme)
        self._apply_view_change()
        return True

    def _toggle_display_mode(self):
        current_mode = self.display_mode
        if current_mode == self.DISPLAY_MODE_UNIFIED:
            self.display_mode = self.DISPLAY_MODE_NODES
        else:
            self.display_mode = self.DISPLAY_MODE_UNIFIED
        self.unified_active_pane = "gpu"
        self.unified_process_panel_hidden = False
        self.unified_selected_process_pid = None
        self.unified_command_scroll = 0
        self._pending_process_signal = None
        self._apply_view_change()

    def _cycle_unified_sort_mode(self):
        current_mode = self._get_unified_sort_mode()
        next_index = (
            self.UNIFIED_SORT_MODES.index(current_mode) + 1
        ) % len(self.UNIFIED_SORT_MODES)
        self._set_unified_sort_mode(self.UNIFIED_SORT_MODES[next_index])

    def _set_unified_sort_mode(self, mode):
        self.unified_sort_mode = mode
        self.unified_selected_gpu = 0
        self.unified_selected_gpu_key = None
        self._reset_process_selection()
        self._apply_view_change()

    def _cycle_unified_filter_mode(self):
        current_mode = self._get_unified_filter_mode()
        next_index = (
            self.UNIFIED_FILTER_MODES.index(current_mode) + 1
        ) % len(self.UNIFIED_FILTER_MODES)
        self._set_unified_filter_mode(self.UNIFIED_FILTER_MODES[next_index])

    def _set_unified_filter_mode(self, mode):
        self.unified_filter_mode = mode
        self.unified_selected_gpu = 0
        self.unified_selected_gpu_key = None
        self.unified_active_pane = "gpu"
        self._reset_process_selection()
        self._apply_view_change()

    def _move_unified_selection(self, delta):
        gpu_count = self._unified_gpu_count
        if gpu_count == 0:
            return False
        current = self.unified_selected_gpu
        selected = max(0, min(current + delta, gpu_count - 1))
        if selected == current:
            return False
        self.unified_selected_gpu = selected
        self.unified_selected_gpu_key = None
        self._reset_process_selection()
        self._request_ui_refresh()
        return True

    def _move_unified_process_selection(self, delta):
        count = self._unified_process_count
        if count == 0:
            return False
        current = self.unified_selected_process
        selected = max(0, min(current + delta, count - 1))
        if selected == current:
            return False
        self.unified_selected_process = selected
        self.unified_selected_process_pid = None
        self.unified_command_scroll = 0
        self._pending_process_signal = None
        self._request_ui_refresh()
        return True

    def _set_unified_process_sort(self, mode, *, toggle=False):
        if mode not in self.UNIFIED_PROCESS_SORT_MODES:
            return False
        current = self._get_unified_process_sort_mode()
        if mode == current and toggle:
            descending = not self.unified_process_sort_descending
        elif mode == current:
            return False
        else:
            descending = mode != "command"

        self.unified_process_sort_mode = mode
        self.unified_process_sort_descending = descending
        self._pending_process_signal = None
        self._request_ui_refresh()
        return True

    def _cycle_unified_process_sort(self):
        current = self._get_unified_process_sort_mode()
        next_index = (
            self.UNIFIED_PROCESS_SORT_MODES.index(current) + 1
        ) % len(self.UNIFIED_PROCESS_SORT_MODES)
        return self._set_unified_process_sort(
            self.UNIFIED_PROCESS_SORT_MODES[next_index]
        )

    def _adjust_unified_process_rows(self, delta):
        if not self._process_panel_visible():
            return False
        current = self.unified_process_rows or max(1, self._unified_process_visible_rows)
        selected = max(1, min(12, current + delta))
        if selected == current:
            return False
        self.unified_process_rows = selected
        self._request_ui_refresh()
        return True

    def _reset_process_selection(self):
        self.unified_selected_process = 0
        self.unified_selected_process_pid = None
        self.unified_command_scroll = 0
        self._pending_process_signal = None

    def _start_unified_process_filter(self):
        if self._unified_gpu_count == 0:
            return False
        entered = self._enter_unified_process_pane()
        changed = not self.unified_process_filter_editing
        self.unified_process_filter_editing = True
        self._pending_process_signal = None
        if changed and not entered:
            self._request_ui_refresh()
        return changed or entered

    def _handle_unified_process_filter_key(self, key_text, key_name):
        """Edit the live process filter while keeping global shortcuts inactive."""
        if not self.unified_process_filter_editing:
            return False

        enter_pressed = (
            key_name == "KEY_ENTER"
            or key_text in {"\n", "\r"}
        )
        if enter_pressed:
            self.unified_process_filter_editing = False
            self._request_ui_refresh()
            return True

        if key_name == "KEY_ESCAPE" or key_text == "\x1b":
            self.unified_process_filter = ""
            self.unified_process_filter_editing = False
            self._reset_process_selection()
            self._request_ui_refresh()
            return True

        if (
            key_name in {"KEY_BACKSPACE", "KEY_DELETE"}
            or key_text in {"\b", "\x7f"}
        ):
            query = self.unified_process_filter
            if query:
                self.unified_process_filter = query[:-1]
                self._reset_process_selection()
                self._request_ui_refresh()
            return True

        if key_text == "\x15":  # Ctrl+U
            self.unified_process_filter = ""
            self._reset_process_selection()
            self._request_ui_refresh()
            return True

        if (
            key_text
            and not str(key_name).startswith("KEY_")
            and all(character.isprintable() for character in key_text)
        ):
            self.unified_process_filter = (self.unified_process_filter + key_text)[:80]
            self._reset_process_selection()
            self._request_ui_refresh()
            return True
        return False

    def _clear_unified_process_filter(self):
        if not self.unified_process_filter and not self.unified_process_filter_editing:
            return False
        self.unified_process_filter = ""
        self.unified_process_filter_editing = False
        self._reset_process_selection()
        self._request_ui_refresh()
        return True

    def _toggle_tui_help(self):
        self.tui_help_visible = not self.tui_help_visible
        self.unified_process_filter_editing = False
        self._pending_process_signal = None
        self._request_ui_refresh()
        return True

    def _scroll_unified_command(self, page_delta):
        line_count = self._unified_command_line_count
        page_size = self._unified_command_page_size
        if page_size <= 0 or line_count <= page_size:
            return False
        current = self.unified_command_scroll
        selected = max(0, min(current + page_delta * page_size, line_count - page_size))
        if selected == current:
            return False
        self.unified_command_scroll = selected
        self._request_ui_refresh()
        return True

    def _process_panel_visible(self):
        if self.unified_process_panel_hidden:
            return False
        return (
            self.display_mode == self.DISPLAY_MODE_UNIFIED
            and (self.unified_detailed or self.unified_show_processes)
            and self._get_unified_filter_mode() != "errors"
        )

    def _enter_unified_process_pane(self):
        """Show the selected GPU's processes and move focus into that pane."""
        if (
            self.display_mode != self.DISPLAY_MODE_UNIFIED
            or self._get_unified_filter_mode() == "errors"
            or self._unified_gpu_count == 0
        ):
            return False

        visibility_changed = False
        persist_visibility = False
        if self.unified_detailed:
            if self.unified_process_panel_hidden:
                self.unified_process_panel_hidden = False
                visibility_changed = True
        elif not self.unified_show_processes:
            self.unified_show_processes = True
            visibility_changed = True
            persist_visibility = True

        if not self._process_panel_visible():
            return False
        focus_changed = self.unified_active_pane != "process"
        if not visibility_changed and not focus_changed:
            return False

        self.unified_active_pane = "process"
        self._pending_process_signal = None
        if persist_visibility:
            self._apply_view_change()
        else:
            self._request_ui_refresh()
        return True

    def _toggle_unified_process_panel(self):
        """Show or hide the process pane without treating Enter as a toggle."""
        if (
            self.display_mode != self.DISPLAY_MODE_UNIFIED
            or self._get_unified_filter_mode() == "errors"
            or self._unified_gpu_count == 0
        ):
            return False

        detailed = self.unified_detailed
        if detailed:
            self.unified_process_panel_hidden = not self.unified_process_panel_hidden
            panel_visible = not self.unified_process_panel_hidden
        else:
            self.unified_process_panel_hidden = False
            panel_visible = not self.unified_show_processes
            self.unified_show_processes = panel_visible

        if not panel_visible:
            self.unified_active_pane = "gpu"
            self._pending_process_signal = None
        if detailed:
            self._request_ui_refresh()
        else:
            self._apply_view_change()
        return True

    def _set_unified_active_pane(self, pane):
        if pane not in {"gpu", "process"}:
            return False
        if pane == "process" and not self._process_panel_visible():
            return False
        if pane == self.unified_active_pane:
            return False
        self.unified_active_pane = pane
        self._pending_process_signal = None
        self._request_ui_refresh()
        return True

    def _toggle_unified_active_pane(self):
        if not self._process_panel_visible():
            return False
        pane = self.unified_active_pane
        return self._set_unified_active_pane(
            "gpu" if pane == "process" else "process"
        )

    def _selected_process_context(self):
        raw_stats = self.cached_raw_stats or {}
        row = self._selected_unified_row(raw_stats)
        if row is None:
            return None
        processes = self._get_sorted_unified_processes(raw_stats, row)
        if not processes:
            return None
        process = processes[self._sync_selected_process(processes)]
        client_index = self._unified_client_index_for_row(row)
        if client_index is None or not 0 <= client_index < len(self.pool):
            return None
        return {
            "row": row,
            "process": process,
            "client_index": client_index,
        }

    @staticmethod
    def _process_command_identity(process):
        """The command text a confirmation must still find on the armed PID."""
        return str(process.get("command") or process.get("process_name") or "")

    def _set_process_action_notice(self, message, color):
        self._process_action_notice = {
            "message": str(message),
            "color": color,
            "expires_at": time.monotonic() + 8,
        }

    def _request_process_signal(self, signal_name):
        """Arm a process signal, or send it when the same action is confirmed."""
        signal_name = str(signal_name).upper()
        if signal_name not in {"INT", "TERM", "KILL"}:
            return False
        if not self._process_panel_visible():
            return False
        if self.unified_active_pane != "process":
            self.unified_active_pane = "process"
            self._pending_process_signal = None
            self._set_process_action_notice(
                f"Process focus enabled; press {signal_name} again to arm",
                "cyan",
            )
            self._request_ui_refresh()
            return True
        context = self._selected_process_context()
        if context is None:
            self._pending_process_signal = None
            self._set_process_action_notice(
                "No selected process is available",
                "yellow",
            )
            self._request_ui_refresh()
            return True

        process = context["process"]
        pid_text = str(process.get("pid", "")).strip()
        if not pid_text.isdigit() or int(pid_text) <= 1:
            self._pending_process_signal = None
            self._set_process_action_notice(
                f"Refusing to signal invalid PID {pid_text or 'N/A'}",
                "red",
            )
            self._request_ui_refresh()
            return True

        pending = self._pending_process_signal
        same_action = (
            pending
            and pending.get("expires_at", 0) > time.monotonic()
            and pending.get("signal") == signal_name
            and pending.get("pid") == int(pid_text)
            and pending.get("client_index") == context["client_index"]
        )
        if same_action:
            return self._confirm_process_signal()

        self._process_action_notice = None
        self._pending_process_signal = {
            "signal": signal_name,
            "pid": int(pid_text),
            "client_index": context["client_index"],
            "node": str(context["row"].get("Node", "N/A")),
            "command": self._process_command_identity(process),
            "expires_at": time.monotonic() + 5,
        }
        self._request_ui_refresh()
        return True

    def _confirm_process_signal(self):
        pending = self._pending_process_signal
        if not pending:
            return False
        if pending.get("expires_at", 0) <= time.monotonic():
            self._pending_process_signal = None
            self._set_process_action_notice(
                "Signal confirmation expired",
                "yellow",
            )
            self._request_ui_refresh()
            return True

        context = self._selected_process_context()
        if context is None:
            self._pending_process_signal = None
            self._set_process_action_notice(
                "The selected process is no longer available",
                "yellow",
            )
            self._request_ui_refresh()
            return True

        pid_text = str(context["process"].get("pid", "")).strip()
        if (
            not pid_text.isdigit()
            or int(pid_text) != pending.get("pid")
            or context["client_index"] != pending.get("client_index")
            or self._process_command_identity(context["process"]) != pending.get("command")
        ):
            self._pending_process_signal = None
            self._set_process_action_notice(
                "Selection changed; signal cancelled",
                "yellow",
            )
            self._request_ui_refresh()
            return True

        signal_name = pending["signal"]
        pid = pending["pid"]
        client = self.pool[pending["client_index"]]
        marker = "__NVIDB_SIGNAL_STATUS__:"
        command = (
            f"kill -s {signal_name} -- {pid} 2>&1; "
            f"status=$?; printf '\\n{marker}%s\\n' \"$status\""
        )
        self._pending_process_signal = None
        if self._node_workers:
            worker = self._signal_worker
            if worker is not None and worker.is_alive():
                self._set_process_action_notice("A signal request is still in progress", "yellow")
            else:
                self._set_process_action_notice(f"Sending SIG{signal_name} to PID {pid}…", "yellow")
                self._signal_worker = threading.Thread(
                    target=self._execute_confirmed_signal,
                    args=(client, command, marker, pending), daemon=True,
                )
                self._signal_worker.start()
            self._request_ui_refresh()
            return True
        return self._execute_confirmed_signal(client, command, marker, pending)

    def _execute_confirmed_signal(self, client, command, marker, pending):
        """Run an already-confirmed action without blocking the UI input loop."""
        signal_name = pending["signal"]
        pid = pending["pid"]
        try:
            output = client.execute_command(command)
            output = "" if output is None else str(output)
            if marker not in output:
                raise RuntimeError("target did not return a signal status")
            message, status_text = output.rsplit(marker, 1)
            status = int(status_text.strip().splitlines()[0])
            if status == 0:
                self._set_process_action_notice(
                    f"SIG{signal_name} sent to PID {pid} on {pending['node']}",
                    "green",
                )
            else:
                error = " ".join(message.strip().split())
                if not error:
                    error = f"kill exited with status {status}"
                self._set_process_action_notice(
                    f"SIG{signal_name} failed for PID {pid}: {error}",
                    "red",
                )
        except Exception as error:
            self._set_process_action_notice(
                f"SIG{signal_name} failed for PID {pid}: {error}",
                "red",
            )
        self._request_ui_refresh()
        return True

    def _move_node_selection(self, delta):
        """Move the nodes-view cursor across the listed nodes.

        The promoted local machine is not in the list, so the cursor
        steps over it instead of landing on a row that is not on screen.
        """
        indices = list(self._node_listed_indices or range(len(self.pool)))
        if not indices:
            return False
        try:
            position = indices.index(self.selected_server)
        except ValueError:
            position = 0
            delta = 0
        position = max(0, min(position + delta, len(indices) - 1))
        if indices[position] == self.selected_server:
            return False
        self.selected_server = indices[position]
        self._request_ui_refresh()
        return True

    def _toggle_server_expansion(self, index):
        if index in self._toggle_disabled_servers:
            return False
        if index in self.expanded_servers:
            self.expanded_servers.discard(index)
        else:
            self.expanded_servers.add(index)
        self._expansion_touched = True
        self._request_ui_refresh()
        return True

    def _handle_mouse_event(self, event):
        """Apply one mouse event and report whether the frame changed."""
        if not self.mouse_enabled:
            return False
        menu = self._tui_menu
        if menu is not None:
            previous_region = menu.hover.region
            result = menu.handle_mouse(event)
            if event.is_motion:
                changed = menu.hover.region != previous_region
                if changed:
                    self._tui_hover_dirty = True
                return changed
            return self._finish_tui_menu_input(result)
        if event.is_motion:
            changed = self._tui_hover.move(
                event, self._click_regions, self._click_targets, self.term.width or 80,
                row_kinds=self._HOVER_ROW_KINDS,
            )
            if changed:
                self._tui_hover_dirty = True
            return changed

        row = event.row - 1
        column = event.column - 1
        target = None
        for region_row, start, end, region_target in (
            self._click_regions or []
        ):
            if region_row == row and start <= column <= end:
                target = region_target
                break
        if target is None:
            target = (self._click_targets or {}).get(row)

        if self.tui_help_visible:
            if (
                event.is_left_press
                and target is not None
                and (target[0] == "close_help" or target == ("key", "?"))
            ):
                return self._toggle_tui_help()
            return False

        if event.is_left_press and target and target[0] in {
            "menu", "view", "key", "gpu_sort", "gpu_filter", "gpu_page", "clear_search",
        }:
            return self._activate_tui_control(*target)

        if event.is_left_press and self.unified_process_filter_editing:
            self.unified_process_filter_editing = False
            self._request_ui_refresh()

        if self.display_mode != self.DISPLAY_MODE_UNIFIED:
            if event.is_wheel_up or event.is_wheel_down:
                delta = -1 if event.is_wheel_up else 1
                return self._move_node_selection(delta)
            if not event.is_left_press:
                return False
            if target is None or target[0] != "server":
                return False
            index = target[1]
            # A closed node opens on the first click; an open node remains
            # visible when selected, and a second click collapses it.
            if index == self.selected_server:
                return self._toggle_server_expansion(index)
            self.selected_server = index
            if index not in self.expanded_servers:
                self._toggle_server_expansion(index)
            self._request_ui_refresh()
            return True

        if event.is_left_press and target and target[0] == "error_filter":
            self.unified_filter_mode = "errors"
            self.unified_selected_gpu = 0
            self.unified_selected_gpu_key = None
            self.unified_active_pane = "gpu"
            self._pending_process_signal = None
            self._request_ui_refresh()
            return True

        if event.is_wheel_up or event.is_wheel_down:
            delta = -1 if event.is_wheel_up else 1
            kind = target[0] if target else None
            if kind in {"gpu", "error"}:
                self.unified_active_pane = "gpu"
                return self._move_unified_selection(delta)
            if kind in {"command", "command_scroll"}:
                self.unified_active_pane = "process"
                return self._scroll_unified_command(delta)
            if kind in {
                "process",
                "process_rows",
                "process_sort",
                "signal",
                "toggle_trends",
            }:
                self.unified_active_pane = "process"
                return self._move_unified_process_selection(delta)
            if self.unified_active_pane == "process":
                return self._move_unified_process_selection(delta)
            return self._move_unified_selection(delta)
        if not event.is_left_press:
            return False

        if target is None:
            return False
        kind, value = target
        if kind == "process_sort":
            self.unified_active_pane = "process"
            return self._set_unified_process_sort(value, toggle=True)
        if kind == "process_rows":
            self.unified_active_pane = "process"
            return self._adjust_unified_process_rows(value)
        if kind == "command_scroll":
            self.unified_active_pane = "process"
            return self._scroll_unified_command(value)
        if kind == "signal":
            self.unified_active_pane = "process"
            return self._request_process_signal(value)
        if kind == "toggle_trends":
            self.unified_active_pane = "process"
            self.unified_show_trends = not self.unified_show_trends
            self._apply_view_change()
            return True
        if kind in {"process", "command"}:
            changed = (
                value != self.unified_selected_process
                or self.unified_active_pane != "process"
            )
            if not changed:
                return False
            self.unified_selected_process = value
            self.unified_selected_process_pid = None
            self.unified_command_scroll = 0
            self.unified_active_pane = "process"
            self._pending_process_signal = None
            self._request_ui_refresh()
            return True
        if kind == "gpu":
            changed = (
                value != self.unified_selected_gpu
                or self.unified_active_pane != "gpu"
            )
            if not changed:
                return self._enter_unified_process_pane()
            self.unified_selected_gpu = value
            self.unified_selected_gpu_key = None
            self.unified_active_pane = "gpu"
            self._reset_process_selection()
            if not self._process_panel_visible():
                self._enter_unified_process_pane()
                self.unified_active_pane = "gpu"
            self._request_ui_refresh()
            return True
        return False

    def _handle_keypress(self, key):
        """Apply one TUI keypress and return whether it changed visible state."""
        key_text = str(key)
        key_name = getattr(key, "name", None) or key_text
        key_lower = key_text.lower()

        if key_text and self._tui_hover.clear():
            self._request_ui_refresh()

        menu = self._tui_menu
        if menu is not None:
            return self._finish_tui_menu_input(menu.handle_key(key_text, key_name))

        if self.tui_help_visible:
            if (
                key_text in {"?", "q", "Q", "\n", "\r", " "}
                or key_name in {"KEY_ESCAPE", "KEY_ENTER"}
                or key_text == "\x1b"
            ):
                return self._toggle_tui_help()
            return False

        if self.unified_process_filter_editing:
            return self._handle_unified_process_filter_key(
                key_text,
                key_name,
            )

        if key_text == "?":
            return self._toggle_tui_help()
        if key_text == "m":
            return self._open_tui_menu("actions")

        if key_text == "C":
            return self._toggle_theme()

        if key_lower == "q":
            self.quit_flag.set()
            return True

        if key_name == "KEY_ESCAPE" or key_text == "\x1b":
            if self._pending_process_signal:
                self._pending_process_signal = None
                self._set_process_action_notice("Signal cancelled", "yellow")
                self._request_ui_refresh()
                return True
            return self._clear_unified_process_filter()

        enter_pressed = (
            key_name == "KEY_ENTER"
            or key_text in {"\n", "\r"}
        )
        if enter_pressed and self._pending_process_signal:
            return self._confirm_process_signal()

        if key_lower == "v":
            self._toggle_display_mode()
            return True
        if self.display_mode == self.DISPLAY_MODE_UNIFIED:
            if key_lower == "d":
                self.unified_detailed = not self.unified_detailed
                self.unified_active_pane = "gpu"
                self.unified_process_panel_hidden = False
                self._reset_process_selection()
                self._apply_view_change()
                return True
            if key_lower == "g":
                self.unified_group_by_node = not self.unified_group_by_node
                self._apply_view_change()
                return True
            if key_lower == "u":
                self.hide_unsupported = not self.hide_unsupported
                self._apply_view_change()
                return True
            if key_lower == "s":
                self._cycle_unified_sort_mode()
                return True
            if key_lower == "f":
                self._cycle_unified_filter_mode()
                return True
            if key_text == "/":
                return self._start_unified_process_filter()
            if key_text == "O":
                entered = self._enter_unified_process_pane()
                changed = self._set_unified_process_sort(
                    self._get_unified_process_sort_mode(),
                    toggle=True,
                )
                return changed or entered
            if key_lower == "o" or key_name == "KEY_F6":
                entered = self._enter_unified_process_pane()
                return self._cycle_unified_process_sort() or entered
            if key_text in {"+", "="}:
                entered = self._enter_unified_process_pane()
                return self._adjust_unified_process_rows(1) or entered
            if key_text == "-":
                entered = self._enter_unified_process_pane()
                return self._adjust_unified_process_rows(-1) or entered
            if key_lower == "p":
                return self._toggle_unified_process_panel()
            if key_text == "i" or key_name == "KEY_F7":
                return self._request_process_signal("INT")
            if key_text == "T" or key_name == "KEY_F8":
                return self._request_process_signal("TERM")
            if key_text == "K" or key_name == "KEY_F9":
                return self._request_process_signal("KILL")
            if key_lower == "t":
                if self._unified_gpu_count == 0:
                    return False
                self.unified_show_trends = not self.unified_show_trends
                self._apply_view_change()
                return True
            if key_name in {"KEY_TAB", "KEY_BTAB"} or key_text == "\t":
                return self._toggle_unified_active_pane()
            if key_name == "KEY_RIGHT" or key_lower == "l":
                return self._enter_unified_process_pane()
            if key_name == "KEY_LEFT" or key_lower == "h":
                return self._set_unified_active_pane("gpu")
            if key_text == "[":
                entered = self._enter_unified_process_pane()
                return self._scroll_unified_command(-1) or entered
            if key_text == "]":
                entered = self._enter_unified_process_pane()
                return self._scroll_unified_command(1) or entered

            process_pane = (
                self.unified_active_pane == "process"
                and self._process_panel_visible()
            )
            if key_lower == "j" or key_name == "KEY_DOWN":
                if process_pane:
                    return self._move_unified_process_selection(1)
                return self._move_unified_selection(1)
            if key_lower == "k" or key_name == "KEY_UP":
                if process_pane:
                    return self._move_unified_process_selection(-1)
                return self._move_unified_selection(-1)
            if key_name in {"KEY_NPAGE", "KEY_PAGEDOWN", "KEY_PGDOWN"}:
                if process_pane:
                    return self._move_unified_process_selection(5)
                return self._move_unified_selection(max(1, self._unified_page_size))
            if key_name in {"KEY_PPAGE", "KEY_PAGEUP", "KEY_PGUP"}:
                if process_pane:
                    return self._move_unified_process_selection(-5)
                return self._move_unified_selection(-max(1, self._unified_page_size))
            if enter_pressed or key_text == " ":
                return self._enter_unified_process_pane()
            return False

        if key_lower == "j" or key_name == "KEY_DOWN":
            return self._move_node_selection(1)
        elif key_lower == "k" or key_name == "KEY_UP":
            return self._move_node_selection(-1)
        elif enter_pressed or key_text == " ":
            return self._toggle_server_expansion(self.selected_server)
        elif key_lower == "a":
            disabled = self._toggle_disabled_servers
            self.expanded_servers = {
                index for index in range(len(self.pool)) if index not in disabled
            }
            self._expansion_touched = True
            self._request_ui_refresh()
            return True
        elif key_lower == "c":
            self.expanded_servers.clear()
            self._expansion_touched = True
            self._request_ui_refresh()
            return True
        return False

    def _start_mouse_reporting(self):
        """Ask the terminal for SGR mouse reports, when the user wants them."""
        self._mouse_reporting = False
        if not self.mouse_enabled:
            return
        try:
            if not sys.stdout.isatty():
                return
            set_mouse_reporting(self.term, True)
            self._mouse_reporting = True
        except Exception as error:
            logging.debug(msg=f"Failed to enable mouse reporting: {error}")

    def _stop_mouse_reporting(self):
        """Always restore normal terminal selection behaviour on the way out."""
        if not self._mouse_reporting:
            return
        self._mouse_reporting = False
        try:
            set_mouse_reporting(self.term, False)
        except Exception:
            pass

    def _request_secret(self, prompt):
        """Workers request credentials; only the UI thread reads the terminal."""
        ready = threading.Event()
        response = []
        self._secret_requests.put((prompt, response, ready))
        self.refresh_needed.set()
        while not self.quit_flag.is_set():
            if ready.wait(0.1):
                if response:
                    return response.pop()
                break
        raise EOFError("Authentication cancelled")

    def _answer_secret_request(self):
        try:
            prompt, response, ready = self._secret_requests.get_nowait()
        except Empty:
            return
        self._stop_mouse_reporting()
        self._cbreak_context.__exit__(None, None, None)
        try:
            # No keyboard listener or repaint can consume/echo the secret.
            print(self.term.normal_cursor + self.term.move_y(self.term.height - 1) + self.term.clear_eol)
            response.append(_prompt_secret(prompt))
        except EOFError:
            pass
        finally:
            ready.set()
            sys.stdout.write(self.term.hide_cursor)
            self._cbreak_context = self.term.cbreak()
            self._cbreak_context.__enter__()
            self._start_mouse_reporting()
            self._tui_diff_screen = None
            self.refresh_needed.set()

    def _publish_node_sample(self, idx, sample, *, phase, started, timings, record=False, sampled_at=None):
        stats, info, users, processes = sample
        info = dict(info)
        now = time.time()
        sampled_at = now if sampled_at is None else sampled_at
        info["_monitor"] = {
            "phase": phase,
            "started": started,
            "updated": sampled_at,
            "timings": dict(timings),
        }
        sample = (stats, info, users, processes)
        # Record only the node that produced this sample, never replay cached
        # samples from other nodes into their history.
        if record and not self.quit_flag.is_set():
            self._record_unified_gpu_history({
                idx: (stats, info),
                "_nvidb": self._sample_metadata({idx: sample}),
            }, timestamp=sampled_at)
        with self._cache_lock:
            if self.quit_flag.is_set():
                return
            self._node_samples[idx] = sample
            raw = {i: (value[0], value[1]) for i, value in self._node_samples.items()}
            raw["_nvidb"] = self._sample_metadata(self._node_samples)
            self.cached_raw_stats = raw
            # Formatting belongs to the UI thread, not concurrent workers.
            self.cached_stats = [""] * len(self.pool)
            self._last_update_time = now
            self._last_fetch_duration = None
            self._last_fetch_error = None
        self.refresh_needed.set()

    def _set_node_phase(self, idx, phase):
        with self._cache_lock:
            if self.quit_flag.is_set():
                return
            stats, info, users, processes = self._node_samples.get(
                idx, (pd.DataFrame(), {"loading": True}, {}, {})
            )
            info = dict(info)
            monitor = dict(info.get("_monitor", {}))
            monitor.update(phase=phase, started=time.time())
            info["_monitor"] = monitor
            self._node_samples[idx] = (stats, info, users, processes)
            raw = dict(self.cached_raw_stats)
            raw[idx] = (stats, info)
            self.cached_raw_stats = raw
        # Phase-only transitions do not change the normal frame. Actual data
        # publications still wake the UI immediately.
        if self.debug:
            self.refresh_needed.set()

    def _refresh_node(self, idx, interval_seconds):
        """One serial worker per node; a slow host never gates other hosts."""
        client = self.pool[idx]
        timings = {}
        if isinstance(client, RemoteClient):
            client.secret_prompt = self._request_secret
        try:
            if not client.connected:
                self._set_node_phase(idx, "connecting")
                start = time.monotonic()
                try:
                    client.connect(announce=False)
                except Exception as error:
                    client._set_connect_error(str(error), error_type="connect")
                timings["ssh"] = time.monotonic() - start
            first_sample = True
            while not self.quit_flag.is_set():
                started = time.time()
                try:
                    self._set_node_phase(idx, "collecting")
                    start = time.monotonic()
                    base = client.get_full_gpu_info(include_advanced=False)
                    sampled_at = time.time()
                    timings["gpu"] = time.monotonic() - start
                    if self.quit_flag.is_set():
                        break
                    # The first GPU rows need neither DCGM, system stats nor ps.
                    if first_sample:
                        quick = self._collect_client_gpu_info(client, snapshot=base, enrich=False)
                        self._publish_node_sample(
                            idx, quick, phase="details", started=started, timings=timings, sampled_at=sampled_at,
                        )
                        first_sample = False
                    self._set_node_phase(idx, "details")
                    start = time.monotonic()
                    sample = self._collect_client_gpu_info(
                        client, snapshot=base,
                        want_process_details=self._process_panel_visible(),
                    )
                    stats, info, users, processes = sample
                    # Do not even probe DCGM in the default live monitor.
                    if self.dcgm:
                        _, info = client._add_advanced_gpu_info(stats, info)
                    stats, info = self._apply_gpu_allowlist(client, stats, info)
                    timings["details"] = time.monotonic() - start
                    phase = "error" if info.get("error") else "ready"
                    self._publish_node_sample(
                        idx, (stats, info, users, processes),
                        phase=phase, started=started, timings=timings, record=True, sampled_at=sampled_at,
                    )
                except Exception as error:
                    self._publish_node_sample(
                        idx, (pd.DataFrame(), {
                            "error": f"{type(error).__name__}: {error}",
                            "error_type": "collect",
                        }, {}, {}),
                        phase="error", started=started, timings=timings,
                    )
                if self.quit_flag.wait(max(0.05, interval_seconds)):
                    break
        finally:
            if isinstance(client, RemoteClient):
                client.secret_prompt = _prompt_secret
                with client._connect_lock:
                    client._close_link_locked()

    def _background_refresh(self, interval_seconds: float = 1.0):
        """Start independent daemon workers; quitting never waits for SSH."""
        if self._node_workers:
            return
        # One worker per node also prevents overlapping commands on the same
        # persistent NVML channel. There is no all-nodes completion barrier.
        for idx in range(len(self.pool)):
            if self.quit_flag.is_set():
                break
            worker = threading.Thread(
                target=self._refresh_node, args=(idx, interval_seconds),
                name=f"nvidb-node-{idx}", daemon=True,
            )
            self._node_workers.append(worker)
            worker.start()
    
    def print_refresh(self):
        """Real-time GPU status display with global keyboard monitoring"""
        print("GPU monitoring starting...")
        print("Controls:")
        print("   j/k or Up/Down : Move the active selection")
        print("   PgUp/PgDn      : Change unified GPU page")
        print("   Enter/Right    : Enter processes; Enter confirms signals")
        print("   p              : Show/hide the unified process pane")
        print("   Space          : Enter processes or toggle per-node details")
        print("   /              : Filter processes by PID, user, or command")
        print("   o / F6 / O     : Cycle process sort / reverse direction")
        print("   + / -          : Show more/fewer process rows")
        print("   ?              : Context-sensitive help")
        print("   a              : Expand all")
        print("   c              : Collapse all")
        print("   v              : Switch unified/per-node view")
        print("   d              : Toggle unified single-line/detailed rows")
        print("   s              : Cycle unified sorting")
        print("   f              : Cycle unified GPU filters")
        print("   g              : Toggle unified per-node grouping")
        print("   u              : Show/hide nodes without GPU support")
        print("   t              : Toggle GPU and selected-process history")
        print("   Tab/Left       : Switch panes or return to GPU selection")
        print("   [ / ]          : Page through a long wrapped command")
        print("   i / T / K      : Confirmed SIGINT / SIGTERM / SIGKILL")
        print("   Mouse          : Select GPUs/processes, use actions, wheel to scroll")
        print("   q              : Quit")
        print("=" * 60)

        # Use cbreak context in main thread for proper cleanup on Ctrl+C
        self._cbreak_context = self.term.cbreak()
        self._cbreak_context.__enter__()
        self._start_mouse_reporting()

        # Background threads (auto-reconnect, etc.) call logging.error/warning/info
        # directly, which write straight to the console. Any such line printed
        # while print_stats is doing cursor-positioned diff rendering corrupts the
        # screen. Each node's own error is already rendered in-frame via
        # _format_error_panel, so muting the console logger here loses nothing.
        logging.disable(logging.CRITICAL)

        try:
            with self.term.hidden_cursor():
                # Draw before starting any network I/O. Input and painting have
                # one owner, so credential prompts cannot race a key listener.
                self.print_stats(use_cache=True)
                self._background_refresh()
                parser = MouseSequenceParser()
                next_draw = time.monotonic() + 1.0

                while not self.quit_flag.is_set():
                    key = self.term.inkey(timeout=0.1)
                    if key:
                        events, keys = parser.feed(key)
                        for event in events:
                            self._handle_mouse_event(event)
                    else:
                        keys = parser.flush()
                    for pending in keys:
                        self._handle_keypress(pending)
                    if self.quit_flag.is_set():
                        break
                    if self.refresh_needed.is_set() or time.monotonic() >= next_draw:
                        self.refresh_needed.clear()
                        self.print_stats(use_cache=True)
                        next_draw = time.monotonic() + 1.0
                    elif self._tui_hover_dirty:
                        if not self._paint_tui_hover():
                            self.print_stats(use_cache=True)
                    self._answer_secret_request()
                
        except KeyboardInterrupt:
            pass  # Silently exit on Ctrl+C
        except Exception as e:
            print(f"\n\n Error occurred: {e}")
        finally:
            self.quit_flag.set()  # Ensure thread exits
            self._stop_mouse_reporting()
            logging.disable(logging.NOTSET)
            # Explicitly exit cbreak mode to restore terminal state
            try:
                self._cbreak_context.__exit__(None, None, None)
            except Exception:
                pass
            # Print newline to ensure clean prompt
            print()

    def print_once(self):
        """Print GPU stats once and exit (no TUI loop)"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        terminal_width = _terminal_size().columns
        if self.compact:
            separator_width = min(80, terminal_width)
        else:
            separator_width = max(20, terminal_width)

        # Get stats
        stats_list, raw_stats_by_client = self.get_client_gpus_info(return_raw=True)

        node_indices = self._compute_node_indices(raw_stats_by_client)
        self._node_listed_indices = node_indices
        server_count = len(node_indices)
        server_label = "Server" if server_count == 1 else "Servers"
        for line in self._format_local_strip(terminal_width, raw_stats_by_client):
            print(line)
        print(f"Time: {current_time} | {server_label}: {server_count}")
        print("-" * separator_width)

        meta = raw_stats_by_client.get("_nvidb") or {}
        user_memory_by_client = meta.get("user_memory_by_client", {}) or {}
        global_user_memory = meta.get("user_memory_global", {}) or {}

        for position, idx in enumerate(node_indices):
            client = self.pool[idx]
            stats_info = stats_list[idx] if idx < len(stats_list) else ""
            stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))

            # Build summary
            summary = self._format_server_summary(self._get_server_summary_data(stats, system_info))

            # Print header with summary
            print(f"[{position + 1}] {client.description}  {summary}")

            # Print the full stats table
            print(stats_info)
            user_summary = user_memory_by_client.get(idx) or {}
            if user_summary:
                print(f"Users: {self._format_user_memory_totals(user_summary)}")
            print()  # Add spacing after each server

        if global_user_memory:
            print(f"Users (all nodes): {self._format_user_memory_totals(global_user_memory)}")
