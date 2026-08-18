from typing import Optional
from blessed import Terminal
import logging
import re
import sys
import os
import subprocess
import getpass
import json
import socket
import textwrap
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import deque
import threading

import paramiko
from paramiko import AuthenticationException
from paramiko.ssh_exception import NoValidConnectionsError, PasswordRequiredException
import pandas as pd
from termcolor import colored
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
    DISABLE_SEQUENCE as MOUSE_DISABLE_SEQUENCE,
    ENABLE_SEQUENCE as MOUSE_ENABLE_SEQUENCE,
    MouseSequenceParser,
)
from .nvml import PynvmlCollector, make_nvml_agent_command
from .ssh_proxy import open_proxyjump_socket
from .tui_theme import (
    DiffScreen,
    display_width,
    fit,
    frame_bottom,
    frame_separator,
    frame_top,
)
from .utils import (
    units_from_str,
    extract_numbers,
    extract_value_and_unit,
    format_bandwidth,
    format_pcie_link,
    get_pcie_load_color,
    get_utilization_color,
    parse_link_number,
    pcie_link_capacity_kib_per_second,
)


def parse_leading_float(value):
    """Return the first numeric component of a text field, or None."""
    if value is None:
        return None
    numbers = extract_numbers(str(value))
    if not numbers:
        return None
    try:
        return float(numbers[0])
    except (TypeError, ValueError):
        return None


class BaseClient(ABC):
    """Base class for both RemoteClient and LocalClient with common functionality"""
    
    def __init__(self):
        self.host = None
        self.port = None
        self.username = None
        self.description = None
        self.connected = False
        self.last_connect_error = None
        self.last_error_type = None
        # A host that answered "no DCGM here" stays that way, and core count
        # never changes; remembering both saves one remote round trip each.
        self._dcgm_probe_failed = False
        self._cpu_cores = None

    def _set_connect_error(self, message: str, error_type: str = "error"):
        self.connected = False
        self.last_connect_error = message
        self.last_error_type = error_type

    def ensure_connected(self) -> bool:
        """Return whether this client can serve a query right now.

        Transports that can break mid-session (SSH) override this to revive
        themselves; a local client is either usable or was never usable.
        """
        return self.connected is not False

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the client (remote or local)"""
        pass
    
    @abstractmethod
    def execute_command(self, command: str) -> str:
        """Execute a command and return the output"""
        pass

    def _chunked(self, items, chunk_size: int):
        for i in range(0, len(items), chunk_size):
            yield items[i : i + chunk_size]

    @staticmethod
    def _unique_pids(pids):
        """Clean and de-duplicate PIDs (preserve order)."""
        seen = set()
        unique_pids = []
        for pid in pids or ():
            pid_str = str(pid).strip()
            if not pid_str or pid_str == "N/A":
                continue
            if pid_str in seen:
                continue
            seen.add(pid_str)
            unique_pids.append(pid_str)
        return unique_pids

    def get_pid_user_map(self, pids, chunk_size: int = 128):
        """Batch query pid -> username mapping via a single ps call (or a few chunked calls)."""
        unique_pids = self._unique_pids(pids)
        if not unique_pids:
            return {}

        pid_to_user = {}
        for pid_chunk in self._chunked(unique_pids, chunk_size):
            pid_list = ",".join(pid_chunk)
            output = self.execute_command(f"ps -o pid=,user= -p {pid_list}")
            if not isinstance(output, str) or not output.strip():
                continue
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                pid_str, username = parts[0].strip(), parts[1].strip()
                if pid_str and username:
                    pid_to_user[pid_str] = username

        return pid_to_user

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
            if not isinstance(output, str) or not output.strip():
                continue
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, len(fields) - 1)
                if len(parts) < len(fields):
                    continue
                values = dict(zip(fields, parts))
                pid_str = values.get("pid", "").strip()
                if not pid_str:
                    continue
                details[pid_str] = {
                    "username": values.get("user") or None,
                    "cpu_percent": parse_leading_float(values.get("pcpu")),
                    "mem_percent": parse_leading_float(values.get("pmem")),
                    "rss_kb": parse_leading_float(values.get("rss")),
                    "elapsed": (values.get("etime") or "").strip() or None,
                    "state": (values.get("stat") or "").strip() or None,
                    "threads": parse_leading_float(values.get("nlwp")),
                    "command": (values.get("args") or "").strip() or None,
                }
        return details

    def get_pid_process_info(self, pids, chunk_size: int = 128):
        """Batch query htop-style details (CPU, RSS, state, full command) per pid."""
        unique_pids = self._unique_pids(pids)
        if not unique_pids:
            return {}

        for fields in self.PROCESS_DETAIL_FIELD_SETS:
            details = self._query_pid_details(unique_pids, fields, chunk_size)
            if details:
                return details
        return {}

    def get_full_gpu_info(self):
        """Get GPU information (base + optional advanced profiling).

        - Base table: NVML, with `nvidia-smi -q -x` used only when NVML is unavailable.
        - Advanced table (DCGM profiling): fetched via DCGM python bindings when available.
        """
        # A dropped link is revived here rather than reported as "no GPUs":
        # every refresh tick gets one (rate-limited) chance to reconnect.
        if not self.ensure_connected():
            error_message = self.last_connect_error or "Not connected"
            error_type = self.last_error_type or "error"
            return pd.DataFrame(), {"error": error_message, "error_type": error_type}

        def safe_get_text(element, path, default="N/A"):
            """Safely get text from XML element, return default if not found."""
            if element is None:
                return default
            found = element.find(path)
            return found.text if found is not None else default

        def _format_percent(value):
            try:
                return f"{int(round(float(value)))} %"
            except (TypeError, ValueError):
                return "N/A"

        def _format_mib(value):
            try:
                return f"{int(round(float(value) / (1024 * 1024)))} MiB"
            except (TypeError, ValueError):
                return "N/A"

        def _format_kib_per_second(value):
            try:
                return f"{int(round(float(value)))} KB/s"
            except (TypeError, ValueError):
                return "N/A"

        def _format_watts(value):
            try:
                watts = float(value) / 1000.0
            except (TypeError, ValueError):
                return "N/A"
            return f"{watts:.2f} W"

        def _format_temperature(value):
            try:
                return f"{int(round(float(value)))} C"
            except (TypeError, ValueError):
                return "N/A"

        def _fetch_nvml_base():
            try:
                payload = self.query_nvml_snapshot()
            except Exception as error:
                logging.debug(msg=f"Failed to query NVML snapshot: {error}")
                return (
                    pd.DataFrame(),
                    {
                        "data_source": "nvml",
                        "nvml_error": f"{type(error).__name__}: {error}",
                    },
                    False,
                )

            if not isinstance(payload, dict) or payload.get("ok") is not True:
                error = (
                    payload.get("error", "NVML snapshot failed")
                    if isinstance(payload, dict)
                    else "NVML helper returned an invalid payload"
                )
                return (
                    pd.DataFrame(),
                    {"data_source": "nvml", "nvml_error": str(error)},
                    False,
                )

            gpus = payload.get("gpus")
            if not isinstance(gpus, list):
                return (
                    pd.DataFrame(),
                    {
                        "data_source": "nvml",
                        "nvml_error": "NVML snapshot did not contain a GPU list",
                    },
                    False,
                )

            rows = []
            for fallback_index, gpu in enumerate(gpus):
                if not isinstance(gpu, dict):
                    continue
                try:
                    gpu_index = int(gpu.get("gpu_index", fallback_index))
                except (TypeError, ValueError):
                    gpu_index = fallback_index

                processes = []
                for process in gpu.get("processes") or []:
                    if not isinstance(process, dict):
                        continue
                    used_memory = _format_mib(process.get("used_gpu_memory_bytes"))
                    processes.append(
                        {
                            "gpu_instance_id": process.get("gpu_instance_id"),
                            "compute_instance_id": process.get("compute_instance_id"),
                            "pid": process.get("pid"),
                            "type": process.get("type") or "N/A",
                            "process_name": process.get("process_name") or "N/A",
                            "used_memory": used_memory,
                            "username": process.get("username"),
                        }
                    )

                performance_state = gpu.get("performance_state")
                try:
                    power_state = f"P{int(performance_state)}"
                except (TypeError, ValueError):
                    power_state = "N/A"

                rows.append(
                    {
                        "gpu_index": gpu_index,
                        "product_name": gpu.get("name") or "Unknown GPU",
                        "product_architecture": gpu.get("architecture") or "N/A",
                        "tx_util": _format_kib_per_second(gpu.get("pcie_tx_kib_per_s")),
                        "rx_util": _format_kib_per_second(gpu.get("pcie_rx_kib_per_s")),
                        "fan_speed": _format_percent(gpu.get("fan_percent")),
                        "total": _format_mib(gpu.get("memory_total_bytes")),
                        "used": _format_mib(gpu.get("memory_used_bytes")),
                        "free": _format_mib(gpu.get("memory_free_bytes")),
                        "gpu_util": _format_percent(gpu.get("gpu_util_percent")),
                        "memory_util": _format_percent(gpu.get("memory_util_percent")),
                        "gpu_temp": _format_temperature(gpu.get("temperature_c")),
                        "power_state": power_state,
                        "power_draw": _format_watts(gpu.get("power_usage_mw")),
                        "current_power_limit": _format_watts(gpu.get("power_limit_mw")),
                        "pcie_link_gen_current": gpu.get("pcie_link_gen_current"),
                        "pcie_link_width_current": gpu.get("pcie_link_width_current"),
                        "pcie_link_gen_max": gpu.get("pcie_link_gen_max"),
                        "pcie_link_width_max": gpu.get("pcie_link_width_max"),
                        "processes": processes,
                    }
                )

            backend = payload.get("backend") or "unknown"
            detail = "nvidia-ml-py" if backend == "pynvml" else "libnvidia-ml.so.1"
            system_info = {
                "driver_version": payload.get("driver_version") or "N/A",
                "cuda_version": payload.get("cuda_version") or "N/A",
                "attached_gpus": str(len(rows)),
                "data_source": "nvml",
                "data_source_detail": detail,
                "nvml_backend": backend,
            }
            return pd.DataFrame(rows), system_info, True

        def _fetch_nvidia_smi_base():
            try:
                result = self.execute_command("nvidia-smi -q -x")
                if not result or not result.strip() or not result.strip().startswith("<?xml"):
                    return pd.DataFrame(), {"data_source": "unsupported", "unsupported": True}

                root = ET.fromstring(result)
                driver_version = safe_get_text(root, "driver_version", "N/A")
                cuda_version = safe_get_text(root, "cuda_version", "N/A")
                attached_gpus = safe_get_text(root, "attached_gpus", "0")
                gpus = root.findall("gpu")
                rows = []

                for gpu_index, gpu in enumerate(gpus):
                    product_name = safe_get_text(gpu, "product_name", "Unknown GPU")
                    product_architecture = safe_get_text(gpu, "product_architecture", "N/A")

                    pci = gpu.find("pci")
                    tx_util = safe_get_text(pci, "tx_util", "N/A")
                    rx_util = safe_get_text(pci, "rx_util", "N/A")
                    # nvidia-smi spells widths as "16x"; parse_link_number copes.
                    link_info = pci.find("pci_gpu_link_info") if pci is not None else None
                    link_gen_current = safe_get_text(
                        link_info, "pcie_gen/current_link_gen", None
                    )
                    link_gen_max = safe_get_text(link_info, "pcie_gen/max_link_gen", None)
                    link_width_current = safe_get_text(
                        link_info, "link_widths/current_link_width", None
                    )
                    link_width_max = safe_get_text(
                        link_info, "link_widths/max_link_width", None
                    )
                    fan_speed = safe_get_text(gpu, "fan_speed", "N/A")

                    fb_memory_usage = gpu.find("fb_memory_usage")
                    total = safe_get_text(fb_memory_usage, "total", "N/A")
                    used = safe_get_text(fb_memory_usage, "used", "N/A")
                    free = safe_get_text(fb_memory_usage, "free", "N/A")

                    utilization = gpu.find("utilization")
                    gpu_util = safe_get_text(utilization, "gpu_util", "N/A")
                    memory_util = safe_get_text(utilization, "memory_util", "N/A")

                    temperature = gpu.find("temperature")
                    gpu_temp = safe_get_text(temperature, "gpu_temp", "N/A")

                    gpu_power_readings = gpu.find("gpu_power_readings")
                    # On newer drivers (e.g. WSL/591.x) the P-state under
                    # <gpu_power_readings><power_state> is deprecated and returns
                    # "Requested functionality has been deprecated"; the real perf
                    # state now lives in <gpu><performance_state>.
                    power_state = safe_get_text(gpu_power_readings, "power_state", "N/A")
                    if not power_state or power_state == "N/A" or "deprecated" in power_state.lower():
                        power_state = safe_get_text(gpu, "performance_state", "N/A")

                    power_draw = safe_get_text(gpu_power_readings, "power_draw", None)
                    if power_draw is None or power_draw == "N/A":
                        power_draw = safe_get_text(gpu_power_readings, "instant_power_draw", "N/A")

                    current_power_limit = safe_get_text(gpu_power_readings, "current_power_limit", "N/A")

                    processes = gpu.find("processes")

                    rows.append(
                        {
                            "gpu_index": gpu_index,
                            "product_name": product_name,
                            "product_architecture": product_architecture,
                            "tx_util": tx_util,
                            "rx_util": rx_util,
                            "fan_speed": fan_speed,
                            "total": total,
                            "used": used,
                            "free": free,
                            "gpu_util": gpu_util,
                            "memory_util": memory_util,
                            "gpu_temp": gpu_temp,
                            "power_state": power_state,
                            "power_draw": power_draw,
                            "current_power_limit": current_power_limit,
                            "pcie_link_gen_current": link_gen_current,
                            "pcie_link_width_current": link_width_current,
                            "pcie_link_gen_max": link_gen_max,
                            "pcie_link_width_max": link_width_max,
                            "processes": processes,
                        }
                    )

                stats = pd.DataFrame(rows)
                system_info = {
                    "driver_version": driver_version,
                    "cuda_version": cuda_version,
                    "attached_gpus": attached_gpus,
                    "data_source": "nvidia-smi",
                    "data_source_detail": "nvidia-smi -q -x",
                }
                return stats, system_info

            except Exception as e:
                logging.debug(msg=f"Failed to get base GPU info via nvidia-smi: {e}")
                return pd.DataFrame(), {"data_source": "unsupported", "unsupported": True}

        stats, system_info, nvml_ok = _fetch_nvml_base()
        if not nvml_ok:
            nvml_error = system_info.get("nvml_error", "NVML unavailable")
            stats, system_info = _fetch_nvidia_smi_base()
            system_info["fallback_reason"] = nvml_error
            if system_info.get("data_source") == "unsupported":
                system_info["nvml_error"] = nvml_error
        if stats.empty:
            # The link can die mid-fetch. Say so, instead of letting one tick
            # look like a machine that never had an NVIDIA GPU.
            if self.connected is False:
                return pd.DataFrame(), {
                    "error": self.last_connect_error or "Not connected",
                    "error_type": self.last_error_type or "error",
                }
            return stats, system_info

        def fmt_percent(value) -> str:
            if value is None:
                return "N/A"
            text = str(value).strip()
            if not text or text in {"N/A", "-"}:
                return "N/A"
            try:
                num = float(text)
            except Exception:
                # Already formatted?
                return text
            # Profiling fields may be 0-1 ratios; normalize to 0-100%.
            if 0.0 <= num <= 1.0:
                num *= 100.0
            if num < 0:
                num = 0.0
            if num > 100:
                num = 100.0
            return f"{num:.0f}%"

        def fmt_temp_c(value) -> str:
            if value is None:
                return "N/A"
            try:
                num = float(value)
            except Exception:
                return "N/A"
            return f"{num:.0f} C"

        def fmt_watts(value) -> str:
            if value is None:
                return "N/A"
            try:
                num = float(value)
            except Exception:
                return "N/A"
            return f"{num:.1f} W" if abs(num - int(num)) > 0.01 else f"{int(num)} W"

        # Fetch advanced profiling metrics via DCGM (optional). A host without
        # DCGM answers the same way on every refresh, so the failure is cached
        # instead of paying the remote python startup per tick forever.
        if self._dcgm_probe_failed:
            return stats, system_info
        try:
            dcgm_cmd = make_dcgm_snapshot_command()
            dcgm_out = self.execute_command(dcgm_cmd)
            payload = None
            if isinstance(dcgm_out, str) and dcgm_out.strip():
                for line in reversed(dcgm_out.splitlines()):
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        payload = json.loads(line)
                        break
                    except Exception:
                        continue

            if isinstance(payload, dict) and payload.get("ok") is True:
                gpus = payload.get("gpus") or []
                if isinstance(gpus, list) and gpus:
                    adv_rows = []

                    for gpu in gpus:
                        try:
                            gpu_index = int(gpu.get("gpu_index"))
                        except Exception:
                            gpu_index = len(adv_rows)

                        sm_active = gpu.get("sm_active")
                        sm_occupancy = gpu.get("sm_occupancy")
                        tensor_active = gpu.get("tensor_active")
                        fp64_active = gpu.get("fp64_active")
                        fp32_active = gpu.get("fp32_active")
                        fp16_active = gpu.get("fp16_active")
                        int_active = gpu.get("int_active")
                        dram_active = gpu.get("dram_active")

                        def pct_num(v):
                            if v is None:
                                return None
                            try:
                                x = float(v)
                            except Exception:
                                return None
                            if 0.0 <= x <= 1.0:
                                x *= 100.0
                            if x < 0:
                                x = 0.0
                            if x > 100:
                                x = 100.0
                            return x

                        pipe_vals = [
                            pct_num(tensor_active),
                            pct_num(fp64_active),
                            pct_num(fp32_active),
                            pct_num(fp16_active),
                            pct_num(int_active),
                        ]
                        pipe_vals = [v for v in pipe_vals if v is not None]
                        instr_throughput = max(pipe_vals) if pipe_vals else None
                        sol = None
                        dram_pct = pct_num(dram_active)
                        if instr_throughput is not None and dram_pct is not None:
                            sol = max(instr_throughput, dram_pct)
                        elif instr_throughput is not None:
                            sol = instr_throughput
                        elif dram_pct is not None:
                            sol = dram_pct

                        adv_rows.append(
                            {
                                "GPU": gpu_index,
                                "sm_active": fmt_percent(sm_active),
                                "sm_occupancy": fmt_percent(sm_occupancy),
                                "instruction_throughput": fmt_percent(instr_throughput),
                                "sol": fmt_percent(sol),
                                "tensor_active": fmt_percent(tensor_active),
                                "fp64_active": fmt_percent(fp64_active),
                                "fp32_active": fmt_percent(fp32_active),
                                "fp16_active": fmt_percent(fp16_active),
                            }
                        )

                    adv_df = pd.DataFrame(adv_rows)
                    advanced_supported = False
                    try:
                        if not adv_df.empty:
                            metric_cols = [c for c in adv_df.columns if c != "GPU"]
                            for col in metric_cols:
                                values = adv_df[col].astype(str).tolist()
                                if any(v not in {"N/A", "-", ""} for v in values):
                                    advanced_supported = True
                                    break
                    except Exception:
                        advanced_supported = False

                    system_info["advanced_source"] = "dcgm"
                    system_info["advanced_source_detail"] = payload.get("connection") or "N/A"
                    system_info["advanced_metrics"] = adv_df
                    system_info["advanced_supported"] = advanced_supported
            else:
                self._dcgm_probe_failed = True
        except Exception:
            # DCGM is optional; ignore failures.
            pass
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
                cpu_output = self.execute_command("nproc")
                if cpu_output and cpu_output.strip().isdigit():
                    self._cpu_cores = int(cpu_output.strip())
                else:
                    # Try macOS sysctl
                    cpu_output = self.execute_command("sysctl -n hw.ncpu")
                    if cpu_output and cpu_output.strip().isdigit():
                        self._cpu_cores = int(cpu_output.strip())
            if self._cpu_cores is not None:
                result["cpu_cores"] = self._cpu_cores

            # Get CPU utilization - try Linux first
            cpu_usage_output = self.execute_command(
                "grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage}'"
            )
            if cpu_usage_output and cpu_usage_output.strip():
                try:
                    result["cpu_percent"] = float(cpu_usage_output.strip())
                except ValueError:
                    pass
            else:
                # Try macOS - use top for CPU usage (user + sys)
                cpu_usage_output = self.execute_command(
                    "top -l 1 -n 0 | grep 'CPU usage' | awk '{print $3, $5}' | tr -d '%,'"
                )
                if cpu_usage_output and cpu_usage_output.strip():
                    try:
                        parts = cpu_usage_output.strip().split()
                        if len(parts) >= 2:
                            user_cpu = float(parts[0])
                            sys_cpu = float(parts[1])
                            result["cpu_percent"] = user_cpu + sys_cpu
                    except ValueError:
                        pass

            # Get memory info - try Linux first
            mem_output = self.execute_command(
                "grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo"
            )
            if mem_output and mem_output.strip():
                mem_info = {}
                for line in mem_output.strip().split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        # Extract numeric value (in kB)
                        value = value.strip().split()[0]
                        try:
                            mem_info[key.strip()] = int(value)
                        except ValueError:
                            pass

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
                if mem_total_output and mem_total_output.strip().isdigit():
                    mem_total_bytes = int(mem_total_output.strip())
                    result["mem_total_gb"] = mem_total_bytes / (1024 * 1024 * 1024)

                    # Get memory usage from vm_stat on macOS
                    vm_stat_output = self.execute_command("vm_stat")
                    if vm_stat_output:
                        page_size = 4096  # Default page size
                        pages_free = 0
                        pages_inactive = 0
                        pages_speculative = 0

                        for line in vm_stat_output.split("\n"):
                            if "page size of" in line:
                                try:
                                    page_size = int(line.split()[-2])
                                except (ValueError, IndexError):
                                    pass
                            elif "Pages free:" in line:
                                try:
                                    pages_free = int(line.split()[-1].rstrip('.'))
                                except (ValueError, IndexError):
                                    pass
                            elif "Pages inactive:" in line:
                                try:
                                    pages_inactive = int(line.split()[-1].rstrip('.'))
                                except (ValueError, IndexError):
                                    pass
                            elif "Pages speculative:" in line:
                                try:
                                    pages_speculative = int(line.split()[-1].rstrip('.'))
                                except (ValueError, IndexError):
                                    pass

                        available_bytes = (pages_free + pages_inactive + pages_speculative) * page_size
                        result["mem_used_gb"] = (mem_total_bytes - available_bytes) / (1024 * 1024 * 1024)

                # Get swap info on macOS
                swap_output = self.execute_command("sysctl -n vm.swapusage")
                if swap_output:
                    # Parse more carefully
                    parts = swap_output.split()
                    for i, part in enumerate(parts):
                        if part == "total" and i + 2 < len(parts):
                            try:
                                val = parts[i + 2].rstrip("M")
                                result["swap_total_gb"] = float(val) / 1024
                            except (ValueError, IndexError):
                                pass
                        elif part == "used" and i + 2 < len(parts):
                            try:
                                val = parts[i + 2].rstrip("M")
                                result["swap_used_gb"] = float(val) / 1024
                            except (ValueError, IndexError):
                                pass

        except Exception as e:
            logging.debug(msg=f"Failed to get system stats: {e}")

        return result

    def get_process_summary(self, gpu_stats=None, detailed: bool = False):
        """Get detailed GPU processes information with user summary.

        With `detailed`, one extra batched `ps` call per host enriches every GPU
        process with htop-style fields (CPU/MEM share, RSS, elapsed time, state,
        thread count) and the full command line.
        """
        def safe_get_text(element, path, default="N/A"):
            """Safely get text from XML element, return default if not found"""
            if element is None:
                return default
            found = element.find(path)
            return found.text if found is not None else default
        
        try:
            # Use provided gpu_stats if available, otherwise get fresh data
            if gpu_stats is None:
                stats, _ = self.get_full_gpu_info()
            else:
                stats = gpu_stats
            
            process_entries = []
            pids = []
            all_pids = []

            # First pass: extract processes and collect all pids
            for idx, row in stats.iterrows():
                processes_element = row.get("processes")
                try:
                    gpu_index = int(row.get("gpu_index", idx))
                except (TypeError, ValueError):
                    gpu_index = idx

                native_processes = (
                    processes_element if isinstance(processes_element, list) else None
                )
                if native_processes is not None:
                    extracted_processes = [
                        process for process in native_processes if isinstance(process, dict)
                    ]
                elif hasattr(processes_element, "findall"):
                    extracted_processes = []
                    for process_info in processes_element.findall("process_info"):
                        extracted_processes.append(
                            {
                                "pid": safe_get_text(process_info, "pid", "N/A"),
                                "process_name": safe_get_text(
                                    process_info, "process_name", "N/A"
                                ),
                                "used_memory": safe_get_text(
                                    process_info, "used_memory", "0 MiB"
                                ),
                                "gpu_instance_id": safe_get_text(
                                    process_info, "gpu_instance_id", "N/A"
                                ),
                                "compute_instance_id": safe_get_text(
                                    process_info, "compute_instance_id", "N/A"
                                ),
                                "type": safe_get_text(process_info, "type", "N/A"),
                                "username": safe_get_text(
                                    process_info, "username", None
                                ),
                            }
                        )
                else:
                    extracted_processes = []

                for process in extracted_processes:
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

                    process_entries.append(
                        {
                            "gpu_instance_id": (
                                gpu_instance_id
                                if gpu_instance_id is not None
                                else "N/A"
                            ),
                            "compute_instance_id": (
                                compute_instance_id
                                if compute_instance_id is not None
                                else "N/A"
                            ),
                            "pid": pid,
                            "type": process.get("type") or "N/A",
                            "process_name": process.get("process_name") or "N/A",
                            "used_memory": used_memory,
                            "username": username,
                            "gpu_index": gpu_index,
                            "_memory_value": memory_value,
                        }
                    )

            pid_details = self.get_pid_process_info(all_pids) if detailed else {}
            missing_user_pids = [
                pid_str
                for pid_str in pids
                if not (pid_details.get(pid_str) or {}).get("username")
            ]
            pid_to_user = self.get_pid_user_map(missing_user_pids)

            all_processes = []
            user_memory_summary = {}
            for entry in process_entries:
                pid_str = str(entry.get("pid", "")).strip()
                details = pid_details.get(pid_str) or {}
                username = entry.get("username")
                if not username or username == "N/A":
                    username = details.get("username")
                if not username or username == "N/A":
                    username = (
                        pid_to_user.get(pid_str, "N/A")
                        if pid_str and pid_str != "N/A"
                        else "N/A"
                    )

                all_processes.append(
                    {
                        "gpu_instance_id": entry.get("gpu_instance_id", "N/A"),
                        "compute_instance_id": entry.get("compute_instance_id", "N/A"),
                        "pid": entry.get("pid", "N/A"),
                        "type": entry.get("type", "N/A"),
                        "process_name": entry.get("process_name", "N/A"),
                        "used_memory": entry.get("used_memory", "0 MiB"),
                        "username": username,
                        "gpu_index": entry.get("gpu_index"),
                        "command": details.get("command") or entry.get("process_name", "N/A"),
                        "cpu_percent": details.get("cpu_percent"),
                        "mem_percent": details.get("mem_percent"),
                        "rss_kb": details.get("rss_kb"),
                        "elapsed": details.get("elapsed"),
                        "state": details.get("state"),
                        "threads": details.get("threads"),
                    }
                )

                memory_value = entry.get("_memory_value", 0) or 0
                if username != "N/A" and memory_value > 0:
                    user_memory_summary[username] = user_memory_summary.get(username, 0) + memory_value
            
            return all_processes, user_memory_summary
            
        except Exception as e:
            logging.error(msg=f"Failed to get process summary: {e}")
            return [], {}
    
    def format_user_memory_compact(self, user_summary):
        """Format user memory summary in compact format like 'qbs(23082M) gdm(4M)' sorted by memory usage desc"""
        if not user_summary:
            return ""
        
        formatted_users = []
        # Sort by memory usage in descending order
        for username, total_memory in sorted(user_summary.items(), key=lambda x: x[1], reverse=True):
            formatted_users.append(f"{username}({total_memory}M)")
        
        return " ".join(formatted_users)


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
        self.identityfile = getattr(server, "identityfile", None)
        self.proxyjump = getattr(server, "proxyjump", None)
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

    def _connect_client(self, *, batch_mode: bool = False, **kwargs):
        """Connect a fresh Paramiko client, optionally through ProxyJump.

        `batch_mode` forbids the ProxyJump helper from prompting, which matters
        on the background reconnect path where nothing can answer a prompt.
        """
        try:
            self.client.close()
        except Exception:
            pass
        self._close_proxy()
        self.client = self._make_ssh_client()

        kwargs.setdefault("timeout", self.CONNECT_TIMEOUT_SECONDS)
        kwargs.setdefault("banner_timeout", self.CONNECT_TIMEOUT_SECONDS)
        kwargs.setdefault("auth_timeout", self.CONNECT_TIMEOUT_SECONDS)

        proxy = None
        try:
            proxy = open_proxyjump_socket(
                self.proxyjump,
                self.host,
                self.port,
                connect_timeout=self.CONNECT_TIMEOUT_SECONDS,
                batch_mode=batch_mode,
            )
            if proxy is not None:
                kwargs["sock"] = proxy
            self.client.connect(**kwargs)
        except Exception:
            try:
                self.client.close()
            except Exception:
                pass
            if proxy is not None:
                try:
                    proxy.close()
                except Exception:
                    pass
            raise
        self._proxy = proxy
        try:
            transport = self.client.get_transport()
            if transport is not None:
                transport.set_keepalive(self.KEEPALIVE_SECONDS)
        except Exception:
            pass

    def _transport_alive(self) -> bool:
        try:
            transport = self.client.get_transport()
        except Exception:
            return False
        return bool(transport is not None and transport.is_active())

    def _close_link_locked(self):
        """Drop the SSH session (and the NVML agent riding on it)."""
        with self._nvml_agent_lock:
            self._close_nvml_agent_locked()
        try:
            self.client.close()
        except Exception:
            pass
        self._close_proxy()

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
                    self.connected = True
                    self.last_connect_error = None
                    self.last_error_type = None
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
            with self._nvml_agent_lock:
                self._close_nvml_agent_locked()
        except Exception:
            pass
        try:
            self.client.close()
            logging.info(msg=f"Connection to {self.host}:{self.port} closed.")
        except Exception:
            pass
        self._close_proxy()

    def _close_nvml_agent_locked(self):
        channel = self._nvml_agent_channel
        if channel is not None:
            try:
                if not channel.closed:
                    channel.sendall(b"quit\n")
            except Exception:
                pass

        for stream_name in (
            "_nvml_agent_stdin",
            "_nvml_agent_stdout",
            "_nvml_agent_stderr",
        ):
            stream = getattr(self, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            setattr(self, stream_name, None)

        if channel is not None:
            try:
                channel.close()
            except Exception:
                pass
        self._nvml_agent_channel = None

    def _start_nvml_agent_locked(self):
        stdin, stdout, stderr = self.client.exec_command(
            command=make_nvml_agent_command()
        )
        channel = stdout.channel
        channel.settimeout(10.0)
        self._nvml_agent_stdin = stdin
        self._nvml_agent_stdout = stdout
        self._nvml_agent_stderr = stderr
        self._nvml_agent_channel = channel

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
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
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
            "error": (
                f"{type(last_error).__name__}: {last_error}"
                if last_error is not None
                else "Remote NVML agent failed"
            ),
        }

    def _authenticate_with_password(
        self,
        max_attempts=3,
        *,
        prompt_only: bool = False,
        allow_prompt: bool = True,
    ) -> bool:
        """Attempt password authentication with retry limit.

        Args:
            max_attempts: Maximum number of password attempts (default: 3)
            prompt_only: Do not try any configured password; always prompt (default: False)
            allow_prompt: Allow asking the user; background reconnects cannot
                (nothing is reading the terminal) and fail instead.

        Returns:
            bool: True if authentication succeeds

        """
        for attempt in range(max_attempts):
            try:
                password = None
                if not prompt_only and self.password is not None and attempt == 0:
                    password = self.password
                elif not allow_prompt:
                    self._set_connect_error(
                        "Password required; restart nvidb to enter it",
                        error_type="auth",
                    )
                    return False
                else:
                    remaining = max_attempts - attempt
                    if attempt > 0:
                        logging.warning(msg=f"Authentication failed. {remaining} attempt(s) remaining.")
                        print(f"  ⚠ Authentication failed. {remaining} attempt(s) remaining.")
                    password = getpass.getpass(prompt=f"Enter password for {self.username}@{self.host}:{self.port} -> ")

                self._connect_client(
                    batch_mode=not allow_prompt,
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    password=password,
                    allow_agent=False,
                    look_for_keys=False,
                )
                self.connected = True
                self.last_connect_error = None
                self.last_error_type = None
                logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username}")
                return True
                    
            except AuthenticationException:
                if attempt == max_attempts - 1:
                    logging.error(
                        msg=f"Password authentication failed after {max_attempts} attempts on {self.description}"
                    )
                    self._set_connect_error("Password incorrect", error_type="auth")
                    return False
                continue
            except Exception as e:
                logging.error(msg=f"Password authentication error on {self.description}: {e}")
                self._set_connect_error(str(e), error_type="connect")
                return False

        self._set_connect_error("Password incorrect", error_type="auth")
        return False

    def connect(self, *, allow_prompt: bool = True, announce: bool = True) -> bool:
        """Open the SSH session.

        `allow_prompt` and `announce` are cleared by the background reconnect,
        which runs under a live TUI: nothing can answer a prompt and any print
        would land in the middle of the rendered screen.
        """
        if announce:
            print(f"Connecting to {self.host}:{self.port} as {self.username}")
        batch_mode = not allow_prompt
        # catch the OSError exception when the host is not reachable
        try:
            identityfile = None
            if self.auth in ("auto", "key") and self.identityfile:
                identityfile = os.path.expanduser(str(self.identityfile))
            if self.auth == "auto":
                # Auto mode: try key-based auth first, then password
                try:
                    self._connect_client(
                        batch_mode=batch_mode,
                        hostname=self.host,
                        port=self.port,
                        username=self.username,
                        allow_agent=False if identityfile else True,
                        look_for_keys=False if identityfile else True,
                        key_filename=identityfile,
                    )
                    self.connected = True
                    self.last_connect_error = None
                    self.last_error_type = None
                    logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username}")
                    return True
                except PasswordRequiredException:
                    if not identityfile:
                        logging.error(msg=f"Key requires passphrase on {self.description}, trying password...")
                        return self._authenticate_with_password(allow_prompt=allow_prompt)
                    if not allow_prompt:
                        self._set_connect_error(
                            "Key passphrase required; restart nvidb to enter it",
                            error_type="auth",
                        )
                        return False
                    passphrase = getpass.getpass(prompt=f"Enter passphrase for key {identityfile} -> ")
                    self._connect_client(
                        batch_mode=batch_mode,
                        hostname=self.host,
                        port=self.port,
                        username=self.username,
                        allow_agent=False,
                        look_for_keys=False,
                        key_filename=identityfile,
                        passphrase=passphrase,
                    )
                    self.connected = True
                    self.last_connect_error = None
                    self.last_error_type = None
                    logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username}")
                    return True
                except AuthenticationException:
                    logging.error(msg=f"Key-based authentication failed on {self.description}, trying password...")
                    # Use the new password authentication method with retry limit
                    return self._authenticate_with_password(allow_prompt=allow_prompt)
                except NoValidConnectionsError as e:
                    logging.error(msg=f"Connection failed: {e}")
                    self._set_connect_error(str(e), error_type="connect")
                    return False
            elif self.auth == "key":
                # Key-based authentication only (no password fallback)
                try:
                    self._connect_client(
                        batch_mode=batch_mode,
                        hostname=self.host,
                        port=self.port,
                        username=self.username,
                        allow_agent=False if identityfile else True,
                        look_for_keys=False if identityfile else True,
                        key_filename=identityfile,
                    )
                    self.connected = True
                    self.last_connect_error = None
                    self.last_error_type = None
                    logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username} (key auth)")
                    return True
                except PasswordRequiredException:
                    if not identityfile:
                        self._set_connect_error("Key requires passphrase; set `identityfile` or use `auth: password`", error_type="auth")
                        return False
                    if not allow_prompt:
                        self._set_connect_error(
                            "Key passphrase required; restart nvidb to enter it",
                            error_type="auth",
                        )
                        return False
                    passphrase = getpass.getpass(prompt=f"Enter passphrase for key {identityfile} -> ")
                    self._connect_client(
                        batch_mode=batch_mode,
                        hostname=self.host,
                        port=self.port,
                        username=self.username,
                        allow_agent=False,
                        look_for_keys=False,
                        key_filename=identityfile,
                        passphrase=passphrase,
                    )
                    self.connected = True
                    self.last_connect_error = None
                    self.last_error_type = None
                    logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username} (key auth)")
                    return True
                except AuthenticationException as e:
                    logging.error(msg=f"Key-based authentication failed on {self.description}: {e}")
                    self._set_connect_error("Key authentication failed", error_type="auth")
                    return False
                except NoValidConnectionsError as e:
                    logging.error(msg=f"Connection failed: {e}")
                    self._set_connect_error(str(e), error_type="connect")
                    return False
            elif self.auth == "password":
                # Password authentication with retry limit
                try:
                    # Password mode: try configured password first, then prompt if needed.
                    return self._authenticate_with_password(allow_prompt=allow_prompt)
                except NoValidConnectionsError as e:
                    logging.error(msg=f"Connection failed: {e}")
                    self._set_connect_error(str(e), error_type="connect")
                    return False
            else:
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
        
    def connect(self) -> bool:
        """Local connection is always successful"""
        logging.info(msg=f"Connected to local machine as {self.username}")
        print(f"Connected to local machine as {self.username}")
        self.connected = True
        self.last_connect_error = None
        self.last_error_type = None
        return True
    
    def execute_command(self, command: str) -> str:
        """Execute local command"""
        try:
            # Use shell=True to support pipes and complex commands
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
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
        "node": "Node",
        "available": "Available",
        "utilization": "Util high",
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
    UNIFIED_PROCESS_SORT_COLUMNS = {
        "vram": "vram",
        "cpu": "cpu",
        "mem": "mem",
        "rss": "rss",
        "time": "time",
        "pid": "pid",
        "command": "command",
    }

    def __init__(
        self,
        server_list: ServerListInfo,
        *,
        compact: bool = False,
        view_settings=None,
    ):
        """Build the client pool.

        `view_settings` restores the layout persisted in config.yml; passing it
        also enables writing later view changes back to that file.
        """
        self.pool = [LocalClient()]
        if server_list is not None:
            self.pool += [RemoteClient(server) for server in server_list]
        logging.info(msg=f"Initialized pool with {len(self.pool)} clients.")
        self.connect_all()
        self.term = Terminal()
        self.compact = bool(compact)
        self.quit_flag = threading.Event()  # Exit flag for inter-thread communication
        # Collapsible display state - every server starts expanded so the
        # dashboard opens showing the whole cluster; the stacked tables
        # then scale to whatever rows the terminal has (see
        # _scale_server_blocks).
        self.expanded_servers = set(range(len(self.pool)))
        self.selected_server = 0  # Currently selected server for navigation
        self._persist_view_enabled = view_settings is not None
        self._default_expansion_applied = False
        self._expansion_touched = False
        settings = nvidb_config.normalize_view_settings(view_settings)
        self.display_mode = (
            self.DISPLAY_MODE_UNIFIED
            if settings["mode"] == "unified"
            else self.DISPLAY_MODE_NODES
        )
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
        self._unified_process_total_count = 0
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
        self._unified_page_start = 0
        self._unified_gpu_count = 0
        self._unified_page_size = 1
        self._unified_gpu_history = {}
        self._unified_process_history = {}
        self._unified_history_lock = threading.Lock()
        self.refresh_needed = threading.Event()  # Flag to trigger immediate refresh after key press
        self.ui_only_refresh = False  # Flag to indicate UI-only refresh (no data fetch)
        self.cached_stats = None  # Cached GPU stats data
        self.cached_raw_stats = {}  # Cached raw stats per client index
        self._cache_lock = threading.Lock()
        self._last_update_time = None
        self._last_fetch_duration = None
        self._last_fetch_error = None
        self._toggle_disabled_servers = set()
    
    def connect_all(self):
        for client in self.pool:
            try:
                client.connect()
            except Exception as e:
                logging.error(msg=f"Failed to connect to {getattr(client, 'description', 'unknown')}: {e}")
                if hasattr(client, "_set_connect_error"):
                    client._set_connect_error(str(e), error_type="connect")

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

    def get_client_gpus_info(self, return_raw: bool = False):
        # Set pandas display options for terminal output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)  # Auto-fill terminal width
        pd.set_option('display.colheader_justify', 'center')  # Center column headers
        
        stats_str = []
        raw_stats_by_client = {}
        user_memory_by_client = {}
        global_user_memory = {}
        process_details_by_client = {}
        # The htop-style process panel needs one extra `ps` call per host, so it
        # is only collected while that panel is on screen.
        want_process_details = self._process_panel_visible()
        for idx, client in enumerate(self.pool):
            stats, system_info = client.get_full_gpu_info()

            # Fetch system stats (CPU, memory, swap) and merge into system_info
            if not (isinstance(system_info, dict) and system_info.get("error")):
                try:
                    sys_stats = client.get_system_stats()
                    system_info["system_stats"] = sys_stats
                except Exception as e:
                    logging.warning(f"Failed to get system stats for {client.description}: {e}")

            user_summary_for_client = {}
            
            # Skip processing if stats is empty
            if stats.empty:
                raw_stats_by_client[idx] = (stats, system_info)
                stats_str.append((stats, system_info))
                user_memory_by_client[idx] = {}
                process_details_by_client[idx] = {}
                continue
            
            # Optimize rx/tx display - split into two columns
            rx_list = []
            tx_list = []
            for _, row in stats.iterrows():
                rx_val, rx_unit = extract_value_and_unit(row['rx_util'])
                tx_val, tx_unit = extract_value_and_unit(row['tx_util'])

                rx_formatted = format_bandwidth(rx_val, rx_unit)
                tx_formatted = format_bandwidth(tx_val, tx_unit)

                # Ensure formatted strings don't exceed column width limit
                rx_list.append(rx_formatted[:11])
                tx_list.append(tx_formatted[:11])

            stats['rx'] = rx_list
            stats['tx'] = tx_list
            self._add_pcie_link_columns(stats)
            stats['power'] = [f"{row['power_state']} {'/'.join(extract_numbers(row['power_draw']))}/{'/'.join(extract_numbers(row['current_power_limit']))}" for _, row in stats.iterrows()]
            stats['memory[used/total]'] = [f"{'/'.join(extract_numbers(row['used']))}/{'/'.join(extract_numbers(row['total']))}" for _, row in stats.iterrows()]

            # Add process information as a column (batch pid->user lookup to avoid per-process SSH calls)
            process_list = []
            try:
                all_processes, user_summary_for_client = client.get_process_summary(
                    stats,
                    detailed=want_process_details,
                )

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

                for gpu_idx, _row in stats.iterrows():
                    user_summary = per_gpu_user_summary.get(gpu_idx, {})
                    if user_summary:
                        process_list.append(client.format_user_memory_compact(user_summary))
                    else:
                        process_list.append("-")
                process_details_by_client[idx] = per_gpu_process_details
            except Exception as e:
                logging.warning(f"Failed to get process info: {e}")
                process_list = ["-" for _ in range(len(stats))]
                user_summary_for_client = {}
                process_details_by_client[idx] = {}
            
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
            # Only drop columns that exist in the DataFrame
            columns_to_drop = [col for col in columns_to_drop if col in stats.columns]
            stats = stats.drop(columns=columns_to_drop)
            
            # Reorder columns: move mem_util before memory[used/total] and processes at the end
            if 'processes' in stats.columns:
                # Define desired column order
                desired_order = ['GPU', 'name', 'fan', 'util', 'temp', 'link', 'rx', 'tx', 'power', 'mem_util', 'memory[used/total]', 'processes']
                # Only keep columns that exist in stats
                ordered_columns = [col for col in desired_order if col in stats.columns]
                # Add any remaining columns that weren't in desired_order
                remaining_columns = [col for col in stats.columns if col not in ordered_columns]
                stats = stats[ordered_columns + remaining_columns]

            stats_str.append((stats, system_info))

            # Cache processed table (not raw XML) and per-user memory stats for this client
            raw_stats_by_client[idx] = (stats.copy() if not stats.empty else stats, system_info)
            user_memory_by_client[idx] = dict(user_summary_for_client or {})
            for username, total_mib in (user_summary_for_client or {}).items():
                try:
                    total_mib_int = int(total_mib)
                except Exception:
                    continue
                if total_mib_int <= 0:
                    continue
                global_user_memory[username] = global_user_memory.get(username, 0) + total_mib_int

        raw_stats_by_client["_nvidb"] = {
            "user_memory_by_client": user_memory_by_client,
            "user_memory_global": global_user_memory,
            "process_details_by_client": process_details_by_client,
        }
        # reformat the str into a single string with fixed width formatting
        # These blocks only ever appear inside the per-node view's panel
        # border, so they reserve its margin up front - narrow terminals
        # drop the border (see `compact_layout`) and so reserve nothing.
        try:
            _fetch_width = os.get_terminal_size().columns
        except OSError:
            _fetch_width = 80
        panel_margin = 0 if _fetch_width < 72 else self._PANEL_BORDER_MARGIN
        formatted_stats = []
        for client, (stats, system_info) in zip(self.pool, stats_str):
            # Create formatted table display (or error panel)
            if isinstance(system_info, dict) and system_info.get("error"):
                error_panel = self._format_error_panel(
                    system_info.get("error", "Error"),
                    error_type=system_info.get("error_type"),
                    outer_margin=panel_margin,
                )
                formatted_stats.append(error_panel)
                continue

            formatted_table = self._format_fixed_width_table(
                stats, border=True, outer_margin=panel_margin
            )

            system_info_header = ""
            if system_info:
                source = system_info.get("data_source", "N/A")
                source_detail = system_info.get("data_source_detail")
                if source_detail and source_detail not in {"N/A", "-", ""}:
                    source_display = f"{source} ({source_detail})"
                else:
                    source_display = str(source)
                # Surface DCGM status: it is easy to mistake a hidden advanced table
                # for "DCGM not connected" when profiling fields are just unsupported
                # (DCP metrics are datacenter-GPU only).
                advanced_display = ""
                adv_source = system_info.get("advanced_source")
                if adv_source:
                    if system_info.get("advanced_supported") is True:
                        adv_detail = system_info.get("advanced_source_detail")
                        if adv_detail and adv_detail not in {"N/A", "-", ""}:
                            advanced_display = f" | Advanced: {adv_source} ({adv_detail})"
                        else:
                            advanced_display = f" | Advanced: {adv_source}"
                    else:
                        advanced_display = f" | Advanced: {adv_source} (profiling unavailable)"
                system_info_header = (
                    f"Driver: {system_info.get('driver_version', 'N/A')} | "
                    f"CUDA: {system_info.get('cuda_version', 'N/A')} | "
                    f"GPUs: {system_info.get('attached_gpus', '0')} | "
                    f"Source: {source_display}{advanced_display}\n"
                )

            advanced_block = ""
            advanced_supported = (
                isinstance(system_info, dict) and system_info.get("advanced_supported") is True
            )
            advanced_df = system_info.get("advanced_metrics") if isinstance(system_info, dict) else None
            if advanced_supported and isinstance(advanced_df, pd.DataFrame) and not advanced_df.empty:
                try:
                    sub_blocks = ["Advanced (DCGM)"]
                    for title, cols in ADVANCED_METRIC_GROUPS:
                        cols_present = present_columns(advanced_df.columns, list(cols))
                        if len(cols_present) < 2:
                            continue
                        sub_df = advanced_df[cols_present].copy()
                        sub_table = self._format_fixed_width_table(
                            sub_df, border=True, outer_margin=panel_margin
                        )
                        sub_blocks.append(f"{title}\n{sub_table}")

                    advanced_block = "\n" + "\n".join(sub_blocks)
                except Exception:
                    advanced_block = ""

            formatted_stats.append(
                f"{system_info_header}{formatted_table}{advanced_block}"
            )
        if return_raw:
            return formatted_stats, raw_stats_by_client
        return formatted_stats

    def _client_table_identity(self, client_index):
        """Return compact node and hostname labels for the unified TUI table."""
        client = self.pool[client_index]
        description = str(getattr(client, "description", "") or "").strip()
        configured_host = str(getattr(client, "host", "") or "").strip()
        is_local = getattr(client, "port", None) == "local" or isinstance(client, LocalClient)

        if is_local:
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
        raw_stats_by_client = raw_stats_by_client if isinstance(raw_stats_by_client, dict) else {}

        for idx in range(len(self.pool)):
            entry = raw_stats_by_client.get(idx)
            if not isinstance(entry, tuple) or len(entry) != 2:
                continue
            stats, _system_info = entry
            if not isinstance(stats, pd.DataFrame) or stats.empty:
                continue

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

    @staticmethod
    def _extract_metric_number(value):
        """Return the first numeric component of a displayed metric."""
        if value is None:
            return None
        try:
            numbers = extract_numbers(str(value))
            return float(numbers[0]) if numbers else None
        except (TypeError, ValueError, IndexError):
            return None

    def _unified_gpu_capacity(self, row):
        """Return utilization and VRAM values used by summaries and sorting."""
        utilization = self._extract_metric_number(row.get("util"))
        memory_value = str(row.get("memory[used/total]", "") or "")
        used_mib = total_mib = None
        if "/" in memory_value:
            used_value, total_value = memory_value.split("/", 1)
            used_mib = self._extract_metric_number(used_value)
            total_mib = self._extract_metric_number(total_value)

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

        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80

        average_display = (
            f"{average_utilization:.0f}%"
            if average_utilization is not None
            else "N/A"
        )

        def _util_color(avg):
            if avg is None:
                return None
            if avg >= 80:
                return "red"
            if avg >= 50:
                return "yellow"
            return "green"

        def _mem_pct_color(pct):
            if pct >= 90:
                return "red"
            if pct >= 75:
                return "yellow"
            return "green"

        sep = colored(" | ", "dark_grey")
        if total_mib > 0 and terminal_width >= 100:
            memory_percent = used_mib / total_mib * 100
            plain_summary = (
                f"Capacity: available {available_count}/{len(table)} | "
                f"busy {busy_count} | avg util {average_display} | "
                f"VRAM {used_mib / 1024:.1f}/{total_mib / 1024:.1f} GiB "
                f"({memory_percent:.0f}%) | free {free_mib / 1024:.1f} GiB"
            )
            if len(plain_summary) > terminal_width:
                return plain_summary[: max(0, terminal_width - 3)] + "..."
            summary = (
                colored("Capacity:", "cyan", attrs=["bold"])
                + " available "
                + colored(f"{available_count}/{len(table)}", "green")
                + sep + "busy "
                + colored(str(busy_count), "yellow")
                + sep + "avg util "
                + colored(average_display, _util_color(average_utilization))
                + sep + "VRAM "
                + colored(
                    f"{used_mib / 1024:.1f}/{total_mib / 1024:.1f} GiB ({memory_percent:.0f}%)",
                    _mem_pct_color(memory_percent),
                )
                + sep + "free "
                + colored(f"{free_mib / 1024:.1f} GiB", "green")
            )
        elif total_mib > 0:
            plain_summary = (
                f"Capacity: avail {available_count}/{len(table)} | "
                f"busy {busy_count} | avg {average_display} | "
                f"VRAM {used_mib / 1024:.0f}/{total_mib / 1024:.0f}G | "
                f"free {free_mib / 1024:.0f}G"
            )
            if len(plain_summary) > terminal_width:
                return plain_summary[: max(0, terminal_width - 3)] + "..."
            memory_percent = used_mib / total_mib * 100
            summary = (
                colored("Capacity:", "cyan", attrs=["bold"])
                + " avail "
                + colored(f"{available_count}/{len(table)}", "green")
                + sep + "busy "
                + colored(str(busy_count), "yellow")
                + sep + "avg "
                + colored(average_display, _util_color(average_utilization))
                + sep + "VRAM "
                + colored(
                    f"{used_mib / 1024:.0f}/{total_mib / 1024:.0f}G",
                    _mem_pct_color(memory_percent),
                )
                + sep + "free "
                + colored(f"{free_mib / 1024:.0f}G", "green")
            )
        else:
            plain_summary = (
                f"Capacity: avail {available_count}/{len(table)} | "
                f"busy {busy_count} | avg {average_display} | VRAM N/A"
            )
            if len(plain_summary) > terminal_width:
                return plain_summary[: max(0, terminal_width - 3)] + "..."
            summary = (
                colored("Capacity:", "cyan", attrs=["bold"])
                + " avail "
                + colored(f"{available_count}/{len(table)}", "green")
                + sep + "busy "
                + colored(str(busy_count), "yellow")
                + sep + "avg "
                + colored(average_display, _util_color(average_utilization))
                + sep + "VRAM N/A"
            )

        return summary

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
        return f"{node} ({hostname})", " | ".join(parts)

    def _unified_section_headers(self, visible_table, scope_table):
        """Map visible row index -> node band text for the grouped views."""
        headers = {}
        if (
            not self._unified_grouping_active()
            or visible_table.empty
            or "Node" not in visible_table.columns
        ):
            return headers

        def identity(row):
            return str(row.get("Node", "")), str(row.get("Hostname", ""))

        previous_key = None
        for position, (_, row) in enumerate(visible_table.iterrows()):
            key = identity(row)
            if key == previous_key:
                continue
            previous_key = key
            node_rows = [
                node_row
                for _, node_row in scope_table.iterrows()
                if identity(node_row) == key
            ]
            headers[position] = self._format_unified_node_band(
                key[0],
                key[1],
                node_rows or [row],
            )
        return headers

    def _paginate_unified_gpu_table(self, table, *, detailed):
        """Return the visible GPU page and selected row within that page."""
        try:
            terminal_height = os.get_terminal_size().lines
        except OSError:
            terminal_height = 24

        show_processes = self._process_panel_visible()
        show_trends = bool(self.unified_show_trends)
        extra_lines = (
            (18 if show_processes else 0)
            + (6 if show_trends else 0)
            # Room for the "filter is hiding GPUs" warning line.
            + (1 if self._get_unified_filter_mode() != "all" else 0)
        )
        # Node bands take screen space too: two extra lines per block in the
        # detailed view (the block reuses one card border), one otherwise.
        group_count = 0
        if (
            not detailed
            and self._unified_grouping_active()
            and not table.empty
            and "Node" in table.columns
        ):
            group_count = len(table[["Node", "Hostname"]].drop_duplicates())
        if detailed:
            reserved_lines = 10 + extra_lines + group_count * 2
            page_size = max(1, (terminal_height - reserved_lines) // 5)
        else:
            reserved_lines = 12 + extra_lines + group_count
            page_size = max(1, terminal_height - reserved_lines)

        total = len(table)
        selected = int(self.unified_selected_gpu or 0)
        selected_key = self.unified_selected_gpu_key
        if selected_key is not None:
            for position, (_, row) in enumerate(table.iterrows()):
                if self._unified_gpu_history_key(row) == selected_key:
                    selected = position
                    break
        selected = max(0, min(selected, max(0, total - 1)))
        self.unified_selected_gpu = selected
        new_selected_key = (
            self._unified_gpu_history_key(table.iloc[selected])
            if total
            else None
        )
        self.unified_selected_gpu_key = new_selected_key
        if (
            selected_key is not None
            and new_selected_key != selected_key
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

        meta = (
            raw_stats_by_client.get("_nvidb", {})
            if isinstance(raw_stats_by_client, dict)
            else {}
        )
        process_details_by_client = (
            meta.get("process_details_by_client", {})
            if isinstance(meta, dict)
            else {}
        )
        client_details = process_details_by_client.get(client_index, {})
        if not isinstance(client_details, dict):
            return []
        gpu_key = str(selected_row.get("GPU", ""))
        processes = client_details.get(gpu_key, [])
        return [
            process
            for process in processes
            if isinstance(process, dict)
        ]

    def _get_sorted_unified_processes(
        self,
        raw_stats_by_client,
        selected_row,
    ):
        processes = self._get_unified_process_details(
            raw_stats_by_client,
            selected_row,
        )
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
        descending = bool(
            getattr(
                self,
                "unified_process_sort_descending",
                mode != "command",
            )
        )
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

    def _unified_process_sort_value(self, process, mode):
        if mode == "vram":
            return self._extract_metric_number(process.get("used_memory"))
        if mode == "cpu":
            return self._extract_metric_number(process.get("cpu_percent"))
        if mode == "mem":
            return self._extract_metric_number(process.get("mem_percent"))
        if mode == "rss":
            return self._extract_metric_number(process.get("rss_kb"))
        if mode == "time":
            return self._process_elapsed_seconds(process.get("elapsed"))
        if mode == "pid":
            return self._extract_metric_number(process.get("pid"))
        if mode == "command":
            command = (
                process.get("command")
                or process.get("process_name")
                or ""
            )
            return str(command).casefold() or None
        return None

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
    def _process_metric_color(value, warning, critical, default="cyan"):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number >= critical:
            return "red"
        if number >= warning:
            return "yellow"
        return default

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
            f"History  GPU {len(gpu_history)}/60 | process {len(history)}/60"
            if include_gpu
            else f"History  {len(history)}/60 samples"
        )
        if not history and not gpu_history:
            plain = f"{title} | waiting for samples"[:width]
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
            spark_width = max(6, width - 8 - len(suffix) - 1)
            spark = self._sparkline(
                values,
                minimum=minimum,
                maximum=max(maximum, peak),
                width=spark_width,
            )
            plain = f"{label:<7}{spark} {suffix}"
            if len(plain) > width:
                plain = plain[: max(0, width - 2)] + ".."
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

        rss_values = [
            float(sample["rss_kb"])
            for sample in history
            if sample.get("rss_kb") is not None
        ]
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
                    max([1] + rss_values),
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
            self._unified_process_total_count = 0
            self._unified_process_count = 0
            self._unified_process_visible_rows = 0
            self._unified_command_line_count = 0
            self._unified_command_page_size = 0
            return "Process details: no GPU selected"

        try:
            terminal_size = os.get_terminal_size()
            terminal_width, terminal_height = terminal_size.columns, terminal_size.lines
        except OSError:
            terminal_width, terminal_height = 80, 24
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
        border_horizontal = "─" if pane_focused else "╌"
        border_vertical = "│" if pane_focused else "┊"
        label_color = "cyan" if pane_focused else "dark_grey"
        process_line_map = process_line_map if process_line_map is not None else {}
        command_line_map = command_line_map if command_line_map is not None else {}
        action_regions = action_regions if action_regions is not None else []

        def trim(text, limit):
            text = str(text)
            if len(text) <= limit:
                return text
            if limit <= 2:
                return text[:limit]
            return text[: limit - 2] + ".."

        def panel_rule(label, *, bottom=False):
            left, right = ("╰", "╯") if bottom else ("├", "┤")
            if not label:
                plain = (
                    left
                    + border_horizontal * max(0, width - 2)
                    + right
                )
                return colored(plain, "dark_grey")
            label = trim(
                f"{border_horizontal} {label} ",
                max(0, width - 2),
            )
            plain = (
                left
                + label
                + border_horizontal * max(0, width - 2 - len(label))
                + right
            )
            return colored(plain, "dark_grey")

        def panel_line(plain, styled=None):
            original_plain = str(plain)
            plain = trim(original_plain, inner_width)
            if styled is None or plain != original_plain:
                styled = plain
            padding = " " * max(0, inner_width - len(plain))
            return (
                colored(f"{border_vertical} ", "dark_grey")
                + (styled if styled is not None else plain)
                + padding
                + colored(f" {border_vertical}", "dark_grey")
            )

        def pack_fields(label, fields):
            """Pack colored detail fields onto as many panel rows as needed."""
            prefix = f"{label:<9}"
            continuation = " " * len(prefix)
            packed = []
            current = []
            current_length = len(prefix)
            for text, color in fields:
                separator_length = 3 if current else 0
                if (
                    current
                    and current_length + separator_length + len(text) > inner_width
                ):
                    packed.append((prefix, current))
                    prefix = continuation
                    current = []
                    current_length = len(prefix)
                    separator_length = 0
                current.append((text, color))
                current_length += separator_length + len(text)
            packed.append((prefix, current))

            rows = []
            for row_prefix, row_fields in packed:
                plain = row_prefix + " | ".join(
                    text for text, _color in row_fields
                )
                styled = colored(
                    row_prefix,
                    label_color,
                    attrs=["bold"] if row_prefix.strip() else None,
                ) + " | ".join(
                    colored(text, color)
                    if color
                    else text
                    for text, color in row_fields
                )
                rows.append((plain, styled))
            return rows

        all_processes = self._get_unified_process_details(
            raw_stats_by_client,
            selected_row,
        )
        processes = self._get_sorted_unified_processes(
            raw_stats_by_client,
            selected_row,
        )
        self._unified_process_total_count = len(all_processes)
        self._unified_process_count = len(processes)
        selected_index = int(
            self.unified_selected_process or 0
        )
        selected_pid = self.unified_selected_process_pid
        if selected_pid is not None:
            for index, process in enumerate(processes):
                if str(process.get("pid", "")) == str(selected_pid):
                    selected_index = index
                    break
        selected_index = max(
            0,
            min(selected_index, max(0, len(processes) - 1)),
        )
        self.unified_selected_process = selected_index
        self.unified_selected_process_pid = (
            str(processes[selected_index].get("pid", ""))
            if processes
            else None
        )
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
        process_sort_arrow = (
            "▼"
            if bool(
                getattr(
                    self,
                    "unified_process_sort_descending",
                    process_sort_mode != "command",
                )
            )
            else "▲"
        )
        focus_label = "ACTIVE" if pane_focused else "inactive · Enter/→"
        filter_title = f" | {filter_label}" if filter_label else ""
        title = (
            f"Processes | {count_label}{filter_title} | "
            f"{node} ({hostname}) | GPU {gpu_index} | "
            f"sort {process_sort_label}{process_sort_arrow} | {focus_label}"
        )
        top_label = trim(
            f"{border_horizontal} {title} ",
            max(0, width - 2),
        )
        top_tail = (
            border_horizontal * max(0, width - 2 - len(top_label))
            + "╮"
        )
        lines = [
            colored("╭", "dark_grey")
            + colored(
                top_label,
                label_color,
                attrs=["bold"] if pane_focused else None,
            )
            + colored(top_tail, "dark_grey")
        ]

        total_process_vram = sum(
            self._extract_metric_number(process.get("used_memory")) or 0
            for process in processes
        )
        total_gpu_mib = capacity.get("total_mib")
        vram_summary = f"{total_process_vram / 1024:.1f} GiB"
        if total_gpu_mib:
            vram_summary += (
                f" ({total_process_vram / total_gpu_mib * 100:.1f}% of GPU VRAM)"
            )
        visible_summary = (
            (
                f"{len(processes)}/{len(all_processes)} match"
                if filter_query or filter_editing
                else f"{len(processes)} active"
            )
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
                panel_line(
                    visible_summary,
                    colored(visible_summary, "light_grey"),
                )
            )

        if not processes:
            self._unified_command_line_count = 0
            self._unified_command_page_size = 0
            self.unified_command_scroll = 0
            lines.append(panel_rule("PROCESS LIST"))
            empty = (
                f"No processes match /{filter_query}"
                if filter_query
                else "No active GPU processes"
            )
            lines.append(panel_line(empty, colored(empty, "dark_grey")))
            lines.append(panel_rule("ACTIONS"))
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
                panel_line(
                    action_plain,
                    "Actions  "
                    + colored(
                        history_label,
                        label_color,
                        attrs=["bold"] if pane_focused else None,
                    ),
                )
            )
            notice = self._process_action_notice
            if notice and notice.get("expires_at", 0) > time.monotonic():
                message = str(notice.get("message", ""))
                lines.append(
                    panel_line(
                        message,
                        colored(
                            message,
                            notice.get("color") or "white",
                            attrs=["bold"],
                        ),
                    )
                )
            elif notice:
                self._process_action_notice = None
            lines.append(panel_rule("", bottom=True))
            return "\n".join(lines)

        active_sort_column = self.UNIFIED_PROCESS_SORT_COLUMNS[
            process_sort_mode
        ]

        def sortable_label(key, label):
            if key == active_sort_column:
                return f"{label}{process_sort_arrow}"
            return label

        column_specs = [
            {"key": "sel", "label": "", "minimum": 1, "align": "left"},
            {
                "key": "pid",
                "label": sortable_label("pid", "PID"),
                "minimum": 7,
                "align": "right",
            },
            {"key": "user", "label": "USER", "minimum": 9, "align": "left"},
            {
                "key": "vram",
                "label": sortable_label("vram", "VRAM"),
                "minimum": 10,
                "align": "right",
            },
            {"key": "gpu", "label": "VRAM%", "minimum": 6, "align": "right"},
            {
                "key": "cpu",
                "label": sortable_label("cpu", "CPU%"),
                "minimum": 6,
                "align": "right",
            },
            {
                "key": "mem",
                "label": sortable_label("mem", "MEM%"),
                "minimum": 6,
                "align": "right",
            },
            {
                "key": "rss",
                "label": sortable_label("rss", "RSS"),
                "minimum": 7,
                "align": "right",
            },
            {
                "key": "time",
                "label": sortable_label("time", "TIME"),
                "minimum": 9,
                "align": "right",
            },
            {"key": "state", "label": "S", "minimum": 3, "align": "left"},
            {
                "key": "command",
                "label": sortable_label("command", "COMMAND"),
                "minimum": 12,
                "align": "left",
            },
        ]

        def columns_width(columns):
            return sum(column["minimum"] for column in columns) + max(
                0, len(columns) - 1
            )

        columns = list(column_specs)
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

        def padded_cell(value, column):
            value = trim(value, column["width"])
            if column["align"] == "right":
                return value.rjust(column["width"])
            return value.ljust(column["width"])

        def process_cells(process, index):
            vram_mib = self._extract_metric_number(process.get("used_memory"))
            gpu_percent = (
                vram_mib / total_gpu_mib * 100
                if vram_mib is not None and total_gpu_mib
                else None
            )
            cpu_percent = self._extract_metric_number(process.get("cpu_percent"))
            mem_percent = self._extract_metric_number(process.get("mem_percent"))
            rss_kb = self._extract_metric_number(process.get("rss_kb"))
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
                "vram": self._process_metric_color(
                    gpu_percent, 50, 80
                ),
                "gpu": self._process_metric_color(
                    gpu_percent, 50, 80
                ),
                "cpu": self._process_metric_color(
                    cpu_percent, 70, 100
                ),
                "mem": self._process_metric_color(
                    mem_percent, 20, 40
                ),
                "rss": "dark_grey",
                "time": "dark_grey",
                "state": (
                    "green"
                    if state.startswith("R")
                    else "red"
                    if state.startswith(("D", "Z", "X"))
                    else "dark_grey"
                ),
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
                "cyan" if column["key"] == active_sort_column else "dark_grey",
                attrs=(
                    ["bold"]
                    if column["key"] == active_sort_column
                    else None
                ),
            )
            for cell, column in zip(header_cells, columns)
        )
        header_line_index = None
        if not ultra_compact:
            if compact_layout:
                header_line_index = len(lines)
                lines.append(panel_line(header_plain, header_styled))
            else:
                lines.append(panel_rule("PROCESS LIST"))
                header_line_index = len(lines)
                lines.append(panel_line(header_plain, header_styled))

        if header_line_index is not None:
            sort_targets = {
                "pid": "pid",
                "vram": "vram",
                "gpu": "vram",
                "cpu": "cpu",
                "mem": "mem",
                "rss": "rss",
                "time": "time",
                "command": "command",
            }
            column_start = 0
            for column in columns:
                sort_target = sort_targets.get(column["key"])
                if sort_target:
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
            lines.append(panel_line(plain, styled))

        if terminal_height >= 28 and (start > 0 or end < len(processes)):
            scroll_status = f"Rows {start + 1}-{end}/{len(processes)} | wheel or j/k"
            lines.append(
                panel_line(
                    scroll_status,
                    colored(scroll_status, "dark_grey"),
                )
            )

        process = processes[selected_index]
        selected_pid = process.get("pid", "N/A")
        if not compact_history:
            selected_label = (
                f"{'SELECTED PROCESS' if pane_focused else 'PROCESS DETAILS'}"
                f" | PID {selected_pid}"
            )
            if compact_layout:
                selected_label += (
                    f" | {process.get('username', 'N/A')}"
                    f" | {process.get('type', 'N/A')}"
                    f" | {process.get('state') or '-'}"
                )
            lines.append(panel_rule(selected_label))
        vram_mib = self._extract_metric_number(process.get("used_memory"))
        gpu_percent = (
            vram_mib / total_gpu_mib * 100
            if vram_mib is not None and total_gpu_mib
            else None
        )
        gpu_used_mib = capacity.get("used_mib")
        used_share = (
            vram_mib / gpu_used_mib * 100
            if vram_mib is not None and gpu_used_mib
            else None
        )
        cpu_percent = self._extract_metric_number(process.get("cpu_percent"))
        mem_percent = self._extract_metric_number(process.get("mem_percent"))
        usage_fields = [
            (
                (
                    f"VRAM {vram_mib:,.0f} MiB"
                    if vram_mib is not None
                    else "VRAM N/A"
                ),
                self._process_metric_color(gpu_percent, 50, 80),
            ),
            (
                (
                    f"{gpu_percent:.1f}% of GPU VRAM"
                    if gpu_percent is not None
                    else "% of GPU VRAM N/A"
                ),
                "cyan",
            ),
            (
                (
                    f"{used_share:.1f}% of used VRAM"
                    if used_share is not None
                    else "% of used VRAM N/A"
                ),
                "cyan",
            ),
            (
                (
                    f"CPU {cpu_percent:.1f}%"
                    if cpu_percent is not None
                    else "CPU N/A"
                ),
                self._process_metric_color(cpu_percent, 70, 100),
            ),
            (
                (
                    f"Host MEM {mem_percent:.1f}%"
                    if mem_percent is not None
                    else "Host MEM N/A"
                ),
                self._process_metric_color(mem_percent, 20, 40),
            ),
            (
                f"RSS {self._format_memory_kb(process.get('rss_kb'))}",
                "dark_grey",
            ),
        ]
        if compact_layout:
            if vram_mib is not None:
                compact_vram = f"VRAM {vram_mib:,.0f}M"
            else:
                compact_vram = "VRAM N/A"
            usage_fields = [
                (
                    compact_vram,
                    self._process_metric_color(
                        gpu_percent,
                        50,
                        80,
                    ),
                ),
            ]
            if gpu_percent is not None:
                usage_fields.append((f"{gpu_percent:.1f}% total", "cyan"))
            if used_share is not None:
                usage_fields.append((f"{used_share:.1f}% used", "cyan"))
            usage_fields.extend(
                [
                    (
                        (
                            f"CPU {cpu_percent:.1f}%"
                            if cpu_percent is not None
                            else "CPU N/A"
                        ),
                        self._process_metric_color(
                            cpu_percent,
                            70,
                            100,
                        ),
                    ),
                    (
                        (
                            f"MEM {mem_percent:.1f}%"
                            if mem_percent is not None
                            else "MEM N/A"
                        ),
                        self._process_metric_color(
                            mem_percent,
                            20,
                            40,
                        ),
                    ),
                    (
                        f"RSS {self._format_memory_kb(process.get('rss_kb'))}",
                        "dark_grey",
                    ),
                ]
            )
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
        state_color = (
            "green"
            if state.startswith("R")
            else "red"
            if state.startswith(("D", "Z", "X"))
            else "dark_grey"
        )
        detail_fields = [
            (f"User {process.get('username', 'N/A')}", "light_grey"),
            (f"Type {process.get('type', 'N/A')}", "dark_grey"),
            (f"State {state}", state_color),
            (f"Threads {threads_display}", "dark_grey"),
            (f"Elapsed {process.get('elapsed') or '-'}", "dark_grey"),
        ]
        if not compact_history:
            for plain, styled in pack_fields("Usage", usage_fields):
                process_line_map[len(lines)] = selected_index
                lines.append(panel_line(plain, styled))
            if not compact_layout:
                for plain, styled in pack_fields("Details", detail_fields):
                    process_line_map[len(lines)] = selected_index
                    lines.append(panel_line(plain, styled))

        command = str(
            process.get("command") or process.get("process_name") or "N/A"
        ).strip()
        command_prefix = "Command  "
        command_width = max(1, inner_width - len(command_prefix))
        all_command_parts = textwrap.wrap(
            command,
            width=command_width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or ["N/A"]
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
                    else " " * len(command_prefix)
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
                lines.append(panel_line(plain, styled))

        if history_rows:
            if history_rule_lines:
                lines.append(panel_rule("SELECTED PROCESS HISTORY"))
            for plain, styled in history_rows:
                lines.append(panel_line(plain, styled))

        if not compact_layout:
            lines.append(panel_rule("ACTIONS"))
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

        history_action_label = (
            f"t:{1 if history_enabled else 0}"
            if tiny_actions
            else f"[t]H:{'ON' if history_enabled else 'OFF'}"
            if compact_actions
            else (
                f"[t]Hist {'ON' if history_enabled else 'OFF'}"
                if command_paged
                else f"[t] History {'ON' if history_enabled else 'OFF'}"
            )
        )
        action_segments = []
        if not compact_actions:
            action_segments.append(("Actions", None, None))
        action_segments.extend(
            [
                (
                    (
                        "i:INT"
                        if tiny_actions
                        else "[i]INT"
                        if compact_actions
                        else "[i] INT"
                    ),
                    action_color("cyan"),
                    ("signal", "INT"),
                ),
                (
                    (
                        "T:TERM"
                        if tiny_actions
                        else "[T]TERM"
                        if compact_actions
                        else "[T] TERM"
                    ),
                    action_color("yellow"),
                    ("signal", "TERM"),
                ),
                (
                    (
                        "K:KILL"
                        if tiny_actions
                        else "[K]KILL"
                        if compact_actions
                        else "[K] KILL"
                    ),
                    action_color("red"),
                    ("signal", "KILL"),
                ),
            ]
        )
        action_segments.append(
            (
                history_action_label,
                action_color("cyan"),
                ("toggle_trends", None),
            )
        )
        if command_paged:
            if tiny_actions:
                action_segments.extend(
                    [
                        (
                            "<",
                            action_color("cyan"),
                            ("command_scroll", -1),
                        ),
                        (
                            f"{command_start + 1}/{len(all_command_parts)}",
                            "dark_grey",
                            None,
                        ),
                        (
                            ">",
                            action_color("cyan"),
                            ("command_scroll", 1),
                        ),
                    ]
                )
            else:
                action_segments.extend(
                    [
                        (
                            "[<]Cmd",
                            action_color("cyan"),
                            ("command_scroll", -1),
                        ),
                        (
                            f"{command_start + 1}-{command_end}/"
                            f"{len(all_command_parts)}",
                            "dark_grey",
                            None,
                        ),
                        (
                            "[>]Cmd",
                            action_color("cyan"),
                            ("command_scroll", 1),
                        ),
                    ]
                )
        action_segments.extend(
            [
                (
                    (
                        "-"
                        if tiny_actions
                        else "-r"
                        if compact_actions
                        else "[-] Rows"
                    ),
                    action_color("cyan"),
                    ("process_rows", -1),
                ),
                (
                    (
                        "+"
                        if tiny_actions
                        else "+r"
                        if compact_actions
                        else "[+] Rows"
                    ),
                    action_color("cyan"),
                    ("process_rows", 1),
                ),
            ]
        )
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
            start_column = action_plain.find(text, search_from)
            search_from = start_column + len(text)
            if target and start_column < inner_width:
                action_regions.append(
                    (
                        action_line_index,
                        2 + start_column,
                        2 + min(inner_width, start_column + len(text)) - 1,
                        target,
                    )
                )
        lines.append(panel_line(action_plain, action_styled))

        pending = self._pending_process_signal
        if pending and pending.get("expires_at", 0) <= time.monotonic():
            self._pending_process_signal = None
            pending = None
        if pending:
            message = (
                f"Confirm SIG{pending['signal']} -> PID {pending['pid']}: "
                "Enter / same action again | Esc cancels"
            )
            lines.append(panel_line(message, colored(message, "yellow", attrs=["bold"])))
        else:
            notice = self._process_action_notice
            if notice and notice.get("expires_at", 0) > time.monotonic():
                message = str(notice.get("message", ""))
                lines.append(
                    panel_line(
                        message,
                        colored(
                            message,
                            notice.get("color") or "white",
                            attrs=["bold"],
                        ),
                    )
                )
            elif notice:
                self._process_action_notice = None

        lines.append(panel_rule("", bottom=True))
        return "\n".join(lines)

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
                        "temperature": self._extract_metric_number(
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
                if not pid or not pid.isdigit():
                    continue
                vram_mib = self._extract_metric_number(
                    process.get("used_memory")
                )
                gpu_vram_percent = None
                if (
                    vram_mib is not None
                    and total_mib is not None
                    and total_mib > 0
                ):
                    gpu_vram_percent = vram_mib / total_mib * 100
                process_samples.append(
                    (
                        self._unified_process_history_key(row, process),
                        {
                            "timestamp": timestamp,
                            "cpu_percent": self._extract_metric_number(
                                process.get("cpu_percent")
                            ),
                            "mem_percent": self._extract_metric_number(
                                process.get("mem_percent")
                            ),
                            "rss_kb": self._extract_metric_number(
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
                    deque(maxlen=60),
                )
                history.append(sample)
            for key, sample in process_samples:
                history = self._unified_process_history.setdefault(
                    key,
                    deque(maxlen=60),
                )
                history.append(sample)

            # PID churn should not grow a long-running TUI without bound. Keep
            # exited-process history around for an hour so a just-finished job
            # can still be inspected.
            stale_before = timestamp - 3600
            stale_keys = [
                key
                for key, history in self._unified_process_history.items()
                if history
                and history[-1].get("timestamp", timestamp) < stale_before
            ]
            for key in stale_keys:
                self._unified_process_history.pop(key, None)

    def _get_unified_gpu_history(self, selected_row):
        if selected_row is None:
            return []
        key = self._unified_gpu_history_key(selected_row)
        with self._unified_history_lock:
            return list(self._unified_gpu_history.get(key, ()))

    def _get_unified_process_history(self, selected_row, process):
        if selected_row is None or process is None:
            return []
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
        title = f"Trends: {node} GPU {gpu_index} | {len(history)}/60 samples"
        if not history:
            return f"{title}\nWaiting for telemetry samples"

        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80

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
                terminal_width - len(label) - len(suffix) - 4,
            )
            spark = self._sparkline(
                values,
                minimum=minimum,
                maximum=maximum,
                width=spark_width,
            )
            line = f"{label:<5} {spark} {suffix}"
            if len(line) > terminal_width:
                return colored(line[:terminal_width], color)
            return (
                colored(f"{label:<5}", color, attrs=["bold"])
                + " "
                + colored(spark, color)
                + f" {suffix}"
            )

        return "\n".join(
            [
                colored(title[:terminal_width], "cyan", attrs=["bold"]),
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
        section_headers=None,
        row_line_map=None,
    ):
        """Format each GPU as a readable, color-aware four-line card.

        `row_line_map` is filled in with line offset -> row index so a click
        anywhere on a card can be traced back to its GPU.
        """
        if table.empty:
            return "No GPU data available"

        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80
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

        def truncate(text, width):
            text = str(text)
            if width <= 0:
                return ""
            if len(text) <= width:
                return text
            if width <= 2:
                return text[:width]
            return text[: width - 2] + ".."

        def utilization_status(value):
            percent = self._extract_metric_number(value)
            if percent is None:
                return "UNKNOWN", None
            if percent >= 80:
                return "HIGH", "red"
            if percent >= 50:
                return "BUSY", "yellow"
            if percent >= 5:
                return "ACTIVE", "cyan"
            return "IDLE", "dark_grey"

        def memory_color(value):
            value = clean(value)
            if "/" not in value:
                return None
            used, total = value.split("/", 1)
            used_value = self._extract_metric_number(used)
            total_value = self._extract_metric_number(total)
            if (
                used_value is None
                or total_value is None
                or total_value <= 0
            ):
                return None
            percent = used_value / total_value * 100
            if percent >= 90:
                return "red"
            if percent >= 75:
                return "yellow"
            return "cyan"

        def threshold_color(value, warning, critical, default="cyan"):
            number = self._extract_metric_number(value)
            if number is None:
                return None
            if number >= critical:
                return "red"
            if number >= warning:
                return "yellow"
            return default

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
                if not active:
                    return 0
                return (
                    sum(min(len(field["text"]), field["minimum"]) for field in active)
                    + len(separator) * (len(active) - 1)
                )

            for key in drop_order:
                if minimum_total() <= content_width:
                    break
                active = [field for field in active if field["key"] != key]

            if not active:
                return " " * content_width

            widths = [len(field["text"]) for field in active]
            excess = (
                sum(widths)
                + len(separator) * (len(active) - 1)
                - content_width
            )
            for key in shrink_order:
                if excess <= 0:
                    break
                for index, field in enumerate(active):
                    if field["key"] != key:
                        continue
                    minimum = min(len(field["text"]), field["minimum"])
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
                truncate(field["text"], width)
                for field, width in zip(active, widths)
            ]
            plain_length = (
                sum(len(part) for part in plain_parts)
                + len(separator) * (len(plain_parts) - 1)
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
            try:
                value = max(0.0, min(100.0, float(percent)))
            except (TypeError, ValueError):
                return ""
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
        section_headers = section_headers or {}
        lines = [top_border]
        for row_index, (_, row) in enumerate(table.iterrows()):
            band = section_headers.get(row_index)
            if band:
                lines.extend(
                    [
                        card_line(
                            self._format_section_band(
                                *band,
                                table_width - 4,
                            )
                        ),
                        middle_border,
                    ]
                )
            utilization = clean(row.get("util")).replace(" ", "")
            status, utilization_color = utilization_status(utilization)
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

            utilization_percent = self._extract_metric_number(row.get("util"))
            memory_percent = self._unified_gpu_capacity(row)["memory_percent"]
            load_fields = [
                make_field("section", "LOAD", 4, color="green", bold=True),
                make_field(
                    "util",
                    f"Util {utilization}{progress_bar(utilization_percent)}",
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
                    color=memory_color(row.get("memory[used/total]")),
                    bold=True,
                ),
                make_field(
                    "temp",
                    f"Temp {clean(row.get('temp'))}",
                    9,
                    color=threshold_color(row.get("temp"), 70, 85),
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

    def _get_unified_node_status_lines(
        self,
        raw_stats_by_client,
        last_update_time,
        *,
        errors_only=False,
    ):
        """Describe nodes that cannot contribute rows to the unified table."""
        if last_update_time is None:
            return []

        hide_unsupported = bool(self.hide_unsupported)
        lines = []
        hidden_count = 0
        raw_stats_by_client = raw_stats_by_client if isinstance(raw_stats_by_client, dict) else {}
        for idx in range(len(self.pool)):
            node, hostname = self._client_table_identity(idx)
            stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))
            if isinstance(system_info, dict) and system_info.get("error"):
                message = system_info.get("error", "Error")
                # A node that dropped off must not read like a node that simply
                # has no NVIDIA GPU, so it keeps the red marker of a real error.
                lines.append(colored(f"! {node} ({hostname}): {message}", "red"))
            elif (
                not errors_only
                and (not isinstance(stats, pd.DataFrame) or stats.empty)
            ):
                # Machines without NVIDIA GPUs (e.g. a macOS laptop) would
                # otherwise repeat the same "no data" line on every refresh.
                if hide_unsupported:
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
        self._default_expansion_applied = True
        if self._expansion_touched:
            return
        if not bool(self.hide_unsupported):
            return

        raw_stats_by_client = raw_stats_by_client if isinstance(raw_stats_by_client, dict) else {}
        expanded = set()
        for idx in range(len(self.pool)):
            stats, _system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))
            if isinstance(stats, pd.DataFrame) and not stats.empty:
                expanded.add(idx)
        self.expanded_servers = expanded

    @staticmethod
    def _screen_line_count(elements):
        """Count the terminal rows a list of frame elements occupies."""
        return sum(len(str(element).split("\n")) for element in elements)

    _ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def _scale_server_blocks(
        self, server_rows, user_memory_by_client, available_rows, *, keep_expanded=None
    ):
        """Fit every expanded server block into the terminal at once.

        All servers open expanded, so the stacked tables have to share
        the rows left under the headers. Blocks keep their natural size
        while they fit; once they do not, GPU rows are dropped from the
        tallest block first, and a block whose table frame cannot survive
        collapses to its header line. The selected server is the last to
        shrink and the last to collapse, so pressing Enter on it always
        shows or hides its table when the window can hold one at all.
        Returns {idx: [block lines]} for the servers that stay expanded.
        """
        blocks = {}
        for idx, _selected, is_expanded, _header, _summary, stats_info in server_rows:
            if not is_expanded:
                continue
            block = list(str(stats_info).splitlines())
            if isinstance(user_memory_by_client, dict):
                users = user_memory_by_client.get(idx, {}) or {}
            else:
                users = {}
            if users:
                user_line = (
                    f"Users: {self._format_user_memory_totals(users, max_users=12)}"
                )
                try:
                    width = os.get_terminal_size().columns
                except OSError:
                    width = 80
                if self.term.length(user_line) > width:
                    user_line = user_line[: max(0, width - 3)] + "..."
                block.append(user_line)
            block.append("")
            blocks[idx] = block

        available_rows = max(0, available_rows)
        natural = {idx: len(block) for idx, block in blocks.items()}
        trimmed_ids = set()
        while sum(len(block) for block in blocks.values()) > available_rows:
            candidates = [idx for idx in blocks if idx != keep_expanded]
            if not candidates:
                candidates = list(blocks)
            tallest = max(candidates, key=lambda idx: len(blocks[idx]))
            trimmed = self._trim_server_block(blocks[tallest], len(blocks[tallest]) - 1)
            if trimmed is None:
                del blocks[tallest]
                continue
            blocks[tallest] = trimmed
            trimmed_ids.add(tallest)
        for idx in trimmed_ids:
            if idx not in blocks:
                continue
            # The marker counts against the block's natural height, not the
            # previous trim step, so the number matches what was dropped.
            hidden = natural[idx] - len(blocks[idx])
            blocks[idx][-1] = colored(
                f"  … {hidden} line(s) hidden — enlarge the terminal or press [c]",
                "dark_grey",
            )
        return blocks

    def _trim_server_block(self, lines, keep):
        """Shorten one expanded block to `keep` lines by dropping GPU rows.

        The table keeps its frame: everything up to the header separator
        stays, as many data rows as fit follow, and the original bottom
        border closes it. Returns None when there is no table to shrink
        or `keep` cannot hold the frame, meaning the caller should
        collapse the server instead.
        """
        if keep >= len(lines):
            return lines
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
        bottom_index = next(
            (
                index
                for index in range(
                    separator_index + 1 if separator_index is not None else 0,
                    len(plain),
                )
                if plain[index].startswith("╰")
            ),
            None,
        )
        if separator_index is None or bottom_index is None:
            return None

        head = lines[: separator_index + 1]
        bottom = lines[bottom_index]
        marker_text = f"  … {len(lines) - keep} line(s) hidden — enlarge the terminal or press [c]"
        marker = colored(marker_text, "dark_grey")
        # A frame with no data rows left says nothing a collapsed header
        # cannot, so one row is the minimum an expansion keeps.
        overhead = len(head) + 3  # bottom border, marker, one data row
        if keep < overhead:
            return None
        rows = lines[separator_index + 1 : bottom_index]
        kept_rows = keep - overhead + 1
        trimmed = head + rows[:kept_rows] + [bottom, marker]
        return trimmed

    def _selected_unified_row(self, raw_stats_by_client):
        """Return the currently selected GPU row across filtering and sorting."""
        table = self._sort_unified_gpu_table(
            self._filter_unified_gpu_table(
                self._build_unified_gpu_table(raw_stats_by_client)
            )
        )
        if table.empty:
            return None
        selected = int(self.unified_selected_gpu or 0)
        selected_key = self.unified_selected_gpu_key
        if selected_key is not None:
            for position, (_, row) in enumerate(table.iterrows()):
                if self._unified_gpu_history_key(row) == selected_key:
                    selected = position
                    break
        selected = max(0, min(selected, len(table) - 1))
        self.unified_selected_gpu = selected
        self.unified_selected_gpu_key = self._unified_gpu_history_key(
            table.iloc[selected]
        )
        self._unified_gpu_count = len(table)
        return table.iloc[selected]

    def _format_tui_help_lines(self, width, height):
        """Render a compact, context-sensitive keyboard and mouse reference."""
        width = max(20, int(width))
        height = max(4, int(height))
        inner_width = max(1, width - 4)
        display_mode = getattr(
            self,
            "display_mode",
            self.DISPLAY_MODE_NODES,
        )
        unified = display_mode == self.DISPLAY_MODE_UNIFIED
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
                ("key", "?", "Close this help"),
                ("key", "q", "Close help; q again quits"),
            ]
        elif pane == "process" and self._process_panel_visible():
            context = "UNIFIED · PROCESS"
            mode = self._get_unified_process_sort_mode()
            arrow = (
                "▼"
                if bool(
                    getattr(
                        self,
                        "unified_process_sort_descending",
                        mode != "command",
                    )
                )
                else "▲"
            )
            query = str(
                self.unified_process_filter or ""
            )
            rows = int(
                self._unified_process_visible_rows or 1
            )
            state = (
                f"sort {self.UNIFIED_PROCESS_SORT_LABELS[mode]}{arrow}"
                f" · filter /{query or '-'} · rows {rows}"
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
                ("key", "v", "Return to the per-node view"),
                ("key", "? / q", "Close this help"),
            ]

        def trim(text, limit):
            text = str(text)
            if len(text) <= limit:
                return text
            if limit <= 1:
                return text[:limit]
            return text[: limit - 1] + "…"

        title = trim(
            f"─ Help · {context} · ?/Esc/q close ",
            width - 2,
        )
        lines = [
            colored("╭", "dark_grey")
            + colored(title, "cyan", attrs=["bold"])
            + colored("─" * max(0, width - 2 - len(title)) + "╮", "dark_grey")
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
                plain = trim(f"── {key} ", inner_width)
                plain += "─" * max(0, inner_width - len(plain))
                styled = colored(plain, "dark_grey")
            else:
                key_text = trim(key, key_width).ljust(key_width)
                description_text = trim(
                    description,
                    description_width,
                ).ljust(description_width)
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

    def _render_unified_gpu_lines(self, raw_stats_by_client, last_update_time):
        """Render the unified TUI body from cached per-node GPU data."""
        self._body_click_targets = {}
        self._body_click_regions = []
        if last_update_time is None:
            return ["Unified GPU table", "Loading GPU data..."]

        source_table = self._build_unified_gpu_table(raw_stats_by_client)
        if source_table.empty:
            node_count = 0
        else:
            node_count = len(
                source_table[["Node", "Hostname"]].drop_duplicates()
            )
        detailed = self.unified_detailed
        try:
            terminal_size = os.get_terminal_size()
            terminal_width = terminal_size.columns
            terminal_height = terminal_size.lines
        except OSError:
            terminal_width = 80
            terminal_height = 24
        filter_mode = self._get_unified_filter_mode()
        show_process_panel = self._process_panel_visible()
        compact_history = (
            show_process_panel
            and terminal_height < 36
            and bool(self.unified_show_trends)
        )
        filtered_table = self._filter_unified_gpu_table(source_table)
        sorted_table = self._sort_unified_gpu_table(filtered_table)
        table, selected_row, page_status = self._paginate_unified_gpu_table(
            sorted_table,
            detailed=detailed,
        )
        title = "Unified GPU table (Detailed)" if detailed else "Unified GPU table"
        # Detailed cards already repeat node identity, so node bands only spend
        # scarce vertical space there. Keep the bands for the compact table.
        section_headers = (
            {}
            if detailed
            else self._unified_section_headers(table, sorted_table)
        )
        row_line_map = {}
        if filter_mode == "errors":
            rendered_table = "Error filter: GPU rows hidden"
        elif table.empty and not source_table.empty:
            filter_label = self.UNIFIED_FILTER_LABELS[filter_mode]
            rendered_table = f"No GPUs match the {filter_label} filter"
        elif detailed:
            rendered_table = self._format_unified_detailed_table(
                table,
                selected_row=selected_row,
                section_headers=section_headers,
                row_line_map=row_line_map,
            )
        else:
            display_table = table.copy()
            if not display_table.empty:
                display_table.insert(
                    0,
                    "sel",
                    [
                        ">" if index == selected_row else " "
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
                border=True,
                column_labels=self.UNIFIED_TABLE_LABELS,
                fixed_width_columns=("sel",),
                section_headers=section_headers,
                row_line_map=row_line_map,
            )
        selected_gpu_row = (
            table.iloc[selected_row]
            if selected_row is not None and not table.empty
            else None
        )
        page_suffix = f" | {page_status}" if page_status else ""
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
                f"{focus_prefix} | D | G:{gpu_count_display} | "
                f"N:{node_count}/{len(self.pool)}"
                + (f" | {compact_page}" if compact_page else "")
            )
        elif detailed and terminal_width < 100:
            title_line = (
                f"{focus_prefix} | Detailed | GPUs: {gpu_count_display} | "
                f"Nodes: {node_count}/{len(self.pool)}{page_suffix}"
            )
        else:
            title_with_focus = (
                f"{focus_prefix} | {title}"
                if detailed
                else title
            )
            title_line = (
                f"{title_with_focus} | GPUs: {gpu_count_display} | "
                f"Nodes with GPU: {node_count}/{len(self.pool)}{page_suffix}"
            )
        if len(title_line) > terminal_width:
            title_line = (
                title_line[: max(0, terminal_width - 2)] + ".."
                if terminal_width >= 2
                else title_line[:terminal_width]
            )
        if detailed and title_line.startswith(focus_prefix):
            title_line = (
                colored(focus_prefix, "cyan", attrs=["bold"])
                + title_line[len(focus_prefix):]
            )
        lines = [title_line]
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
        page_start = int(self._unified_page_start or 0)
        self._body_click_targets = {
            table_offset + line_index: ("gpu", page_start + row_index)
            for line_index, row_index in row_line_map.items()
        }
        lines.append(rendered_table)

        if show_process_panel:
            process_line_map = {}
            command_line_map = {}
            action_regions = []
            process_offset = self._screen_line_count(lines)
            compact_frame = detailed and (
                terminal_height < 28
                or (
                    terminal_height < 36
                    and bool(self.unified_show_trends)
                )
            )
            frame_reserve = 3 if compact_frame else 4
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
                terminal_height
                - frame_reserve
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
            self._unified_process_total_count = 0
            self._unified_process_count = 0
            self._unified_process_visible_rows = 0
            self._unified_command_line_count = 0
            self._unified_command_page_size = 0
            self.unified_command_scroll = 0
            if self.unified_active_pane == "process":
                self.unified_active_pane = "gpu"

        if (
            self.unified_show_trends
            and filter_mode != "errors"
            and (
                not compact_history
                or int(self._unified_process_count or 0) == 0
            )
        ):
            lines.append(
                self._format_unified_gpu_trends(selected_gpu_row)
            )

        status_lines = self._get_unified_node_status_lines(
            raw_stats_by_client,
            last_update_time,
            # Detailed mode reserves the lower half for process interaction.
            # Still show real node errors, but omit the routine "unsupported
            # node hidden" reminder that is already reflected in the title.
            errors_only=(filter_mode == "errors" or detailed),
        )
        if filter_mode == "errors" and not status_lines:
            status_lines = ["No node errors"]
        if status_lines:
            lines.append("Node status:")
            lines.extend(status_lines)
        return lines

    def _write_tui_lines(self, output_lines):
        """Paint one TUI frame, rewriting only the rows that changed.

        Whole-screen clears are what made refreshes flicker, especially
        over SSH; diffing against the previous frame keeps a refresh
        invisible when nothing moved.
        """
        rows = []
        for line in output_lines:
            rows.extend(str(line).split("\n"))
        screen = getattr(self, "_tui_diff_screen", None)
        if screen is None:
            screen = DiffScreen(self.term)
            self._tui_diff_screen = screen
        screen.paint(rows)

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
        try:
            width = os.get_terminal_size().columns
        except OSError:
            width = 80
        if outer_margin:
            width = max(20, width - outer_margin)

        if error_type == "auth" and str(message).strip() == "Password incorrect":
            title = "Password incorrect"
        elif error_type == "auth":
            title = "Authentication failed"
        else:
            title = "Error"
        line1 = f" {title} ".center(width, " ")
        line2 = f" {message} ".center(width, " ")
        line3 = " ".center(width, " ")
        return "\n".join(
            [
                colored(line1[:width], "white", "on_red", attrs=["bold"]),
                colored(line2[:width], "white", "on_red", attrs=["bold"]),
                colored(line3[:width], "white", "on_red", attrs=["bold"]),
            ]
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

        if len(label) >= width:
            label = label[: width - 2] + ".." if width > 2 else label[:width]
            separator = ""
            detail = ""
        elif len(label) + len(separator) + len(detail) > width:
            room = width - len(label) - len(separator)
            if room <= 3:
                separator = ""
                detail = ""
            else:
                detail = detail[: room - 2] + ".."

        pieces = [colored(label, "cyan", attrs=["bold"])]
        if separator:
            pieces.append(colored(separator, attrs=["dark"]))
        if detail:
            pieces.append(detail)
        tail = width - len(label) - len(separator) - len(detail)
        if tail == 1:
            pieces.append(" ")
        elif tail > 1:
            pieces.append(" " + colored("─" * (tail - 1), attrs=["dark"]))
        return "".join(pieces)

    def _format_fixed_width_table(
        self,
        df,
        border: bool = False,
        column_labels=None,
        *,
        min_width_overrides=None,
        importance_overrides=None,
        must_keep_columns=None,
        left_align_columns=None,
        fill_columns=None,
        fixed_width_columns=(),
        section_headers=None,
        row_line_map=None,
        outer_margin: int = 0,
    ):
        """Format fixed-width table display.

        Optional keyword arguments let callers reuse the layout engine for
        non-GPU tables: column minimum widths, drop priority, columns that must
        survive on narrow terminals, left-aligned and space-absorbing columns,
        plus `section_headers` (row index -> banner text) for grouped views.
        `row_line_map` is filled in with line offset -> data row index so callers
        can turn a mouse click into a row. `outer_margin` shrinks the table to
        leave room for a caller-drawn frame around it (e.g. the per-node view's
        panel border) instead of using the terminal's full width.
        """
        if df.empty:
            return "No GPU data available"

        column_labels = column_labels or {}
        section_headers = section_headers or {}
        left_align_columns = set(left_align_columns or ()) | {"Node", "Hostname"}
        fixed_width_columns = set(fixed_width_columns or ())

        # Get terminal width
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 120  # Default width
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
        min_widths.update(min_width_overrides or {})

        # Calculate widths based on terminal space
        separator_width = 3
        outer_width = 4 if border else 0  # "| " + " |"
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
        importance.update(importance_overrides or {})
        if must_keep_columns is not None:
            must_keep_names = tuple(must_keep_columns)
        elif is_unified_table:
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
            min_widths_for_cols[col] = max(min_width, len(str(label)))

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
            max_len = len(str(col))
            for value in df_display[col]:
                value_len = len(str(value))
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
        if fill_columns:
            fill_priority = list(fill_columns) + [
                col for col in fill_priority if col not in fill_columns
            ]

        if available_width >= desired_total:
            column_widths = dict(desired_widths)
            extra = available_width - desired_total
            if extra > 0 and not self.compact:
                if fill_columns:
                    spread_columns = [c for c in fill_columns if c in selected_columns]
                else:
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
            if width <= 0:
                return ""
            if text is None:
                text = ""
            text = str(text)
            if len(text) <= width:
                return text
            if width <= 2:
                return text[:width]
            if tail_preserve:
                return ".." + text[-(width - 2):]
            return text[:width - 2] + ".."

        def parse_percent(value: str) -> Optional[float]:
            if value is None:
                return None
            value_str = str(value).strip()
            if not value_str or value_str in {"N/A", "-"}:
                return None
            numbers = extract_numbers(value_str)
            if not numbers:
                return None
            try:
                return float(numbers[0])
            except Exception:
                return None

        def parse_ratio_percent(value: str) -> Optional[float]:
            if value is None:
                return None
            value_str = str(value).strip()
            if not value_str or value_str in {"N/A", "-"} or "/" not in value_str:
                return None
            used_str, total_str = value_str.split("/", 1)
            try:
                used_numbers = extract_numbers(used_str)
                total_numbers = extract_numbers(total_str)
                if not used_numbers or not total_numbers:
                    return None
                used_val = float(used_numbers[0])
                total_val = float(total_numbers[0])
                if total_val <= 0:
                    return None
                return (used_val / total_val) * 100
            except Exception:
                return None

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
                return float(df.iloc[row_index][name])
            except (TypeError, ValueError):
                return None

        def format_bar_cell(text: str, percent: Optional[float], width: int, bar_bg=None) -> str:
            if width <= 0:
                return ""
            text = truncate_text(text, width)

            chars = [" "] * width
            start = max(0, (width - len(text)) // 2)
            for i, ch in enumerate(text):
                pos = start + i
                if 0 <= pos < width:
                    chars[pos] = ch

            if percent is None:
                return "".join(chars)

            try:
                p = max(0.0, min(100.0, float(percent)))
            except Exception:
                return "".join(chars)

            fill_len = int(round((p / 100.0) * width))
            if p > 0 and fill_len == 0:
                fill_len = 1
            fill_len = max(0, min(width, fill_len))

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
        for col in selected_columns:
            width = column_widths.get(col, 12)
            col_name = truncate_text(str(column_labels.get(col, col)), width)
            cell = f"{col_name:^{width}}"
            header_parts.append(cell)
            header_styled_parts.append(colored(cell, "cyan", attrs=["bold"]))
        header_plain = " │ ".join(header_parts)
        header = cell_separator.join(header_styled_parts)

        # Format separator line
        separator_parts = []
        for col in selected_columns:
            width = column_widths.get(col, 12)
            separator_parts.append("─" * width)
        separator = colored("─┼─", "dark_grey").join(separator_parts)

        # Format data rows
        inner_width = len(header_plain)
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
                    row_parts.append(f"{value:<{width}}")
                    continue

                if col == "GPU":
                    cell = f"{value:^{width}}"
                    if row_util_color:
                        cell = colored(cell, row_util_color, attrs=["bold"])
                    row_parts.append(cell)
                    continue

                if col == "util" and value != "N/A":
                    percent = parse_percent(raw_value)
                    display_text = str(raw_value).replace(" ", "")
                    if percent is not None:
                        display_text = f"{int(round(percent))}%"
                    row_parts.append(format_bar_cell(display_text, percent, width))
                    continue

                if col == "mem_util" and value != "N/A":
                    percent = parse_percent(raw_value)
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

                # Center-align all other columns
                row_parts.append(f"{value:^{width}}")

            data_lines.append(cell_separator.join(row_parts))

        result_lines = [header, separator] + data_lines
        data_offset = 2
        if border:
            # A rounded hairline frame instead of ASCII `+--+` art: the
            # corners flow into the rules and the whole border stays one
            # dim grey, so the frame reads as a hairline, not a cage.
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
            data_offset = 3

        if row_line_map is not None:
            for line_index, row_index in row_lines.items():
                row_line_map[data_offset + line_index] = row_index

        return "\n".join(result_lines)
    
    def _get_server_summary(self, stats, system_info):
        """Generate compact summary for collapsed server view."""
        summary_data = self._get_server_summary_data(stats, system_info)
        return self._format_server_summary(summary_data)

    def _get_server_summary_data(self, stats, system_info=None):
        """Compute summary data for formatting/alignment."""
        # Extract system stats from system_info if available
        sys_stats = {}
        data_source = None
        if system_info and isinstance(system_info, dict):
            data_source = system_info.get("data_source")
        if system_info and isinstance(system_info, dict):
            sys_stats = system_info.get("system_stats", {})

        if stats.empty:
            # Return system stats even when no GPU data
            return {
                "gpu_count": 0,
                "idle_count": 0,
                "avg_util": 0,
                "util_color": None,
                "mem_display": "N/A",
                "no_gpu": True,
                # System stats
                "cpu_cores": sys_stats.get("cpu_cores", 0),
                "cpu_percent": sys_stats.get("cpu_percent", 0.0),
                "mem_used_gb": sys_stats.get("mem_used_gb", 0.0),
                "mem_total_gb": sys_stats.get("mem_total_gb", 0.0),
                "swap_used_gb": sys_stats.get("swap_used_gb", 0.0),
                "swap_total_gb": sys_stats.get("swap_total_gb", 0.0),
                "data_source": data_source,
            }

        def mem_to_mib(value_str) -> int:
            if not value_str or str(value_str).strip() in {"N/A", ""}:
                return 0
            try:
                numbers = extract_numbers(str(value_str))
                if not numbers:
                    return 0
                value = float(numbers[0])
                unit = units_from_str(str(value_str)).lower()
                if unit in {"mib", "mb"}:
                    return int(value)
                if unit in {"gib", "gb"}:
                    return int(value * 1024)
                if unit in {"kib", "kb"}:
                    return int(value / 1024)
                if unit in {"b"}:
                    return int(value / (1024 * 1024))
                return int(value)
            except Exception:
                return 0

        gpu_count = len(stats)
        idle_count = 0
        total_util = 0
        total_mem_used_mib = 0
        total_mem_total_mib = 0

        for _, row in stats.iterrows():
            util_str = str(row.get("util", row.get("gpu_util", "0")))
            try:
                util_val = int(util_str.replace("%", "").strip())
                total_util += util_val
                if util_val < 5:
                    idle_count += 1
            except (ValueError, AttributeError):
                pass

            # Prefer explicit used/total fields (from nvidia-smi -q -x); fallback to pre-formatted column
            used_mib = mem_to_mib(row.get("used", ""))
            total_mib = mem_to_mib(row.get("total", ""))

            if used_mib == 0 and total_mib == 0:
                mem_col = row.get("memory[used/total]", "")
                if mem_col and "/" in str(mem_col):
                    try:
                        used_str, total_str = str(mem_col).split("/", 1)
                        used_mib = int("".join(filter(str.isdigit, used_str)) or 0)
                        total_mib = int("".join(filter(str.isdigit, total_str)) or 0)
                    except Exception:
                        used_mib = 0
                        total_mib = 0

            total_mem_used_mib += used_mib
            total_mem_total_mib += total_mib

        avg_util = total_util // gpu_count if gpu_count > 0 else 0

        if total_mem_total_mib > 0:
            if total_mem_total_mib >= 1024:
                mem_display = f"{round(total_mem_used_mib/1024)}GB/{round(total_mem_total_mib/1024)}GB"
            else:
                mem_display = f"{total_mem_used_mib}MB/{total_mem_total_mib}MB"
        else:
            mem_display = "N/A"

        if avg_util >= 80:
            util_color = "red"
        elif avg_util >= 50:
            util_color = "yellow"
        else:
            util_color = "green"

        return {
            "gpu_count": gpu_count,
            "idle_count": idle_count,
            "avg_util": avg_util,
            "util_color": util_color,
            "mem_display": mem_display,
            # System stats
            "cpu_cores": sys_stats.get("cpu_cores", 0),
            "cpu_percent": sys_stats.get("cpu_percent", 0.0),
            "mem_used_gb": sys_stats.get("mem_used_gb", 0.0),
            "mem_total_gb": sys_stats.get("mem_total_gb", 0.0),
            "swap_used_gb": sys_stats.get("swap_used_gb", 0.0),
            "swap_total_gb": sys_stats.get("swap_total_gb", 0.0),
            "data_source": data_source,
        }

    @staticmethod
    def _server_icon_color(expand_icon, toggle_disabled):
        """Expand icon colour: green = open, red = blocked, dim = closed."""
        if toggle_disabled:
            return "red"
        if expand_icon == "v":
            return "green"
        return "dark_grey"

    @staticmethod
    def _server_status_color(summary_data):
        """Server-name colour mirrors the summary's own health colour."""
        if not isinstance(summary_data, dict):
            return None
        status = summary_data.get("status")
        if status == "error":
            return "red"
        if status in ("loading", "empty"):
            return "dark_grey"
        if summary_data.get("no_gpu"):
            return "dark_grey"
        return summary_data.get("util_color")

    def _format_server_summary(self, summary_data, widths=None, *, compact=False) -> str:
        """Format summary string; if widths provided, columns are aligned.

        `compact` renders the plain short form for narrow terminals, where
        the aligned form would push a server header past the terminal edge.
        """
        if compact:
            if not summary_data:
                return "no data"
            status = (
                summary_data.get("status")
                if isinstance(summary_data, dict)
                else None
            )
            if status == "loading":
                return "loading…"
            if status == "error":
                message = " ".join(str(summary_data.get("message", "Error")).split())
                return f"err: {message or 'error'}"
            if status == "empty":
                return "no data"
            if summary_data.get("no_gpu"):
                source = summary_data.get("data_source")
                return "unsupported" if source == "unsupported" else "no GPU"
            return (
                f"{summary_data['gpu_count']}G · {summary_data['avg_util']}% · "
                f"{summary_data['mem_display']}"
            )
        if not summary_data:
            return "No GPU data available"
        if isinstance(summary_data, dict) and summary_data.get("status") == "loading":
            return "Loading..."
        if isinstance(summary_data, dict) and summary_data.get("status") == "error":
            message = summary_data.get("message", "Error")
            error_type = summary_data.get("error_type")
            if error_type == "auth" and str(message).strip() == "Password incorrect":
                return colored("Password incorrect", "red", attrs=["bold"])
            return colored(message, "red", attrs=["bold"])
        if isinstance(summary_data, dict) and summary_data.get("status") == "empty":
            return "No GPU data available"

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
            if cpu_percent_int >= 80:
                cpu_color = "red"
            elif cpu_percent_int >= 50:
                cpu_color = "yellow"
            else:
                cpu_color = "green"
            cpu_util_str = colored(f"{cpu_percent_int:>3}%", cpu_color)
            cpu_part = f"CPU: {cpu_util_str}({cpu_cores}C)"
        else:
            cpu_part = ""

        # Memory: used/total with swap in brackets
        if mem_total_gb > 0:
            if swap_total_gb > 0:
                mem_sys_part = f"Mem: {mem_used_gb:.0f}/{mem_total_gb:.0f}G(Swap:{swap_used_gb:.1f}/{swap_total_gb:.0f}G)"
            else:
                mem_sys_part = f"Mem: {mem_used_gb:.0f}/{mem_total_gb:.0f}G"
        else:
            mem_sys_part = ""

        # Handle no GPU case - only show system stats
        if no_gpu:
            source = summary_data.get("data_source") or "unknown"
            label = "Unsupported" if source == "unsupported" else "No GPU"
            sys_parts = [p for p in [cpu_part, mem_sys_part] if p]
            if sys_parts:
                return f"{label} | src:{source} | " + " | ".join(sys_parts)
            return f"{label} | src:{source}"

        if widths:
            gpu_digits = widths.get("gpu_digits", len(str(gpu_count)))
            idle_digits = widths.get("idle_digits", len(str(idle_count)))
            util_digits = widths.get("util_digits", len(str(avg_util)))
            mem_width = widths.get("mem_width", len(str(mem_display)))
        else:
            gpu_digits = len(str(gpu_count))
            idle_digits = len(str(idle_count))
            util_digits = len(str(avg_util))
            mem_width = len(str(mem_display))

        gpu_part = f"{gpu_count:>{gpu_digits}} GPUs"
        idle_part = f"{idle_count:>{idle_digits}} idle"
        util_plain = f"{avg_util:>{util_digits}}%"
        util_part = colored(util_plain, util_color) if util_color else util_plain
        mem_part = f"{str(mem_display):>{mem_width}}"

        # Build the summary string
        parts = [f"{gpu_part} | {idle_part} | {util_part} avg | {mem_part}"]
        sys_parts = [p for p in [cpu_part, mem_sys_part] if p]
        if sys_parts:
            parts.append(" | ".join(sys_parts))

        source = summary_data.get("data_source")
        if source:
            parts.append(f"src:{source}")

        return " | ".join(parts)

    def _format_user_memory_totals(self, user_summary: dict, *, max_users: int = 10) -> str:
        """Format per-user total VRAM usage (MiB) as a compact single-line string."""
        if not user_summary:
            return ""

        items = []
        for user, mib in user_summary.items():
            try:
                mib_int = int(mib)
            except Exception:
                continue
            if mib_int <= 0:
                continue
            items.append((str(user), mib_int))

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
        if not isinstance(raw_stats_by_client, dict):
            raw_stats_by_client = {}

        self._apply_default_expansion(raw_stats_by_client, last_update_time)

        # Disable expand/collapse for servers with auth errors (e.g., password incorrect)
        try:
            disabled = set()
            for idx in range(len(self.pool)):
                _stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))
                if isinstance(system_info, dict) and system_info.get("error_type") == "auth":
                    disabled.add(idx)
            self._toggle_disabled_servers = disabled
        except Exception:
            self._toggle_disabled_servers = set()
        
        current_time = time.strftime("%H:%M:%S")
        
        # Ensure selected_server is within bounds
        if self.selected_server >= len(self.pool):
            self.selected_server = len(self.pool) - 1
        if self.selected_server < 0:
            self.selected_server = 0

        update_display = "--:--:--"
        if last_update_time:
            update_display = time.strftime("%H:%M:%S", time.localtime(last_update_time))
        fetch_display = ""
        if isinstance(last_fetch_duration, (int, float)):
            fetch_display = f" ({last_fetch_duration:.1f}s)"
        warn_display = " | WARN: refresh failed" if last_fetch_error else ""

        output_lines = []
        server_count = len(self.pool)
        server_label = "Server" if server_count == 1 else "Servers"
        try:
            terminal_size = os.get_terminal_size()
            terminal_width = terminal_size.columns
            terminal_height = terminal_size.lines
        except OSError:
            terminal_width = 80
            terminal_height = 24

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
            and not bool(self.tui_help_visible)
        )
        if show_panel_border:
            terminal_width -= self._PANEL_BORDER_MARGIN
        if display_mode == self.DISPLAY_MODE_UNIFIED:
            detailed = self.unified_detailed
            view_label = "Unified/Detailed" if detailed else "Unified/Single-line"
            detail_action = "Single-line" if detailed else "Detailed"
            process_action = (
                "HideProc" if self._process_panel_visible() else "ShowProc"
            )
            sort_mode = self._get_unified_sort_mode()
            sort_label = {
                "node": "Node",
                "available": "Free",
                "utilization": "Util",
            }[sort_mode]
            filter_mode = self._get_unified_filter_mode()
            filter_label = self.UNIFIED_FILTER_LABELS[filter_mode]
            group_label = (
                "Group" if self.unified_group_by_node else "Flat"
            )
            unsupported_label = (
                "Show" if self.hide_unsupported else "Hide"
            )
            process_focused = (
                self.unified_active_pane == "process"
                and self._process_panel_visible()
            )
            if process_focused:
                process_sort_mode = self._get_unified_process_sort_mode()
                process_sort_label = self.UNIFIED_PROCESS_SORT_LABELS[
                    process_sort_mode
                ]
                process_sort_arrow = (
                    "▼" if self.unified_process_sort_descending else "▲"
                )
                process_filter = str(
                    self.unified_process_filter or ""
                )
                find_label = (
                    f"Find:{process_filter[:12]}"
                    if process_filter
                    else "Find"
                )
                controls = (
                    f"[?]Help [/]{find_label} "
                    f"[o]{process_sort_label}{process_sort_arrow} "
                    f"[+/-]Rows [←]GPU [j/k]Select [i/T/K]Signal "
                    f"[t]History [p]{process_action} [q]"
                )
                if compact_layout:
                    controls = (
                        f"? · /{find_label} · o sort · j/k sel · "
                        f"t hist · p · q"
                    )
            else:
                controls = (
                    f"[?]Help [Enter/→]Proc [j/k]GPU [p]{process_action} "
                    f"[d]{detail_action} [s]{sort_label} [f]{filter_label} "
                    f"[q] [g]{group_label} [u]{unsupported_label} [v]Nodes"
                )
                if compact_layout:
                    controls = f"? · ⏎ proc · j/k GPU · d view · q quit"
            if len(controls) > terminal_width:
                controls = controls[: max(0, terminal_width - 3)] + "..."
        else:
            view_label = "Per-node"
            controls = (
                "[?] Help  [v] Unified view  [j/k] Select  "
                "[Enter] Toggle  [a/c] Expand/Collapse  [q] Quit"
            )
            if compact_layout:
                controls = "? help · v view · j/k sel · ⏎ exp · q quit"
        if bool(self.tui_help_visible):
            controls = "[? / Esc / q] Close help"
        if compact_layout:
            plain_title = (
                f"nvidb · {server_count} {server_label.lower()} · {current_time}"
                + (" · refresh failed" if last_fetch_error else "")
            )
            title_line = (
                fit(plain_title, terminal_width)
                if len(plain_title) > terminal_width
                else colored("nvidb", "green", attrs=["bold"])
                + colored(" · ", "dark_grey")
                + colored(f"{server_count}", "magenta")
                + f" {server_label.lower()}"
                + colored(" · ", "dark_grey")
                + colored(current_time, "cyan")
                + (colored(" · refresh failed", "red") if last_fetch_error else "")
            )
        else:
            plain_title = (
                f"Time: {current_time} | Updated: {update_display}{fetch_display} | "
                f"{server_label}: {server_count} | View: {view_label}{warn_display}"
            )
            title_line = (
                fit(plain_title, terminal_width)
                if len(plain_title) > terminal_width
                else "Time: " + colored(current_time, "cyan")
                + " | Updated: " + colored(f"{update_display}{fetch_display}", "cyan")
                + " | " + f"{server_label}: " + colored(str(server_count), "magenta")
                + " | View: " + colored(view_label, "green")
                + (colored(warn_display, "red") if warn_display else "")
            )
        output_lines.append(title_line)

        def _colorize_controls(text):
            """Highlight [key] shortcut patterns in cyan bold."""
            parts = re.split(r'(\[[^\]]+\])', text)
            result = []
            for part in parts:
                if part.startswith('[') and part.endswith(']'):
                    result.append(colored(part, "cyan", attrs=["bold"]))
                else:
                    result.append(part)
            return "".join(result)

        output_lines.append(_colorize_controls(fit(controls, terminal_width)))

        if self.compact:
            separator_width = min(80, terminal_width)
        else:
            separator_width = max(20, terminal_width)
        output_lines.append(colored("─" * separator_width, "dark_grey"))

        if bool(self.tui_help_visible):
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
            self._click_regions = []
            self._body_click_targets = {}
            self._body_click_regions = []
            self._write_tui_lines(output_lines)
            return

        meta = {}
        if isinstance(raw_stats_by_client, dict):
            meta = raw_stats_by_client.get("_nvidb", {}) or {}
        user_memory_by_client = meta.get("user_memory_by_client", {}) or {}
        global_user_memory = meta.get("user_memory_global", {}) or {}

        compact_detailed_screen = (
            display_mode == self.DISPLAY_MODE_UNIFIED
            and bool(self.unified_detailed)
            and (
                terminal_height < 28
                or (
                    terminal_height < 36
                    and bool(self.unified_show_trends)
                )
            )
        )
        if global_user_memory and not compact_detailed_screen:
            user_totals = self._format_user_memory_totals(global_user_memory, max_users=12)
            plain_line = f"Users (all nodes): {user_totals}"
            if self.term.length(plain_line) > terminal_width:
                plain_line = plain_line[: max(0, terminal_width - 3)] + "..."
                output_lines.append(plain_line)
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
            )
            output_lines.extend(body)
            # Translate the body-relative hit boxes into absolute screen rows.
            self._click_targets = {
                body_offset + line: target
                for line, target in (self._body_click_targets or {}).items()
            }
            self._click_regions = [
                (body_offset + line, start, end, target)
                for line, start, end, target in (
                    self._body_click_regions or []
                )
            ]
            self._write_tui_lines(output_lines)
            return

        self._click_targets = {}
        self._click_regions = []
        self._body_click_targets = {}
        self._body_click_regions = []

        # Align the collapsed server list by padding headers to the same display width
        index_width = len(str(len(self.pool)))
        server_rows = []
        max_header_width = 0

        summary_rows = []
        for idx, (client, stats_info) in enumerate(zip(self.pool, stats_list)):
            is_selected = idx == self.selected_server
            is_expanded = idx in self.expanded_servers
            toggle_disabled = idx in self._toggle_disabled_servers

            # Use cached raw stats for summary
            stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))

            # Build summary
            if isinstance(system_info, dict) and system_info.get("error"):
                summary_data = {
                    "status": "error",
                    "message": system_info.get("error", "Error"),
                    "error_type": system_info.get("error_type"),
                }
            elif last_update_time is None and stats.empty:
                summary_data = {"status": "loading"}
            else:
                summary_data = self._get_server_summary_data(stats, system_info) or {"status": "empty"}

            # Format the header line (index padded for consistent width)
            if toggle_disabled:
                expand_icon = "!"
            else:
                expand_icon = "v" if is_expanded else ">"
            selector = "*" if is_selected else " "
            header_plain = f"{selector} {expand_icon} [{idx + 1:{index_width}d}] {client.description}"

            header_width = self.term.length(header_plain)
            max_header_width = max(max_header_width, header_width)

            summary_rows.append(summary_data)
            server_rows.append((idx, is_selected, is_expanded, header_plain, summary_data, stats_info))

        non_empty_summaries = [s for s in summary_rows if isinstance(s, dict) and "gpu_count" in s]
        if non_empty_summaries:
            widths = {
                "gpu_digits": max(len(str(s["gpu_count"])) for s in non_empty_summaries),
                "idle_digits": max(len(str(s["idle_count"])) for s in non_empty_summaries),
                "util_digits": max(len(str(s["avg_util"])) for s in non_empty_summaries),
                "mem_width": max(len(str(s["mem_display"])) for s in non_empty_summaries),
            }
        else:
            widths = None

        # All-expanded by default means the stacked tables must share the
        # terminal: scale them to the rows that remain under the headers
        # so the whole cluster stays on one screen.
        border_rows = 2 if show_panel_border else 0
        separator_count = max(0, len(server_rows) - 1) if show_panel_border else 0
        available_rows = (
            (terminal_height - 1)
            - self._screen_line_count(output_lines)
            - len(server_rows)
            - border_rows
        )
        # Dividers between servers are dropped first when the terminal is
        # short, so a tight window still favours GPU data over rules.
        show_separators = separator_count > 0 and available_rows - separator_count >= 0
        if show_separators:
            available_rows -= separator_count
        visible_blocks = self._scale_server_blocks(
            server_rows,
            user_memory_by_client,
            available_rows,
            keep_expanded=self.selected_server,
        )

        for position, (idx, is_selected, _is_expanded, _header_plain, summary_data, _stats_info) in enumerate(server_rows):
            toggle_disabled = idx in self._toggle_disabled_servers
            # A server the scaler collapsed still gets its header, just
            # with the collapsed icon.
            expand_icon = (
                "!"
                if toggle_disabled
                else ("v" if idx in visible_blocks else ">")
            )
            selector = "*" if is_selected else " "
            summary = self._format_server_summary(
                summary_data, widths=widths, compact=compact_layout
            )
            # A server header is exactly one terminal line. The description
            # gives ground to the summary when both cannot fit, so the
            # terminal never soft-wraps the row and shears the click map.
            summary_room = terminal_width - 2 - len(self._ANSI_ESCAPE_RE.sub("", summary))
            # Colour is skipped on the selected row: it is already picked
            # out with reverse video, and termcolor's per-segment reset
            # codes would cut that reverse video short partway through.
            icon_color = None if is_selected else self._server_icon_color(
                expand_icon, toggle_disabled
            )
            status_color = None if is_selected else self._server_status_color(
                summary_data
            )
            if compact_layout:
                prefix = f"{selector} {expand_icon} {idx + 1} "
                description = fit(
                    self.pool[idx].description,
                    max(1, summary_room - len(prefix)),
                )
                if is_selected:
                    header_display = (
                        self.term.reverse + prefix + description + self.term.normal
                    )
                else:
                    icon_display = (
                        colored(expand_icon, icon_color) if icon_color else expand_icon
                    )
                    description_display = (
                        colored(description, status_color, attrs=["bold"])
                        if status_color
                        else description
                    )
                    header_display = (
                        f"{selector} {icon_display} {idx + 1} {description_display}"
                    )
            else:
                header_plain = (
                    f"{selector} {expand_icon} "
                    f"[{idx + 1:{index_width}d}] {self.pool[idx].description}"
                )
                pad = max_header_width - self.term.length(header_plain)
                header_padded = header_plain + (" " * pad if pad > 0 else "")
                header_padded = header_padded[: max(1, summary_room)]
                if is_selected:
                    header_display = (
                        self.term.reverse + header_padded + self.term.normal
                    )
                else:
                    # The icon and index prefix are a handful of ASCII
                    # characters, always shorter than summary_room in
                    # practice (this branch only runs at width >= 72); the
                    # rest (description + trailing pad) takes the status
                    # colour as one block so a mid-word cut never splits
                    # an ANSI code.
                    prefix_plain = (
                        f"{selector} {expand_icon} [{idx + 1:{index_width}d}] "
                    )
                    if len(header_padded) >= len(prefix_plain):
                        rest = header_padded[len(prefix_plain):]
                        icon_display = (
                            colored(expand_icon, icon_color)
                            if icon_color
                            else expand_icon
                        )
                        rest_display = (
                            colored(rest, status_color, attrs=["bold"])
                            if status_color
                            else rest
                        )
                        header_display = (
                            f"{selector} {icon_display} "
                            f"[{idx + 1:{index_width}d}] {rest_display}"
                        )
                    else:
                        header_display = header_padded
            row = f"{header_display}  {summary}"
            if len(self._ANSI_ESCAPE_RE.sub("", row)) > terminal_width:
                # A pathologically long error message is the one input that
                # can still overflow; dropping its colour beats wrapping.
                row = fit(self._ANSI_ESCAPE_RE.sub("", row), terminal_width)
            self._click_targets[self._screen_line_count(output_lines)] = ("server", idx)
            output_lines.append(row)

            # The full stats table, already formatted and scaled to fit.
            block = visible_blocks.get(idx)
            if block is not None:
                output_lines.extend(block)

            if show_separators and position < len(server_rows) - 1:
                output_lines.append(self._PANEL_SEPARATOR)

        if show_panel_border:
            output_lines = self._wrap_panel_border(output_lines, terminal_width)
            self._click_targets = {
                row + 1: target for row, target in self._click_targets.items()
            }

        self._write_tui_lines(output_lines)

    def _request_ui_refresh(self):
        self.ui_only_refresh = True
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
        }

    def _persist_view_settings(self):
        """Write the current layout to config.yml; never break the TUI on error."""
        if not self._persist_view_enabled:
            return
        try:
            nvidb_config.save_view_settings(self._current_view_settings())
        except Exception as error:
            logging.debug(msg=f"Failed to persist view settings: {error}")

    def _apply_view_change(self):
        self._request_ui_refresh()
        self._persist_view_settings()

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
        self.unified_sort_mode = self.UNIFIED_SORT_MODES[next_index]
        self.unified_selected_gpu = 0
        self.unified_selected_gpu_key = None
        self.unified_selected_process = 0
        self.unified_selected_process_pid = None
        self.unified_command_scroll = 0
        self._pending_process_signal = None
        self._apply_view_change()

    def _cycle_unified_filter_mode(self):
        current_mode = self._get_unified_filter_mode()
        next_index = (
            self.UNIFIED_FILTER_MODES.index(current_mode) + 1
        ) % len(self.UNIFIED_FILTER_MODES)
        self.unified_filter_mode = self.UNIFIED_FILTER_MODES[next_index]
        self.unified_selected_gpu = 0
        self.unified_selected_gpu_key = None
        self.unified_selected_process = 0
        self.unified_selected_process_pid = None
        self.unified_command_scroll = 0
        self.unified_active_pane = "gpu"
        self._pending_process_signal = None
        self._apply_view_change()

    def _move_unified_selection(self, delta):
        gpu_count = max(0, int(self._unified_gpu_count or 0))
        if gpu_count == 0:
            return False
        current = int(self.unified_selected_gpu or 0)
        selected = max(0, min(current + delta, gpu_count - 1))
        if selected == current:
            return False
        self.unified_selected_gpu = selected
        self.unified_selected_gpu_key = None
        self.unified_selected_process = 0
        self.unified_selected_process_pid = None
        self.unified_command_scroll = 0
        self._pending_process_signal = None
        self._request_ui_refresh()
        return True

    def _move_unified_process_selection(self, delta):
        count = max(0, int(self._unified_process_count or 0))
        if count == 0:
            return False
        current = int(self.unified_selected_process or 0)
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
        current_descending = bool(
            getattr(
                self,
                "unified_process_sort_descending",
                current != "command",
            )
        )
        if mode == current and toggle:
            descending = not current_descending
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
        current = int(self.unified_process_rows or 0)
        if current <= 0:
            current = max(
                1,
                int(
                    getattr(
                        self,
                        "_unified_process_visible_rows",
                        1,
                    )
                    or 1
                ),
            )
        selected = max(1, min(12, current + int(delta)))
        if selected == current:
            return False
        self.unified_process_rows = selected
        self._request_ui_refresh()
        return True

    def _reset_unified_process_filter_selection(self):
        self.unified_selected_process = 0
        self.unified_selected_process_pid = None
        self.unified_command_scroll = 0
        self._pending_process_signal = None

    def _start_unified_process_filter(self):
        if max(0, int(self._unified_gpu_count or 0)) == 0:
            return False
        entered = self._enter_unified_process_pane()
        changed = not bool(
            self.unified_process_filter_editing
        )
        self.unified_process_filter_editing = True
        self._pending_process_signal = None
        if changed and not entered:
            self._request_ui_refresh()
        return changed or entered

    def _handle_unified_process_filter_key(self, key_text, key_name):
        """Edit the live process filter while keeping global shortcuts inactive."""
        if not bool(
            self.unified_process_filter_editing
        ):
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
            self._reset_unified_process_filter_selection()
            self._request_ui_refresh()
            return True

        if (
            key_name in {"KEY_BACKSPACE", "KEY_DELETE"}
            or key_text in {"\b", "\x7f"}
        ):
            query = str(
                self.unified_process_filter or ""
            )
            if query:
                self.unified_process_filter = query[:-1]
                self._reset_unified_process_filter_selection()
                self._request_ui_refresh()
            return True

        if key_text == "\x15":  # Ctrl+U
            self.unified_process_filter = ""
            self._reset_unified_process_filter_selection()
            self._request_ui_refresh()
            return True

        if (
            key_text
            and not str(key_name).startswith("KEY_")
            and all(character.isprintable() for character in key_text)
        ):
            query = str(
                self.unified_process_filter or ""
            )
            self.unified_process_filter = (query + key_text)[:80]
            self._reset_unified_process_filter_selection()
            self._request_ui_refresh()
            return True
        return False

    def _clear_unified_process_filter(self):
        query = str(self.unified_process_filter or "")
        editing = bool(
            self.unified_process_filter_editing
        )
        if not query and not editing:
            return False
        self.unified_process_filter = ""
        self.unified_process_filter_editing = False
        self._reset_unified_process_filter_selection()
        self._request_ui_refresh()
        return True

    def _toggle_tui_help(self):
        self.tui_help_visible = not bool(
            self.tui_help_visible
        )
        self.unified_process_filter_editing = False
        self._pending_process_signal = None
        self._request_ui_refresh()
        return True

    def _scroll_unified_command(self, page_delta):
        line_count = max(
            0,
            int(self._unified_command_line_count or 0),
        )
        page_size = max(
            0,
            int(self._unified_command_page_size or 0),
        )
        if page_size <= 0 or line_count <= page_size:
            return False
        current = max(
            0,
            int(self.unified_command_scroll or 0),
        )
        selected = max(
            0,
            min(
                current + int(page_delta) * page_size,
                line_count - page_size,
            ),
        )
        if selected == current:
            return False
        self.unified_command_scroll = selected
        self._request_ui_refresh()
        return True

    def _process_panel_visible(self):
        if bool(self.unified_process_panel_hidden):
            return False
        return (
            self.display_mode
            == self.DISPLAY_MODE_UNIFIED
            and (
                bool(self.unified_detailed)
                or bool(self.unified_show_processes)
            )
            and self._get_unified_filter_mode() != "errors"
        )

    def _enter_unified_process_pane(self):
        """Show the selected GPU's processes and move focus into that pane."""
        if (
            self.display_mode
            != self.DISPLAY_MODE_UNIFIED
            or self._get_unified_filter_mode() == "errors"
            or max(0, int(self._unified_gpu_count or 0)) == 0
        ):
            return False

        visibility_changed = False
        persist_visibility = False
        if bool(self.unified_detailed):
            if bool(
                self.unified_process_panel_hidden
            ):
                self.unified_process_panel_hidden = False
                visibility_changed = True
        elif not bool(self.unified_show_processes):
            self.unified_show_processes = True
            visibility_changed = True
            persist_visibility = True

        if not self._process_panel_visible():
            return False
        focus_changed = (
            self.unified_active_pane != "process"
        )
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
            self.display_mode
            != self.DISPLAY_MODE_UNIFIED
            or self._get_unified_filter_mode() == "errors"
            or max(0, int(self._unified_gpu_count or 0)) == 0
        ):
            return False

        detailed = bool(self.unified_detailed)
        if detailed:
            self.unified_process_panel_hidden = not bool(
                self.unified_process_panel_hidden
            )
            panel_visible = not self.unified_process_panel_hidden
        else:
            self.unified_process_panel_hidden = False
            panel_visible = not bool(
                self.unified_show_processes
            )
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
        index = int(self.unified_selected_process or 0)
        selected_pid = self.unified_selected_process_pid
        if selected_pid is not None:
            for process_index, process in enumerate(processes):
                if str(process.get("pid", "")) == str(selected_pid):
                    index = process_index
                    break
        index = max(
            0,
            min(index, len(processes) - 1),
        )
        process = processes[index]
        self.unified_selected_process = index
        self.unified_selected_process_pid = str(process.get("pid", ""))
        client_index = self._unified_client_index_for_row(row)
        if client_index is None or not 0 <= client_index < len(self.pool):
            return None
        return {
            "row": row,
            "process": process,
            "client_index": client_index,
        }

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
            "command": str(
                process.get("command")
                or process.get("process_name")
                or ""
            ),
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
            or str(
                context["process"].get("command")
                or context["process"].get("process_name")
                or ""
            )
            != pending.get("command")
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

        if bool(self.tui_help_visible):
            if (
                event.is_left_press
                and target is not None
                and target[0] == "close_help"
            ):
                return self._toggle_tui_help()
            return False

        if self.display_mode != self.DISPLAY_MODE_UNIFIED:
            if event.is_wheel_up or event.is_wheel_down:
                delta = -1 if event.is_wheel_up else 1
                selected = max(
                    0,
                    min(self.selected_server + delta, len(self.pool) - 1),
                )
                if selected == self.selected_server:
                    return False
                self.selected_server = selected
                self._request_ui_refresh()
                return True
            if not event.is_left_press:
                return False
            if target is None or target[0] != "server":
                return False
            index = target[1]
            # First click selects the node; clicking it again expands/collapses.
            if index == self.selected_server:
                return self._toggle_server_expansion(index)
            self.selected_server = index
            self._request_ui_refresh()
            return True

        if event.is_wheel_up or event.is_wheel_down:
            delta = -1 if event.is_wheel_up else 1
            kind = target[0] if target else None
            if kind == "gpu":
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
            self.unified_show_trends = not getattr(
                self,
                "unified_show_trends",
                False,
            )
            self._apply_view_change()
            return True
        if kind == "process":
            changed = (
                value
                != int(self.unified_selected_process or 0)
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
        if kind == "command":
            changed = (
                value
                != int(self.unified_selected_process or 0)
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
                value != int(self.unified_selected_gpu or 0)
                or self.unified_active_pane != "gpu"
            )
            if not changed:
                return False
            self.unified_selected_gpu = value
            self.unified_selected_gpu_key = None
            self.unified_selected_process = 0
            self.unified_selected_process_pid = None
            self.unified_command_scroll = 0
            self.unified_active_pane = "gpu"
            self._pending_process_signal = None
            self._request_ui_refresh()
            return True
        return False

    def _handle_keypress(self, key):
        """Apply one TUI keypress and return whether it changed visible state."""
        key_text = str(key)
        key_name = key.name if hasattr(key, "name") and key.name else key_text
        key_lower = key_text.lower()

        if bool(self.tui_help_visible):
            if (
                key_text in {"?", "q", "Q", "\n", "\r", " "}
                or key_name in {"KEY_ESCAPE", "KEY_ENTER"}
                or key_text == "\x1b"
            ):
                return self._toggle_tui_help()
            return False

        if bool(
            self.unified_process_filter_editing
        ):
            return self._handle_unified_process_filter_key(
                key_text,
                key_name,
            )

        if key_text == "?":
            return self._toggle_tui_help()

        if key_lower == "q":
            self.quit_flag.set()
            return True

        if key_name == "KEY_ESCAPE" or key_text == "\x1b":
            if self._pending_process_signal:
                self._pending_process_signal = None
                self._set_process_action_notice("Signal cancelled", "yellow")
                self._request_ui_refresh()
                return True
            if self._clear_unified_process_filter():
                return True
            return False

        enter_pressed = (
            key_name == "KEY_ENTER"
            or key_text in {"\n", "\r"}
        )
        if enter_pressed and self._pending_process_signal:
            return self._confirm_process_signal()

        if key_lower == "v":
            self._toggle_display_mode()
            return True
        if key_lower == "d":
            if self.display_mode != self.DISPLAY_MODE_UNIFIED:
                return False
            self.unified_detailed = not self.unified_detailed
            self.unified_active_pane = "gpu"
            self.unified_process_panel_hidden = False
            self.unified_selected_process = 0
            self.unified_selected_process_pid = None
            self.unified_command_scroll = 0
            self._pending_process_signal = None
            self._apply_view_change()
            return True
        if key_lower == "g":
            if self.display_mode != self.DISPLAY_MODE_UNIFIED:
                return False
            self.unified_group_by_node = not getattr(
                self,
                "unified_group_by_node",
                True,
            )
            self._apply_view_change()
            return True
        if key_lower == "u":
            if self.display_mode != self.DISPLAY_MODE_UNIFIED:
                return False
            self.hide_unsupported = not self.hide_unsupported
            self._apply_view_change()
            return True
        if key_lower == "s":
            if self.display_mode != self.DISPLAY_MODE_UNIFIED:
                return False
            self._cycle_unified_sort_mode()
            return True
        if key_lower == "f":
            if self.display_mode != self.DISPLAY_MODE_UNIFIED:
                return False
            self._cycle_unified_filter_mode()
            return True
        if self.display_mode == self.DISPLAY_MODE_UNIFIED:
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
                if max(0, int(self._unified_gpu_count or 0)) == 0:
                    return False
                self.unified_show_trends = not getattr(
                    self,
                    "unified_show_trends",
                    False,
                )
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
            if key_name in {"KEY_NPAGE", "KEY_PAGEDOWN"}:
                if process_pane:
                    return self._move_unified_process_selection(5)
                return self._move_unified_selection(
                    max(1, int(self._unified_page_size or 1))
                )
            if key_name in {"KEY_PPAGE", "KEY_PAGEUP"}:
                if process_pane:
                    return self._move_unified_process_selection(-5)
                return self._move_unified_selection(
                    -max(1, int(self._unified_page_size or 1))
                )
            if enter_pressed or key_text == " ":
                return self._enter_unified_process_pane()
            return False

        if key_lower == "j" or key_name == "KEY_DOWN":
            if self.selected_server < len(self.pool) - 1:
                self.selected_server += 1
                self._request_ui_refresh()
                return True
        elif key_lower == "k" or key_name == "KEY_UP":
            if self.selected_server > 0:
                self.selected_server -= 1
                self._request_ui_refresh()
                return True
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

    def _keyboard_listener(self, cbreak_context):
        """Real-time keyboard listener thread, monitors keys for navigation and control."""
        parser = MouseSequenceParser()
        try:
            while not self.quit_flag.is_set():
                try:
                    key = self.term.inkey(timeout=0.1)
                    if not key:
                        # An escape that never became a mouse report is just Esc.
                        for pending in parser.flush():
                            self._handle_keypress(pending)
                        continue
                    events, keys = parser.feed(key)
                    for event in events:
                        self._handle_mouse_event(event)
                    for pending in keys:
                        self._handle_keypress(pending)
                except KeyboardInterrupt:
                    break
        except Exception as error:
            # Swallowing this silently would hide every input bug behind a
            # dead keyboard, so at least log it before the listener dies.
            logging.error(msg=f"Keyboard listener crashed: {error}")

    def _start_mouse_reporting(self):
        """Ask the terminal for SGR mouse reports, when the user wants them."""
        self._mouse_reporting = False
        if not self.mouse_enabled:
            return
        try:
            if not sys.stdout.isatty():
                return
            sys.stdout.write(MOUSE_ENABLE_SEQUENCE)
            sys.stdout.flush()
            self._mouse_reporting = True
        except Exception as error:
            logging.debug(msg=f"Failed to enable mouse reporting: {error}")

    def _stop_mouse_reporting(self):
        """Always restore normal terminal selection behaviour on the way out."""
        if not self._mouse_reporting:
            return
        self._mouse_reporting = False
        try:
            sys.stdout.write(MOUSE_DISABLE_SEQUENCE)
            sys.stdout.flush()
        except Exception:
            pass

    def _background_refresh(self, interval_seconds: float = 1.0):
        """Background thread: fetch stats periodically so UI thread stays responsive."""
        while not self.quit_flag.is_set():
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

            # Trigger UI refresh (both for new data and errors)
            self.refresh_needed.set()

            # Keep a minimum idle interval between refreshes to avoid hammering remote hosts
            sleep_seconds = max(0.0, interval_seconds)
            end_time = time.time() + sleep_seconds
            while time.time() < end_time and not self.quit_flag.is_set():
                time.sleep(0.1)
    
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
        cbreak_ctx = self.term.cbreak()
        cbreak_ctx.__enter__()
        self._start_mouse_reporting()

        # Background threads (auto-reconnect, etc.) call logging.error/warning/info
        # directly, which write straight to the console. Any such line printed
        # while print_stats is doing cursor-positioned diff rendering corrupts the
        # screen. Each node's own error is already rendered in-frame via
        # _format_error_panel, so muting the console logger here loses nothing.
        logging.disable(logging.CRITICAL)

        try:
            # Start keyboard listener thread (uses cbreak mode from main thread)
            keyboard_thread = threading.Thread(target=self._keyboard_listener, args=(cbreak_ctx,), daemon=True)
            keyboard_thread.start()

            # Start background data refresh thread (prevents remote SSH calls from blocking UI)
            refresh_thread = threading.Thread(target=self._background_refresh, daemon=True)
            refresh_thread.start()
            
            with self.term.hidden_cursor():
                # Initial draw (may show "Loading..." until first background refresh completes)
                self.print_stats(use_cache=True)

                while not self.quit_flag.is_set():
                    # Event-driven redraw: avoids back-to-back full redraws (less flicker)
                    self.refresh_needed.wait(timeout=1.0)
                    self.refresh_needed.clear()
                    if self.quit_flag.is_set():
                        break
                    self.print_stats(use_cache=True)
                
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
                cbreak_ctx.__exit__(None, None, None)
            except Exception:
                pass
            # Print newline to ensure clean prompt
            print()

    def print_once(self):
        """Print GPU stats once and exit (no TUI loop)"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        server_count = len(self.pool)
        server_label = "Server" if server_count == 1 else "Servers"
        print(f"Time: {current_time} | {server_label}: {server_count}")
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80
        if self.compact:
            separator_width = min(80, terminal_width)
        else:
            separator_width = max(20, terminal_width)
        print("-" * separator_width)
        
        # Get stats
        stats_list, raw_stats_by_client = self.get_client_gpus_info(return_raw=True)

        meta = {}
        if isinstance(raw_stats_by_client, dict):
            meta = raw_stats_by_client.get("_nvidb", {}) or {}
        user_memory_by_client = meta.get("user_memory_by_client", {}) or {}
        global_user_memory = meta.get("user_memory_global", {}) or {}
        
        for idx, (client, stats_info) in enumerate(zip(self.pool, stats_list)):
            stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))
            
            # Build summary
            summary = self._get_server_summary(stats, system_info)
            
            # Print header with summary
            print(f"[{idx + 1}] {client.description}  {summary}")
            
            # Print the full stats table
            print(stats_info)
            if isinstance(user_memory_by_client, dict):
                user_summary = user_memory_by_client.get(idx, {}) or {}
            else:
                user_summary = {}
            if user_summary:
                print(f"Users: {self._format_user_memory_totals(user_summary, max_users=12)}")
            print()  # Add spacing after each server

        if global_user_memory:
            print(f"Users (all nodes): {self._format_user_memory_totals(global_user_memory, max_users=12)}")
