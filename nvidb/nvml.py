from __future__ import annotations

import base64
import os
import textwrap
import threading
import time
from typing import Any, Optional

import pynvml


_ARCHITECTURE_NAMES = {
    2: "Kepler",
    3: "Maxwell",
    4: "Pascal",
    5: "Volta",
    6: "Turing",
    7: "Ampere",
    8: "Ada",
    9: "Hopper",
    10: "Blackwell",
    11: "T23x",
}
_UINT_MAX = (1 << 32) - 1
_ULONG_LONG_MAX = (1 << 64) - 1


def _decode(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _cuda_version(value: Any) -> Optional[str]:
    try:
        version = int(value)
    except (TypeError, ValueError):
        return None
    major = version // 1000
    minor = (version % 1000) // 10
    revision = version % 10
    return f"{major}.{minor}.{revision}" if revision else f"{major}.{minor}"


def _process_username(pid: int) -> Optional[str]:
    try:
        import pwd

        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (ImportError, KeyError, OSError):
        return None


def _process_name(pid: int) -> Optional[str]:
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


class PynvmlCollector:
    """Keep one local NVML context alive and return lightweight snapshots."""

    def __init__(self, *, slow_metric_ttl: float = 5.0):
        self.slow_metric_ttl = max(0.0, float(slow_metric_ttl))
        self._lock = threading.RLock()
        self._initialized = False
        self._handles = []
        self._device_static = []
        self._slow_metrics = {}
        self._driver_version = None
        self._cuda_version = None

    @staticmethod
    def _safe(func, *args, default=None):
        if not callable(func):
            return default
        try:
            return func(*args)
        except Exception:
            return default

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        pynvml.nvmlInit()
        self._initialized = True
        self._driver_version = _decode(
            self._safe(getattr(pynvml, "nvmlSystemGetDriverVersion", None))
        )

        cuda_value = None
        for name in ("nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion"):
            cuda_value = self._safe(getattr(pynvml, name, None))
            if cuda_value is not None:
                break
        self._cuda_version = _cuda_version(cuda_value)
        self._refresh_handles()

    def _refresh_handles(self) -> None:
        count = int(pynvml.nvmlDeviceGetCount())
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)]
        self._device_static = []
        self._slow_metrics.clear()

        architecture_func = getattr(pynvml, "nvmlDeviceGetArchitecture", None)
        for index, handle in enumerate(self._handles):
            name = _decode(self._safe(getattr(pynvml, "nvmlDeviceGetName", None), handle))
            architecture = self._safe(architecture_func, handle)
            self._device_static.append(
                {
                    "gpu_index": index,
                    "name": name,
                    "architecture": _ARCHITECTURE_NAMES.get(architecture),
                    # The widest, fastest link the card itself supports. It is
                    # context, not a ceiling: a card in a narrower slot still
                    # reports its own maximum here.
                    "pcie_link_gen_max": self._safe(
                        getattr(pynvml, "nvmlDeviceGetMaxPcieLinkGeneration", None),
                        handle,
                    ),
                    "pcie_link_width_max": self._safe(
                        getattr(pynvml, "nvmlDeviceGetMaxPcieLinkWidth", None),
                        handle,
                    ),
                }
            )

    def _get_slow_metrics(self, index: int, handle: Any, now: float) -> dict:
        cached = self._slow_metrics.get(index)
        if cached is not None and now - cached["timestamp"] < self.slow_metric_ttl:
            return cached["values"]

        performance_state = self._safe(
            getattr(pynvml, "nvmlDeviceGetPerformanceState", None), handle
        )
        values = {
            "fan_percent": self._safe(
                getattr(pynvml, "nvmlDeviceGetFanSpeed", None), handle
            ),
            "temperature_c": self._safe(
                getattr(pynvml, "nvmlDeviceGetTemperature", None),
                handle,
                getattr(pynvml, "NVML_TEMPERATURE_GPU", 0),
            ),
            "power_limit_mw": self._safe(
                getattr(pynvml, "nvmlDeviceGetPowerManagementLimit", None), handle
            ),
            "performance_state": (
                int(performance_state) if isinstance(performance_state, int) else None
            ),
        }
        self._slow_metrics[index] = {"timestamp": now, "values": values}
        return values

    def _get_processes(self, handle: Any) -> list[dict]:
        processes = {}
        for process_type, function_name in (
            ("C", "nvmlDeviceGetComputeRunningProcesses"),
            ("G", "nvmlDeviceGetGraphicsRunningProcesses"),
        ):
            function = getattr(pynvml, function_name, None)
            for process in self._safe(function, handle, default=()) or ():
                try:
                    pid = int(process.pid)
                except (AttributeError, TypeError, ValueError):
                    continue

                used_memory = getattr(process, "usedGpuMemory", None)
                if not isinstance(used_memory, int) or used_memory >= _ULONG_LONG_MAX:
                    used_memory = None

                entry = processes.setdefault(
                    pid,
                    {
                        "pid": pid,
                        "type": "",
                        "process_name": None,
                        "username": _process_username(pid),
                        "used_gpu_memory_bytes": used_memory,
                        "gpu_instance_id": None,
                        "compute_instance_id": None,
                    },
                )
                if process_type not in entry["type"]:
                    entry["type"] += process_type
                if entry["used_gpu_memory_bytes"] is None and used_memory is not None:
                    entry["used_gpu_memory_bytes"] = used_memory

                gpu_instance_id = getattr(process, "gpuInstanceId", _UINT_MAX)
                compute_instance_id = getattr(process, "computeInstanceId", _UINT_MAX)
                if isinstance(gpu_instance_id, int) and gpu_instance_id != _UINT_MAX:
                    entry["gpu_instance_id"] = gpu_instance_id
                if isinstance(compute_instance_id, int) and compute_instance_id != _UINT_MAX:
                    entry["compute_instance_id"] = compute_instance_id

                process_name = self._safe(
                    getattr(pynvml, "nvmlSystemGetProcessName", None), pid
                )
                entry["process_name"] = _decode(process_name) or _process_name(pid)

        return list(processes.values())

    def collect(self) -> dict:
        with self._lock:
            try:
                self._ensure_initialized()
                count = int(pynvml.nvmlDeviceGetCount())
                if count != len(self._handles):
                    self._refresh_handles()

                now = time.monotonic()
                gpus = []
                for index, handle in enumerate(self._handles):
                    memory_function = getattr(
                        pynvml, "nvmlDeviceGetMemoryInfo", None
                    )
                    memory = self._safe(
                        memory_function,
                        handle,
                        getattr(pynvml, "nvmlMemory_v2", None),
                    )
                    if memory is None:
                        memory = self._safe(memory_function, handle)
                    utilization = self._safe(
                        getattr(pynvml, "nvmlDeviceGetUtilizationRates", None), handle
                    )
                    slow_metrics = self._get_slow_metrics(index, handle, now)

                    gpu = dict(self._device_static[index])
                    gpu.update(
                        {
                            "memory_total_bytes": getattr(memory, "total", None),
                            "memory_used_bytes": getattr(memory, "used", None),
                            "memory_free_bytes": getattr(memory, "free", None),
                            "memory_reserved_bytes": getattr(
                                memory, "reserved", None
                            ),
                            "gpu_util_percent": getattr(utilization, "gpu", None),
                            "memory_util_percent": getattr(utilization, "memory", None),
                            "pcie_tx_kib_per_s": self._safe(
                                getattr(pynvml, "nvmlDeviceGetPcieThroughput", None),
                                handle,
                                getattr(pynvml, "NVML_PCIE_UTIL_TX_BYTES", 0),
                            ),
                            "pcie_rx_kib_per_s": self._safe(
                                getattr(pynvml, "nvmlDeviceGetPcieThroughput", None),
                                handle,
                                getattr(pynvml, "NVML_PCIE_UTIL_RX_BYTES", 1),
                            ),
                            # Read every tick, not cached with the slow metrics:
                            # an idle link drops to gen1 and narrows its width,
                            # and stale link state would misreport the load.
                            "pcie_link_gen_current": self._safe(
                                getattr(
                                    pynvml, "nvmlDeviceGetCurrPcieLinkGeneration", None
                                ),
                                handle,
                            ),
                            "pcie_link_width_current": self._safe(
                                getattr(pynvml, "nvmlDeviceGetCurrPcieLinkWidth", None),
                                handle,
                            ),
                            "power_usage_mw": self._safe(
                                getattr(pynvml, "nvmlDeviceGetPowerUsage", None), handle
                            ),
                            "processes": self._get_processes(handle),
                            **slow_metrics,
                        }
                    )
                    gpus.append(gpu)

                return {
                    "ok": True,
                    "backend": "pynvml",
                    "driver_version": self._driver_version,
                    "cuda_version": self._cuda_version,
                    "gpus": gpus,
                }
            except Exception as error:
                return {
                    "ok": False,
                    "backend": "pynvml",
                    "error": f"{type(error).__name__}: {error}",
                }

    def close(self) -> None:
        with self._lock:
            if not self._initialized:
                return
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False
            self._handles = []
            self._device_static = []
            self._slow_metrics.clear()


REMOTE_NVML_AGENT_SCRIPT = textwrap.dedent(
    r"""
    import ctypes
    import ctypes.util
    import json
    import os
    import sys
    import time

    UINT_MAX = (1 << 32) - 1
    ULONG_LONG_MAX = (1 << 64) - 1
    ARCHITECTURES = {
        2: "Kepler",
        3: "Maxwell",
        4: "Pascal",
        5: "Volta",
        6: "Turing",
        7: "Ampere",
        8: "Ada",
        9: "Hopper",
        10: "Blackwell",
        11: "T23x",
    }

    class Memory(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    class MemoryV2(ctypes.Structure):
        _fields_ = [
            ("version", ctypes.c_uint),
            ("total", ctypes.c_ulonglong),
            ("reserved", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    class Utilization(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class ProcessInfoV1(ctypes.Structure):
        _fields_ = [
            ("pid", ctypes.c_uint),
            ("usedGpuMemory", ctypes.c_ulonglong),
        ]

    class ProcessInfoV2(ctypes.Structure):
        _fields_ = [
            ("pid", ctypes.c_uint),
            ("usedGpuMemory", ctypes.c_ulonglong),
            ("gpuInstanceId", ctypes.c_uint),
            ("computeInstanceId", ctypes.c_uint),
        ]

    def decode(value):
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def format_cuda_version(value):
        if value is None:
            return None
        version = int(value)
        major = version // 1000
        minor = (version % 1000) // 10
        revision = version % 10
        return f"{major}.{minor}.{revision}" if revision else f"{major}.{minor}"

    def process_username(pid):
        try:
            import pwd
            return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
        except (ImportError, KeyError, OSError):
            return None

    def process_name_from_proc(pid):
        try:
            with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as handle:
                return handle.read().strip() or None
        except OSError:
            return None

    class Nvml:
        def __init__(self, slow_metric_ttl=5.0):
            self.lib = self._load_library()
            self.slow_metric_ttl = float(slow_metric_ttl)
            self.slow_metrics = {}
            self.init_function = self._bind(
                ("nvmlInit_v2", "nvmlInit"),
                [],
            )
            self.shutdown_function = self._bind(("nvmlShutdown",), [])
            result = self.init_function()
            if result != 0:
                raise RuntimeError(self.error_string(result))

            self.driver_version = self._system_string("nvmlSystemGetDriverVersion", 128)
            self.cuda_version = self._get_cuda_version()
            self.devices = self._get_devices()

        @staticmethod
        def _load_library():
            candidates = [
                ctypes.util.find_library("nvidia-ml"),
                "libnvidia-ml.so.1",
                "/usr/lib/wsl/lib/libnvidia-ml.so.1",
            ]
            errors = []
            for candidate in candidates:
                if not candidate:
                    continue
                try:
                    return ctypes.CDLL(candidate)
                except OSError as error:
                    errors.append(str(error))
            raise OSError("; ".join(errors) or "libnvidia-ml.so.1 not found")

        def _bind(self, names, argtypes):
            for name in names:
                try:
                    function = getattr(self.lib, name)
                except AttributeError:
                    continue
                function.argtypes = argtypes
                function.restype = ctypes.c_int
                return function
            raise AttributeError(names[0])

        def _optional_bind(self, names, argtypes):
            try:
                return self._bind(names, argtypes)
            except AttributeError:
                return None

        def error_string(self, result):
            function = self._optional_bind(
                ("nvmlErrorString",),
                [ctypes.c_int],
            )
            if function is None:
                return f"NVML error {result}"
            function.restype = ctypes.c_char_p
            return decode(function(result)) or f"NVML error {result}"

        def _system_string(self, name, size):
            function = self._optional_bind(
                (name,),
                [ctypes.c_char_p, ctypes.c_uint],
            )
            if function is None:
                return None
            buffer = ctypes.create_string_buffer(size)
            result = function(buffer, size)
            return decode(buffer.value) if result == 0 else None

        def _get_cuda_version(self):
            function = self._optional_bind(
                ("nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion"),
                [ctypes.POINTER(ctypes.c_int)],
            )
            if function is None:
                return None
            value = ctypes.c_int()
            return format_cuda_version(value.value) if function(ctypes.byref(value)) == 0 else None

        def _get_devices(self):
            count_function = self._bind(
                ("nvmlDeviceGetCount_v2", "nvmlDeviceGetCount"),
                [ctypes.POINTER(ctypes.c_uint)],
            )
            count = ctypes.c_uint()
            result = count_function(ctypes.byref(count))
            if result != 0:
                raise RuntimeError(self.error_string(result))

            handle_function = self._bind(
                ("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex"),
                [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
            )
            devices = []
            for index in range(count.value):
                handle = ctypes.c_void_p()
                result = handle_function(index, ctypes.byref(handle))
                if result != 0:
                    continue
                devices.append(
                    {
                        "gpu_index": index,
                        "handle": handle,
                        "name": self._device_string(
                            "nvmlDeviceGetName",
                            handle,
                            256,
                        ),
                        "architecture": self._architecture(handle),
                        "pcie_link_gen_max": self._uint_query(
                            "nvmlDeviceGetMaxPcieLinkGeneration",
                            handle,
                        ),
                        "pcie_link_width_max": self._uint_query(
                            "nvmlDeviceGetMaxPcieLinkWidth",
                            handle,
                        ),
                    }
                )
            return devices

        def _device_string(self, name, handle, size):
            function = self._optional_bind(
                (name,),
                [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
            )
            if function is None:
                return None
            buffer = ctypes.create_string_buffer(size)
            result = function(handle, buffer, size)
            return decode(buffer.value) if result == 0 else None

        def _architecture(self, handle):
            value = self._uint_query("nvmlDeviceGetArchitecture", handle)
            return ARCHITECTURES.get(value)

        def _uint_query(self, name, handle, *arguments):
            argtypes = [ctypes.c_void_p]
            argtypes.extend(ctypes.c_uint for _ in arguments)
            argtypes.append(ctypes.POINTER(ctypes.c_uint))
            function = self._optional_bind((name,), argtypes)
            if function is None:
                return None
            value = ctypes.c_uint()
            result = function(handle, *arguments, ctypes.byref(value))
            return int(value.value) if result == 0 else None

        def _memory_info(self, handle):
            versioned_function = self._optional_bind(
                ("nvmlDeviceGetMemoryInfo_v2",),
                [ctypes.c_void_p, ctypes.POINTER(MemoryV2)],
            )
            if versioned_function is not None:
                versioned_value = MemoryV2()
                versioned_value.version = (2 << 24) | ctypes.sizeof(MemoryV2)
                if (
                    versioned_function(handle, ctypes.byref(versioned_value))
                    == 0
                ):
                    return versioned_value

            function = self._optional_bind(
                ("nvmlDeviceGetMemoryInfo",),
                [ctypes.c_void_p, ctypes.POINTER(Memory)],
            )
            if function is None:
                return None
            value = Memory()
            return value if function(handle, ctypes.byref(value)) == 0 else None

        def _utilization(self, handle):
            function = self._optional_bind(
                ("nvmlDeviceGetUtilizationRates",),
                [ctypes.c_void_p, ctypes.POINTER(Utilization)],
            )
            if function is None:
                return None
            value = Utilization()
            return value if function(handle, ctypes.byref(value)) == 0 else None

        def _process_name(self, pid):
            function = self._optional_bind(
                ("nvmlSystemGetProcessName",),
                [ctypes.c_uint, ctypes.c_char_p, ctypes.c_uint],
            )
            if function is not None:
                buffer = ctypes.create_string_buffer(1024)
                if function(pid, buffer, len(buffer)) == 0:
                    return decode(buffer.value)
            return process_name_from_proc(pid)

        def _running_processes(self, handle, prefix):
            function = self._optional_bind(
                (f"{prefix}_v2",),
                [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint),
                    ctypes.POINTER(ProcessInfoV2),
                ],
            )
            struct_type = ProcessInfoV2
            if function is None:
                function = self._optional_bind(
                    (prefix,),
                    [
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_uint),
                        ctypes.POINTER(ProcessInfoV1),
                    ],
                )
                struct_type = ProcessInfoV1
            if function is None:
                return []

            count = ctypes.c_uint()
            result = function(handle, ctypes.byref(count), None)
            if result == 0 or count.value == 0:
                return []

            capacity = min(int(count.value) + 16, 100000)
            for _ in range(2):
                array_type = struct_type * capacity
                values = array_type()
                actual = ctypes.c_uint(capacity)
                result = function(handle, ctypes.byref(actual), values)
                if result == 0:
                    return list(values[: actual.value])
                if actual.value <= capacity or actual.value > 100000:
                    break
                capacity = int(actual.value) + 16
            return []

        def _processes(self, handle):
            processes = {}
            for process_type, prefix in (
                ("C", "nvmlDeviceGetComputeRunningProcesses"),
                ("G", "nvmlDeviceGetGraphicsRunningProcesses"),
            ):
                for process in self._running_processes(handle, prefix):
                    pid = int(process.pid)
                    used_memory = int(process.usedGpuMemory)
                    if used_memory >= ULONG_LONG_MAX:
                        used_memory = None

                    entry = processes.setdefault(
                        pid,
                        {
                            "pid": pid,
                            "type": "",
                            "process_name": self._process_name(pid),
                            "username": process_username(pid),
                            "used_gpu_memory_bytes": used_memory,
                            "gpu_instance_id": None,
                            "compute_instance_id": None,
                        },
                    )
                    if process_type not in entry["type"]:
                        entry["type"] += process_type
                    if entry["used_gpu_memory_bytes"] is None and used_memory is not None:
                        entry["used_gpu_memory_bytes"] = used_memory

                    gpu_instance_id = getattr(process, "gpuInstanceId", UINT_MAX)
                    compute_instance_id = getattr(process, "computeInstanceId", UINT_MAX)
                    if gpu_instance_id != UINT_MAX:
                        entry["gpu_instance_id"] = int(gpu_instance_id)
                    if compute_instance_id != UINT_MAX:
                        entry["compute_instance_id"] = int(compute_instance_id)

            return list(processes.values())

        def _get_slow_metrics(self, index, handle, now):
            cached = self.slow_metrics.get(index)
            if cached is not None and now - cached["timestamp"] < self.slow_metric_ttl:
                return cached["values"]

            values = {
                "fan_percent": self._uint_query("nvmlDeviceGetFanSpeed", handle),
                "temperature_c": self._uint_query(
                    "nvmlDeviceGetTemperature",
                    handle,
                    0,
                ),
                "power_limit_mw": self._uint_query(
                    "nvmlDeviceGetPowerManagementLimit",
                    handle,
                ),
                "performance_state": self._uint_query(
                    "nvmlDeviceGetPerformanceState",
                    handle,
                ),
            }
            self.slow_metrics[index] = {"timestamp": now, "values": values}
            return values

        def collect(self):
            now = time.monotonic()
            gpus = []
            for device in self.devices:
                index = device["gpu_index"]
                handle = device["handle"]
                memory = self._memory_info(handle)
                utilization = self._utilization(handle)
                gpu = {
                    "gpu_index": index,
                    "name": device["name"],
                    "architecture": device["architecture"],
                    "memory_total_bytes": getattr(memory, "total", None),
                    "memory_used_bytes": getattr(memory, "used", None),
                    "memory_free_bytes": getattr(memory, "free", None),
                    "memory_reserved_bytes": getattr(memory, "reserved", None),
                    "gpu_util_percent": getattr(utilization, "gpu", None),
                    "memory_util_percent": getattr(utilization, "memory", None),
                    "pcie_tx_kib_per_s": self._uint_query(
                        "nvmlDeviceGetPcieThroughput",
                        handle,
                        0,
                    ),
                    "pcie_rx_kib_per_s": self._uint_query(
                        "nvmlDeviceGetPcieThroughput",
                        handle,
                        1,
                    ),
                    "pcie_link_gen_max": device["pcie_link_gen_max"],
                    "pcie_link_width_max": device["pcie_link_width_max"],
                    # Read every tick: an idle link drops to gen1 and narrows
                    # its width, so cached link state would misreport the load.
                    "pcie_link_gen_current": self._uint_query(
                        "nvmlDeviceGetCurrPcieLinkGeneration",
                        handle,
                    ),
                    "pcie_link_width_current": self._uint_query(
                        "nvmlDeviceGetCurrPcieLinkWidth",
                        handle,
                    ),
                    "power_usage_mw": self._uint_query(
                        "nvmlDeviceGetPowerUsage",
                        handle,
                    ),
                    "processes": self._processes(handle),
                }
                gpu.update(self._get_slow_metrics(index, handle, now))
                gpus.append(gpu)

            return {
                "ok": True,
                "backend": "ctypes",
                "driver_version": self.driver_version,
                "cuda_version": self.cuda_version,
                "gpus": gpus,
            }

        def close(self):
            try:
                self.shutdown_function()
            except Exception:
                pass

    backend = None
    init_error = None
    try:
        backend = Nvml()
    except Exception as error:
        init_error = f"{type(error).__name__}: {error}"

    def emit_snapshot():
        if backend is None:
            payload = {
                "ok": False,
                "backend": "ctypes",
                "error": init_error or "NVML initialization failed",
            }
        else:
            try:
                payload = backend.collect()
            except Exception as error:
                payload = {
                    "ok": False,
                    "backend": "ctypes",
                    "error": f"{type(error).__name__}: {error}",
                }
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)

    try:
        if len(sys.argv) > 1 and sys.argv[1] == "once":
            emit_snapshot()
        else:
            for command in sys.stdin:
                command = command.strip().lower()
                if command == "snapshot":
                    emit_snapshot()
                elif command in {"quit", "exit"}:
                    break
    finally:
        if backend is not None:
            backend.close()
    """
).strip()


def make_nvml_agent_command(*, python_exe: str = "python3", once: bool = False) -> str:
    encoded = base64.b64encode(REMOTE_NVML_AGENT_SCRIPT.encode("utf-8")).decode("ascii")
    argument = " once" if once else ""
    return (
        f"{python_exe} -S -u -c "
        f"\"import base64; exec(compile(base64.b64decode('{encoded}').decode('utf-8'), "
        f"'<nvidb-nvml-agent>', 'exec'))\"{argument}"
    )
