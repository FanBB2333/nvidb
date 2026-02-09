from __future__ import annotations

import base64
import textwrap

DCGM_PYTHON_BINDINGS_PATHS = (
    "/usr/share/datacenter-gpu-manager-4/bindings/python3",
    "/usr/share/datacenter-gpu-manager/bindings/python3",
    "/usr/share/datacenter-gpu-manager-3/bindings/python3",
    "/usr/share/datacenter-gpu-manager-2/bindings/python3",
)

DCGM_TCP_HOST_CANDIDATES = (
    "127.0.0.1:5555",
    "localhost:5555",
    # Fallbacks (some builds may default the port internally)
    "127.0.0.1",
    "localhost",
)

DCGM_UNIX_SOCKET_CANDIDATES = (
    "/var/run/dcgm/dcgm.sock",
    "/run/dcgm/dcgm.sock",
    "/var/run/nvidia-dcgm/dcgm.sock",
    "/run/nvidia-dcgm/dcgm.sock",
)


def make_dcgm_snapshot_command(*, python_exe: str = "python3") -> str:
    """Return a shell command that prints a DCGM snapshot as JSON on stdout.

    The command is intended to be executed *on the target node* (local or via SSH).
    """

    script = textwrap.dedent(
        f"""
        import json
        import os
        import sys
        import time

        BINDINGS = {list(DCGM_PYTHON_BINDINGS_PATHS)!r}
        for p in BINDINGS:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        try:
            import pydcgm
            import dcgm_fields
        except Exception as e:
            print(json.dumps({{"ok": False, "error": "dcgm_import_failed", "detail": str(e)}}))
            raise SystemExit(0)

        def _try_connect():
            errors = []
            # Prefer TCP hostengine (matches `dcgmi` default).
            for host in {list(DCGM_TCP_HOST_CANDIDATES)!r}:
                try:
                    handle = pydcgm.DcgmHandle(ipAddress=host)
                    return handle, f"tcp://{{host}}", errors
                except Exception as e:
                    errors.append(f"tcp://{{host}}:{{type(e).__name__}}")

            # Try unix domain socket if hostengine was started with -d.
            for sock in {list(DCGM_UNIX_SOCKET_CANDIDATES)!r}:
                if not os.path.exists(sock):
                    continue
                try:
                    handle = pydcgm.DcgmHandle(unixSocketPath=sock)
                    return handle, f"unix://{{sock}}", errors
                except Exception as e:
                    errors.append(f"unix://{{sock}}:{{type(e).__name__}}")

            # As a last resort, try embedded mode.
            try:
                handle = pydcgm.DcgmHandle()
                return handle, "embedded", errors
            except Exception as e:
                errors.append(f"embedded:{{type(e).__name__}}")
                return None, None, errors

        handle, connection, connect_errors = _try_connect()
        if handle is None:
            print(
                json.dumps(
                    {{
                        "ok": False,
                        "error": "dcgm_connect_failed",
                        "detail": connect_errors[-6:],
                    }}
                )
            )
            raise SystemExit(0)

        system = handle.GetSystem()

        gpu_ids = []
        try:
            gpu_ids = system.discovery.GetAllSupportedGpuIds() or []
        except Exception:
            gpu_ids = []
        if not gpu_ids:
            try:
                gpu_ids = system.discovery.GetAllGpuIds() or []
            except Exception:
                gpu_ids = []

        if not gpu_ids:
            print(json.dumps({{"ok": True, "connection": connection, "gpus": []}}))
            raise SystemExit(0)

        # Base + profiling fields (keep this minimal; consumers can format/derive as needed).
        base_fields = [
            dcgm_fields.DCGM_FI_DRIVER_VERSION,
            dcgm_fields.DCGM_FI_CUDA_DRIVER_VERSION,
            dcgm_fields.DCGM_FI_DEV_NAME,
            dcgm_fields.DCGM_FI_DEV_PCI_BUSID,
            dcgm_fields.DCGM_FI_DEV_FAN_SPEED,
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
            dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,
            dcgm_fields.DCGM_FI_DEV_PSTATE,
            dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
            dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT,
            dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
            dcgm_fields.DCGM_FI_DEV_FB_USED,
            dcgm_fields.DCGM_FI_DEV_FB_FREE,
            dcgm_fields.DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
            dcgm_fields.DCGM_FI_DEV_PCIE_RX_THROUGHPUT,
        ]

        prof_fields = [
            dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
            dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PIPE_INT_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
        ]

        field_ids = sorted(set(base_fields + prof_fields))
        group_name = f"nvidb_{{os.getpid()}}_{{int(time.time() * 1000)}}"
        field_group_name = group_name + "_fields"

        try:
            group = system.GetGroupWithGpuIds(group_name, gpu_ids)
            field_group = pydcgm.DcgmFieldGroup(handle, field_group_name, field_ids)
        except Exception as e:
            print(json.dumps({{"ok": False, "error": "dcgm_group_failed", "detail": str(e)}}))
            raise SystemExit(0)

        try:
            group.samples.WatchFields(field_group, 1_000_000, 30.0, 1)
            system.UpdateAllFields(True)
        except Exception:
            # Some fields (especially profiling) may fail to watch on unsupported GPUs.
            pass

        try:
            fvc = group.samples.GetLatest(field_group)
        except Exception as e:
            print(json.dumps({{"ok": False, "error": "dcgm_get_latest_failed", "detail": str(e)}}))
            raise SystemExit(0)

        def to_jsonable(value):
            if value is None:
                return None
            # ctypes scalar wrappers
            if hasattr(value, "value") and not isinstance(
                value, (str, bytes, bytearray, int, float, bool)
            ):
                try:
                    value = value.value
                except Exception:
                    pass
            if isinstance(value, (bytes, bytearray)):
                try:
                    return value.decode("utf-8", errors="replace")
                except Exception:
                    return str(value)
            if isinstance(value, (int, float, str, bool)):
                return value
            try:
                return float(value)
            except Exception:
                return str(value)

        def get_value(gid: int, fid: int):
            try:
                vals = fvc.values.get(gid, {{}}).get(fid)
                if not vals:
                    return None
                v = vals[-1]
                if getattr(v, "isBlank", False):
                    return None
                return to_jsonable(v.value)
            except Exception:
                return None

        driver_version = None
        cuda_driver_version = None
        for gid in gpu_ids:
            if driver_version is None:
                driver_version = get_value(gid, dcgm_fields.DCGM_FI_DRIVER_VERSION)
            if cuda_driver_version is None:
                cuda_driver_version = get_value(gid, dcgm_fields.DCGM_FI_CUDA_DRIVER_VERSION)

        gpus = []
        for gid in gpu_ids:
            gpus.append(
                {{
                    "gpu_index": int(gid),
                    "name": get_value(gid, dcgm_fields.DCGM_FI_DEV_NAME),
                    "pci_bus_id": get_value(gid, dcgm_fields.DCGM_FI_DEV_PCI_BUSID),
                    "fan_speed": get_value(gid, dcgm_fields.DCGM_FI_DEV_FAN_SPEED),
                    "gpu_temp": get_value(gid, dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
                    "gpu_util": get_value(gid, dcgm_fields.DCGM_FI_DEV_GPU_UTIL),
                    "mem_util": get_value(gid, dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL),
                    "pstate": get_value(gid, dcgm_fields.DCGM_FI_DEV_PSTATE),
                    "power_w": get_value(gid, dcgm_fields.DCGM_FI_DEV_POWER_USAGE),
                    "power_limit_w": get_value(gid, dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT),
                    "fb_total_mb": get_value(gid, dcgm_fields.DCGM_FI_DEV_FB_TOTAL),
                    "fb_used_mb": get_value(gid, dcgm_fields.DCGM_FI_DEV_FB_USED),
                    "fb_free_mb": get_value(gid, dcgm_fields.DCGM_FI_DEV_FB_FREE),
                    "pcie_tx_kbps": get_value(gid, dcgm_fields.DCGM_FI_DEV_PCIE_TX_THROUGHPUT),
                    "pcie_rx_kbps": get_value(gid, dcgm_fields.DCGM_FI_DEV_PCIE_RX_THROUGHPUT),
                    # Profiling fields (ratios; typically 0-1 or 0-100 depending on version/GPU).
                    "sm_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_SM_ACTIVE),
                    "sm_occupancy": get_value(gid, dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY),
                    "tensor_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE),
                    "fp64_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE),
                    "fp32_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE),
                    "fp16_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE),
                    "int_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_PIPE_INT_ACTIVE),
                    "dram_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE),
                }}
            )

        payload = {{
            "ok": True,
            "connection": connection,
            "driver_version": driver_version,
            "cuda_driver_version": cuda_driver_version,
            "gpus": gpus,
        }}
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        """
    ).strip()

    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    # Keep quoting simple & portable for remote shells.
    return (
        f"{python_exe} -c "
        f"\"import base64; exec(compile(base64.b64decode('{encoded}').decode('utf-8'), '<nvidb-dcgm>', 'exec'))\""
    )
