from __future__ import annotations

import textwrap

from .nvml import _python_exec_command

DCGM_SNAPSHOT_SCRIPT = textwrap.dedent(
    r"""
    import json
    import os
    import sys
    import time
    import warnings

    # DCGM 4.6+ bindings emit DeprecationWarnings for legacy field names;
    # keep stderr clean for callers that log it.
    warnings.filterwarnings("ignore")

    BINDINGS = [
        "/usr/share/datacenter-gpu-manager-4/bindings/python3",
        "/usr/share/datacenter-gpu-manager/bindings/python3",
        "/usr/share/datacenter-gpu-manager-3/bindings/python3",
        "/usr/share/datacenter-gpu-manager-2/bindings/python3",
    ]
    for p in BINDINGS:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        import pydcgm
        import dcgm_fields
    except Exception as e:
        print(json.dumps({"ok": False, "error": "dcgm_import_failed", "detail": str(e)}))
        raise SystemExit(0)

    def _try_connect():
        errors = []
        # Prefer TCP hostengine (matches `dcgmi` default).
        for host in [
            "127.0.0.1:5555",
            "localhost:5555",
            # Fallbacks (some builds may default the port internally)
            "127.0.0.1",
            "localhost",
        ]:
            try:
                handle = pydcgm.DcgmHandle(ipAddress=host)
                return handle, f"tcp://{host}", errors
            except Exception as e:
                errors.append(f"tcp://{host}:{type(e).__name__}")

        # Try unix domain socket if hostengine was started with -d.
        for sock in [
            "/var/run/dcgm/dcgm.sock",
            "/run/dcgm/dcgm.sock",
            "/var/run/nvidia-dcgm/dcgm.sock",
            "/run/nvidia-dcgm/dcgm.sock",
        ]:
            if not os.path.exists(sock):
                continue
            try:
                handle = pydcgm.DcgmHandle(unixSocketPath=sock)
                return handle, f"unix://{sock}", errors
            except Exception as e:
                errors.append(f"unix://{sock}:{type(e).__name__}")

        # As a last resort, try embedded mode.
        try:
            handle = pydcgm.DcgmHandle()
            return handle, "embedded", errors
        except Exception as e:
            errors.append(f"embedded:{type(e).__name__}")
            return None, None, errors

    handle, connection, connect_errors = _try_connect()
    if handle is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "dcgm_connect_failed",
                    "detail": connect_errors[-6:],
                }
            )
        )
        raise SystemExit(0)

    system = handle.GetSystem()

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
        print(json.dumps({"ok": True, "connection": connection, "gpus": []}))
        raise SystemExit(0)

    # Only the profiling fields: NVML already supplies the basic metrics.
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

    field_ids = sorted(set(prof_fields))
    group_name = f"nvidb_{os.getpid()}_{int(time.time() * 1000)}"
    field_group_name = group_name + "_fields"

    try:
        group = system.GetGroupWithGpuIds(group_name, gpu_ids)
        field_group = pydcgm.DcgmFieldGroup(handle, field_group_name, field_ids)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "dcgm_group_failed", "detail": str(e)}))
        raise SystemExit(0)

    try:
        group.samples.WatchFields(field_group, 1_000_000, 30.0, 1)
    except Exception:
        # Some fields (especially profiling) may fail to watch on unsupported GPUs.
        pass

    try:
        system.UpdateAllFields(True)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "dcgm_update_failed", "detail": str(e)}))
        raise SystemExit(0)

    try:
        fvc = group.samples.GetLatest(field_group)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "dcgm_get_latest_failed", "detail": str(e)}))
        raise SystemExit(0)

    def to_jsonable(value):
        if value is None:
            return None
        # ctypes scalar wrappers
        if hasattr(value, "value") and not isinstance(
            value, (str, bytes, bytearray, int, float)
        ):
            try:
                value = value.value
            except Exception:
                pass
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (int, float, str)):
            return value
        try:
            return float(value)
        except Exception:
            return str(value)

    def get_value(gid: int, fid: int):
        try:
            vals = fvc.values.get(gid, {}).get(fid)
            if not vals:
                return None
            v = vals[-1]
            if getattr(v, "isBlank", False):
                return None
            return to_jsonable(v.value)
        except Exception:
            return None

    gpus = []
    for gid in gpu_ids:
        gpus.append(
            {
                "gpu_index": int(gid),
                # Profiling fields (ratios; typically 0-1 or 0-100 depending on version/GPU).
                "sm_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_SM_ACTIVE),
                "sm_occupancy": get_value(gid, dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY),
                "tensor_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE),
                "fp64_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE),
                "fp32_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE),
                "fp16_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE),
                "int_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_PIPE_INT_ACTIVE),
                "dram_active": get_value(gid, dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE),
            }
        )

    payload = {
        "ok": True,
        "connection": connection,
        "gpus": gpus,
    }
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    """
).strip()


def make_dcgm_snapshot_command() -> str:
    """Return a shell command that prints a DCGM snapshot as JSON on stdout.

    The command is intended to be executed *on the target node* (local or via SSH).
    """
    return _python_exec_command(DCGM_SNAPSHOT_SCRIPT, "<nvidb-dcgm>")
