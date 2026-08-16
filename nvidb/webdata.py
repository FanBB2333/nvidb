"""
Data layer for nvidb's web dashboard (Live GPU + Log viewer).

The Dash-based UI lives in `nvidb.dashboard`; this module owns the
UI-agnostic parsing, formatting, and SQLite log-reading helpers it imports.
"""

from datetime import datetime
import math
import re
import sqlite3
from pathlib import Path

import pandas as pd

from . import config


def get_db_path():
    return config.get_db_path()


def _as_path(value):
    if value is None:
        return None
    return Path(value).expanduser()


def load_sessions(db_path):
    db_path = _as_path(db_path)
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        """
        SELECT
            s.id,
            s.start_time,
            s.end_time,
            s.status,
            s.interval_seconds,
            s.include_remote,
            COUNT(g.id) as record_count,
            COUNT(DISTINCT g.timestamp) as snapshot_count
        FROM log_sessions s
        LEFT JOIN gpu_logs g ON s.id = g.session_id
        GROUP BY s.id
        ORDER BY s.id ASC
        """,
        conn,
    )
    conn.close()
    return df


def load_session_logs(db_path, session_id):
    db_path = _as_path(db_path)
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        """
        SELECT
            timestamp,
            node,
            gpu_id,
            name,
            fan_speed,
            util_gpu,
            util_mem,
            temperature,
            rx,
            tx,
            power,
            memory_used,
            memory_total,
            processes
        FROM gpu_logs
        WHERE session_id = ?
        ORDER BY timestamp, node, gpu_id
        """,
        conn,
        params=(session_id,),
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _parse_percent(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("N/A", "-"):
        return None
    match = re.search(r"(\d+\.?\d*)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _parse_mib_pair(value):
    if value is None:
        return None, None
    text = str(value).strip()
    if not text or text in ("N/A", "-"):
        return None, None
    parts = text.split("/", 1)
    if len(parts) != 2:
        return None, None
    try:
        used = float(parts[0])
        total = float(parts[1])
        return used, total
    except Exception:
        return None, None


def _parse_temperature_c(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("N/A", "-"):
        return None
    match = re.search(r"(\d+\.?\d*)\s*C", text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    match = re.search(r"(\d+\.?\d*)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


_MEM_UNIT_TO_MIB = {
    "b": 1 / 1024 / 1024,
    "kb": 1 / 1024,
    "kib": 1 / 1024,
    "mb": 1,
    "mib": 1,
    "gb": 1024,
    "gib": 1024,
    "tb": 1024 * 1024,
    "tib": 1024 * 1024,
}


def _parse_memory_gib(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("N/A", "-"):
        return None

    match = re.search(r"(\d+\.?\d*)\s*([a-zA-Z]+)", text)
    if match:
        number_str, unit = match.group(1), match.group(2)
        unit_key = unit.strip().lower()
    else:
        number_match = re.search(r"(\d+\.?\d*)", text)
        if not number_match:
            return None
        number_str = number_match.group(1)
        unit_key = "mib"

    try:
        number = float(number_str)
    except Exception:
        return None

    mib_multiplier = _MEM_UNIT_TO_MIB.get(unit_key)
    if mib_multiplier is None:
        mib_multiplier = _MEM_UNIT_TO_MIB.get(unit_key.replace("bytes", "b"), None)
    if mib_multiplier is None:
        mib_multiplier = 1
    mib = number * mib_multiplier
    return mib / 1024.0


_BW_UNIT_TO_MBPS = {
    "b/s": 1 / 1024 / 1024,
    "kb/s": 1 / 1024,
    "kib/s": 1 / 1024,
    "mb/s": 1,
    "mib/s": 1,
    "gb/s": 1024,
    "gib/s": 1024,
}


def _parse_bandwidth_mbps(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("N/A", "-"):
        return None
    match = re.search(r"(\d+\.?\d*)\s*([a-zA-Z/]+)", text)
    if match:
        number_str, unit = match.group(1), match.group(2)
        unit_key = unit.strip().lower()
    else:
        number_match = re.search(r"(\d+\.?\d*)", text)
        if not number_match:
            return None
        number_str = number_match.group(1)
        unit_key = "mb/s"

    try:
        number = float(number_str)
    except Exception:
        return None

    multiplier = _BW_UNIT_TO_MBPS.get(unit_key)
    if multiplier is None:
        multiplier = 1
    return number * multiplier


def _parse_power_watts(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("N/A", "-"):
        return None
    match = re.search(r"(\d+\.?\d*)\s*W", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _as_datetime(value):
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    try:
        return ts.to_pydatetime()
    except Exception:
        return ts


def _format_datetime(value, *, include_seconds: bool = True):
    dt = _as_datetime(value)
    if dt is None:
        return "N/A"
    fmt = "%Y-%m-%d %H:%M:%S" if include_seconds else "%Y-%m-%d %H:%M"
    try:
        return dt.strftime(fmt)
    except Exception:
        text = str(value)
        text = text.replace("T", " ")
        return text[:19] if include_seconds else text[:16]


def _format_duration(start_value, end_value):
    start_dt = _as_datetime(start_value)
    end_dt = _as_datetime(end_value)
    if start_dt is None:
        return "N/A"
    if end_dt is None:
        end_dt = datetime.now()
    try:
        total_seconds = int((end_dt - start_dt).total_seconds())
    except Exception:
        return "N/A"
    total_seconds = max(0, total_seconds)

    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    if days > 0:
        return f"{days}d{hours}h"
    if hours > 0:
        return f"{hours}h{minutes}m"
    if minutes > 0:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


def _format_gb(mib):
    if mib is None:
        return "N/A"
    gb = mib / 1024.0
    if gb >= 10:
        return f"{gb:.0f}GB"
    return f"{gb:.1f}GB"


def _server_summary(table, system_info=None):
    source = None
    if system_info and isinstance(system_info, dict):
        source = system_info.get("data_source")

    if table is None or table.empty:
        if source == "unsupported":
            return "Unsupported"
        if source:
            return f"No GPUs | src:{source}"
        return "No GPUs"

    utils = []
    used_total_pairs = []
    for _, row in table.iterrows():
        utils.append(_parse_percent(row.get("util")))
        used_total_pairs.append(_parse_mib_pair(row.get("memory[used/total]")))

    util_values = [u for u in utils if u is not None]
    avg_util = (sum(util_values) / len(util_values)) if util_values else 0.0

    idle = 0
    for u in util_values:
        if u < 5:
            idle += 1

    used_sum = 0.0
    total_sum = 0.0
    for used, total in used_total_pairs:
        if used is None or total is None:
            continue
        used_sum += used
        total_sum += total

    mem_str = f"{_format_gb(used_sum)}/{_format_gb(total_sum)}" if total_sum else "N/A"

    # Build GPU summary part
    gpu_summary = f"{len(table)} GPUs | {idle} idle | {avg_util:.0f}% avg | {mem_str}"

    # Extract system stats if available
    sys_stats = {}
    if system_info and isinstance(system_info, dict):
        sys_stats = system_info.get("system_stats", {})

    cpu_cores = sys_stats.get("cpu_cores", 0)
    cpu_percent = sys_stats.get("cpu_percent", 0.0)
    mem_used_gb = sys_stats.get("mem_used_gb", 0.0)
    mem_total_gb = sys_stats.get("mem_total_gb", 0.0)
    swap_used_gb = sys_stats.get("swap_used_gb", 0.0)
    swap_total_gb = sys_stats.get("swap_total_gb", 0.0)

    sys_parts = []

    # CPU: utilization first, cores in brackets
    if cpu_cores > 0:
        cpu_percent_int = int(round(cpu_percent))
        sys_parts.append(f"CPU: {cpu_percent_int:>3}%({cpu_cores}C)")

    # Memory: used/total with swap in brackets
    if mem_total_gb > 0:
        if swap_total_gb > 0:
            sys_parts.append(f"Mem: {mem_used_gb:.0f}/{mem_total_gb:.0f}G(Swap:{swap_used_gb:.1f}/{swap_total_gb:.0f}G)")
        else:
            sys_parts.append(f"Mem: {mem_used_gb:.0f}/{mem_total_gb:.0f}G")

    if sys_parts:
        summary = f"{gpu_summary} | {' | '.join(sys_parts)}"
    else:
        summary = gpu_summary

    if source:
        summary = f"{summary} | src:{source}"
    return summary


def _strip_gpu_name(value):
    text = "" if value is None else str(value)
    return text.replace("NVIDIA", "").replace("GeForce", "").strip()


def _format_mib(mib: int) -> str:
    try:
        mib_int = int(mib)
    except Exception:
        return "N/A"
    if mib_int >= 1024:
        return f"{mib_int / 1024:.1f} GB"
    return f"{mib_int:d} MiB"


def _user_summary_df(user_summary: dict) -> pd.DataFrame:
    rows = []
    for user, mib in (user_summary or {}).items():
        try:
            mib_int = int(mib)
        except Exception:
            continue
        if mib_int <= 0:
            continue
        rows.append({"user": str(user), "vram_mib": mib_int, "vram": _format_mib(mib_int)})
    if not rows:
        return pd.DataFrame(columns=["user", "vram", "vram_mib"])
    df = pd.DataFrame(rows).sort_values(by="vram_mib", ascending=False, kind="stable").reset_index(drop=True)
    return df


_USER_MEM_RE = re.compile(r"([^\s()]+)\((\d+)\s*M\)", flags=re.IGNORECASE)


def _parse_user_memory_compact(value) -> dict:
    if value is None:
        return {}
    text = str(value).strip()
    if not text or text in ("-", "N/A"):
        return {}
    users = {}
    for username, mib_str in _USER_MEM_RE.findall(text):
        username = str(username).strip()
        if not username or username == "N/A":
            continue
        try:
            mib = int(mib_str)
        except Exception:
            continue
        if mib <= 0:
            continue
        users[username] = users.get(username, 0) + mib
    return users


def _user_memory_from_df(df: pd.DataFrame) -> dict:
    totals = {}
    if df is None or df.empty:
        return totals
    if "processes" not in df.columns:
        return totals
    for value in df["processes"]:
        for user, mib in _parse_user_memory_compact(value).items():
            totals[user] = totals.get(user, 0) + mib
    return totals


def _user_time_share_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["user", "snapshots", "share"])
    if "timestamp" not in df.columns or "processes" not in df.columns:
        return pd.DataFrame(columns=["user", "snapshots", "share"])

    timestamps = df["timestamp"].dropna().drop_duplicates().sort_values()
    total = int(len(timestamps))
    if total <= 0:
        return pd.DataFrame(columns=["user", "snapshots", "share"])

    counts = {}
    for ts, group in df.groupby("timestamp", sort=False):
        if pd.isna(ts):
            continue
        users = set()
        for value in group["processes"]:
            users.update(_parse_user_memory_compact(value).keys())
        for user in users:
            counts[user] = counts.get(user, 0) + 1

    rows = []
    for user, snaps in counts.items():
        try:
            snaps_int = int(snaps)
        except Exception:
            continue
        share = (snaps_int / total) * 100.0
        rows.append({"user": user, "snapshots": snaps_int, "share": share})
    if not rows:
        return pd.DataFrame(columns=["user", "snapshots", "share"])
    return pd.DataFrame(rows).sort_values(by="share", ascending=False, kind="stable").reset_index(drop=True)


def _load_server_list():
    from .data_modules import ServerListInfo

    cfg_path = Path(config.get_config_path()).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return ServerListInfo.from_yaml(cfg_path)


def _build_log_snapshot_table(df):
    from .utils import extract_numbers

    if df.empty:
        return pd.DataFrame()

    used_total = []
    for _, row in df.iterrows():
        used = row.get("memory_used", "N/A")
        total = row.get("memory_total", "N/A")
        used_str = "/".join(extract_numbers(str(used))) or "N/A"
        total_str = "/".join(extract_numbers(str(total))) or "N/A"
        used_total.append(f"{used_str}/{total_str}")

    table = pd.DataFrame(
        {
            "GPU": df["gpu_id"],
            "name": df["name"].map(_strip_gpu_name),
            "fan": df.get("fan_speed", "-"),
            "util": df.get("util_gpu", "-"),
            "mem_util": df.get("util_mem", "-"),
            "temp": df.get("temperature", "-"),
            "rx": df.get("rx", "-"),
            "tx": df.get("tx", "-"),
            "power": df.get("power", "-"),
            "memory[used/total]": used_total,
            "processes": df.get("processes", "-"),
        }
    )
    return table.sort_values(by="GPU", kind="stable")


_LOG_METRICS = {
    "util_gpu": {
        "label": "GPU Util (%)",
        "source": "util_gpu",
        "parser": _parse_percent,
        "tooltip_format": ".1f",
        "default": True,
        "height": 220,
    },
    "memory_used": {
        "label": "VRAM Used (GiB)",
        "source": "memory_used",
        "parser": _parse_memory_gib,
        "tooltip_format": ".2f",
        "default": True,
        "height": 220,
    },
    "temperature": {
        "label": "Temperature (°C)",
        "source": "temperature",
        "parser": _parse_temperature_c,
        "tooltip_format": ".1f",
        "default": True,
        "height": 220,
    },
    "power": {
        "label": "Power Draw (W)",
        "source": "power",
        "parser": _parse_power_watts,
        "tooltip_format": ".1f",
        "default": True,
        "height": 220,
    },
    "util_mem": {
        "label": "Memory Util (%)",
        "source": "util_mem",
        "parser": _parse_percent,
        "tooltip_format": ".1f",
        "default": False,
        "height": 220,
    },
    "rx": {
        "label": "PCIe RX (MB/s)",
        "source": "rx",
        "parser": _parse_bandwidth_mbps,
        "tooltip_format": ".2f",
        "default": True,
        "height": 220,
    },
    "tx": {
        "label": "PCIe TX (MB/s)",
        "source": "tx",
        "parser": _parse_bandwidth_mbps,
        "tooltip_format": ".2f",
        "default": True,
        "height": 220,
    },
    "fan_speed": {
        "label": "Fan Speed (%)",
        "source": "fan_speed",
        "parser": _parse_percent,
        "tooltip_format": ".0f",
        "default": False,
        "height": 220,
    },
}


def _downsample_per_gpu(df: pd.DataFrame, *, max_points_per_gpu: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        max_points = int(max_points_per_gpu)
    except Exception:
        max_points = 600
    if max_points <= 0:
        return df

    frames = []
    for _gpu_id, group in df.groupby("gpu_id", sort=False):
        group_sorted = group.sort_values(by="timestamp", kind="stable")
        n = len(group_sorted)
        if n <= max_points:
            frames.append(group_sorted)
            continue
        step = int(math.ceil(n / max_points))
        frames.append(group_sorted.iloc[::step])
    if not frames:
        return df.iloc[:0]
    return pd.concat(frames, ignore_index=True)
