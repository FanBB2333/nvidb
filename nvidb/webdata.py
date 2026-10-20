"""
Data layer for nvidb's web dashboard (Live GPU + Log viewer).

The Dash-based UI lives in `nvidb.dashboard`; this module owns the
UI-agnostic parsing, formatting, and SQLite log-reading helpers it imports.
"""

import contextlib
from datetime import datetime
import math
import re
import sqlite3
from pathlib import Path

import pandas as pd

from . import config


def _open_log_db(db_path):
    # sqlite3.connect would create a missing file, leaving an empty DB that
    # later reads as "no such table" instead of "no log database".
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Log database not found: {db_path}")
    return contextlib.closing(sqlite3.connect(db_path))


def load_sessions(db_path):
    with _open_log_db(db_path) as conn:
        return pd.read_sql_query(
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


def load_session_logs(db_path, session_id):
    with _open_log_db(db_path) as conn:
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# Every parser below needs a digit, so None ("None"), "", "N/A", "-" and NaN
# ("nan") all fall through to None without a separate check.
_NUMBER_RE = re.compile(r"(\d+\.?\d*)")


def _parse_percent(value):
    match = _NUMBER_RE.search(str(value))
    return float(match.group(1)) if match else None


def _parse_mib_pair(value):
    parts = str(value).split("/", 1)
    if len(parts) != 2:
        return None, None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None, None


def _parse_temperature_c(value):
    text = str(value)
    match = re.search(r"(\d+\.?\d*)\s*C", text, flags=re.IGNORECASE) or _NUMBER_RE.search(text)
    return float(match.group(1)) if match else None


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
    text = str(value)
    match = re.search(r"(\d+\.?\d*)\s*([a-zA-Z]+)", text)
    if match:
        number_str, unit_key = match.group(1), match.group(2).lower()
    else:
        number_match = _NUMBER_RE.search(text)
        if not number_match:
            return None
        number_str, unit_key = number_match.group(1), "mib"

    mib_multiplier = _MEM_UNIT_TO_MIB.get(unit_key) or _MEM_UNIT_TO_MIB.get(unit_key.replace("bytes", "b"), 1)
    mib = float(number_str) * mib_multiplier
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
    text = str(value)
    match = re.search(r"(\d+\.?\d*)\s*([a-zA-Z/]+)", text)
    if match:
        number_str, unit_key = match.group(1), match.group(2).lower()
    else:
        number_match = _NUMBER_RE.search(text)
        if not number_match:
            return None
        number_str, unit_key = number_match.group(1), "mb/s"
    return float(number_str) * _BW_UNIT_TO_MBPS.get(unit_key, 1)


def _parse_power_watts(value):
    match = re.search(r"(\d+\.?\d*)\s*W", str(value), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _as_datetime(value):
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _format_datetime(value, *, include_seconds: bool = True):
    dt = _as_datetime(value)
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M:%S" if include_seconds else "%Y-%m-%d %H:%M")


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


def _gpu_table_totals(table):
    """(known util values, used MiB sum, total MiB sum) over a GPU table.

    Rows whose memory pair does not parse are left out of both sums.
    """
    util_values = []
    used_sum = 0.0
    total_sum = 0.0
    for _, row in table.iterrows():
        util = _parse_percent(row.get("util"))
        if util is not None:
            util_values.append(util)
        used, total = _parse_mib_pair(row.get("memory[used/total]"))
        if used is not None and total is not None:
            used_sum += used
            total_sum += total
    return util_values, used_sum, total_sum


def _server_summary(table, system_info=None):
    info = system_info or {}
    source = info.get("data_source")

    if table.empty:
        if source == "unsupported":
            return "Unsupported"
        if source:
            return f"No GPUs | src:{source}"
        return "No GPUs"

    util_values, used_sum, total_sum = _gpu_table_totals(table)
    avg_util = (sum(util_values) / len(util_values)) if util_values else 0.0
    idle = sum(1 for u in util_values if u < 5)
    mem_str = f"{_format_gb(used_sum)}/{_format_gb(total_sum)}" if total_sum else "N/A"

    # Build GPU summary part
    gpu_summary = f"{len(table)} GPUs | {idle} idle | {avg_util:.0f}% avg | {mem_str}"

    sys_stats = info.get("system_stats", {})
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
        swap = f"(Swap:{swap_used_gb:.1f}/{swap_total_gb:.0f}G)" if swap_total_gb > 0 else ""
        sys_parts.append(f"Mem: {mem_used_gb:.0f}/{mem_total_gb:.0f}G{swap}")

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
        mib_int = int(mib)
        if mib_int <= 0:
            continue
        rows.append({"user": str(user), "vram_mib": mib_int, "vram": _format_mib(mib_int)})
    if not rows:
        return pd.DataFrame(columns=["user", "vram", "vram_mib"])
    df = pd.DataFrame(rows).sort_values(by="vram_mib", ascending=False, kind="stable").reset_index(drop=True)
    return df


_USER_MEM_RE = re.compile(r"([^\s()]+)\((\d+)\s*M\)", flags=re.IGNORECASE)


def _parse_user_memory_compact(value) -> dict:
    users = {}
    for username, mib_str in _USER_MEM_RE.findall(str(value)):
        mib = int(mib_str)
        if username == "N/A" or mib <= 0:
            continue
        users[username] = users.get(username, 0) + mib
    return users


def _user_memory_from_df(df: pd.DataFrame) -> dict:
    totals = {}
    if df.empty or "processes" not in df.columns:
        return totals
    for value in df["processes"]:
        for user, mib in _parse_user_memory_compact(value).items():
            totals[user] = totals.get(user, 0) + mib
    return totals


_EMPTY_SHARE_COLUMNS = ["user", "snapshots", "share"]


def _user_time_share_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns or "processes" not in df.columns:
        return pd.DataFrame(columns=_EMPTY_SHARE_COLUMNS)

    total = int(df["timestamp"].nunique())
    if total <= 0:
        return pd.DataFrame(columns=_EMPTY_SHARE_COLUMNS)

    counts = {}
    # groupby drops NaT keys, matching the nunique() total above.
    for _ts, group in df.groupby("timestamp", sort=False):
        users = set()
        for value in group["processes"]:
            users.update(_parse_user_memory_compact(value).keys())
        for user in users:
            counts[user] = counts.get(user, 0) + 1

    rows = [{"user": user, "snapshots": snaps, "share": (snaps / total) * 100.0} for user, snaps in counts.items()]
    if not rows:
        return pd.DataFrame(columns=_EMPTY_SHARE_COLUMNS)
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


# Log viewer charts: label, value parser, shown by default, and a fixed
# y-axis top for percentage metrics.
_LOG_METRICS = {
    "util_gpu": {"label": "GPU Util (%)", "parser": _parse_percent, "default": True, "ymax": 100},
    "memory_used": {"label": "VRAM Used (GiB)", "parser": _parse_memory_gib, "default": True},
    "temperature": {"label": "Temperature (°C)", "parser": _parse_temperature_c, "default": True},
    "power": {"label": "Power Draw (W)", "parser": _parse_power_watts, "default": True},
    "util_mem": {"label": "Memory Util (%)", "parser": _parse_percent, "default": False, "ymax": 100},
    "rx": {"label": "PCIe RX (MB/s)", "parser": _parse_bandwidth_mbps, "default": True},
    "tx": {"label": "PCIe TX (MB/s)", "parser": _parse_bandwidth_mbps, "default": True},
    "fan_speed": {"label": "Fan Speed (%)", "parser": _parse_percent, "default": False, "ymax": 100},
}


def _downsample_per_gpu(df: pd.DataFrame, *, max_points_per_gpu: int) -> pd.DataFrame:
    frames = []
    for _gpu_id, group in df.groupby("gpu_id", sort=False):
        group_sorted = group.sort_values(by="timestamp", kind="stable")
        n = len(group_sorted)
        if n <= max_points_per_gpu:
            frames.append(group_sorted)
            continue
        step = int(math.ceil(n / max_points_per_gpu))
        frames.append(group_sorted.iloc[::step])
    if not frames:
        return df.iloc[:0]
    return pd.concat(frames, ignore_index=True)
