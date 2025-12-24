"""
Streamlit web interface for nvidb (Live GPU + Log viewer).

Usage:
    nvidb web                 # Web dashboard (Live + Logs)
    # Select log sessions from the left sidebar after the server starts.
"""

import argparse
from datetime import datetime
import importlib.util
import math
import platform
import re
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

try:
    from nvidb import config
except ImportError:  # pragma: no cover
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
    if table is None or table.empty:
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
        return f"{gpu_summary} | {' | '.join(sys_parts)}"
    return gpu_summary


def _strip_gpu_name(value):
    text = "" if value is None else str(value)
    return text.replace("NVIDIA", "").replace("GeForce", "").strip()


def _get_local_system_info():
    import os

    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }

    mem_gb = None
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    mem_gb = kb / 1024.0 / 1024.0
                    break
    except Exception:
        pass

    if mem_gb is not None:
        info["memory_total_gb"] = round(mem_gb, 2)
    return info


def _ensure_streamlit():
    if st is None:
        raise RuntimeError("streamlit is required for `nvidb web` (install with `pip install streamlit`).")


_TEAL_PRIMARY = "#00BFA5"
_SIDEBAR_WIDTH_PX = 360
_CHART_COLOR_RANGE = [
    "#00BFA5",
    "#00ACC1",
    "#1E88E5",
    "#26C6DA",
    "#5C6BC0",
    "#00897B",
    "#039BE5",
    "#4DD0E1",
    "#80CBC4",
    "#26A69A",
]


def _apply_streamlit_theme(*, theme_mode: str) -> None:
    _ensure_streamlit()
    theme_mode_norm = str(theme_mode or "Light").strip().lower()
    desired_base = "dark" if theme_mode_norm == "dark" else "light"

    try:
        current_base = st._config.get_option("theme.base")
    except Exception:  # pragma: no cover
        current_base = None

    current_base_norm = str(current_base).strip().lower() if current_base else None
    if current_base_norm == desired_base:
        return

    st._config.set_option("theme.base", desired_base)
    _trigger_rerun()


def _apply_app_styles():
    _ensure_streamlit()

    # Base styles for sidebar and controls (theme colors come from Streamlit).
    base_css = f"""
        section[data-testid="stSidebar"] {{
          min-width: {_SIDEBAR_WIDTH_PX}px !important;
          width: {_SIDEBAR_WIDTH_PX}px !important;
        }}
        section[data-testid="stSidebar"] > div {{
          min-width: {_SIDEBAR_WIDTH_PX}px !important;
          width: {_SIDEBAR_WIDTH_PX}px !important;
        }}
        section[data-testid="stSidebar"] .stRadio label {{
          font-size: 0.98rem !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
          font-size: 0.98rem !important;
        }}
        div[data-testid="stSegmentedControl"] button {{
          font-size: 1.05rem !important;
          padding: 0.35rem 0.85rem !important;
        }}
    """

    st.markdown(
        f"""
        <style>
        {base_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _trigger_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    if hasattr(st, "experimental_rerun"):  # pragma: no cover
        st.experimental_rerun()


def _maybe_cache_data(**cache_kwargs):
    def decorator(func):
        if st is None:
            return func
        try:
            import streamlit.runtime as st_runtime
        except Exception:  # pragma: no cover
            st_runtime = None
        if st_runtime is not None:
            try:
                if not st_runtime.exists():
                    return func
            except Exception:  # pragma: no cover
                return func
        cache = getattr(st, "cache_data", None)
        if cache is None:
            return func
        return cache(**cache_kwargs)(func)

    return decorator


def _autorefresh(seconds: int, *, enabled: bool, key: str):
    _ensure_streamlit()
    if not enabled:
        return
    try:
        seconds_int = int(seconds)
    except Exception:
        seconds_int = 5
    seconds_int = max(1, min(3600, seconds_int))

    try:
        import streamlit.components.v1 as components
    except Exception:  # pragma: no cover
        return

    interval_ms = seconds_int * 1000
    components.html(
        f"""
        <script>
        const refreshKey = {key!r};
        const interval = {interval_ms};
        const sendMessage = (type, data) => {{
          window.parent.postMessage({{ isStreamlitMessage: true, type, ...data }}, "*");
        }};
        // Register as a Streamlit component so setComponentValue triggers a rerun.
        sendMessage("streamlit:componentReady", {{ apiVersion: 1 }});

        window.__nvidbAutoRefreshTimers = window.__nvidbAutoRefreshTimers || {{}};
        if (window.__nvidbAutoRefreshTimers[refreshKey]) {{
          clearTimeout(window.__nvidbAutoRefreshTimers[refreshKey]);
        }}
        window.__nvidbAutoRefreshTimers[refreshKey] = setTimeout(() => {{
          sendMessage("streamlit:setComponentValue", {{ value: Date.now() }});
        }}, interval);
        </script>
        """,
        height=0,
        width=0,
    )


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


_GPU_TABLE_COLUMNS = [
    ("GPU", "GPU"),
    ("name", "name"),
    ("fan", "fan"),
    ("util", "util (GPU%)"),
    ("temp", "temp"),
    ("rx", "rx"),
    ("tx", "tx"),
    ("power", "power"),
    ("mem_util", "mem_util (mem%)"),
    ("memory[used/total]", "mem"),
    ("processes", "processes"),
]

_DEFAULT_GPU_TABLE_COLUMNS = [col_name for col_name, _label in _GPU_TABLE_COLUMNS]


def _center_dataframe(df: pd.DataFrame, *, max_cells: int = 8000):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    try:
        cells = int(df.shape[0]) * int(df.shape[1])
    except Exception:  # pragma: no cover
        cells = max_cells + 1
    if cells > max_cells:
        return df

    styled = df.style.set_properties(**{"text-align": "center"})
    styled = styled.set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}],
        overwrite=False,
    )
    return styled


def _render_gpu_column_checkboxes(available_columns, *, key_prefix: str):
    _ensure_streamlit()
    available = set(available_columns or [])
    controls = st.container()
    controls.markdown("**Columns**")
    grid = controls.columns(6)

    selected = []
    for idx, (col_name, label) in enumerate(_GPU_TABLE_COLUMNS):
        default_checked = col_name in _DEFAULT_GPU_TABLE_COLUMNS
        disabled = False
        if available:
            disabled = col_name not in available
            if disabled:
                default_checked = False
        checked = grid[idx % 6].checkbox(
            label,
            value=default_checked,
            key=f"{key_prefix}_{col_name}",
            disabled=disabled,
        )
        if checked and not disabled:
            selected.append(col_name)

    if not selected:
        selected = list(_DEFAULT_GPU_TABLE_COLUMNS)
    return selected


def _render_progress_bar_html(value, max_value=100):
    """Generate HTML for a progress bar with percentage"""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        pct = float(value)
        pct = max(0, min(100, pct))
    except Exception:
        return "N/A"

    # Color based on percentage using CSS variables
    if pct >= 80:
        color = "var(--nvidb-progress-high, #c92a2a)"
    elif pct >= 50:
        color = "var(--nvidb-progress-medium, #e67700)"
    else:
        color = "var(--nvidb-progress-low, #2b8a3e)"

    return f'''<div style="display:flex;align-items:center;justify-content:center;gap:6px;">
        <div style="width:60px;height:8px;background:var(--nvidb-progress-bg, #ddd);border-radius:4px;overflow:hidden;">
            <div style="width:{pct:.0f}%;height:100%;background:{color};"></div>
        </div>
        <span style="font-size:0.85em;min-width:35px;">{pct:.0f}%</span>
    </div>'''


def _render_gpu_table(df: pd.DataFrame, *, visible_columns=None):
    _ensure_streamlit()
    if df is None or df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
        return

    table = df.copy()
    available_cols = list(table.columns)
    desired_cols = list(visible_columns) if visible_columns else list(available_cols)

    # Remove mem% from desired_cols since we'll merge it with memory[used/total]
    if "mem%" in desired_cols:
        desired_cols.remove("mem%")

    def ratio_percent(value):
        used, total = _parse_mib_pair(value)
        if used is None or total in (None, 0):
            return None
        return (float(used) / float(total)) * 100

    column_order = []
    for col_name in desired_cols:
        if col_name == "mem%":
            continue
        if col_name in table.columns:
            column_order.append(col_name)

    if not column_order:
        column_order = list(available_cols)

    # Create display dataframe
    display = table[column_order].copy()

    # Process util/mem columns for Streamlit-native rendering.
    column_config = {}
    if "util" in display.columns:
        display["util"] = display["util"].map(_parse_percent)
        column_config["util"] = st.column_config.ProgressColumn(
            "util",
            min_value=0,
            max_value=100,
            format="%.0f%%",
            color="primary",
        )

    if "mem_util" in display.columns:
        display["mem_util"] = display["mem_util"].map(_parse_percent)
        column_config["mem_util"] = st.column_config.ProgressColumn(
            "mem_util",
            min_value=0,
            max_value=100,
            format="%.0f%%",
            color="primary",
        )

    if "memory[used/total]" in display.columns:
        display["memory[used/total]"] = display["memory[used/total]"].map(ratio_percent)
        display = display.rename(columns={"memory[used/total]": "mem"})
        column_order = ["mem" if col == "memory[used/total]" else col for col in column_order]
        display = display[column_order]
        column_config["mem"] = st.column_config.ProgressColumn(
            "mem",
            min_value=0,
            max_value=100,
            format="%.0f%%",
            color="primary",
        )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )


def _load_server_list():
    try:
        from nvidb.data_modules import ServerListInfo
    except ImportError:  # pragma: no cover
        from .data_modules import ServerListInfo

    cfg_path = Path(config.get_config_path()).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return ServerListInfo.from_yaml(cfg_path)


def _get_pool(include_remote):
    _ensure_streamlit()

    try:
        from nvidb.connection import NVClientPool
    except ImportError:  # pragma: no cover
        from .connection import NVClientPool

    key = f"_nvidb_pool_remote_{bool(include_remote)}"
    if key in st.session_state:
        return st.session_state[key]

    server_list = None
    if include_remote:
        server_list = _load_server_list()

    pool = NVClientPool(server_list)
    st.session_state[key] = pool
    return pool


class _LiveStatsCache:
    def __init__(self, pool, *, interval_seconds: int, enabled: bool):
        self.pool = pool
        self._lock = threading.Lock()
        self._interval_seconds = 5
        self._enabled = bool(enabled)

        self.raw_stats_by_client = None
        self.last_updated = None
        self.last_error = None
        self.last_fetch_duration_s = None
        self.fetch_in_progress = False

        self._stop_event = threading.Event()
        self._refresh_event = threading.Event()

        self.set_schedule(interval_seconds=interval_seconds, enabled=enabled, trigger_refresh=True)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_schedule(self, *, interval_seconds: int, enabled: bool, trigger_refresh: bool = False):
        try:
            interval_seconds_int = int(interval_seconds)
        except Exception:
            interval_seconds_int = 5
        interval_seconds_int = max(1, min(3600, interval_seconds_int))

        with self._lock:
            changed = (
                interval_seconds_int != self._interval_seconds
                or bool(enabled) != self._enabled
            )
            self._interval_seconds = interval_seconds_int
            self._enabled = bool(enabled)

        if changed or trigger_refresh:
            self._refresh_event.set()

    def request_refresh(self):
        self._refresh_event.set()

    def snapshot(self):
        with self._lock:
            return {
                "raw_stats_by_client": self.raw_stats_by_client,
                "last_updated": self.last_updated,
                "last_error": self.last_error,
                "last_fetch_duration_s": self.last_fetch_duration_s,
                "fetch_in_progress": self.fetch_in_progress,
                "interval_seconds": self._interval_seconds,
                "enabled": self._enabled,
            }

    def stop(self):  # pragma: no cover
        self._stop_event.set()
        self._refresh_event.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass

    def _run(self):  # pragma: no cover
        while not self._stop_event.is_set():
            with self._lock:
                enabled = self._enabled
                interval = self._interval_seconds

            if enabled:
                self._refresh_event.wait(timeout=interval)
            else:
                self._refresh_event.wait()

            self._refresh_event.clear()
            if self._stop_event.is_set():
                break

            start = time.monotonic()
            with self._lock:
                self.fetch_in_progress = True

            try:
                _formatted, raw_stats = self.pool.get_client_gpus_info(return_raw=True)
                now = datetime.now()
                with self._lock:
                    self.raw_stats_by_client = raw_stats
                    self.last_updated = now
                    self.last_error = None
            except Exception as e:
                with self._lock:
                    self.last_error = str(e)
            finally:
                duration = time.monotonic() - start
                with self._lock:
                    self.last_fetch_duration_s = duration
                    self.fetch_in_progress = False


def _get_live_cache(pool, *, interval_seconds: int, enabled: bool):
    _ensure_streamlit()
    key = f"_nvidb_live_cache_{id(pool)}"
    cache = st.session_state.get(key)
    if cache is None or getattr(cache, "pool", None) is not pool:
        cache = _LiveStatsCache(pool, interval_seconds=interval_seconds, enabled=enabled)
        st.session_state[key] = cache
        return cache

    cache.set_schedule(interval_seconds=interval_seconds, enabled=enabled)
    return cache


def show_live_dashboard(*, include_remote):
    _ensure_streamlit()

    st.header("Live GPU")

    try:
        controls = st.container(border=True)
    except TypeError:  # pragma: no cover
        controls = st.container()
    col1, col2, col3 = controls.columns(3)
    with col1:
        refresh_interval = st.slider(
            "Refresh interval (seconds)",
            1,
            30,
            5,
            key="_nvidb_live_refresh_interval",
        )
    with col2:
        auto_refresh = st.checkbox(
            "Auto refresh",
            value=True,
            key="_nvidb_live_auto_refresh",
        )
    with col3:
        refresh_now = st.button(
            "Refresh now",
            key="_nvidb_live_refresh_now",
        )
    use_fragment = hasattr(st, "fragment")

    run_every = refresh_interval if auto_refresh else None

    effective_remote = bool(include_remote)
    try:
        pool = _get_pool(include_remote)
    except Exception as e:
        effective_remote = False
        st.error(str(e))
        pool = _get_pool(False)

    cache = _get_live_cache(pool, interval_seconds=refresh_interval, enabled=auto_refresh)
    if refresh_now:
        cache.request_refresh()

    def _render():
        if effective_remote:
            st.caption("Remote: enabled")
        else:
            st.caption("Remote: disabled (switch to `Live-remote` to include remote servers)")

        snapshot = cache.snapshot()
        raw_stats_by_client = snapshot.get("raw_stats_by_client")
        last_updated = snapshot.get("last_updated")
        last_error = snapshot.get("last_error")
        fetch_in_progress = snapshot.get("fetch_in_progress")
        last_fetch_duration_s = snapshot.get("last_fetch_duration_s")

        if last_error:
            st.warning(f"Last fetch error: {last_error}")

        available_columns = set()
        if isinstance(raw_stats_by_client, dict):
            for idx in range(len(pool.pool)):
                try:
                    table, _system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))
                except Exception:
                    continue
                if isinstance(table, pd.DataFrame) and not table.empty:
                    available_columns.update(table.columns)

        visible_columns = _render_gpu_column_checkboxes(
            available_columns,
            key_prefix="_nvidb_live_cols_v2",
        )

        if raw_stats_by_client is None:
            st.info("Fetching GPU data...")
            return

        meta = {}
        if isinstance(raw_stats_by_client, dict):
            meta = raw_stats_by_client.get("_nvidb", {}) or {}
        user_memory_by_client = meta.get("user_memory_by_client", {}) or {}
        global_user_memory = meta.get("user_memory_global", {}) or {}

        if global_user_memory:
            with st.expander("User VRAM totals (all nodes)", expanded=False):
                summary_df = _user_summary_df(global_user_memory)
                st.dataframe(
                    _center_dataframe(summary_df[["user", "vram"]]),
                    width="stretch",
                    hide_index=True,
                )

        multi = len(pool.pool) > 1
        for idx, client in enumerate(pool.pool):
            description = getattr(client, "description", "unknown")
            if isinstance(raw_stats_by_client, dict):
                table, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))
            else:
                table, system_info = pd.DataFrame(), {}

            user_summary = {}
            if isinstance(user_memory_by_client, dict):
                user_summary = user_memory_by_client.get(idx, {}) or {}

            if isinstance(system_info, dict) and system_info.get("error"):
                title = f"[{idx + 1}] {description} | Error"

                def body(system_info=system_info):
                    st.error(system_info.get("error", "Unknown error"))

            elif table is None or table.empty:
                title = f"[{idx + 1}] {description} | No GPUs"

                def body(system_info=system_info, description=description):
                    payload = {"nvidia": system_info or {}}
                    if str(description).startswith("Local Machine"):
                        payload["system"] = _get_local_system_info()
                    st.json(payload)

            else:
                title = f"[{idx + 1}] {description} | {_server_summary(table, system_info)}"

                def body(table=table, system_info=system_info, user_summary=user_summary, visible_columns=visible_columns):
                    if (
                        isinstance(system_info, dict)
                        and any(k in system_info for k in ("driver_version", "cuda_version", "attached_gpus"))
                    ):
                        st.caption(
                            f"Driver: {system_info.get('driver_version', 'N/A')} | "
                            f"CUDA: {system_info.get('cuda_version', 'N/A')} | "
                            f"GPUs: {system_info.get('attached_gpus', '0')}"
                        )

                    _render_gpu_table(table, visible_columns=visible_columns)

                    if user_summary:
                        with st.expander("User VRAM totals (this node)", expanded=False):
                            summary_df = _user_summary_df(user_summary)
                            st.dataframe(
                                _center_dataframe(summary_df[["user", "vram"]]),
                                width="stretch",
                                hide_index=True,
                            )

            if multi:
                with st.expander(title, expanded=(idx == 0)):
                    body()
            else:
                st.subheader(description)
                body()

        status_parts = []
        if fetch_in_progress:
            status_parts.append("Updating…")
        if last_updated is not None:
            status_parts.append(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        if last_fetch_duration_s is not None:
            status_parts.append(f"Fetch: {last_fetch_duration_s:.2f}s")
        if status_parts:
            st.caption(" | ".join(status_parts))

    if use_fragment:
        render = st.fragment(_render, run_every=run_every)
        render()
    else:
        _autorefresh(refresh_interval, enabled=auto_refresh, key="_nvidb_autorefresh_live")
        _render()


def _build_log_snapshot_table(df):
    try:
        from nvidb.utils import extract_numbers
    except ImportError:  # pragma: no cover
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


@_maybe_cache_data(ttl=5)
def _load_sessions_cached(db_path: str) -> pd.DataFrame:
    return load_sessions(db_path)


@_maybe_cache_data(ttl=5)
def _load_session_logs_cached(db_path: str, session_id: int) -> pd.DataFrame:
    return load_session_logs(db_path, session_id)


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


def _render_timeseries_chart(
    df_long: pd.DataFrame,
    *,
    title: str,
    tooltip_format: str,
    height: int = 220,
    series_field: str = "gpu_id",
    series_title: str = "GPU",
):
    _ensure_streamlit()
    if df_long is None or df_long.empty:
        st.info("No data to plot.")
        return
    if alt is None:  # pragma: no cover
        st.warning("altair is required for charts.")
        return

    theme_mode_norm = "light"
    try:
        theme_mode_norm = str(st.session_state.get("_nvidb_theme_mode", "Light") or "Light").strip().lower()
    except Exception:  # pragma: no cover
        theme_mode_norm = "light"
    if theme_mode_norm == "dark":
        axis_label_color = "#cbd5e1"
        title_color = "#e2e8f0"
        grid_color = "rgba(148, 163, 184, 0.18)"
        domain_color = "rgba(148, 163, 184, 0.35)"
    else:
        axis_label_color = "#334155"
        title_color = "#0f172a"
        grid_color = "rgba(15, 23, 42, 0.08)"
        domain_color = "rgba(15, 23, 42, 0.18)"

    data = df_long.dropna(subset=["timestamp", "gpu_id", "value"]).copy()
    if data.empty:
        st.info("No data to plot.")
        return
    series_field = str(series_field or "gpu_id")
    if series_field not in data.columns:
        series_field = "gpu_id"
    data[series_field] = data[series_field].astype(str)

    hover = alt.selection_point(
        fields=[series_field],
        nearest=True,
        on="mouseover",
        clear="mouseout",
        empty="none",
    )

    color_scale = alt.Scale(range=_CHART_COLOR_RANGE)

    x = alt.X(
        "timestamp:T",
        title=None,
        axis=alt.Axis(
            tickCount=6,
            labelOverlap="greedy",
            format="%m-%d %H:%M",
        ),
    )

    lines = alt.Chart(data).mark_line().encode(
        x=x,
        y=alt.Y("value:Q", title=None),
        color=alt.Color(f"{series_field}:N", title=str(series_title or "GPU"), scale=color_scale),
    )

    highlight = alt.Chart(data).mark_line(size=4).encode(
        x=x,
        y=alt.Y("value:Q", title=None),
        color=alt.Color(f"{series_field}:N", legend=None, scale=color_scale),
    ).transform_filter(hover)

    points = alt.Chart(data).mark_circle(size=60, opacity=0).encode(
        x=x,
        y="value:Q",
        color=alt.Color(f"{series_field}:N", legend=None, scale=color_scale),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Time"),
            alt.Tooltip(f"{series_field}:N", title=str(series_title or "GPU")),
            alt.Tooltip("value:Q", title=title, format=str(tooltip_format or "")),
        ],
    )

    chart = (
        alt.layer(lines, highlight, points)
        .add_params(hover)
        .properties(height=height, title=title, background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor=axis_label_color,
            titleColor=title_color,
            gridColor=grid_color,
            domainColor=domain_color,
            tickColor=domain_color,
        )
        .configure_legend(labelColor=axis_label_color, titleColor=title_color)
        .configure_title(color=title_color)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")


def show_logs_dashboard(db_path, *, initial_session_id=None):
    _ensure_streamlit()

    db_path = _as_path(db_path or get_db_path())
    if db_path is None or not db_path.exists():
        st.error(f"Database not found: {db_path}")
        st.info("Run `nvidb log` first to start logging.")
        return

    sessions_df = _load_sessions_cached(str(db_path))
    if sessions_df.empty:
        st.warning("No log sessions found.")
        return

    st.sidebar.subheader("Sessions")
    query = st.sidebar.text_input("Search", value="", placeholder="id / time / status")

    sessions_df = sessions_df.sort_values(by="id", ascending=True, kind="stable")

    session_options = []
    for _, row in sessions_df.iterrows():
        session_id = int(row.get("id"))
        start_time = _format_datetime(row.get("start_time"), include_seconds=False)
        duration = _format_duration(row.get("start_time"), row.get("end_time"))
        status = str(row.get("status") or "unknown")
        snapshot_count = int(row.get("snapshot_count") or 0)
        label = f"#{session_id} | {start_time} | {duration} | {status} | {snapshot_count} snaps"
        session_options.append((label, session_id))

    if query.strip():
        q = query.strip().lower()
        session_options = [(label, sid) for label, sid in session_options if q in label.lower() or q in str(sid)]

    if not session_options:
        st.sidebar.info("No sessions match the current search.")
        return

    labels = [label for label, _sid in session_options]
    label_to_id = {label: sid for label, sid in session_options}

    default_index = max(0, len(session_options) - 1)
    if initial_session_id is not None:
        try:
            initial_session_id_int = int(initial_session_id)
        except Exception:
            initial_session_id_int = None
        if initial_session_id_int is not None:
            for idx, (_label, sid) in enumerate(session_options):
                if sid == initial_session_id_int:
                    default_index = idx
                    break

    st.sidebar.markdown("`id | start | dur | status | snaps`")
    try:
        session_container = st.sidebar.container(height=520, border=True)
    except TypeError:  # pragma: no cover
        session_container = st.sidebar.container()

    selected_label = session_container.radio(
        "Session",
        labels,
        index=default_index,
        key="_nvidb_session_selector_v1",
        label_visibility="collapsed",
    )
    session_id = label_to_id[selected_label]

    df = _load_session_logs_cached(str(db_path), int(session_id))
    if df.empty:
        st.warning("No logs found for this session.")
        return

    session_info = sessions_df[sessions_df["id"] == session_id].iloc[0]
    st.header(f"Session {session_id}")
    start_time_raw = session_info.get("start_time")
    end_time_raw = session_info.get("end_time")
    if end_time_raw in ("None", ""):
        end_time_raw = None

    status = str(session_info.get("status") or "unknown")
    interval_seconds = session_info.get("interval_seconds")
    include_remote = bool(session_info.get("include_remote"))
    record_count = int(session_info.get("record_count") or 0)
    snapshot_count = int(session_info.get("snapshot_count") or 0)

    duration_str = _format_duration(start_time_raw, end_time_raw)
    start_str = _format_datetime(start_time_raw, include_seconds=True)
    end_str = _format_datetime(end_time_raw, include_seconds=True) if end_time_raw is not None else "running"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Status", status)
    with col2:
        st.metric("Interval", f"{interval_seconds}s" if interval_seconds else "N/A")
    with col3:
        st.metric("Snapshots", snapshot_count)
    with col4:
        st.metric("Records", record_count)
    with col5:
        st.metric("Remote", "Yes" if include_remote else "No")

    st.markdown(f"**Start:** `{start_str}`  \n**End:** `{end_str}`  \n**Duration:** `{duration_str}`")

    try:
        controls = st.container(border=True)
    except TypeError:  # pragma: no cover
        controls = st.container()
    controls.subheader("Display Options")

    nodes_all = sorted({str(n) for n in df["node"].dropna().unique()})
    nodes_col, time_col = controls.columns([1, 2])
    with nodes_col:
        selected_nodes = st.multiselect(
            "Nodes",
            options=nodes_all,
            default=nodes_all,
            key=f"_nvidb_nodes_{session_id}",
        )
    if not selected_nodes:
        st.info("Select at least one node to display.")
        return
    df = df[df["node"].isin(selected_nodes)]

    min_ts = df["timestamp"].dropna().min()
    max_ts = df["timestamp"].dropna().max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        st.warning("No timestamps available in this session.")
        return

    with time_col:
        start_ts, end_ts = st.slider(
            "Time range",
            min_value=min_ts.to_pydatetime(),
            max_value=max_ts.to_pydatetime(),
            value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()),
            key=f"_nvidb_time_range_{session_id}",
        )
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]

    max_points = controls.slider(
        "Max points / GPU (charts)",
        50,
        2000,
        600,
        step=50,
        key=f"_nvidb_max_points_{session_id}",
    )

    metric_keys = list(_LOG_METRICS.keys())
    default_metrics = [k for k, spec in _LOG_METRICS.items() if spec.get("default")]
    selected_metrics = controls.multiselect(
        "Metrics",
        options=metric_keys,
        default=default_metrics,
        format_func=lambda k: (_LOG_METRICS.get(k) or {}).get("label", k),
        key=f"_nvidb_metrics_{session_id}",
    )

    timestamps = df["timestamp"].dropna().drop_duplicates().sort_values()
    if timestamps.empty:
        st.warning("No timestamps available in the selected time range.")
        return

    ts_list = list(timestamps)
    ts_idx = controls.slider(
        "Snapshot index",
        0,
        len(ts_list) - 1,
        len(ts_list) - 1,
        key=f"_nvidb_snapshot_idx_{session_id}",
    )
    selected_ts = ts_list[ts_idx]

    snapshot = df[df["timestamp"] == selected_ts]
    st.subheader(f"Snapshot @ {selected_ts}")

    snapshot_nodes = sorted({str(n) for n in snapshot["node"].dropna().unique()})
    if snapshot_nodes:
        node_colors = {node: _CHART_COLOR_RANGE[idx % len(_CHART_COLOR_RANGE)] for idx, node in enumerate(snapshot_nodes)}
    else:
        node_colors = {}

    if len(snapshot_nodes) > 1:
        st.caption("Nodes")
        chips = []
        for node in snapshot_nodes:
            color = node_colors.get(node, _TEAL_PRIMARY)
            safe_label = str(node).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            chips.append(
                f"""<span style="display:inline-flex;align-items:center;margin:0 10px 8px 0;">
                    <span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:{color};margin-right:6px;"></span>
                    <span style="font-size:0.95rem;">{safe_label}</span>
                </span>"""
            )
        st.markdown("""<div style="display:flex;flex-wrap:wrap;">""" + "".join(chips) + "</div>", unsafe_allow_html=True)

    overview_rows = []
    for node in snapshot_nodes:
        node_snapshot = snapshot[snapshot["node"] == node]
        node_table = _build_log_snapshot_table(node_snapshot)
        if node_table.empty:
            continue

        util_values = []
        used_sum = 0.0
        total_sum = 0.0
        for _, row in node_table.iterrows():
            util_values.append(_parse_percent(row.get("util")))
            used, total = _parse_mib_pair(row.get("memory[used/total]"))
            if used is not None and total is not None:
                used_sum += float(used)
                total_sum += float(total)

        util_clean = [u for u in util_values if u is not None]
        idle = sum(1 for u in util_clean if u < 5)
        avg_util = (sum(util_clean) / len(util_clean)) if util_clean else 0.0
        mem_str = f"{_format_gb(used_sum)}/{_format_gb(total_sum)}" if total_sum else "N/A"

        overview_rows.append(
            {
                "node": node,
                "gpus": int(len(node_table)),
                "idle": int(idle),
                "avg_util": f"{avg_util:.0f}%",
                "vram_used/total": mem_str,
            }
        )

    if overview_rows:
        overview_df = pd.DataFrame(overview_rows)
        st.dataframe(_center_dataframe(overview_df), width="stretch", hide_index=True)

    snapshot_user_summary = _user_memory_from_df(snapshot)
    if snapshot_user_summary:
        with st.expander("User VRAM totals @ snapshot (all nodes)", expanded=False):
            summary_df = _user_summary_df(snapshot_user_summary)
            st.dataframe(
                _center_dataframe(summary_df[["user", "vram"]]),
                width="stretch",
                hide_index=True,
            )

    parsed_cols = {}
    parsed_sources = []
    for metric_key in selected_metrics:
        spec = _LOG_METRICS.get(metric_key)
        if not spec:
            continue
        parsed_sources.append(spec.get("source", metric_key))
    parsed_sources = sorted({c for c in parsed_sources if c in df.columns})

    parsed = df[["timestamp", "node", "gpu_id", *parsed_sources]].copy()
    for metric_key in selected_metrics:
        spec = _LOG_METRICS.get(metric_key)
        if not spec:
            continue
        source = spec.get("source", metric_key)
        parser = spec.get("parser")
        if source not in parsed.columns or parser is None:
            continue
        out_col = f"_nvidb_{metric_key}"
        parsed[out_col] = parsed[source].map(parser)
        parsed_cols[metric_key] = out_col

    nodes = sorted({str(n) for n in parsed["node"].dropna().unique()})
    for node_idx, node in enumerate(nodes):
        node_snapshot = snapshot[snapshot["node"] == node]
        node_table = _build_log_snapshot_table(node_snapshot)
        title = f"{node} | {_server_summary(node_table)}"

        with st.expander(title, expanded=(node_idx == 0)):
            node_df = parsed[parsed["node"] == node]
            if node_df.empty:
                st.info("No timeseries data for this node.")
                if node_table.empty:
                    st.info("No snapshot data for this node at the selected time.")
                else:
                    _render_gpu_table(node_table)
                continue

            node_gpu_ids = sorted({int(g) for g in node_df["gpu_id"].dropna().unique()})
            visible_node_gpus = list(node_gpu_ids)
            if len(node_gpu_ids) > 1:
                visible_node_gpus = st.multiselect(
                    "Visible GPUs (this node)",
                    options=node_gpu_ids,
                    default=node_gpu_ids,
                    key=f"_nvidb_node_visible_gpus_{session_id}_{node}",
                )

            node_time_df = df[df["node"] == node]
            if visible_node_gpus:
                node_time_df = node_time_df[node_time_df["gpu_id"].isin(visible_node_gpus)]
            time_share_df = _user_time_share_df(node_time_df)

            node_snapshot_users = _user_memory_from_df(node_snapshot)

            gpu_names = {}
            gpu_name_source = node_snapshot if not node_snapshot.empty else node_time_df
            if gpu_name_source is not None and not gpu_name_source.empty and "name" in gpu_name_source.columns:
                for _, row in gpu_name_source[["gpu_id", "name"]].dropna().drop_duplicates().iterrows():
                    try:
                        gpu_id_int = int(row.get("gpu_id"))
                    except Exception:
                        continue
                    name = _strip_gpu_name(row.get("name"))
                    if name and name != "N/A" and gpu_id_int not in gpu_names:
                        gpu_names[gpu_id_int] = name

            gpu_label_map = {}
            for gpu_id_int in node_gpu_ids:
                name = gpu_names.get(gpu_id_int)
                if name:
                    gpu_label_map[gpu_id_int] = f"GPU {gpu_id_int} ({name})"
                else:
                    gpu_label_map[gpu_id_int] = f"GPU {gpu_id_int}"

            if selected_metrics:
                node_df_for_charts = node_df
                if visible_node_gpus:
                    node_df_for_charts = node_df[node_df["gpu_id"].isin(visible_node_gpus)]
                else:
                    st.info("Select at least one GPU to show charts for this node.")
                    node_df_for_charts = node_df.iloc[:0]

                if not node_df_for_charts.empty:
                    cols = st.columns(2)
                    chart_slot = 0
                    for metric_key in selected_metrics:
                        out_col = parsed_cols.get(metric_key)
                        spec = _LOG_METRICS.get(metric_key) or {}
                        if out_col is None or out_col not in node_df_for_charts.columns:
                            continue

                        long_df = node_df_for_charts[["timestamp", "gpu_id", out_col]].rename(columns={out_col: "value"})
                        def _label_for_gpu(gid):
                            if pd.isna(gid):
                                return "GPU ?"
                            try:
                                gid_int = int(gid)
                            except Exception:
                                return f"GPU {gid}"
                            return gpu_label_map.get(gid_int, f"GPU {gid_int}")

                        long_df["gpu_label"] = long_df["gpu_id"].map(_label_for_gpu)
                        long_df = long_df.dropna(subset=["value"])
                        if long_df.empty:
                            continue
                        long_df = _downsample_per_gpu(long_df, max_points_per_gpu=max_points)
                        if long_df.empty:
                            continue

                        with cols[chart_slot % 2]:
                            metric_label = str(spec.get("label") or metric_key)
                            chart_title = metric_label if len(nodes) <= 1 else f"{node} • {metric_label}"
                            _render_timeseries_chart(
                                long_df,
                                title=chart_title,
                                tooltip_format=str(spec.get("tooltip_format") or ""),
                                height=int(spec.get("height") or 220),
                                series_field="gpu_label",
                                series_title="GPU",
                            )
                        chart_slot += 1
                else:
                    st.info("No chart data available for the current GPU selection.")
            else:
                st.info("Select metrics above to show trend charts.")

            if not time_share_df.empty:
                st.markdown("**User GPU-time share (selected time range)**")
                head = time_share_df.head(12)
                st.dataframe(
                    _center_dataframe(head),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "share": st.column_config.ProgressColumn(
                            "share",
                            min_value=0,
                            max_value=100,
                            format="%.0f%%",
                            color="primary",
                        )
                    },
                )
                if len(time_share_df) > len(head):
                    with st.expander("Show all users", expanded=False):
                        st.dataframe(
                            _center_dataframe(time_share_df),
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "share": st.column_config.ProgressColumn(
                                    "share",
                                    min_value=0,
                                    max_value=100,
                                    format="%.0f%%",
                                    color="primary",
                                )
                            },
                        )
            else:
                st.caption("No user process records in the selected range.")

            if node_snapshot_users:
                with st.expander("User VRAM totals @ snapshot (this node)", expanded=False):
                    summary_df = _user_summary_df(node_snapshot_users)
                    st.dataframe(
                        _center_dataframe(summary_df[["user", "vram"]]),
                        width="stretch",
                        hide_index=True,
                    )

            if node_table.empty:
                st.info("No snapshot data for this node at the selected time.")
            else:
                _render_gpu_table(node_table)

    with st.expander("View raw data", expanded=False):
        st.dataframe(_center_dataframe(df), width="stretch")


def main(*, session_id=None, db_path=None):
    _ensure_streamlit()

    st.set_page_config(page_title="nvidb web", page_icon="🖥️", layout="wide")
    theme_mode = st.session_state.get("_nvidb_theme_mode", "Light")
    theme_mode_norm = str(theme_mode or "Light").strip().lower()
    theme_mode = "Dark" if theme_mode_norm == "dark" else "Light"
    st.session_state["_nvidb_theme_mode"] = theme_mode
    _apply_streamlit_theme(theme_mode=theme_mode)
    _apply_app_styles()

    st.title("nvidb web")

    header_cols = st.columns([3, 2])

    default_view = "Logs" if session_id is not None else "Live-local"
    with header_cols[0]:
        if hasattr(st, "segmented_control"):
            view = st.segmented_control(
                "View",
                options=["Live-local", "Live-remote", "Logs"],
                default=default_view,
                key="_nvidb_view_v1",
                format_func=lambda v: (
                    "🟢 Live-local"
                    if v == "Live-local"
                    else ("🌐 Live-remote" if v == "Live-remote" else "📜 Logs")
                ),
            )
        else:  # pragma: no cover
            default_index = 2 if default_view == "Logs" else 0
            view = st.radio(
                "View",
                options=["Live-local", "Live-remote", "Logs"],
                index=default_index,
                horizontal=True,
                key="_nvidb_view_v1",
                format_func=lambda v: (
                    "🟢 Live-local"
                    if v == "Live-local"
                    else ("🌐 Live-remote" if v == "Live-remote" else "📜 Logs")
                ),
            )

    with header_cols[1]:
        if hasattr(st, "segmented_control"):
            st.segmented_control(
                "Theme",
                options=["Light", "Dark"],
                default=str(theme_mode or "Light"),
                key="_nvidb_theme_mode",
                format_func=lambda v: {"Light": "☀️ Light", "Dark": "🌙 Dark"}.get(v, str(v)),
            )
        else:  # pragma: no cover
            theme_index = 1 if theme_mode == "Dark" else 0
            st.selectbox(
                "Theme",
                options=["Light", "Dark"],
                index=theme_index,
                key="_nvidb_theme_mode",
            )

    if view == "Live-local":
        show_live_dashboard(include_remote=False)
    elif view == "Live-remote":
        show_live_dashboard(include_remote=True)
    else:
        show_logs_dashboard(db_path=db_path, initial_session_id=session_id)


def run_streamlit_app(*, session_id=None, db_path=None, port=8501, include_remote=False):
    import subprocess
    import sys

    if importlib.util.find_spec("streamlit") is None:
        print("streamlit is required for `nvidb web`.\nInstall: pip install streamlit")
        raise SystemExit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        __file__,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--",
    ]

    if db_path is not None:
        cmd.extend(["--db-path", str(db_path)])
    print(f"Starting Streamlit on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()
    main(db_path=args.db_path)
