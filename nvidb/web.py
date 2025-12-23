"""
Streamlit web interface for nvidb (Live GPU + Log viewer).

Usage:
    nvidb web                 # Live view (local)
    nvidb web --remote        # Live view (local + remote)
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


def _server_summary(table):
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
    return f"{len(table)} GPUs | {idle} idle | {avg_util:.0f}% avg | {mem_str}"


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


def _apply_app_styles():
    _ensure_streamlit()
    st.markdown(
        f"""
        <style>
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
        section[data-testid="stSidebar"] [data-testid=\"stMarkdownContainer\"] p {{
          font-size: 0.98rem !important;
        }}
        div[data-testid="stSegmentedControl"] button {{
          font-size: 1.05rem !important;
          padding: 0.35rem 0.85rem !important;
        }}
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


_GPU_TABLE_COLUMNS = [
    ("GPU", "GPU"),
    ("name", "name"),
    ("fan", "fan"),
    ("util", "util (GPU%)"),
    ("mem_util", "mem_util (mem%)"),
    ("temp", "temp"),
    ("rx", "rx"),
    ("tx", "tx"),
    ("power", "power"),
    ("memory[used/total]", "memory[used/total]"),
    ("mem%", "mem%"),
    ("processes", "processes"),
]

_DEFAULT_GPU_TABLE_COLUMNS = [col_name for col_name, _label in _GPU_TABLE_COLUMNS]


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
            if col_name == "mem%":
                disabled = "memory[used/total]" not in available
            else:
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


def _render_gpu_table(df: pd.DataFrame, *, visible_columns=None):
    _ensure_streamlit()
    if df is None or df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    table = df.copy()
    available_cols = list(table.columns)
    desired_cols = list(visible_columns) if visible_columns else list(available_cols)
    if visible_columns is None and "memory[used/total]" in available_cols and "mem%" not in desired_cols:
        try:
            desired_cols.insert(desired_cols.index("memory[used/total]") + 1, "mem%")
        except ValueError:
            desired_cols.append("mem%")

    column_config = {}

    def ratio_percent(value):
        used, total = _parse_mib_pair(value)
        if used is None or total in (None, 0):
            return None
        return (float(used) / float(total)) * 100

    mem_pct_col = None
    if "mem%" in desired_cols and "memory[used/total]" in table.columns:
        mem_pct_col = "_nvidb_mem_pct"
        table[mem_pct_col] = pd.to_numeric(table["memory[used/total]"].map(ratio_percent), errors="coerce")
        column_config[mem_pct_col] = st.column_config.ProgressColumn(
            "mem%",
            width="small",
            min_value=0,
            max_value=100,
            format="%.0f%%",
            color="primary",
        )

    column_order = []
    for col_name in desired_cols:
        if col_name == "mem%":
            if mem_pct_col is not None:
                column_order.append(mem_pct_col)
            continue
        if col_name in table.columns:
            column_order.append(col_name)

    if not column_order:
        column_order = list(available_cols)

    if "util" in column_order and "util" in table.columns:
        table["util"] = pd.to_numeric(table["util"].map(_parse_percent), errors="coerce")
        column_config["util"] = st.column_config.ProgressColumn(
            "util",
            width="small",
            min_value=0,
            max_value=100,
            format="%.0f%%",
            color="primary",
        )

    if "mem_util" in column_order and "mem_util" in table.columns:
        table["mem_util"] = pd.to_numeric(table["mem_util"].map(_parse_percent), errors="coerce")
        column_config["mem_util"] = st.column_config.ProgressColumn(
            "mem_util",
            width="small",
            min_value=0,
            max_value=100,
            format="%.0f%%",
            color="primary",
        )

    if "GPU" in column_order:
        column_config.setdefault("GPU", st.column_config.NumberColumn("GPU", width="small"))
    if "name" in column_order:
        column_config.setdefault("name", st.column_config.TextColumn("name", width="small", max_chars=20))
    if "fan" in column_order:
        column_config.setdefault("fan", st.column_config.TextColumn("fan", width="small"))
    if "temp" in column_order:
        column_config.setdefault("temp", st.column_config.TextColumn("temp", width="small"))
    if "rx" in column_order:
        column_config.setdefault("rx", st.column_config.TextColumn("rx", width="small"))
    if "tx" in column_order:
        column_config.setdefault("tx", st.column_config.TextColumn("tx", width="small"))
    if "power" in column_order:
        column_config.setdefault("power", st.column_config.TextColumn("power", width="medium", max_chars=18))
    if "memory[used/total]" in column_order:
        column_config.setdefault(
            "memory[used/total]",
            st.column_config.TextColumn("memory[used/total]", width="small", max_chars=16),
        )
    if "processes" in column_order:
        column_config.setdefault(
            "processes",
            st.column_config.TextColumn("processes", width="small", max_chars=24),
        )

    st.dataframe(
        table[column_order],
        use_container_width=True,
        hide_index=True,
        column_order=column_order,
        column_config=column_config or None,
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
            st.caption("Remote: disabled (run `nvidb web --remote` to include remote servers)")

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
                st.dataframe(summary_df[["user", "vram"]], use_container_width=True)

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
                title = f"[{idx + 1}] {description} | {_server_summary(table)}"

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
                            st.dataframe(summary_df[["user", "vram"]], use_container_width=True)

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
        "default": False,
        "height": 220,
    },
    "tx": {
        "label": "PCIe TX (MB/s)",
        "source": "tx",
        "parser": _parse_bandwidth_mbps,
        "tooltip_format": ".2f",
        "default": False,
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


def _render_timeseries_chart(df_long: pd.DataFrame, *, title: str, tooltip_format: str, height: int = 220):
    _ensure_streamlit()
    if df_long is None or df_long.empty:
        st.info("No data to plot.")
        return
    if alt is None:  # pragma: no cover
        st.warning("altair is required for charts.")
        return

    data = df_long.dropna(subset=["timestamp", "gpu_id", "value"]).copy()
    if data.empty:
        st.info("No data to plot.")
        return
    data["gpu_id"] = data["gpu_id"].astype(str)

    hover = alt.selection_point(
        fields=["gpu_id"],
        nearest=True,
        on="mouseover",
        clear="mouseout",
        empty="none",
    )

    color_scale = alt.Scale(range=_CHART_COLOR_RANGE)

    lines = alt.Chart(data).mark_line().encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("value:Q", title=None),
        color=alt.Color("gpu_id:N", title="GPU", scale=color_scale),
    )

    highlight = alt.Chart(data).mark_line(size=4).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("value:Q", title=None),
        color=alt.Color("gpu_id:N", legend=None, scale=color_scale),
    ).transform_filter(hover)

    points = alt.Chart(data).mark_circle(size=60, opacity=0).encode(
        x="timestamp:T",
        y="value:Q",
        color=alt.Color("gpu_id:N", legend=None, scale=color_scale),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Time"),
            alt.Tooltip("gpu_id:N", title="GPU"),
            alt.Tooltip("value:Q", title=title, format=str(tooltip_format or "")),
        ],
    )

    chart = (
        alt.layer(lines, highlight, points)
        .add_params(hover)
        .properties(height=height, title=title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


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

    gpu_ids_all = sorted({int(g) for g in df["gpu_id"].dropna().unique()})
    gpu_col, points_col = controls.columns(2)
    with gpu_col:
        visible_gpus = st.multiselect(
            "Visible GPUs",
            options=gpu_ids_all,
            default=gpu_ids_all,
            key=f"_nvidb_visible_gpus_{session_id}",
        )
    with points_col:
        max_points = st.slider(
            "Max points / GPU",
            50,
            2000,
            600,
            step=50,
            key=f"_nvidb_max_points_{session_id}",
        )
    if not visible_gpus:
        st.info("Select at least one GPU to display.")
        return
    df = df[df["gpu_id"].isin(visible_gpus)]

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
            if node_table.empty:
                st.info("No snapshot data for this node at the selected time.")
            else:
                _render_gpu_table(node_table)

            if not selected_metrics:
                st.info("Select metrics above to show trend charts.")
                continue

            node_df = parsed[parsed["node"] == node]
            if node_df.empty:
                st.info("No timeseries data for this node.")
                continue

            cols = st.columns(2)
            chart_slot = 0
            for metric_key in selected_metrics:
                out_col = parsed_cols.get(metric_key)
                spec = _LOG_METRICS.get(metric_key) or {}
                if out_col is None or out_col not in node_df.columns:
                    continue

                long_df = node_df[["timestamp", "gpu_id", out_col]].rename(columns={out_col: "value"})
                long_df = long_df.dropna(subset=["value"])
                long_df = _downsample_per_gpu(long_df, max_points_per_gpu=max_points)

                with cols[chart_slot % 2]:
                    _render_timeseries_chart(
                        long_df,
                        title=str(spec.get("label") or metric_key),
                        tooltip_format=str(spec.get("tooltip_format") or ""),
                        height=int(spec.get("height") or 220),
                    )
                chart_slot += 1

    with st.expander("View raw data", expanded=False):
        st.dataframe(df, use_container_width=True)


def main(*, session_id=None, db_path=None, include_remote=False):
    _ensure_streamlit()

    st.set_page_config(page_title="nvidb web", page_icon="🖥️", layout="wide")
    _apply_app_styles()
    st.title("nvidb web")

    default_view = "Logs" if session_id is not None else "Live"
    if hasattr(st, "segmented_control"):
        view = st.segmented_control(
            "View",
            options=["Live", "Logs"],
            default=default_view,
            key="_nvidb_view_v1",
            format_func=lambda v: "🟢 Live" if v == "Live" else "📜 Logs",
        )
    else:  # pragma: no cover
        default_index = 1 if default_view == "Logs" else 0
        view = st.radio(
            "View",
            options=["Live", "Logs"],
            index=default_index,
            horizontal=True,
            key="_nvidb_view_v1",
            format_func=lambda v: "🟢 Live" if v == "Live" else "📜 Logs",
        )

    if view == "Live":
        show_live_dashboard(include_remote=include_remote)
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
    if include_remote:
        cmd.append("--include-remote")

    print(f"Starting Streamlit on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--include-remote", action="store_true", default=False)
    args = parser.parse_args()
    main(db_path=args.db_path, include_remote=bool(args.include_remote))
