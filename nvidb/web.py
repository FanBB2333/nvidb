"""
Streamlit web interface for nvidb (Live GPU + Log viewer).

Usage:
    nvidb web                 # Live view (local)
    nvidb web --remote        # Live view (local + remote)
    nvidb web 1               # Open session 1 (Logs view)
"""

import argparse
from datetime import datetime
import importlib.util
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
            COUNT(g.id) as record_count
        FROM log_sessions s
        LEFT JOIN gpu_logs g ON s.id = g.session_id
        GROUP BY s.id
        ORDER BY s.id DESC
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


def _trigger_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    if hasattr(st, "experimental_rerun"):  # pragma: no cover
        st.experimental_rerun()


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
            color="auto-inverse",
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
            color="auto-inverse",
        )

    if "mem_util" in column_order and "mem_util" in table.columns:
        table["mem_util"] = pd.to_numeric(table["mem_util"].map(_parse_percent), errors="coerce")
        column_config["mem_util"] = st.column_config.ProgressColumn(
            "mem_util",
            width="small",
            min_value=0,
            max_value=100,
            format="%.0f%%",
            color="auto-inverse",
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

    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)
    auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
    use_fragment = hasattr(st, "fragment")
    refresh_now = st.sidebar.button("Refresh now")

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


def show_logs_dashboard(db_path, session_id=None):
    _ensure_streamlit()

    db_path = _as_path(db_path or get_db_path())
    if db_path is None or not db_path.exists():
        st.error(f"Database not found: {db_path}")
        st.info("Run `nvidb log` first to start logging.")
        return

    sessions_df = load_sessions(db_path)
    if sessions_df.empty:
        st.warning("No log sessions found.")
        return

    if session_id is None:
        session_options = {
            f"Session {row['id']} ({str(row['start_time'])[:19]}) - {row['record_count']} records": row["id"]
            for _, row in sessions_df.iterrows()
        }
        selected = st.sidebar.selectbox("Select Session", list(session_options.keys()))
        session_id = session_options[selected]
    else:
        st.sidebar.info(f"Viewing Session {session_id}")

    df = load_session_logs(db_path, session_id)
    if df.empty:
        st.warning("No logs found for this session.")
        return

    session_info = sessions_df[sessions_df["id"] == session_id].iloc[0]
    st.header(f"Session {session_id}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", int(session_info["record_count"]))
    with col2:
        st.metric("Interval", f"{session_info['interval_seconds']}s")
    with col3:
        st.metric("Status", session_info["status"])
    with col4:
        st.metric("Remote", "Yes" if session_info["include_remote"] else "No")

    timestamps = df["timestamp"].dropna().drop_duplicates().sort_values()
    if timestamps.empty:
        st.warning("No timestamps available in this session.")
        return

    ts_list = list(timestamps)
    ts_idx = st.sidebar.slider("Record index", 0, len(ts_list) - 1, len(ts_list) - 1)
    selected_ts = ts_list[ts_idx]

    st.subheader(f"Snapshot @ {selected_ts}")
    snapshot = df[df["timestamp"] == selected_ts]

    nodes = list(snapshot["node"].dropna().drop_duplicates())
    multi = len(nodes) > 1
    for node in nodes:
        node_df = snapshot[snapshot["node"] == node]
        table = _build_log_snapshot_table(node_df)
        title = f"{node} | {_server_summary(table)}"

        if multi:
            with st.expander(title, expanded=(node == nodes[0])):
                _render_gpu_table(table)
        else:
            st.markdown(f"**{node}**")
            _render_gpu_table(table)

    with st.expander("View raw data"):
        st.dataframe(df, use_container_width=True)


def main(*, session_id=None, db_path=None, include_remote=False):
    _ensure_streamlit()

    st.set_page_config(page_title="nvidb web", page_icon="🖥️", layout="wide")
    st.title("nvidb web")

    if session_id is None:
        view = st.sidebar.radio("View", ["Live", "Logs"])
    else:
        view = "Logs"

    if view == "Live":
        show_live_dashboard(include_remote=include_remote)
    else:
        show_logs_dashboard(db_path=db_path, session_id=session_id)


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

    if session_id is not None:
        cmd.extend(["--session-id", str(session_id)])
    if db_path is not None:
        cmd.extend(["--db-path", str(db_path)])
    if include_remote:
        cmd.append("--include-remote")

    print(f"Starting Streamlit on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--include-remote", action="store_true", default=False)
    args = parser.parse_args()
    main(session_id=args.session_id, db_path=args.db_path, include_remote=bool(args.include_remote))
