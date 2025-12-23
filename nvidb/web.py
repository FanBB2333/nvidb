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
from pathlib import Path
from typing import Optional

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


def _bar_css(percent: Optional[float]) -> str:
    if percent is None:
        return ""
    try:
        p = float(percent)
    except Exception:
        return ""
    p = max(0.0, min(100.0, p))
    if p >= 80:
        fill = "rgba(255, 75, 75, 0.35)"
        text = "#ff4b4b"
    elif p >= 50:
        fill = "rgba(249, 199, 79, 0.35)"
        text = "#f9c74f"
    elif p >= 5:
        fill = "rgba(67, 170, 139, 0.35)"
        text = "#43aa8b"
    else:
        fill = "transparent"
        text = "inherit"
    return f"background: linear-gradient(90deg, {fill} {p}%, transparent {p}%); color: {text};"


def _style_gpu_table(df: pd.DataFrame):
    if df is None or df.empty:
        return df

    def ratio_percent(value):
        used, total = _parse_mib_pair(value)
        if used is None or total in (None, 0):
            return None
        return (float(used) / float(total)) * 100

    styler = df.style
    styler_map = getattr(styler, "map", None)
    if styler_map is None:  # pandas<2.0 compatibility
        styler_map = getattr(styler, "applymap", None)
    if styler_map is None:
        return df
    if "util" in df.columns:
        styler = styler_map(lambda v: _bar_css(_parse_percent(v)), subset=["util"])
    if "mem_util" in df.columns:
        styler = styler_map(lambda v: _bar_css(_parse_percent(v)), subset=["mem_util"])
    if "memory[used/total]" in df.columns:
        styler = styler_map(lambda v: _bar_css(ratio_percent(v)), subset=["memory[used/total]"])
    return styler


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


def show_live_dashboard(*, include_remote):
    _ensure_streamlit()

    st.header("Live GPU")

    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)
    auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
    if st.sidebar.button("Refresh now"):
        _trigger_rerun()
    _autorefresh(refresh_interval, enabled=auto_refresh, key="_nvidb_autorefresh_live")

    effective_remote = bool(include_remote)
    try:
        pool = _get_pool(include_remote)
    except Exception as e:
        effective_remote = False
        st.error(str(e))
        pool = _get_pool(False)

    if effective_remote:
        st.caption("Remote: enabled")
    else:
        st.caption("Remote: disabled (run `nvidb web --remote` to include remote servers)")

    try:
        _formatted, raw_stats_by_client = pool.get_client_gpus_info(return_raw=True)
    except Exception as e:
        st.error(f"Failed to fetch GPU data: {e}")
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

            def body(table=table, system_info=system_info, user_summary=user_summary):
                if (
                    isinstance(system_info, dict)
                    and any(k in system_info for k in ("driver_version", "cuda_version", "attached_gpus"))
                ):
                    st.caption(
                        f"Driver: {system_info.get('driver_version', 'N/A')} | "
                        f"CUDA: {system_info.get('cuda_version', 'N/A')} | "
                        f"GPUs: {system_info.get('attached_gpus', '0')}"
                    )

                st.dataframe(_style_gpu_table(table), use_container_width=True)

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

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


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
                st.dataframe(_style_gpu_table(table), use_container_width=True)
        else:
            st.markdown(f"**{node}**")
            st.dataframe(_style_gpu_table(table), use_container_width=True)

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
