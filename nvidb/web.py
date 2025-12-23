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


def _build_live_table(client, stats):
    try:
        from nvidb.utils import extract_numbers, extract_value_and_unit, format_bandwidth
    except ImportError:  # pragma: no cover
        from .utils import extract_numbers, extract_value_and_unit, format_bandwidth

    if stats is None or stats.empty:
        return pd.DataFrame()

    table = stats.copy()

    rx_list = []
    tx_list = []
    for _, row in table.iterrows():
        rx_val, rx_unit = extract_value_and_unit(row.get("rx_util", "0"))
        tx_val, tx_unit = extract_value_and_unit(row.get("tx_util", "0"))
        rx_list.append(format_bandwidth(rx_val, rx_unit))
        tx_list.append(format_bandwidth(tx_val, tx_unit))

    table["rx"] = rx_list
    table["tx"] = tx_list
    table["power"] = [
        f"{row.get('power_state', 'N/A')} "
        f"{'/'.join(extract_numbers(str(row.get('power_draw', 'N/A'))))}/"
        f"{'/'.join(extract_numbers(str(row.get('current_power_limit', 'N/A'))))}"
        for _, row in table.iterrows()
    ]
    table["memory[used/total]"] = [
        f"{'/'.join(extract_numbers(str(row.get('used', 'N/A'))))}/"
        f"{'/'.join(extract_numbers(str(row.get('total', 'N/A'))))}"
        for _, row in table.iterrows()
    ]

    process_list = []
    try:
        all_processes, _ = client.get_process_summary(table)
        per_gpu_user_summary = {}
        for proc in all_processes:
            gpu_idx = proc.get("gpu_index")
            username = proc.get("username")
            used_memory = proc.get("used_memory", "0 MiB")
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

        for _, row in table.iterrows():
            gpu_idx = row.get("gpu_index")
            user_summary = per_gpu_user_summary.get(gpu_idx, {})
            if user_summary:
                process_list.append(client.format_user_memory_compact(user_summary))
            else:
                process_list.append("-")
    except Exception:
        process_list = ["-" for _ in range(len(table))]

    table["processes"] = process_list

    table = table.rename(
        columns={
            "product_name": "name",
            "gpu_temp": "temp",
            "fan_speed": "fan",
            "memory_util": "mem_util",
            "gpu_util": "util",
            "gpu_index": "GPU",
        }
    )
    if "name" in table.columns:
        table["name"] = table["name"].map(_strip_gpu_name)

    columns_to_drop = [
        "product_architecture",
        "rx_util",
        "tx_util",
        "power_state",
        "power_draw",
        "current_power_limit",
        "used",
        "total",
        "free",
    ]
    table = table.drop(columns=[c for c in columns_to_drop if c in table.columns])

    desired_order = [
        "GPU",
        "name",
        "fan",
        "util",
        "mem_util",
        "temp",
        "rx",
        "tx",
        "power",
        "memory[used/total]",
        "processes",
    ]
    ordered = [c for c in desired_order if c in table.columns]
    tail = [c for c in table.columns if c not in ordered]
    return table[ordered + tail]


def show_live_dashboard(*, include_remote):
    _ensure_streamlit()

    st.header("Live GPU")

    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)
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

    results = []
    for client in pool.pool:
        result = client.get_full_gpu_info()
        if isinstance(result, tuple) and len(result) == 2:
            stats_df, system_info = result
        else:
            stats_df = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
            system_info = {}

        results.append(
            {
                "client": client,
                "description": getattr(client, "description", "unknown"),
                "system_info": system_info,
                "stats": stats_df,
            }
        )

    multi = len(results) > 1
    for idx, item in enumerate(results):
        description = item["description"]
        system_info = item["system_info"] or {}
        stats_df = item["stats"]

        if isinstance(system_info, dict) and system_info.get("error"):
            body = lambda: st.error(system_info.get("error", "Unknown error"))
            title = f"[{idx + 1}] {description} | Error"
        else:
            table = _build_live_table(item["client"], stats_df)
            if table.empty:
                title = f"[{idx + 1}] {description} | No GPUs"
                body = lambda: st.json(
                    {
                        "nvidia": system_info or {},
                        "system": _get_local_system_info()
                        if description.startswith("Local Machine")
                        else {"os": item["client"].get_os_info()},
                    }
                )
            else:
                title = f"[{idx + 1}] {description} | {_server_summary(table)}"

                def body(table=table, system_info=system_info):
                    if system_info:
                        st.caption(
                            f"Driver: {system_info.get('driver_version', 'N/A')} | "
                            f"CUDA: {system_info.get('cuda_version', 'N/A')} | "
                            f"GPUs: {system_info.get('attached_gpus', '0')}"
                        )
                    st.dataframe(table, use_container_width=True)

        if multi:
            with st.expander(title, expanded=(idx == 0)):
                body()
        else:
            st.subheader(description)
            body()

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(refresh_interval)
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover
        st.experimental_rerun()


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
                st.dataframe(table, use_container_width=True)
        else:
            st.markdown(f"**{node}**")
            st.dataframe(table, use_container_width=True)

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
