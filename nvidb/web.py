"""
Streamlit web interface for nvidb GPU monitoring.

Usage:
    nvidb log web           # Live GPU monitoring
    nvidb log web 1         # View session 1 historical data
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
import time

# Import config with fallback for standalone execution
try:
    from nvidb import config
except ImportError:
    from . import config


def get_db_path():
    """Get the default database path."""
    return config.get_db_path()


def load_sessions(db_path):
    """Load all sessions from database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('''
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
    ''', conn)
    conn.close()
    return df


def load_session_logs(db_path, session_id):
    """Load GPU logs for a specific session."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('''
        SELECT 
            timestamp,
            node,
            gpu_id,
            name,
            fan_speed,
            util_gpu,
            util_mem,
            temperature,
            power,
            memory_used,
            memory_total,
            processes
        FROM gpu_logs
        WHERE session_id = ?
        ORDER BY timestamp
    ''', conn, params=(session_id,))
    conn.close()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Parse numeric values
    for col in ['util_gpu', 'util_mem', 'temperature']:
        df[col] = pd.to_numeric(df[col].str.replace('%', '').str.replace(' ', ''), errors='coerce')
    
    # Parse memory values (e.g., "1234 MiB" -> 1234)
    for col in ['memory_used', 'memory_total']:
        df[col] = df[col].apply(parse_memory_value)
    
    return df


def parse_memory_value(val):
    """Parse memory string like '1234 MiB' to float in MB."""
    if pd.isna(val) or val in ('N/A', '-'):
        return None
    try:
        val = str(val).strip()
        if 'GiB' in val or 'GB' in val:
            return float(val.replace('GiB', '').replace('GB', '').strip()) * 1024
        elif 'MiB' in val or 'MB' in val:
            return float(val.replace('MiB', '').replace('MB', '').strip())
        else:
            return float(val)
    except:
        return None


def get_live_gpu_data():
    """Get current GPU data for live monitoring."""
    try:
        # Try absolute import first, then relative
        try:
            from nvidb.connection import NVClientPool
        except ImportError:
            from .connection import NVClientPool
        
        pool = NVClientPool(None)  # Local only
        
        data = []
        timestamp = datetime.now()
        
        for client in pool.pool:
            try:
                result = client.get_full_gpu_info()
                if isinstance(result, tuple) and len(result) == 2:
                    stats_df, system_info = result
                else:
                    stats_df = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
                
                if not stats_df.empty:
                    for idx, row in stats_df.iterrows():
                        data.append({
                            'timestamp': timestamp,
                            'node': getattr(client, 'description', 'unknown'),
                            'gpu_id': row.get('gpu_index', idx),
                            'name': row.get('product_name', 'N/A'),
                            'util_gpu': row.get('gpu_util', 'N/A'),
                            'util_mem': row.get('memory_util', 'N/A'),
                            'temperature': row.get('gpu_temp', 'N/A'),
                            'memory_used': row.get('used', 'N/A'),
                            'memory_total': row.get('total', 'N/A'),
                            'power': f"{row.get('power_state', 'N/A')} {row.get('power_draw', 'N/A')}/{row.get('current_power_limit', 'N/A')}",
                        })
            except Exception as e:
                st.warning(f"Error getting data from client: {e}")
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Failed to get live GPU data: {e}")
        return pd.DataFrame()


def parse_user_stats(df, interval=5):
    """Parse user statistics from processes column."""
    user_time = {}
    user_max_memory = {}
    
    for _, row in df.iterrows():
        processes = row.get('processes', '')
        node = row.get('node', 'unknown')
        
        if not processes or processes in ('-', 'N/A'):
            continue
        
        for part in str(processes).split():
            if ':' in part:
                try:
                    user, mem_str = part.split(':', 1)
                    mem_mb = 0
                    if 'MB' in mem_str.upper():
                        mem_mb = float(mem_str.upper().replace('MB', '').replace('MIB', ''))
                    elif 'GB' in mem_str.upper():
                        mem_mb = float(mem_str.upper().replace('GB', '').replace('GIB', '')) * 1024
                    
                    key = f"{user}@{node}"
                    user_time[key] = user_time.get(key, 0) + interval
                    
                    if key not in user_max_memory or mem_mb > user_max_memory[key]:
                        user_max_memory[key] = mem_mb
                except:
                    continue
    
    return user_time, user_max_memory


def format_duration(seconds):
    """Format seconds to human readable duration."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def main(session_id=None, db_path=None):
    """Main Streamlit application."""
    st.set_page_config(
        page_title="nvidb GPU Monitor",
        page_icon="🖥️",
        layout="wide"
    )
    
    st.title("nvidb GPU Monitor")
    
    db_path = db_path or get_db_path()
    
    # Mode selection
    if session_id is None:
        mode = st.sidebar.radio("Mode", ["Live", "Historical"])
    else:
        mode = "Historical"
        st.sidebar.info(f"Viewing Session {session_id}")
    
    if mode == "Live":
        show_live_dashboard()
    else:
        show_historical_dashboard(db_path, session_id)


def show_live_dashboard():
    """Display live GPU monitoring dashboard."""
    st.header("Live GPU Monitoring")
    
    # Auto-refresh control
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)
    placeholder = st.empty()
    
    while True:
        with placeholder.container():
            df = get_live_gpu_data()
            
            if df.empty:
                st.warning("No GPU data available")
            else:
                # Display GPU cards
                cols = st.columns(min(len(df), 4))
                for i, (_, row) in enumerate(df.iterrows()):
                    with cols[i % len(cols)]:
                        st.metric(
                            label=f"GPU {row['gpu_id']} - {row['name'][:20]}",
                            value=f"{row.get('util_gpu', 'N/A')}%",
                            delta=f"Mem: {row.get('memory_used', 'N/A')}"
                        )
                
                # Display detailed table
                st.subheader("GPU Details")
                display_cols = ['node', 'gpu_id', 'name', 'util_gpu', 'util_mem', 
                              'temperature', 'memory_used', 'memory_total']
                st.dataframe(df[display_cols], use_container_width=True)
            
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        time.sleep(refresh_interval)
        st.rerun()


def show_historical_dashboard(db_path, session_id=None):
    """Display historical log dashboard."""
    
    if not Path(db_path).exists():
        st.error(f"Database not found: {db_path}")
        st.info("Run 'nvidb log' first to start logging.")
        return
    
    # Load sessions
    sessions_df = load_sessions(db_path)
    
    if sessions_df.empty:
        st.warning("No log sessions found.")
        return
    
    # Session selector
    if session_id is None:
        session_options = {
            f"Session {row['id']} ({row['start_time'][:19]}) - {row['record_count']} records": row['id']
            for _, row in sessions_df.iterrows()
        }
        selected = st.sidebar.selectbox("Select Session", list(session_options.keys()))
        session_id = session_options[selected]
    
    # Load session data
    df = load_session_logs(db_path, session_id)
    
    if df.empty:
        st.warning("No logs found for this session.")
        return
    
    # Session info
    session_info = sessions_df[sessions_df['id'] == session_id].iloc[0]
    
    st.header(f"Session {session_id}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", session_info['record_count'])
    with col2:
        st.metric("Interval", f"{session_info['interval_seconds']}s")
    with col3:
        st.metric("Status", session_info['status'])
    with col4:
        st.metric("Remote", "Yes" if session_info['include_remote'] else "No")
    
    st.caption(f"Start: {session_info['start_time']} | End: {session_info['end_time'] or 'Running'}")
    
    # GPU Utilization Chart
    st.subheader("GPU Utilization Over Time")
    
    # Pivot data for charting
    nodes = df['node'].unique()
    gpu_ids = df['gpu_id'].unique()
    
    for node in nodes:
        node_df = df[df['node'] == node]
        
        st.markdown(f"**{node}**")
        
        # Create chart data
        chart_data = node_df.pivot_table(
            index='timestamp', 
            columns='gpu_id', 
            values='util_gpu',
            aggfunc='first'
        )
        
        if not chart_data.empty:
            chart_data.columns = [f"GPU {col}" for col in chart_data.columns]
            st.line_chart(chart_data)
    
    # Memory Usage Chart
    st.subheader("Memory Usage Over Time")
    
    for node in nodes:
        node_df = df[df['node'] == node]
        
        st.markdown(f"**{node}**")
        
        chart_data = node_df.pivot_table(
            index='timestamp', 
            columns='gpu_id', 
            values='memory_used',
            aggfunc='first'
        )
        
        if not chart_data.empty:
            chart_data.columns = [f"GPU {col} (MB)" for col in chart_data.columns]
            st.line_chart(chart_data)
    
    # User Statistics
    st.subheader("User Statistics")
    
    user_time, user_max_memory = parse_user_stats(df, session_info['interval_seconds'])
    
    if user_time:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Users by GPU Time**")
            sorted_time = sorted(user_time.items(), key=lambda x: x[1], reverse=True)[:10]
            time_df = pd.DataFrame(sorted_time, columns=['User', 'Time (s)'])
            time_df['Duration'] = time_df['Time (s)'].apply(format_duration)
            st.dataframe(time_df[['User', 'Duration']], use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Top Users by Max Memory**")
            sorted_mem = sorted(user_max_memory.items(), key=lambda x: x[1], reverse=True)[:10]
            mem_df = pd.DataFrame(sorted_mem, columns=['User', 'Memory (MB)'])
            mem_df['Memory'] = mem_df['Memory (MB)'].apply(
                lambda x: f"{x/1024:.1f} GB" if x >= 1024 else f"{x:.0f} MB"
            )
            st.dataframe(mem_df[['User', 'Memory']], use_container_width=True, hide_index=True)
    else:
        st.info("No user process data available for this session.")
    
    # Raw Data
    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True)


def run_streamlit_app(session_id=None, db_path=None, port=8501):
    """Launch the Streamlit app."""
    import sys
    import subprocess
    
    # Build command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        __file__,
        "--server.port", str(port),
        "--server.headless", "true",
        "--"
    ]
    
    if session_id is not None:
        cmd.extend(["--session-id", str(session_id)])
    if db_path is not None:
        cmd.extend(["--db-path", str(db_path)])
    
    print(f"Starting Streamlit on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    
    subprocess.run(cmd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()
    
    main(session_id=args.session_id, db_path=args.db_path)
