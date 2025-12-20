import pandas as pd
import logging
import time
import threading
import sys
from datetime import datetime
from pathlib import Path
from .connection import NVClientPool
from .data_modules import ServerListInfo
from . import config


class NVLoggerSQLite:
    """GPU Logger with SQLite3 storage for persistent data collection."""
    
    DEFAULT_INTERVAL = 5  # seconds
    
    def __init__(self, db_path=None, server_list=None, interval=None):
        self.db_path = db_path or config.get_db_path()
        self.interval = interval or self.DEFAULT_INTERVAL
        self.server_list = server_list
        self.client_pool = None
        self.logging_active = False
        self.logging_thread = None
        self.session_id = None
        
        # Buffer for collected data (written to DB on exit)
        self.data_buffer = []
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create log_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS log_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT DEFAULT 'running',
                interval_seconds INTEGER,
                include_remote INTEGER DEFAULT 0
            )
        ''')
        
        # Create gpu_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gpu_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                node TEXT NOT NULL,
                gpu_id INTEGER,
                name TEXT,
                fan_speed TEXT,
                util_gpu TEXT,
                util_mem TEXT,
                temperature TEXT,
                rx TEXT,
                tx TEXT,
                power TEXT,
                memory_used TEXT,
                memory_total TEXT,
                processes TEXT,
                FOREIGN KEY (session_id) REFERENCES log_sessions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info(f"Database initialized at {self.db_path}")
    
    def _start_session(self):
        """Create a new logging session and return its ID."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_time = datetime.now().isoformat()
        include_remote = 1 if self.server_list is not None else 0
        
        cursor.execute('''
            INSERT INTO log_sessions (start_time, status, interval_seconds, include_remote)
            VALUES (?, 'running', ?, ?)
        ''', (start_time, self.interval, include_remote))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logging.info(f"Started logging session {session_id}")
        return session_id
    
    def _end_session(self, status='completed'):
        """Mark the current session as ended."""
        if self.session_id is None:
            return
        
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_time = datetime.now().isoformat()
        cursor.execute('''
            UPDATE log_sessions SET end_time = ?, status = ? WHERE id = ?
        ''', (end_time, status, self.session_id))
        
        conn.commit()
        conn.close()
        logging.info(f"Ended logging session {self.session_id} with status: {status}")
    
    def _flush_buffer(self):
        """Write buffered data to database."""
        if not self.data_buffer:
            return
        
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executemany('''
            INSERT INTO gpu_logs (
                session_id, timestamp, node, gpu_id, name, fan_speed,
                util_gpu, util_mem, temperature, rx, tx, power,
                memory_used, memory_total, processes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', self.data_buffer)
        
        conn.commit()
        conn.close()
        
        buffer_count = len(self.data_buffer)
        self.data_buffer = []
        logging.info(f"Flushed {buffer_count} records to database")
    
    def initialize_client_pool(self):
        """Initialize NVClientPool for data collection."""
        try:
            self.client_pool = NVClientPool(self.server_list)
            logging.info("NVClientPool initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize NVClientPool: {e}")
            return False
    
    def start_logging(self):
        """Start continuous logging in a separate thread."""
        if self.logging_active:
            logging.warning("Logging is already active")
            return False
        
        if not self.initialize_client_pool():
            return False
        
        self.session_id = self._start_session()
        self.logging_active = True
        self.logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
        self.logging_thread.start()
        
        logging.info(f"Started logging to {self.db_path} with interval {self.interval} seconds")
        return True
    
    def stop_logging(self):
        """Stop continuous logging and flush data to database."""
        self.logging_active = False
        
        if self.logging_thread and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=self.interval + 1)
        
        # Flush remaining data and end session
        self._flush_buffer()
        self._end_session('completed')
        
        logging.info("Stopped logging")
    
    def _logging_loop(self):
        """Main logging loop that runs in a separate thread."""
        while self.logging_active:
            try:
                self._collect_data()
                time.sleep(self.interval)
            except Exception as e:
                logging.error(f"Error during data collection: {e}")
                time.sleep(self.interval)
    
    def _collect_data(self):
        """Collect data from NVClientPool and buffer it."""
        if not self.client_pool:
            return
        
        timestamp = datetime.now().isoformat()
        
        try:
            for client in self.client_pool.pool:
                try:
                    result = client.get_full_gpu_info()
                    
                    if isinstance(result, tuple) and len(result) == 2:
                        stats_df, system_info = result
                    else:
                        stats_df = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
                        system_info = {}
                    
                    if not stats_df.empty:
                        self._buffer_client_data(client, stats_df, timestamp)
                        
                except Exception as e:
                    logging.error(f"Error getting data from client {getattr(client, 'description', 'unknown')}: {e}")
                    
        except Exception as e:
            logging.error(f"Error collecting data from client pool: {e}")
    
    def _buffer_client_data(self, client, stats_df, timestamp):
        """Buffer data for a single client."""
        node_name = getattr(client, 'description', 'unknown')
        
        for idx, row in stats_df.iterrows():
            try:
                # Get process information
                processes_info = "-"
                processes_element = row.get('processes')
                if processes_element is not None:
                    try:
                        processes, user_summary = client.get_process_summary(stats_df.iloc[[idx]])
                        if user_summary:
                            processes_info = client.format_user_memory_compact(user_summary)
                    except Exception as e:
                        logging.warning(f"Failed to get process info for GPU {idx}: {e}")
                
                # Create record tuple for bulk insert
                record = (
                    self.session_id,
                    timestamp,
                    node_name,
                    row.get('gpu_index', idx),
                    row.get('product_name', 'N/A'),
                    row.get('fan_speed', 'N/A'),
                    row.get('gpu_util', 'N/A'),
                    row.get('memory_util', 'N/A'),
                    row.get('gpu_temp', 'N/A'),
                    row.get('rx_util', 'N/A'),
                    row.get('tx_util', 'N/A'),
                    f"{row.get('power_state', 'N/A')} {row.get('power_draw', 'N/A')}/{row.get('current_power_limit', 'N/A')}",
                    row.get('used', 'N/A'),
                    row.get('total', 'N/A'),
                    processes_info
                )
                
                self.data_buffer.append(record)
                
            except Exception as e:
                logging.error(f"Error processing GPU data for {node_name}, GPU {idx}: {e}")


def run_sqlite_logger(server_list=None, interval=5, db_path=None):
    """Run the SQLite logger as a foreground process with signal handling."""
    import signal
    
    logger = NVLoggerSQLite(
        db_path=db_path,
        server_list=server_list,
        interval=interval
    )
    
    def signal_handler(sig, frame):
        print("\nStopping logging...")
        logger.stop_logging()
        print("Logging stopped. Data saved to database.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"Starting GPU logging to: {logger.db_path}")
    print(f"Interval: {logger.interval} seconds")
    print(f"Remote servers: {'Yes' if server_list else 'No (local only)'}")
    print("Press Ctrl+C to stop and save data\n")
    
    if logger.start_logging():
        try:
            while logger.logging_active:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("Failed to start logging")
        sys.exit(1)