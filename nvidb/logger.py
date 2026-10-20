import contextlib
import logging
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from .connection import NVClientPool
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
        
        # Rows not yet written: flushed every tick, and kept for the next
        # attempt when a write fails (disk full, DB locked).
        self.data_buffer = []
        self._buffer_lock = threading.Lock()
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _execute(self, sql, params=(), *, many=False):
        """Run one statement in its own committed transaction; return lastrowid."""
        # The inner `conn` context commits, or rolls back if the statement fails.
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn, conn:
            cursor = conn.executemany(sql, params) if many else conn.execute(sql, params)
            return cursor.lastrowid
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        # Create log_sessions table
        self._execute('''
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
        self._execute('''
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
        logging.info(f"Database initialized at {self.db_path}")
    
    def _start_session(self):
        """Create a new logging session and return its ID."""
        start_time = datetime.now().isoformat()
        include_remote = 1 if self.server_list is not None else 0
        
        session_id = self._execute('''
            INSERT INTO log_sessions (start_time, status, interval_seconds, include_remote)
            VALUES (?, 'running', ?, ?)
        ''', (start_time, self.interval, include_remote))
        
        logging.info(f"Started logging session {session_id}")
        return session_id
    
    def _end_session(self, status='completed'):
        """Mark the current session as ended."""
        if self.session_id is None:
            return
        
        end_time = datetime.now().isoformat()
        self._execute('''
            UPDATE log_sessions SET end_time = ?, status = ? WHERE id = ?
        ''', (end_time, status, self.session_id))
        logging.info(f"Ended logging session {self.session_id} with status: {status}")
    
    def _flush_buffer(self):
        """Write buffered data to database; keep it for a retry if that fails."""
        with self._buffer_lock:
            if not self.data_buffer:
                return
            try:
                self._execute('''
                    INSERT INTO gpu_logs (
                        session_id, timestamp, node, gpu_id, name, fan_speed,
                        util_gpu, util_mem, temperature, rx, tx, power,
                        memory_used, memory_total, processes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', self.data_buffer, many=True)
            except sqlite3.Error as e:
                logging.error(f"Failed to write {len(self.data_buffer)} records (kept for retry): {e}")
                return
            buffer_count = len(self.data_buffer)
            self.data_buffer = []
        logging.debug(f"Flushed {buffer_count} records to database")
    
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
            self._collect_data()
            self._flush_buffer()
            time.sleep(self.interval)
    
    def _collect_data(self):
        """Collect one sample from every client into the buffer."""
        timestamp = datetime.now().isoformat()
        
        for client in self.client_pool.pool:
            try:
                # Advanced (DCGM) metrics live only in system_info, which is
                # not logged, so they are not fetched. The allowlist is the
                # same `gpu_ids` rule the monitor and the queue apply.
                stats_df, _system_info = NVClientPool._apply_gpu_allowlist(
                    client, *client.get_full_gpu_info(include_advanced=False)
                )
                records = self._client_records(client, stats_df, timestamp)
            except Exception as e:
                logging.error(f"Error getting data from client {client.description}: {e}")
                continue
            with self._buffer_lock:
                self.data_buffer.extend(records)
    
    def _client_records(self, client, stats_df, timestamp):
        """Record tuples for one client's GPUs."""
        records = []
        for idx, row in stats_df.iterrows():
            # Get process information
            processes_info = "-"
            try:
                # .loc: an allowlisted frame keeps the host's row labels.
                _processes, user_summary = client.get_process_summary(stats_df.loc[[idx]])
                if user_summary:
                    processes_info = client.format_user_memory_compact(user_summary)
            except Exception as e:
                logging.warning(f"Failed to get process info for GPU {idx}: {e}")
            
            # Create record tuple for bulk insert
            records.append((
                self.session_id,
                timestamp,
                client.description,
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
            ))
        return records


def run_sqlite_logger(server_list=None, interval=5, db_path=None):
    """Run the SQLite logger as a foreground process with signal handling."""
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
    
    if not logger.start_logging():
        print("Failed to start logging")
        sys.exit(1)
    # signal_handler stops the logger, saves the last rows and exits.
    while logger.logging_active:
        time.sleep(1)
