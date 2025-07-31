import pandas as pd
import logging
import time
import threading
from pathlib import Path
from .connection import NviClientPool
from .data_modules import ServerListInfo

LOG_INTERVAL = 5  # seconds

class NVLogger:
    def __init__(self, filename="nvidia_stats.csv", server_list=None):
        self.filename = filename
        self.server_list = server_list
        self.client_pool = None
        self.logging_active = False
        self.logging_thread = None
        
        # columns: timestamp, node, GPU  |     name     |   fan    |   util   | mem_util |   temp   |     rx     |     tx     |       power        | memory[used/total] |      processes 
        self.data = pd.DataFrame(columns=[
            'timestamp', 'node', 'gpu_id', 'name', 'fan_speed', 'util_gpu', 'util_mem', 'temperature', 'rx', 'tx', 'power', 'memory_used', 'memory_total', 'processes'
        ])

    def initialize_client_pool(self):
        """Initialize NviClientPool for data collection"""
        try:
            self.client_pool = NviClientPool(self.server_list)
            logging.info("NviClientPool initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize NviClientPool: {e}")
            return False
    
    def start_logging(self, log_path=None):
        """Start continuous logging in a separate thread"""
        if self.logging_active:
            logging.warning("Logging is already active")
            return False
            
        if log_path:
            self.filename = log_path
            
        if not self.initialize_client_pool():
            return False
            
        self.logging_active = True
        self.logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
        self.logging_thread.start()
        logging.info(f"Started logging to {self.filename} with interval {LOG_INTERVAL} seconds")
        return True
    
    def stop_logging(self):
        """Stop continuous logging"""
        self.logging_active = False
        if self.logging_thread and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=LOG_INTERVAL + 1)
        logging.info("Stopped logging")
    
    def _logging_loop(self):
        """Main logging loop that runs in a separate thread"""
        while self.logging_active:
            try:
                self._collect_and_log_data()
                time.sleep(LOG_INTERVAL)
            except Exception as e:
                logging.error(f"Error during data collection: {e}")
                time.sleep(LOG_INTERVAL)
    
    def _collect_and_log_data(self):
        """Collect data from NviClientPool and log it"""
        if not self.client_pool:
            return
            
        try:
            # Get raw data from each client in the pool
            for client in self.client_pool.pool:
                try:
                    # Get raw GPU info directly from client
                    result = client.get_full_gpu_info()
                    
                    # Handle the tuple return from get_full_gpu_info
                    if isinstance(result, tuple) and len(result) == 2:
                        stats_df, system_info = result
                    else:
                        # Fallback for backward compatibility
                        stats_df = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
                        system_info = {}
                    
                    if not stats_df.empty:
                        self._log_client_data(client, stats_df, system_info)
                        
                except Exception as e:
                    logging.error(f"Error getting data from client {getattr(client, 'description', 'unknown')}: {e}")
                    
        except Exception as e:
            logging.error(f"Error collecting data from client pool: {e}")
    
    def _parse_client_stats(self, client_stat_str):
        """Parse client statistics string - this needs to be adapted based on actual format"""
        # This method is no longer needed as we get raw data directly
        return pd.DataFrame(), {}
    
    def _log_client_data(self, client, stats_df, system_info):
        """Log data for a single client"""
        timestamp = pd.Timestamp.now()
        node_name = getattr(client, 'description', 'unknown')
        
        for idx, row in stats_df.iterrows():
            try:
                # Parse memory information from raw data
                memory_used = row.get('used', 'N/A')
                memory_total = row.get('total', 'N/A')
                
                # Get process information
                processes_element = row.get('processes')
                processes_info = "-"
                if processes_element is not None:
                    try:
                        # Get process summary for this GPU row
                        processes, user_summary = client.get_process_summary(stats_df.iloc[[idx]])
                        if user_summary:
                            processes_info = client.format_user_memory_compact(user_summary)
                    except Exception as e:
                        logging.warning(f"Failed to get process info for GPU {idx}: {e}")
                
                new_entry = {
                    'timestamp': timestamp,
                    'node': node_name,
                    'gpu_id': row.get('gpu_index', idx),
                    'name': row.get('product_name', 'N/A'),
                    'fan_speed': row.get('fan_speed', 'N/A'),
                    'util_gpu': row.get('gpu_util', 'N/A'),
                    'util_mem': row.get('memory_util', 'N/A'),
                    'temperature': row.get('gpu_temp', 'N/A'),
                    'rx': row.get('rx_util', 'N/A'),
                    'tx': row.get('tx_util', 'N/A'),
                    'power': f"{row.get('power_state', 'N/A')} {row.get('power_draw', 'N/A')}/{row.get('current_power_limit', 'N/A')}",
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'processes': processes_info,
                }
                
                # Use concat instead of append (deprecated)
                new_df = pd.DataFrame([new_entry])
                if self.data.empty:
                    self.data = new_df
                else:
                    self.data = pd.concat([self.data, new_df], ignore_index=True)
                
            except Exception as e:
                logging.error(f"Error processing GPU data for {node_name}, GPU {idx}: {e}")
        
        # Save to CSV after each client's data is processed
        self._save_to_csv()
    
    def _save_to_csv(self):
        """Save current data to CSV file"""
        try:
            self.data.to_csv(self.filename, index=False)
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")

    def log(self, node, gpu_id, name, fan_speed, util_gpu, util_mem, temperature, rx, tx, power, memory_used, memory_total, processes):
        """Manual logging method (for backward compatibility)"""
        timestamp = pd.Timestamp.now()
        new_entry = {
            'timestamp': timestamp,
            'node': node,
            'gpu_id': gpu_id,
            'name': name,
            'fan_speed': fan_speed,
            'util_gpu': util_gpu,
            'util_mem': util_mem,
            'temperature': temperature,
            'rx': rx,
            'tx': tx,
            'power': power,
            'memory_used': memory_used,
            'memory_total': memory_total,
            'processes': processes,
        }
        new_df = pd.DataFrame([new_entry])
        if self.data.empty:
            self.data = new_df
        else:
            self.data = pd.concat([self.data, new_df], ignore_index=True)
        self._save_to_csv()


if __name__ == "__main__":
    # 示例：初始化 logger 并开始记录数据
    import sys
    import signal
    from .data_modules import ServerListInfo, ServerInfo
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建 server list（可以为 None 只使用本地 GPU）
    server_list = None  # 使用 None 只监控本地 GPU
    
    # 如果需要监控远程服务器，可以这样配置：
    # server_list = ServerListInfo()
    # server_list.add_server(ServerInfo(
    #     host='remote_host_ip',
    #     port=22,
    #     username='username',
    #     description='Remote GPU Server'
    # ))
    
    # 指定保存路径
    log_file = sys.argv[1] if len(sys.argv) > 1 else "nvidia_stats.csv"
    
    # 创建 logger 实例
    logger = NVLogger(filename=log_file, server_list=server_list)
    
    # 设置优雅退出
    def signal_handler(sig, frame):
        print("\n正在停止日志记录...")
        logger.stop_logging()
        print("日志记录已停止")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 开始日志记录
    print(f"开始记录 GPU 数据到文件: {log_file}")
    print(f"记录间隔: {LOG_INTERVAL} 秒")
    print("按 Ctrl+C 停止记录")
    
    if logger.start_logging():
        try:
            # 保持主线程运行
            while logger.logging_active:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("启动日志记录失败")
        sys.exit(1)