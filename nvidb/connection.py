from typing import Literal
from blessed import Terminal
import logging
import sys
import os
import subprocess
import getpass
import json
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from io import StringIO
import threading

import pynvml
import paramiko
from paramiko import AuthenticationException
from paramiko.client import SSHClient, AutoAddPolicy
from paramiko.ssh_exception import NoValidConnectionsError
import pandas as pd
from termcolor import colored, cprint
from .data_modules import ServerInfo, ServerListInfo
from .utils import xml_to_dict, num_from_str, units_from_str, extract_numbers, extract_value_and_unit, format_bandwidth, get_utilization_color, get_memory_color, get_memory_ratio_color


def nvidbInit():
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print("GPU", i, "Name:", name)
        print("GPU", i, "Temperature:", temperature, "C")


class BaseClient(ABC):
    """Base class for both RemoteClient and LocalClient with common functionality"""
    
    def __init__(self):
        self.host = None
        self.port = None
        self.username = None
        self.description = None
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the client (remote or local)"""
        pass
    
    @abstractmethod
    def execute_command(self, command: str) -> str:
        """Execute a command and return the output"""
        pass

    def _chunked(self, items, chunk_size: int):
        for i in range(0, len(items), chunk_size):
            yield items[i : i + chunk_size]

    def get_pid_user_map(self, pids, chunk_size: int = 128):
        """Batch query pid -> username mapping via a single ps call (or a few chunked calls)."""
        if not pids:
            return {}

        # Clean and de-duplicate PIDs (preserve order)
        seen = set()
        unique_pids = []
        for pid in pids:
            pid_str = str(pid).strip()
            if not pid_str or pid_str == "N/A":
                continue
            if pid_str in seen:
                continue
            seen.add(pid_str)
            unique_pids.append(pid_str)

        if not unique_pids:
            return {}

        pid_to_user = {}
        for pid_chunk in self._chunked(unique_pids, chunk_size):
            pid_list = ",".join(pid_chunk)
            output = self.execute_command(f"ps -o pid=,user= -p {pid_list}")
            if not isinstance(output, str) or not output.strip():
                continue
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                pid_str, username = parts[0].strip(), parts[1].strip()
                if pid_str and username:
                    pid_to_user[pid_str] = username

        return pid_to_user
    
    def test(self):
        """Test connection with ls command"""
        try:
            result = self.execute_command('ls')
            logging.info(msg=f"Test result: {result}")
            return result
        except Exception as e:
            logging.error(msg=f"Test command failed: {e}")
            return ""
    
    def get_os_info(self) -> str:
        """Get operating system information"""
        try:
            result = self.execute_command('uname -a')
            return result
        except Exception as e:
            logging.error(msg=f"Failed to get OS info: {e}")
            return ""
    
    def get_gpu_stats(self, command='nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv') -> pd.DataFrame:
        """Get GPU statistics in CSV format"""
        try:
            result = self.execute_command(command)
            stats = pd.read_csv(StringIO(result), header=0)
            return stats
        except Exception as e:
            logging.error(msg=f"Failed to get GPU stats: {e}")
            return pd.DataFrame()
    
    def get_full_gpu_info(self):
        """Get complete GPU information from nvidia-smi XML output"""
        def safe_get_text(element, path, default="N/A"):
            """Safely get text from XML element, return default if not found"""
            if element is None:
                return default
            found = element.find(path)
            return found.text if found is not None else default
        
        try:
            result = self.execute_command('nvidia-smi -q -x')
            root = ET.fromstring(result)
            driver_version = safe_get_text(root, 'driver_version', 'N/A')
            cuda_version = safe_get_text(root, 'cuda_version', 'N/A')
            attached_gpus = safe_get_text(root, 'attached_gpus', '0')
            gpus = root.findall('gpu')
            stats = []
            
            for gpu_index, gpu in enumerate(gpus):
                # Safely extract all values with default fallbacks
                minor_number = safe_get_text(gpu, 'minor_number', 'N/A')
                product_name = safe_get_text(gpu, 'product_name', 'Unknown GPU')
                product_architecture = safe_get_text(gpu, 'product_architecture', 'N/A')
                
                pci = gpu.find('pci')
                tx_util = safe_get_text(pci, 'tx_util', 'N/A')
                rx_util = safe_get_text(pci, 'rx_util', 'N/A')
                fan_speed = safe_get_text(gpu, 'fan_speed', 'N/A')
                
                fb_memory_usage = gpu.find('fb_memory_usage')
                total = safe_get_text(fb_memory_usage, 'total', 'N/A')
                used = safe_get_text(fb_memory_usage, 'used', 'N/A')
                free = safe_get_text(fb_memory_usage, 'free', 'N/A')
                
                utilization = gpu.find('utilization')
                gpu_util = safe_get_text(utilization, 'gpu_util', 'N/A')
                memory_util = safe_get_text(utilization, 'memory_util', 'N/A')
                
                temperature = gpu.find('temperature')
                gpu_temp = safe_get_text(temperature, 'gpu_temp', 'N/A')

                gpu_power_readings = gpu.find('gpu_power_readings')
                power_state = safe_get_text(gpu_power_readings, 'power_state', 'N/A')
                
                # Try to get power_draw, if not available, try instant_power_draw as fallback
                power_draw = safe_get_text(gpu_power_readings, 'power_draw', None)
                if power_draw is None or power_draw == 'N/A':
                    power_draw = safe_get_text(gpu_power_readings, 'instant_power_draw', 'N/A')
                
                current_power_limit = safe_get_text(gpu_power_readings, 'current_power_limit', 'N/A')
                
                processes = gpu.find('processes')
                
                stats.append({
                    'gpu_index': gpu_index,
                    'product_name': product_name, 
                    'product_architecture': product_architecture, 
                    'tx_util': tx_util, 
                    'rx_util': rx_util, 
                    'fan_speed': fan_speed, 
                    'total': total, 
                    'used': used, 
                    'free': free, 
                    'gpu_util': gpu_util, 
                    'memory_util': memory_util, 
                    'gpu_temp': gpu_temp, 
                    'power_state': power_state, 
                    'power_draw': power_draw, 
                    'current_power_limit': current_power_limit,
                    'processes': processes
                })
            
            stats = pd.DataFrame(stats)
            
            # Return both stats and system info
            system_info = {
                'driver_version': driver_version,
                'cuda_version': cuda_version,
                'attached_gpus': attached_gpus
            }
            return stats, system_info
            
        except Exception as e:
            logging.error(msg=f"Failed to get full GPU info: {e}")
            return pd.DataFrame(), {}
    
    def get_client(self):
        """Return the client object"""
        return self
    
    def get_process_summary(self, gpu_stats=None):
        """Get detailed GPU processes information with user summary"""
        def safe_get_text(element, path, default="N/A"):
            """Safely get text from XML element, return default if not found"""
            if element is None:
                return default
            found = element.find(path)
            return found.text if found is not None else default
        
        try:
            # Use provided gpu_stats if available, otherwise get fresh data
            if gpu_stats is None:
                stats, _ = self.get_full_gpu_info()
            else:
                stats = gpu_stats
            
            process_entries = []
            pids = []

            # First pass: extract processes and collect all pids
            for idx, row in stats.iterrows():
                processes_element = row.get("processes")
                if processes_element is None or not hasattr(processes_element, "findall"):
                    continue
                for process_info in processes_element.findall("process_info"):
                    pid = safe_get_text(process_info, "pid", "N/A")
                    process_name = safe_get_text(process_info, "process_name", "N/A")
                    used_memory = safe_get_text(process_info, "used_memory", "0 MiB")
                    gpu_instance_id = safe_get_text(process_info, "gpu_instance_id", "N/A")
                    compute_instance_id = safe_get_text(process_info, "compute_instance_id", "N/A")
                    process_type = safe_get_text(process_info, "type", "N/A")

                    # Extract memory value in MiB
                    memory_value = 0
                    if used_memory != "N/A":
                        try:
                            memory_value = int(str(used_memory).replace("MiB", "").strip())
                        except Exception:
                            memory_value = 0

                    pid_str = str(pid).strip()
                    if pid_str and pid_str != "N/A":
                        pids.append(pid_str)

                    process_entries.append(
                        {
                            "gpu_instance_id": gpu_instance_id,
                            "compute_instance_id": compute_instance_id,
                            "pid": pid,
                            "type": process_type,
                            "process_name": process_name,
                            "used_memory": used_memory,
                            "gpu_index": idx,
                            "_memory_value": memory_value,
                        }
                    )

            pid_to_user = self.get_pid_user_map(pids)

            all_processes = []
            user_memory_summary = {}
            for entry in process_entries:
                pid_str = str(entry.get("pid", "")).strip()
                username = pid_to_user.get(pid_str, "N/A") if pid_str and pid_str != "N/A" else "N/A"

                all_processes.append(
                    {
                        "gpu_instance_id": entry.get("gpu_instance_id", "N/A"),
                        "compute_instance_id": entry.get("compute_instance_id", "N/A"),
                        "pid": entry.get("pid", "N/A"),
                        "type": entry.get("type", "N/A"),
                        "process_name": entry.get("process_name", "N/A"),
                        "used_memory": entry.get("used_memory", "0 MiB"),
                        "username": username,
                        "gpu_index": entry.get("gpu_index"),
                    }
                )

                memory_value = entry.get("_memory_value", 0) or 0
                if username != "N/A" and memory_value > 0:
                    user_memory_summary[username] = user_memory_summary.get(username, 0) + memory_value
            
            return all_processes, user_memory_summary
            
        except Exception as e:
            logging.error(msg=f"Failed to get process summary: {e}")
            return [], {}
    
    def format_process_summary_xml(self):
        """Format process summary as XML similar to nvidia-smi output"""
        processes, user_summary = self.get_process_summary()
        
        if not processes:
            return ""
        
        xml_output = []
        xml_output.append("        <processes>")
        
        for process in processes:
            xml_output.append("            <process_info>")
            xml_output.append(f"                <gpu_instance_id>{process['gpu_instance_id']}</gpu_instance_id>")
            xml_output.append(f"                <compute_instance_id>{process['compute_instance_id']}</compute_instance_id>")
            xml_output.append(f"                <pid>{process['pid']}</pid>")
            xml_output.append(f"                <type>{process['type']}</type>")
            xml_output.append(f"                <process_name>{process['process_name']}</process_name>")
            xml_output.append(f"                <used_memory>{process['used_memory']}</used_memory>")
            xml_output.append(f"                <username>{process['username']}</username>")
            xml_output.append("            </process_info>")
        
        xml_output.append("        </processes>")
        
        # Add user memory summary
        if user_summary:
            xml_output.append("        <user_memory_summary>")
            for username, total_memory in user_summary.items():
                xml_output.append("            <user_info>")
                xml_output.append(f"                <username>{username}</username>")
                xml_output.append(f"                <total_memory>{total_memory} MiB</total_memory>")
                xml_output.append("            </user_info>")
            xml_output.append("        </user_memory_summary>")
        
        return "\n".join(xml_output)
    
    def format_user_memory_compact(self, user_summary):
        """Format user memory summary in compact format like 'qbs(23082M) gdm(4M)' sorted by memory usage desc"""
        if not user_summary:
            return ""
        
        formatted_users = []
        # Sort by memory usage in descending order
        for username, total_memory in sorted(user_summary.items(), key=lambda x: x[1], reverse=True):
            formatted_users.append(f"{username}({total_memory}M)")
        
        return " ".join(formatted_users)


class RemoteClient(BaseClient):
    def __init__(self, server: ServerInfo):
        super().__init__()
        self.host = server.host
        self.port = server.port
        self.username = server.username
        self.auth = server.auth
        self.description = server.description
        self.password = server.password
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.load_system_host_keys()
    
    def __del__(self):
        self.client.close()
        logging.info(msg=f"Connection to {self.host}:{self.port} closed.")

    def connect(self) -> bool:
        print(f"Connecting to {self.host}:{self.port} as {self.username}")
        # catch the OSError exception when the host is not reachable
        try:
            if "auto" == self.auth:
                try:
                    if self.password is not None:
                        self.client.connect(hostname=self.host, port=self.port, username=self.username, password=self.password)
                    else:
                        self.client.connect(hostname=self.host, port=self.port, username=self.username)
                    logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username}")
                    return True
                except AuthenticationException as e:
                    logging.error(msg=f"Authentication failed on {self.description}")
                    try:
                        # prompt to input password
                        password = getpass.getpass(prompt=f'Enter password for {self.username}@{self.host}:{self.port} -> ')
                        self.client.connect(hostname=self.host, port=self.port, username=self.username, password=password)
                    except AuthenticationException as e:
                        logging.error(msg=f"Password authentication failed on {self.description}, exiting...")
                        sys.exit(1)
                except NoValidConnectionsError as e:
                    logging.error(msg=f"Connection failed: {e}")
                    sys.exit(1)
            else:
                logging.error(msg=f"Unsupported authentication method: {self.auth}, please use 'auto' strategy.")
        except OSError as e:
            logging.error(msg=f"Connection failed: {e}")
            return False
    
    def execute_command(self, command: str) -> str:
        """Execute command on remote server"""
        stdin, stdout, stderr = self.client.exec_command(command=command)
        result = stdout.read().decode()
        return result
    
    def get_client(self) -> SSHClient:
        return self.client


class LocalClient(BaseClient):
    def __init__(self):
        super().__init__()
        self.host = "localhost"
        self.port = "local"
        self.username = getpass.getuser()
        self.description = f"Local Machine ({self.username}@{self.host})"
        
    def connect(self) -> bool:
        """Local connection is always successful"""
        logging.info(msg=f"Connected to local machine as {self.username}")
        print(f"Connected to local machine as {self.username}")
        return True
    
    def execute_command(self, command: str) -> str:
        """Execute local command"""
        try:
            # Use shell=True to support pipes and complex commands
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logging.error(msg=f"Command '{command}' execution failed with return code {result.returncode}")
                return "N/A"
            return result.stdout
        except subprocess.CalledProcessError as e:
            logging.error(msg=f"Command '{command}' execution failed: {e}")
            return f"Error: {e.stderr}" if e.stderr else f"Command failed with return code {e.returncode}"
        except Exception as e:
            logging.error(msg=f"Unexpected error executing command: {e}")
            return f"Unexpected error: {str(e)}"


class NVClientPool:
    def __init__(self, server_list: ServerListInfo):
        self.pool = [LocalClient()]
        if server_list is not None:
            self.pool += [RemoteClient(server) for server in server_list]
        logging.info(msg=f"Initialized pool with {len(self.pool)} clients.")
        self.connect_all()
        self.term = Terminal()
        self.quit_flag = threading.Event()  # Exit flag for inter-thread communication
        # Collapsible display state - only first server expanded by default
        self.expanded_servers = {0}  # Only first server expanded by default
        self.selected_server = 0  # Currently selected server for navigation
        self.refresh_needed = threading.Event()  # Flag to trigger immediate refresh after key press
        self.ui_only_refresh = False  # Flag to indicate UI-only refresh (no data fetch)
        self.cached_stats = None  # Cached GPU stats data
        self.cached_raw_stats = {}  # Cached raw stats per client index
        self._cache_lock = threading.Lock()
        self._last_update_time = None
        self._last_fetch_duration = None
        self._last_fetch_error = None
    
    def connect_all(self):
        self.pool = [client for client in self.pool if client.connect()]

    def test(self):
        pass
    
    def execute_command(self, command):
        for idx, client in enumerate(self.pool):
            # cprint(f"Output on {client.description}", 'yellow')
            # logging.info(msg=f"Executing command on {client.description}")
            result = client.execute_command(command)
            logging.info(msg=colored(f"{client.description}", 'yellow'))
            print(colored(f"{client.description}", 'yellow'))
            print(result)
    
    def execute_command_parse(self, command, type: Literal['csv', 'xml', 'json']='xml'):
        for idx, client in enumerate(self.pool):
            result = client.execute_command(command)
            logging.info(msg=colored(f"{client.description}", 'yellow'))
            # if result is the instance of dict
            if isinstance(result, dict):
                pass
            elif isinstance(result, str):
                if 'xml' == type:
                    result = xml_to_dict(result)
                elif 'csv' == type:
                    result = pd.read_csv(filepath_or_buffer=result, header=0)
                elif 'json' == type:
                    result = json.loads(result)
                else:
                    logging.error(msg=f"Unsupported type: {type}")
            else:
                logging.error(msg=f"Unsupported result type: {type(result)}")
        return result
    
    def get_client_gpus_info(self, return_raw: bool = False):
        # Set pandas display options for terminal output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)  # Auto-fill terminal width
        pd.set_option('display.colheader_justify', 'center')  # Center column headers
        
        stats_str = []
        raw_stats_by_client = {}
        for idx, client in enumerate(self.pool):
            result = client.get_full_gpu_info()
            
            # Handle the tuple return from get_full_gpu_info
            if isinstance(result, tuple) and len(result) == 2:
                stats, system_info = result
            else:
                # Fallback for backward compatibility
                stats = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
                system_info = {}
            
            # Cache raw stats for summary display (avoids redundant SSH calls)
            raw_stats_by_client[idx] = (stats.copy() if not stats.empty else stats, system_info)
            
            # Skip processing if stats is empty
            if stats.empty:
                stats_str.append((stats, system_info))
                continue
            
            # Optimize rx/tx display - split into two columns
            rx_list = []
            tx_list = []
            for _, row in stats.iterrows():
                rx_val, rx_unit = extract_value_and_unit(row['rx_util'])
                tx_val, tx_unit = extract_value_and_unit(row['tx_util'])
                
                rx_formatted = format_bandwidth(rx_val, rx_unit)
                tx_formatted = format_bandwidth(tx_val, tx_unit)
                
                # Ensure formatted strings don't exceed column width limit
                rx_list.append(rx_formatted[:11] if len(rx_formatted) > 11 else rx_formatted)
                tx_list.append(tx_formatted[:11] if len(tx_formatted) > 11 else tx_formatted)
            
            stats['rx'] = rx_list
            stats['tx'] = tx_list
            stats['power'] = [f"{row['power_state']} {'/'.join(extract_numbers(row['power_draw']))}/{'/'.join(extract_numbers(row['current_power_limit']))}" for _, row in stats.iterrows()]
            stats['memory[used/total]'] = [f"{'/'.join(extract_numbers(row['used']))}/{'/'.join(extract_numbers(row['total']))}" for _, row in stats.iterrows()]

            # Add process information as a column (batch pid->user lookup to avoid per-process SSH calls)
            process_list = []
            try:
                all_processes, _ = client.get_process_summary(stats)

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

                for gpu_idx, _row in stats.iterrows():
                    user_summary = per_gpu_user_summary.get(gpu_idx, {})
                    if user_summary:
                        process_list.append(client.format_user_memory_compact(user_summary))
                    else:
                        process_list.append("-")
            except Exception as e:
                logging.warning(f"Failed to get process info: {e}")
                process_list = ["-" for _ in range(len(stats))]
            
            stats['processes'] = process_list

            # rename columns: product_name -> name, gpu_temp -> temp, fan_speed -> fan, memory_util -> mem_util, gpu_util -> util, gpu_index -> GPU
            stats = stats.rename(columns={'product_name': 'name', 'gpu_temp': 'temp', 'fan_speed': 'fan', 'memory_util': 'mem_util', 'gpu_util': 'util', 'gpu_index': 'GPU'})
            
            # replace the NVIDIA/GeForce with "" in name column
            stats['name'] = stats['name'].str.replace('NVIDIA', '').str.replace('GeForce', '').str.strip()
            
            # remove rows: product_architecture, rx_util, tx_util, power_state, power_draw, current_power_limit, used, total, free
            # but keep the new processes column
            columns_to_drop = ['product_architecture', 'rx_util', 'tx_util', 'power_state', 'power_draw', 'current_power_limit', 'used', 'total', 'free']
            # Only drop columns that exist in the DataFrame
            columns_to_drop = [col for col in columns_to_drop if col in stats.columns]
            stats = stats.drop(columns=columns_to_drop)
            
            # Reorder columns to put processes at the end
            if 'processes' in stats.columns:
                other_columns = [col for col in stats.columns if col != 'processes']
                stats = stats[other_columns + ['processes']]

            stats_str.append((stats, system_info))
        # reformat the str into a single string with fixed width formatting
        formatted_stats = []
        for client, (stats, system_info) in zip(self.pool, stats_str):
            # Create formatted table display
            formatted_table = self._format_fixed_width_table(stats)
            
            # Create system info header
            system_info_header = ""
            if system_info:
                system_info_header = f"Driver: {system_info.get('driver_version', 'N/A')} | CUDA: {system_info.get('cuda_version', 'N/A')} | GPUs: {system_info.get('attached_gpus', '0')}\n"
            
            formatted_stats.append(f"\n{colored(client.description, 'yellow')}\n{system_info_header}{formatted_table}")
        if return_raw:
            return formatted_stats, raw_stats_by_client
        return formatted_stats
    
    def _format_fixed_width_table(self, df):
        """Format fixed-width table display"""
        if df.empty:
            return "No GPU data available"
        
        # Get terminal width
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 120  # Default width
        
        # Define minimum width for each column
        min_widths = {
            'GPU': 4,
            'name': 12,
            'temp': 8,
            'fan': 8,
            'util': 8,
            'mem_util': 8,
            'rx': 10,
            'tx': 10,
            'power': 18,
            'memory[used/total]': 16,
            'processes': 20  # Add width for processes column
        }

        # Compute content width for each column
        content_widths = {}
        for col in df.columns:
            max_len = len(str(col))
            for value in df[col]:
                value_len = len(str(value))
                if value_len > max_len:
                    max_len = value_len
            content_widths[col] = max_len

        # Calculate widths based on terminal space
        separator_width = 3
        total_separator_width = separator_width * (len(df.columns) - 1)
        available_width = terminal_width - total_separator_width

        min_widths_for_cols = {}
        desired_widths = {}
        for col in df.columns:
            min_width = min_widths.get(col, 12)
            min_width = max(min_width, len(str(col)))
            min_widths_for_cols[col] = min_width
            desired_widths[col] = max(min_width, content_widths.get(col, min_width))

        min_total = sum(min_widths_for_cols.values())
        desired_total = sum(desired_widths.values())

        if available_width <= min_total:
            column_widths = min_widths_for_cols
        elif available_width >= desired_total:
            column_widths = desired_widths
        else:
            column_widths = dict(min_widths_for_cols)
            extra = available_width - min_total
            priority = [
                'name',
                'processes',
                'power',
                'memory[used/total]',
                'rx',
                'tx',
                'temp',
                'fan',
                'util',
                'mem_util',
                'GPU',
            ]
            for col in [c for c in priority if c in df.columns]:
                if extra <= 0:
                    break
                need = desired_widths[col] - column_widths[col]
                if need > 0:
                    add = min(extra, need)
                    column_widths[col] += add
                    extra -= add
            if extra > 0:
                for col in df.columns:
                    if extra <= 0:
                        break
                    if col in priority:
                        continue
                    need = desired_widths[col] - column_widths[col]
                    if need > 0:
                        add = min(extra, need)
                        column_widths[col] += add
                        extra -= add
        
        # Format table header
        header_parts = []
        for col in df.columns:
            width = column_widths.get(col, 12)
            header_parts.append(f"{col:^{width}}")
        header = " | ".join(header_parts)
        
        # Format separator line
        separator_parts = []
        for col in df.columns:
            width = column_widths.get(col, 12)
            separator_parts.append("-" * width)
        separator = "-+-".join(separator_parts)
        
        # Format data rows
        data_lines = []
        for _, row in df.iterrows():
            row_parts = []
            for col in df.columns:
                width = column_widths.get(col, 12)
                value = str(row[col])
                # Truncate long values and add ellipsis
                if len(value) > width:
                    if col == 'name' and width > 2:
                        value = ".." + value[-(width - 2):]
                    elif width > 2:
                        value = value[:width - 2] + ".."
                    else:
                        value = value[:width]
                
                # Add color formatting for different columns
                if col == 'util' and value != 'N/A':
                    # GPU utilization coloring
                    color = get_utilization_color(value)
                    if color:
                        value = colored(f"{value:^{width}}", color)
                    else:
                        value = f"{value:^{width}}"
                elif col == 'mem' and value != 'N/A':
                    # Memory utilization coloring
                    color = get_memory_color(value)
                    if color:
                        value = colored(f"{value:^{width}}", color)
                    else:
                        value = f"{value:^{width}}"
                elif col == 'memory[used/total]' and value != 'N/A':
                    # Memory ratio coloring
                    try:
                        if '/' in value:
                            used_str, total_str = value.split('/')
                            color = get_memory_ratio_color(used_str, total_str)
                            if color:
                                value = colored(f"{value:^{width}}", color)
                            else:
                                value = f"{value:^{width}}"
                        else:
                            value = f"{value:^{width}}"
                    except (ValueError, AttributeError, IndexError):
                        value = f"{value:^{width}}"
                else:
                    # Center-align all other columns
                    value = f"{value:^{width}}"
                
                row_parts.append(value)
            data_lines.append(" | ".join(row_parts))
        
        # Combine all parts
        result = [header, separator] + data_lines
        return "\n".join(result)
    
    def _get_server_summary(self, stats, system_info):
        """Generate compact summary for collapsed server view."""
        summary_data = self._get_server_summary_data(stats)
        return self._format_server_summary(summary_data)

    def _get_server_summary_data(self, stats):
        """Compute summary data for formatting/alignment."""
        if stats.empty:
            return None

        def mem_to_mib(value_str) -> int:
            if not value_str or str(value_str).strip() in {"N/A", ""}:
                return 0
            try:
                numbers = extract_numbers(str(value_str))
                if not numbers:
                    return 0
                value = float(numbers[0])
                unit = units_from_str(str(value_str)).lower()
                if unit in {"mib", "mb"}:
                    return int(value)
                if unit in {"gib", "gb"}:
                    return int(value * 1024)
                if unit in {"kib", "kb"}:
                    return int(value / 1024)
                if unit in {"b"}:
                    return int(value / (1024 * 1024))
                return int(value)
            except Exception:
                return 0

        gpu_count = len(stats)
        idle_count = 0
        total_util = 0
        total_mem_used_mib = 0
        total_mem_total_mib = 0

        for _, row in stats.iterrows():
            util_str = str(row.get("util", row.get("gpu_util", "0")))
            try:
                util_val = int(util_str.replace("%", "").strip())
                total_util += util_val
                if util_val < 5:
                    idle_count += 1
            except (ValueError, AttributeError):
                pass

            # Prefer explicit used/total fields (from nvidia-smi -q -x); fallback to pre-formatted column
            used_mib = mem_to_mib(row.get("used", ""))
            total_mib = mem_to_mib(row.get("total", ""))

            if used_mib == 0 and total_mib == 0:
                mem_col = row.get("memory[used/total]", "")
                if mem_col and "/" in str(mem_col):
                    try:
                        used_str, total_str = str(mem_col).split("/", 1)
                        used_mib = int("".join(filter(str.isdigit, used_str)) or 0)
                        total_mib = int("".join(filter(str.isdigit, total_str)) or 0)
                    except Exception:
                        used_mib = 0
                        total_mib = 0

            total_mem_used_mib += used_mib
            total_mem_total_mib += total_mib

        avg_util = total_util // gpu_count if gpu_count > 0 else 0

        if total_mem_total_mib > 0:
            if total_mem_total_mib >= 1024:
                mem_display = f"{total_mem_used_mib//1024}GB/{total_mem_total_mib//1024}GB"
            else:
                mem_display = f"{total_mem_used_mib}MB/{total_mem_total_mib}MB"
        else:
            mem_display = "N/A"

        if avg_util >= 80:
            util_color = "red"
        elif avg_util >= 50:
            util_color = "yellow"
        else:
            util_color = "green"

        return {
            "gpu_count": gpu_count,
            "idle_count": idle_count,
            "avg_util": avg_util,
            "util_color": util_color,
            "mem_display": mem_display,
        }

    def _format_server_summary(self, summary_data, widths=None) -> str:
        """Format summary string; if widths provided, columns are aligned."""
        if not summary_data:
            return "No GPU data available"
        if isinstance(summary_data, dict) and summary_data.get("status") == "loading":
            return "Loading..."
        if isinstance(summary_data, dict) and summary_data.get("status") == "empty":
            return "No GPU data available"

        gpu_count = summary_data["gpu_count"]
        idle_count = summary_data["idle_count"]
        avg_util = summary_data["avg_util"]
        util_color = summary_data["util_color"]
        mem_display = summary_data["mem_display"]

        if widths:
            gpu_digits = widths.get("gpu_digits", len(str(gpu_count)))
            idle_digits = widths.get("idle_digits", len(str(idle_count)))
            util_digits = widths.get("util_digits", len(str(avg_util)))
            mem_width = widths.get("mem_width", len(str(mem_display)))
        else:
            gpu_digits = len(str(gpu_count))
            idle_digits = len(str(idle_count))
            util_digits = len(str(avg_util))
            mem_width = len(str(mem_display))

        gpu_part = f"{gpu_count:>{gpu_digits}} GPUs"
        idle_part = f"{idle_count:>{idle_digits}} idle"
        util_plain = f"{avg_util:>{util_digits}}%"
        util_part = colored(util_plain, util_color) if util_color else util_plain
        mem_part = f"{str(mem_display):>{mem_width}}"

        return f"{gpu_part} | {idle_part} | {util_part} avg | {mem_part}"
    
    def print_stats(self, use_cache=False):
        """Print GPU stats with collapsible server view."""
        # Fetch new data or use cache
        if use_cache:
            with self._cache_lock:
                stats_list = self.cached_stats
                raw_stats_by_client = self.cached_raw_stats
                last_update_time = self._last_update_time
                last_fetch_duration = self._last_fetch_duration
                last_fetch_error = self._last_fetch_error
        else:
            fetch_started_at = time.time()
            try:
                stats_list, raw_stats_by_client = self.get_client_gpus_info(return_raw=True)
                fetch_error = None
            except Exception as e:
                stats_list, raw_stats_by_client = None, {}
                fetch_error = e
            fetch_duration = time.time() - fetch_started_at

            with self._cache_lock:
                if stats_list is not None:
                    self.cached_stats = stats_list
                    self.cached_raw_stats = raw_stats_by_client
                    self._last_update_time = time.time()
                    self._last_fetch_error = None
                else:
                    self._last_fetch_error = fetch_error
                self._last_fetch_duration = fetch_duration

            last_update_time = self._last_update_time
            last_fetch_duration = self._last_fetch_duration
            last_fetch_error = self._last_fetch_error

        if stats_list is None:
            stats_list = ["Loading..."] * len(self.pool)
        if not isinstance(raw_stats_by_client, dict):
            raw_stats_by_client = {}
        
        current_time = time.strftime("%H:%M:%S")
        
        # Ensure selected_server is within bounds
        if self.selected_server >= len(self.pool):
            self.selected_server = len(self.pool) - 1
        if self.selected_server < 0:
            self.selected_server = 0

        update_display = "--:--:--"
        if last_update_time:
            update_display = time.strftime("%H:%M:%S", time.localtime(last_update_time))
        fetch_display = ""
        if isinstance(last_fetch_duration, (int, float)):
            fetch_display = f" ({last_fetch_duration:.1f}s)"
        warn_display = " | WARN: refresh failed" if last_fetch_error else ""

        output_lines = []
        output_lines.append(
            f"Time: {current_time} | Updated: {update_display}{fetch_display} | Servers: {len(self.pool)} | [j/k] Navigate [Enter] Toggle [a] Expand All [c] Collapse All [q] Quit{warn_display}"
        )
        output_lines.append("-" * 80)

        # Align the collapsed server list by padding headers to the same display width
        index_width = len(str(len(self.pool)))
        server_rows = []
        max_header_width = 0

        summary_rows = []
        for idx, (client, stats_info) in enumerate(zip(self.pool, stats_list)):
            is_selected = idx == self.selected_server
            is_expanded = idx in self.expanded_servers

            # Use cached raw stats for summary
            stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))

            # Build summary
            if last_update_time is None and stats.empty:
                summary_data = {"status": "loading"}
            else:
                summary_data = self._get_server_summary_data(stats) or {"status": "empty"}

            # Format the header line (index padded for consistent width)
            expand_icon = "v" if is_expanded else ">"
            selector = "*" if is_selected else " "
            header_plain = f"{selector} {expand_icon} [{idx + 1:{index_width}d}] {client.description}"

            header_width = self.term.length(header_plain)
            max_header_width = max(max_header_width, header_width)

            summary_rows.append(summary_data)
            server_rows.append((is_selected, is_expanded, header_plain, summary_data, stats_info))

        non_empty_summaries = [s for s in summary_rows if isinstance(s, dict) and "gpu_count" in s]
        if non_empty_summaries:
            widths = {
                "gpu_digits": max(len(str(s["gpu_count"])) for s in non_empty_summaries),
                "idle_digits": max(len(str(s["idle_count"])) for s in non_empty_summaries),
                "util_digits": max(len(str(s["avg_util"])) for s in non_empty_summaries),
                "mem_width": max(len(str(s["mem_display"])) for s in non_empty_summaries),
            }
        else:
            widths = None

        for is_selected, is_expanded, header_plain, summary_data, stats_info in server_rows:
            pad = max_header_width - self.term.length(header_plain)
            header_padded = header_plain + (" " * pad if pad > 0 else "")

            if is_selected:
                header_display = self.term.reverse + header_padded + self.term.normal
            else:
                header_display = header_padded

            # Header with summary on same line (summary aligned)
            summary = self._format_server_summary(summary_data, widths=widths)
            output_lines.append(f"{header_display}  {summary}")

            # If expanded, print the full stats table (already formatted from get_client_gpus_info)
            if is_expanded:
                output_lines.extend(str(stats_info).splitlines())
                output_lines.append("")  # Add spacing after expanded server

        # Single write reduces visible flicker vs. clearing + many print() calls
        screen = self.term.home + "\n".join(self.term.clear_eol + line for line in output_lines) + "\n" + self.term.clear_eos
        sys.stdout.write(screen)
        sys.stdout.flush()

    def _keyboard_listener(self):
        """Real-time keyboard listener thread, monitors keys for navigation and control"""
        try:
            with self.term.cbreak():  # Enable character-by-character input
                while not self.quit_flag.is_set():
                    try:
                        key = self.term.inkey(timeout=0.1)  # Non-blocking input with timeout
                        if key:
                            key_name = key.name if hasattr(key, 'name') and key.name else str(key)
                            key_lower = str(key).lower()
                            
                            if key_lower == 'q':
                                self.quit_flag.set()
                                break
                            elif key_lower == 'h':
                                # Show help - will be overwritten on next refresh
                                pass
                            elif key_lower == 'j' or key_name == 'KEY_DOWN':
                                # Move selection down
                                if self.selected_server < len(self.pool) - 1:
                                    self.selected_server += 1
                                    self.ui_only_refresh = True  # Use cached data for fast UI update
                                    self.refresh_needed.set()  # Trigger immediate refresh
                            elif key_lower == 'k' or key_name == 'KEY_UP':
                                # Move selection up
                                if self.selected_server > 0:
                                    self.selected_server -= 1
                                    self.ui_only_refresh = True  # Use cached data for fast UI update
                                    self.refresh_needed.set()  # Trigger immediate refresh
                            elif key_name == 'KEY_ENTER' or key_lower == ' ' or key == '\n' or key == '\r':
                                # Toggle expand/collapse for selected server
                                if self.selected_server in self.expanded_servers:
                                    self.expanded_servers.discard(self.selected_server)
                                else:
                                    self.expanded_servers.add(self.selected_server)
                                self.ui_only_refresh = True  # Use cached data for fast UI update
                                self.refresh_needed.set()  # Trigger immediate refresh
                            elif key_lower == 'a':
                                # Expand all servers
                                self.expanded_servers = set(range(len(self.pool)))
                                self.ui_only_refresh = True  # Use cached data for fast UI update
                                self.refresh_needed.set()  # Trigger immediate refresh
                            elif key_lower == 'c':
                                # Collapse all servers
                                self.expanded_servers.clear()
                                self.ui_only_refresh = True  # Use cached data for fast UI update
                                self.refresh_needed.set()  # Trigger immediate refresh
                    except KeyboardInterrupt:
                        break
        except:
            pass

    def _background_refresh(self, interval_seconds: float = 1.0):
        """Background thread: fetch stats periodically so UI thread stays responsive."""
        while not self.quit_flag.is_set():
            fetch_started_at = time.time()
            try:
                stats_list, raw_stats_by_client = self.get_client_gpus_info(return_raw=True)
                fetch_error = None
            except Exception as e:
                stats_list, raw_stats_by_client = None, {}
                fetch_error = e
            fetch_duration = time.time() - fetch_started_at

            with self._cache_lock:
                if stats_list is not None:
                    self.cached_stats = stats_list
                    self.cached_raw_stats = raw_stats_by_client
                    self._last_update_time = time.time()
                    self._last_fetch_error = None
                else:
                    self._last_fetch_error = fetch_error
                self._last_fetch_duration = fetch_duration

            # Trigger UI refresh (both for new data and errors)
            self.refresh_needed.set()

            # Keep a minimum idle interval between refreshes to avoid hammering remote hosts
            sleep_seconds = max(0.0, interval_seconds)
            end_time = time.time() + sleep_seconds
            while time.time() < end_time and not self.quit_flag.is_set():
                time.sleep(0.1)
    
    def print_refresh(self):
        """Real-time GPU status display with global keyboard monitoring"""
        print("GPU monitoring starting...")
        print("Controls:")
        print("   j/k or Up/Down : Navigate between servers")
        print("   Enter/Space    : Toggle expand/collapse")
        print("   a              : Expand all")
        print("   c              : Collapse all")
        print("   q              : Quit")
        print("=" * 60)
        
        # Start keyboard listener thread
        keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        keyboard_thread.start()

        # Start background data refresh thread (prevents remote SSH calls from blocking UI)
        refresh_thread = threading.Thread(target=self._background_refresh, daemon=True)
        refresh_thread.start()
        
        try:
            with self.term.hidden_cursor():
                # Initial draw (may show "Loading..." until first background refresh completes)
                self.print_stats(use_cache=True)

                while not self.quit_flag.is_set():
                    # Event-driven redraw: avoids back-to-back full redraws (less flicker)
                    self.refresh_needed.wait(timeout=1.0)
                    self.refresh_needed.clear()
                    if self.quit_flag.is_set():
                        break
                    self.print_stats(use_cache=True)
                
        except KeyboardInterrupt:
            print("\n\n👋 Detected Ctrl+C, exiting program...")
        except Exception as e:
            print(f"\n\n❌ Error occurred: {e}")
        finally:
            self.quit_flag.set()  # Ensure thread exits

    def print_once(self):
        """Print GPU stats once and exit (no TUI loop)"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Time: {current_time} | Servers: {len(self.pool)}")
        print("=" * 80)
        
        # Get stats
        stats_list, raw_stats_by_client = self.get_client_gpus_info(return_raw=True)
        
        for idx, (client, stats_info) in enumerate(zip(self.pool, stats_list)):
            stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))
            
            # Build summary
            summary = self._get_server_summary(stats, system_info)
            
            # Print header with summary
            print(f"[{idx + 1}] {client.description}  {summary}")
            
            # Print the full stats table
            print(stats_info)
            print()  # Add spacing after each server
