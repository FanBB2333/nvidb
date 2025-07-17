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
from .utils import xml_to_dict, num_from_str, units_from_str, extract_numbers, extract_value_and_unit, format_bandwidth


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
            gpus = root.findall('gpu')
            stats = []
            
            for gpu in gpus:
                # Safely extract all values with default fallbacks
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
                    'current_power_limit': current_power_limit
                })
            
            stats = pd.DataFrame(stats)
            return stats
            
        except Exception as e:
            logging.error(msg=f"Failed to get full GPU info: {e}")
            return pd.DataFrame()
    
    def get_client(self):
        """Return the client object"""
        return self


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
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            logging.error(msg=f"Command execution failed: {e}")
            return f"Error: {e.stderr}" if e.stderr else f"Command failed with return code {e.returncode}"
        except Exception as e:
            logging.error(msg=f"Unexpected error executing command: {e}")
            return f"Unexpected error: {str(e)}"


class NviClientPool:
    def __init__(self, server_list: ServerListInfo):
        self.pool = [LocalClient()]
        if server_list is not None:
            self.pool += [RemoteClient(server) for server in server_list]
        logging.info(msg=f"Initialized pool with {len(self.pool)} clients.")
        self.connect_all()
        self.term = Terminal()
        self.quit_flag = threading.Event()  # Exit flag for inter-thread communication
    
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
    
    def get_client_gpus_info(self):
        # Set pandas display options - fixed column width and fill horizontally
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)  # Auto-fill terminal width
        pd.set_option('display.max_colwidth', 12)  # Fixed max column width to 12 characters
        pd.set_option('display.colheader_justify', 'center')  # Center column headers
        
        stats_str = []
        for idx, client in enumerate(self.pool):
            # logging.info(msg=colored(f"{client.description}", 'yellow'))
            # print(colored(f"{client.description}", 'yellow'))
            stats = client.get_full_gpu_info()
            
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

            # rename columns: product_name -> name, gpu_temp -> temp, fan_speed -> fan, memory_util -> mem_util, gpu_util -> gpu_util
            stats = stats.rename(columns={'product_name': 'name', 'gpu_temp': 'temp', 'fan_speed': 'fan', 'memory_util': 'mem', 'gpu_util': 'util'})
            
            # replace the NVIDIA/GeForce with "" in name column
            stats['name'] = stats['name'].str.replace('NVIDIA', '').str.replace('GeForce', '').str.strip()
            
            # Ensure name column doesn't exceed column width limit
            stats['name'] = stats['name'].apply(lambda x: x[:11] if len(str(x)) > 11 else x)
            
            # remove rows: product_architecture, rx_util, tx_util, power_state, power_draw, current_power_limit, used, total, free
            stats = stats.drop(columns=['product_architecture', 'rx_util', 'tx_util', 'power_state', 'power_draw', 'current_power_limit', 'used', 'total', 'free'])

            stats_str.append(stats)
        # reformat the str into a single string with fixed width formatting
        formatted_stats = []
        for client, stats in zip(self.pool, stats_str):
            # Create formatted table display
            formatted_table = self._format_fixed_width_table(stats)
            formatted_stats.append(f"\n{colored(client.description, 'yellow')}\n{formatted_table}")
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
        
        # Define fixed width for each column - adaptive to terminal width
        base_widths = {
            'name': 12,
            'temp': 8,
            'fan': 8,
            'util': 6,
            'mem': 6,
            'rx': 10,
            'tx': 10,
            'power': 15,
            'memory[used/total]': 18
        }
        
        # Calculate minimum width required
        min_width_needed = sum(base_widths.get(col, 12) for col in df.columns) + 3 * (len(df.columns) - 1)  # Add separators
        
        # If terminal width is sufficient, appropriately increase width of certain columns
        if terminal_width > min_width_needed + 20:
            extra_space = terminal_width - min_width_needed - 10  # Leave some margin
            # Priority to increase name column width
            if 'name' in base_widths:
                base_widths['name'] += min(extra_space // 2, 8)
            # Increase memory column width
            if 'memory[used/total]' in base_widths:
                base_widths['memory[used/total]'] += min(extra_space // 4, 6)
        
        column_widths = base_widths
        
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
                    value = value[:width-2] + ".."
                # Right-align numeric columns, left-align text columns
                if col in ['temp', 'fan', 'util', 'mem', 'rx', 'tx']:
                    row_parts.append(f"{value:>{width}}")
                else:
                    row_parts.append(f"{value:<{width}}")
            data_lines.append(" | ".join(row_parts))
        
        # Combine all parts
        result = [header, separator] + data_lines
        return "\n".join(result)
    
    def print_stats(self):
        stats_str = self.get_client_gpus_info()
        current_time = time.strftime("%H:%M:%S")
        print(self.term.home + self.term.clear)
        print(f"⏰ Time: {current_time}")
        for stats in stats_str:
            print(stats)

    def _keyboard_listener(self):
        """Real-time keyboard listener thread, monitors 'q' key to exit"""
        try:
            with self.term.cbreak():  # Enable character-by-character input
                while not self.quit_flag.is_set():
                    try:
                        key = self.term.inkey(timeout=0.1)  # Non-blocking input with timeout
                        if key:
                            if key.lower() == 'q':
                                self.quit_flag.set()
                                break
                            elif key.lower() == 'h':
                                # Temporarily show help overlay
                                print(self.term.move_yx(0, 0) + self.term.clear_eol + 
                                      self.term.bold + "💡 Help: Press 'q' to exit | Press any key to continue" + self.term.normal)
                    except KeyboardInterrupt:
                        break
        except:
            pass
    
    def print_refresh(self):
        """Real-time GPU status display with global keyboard monitoring"""
        print("🖥️  GPU monitoring starting...")
        print("💡 Tips:")
        print("   - Press 'q' key to exit program (no Enter required)")
        print("   - Press 'h' key to show help") 
        print("   - Press Ctrl+C to force exit")
        print("=" * 60)
        
        # Start keyboard listener thread
        keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        try:
            while not self.quit_flag.is_set():
                # Display GPU status with time at top
                self.print_stats()
                
                # Wait 2 seconds or until exit flag is set
                for _ in range(20):  # 2 seconds divided into 20 x 0.1 seconds
                    if self.quit_flag.is_set():
                        break
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n👋 Detected Ctrl+C, exiting program...")
        except Exception as e:
            print(f"\n\n❌ Error occurred: {e}")
        finally:
            self.quit_flag.set()  # Ensure thread exits

