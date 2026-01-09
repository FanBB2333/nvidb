from typing import Literal, Optional
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
from paramiko.ssh_exception import NoValidConnectionsError, PasswordRequiredException
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
        self.connected = False
        self.last_connect_error = None
        self.last_error_type = None

    def _set_connect_error(self, message: str, error_type: str = "error"):
        self.connected = False
        self.last_connect_error = message
        self.last_error_type = error_type
    
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
        if getattr(self, "connected", True) is False:
            error_message = getattr(self, "last_connect_error", None) or "Not connected"
            error_type = getattr(self, "last_error_type", None) or "error"
            return pd.DataFrame(), {"error": error_message, "error_type": error_type}

        def safe_get_text(element, path, default="N/A"):
            """Safely get text from XML element, return default if not found"""
            if element is None:
                return default
            found = element.find(path)
            return found.text if found is not None else default

        try:
            result = self.execute_command('nvidia-smi -q -x')
            # Check if we got valid output before trying to parse
            if not result or not result.strip() or not result.strip().startswith('<?xml'):
                # No NVIDIA GPU or nvidia-smi not available - this is expected on some systems
                return pd.DataFrame(), {}
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
            # Only log at debug level - this is expected on systems without NVIDIA GPUs
            logging.debug(msg=f"Failed to get full GPU info: {e}")
            return pd.DataFrame(), {}

    def get_system_stats(self) -> dict:
        """Get system statistics: CPU cores, CPU usage, memory usage, and swap usage.

        Returns:
            dict: System statistics with keys:
                - cpu_cores: Number of CPU cores
                - cpu_percent: Total CPU utilization percentage
                - mem_used_gb: Used memory in GB
                - mem_total_gb: Total memory in GB
                - swap_used_gb: Used swap in GB
                - swap_total_gb: Total swap in GB
        """
        result = {
            "cpu_cores": 0,
            "cpu_percent": 0.0,
            "mem_used_gb": 0.0,
            "mem_total_gb": 0.0,
            "swap_used_gb": 0.0,
            "swap_total_gb": 0.0,
        }

        try:
            # Get CPU cores - try Linux first, then macOS
            cpu_output = self.execute_command("nproc")
            if cpu_output and cpu_output.strip().isdigit():
                result["cpu_cores"] = int(cpu_output.strip())
            else:
                # Try macOS sysctl
                cpu_output = self.execute_command("sysctl -n hw.ncpu")
                if cpu_output and cpu_output.strip().isdigit():
                    result["cpu_cores"] = int(cpu_output.strip())

            # Get CPU utilization - try Linux first
            cpu_usage_output = self.execute_command(
                "grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage}'"
            )
            if cpu_usage_output and cpu_usage_output.strip():
                try:
                    result["cpu_percent"] = float(cpu_usage_output.strip())
                except ValueError:
                    pass
            else:
                # Try macOS - use top for CPU usage (user + sys)
                cpu_usage_output = self.execute_command(
                    "top -l 1 -n 0 | grep 'CPU usage' | awk '{print $3, $5}' | tr -d '%,'"
                )
                if cpu_usage_output and cpu_usage_output.strip():
                    try:
                        parts = cpu_usage_output.strip().split()
                        if len(parts) >= 2:
                            user_cpu = float(parts[0])
                            sys_cpu = float(parts[1])
                            result["cpu_percent"] = user_cpu + sys_cpu
                    except ValueError:
                        pass

            # Get memory info - try Linux first
            mem_output = self.execute_command(
                "grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo"
            )
            if mem_output and mem_output.strip():
                mem_info = {}
                for line in mem_output.strip().split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        # Extract numeric value (in kB)
                        value = value.strip().split()[0]
                        try:
                            mem_info[key.strip()] = int(value)
                        except ValueError:
                            pass

                mem_total_kb = mem_info.get("MemTotal", 0)
                mem_available_kb = mem_info.get("MemAvailable", 0)
                swap_total_kb = mem_info.get("SwapTotal", 0)
                swap_free_kb = mem_info.get("SwapFree", 0)

                result["mem_total_gb"] = mem_total_kb / (1024 * 1024)
                result["mem_used_gb"] = (mem_total_kb - mem_available_kb) / (1024 * 1024)
                result["swap_total_gb"] = swap_total_kb / (1024 * 1024)
                result["swap_used_gb"] = (swap_total_kb - swap_free_kb) / (1024 * 1024)
            else:
                # Try macOS - use sysctl for memory
                mem_total_output = self.execute_command("sysctl -n hw.memsize")
                if mem_total_output and mem_total_output.strip().isdigit():
                    mem_total_bytes = int(mem_total_output.strip())
                    result["mem_total_gb"] = mem_total_bytes / (1024 * 1024 * 1024)

                    # Get memory usage from vm_stat on macOS
                    vm_stat_output = self.execute_command("vm_stat")
                    if vm_stat_output:
                        page_size = 4096  # Default page size
                        pages_free = 0
                        pages_inactive = 0
                        pages_speculative = 0

                        for line in vm_stat_output.split("\n"):
                            if "page size of" in line:
                                try:
                                    page_size = int(line.split()[-2])
                                except (ValueError, IndexError):
                                    pass
                            elif "Pages free:" in line:
                                try:
                                    pages_free = int(line.split()[-1].rstrip('.'))
                                except (ValueError, IndexError):
                                    pass
                            elif "Pages inactive:" in line:
                                try:
                                    pages_inactive = int(line.split()[-1].rstrip('.'))
                                except (ValueError, IndexError):
                                    pass
                            elif "Pages speculative:" in line:
                                try:
                                    pages_speculative = int(line.split()[-1].rstrip('.'))
                                except (ValueError, IndexError):
                                    pass

                        available_bytes = (pages_free + pages_inactive + pages_speculative) * page_size
                        result["mem_used_gb"] = (mem_total_bytes - available_bytes) / (1024 * 1024 * 1024)

                # Get swap info on macOS
                swap_output = self.execute_command("sysctl -n vm.swapusage")
                if swap_output:
                    # Format: "total = 2048.00M  used = 1024.00M  free = 1024.00M"
                    for part in swap_output.split():
                        if part.endswith("M"):
                            try:
                                value = float(part[:-1])
                                # Determine which field based on position
                            except ValueError:
                                pass
                    # Parse more carefully
                    parts = swap_output.split()
                    for i, part in enumerate(parts):
                        if part == "total" and i + 2 < len(parts):
                            try:
                                val = parts[i + 2].rstrip("M")
                                result["swap_total_gb"] = float(val) / 1024
                            except (ValueError, IndexError):
                                pass
                        elif part == "used" and i + 2 < len(parts):
                            try:
                                val = parts[i + 2].rstrip("M")
                                result["swap_used_gb"] = float(val) / 1024
                            except (ValueError, IndexError):
                                pass

        except Exception as e:
            logging.debug(msg=f"Failed to get system stats: {e}")

        return result

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
        self.identityfile = getattr(server, "identityfile", None)
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if self.auth in ['auto', 'key']:
            self.client.load_system_host_keys()
    
    def __del__(self):
        self.client.close()
        logging.info(msg=f"Connection to {self.host}:{self.port} closed.")
    
    def _authenticate_with_password(self, max_attempts=3, *, prompt_only: bool = False) -> bool:
        """Attempt password authentication with retry limit.
        
        Args:
            max_attempts: Maximum number of password attempts (default: 3)
            prompt_only: Do not try any configured password; always prompt (default: False)
            
        Returns:
            bool: True if authentication succeeds
            
        """
        for attempt in range(max_attempts):
            try:
                password = None
                if not prompt_only and self.password is not None and attempt == 0:
                    password = self.password
                else:
                    remaining = max_attempts - attempt
                    if attempt > 0:
                        logging.warning(msg=f"Authentication failed. {remaining} attempt(s) remaining.")
                        print(f"  ⚠ Authentication failed. {remaining} attempt(s) remaining.")
                    password = getpass.getpass(prompt=f"Enter password for {self.username}@{self.host}:{self.port} -> ")

                self.client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    password=password,
                    allow_agent=False,
                    look_for_keys=False,
                )
                self.connected = True
                self.last_connect_error = None
                self.last_error_type = None
                logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username}")
                return True
                    
            except AuthenticationException as e:
                if attempt == max_attempts - 1:
                    logging.error(
                        msg=f"Password authentication failed after {max_attempts} attempts on {self.description}"
                    )
                    self._set_connect_error("Password incorrect", error_type="auth")
                    return False
                continue
            except Exception as e:
                logging.error(msg=f"Password authentication error on {self.description}: {e}")
                self._set_connect_error(str(e), error_type="connect")
                return False

        self._set_connect_error("Password incorrect", error_type="auth")
        return False

    def connect(self) -> bool:
        print(f"Connecting to {self.host}:{self.port} as {self.username}")
        # catch the OSError exception when the host is not reachable
        try:
            identityfile = None
            if self.auth in ("auto", "key") and self.identityfile:
                identityfile = os.path.expanduser(str(self.identityfile))
            if self.auth == "auto":
                # Auto mode: try key-based auth first, then password
                try:
                    self.client.connect(
                        hostname=self.host,
                        port=self.port,
                        username=self.username,
                        allow_agent=False if identityfile else True,
                        look_for_keys=False if identityfile else True,
                        key_filename=identityfile,
                    )
                    self.connected = True
                    self.last_connect_error = None
                    self.last_error_type = None
                    logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username}")
                    return True
                except PasswordRequiredException:
                    if not identityfile:
                        logging.error(msg=f"Key requires passphrase on {self.description}, trying password...")
                        return self._authenticate_with_password()
                    passphrase = getpass.getpass(prompt=f"Enter passphrase for key {identityfile} -> ")
                    self.client.connect(
                        hostname=self.host,
                        port=self.port,
                        username=self.username,
                        allow_agent=False,
                        look_for_keys=False,
                        key_filename=identityfile,
                        passphrase=passphrase,
                    )
                    self.connected = True
                    self.last_connect_error = None
                    self.last_error_type = None
                    logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username}")
                    return True
                except AuthenticationException as e:
                    logging.error(msg=f"Key-based authentication failed on {self.description}, trying password...")
                    # Use the new password authentication method with retry limit
                    return self._authenticate_with_password()
                except NoValidConnectionsError as e:
                    logging.error(msg=f"Connection failed: {e}")
                    self._set_connect_error(str(e), error_type="connect")
                    return False
            elif self.auth == "key":
                # Key-based authentication only (no password fallback)
                try:
                    self.client.connect(
                        hostname=self.host,
                        port=self.port,
                        username=self.username,
                        allow_agent=False if identityfile else True,
                        look_for_keys=False if identityfile else True,
                        key_filename=identityfile,
                    )
                    self.connected = True
                    self.last_connect_error = None
                    self.last_error_type = None
                    logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username} (key auth)")
                    return True
                except PasswordRequiredException:
                    if not identityfile:
                        self._set_connect_error("Key requires passphrase; set `identityfile` or use `auth: password`", error_type="auth")
                        return False
                    passphrase = getpass.getpass(prompt=f"Enter passphrase for key {identityfile} -> ")
                    self.client.connect(
                        hostname=self.host,
                        port=self.port,
                        username=self.username,
                        allow_agent=False,
                        look_for_keys=False,
                        key_filename=identityfile,
                        passphrase=passphrase,
                    )
                    self.connected = True
                    self.last_connect_error = None
                    self.last_error_type = None
                    logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username} (key auth)")
                    return True
                except AuthenticationException as e:
                    logging.error(msg=f"Key-based authentication failed on {self.description}: {e}")
                    self._set_connect_error("Key authentication failed", error_type="auth")
                    return False
                except NoValidConnectionsError as e:
                    logging.error(msg=f"Connection failed: {e}")
                    self._set_connect_error(str(e), error_type="connect")
                    return False
            elif self.auth == "password":
                # Password authentication with retry limit
                try:
                    # Password mode: try configured password first, then prompt if needed.
                    return self._authenticate_with_password()
                except NoValidConnectionsError as e:
                    logging.error(msg=f"Connection failed: {e}")
                    self._set_connect_error(str(e), error_type="connect")
                    return False
            else:
                logging.error(msg=f"Unsupported authentication method: {self.auth}, please use 'auto', 'key', or 'password'.")
                self._set_connect_error(f"Unsupported auth method: {self.auth}", error_type="error")
                return False
        except OSError as e:
            logging.error(msg=f"Connection failed: {e}")
            self._set_connect_error(str(e), error_type="connect")
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
        self.connected = True
        self.last_connect_error = None
        self.last_error_type = None
        return True
    
    def execute_command(self, command: str) -> str:
        """Execute local command"""
        try:
            # Use shell=True to support pipes and complex commands
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                # Only log at debug level for expected failures on systems without NVIDIA GPUs
                logging.debug(msg=f"Command '{command}' execution failed with return code {result.returncode}")
                return ""
            return result.stdout
        except subprocess.CalledProcessError as e:
            logging.debug(msg=f"Command '{command}' execution failed: {e}")
            return ""
        except Exception as e:
            logging.debug(msg=f"Unexpected error executing command: {e}")
            return ""


class NVClientPool:
    def __init__(self, server_list: ServerListInfo, *, compact: bool = False):
        self.pool = [LocalClient()]
        if server_list is not None:
            self.pool += [RemoteClient(server) for server in server_list]
        logging.info(msg=f"Initialized pool with {len(self.pool)} clients.")
        self.connect_all()
        self.term = Terminal()
        self.compact = bool(compact)
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
        self._toggle_disabled_servers = set()
    
    def connect_all(self):
        for client in self.pool:
            try:
                client.connect()
            except Exception as e:
                logging.error(msg=f"Failed to connect to {getattr(client, 'description', 'unknown')}: {e}")
                if hasattr(client, "_set_connect_error"):
                    client._set_connect_error(str(e), error_type="connect")

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

    def get_all_system_stats(self) -> dict:
        """Get system statistics for all clients in the pool.

        Returns:
            dict: Mapping of client index to system stats dict.
                  Each system stats dict contains:
                  - cpu_cores: Number of CPU cores
                  - cpu_percent: Total CPU utilization percentage
                  - mem_used_gb: Used memory in GB
                  - mem_total_gb: Total memory in GB
                  - swap_used_gb: Used swap in GB
                  - swap_total_gb: Total swap in GB
        """
        result = {}
        for idx, client in enumerate(self.pool):
            try:
                result[idx] = client.get_system_stats()
            except Exception as e:
                logging.warning(f"Failed to get system stats for {client.description}: {e}")
                result[idx] = {}
        return result

    def get_client_gpus_info(self, return_raw: bool = False):
        # Set pandas display options for terminal output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)  # Auto-fill terminal width
        pd.set_option('display.colheader_justify', 'center')  # Center column headers
        
        stats_str = []
        raw_stats_by_client = {}
        user_memory_by_client = {}
        global_user_memory = {}
        for idx, client in enumerate(self.pool):
            result = client.get_full_gpu_info()
            
            # Handle the tuple return from get_full_gpu_info
            if isinstance(result, tuple) and len(result) == 2:
                stats, system_info = result
            else:
                # Fallback for backward compatibility
                stats = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
                system_info = {}

            # Fetch system stats (CPU, memory, swap) and merge into system_info
            if not (isinstance(system_info, dict) and system_info.get("error")):
                try:
                    sys_stats = client.get_system_stats()
                    system_info["system_stats"] = sys_stats
                except Exception as e:
                    logging.warning(f"Failed to get system stats for {client.description}: {e}")

            user_summary_for_client = {}
            
            # Skip processing if stats is empty
            if stats.empty:
                raw_stats_by_client[idx] = (stats.copy() if not stats.empty else stats, system_info)
                stats_str.append((stats, system_info))
                user_memory_by_client[idx] = {}
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
                all_processes, user_summary_for_client = client.get_process_summary(stats)

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
                user_summary_for_client = {}
            
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
            
            # Reorder columns: move mem_util before memory[used/total] and processes at the end
            if 'processes' in stats.columns:
                # Define desired column order
                desired_order = ['GPU', 'name', 'fan', 'util', 'temp', 'rx', 'tx', 'power', 'mem_util', 'memory[used/total]', 'processes']
                # Only keep columns that exist in stats
                ordered_columns = [col for col in desired_order if col in stats.columns]
                # Add any remaining columns that weren't in desired_order
                remaining_columns = [col for col in stats.columns if col not in ordered_columns]
                stats = stats[ordered_columns + remaining_columns]

            stats_str.append((stats, system_info))

            # Cache processed table (not raw XML) and per-user memory stats for this client
            raw_stats_by_client[idx] = (stats.copy() if not stats.empty else stats, system_info)
            user_memory_by_client[idx] = dict(user_summary_for_client or {})
            for username, total_mib in (user_summary_for_client or {}).items():
                try:
                    total_mib_int = int(total_mib)
                except Exception:
                    continue
                if total_mib_int <= 0:
                    continue
                global_user_memory[username] = global_user_memory.get(username, 0) + total_mib_int

        raw_stats_by_client["_nvidb"] = {
            "user_memory_by_client": user_memory_by_client,
            "user_memory_global": global_user_memory,
        }
        # reformat the str into a single string with fixed width formatting
        formatted_stats = []
        for client, (stats, system_info) in zip(self.pool, stats_str):
            # Create formatted table display (or error panel)
            if isinstance(system_info, dict) and system_info.get("error"):
                error_panel = self._format_error_panel(
                    system_info.get("error", "Error"),
                    error_type=system_info.get("error_type"),
                )
                formatted_stats.append(f"\n{colored(client.description, 'yellow')}\n{error_panel}")
                continue

            formatted_table = self._format_fixed_width_table(stats, border=True)

            system_info_header = ""
            if system_info:
                system_info_header = (
                    f"Driver: {system_info.get('driver_version', 'N/A')} | "
                    f"CUDA: {system_info.get('cuda_version', 'N/A')} | "
                    f"GPUs: {system_info.get('attached_gpus', '0')}\n"
                )

            formatted_stats.append(f"\n{colored(client.description, 'yellow')}\n{system_info_header}{formatted_table}")
        if return_raw:
            return formatted_stats, raw_stats_by_client
        return formatted_stats

    def _format_error_panel(self, message: str, error_type: Optional[str] = None) -> str:
        try:
            width = os.get_terminal_size().columns
        except OSError:
            width = 80

        if error_type == "auth" and str(message).strip() == "Password incorrect":
            title = "Password incorrect"
        elif error_type == "auth":
            title = "Authentication failed"
        else:
            title = "Error"
        line1 = f" {title} ".center(width, " ")
        line2 = f" {message} ".center(width, " ")
        line3 = " ".center(width, " ")
        return "\n".join(
            [
                colored(line1[:width], "white", "on_red", attrs=["bold"]),
                colored(line2[:width], "white", "on_red", attrs=["bold"]),
                colored(line3[:width], "white", "on_red", attrs=["bold"]),
            ]
        )
    
    def _format_fixed_width_table(self, df, border: bool = False):
        """Format fixed-width table display."""
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

        all_columns = list(df.columns)

        # Calculate widths based on terminal space
        separator_width = 3
        outer_width = 4 if border else 0  # "| " + " |"
        max_content_width = max(0, terminal_width - outer_width)

        # Column importance: larger => dropped earlier when narrow
        importance = {
            "GPU": 0,
            "util": 1,
            "mem_util": 2,
            "memory[used/total]": 3,
            "name": 4,
            "processes": 5,
            "temp": 6,
            "power": 7,
            "rx": 8,
            "tx": 9,
            "fan": 10,
        }
        must_keep = [c for c in ("GPU", "util", "mem_util") if c in all_columns]

        min_widths_for_cols = {}
        for col in all_columns:
            min_width = min_widths.get(col, 12)
            min_widths_for_cols[col] = max(min_width, len(str(col)))

        def min_table_width(cols) -> int:
            if not cols:
                return 0
            return sum(min_widths_for_cols[c] for c in cols) + separator_width * (len(cols) - 1)

        # Drop least important columns until it fits the terminal width
        selected_columns = list(all_columns)
        while selected_columns and min_table_width(selected_columns) > max_content_width:
            droppable = [c for c in selected_columns if c not in must_keep]
            if not droppable:
                droppable = list(selected_columns)
            drop_col = max(
                droppable,
                key=lambda c: (importance.get(c, 100), selected_columns.index(c)),
            )
            selected_columns.remove(drop_col)

        if not selected_columns:
            selected_columns = all_columns[:1]

        df_display = df[selected_columns]

        # Compute content width for each column
        content_widths = {}
        for col in selected_columns:
            max_len = len(str(col))
            for value in df_display[col]:
                value_len = len(str(value))
                if value_len > max_len:
                    max_len = value_len
            content_widths[col] = max_len

        desired_widths = {}
        min_widths_selected = {}
        for col in selected_columns:
            min_width = min_widths_for_cols.get(col, 12)
            min_widths_selected[col] = min_width
            desired_widths[col] = max(min_width, content_widths.get(col, min_width))

        total_separator_width = separator_width * (len(selected_columns) - 1)
        available_width = max_content_width - total_separator_width  # available for column contents
        if available_width < 1:
            available_width = 1

        min_total = sum(min_widths_selected.values())
        desired_total = sum(desired_widths.values())

        fill_priority = [
            "processes",
            "name",
            "power",
            "memory[used/total]",
            "rx",
            "tx",
            "temp",
            "fan",
            "util",
            "mem_util",
            "GPU",
        ]

        if available_width >= desired_total:
            column_widths = dict(desired_widths)
            extra = available_width - desired_total
            if extra > 0 and not self.compact:
                fill_columns = list(selected_columns)
                share, remainder = divmod(extra, len(fill_columns))
                if share:
                    for col in fill_columns:
                        column_widths[col] += share
                if remainder:
                    offset = terminal_width % len(fill_columns)
                    for i in range(remainder):
                        column_widths[fill_columns[(offset + i) % len(fill_columns)]] += 1
        elif available_width >= min_total:
            column_widths = dict(min_widths_selected)
            extra = available_width - min_total
            for col in (c for c in fill_priority if c in selected_columns):
                if extra <= 0:
                    break
                need = desired_widths[col] - column_widths[col]
                if need > 0:
                    add = min(extra, need)
                    column_widths[col] += add
                    extra -= add
        else:
            # Extremely narrow terminal: shrink columns below minimums, but keep at least 1 char
            column_widths = dict(min_widths_selected)
            excess = sum(column_widths.values()) - available_width
            if excess > 0:
                shrink_order = sorted(
                    selected_columns,
                    key=lambda c: (importance.get(c, 100), selected_columns.index(c)),
                    reverse=True,
                )
                for col in shrink_order:
                    if excess <= 0:
                        break
                    reducible = column_widths[col] - 1
                    if reducible <= 0:
                        continue
                    reduce_by = min(excess, reducible)
                    column_widths[col] -= reduce_by
                    excess -= reduce_by

        def truncate_text(text: str, width: int, tail_preserve: bool = False) -> str:
            if width <= 0:
                return ""
            if text is None:
                text = ""
            text = str(text)
            if len(text) <= width:
                return text
            if width <= 2:
                return text[:width]
            if tail_preserve:
                return ".." + text[-(width - 2):]
            return text[:width - 2] + ".."

        def parse_percent(value: str) -> Optional[float]:
            if value is None:
                return None
            value_str = str(value).strip()
            if not value_str or value_str in {"N/A", "-"}:
                return None
            numbers = extract_numbers(value_str)
            if not numbers:
                return None
            try:
                return float(numbers[0])
            except Exception:
                return None

        def parse_ratio_percent(value: str) -> Optional[float]:
            if value is None:
                return None
            value_str = str(value).strip()
            if not value_str or value_str in {"N/A", "-"} or "/" not in value_str:
                return None
            used_str, total_str = value_str.split("/", 1)
            try:
                used_numbers = extract_numbers(used_str)
                total_numbers = extract_numbers(total_str)
                if not used_numbers or not total_numbers:
                    return None
                used_val = float(used_numbers[0])
                total_val = float(total_numbers[0])
                if total_val <= 0:
                    return None
                return (used_val / total_val) * 100
            except Exception:
                return None

        def mem_bar_bg(percent: Optional[float]) -> Optional[str]:
            if percent is None:
                return None
            if percent >= 60:
                return "on_red"
            if percent >= 10:
                return "on_yellow"
            if percent > 0:
                return "on_green"
            return None

        def format_bar_cell(text: str, percent: Optional[float], width: int) -> str:
            if width <= 0:
                return ""
            text = truncate_text(text, width)

            chars = [" "] * width
            start = max(0, (width - len(text)) // 2)
            for i, ch in enumerate(text):
                pos = start + i
                if 0 <= pos < width:
                    chars[pos] = ch

            if percent is None:
                return "".join(chars)

            try:
                p = max(0.0, min(100.0, float(percent)))
            except Exception:
                return "".join(chars)

            fill_len = int(round((p / 100.0) * width))
            if p > 0 and fill_len == 0:
                fill_len = 1
            fill_len = max(0, min(width, fill_len))

            # Keep the unfilled portion on the terminal default background.
            # `termcolor`'s "on_grey" maps to ANSI 40 (black), which looks wrong
            # on light/gray terminal themes.
            fill_bg = mem_bar_bg(p)

            left = "".join(chars[:fill_len])
            right = "".join(chars[fill_len:])
            if fill_len > 0 and fill_bg:
                left = colored(left, on_color=fill_bg)
            return left + right

        # Format table header
        header_parts = []
        for col in selected_columns:
            width = column_widths.get(col, 12)
            col_name = truncate_text(str(col), width)
            header_parts.append(f"{col_name:^{width}}")
        header = " | ".join(header_parts)

        # Format separator line
        separator_parts = []
        for col in selected_columns:
            width = column_widths.get(col, 12)
            separator_parts.append("-" * width)
        separator = "-+-".join(separator_parts)

        # Format data rows
        data_lines = []
        for _, row in df_display.iterrows():
            row_parts = []
            row_util = str(row.get("util", "0"))
            row_util_color = get_utilization_color(row_util)
            for col in selected_columns:
                width = column_widths.get(col, 12)
                raw_value = row.get(col, "")
                value = truncate_text(raw_value, width, tail_preserve=(col == "name"))

                if col == "GPU":
                    cell = f"{value:^{width}}"
                    if row_util_color:
                        cell = colored(cell, row_util_color, attrs=["bold"])
                    row_parts.append(cell)
                    continue

                if col == "util" and value != "N/A":
                    percent = parse_percent(raw_value)
                    display_text = str(raw_value).replace(" ", "")
                    if percent is not None:
                        display_text = f"{int(round(percent))}%"
                    row_parts.append(format_bar_cell(display_text, percent, width))
                    continue

                if col == "mem_util" and value != "N/A":
                    percent = parse_percent(raw_value)
                    display_text = str(raw_value).replace(" ", "")
                    if percent is not None:
                        display_text = f"{int(round(percent))}%"
                    row_parts.append(format_bar_cell(display_text, percent, width))
                    continue

                if col == "memory[used/total]" and value != "N/A":
                    percent = parse_ratio_percent(raw_value)
                    row_parts.append(format_bar_cell(value, percent, width))
                    continue

                # Center-align all other columns
                row_parts.append(f"{value:^{width}}")

            data_lines.append(" | ".join(row_parts))

        result_lines = [header, separator] + data_lines
        if border:
            inner_width = len(header)
            top = "+" + "-" * (inner_width + 2) + "+"
            bottom = top
            result_lines = [top] + [f"| {line} |" for line in result_lines] + [bottom]

        return "\n".join(result_lines)
    
    def _get_server_summary(self, stats, system_info):
        """Generate compact summary for collapsed server view."""
        summary_data = self._get_server_summary_data(stats, system_info)
        return self._format_server_summary(summary_data)

    def _get_server_summary_data(self, stats, system_info=None):
        """Compute summary data for formatting/alignment."""
        # Extract system stats from system_info if available
        sys_stats = {}
        if system_info and isinstance(system_info, dict):
            sys_stats = system_info.get("system_stats", {})

        if stats.empty:
            # Return system stats even when no GPU data
            return {
                "gpu_count": 0,
                "idle_count": 0,
                "avg_util": 0,
                "util_color": None,
                "mem_display": "N/A",
                "no_gpu": True,
                # System stats
                "cpu_cores": sys_stats.get("cpu_cores", 0),
                "cpu_percent": sys_stats.get("cpu_percent", 0.0),
                "mem_used_gb": sys_stats.get("mem_used_gb", 0.0),
                "mem_total_gb": sys_stats.get("mem_total_gb", 0.0),
                "swap_used_gb": sys_stats.get("swap_used_gb", 0.0),
                "swap_total_gb": sys_stats.get("swap_total_gb", 0.0),
            }

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
                mem_display = f"{round(total_mem_used_mib/1024)}GB/{round(total_mem_total_mib/1024)}GB"
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
            # System stats
            "cpu_cores": sys_stats.get("cpu_cores", 0),
            "cpu_percent": sys_stats.get("cpu_percent", 0.0),
            "mem_used_gb": sys_stats.get("mem_used_gb", 0.0),
            "mem_total_gb": sys_stats.get("mem_total_gb", 0.0),
            "swap_used_gb": sys_stats.get("swap_used_gb", 0.0),
            "swap_total_gb": sys_stats.get("swap_total_gb", 0.0),
        }

    def _format_server_summary(self, summary_data, widths=None) -> str:
        """Format summary string; if widths provided, columns are aligned."""
        if not summary_data:
            return "No GPU data available"
        if isinstance(summary_data, dict) and summary_data.get("status") == "loading":
            return "Loading..."
        if isinstance(summary_data, dict) and summary_data.get("status") == "error":
            message = summary_data.get("message", "Error")
            error_type = summary_data.get("error_type")
            if error_type == "auth" and str(message).strip() == "Password incorrect":
                return colored("Password incorrect", "red", attrs=["bold"])
            return colored(message, "red", attrs=["bold"])
        if isinstance(summary_data, dict) and summary_data.get("status") == "empty":
            return "No GPU data available"

        gpu_count = summary_data["gpu_count"]
        idle_count = summary_data["idle_count"]
        avg_util = summary_data["avg_util"]
        util_color = summary_data["util_color"]
        mem_display = summary_data["mem_display"]
        no_gpu = summary_data.get("no_gpu", False)

        # System stats
        cpu_cores = summary_data.get("cpu_cores", 0)
        cpu_percent = summary_data.get("cpu_percent", 0.0)
        mem_used_gb = summary_data.get("mem_used_gb", 0.0)
        mem_total_gb = summary_data.get("mem_total_gb", 0.0)
        swap_used_gb = summary_data.get("swap_used_gb", 0.0)
        swap_total_gb = summary_data.get("swap_total_gb", 0.0)

        # Format system stats parts
        # CPU: utilization first, cores in brackets
        if cpu_cores > 0:
            cpu_percent_int = int(round(cpu_percent))
            if cpu_percent_int >= 80:
                cpu_color = "red"
            elif cpu_percent_int >= 50:
                cpu_color = "yellow"
            else:
                cpu_color = "green"
            cpu_util_str = colored(f"{cpu_percent_int:>3}%", cpu_color)
            cpu_part = f"CPU: {cpu_util_str}({cpu_cores}C)"
        else:
            cpu_part = ""

        # Memory: used/total with swap in brackets
        if mem_total_gb > 0:
            if swap_total_gb > 0:
                mem_sys_part = f"Mem: {mem_used_gb:.0f}/{mem_total_gb:.0f}G(Swap:{swap_used_gb:.1f}/{swap_total_gb:.0f}G)"
            else:
                mem_sys_part = f"Mem: {mem_used_gb:.0f}/{mem_total_gb:.0f}G"
        else:
            mem_sys_part = ""

        # Handle no GPU case - only show system stats
        if no_gpu:
            sys_parts = [p for p in [cpu_part, mem_sys_part] if p]
            if sys_parts:
                return "No GPU | " + " | ".join(sys_parts)
            return "No GPU"

        if widths:
            gpu_digits = widths.get("gpu_digits", len(str(gpu_count)))
            idle_digits = widths.get("idle_digits", len(str(idle_count)))
            util_digits = widths.get("util_digits", len(str(avg_util)))
            mem_width = widths.get("mem_width", len(str(mem_display)))
            cpu_cores_digits = widths.get("cpu_cores_digits", len(str(cpu_cores)))
        else:
            gpu_digits = len(str(gpu_count))
            idle_digits = len(str(idle_count))
            util_digits = len(str(avg_util))
            mem_width = len(str(mem_display))
            cpu_cores_digits = len(str(cpu_cores))

        gpu_part = f"{gpu_count:>{gpu_digits}} GPUs"
        idle_part = f"{idle_count:>{idle_digits}} idle"
        util_plain = f"{avg_util:>{util_digits}}%"
        util_part = colored(util_plain, util_color) if util_color else util_plain
        mem_part = f"{str(mem_display):>{mem_width}}"

        # Build the summary string
        parts = [f"{gpu_part} | {idle_part} | {util_part} avg | {mem_part}"]
        sys_parts = [p for p in [cpu_part, mem_sys_part] if p]
        if sys_parts:
            parts.append(" | ".join(sys_parts))

        return " | ".join(parts)

    def _format_user_memory_totals(self, user_summary: dict, *, max_users: int = 10) -> str:
        """Format per-user total VRAM usage (MiB) as a compact single-line string."""
        if not user_summary:
            return ""

        items = []
        for user, mib in user_summary.items():
            try:
                mib_int = int(mib)
            except Exception:
                continue
            if mib_int <= 0:
                continue
            items.append((str(user), mib_int))

        if not items:
            return ""

        items.sort(key=lambda x: x[1], reverse=True)
        parts = []
        for user, mib_int in items[:max_users]:
            if mib_int >= 1024:
                parts.append(f"{user}({mib_int/1024:.1f}G)")
            else:
                parts.append(f"{user}({mib_int}M)")

        remaining = len(items) - max_users
        if remaining > 0:
            parts.append(f"+{remaining} more")
        return " ".join(parts)
    
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

        # Disable expand/collapse for servers with auth errors (e.g., password incorrect)
        try:
            disabled = set()
            for idx in range(len(self.pool)):
                _stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))
                if isinstance(system_info, dict) and system_info.get("error_type") == "auth":
                    disabled.add(idx)
            self._toggle_disabled_servers = disabled
        except Exception:
            self._toggle_disabled_servers = set()
        
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
        server_count = len(self.pool)
        server_label = "Server" if server_count == 1 else "Servers"
        output_lines.append(
            f"Time: {current_time} | Updated: {update_display}{fetch_display} | {server_label}: {server_count} | [j/k] Navigate [Enter] Toggle [a] Expand All [c] Collapse All [q] Quit{warn_display}"
        )
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80
        if self.compact:
            separator_width = min(80, terminal_width)
        else:
            separator_width = max(20, terminal_width)
        output_lines.append("-" * separator_width)

        meta = {}
        if isinstance(raw_stats_by_client, dict):
            meta = raw_stats_by_client.get("_nvidb", {}) or {}
        user_memory_by_client = meta.get("user_memory_by_client", {}) or {}
        global_user_memory = meta.get("user_memory_global", {}) or {}

        if global_user_memory:
            global_line = f"Users (all nodes): {self._format_user_memory_totals(global_user_memory, max_users=12)}"
            if self.term.length(global_line) > terminal_width:
                global_line = global_line[: max(0, terminal_width - 3)] + "..."
            output_lines.append(global_line)

        # Align the collapsed server list by padding headers to the same display width
        index_width = len(str(len(self.pool)))
        server_rows = []
        max_header_width = 0

        summary_rows = []
        for idx, (client, stats_info) in enumerate(zip(self.pool, stats_list)):
            is_selected = idx == self.selected_server
            is_expanded = idx in self.expanded_servers
            toggle_disabled = idx in self._toggle_disabled_servers

            # Use cached raw stats for summary
            stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))

            # Build summary
            if isinstance(system_info, dict) and system_info.get("error"):
                summary_data = {
                    "status": "error",
                    "message": system_info.get("error", "Error"),
                    "error_type": system_info.get("error_type"),
                }
            elif last_update_time is None and stats.empty:
                summary_data = {"status": "loading"}
            else:
                summary_data = self._get_server_summary_data(stats, system_info) or {"status": "empty"}

            # Format the header line (index padded for consistent width)
            if toggle_disabled:
                expand_icon = "!"
            else:
                expand_icon = "v" if is_expanded else ">"
            selector = "*" if is_selected else " "
            header_plain = f"{selector} {expand_icon} [{idx + 1:{index_width}d}] {client.description}"

            header_width = self.term.length(header_plain)
            max_header_width = max(max_header_width, header_width)

            summary_rows.append(summary_data)
            server_rows.append((idx, is_selected, is_expanded, header_plain, summary_data, stats_info))

        non_empty_summaries = [s for s in summary_rows if isinstance(s, dict) and "gpu_count" in s]
        if non_empty_summaries:
            widths = {
                "gpu_digits": max(len(str(s["gpu_count"])) for s in non_empty_summaries),
                "idle_digits": max(len(str(s["idle_count"])) for s in non_empty_summaries),
                "util_digits": max(len(str(s["avg_util"])) for s in non_empty_summaries),
                "mem_width": max(len(str(s["mem_display"])) for s in non_empty_summaries),
                "cpu_cores_digits": max(len(str(s.get("cpu_cores", 0))) for s in non_empty_summaries),
            }
        else:
            widths = None

        for idx, is_selected, is_expanded, header_plain, summary_data, stats_info in server_rows:
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
                if isinstance(user_memory_by_client, dict):
                    user_summary = user_memory_by_client.get(idx, {}) or {}
                else:
                    user_summary = {}
                if user_summary:
                    user_line = f"Users: {self._format_user_memory_totals(user_summary, max_users=12)}"
                    if self.term.length(user_line) > terminal_width:
                        user_line = user_line[: max(0, terminal_width - 3)] + "..."
                    output_lines.append(user_line)
                output_lines.append("")  # Add spacing after expanded server

        # Single write reduces visible flicker vs. clearing + many print() calls
        screen = self.term.home + "\n".join(self.term.clear_eol + line for line in output_lines) + "\n" + self.term.clear_eos
        sys.stdout.write(screen)
        sys.stdout.flush()

    def _keyboard_listener(self, cbreak_context):
        """Real-time keyboard listener thread, monitors keys for navigation and control"""
        try:
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
                            if self.selected_server in getattr(self, "_toggle_disabled_servers", set()):
                                continue
                            if self.selected_server in self.expanded_servers:
                                self.expanded_servers.discard(self.selected_server)
                            else:
                                self.expanded_servers.add(self.selected_server)
                            self.ui_only_refresh = True  # Use cached data for fast UI update
                            self.refresh_needed.set()  # Trigger immediate refresh
                        elif key_lower == 'a':
                            # Expand all servers
                            disabled = getattr(self, "_toggle_disabled_servers", set())
                            self.expanded_servers = set(i for i in range(len(self.pool)) if i not in disabled)
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
        
        # Use cbreak context in main thread for proper cleanup on Ctrl+C
        cbreak_ctx = self.term.cbreak()
        cbreak_ctx.__enter__()
        
        try:
            # Start keyboard listener thread (uses cbreak mode from main thread)
            keyboard_thread = threading.Thread(target=self._keyboard_listener, args=(cbreak_ctx,), daemon=True)
            keyboard_thread.start()

            # Start background data refresh thread (prevents remote SSH calls from blocking UI)
            refresh_thread = threading.Thread(target=self._background_refresh, daemon=True)
            refresh_thread.start()
            
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
            pass  # Silently exit on Ctrl+C
        except Exception as e:
            print(f"\n\n Error occurred: {e}")
        finally:
            self.quit_flag.set()  # Ensure thread exits
            # Explicitly exit cbreak mode to restore terminal state
            try:
                cbreak_ctx.__exit__(None, None, None)
            except Exception:
                pass
            # Print newline to ensure clean prompt
            print()

    def print_once(self):
        """Print GPU stats once and exit (no TUI loop)"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        server_count = len(self.pool)
        server_label = "Server" if server_count == 1 else "Servers"
        print(f"Time: {current_time} | {server_label}: {server_count}")
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80
        if self.compact:
            separator_width = min(80, terminal_width)
        else:
            separator_width = max(20, terminal_width)
        print("-" * separator_width)
        
        # Get stats
        stats_list, raw_stats_by_client = self.get_client_gpus_info(return_raw=True)

        meta = {}
        if isinstance(raw_stats_by_client, dict):
            meta = raw_stats_by_client.get("_nvidb", {}) or {}
        user_memory_by_client = meta.get("user_memory_by_client", {}) or {}
        global_user_memory = meta.get("user_memory_global", {}) or {}
        
        for idx, (client, stats_info) in enumerate(zip(self.pool, stats_list)):
            stats, system_info = raw_stats_by_client.get(idx, (pd.DataFrame(), {}))
            
            # Build summary
            summary = self._get_server_summary(stats, system_info)
            
            # Print header with summary
            print(f"[{idx + 1}] {client.description}  {summary}")
            
            # Print the full stats table
            print(stats_info)
            if isinstance(user_memory_by_client, dict):
                user_summary = user_memory_by_client.get(idx, {}) or {}
            else:
                user_summary = {}
            if user_summary:
                print(f"Users: {self._format_user_memory_totals(user_summary, max_users=12)}")
            print()  # Add spacing after each server

        if global_user_memory:
            print(f"Users (all nodes): {self._format_user_memory_totals(global_user_memory, max_users=12)}")
