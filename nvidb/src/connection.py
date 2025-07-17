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

import pynvml
import paramiko
from paramiko import AuthenticationException
from paramiko.client import SSHClient, AutoAddPolicy
from paramiko.ssh_exception import NoValidConnectionsError
import pandas as pd
from termcolor import colored, cprint
from .data_modules import ServerInfo, ServerListInfo
from .utils import xml_to_dict, num_from_str, units_from_str, extract_numbers


def nvidbInit():
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print("GPU", i, "Name:", name)
        print("GPU", i, "Temperature:", temperature, "C")


class RemoteClient:
    def __init__(self, server: ServerInfo):
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
        
    def test(self):
        # test with ls command
        stdin, stdout, stderr = self.client.exec_command(command='ls')
        result = stdout.read().decode()
        logging.info(msg=f"Test result: {result}")
        return result
    
    
    def get_os_info(self) -> str:
        stdin, stdout, stderr = self.client.exec_command(command='uname -a')
        result = stdout.read().decode()
        return result
    
    def get_gpu_stats(self, command = 'nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv') -> str:
        stdin, stdout, stderr = self.client.exec_command(command=command)
        stats = pd.read_csv(filepath_or_buffer=stdout, header=0)
        return stats
    
    def get_full_gpu_info(self):
        stdin, stdout, stderr = self.client.exec_command(command='nvidia-smi -q -x')
        root = ET.fromstring(stdout.read().decode())
        gpus = root.findall('gpu')
        stats = []
        for gpu in gpus:
            product_name = gpu.find('product_name').text
            product_architecture = gpu.find('product_architecture').text
            
            pci = gpu.find('pci')
            tx_util = pci.find('tx_util').text
            rx_util = pci.find('rx_util').text
            fan_speed = gpu.find('fan_speed').text
            
            fb_memory_usage = gpu.find('fb_memory_usage')
            total = fb_memory_usage.find('total').text
            used = fb_memory_usage.find('used').text
            free = fb_memory_usage.find('free').text
            
            utilization = gpu.find('utilization')
            gpu_util = utilization.find('gpu_util').text
            memory_util = utilization.find('memory_util').text
            
            temperature = gpu.find('temperature')
            gpu_temp = temperature.find('gpu_temp').text

            gpu_power_readings = gpu.find('gpu_power_readings')
            power_state = gpu_power_readings.find('power_state').text
            power_draw = gpu_power_readings.find('power_draw').text
            current_power_limit = gpu_power_readings.find('current_power_limit').text
            
            processes = gpu.find('processes')
            # add new line to the dataframe
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
    
    def execute_command(self, command: str) -> str:
        stdin, stdout, stderr = self.client.exec_command(command=command)
        result = stdout.read().decode()
        return result
    
    def get_client(self) -> SSHClient:
        return self.client


class LocalClient:
    def __init__(self):
        self.host = "localhost"
        self.port = "local"
        self.username = getpass.getuser()
        self.description = f"Local Machine ({self.username}@{self.host})"
        
    def connect(self) -> bool:
        """Local connection is always successful"""
        logging.info(msg=f"Connected to local machine as {self.username}")
        print(f"Connected to local machine as {self.username}")
        return True
    
    def test(self):
        """Test local connection with ls command"""
        try:
            result = subprocess.run(['ls'], capture_output=True, text=True, check=True)
            output = result.stdout
            logging.info(msg=f"Test result: {output}")
            return output
        except subprocess.CalledProcessError as e:
            logging.error(msg=f"Test command failed: {e}")
            return ""
    
    def get_os_info(self) -> str:
        """Get local operating system information"""
        try:
            result = subprocess.run(['uname', '-a'], capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            logging.error(msg=f"Failed to get OS info: {e}")
            return ""
    
    def get_gpu_stats(self, command='nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv') -> pd.DataFrame:
        """Get local GPU statistics"""
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
            # Use StringIO to simulate file object for pandas
            from io import StringIO
            stats = pd.read_csv(StringIO(result.stdout), header=0)
            return stats
        except subprocess.CalledProcessError as e:
            logging.error(msg=f"Failed to get GPU stats: {e}")
            return pd.DataFrame()
        except FileNotFoundError:
            logging.error(msg="nvidia-smi command not found")
            return pd.DataFrame()
    
    def get_full_gpu_info(self):
        """Get complete GPU information"""
        try:
            result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True, check=True)
            root = ET.fromstring(result.stdout)
            gpus = root.findall('gpu')
            stats = []
            
            for gpu in gpus:
                product_name = gpu.find('product_name').text
                product_architecture = gpu.find('product_architecture').text
                
                pci = gpu.find('pci')
                tx_util = pci.find('tx_util').text
                rx_util = pci.find('rx_util').text
                fan_speed = gpu.find('fan_speed').text
                
                fb_memory_usage = gpu.find('fb_memory_usage')
                total = fb_memory_usage.find('total').text
                used = fb_memory_usage.find('used').text
                free = fb_memory_usage.find('free').text
                
                utilization = gpu.find('utilization')
                gpu_util = utilization.find('gpu_util').text
                memory_util = utilization.find('memory_util').text
                
                temperature = gpu.find('temperature')
                gpu_temp = temperature.find('gpu_temp').text

                gpu_power_readings = gpu.find('gpu_power_readings')
                power_state = gpu_power_readings.find('power_state').text
                power_draw = gpu_power_readings.find('power_draw').text
                current_power_limit = gpu_power_readings.find('current_power_limit').text
                
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
            
        except subprocess.CalledProcessError as e:
            logging.error(msg=f"Failed to get full GPU info: {e}")
            return pd.DataFrame()
        except FileNotFoundError:
            logging.error(msg="nvidia-smi command not found")
            return pd.DataFrame()
        except ET.ParseError as e:
            logging.error(msg=f"Failed to parse XML output: {e}")
            return pd.DataFrame()
    
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
    
    def get_client(self):
        """Return self to maintain interface consistency with RemoteClient"""
        return self


class NviClientPool:
    def __init__(self, server_list: ServerListInfo, add_local: bool = True):
        self.pool = [LocalClient()] + [RemoteClient(server) for server in server_list]
        logging.info(msg=f"Initialized pool with {len(self.pool)} clients.")
        self.connect_all()
        self.term = Terminal()
    
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
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 0)
        stats_str = []
        for idx, client in enumerate(self.pool):
            # logging.info(msg=colored(f"{client.description}", 'yellow'))
            # print(colored(f"{client.description}", 'yellow'))
            stats = client.get_full_gpu_info()
            # apply from_str to rx_util, tx_util, power_state, power_draw, current_power_limit, used, total, free
            stats['rx/tx'] = [f"{'/'.join(extract_numbers(row['rx_util']))}/{'/'.join(extract_numbers(row['tx_util']))}" for _, row in stats.iterrows()]
            stats['power'] = [f"{row['power_state']} {'/'.join(extract_numbers(row['power_draw']))}/{'/'.join(extract_numbers(row['current_power_limit']))}" for _, row in stats.iterrows()]
            stats['memory[used/total]'] = [f"{'/'.join(extract_numbers(row['used']))}/{'/'.join(extract_numbers(row['total']))}" for _, row in stats.iterrows()]

            # rename columns: product_name -> name, gpu_temp -> temp, fan_speed -> fan, memory_util -> mem_util, gpu_util -> gpu_util
            stats = stats.rename(columns={'product_name': 'name', 'gpu_temp': 'temp', 'fan_speed': 'fan', 'memory_util': 'mem', 'gpu_util': 'util'})
            
            # replace the NVIDIA/GeForce with "" in name column
            stats['name'] = stats['name'].str.replace('NVIDIA', '').str.replace('GeForce', '').str.strip()
            
            # remove rows: product_architecture, rx_util, tx_util, power_state, power_draw, current_power_limit, used, total, free
            stats = stats.drop(columns=['product_architecture', 'rx_util', 'tx_util', 'power_state', 'power_draw', 'current_power_limit', 'used', 'total', 'free'])

            stats_str.append(stats)
        # reformat the str into a single string
        stats_str = [f"\n{colored(client.description)}\n{str(stats)}" for client, stats in zip(self.pool, stats_str)]
        return stats_str
    
    def print_stats(self):
        stats_str = self.get_client_gpus_info()
        print(self.term.clear)
        for stats in stats_str:
            print(stats)

    def print_refresh(self):
        while True:
            # print(self.term.clear)
            self.print_stats()
            # time.sleep(1)

