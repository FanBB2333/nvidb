from typing import Literal
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


class NviClient:
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


class NviClientPool:
    def __init__(self, server_list: ServerListInfo):
        self.pool = [NviClient(server) for server in server_list]
        logging.info(msg=f"Initialized pool with {len(self.pool)} clients.")
        self.connect_all()
    
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
        for idx, client in enumerate(self.pool):
            logging.info(msg=colored(f"{client.description}", 'yellow'))
            print(colored(f"{client.description}", 'yellow'))
            stats = client.get_full_gpu_info()
            # apply from_str to rx_util, tx_util, power_state, power_draw, current_power_limit, used, total, free
            stats['rx/tx'] = [f"{'/'.join(extract_numbers(row['rx_util']))}/{'/'.join(extract_numbers(row['tx_util']))}" for _, row in stats.iterrows()]
            stats['power'] = [f"{row['power_state']} {'/'.join(extract_numbers(row['power_draw']))}/{'/'.join(extract_numbers(row['current_power_limit']))}" for _, row in stats.iterrows()]
            stats['memory[used/total]'] = [f"{'/'.join(extract_numbers(row['used']))}/{'/'.join(extract_numbers(row['total']))}" for _, row in stats.iterrows()]

            # remove rows: product_architecture, rx_util, tx_util, power_state, power_draw, current_power_limit, used, total, free
            stats = stats.drop(columns=['product_architecture', 'rx_util', 'tx_util', 'power_state', 'power_draw', 'current_power_limit', 'used', 'total', 'free'])

            print(stats)

