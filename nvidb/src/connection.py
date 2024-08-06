from typing import Literal
import logging
import sys
import os
import subprocess
import getpass
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
        self.connect()
    
    def __del__(self):
        self.client.close()
        logging.info(msg=f"Connection to {self.host}:{self.port} closed.")

    def connect(self) -> None:
        if "auto" == self.auth:
            try:
                if self.password is not None:
                    self.client.connect(hostname=self.host, port=self.port, username=self.username, password=self.password)
                else:
                    self.client.connect(hostname=self.host, port=self.port, username=self.username)
                logging.info(msg=f"Connected to {self.host}:{self.port} as {self.username}")
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
        result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True)
        response = result.stdout
        root = ET.fromstring(response)
        gpus = root.findall('gpu')
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

