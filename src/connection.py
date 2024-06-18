from typing import Literal
import logging
import sys
import os
import getpass

import pynvml
import paramiko
from paramiko import AuthenticationException
from paramiko.client import SSHClient, AutoAddPolicy
from paramiko.ssh_exception import NoValidConnectionsError

from data_modules import ServerInfo


def nvidbInit():
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print("GPU", i, "Name:", name)
        print("GPU", i, "Temperature:", temperature, "C")


def NviClient():
    def __init__(self, server: ServerInfo):
        self.host = server.host
        self.port = server.port
        self.username = server.username
        self.auth = server.auth
        self.description = server.description
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()


    def connect(self, auth: Literal['password', 'key'] = 'password') -> None:
        if "password" == auth:
            try:
                # prompt to input password
                password = getpass.getpass(prompt=f'Enter password for {self.username}@{self.host}:{self.port} -> ')
                self.client.connect(hostname=self.host, port=self.port, username=self.username, password=password)
            except AuthenticationException as e:
                logging.error(msg=f"Authentication failed: {e}")
                sys.exit(1)
            except NoValidConnectionsError as e:
                logging.error(msg=f"Connection failed: {e}")
                sys.exit(1)
        elif "key" == auth:
            pass
    
    def test(self):
        # test with ls command
        stdin, stdout, stderr = self.client.exec_command(command='ls')
        print(stdout.read().decode())
    
    
    def get_client(self) -> SSHClient:
        return self.client

    
    def close(self) -> None:
        self.client.close()
        logging.info(msg=f"Connection to {self.host}:{self.port} closed.")
        

def NviClientPool():
    def __init__(self, host: str, port: int, username: str, password: str, pool_size: int):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.pool_size = pool_size
        self.client_pool = [NviClient(host=host, port=port, username=username, password=password) for _ in range(pool_size)]

    def test(self):
        pass