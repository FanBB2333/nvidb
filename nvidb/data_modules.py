import logging
import sys
import os
from typing import Optional
from typing import Literal
import getpass
from dataclasses import dataclass, asdict, field


import paramiko
from paramiko import AuthenticationException
from paramiko.client import SSHClient, AutoAddPolicy
from paramiko.ssh_exception import NoValidConnectionsError
import pandas as pd


    
# product_name = gpu.find('product_name').text
# product_architecture = gpu.find('product_architecture').text

# pci = gpu.find('pci')
# tx_util = pci.find('tx_util').text
# rx_util = pci.find('rx_util').text
# fan_speed = gpu.find('fan_speed').text

# fb_memory_usage = gpu.find('fb_memory_usage')
# total = fb_memory_usage.find('total').text
# used = fb_memory_usage.find('used').text
# free = fb_memory_usage.find('free').text

# utilization = gpu.find('utilization')
# gpu_util = utilization.find('gpu_util').text
# memory_util = utilization.find('memory_util').text

# temperature = gpu.find('temperature')
# gpu_temp = temperature.find('gpu_temp').text

# gpu_power_readings = gpu.find('gpu_power_readings')
# power_state = gpu_power_readings.find('power_state').text
# power_draw = gpu_power_readings.find('power_draw').text
# current_power_limit = gpu_power_readings.find('current_power_limit').text

# processes = gpu.find('processes')

@dataclass
class GPUProcess:
    pid: str
    username: str
    gpu_memory: str
    gpu_util: str
    memory_util: str
    command: str
    gpu_name: str
    gpu_index: str
    
    def __post_init__(self):
        self.gpu_memory = int(self.gpu_memory.replace('MiB', '').strip())
        self.gpu_util = int(self.gpu_util.replace('%', '').strip())
        self.memory_util = int(self.memory_util.replace('%', '').strip())
        self.gpu_index = int(self.gpu_index)
        
    def __repr__(self):
        return f'GPUProcess(pid={self.pid}, username={self.username}, gpu_memory={self.gpu_memory}, gpu_util={self.gpu_util}, memory_util={self.memory_util}, command={self.command}, gpu_name={self.gpu_name}, gpu_index={self.gpu_index})'
    
    def __str__(self):
        return f'PID: {self.pid}, User: {self.username}, GPU Memory: {self.gpu_memory}, GPU Util: {self.gpu_util}, Memory Util: {self.memory_util}, Command: {self.command}, GPU Name: {self.gpu_name}, GPU Index: {self.gpu_index}'
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, process):
        return cls(**process)

@dataclass
class GPUInfo:
    index: str
    product_name: str
    product_architecture: str
    tx_util: str
    rx_util: str
    fan_speed: str
    mem_total: str
    mem_used: str
    mem_free: str
    gpu_util: str
    memory_util: str
    gpu_temp: str
    power_state: str
    power_draw: str
    current_power_limit: str
    processes: list[GPUProcess]
    
@dataclass
class ServerInfo:
    host: str
    port: int
    username: str
    description: Optional[str] = None
    identityfile: Optional[str] = None
    password: str = field(repr=False, default=None)
    auth: Literal['password', 'key', 'auto'] = 'auto'
    
    def __post_init__(self):
        if self.description is None:
            self.description = f'{self.username}@{self.host}:{self.port}'

# List of ServerInfo
class ServerListInfo:
    _deprecated_warnings_emitted = set()

    def __init__(self):
        self.servers = []
        
    def add_server(self, server_info):
        self.servers.append(server_info)
    
    def remove_server(self, idx):
        del self.servers[idx]
    
    def __iter__(self):
        return iter(self.servers)
    
    def __len__(self):
        return len(self.servers)
    
    def __getitem__(self, index):
        return self.servers[index]
    
    def __repr__(self):
        return f'ServerList({self.servers})'
    
    def __str__(self):
        return '\n'.join([f"{idx}: {server.description}" for idx, server in enumerate(self.servers)])
    
    def to_dict(self):
        servers = []
        for server in self.servers:
            data = asdict(server)
            data["hostname"] = data.pop("host", None)
            data["nickname"] = data.pop("description", None)
            for key in [k for k, v in data.items() if v is None]:
                data.pop(key, None)
            servers.append(data)
        return servers

    @classmethod
    def _warn_deprecated_key(cls, old_key: str, new_key: str):
        token = (old_key, new_key)
        if token in cls._deprecated_warnings_emitted:
            return
        cls._deprecated_warnings_emitted.add(token)
        logging.warning("Config key `%s` is deprecated; please use `%s` instead.", old_key, new_key)

    @staticmethod
    def _normalize_server_dict(server: dict) -> dict:
        server = dict(server or {})
        hostname = server.get("hostname")
        if "host" in server:
            ServerListInfo._warn_deprecated_key("host", "hostname")
        if hostname is not None:
            server["host"] = hostname
        server.pop("hostname", None)

        nickname = server.get("nickname")
        if "description" in server:
            ServerListInfo._warn_deprecated_key("description", "nickname")
        if nickname is not None:
            server["description"] = nickname
        server.pop("nickname", None)
        return server
    
    @classmethod
    def from_dict(cls, server_list):
        instance = cls()
        for server in server_list:
            instance.add_server(ServerInfo(**cls._normalize_server_dict(server)))
        return instance
    
    @classmethod
    def from_yaml(cls, file):
        import yaml
        with open(file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_dict((config or {}).get('servers', []))
    
    def to_yaml(self, file):
        import yaml
        with open(file, 'w') as f:
            yaml.dump({"servers": self.to_dict()}, f, sort_keys=False)
