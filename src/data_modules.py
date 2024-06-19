from typing import Literal
from dataclasses import dataclass, asdict, field


@dataclass
class ServerInfo:
    host: str
    port: int
    username: str
    description: str
    auth: Literal['password', 'key', 'auto'] = 'auto'
    password: str = field(init=False, repr=False)
    
    def __post_init__(self):
        if self.description is None:
            self.description = f'{self.username}@{self.host}:{self.port}'


class ServerList:
    def __init__(self):
        self.servers = []
        
    def add_server(self, server_info):
        self.servers.append(server_info)
        
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
        return [asdict(server) for server in self.servers]
    
    @classmethod
    def from_dict(cls, server_list):
        instance = cls()
        for server in server_list:
            instance.add_server(ServerInfo(**server))
        return instance
    
    @classmethod
    def from_yaml(cls, file):
        import yaml
        with open(file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_dict(config['servers'])