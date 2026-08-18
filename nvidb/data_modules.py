import logging
from typing import List, Optional
from typing import Literal
from dataclasses import dataclass, asdict, field

from .config import normalize_gpu_indices


@dataclass
class ServerInfo:
    host: str
    port: int
    username: str
    description: Optional[str] = None
    identityfile: Optional[str] = None
    password: str = field(repr=False, default=None)
    auth: Literal['password', 'key', 'auto'] = 'auto'
    proxyjump: Optional[str] = None
    # Only these GPU indices are nvidb's to watch and schedule onto; None
    # means the whole host. `gpus` is the older spelling of the same key.
    gpu_ids: Optional[List[int]] = None
    gpus: Optional[List[int]] = None

    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = self.gpus
        self.gpu_ids = normalize_gpu_indices(self.gpu_ids)
        self.gpus = self.gpu_ids
        if self.description is None:
            self.description = f'{self.username}@{self.host}:{self.port}'

# List of ServerInfo
class ServerListInfo:
    _deprecated_warnings_emitted = set()

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

    def to_dict(self):
        servers = []
        for server in self.servers:
            data = asdict(server)
            data["hostname"] = data.pop("host", None)
            data["nickname"] = data.pop("description", None)
            # `gpus` is only an input alias; one canonical key goes back out.
            data.pop("gpus", None)
            for key in [k for k, v in data.items() if v is None]:
                data.pop(key, None)
            servers.append(data)
        return servers

    def __str__(self):
        return '\n'.join([f"{idx}: {server.description}" for idx, server in enumerate(self.servers)])

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
