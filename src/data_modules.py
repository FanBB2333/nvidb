from dataclasses import dataclass, asdict, field


@dataclass
class ServerInfo:
    host: str
    port: int
    username: str
    auth: str = 'password'
    password: str = field(init=False, repr=False)
    description: str = field(init=False)
    
    def __post_init__(self):
        self.description = f'{self.username}@{self.host}:{self.port}'