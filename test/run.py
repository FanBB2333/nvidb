import pytest
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from src.connection import NviClient
from src.data_modules import ServerInfo, ServerList


cli: NviClient = None

def init():
    global test_server, cli
    with open('test/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # test_server = ServerInfo(**config['servers'][0])
    # server_list = ServerList.from_dict(config['servers'])
    server_list = ServerList.from_yaml('test/config.yaml')
    cli = NviClient(server_list[0])


def test_connection():
    cli.connect()

def test_get_os_info():
    logging.info(msg=cli.get_os_info())

def test_get_gpu_stats():
    s = cli.get_gpu_stats()
    logging.info(msg=s)

def test_get_info():
    pass


if __name__ == '__main__':
    # pytest.main()
    init()
    # test_get_gpu_stats()