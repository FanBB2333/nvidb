import pytest
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from src.connection import NviClient
from src.data_modules import ServerInfo


test_server: ServerInfo = None

def init():
    global test_server
    with open('test/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    test_server = ServerInfo(**config['servers'][0])


def test_connection():
    cli: NviClient = NviClient(test_server)
    cli.connect()
    cli.test()

def test_get_os_info():
    cli = NviClient(test_server)
    cli.connect()
    logging.info(msg=cli.get_os_info())

def test_get_gpu_stats():
    cli = NviClient(test_server)
    s = cli.get_gpu_stats()
    print(s)

def test_get_info():
    pass


if __name__ == '__main__':
    # pytest.main()
    init()
    test_get_gpu_stats()