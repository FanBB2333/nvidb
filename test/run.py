import pytest
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from src.connection import NviClient, NviClientPool
from src.data_modules import ServerInfo, ServerListInfo


cli: NviClient = None
server_list: ServerListInfo = None

def init():
    global test_server, cli, server_list
    with open('test/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # test_server = ServerInfo(**config['servers'][0])
    # server_list = ServerList.from_dict(config['servers'])
    server_list = ServerListInfo.from_yaml('test/config.yml')
    cli = NviClient(server_list[0])


def test_connection():
    cli.connect()

def test_get_os_info():
    logging.info(msg=cli.get_os_info())

def test_get_gpu_stats():
    # s = cli.get_gpu_stats(command="nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv")
    s = cli.get_gpu_stats()
    logging.info(msg=s)

def test_get_all_stats():
    server_list
    


if __name__ == '__main__':
    # pytest.main()
    init()
    test_get_gpu_stats()