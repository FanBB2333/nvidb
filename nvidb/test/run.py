import pytest
import os
import yaml
import logging
from ..src.connection import NviClient, NviClientPool
from ..src.data_modules import ServerInfo, ServerListInfo


cli: NviClient = None
server_list: ServerListInfo = None

def init():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    global test_server, cli, server_list
    # config_path = 'nvidb/test/config.yml'
    config_path = '~/.nvidb/config.yml'
    config_path = os.path.expanduser(config_path)
    # mkdir if not exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # test_server = ServerInfo(**config['servers'][0])
    # server_list = ServerList.from_dict(config['servers'])
    server_list = ServerListInfo.from_yaml(config_path)
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
    pool = NviClientPool(server_list)
    # pool.execute_command(command='nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv')
    # pool.execute_command(command='nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.memory,utilization.gpu --format=csv,nounits')
    pool.execute_command(command='nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.memory,utilization.gpu,memory.used,memory.total,power.draw --format=csv')
    # pool.execute_command(command='gpustat')
    
    
def main():
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    init()
    test_get_all_stats()

if __name__ == "__main__":
    init()
    test_get_all_stats()