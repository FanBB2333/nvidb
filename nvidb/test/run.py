import pytest
import os
import yaml
import logging
import argparse
from ..src.connection import RemoteClient, NviClientPool
from ..src.data_modules import ServerInfo, ServerListInfo


cli: RemoteClient = None

def init(config_path = '~/.nvidb/config.yml'):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    global test_server, cli
    # config_path = 'nvidb/test/config.yml'
    config_path = os.path.expanduser(config_path)
    # mkdir if not exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # test_server = ServerInfo(**config['servers'][0])
    # server_list = ServerList.from_dict(config['servers'])
    server_list: ServerListInfo = ServerListInfo.from_yaml(config_path)
    cli = RemoteClient(server_list[0])
    return server_list


def test_connection():
    cli.connect()

def test_get_os_info():
    logging.info(msg=cli.get_os_info())

def test_get_gpu_stats():
    # s = cli.get_gpu_stats(command="nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv")
    s = cli.get_gpu_stats()
    logging.info(msg=s)

def test_get_all_stats(server_list):
    pool = NviClientPool(server_list)
    pool.print_refresh()
    # pool.execute_command(command='nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv')
    # pool.execute_command(command='nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.memory,utilization.gpu --format=csv,nounits')
    # pool.execute_command(command='nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.memory,utilization.gpu,memory.used,memory.total,power.draw --format=csv')
    # pool.execute_command(command='gpustat')
    # pool.execute_command_parse('nvidia-smi -q -x', type='xml')
    
    
def main():
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    server_list = init()
    test_get_all_stats(server_list)
    # 创建主解析器
    parser = argparse.ArgumentParser(prog="nvidb", description="A simple tool to manage Nvidia GPU servers.")
    
    # 添加子解析器
    subparsers = parser.add_subparsers(dest='command')  # 子命令存储在 args.command 中

    # 添加 `ls` 子命令
    ls_parser = subparsers.add_parser('ls', help='List items')
    ls_parser.add_argument('--detail', action='store_true', help='Show detailed list')

    # 添加 `add` 子命令
    add_parser = subparsers.add_parser('add', help='Add an item')
    add_parser.add_argument('name', type=str, help='Name of the item to add')

    # 解析命令行参数
    args = parser.parse_args()

    # 根据不同的子命令执行不同的功能
    if args.command == 'ls':
        if args.detail:
            print("Showing detailed list of items.")
        else:
            print("Showing list of items.")
    elif args.command == 'add':
        print(f"Adding item: {args.name}")
    else:
        parser.print_help()

if __name__ == "__main__":
    # python -m nvidb.test.run
    print("Running test")
    server_list = init()
    test_get_all_stats(server_list)
    print("Test complete")