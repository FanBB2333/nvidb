import pytest
import os
import yaml
import logging
import argparse
from ..connection import RemoteClient, NviClientPool
from ..data_modules import ServerInfo, ServerListInfo


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


def interactive_add_server(config_path='~/.nvidb/config.yml'):
    """Interactively add a new server to the configuration."""
    print("\n" + "=" * 50)
    print("       Add New Server Configuration")
    print("=" * 50 + "\n")
    
    # Host (required)
    while True:
        host = input("Host (IP or hostname): ").strip()
        if host:
            break
        print("  ⚠ Host is required. Please enter a valid host.")
    
    # Port (required, default 22)
    while True:
        port_input = input("Port [22]: ").strip()
        if not port_input:
            port = 22
            break
        try:
            port = int(port_input)
            if 1 <= port <= 65535:
                break
            print("  ⚠ Port must be between 1 and 65535.")
        except ValueError:
            print("  ⚠ Please enter a valid integer for port.")
    
    # Username (required)
    while True:
        username = input("Username: ").strip()
        if username:
            break
        print("  ⚠ Username is required.")
    
    # Description (optional, has default)
    default_desc = f"{username}@{host}:{port}"
    description = input(f"Description [{default_desc}]: ").strip()
    if not description:
        description = default_desc
    
    # Auth method
    print("\nAuthentication method:")
    print("  1. auto (try key first, then password)")
    print("  2. key (SSH key only)")
    print("  3. password (password only)")
    while True:
        auth_choice = input("Choose auth method [1]: ").strip()
        if not auth_choice or auth_choice == '1':
            auth = 'auto'
            break
        elif auth_choice == '2':
            auth = 'key'
            break
        elif auth_choice == '3':
            auth = 'password'
            break
        print("  ⚠ Please enter 1, 2, or 3.")
    
    # Password (optional)
    password = None
    if auth in ['auto', 'password']:
        password_prompt = "Password (leave empty for key-based auth): " if auth == 'auto' else "Password: "
        password = getpass.getpass(password_prompt)
        if not password:
            password = None
    
    # Create the server info
    server_info = ServerInfo(
        host=host,
        port=port,
        username=username,
        description=description,
        password=password,
        auth=auth
    )
    
    # Display summary
    print("\n" + "-" * 50)
    print("Server Configuration Summary:")
    print("-" * 50)
    print(f"  Host:        {host}")
    print(f"  Port:        {port}")
    print(f"  Username:    {username}")
    print(f"  Description: {description}")
    print(f"  Auth:        {auth}")
    print(f"  Password:    {'***' if password else '(not set)'}")
    print("-" * 50)
    
    # Confirm and save
    confirm = input("\nSave this server configuration? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        config_path = os.path.expanduser(config_path)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Load existing config or create new one
        import yaml
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader) or {}
        else:
            config = {}
        
        if 'servers' not in config:
            config['servers'] = []
        
        # Add the new server
        server_dict = {
            'host': host,
            'port': port,
            'username': username,
            'description': description,
            'auth': auth
        }
        if password:
            server_dict['password'] = password
        
        config['servers'].append(server_dict)
        
        # Save to file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n✓ Server added successfully to {config_path}")
    else:
        print("\n✗ Operation cancelled.")


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
    
    # 创建主解析器
    parser = argparse.ArgumentParser(prog="nvidb", description="A simple tool to manage Nvidia GPU servers.")
    parser.add_argument('--remote', action='store_true', help='Use remote servers')
    
    subparsers = parser.add_subparsers(dest='command')
    ls_parser = subparsers.add_parser('ls', help='List items')
    ls_parser.add_argument('--detail', action='store_true', help='Show detailed list')
    add_parser = subparsers.add_parser('add', help='Add a server interactively')
    args = parser.parse_args()
    
    if args.remote:
        server_list = init()
    else:
        server_list = None
    
    if args.command == 'ls':
        if args.detail:
            print("Showing detailed list of items.")
        else:
            print("Showing list of items.")
    elif args.command == 'add':
        interactive_add_server()
    else:
        # Default action: run interactive monitoring
        pool = NviClientPool(server_list)
        pool.print_refresh()

if __name__ == "__main__":
    # python -m nvidb.test.run
    print("Running test")
    main()
    print("Test complete")