import yaml
import getpass
import logging
import argparse
import shutil
import json
from pathlib import Path
from ..connection import RemoteClient, NVClientPool
from ..data_modules import ServerInfo, ServerListInfo
from ..logger import run_sqlite_logger
from .. import config


cli: RemoteClient = None

def _warn_if_deprecated_config_keys(servers):
    for server in servers or []:
        try:
            ServerListInfo._normalize_server_dict(dict(server))
        except Exception:
            pass

def init(config_path=None):
    config_path = config_path or config.get_config_path()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    global test_server, cli
    # config_path = 'nvidb/test/config.yml'
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # test_server = ServerInfo(**config['servers'][0])
    # server_list = ServerList.from_dict(config['servers'])
    server_list: ServerListInfo = ServerListInfo.from_yaml(config_path)
    cli = RemoteClient(server_list[0])
    return server_list


def _dq(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _format_servers_yaml(servers) -> str:
    lines = ["servers:"]

    for i, server in enumerate(servers):
        if i > 0:
            lines.append("")
        hostname = server.get("hostname") or server.get("host") or ""
        lines.append(f"  - hostname: {_dq(hostname)}")

        port = server.get("port", 22)
        try:
            port_int = int(port)
        except Exception:
            port_int = 22
        lines.append(f"    port: {port_int}")

        username = server.get("username")
        if username is not None:
            lines.append(f"    username: {_dq(username)}")

        password = server.get("password")
        if password:
            lines.append(f"    password: {_dq(password)}")

        nickname = server.get("nickname")
        if nickname is None:
            nickname = server.get("description")
        if nickname is not None:
            lines.append(f"    nickname: {_dq(nickname)}")

        auth = server.get("auth", "auto")
        if auth is not None:
            lines.append(f"    auth: {_dq(auth)}")

    return "\n".join(lines) + "\n"


def _write_config_yaml(config_path: Path, cfg: dict):
    config_path = Path(config_path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    servers = cfg.get("servers", []) or []
    config_path.write_text(_format_servers_yaml(servers), encoding="utf-8")


def interactive_add_server(config_path=None):
    """Interactively add a new server to the configuration."""
    config_path = config_path or config.get_config_path()
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
    
    # Nickname (optional, has default)
    default_desc = f"{username}@{host}:{port}"
    nickname = input(f"Nickname [{default_desc}]: ").strip()
    if not nickname:
        nickname = default_desc
    
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
    
    # Create the server info
    server_info = ServerInfo(
        host=host,
        port=port,
        username=username,
        description=nickname,
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
    print(f"  Nickname:    {nickname}")
    print(f"  Auth:        {auth}")
    print(f"  Password:    {'***' if password else '(not set)'}")
    print("-" * 50)
    
    # Confirm and save
    confirm = input("\nSave this server configuration? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        config_path = Path(config_path).expanduser()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config or create new one
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
        else:
            cfg = {}
        
        if 'servers' not in cfg:
            cfg['servers'] = []
        
        # Add the new server
        server_dict = {
            'hostname': host,
            'port': port,
            'username': username,
            'nickname': nickname,
            'auth': auth
        }
        if password:
            server_dict['password'] = password
        
        cfg['servers'].append(server_dict)
        
        _write_config_yaml(config_path, cfg)
        
        print(f"\nServer added successfully to {config_path}")
    else:
        print("\nOperation cancelled.")


def _get_directory_size(path):
    """Calculate the total size of a directory in bytes."""
    total_size = 0
    path = Path(path)
    if not path.exists():
        return 0
    for file_path in path.rglob('*'):
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except (OSError, PermissionError):
                pass
    return total_size


def _format_size(size_bytes):
    """Format size in bytes to human-readable string."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} bytes"


def show_info(config_path=None):
    """Show configuration information."""
    config_path = config_path or config.get_config_path()
    
    print("\n" + "-" * 50)
    print("         nvidb Configuration Info")
    print("-" * 50 + "\n")
    
    # Working directory info
    print(f"Working Directory: {config.WORKING_DIR}")
    
    # Disk usage statistics
    if Path(config.WORKING_DIR).exists():
        total_size = _get_directory_size(config.WORKING_DIR)
        print(f"   Disk Usage: {_format_size(total_size)}")
    else:
        print(f"   Disk Usage: 0 bytes (directory not created)")
    print()
    
    # Config file info
    print(f"Config File: {config_path}")
    
    if not Path(config_path).exists():
        print(f"   Status: Not found")
    else:
        print(f"   Status: Exists")
    
    # Database file info
    db_path = config.get_db_path()
    print(f"\nDatabase File: {db_path}")
    if Path(db_path).exists():
        print(f"   Status: Exists")
    else:
        print(f"   Status: Not created yet")
    
    print("\n" + "-" * 50)


def show_servers(config_path=None, detail=False):
    """Show configured server list."""
    config_path = config_path or config.get_config_path()
    
    if not Path(config_path).exists():
        print("\nNo servers configured yet.")
        print("Run 'nvidb add' to add your first server.")
        return
    
    # Load server info
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
    except Exception as e:
        print(f"Error reading config: {e}")
        return
    
    servers = cfg.get('servers', [])
    _warn_if_deprecated_config_keys(servers)
    server_count = len(servers)
    
    if server_count == 0:
        print("\nNo servers configured yet.")
        print("Run 'nvidb add' to add a server.")
        return
    
    print("\n" + "-" * 50)
    print(f"         Server List ({server_count} servers)")
    print("-" * 50)
    
    for idx, server in enumerate(servers):
        host = server.get('hostname') or server.get('host', 'N/A')
        port = server.get('port', 22)
        username = server.get('username', 'N/A')
        nickname = server.get('nickname') or server.get('description', f'{username}@{host}:{port}')
        
        if detail:
            auth = server.get('auth', 'auto')
            has_password = 'Yes' if server.get('password') else 'No'
            
            print(f"\n  [{idx + 1}] {nickname}")
            print(f"      Host:     {host}:{port}")
            print(f"      User:     {username}")
            print(f"      Auth:     {auth}")
            print(f"      Password: {has_password}")
        else:
            print(f"  [{idx + 1}] {nickname} ({host}:{port})")
    
    print("\n" + "-" * 50)


def interactive_clean(clean_all=False):
    """Interactively clean server configurations or log data."""
    config_path = config.get_config_path()
    db_path = config.get_db_path()
    working_dir = config.WORKING_DIR
    
    if clean_all:
        # Clean all: delete entire working directory
        print("\n" + "=" * 50)
        print("         Clean All Data")
        print("=" * 50 + "\n")
        
        print(f"This will delete the entire working directory:")
        print(f"  {working_dir}")
        print("\nThis includes:")
        print(f"  - Configuration file: {config_path}")
        print(f"  - Database file: {db_path}")
        print("  - All other files in the directory")
        
        confirm = input("\nAre you sure you want to delete ALL data? [y/N]: ").strip().lower()
        if confirm in ['y', 'yes']:
            confirm2 = input("Type 'DELETE' to confirm: ").strip()
            if confirm2 == 'DELETE':
                if Path(working_dir).exists():
                    shutil.rmtree(working_dir)
                    print(f"\nDeleted: {working_dir}")
                else:
                    print(f"\nDirectory does not exist: {working_dir}")
            else:
                print("\nOperation cancelled.")
        else:
            print("\nOperation cancelled.")
        return
    
    # Interactive clean menu
    print("\n" + "=" * 50)
    print("         Clean Data")
    print("=" * 50 + "\n")
    
    print("What would you like to clean?")
    print("  1. Remove a server from configuration")
    print("  2. Delete log database")
    print("  3. Cancel")
    
    choice = input("\nEnter your choice [1-3]: ").strip()
    
    if choice == '1':
        # Remove a server
        _clean_server(config_path)
    elif choice == '2':
        # Delete database
        _clean_database(db_path)
    else:
        print("\nOperation cancelled.")


def _clean_server(config_path):
    """Remove a server from configuration."""
    if not Path(config_path).exists():
        print(f"\nConfiguration file not found: {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
    except Exception as e:
        print(f"\nError reading config: {e}")
        return
    
    servers = cfg.get('servers', [])
    _warn_if_deprecated_config_keys(servers)
    if not servers:
        print("\nNo servers configured.")
        return
    
    print("\nConfigured servers:")
    print("-" * 50)
    for idx, server in enumerate(servers):
        host = server.get('hostname') or server.get('host', 'N/A')
        port = server.get('port', 22)
        nickname = server.get('nickname') or server.get('description', f"{server.get('username', 'N/A')}@{host}:{port}")
        print(f"  [{idx + 1}] {nickname} ({host}:{port})")
    
    print(f"  [0] Cancel")
    
    try:
        choice = input("\nEnter the number of the server to remove: ").strip()
        if choice == '0' or not choice:
            print("\nOperation cancelled.")
            return
        
        idx = int(choice) - 1
        if idx < 0 or idx >= len(servers):
            print("\nInvalid selection.")
            return
        
        server = servers[idx]
        server_host = server.get('hostname') or server.get('host', 'N/A')
        nickname = server.get('nickname') or server.get('description', f"{server.get('username', 'N/A')}@{server_host}")
        
        confirm = input(f"\nRemove server '{nickname}'? [y/N]: ").strip().lower()
        if confirm in ['y', 'yes']:
            servers.pop(idx)
            cfg['servers'] = servers
            
            _write_config_yaml(config_path, cfg)
            
            print(f"\nServer removed successfully.")
        else:
            print("\nOperation cancelled.")
            
    except ValueError:
        print("\nInvalid input.")


def _clean_database(db_path):
    """Delete the log database."""
    db_path = Path(db_path)
    
    if not db_path.exists():
        print(f"\nDatabase file not found: {db_path}")
        return
    
    # Show database info
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get session count
        cursor.execute("SELECT COUNT(*) FROM log_sessions")
        session_count = cursor.fetchone()[0]
        
        # Get log count
        cursor.execute("SELECT COUNT(*) FROM gpu_logs")
        log_count = cursor.fetchone()[0]
        
        # Get file size
        file_size = db_path.stat().st_size
        if file_size >= 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.2f} MB"
        elif file_size >= 1024:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size} bytes"
        
        conn.close()
        
        print(f"\nDatabase: {db_path}")
        print(f"  Size: {size_str}")
        print(f"  Sessions: {session_count}")
        print(f"  Log entries: {log_count}")
        
    except Exception as e:
        print(f"\nDatabase: {db_path}")
        print(f"  Error reading database: {e}")
    
    confirm = input("\nDelete this database? [y/N]: ").strip().lower()
    if confirm in ['y', 'yes']:
        db_path.unlink()
        print(f"\nDatabase deleted successfully.")
    else:
        print("\nOperation cancelled.")


def test_connection():
    cli.connect()

def test_get_os_info():
    logging.info(msg=cli.get_os_info())

def test_get_gpu_stats():
    # s = cli.get_gpu_stats(command="nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv")
    s = cli.get_gpu_stats()
    logging.info(msg=s)

def test_get_all_stats(server_list):
    pool = NVClientPool(server_list)
    pool.print_refresh()
    # pool.execute_command(command='nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv')
    # pool.execute_command(command='nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.memory,utilization.gpu --format=csv,nounits')
    # pool.execute_command(command='nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.memory,utilization.gpu,memory.used,memory.total,power.draw --format=csv')
    # pool.execute_command(command='gpustat')
    # pool.execute_command_parse('nvidia-smi -q -x', type='xml')
    
    
def main():
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建主解析器
    parser = argparse.ArgumentParser(prog="nvidb", description="A simple tool to manage NVIDIA GPU servers.")
    parser.add_argument('--version', action='version', version=f'nvidb {config.VERSION}')
    parser.add_argument('--remote', action='store_true', help='Use remote servers')
    parser.add_argument('--once', action='store_true', help='Print GPU stats once and exit (no TUI loop)')
    
    subparsers = parser.add_subparsers(dest='command')
    ls_parser = subparsers.add_parser('ls', help='List configured servers')
    ls_parser.add_argument('--detail', action='store_true', help='Show detailed server information')
    add_parser = subparsers.add_parser('add', help='Add a server interactively')
    info_parser = subparsers.add_parser('info', help='Show configuration info')
    log_parser = subparsers.add_parser('log', help='Log GPU stats to SQLite database')
    log_parser.add_argument('--interval', type=int, default=5, help='Logging interval in seconds (default: 5)')
    log_parser.add_argument('--db-path', type=str, default=None, help='Database path (default: $WORKING_DIR/gpu_log.db)')
    clean_parser = subparsers.add_parser('clean', help='Clean server configurations or log data')
    clean_parser.add_argument('target', nargs='?', default=None, help="'all' to delete everything")
    args = parser.parse_args()
    
    if args.remote:
        server_list = init()
    else:
        server_list = None
    
    if args.command == 'ls':
        show_servers(detail=args.detail)
    elif args.command == 'add':
        interactive_add_server()
    elif args.command == 'info':
        show_info()
    elif args.command == 'log':
        run_sqlite_logger(
            server_list=server_list,
            interval=args.interval,
            db_path=getattr(args, 'db_path', None)
        )
    elif args.command == 'clean':
        interactive_clean(clean_all=(args.target == 'all'))
    else:
        # Default action: run interactive monitoring
        pool = NVClientPool(server_list)
        if args.once:
            pool.print_once()
        else:
            pool.print_refresh()

if __name__ == "__main__":
    # python -m nvidb.test.run
    print("Running test")
    main()
    print("Test complete")
