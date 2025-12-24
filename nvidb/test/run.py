import yaml
import getpass
import logging
import argparse
import shutil
import json
import re
from pathlib import Path
from paramiko import SSHConfig
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


def _load_config_yaml(config_path=None) -> dict:
    config_path = Path(config_path or config.get_config_path()).expanduser()
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _get_basic_compact(cfg: dict) -> bool:
    basic = (cfg or {}).get("basic", {})
    if not isinstance(basic, dict):
        basic = {}
    return bool(basic.get("compact", False))


def _is_specific_ssh_host(name: str) -> bool:
    if not name:
        return False
    if name.startswith("!"):
        return False
    if any(ch in name for ch in ["*", "?"]):
        return False
    return True


def _format_import_candidate(candidate: dict) -> str:
    nickname = candidate.get("nickname") or candidate.get("hostname")
    hostname = candidate.get("hostname")
    port = candidate.get("port")
    username = candidate.get("username")
    identityfile = candidate.get("identityfile")
    label = f"{nickname} ({username}@{hostname}:{port})"
    if identityfile:
        label += f" [key: {identityfile}]"
    return label


def _prompt_import_selection(candidates):
    if not candidates:
        return []
    
    from blessed import Terminal
    
    term = Terminal()
    selected = [True] * len(candidates)
    cursor_pos = 0
    
    def draw_menu():
        # Clear screen area and draw menu
        lines = []
        lines.append("Select servers to import (use arrow keys to navigate, space to toggle):")
        lines.append("")
        
        for idx, candidate in enumerate(candidates):
            mark = "[x]" if selected[idx] else "[ ]"
            label = _format_import_candidate(candidate)
            line = f"  {mark} {idx + 1}. {label}"
            
            if idx == cursor_pos:
                lines.append(term.reverse(line))
            else:
                lines.append(line)
        
        lines.append("")
        lines.append("Space: toggle | a: all | n: none | i: invert | s/Enter: save | q: cancel")
        
        # Move cursor up to redraw
        total_lines = len(lines)
        print(term.move_up(total_lines) + term.clear_eos, end='')
        for line in lines:
            print(line)
        
    # Print initial empty lines to reserve space
    total_lines = len(candidates) + 4
    for _ in range(total_lines):
        print()
    
    cbreak_ctx = term.cbreak()
    cbreak_ctx.__enter__()
    
    try:
        with term.hidden_cursor():
            draw_menu()
            
            while True:
                key = term.inkey(timeout=None)
                
                if key.code == term.KEY_UP or key == 'k':
                    cursor_pos = max(0, cursor_pos - 1)
                elif key.code == term.KEY_DOWN or key == 'j':
                    cursor_pos = min(len(candidates) - 1, cursor_pos + 1)
                elif key == ' ':  # Space to toggle
                    selected[cursor_pos] = not selected[cursor_pos]
                elif key.lower() == 'a':  # Select all
                    selected[:] = [True] * len(candidates)
                elif key.lower() == 'n':  # Select none
                    selected[:] = [False] * len(candidates)
                elif key.lower() == 'i':  # Invert selection
                    selected[:] = [not s for s in selected]
                elif key.lower() == 's' or key.code in (term.KEY_ENTER, 10, 13) or key == '\n' or key == '\r':  # Save
                    return selected
                elif key.lower() == 'q' or key.code == term.KEY_ESCAPE:  # Cancel
                    return None
                
                draw_menu()
    except KeyboardInterrupt:
        return None
    finally:
        try:
            cbreak_ctx.__exit__(None, None, None)
        except Exception:
            pass
        print()  # Clean newline


def import_ssh_config(ssh_config_path=None, config_path=None):
    config_path = config_path or config.get_config_path()
    ssh_config_path = Path(ssh_config_path or "~/.ssh/config").expanduser()
    if not ssh_config_path.exists():
        print(f"\nSSH config not found: {ssh_config_path}")
        return

    ssh_config = SSHConfig()
    with open(ssh_config_path, "r") as f:
        ssh_config.parse(f)

    hostnames = sorted(name for name in ssh_config.get_hostnames() if _is_specific_ssh_host(name))
    if not hostnames:
        print("\nNo importable hosts found in SSH config.")
        return

    candidates = []
    for alias in hostnames:
        data = ssh_config.lookup(alias) or {}
        hostname = data.get("hostname") or alias
        port = data.get("port", 22)
        try:
            port = int(port)
        except Exception:
            port = 22
        username = data.get("user") or getpass.getuser()
        identityfile = None
        identityfiles = data.get("identityfile")
        if isinstance(identityfiles, (list, tuple)):
            if identityfiles:
                identityfile = identityfiles[0]
        elif isinstance(identityfiles, str):
            identityfile = identityfiles

        candidate = {
            "hostname": hostname,
            "port": port,
            "username": username,
            "nickname": alias,
            "auth": "auto",
        }
        if identityfile:
            candidate["identityfile"] = identityfile
        candidates.append(candidate)

    selection = _prompt_import_selection(candidates)
    if selection is None:
        print("\nImport cancelled.")
        return

    selected_candidates = [c for c, chosen in zip(candidates, selection) if chosen]
    if not selected_candidates:
        print("\nNo servers selected for import.")
        return

    confirm = input(f"\nImport {len(selected_candidates)} server(s) into {config_path}? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes"):
        print("\nImport cancelled.")
        return

    config_path = Path(config_path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
    else:
        cfg = {}

    servers = cfg.get("servers", []) or []
    existing_keys = set()
    for server in servers:
        normalized = ServerListInfo._normalize_server_dict(dict(server))
        host = normalized.get("host") or normalized.get("hostname")
        port = normalized.get("port", 22)
        try:
            port = int(port)
        except Exception:
            port = 22
        username = normalized.get("username")
        existing_keys.add((host, port, username))

    added = 0
    skipped = 0
    for candidate in selected_candidates:
        key = (candidate.get("hostname"), candidate.get("port"), candidate.get("username"))
        if key in existing_keys:
            skipped += 1
            continue
        servers.append(candidate)
        existing_keys.add(key)
        added += 1

    cfg["servers"] = servers
    _write_config_yaml(config_path, cfg)

    print(f"\nImport complete: {added} added, {skipped} skipped (already exists).")

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

        identityfile = server.get("identityfile")
        if auth in ("auto", "key") and identityfile:
            lines.append(f"    identityfile: {_dq(identityfile)}")

    return "\n".join(lines) + "\n"


def _format_config_yaml(cfg: dict) -> str:
    if not isinstance(cfg, dict):
        cfg = {}

    servers = cfg.get("servers", []) or []

    header_cfg = {}
    if "basic" in cfg:
        header_cfg["basic"] = cfg.get("basic")
    else:
        header_cfg["basic"] = {"compact": False}
    for key, value in cfg.items():
        if key in {"basic", "servers"}:
            continue
        header_cfg[key] = value

    header_text = yaml.safe_dump(
        header_cfg,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).rstrip("\n")
    servers_text = _format_servers_yaml(servers).rstrip("\n")
    return f"{header_text}\n\n{servers_text}\n"


def _write_config_yaml(config_path: Path, cfg: dict):
    config_path = Path(config_path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_format_config_yaml(cfg), encoding="utf-8")


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

    identityfile = None
    if auth in ("auto", "key"):
        identityfile_input = input("IdentityFile (SSH private key path, optional): ").strip()
        if identityfile_input:
            identityfile = identityfile_input
    
    # Password (optional)
    password = None
    
    # Create the server info
    server_info = ServerInfo(
        host=host,
        port=port,
        username=username,
        description=nickname,
        identityfile=identityfile,
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
    if auth in ("auto", "key"):
        print(f"  IdentityFile:{' (default)' if not identityfile else ' ' + identityfile}")
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
        if identityfile and auth in ("auto", "key"):
            server_dict['identityfile'] = identityfile
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


def list_log_sessions(db_path=None):
    """List all log sessions from the database."""
    import sqlite3
    from datetime import datetime
    
    db_path = Path(db_path or config.get_db_path())
    
    if not db_path.exists():
        print(f"\nDatabase file not found: {db_path}")
        print("Run 'nvidb log' to start logging first.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                s.id,
                s.start_time,
                s.end_time,
                s.status,
                s.interval_seconds,
                s.include_remote,
                COUNT(g.id) as log_count
            FROM log_sessions s
            LEFT JOIN gpu_logs g ON s.id = g.session_id
            GROUP BY s.id
            ORDER BY s.id DESC
        ''')
        
        sessions = cursor.fetchall()
        conn.close()
        
        if not sessions:
            print("\nNo log sessions found.")
            return
        
        print("\n" + "-" * 70)
        print("                         Log Sessions")
        print("-" * 70)
        print(f"{'ID':>4}  {'Start Time':<20}  {'End Time':<20}  {'Status':<10}  {'Records':>8}")
        print("-" * 70)
        
        for session in sessions:
            session_id, start_time, end_time, status, interval, include_remote, log_count = session
            
            # Format times
            try:
                start_dt = datetime.fromisoformat(start_time)
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                start_str = start_time[:19] if start_time else "N/A"
            
            if end_time:
                try:
                    end_dt = datetime.fromisoformat(end_time)
                    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    end_str = end_time[:19] if end_time else "N/A"
            else:
                end_str = "(running)"
            
            print(f"{session_id:>4}  {start_str:<20}  {end_str:<20}  {status:<10}  {log_count:>8}")
        
        print("-" * 70)
        print(f"\nTotal: {len(sessions)} session(s)")
        
    except Exception as e:
        print(f"\nError reading database: {e}")


def show_log_info(session_id=None, db_path=None):
    """Show statistics for a log session."""
    import sqlite3
    from datetime import datetime
    
    db_path = Path(db_path or config.get_db_path())
    
    if not db_path.exists():
        print(f"\nDatabase file not found: {db_path}")
        print("Run 'nvidb log' to start logging first.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # If no session_id provided, use the latest session
        if session_id is None:
            cursor.execute("SELECT id FROM log_sessions ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            if not result:
                print("\nNo log sessions found.")
                conn.close()
                return
            session_id = result[0]
        
        # Get session info
        cursor.execute('''
            SELECT id, start_time, end_time, status, interval_seconds, include_remote
            FROM log_sessions WHERE id = ?
        ''', (session_id,))
        
        session = cursor.fetchone()
        if not session:
            print(f"\nSession {session_id} not found.")
            conn.close()
            return
        
        sid, start_time, end_time, status, interval, include_remote = session
        
        # Get log count
        cursor.execute("SELECT COUNT(*) FROM gpu_logs WHERE session_id = ?", (session_id,))
        log_count = cursor.fetchone()[0]
        
        # Calculate duration
        try:
            start_dt = datetime.fromisoformat(start_time)
            if end_time:
                end_dt = datetime.fromisoformat(end_time)
            else:
                end_dt = datetime.now()
            duration = end_dt - start_dt
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        except:
            duration_str = "N/A"
        
        print("\n" + "-" * 60)
        print(f"             Session {session_id} Statistics")
        print("-" * 60)
        
        # Session metadata
        print(f"\nSession Info:")
        print(f"  Start Time:     {start_time}")
        print(f"  End Time:       {end_time or '(still running)'}")
        print(f"  Duration:       {duration_str}")
        print(f"  Status:         {status}")
        print(f"  Interval:       {interval} seconds")
        print(f"  Remote:         {'Yes' if include_remote else 'No'}")
        print(f"  Total Records:  {log_count}")
        
        if log_count == 0:
            print("\nNo log data available for analysis.")
            conn.close()
            return
        
        # Get unique nodes
        cursor.execute('''
            SELECT DISTINCT node FROM gpu_logs WHERE session_id = ?
        ''', (session_id,))
        nodes = [row[0] for row in cursor.fetchall()]
        
        print(f"\n  Nodes:          {', '.join(nodes)}")
        
        # Parse processes to get user statistics
        cursor.execute('''
            SELECT timestamp, node, gpu_id, processes, memory_used
            FROM gpu_logs 
            WHERE session_id = ? AND processes != '-' AND processes != 'N/A'
        ''', (session_id,))
        
        process_rows = cursor.fetchall()
        
        # Track user GPU time and memory usage
        user_time = {}  # user -> total_seconds
        user_max_memory = {}  # user -> max_memory_mb
        
        for timestamp, node, gpu_id, processes, memory_used in process_rows:
            if not processes or processes in ('-', 'N/A'):
                continue
            
            # Parse processes string like "user1:1234MB user2:567MB"
            for part in processes.split():
                if ':' in part:
                    try:
                        user, mem_str = part.split(':', 1)
                        # Extract memory value
                        mem_mb = 0
                        if 'MB' in mem_str.upper():
                            mem_mb = float(mem_str.upper().replace('MB', '').replace('MIB', ''))
                        elif 'GB' in mem_str.upper():
                            mem_mb = float(mem_str.upper().replace('GB', '').replace('GIB', '')) * 1024
                        
                        key = f"{user}@{node}"
                        
                        # Add time (interval seconds per record)
                        user_time[key] = user_time.get(key, 0) + interval
                        
                        # Track max memory
                        if key not in user_max_memory or mem_mb > user_max_memory[key]:
                            user_max_memory[key] = mem_mb
                    except:
                        continue
        
        # Display top users by GPU time
        if user_time:
            print("\n" + "-" * 60)
            print("Top Users by GPU Time:")
            print("-" * 60)
            
            sorted_by_time = sorted(user_time.items(), key=lambda x: x[1], reverse=True)[:10]
            for user, seconds in sorted_by_time:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                secs = seconds % 60
                if hours > 0:
                    time_str = f"{hours}h {minutes}m {secs}s"
                elif minutes > 0:
                    time_str = f"{minutes}m {secs}s"
                else:
                    time_str = f"{secs}s"
                print(f"  {user:<30}  {time_str}")
        
        # Display top users by memory usage
        if user_max_memory:
            print("\n" + "-" * 60)
            print("Top Users by Max Memory Usage:")
            print("-" * 60)
            
            sorted_by_mem = sorted(user_max_memory.items(), key=lambda x: x[1], reverse=True)[:10]
            for user, mem_mb in sorted_by_mem:
                if mem_mb >= 1024:
                    mem_str = f"{mem_mb/1024:.1f} GB"
                else:
                    mem_str = f"{mem_mb:.0f} MB"
                print(f"  {user:<30}  {mem_str}")
        
        if not user_time and not user_max_memory:
            print("\n  No user process data available.")
        
        print("\n" + "-" * 60)
        
        conn.close()
        
    except Exception as e:
        print(f"\nError analyzing session: {e}")


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
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Use compact display (do not stretch columns to fill terminal width)',
    )
    
    subparsers = parser.add_subparsers(dest='command')
    ls_parser = subparsers.add_parser('ls', help='List configured servers')
    ls_parser.add_argument('--detail', action='store_true', help='Show detailed server information')
    add_parser = subparsers.add_parser('add', help='Add a server interactively')
    import_parser = subparsers.add_parser('import', help='Import servers from SSH config')
    import_parser.add_argument('path', nargs='?', default=None, help='Path to SSH config (default: ~/.ssh/config)')
    info_parser = subparsers.add_parser('info', help='Show configuration info')
    log_parser = subparsers.add_parser('log', help='Log GPU stats to SQLite database')
    # Also accept `--remote` after the subcommand (e.g. `nvidb log --remote`)
    log_parser.add_argument('--remote', action='store_true', default=argparse.SUPPRESS, help='Use remote servers')
    log_parser.add_argument('--interval', type=int, default=5, help='Logging interval in seconds (default: 5)')
    log_parser.add_argument('--db-path', type=str, default=None, help='Database path (default: $WORKING_DIR/gpu_log.db)')
    log_subparsers = log_parser.add_subparsers(dest='log_command')
    log_ls_parser = log_subparsers.add_parser('ls', help='List all log sessions')
    log_ls_parser.add_argument('--db-path', type=str, default=None, help='Database path')
    log_info_parser = log_subparsers.add_parser('info', help='Show statistics for a log session')
    log_info_parser.add_argument('session_id', nargs='?', type=int, default=None, help='Session ID (default: latest)')
    log_info_parser.add_argument('--db-path', type=str, default=None, help='Database path')
    # Deprecated alias (use `nvidb web` instead)
    log_web_parser = log_subparsers.add_parser('web', help='(deprecated) Use `nvidb web`')
    log_web_parser.add_argument('--db-path', type=str, default=None, help='Database path')
    log_web_parser.add_argument('--port', type=int, default=8501, help='Streamlit port (default: 8501)')
    web_parser = subparsers.add_parser('web', help='Open web dashboard (Live + Logs)')
    web_parser.add_argument('--db-path', type=str, default=None, help='Database path (default: $WORKING_DIR/gpu_log.db)')
    web_parser.add_argument('--port', type=int, default=8501, help='Streamlit port (default: 8501)')
    clean_parser = subparsers.add_parser('clean', help='Clean server configurations or log data')
    clean_parser.add_argument('target', nargs='?', default=None, help="'all' to delete everything")
    args = parser.parse_args()

    cfg = _load_config_yaml()
    compact = bool(args.compact or _get_basic_compact(cfg))

    server_list = None
    remote_requested = bool(getattr(args, "remote", False))
    if remote_requested:
        if args.command is None:
            server_list = init()
        elif args.command == "log" and getattr(args, "log_command", None) not in {"ls", "info", "web"}:
            server_list = init()
    
    if args.command == 'ls':
        show_servers(detail=args.detail)
    elif args.command == 'add':
        interactive_add_server()
    elif args.command == 'import':
        import_ssh_config(args.path)
    elif args.command == 'info':
        show_info()
    elif args.command == 'log':
        db_path = getattr(args, 'db_path', None)
        if args.log_command == 'ls':
            list_log_sessions(db_path=db_path)
        elif args.log_command == 'info':
            show_log_info(session_id=args.session_id, db_path=db_path)
        elif args.log_command == 'web':
            print("Deprecated: use `nvidb web` instead of `nvidb log web`.")
            from ..web import run_streamlit_app
            run_streamlit_app(
                db_path=db_path,
                port=args.port,
            )
        else:
            # Default: start logging
            run_sqlite_logger(
                server_list=server_list,
                interval=args.interval,
                db_path=db_path
            )
    elif args.command == 'web':
        from ..web import run_streamlit_app
        run_streamlit_app(
            db_path=getattr(args, "db_path", None),
            port=args.port,
        )
    elif args.command == 'clean':
        interactive_clean(clean_all=(args.target == 'all'))
    else:
        # Default action: run interactive monitoring
        pool = NVClientPool(server_list, compact=compact)
        if args.once:
            pool.print_once()
        else:
            pool.print_refresh()

if __name__ == "__main__":
    # python -m nvidb.test.run
    print("Running test")
    main()
    print("Test complete")
