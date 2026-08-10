"""
nvidb configuration module.
Centralized configuration for working directory and other settings.
"""
import json
import os
from pathlib import Path

# Package version
VERSION = "1.8.0"

# Default working directory for nvidb data (config, logs, database)
# Can be overridden by NVIDB_HOME environment variable
WORKING_DIR = Path(os.environ.get('NVIDB_HOME', '~/.nvidb')).expanduser()

# TUI display state persisted under the `view` section of config.yml so the
# next run starts with the layout the user left behind.
DEFAULT_VIEW_SETTINGS = {
    "mode": "nodes",
    "detailed": False,
    "sort": "node",
    "filter": "all",
    "processes": False,
    "trends": False,
    "group_by_node": True,
    "hide_unsupported": True,
    "mouse": True,
}

VIEW_SETTING_CHOICES = {
    "mode": ("nodes", "unified"),
    "sort": ("node", "available", "utilization"),
    "filter": ("all", "available", "busy", "errors"),
}

def get_config_path():
    """Get the path to the config file."""
    return WORKING_DIR / 'config.yml'

def get_db_path():
    """Get the default path to the SQLite database."""
    return WORKING_DIR / 'gpu_log.db'

def ensure_working_dir():
    """Ensure the working directory exists."""
    WORKING_DIR.mkdir(parents=True, exist_ok=True)


def get_queue_config_path():
    """Path of the queue's own config file, overridable for tests and sandboxes."""
    override = os.environ.get('NVIDB_QUEUE_CONFIG')
    if override:
        return Path(override).expanduser()
    return WORKING_DIR / 'queue.yml'


def _read_yaml(path, *, strict=False) -> dict:
    """Load a YAML mapping, optionally reporting invalid or unreadable files."""
    import yaml

    path = Path(path).expanduser()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=yaml.FullLoader) or {}
    except Exception as exc:
        if strict:
            raise ValueError(f"could not read YAML configuration {path}: {exc}") from exc
        return {}
    if not isinstance(data, dict):
        if strict:
            raise ValueError(f"YAML configuration {path} must contain a mapping")
        return {}
    return data


def load_config(config_path=None) -> dict:
    """Load config.yml, returning an empty dict when it is missing or invalid."""
    return _read_yaml(config_path or get_config_path())


def load_queue_config(config_path=None) -> dict:
    """Configuration governing the job queue: queue.yml when it exists, else config.yml.

    The queue needs a different server list from the monitor. Every entry it can
    see becomes a node it probes and may dispatch work onto, which is not what
    someone who added a colleague's machine to `nvidb` was asking for. Giving the
    queue its own file lets one `~/.nvidb` serve both; leaving the file out keeps
    the original behaviour exactly.

    The result has the same shape as config.yml, so everything downstream reads
    `servers` and `queue` without knowing which file they came from.
    """
    main = load_config()
    queue_path = Path(config_path or get_queue_config_path()).expanduser()
    if not queue_path.exists():
        return main

    # An invalid queue file must not silently fall back to config.yml: every
    # server in that file is a potential dispatch target.
    queue_cfg = _read_yaml(queue_path, strict=True)
    if not queue_cfg:
        return main

    merged = dict(queue_cfg)

    # `nodes` reads better in a file that is only about the queue, but `servers`
    # keeps blocks copy-pasteable from config.yml. Both are accepted.
    servers = queue_cfg.get("servers")
    if servers is None:
        servers = queue_cfg.get("nodes")
    if servers is None:
        # No list of its own: the queue schedules onto whatever the monitor
        # watches, which is what a single-machine setup wants.
        servers = main.get("servers") or []
    merged.pop("nodes", None)
    merged["servers"] = servers

    settings = dict((main.get("queue") or {}))
    overrides = queue_cfg.get("queue") or {}
    if isinstance(overrides, dict):
        notify = {**(settings.get("notify") or {}), **(overrides.get("notify") or {})}
        settings.update(overrides)
        if notify:
            # Merged one level deeper: a queue.yml that only turns off desktop
            # notifications must not drop the command hook set in config.yml.
            settings["notify"] = notify
    merged["queue"] = settings
    return merged


def _dq(value) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def format_servers_yaml(servers) -> str:
    """Render the `servers` section with one readable block per server."""
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

        proxyjump = server.get("proxyjump")
        if proxyjump:
            lines.append(f"    proxyjump: {_dq(proxyjump)}")

    return "\n".join(lines) + "\n"


def format_config_yaml(cfg: dict) -> str:
    """Render the whole config file: `basic`, other sections, then `servers`."""
    import yaml

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
    servers_text = format_servers_yaml(servers).rstrip("\n")
    return f"{header_text}\n\n{servers_text}\n"


def _write_text_atomic(path: Path, text: str):
    """Replace a file in one step so an interrupted write cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def write_config(cfg: dict, config_path=None):
    """Write config.yml, creating the working directory when needed."""
    path = Path(config_path or get_config_path()).expanduser()
    _write_text_atomic(path, format_config_yaml(cfg))


def normalize_view_settings(raw) -> dict:
    """Merge stored view settings over the defaults, dropping invalid values."""
    settings = dict(DEFAULT_VIEW_SETTINGS)
    if not isinstance(raw, dict):
        return settings

    for key, default in DEFAULT_VIEW_SETTINGS.items():
        if key not in raw:
            continue
        value = raw[key]
        choices = VIEW_SETTING_CHOICES.get(key)
        if choices is not None:
            if isinstance(value, str) and value in choices:
                settings[key] = value
            continue
        if isinstance(default, bool):
            settings[key] = bool(value)
    return settings


def load_view_settings(config_path=None, cfg=None) -> dict:
    """Return persisted TUI view settings merged over the defaults."""
    if cfg is None:
        cfg = load_config(config_path)
    return normalize_view_settings((cfg or {}).get("view"))


def _render_view_block(settings) -> str:
    lines = ["view:"]
    for key in DEFAULT_VIEW_SETTINGS:
        value = settings[key]
        if isinstance(value, bool):
            value = "true" if value else "false"
        lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def _is_block_continuation(line: str) -> bool:
    """True while a line still belongs to the current top-level block."""
    return not line.strip() or line[0].isspace() or line.lstrip().startswith("#")


def _replace_top_level_block(text: str, key: str, block: str) -> str:
    """Swap one top-level YAML block, leaving every other line untouched.

    The view section is rewritten on each key press, so a full re-serialization
    would silently drop the comments users keep next to their server entries.
    """
    lines = text.splitlines(keepends=True)

    start = None
    for index, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            start = index
            break

    if start is None:
        insert_at = len(lines)
        for index, line in enumerate(lines):
            if line.startswith("servers:"):
                insert_at = index
                # Keep comments that introduce the servers section with it.
                while insert_at > 0 and lines[insert_at - 1].lstrip().startswith("#"):
                    insert_at -= 1
                break
        head = "".join(lines[:insert_at])
        if head and not head.endswith("\n"):
            head += "\n"
        tail = "".join(lines[insert_at:])
        return f"{head}{block}\n{tail}" if tail else f"{head}{block}"

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if not _is_block_continuation(lines[index]):
            end = index
            break
    # Blank lines and comments trailing the block belong to whatever follows.
    while end > start + 1 and (
        not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")
    ):
        end -= 1
    return "".join(lines[:start]) + block + "".join(lines[end:])


def save_view_settings(settings, config_path=None) -> bool:
    """Persist TUI view settings, keeping the rest of config.yml untouched."""
    path = Path(config_path or get_config_path()).expanduser()
    settings = normalize_view_settings(settings)
    try:
        if not path.exists():
            write_config({"view": settings}, path)
            return True
        text = path.read_text(encoding="utf-8")
        updated = _replace_top_level_block(text, "view", _render_view_block(settings))
        if updated != text:
            _write_text_atomic(path, updated)
        return True
    except Exception:
        return False
