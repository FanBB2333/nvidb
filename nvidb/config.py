"""
nvidb configuration module.
Centralized configuration for working directory and other settings.
"""
import json
import os
import secrets
import stat
from pathlib import Path

# Package version
VERSION = "1.10.0"

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
    # Colour theme shared by the TUIs: "classic" is the monitor's plain
    # high-contrast ANSI palette, "muted" softens routine accents in both UIs.
    "theme": "classic",
}

VIEW_SETTING_CHOICES = {
    "mode": ("nodes", "unified"),
    "sort": ("node", "available", "utilization"),
    "filter": ("all", "available", "busy", "errors"),
    "theme": ("classic", "muted"),
}

def get_config_path():
    """Get the path to the config file."""
    return WORKING_DIR / 'config.yml'

def get_db_path():
    """Get the default path to the SQLite database."""
    return WORKING_DIR / 'gpu_log.db'

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
        notify = settings.get("notify")
        settings.update(overrides)
        override = overrides.get("notify")
        if isinstance(notify, dict) and isinstance(override, dict):
            # Merged one level deeper: a queue.yml that only turns off desktop
            # notifications must not drop the command hook set in config.yml.
            settings["notify"] = {**notify, **override}
        elif override is None and notify is not None:
            # An empty `notify:` in queue.yml overrides nothing.
            settings["notify"] = notify
    merged["queue"] = settings
    return merged


_TRUE_STRINGS = {"true", "yes", "on", "1"}
_FALSE_STRINGS = {"false", "no", "off", "0", ""}


def _parse_bool(value) -> bool:
    """Read a boolean setting; a quoted "false" must not count as true."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        raise ValueError(f"not a boolean: {value!r}")
    return bool(value)


def merge_settings(defaults, raw, *, converters=None) -> dict:
    """Overlay known settings on defaults, coercing only declared value types."""
    settings = dict(defaults)
    if not isinstance(raw, dict):
        return settings

    converters = converters or {}
    for key, default in defaults.items():
        if key not in raw:
            continue
        converter = converters.get(key)
        if converter is None and isinstance(default, bool):
            converter = _parse_bool
        try:
            settings[key] = converter(raw[key]) if converter else raw[key]
        except (TypeError, ValueError):
            continue
    return settings


def _dq(value) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def normalize_gpu_indices(value, *, label="gpu_ids"):
    """Validate and normalize an optional GPU-index allowlist."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of non-negative GPU indices")

    normalized = []
    seen = set()
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{label} must contain only non-negative integers")
        if index not in seen:
            normalized.append(index)
            seen.add(index)
    return normalized


def server_gpu_ids(server, *, context=""):
    """Which of a server's GPUs nvidb uses, or None for all of them.

    One key decides this everywhere - the monitor's tables and the
    queue's placements alike:

        servers:
          - nickname: "shared-box"
            gpu_ids: [2]   # ignore every other GPU on this host

    Listed indices are the host's own NVML indices and keep those numbers
    everywhere nvidb shows them, so GPU 2 stays "GPU 2" rather than being
    renumbered to 0. `context` prefixes the error message so a bad value
    names its server.
    """
    if not isinstance(server, dict):
        return None
    return normalize_gpu_indices(
        server.get("gpu_ids"), label=f"{context} gpu_ids".strip()
    )


def format_servers_yaml(servers) -> str:
    """Render the `servers` section with one readable block per server."""
    if not servers:
        # A bare `servers:` would read back as None rather than a list.
        return "servers: []\n"
    lines = ["servers:"]
    for i, server in enumerate(servers):
        if i > 0:
            lines.append("")
        hostname = server.get("hostname") or server.get("host") or ""
        lines.append(f"  - hostname: {_dq(hostname)}")

        port = server.get("port", 22)
        try:
            port_int = int(port)
        except (TypeError, ValueError):
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

        gpu_ids = server_gpu_ids(server)
        if gpu_ids is not None:
            lines.append(f"    gpu_ids: {json.dumps(gpu_ids)}")

    return "\n".join(lines) + "\n"


def format_config_yaml(cfg: dict) -> str:
    """Render the whole config file: `basic`, other sections, then `servers`."""
    import yaml

    header_cfg = {"basic": cfg.get("basic", {"compact": False})}
    header_cfg.update((key, value) for key, value in cfg.items() if key not in {"basic", "servers"})

    header_text = yaml.safe_dump(
        header_cfg,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).rstrip("\n")
    servers_text = format_servers_yaml(cfg.get("servers") or []).rstrip("\n")
    return f"{header_text}\n\n{servers_text}\n"


def write_text_atomic(path: Path, text: str, *, mode=None):
    """Replace a file in one step so an interrupted write cannot truncate it.

    Each write goes through its own temporary file, so concurrent writers never
    unlink or truncate each other's. Without an explicit `mode` an existing
    file keeps its permissions (a config.yml holding passwords may be 0600);
    a new one gets the umask's.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode is None and path.exists():
        mode = stat.S_IMODE(path.stat().st_mode)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    # O_EXCL rather than tempfile.mkstemp: mkstemp's 0600 would override the
    # umask a new file is meant to get.
    descriptor = os.open(str(temp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with open(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        if mode is not None:
            temp_path.chmod(mode)
        os.replace(str(temp_path), str(path))
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def write_config(cfg: dict, config_path=None):
    """Write config.yml, creating the working directory when needed."""
    path = Path(config_path or get_config_path()).expanduser()
    write_text_atomic(path, format_config_yaml(cfg))


def _choice(options):
    def convert(value):
        if isinstance(value, str) and value in options:
            return value
        raise ValueError(f"not one of {options}: {value!r}")
    return convert


_VIEW_CONVERTERS = {key: _choice(options) for key, options in VIEW_SETTING_CHOICES.items()}


def normalize_view_settings(raw) -> dict:
    """Merge stored view settings over the defaults, dropping invalid values."""
    return merge_settings(DEFAULT_VIEW_SETTINGS, raw, converters=_VIEW_CONVERTERS)


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


def save_view_settings(settings, config_path=None):
    """Persist TUI view settings, keeping the rest of config.yml untouched."""
    path = Path(config_path or get_config_path()).expanduser()
    settings = normalize_view_settings(settings)
    if not path.exists():
        write_config({"view": settings}, path)
        return
    text = path.read_text(encoding="utf-8")
    updated = _replace_top_level_block(text, "view", _render_view_block(settings))
    if updated != text:
        write_text_atomic(path, updated)
