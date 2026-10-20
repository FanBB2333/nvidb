"""The keeper: a shell script in `~/.nvidb` that keeps the queue moving.

Every nvidb client advances the queue by ticking, so a queue whose clients come
and go still works - as long as somebody runs one. That stops being true the
moment the point of the queue is to keep scheduling while nobody is connected:
a job submitted before a laptop is closed stays `pending` until some client
looks again.

The keeper closes that gap on whichever machine holds the queue. Its portable
mode is a small `sh` supervisor because these machines are often ones the user
does not administer. Where a systemd user session is available, the same
commands can manage a user service that returns after reboot.
"""
from __future__ import annotations

import os
import plistlib
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config as nvidb_config

SCRIPT_NAME = "queue-keeper.sh"
PID_NAME = "queue-keeper.pid"
LOG_NAME = "queue-keeper.log"
LOCK_NAME = "queue-keeper.lock"
TOKEN_NAME = "queue-keeper.token"
MANAGER_NAME = "queue-keeper.manager"
STOPPED_NAME = "queue-keeper.stopped"
SYSTEMD_UNIT_NAME = "nvidb-queue.service"
LAUNCHD_LABEL = "com.nvidb.queue-keeper"

DEFAULT_INTERVAL = 15
ACTIONS = ("start", "ensure", "stop", "restart", "status", "logs")
MANAGERS = ("shell", "systemd", "launchd")

TEMPLATE_PATH = Path(__file__).with_name("keeper.sh")


def home() -> Path:
    return Path(nvidb_config.WORKING_DIR).expanduser()


def script_path() -> Path:
    return home() / SCRIPT_NAME


def pid_file() -> Path:
    return home() / PID_NAME


def log_file() -> Path:
    return home() / LOG_NAME


def token_file() -> Path:
    return home() / TOKEN_NAME


def manager_file() -> Path:
    return home() / MANAGER_NAME


def stopped_file() -> Path:
    return home() / STOPPED_NAME


def launchd_unit_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def default_manager() -> str:
    if manager_file().exists():
        return manager()
    if sys.platform == "darwin":
        return "launchd"
    if shutil.which("systemctl"):
        try:
            if _systemctl("show-environment").returncode == 0:
                return "systemd"
        except (OSError, subprocess.SubprocessError):
            pass
    return "shell"


def systemd_unit_path() -> Path:
    """The current user's systemd unit directory, respecting XDG_CONFIG_HOME."""
    configured = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(configured).expanduser() if configured else Path.home() / ".config"
    return config_home / "systemd" / "user" / SYSTEMD_UNIT_NAME


def manager() -> str:
    try:
        value = manager_file().read_text(encoding="utf-8").strip()
    except OSError:
        return "shell"
    return value if value in MANAGERS else "shell"


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *[str(arg) for arg in args]],
        capture_output=True,
        text=True,
        timeout=15,
    )


def resolve_nvidb_binary(explicit: Optional[str] = None) -> str:
    """Find the absolute path of the `nvidb` the keeper should run.

    An absolute path is baked into the script because the keeper is normally
    installed over SSH, and `ssh host 'nvidb ...'` runs a non-interactive shell
    whose PATH usually lacks the `~/.local/bin` that a `pip install --user`
    writes to. Discovering that at install time is a clear error; discovering it
    at 3am is a queue that silently stopped scheduling.
    """
    candidates: List[str] = []
    if explicit:
        candidates.append(str(Path(explicit).expanduser()))
    else:
        found = shutil.which("nvidb")
        if found:
            candidates.append(found)
        # Running as `nvidb queue keeper install` from a directory that is not
        # on PATH still knows where it came from.
        argv0 = Path(sys.argv[0] or "")
        if argv0.name.startswith("nvidb"):
            candidates.append(str(argv0.resolve()))

    for candidate in candidates:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())

    if explicit:
        raise ValueError(f"{explicit} is not an executable file")
    raise ValueError(
        "could not find the nvidb executable; pass --nvidb /path/to/nvidb "
        "(a `pip install --user` puts it in ~/.local/bin)"
    )


def render(*, nvidb_bin: str, interval: int, manager_name: str, token: str) -> str:
    """Fill the shipped template in for this machine."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("__NVIDB_HOME__", shlex.quote(str(home())))
        .replace("__NVIDB_BIN__", shlex.quote(str(nvidb_bin)))
        .replace("__INTERVAL__", str(int(interval)))
        .replace("__MANAGER__", shlex.quote(manager_name))
        .replace("__KEEPER_TOKEN__", shlex.quote(token))
        .replace("__SYSTEMD_UNIT__", shlex.quote(SYSTEMD_UNIT_NAME))
        .replace("__LAUNCHD_LABEL__", shlex.quote(LAUNCHD_LABEL))
        .replace("__LAUNCHD_PLIST__", shlex.quote(str(launchd_unit_path())))
    )


def render_launchd_unit(*, nvidb_bin: str, interval: int) -> str:
    return plistlib.dumps({
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [nvidb_bin, "queue", "daemon", "--interval", str(int(interval))],
        "EnvironmentVariables": {"NVIDB_HOME": str(home()), "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        "KeepAlive": True,
        "RunAtLoad": True,
        "ProcessType": "Background",
        "ThrottleInterval": 5,
        "StandardOutPath": str(log_file()),
        "StandardErrorPath": str(log_file()),
    }).decode("utf-8")


def render_systemd_unit(*, nvidb_bin: str, interval: int) -> str:
    """Render a user unit that lets systemd supervise the daemon directly."""

    def quote(value: str) -> str:
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
        return f'"{escaped}"'

    return "\n".join(
        [
            "[Unit]",
            "Description=nvidb queue scheduler",
            "",
            "[Service]",
            "Type=simple",
            f"Environment={quote(f'NVIDB_HOME={home()}')}",
            f"ExecStart={quote(nvidb_bin)} queue daemon --interval {int(interval)}",
            "Restart=always",
            "RestartSec=5",
            "TimeoutStopSec=30",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def _write_atomic(path: Path, text: str, *, mode: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        if mode is not None:
            temporary.chmod(mode)
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _command_error(result: subprocess.CompletedProcess) -> str:
    return (result.stderr or result.stdout or f"exit {result.returncode}").strip()


def _daemon_reload() -> None:
    reloaded = _systemctl("daemon-reload")
    if reloaded.returncode != 0:
        raise RuntimeError(
            f"systemctl --user daemon-reload failed: {_command_error(reloaded)}"
        )


def _log_location(manager_name: str) -> str:
    if manager_name == "systemd":
        return f"journalctl --user -u {SYSTEMD_UNIT_NAME}"
    return str(log_file())


def _unit_location(manager_name: str) -> Optional[str]:
    if manager_name == "systemd":
        return str(systemd_unit_path())
    if manager_name == "launchd":
        return str(launchd_unit_path())
    return None


def install(
    *,
    nvidb_bin: Optional[str] = None,
    interval: int = DEFAULT_INTERVAL,
    manager_name: str = "shell",
) -> Dict[str, Any]:
    """Install the shell keeper or an optional systemd user service."""
    if manager_name not in MANAGERS:
        raise ValueError(f"unknown keeper manager: {manager_name}")
    if manager_name == "systemd" and not shutil.which("systemctl"):
        raise ValueError("systemctl is not available; install the shell keeper instead")
    if manager_name == "launchd" and (sys.platform != "darwin" or not shutil.which("launchctl")):
        raise ValueError("launchd requires macOS and launchctl")

    resolved = resolve_nvidb_binary(nvidb_bin)
    target = script_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    previous_manager = manager()

    # Switching supervisors while the old one is still active would leave two
    # daemons ticking the same database. Ask the old installation to stop
    # before replacing the script that knows how to control it.
    if existed and previous_manager != manager_name:
        stopped = run("stop")
        if stopped.returncode != 0:
            raise RuntimeError(
                f"could not stop the existing {previous_manager} keeper: "
                f"{_command_error(stopped)}"
            )

    token = secrets.token_hex(16)
    _write_atomic(
        target,
        render(
            nvidb_bin=resolved,
            interval=interval,
            manager_name=manager_name,
            token=token,
        ),
        mode=0o755,
    )
    _write_atomic(manager_file(), f"{manager_name}\n", mode=0o600)

    unit = systemd_unit_path()
    if manager_name == "launchd":
        _write_atomic(
            launchd_unit_path(),
            render_launchd_unit(nvidb_bin=resolved, interval=interval),
            mode=0o644,
        )
    elif previous_manager == "launchd" and launchd_unit_path().exists():
        launchd_unit_path().unlink()
    if manager_name == "systemd":
        _write_atomic(
            unit,
            render_systemd_unit(nvidb_bin=resolved, interval=interval),
            mode=0o644,
        )
        _daemon_reload()
    elif previous_manager == "systemd" and unit.exists():
        unit.unlink()
        if shutil.which("systemctl"):
            _daemon_reload()

    _reset_status_cache()
    return {
        "script": str(target),
        "nvidb": resolved,
        "interval": int(interval),
        "replaced": existed,
        "manager": manager_name,
        "unit": _unit_location(manager_name),
        "log": _log_location(manager_name),
    }


def _pid() -> Optional[int]:
    try:
        raw = pid_file().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _token() -> Optional[str]:
    try:
        value = token_file().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _shell_process_matches(pid: int, token: Optional[str]) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, OverflowError):
        return False
    try:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    command = result.stdout or ""
    if token:
        return f"_loop {token}" in command
    # Compatibility with a keeper installed before identity tokens existed.
    # The script path still distinguishes it from an unrelated reused pid.
    return f"{script_path()} _loop" in command


def _systemd_status() -> tuple:
    try:
        active = _systemctl("is-active", "--quiet", SYSTEMD_UNIT_NAME)
    except (OSError, subprocess.SubprocessError):
        return False, None
    if active.returncode != 0:
        return False, None
    try:
        shown = _systemctl(
            "show", "--property", "MainPID", "--value", SYSTEMD_UNIT_NAME
        )
        pid = int((shown.stdout or "").strip()) if shown.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        pid = None
    return True, pid or None


def _launchd_status() -> tuple:
    try:
        result = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False, None
    if result.returncode != 0:
        return False, None
    for line in (result.stdout or "").splitlines():
        key, sep, value = line.strip().partition(" = ")
        if key == "pid" and sep and value.isdigit():
            return True, int(value)
    return False, None


_STATUS_CACHE_TTL = 10.0
_status_cache: Dict[str, Any] = {"at": 0.0, "value": None}


def _reset_status_cache() -> None:
    """Tests poke the filesystem directly, so they drop the cache explicitly."""
    _status_cache["at"] = 0.0
    _status_cache["value"] = None


def status() -> Dict[str, Any]:
    """Whether a keeper is installed here and whether it is running.

    Shell mode verifies both the pid and the unique token in its command line,
    so a stale pid file cannot mistake an unrelated process for the keeper.
    The answer costs one or two subprocess spawns and only changes when the
    user starts or stops the keeper, so it is cached for a few seconds.
    """
    now = time.monotonic()
    if _status_cache["value"] is not None and now - _status_cache["at"] < _STATUS_CACHE_TTL:
        return dict(_status_cache["value"])
    manager_name = manager()
    if manager_name == "systemd":
        running, pid = _systemd_status()
    elif manager_name == "launchd":
        running, pid = _launchd_status()
    else:
        pid = _pid()
        running = bool(pid and _shell_process_matches(pid, _token()))
    value = {
        "installed": script_path().exists(),
        "running": running,
        "pid": pid if running else None,
        "manager": manager_name,
        "stopped": stopped_file().exists(),
        "script": str(script_path()),
        "unit": _unit_location(manager_name),
        "log": _log_location(manager_name),
    }
    _status_cache["at"] = now
    _status_cache["value"] = dict(value)
    return value


def run(action: str, *extra: str) -> subprocess.CompletedProcess:
    """Run the installed script. Raises FileNotFoundError when it is missing."""
    target = script_path()
    if not target.exists():
        raise FileNotFoundError(str(target))
    if action == "ensure" and stopped_file().exists():
        return subprocess.CompletedProcess([], 0, "explicitly stopped\n", "")
    if action == "stop":
        _write_atomic(stopped_file(), "stopped\n", mode=0o600)
    elif action in ("start", "restart"):
        stopped_file().unlink(missing_ok=True)
    _reset_status_cache()
    result = subprocess.run(
        ["sh", str(target), action, *[str(item) for item in extra]],
        capture_output=True,
        text=True,
    )
    _reset_status_cache()
    return result


def ensure_local() -> Optional[str]:
    """Recover an installed keeper after a crash, respecting an explicit stop."""
    if not manager_file().exists() or stopped_file().exists():
        return None
    state = status()
    if not state["installed"] or state["running"]:
        return None
    try:
        result = run("ensure")
        return _command_error(result) if result.returncode else None
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)
