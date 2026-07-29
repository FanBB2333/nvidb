"""The keeper: a shell script in `~/.nvidb` that keeps the queue moving.

Every nvidb client advances the queue by ticking, so a queue whose clients come
and go still works - as long as somebody runs one. That stops being true the
moment the point of the queue is to keep scheduling while nobody is connected:
a job submitted before a laptop is closed stays `pending` until some client
looks again.

The keeper closes that gap on whichever machine holds the queue. It is a small
`sh` script rather than a service unit because the machines this runs on are
often ones the user does not administer: no root, no lingering user session, no
cron. What it needs is a shell and the ability to leave a process behind.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config as nvidb_config

SCRIPT_NAME = "queue-keeper.sh"
PID_NAME = "queue-keeper.pid"
LOG_NAME = "queue-keeper.log"
SESSION_NAME = "queue-keeper.session"
LOCK_NAME = "queue-keeper.lock"

DEFAULT_INTERVAL = 15
ACTIONS = ("start", "ensure", "stop", "restart", "status", "logs")

TEMPLATE_PATH = Path(__file__).with_name("keeper.sh")


def home() -> Path:
    return Path(nvidb_config.WORKING_DIR).expanduser()


def script_path() -> Path:
    return home() / SCRIPT_NAME


def pid_file() -> Path:
    return home() / PID_NAME


def log_file() -> Path:
    return home() / LOG_NAME


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


def render(*, nvidb_bin: str, interval: int = DEFAULT_INTERVAL) -> str:
    """Fill the shipped template in for this machine."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("__NVIDB_HOME__", str(home()))
        .replace("__NVIDB_BIN__", str(nvidb_bin))
        .replace("__INTERVAL__", str(int(interval)))
    )


def install(
    *,
    nvidb_bin: Optional[str] = None,
    interval: int = DEFAULT_INTERVAL,
) -> Dict[str, Any]:
    """Write the keeper script into the nvidb working directory."""
    resolved = resolve_nvidb_binary(nvidb_bin)
    target = script_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(render(nvidb_bin=resolved, interval=interval), encoding="utf-8")
    target.chmod(0o755)
    return {
        "script": str(target),
        "nvidb": resolved,
        "interval": int(interval),
        "replaced": existed,
        "log": str(log_file()),
    }


def _pid() -> Optional[int]:
    try:
        raw = pid_file().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def status() -> Dict[str, Any]:
    """Whether a keeper is installed here and whether it is running.

    Cheap enough to call on every status render: one small file read and one
    signal-free `kill`.
    """
    pid = _pid()
    running = False
    if pid:
        try:
            os.kill(pid, 0)
            running = True
        except ProcessLookupError:
            running = False
        except PermissionError:
            # Someone else's process holds the pid; not ours to judge, but it
            # certainly exists.
            running = True
        except OSError:
            running = False
    return {
        "installed": script_path().exists(),
        "running": running,
        "pid": pid if running else None,
        "script": str(script_path()),
        "log": str(log_file()),
    }


def run(action: str, *extra: str) -> subprocess.CompletedProcess:
    """Run the installed script. Raises FileNotFoundError when it is missing."""
    target = script_path()
    if not target.exists():
        raise FileNotFoundError(str(target))
    return subprocess.run(
        ["sh", str(target), action, *[str(item) for item in extra]],
        capture_output=True,
        text=True,
    )
