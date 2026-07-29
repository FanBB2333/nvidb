"""Forwarding queue commands to the machine that holds the queue.

When the scheduler lives on a always-on node rather than on the laptop, the
laptop should not open a second, private queue database: there is one queue, and
it is over there. So a client with `remote:` configured stops opening a database
at all and hands the whole command line to the same `nvidb` on the far side.

Running the real CLI remotely, rather than reimplementing it over some protocol,
is what keeps `job submit`, `job wait`, `logs -f`, the TUI and the exit codes
behaving identically whether or not the queue happens to be local.
"""
from __future__ import annotations

import base64
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import config as nvidb_config
from . import keeper as keeper_mod

# Set on the far side so a misconfigured remote pointing at itself fails as a
# plain command instead of looping forever.
NO_REMOTE_ENV = "NVIDB_QUEUE_NO_REMOTE"

# Flags that mean something here and nothing there.
_LOCAL_ONLY_FLAGS = ("--db-path", "--local")

# Stands in for the remote path of a locally-read `--script` file until the
# command line is assembled, since that one argument must reach the far side as
# a shell expansion rather than as a quoted literal.
_SCRIPT_SENTINEL = "\x00nvidb-script\x00"


def load_target(cfg: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """The configured queue host, or None when this machine owns the queue."""
    if os.environ.get(NO_REMOTE_ENV):
        return None
    if cfg is None:
        cfg = nvidb_config.load_queue_config()
    raw = (cfg or {}).get("remote")
    if raw is None:
        raw = ((cfg or {}).get("queue") or {}).get("remote")
    if not isinstance(raw, dict):
        return None
    host = raw.get("host") or raw.get("hostname")
    if not host:
        return None
    port = raw.get("port")
    return {
        "host": str(host),
        "port": int(port) if port else None,
        "username": raw.get("username") or raw.get("user"),
        "nvidb": str(raw.get("nvidb") or "nvidb"),
        # "ensure" starts the keeper as a side effect of any command, which is
        # how a queue host that rebooted comes back without a cron entry.
        "keeper": str(raw.get("keeper") or "ensure"),
        "nvidb_home": raw.get("nvidb_home"),
        "ssh_options": [str(option) for option in (raw.get("ssh_options") or [])],
    }


def split_command(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split at the first `--`, which separates our flags from the job's command.

    Everything after it belongs to the user's command and is passed through
    untouched - a job that itself takes `--db-path` must arrive intact.
    """
    argv = list(argv)
    for index, token in enumerate(argv):
        if token == "--":
            return argv[:index], argv[index:]
    return argv, []


def _strip_local_flags(head: Sequence[str]) -> List[str]:
    out: List[str] = []
    skip = False
    for token in head:
        if skip:
            skip = False
            continue
        if token in _LOCAL_ONLY_FLAGS:
            # --local takes no value; --db-path does. Only the latter consumes
            # the next token.
            skip = token == "--db-path"
            continue
        if any(token.startswith(f"{flag}=") for flag in _LOCAL_ONLY_FLAGS):
            continue
        out.append(token)
    return out


def _point_script_at_temp_file(head: List[str]) -> List[str]:
    """Rewrite `--script FILE` so it names the file we are about to write there."""
    out: List[str] = []
    replace_next = False
    for token in head:
        if replace_next:
            out.append(_SCRIPT_SENTINEL)
            replace_next = False
            continue
        if token == "--script":
            out.append(token)
            replace_next = True
            continue
        if token.startswith("--script="):
            out.append(f"--script={_SCRIPT_SENTINEL}")
            continue
        out.append(token)
    return out


def _quote(token: str) -> str:
    if token == _SCRIPT_SENTINEL:
        return '"$nvidb_script"'
    if token == f"--script={_SCRIPT_SENTINEL}":
        return '--script="$nvidb_script"'
    return shlex.quote(token)


def keeper_script_path(target: Dict[str, Any]) -> str:
    """Where the keeper lives on the far side, as a shell word."""
    home = target.get("nvidb_home")
    if home:
        return f"{str(home).rstrip('/')}/{keeper_mod.SCRIPT_NAME}"
    return f"$HOME/.nvidb/{keeper_mod.SCRIPT_NAME}"


def build_command(
    target: Dict[str, Any],
    argv: Sequence[str],
    *,
    script_text: Optional[str] = None,
) -> str:
    """The shell command to run on the queue host for this invocation."""
    head, tail = split_command(argv)
    head = _strip_local_flags(head)
    if script_text is not None:
        head = _point_script_at_temp_file(head)

    lines: List[str] = []
    if target.get("keeper") == "ensure":
        # Folded into the same command, so keeping the queue alive costs no
        # extra round trip. A keeper that is already up exits immediately.
        lines.append(f"sh {keeper_script_path(target)} ensure >/dev/null 2>&1 || true")

    if script_text is not None:
        # The same base64 hop the executor uses to put a run script on a node:
        # a job body may contain quotes, newlines and non-ASCII.
        encoded = base64.b64encode(script_text.encode("utf-8")).decode("ascii")
        lines.append("nvidb_script=$(mktemp)")
        lines.append(
            f"printf '%s' {shlex.quote(encoded)} | base64 -d > \"$nvidb_script\""
        )

    env = [f"{NO_REMOTE_ENV}=1"]
    if target.get("nvidb_home"):
        env.append(f"NVIDB_HOME={shlex.quote(str(target['nvidb_home']))}")
    command = " ".join(
        [*env, shlex.quote(str(target["nvidb"])), *[_quote(token) for token in [*head, *tail]]]
    )

    if script_text is None:
        lines.append(command)
    else:
        # The temporary file has to go whatever the command did, and the
        # command's status is what the caller is waiting for.
        lines.append("nvidb_rc=0")
        lines.append(f"{command} || nvidb_rc=$?")
        lines.append('rm -f "$nvidb_script"')
        lines.append("exit $nvidb_rc")
    return "\n".join(lines)


def ssh_argv(target: Dict[str, Any], command: str, *, tty: bool = False) -> List[str]:
    argv = ["ssh"]
    if tty:
        argv.append("-t")
    if target.get("port"):
        argv += ["-p", str(target["port"])]
    for option in target.get("ssh_options") or []:
        argv += ["-o", option]
    host = str(target["host"])
    username = target.get("username")
    argv.append(f"{username}@{host}" if username else host)
    argv.append(command)
    return argv


def forward(
    target: Dict[str, Any],
    argv: Sequence[str],
    *,
    tty: bool = False,
    script_text: Optional[str] = None,
) -> int:
    """Run this invocation on the queue host, inheriting our streams.

    stdin, stdout and stderr are passed straight through, so `logs --follow`,
    `job wait` and the TUI behave as if they were running here, and the remote
    exit status - which `queue alerts` and `job wait` use to say what happened -
    is what this process returns.
    """
    command = build_command(target, argv, script_text=script_text)
    try:
        return subprocess.call(ssh_argv(target, command, tty=tty))
    except FileNotFoundError:
        print("error: ssh not found on this machine", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
