"""OpenSSH ProxyJump support for Paramiko connections.

The jump is an `ssh -W` stdio forward wired to one end of a socketpair, so
Paramiko talks to a real kernel socket. `paramiko.ProxyCommand` is avoided
deliberately: it pumps the subprocess pipes through a select()+os.read()
loop that spins at full CPU forever once the ssh process exits, because EOF
keeps select() reporting readable while os.read() returns nothing - a stuck
daemon burning a whole core until someone notices. With a socketpair the
subprocess exiting is an ordinary connection close.
"""
from __future__ import annotations

import math
import shlex
import shutil
import socket
import subprocess
from typing import List, Optional


class ProxyJumpError(OSError):
    """Raised when a ProxyJump setting cannot be used."""


def normalize_proxyjump(proxyjump) -> Optional[str]:
    """Return a normalized comma-separated ProxyJump chain."""
    if proxyjump is None:
        return None

    if isinstance(proxyjump, (list, tuple)):
        parts = [str(part).strip() for part in proxyjump]
    else:
        value = str(proxyjump).strip()
        if not value or value.lower() == "none":
            return None
        parts = [part.strip() for part in value.split(",")]

    if not parts or any(not part for part in parts):
        raise ProxyJumpError(f"Invalid ProxyJump value: {proxyjump!r}")
    if len(parts) == 1 and parts[0].lower() == "none":
        return None
    if any(part.lower() == "none" for part in parts):
        raise ProxyJumpError("ProxyJump 'none' cannot be combined with other hosts")
    return ",".join(parts)


def _target_endpoint(hostname: str, port: int) -> str:
    host = str(hostname).strip()
    if not host:
        raise ProxyJumpError("ProxyJump requires a target hostname")
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        host = f"[{host}]"
    return f"{host}:{int(port or 22)}"


def _final_jump_destination(spec: str):
    """Convert ProxyJump's [user@]host[:port] into ssh destination options."""
    user = None
    host_and_port = spec
    if "@" in spec:
        user, host_and_port = spec.rsplit("@", 1)
        if not user or not host_and_port:
            raise ProxyJumpError(f"Invalid ProxyJump destination: {spec!r}")

    port = None
    host = host_and_port
    if host_and_port.startswith("["):
        closing = host_and_port.find("]")
        if closing < 0:
            raise ProxyJumpError(f"Invalid ProxyJump IPv6 destination: {spec!r}")
        host = host_and_port[: closing + 1]
        suffix = host_and_port[closing + 1 :]
        if suffix:
            if not suffix.startswith(":") or not suffix[1:].isdigit():
                raise ProxyJumpError(f"Invalid ProxyJump port: {spec!r}")
            port = int(suffix[1:])
    elif host_and_port.count(":") == 1:
        host, port_text = host_and_port.rsplit(":", 1)
        if not host or not port_text.isdigit():
            raise ProxyJumpError(f"Invalid ProxyJump port: {spec!r}")
        port = int(port_text)

    if not host:
        raise ProxyJumpError(f"Invalid ProxyJump destination: {spec!r}")
    if port is not None and not 1 <= port <= 65535:
        raise ProxyJumpError(f"Invalid ProxyJump port: {port}")

    destination = f"{user}@{host}" if user else host
    return destination, port


def _proxy_command_args(
    proxyjump,
    hostname: str,
    port: int = 22,
    *,
    ssh_executable: str = "ssh",
    connect_timeout: Optional[float] = None,
    batch_mode: bool = False,
) -> List[str]:
    """The OpenSSH argv for the stdio forward."""
    normalized = normalize_proxyjump(proxyjump)
    if normalized is None:
        raise ProxyJumpError("ProxyJump is empty")

    jumps = normalized.split(",")
    destination, jump_port = _final_jump_destination(jumps[-1])

    args = [str(ssh_executable)]
    if batch_mode:
        args.extend(["-o", "BatchMode=yes"])
    args.extend(["-o", "ConnectionAttempts=1", "-o", "ExitOnForwardFailure=yes"])
    if connect_timeout is not None and float(connect_timeout) > 0:
        args.extend(["-o", f"ConnectTimeout={max(1, math.ceil(float(connect_timeout)))}"])
    args.extend(["-W", _target_endpoint(hostname, port)])
    if len(jumps) > 1:
        args.extend(["-J", ",".join(jumps[:-1])])
    if jump_port is not None:
        args.extend(["-p", str(jump_port)])
    args.extend(["--", destination])
    return args


def build_proxy_command(
    proxyjump,
    hostname: str,
    port: int = 22,
    *,
    ssh_executable: str = "ssh",
    connect_timeout: Optional[float] = None,
    batch_mode: bool = False,
) -> str:
    """The stdio-forward command as one shell-quoted string (for display)."""
    return shlex.join(
        _proxy_command_args(
            proxyjump,
            hostname,
            port,
            ssh_executable=ssh_executable,
            connect_timeout=connect_timeout,
            batch_mode=batch_mode,
        )
    )


class _ProxyJumpSocket(socket.socket):
    """One end of a socketpair whose other end is the `ssh -W` process.

    Closing the socket also ends the subprocess: terminate covers an ssh
    that outlives its connection, and the wait() reaps it so a long-running
    daemon cannot accumulate zombies.
    """

    def __init__(self, fileno: int, process: subprocess.Popen):
        super().__init__(socket.AF_UNIX, socket.SOCK_STREAM, 0, fileno)
        self._process: Optional[subprocess.Popen] = process

    def close(self) -> None:
        try:
            super().close()
        finally:
            process = getattr(self, "_process", None)
            self._process = None
            if process is None:
                return
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    process.wait()
            else:
                process.wait()


def open_proxyjump_socket(
    proxyjump,
    hostname: str,
    port: int = 22,
    *,
    connect_timeout: Optional[float] = None,
    batch_mode: bool = False,
):
    """Start an OpenSSH stdio forward suitable for ``SSHClient.connect(sock=...)``.

    Returns a real socket: the ssh process reads and writes the other end of
    a socketpair, the kernel moves the bytes, and the process exiting shows
    up as a clean EOF instead of wedging a reader.
    """
    normalized = normalize_proxyjump(proxyjump)
    if normalized is None:
        return None

    ssh_executable = shutil.which("ssh")
    if not ssh_executable:
        raise ProxyJumpError(
            "ProxyJump requires the OpenSSH client (`ssh`) on the local machine"
        )
    args = _proxy_command_args(
        normalized,
        hostname,
        port,
        ssh_executable=ssh_executable,
        connect_timeout=connect_timeout,
        batch_mode=batch_mode,
    )
    ours = None
    theirs = None
    try:
        ours, theirs = socket.socketpair()
        process = subprocess.Popen(
            args,
            stdin=theirs,
            stdout=theirs,
            # ssh's own diagnostics would interleave with TUI output; the
            # errors that matter surface through Paramiko as banner/EOF
            # failures either way.
            stderr=subprocess.DEVNULL,
        )
    except Exception as error:
        if ours is not None:
            ours.close()
        raise ProxyJumpError(
            f"Could not start ProxyJump command: {type(error).__name__}: {error}"
        ) from error
    finally:
        if theirs is not None:
            theirs.close()
    return _ProxyJumpSocket(ours.detach(), process)
