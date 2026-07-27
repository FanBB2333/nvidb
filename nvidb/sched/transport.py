"""Command transports for the job queue.

The scheduler runs unattended inside other tools, so these transports are
strictly non-interactive: they never prompt for a password or a key passphrase,
and every call is bounded by a timeout. That is the main reason the queue does
not reuse `RemoteClient`, which is built for a human sitting at a terminal.
"""
from __future__ import annotations

import base64
import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Dict, Optional

DEFAULT_CONNECT_TIMEOUT = 8.0
DEFAULT_COMMAND_TIMEOUT = 30.0


@dataclass
class CommandResult:
    exit_status: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_status == 0


class TransportError(RuntimeError):
    """Raised when a transport cannot reach or authenticate to a node."""


class Transport:
    """Runs shell commands somewhere. Subclasses decide where."""

    name = "transport"

    def run(self, command: str, timeout: Optional[float] = None) -> CommandResult:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # --- shared helpers ---------------------------------------------------

    def write_file(self, path: str, content: str, *, mode: Optional[str] = None) -> None:
        """Create a remote file from a local string.

        The payload travels base64-encoded so arbitrary job commands - quotes,
        newlines, `$`, non-ASCII - survive the trip through a shell.
        """
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        quoted = shlex.quote(path)
        script = (
            f"mkdir -p {shlex.quote(_dirname(path))} && "
            f"printf '%s' {shlex.quote(encoded)} | base64 -d > {quoted}"
        )
        if mode:
            script += f" && chmod {mode} {quoted}"
        result = self.run(script)
        if not result.ok:
            raise TransportError(
                f"Failed to write {path} on {self.name}: {result.stderr.strip() or result.stdout.strip()}"
            )

    def read_file(self, path: str, *, tail_lines: Optional[int] = None) -> str:
        """Read a remote file, optionally only its last N lines."""
        quoted = shlex.quote(path)
        if tail_lines:
            command = f"tail -n {int(tail_lines)} {quoted} 2>/dev/null || true"
        else:
            command = f"cat {quoted} 2>/dev/null || true"
        return self.run(command).stdout


def _dirname(path: str) -> str:
    if "/" not in path:
        return "."
    return path.rsplit("/", 1)[0] or "/"


class LocalTransport(Transport):
    """Runs commands on this machine. Used for local nodes and in tests."""

    def __init__(self, name: str = "local"):
        self.name = name

    def run(self, command: str, timeout: Optional[float] = None) -> CommandResult:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout or DEFAULT_COMMAND_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", f"timed out after {timeout}s")
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class SSHTransport(Transport):
    """A lazily connected, auto-reconnecting, non-interactive SSH channel."""

    def __init__(
        self,
        hostname: str,
        port: int = 22,
        username: Optional[str] = None,
        *,
        name: Optional[str] = None,
        auth: str = "auto",
        identityfile: Optional[str] = None,
        password: Optional[str] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ):
        self.hostname = hostname
        self.port = int(port or 22)
        self.username = username
        self.name = name or f"{username}@{hostname}:{port}"
        self.auth = auth or "auto"
        self.identityfile = identityfile
        self.password = password
        self.connect_timeout = connect_timeout
        self._client = None
        self._lock = threading.RLock()

    # --- connection -------------------------------------------------------

    def _connect_locked(self):
        import os

        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.load_system_host_keys()
        except Exception:
            pass

        identityfile = None
        if self.identityfile and self.auth in ("auto", "key"):
            identityfile = os.path.expanduser(str(self.identityfile))

        kwargs = {
            "hostname": self.hostname,
            "port": self.port,
            "username": self.username,
            "timeout": self.connect_timeout,
            "banner_timeout": self.connect_timeout,
            "auth_timeout": self.connect_timeout,
        }
        if self.auth == "password":
            if not self.password:
                raise TransportError(
                    f"{self.name}: auth 'password' needs a stored password "
                    "(the queue never prompts)"
                )
            kwargs.update(password=self.password, allow_agent=False, look_for_keys=False)
        else:
            kwargs.update(
                key_filename=identityfile,
                allow_agent=True,
                look_for_keys=True,
            )
            if self.password and self.auth == "auto":
                kwargs["password"] = self.password

        try:
            client.connect(**kwargs)
        except Exception as error:
            try:
                client.close()
            except Exception:
                pass
            raise TransportError(f"{self.name}: {type(error).__name__}: {error}") from error

        self._client = client
        return client

    def _ensure_client(self):
        client = self._client
        if client is not None:
            transport = client.get_transport()
            if transport is not None and transport.is_active():
                return client
            self._close_locked()
        return self._connect_locked()

    def _close_locked(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    @property
    def connected(self) -> bool:
        client = self._client
        if client is None:
            return False
        transport = client.get_transport()
        return bool(transport is not None and transport.is_active())

    # --- execution --------------------------------------------------------

    def run(self, command: str, timeout: Optional[float] = None) -> CommandResult:
        timeout = timeout or DEFAULT_COMMAND_TIMEOUT
        with self._lock:
            last_error: Optional[Exception] = None
            # One retry: a channel that died between ticks is routine, not fatal.
            for attempt in range(2):
                try:
                    client = self._ensure_client()
                    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
                    stdin.close()
                    out = stdout.read().decode("utf-8", errors="replace")
                    err = stderr.read().decode("utf-8", errors="replace")
                    status = stdout.channel.recv_exit_status()
                    return CommandResult(status, out, err)
                except TransportError:
                    raise
                except Exception as error:
                    last_error = error
                    self._close_locked()
                    if attempt == 1:
                        break
            raise TransportError(
                f"{self.name}: command failed: {type(last_error).__name__}: {last_error}"
            )


class TransportPool:
    """Keeps one transport per node alive for the lifetime of a process.

    A long-lived client such as the TUI pays the SSH handshake once; a one-shot
    CLI call pays it once per invocation and then exits.
    """

    def __init__(self):
        self._transports: Dict[str, Transport] = {}
        self._lock = threading.RLock()

    def get(self, node_name: str, factory) -> Transport:
        with self._lock:
            transport = self._transports.get(node_name)
            if transport is None:
                transport = factory()
                self._transports[node_name] = transport
            return transport

    def drop(self, node_name: str) -> None:
        with self._lock:
            transport = self._transports.pop(node_name, None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            transports = list(self._transports.values())
            self._transports.clear()
        for transport in transports:
            try:
                transport.close()
            except Exception:
                pass
