"""Command transports for the job queue.

The scheduler runs unattended inside other tools, so these transports are
strictly non-interactive: they never prompt for a password or a key passphrase,
and every call is bounded by a timeout. That is the main reason the queue does
not reuse `RemoteClient`, which is built for a human sitting at a terminal.
"""
from __future__ import annotations

import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

from ..ssh_proxy import open_proxyjump_socket

DEFAULT_CONNECT_TIMEOUT = 8.0
DEFAULT_COMMAND_TIMEOUT = 30.0


@dataclass
class CommandResult:
    exit_status: int
    stdout: str
    stderr: str


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

    def read_file(self, path: str, *, tail_lines: Optional[int] = None) -> str:
        """Read a remote file, optionally only its last N lines."""
        quoted = shlex.quote(path)
        if tail_lines:
            command = f"tail -n {int(tail_lines)} {quoted} 2>/dev/null || true"
        else:
            command = f"cat {quoted} 2>/dev/null || true"
        return self.run(command).stdout


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
            # A hung command is a transport fault, not a command that failed:
            # returning a status here would have the scheduler read the empty
            # output as fact - no pid, no processes - instead of as "unknown".
            raise TransportError(
                f"{self.name}: command timed out after {timeout or DEFAULT_COMMAND_TIMEOUT}s"
            ) from None
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
        proxyjump: Optional[str] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ):
        self.hostname = hostname
        self.port = int(port or 22)
        self.username = username
        self.name = name or f"{username}@{hostname}:{port}"
        self.auth = auth or "auto"
        self.identityfile = identityfile
        self.password = password
        self.proxyjump = proxyjump
        self.connect_timeout = connect_timeout
        self._client = None
        self._proxy = None
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

        proxy = None
        try:
            proxy = open_proxyjump_socket(
                self.proxyjump,
                self.hostname,
                self.port,
                connect_timeout=self.connect_timeout,
                batch_mode=True,
            )
            if proxy is not None:
                kwargs["sock"] = proxy
            client.connect(**kwargs)
        except Exception as error:
            try:
                client.close()
            except Exception:
                pass
            if proxy is not None:
                try:
                    proxy.close()
                except Exception:
                    pass
            raise TransportError(f"{self.name}: {type(error).__name__}: {error}") from error

        self._client = client
        self._proxy = proxy
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
        if self._proxy is not None:
            try:
                self._proxy.close()
            except Exception:
                pass
        self._proxy = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()

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


