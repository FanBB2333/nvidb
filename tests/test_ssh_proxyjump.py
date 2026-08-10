import shlex
from pathlib import Path

import pytest
import yaml

from nvidb import config
from nvidb.connection import RemoteClient
from nvidb.data_modules import ServerInfo, ServerListInfo
from nvidb.sched.model import Node
from nvidb.sched.scheduler import Scheduler
from nvidb.sched.transport import SSHTransport, TransportError
from nvidb.ssh_proxy import (
    ProxyJumpError,
    build_proxy_command,
    normalize_proxyjump,
)
from nvidb.test import run as cli_run


def test_proxyjump_command_uses_openssh_stdio_forward():
    command = build_proxy_command(
        "login-a,alice@login-b:2222",
        "10.0.0.42",
        2200,
        ssh_executable="/usr/bin/ssh",
        connect_timeout=7.2,
        batch_mode=True,
    )

    assert shlex.split(command) == [
        "/usr/bin/ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=8",
        "-W",
        "10.0.0.42:2200",
        "-J",
        "login-a",
        "-p",
        "2222",
        "--",
        "alice@login-b",
    ]


def test_proxyjump_command_brackets_an_ipv6_target():
    command = build_proxy_command(
        "jump",
        "2001:db8::42",
        ssh_executable="ssh",
    )

    assert shlex.split(command)[-3:] == ["[2001:db8::42]:22", "--", "jump"]


@pytest.mark.parametrize("value", [None, "", "  ", "none", "NONE"])
def test_empty_or_disabled_proxyjump_is_normalized_to_none(value):
    assert normalize_proxyjump(value) is None


@pytest.mark.parametrize("value", ["jump,", ",jump", "jump,none", "jump:bad"])
def test_invalid_proxyjump_is_rejected(value):
    if value == "jump:bad":
        with pytest.raises(ProxyJumpError):
            build_proxy_command(value, "gpu", ssh_executable="ssh")
    else:
        with pytest.raises(ProxyJumpError):
            normalize_proxyjump(value)


def _open_proxy_socket_running(monkeypatch, argv):
    """Open a proxy socket whose 'ssh' is an arbitrary local command."""
    from nvidb import ssh_proxy

    monkeypatch.setattr(ssh_proxy.shutil, "which", lambda name: "/bin/sh")
    monkeypatch.setattr(
        ssh_proxy, "_proxy_command_args", lambda *args, **kwargs: argv
    )
    return ssh_proxy.open_proxyjump_socket("jump", "10.0.0.42")


def test_proxy_socket_is_a_real_socket_wired_to_the_command(monkeypatch):
    sock = _open_proxy_socket_running(monkeypatch, ["/bin/cat"])
    try:
        sock.settimeout(10)
        sock.sendall(b"hello through the jump\n")
        assert sock.recv(64) == b"hello through the jump\n"
    finally:
        sock.close()
    # Closing the socket also ended and reaped the subprocess.
    assert sock._process is None


def test_a_dead_proxy_command_reads_as_eof_not_a_busy_loop(monkeypatch):
    """The reason ProxyCommand was replaced: its recv() spins at full CPU
    forever once the ssh process exits. A real socketpair must deliver a
    plain EOF instead."""
    sock = _open_proxy_socket_running(monkeypatch, ["/bin/sh", "-c", "exit 0"])
    try:
        sock.settimeout(10)  # a regression would raise timeout, not hang CI
        assert sock.recv(1) == b""
    finally:
        sock.close()


class _FakeProxy:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeTransportState:
    def is_active(self):
        return True


class _FakeSSHClient:
    instances = []
    connect_error = None

    def __init__(self):
        self.connect_kwargs = None
        self.closed = False
        self.__class__.instances.append(self)

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def load_system_host_keys(self):
        self.loaded_system_keys = True

    def connect(self, **kwargs):
        self.connect_kwargs = kwargs
        if self.connect_error:
            raise self.connect_error

    def close(self):
        self.closed = True

    def get_transport(self):
        return _FakeTransportState()


def test_scheduler_transport_passes_proxy_socket_and_closes_it(monkeypatch):
    _FakeSSHClient.instances = []
    _FakeSSHClient.connect_error = None
    proxy = _FakeProxy()
    monkeypatch.setattr("paramiko.SSHClient", _FakeSSHClient)
    monkeypatch.setattr(
        "nvidb.sched.transport.open_proxyjump_socket",
        lambda *args, **kwargs: proxy,
    )
    transport = SSHTransport(
        "10.0.0.42",
        username="alice",
        proxyjump="login",
    )

    client = transport._connect_locked()

    assert client.connect_kwargs["sock"] is proxy
    assert client.connect_kwargs["hostname"] == "10.0.0.42"
    transport.close()
    assert client.closed is True
    assert proxy.closed is True


def test_scheduler_transport_closes_proxy_after_connection_failure(monkeypatch):
    _FakeSSHClient.instances = []
    _FakeSSHClient.connect_error = RuntimeError("unreachable")
    proxy = _FakeProxy()
    monkeypatch.setattr("paramiko.SSHClient", _FakeSSHClient)
    monkeypatch.setattr(
        "nvidb.sched.transport.open_proxyjump_socket",
        lambda *args, **kwargs: proxy,
    )
    transport = SSHTransport(
        "10.0.0.42",
        username="alice",
        proxyjump="login",
    )

    with pytest.raises(TransportError, match="unreachable"):
        transport._connect_locked()
    assert proxy.closed is True
    _FakeSSHClient.connect_error = None


def test_remote_monitor_passes_proxy_socket_to_paramiko(monkeypatch):
    _FakeSSHClient.instances = []
    _FakeSSHClient.connect_error = None
    proxy = _FakeProxy()
    monkeypatch.setattr("nvidb.connection.paramiko.SSHClient", _FakeSSHClient)
    monkeypatch.setattr(
        "nvidb.connection.open_proxyjump_socket",
        lambda *args, **kwargs: proxy,
    )
    client = RemoteClient(
        ServerInfo(
            host="10.0.0.42",
            port=22,
            username="alice",
            auth="key",
            proxyjump="login",
        )
    )

    assert client.connect() is True
    assert client.client.connect_kwargs["sock"] is proxy
    assert client.client.connect_kwargs["username"] == "alice"
    client.client.close()
    client._close_proxy()


def test_scheduler_copies_proxyjump_from_server_config():
    scheduler = Scheduler(
        None,
        cfg={
            "servers": [
                {
                    "hostname": "10.0.0.42",
                    "username": "alice",
                    "nickname": "gpu",
                    "proxyjump": "login",
                }
            ]
        },
    )

    backend = scheduler._default_backend_factory(Node(name="gpu"))

    assert backend.transport.proxyjump == "login"
    scheduler.close()


def test_server_config_round_trips_proxyjump():
    servers = ServerListInfo()
    servers.add_server(
        ServerInfo(
            host="10.0.0.42",
            port=22,
            username="alice",
            proxyjump="login",
        )
    )

    serialized = servers.to_dict()

    assert serialized[0]["proxyjump"] == "login"
    assert ServerListInfo.from_dict(serialized)[0].proxyjump == "login"
    assert '    proxyjump: "login"' in config.format_servers_yaml(serialized)


def test_ssh_config_import_keeps_proxyjump(monkeypatch, tmp_path):
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text(
        "\n".join(
            [
                "Host login",
                "  HostName login.example.com",
                "  User jump-user",
                "Host training-a100",
                "  HostName 10.0.0.42",
                "  User alice",
                "  IdentityFile ~/.ssh/id_ed25519",
                "  ProxyJump login",
                "",
            ]
        )
    )
    output = tmp_path / "config.yml"
    monkeypatch.setattr(
        cli_run,
        "_prompt_import_selection",
        lambda candidates: [
            candidate["nickname"] == "training-a100" for candidate in candidates
        ],
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    cli_run.import_ssh_config(ssh_config, output)

    stored = yaml.safe_load(output.read_text())
    assert stored["servers"] == [
        {
            "hostname": "10.0.0.42",
            "port": 22,
            "username": "alice",
            "nickname": "training-a100",
            "auth": "auto",
            "identityfile": str(Path("~/.ssh/id_ed25519").expanduser()),
            "proxyjump": "login",
        }
    ]
