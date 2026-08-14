"""A remote node that drops off the network must come back on its own.

The TUI keeps one SSH session per node for its whole lifetime, so a Wi-Fi
hiccup, a suspended laptop or a rebooted jump host used to leave a permanently
dead Paramiko transport behind: every later refresh failed, the empty table was
reported as "No GPU data available" - indistinguishable from a machine without
NVIDIA GPUs - and reconnecting the network changed nothing.
"""
import getpass
import json

import pandas as pd
import pytest

from nvidb.connection import RemoteClient
from nvidb.data_modules import ServerInfo


SNAPSHOT = {
    "ok": True,
    "backend": "ctypes",
    "driver_version": "550.90.07",
    "cuda_version": "12.4",
    "gpus": [
        {
            "gpu_index": 0,
            "name": "NVIDIA RTX PRO 5000",
            "architecture": "Blackwell",
            "memory_total_bytes": 48 * 1024**3,
            "memory_used_bytes": 4 * 1024**3,
            "memory_free_bytes": 44 * 1024**3,
            "gpu_util_percent": 42,
            "memory_util_percent": 10,
            "temperature_c": 51,
            "fan_percent": 35,
            "power_usage_mw": 120_000,
            "power_limit_mw": 300_000,
            "performance_state": 2,
            "processes": [],
        }
    ],
}


class _FakeTransport:
    def __init__(self):
        self.active = True

    def is_active(self):
        return self.active

    def set_keepalive(self, interval):
        self.keepalive = interval


class _FakeStdout:
    """Stands in for the NVML agent's stdout, and for any other command."""

    def __init__(self, client):
        self._client = client
        self.channel = _FakeChannel(client)

    def readline(self):
        self._client.require_link()
        return json.dumps(SNAPSHOT) + "\n"

    def read(self):
        self._client.require_link()
        return b""

    def close(self):
        pass


class _FakeChannel:
    def __init__(self, client):
        self._client = client
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def exit_status_ready(self):
        return False

    def sendall(self, data):
        self._client.require_link()

    def close(self):
        self.closed = True


class _FakeSSHClient:
    """A Paramiko stand-in whose link can be cut and restored at will."""

    network_up = True
    connect_attempts = 0
    live_transports = []

    def __init__(self):
        self._transport = None

    def set_missing_host_key_policy(self, policy):
        pass

    def load_system_host_keys(self):
        pass

    def connect(self, **kwargs):
        type(self).connect_attempts += 1
        if not type(self).network_up:
            raise OSError("No route to host")
        self._transport = _FakeTransport()
        type(self).live_transports.append(self._transport)

    def get_transport(self):
        return self._transport

    def close(self):
        self._transport = None

    def exec_command(self, command=None, timeout=None):
        self.require_link()
        stdout = _FakeStdout(self)
        return object(), stdout, _FakeStdout(self)

    def require_link(self):
        if self._transport is None or not self._transport.is_active():
            raise OSError("Socket is closed")

    # --- test controls ----------------------------------------------------

    @classmethod
    def reset(cls):
        cls.network_up = True
        cls.connect_attempts = 0
        cls.live_transports = []

    @classmethod
    def cut_network(cls):
        cls.network_up = False
        for transport in cls.live_transports:
            transport.active = False


@pytest.fixture
def client(monkeypatch):
    _FakeSSHClient.reset()
    monkeypatch.setattr("nvidb.connection.paramiko.SSHClient", _FakeSSHClient)
    monkeypatch.setattr(
        getpass,
        "getpass",
        lambda *args, **kwargs: pytest.fail("a background reconnect must never prompt"),
    )
    remote = RemoteClient(
        ServerInfo(host="10.0.0.42", port=22, username="alice", auth="key")
    )
    assert remote.connect(announce=False) is True
    return remote


def _gpu_table(remote):
    stats, system_info = remote.get_full_gpu_info()
    return stats, system_info


def test_a_healthy_link_reports_gpus(client):
    stats, system_info = _gpu_table(client)

    assert list(stats["gpu_index"]) == [0]
    assert system_info["driver_version"] == "550.90.07"


def test_a_dropped_link_is_reported_as_an_error_not_as_a_missing_gpu(client):
    _FakeSSHClient.cut_network()

    stats, system_info = _gpu_table(client)

    # An empty table with no error reads as "this machine has no NVIDIA GPU",
    # which the unified view hides behind the [u] toggle.
    assert stats.empty
    assert system_info["error_type"] == "connect"
    assert "No route to host" in system_info["error"]
    assert client.connected is False


def test_the_table_returns_once_the_network_is_back(client, monkeypatch):
    # Backoff is covered on its own below; here every refresh may retry.
    monkeypatch.setattr(RemoteClient, "RECONNECT_BACKOFF_SECONDS", (0.0,))
    _FakeSSHClient.cut_network()
    assert _gpu_table(client)[0].empty

    _FakeSSHClient.network_up = True
    stats, system_info = _gpu_table(client)

    assert list(stats["gpu_index"]) == [0]
    assert client.connected is True
    assert client.last_connect_error is None


def test_an_unreachable_node_is_retried_on_a_backoff_not_every_refresh(client):
    _FakeSSHClient.cut_network()
    _FakeSSHClient.connect_attempts = 0

    # First refresh after the drop retries immediately: most drops are over by
    # the time anyone looks at the screen.
    assert client.ensure_connected() is False
    assert _FakeSSHClient.connect_attempts == 1

    # The refreshes that follow within the backoff window must not re-handshake.
    for _ in range(5):
        assert client.ensure_connected() is False
    assert _FakeSSHClient.connect_attempts == 1
    assert "retrying" in client.last_connect_error


def test_credentials_that_only_a_human_can_supply_are_not_retried(client, monkeypatch):
    """Reconnecting cannot invent a password, so it stops instead of looping."""
    client.auth = "password"
    client.password = None
    _FakeSSHClient.cut_network()
    _FakeSSHClient.connect_attempts = 0

    assert client.ensure_connected() is False
    assert client.last_error_type == "auth"
    assert client.ensure_connected() is False
    assert _FakeSSHClient.connect_attempts == 0


def test_a_reconnect_never_blocks_the_ui_thread_forever(client):
    """Every SSH call is bounded, so one dead node cannot freeze the refresh."""
    connect_kwargs = {}
    monkey = _FakeSSHClient.connect

    def record(self, **kwargs):
        connect_kwargs.update(kwargs)
        return monkey(self, **kwargs)

    _FakeSSHClient.connect = record
    try:
        client._close_link_locked()
        assert client.ensure_connected() is True
    finally:
        _FakeSSHClient.connect = monkey

    assert connect_kwargs["timeout"] == RemoteClient.CONNECT_TIMEOUT_SECONDS
    assert connect_kwargs["banner_timeout"] == RemoteClient.CONNECT_TIMEOUT_SECONDS
    assert connect_kwargs["auth_timeout"] == RemoteClient.CONNECT_TIMEOUT_SECONDS


def test_a_dropped_node_keeps_its_error_line_in_the_unified_view():
    """The status line must not read like a node that simply has no GPU."""
    from test_tui_views import _pool, _without_ansi

    pool = _pool()
    raw_stats = {
        0: (pd.DataFrame(), {}),
        1: (
            pd.DataFrame(),
            {
                "error": "Connection lost (OSError: Socket is closed); reconnecting",
                "error_type": "connect",
            },
        ),
    }

    lines = [
        _without_ansi(line)
        for line in pool._get_unified_node_status_lines(raw_stats, last_update_time=1)
    ]

    assert any("! training-node (100.64.0.42): Connection lost" in line for line in lines)
    assert not any("No GPU data available" in line for line in lines)
