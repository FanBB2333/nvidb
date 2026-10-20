"""CPU utilization must describe the last refresh interval, cheaply."""
import sys

import pytest

from nvidb import connection
from nvidb.connection import BaseClient, LocalClient


class ProcStatClient(BaseClient):
    def __init__(self, stat_lines):
        super().__init__()
        self.connected = True
        self.stat_lines = list(stat_lines)
        self.commands = []

    def connect(self, **kwargs):
        return True

    def execute_command(self, command):
        self.commands.append(command)
        if "/proc/stat" in command:
            return self.stat_lines.pop(0)
        return ""


def test_linux_cpu_percent_covers_the_last_interval_not_the_uptime():
    client = ProcStatClient(
        [
            "cpu  9000 0 1000 90000 0 0 0 0 0 0\n",
            # 50 busy ticks out of 100 since the first sample; the since-boot
            # average would still read about 10%.
            "cpu  9050 0 1000 90050 0 0 0 0 0 0\n",
        ]
    )

    assert client.get_system_stats()["cpu_percent"] == pytest.approx(10.0)
    assert client.get_system_stats()["cpu_percent"] == pytest.approx(50.0)
    assert not any(command.startswith("top") for command in client.commands)


def test_linux_cpu_percent_counts_niced_work_but_not_iowait():
    # Two samples 3 s apart on a 12-core node whose load runs niced.
    client = ProcStatClient(
        [
            "cpu  199345 9509371 799488 45806320 345518 0 4765 0 0 0\n",
            "cpu  199353 9510552 799538 45808651 345544 0 4765 0 0 0\n",
        ]
    )

    client.get_system_stats()
    busy = 8 + 1181 + 50
    assert client.get_system_stats()["cpu_percent"] == pytest.approx(busy * 100 / (busy + 2331 + 26))


def test_local_macos_cpu_percent_reads_mach_counters_instead_of_top(monkeypatch):
    samples = iter([(100, 1000), (150, 1100)])
    monkeypatch.setattr(connection.sys, "platform", "darwin")
    monkeypatch.setattr(connection, "_darwin_cpu_ticks", lambda: next(samples))
    client = LocalClient()
    commands = []
    monkeypatch.setattr(client, "execute_command", lambda command: commands.append(command) or "")

    assert client.get_system_stats()["cpu_percent"] == pytest.approx(10.0)
    assert client.get_system_stats()["cpu_percent"] == pytest.approx(50.0)
    assert not any("top" in command or "/proc/stat" in command for command in commands)


@pytest.mark.skipif(sys.platform != "darwin", reason="Mach host statistics are macOS-only")
def test_mach_cpu_ticks_are_readable():
    busy, total = connection._darwin_cpu_ticks()
    assert 0 <= busy <= total and total > 0
