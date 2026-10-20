"""Tests for the @nvidb.monitor decorator against a fake NVML."""

import asyncio
import importlib
from types import SimpleNamespace

import pytest

monitor_mod = importlib.import_module("nvidb.monitor")


@pytest.fixture
def fake_nvml(monkeypatch):
    used = iter(range(1000, 10**9, 1000))
    nvml = monitor_mod.pynvml
    monkeypatch.setattr(nvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(nvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(nvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(nvml, "nvmlDeviceGetHandleByIndex", lambda idx: idx)
    monkeypatch.setattr(nvml, "nvmlDeviceGetName", lambda handle: "Fake GPU")
    monkeypatch.setattr(
        nvml, "nvmlDeviceGetMemoryInfo", lambda handle: SimpleNamespace(used=next(used), total=8 << 30)
    )
    monkeypatch.setattr(nvml, "nvmlDeviceGetUtilizationRates", lambda handle: SimpleNamespace(gpu=50, memory=10))
    monkeypatch.setattr(nvml, "nvmlDeviceGetTemperature", lambda handle, sensor: 60)
    monkeypatch.setattr(nvml, "nvmlDeviceGetPowerUsage", lambda handle: 100000)


def test_format_delta_keeps_the_sign_of_freed_memory():
    assert monitor_mod._format_delta(-1024) == "-1.00 KiB"
    assert monitor_mod._format_delta(2048) == "+2.00 KiB"
    assert monitor_mod._format_delta(0) == "+0 B"


def test_sync_and_async_functions_report_and_return(fake_nvml, capsys):
    @monitor_mod.monitor(sample_interval=0.01)
    def work(x):
        return x * 2

    @monitor_mod.monitor
    async def async_work(x):
        await asyncio.sleep(0.02)
        return x + 1

    assert work(21) == 42
    assert work.__name__ == "work"
    assert asyncio.run(async_work(1)) == 2
    out = capsys.readouterr().out
    assert "Function completed: work" in out
    assert "Function completed: async_work" in out
    assert "GPU 0: Fake GPU" in out
    assert "Power:" in out


def test_exceptions_propagate_after_the_report(fake_nvml, capsys):
    @monitor_mod.monitor
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        boom()
    assert "Function completed: boom" in capsys.readouterr().out


def test_disabled_monitor_leaves_the_function_alone():
    def work():
        return 1

    assert monitor_mod.monitor(enabled=False)(work) is work
