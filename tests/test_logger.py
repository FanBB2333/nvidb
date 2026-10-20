"""Tests for the SQLite GPU logger against fake clients (no NVML, no SSH)."""

import sqlite3
import time
from types import SimpleNamespace

import pandas as pd

from nvidb import logger as nvlogger


class _FakeClient:
    def __init__(self, description="node", gpu_ids=None, gpu_count=2):
        self.description = description
        self.gpu_ids = gpu_ids
        self.gpu_count = gpu_count
        self.full_info_calls = []

    def get_full_gpu_info(self, *, include_advanced=True):
        self.full_info_calls.append(include_advanced)
        rows = [
            {
                "gpu_index": i,
                "product_name": f"GPU{i}",
                "fan_speed": "30 %",
                "gpu_util": f"{10 * i} %",
                "memory_util": "5 %",
                "gpu_temp": "50 C",
                "rx_util": "1 KB/s",
                "tx_util": "2 KB/s",
                "power_state": "P2",
                "power_draw": "100.00 W",
                "current_power_limit": "300.00 W",
                "used": "1000 MiB",
                "total": "24564 MiB",
                "processes": [{"pid": 100 + i, "used_memory": "500 MiB", "username": f"user{i}"}],
            }
            for i in range(self.gpu_count)
        ]
        return pd.DataFrame(rows), {"attached_gpus": str(self.gpu_count)}

    def get_process_summary(self, gpu_stats):
        row = gpu_stats.iloc[0]
        return [], {row["processes"][0]["username"]: 500}

    def format_user_memory_compact(self, user_summary):
        return " ".join(f"{u}({m}M)" for u, m in user_summary.items())


def _logger(tmp_path, *clients, interval=5):
    log = nvlogger.NVLoggerSQLite(db_path=str(tmp_path / "gpu_log.db"), interval=interval)
    log.client_pool = SimpleNamespace(pool=list(clients))
    log.session_id = log._start_session()
    return log


def _rows(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT node, gpu_id, name, power, processes FROM gpu_logs ORDER BY id").fetchall()


def test_rows_reach_the_db_while_the_session_runs(tmp_path, monkeypatch):
    client = _FakeClient()
    log = _logger(tmp_path, client, interval=0.05)
    monkeypatch.setattr(log, "initialize_client_pool", lambda: True)
    log.session_id = None
    assert log.start_logging()
    deadline = time.monotonic() + 3
    while len(_rows(log.db_path)) < 6 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(_rows(log.db_path)) >= 6  # three ticks written before stop
    log.stop_logging()
    assert log.data_buffer == []
    assert _rows(log.db_path)[0] == ("node", 0, "GPU0", "P2 100.00 W/300.00 W", "user0(500M)")
    assert set(client.full_info_calls) == {False}
    with sqlite3.connect(log.db_path) as conn:
        assert conn.execute("SELECT status FROM log_sessions WHERE id = ?", (log.session_id,)).fetchone() == (
            "completed",
        )


def test_only_allowlisted_gpus_are_logged(tmp_path):
    log = _logger(tmp_path, _FakeClient(gpu_ids=[1], gpu_count=3))
    log._collect_data()
    log._flush_buffer()
    assert _rows(log.db_path) == [("node", 1, "GPU1", "P2 100.00 W/300.00 W", "user1(500M)")]


def test_a_failed_write_keeps_the_rows_for_the_next_tick(tmp_path, monkeypatch):
    log = _logger(tmp_path, _FakeClient(gpu_count=1))
    real_execute = log._execute

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    log._collect_data()
    monkeypatch.setattr(log, "_execute", locked)
    log._flush_buffer()
    assert len(log.data_buffer) == 1
    monkeypatch.setattr(log, "_execute", real_execute)
    log._collect_data()
    log._flush_buffer()
    assert log.data_buffer == []
    assert len(_rows(log.db_path)) == 2
