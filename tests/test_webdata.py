"""Characterization tests for nvidb.webdata (the data layer nvidb.dashboard reuses).

This module had zero test coverage before it was split out of the legacy
web.py (which mixed it with a dead Streamlit UI). These tests pin its
observable behavior so future refactors of dashboard.py / webdata.py don't
silently change parsing or formatting output.
"""

import sqlite3

import pandas as pd
import pytest

from nvidb import webdata


def _make_log_db(path):
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE log_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT DEFAULT 'running',
            interval_seconds INTEGER,
            include_remote INTEGER DEFAULT 0
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE gpu_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            node TEXT NOT NULL,
            gpu_id INTEGER,
            name TEXT,
            fan_speed TEXT,
            util_gpu TEXT,
            util_mem TEXT,
            temperature TEXT,
            rx TEXT,
            tx TEXT,
            power TEXT,
            memory_used TEXT,
            memory_total TEXT,
            processes TEXT
        )
        """
    )
    cursor.execute(
        "INSERT INTO log_sessions (id, start_time, end_time, status, interval_seconds, include_remote) "
        "VALUES (1, '2026-01-01T00:00:00', '2026-01-01T00:01:00', 'stopped', 5, 0)"
    )
    cursor.execute(
        """
        INSERT INTO gpu_logs
            (session_id, timestamp, node, gpu_id, name, fan_speed, util_gpu, util_mem,
             temperature, rx, tx, power, memory_used, memory_total, processes)
        VALUES
            (1, '2026-01-01T00:00:00', 'local', 0, 'NVIDIA GeForce RTX 3090', '40%', '85%', '30%',
             '65C', '12.3MB/s', '4.1MB/s', '210.5W', '19045', '24564', 'alice(19000M)')
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture
def log_db(tmp_path):
    db_path = tmp_path / "gpu_log.db"
    _make_log_db(db_path)
    return db_path


def test_load_sessions(log_db):
    df = webdata.load_sessions(log_db)
    assert list(df["id"]) == [1]
    assert df.iloc[0]["record_count"] == 1
    assert df.iloc[0]["snapshot_count"] == 1


def test_load_session_logs(log_db):
    df = webdata.load_session_logs(log_db, 1)
    assert len(df) == 1
    assert df.iloc[0]["node"] == "local"
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("N/A", None),
        ("-", None),
        ("85%", 85.0),
        ("85.5", 85.5),
    ],
)
def test_parse_percent(value, expected):
    assert webdata._parse_percent(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, (None, None)),
        ("N/A", (None, None)),
        ("19045/24564", (19045.0, 24564.0)),
        ("not-a-pair", (None, None)),
    ],
)
def test_parse_mib_pair(value, expected):
    assert webdata._parse_mib_pair(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("N/A", None),
        ("65C", 65.0),
        ("65 C", 65.0),
        ("65", 65.0),
    ],
)
def test_parse_temperature_c(value, expected):
    assert webdata._parse_temperature_c(value) == expected


@pytest.mark.parametrize(
    "value,expected_gib",
    [
        (None, None),
        ("1024MiB", 1.0),
        ("1GiB", 1.0),
        ("1024", 1.0),  # bare number defaults to MiB
    ],
)
def test_parse_memory_gib(value, expected_gib):
    result = webdata._parse_memory_gib(value)
    if expected_gib is None:
        assert result is None
    else:
        assert result == pytest.approx(expected_gib)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("N/A", None),
        ("10MB/s", 10.0),
        ("1GB/s", 1024.0),
    ],
)
def test_parse_bandwidth_mbps(value, expected):
    result = webdata._parse_bandwidth_mbps(value)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("N/A", None),
        ("210.5W", 210.5),
        ("no watts here", None),
    ],
)
def test_parse_power_watts(value, expected):
    assert webdata._parse_power_watts(value) == expected


def test_format_gb():
    assert webdata._format_gb(None) == "N/A"
    assert webdata._format_gb(1024) == "1.0GB"
    assert webdata._format_gb(1024 * 20) == "20GB"


def test_format_mib():
    assert webdata._format_mib(512) == "512 MiB"
    assert webdata._format_mib(2048) == "2.0 GB"
    assert webdata._format_mib("not-a-number") == "N/A"


def test_format_duration():
    assert webdata._format_duration(None, None) == "N/A"
    assert webdata._format_duration("2026-01-01T00:00:00", "2026-01-01T00:01:05") == "1m5s"
    assert webdata._format_duration("2026-01-01T00:00:00", "2026-01-02T01:00:00") == "1d1h"


def test_strip_gpu_name():
    assert webdata._strip_gpu_name("NVIDIA GeForce RTX 3090") == "RTX 3090"
    assert webdata._strip_gpu_name(None) == ""


def test_server_summary_empty():
    empty = pd.DataFrame()
    assert webdata._server_summary(empty) == "No GPUs"
    assert webdata._server_summary(empty, {"data_source": "unsupported"}) == "Unsupported"
    assert webdata._server_summary(empty, {"data_source": "nvml"}) == "No GPUs | src:nvml"


def test_server_summary_with_data():
    table = pd.DataFrame(
        [
            {"util": "0%", "memory[used/total]": "0/24564"},
            {"util": "100%", "memory[used/total]": "19045/24564"},
        ]
    )
    system_info = {
        "data_source": "nvml",
        "system_stats": {
            "cpu_cores": 8,
            "cpu_percent": 12.4,
            "mem_used_gb": 10.0,
            "mem_total_gb": 32.0,
            "swap_used_gb": 0.0,
            "swap_total_gb": 0.0,
        },
    }
    summary = webdata._server_summary(table, system_info)
    assert "2 GPUs" in summary
    assert "1 idle" in summary
    assert "50% avg" in summary
    assert "CPU:  12%(8C)" in summary
    assert "Mem: 10/32G" in summary
    assert summary.endswith("src:nvml")


def test_parse_user_memory_compact():
    assert webdata._parse_user_memory_compact(None) == {}
    assert webdata._parse_user_memory_compact("-") == {}
    assert webdata._parse_user_memory_compact("alice(1000M) bob(2000M)") == {
        "alice": 1000,
        "bob": 2000,
    }
    # Same user appearing twice (e.g. two processes) accumulates.
    assert webdata._parse_user_memory_compact("alice(1000M) alice(500M)") == {"alice": 1500}


def test_user_memory_from_df():
    df = pd.DataFrame({"processes": ["alice(1000M)", "bob(2000M) alice(500M)", "-"]})
    assert webdata._user_memory_from_df(df) == {"alice": 1500, "bob": 2000}
    assert webdata._user_memory_from_df(pd.DataFrame()) == {}


def test_user_summary_df():
    df = webdata._user_summary_df({"alice": 1500, "bob": 2000, "zero_user": 0})
    assert list(df["user"]) == ["bob", "alice"]
    assert list(df["vram_mib"]) == [2000, 1500]


def test_user_time_share_df():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00", "2026-01-01T00:00:00", "2026-01-01T00:00:05"]
            ),
            "processes": ["alice(1000M)", "bob(500M)", "alice(1000M)"],
        }
    )
    share = webdata._user_time_share_df(df)
    row_by_user = {row["user"]: row for _, row in share.iterrows()}
    assert row_by_user["alice"]["snapshots"] == 2
    assert row_by_user["alice"]["share"] == pytest.approx(100.0)
    assert row_by_user["bob"]["snapshots"] == 1
    assert row_by_user["bob"]["share"] == pytest.approx(50.0)


def test_build_log_snapshot_table(log_db):
    df = webdata.load_session_logs(log_db, 1)
    table = webdata._build_log_snapshot_table(df)
    assert list(table["GPU"]) == [0]
    assert table.iloc[0]["name"] == "RTX 3090"
    assert table.iloc[0]["memory[used/total]"] == "19045/24564"


def test_downsample_per_gpu():
    df = pd.DataFrame(
        {
            "gpu_id": [0] * 10,
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="s"),
            "value": range(10),
        }
    )
    downsampled = webdata._downsample_per_gpu(df, max_points_per_gpu=3)
    assert len(downsampled) <= 4  # ceil(10/3) step => 4 points
    # Below the cap, nothing is dropped.
    assert len(webdata._downsample_per_gpu(df, max_points_per_gpu=100)) == 10


def test_get_db_path_delegates_to_config(monkeypatch):
    monkeypatch.setattr(webdata.config, "get_db_path", lambda: "/tmp/fake.db")
    assert webdata.get_db_path() == "/tmp/fake.db"


def test_load_server_list_missing_config(tmp_path, monkeypatch):
    monkeypatch.setattr(webdata.config, "get_config_path", lambda: str(tmp_path / "absent.yml"))
    with pytest.raises(FileNotFoundError):
        webdata._load_server_list()
