import os
import re
import threading
from types import SimpleNamespace

import pandas as pd

from nvidb.connection import NVClientPool


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _without_ansi(value):
    return ANSI_ESCAPE.sub("", value)


def _gpu_row(gpu_index, name, utilization, memory, processes="-"):
    return {
        "GPU": gpu_index,
        "name": name,
        "fan": "30 %",
        "util": utilization,
        "temp": "48 C",
        "rx": "1 KB/s",
        "tx": "2 KB/s",
        "power": "P2 80/200",
        "mem_util": "25 %",
        "memory[used/total]": memory,
        "processes": processes,
    }


def _pool():
    pool = NVClientPool.__new__(NVClientPool)
    pool.pool = [
        SimpleNamespace(
            description="Local Machine",
            host="localhost",
            port="local",
        ),
        SimpleNamespace(
            description="training-node",
            host="100.64.0.42",
            port=22,
        ),
    ]
    pool.compact = False
    pool.display_mode = pool.DISPLAY_MODE_NODES
    pool.unified_detailed = False
    pool.unified_sort_mode = "node"
    pool.unified_filter_mode = "all"
    pool.unified_selected_gpu = 0
    pool.unified_show_processes = False
    pool.unified_show_trends = False
    pool._unified_gpu_count = 0
    pool._unified_page_size = 1
    pool._unified_gpu_history = {}
    pool._unified_history_lock = threading.Lock()
    pool.selected_server = 0
    pool.expanded_servers = {0}
    pool._toggle_disabled_servers = set()
    pool.quit_flag = threading.Event()
    pool.refresh_needed = threading.Event()
    pool.ui_only_refresh = False
    return pool


def test_unified_table_adds_node_and_hostname_and_uses_summary_columns():
    pool = _pool()
    raw_stats = {
        0: (
            pd.DataFrame([_gpu_row(0, "RTX 4090", "10 %", "1024/24576")]),
            {},
        ),
        1: (
            pd.DataFrame(
                [
                    _gpu_row(0, "RTX 6000 Ada", "75 %", "24000/49140", "alice(24G)"),
                    _gpu_row(1, "RTX 6000 Ada", "0 %", "0/49140"),
                ]
            ),
            {},
        ),
    }

    table = pool._build_unified_gpu_table(raw_stats)

    assert list(table.columns) == list(pool.UNIFIED_TABLE_COLUMNS)
    assert table["Node"].tolist() == ["Local", "training-node", "training-node"]
    assert table.loc[0, "Hostname"]
    assert table.loc[1:, "Hostname"].tolist() == ["100.64.0.42", "100.64.0.42"]
    assert table.loc[1, "rx"] == "1 KB/s"
    assert table.loc[1, "tx"] == "2 KB/s"
    assert "mem_util" not in table.columns


def test_unified_table_keeps_identity_and_core_metrics_on_narrow_terminals(monkeypatch):
    pool = _pool()
    raw_stats = {
        0: (
            pd.DataFrame([_gpu_row(0, "RTX 4090", "10 %", "1024/24576")]),
            {},
        ),
        1: (
            pd.DataFrame([_gpu_row(0, "RTX 6000 Ada", "75 %", "24000/49140")]),
            {},
        ),
    }
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((80, 30)))

    rendered = "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))

    assert "Unified GPU table | GPUs: 2 | Nodes with GPU: 2/2" in rendered
    assert "Capacity: avail 0/2 | busy 1 | avg 42%" in rendered
    assert "Node" in rendered
    assert "Hostname/IP" in rendered
    assert "GPU" in rendered
    assert "Model" in rendered
    assert "Util" in rendered
    assert "VRAM U/T MiB" in rendered
    assert "train" in rendered
    assert "100.64.0.42" in rendered
    assert "Processes" not in rendered


def test_unified_view_reports_nodes_without_gpu_rows():
    pool = _pool()
    raw_stats = {
        0: (
            pd.DataFrame([_gpu_row(0, "RTX 4090", "10 %", "1024/24576")]),
            {},
        ),
        1: (
            pd.DataFrame(),
            {"error": "Connection timed out", "error_type": "connect"},
        ),
    }

    rendered = "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))

    assert "Nodes with GPU: 1/2" in rendered
    assert "! training-node (100.64.0.42): Connection timed out" in rendered


def test_unified_capacity_summary_and_sort_modes(monkeypatch):
    pool = _pool()
    table = pd.DataFrame(
        [
            {
                **_gpu_row(0, "busy", "90 %", "20000/24576"),
                "Node": "node-a",
                "Hostname": "10.0.0.1",
            },
            {
                **_gpu_row(0, "small-free", "0 %", "0/24576"),
                "Node": "node-b",
                "Hostname": "10.0.0.2",
            },
            {
                **_gpu_row(0, "large-free", "0 %", "4/49140"),
                "Node": "node-c",
                "Hostname": "10.0.0.3",
            },
        ]
    )
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((80, 30)))

    summary = pool._format_unified_capacity_summary(table)

    assert "avail 2/3" in summary
    assert "busy 1" in summary
    assert "avg 30%" in summary
    assert "VRAM 20/96G" in summary
    assert "free 76G" in summary
    assert pool._sort_unified_gpu_table(table)["name"].tolist() == [
        "busy",
        "small-free",
        "large-free",
    ]

    pool.unified_sort_mode = "available"
    assert pool._sort_unified_gpu_table(table)["name"].tolist() == [
        "large-free",
        "small-free",
        "busy",
    ]

    pool.unified_sort_mode = "utilization"
    assert pool._sort_unified_gpu_table(table)["name"].tolist() == [
        "busy",
        "large-free",
        "small-free",
    ]


def test_unified_filter_modes_and_error_view():
    pool = _pool()
    table = pd.DataFrame(
        [
            _gpu_row(0, "available", "0 %", "0/24576"),
            _gpu_row(1, "reserved", "0 %", "12000/24576"),
            _gpu_row(2, "busy", "75 %", "20000/24576"),
        ]
    )

    assert pool._filter_unified_gpu_table(table)["name"].tolist() == [
        "available",
        "reserved",
        "busy",
    ]

    pool.unified_filter_mode = "available"
    assert pool._filter_unified_gpu_table(table)["name"].tolist() == ["available"]

    pool.unified_filter_mode = "busy"
    assert pool._filter_unified_gpu_table(table)["name"].tolist() == ["busy"]

    pool.unified_filter_mode = "errors"
    assert pool._filter_unified_gpu_table(table).empty

    raw_stats = {
        0: (pd.DataFrame(), {}),
        1: (
            pd.DataFrame(),
            {"error": "Connection timed out", "error_type": "connect"},
        ),
    }
    rendered = "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))
    assert "GPUs: 0/0" in rendered
    assert "Error filter: GPU rows hidden" in rendered
    assert "Connection timed out" in rendered
    assert "Local" not in "\n".join(
        pool._get_unified_node_status_lines(
            raw_stats,
            last_update_time=1,
            errors_only=True,
        )
    )


def test_unified_detailed_view_paginates_and_scrolls(monkeypatch):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    raw_stats = {
        1: (
            pd.DataFrame(
                [
                    _gpu_row(0, "GPU A", "0 %", "0/24576"),
                    _gpu_row(1, "GPU B", "10 %", "1024/24576"),
                    _gpu_row(2, "GPU C", "90 %", "20000/24576"),
                ]
            ),
            {},
        ),
    }
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((80, 16)))

    first_page = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )

    assert "Rows 1-1/3" in first_page
    assert "> GPU 0 [IDLE]" in first_page
    assert "Model GPU A" in first_page
    assert "Model GPU B" not in first_page
    assert pool._unified_page_size == 1

    assert pool._handle_keypress("j") is True
    second_page = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )

    assert "Rows 2-2/3" in second_page
    assert "> GPU 1 [ACTIVE]" in second_page
    assert "Model GPU B" in second_page

    page_down = SimpleNamespace(name="KEY_NPAGE")
    assert pool._handle_keypress(page_down) is True
    third_page = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )
    assert "Rows 3-3/3" in third_page
    assert "> GPU 2 [HIGH]" in third_page


def test_unified_selected_gpu_process_details(monkeypatch):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool.unified_show_processes = True
    raw_stats = {
        1: (
            pd.DataFrame(
                [
                    _gpu_row(
                        0,
                        "RTX 6000 Ada",
                        "75 %",
                        "24000/49140",
                        "alice(24G)",
                    )
                ]
            ),
            {},
        ),
        "_nvidb": {
            "process_details_by_client": {
                1: {
                    "0": [
                        {
                            "pid": 4242,
                            "username": "alice",
                            "used_memory": "16384 MiB",
                            "type": "C",
                            "process_name": "python",
                        },
                        {
                            "pid": 5252,
                            "username": "bob",
                            "used_memory": "4096 MiB",
                            "type": "C",
                            "process_name": "torchrun",
                        },
                    ]
                }
            }
        },
    }
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((80, 30)))

    rendered = "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))

    assert "Processes: training-node (100.64.0.42) GPU 0" in rendered
    assert "PID" in rendered
    assert "User" in rendered
    assert "VRAM" in rendered
    assert "Command" in rendered
    assert "4242" in rendered
    assert "alice" in rendered
    assert "16384 MiB" in rendered
    assert "python" in rendered


def test_unified_gpu_trends_keep_latest_sixty_samples(monkeypatch):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool.unified_show_trends = True
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((80, 30)))

    raw_stats = {}
    for sample_index in range(65):
        raw_stats = {
            1: (
                pd.DataFrame(
                    [
                        {
                            **_gpu_row(
                                0,
                                "RTX 6000 Ada",
                                f"{sample_index % 100} %",
                                f"{sample_index * 100}/49140",
                            ),
                            "temp": f"{40 + sample_index % 20} C",
                        }
                    ]
                ),
                {},
            )
        }
        pool._record_unified_gpu_history(
            raw_stats,
            timestamp=sample_index,
        )

    history = next(iter(pool._unified_gpu_history.values()))
    assert len(history) == 60
    assert history[0]["timestamp"] == 5
    assert history[-1]["timestamp"] == 64

    rendered = "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))
    assert "Trends: training-node GPU 0 | 60/60 samples" in rendered
    assert "Util" in rendered
    assert "VRAM" in rendered
    assert "Temp" in rendered
    assert any(block in rendered for block in "▁▂▃▄▅▆▇█")


def test_detailed_view_uses_readable_cards_on_narrow_terminals(monkeypatch):
    pool = _pool()
    pool.unified_detailed = True
    raw_stats = {
        0: (
            pd.DataFrame([_gpu_row(0, "RTX 4090", "10 %", "1024/24576")]),
            {},
        ),
        1: (
            pd.DataFrame(
                [
                    _gpu_row(0, "RTX 6000 Ada", "75 %", "24000/49140", "alice(24G)"),
                    _gpu_row(1, "RTX 6000 Ada", "0 %", "0/49140"),
                ]
            ),
            {},
        ),
    }
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((80, 30)))

    rendered = "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))
    table = pool._build_unified_gpu_table(raw_stats)
    detailed_lines = pool._format_unified_detailed_table(table).splitlines()
    content_lines = [line for line in detailed_lines if line.startswith("|")]

    assert "Unified GPU table (Detailed) | GPUs: 3" in rendered
    assert len(content_lines) == 9
    assert all(len(_without_ansi(line)) == 80 for line in detailed_lines)
    assert "Node training-node (100.64.0.42)" in rendered
    assert "GPU 0 [BUSY]" in rendered
    assert "Model RTX 6000 Ada" in rendered
    assert "Util 75%" in rendered
    assert "VRAM 24000/49140 MiB" in rendered
    assert "Temp 48 C" in rendered
    assert "Power P2 80/200 W" in rendered
    assert "Proc alice(24G)" in rendered
    assert "Fan 30 %" in rendered
    assert "RX 1 KB/s" in rendered
    assert "TX 2 KB/s" in rendered


def test_detailed_view_keeps_full_auxiliary_metrics_on_wide_terminals(monkeypatch):
    pool = _pool()
    pool.unified_detailed = True
    raw_stats = {
        1: (
            pd.DataFrame([_gpu_row(0, "RTX 6000 Ada", "75 %", "24000/49140")]),
            {},
        ),
    }
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((140, 40)))

    rendered = "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))

    assert "Fan 30 %" in rendered
    assert "RX 1 KB/s" in rendered
    assert "TX 2 KB/s" in rendered


def test_detailed_view_colors_status_without_breaking_alignment(monkeypatch):
    pool = _pool()
    pool.unified_detailed = True
    table = pd.DataFrame(
        [
            _gpu_row(0, "RTX 6000 Ada", "85 %", "40000/49140"),
            _gpu_row(1, "RTX 6000 Ada", "0 %", "0/49140"),
        ]
    )
    table["Node"] = "training-node"
    table["Hostname"] = "100.64.0.42"
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((80, 30)))
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("ANSI_COLORS_DISABLED", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    rendered = pool._format_unified_detailed_table(table)
    detailed_lines = rendered.splitlines()
    plain_rendered = _without_ansi(rendered)

    assert "\x1b[31m" in rendered
    assert "\x1b[36m" in rendered
    assert "GPU 0 [HIGH]" in plain_rendered
    assert "GPU 1 [IDLE]" in plain_rendered
    assert all(len(_without_ansi(line)) == 80 for line in detailed_lines)


def test_v_and_d_keys_switch_views_and_disable_node_navigation_in_unified_view():
    pool = _pool()

    assert pool._handle_keypress("d") is False
    assert pool._handle_keypress("s") is False
    assert pool._handle_keypress("f") is False
    assert pool._handle_keypress("t") is False
    assert pool.unified_detailed is False
    assert pool.unified_sort_mode == "node"
    assert pool.unified_filter_mode == "all"

    assert pool._handle_keypress("v") is True
    assert pool.display_mode == pool.DISPLAY_MODE_UNIFIED
    assert pool.refresh_needed.is_set()

    pool.refresh_needed.clear()
    assert pool._handle_keypress("d") is True
    assert pool.unified_detailed is True
    assert pool.refresh_needed.is_set()

    pool.refresh_needed.clear()
    assert pool._handle_keypress("s") is True
    assert pool.unified_sort_mode == "available"
    assert pool.refresh_needed.is_set()

    pool.refresh_needed.clear()
    assert pool._handle_keypress("s") is True
    assert pool.unified_sort_mode == "utilization"
    assert pool.refresh_needed.is_set()

    pool.refresh_needed.clear()
    assert pool._handle_keypress("s") is True
    assert pool.unified_sort_mode == "node"
    assert pool.refresh_needed.is_set()

    pool.refresh_needed.clear()
    assert pool._handle_keypress("f") is True
    assert pool.unified_filter_mode == "available"
    assert pool.refresh_needed.is_set()

    pool.refresh_needed.clear()
    assert pool._handle_keypress("f") is True
    assert pool.unified_filter_mode == "busy"
    assert pool.refresh_needed.is_set()

    pool.refresh_needed.clear()
    assert pool._handle_keypress("f") is True
    assert pool.unified_filter_mode == "errors"
    assert pool.refresh_needed.is_set()

    pool.refresh_needed.clear()
    assert pool._handle_keypress("f") is True
    assert pool.unified_filter_mode == "all"
    assert pool.refresh_needed.is_set()

    pool.refresh_needed.clear()
    assert pool._handle_keypress("j") is False
    assert pool.selected_server == 0
    assert not pool.refresh_needed.is_set()

    pool._unified_gpu_count = 1
    assert pool._handle_keypress("\n") is True
    assert pool.unified_show_processes is True
    assert pool.refresh_needed.is_set()

    pool.refresh_needed.clear()
    assert pool._handle_keypress("t") is True
    assert pool.unified_show_trends is True
    assert pool.refresh_needed.is_set()

    assert pool._handle_keypress("d") is True
    assert pool.unified_detailed is False
    assert pool._handle_keypress("v") is True
    assert pool.display_mode == pool.DISPLAY_MODE_NODES
    pool.refresh_needed.clear()

    assert pool._handle_keypress("j") is True
    assert pool.selected_server == 1
    assert pool.refresh_needed.is_set()
