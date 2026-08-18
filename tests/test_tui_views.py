import logging
import os
import re
import threading
from types import SimpleNamespace

import pandas as pd
from blessed import Terminal

from nvidb.connection import NVClientPool
from nvidb.mouse import MouseEvent


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
        "link": "4.0x16",
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
    pool.unified_selected_gpu_key = None
    pool.unified_show_processes = False
    pool.unified_show_trends = False
    pool.unified_group_by_node = True
    pool.hide_unsupported = True
    pool.mouse_enabled = True
    pool.unified_active_pane = "gpu"
    pool.unified_process_panel_hidden = False
    pool.unified_selected_process = 0
    pool.unified_selected_process_pid = None
    pool._unified_process_count = 0
    pool.unified_process_filter = ""
    pool.unified_process_filter_editing = False
    pool.unified_process_sort_mode = "vram"
    pool.unified_process_sort_descending = True
    pool.unified_process_rows = 0
    pool._unified_process_visible_rows = 1
    pool._unified_process_total_count = 0
    pool.unified_command_scroll = 0
    pool._unified_command_line_count = 0
    pool._unified_command_page_size = 0
    pool._pending_process_signal = None
    pool._process_action_notice = None
    pool._click_targets = {}
    pool._click_regions = []
    pool._body_click_targets = {}
    pool._body_click_regions = []
    pool._unified_page_start = 0
    pool._unified_gpu_count = 0
    pool._unified_page_size = 1
    pool._unified_gpu_history = {}
    pool._unified_process_history = {}
    pool._unified_history_lock = threading.Lock()
    pool.tui_help_visible = False
    pool.selected_server = 0
    pool.expanded_servers = {0}
    pool._toggle_disabled_servers = set()
    pool._default_expansion_applied = False
    pool._expansion_touched = False
    pool._persist_view_enabled = False
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
    pool.unified_group_by_node = False
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

    assert "Unified GPU table · GPUs 2 · Nodes with GPU 2/2" in rendered
    assert "Capacity: avail 0/2 · busy 1 · avg 42%" in rendered
    assert "Node" in rendered
    assert "Hostname/IP" in rendered
    assert "GPU" in rendered
    assert "Model" in rendered
    assert "Util" in rendered
    assert "VRAM U/T MiB" in rendered
    assert "train" in rendered
    assert "100.64.0.42" in rendered
    assert "Processes" not in rendered


def test_grouped_unified_table_uses_node_bands_instead_of_node_columns(monkeypatch):
    pool = _pool()
    raw_stats = {
        0: (
            pd.DataFrame([_gpu_row(0, "RTX 4090", "10 %", "1024/24576")]),
            {},
        ),
        1: (
            pd.DataFrame(
                [
                    _gpu_row(0, "RTX 6000 Ada", "75 %", "24000/49140"),
                    _gpu_row(1, "RTX 6000 Ada", "0 %", "0/49140"),
                ]
            ),
            {},
        ),
    }
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((100, 30)))

    lines = pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    rendered = _without_ansi("\n".join(lines))

    assert "training-node (100.64.0.42) · 2 GPUs · free 1 · avg 38%" in rendered
    assert "VRAM 23/96G" in rendered
    # Node identity moved into the band, freeing the columns for GPU metrics.
    assert "Hostname/IP" not in rendered
    assert all(
        len(_without_ansi(line)) == 100
        for line in rendered.splitlines()
        if line.startswith("│")
    )

    # Grouping only applies while rows follow node order.
    pool.unified_sort_mode = "utilization"
    flat = _without_ansi("\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1)))
    assert "Hostname/IP" in flat
    assert "training-node (100.64.0.42) · 2 GPUs" not in flat


def test_node_band_uses_a_dim_rule_instead_of_a_background_block(monkeypatch):
    pool = _pool()
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("ANSI_COLORS_DISABLED", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    band = pool._format_section_band("training-node (10.0.0.1)", "2 GPUs | free 1", 60)

    assert len(_without_ansi(band)) == 60
    assert _without_ansi(band).startswith("training-node (10.0.0.1) · 2 GPUs | free 1 ───")
    assert "\x1b[36m" in band  # cyan node name
    assert "\x1b[2m" in band  # dim separator and rule
    assert "on_blue" not in band and "\x1b[44m" not in band

    # Narrow bands drop the stats before they touch the node name.
    narrow = _without_ansi(pool._format_section_band("training-node", "2 GPUs | free 1", 20))
    assert narrow.startswith("training-node")
    assert len(narrow) == 20


def test_active_filter_warns_when_it_hides_gpus(monkeypatch):
    pool = _pool()
    pool.unified_filter_mode = "available"
    raw_stats = {
        1: (
            pd.DataFrame(
                [
                    _gpu_row(0, "RTX 6000 Ada", "75 %", "24000/49140"),
                    _gpu_row(1, "RTX 6000 Ada", "0 %", "0/49140"),
                ]
            ),
            {},
        ),
    }
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((100, 30)))

    rendered = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))
    )
    assert "Filter Available: 1 of 2 GPUs hidden ([f] to change)" in rendered

    pool.unified_filter_mode = "all"
    unfiltered = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))
    )
    assert "GPUs hidden" not in unfiltered


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

    assert "Nodes with GPU 1/2" in rendered
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


def test_gpu_selection_stays_on_the_same_card_when_live_sort_order_changes(
    monkeypatch,
):
    pool = _pool()
    pool.unified_sort_mode = "utilization"
    pool.unified_selected_gpu = 1
    raw_stats = {
        1: (
            pd.DataFrame(
                [
                    _gpu_row(0, "GPU A", "90 %", "20000/49140"),
                    _gpu_row(1, "GPU B", "10 %", "4000/49140"),
                ]
            ),
            {},
        )
    }
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((100, 30)))

    pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    assert pool.unified_selected_gpu == 1
    assert pool.unified_selected_gpu_key[-1] == "1"

    raw_stats[1][0].loc[0, "util"] = "5 %"
    raw_stats[1][0].loc[1, "util"] = "99 %"
    pool._render_unified_gpu_lines(raw_stats, last_update_time=2)
    assert pool.unified_selected_gpu == 0
    assert pool.unified_selected_gpu_key[-1] == "1"


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
    assert "GPUs 0/0" in rendered
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
    assert "› training-node GPU 0 [IDLE]" in first_page
    assert "Model GPU A" in first_page
    assert "Model GPU B" not in first_page
    assert pool._unified_page_size == 1

    assert pool._handle_keypress("j") is True
    second_page = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )

    assert "Rows 2-2/3" in second_page
    assert "› training-node GPU 1 [ACTIVE]" in second_page
    assert "Model GPU B" in second_page

    page_down = SimpleNamespace(name="KEY_NPAGE")
    assert pool._handle_keypress(page_down) is True
    third_page = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )
    assert "Rows 3-3/3" in third_page
    assert "› training-node GPU 2 [HIGH]" in third_page


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

    assert (
        "Processes | 2 active | training-node (100.64.0.42) | GPU 0"
        in rendered
    )
    assert "PID" in rendered
    assert "USER" in rendered
    assert "VRAM" in rendered
    assert "COMMAND" in rendered
    assert "4242" in rendered
    assert "alice" in rendered
    assert "16,384 MiB" in rendered
    assert "33.3% of GPU VRAM" in rendered
    assert "68.3% of used VRAM" in rendered
    assert "python" in rendered


def test_unified_process_panel_shows_htop_fields_and_full_command(monkeypatch):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_show_processes = True
    pool.unified_active_pane = "process"
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("ANSI_COLORS_DISABLED", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    command = (
        "/usr/bin/python3 train.py --config configs/qwen3-vl-32b.yaml --deepspeed zero3"
    )
    raw_stats = {
        1: (
            pd.DataFrame([_gpu_row(0, "RTX 6000 Ada", "75 %", "24000/49140")]),
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
                            "process_name": "python3",
                            "command": command,
                            "cpu_percent": 412.5,
                            "mem_percent": 18.4,
                            "rss_kb": 12582912,
                            "elapsed": "2-03:21:07",
                            "state": "Rl",
                            "threads": 57,
                        }
                    ]
                }
            }
        },
    }
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((120, 40)))

    styled = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )
    rendered = _without_ansi(styled)

    for header in ("PID", "USER", "VRAM", "CPU%", "MEM%", "RSS", "TIME", "COMMAND"):
        assert header in rendered
    assert "412.5" in rendered
    assert "18.4" in rendered
    assert "12.0G" in rendered
    assert "2-03:21:07" in rendered
    # Truncated inside the table row, wrapped in full in the selected block.
    for fragment in command.split():
        assert fragment in rendered
    assert "\x1b[100m" in styled  # low-contrast grey selected row
    assert "\x1b[31m" in styled  # high CPU value
    assert "\x1b[35m" not in styled  # no magenta PID/user accents
    assert "\x1b[34m" not in styled  # no blue PID/command accents
    assert all(
        len(line) == 120
        for line in rendered.splitlines()
        if line.startswith(("┌", "├", "└", "│", "┊"))
    )


def test_process_details_are_only_collected_while_the_panel_is_open():
    pool = _pool()
    calls = []

    class FakeClient:
        description = "training-node"
        host = "100.64.0.42"
        port = 22

        def get_full_gpu_info(self):
            return (
                pd.DataFrame(
                    [
                        {
                            "gpu_index": 0,
                            "product_name": "NVIDIA RTX 6000 Ada",
                            "product_architecture": "Ada",
                            "tx_util": "2 KB/s",
                            "rx_util": "1 KB/s",
                            "fan_speed": "30 %",
                            "total": "49140 MiB",
                            "used": "24000 MiB",
                            "free": "25140 MiB",
                            "gpu_util": "75 %",
                            "memory_util": "25 %",
                            "gpu_temp": "48 C",
                            "power_state": "P2",
                            "power_draw": "80.00 W",
                            "current_power_limit": "200.00 W",
                            "processes": [],
                        }
                    ]
                ),
                {},
            )

        def get_system_stats(self):
            return {}

        def get_process_summary(self, stats=None, detailed=False):
            calls.append(detailed)
            return [], {}

    pool.pool = [FakeClient()]

    pool.get_client_gpus_info(return_raw=True)
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.get_client_gpus_info(return_raw=True)
    pool.unified_detailed = True
    pool.get_client_gpus_info(return_raw=True)
    pool.unified_process_panel_hidden = True
    pool.get_client_gpus_info(return_raw=True)
    pool.unified_process_panel_hidden = False
    pool.unified_detailed = False
    pool.unified_show_processes = True
    pool.get_client_gpus_info(return_raw=True)

    # The extra `ps` call per host is only worth paying for while the panel is up.
    assert calls == [False, False, True, False, True]


def test_process_selection_stays_on_the_same_pid_when_vram_sorting_changes(
    monkeypatch,
):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool.unified_selected_process = 1
    raw_stats = _focus_raw_stats("python train.py")
    processes = raw_stats["_nvidb"]["process_details_by_client"][1]["0"]
    processes.append(
        {
            **processes[0],
            "pid": 5252,
            "used_memory": "4000 MiB",
            "command": "python eval.py",
        }
    )
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((100, 40)))

    pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    assert pool.unified_selected_process == 1
    assert pool.unified_selected_process_pid == "5252"

    processes[1]["used_memory"] = "30000 MiB"
    pool._render_unified_gpu_lines(raw_stats, last_update_time=2)
    assert pool.unified_selected_process == 0
    assert pool.unified_selected_process_pid == "5252"


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


def test_selected_process_history_tracks_cpu_vram_ram_and_rss(monkeypatch):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool.unified_show_trends = True
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((100, 40)))

    raw_stats = _focus_raw_stats("python train.py --steps 100")
    process = raw_stats["_nvidb"]["process_details_by_client"][1]["0"][0]
    for sample_index in range(4):
        process["cpu_percent"] = 40 + sample_index * 10
        process["mem_percent"] = 2 + sample_index
        process["rss_kb"] = 1024 * (1000 + sample_index * 100)
        process["used_memory"] = f"{20000 + sample_index * 1000} MiB"
        pool._record_unified_gpu_history(raw_stats, timestamp=sample_index)

    history = next(iter(pool._unified_process_history.values()))
    assert len(history) == 4
    assert history[-1]["cpu_percent"] == 70
    assert history[-1]["mem_percent"] == 5
    assert history[-1]["rss_kb"] == 1331200
    assert history[-1]["vram_mib"] == 23000

    rendered = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))
    )
    assert "SELECTED PROCESS HISTORY" in rendered
    assert "History  4/60 samples" in rendered
    for label in ("CPU", "VRAM", "RAM", "RSS"):
        assert label in rendered
    assert any(block in rendered for block in "▁▂▃▄▅▆▇█")


def test_detailed_view_uses_readable_cards_on_narrow_terminals(monkeypatch):
    pool = _pool()
    pool.unified_detailed = True
    pool.unified_selected_gpu = 1
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
    content_lines = [
        line
        for line in detailed_lines
        if _without_ansi(line).startswith(("│", "┊"))
    ]

    assert "Focus GPU/node · Detailed · GPUs 3" in rendered
    assert len(content_lines) == 12
    assert all(len(_without_ansi(line)) == 80 for line in detailed_lines)
    assert "training-node (100.64.0.42) -- 2 GPUs" not in _without_ansi(rendered)
    assert "training-node GPU 0 [BUSY]" in rendered
    assert "100.64.0.42" in rendered
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


def test_detailed_focus_uses_quiet_highlight_and_solid_dashed_borders(
    monkeypatch,
):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    raw_stats = _focus_raw_stats("python train.py")
    monkeypatch.setattr(
        os,
        "get_terminal_size",
        lambda: os.terminal_size((100, 32)),
    )
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("ANSI_COLORS_DISABLED", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    pool.unified_active_pane = "gpu"
    gpu_styled = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )
    gpu_plain = _without_ansi(gpu_styled)
    gpu_highlights = [
        _without_ansi(line)
        for line in gpu_styled.splitlines()
        if "\x1b[100m" in line
    ]

    assert gpu_plain.startswith("Focus GPU/node")
    assert any(line.startswith("╭─") for line in gpu_plain.splitlines())
    assert any(
        line.startswith("╭╌ Processes")
        for line in gpu_plain.splitlines()
    )
    assert len(gpu_highlights) == 1
    assert "GPU  › training-node GPU 0" in gpu_highlights[0]
    assert "›    4242" not in gpu_plain

    pool.unified_active_pane = "process"
    process_styled = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )
    process_plain = _without_ansi(process_styled)
    process_highlights = [
        _without_ansi(line)
        for line in process_styled.splitlines()
        if "\x1b[100m" in line
    ]

    assert process_plain.startswith("Focus process")
    assert any(
        line.startswith("╭╌╌")
        for line in process_plain.splitlines()
    )
    assert any(
        line.startswith("╭─ Processes")
        for line in process_plain.splitlines()
    )
    assert len(process_highlights) == 1
    assert "›    4242" in process_highlights[0]
    assert "› training-node GPU 0" not in process_plain
    assert "\x1b[34m" not in process_styled
    assert "\x1b[35m" not in process_styled


def test_detailed_cards_stay_aligned_with_progress_bars_on_wide_terminals(monkeypatch):
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
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((140, 40)))

    rendered = pool._format_unified_detailed_table(
        table,
        selected_row=0,
        section_headers={0: ("training-node (100.64.0.42)", "2 GPUs | free 1")},
    )
    plain = _without_ansi(rendered)

    assert "training-node (100.64.0.42) · 2 GPUs | free 1" in plain
    assert "━" in plain and "─" in plain
    assert all(len(line) == 140 for line in plain.splitlines())


def test_unsupported_nodes_are_hidden_until_toggled(monkeypatch):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    raw_stats = {
        0: (pd.DataFrame(), {"data_source": "unsupported", "unsupported": True}),
        1: (
            pd.DataFrame([_gpu_row(0, "RTX 6000 Ada", "75 %", "24000/49140")]),
            {},
        ),
    }
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((100, 30)))

    hidden = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))
    )
    assert "1 node without GPU support hidden ([u] to show)" in hidden
    assert "No GPU data available" not in hidden

    assert pool._handle_keypress("u") is True
    assert pool.hide_unsupported is False
    shown = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, last_update_time=1))
    )
    assert "Local" in shown
    assert "No GPU data available" in shown


def test_default_expansion_skips_nodes_without_gpus():
    pool = _pool()
    raw_stats = {
        0: (pd.DataFrame(), {"data_source": "unsupported", "unsupported": True}),
        1: (
            pd.DataFrame([_gpu_row(0, "RTX 6000 Ada", "75 %", "24000/49140")]),
            {},
        ),
    }

    pool._apply_default_expansion(raw_stats, last_update_time=1)
    assert pool.expanded_servers == {1}

    # Every GPU-bearing node opens expanded, not just the first one.
    pool.pool.append(SimpleNamespace(description="third-node", host="10.0.0.3", port=22))
    raw_stats[2] = (
        pd.DataFrame([_gpu_row(0, "RTX 4090", "10 %", "1024/24576")]),
        {},
    )
    pool._default_expansion_applied = False
    pool._apply_default_expansion(raw_stats, last_update_time=1)
    assert pool.expanded_servers == {1, 2}

    # Applied once only, and never against a user's own expand/collapse choice.
    pool.expanded_servers = {0}
    pool._apply_default_expansion(raw_stats, last_update_time=1)
    assert pool.expanded_servers == {0}

    touched = _pool()
    touched._expansion_touched = True
    touched._apply_default_expansion(raw_stats, last_update_time=1)
    assert touched.expanded_servers == {0}


def _server_block(pool, gpu_count):
    """A realistic expanded-server block: description, driver line, table."""
    rows = [
        _gpu_row(index, "RTX 4090", "10 %", "1024/24576")
        for index in range(gpu_count)
    ]
    table = pool._format_fixed_width_table(pd.DataFrame(rows), border=True)
    return f"Driver: 550.0 | CUDA: 12.4\n{table}"


def _server_rows(pool, specs):
    return [
        (idx, False, True, f"hdr {idx}", {}, stats_info)
        for idx, stats_info in enumerate(specs)
    ]


def test_scale_server_blocks_keeps_every_table_when_the_terminal_is_tall():
    pool = _pool()
    specs = [_server_block(pool, 6), _server_block(pool, 2)]

    blocks = pool._scale_server_blocks(_server_rows(pool, specs), {}, 100)

    assert set(blocks) == {0, 1}
    assert blocks[0] == specs[0].splitlines() + [""]
    assert not any("hidden" in line for line in blocks[0])


def test_scale_server_blocks_trims_gpu_rows_but_keeps_the_frame(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    pool = _pool()
    specs = [_server_block(pool, 8), _server_block(pool, 2)]

    blocks = pool._scale_server_blocks(_server_rows(pool, specs), {}, 20)

    # Everything still expanded, and the total fits the budget exactly.
    assert set(blocks) == {0, 1}
    assert sum(len(block) for block in blocks.values()) <= 20
    plain = [
        [_without_ansi(line) for line in block]
        for block in blocks.values()
    ]
    for block in plain:
        # The rounded frame survives trimming: separator, rows, bottom.
        assert any(line.startswith("├") for line in block)
        assert any(line.startswith("╰") for line in block)
    trimmed = [
        block
        for block in plain
        if any("line(s) hidden" in line for line in block)
    ]
    assert trimmed, "the tallest table should say how much it dropped"


def test_scale_server_blocks_collapses_servers_when_the_screen_is_tiny():
    pool = _pool()
    specs = [_server_block(pool, 8), _server_block(pool, 2)]

    # Too short for even one framed table below the headers.
    blocks = pool._scale_server_blocks(_server_rows(pool, specs), {}, 3)

    assert blocks == {}


def test_the_selected_server_is_the_last_to_collapse():
    pool = _pool()
    specs = [_server_block(pool, 8), _server_block(pool, 8), _server_block(pool, 8)]

    # Room for exactly one minimum block: the selected server keeps it,
    # so Enter on it always has a visible effect.
    blocks = pool._scale_server_blocks(
        _server_rows(pool, specs), {}, 10, keep_expanded=1
    )

    assert set(blocks) == {1}
    assert any("line(s) hidden" in _without_ansi(line) for line in blocks[1])


def test_nodes_view_fits_the_terminal_with_every_server_expanded(
    monkeypatch, capsys
):
    pool = _pool()
    pool.term = Terminal(force_styling=False)
    pool._default_expansion_applied = True
    pool.expanded_servers = {0, 1}
    specs = [_server_block(pool, 8), _server_block(pool, 4)]
    pool.cached_stats = specs
    pool.cached_raw_stats = {
        0: (
            pd.DataFrame(
                [
                    _gpu_row(index, "RTX 4090", "10 %", "1024/24576")
                    for index in range(8)
                ]
            ),
            {},
        ),
        1: (
            pd.DataFrame(
                [
                    _gpu_row(index, "RTX 4090", "10 %", "1024/24576")
                    for index in range(4)
                ]
            ),
            {},
        ),
    }
    pool._last_update_time = 1
    pool._last_fetch_duration = 0.1
    pool._last_fetch_error = None
    pool._cache_lock = threading.Lock()
    # +3 over the tight 20-line case: the panel border (top+bottom) and the
    # divider between the two servers each cost one row.
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((120, 23)))

    pool.print_stats(use_cache=True)
    capsys.readouterr()

    frame = pool._tui_diff_screen._previous
    # One screen, no scrolling: the scaled frame fits the 23-line window.
    assert len(frame) <= 22
    plain = [_without_ansi(line) for line in frame]
    # The block no longer repeats the server's name, so the selected
    # server's full 8-GPU table fits exactly; the other one collapses.
    assert any("│   7 " in line for line in plain)
    assert any("▸ [2] " in line for line in plain)
    assert not any(line.strip() == "node description" for line in plain)
    # Both server headers stay reachable for clicks.
    assert {kind for kind, _ in pool._click_targets.values()} == {"server"}


def _nodes_pool(cached_stats, raw_stats, *, updated=1):
    pool = _pool()
    pool.term = Terminal(force_styling=False)
    pool._default_expansion_applied = True
    pool.expanded_servers = {0, 1}
    pool.cached_stats = cached_stats
    pool.cached_raw_stats = raw_stats
    pool._last_update_time = updated
    pool._last_fetch_duration = 0.1
    pool._last_fetch_error = None
    pool._cache_lock = threading.Lock()
    return pool


def _raw_stats_for(gpu_counts):
    return {
        idx: (
            pd.DataFrame(
                [
                    _gpu_row(index, "RTX 4090", "10 %", "1024/24576")
                    for index in range(count)
                ]
            ),
            {},
        )
        for idx, count in enumerate(gpu_counts)
    }


def test_narrow_windows_never_wrap_a_server_row(monkeypatch, capsys):
    # The blocks format to the live terminal width, so narrow it first.
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((58, 30)))
    pool = _pool()
    pool.pool[1].description = "training-node-with-a-very-long-name"
    pool = _nodes_pool(
        [_server_block(pool, 2) for _ in range(2)], _raw_stats_for([2, 2])
    )
    pool.pool[1].description = "training-node-with-a-very-long-name"

    pool.print_stats(use_cache=True)
    capsys.readouterr()

    frame = pool._tui_diff_screen._previous
    plain = [_without_ansi(line) for line in frame]
    assert all(len(line) <= 58 for line in plain), max(map(len, plain))
    # Compact chrome: short title and hint line instead of the wide ones.
    assert plain[0].startswith("nvidb · 2 servers ·")
    assert "v view" in plain[1] and "⏎ exp" in plain[1]
    assert "[v] Unified view" not in plain[1]
    # The long description yields to the summary instead of wrapping.
    header = next(line for line in plain if "training-node" in line)
    assert header.count("training-node") == 1
    assert "…" in header


def test_loading_is_said_once_per_server(monkeypatch, capsys):
    pool = _nodes_pool(None, _raw_stats_for([0, 0]), updated=None)
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((120, 30)))

    pool.print_stats(use_cache=True)
    capsys.readouterr()

    frame = [_without_ansi(line) for line in pool._tui_diff_screen._previous]
    # Once per server, in the header summary - not again in a block below.
    assert sum("Loading..." in line for line in frame) == len(pool.pool)
    assert not any(line.strip() == "Loading..." for line in frame[3:])


def _click(row):
    return MouseEvent(button=0, column=4, row=row, pressed=True)


def _focus_raw_stats(command):
    return {
        1: (
            pd.DataFrame(
                [
                    _gpu_row(0, "RTX 6000 Ada", "94 %", "24000/49140", "alice(23G)"),
                    _gpu_row(1, "RTX 6000 Ada", "0 %", "0/49140"),
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
                            "used_memory": "23000 MiB",
                            "type": "C",
                            "process_name": "python",
                            "command": command,
                            "cpu_percent": 99.4,
                            "mem_percent": 2.6,
                            "rss_kb": 1677721,
                            "elapsed": "32:05",
                            "state": "RNl",
                            "threads": 33,
                        }
                    ]
                }
            }
        },
    }


def _multi_process_raw_stats():
    raw_stats = _focus_raw_stats("python train.py --model llama")
    processes = raw_stats["_nvidb"]["process_details_by_client"][1]["0"]
    processes.extend(
        [
            {
                **processes[0],
                "pid": 5252,
                "username": "bob",
                "used_memory": "4000 MiB",
                "command": "torchrun eval.py --suite math",
                "cpu_percent": 250.0,
                "mem_percent": 8.0,
                "rss_kb": 3000000,
                "elapsed": "1-02:03:04",
                "state": "S",
            },
            {
                **processes[0],
                "pid": 6262,
                "username": "carol",
                "used_memory": "8000 MiB",
                "command": "python qwen serve.py",
                "cpu_percent": 10.0,
                "mem_percent": 1.0,
                "rss_kb": 500000,
                "elapsed": "02:00",
                "state": "R",
            },
        ]
    )
    return raw_stats


def test_detailed_process_pane_shows_the_whole_command_wrapped(monkeypatch):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    command = (
        "/home/alice/anaconda3/envs/eval/bin/python /home/alice/scripts/"
        "diag_math500_format.py --model /data/models/Qwen3-1.7B --math500 "
        "/data/datasets/MATH-500/test.jsonl --out /data/results/qwen3.jsonl"
    )
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((90, 30)))

    rendered = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(_focus_raw_stats(command), 1))
    )

    assert "training-node GPU 0 [HIGH]" in rendered
    assert (
        "Processes | 1 active | training-node (100.64.0.42) | GPU 0"
        in rendered
    )
    assert "VRAM 23,000 MiB" in rendered
    assert "CPU 99.4%" in rendered and "Threads 33" in rendered
    assert "GPU focus" not in rendered
    # The command is wrapped over several lines rather than cut off.
    assert command not in rendered
    for fragment in command.split():
        assert fragment in rendered
    assert all(len(line) <= 90 for line in rendered.splitlines())


def test_detailed_process_actions_stay_visible_on_a_24_line_terminal(
    monkeypatch,
):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    command = (
        "/opt/conda/bin/python train.py --model /models/Qwen "
        "--dataset /data/test.jsonl --batch 16 --max-new 2048"
    )
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((80, 24)))
    raw_stats = _focus_raw_stats(command)
    raw_stats["_nvidb"]["process_details_by_client"][1]["0"].append(
        {
            "pid": 5252,
            "username": "bob",
            "used_memory": "1000 MiB",
            "type": "C",
            "process_name": "python",
            "command": "python eval.py",
        }
    )

    rendered = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, 1))
    )

    # print_stats adds three fixed header rows above this body.
    assert len(rendered.splitlines()) <= 21
    assert "2 active" in rendered
    assert "[i] INT  [T] TERM  [K] KILL" in rendered
    for fragment in command.split():
        assert fragment in rendered

    pool.unified_show_trends = True
    for sample_index in range(3):
        pool._record_unified_gpu_history(raw_stats, timestamp=sample_index)
    history_view = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, 2))
    )
    assert len(history_view.splitlines()) <= 21
    assert "GPU U" in history_view
    assert "CPU" in history_view
    assert "[i] INT  [T] TERM  [K] KILL" in history_view


def test_long_command_uses_scrollable_pages_on_short_terminals(monkeypatch):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool.unified_active_pane = "process"
    command = (
        "/home/alice/conda/bin/python /workspace/train.py "
        "--model /models/Qwen3-32B --adapter /experiments/checkpoints/adapter "
        "--dataset /datasets/math/test.jsonl "
        "--output /experiments/results/long-name/result.jsonl "
        "--batch-size 16 --max-new-tokens 4096"
    )
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((60, 24)))
    raw_stats = _focus_raw_stats(command)

    first_page = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, 1))
    )
    assert len(first_page.splitlines()) <= 21
    assert "[<]Cmd" in first_page and "[>]Cmd" in first_page
    assert pool._unified_command_line_count > pool._unified_command_page_size
    assert "--max-new-tokens 4096" not in first_page

    pool._click_targets = dict(pool._body_click_targets)
    command_row = next(
        row
        for row, (kind, _value) in pool._click_targets.items()
        if kind == "command"
    )
    assert pool._handle_mouse_event(
        MouseEvent(button=65, column=10, row=command_row + 1, pressed=True)
    ) is True
    assert pool.unified_command_scroll > 0

    pool.unified_command_scroll = 0
    assert pool._handle_keypress("]") is True
    last_page = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, 2))
    )
    assert len(last_page.splitlines()) <= 21
    assert "/experiments/results/long-name/result.jsonl" in last_page
    assert "--max-new-tokens 4096" in last_page


def test_detailed_process_layout_stays_inside_terminal_bounds(monkeypatch):
    command = " ".join(
        ["/opt/conda/bin/python", "/workspace/train.py"]
        + [
            f"--option-{index} /very/long/path/to/checkpoint-{index}/adapter.bin"
            for index in range(18)
        ]
    )
    cases = (
        (40, 20, False),
        (40, 20, True),
        (40, 28, False),
        (60, 36, True),
        (80, 36, True),
        (120, 40, False),
    )

    for width, height, history_enabled in cases:
        pool = _pool()
        pool.display_mode = pool.DISPLAY_MODE_UNIFIED
        pool.unified_detailed = True
        pool.unified_active_pane = "process"
        pool.unified_show_trends = history_enabled
        raw_stats = _focus_raw_stats(command)
        monkeypatch.setattr(
            os,
            "get_terminal_size",
            lambda width=width, height=height: os.terminal_size(
                (width, height)
            ),
        )
        if history_enabled:
            for sample_index in range(3):
                pool._record_unified_gpu_history(
                    raw_stats,
                    timestamp=sample_index,
                )

        rendered = _without_ansi(
            "\n".join(pool._render_unified_gpu_lines(raw_stats, 1))
        )
        lines = rendered.splitlines()
        frame_reserve = (
            3
            if height < 28 or (height < 36 and history_enabled)
            else 4
        )

        assert all(len(line) <= width for line in lines)
        assert len(lines) <= height - frame_reserve
        signal_targets = {
            target
            for _row, _start, _end, target in pool._body_click_regions
            if target[0] == "signal"
        }
        assert signal_targets == {
            ("signal", "INT"),
            ("signal", "TERM"),
            ("signal", "KILL"),
        }
        if not history_enabled:
            assert (
                pool._unified_command_line_count
                > pool._unified_command_page_size
            )
            pager_targets = {
                target
                for _row, _start, _end, target in pool._body_click_regions
                if target[0] == "command_scroll"
            }
            assert pager_targets == {
                ("command_scroll", -1),
                ("command_scroll", 1),
            }


def test_process_filter_is_live_and_keeps_shortcuts_out_of_the_query(
    monkeypatch,
):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool._unified_gpu_count = 2
    raw_stats = _multi_process_raw_stats()
    monkeypatch.setattr(
        os,
        "get_terminal_size",
        lambda: os.terminal_size((100, 40)),
    )

    assert pool._handle_keypress("/") is True
    assert pool.unified_active_pane == "process"
    assert pool.unified_process_filter_editing is True

    for character in "bob":
        assert pool._handle_keypress(character) is True
    assert pool._handle_keypress("q") is True
    assert not pool.quit_flag.is_set()
    assert pool.unified_process_filter == "bobq"
    assert pool._handle_keypress(
        SimpleNamespace(name="KEY_BACKSPACE")
    ) is True

    rendered = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, 1))
    )
    assert "Processes | 1/3 match" in rendered
    assert "Processes | 1/3 match | /bob_" in rendered
    assert "torchrun eval.py --suite math" in rendered
    assert pool._unified_process_count == 1
    assert pool._unified_process_total_count == 3

    assert pool._handle_keypress("\n") is True
    assert pool.unified_process_filter_editing is False
    assert pool.unified_process_filter == "bob"
    assert pool._handle_keypress(SimpleNamespace(name="KEY_ESCAPE")) is True
    assert pool.unified_process_filter == ""

    pool._render_unified_gpu_lines(raw_stats, 2)
    assert pool._unified_process_count == 3


def test_process_sort_supports_keyboard_and_clickable_headers(monkeypatch):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool.unified_active_pane = "process"
    pool._unified_gpu_count = 2
    raw_stats = _multi_process_raw_stats()
    selected_row = pool._build_unified_gpu_table(raw_stats).iloc[0]
    monkeypatch.setattr(
        os,
        "get_terminal_size",
        lambda: os.terminal_size((120, 44)),
    )

    def sorted_pids():
        return [
            process["pid"]
            for process in pool._get_sorted_unified_processes(
                raw_stats,
                selected_row,
            )
        ]

    assert sorted_pids() == [4242, 6262, 5252]
    assert pool._handle_keypress("o") is True
    assert pool.unified_process_sort_mode == "cpu"
    assert pool.unified_process_sort_descending is True
    assert sorted_pids() == [5252, 4242, 6262]

    assert pool._handle_keypress("O") is True
    assert pool.unified_process_sort_descending is False
    assert sorted_pids() == [6262, 4242, 5252]

    pool._render_unified_gpu_lines(raw_stats, 1)
    rss_region = next(
        region
        for region in pool._body_click_regions
        if region[3] == ("process_sort", "rss")
    )
    pool._click_regions = list(pool._body_click_regions)
    row, start, _end, _target = rss_region
    click = MouseEvent(
        button=0,
        column=start + 1,
        row=row + 1,
        pressed=True,
    )
    assert pool._handle_mouse_event(click) is True
    assert pool.unified_process_sort_mode == "rss"
    assert pool.unified_process_sort_descending is True
    assert sorted_pids() == [5252, 4242, 6262]

    assert pool._handle_mouse_event(click) is True
    assert pool.unified_process_sort_descending is False
    assert sorted_pids() == [6262, 4242, 5252]


def test_process_row_count_can_be_resized_without_exceeding_the_screen(
    monkeypatch,
):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool.unified_active_pane = "process"
    raw_stats = _multi_process_raw_stats()
    processes = raw_stats["_nvidb"]["process_details_by_client"][1]["0"]
    template = processes[0]
    for index in range(9):
        processes.append(
            {
                **template,
                "pid": 7000 + index,
                "username": f"worker{index}",
                "used_memory": f"{3000 - index * 100} MiB",
                "command": f"python worker.py --rank {index}",
                "cpu_percent": 50.0 + index,
            }
        )
    width, height = 120, 70
    monkeypatch.setattr(
        os,
        "get_terminal_size",
        lambda: os.terminal_size((width, height)),
    )

    initial = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, 1))
    )
    initial_rows = pool._unified_process_visible_rows
    assert initial_rows >= 2
    assert pool._unified_process_count == 12

    assert pool._handle_keypress("+") is True
    expanded = _without_ansi(
        "\n".join(pool._render_unified_gpu_lines(raw_stats, 2))
    )
    assert pool._unified_process_visible_rows == initial_rows + 1
    assert len(expanded.splitlines()) <= height - 4

    assert pool._handle_keypress("-") is True
    pool._render_unified_gpu_lines(raw_stats, 3)
    assert pool._unified_process_visible_rows == initial_rows

    plus_region = next(
        region
        for region in pool._body_click_regions
        if region[3] == ("process_rows", 1)
    )
    pool._click_regions = list(pool._body_click_regions)
    row, start, _end, _target = plus_region
    assert pool._handle_mouse_event(
        MouseEvent(
            button=0,
            column=start + 1,
            row=row + 1,
            pressed=True,
        )
    ) is True
    assert pool.unified_process_rows == initial_rows + 1
    assert len(initial.splitlines()) <= height - 4


def test_help_overlay_tracks_focus_and_closes_with_mouse(monkeypatch, capsys):
    pool = _pool()
    pool.term = Terminal(force_styling=False)
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool.cached_stats = ["", ""]
    pool.cached_raw_stats = _multi_process_raw_stats()
    pool._last_update_time = 1
    pool._last_fetch_duration = 0.1
    pool._last_fetch_error = None
    pool._cache_lock = threading.Lock()
    monkeypatch.setattr(
        os,
        "get_terminal_size",
        lambda: os.terminal_size((100, 36)),
    )

    pool.print_stats(use_cache=True)
    capsys.readouterr()
    assert pool._handle_keypress("?") is True
    pool.print_stats(use_cache=True)
    gpu_help = _without_ansi(capsys.readouterr().out)
    assert "Help · UNIFIED · GPU/NODE" in gpu_help
    assert "Enter/→/l" in gpu_help
    assert set(pool._click_targets.values()) == {("close_help", None)}

    help_row = next(iter(pool._click_targets))
    assert pool._handle_mouse_event(_click(help_row + 1)) is True
    assert pool.tui_help_visible is False

    assert pool._handle_keypress("\n") is True
    assert pool.unified_active_pane == "process"
    assert pool._handle_keypress("?") is True
    pool.print_stats(use_cache=True)
    process_help = _without_ansi(capsys.readouterr().out)
    assert "Help · UNIFIED · PROCESS" in process_help
    assert "FIND AND ORDER" in process_help
    assert "Mouse header" in process_help
    assert "+ / -" in process_help

    assert pool._handle_keypress("q") is True
    assert pool.tui_help_visible is False
    assert not pool.quit_flag.is_set()


def test_arrow_keys_switch_gpu_and_process_panes_without_leaving_detailed_view():
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool._unified_gpu_count = 2
    pool._unified_process_count = 3

    right = SimpleNamespace(name="KEY_RIGHT")
    assert pool._handle_keypress(right) is True
    assert pool.unified_active_pane == "process"
    assert pool.unified_detailed is True
    assert pool._handle_keypress(right) is False
    assert pool._handle_keypress("\n") is False
    assert pool.unified_active_pane == "process"

    assert pool._handle_keypress("j") is True
    assert pool.unified_selected_process == 1
    assert pool._handle_keypress("k") is True
    assert pool.unified_selected_process == 0

    assert pool._handle_keypress(SimpleNamespace(name="KEY_LEFT")) is True
    assert pool.unified_active_pane == "gpu"
    assert pool._handle_keypress("j") is True
    assert pool.unified_selected_gpu == 1
    assert pool._handle_keypress("h") is False


def test_enter_opens_processes_and_p_is_the_only_visibility_toggle():
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool._unified_gpu_count = 2
    pool._unified_process_count = 3

    assert pool._process_panel_visible() is True
    assert pool._handle_keypress("\n") is True
    assert pool.unified_active_pane == "process"

    pool.refresh_needed.clear()
    assert pool._handle_keypress("\n") is False
    assert pool._process_panel_visible() is True
    assert pool.unified_active_pane == "process"
    assert not pool.refresh_needed.is_set()

    assert pool._handle_keypress("p") is True
    assert pool.unified_process_panel_hidden is True
    assert pool._process_panel_visible() is False
    assert pool.unified_active_pane == "gpu"

    # Enter and Right both reveal the pane and enter its task list.
    assert pool._handle_keypress("\n") is True
    assert pool.unified_process_panel_hidden is False
    assert pool._process_panel_visible() is True
    assert pool.unified_active_pane == "process"

    assert pool._handle_keypress("p") is True
    assert pool._handle_keypress(SimpleNamespace(name="KEY_RIGHT")) is True
    assert pool.unified_process_panel_hidden is False
    assert pool.unified_active_pane == "process"


def test_single_line_enter_opens_processes_while_p_shows_and_hides_them():
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool._unified_gpu_count = 1
    pool._unified_process_count = 1

    assert pool._process_panel_visible() is False
    assert pool._handle_keypress("\n") is True
    assert pool.unified_show_processes is True
    assert pool.unified_active_pane == "process"
    assert pool._handle_keypress("\n") is False
    assert pool.unified_show_processes is True

    assert pool._handle_keypress("p") is True
    assert pool.unified_show_processes is False
    assert pool.unified_active_pane == "gpu"
    assert pool._process_panel_visible() is False

    # Showing the panel does not implicitly enter it; Enter performs that step.
    assert pool._handle_keypress("p") is True
    assert pool.unified_show_processes is True
    assert pool.unified_active_pane == "gpu"
    assert pool._handle_keypress(" ") is True
    assert pool.unified_active_pane == "process"


def test_hidden_detailed_process_panel_is_restored_by_enter(monkeypatch):
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    raw_stats = _focus_raw_stats("python train.py")
    monkeypatch.setattr(
        os,
        "get_terminal_size",
        lambda: os.terminal_size((100, 32)),
    )

    visible = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )
    assert "Processes | 1 active" in visible

    assert pool._handle_keypress("p") is True
    hidden = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )
    assert "Processes | 1 active" not in hidden
    assert pool._unified_process_count == 0

    assert pool._handle_keypress("\n") is True
    restored = "\n".join(
        pool._render_unified_gpu_lines(raw_stats, last_update_time=1)
    )
    assert restored.startswith("Focus process")
    assert "Processes | 1 active" in restored


def test_mouse_selects_gpus_processes_and_signal_actions(monkeypatch, capsys):
    pool = _pool()
    pool.term = Terminal(force_styling=False)
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    pool.cached_stats = ["", ""]
    pool.cached_raw_stats = _focus_raw_stats("python train.py")
    pool._last_update_time = 1
    pool._last_fetch_duration = 0.1
    pool._last_fetch_error = None
    pool._cache_lock = threading.Lock()
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((120, 40)))

    pool.print_stats(use_cache=True)
    capsys.readouterr()

    gpu_rows = {
        gpu: row
        for row, (kind, gpu) in pool._click_targets.items()
        if kind == "gpu"
    }
    assert set(gpu_rows) == {0, 1}

    # A click on another GPU only moves the selection.
    assert pool._handle_mouse_event(_click(gpu_rows[1] + 1)) is True
    assert pool.unified_selected_gpu == 1

    # Clicking the selected GPU is inert; no separate modal view is opened.
    assert pool._handle_mouse_event(_click(gpu_rows[1] + 1)) is False
    assert pool.unified_active_pane == "gpu"
    assert pool._handle_mouse_event(_click(gpu_rows[0] + 1)) is True
    assert pool.unified_selected_gpu == 0

    pool.print_stats(use_cache=True)
    capsys.readouterr()
    process_rows = [
        row
        for row, (kind, _value) in pool._click_targets.items()
        if kind == "process"
    ]
    assert process_rows
    assert pool._handle_mouse_event(_click(process_rows[0] + 1)) is True
    assert pool.unified_active_pane == "process"
    assert pool.unified_selected_process == 0

    term_region = next(
        region
        for region in pool._click_regions
        if region[3] == ("signal", "TERM")
    )
    row, start, _end, _target = term_region
    assert pool._handle_mouse_event(
        MouseEvent(button=0, column=start + 1, row=row + 1, pressed=True)
    ) is True
    assert pool._pending_process_signal["signal"] == "TERM"
    assert pool._pending_process_signal["pid"] == 4242

    # The wheel over a GPU card moves the GPU selection.
    assert pool._handle_mouse_event(
        MouseEvent(button=65, column=4, row=gpu_rows[0] + 1, pressed=True)
    ) is True
    assert pool.unified_selected_gpu == 1


def test_clicking_a_server_row_selects_then_expands_it(monkeypatch, capsys):
    pool = _pool()
    pool.term = Terminal(force_styling=False)
    pool.expanded_servers = set()
    pool._default_expansion_applied = True  # isolate clicks from the startup default
    pool.cached_stats = ["", ""]
    pool.cached_raw_stats = {
        0: (pd.DataFrame(), {}),
        1: (
            pd.DataFrame([_gpu_row(0, "RTX 6000 Ada", "75 %", "24000/49140")]),
            {},
        ),
    }
    pool._last_update_time = 1
    pool._last_fetch_duration = 0.1
    pool._last_fetch_error = None
    pool._cache_lock = threading.Lock()
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((120, 40)))

    pool.print_stats(use_cache=True)
    capsys.readouterr()

    server_rows = {
        index: row
        for row, (kind, index) in pool._click_targets.items()
        if kind == "server"
    }
    assert set(server_rows) == {0, 1}

    assert pool._handle_mouse_event(_click(server_rows[1] + 1)) is True
    assert pool.selected_server == 1
    assert pool.expanded_servers == set()

    assert pool._handle_mouse_event(_click(server_rows[1] + 1)) is True
    assert pool.expanded_servers == {1}


def test_mouse_can_be_turned_off():
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.mouse_enabled = False
    pool._click_targets = {5: ("gpu", 1)}

    assert pool._handle_mouse_event(_click(6)) is False
    assert pool.unified_selected_gpu == 0


def test_process_signals_require_confirmation_and_target_the_selected_host():
    pool = _pool()
    pool.display_mode = pool.DISPLAY_MODE_UNIFIED
    pool.unified_detailed = True
    commands = []

    class SignalClient:
        description = "training-node"
        host = "100.64.0.42"
        port = 22
        status = 0

        def execute_command(self, command):
            commands.append(command)
            if self.status:
                return (
                    "kill: Operation not permitted\n"
                    f"__NVIDB_SIGNAL_STATUS__:{self.status}\n"
                )
            return "\n__NVIDB_SIGNAL_STATUS__:0\n"

    signal_client = SignalClient()
    pool.pool[1] = signal_client
    pool.cached_raw_stats = _focus_raw_stats("python train.py")
    pool._unified_gpu_count = 2
    pool._unified_process_count = 1

    # A signal key first transfers focus from GPU selection to processes.
    assert pool._handle_keypress("T") is True
    assert pool.unified_active_pane == "process"
    assert commands == []
    assert pool._pending_process_signal is None
    assert "press TERM again to arm" in pool._process_action_notice["message"]

    assert pool._handle_keypress("T") is True
    assert pool._pending_process_signal["signal"] == "TERM"
    assert pool._pending_process_signal["pid"] == 4242

    assert pool._handle_keypress("\n") is True
    assert len(commands) == 1
    assert "kill -s TERM -- 4242" in commands[0]
    assert "__NVIDB_SIGNAL_STATUS__" in commands[0]
    assert pool._pending_process_signal is None
    assert "SIGTERM sent to PID 4242 on training-node" in (
        pool._process_action_notice["message"]
    )

    assert pool._handle_keypress("K") is True
    assert pool._pending_process_signal["signal"] == "KILL"
    assert pool._handle_keypress(SimpleNamespace(name="KEY_ESCAPE")) is True
    assert pool._pending_process_signal is None
    assert len(commands) == 1

    signal_client.status = 1
    assert pool._handle_keypress("i") is True
    assert pool._handle_keypress("i") is True
    assert len(commands) == 2
    assert "kill -s INT -- 4242" in commands[-1]
    assert "Operation not permitted" in pool._process_action_notice["message"]
    assert pool._process_action_notice["color"] == "red"


def test_view_changes_are_persisted_to_config(monkeypatch):
    pool = _pool()
    pool._persist_view_enabled = True
    saved = []
    monkeypatch.setattr(
        "nvidb.connection.nvidb_config.save_view_settings",
        lambda settings: saved.append(settings) or True,
    )
    # Keep the test hermetic: never read the developer's real config.yml.
    monkeypatch.setattr(
        "nvidb.connection.nvidb_config.load_view_settings",
        lambda *a, **k: {"theme": "classic"},
    )

    assert pool._handle_keypress("v") is True
    assert pool._handle_keypress("d") is True
    assert pool._handle_keypress("g") is True

    assert pool.unified_group_by_node is False
    assert saved[-1] == {
        "mode": "unified",
        "detailed": True,
        "sort": "node",
        "filter": "all",
        "processes": False,
        "trends": False,
        "group_by_node": False,
        "hide_unsupported": True,
        "mouse": True,
        # Carried over from the stored settings: this TUI does not manage
        # the theme but must not clobber the queue TUI's choice.
        "theme": "classic",
    }
    pool._unified_gpu_count = 4
    saved.clear()
    assert pool._handle_keypress("p") is True
    assert pool.unified_process_panel_hidden is True
    assert saved == []

    assert pool._handle_keypress("p") is True
    pool.unified_detailed = False
    assert pool._handle_keypress("p") is True
    assert pool.unified_show_processes is True
    assert saved[-1]["processes"] is True

    # Moving the selection is not a layout change and must not rewrite the file.
    saved.clear()
    assert pool._handle_keypress("j") is True
    assert saved == []


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
    assert pool.unified_active_pane == "process"
    assert pool.unified_show_processes is False
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


def test_print_refresh_mutes_console_logging_while_it_owns_the_screen(monkeypatch):
    # Background threads (auto-reconnect, etc.) call logging.error/warning/info
    # directly. A raw log line printed while print_stats does cursor-positioned
    # diff rendering corrupts the screen, so print_refresh must silence the
    # console logger for as long as it owns the terminal, and restore it after.
    pool = _pool()
    pool.term = Terminal(force_styling=False)
    pool.mouse_enabled = False
    pool.quit_flag.set()  # exit right after the first draw

    seen_disabled_during_draw = []

    def fake_print_stats(use_cache=True):
        seen_disabled_during_draw.append(not logging.getLogger().isEnabledFor(logging.ERROR))

    monkeypatch.setattr(pool, "print_stats", fake_print_stats)

    assert logging.getLogger().isEnabledFor(logging.ERROR)
    pool.print_refresh()
    assert logging.getLogger().isEnabledFor(logging.ERROR)

    assert seen_disabled_during_draw == [True]
