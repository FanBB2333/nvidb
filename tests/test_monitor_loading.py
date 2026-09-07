"""Fast nodes, slow nodes and authentication must not share a refresh barrier."""

from contextlib import nullcontext
import os
import threading
import time

import pandas as pd
import pytest

from nvidb import connection
from nvidb.connection import BaseClient, NVClientPool


class Node(BaseClient):
    def __init__(self, name, *, connect_gate=None, gpu_gate=None, details_gate=None):
        super().__init__()
        self.description = self.host = name
        self.port = 22
        self.connect_gate = connect_gate
        self.gpu_gate = gpu_gate
        self.details_gate = details_gate
        self.calls = 0
        self.connect_entered = threading.Event()
        self.gpu_entered = threading.Event()
        self.details_entered = threading.Event()

    def connect(self, **kwargs):
        self.connect_entered.set()
        if self.connect_gate:
            assert self.connect_gate.wait(5)
        self.connected = True
        return True

    def query_nvml_snapshot(self):
        self.calls += 1
        self.gpu_entered.set()
        if self.gpu_gate:
            assert self.gpu_gate.wait(5)
        return {
            "ok": True, "backend": "ctypes", "gpus": [{
                "gpu_index": 0, "name": "NVIDIA Test GPU",
                "memory_total_bytes": 8 * 1024**3,
                "memory_used_bytes": 1024**3,
                "memory_free_bytes": 7 * 1024**3,
                "gpu_util_percent": 25, "processes": [],
            }],
        }

    def get_system_stats(self):
        self.details_entered.set()
        if self.details_gate:
            assert self.details_gate.wait(5)
        return {"cpu_cores": 8}

    def execute_command(self, command):
        return '{"ok": false}'


@pytest.fixture
def make_pool():
    pools = []

    def make(*nodes):
        pool = NVClientPool(None, defer_connect=True)
        pool.pool = list(nodes)
        pool.expanded_servers = set(range(len(nodes)))
        pools.append(pool)
        return pool

    yield make
    for pool in pools:
        pool.quit_flag.set()
        for node in pool.pool:
            for name in ("connect_gate", "gpu_gate", "details_gate"):
                gate = getattr(node, name, None)
                if gate:
                    gate.set()
        for worker in pool._node_workers:
            worker.join(1)
            assert not worker.is_alive()


def wait_for(pool, predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        pool.refresh_needed.clear()
        with pool._cache_lock:
            if predicate(pool.cached_raw_stats):
                return
        pool.refresh_needed.wait(0.05)
    pytest.fail("node did not publish its sample")


def test_interactive_constructor_defers_connections(monkeypatch):
    calls = []
    monkeypatch.setattr(NVClientPool, "connect_all", lambda self: calls.append(True))
    NVClientPool(None, defer_connect=True)
    assert calls == []
    NVClientPool(None)
    assert calls == [True]  # --once/log/web keep synchronous compatibility.


@pytest.mark.parametrize("stage", ["connect_gate", "gpu_gate"])
def test_slow_node_does_not_delay_fast_node_or_subsequent_refreshes(make_pool, stage):
    slow = Node("slow", **{stage: threading.Event()})
    fast = Node("fast")
    pool = make_pool(slow, fast)
    pool._background_refresh(0.05)
    wait_for(pool, lambda raw: fast.calls >= 2 and 1 in raw and not raw[1][0].empty)
    assert pool.cached_raw_stats[0][1]["loading"] is True
    assert pool.cached_raw_stats[1][0].iloc[0]["name"] == "Test GPU"
    assert len(pool._node_workers) == 2
    assert all(worker.daemon for worker in pool._node_workers)


def test_gpu_rows_appear_before_slow_details_and_sample_age_is_preserved(make_pool):
    node = Node("details-slow", details_gate=threading.Event())
    pool = make_pool(node)
    pool._background_refresh(60)
    assert node.details_entered.wait(2)
    with pool._cache_lock:
        quick = pool.cached_raw_stats[0]
    assert not quick[0].empty
    assert "system_stats" not in quick[1]
    sampled_at = quick[1]["_monitor"]["updated"]
    assert quick[1]["_monitor"]["phase"] == "details"
    node.details_gate.set()
    wait_for(pool, lambda raw: raw[0][1]["_monitor"]["phase"] == "ready")
    complete = pool.cached_raw_stats[0]
    assert complete[1]["system_stats"]["cpu_cores"] == 8
    assert complete[1]["_monitor"]["updated"] == sampled_at
    assert "system_stats" not in quick[1]  # Published snapshots are immutable.
    assert set(complete[1]["_monitor"]["timings"]) == {"ssh", "gpu", "details"}


def test_failed_collection_is_local_and_recovers(make_pool):
    node = Node("broken-once")
    original = node.get_full_gpu_info
    attempts = []

    def fail_once(**kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("test collection error")
        return original(**kwargs)

    node.get_full_gpu_info = fail_once
    pool = make_pool(node, Node("healthy"))
    pool._background_refresh(0.2)
    wait_for(pool, lambda raw: 0 in raw and raw[0][1].get("error"))
    assert "test collection error" in pool.cached_raw_stats[0][1]["error"]
    wait_for(pool, lambda raw: 0 in raw and not raw[0][0].empty and 1 in raw and not raw[1][0].empty)


def test_pending_nodes_remain_visible_and_expand_when_they_arrive(make_pool, monkeypatch, capsys):
    pool = make_pool(Node("fast"), Node("slow"))
    pool.debug = True
    pool._set_node_phase(1, "connecting")
    pool.pool[0].connect()
    sample = pool._collect_client_gpu_info(pool.pool[0])
    pool._publish_node_sample(0, sample, phase="ready", started=time.time(), timings={})
    pool._node_workers = [threading.Thread()]  # Use live-snapshot formatting.
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((140, 40)))
    try:
        pool.print_stats(use_cache=True)
        capsys.readouterr()
        frame = "\n".join(pool._tui_diff_screen._previous)
        assert "Connecting SSH" in frame
        assert "Loading 1: slow" in frame
        assert "Test GPU" in frame
        assert pool.expanded_servers == {0, 1}
        assert not pool._default_expansion_applied
        status = "\n".join(pool._get_unified_node_status_lines(pool.cached_raw_stats, time.time()))
        assert "slow" in status and "Connecting SSH" in status
        assert "without GPU support" not in status
        pool._publish_node_sample(1, sample, phase="ready", started=time.time(), timings={})
        pool.print_stats(use_cache=True)
        assert pool.expanded_servers == {0, 1}
        assert pool._default_expansion_applied
    finally:
        pool._node_workers = []


def test_publishing_one_node_does_not_duplicate_another_nodes_history(make_pool):
    pool = make_pool(Node("one"), Node("two"))
    for node in pool.pool:
        node.connect()
    one = pool._collect_client_gpu_info(pool.pool[0])
    two = pool._collect_client_gpu_info(pool.pool[1])
    for idx, sample in [(0, one), (1, two), (1, two)]:
        pool._publish_node_sample(idx, sample, phase="ready", started=time.time(), timings={}, record=True)
    assert sorted(map(len, pool._unified_gpu_history.values())) == [1, 2]


def test_late_worker_results_are_ignored_after_quit(make_pool):
    pool = make_pool(Node("one"))
    pool.quit_flag.set()
    pool._publish_node_sample(0, (pd.DataFrame(), {}, {}, {}), phase="ready", started=0, timings={})
    assert pool.cached_raw_stats == {}
    assert not pool.refresh_needed.is_set()


def test_tui_draws_before_connecting_and_quits_with_ssh_still_pending(make_pool, monkeypatch, capsys):
    slow = Node("slow", connect_gate=threading.Event())
    pool = make_pool(slow)
    frames = []

    def draw(**kwargs):
        frames.append(slow.connect_entered.is_set())

    def key(**kwargs):
        assert slow.connect_entered.wait(1)
        return "q"

    monkeypatch.setattr(pool, "print_stats", draw)
    monkeypatch.setattr(pool.term, "inkey", key)
    pool.print_refresh()
    output = capsys.readouterr().out
    assert "Error occurred" not in output
    assert frames[0] is False
    assert pool.quit_flag.is_set()
    assert pool._node_workers[0].is_alive()


def test_deferred_advanced_metrics_still_respect_gpu_allowlist(make_pool):
    node = Node("restricted")
    node.gpu_ids = [0]

    def advanced(stats, info):
        return stats, {**info, "advanced_metrics": pd.DataFrame([{"GPU": 0}, {"GPU": 1}])}

    node._add_advanced_gpu_info = advanced
    pool = make_pool(node)
    pool.dcgm = True
    pool._background_refresh(60)
    pool._background_refresh(60)
    assert len(pool._node_workers) == 1
    wait_for(pool, lambda raw: 0 in raw and raw[0][1]["_monitor"]["phase"] == "ready")
    assert pool.cached_raw_stats[0][1]["advanced_metrics"]["GPU"].tolist() == [0]


@pytest.mark.parametrize("debug", [False, True])
def test_live_monitor_does_not_query_dcgm_by_default(make_pool, debug):
    node = Node("one")
    calls = []

    def advanced(stats, info):
        calls.append(True)
        return stats, {**info, "advanced_source": "dcgm", "advanced_supported": True}

    node._add_advanced_gpu_info = advanced
    pool = make_pool(node)
    pool.debug = debug
    assert pool.dcgm is False
    pool._background_refresh(0.05)
    wait_for(pool, lambda raw: node.calls >= 2 and 0 in raw and raw[0][1]["_monitor"]["phase"] == "ready")
    assert calls == []
    stats, info = pool.cached_raw_stats[0]
    assert not stats.empty
    assert "system_stats" in info
    assert "advanced_source" not in info
    assert "advanced_metrics" not in info
    assert "DCGM" not in "\n".join(pool._format_client_blocks([(stats, info)]))


def test_quick_snapshot_keeps_known_processes_without_ps(make_pool):
    node = Node("busy")
    node.connect()
    payload = node.query_nvml_snapshot()
    payload["gpus"][0]["processes"] = [{
        "pid": 123, "process_name": "python", "used_gpu_memory_bytes": 1024**3,
    }]
    node.query_nvml_snapshot = lambda: payload
    node.get_pid_user_map = lambda pids: pytest.fail("quick sample must not run ps")
    pool = make_pool(node)
    base = node.get_full_gpu_info(include_advanced=False)
    sample = pool._collect_client_gpu_info(node, snapshot=base, enrich=False)
    assert sample[3]["0"][0]["pid"] == 123
    assert sample[3]["0"][0]["username"] == "N/A"
    assert not node.details_entered.is_set()


def test_cached_node_blocks_avoid_reformatting_unchanged_nodes(make_pool, monkeypatch):
    node = Node("one")
    node.connect()
    pool = make_pool(node)
    sample = pool._collect_client_gpu_info(node)
    calls = []
    original = pool._format_fixed_width_table

    def format_table(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(pool, "_format_fixed_width_table", format_table)
    pool._format_client_blocks([sample[:2]])
    pool._format_client_blocks([sample[:2]])
    assert len(calls) == 1
    monkeypatch.setattr(connection, "_monitor_theme", "muted")
    pool._format_client_blocks([sample[:2]])
    assert len(calls) == 2


@pytest.mark.parametrize("mode", ["nodes", "unified"])
@pytest.mark.parametrize("width", [48, 120])
def test_collection_phases_do_not_change_normal_frames(make_pool, monkeypatch, capsys, mode, width):
    pool = make_pool(Node("one"))
    assert pool.debug is False
    pool.display_mode = mode
    pool.pool[0].connect()
    sample = pool._collect_client_gpu_info(pool.pool[0])
    monkeypatch.setattr(connection.time, "time", lambda: 100.0)
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((width, 40)))
    pool._publish_node_sample(0, sample, phase="ready", started=100, timings={"ssh": 0.1})
    pool._node_workers = [threading.Thread()]
    try:
        pool.print_stats(use_cache=True)
        before = list(pool._tui_diff_screen._previous)
        assert "Updated" in "\n".join(before)
        for phase in ("collecting", "details", "ready"):
            pool.refresh_needed.clear()
            monkeypatch.setattr(connection.time, "time", lambda: 110.0)
            pool._set_node_phase(0, phase)
            assert not pool.refresh_needed.is_set()
            pool.print_stats(use_cache=True)
            assert pool._tui_diff_screen._previous == before
        text = "\n".join(before).lower()
        assert not any(word in text for word in ("telemetry", "loading", "collecting", "stale", "ssh 0.1s"))
    finally:
        capsys.readouterr()
        pool._node_workers = []


def test_debug_enables_diagnostics_without_affecting_cached_tables(make_pool, monkeypatch, capsys):
    pool = make_pool(Node("one"))
    pool.pool[0].connect()
    sample = pool._collect_client_gpu_info(pool.pool[0])
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((140, 40)))
    pool._publish_node_sample(0, sample, phase="ready", started=time.time(), timings={"ssh": 0.2})
    pool._node_workers = [threading.Thread()]
    try:
        pool.print_stats(use_cache=True)
        assert "Telemetry" not in "\n".join(pool._tui_diff_screen._previous)
        pool.debug = True
        pool.print_stats(use_cache=True)
        assert "Telemetry" in "\n".join(pool._tui_diff_screen._previous)
        assert "SSH 0.2s" in "\n".join(pool._tui_diff_screen._previous)
        pool.refresh_needed.clear()
        pool._set_node_phase(0, "collecting")
        assert pool.refresh_needed.is_set()
        pool.print_stats(use_cache=True)
        assert "Collecting GPU" in "\n".join(pool._tui_diff_screen._previous)
        pool.debug = False
        pool.print_stats(use_cache=True)
        assert "Collecting GPU" not in "\n".join(pool._tui_diff_screen._previous)
    finally:
        capsys.readouterr()
        pool._node_workers = []


def test_errors_remain_visible_without_debug(make_pool, monkeypatch, capsys):
    pool = make_pool(Node("offline"))
    pool._publish_node_sample(0, (pd.DataFrame(), {"error": "Connection timed out", "error_type": "connect"}, {}, {}), phase="error", started=time.time(), timings={})
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((120, 40)))
    for mode in ("nodes", "unified"):
        pool.display_mode = mode
        pool.print_stats(use_cache=True)
        assert "Connection timed out" in "\n".join(pool._tui_diff_screen._previous)
    capsys.readouterr()


def test_initial_loading_messages_are_debug_only(make_pool, monkeypatch, capsys):
    pool = make_pool(Node("pending"))
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((120, 40)))
    for mode in ("nodes", "unified"):
        pool.display_mode = mode
        pool.print_stats(use_cache=True)
        assert "loading" not in "\n".join(pool._tui_diff_screen._previous).lower()
    pool.debug = True
    pool.print_stats(use_cache=True)
    assert "Loading GPU data" in "\n".join(pool._tui_diff_screen._previous)
    capsys.readouterr()


def test_debug_does_not_change_the_existing_d_binding_or_saved_settings(make_pool):
    pool = make_pool(Node("one"))
    pool.debug = True
    assert pool._handle_keypress("d") is False  # No binding in per-node view.
    assert pool._handle_keypress("v")
    was_detailed = pool.unified_detailed
    assert pool._handle_keypress("d")
    assert pool.unified_detailed is not was_detailed
    assert pool.debug is True
    assert "debug" not in pool._current_view_settings()


def test_confirmed_signal_does_not_block_the_live_input_loop(make_pool, monkeypatch):
    node = Node("one")
    pool = make_pool(node)
    entered = threading.Event()
    release = threading.Event()
    commands = []

    def execute(command):
        commands.append(command)
        entered.set()
        assert release.wait(5)
        return "__NVIDB_SIGNAL_STATUS__:0\n"

    node.execute_command = execute
    monkeypatch.setattr(pool, "_selected_process_context", lambda: {
        "process": {"pid": 123, "command": "python train.py"}, "client_index": 0,
    })
    pending = {
        "signal": "TERM", "pid": 123, "client_index": 0, "node": "one",
        "command": "python train.py", "expires_at": time.monotonic() + 5,
    }
    pool._node_workers = [threading.Thread()]
    try:
        pool._pending_process_signal = pending
        assert pool._confirm_process_signal()
        assert entered.wait(1)
        pool._pending_process_signal = pending
        assert pool._confirm_process_signal()
        assert len(commands) == 1  # At most one pending confirmed action.
        assert pool._handle_keypress("q")
        assert pool.quit_flag.is_set()
    finally:
        release.set()
        pool._signal_worker.join(1)
        pool._node_workers = []


def test_secret_request_is_answered_only_on_ui_thread(make_pool, monkeypatch):
    pool = make_pool(Node("one"))
    pool._mouse_reporting = False
    pool.mouse_enabled = False
    pool._cbreak_context = nullcontext()
    seen = []
    answers = []

    def prompt(text):
        seen.append(threading.current_thread())
        return "test secret"

    monkeypatch.setattr(connection, "_prompt_secret", prompt)
    worker = threading.Thread(target=lambda: answers.append(pool._request_secret("Password: ")))
    worker.start()
    try:
        assert pool.refresh_needed.wait(1)
        pool._answer_secret_request()
        worker.join(1)
        assert not worker.is_alive()
        assert answers == ["test secret"]
        assert seen == [threading.current_thread()]
    finally:
        pool.quit_flag.set()
        worker.join(1)
        pool._cbreak_context.__exit__(None, None, None)


def test_waiting_for_credentials_can_be_cancelled(make_pool):
    pool = make_pool(Node("one"))
    cancelled = threading.Event()

    def request():
        try:
            pool._request_secret("Password: ")
        except EOFError:
            cancelled.set()

    worker = threading.Thread(target=request)
    worker.start()
    assert pool.refresh_needed.wait(1)
    pool.quit_flag.set()
    worker.join(1)
    assert cancelled.is_set()
