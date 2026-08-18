"""Tests for the queue TUI's rendering and key handling.

The worker thread is never started here: the interface is driven from a plain
snapshot dictionary, which is exactly the contract it renders from.
"""
import os
import re

import pytest
from blessed.keyboard import Keystroke

from nvidb.mouse import MouseEvent
from nvidb.sched.tui import QueueTUI, _Worker

ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_MISSING = object()


def _plain(text):
    return ANSI.sub("", text)


@pytest.fixture(autouse=True)
def wide_terminal():
    """blessed reads COLUMNS/LINES when it has no tty, which is the case here."""
    previous = {key: os.environ.get(key) for key in ("COLUMNS", "LINES")}
    os.environ["COLUMNS"] = "150"
    os.environ["LINES"] = "42"
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _gpu(index=0, name="RTX 3090 Ti", total=24564, used=0, external=0, reserved=0,
         free=None, jobs=0, attribution="processes", procs=0, processes=None,
         queue_mem=0):
    return {
        "index": index,
        "name": name,
        "mem_total_mb": total,
        "mem_used_mb": used,
        "mem_used_percent": int(round(100 * used / total)) if total else None,
        "external_mem_mb": external,
        "external_procs": procs,
        "queue_mem_mb": queue_mem,
        "attribution": attribution,
        "reserved_mb": reserved,
        "queue_jobs": jobs,
        "free_mb": total - external - reserved - 512 if free is None else free,
        "util_percent": 30,
        "mem_util_percent": 12,
        "temperature_c": 50,
        "processes": processes or [],
        "updated_at": None,
    }


def _proc(pid=999, mem=4096, name="python train.py", user="alice", job_id=None,
          job_name=None):
    return {
        "pid": pid,
        "mem_mb": mem,
        "name": name,
        "user": user,
        "type": "C",
        "job_id": job_id,
        "job_name": job_name,
        "managed": job_id is not None,
    }


def _job(job_id, state="running", name="train", node="small-node", **overrides):
    job = {
        "id": job_id,
        "name": name,
        "state": state,
        "command": "python train.py --epochs 10",
        "workdir": "/data",
        "env": {},
        "priority": 0,
        "gpus": 1,
        "vram_mb": 20480,
        "node_constraint": None,
        "depends_on": [],
        "max_runtime_s": None,
        "submitter": "client-a",
        "tags": [],
        "created_at": "2026-07-27T10:00:00+00:00",
        "started_at": "2026-07-27T10:00:01+00:00",
        "finished_at": None,
        "elapsed_s": 125.0,
        "node": node,
        "gpu_ids": [0],
        "remote_pid": 4321,
        "run_dir": "/home/u/.nvidb/jobs/1",
        "exit_code": None,
        "attempt": 1,
        "max_retries": 0,
        "last_error": None,
        "heartbeat_at": None,
        "gpu_mem_mb": None,
        "result": None,
        "notes": None,
        "progress": None,
        "progress_at": None,
    }
    job.update(overrides)
    return job


def _snapshot(jobs=None, recent=None, nodes=None, counts=_MISSING):
    return {
        "generated_at": "2026-07-27T10:02:00+00:00",
        "db_path": "/tmp/queue.db",
        "last_tick_at": "2026-07-27T10:01:58+00:00",
        "tick_lock": None,
        "settings": {"headroom_mb": 512, "max_jobs_per_gpu": 4, "placement": "spread"},
        "counts": {"running": 1, "pending": 1} if counts is _MISSING else counts,
        "nodes": nodes
        if nodes is not None
        else [
            {
                "name": "big-node",
                "hostname": "10.0.0.1",
                "port": 22,
                "username": "u",
                "state": "up",
                "enabled": True,
                "last_seen": "2026-07-27T10:01:58+00:00",
                "last_error": None,
                "gpu_count": 1,
                "probed_at": None,
                "gpus": [_gpu(name="RTX PRO 5000", total=73415, used=69000, external=69000, procs=2)],
            },
            {
                "name": "small-node",
                "hostname": "10.0.0.2",
                "port": 2222,
                "username": "u",
                "state": "up",
                "enabled": True,
                "last_seen": "2026-07-27T10:01:58+00:00",
                "last_error": None,
                "gpu_count": 1,
                "probed_at": None,
                "gpus": [_gpu()],
            },
        ],
        "jobs": jobs if jobs is not None else [_job(1), _job(2, state="pending", name="eval")],
        "recent": recent or [],
    }


def _state(snapshot=_MISSING, **overrides):
    state = {
        "snapshot": _snapshot() if snapshot is _MISSING else snapshot,
        "notice": None,
        "log_text": "",
        "log_job": None,
        "busy": False,
        "auto_tick": True,
        "error": None,
    }
    state.update(overrides)
    return state


def _tui(width=150):
    os.environ["COLUMNS"] = str(width)
    return QueueTUI()


def _render(tui, state=None):
    return _plain(tui.render(state or _state()))


def _mouse_at(output, text, *, button=0, occurrence=0):
    matches = []
    for row, line in enumerate(output.splitlines()):
        start = 0
        while True:
            column = line.find(text, start)
            if column < 0:
                break
            matches.append((row, column))
            start = column + 1
    row, column = matches[occurrence]
    return MouseEvent(
        button=button,
        column=column + 1,
        row=row + 1,
        pressed=True,
    )


# --- rendering -------------------------------------------------------------

def test_the_screen_shows_nodes_capacity_and_jobs():
    output = _render(_tui())
    assert "big-node" in output and "small-node" in output
    # The GPU model lives on the node header now, once per node.
    assert "1× RTX PRO 5000" in output
    # The grid cell keeps the free amount and names the foreign processes.
    assert "free" in output and "·2p" in output
    assert "python train.py" in output
    # The keybinding footer is always reachable.
    assert "q quit" in output


def test_gpus_share_lines_instead_of_taking_one_each():
    snapshot = _snapshot()
    snapshot["nodes"][0]["gpus"] = [
        _gpu(index, name="RTX PRO 5000", total=73415) for index in range(4)
    ]
    tui = _tui(width=150)
    output = _plain(tui.render(_state(snapshot)))
    gpu_rows = [line for line in output.splitlines() if "G0 " in line or "G3 " in line]
    # Four GPUs at 150 columns fit on a single grid line.
    assert any("G0 " in line and "G3 " in line for line in gpu_rows)


def test_external_processes_collapse_to_one_summary_line():
    snapshot = _snapshot()
    snapshot["nodes"][0]["gpus"] = [
        _gpu(
            0,
            total=73415,
            used=69000,
            external=69000,
            procs=3,
            processes=[
                _proc(100, 40000, "python train.py", "alice"),
                _proc(101, 20000, "python infer.py", "bob"),
                _proc(102, 5000, "jupyter", "carol"),
                _proc(103, 4000, "python small.py", "dave"),
            ],
        )
    ]
    output = _plain(_tui().render(_state(snapshot)))
    ext_lines = [line for line in output.splitlines() if line.lstrip().startswith("ext ")]
    assert len(ext_lines) == 1
    # Biggest foreign process first, overflow counted instead of listed.
    assert "alice" in ext_lines[0] and "39.1G" in ext_lines[0]
    assert "+1 more" in ext_lines[0]


def test_procs_all_still_lists_every_process_line():
    snapshot = _snapshot()
    snapshot["nodes"][0]["gpus"] = [
        _gpu(
            0,
            total=73415,
            used=69000,
            external=69000,
            procs=2,
            processes=[
                _proc(100, 40000, "python train.py", "alice"),
                _proc(101, 20000, "python infer.py", "bob"),
            ],
        )
    ]
    tui = _tui()
    tui.proc_view = "all"
    output = _plain(tui.render(_state(snapshot)))
    assert "alice" in output and "bob" in output
    assert "unmanaged" in output


def test_capacity_reflects_foreign_usage():
    output = _render(_tui())
    # 73415 total - 69000 foreign - 512 headroom, rendered compactly.
    assert "3.8G" in output


def test_a_blind_node_says_so_instead_of_claiming_zero_processes():
    snapshot = _snapshot()
    snapshot["nodes"][1]["gpus"] = [_gpu(used=8000, external=0, reserved=8192,
                                         attribution="blind", jobs=1)]
    output = _render(_tui(), _state(snapshot))
    assert "·blind" in output


def test_progress_replaces_the_command_in_the_job_row():
    snapshot = _snapshot(jobs=[_job(1, progress="epoch 3/10 loss 0.42")])
    output = _render(_tui(), _state(snapshot))
    assert "PROGRESS / COMMAND" in output
    assert "▸ epoch 3/10 loss 0.42" in output
    # The command is still reachable in the detail pane.
    assert "python train.py" in output


def test_a_job_without_progress_still_shows_its_command():
    output = _render(_tui())
    assert "PROGRESS" not in output
    assert "python train.py --epochs 10" in output


def test_the_detail_pane_shows_both_the_note_and_the_live_status():
    snapshot = _snapshot(
        jobs=[_job(1, progress="epoch 3/10", notes="baseline A, lr=1e-4")]
    )
    output = _render(_tui(), _state(snapshot))
    assert "note baseline A, lr=1e-4" in output
    assert "live " in output


def test_a_long_note_wraps_instead_of_being_truncated():
    from nvidb.sched.model import display_width

    os.environ["LINES"] = "32"
    note = (
        "开头标记：这个任务用于验证详情区域中的长说明能够根据终端宽度自动换行，"
        "其中包含中文、English words 和参数 --learning-rate 1e-4。"
        "继续添加内容以确保单行无法容纳，最后保留一个容易检查的结尾标记。"
        "结尾标记：NOTE-COMPLETE"
    )
    snapshot = _snapshot(jobs=[_job(1, notes=note)], nodes=[])
    tui = _tui(width=60)

    output = _render(tui, _state(snapshot))

    assert "开头标记" in output
    assert "NOTE-COMPLETE" in output
    assert tui.detail_pages == 1
    for line in output.splitlines():
        assert display_width(line) <= 60


def test_a_very_long_note_can_be_paged_to_its_end():
    os.environ["LINES"] = "20"
    note = "\n".join(
        [f"note line {index:02d}: retained detail text" for index in range(30)]
        + ["END-OF-NOTE"]
    )
    snapshot = _snapshot(jobs=[_job(1, notes=note)], nodes=[])
    state = _state(snapshot)
    tui = _tui(width=70)

    first_page = _render(tui, state)

    assert tui.detail_pages > 1
    assert "[1/" in first_page
    assert "END-OF-NOTE" not in first_page

    for _ in range(tui.detail_pages - 1):
        _press(tui, "]")
        last_page = _render(tui, state)

    assert f"[{tui.detail_pages}/{tui.detail_pages}" in last_page
    assert "END-OF-NOTE" in last_page


def test_an_unreachable_node_shows_its_error():
    snapshot = _snapshot()
    snapshot["nodes"][0].update(state="down", last_error="timed out", gpus=[])
    output = _render(_tui(), _state(snapshot))
    assert "down" in output
    assert "timed out" in output


def test_every_line_fits_the_terminal_width():
    snapshot = _snapshot(jobs=[_job(1, name="a-very-long-job-name-that-overflows")])
    tui = _tui(width=80)
    for line in _plain(tui.render(_state(snapshot))).splitlines():
        assert len(line) <= 80, line


def test_wide_characters_do_not_push_lines_past_the_edge():
    from nvidb.sched.model import display_width

    snapshot = _snapshot(
        jobs=[
            _job(
                1,
                name="基线实验对照组",
                notes="复现 issue #12，学习率 1e-4",
                progress="第 3 轮 / 共 10 轮，损失 0.42",
            )
        ]
    )
    snapshot["nodes"][0]["name"] = "训练节点一号"
    tui = _tui(width=100)
    for line in _plain(tui.render(_state(snapshot))).splitlines():
        assert display_width(line) <= 100, line


def test_the_screen_survives_an_empty_queue():
    output = _render(_tui(), _state(_snapshot(jobs=[], counts={})))
    assert "queue empty" in output
    assert "no jobs match this filter" in output


def test_before_the_first_refresh_the_screen_explains_itself():
    output = _plain(_tui().render(_state(snapshot=None)))
    assert "connecting to nodes" in output


def test_a_worker_error_is_shown_rather_than_swallowed():
    output = _render(_tui(), _state(error="TransportError: host unreachable"))
    assert "host unreachable" in output


# --- selection and filtering ----------------------------------------------

def _press(tui, char, name=None):
    key = Keystroke(ucs=char if name is None else "", code=None, name=name)
    return tui.handle_key(key)


def test_j_and_k_move_the_job_selection():
    tui = _tui()
    _render(tui)
    assert tui.job_index == 0
    _press(tui, "j")
    assert tui.job_index == 1
    _press(tui, "j")  # already at the end
    assert tui.job_index == 1
    _press(tui, "k")
    assert tui.job_index == 0


def test_tab_switches_which_pane_has_focus():
    tui = _tui()
    _render(tui)
    assert tui.focus == "jobs"
    _press(tui, "", name="KEY_TAB")
    assert tui.focus == "nodes"
    _press(tui, "j")
    assert tui.node_index == 1
    assert tui.job_index == 0  # the job pane did not move


def test_enter_has_visible_expand_and_collapse_states():
    snapshot = _snapshot(jobs=[_job(1, notes="visible detail")], nodes=[])
    state = _state(snapshot)
    tui = _tui()

    expanded = _render(tui, state)
    assert "▼ JOB 1 DETAIL" in expanded
    assert "Enter hide detail" in expanded
    assert "note visible detail" in expanded

    _press(tui, "", name="KEY_ENTER")
    collapsed = _render(tui, state)
    assert "▶ JOB 1 DETAIL HIDDEN" in collapsed
    assert "Enter show detail" in collapsed
    assert "note visible detail" not in collapsed

    # Some terminal definitions report Return separately from KEY_ENTER.
    _press(tui, "", name="KEY_RETURN")
    reopened = _render(tui, state)
    assert "▼ JOB 1 DETAIL" in reopened
    assert "note visible detail" in reopened


def test_the_filter_cycles_through_the_useful_views():
    tui = _tui()
    snapshot = _snapshot(
        jobs=[_job(1), _job(2, state="pending")],
        recent=[_job(3, state="completed", exit_code=0)],
    )
    _render(tui, _state(snapshot))
    assert {job["id"] for job in tui.jobs} == {1, 2}

    _press(tui, "f")  # all
    _render(tui, _state(snapshot))
    assert {job["id"] for job in tui.jobs} == {1, 2, 3}

    _press(tui, "f")  # running
    _render(tui, _state(snapshot))
    assert {job["id"] for job in tui.jobs} == {1}

    _press(tui, "f")  # pending
    _render(tui, _state(snapshot))
    assert {job["id"] for job in tui.jobs} == {2}

    _press(tui, "f")  # finished
    _render(tui, _state(snapshot))
    assert {job["id"] for job in tui.jobs} == {3}


def test_T_cycles_the_theme_and_persists_it(monkeypatch):
    from nvidb.sched import tui as tui_module

    saved = {}
    monkeypatch.setattr(
        tui_module.nvidb_config, "load_view_settings", lambda *a, **k: {"mouse": True}
    )
    monkeypatch.setattr(
        tui_module.nvidb_config,
        "save_view_settings",
        lambda settings, *a, **k: saved.update(settings) or True,
    )
    tui = _tui()
    _render(tui)
    assert tui.theme == "classic"
    _press(tui, "T")
    assert tui.theme == "muted"
    assert saved["theme"] == "muted"
    assert "T theme:muted" in _render(tui)
    _press(tui, "T")
    assert tui.theme == "classic"


def test_q_quits_and_other_keys_do_not():
    tui = _tui()
    _render(tui)
    assert _press(tui, "j") is True
    assert _press(tui, "q") is False


def test_help_opens_and_closes_without_quitting():
    tui = _tui()
    _render(tui)
    _press(tui, "?")
    assert "Force a scheduler tick now" in _render(tui)
    # `q` inside help closes the panel rather than exiting the program.
    assert _press(tui, "q") is True
    assert tui.show_help is False


# --- mouse interaction -----------------------------------------------------

def test_clicking_job_rows_selects_then_toggles_the_current_detail():
    tui = _tui()
    output = _render(tui)

    click = _mouse_at(output, "eval")
    assert tui.handle_mouse(click) is True
    assert tui.focus == "jobs"
    assert tui.selected_job()["id"] == 2

    assert tui.show_detail is True
    assert tui.handle_mouse(click) is True
    assert tui.show_detail is False


def test_clicking_any_line_in_a_node_card_selects_that_node():
    tui = _tui()
    output = _render(tui)

    assert tui.handle_mouse(_mouse_at(output, "RTX 3090 Ti")) is True
    assert tui.focus == "nodes"
    assert tui.selected_node()["name"] == "small-node"


def test_wheel_routes_to_the_pane_under_the_pointer():
    tui = _tui()
    output = _render(tui)

    assert tui.handle_mouse(_mouse_at(output, "JOBS (2)", button=65)) is True
    assert tui.focus == "jobs"
    assert tui.selected_job()["id"] == 2

    assert tui.handle_mouse(_mouse_at(output, "NODES", button=65)) is True
    assert tui.focus == "nodes"
    assert tui.selected_node()["name"] == "small-node"


def test_wheel_pages_wrapped_detail_and_scrolls_back_through_logs():
    os.environ["LINES"] = "20"
    note = "\n".join(f"detail line {index}" for index in range(30))
    snapshot = _snapshot(jobs=[_job(1, notes=note)], nodes=[])
    state = _state(snapshot)
    tui = _tui(width=80)
    output = _render(tui, state)

    assert tui.detail_pages > 1
    assert tui.handle_mouse(_mouse_at(output, "JOB 1 DETAIL", button=65)) is True
    assert tui.detail_page == 1

    _press(tui, "L")
    log_state = _state(
        snapshot,
        log_text="\n".join(f"log line {index}" for index in range(60)),
        log_job=1,
    )
    output = _render(tui, log_state)
    assert tui.handle_mouse(_mouse_at(output, "LOG job 1", button=64)) is True
    assert tui.log_offset == 3
    assert "3 line(s) above tail" in _render(tui, log_state)


def test_action_bar_buttons_post_actions_and_keep_cancel_confirmation():
    tui = _tui()
    output = _render(tui)

    assert tui.handle_mouse(_mouse_at(output, "[t tick]")) is True
    assert tui.worker.actions.get_nowait() == ("tick",)

    assert tui.handle_mouse(_mouse_at(output, "[c cancel]")) is True
    assert tui.worker.actions.empty()
    assert tui.pending_confirm is not None

    output = _render(tui)
    assert tui.handle_mouse(_mouse_at(output, "[c confirm cancel]")) is True
    assert tui.worker.actions.get_nowait() == ("cancel", 1)


def test_status_counts_and_alert_rows_are_clickable():
    tui = _tui()
    output = _render(tui)

    assert tui.handle_mouse(_mouse_at(output, "pending 1")) is True
    assert tui.filter == "pending"

    failed = _job(7, state="failed", exit_code=1, finished_at="now")
    snapshot = _snapshot(jobs=[], recent=[failed], counts={"failed": 1})
    snapshot["alerts"] = [
        {
            "id": 3,
            "kind": "job_failed",
            "severity": "error",
            "job_id": 7,
            "node": "small-node",
            "title": "job 7 failed",
            "acknowledged_at": None,
        }
    ]
    output = _render(tui, _state(snapshot))

    assert tui.handle_mouse(_mouse_at(output, "job 7 failed")) is True
    assert tui.filter == "all"
    assert tui.selected_job()["id"] == 7
    assert tui.show_log is True
    assert tui.worker.log_request == (7, "stdout")


def test_help_and_quit_buttons_are_clickable():
    tui = _tui()
    output = _render(tui)

    assert tui.handle_mouse(_mouse_at(output, "[? help]")) is True
    assert tui.show_help is True
    help_output = _render(tui)
    assert tui.handle_mouse(_mouse_at(help_output, "Mouse click")) is True
    assert tui.show_help is False

    output = _render(tui)
    assert tui.handle_mouse(_mouse_at(output, "[q quit]")) is False


def test_mouse_events_are_ignored_when_mouse_support_is_disabled():
    tui = QueueTUI(mouse_enabled=False)
    output = _render(tui)

    assert tui.handle_mouse(_mouse_at(output, "[q quit]")) is True


def test_worker_shutdown_can_join_without_shadowing_thread_stop():
    worker = _Worker()
    worker.run = lambda: None

    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive()


# --- actions ---------------------------------------------------------------

def test_cancelling_needs_a_second_keypress():
    tui = _tui()
    _render(tui)
    _press(tui, "c")
    assert tui.worker.actions.empty()
    assert tui.pending_confirm is not None
    assert "press c again" in _render(tui)

    _press(tui, "c")
    assert tui.worker.actions.get_nowait() == ("cancel", 1)


def test_escape_abandons_a_pending_confirmation():
    tui = _tui()
    _render(tui)
    _press(tui, "c")
    _press(tui, "", name="KEY_ESCAPE")
    assert tui.pending_confirm is None
    _press(tui, "c")
    assert tui.worker.actions.empty()  # the counter restarted


def test_requeue_and_tick_are_posted_to_the_worker():
    tui = _tui()
    _render(tui)
    _press(tui, "r")
    assert tui.worker.actions.get_nowait() == ("requeue", 1)
    _press(tui, "t")
    assert tui.worker.actions.get_nowait() == ("tick",)


def test_d_drains_the_selected_node_and_resumes_a_drained_one():
    tui = _tui()
    snapshot = _snapshot()
    _render(tui, _state(snapshot))
    _press(tui, "", name="KEY_TAB")
    _press(tui, "d")
    assert tui.worker.actions.get_nowait() == ("drain", "big-node")

    snapshot["nodes"][0]["enabled"] = False
    _render(tui, _state(snapshot))
    _press(tui, "d")
    assert tui.worker.actions.get_nowait() == ("resume", "big-node")


def test_the_log_view_asks_the_worker_for_the_selected_job():
    tui = _tui()
    _render(tui)
    _press(tui, "L")
    assert tui.worker.log_request == (1, "stdout")
    output = _render(tui, _state(log_text="step 1\nstep 2\n", log_job=1))
    assert "LOG job 1" in output
    assert "step 2" in output

    _press(tui, "L")
    assert tui.worker.log_request is None


# --- sorting, priority and reordering --------------------------------------

def test_the_default_sort_is_dispatch_order():
    tui = _tui()
    snapshot = _snapshot(
        jobs=[
            _job(1, state="pending", name="low", priority=0),
            _job(2, state="pending", name="high", priority=5),
            _job(3, state="running", name="busy"),
        ]
    )
    _render(tui, _state(snapshot))
    # Running first, then pending exactly as the scheduler would dispatch.
    assert [job["id"] for job in tui.jobs] == [3, 2, 1]


def test_the_job_table_shows_priority_and_compact_states():
    tui = _tui()
    snapshot = _snapshot(
        jobs=[_job(1), _job(2, state="pending", name="eval", priority=3)]
    )
    output = _render(tui, _state(snapshot))
    assert "PRI" in output
    assert "● run" in output and "· pend" in output


def test_s_cycles_the_sort_and_S_flips_it():
    tui = _tui()
    _render(tui)
    assert tui.sort_key == "queue" and tui.sort_reverse is False
    _press(tui, "s")
    assert tui.sort_key == "id"
    _press(tui, "S")
    assert tui.sort_reverse is True
    _render(tui)
    assert [job["id"] for job in tui.jobs] == [2, 1]


def test_clicking_a_column_header_sorts_then_flips():
    tui = _tui()
    output = _render(tui)
    assert tui.handle_mouse(_mouse_at(output, "NAME")) is True
    assert tui.sort_key == "name" and tui.sort_reverse is False

    output = _render(tui)
    assert "NAME▼" in output
    assert tui.handle_mouse(_mouse_at(output, "NAME▼")) is True
    assert tui.sort_reverse is True


def test_clicking_the_sort_label_cycles_the_sort():
    tui = _tui()
    output = _render(tui)
    assert tui.handle_mouse(_mouse_at(output, "sort queue▼")) is True
    assert tui.sort_key == "id"


def test_the_selection_follows_a_job_when_the_order_changes():
    tui = _tui()
    snapshot = _snapshot(
        jobs=[
            _job(1, state="pending", name="a", priority=0),
            _job(2, state="pending", name="b", priority=0),
        ]
    )
    _render(tui, _state(snapshot))
    _press(tui, "j")
    assert tui.selected_job()["id"] == 2

    # Job 2 gains priority and jumps to the head; the cursor rides along.
    snapshot["jobs"][1]["priority"] = 5
    _render(tui, _state(snapshot))
    assert tui.selected_job()["id"] == 2
    assert tui.job_index == 0


def test_priority_keys_post_adjustments_for_the_selected_job():
    tui = _tui()
    _render(tui)
    _press(tui, "+")
    assert tui.worker.actions.get_nowait() == ("priority", 1, 1)
    _press(tui, "-")
    assert tui.worker.actions.get_nowait() == ("priority", 1, -1)


def test_priority_keys_refuse_finished_jobs():
    tui = _tui()
    tui.filter = "finished"
    snapshot = _snapshot(
        jobs=[],
        recent=[_job(9, state="failed", exit_code=1)],
        counts={"failed": 1},
    )
    _render(tui, _state(snapshot))
    _press(tui, "+")
    assert tui.worker.actions.empty()
    assert "no priority" in tui.worker.notice[0]


def test_move_keys_reorder_only_pending_jobs():
    tui = _tui()
    _render(tui)  # job 1 (running) is selected
    _press(tui, "K")
    assert tui.worker.actions.empty()
    assert "only pending" in tui.worker.notice[0]

    _press(tui, "j")  # job 2 is pending
    _press(tui, "K")
    assert tui.worker.actions.get_nowait() == ("move", 2, -1)
    _press(tui, "J")
    assert tui.worker.actions.get_nowait() == ("move", 2, 1)


def test_moving_a_job_snaps_the_table_back_to_queue_order():
    tui = _tui()
    _render(tui)
    _press(tui, "s")  # sort by id instead
    _press(tui, "j")  # select the pending job
    _press(tui, "K")
    assert tui.sort_key == "queue" and tui.sort_reverse is False
    assert tui.worker.actions.get_nowait() == ("move", 2, -1)


def test_priority_and_move_buttons_are_clickable():
    tui = _tui()
    _render(tui)
    _press(tui, "j")  # the pending job has both button groups
    output = _render(tui)
    assert tui.handle_mouse(_mouse_at(output, "[+ pri:0]")) is True
    assert tui.worker.actions.get_nowait() == ("priority", 2, 1)
    assert tui.handle_mouse(_mouse_at(output, "[K ▲queue]")) is True
    assert tui.worker.actions.get_nowait() == ("move", 2, -1)


def test_the_row_marker_tracks_the_detail_state():
    tui = _tui()
    output = _render(tui)
    row = next(
        line for line in output.splitlines() if "train" in line and "python" in line
    )
    assert row.startswith("▼")

    _press(tui, "", name="KEY_ENTER")
    output = _render(tui)
    row = next(
        line for line in output.splitlines() if "train" in line and "python" in line
    )
    assert row.startswith("▶")
