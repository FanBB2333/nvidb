"""Tests for the queue TUI's rendering and key handling.

The worker thread is never started here: the interface is driven from a plain
snapshot dictionary, which is exactly the contract it renders from.
"""
import os
import re
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from blessed import Terminal
from blessed.keyboard import Keystroke

from nvidb.mouse import MouseEvent
from nvidb.tui_theme import display_width
from nvidb.sched.tui import COMMANDS, CONFIRM_SECONDS, QueueTUI, _Worker

ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_MISSING = object()


def _plain(text):
    return ANSI.sub("", text)


def _content_line(line):
    """Remove the optional panel sides without changing content indentation."""
    if line.startswith("│ ") and line.endswith(" │"):
        return line[2:-2]
    return line


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
        "depends_any": [],
        "max_runtime_s": None,
        "submitter": "client-a",
        "tags": [],
        "created_at": "2026-07-27T10:00:00+00:00",
        "started_at": "2026-07-27T10:00:01+00:00",
        "finished_at": None,
        "elapsed_s": 125.0,
        "node": node,
        "gpu_ids": [0],
        "lane": None,
        "lane_seq": None,
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
        "held_reason": None,
        "held": False,
    }
    job.update(overrides)
    return job


def _snapshot(jobs=None, recent=None, nodes=None, lanes=None, counts=_MISSING):
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
        "lanes": lanes or [],
    }


def _lane(
    node,
    gpu_ids,
    *,
    running=None,
    queued=None,
    paused=False,
    blocked=None,
    runner_up=True,
):
    suffix = ",".join(str(index) for index in gpu_ids)
    return {
        "name": f"{node}:{suffix}",
        "node": node,
        "gpu_ids": list(gpu_ids),
        "paused": paused,
        "concurrency": 1,
        "runner": True,
        "runner_up": runner_up,
        "runner_state": {},
        "runner_seen_at": None,
        "node_state": "up",
        "running": list(running or []),
        "queued": list(queued or []),
        "blocked": blocked,
    }


def _flow_snapshot():
    running = _job(
        10,
        name="train",
        node="small-node",
        gpu_ids=[0],
        lane="small-node:0",
        lane_seq=1,
        elapsed_s=3600,
    )
    evaluate = _job(
        11,
        state="pending",
        name="evaluate",
        node=None,
        gpu_ids=[],
        lane="small-node:0",
        lane_seq=2,
        priority=5,
        depends_on=[10],
        started_at=None,
        elapsed_s=None,
    )
    report = _job(
        12,
        state="pending",
        name="report",
        node=None,
        gpu_ids=[],
        lane="small-node:0",
        lane_seq=3,
        depends_any=[11],
        started_at=None,
        elapsed_s=None,
    )
    held = _job(
        13,
        state="pending",
        name="archive",
        node=None,
        gpu_ids=[],
        lane="small-node:0",
        lane_seq=4,
        priority=-1,
        depends_on=[12],
        held=True,
        held_reason="job 12 failed",
        started_at=None,
        elapsed_s=None,
    )
    second_gpu = _job(
        14,
        state="pending",
        name="sweep-b",
        node=None,
        gpu_ids=[],
        lane="small-node:1",
        lane_seq=1,
        priority=2,
        started_at=None,
        elapsed_s=None,
    )
    shared = _job(
        20,
        state="pending",
        name="urgent-free",
        node=None,
        gpu_ids=[],
        priority=8,
        started_at=None,
        elapsed_s=None,
    )
    snapshot = _snapshot(
        jobs=[running, evaluate, report, held, second_gpu, shared],
        lanes=[
            _lane(
                "small-node",
                [0],
                running=[running],
                queued=[evaluate, report, held],
                blocked="job 13 is held: job 12 failed",
            ),
            _lane(
                "small-node",
                [1],
                queued=[second_gpu],
                paused=True,
                blocked="paused",
            ),
        ],
        counts={"running": 1, "pending": 5},
    )
    snapshot["nodes"][1]["gpus"] = [_gpu(0), _gpu(1)]
    snapshot["nodes"][1]["gpus"][1]["util_percent"] = 0
    return snapshot


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


def _tui(width=150, height=None, *, resource_view=None):
    os.environ["COLUMNS"] = str(width)
    if height is not None:
        os.environ["LINES"] = str(height)
    tui = QueueTUI()
    if resource_view is not None:
        tui.resource_view = resource_view
    return tui


def _render(tui, state=None):
    return _plain(tui.render(state or _state()))


def _mouse_at(output, text, *, button=0, occurrence=0):
    from nvidb.sched.model import display_width

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
    column = display_width(output.splitlines()[row][:column])
    return MouseEvent(
        button=button,
        column=column + 1,
        row=row + 1,
        pressed=True,
    )


def _click_target(tui, target):
    row, start, _end, _target = next(region for region in tui._click_regions if region[3] == target)
    return tui.handle_mouse(MouseEvent(0, start + 1, row + 1, True))


def _choose_menu_target(tui, state, target):
    menu = tui._menu
    index = next(index for index, (_label, value) in enumerate(menu.entries) if value == target)
    menu.selected = index
    _render(tui, state)
    row, start, _end, _target = next(region for region in menu.regions if region[3] == ("choose", index))
    return tui.handle_mouse(MouseEvent(0, start + 1, row + 1, True))


# --- rendering -------------------------------------------------------------

def test_overview_is_default_and_lists_each_job_once_in_queue_order():
    snapshot = _flow_snapshot()
    snapshot["jobs"][3]["priority"] = 99  # Lane order still takes precedence.
    tui = _tui()
    output = _render(tui, _state(snapshot))

    assert tui.resource_view == "overview"
    assert tui.sort_key == "queue" and not tui.sort_reverse
    assert [job["id"] for job in tui.jobs] == [10, 20, 11, 12, 13, 14]
    assert "JOB" in output and "STATUS" in output and "WHERE" in output
    assert "PROGRESS / WAIT" in output
    assert "TASK FLOW" not in output and "DEPS" not in output
    assert "python train.py" not in output
    assert "DETAIL HIDDEN" not in output
    for job in snapshot["jobs"]:
        assert output.count(f"#{job['id']} {job['name']}") == 1


@pytest.mark.parametrize("width", [24, 36, 40, 60, 80, 150])
def test_overview_keeps_keyboard_navigation_and_more_discoverable(width):
    tui = _tui(width=width, height=24)
    output = _render(tui)
    lines = output.splitlines()
    hint = _content_line(lines[-2] if lines[-1].startswith("╰") else lines[-1])
    assert "Enter" in hint and "m" in hint and "?" in hint and "q" in hint
    assert "j/k" in hint or "jk" in hint
    assert display_width(hint) <= width
    assert any(target == ("menu", "actions") for _row, _start, _end, target in tui._click_regions)
    assert _press(tui, "m")
    assert tui._menu is not None
    assert any(target == ("quit", None) for _label, target in tui._menu.entries)


def test_overview_unknown_wait_does_not_guess_a_resource_or_dependency_reason():
    pending = _job(8, state="pending", name="unassigned", node=None,
                   gpu_ids=[], started_at=None, elapsed_s=None)
    snapshot = _snapshot(jobs=[pending], counts={"pending": 1})
    tui = _tui()
    output = _render(tui, _state(snapshot))
    row = next(line for line in output.splitlines() if "#8 unassigned" in line)

    assert "Waiting" in row and "Any GPU" in row
    assert "Waiting for scheduling" in row
    assert "Lane position" not in row
    assert "After #" not in row and "memory" not in row.lower()


def test_overview_wait_reasons_preserve_holds_pauses_dependencies_and_lane_order():
    snapshot = _flow_snapshot()
    snapshot["jobs"][3]["lane_seq"] = 60  # Position is 3, not the stored sequence.
    output = _render(_tui(), _state(snapshot))
    cases = [
        (11, "evaluate", "Waiting", "After #10 succeeds", 1),
        (12, "report", "Waiting", "After #11 finishes", 2),
        (13, "archive", "Held", "job 12 failed", 3),
        (14, "sweep-b", "Waiting", "Queue paused", 1),
    ]
    for job_id, name, status, reason, position in cases:
        row = next(line for line in output.splitlines() if f"#{job_id} {name}" in line)
        assert status in row and reason in row
        assert f"Lane position {position}" in row


@pytest.mark.parametrize("field,upstream_state", [("depends_on", "completed"), ("depends_any", "failed")])
def test_overview_does_not_show_satisfied_dependencies_as_waiting(field, upstream_state):
    job = _job(8, state="pending", name="ready", node=None, gpu_ids=[],
               started_at=None, elapsed_s=None, **{field: [99]})
    snapshot = _snapshot(jobs=[job], counts={"pending": 1})
    snapshot["dependencies"] = [{"id": 99, "state": upstream_state, "name": "upstream"}]
    output = _render(_tui(), _state(snapshot))
    row = next(line for line in output.splitlines() if "#8 ready" in line)
    assert "Waiting for scheduling" in row
    assert "After #99" not in row


@pytest.mark.parametrize("field,upstream_state,reason", [
    ("depends_on", None, "Dependency #99 status unavailable"),
    ("depends_any", None, "Dependency #99 status unavailable"),
    ("depends_on", "failed", "Dependency #99 failed"),
    ("depends_on", "cancelled", "Dependency #99 cancelled"),
    ("depends_on", "timeout", "Dependency #99 timeout"),
    ("depends_on", "lost", "Dependency #99 lost"),
])
def test_overview_reports_unknown_or_failed_dependencies_without_predicting_completion(
    field, upstream_state, reason,
):
    job = _job(8, state="pending", name="blocked", node=None, gpu_ids=[],
               held=False, held_reason=None, started_at=None, elapsed_s=None,
               **{field: [99]})
    snapshot = _snapshot(jobs=[job], counts={"pending": 1})
    if upstream_state is not None:
        snapshot["dependencies"] = [{"id": 99, "state": upstream_state, "name": "upstream"}]
    output = _render(_tui(), _state(snapshot))
    row = next(line for line in output.splitlines() if "#8 blocked" in line)

    assert reason in row
    assert "After #99" not in row
    assert "Waiting for scheduling" not in row
    assert not job["held"]  # Snapshot state is not silently changed by rendering.


@pytest.mark.parametrize("width,height", [(36, 15), (60, 20), (80, 24), (150, 42)])
def test_overview_keeps_jobs_and_more_visible_in_small_and_large_terminals(width, height):
    from nvidb.sched.model import display_width

    tui = _tui(width=width, height=height)
    output = _render(tui)
    lines = output.splitlines()

    assert "#1 train" in output
    assert "Running" in output
    assert len(lines) <= height - 1
    assert all(display_width(line) <= width for line in lines)
    more = [region for region in tui._click_regions if region[3] == ("menu", "actions")]
    assert more
    assert all(0 <= row < len(lines) and 0 <= start <= end < width for row, start, end, _ in more)
    if width < 72:
        assert sum(target == ("job", 0) for target in tui._row_targets.values()) == 2


@pytest.mark.parametrize("state,label", [
    ("running", "Running"), ("pending", "Waiting"), ("completed", "Done"),
    ("failed", "Failed"), ("cancelled", "Cancelled"), ("timeout", "Timed out"),
    ("lost", "Lost"),
])
def test_narrow_overview_spells_out_job_states(state, label):
    tui = _tui(width=60, height=20)
    tui.filter = "all"
    snapshot = _snapshot(jobs=[_job(7, state=state, name="状态测试")], counts={state: 1})
    output = _render(tui, _state(snapshot))

    assert label in output
    assert "#7 状态测试" in output
    assert sum(target == ("job", 0) for target in tui._row_targets.values()) == 2


def test_narrow_overview_keeps_known_wait_reason_on_the_second_line():
    job = _job(7, state="pending", name="等待实验", held=True,
               held_reason="Manual review", node=None, gpu_ids=[])
    tui = _tui(width=60, height=20)
    output = _render(tui, _state(_snapshot(jobs=[job], counts={"pending": 1})))
    rows = [row for row, target in tui._row_targets.items() if target == ("job", 0)]

    assert len(rows) == 2
    assert "Held" in output.splitlines()[rows[0]]
    assert "Manual review" in output.splitlines()[rows[1]]


def test_overview_tab_opens_visible_server_controls_before_moving_node_focus():
    tui = _tui()
    _render(tui)
    _press(tui, "", name="KEY_TAB")
    assert tui.resource_view == "servers"
    assert tui.focus == "nodes"
    assert "SERVERS (2)" in _render(tui)
    _press(tui, "j")
    assert tui.selected_node()["name"] == "small-node"


def test_overview_chinese_node_click_scopes_running_jobs_and_keeps_job_actions_ready():
    from nvidb.sched.model import display_width

    node_name = "训练节点一"
    snapshot = _snapshot(jobs=[
        _job(1, name="当前训练", node=node_name),
        _job(2, name="等待训练", state="pending", node=node_name),
        _job(3, name="其它训练", node="small-node"),
    ], counts={"running": 2, "pending": 1})
    snapshot["nodes"][0]["name"] = node_name
    state = _state(snapshot)
    tui = _tui()
    output = _render(tui, state)
    row = next(row for row, target in tui._row_targets.items() if target == ("node", 0))
    line = output.splitlines()[row]
    offset = line.index(node_name) + len(node_name) - 1
    column = display_width(line[:offset]) + 1

    assert tui.handle_mouse(MouseEvent(0, column, row + 1, True))
    assert tui.resource_view == "overview"
    assert tui.focus == "jobs"
    assert tui.job_scope_node == node_name
    assert [job["id"] for job in tui.jobs] == [1]
    _render(tui, state)
    assert _click_target(tui, ("scope_all", None))
    assert tui.job_scope_node is None
    assert [job["id"] for job in tui.jobs] == [1, 3, 2]

    _render(tui, state)
    row = next(row for row, target in tui._row_targets.items() if target == ("node", 0))
    tui.handle_mouse(MouseEvent(0, column, row + 1, True))
    _render(tui, state)
    assert _click_target(tui, ("filter", "pending"))
    assert tui.job_scope_node is None
    assert [job["id"] for job in tui.jobs] == [2]


@pytest.mark.parametrize("width", [36, 80, 150])
def test_overview_more_keeps_every_secondary_action_reachable(width):
    tui = _tui(width=width, height=24)
    state = _state()
    _render(tui, state)
    _press(tui, "j")  # Pending jobs expose both priority and ordering controls.
    _render(tui, state)
    visible = {target for _row, _start, _end, target in tui._click_regions}
    assert _click_target(tui, ("menu", "actions"))
    menu = tui._menu
    hidden = [target for _label, target in menu.entries]
    expected = {
        ("detail", None), ("log", None), ("cancel", None),
        ("priority", 1), ("priority", -1), ("move", -1), ("move", 1),
        ("menu", "filter"), ("menu", "sort"), ("menu", "view"),
        ("procs", None), ("tick", None), ("auto", False), ("theme", None),
        ("switch_pane", None), ("help", None), ("quit", None),
    }
    assert expected <= visible | set(hidden)
    assert len(hidden) == len(set(hidden))
    for index in range(len(menu.entries)):
        menu.selected = index
        _render(tui, state)
        assert any(region[3] == ("choose", index) for region in menu.regions)
    assert tui.worker.actions.empty()


@pytest.mark.parametrize("width", [36, 60, 150])
def test_overview_cancel_confirmation_stays_visible_and_requires_a_second_click(width):
    tui = _tui(width=width, height=15)
    state = _state()
    _render(tui, state)
    if not any(region[3] == ("cancel", None) for region in tui._click_regions):
        _click_target(tui, ("menu", "actions"))
        _choose_menu_target(tui, state, ("cancel", None))
    else:
        _click_target(tui, ("cancel", None))
    assert tui.pending_confirm is not None
    assert tui.worker.actions.empty()

    output = _render(tui, state)
    assert "Cancel #1?" in output
    assert "[Confirm]" in output
    assert _click_target(tui, ("cancel", None))
    assert tui.worker.actions.get_nowait() == ("cancel", 1)
    assert tui.pending_confirm is None


def test_overview_view_menu_selects_each_view_without_cycling():
    tui = _tui()
    state = _state()
    for view in ("servers", "flow", "overview"):
        _render(tui, state)
        assert _click_target(tui, ("menu", "view"))
        assert _choose_menu_target(tui, state, ("resource_view", view))
        assert tui.resource_view == view
        assert tui._menu is None


def test_overview_resource_summary_stays_short_without_hiding_the_job_list():
    base = _snapshot()["nodes"][1]
    nodes = [dict(base, name=f"server-{index}") for index in range(10)]
    jobs = [_job(index, name=f"experiment-{index}", node="server-0") for index in range(1, 13)]
    state = _state(_snapshot(jobs=jobs, nodes=nodes, counts={"running": len(jobs)}))
    tui = _tui()
    output = _render(tui, state)
    resource_rows = [row for row, target in tui._row_targets.items()
                     if target[0] == "node" or target == ("pane", "nodes")]

    assert 1 <= len(resource_rows) <= 4
    assert "more line(s)" in output
    for job in jobs:
        assert f"#{job['id']} {job['name']}" in output
    assert not any(target[0] == "gpu" for _row, _start, _end, target in tui._click_regions)

    _press(tui, "v")
    _render(tui, state)
    assert any(target[0] == "gpu" for _row, _start, _end, target in tui._click_regions)
    _click_target(tui, ("menu", "view"))
    _choose_menu_target(tui, state, ("resource_view", "overview"))
    _render(tui, state)
    assert not any(target[0] == "gpu" for _row, _start, _end, target in tui._click_regions)


def test_overview_details_retain_command_priority_memory_and_exit_information():
    job = _job(7, state="failed", name="failed-run", priority=3,
               gpu_mem_mb=8192, exit_code=2)
    state = _state(_snapshot(jobs=[], recent=[job], nodes=[], counts={"failed": 1}))
    tui = _tui(width=80, height=20)
    tui.filter = "finished"
    _render(tui, state)
    _press(tui, "", name="KEY_ENTER")
    pages = [_render(tui, state)]
    for _ in range(tui.detail_pages - 1):
        _press(tui, "]")
        pages.append(_render(tui, state))
    detail = "\n".join(pages)

    assert "python train.py --epochs 10" in detail
    assert "priority 3" in detail and "runtime" in detail
    assert "GPU memory used 8.0G" in detail
    assert "exit code 2" in detail
    assert "vram=20.0G" in detail


def test_task_flow_remains_available_and_shows_each_gpu_lane():
    snapshot = _flow_snapshot()
    tui = _tui(resource_view="flow")
    output = _render(tui, _state(snapshot))
    lines = [_content_line(line) for line in output.splitlines()]

    assert tui.resource_view == "flow"
    assert "TASK FLOW · 2N 3G · R1 Q5" in output
    gpu0 = next(line for line in lines if line.startswith("    G0 ") and "RUN #10" in line)
    assert "[ RUN #10 train 01:00:00 ]" in gpu0
    assert "{ lane Q3 }" in gpu0
    assert "[1:#11 P+5 evaluate ←✓#10]" in gpu0
    assert "[2:#12 P0 report ←◇#11]" in gpu0
    assert "+1" in gpu0

    gpu1 = next(line for line in lines if line.startswith("    G1 "))
    assert "[ IDLE ]" in gpu1
    assert "{ lane Q1 }" in gpu1
    assert "[1:#14 P+2 sweep-b]" in gpu1
    assert "! PAUSED" in gpu1


def test_task_flow_keeps_unassigned_jobs_in_the_shared_priority_pool():
    output = _render(_tui(resource_view="flow"), _state(_flow_snapshot()))
    lines = [_content_line(line) for line in output.splitlines()]
    shared = next(line for line in lines if line.startswith("  ANY GPU"))

    assert "{ Q1 · priority }" in shared
    assert "[1:#20 P+8 urgent-free]" in shared
    assert all("#20" not in line for line in lines if line.startswith("    G"))


@pytest.mark.parametrize("gpu_state,node_state,expected", [
    ({"external_mem_mb": 24000, "external_procs": 1, "util_percent": 99}, "up", "EXTERNAL"),
    ({"util_percent": 75}, "up", "BUSY"),
    ({"util_percent": 0}, "up", "IDLE"),
    ({"util_percent": None}, "up", "UNKNOWN"),
    ({"util_percent": 0}, "down", "UNKNOWN"),
])
def test_flow_activity_distinguishes_external_idle_and_unknown(gpu_state, node_state, expected):
    snapshot = _snapshot(jobs=[], nodes=[_snapshot()["nodes"][1]])
    node = snapshot["nodes"][0]
    node["state"] = node_state
    node["gpus"][0].update(gpu_state)
    output = _render(_tui(resource_view="flow"), _state(snapshot))
    assert f"[ {expected} ]" in output


@pytest.mark.parametrize("target", ["{ lane Q3 }", "+3"])
def test_task_pool_and_overflow_open_complete_lane_queue(target):
    snapshot = _flow_snapshot()
    tui = _tui(width=100, height=42, resource_view="flow")
    state = _state(snapshot)
    output = _render(tui, state)
    tui.handle_mouse(_mouse_at(output, target))
    assert [job["id"] for job in tui.jobs] == [11, 12, 13]
    assert "queued in lane small-node:0" in _render(tui, state)
    _press(tui, "x")
    assert tui.job_scope_pool is None
    assert len(tui.jobs) == 6


def test_shared_pool_excludes_running_and_cpu_jobs_and_can_open_job_detail():
    snapshot = _flow_snapshot()
    snapshot["jobs"].extend([
        _job(21, state="pending", gpus=0), _job(22, state="running", lane=None),
    ])
    tui = _tui(resource_view="flow")
    state = _state(snapshot)
    output = _render(tui, state)
    tui.handle_mouse(_mouse_at(output, "{ Q1 · priority }"))
    assert [job["id"] for job in tui.jobs] == [20]
    _press(tui, "", name="KEY_ENTER")
    assert "JOB 20 DETAIL" in _render(tui, state)
    _press(tui, "f")
    assert tui.job_scope_pool is None


def test_job_details_page_through_full_commands_and_dependency_states():
    snapshot = _flow_snapshot()
    job = snapshot["jobs"][1]
    job["command"] = "python train.py " + "--checkpoint /models/team/checkpoint " * 12 + "--LAST-ARG"
    job["workdir"] = "/workspace/" + "nested/" * 20 + "END-PATH"
    job["depends_any"] = [999]
    snapshot["dependencies"] = [{"id": 999, "name": "old", "state": "failed"}]
    tui = _tui(width=60, height=20)
    state = _state(snapshot)
    _render(tui, state)
    tui._select_job_id(11)
    tui.show_detail = True
    pages = [_render(tui, state)]
    for _ in range(tui.detail_pages - 1):
        _press(tui, "]")
        pages.append(_render(tui, state))
    output = "\n".join(pages)
    assert "--LAST-ARG" in output and "END-PATH" in output
    assert "queued position 1/3" in output
    assert "#10 · running · requires success" in output
    assert "#999 · failed · requires any terminal result" in output


def test_task_flow_draws_success_and_any_result_dependencies():
    output = _render(_tui(resource_view="flow"), _state(_flow_snapshot()))
    dependency = next(line for line in output.splitlines() if "DEPS" in line)

    assert "#10 ─✓▶ #11" in dependency
    assert "#11 ─◇▶ #12" in dependency
    assert "#12 ─✓▶ #13" in dependency


def test_task_flow_job_tokens_open_the_corresponding_job_detail():
    snapshot = _flow_snapshot()
    state = _state(snapshot)
    tui = _tui(resource_view="flow")
    output = _render(tui, state)
    click = _mouse_at(output, "[1:#14 P+2 sweep-b]")

    assert tui.handle_mouse(click) is True
    assert tui.focus == "jobs"
    assert tui.selected_job()["id"] == 14
    assert "JOB 14 DETAIL" in _render(tui, state)

    # Repeating the click follows job-row behaviour and collapses its detail.
    assert tui.handle_mouse(click) is True
    assert tui.show_detail is False


def test_clicking_a_flow_gpu_scopes_the_job_list_to_that_card():
    snapshot = _flow_snapshot()
    state = _state(snapshot)
    tui = _tui(resource_view="flow")
    output = _render(tui, state)

    assert tui.handle_mouse(_mouse_at(output, "G0 ", occurrence=1)) is True
    assert tui.job_scope_node == "small-node"
    assert tui.job_scope_gpu == 0
    assert [job["id"] for job in tui.jobs] == [10]
    assert "JOBS (1) · running on small-node/GPU0" in _render(tui, state)


def test_v_cycles_overview_task_flow_and_server_capacity_views():
    tui = _tui()
    state = _state(_flow_snapshot())
    _render(tui, state)
    assert tui.resource_view == "overview"

    _press(tui, "v")
    assert tui.resource_view == "flow"
    assert "TASK FLOW" in _render(tui, state)

    _press(tui, "v")
    assert tui.resource_view == "servers"
    assert "SERVERS (2)" in _render(tui, state)

    _render(tui, state)
    assert _click_target(tui, ("menu", "view"))
    assert _choose_menu_target(tui, state, ("resource_view", "overview"))
    assert tui.resource_view == "overview"


def test_task_flow_remains_readable_on_a_narrow_terminal():
    from nvidb.sched.model import display_width

    tui = _tui(width=60, resource_view="flow")
    output = _render(tui, _state(_flow_snapshot()))

    assert "TASK FLOW" in output
    assert "✓=success ◇=any-result" in output
    assert "{ lane Q3 }" in output
    assert "{ lane Q1 }" in output
    assert "+" in output  # queue/dependency overflow is counted, not wrapped
    for line in output.splitlines():
        assert display_width(line) <= 60, line


@pytest.mark.parametrize("width,height", [(40, 15), (60, 20), (80, 24)])
def test_frame_respects_the_physical_terminal_size(width, height):
    from nvidb.sched.model import display_width

    output = _render(_tui(width=width, height=height))
    lines = output.splitlines()

    assert len(lines) <= height - 1
    assert all(display_width(line) <= width for line in lines)


def test_compact_layout_keeps_jobs_visible_and_limits_the_footer():
    tui = _tui(width=60, height=20)
    state = _state()
    output = _render(tui, state)

    assert "JOBS (2)" in output
    assert "train" in output
    assert len(tui._footer_lines(state, 60, max_rows=2)) <= 2


def test_resource_viewport_follows_the_selected_node():
    nodes = []
    for index in range(10):
        nodes.append(
            {
                "name": f"node-{index}",
                "hostname": f"10.0.0.{index}",
                "port": 22,
                "username": "u",
                "state": "up",
                "enabled": True,
                "last_seen": "2026-07-27T10:01:58+00:00",
                "last_error": None,
                "gpu_count": 1,
                "probed_at": None,
                "gpus": [_gpu()],
            }
        )
    snapshot = _snapshot(nodes=nodes)
    tui = _tui(width=60, height=20, resource_view="servers")
    tui.focus = "nodes"
    tui.node_index = len(nodes) - 1

    output = _render(tui, _state(snapshot))

    assert "node-9" in output
    assert "above" in output


def test_all_gpu_rows_and_shared_pool_are_reachable_on_one_large_node():
    snapshot = _flow_snapshot()
    node = snapshot["nodes"][1]
    node["gpus"] = [_gpu(index) for index in range(8)]
    snapshot["nodes"] = [node]
    tui = _tui(width=100, height=30, resource_view="flow")
    tui.focus = "nodes"
    state = _state(snapshot)
    output = _render(tui, state)
    assert "G7 " not in output
    _press(tui, "", name="KEY_PGDOWN")
    output = _render(tui, state)
    assert "G7 " in output
    assert "ANY GPU" in output
    _press(tui, "", name="KEY_PGDOWN")
    output = _render(tui, state)
    assert "DEPS" in output
    tui.handle_mouse(_mouse_at(output, "G7 ", button=64))
    output = _render(tui, state)
    assert "G7 " in output
    tui.handle_mouse(_mouse_at(output, "G7 "))
    assert tui.job_scope_gpu == 7
    _press(tui, "", name="KEY_LEFT")
    output = _render(tui, state)
    assert "❯ G6 " in output
    _press(tui, "", name="KEY_ENTER")
    assert tui.job_scope_gpu == 6


def test_keyboard_gpu_cursor_reaches_the_last_card_and_survives_removal():
    snapshot = _snapshot(nodes=[_snapshot()["nodes"][1]])
    snapshot["nodes"][0]["gpus"] = [_gpu(index) for index in (0, 2, 7)]
    tui = _tui(width=80, height=24, resource_view="flow")
    tui.focus = "nodes"
    _render(tui, _state(snapshot))
    for _ in range(3):
        _press(tui, "", name="KEY_RIGHT")
    assert "❯ G7 " in _render(tui, _state(snapshot))
    _press(tui, "", name="KEY_ENTER")
    assert tui.job_scope_gpu == 7
    snapshot["nodes"][0]["gpus"] = [_gpu(0)]
    _render(tui, _state(snapshot))
    assert tui._cursor_gpu is None
    assert tui.job_scope_gpu is None


def test_the_screen_shows_nodes_capacity_and_jobs():
    output = _render(_tui(resource_view="servers"))
    assert "SERVERS (2)" in output
    assert "▾ [1]" in output and "▾ [2]" in output
    assert "big-node" in output and "small-node" in output
    assert output.index("big-node") < output.index("small-node")
    # The GPU model lives on the node header now, once per node.
    assert "1× RTX PRO 5000" in output
    # The grid cell shows used/total VRAM and the running job's id.
    assert "67.4G/71.7G" in output
    assert "#1" in output
    assert "python train.py" in output
    # The keybinding footer is always reachable.
    assert "q quit" in output


def test_gpus_share_lines_instead_of_taking_one_each():
    snapshot = _snapshot()
    snapshot["nodes"][0]["gpus"] = [
        _gpu(index, name="RTX PRO 5000", total=73415) for index in range(4)
    ]
    tui = _tui(width=150, resource_view="servers")
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
    output = _plain(_tui(resource_view="servers").render(_state(snapshot)))
    ext_lines = [
        line for line in output.splitlines()
        if _content_line(line).lstrip().startswith("ext ")
    ]
    assert len(ext_lines) == 1
    # Biggest foreign process first, overflow counted instead of listed.
    assert "alice" in ext_lines[0] and "39.1G" in ext_lines[0]
    assert "+1 more" in ext_lines[0]


def test_a_node_compresses_to_a_single_line_when_the_width_allows():
    """Name, model, GPU occupancy, hostname and state all share one line, so
    a fullscreen terminal is not mostly blank to the right of each node."""
    output = _render(_tui(resource_view="servers"))
    line = next(line for line in output.splitlines() if "big-node" in line)
    assert "1× RTX PRO 5000" in line
    assert "G0 " in line and "67.4G/71.7G" in line
    assert "10.0.0.1" in line and _content_line(line).rstrip().endswith("up")


def test_a_thin_rule_separates_node_rows():
    output = _render(_tui(resource_view="servers"))
    lines = output.splitlines()
    first = next(i for i, line in enumerate(lines) if "big-node" in line)
    second = next(i for i, line in enumerate(lines) if "small-node" in line)
    assert any("┄┄┄" in lines[i] for i in range(first + 1, second))


def test_the_external_summary_joins_the_node_line_when_it_fits():
    snapshot = _snapshot()
    snapshot["nodes"][0]["gpus"] = [
        _gpu(
            0,
            name="RTX PRO 5000",
            total=73415,
            used=40000,
            external=40000,
            procs=1,
            processes=[_proc(100, 40000, "python train.py", "alice")],
        )
    ]
    output = _plain(_tui(resource_view="servers").render(_state(snapshot)))
    line = next(line for line in output.splitlines() if "big-node" in line)
    assert "ext " in line and "alice" in line
    # Everything moved inline, so no stacked ext line remains.
    assert not any(
        _content_line(line).lstrip().startswith("ext ") for line in output.splitlines()
    )


def test_a_narrow_screen_stacks_the_node_back_into_a_grid():
    snapshot = _snapshot()
    snapshot["nodes"][0]["gpus"] = [
        _gpu(index, name="RTX PRO 5000", total=73415) for index in range(3)
    ]
    tui = _tui(width=100, resource_view="servers")
    output = _plain(tui.render(_state(snapshot)))
    lines = output.splitlines()
    header = next(line for line in lines if "big-node" in line)
    # Three cells plus the head cannot share 100 columns, so the node falls
    # back to the stacked layout: header line, grid lines below.
    assert "G0 " not in header
    assert any(_content_line(line).startswith("    G0 ") for line in lines)


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
    tui = _tui(resource_view="servers")
    tui.proc_view = "all"
    output = _plain(tui.render(_state(snapshot)))
    assert "alice" in output and "bob" in output
    assert "unmanaged" in output


def test_capacity_reflects_foreign_usage():
    output = _render(_tui(resource_view="servers"))
    # The 69000MB of foreign memory shows up as the card's used total.
    assert "67.4G/71.7G" in output


def test_a_blind_node_says_so_instead_of_claiming_zero_processes():
    snapshot = _snapshot()
    snapshot["nodes"][1]["gpus"] = [_gpu(used=8000, external=0, reserved=8192,
                                         attribution="blind", jobs=1)]
    output = _render(_tui(resource_view="servers"), _state(snapshot))
    # A `~` before the used amount marks the split as inferred.
    assert "~7.8G/24.0G" in output


def test_progress_replaces_the_command_in_the_job_row():
    snapshot = _snapshot(jobs=[_job(1, progress="epoch 3/10 loss 0.42")])
    tui = _tui(resource_view="flow")
    tui.show_detail = True
    output = _render(tui, _state(snapshot))
    assert "PROGRESS / COMMAND" in output
    assert "▸ epoch 3/10 loss 0.42" in output
    # The command is still reachable in the detail pane.
    assert "python train.py" in output


def test_a_job_without_progress_still_shows_its_command():
    output = _render(_tui(resource_view="flow"))
    assert "PROGRESS" not in output
    assert "python train.py --epochs 10" in output


def test_the_detail_pane_shows_both_the_note_and_the_live_status():
    snapshot = _snapshot(
        jobs=[_job(1, progress="epoch 3/10", notes="baseline A, lr=1e-4")]
    )
    tui = _tui()
    tui.show_detail = True
    output = _render(tui, _state(snapshot))
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
    tui.show_detail = True

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
    tui.show_detail = True

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
    output = _render(_tui(resource_view="servers"), _state(snapshot))
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
    assert "Active 0" in output
    assert "No active jobs" in output


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
    tui = _tui(resource_view="flow")

    collapsed = _render(tui, state)
    assert "▶ JOB 1 DETAIL HIDDEN" in collapsed
    assert "Enter show detail" in collapsed
    assert "note visible detail" not in collapsed

    _press(tui, "", name="KEY_ENTER")
    expanded = _render(tui, state)
    assert "▼ JOB 1 DETAIL" in expanded
    assert "Enter hide detail" in expanded
    assert "note visible detail" in expanded

    # Some terminal definitions report Return separately from KEY_ENTER.
    _press(tui, "", name="KEY_RETURN")
    hidden_again = _render(tui, state)
    assert "▶ JOB 1 DETAIL HIDDEN" in hidden_again
    assert "note visible detail" not in hidden_again


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
    _render(tui)
    assert _click_target(tui, ("menu", "actions"))
    assert any("T theme:muted" == label for label, _target in tui._menu.entries)
    _press(tui, "", name="KEY_ESCAPE")
    _press(tui, "T")
    assert tui.theme == "classic"


def test_q_quits_and_other_keys_do_not():
    tui = _tui()
    _render(tui)
    assert _press(tui, "j") is True
    assert _press(tui, "q") is False


def test_command_registry_has_unique_bindings_in_each_context():
    seen = set()
    for command in COMMANDS:
        if command.help_key is not None:
            assert command.description
        for context in command.contexts:
            for kind, keys in (("text", command.text_keys), ("name", command.key_names)):
                for key in keys:
                    binding = (context, kind, key)
                    assert binding not in seen
                    seen.add(binding)


def test_g_and_G_follow_the_focused_pane():
    snapshot = _snapshot(
        nodes=[
            {
                **_snapshot()["nodes"][0],
                "name": f"node-{index}",
                "hostname": f"10.0.0.{index}",
            }
            for index in range(5)
        ]
    )
    tui = _tui()
    _render(tui, _state(snapshot))
    _press(tui, "", name="KEY_TAB")

    _press(tui, "G")
    assert tui.selected_node()["name"] == "node-4"
    _press(tui, "g")
    assert tui.selected_node()["name"] == "node-0"


def test_help_opens_and_closes_without_quitting():
    tui = _tui()
    _render(tui)
    _press(tui, "?")
    assert "Force a scheduler tick now" in _render(tui)
    # `q` inside help closes the panel rather than exiting the program.
    assert _press(tui, "q") is True
    assert tui.show_help is False


def test_help_is_contextual_and_fits_the_terminal():
    tui = _tui(width=60, height=20)
    _render(tui)

    _press(tui, "?")
    job_help = _render(tui)
    assert "Cancel the selected job" in job_help
    assert "Drain or resume" not in job_help
    assert len(job_help.splitlines()) <= 19

    _press(tui, "?")
    _press(tui, "", name="KEY_TAB")
    _press(tui, "?")
    node_help = _render(tui)
    assert "Drain or resume" in node_help
    assert "Cancel the selected job" not in node_help


def test_help_scrolls_with_keyboard_and_wheel_without_running_commands():
    tui = _tui(width=80, height=24)
    state = _state()
    _render(tui, state)
    _press(tui, "?")
    first = _render(tui, state)
    _press(tui, "", name="KEY_PGDOWN")
    last = _render(tui, state)
    assert first != last and "Mouse job" in last
    assert len(last.splitlines()) <= 23
    _press(tui, "c")
    assert tui.pending_confirm is None
    tui.handle_mouse(_mouse_at(last, "HELP", button=64))
    assert tui.help_offset < tui._help_max_offset
    _press(tui, "g")
    assert _render(tui, state) == first
    _press(tui, "", name="KEY_ESCAPE")
    assert not tui.show_help


# --- mouse interaction -----------------------------------------------------

@pytest.fixture
def hover_tui(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr("nvidb.sched.tui.age_seconds", lambda value: 125)
    tui = _tui(resource_view="servers")
    tui.term = Terminal(kind="xterm-256color", force_styling=True)
    return tui


@pytest.mark.parametrize("target", [
    ("tick", None), ("cancel", None), ("quit", None), ("menu", "filter"),
    ("job", 1), ("node", 1), ("gpu", (1, 0)),
])
def test_hover_highlights_queue_controls_and_rows_without_activating(hover_tui, target):
    tui = hover_tui
    before = tui.render(_state())
    region = next((region for region in tui._click_regions if region[3] == target), None)
    if region is not None:
        row, column, _end, _target = region
    else:
        row = next(row for row, value in tui._row_targets.items() if value == target)
        column = 0

    def selection_state():
        return (
            tui.focus, tui.job_index, tui.node_index, tui._selected_job_id,
            tui._cursor_gpu, tui.job_scope_node, tui.job_scope_gpu,
            tui.filter, tui.sort_key, tui.sort_reverse, tui.show_detail,
            tui.show_log, tui.show_help, tui.pending_confirm, tui._resource_scroll,
        )

    initial_selection = selection_state()
    assert tui.handle_mouse(MouseEvent(35, column + 1, row + 1, True)) is True
    hovered = tui.render(_state())
    assert _plain(hovered) == _plain(before)
    assert [index for index, (old, new) in enumerate(
        zip(before.splitlines(), hovered.splitlines())
    ) if old != new] == [row]
    assert selection_state() == initial_selection
    assert tui._menu is None
    assert tui.worker.actions.empty()
    assert tui.worker.log_request is None

    # Leaving every hit target restores the row without changing selection.
    assert tui.handle_mouse(MouseEvent(35, 1, tui.term.height, True)) is True
    assert tui.render(_state()) == before


def test_hover_ignores_non_actionable_queue_pane_space(hover_tui):
    tui = hover_tui
    before = tui.render(_state())
    row = next(
        row for row, target in tui._row_targets.items()
        if target == ("pane", "jobs")
        and not any(r == row and start <= tui.term.width - 1 <= end
                    for r, start, end, _ in tui._click_regions)
    )

    assert tui.handle_mouse(MouseEvent(35, tui.term.width, row + 1, True)) is True
    assert tui.render(_state()) == before
    assert tui.focus == "jobs"


def test_queue_hover_is_cleared_by_keyboard_input_and_opening_a_menu(hover_tui):
    tui = hover_tui
    before = tui.render(_state())
    motion = _mouse_at(_plain(before), "[t tick]", button=35)
    tui.handle_mouse(motion)
    assert tui.render(_state()) != before

    _press(tui, "z")
    assert tui.render(_state()) == before

    tui.handle_mouse(motion)
    assert tui.render(_state()) != before
    tui._activate("menu", "filter")
    _press(tui, "", "KEY_ESCAPE")
    assert tui._menu is None
    assert tui.render(_state()) == before


@pytest.mark.parametrize("disabled", ["mouse", "styling"])
def test_queue_hover_respects_disabled_mouse_and_plain_terminals(hover_tui, disabled, monkeypatch):
    tui = hover_tui
    if disabled == "mouse":
        tui.mouse_enabled = False
    else:
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        tui.term = Terminal(force_styling=False)
    before = tui.render(_state())

    tui.handle_mouse(_mouse_at(_plain(before), "[t tick]", button=35))

    assert tui.render(_state()) == before
    assert tui.worker.actions.empty()


def test_hovering_alert_rows_does_not_open_logs(hover_tui):
    tui = hover_tui
    snapshot = _snapshot()
    snapshot["alerts"] = [{
        "id": 3, "kind": "job_failed", "severity": "error", "job_id": 1,
        "node": "small-node", "title": "job 1 failed", "acknowledged_at": None,
    }]
    state = _state(snapshot)
    before = tui.render(state)
    motion = _mouse_at(_plain(before), "job 1 failed", button=35)

    tui.handle_mouse(motion)
    hovered = tui.render(state)

    assert _plain(hovered) == _plain(before)
    assert hovered.splitlines()[motion.row - 1] != before.splitlines()[motion.row - 1]
    assert tui.filter == "active"
    assert tui.show_log is False
    assert tui.worker.log_request is None
    assert tui.worker.actions.empty()


@pytest.fixture
def run_queue_input(monkeypatch):
    """Run the real input loop against deterministic terminal input and time."""
    def run(tui, steps):
        clock = 0.0
        consumed = 0
        frames = []
        full_renders = []
        state_reads = []
        lifecycle = []
        steps = iter(steps)

        def inkey(timeout):
            nonlocal clock, consumed
            assert 0 <= timeout <= 0.4 + 1e-9
            text, elapsed = next(steps)
            clock += elapsed
            consumed += 1
            return Keystroke(text)

        def frame(state):
            full_renders.append(consumed)
            tui._click_regions = [(0, 0, 8, ("tick", None))]
            return tui._paint_hover(["queue"])

        def read_state():
            state_reads.append(consumed)
            return _state()

        tui.term = SimpleNamespace(
            width=80, height=42, does_styling=False,
            fullscreen=nullcontext, cbreak=nullcontext,
            hidden_cursor=nullcontext, inkey=inkey,
        )
        monkeypatch.setattr("nvidb.sched.tui.time.monotonic", lambda: clock)
        monkeypatch.setattr("nvidb.sched.tui.time.time", lambda: clock)
        monkeypatch.setattr("nvidb.sched.tui.DiffScreen", lambda term: SimpleNamespace(
            paint=lambda lines: frames.append((consumed, clock, tui.pending_confirm)),
        ))
        monkeypatch.setattr(tui, "_frame_lines", frame)
        monkeypatch.setattr(tui, "_start_mouse_reporting", lambda: None)
        monkeypatch.setattr(tui, "_stop_mouse_reporting", lambda: None)
        monkeypatch.setattr(tui.worker, "read_state", read_state)
        monkeypatch.setattr(tui.worker, "start", lambda: lifecycle.append("start"))
        monkeypatch.setattr(tui.worker, "stop", lambda: lifecycle.append("stop"))
        monkeypatch.setattr(tui.worker, "join", lambda timeout: lifecycle.append("join"))
        assert tui.run() == 0
        assert lifecycle == ["start", "stop", "join"]
        return SimpleNamespace(frames=frames, full_renders=full_renders, state_reads=state_reads)
    return run


def test_queue_loop_batches_mouse_chunks_and_unchanged_hover(run_queue_input):
    tui = _tui()
    enter = "\x1b[<35;1;1M"
    inside = "\x1b[<35;3;1M"
    leave = "\x1b[<35;1;5M"
    report_chunks = enter + inside + leave
    steps = [(char, 0.001) for char in report_chunks]
    steps.extend([("z", 0.001), ("", 0.4), ("q", 0.001)])

    result = run_queue_input(tui, steps)

    assert [count for count, _time, _pending in result.frames] == [
        0, len(enter), len(report_chunks), len(report_chunks) + 1, len(report_chunks) + 2,
    ]
    assert result.full_renders == result.state_reads == [
        0, len(report_chunks) + 1, len(report_chunks) + 2,
    ]
    assert tui.worker.actions.empty()


def test_queue_loop_refreshes_worker_data_during_continuous_mouse_input(run_queue_input):
    tui = _tui()
    report = "\x1b[<35;1;1M"
    steps = [(char, 0.11) for char in report] + [("q", 0)]

    result = run_queue_input(tui, steps)

    # Refreshes keep happening even when no complete event has arrived yet.
    assert [count for count, _time, _pending in result.frames] == [0, 4, 8, len(report)]
    assert result.full_renders == result.state_reads == [0, 4, 8]


def test_queue_loop_expires_confirmation_during_partial_mouse_input(run_queue_input):
    tui = _tui()
    tui.pending_confirm = ("cancel", 1, -CONFIRM_SECONDS + 0.1)

    result = run_queue_input(tui, [
        ("\x1b", 0.2), ("[<35;1;1M", 0), ("q", 0),
    ])

    assert result.frames[0][2] is not None
    assert result.frames[1] == (1, 0.2, None)
    assert tui.pending_confirm is None
    assert tui.worker.actions.empty()


def test_queue_cached_hover_keeps_the_base_frame_clean(hover_tui):
    tui = hover_tui
    original = tui._frame_lines(_state())
    motion = _mouse_at("\n".join(_plain(line) for line in original), "[t tick]", button=35)

    tui.handle_mouse(motion)
    hovered = tui._paint_hover()

    assert hovered != original
    assert tui._base_frame == original
    assert hovered == tui._frame_lines(_state())
    tui.handle_mouse(MouseEvent(35, 1, tui.term.height, True))
    assert tui._paint_hover() == original


def test_queue_cached_hover_falls_back_after_resize_or_menu_change(hover_tui, monkeypatch):
    tui = hover_tui
    tui._frame_lines(_state())
    monkeypatch.setenv("COLUMNS", "100")
    assert tui._paint_hover() is None

    tui._frame_lines(_state())
    tui._activate("menu", "filter")
    assert tui._paint_hover() is None
    menu_lines = tui._frame_lines(_state())
    row, start, _end, _target = tui._menu.regions[0]
    tui.handle_mouse(MouseEvent(35, start + 1, row + 1, True))
    assert tui._paint_hover() != menu_lines
    assert tui._paint_hover() == tui._frame_lines(_state())
    _press(tui, "", "KEY_ESCAPE")
    assert tui._paint_hover() is None


def test_clicking_job_rows_selects_then_toggles_the_current_detail():
    tui = _tui(resource_view="flow")
    output = _render(tui)

    click = _mouse_at(output, "eval", occurrence=1)
    assert tui.handle_mouse(click) is True
    assert tui.focus == "jobs"
    assert tui.selected_job()["id"] == 2

    assert tui.show_detail is False
    assert tui.handle_mouse(click) is True
    assert tui.show_detail is True


def test_clicking_any_line_in_a_node_card_selects_that_node():
    tui = _tui(resource_view="servers")
    output = _render(tui)

    assert tui.handle_mouse(_mouse_at(output, "RTX 3090 Ti")) is True
    assert tui.focus == "nodes"
    assert tui.selected_node()["name"] == "small-node"
    # A server scope shows only work that is actually running there; a pending
    # job constrained to the server has not landed on it yet.
    assert [job["id"] for job in tui.jobs] == [1]


def test_server_and_gpu_clicks_drill_down_the_running_job_list():
    snapshot = _snapshot(
        jobs=[
            _job(1, node="small-node", gpu_ids=[0], elapsed_s=400),
            _job(2, node="small-node", gpu_ids=[1], elapsed_s=300),
            _job(3, node="small-node", gpu_ids=[0, 1], elapsed_s=200),
            _job(4, node="big-node", gpu_ids=[0], elapsed_s=500),
            _job(
                5,
                state="pending",
                node="small-node",
                gpu_ids=[1],
                started_at=None,
                elapsed_s=None,
            ),
        ]
    )
    snapshot["nodes"][1]["gpus"] = [_gpu(0), _gpu(1)]
    state = _state(snapshot)
    tui = _tui(resource_view="servers")
    output = _render(tui, state)

    assert tui.handle_mouse(_mouse_at(output, "small-node")) is True
    assert tui.job_scope_node == "small-node"
    assert tui.job_scope_gpu is None
    assert [job["id"] for job in tui.jobs] == [1, 2, 3]

    output = _render(tui, state)
    lines = output.splitlines()
    gpu_row = next(
        row
        for row, line in enumerate(lines)
        if "▾ [2]" in line and "G1 " in line
    )
    gpu_line = lines[gpu_row]
    gpu_start = gpu_line.index("G1 ")
    # Click the memory reading near the far end of the cell, not merely its
    # label, to verify that the whole rendered GPU card is a hit target.
    memory_column = gpu_line.index("0M/24.0G", gpu_start)
    assert tui.handle_mouse(
        MouseEvent(
            button=0,
            column=memory_column + 1,
            row=gpu_row + 1,
            pressed=True,
        )
    ) is True
    assert tui.job_scope_node == "small-node"
    assert tui.job_scope_gpu == 1
    assert [job["id"] for job in tui.jobs] == [2, 3]

    output = _render(tui, state)
    assert "JOBS (2) · running on small-node/GPU1" in output
    assert tui.handle_mouse(_mouse_at(output, "[all jobs]")) is True
    assert tui.job_scope_node is None and tui.job_scope_gpu is None
    assert [job["id"] for job in tui.jobs] == [1, 2, 3, 4, 5]


def test_gpu_click_regions_follow_the_stacked_narrow_layout():
    snapshot = _snapshot(
        jobs=[_job(7, node="small-node", gpu_ids=[10], elapsed_s=60)]
    )
    small = snapshot["nodes"][1]
    small["gpus"] = [_gpu(0), _gpu(1), _gpu(10)]
    snapshot["nodes"] = [small]
    tui = _tui(width=80, resource_view="servers")
    output = _render(tui, _state(snapshot))
    lines = output.splitlines()
    gpu_row = next(
        row for row, line in enumerate(lines)
        if _content_line(line).startswith("    G10")
    )
    memory_column = lines[gpu_row].index("0M/24.0G")

    assert tui.handle_mouse(
        MouseEvent(
            button=0,
            column=memory_column + 1,
            row=gpu_row + 1,
            pressed=True,
        )
    ) is True
    assert tui.job_scope_node == "small-node"
    assert tui.job_scope_gpu == 10
    assert [job["id"] for job in tui.jobs] == [7]


def test_wheel_routes_to_the_pane_under_the_pointer():
    tui = _tui(resource_view="flow")
    output = _render(tui)

    assert tui.handle_mouse(_mouse_at(output, "JOBS (2)", button=65)) is True
    assert tui.focus == "jobs"
    assert tui.selected_job()["id"] == 2

    assert tui.handle_mouse(_mouse_at(output, "TASK FLOW", button=65)) is True
    assert tui.focus == "nodes"
    assert tui.selected_node()["name"] == "big-node"
    assert tui._resource_scroll is not None


def test_wheel_pages_wrapped_detail_and_scrolls_back_through_logs():
    os.environ["LINES"] = "20"
    note = "\n".join(f"detail line {index}" for index in range(30))
    snapshot = _snapshot(jobs=[_job(1, notes=note)], nodes=[])
    state = _state(snapshot)
    tui = _tui(width=80)
    tui.show_detail = True
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
    tui = _tui(resource_view="flow")
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
    tui = _tui(resource_view="flow")
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
    tui = _tui(resource_view="flow")
    output = _render(tui)

    assert tui.handle_mouse(_mouse_at(output, "[? help]")) is True
    assert tui.show_help is True
    help_output = _render(tui)
    assert tui.handle_mouse(_mouse_at(help_output, "Mouse job")) is True
    assert tui.show_help is False

    output = _render(tui)
    assert tui.handle_mouse(_mouse_at(output, "[q quit]")) is False


def test_mouse_events_are_ignored_when_mouse_support_is_disabled():
    tui = QueueTUI(mouse_enabled=False)
    tui.resource_view = "flow"
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
    assert "Cancel #1?" in _render(tui)

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


def test_node_actions_only_apply_while_the_node_pane_has_focus():
    tui = _tui()
    output = _render(tui)

    assert "drain:big-node" not in output
    _press(tui, "d")
    assert tui.worker.actions.empty()

    _press(tui, "", name="KEY_TAB")
    assert "drain:big-node" in _render(tui)
    _press(tui, "d")
    assert tui.worker.actions.get_nowait() == ("drain", "big-node")


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

def test_queue_sort_preserves_lane_positions_even_when_priorities_differ():
    snapshot = _flow_snapshot()
    snapshot["jobs"][3]["priority"] = 99
    tui = _tui()
    tui.sort_key = "queue"
    _render(tui, _state(snapshot))
    assert [job["id"] for job in tui.jobs if job.get("lane") == "small-node:0"
            and job["state"] == "pending"] == [11, 12, 13]
    assert tui.jobs[1]["id"] == 20  # the free pool still uses priority


def test_runtime_sort_is_available_from_longest_to_shortest():
    tui = _tui()
    tui.sort_key = "time"
    snapshot = _snapshot(
        jobs=[
            _job(1, name="short", elapsed_s=15),
            _job(2, name="long", elapsed_s=900),
            _job(
                3,
                state="pending",
                name="waiting",
                started_at=None,
                elapsed_s=None,
            ),
            _job(4, name="middle", elapsed_s=120),
        ]
    )
    _render(tui, _state(snapshot))
    assert tui.sort_key == "time" and tui.sort_reverse is False
    assert [job["id"] for job in tui.jobs] == [2, 4, 1, 3]


def test_the_job_table_shows_priority_and_compact_states():
    tui = _tui(resource_view="flow")
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
    tui = _tui(resource_view="flow")
    output = _render(tui)
    assert tui.handle_mouse(_mouse_at(output, "NAME")) is True
    assert tui.sort_key == "name" and tui.sort_reverse is False

    output = _render(tui)
    assert "NAME▼" in output
    assert tui.handle_mouse(_mouse_at(output, "NAME▼")) is True
    assert tui.sort_reverse is True


def test_clicking_the_sort_label_cycles_the_sort():
    tui = _tui(resource_view="flow")
    output = _render(tui)
    assert tui.handle_mouse(_mouse_at(output, "sort queue▼")) is True
    assert tui.sort_key == "id"


def test_the_selection_follows_a_job_when_the_order_changes():
    tui = _tui()
    tui.sort_key = "queue"
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
    tui = _tui(resource_view="flow")
    _render(tui)
    _press(tui, "j")  # the pending job has both button groups
    output = _render(tui)
    assert tui.handle_mouse(_mouse_at(output, "[+ pri:0]")) is True
    assert tui.worker.actions.get_nowait() == ("priority", 2, 1)
    assert tui.handle_mouse(_mouse_at(output, "[K ▲queue]")) is True
    assert tui.worker.actions.get_nowait() == ("move", 2, -1)


def test_the_row_marker_tracks_the_detail_state():
    tui = _tui(resource_view="flow")
    output = _render(tui)
    row = next(
        line for line in output.splitlines() if "train" in line and "python" in line
    )
    assert _content_line(row).startswith("▶")

    _press(tui, "", name="KEY_ENTER")
    output = _render(tui)
    row = next(
        line for line in output.splitlines() if "train" in line and "python" in line
    )
    assert _content_line(row).startswith("▼")


@pytest.mark.parametrize("width", [36, 80, 150])
def test_queue_overflow_opens_direct_filter_menu_without_keyboard(width):
    tui = _tui(width=width)
    state = _state(_snapshot(jobs=[_job(1), _job(2, state="pending")]))
    _render(tui, state)

    def click_target(target):
        row, start, _, _ = next(r for r in tui._click_regions if r[3] == target)
        assert tui.handle_mouse(MouseEvent(0, start + 1, row + 1, True))

    def choose(target):
        menu = tui._menu
        index = next(i for i, (_, value) in enumerate(menu.entries) if value == target)
        menu.selected = index
        _render(tui, state)
        row, start, _, _ = next(r for r in menu.regions if r[3] == ("choose", index))
        assert tui.handle_mouse(MouseEvent(0, start + 1, row + 1, True))

    if not any(r[3] == ("menu", "filter") for r in tui._click_regions):
        click_target(("menu", "actions"))
        choose(("menu", "filter"))
    else:
        click_target(("menu", "filter"))
    choose(("filter", "pending"))
    assert tui.filter == "pending"
    assert [job["id"] for job in tui.jobs] == [2]
    assert tui._menu is None
    _render(tui, state)
    if not any(r[3] == ("menu", "sort") for r in tui._click_regions):
        click_target(("menu", "actions"))
        choose(("menu", "sort"))
    else:
        click_target(("menu", "sort"))
    choose(("sort", "pri"))
    assert tui.sort_key == "pri"
    assert tui.worker.actions.empty()


@pytest.mark.parametrize("input_kind", ["wheel_up", "wheel_down", "pane_click"])
def test_overview_server_mouse_input_keeps_keyboard_on_jobs(input_kind):
    tui = _tui(width=80, height=24)
    snapshot = _snapshot(jobs=[_job(1), _job(2), _job(3)])
    node = snapshot["nodes"][0]
    snapshot["nodes"] = [{**node, "name": f"node-{index}"} for index in range(6)]
    state = _state(snapshot)
    _render(tui, state)
    target = ("pane", "nodes") if input_kind == "pane_click" else ("node", 0)
    row = next(row for row, value in tui._row_targets.items() if value == target)
    button = {"wheel_up": 64, "wheel_down": 65, "pane_click": 0}[input_kind]

    tui.handle_mouse(MouseEvent(35, 3, row + 1, True))
    tui.handle_mouse(MouseEvent(button, 3, row + 1, True))
    _render(tui, state)
    assert tui.focus == "jobs"
    assert tui.resource_view == "overview"
    assert tui.selected_job()["id"] == 1
    if input_kind == "wheel_down":
        assert tui._resource_scroll > 0

    for text, name, job_id in [
        ("\x1b[B", "KEY_DOWN", 2), ("j", None, 3),
        ("k", None, 2), ("\x1b[A", "KEY_UP", 1),
    ]:
        assert tui.handle_key(Keystroke(text, name=name))
        output = _render(tui, state)
        assert tui.selected_job()["id"] == job_id
        assert f"#{job_id} train" in output
    assert tui.handle_key(Keystroke("\r", name="KEY_ENTER"))
    _render(tui, state)
    assert tui.show_detail
    assert tui.handle_key(Keystroke("L"))
    _render(tui, state)
    assert tui.show_log and tui.worker.log_request == (1, "stdout")
    assert tui.handle_key(Keystroke("c"))
    _render(tui, state)
    assert tui.pending_confirm[:2] == ("cancel", 1)
    assert tui.worker.actions.empty()
    assert tui.handle_key(Keystroke("q")) is False


@pytest.mark.parametrize("switch_method", ["shortcut", "menu"])
def test_returning_from_server_focus_to_overview_restores_job_keyboard(switch_method):
    tui = _tui(width=80, height=24)
    state = _state(_flow_snapshot())
    _render(tui, state)
    first_job = tui.selected_job()["id"]
    assert tui.handle_key(Keystroke("\t", name="KEY_TAB"))
    _render(tui, state)
    assert tui.resource_view == "servers" and tui.focus == "nodes"
    assert tui.handle_key(Keystroke("j"))
    _render(tui, state)
    assert tui.node_index == 1 and tui.selected_job()["id"] == first_job

    if switch_method == "shortcut":
        assert tui.handle_key(Keystroke("v"))
    else:
        assert _click_target(tui, ("menu", "view"))
        _render(tui, state)
        assert tui.handle_key(Keystroke("g"))
        _render(tui, state)
        menu = tui._menu
        row, start, _end, _target = next(region for region in menu.regions if region[3] == ("choose", 1))
        assert tui.handle_mouse(MouseEvent(35, start + 1, row + 1, True))
        _render(tui, state)
        assert menu.selected == 0  # Pointer feedback does not replace keyboard selection.
        assert tui.handle_key(Keystroke("\r", name="KEY_ENTER"))
    _render(tui, state)
    assert tui._menu is None
    assert tui.resource_view == "overview" and tui.focus == "jobs"
    assert tui.selected_job()["id"] == first_job

    assert tui.handle_key(Keystroke("\x1b[B", name="KEY_DOWN"))
    _render(tui, state)
    assert tui.selected_job()["id"] == 20
    assert tui.handle_key(Keystroke("\r", name="KEY_ENTER"))
    _render(tui, state)
    assert tui.show_detail
    assert tui.worker.actions.empty()
    assert tui.handle_key(Keystroke("q")) is False


def test_overview_node_alert_opens_visible_server_diagnostics_and_node_keyboard():
    tui = _tui(width=60, height=24)
    snapshot = _snapshot()
    snapshot["nodes"][1].update(state="down", last_error="Connection refused")
    snapshot["alerts"] = [{
        "id": 7, "kind": "node_down", "severity": "error", "job_id": None,
        "node": "small-node", "title": "small-node unreachable", "acknowledged_at": None,
    }]
    state = _state(snapshot)
    tui.show_detail = True  # Compact detail normally hides the resource pane.
    _render(tui, state)
    row = next(row for row, target in tui._row_targets.items() if target == ("node_name", "small-node"))

    assert tui.handle_mouse(MouseEvent(0, 1, row + 1, True))
    output = _render(tui, state)

    assert tui.resource_view == "servers" and tui.focus == "nodes"
    assert tui.selected_node()["name"] == "small-node"
    assert tui.show_detail is False
    assert "SERVERS (2)" in output and "Connection refused" in output
    assert tui.handle_key(Keystroke("k"))
    _render(tui, state)
    assert tui.selected_node()["name"] == "big-node"
    assert tui.handle_key(Keystroke("\t", name="KEY_TAB"))
    _render(tui, state)
    assert tui.focus == "jobs"
    assert tui.worker.actions.empty()
