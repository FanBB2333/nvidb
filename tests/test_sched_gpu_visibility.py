"""The queue must report what is on a GPU, not only what it put there.

Both machines run hand-launched work, so capacity numbers that only describe
queue jobs are misleading: a card can be full with nothing of ours on it. These
tests cover the whole path - NVML payload, database round trip, JSON, the CLI
tables and the TUI panes.
"""
import argparse
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fake_cluster import FakeCluster, FakeGpu, FakeNode  # noqa: E402

from nvidb.sched import cli as sched_cli  # noqa: E402
from nvidb.sched import db as dbm  # noqa: E402
from nvidb.sched.model import GpuProcess  # noqa: E402
from nvidb.sched.scheduler import MAX_RECORDED_GPU_PROCESSES, Scheduler  # noqa: E402


@pytest.fixture
def cluster():
    big = FakeNode("big-node", [FakeGpu(0, "RTX PRO 5000 72GB", 73415)])
    small = FakeNode("small-node", [FakeGpu(0, "RTX 3090 Ti", 24564)])
    return FakeCluster(big, small)


@pytest.fixture
def scheduler(tmp_path, cluster):
    conn = dbm.open_db(tmp_path / "queue.db")
    sched = Scheduler(
        conn,
        cfg=cluster.config(),
        settings={
            "headroom_mb": 512,
            "max_jobs_per_gpu": 4,
            "probe_timeout": 5,
            "launch_timeout": 5,
            "tick_min_interval": 0,
            "lock_ttl": 60,
            "job_root": ".nvidb/jobs",
            "default_vram": "0",
            "placement": "spread",
            "include_local": False,
        },
        backend_factory=cluster.backend_factory,
        owner="test",
    )
    sched.sync_nodes_from_config()
    yield sched
    sched.close()
    conn.close()


def _gpu(scheduler, node="big-node", index=0):
    return dbm.get_node(scheduler.conn, node).gpus[index]


# --- what the scheduler records -------------------------------------------

def test_foreign_processes_are_named_not_just_counted(scheduler, cluster):
    cluster["big-node"].add_foreign_process(
        0, 69000, pid=4242, name="python train.py", username="alice"
    )
    scheduler.tick(force=True)

    gpu = _gpu(scheduler)
    assert gpu.external_procs == 1
    assert len(gpu.processes) == 1
    process = gpu.processes[0]
    assert process.pid == 4242
    assert process.name == "python train.py"
    assert process.username == "alice"
    assert process.mem_mb == 69000
    assert process.managed is False
    assert process.owner == "external"


def test_a_queue_job_is_labelled_with_the_job_it_belongs_to(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("python train.py", name="train", vram="4G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].job_allocates(job_id, 0, 4000)
    scheduler.tick(force=True)

    gpu = _gpu(scheduler, "small-node")
    assert [process.job_id for process in gpu.processes] == [job_id]
    assert gpu.processes[0].owner == f"job {job_id} train"
    assert gpu.processes[0].managed is True
    assert gpu.external_processes == []
    # Our own usage is reported separately from foreign usage.
    assert gpu.queue_mem_mb == 4000
    assert gpu.external_mem_mb == 0


def test_both_kinds_of_work_show_up_on_one_card(scheduler, cluster):
    scheduler.tick(force=True)
    cluster["small-node"].add_foreign_process(0, 6000, name="ollama", username="root")
    job_id = scheduler.submit("a", name="mine", vram="4G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].job_allocates(job_id, 0, 4000)
    scheduler.tick(force=True)

    gpu = _gpu(scheduler, "small-node")
    assert gpu.mem_used_mb == 10000
    assert gpu.queue_mem_mb == 4000
    assert gpu.external_mem_mb == 6000
    # Biggest consumer first, whoever owns it.
    assert [process.mem_mb for process in gpu.processes] == [6000, 4000]
    assert [process.managed for process in gpu.processes] == [False, True]


def test_the_recorded_process_list_is_bounded_and_keeps_the_biggest(scheduler, cluster):
    for index in range(MAX_RECORDED_GPU_PROCESSES + 8):
        cluster["big-node"].add_foreign_process(0, 100 + index, name=f"proc{index}")
    scheduler.tick(force=True)

    gpu = _gpu(scheduler)
    assert gpu.external_procs == MAX_RECORDED_GPU_PROCESSES + 8  # counted in full
    assert len(gpu.processes) == MAX_RECORDED_GPU_PROCESSES  # but not all stored
    assert gpu.processes[0].mem_mb == 100 + MAX_RECORDED_GPU_PROCESSES + 7


def test_a_blind_node_reports_memory_without_inventing_processes(scheduler, cluster):
    wsl = cluster["small-node"]
    wsl.hide_processes = True
    scheduler.tick(force=True)
    job_id = scheduler.submit("a", vram="8G", node="small-node")
    scheduler.tick(force=True)
    wsl.job_allocates(job_id, 0, 8000)
    scheduler.tick(force=True)

    gpu = _gpu(scheduler, "small-node")
    assert gpu.attribution == "blind"
    assert gpu.processes == []
    assert gpu.mem_used_mb == 8000  # the card is busy even if nothing is named


def test_a_driver_that_lists_processes_without_memory_is_still_blind(scheduler, cluster):
    """WSL's NVML names the process but reports no memory for it.

    Taken at face value the job's own footprint is charged twice - once as
    foreign usage, again as its reservation - which is what wedged the real
    3090 Ti at zero free memory.
    """
    wsl = cluster["small-node"]
    wsl.hide_process_memory = True
    scheduler.tick(force=True)
    job_id = scheduler.submit("a", name="grid", vram="8G", node="small-node")
    scheduler.tick(force=True)
    wsl.job_allocates(job_id, 0, 22000)
    scheduler.tick(force=True)

    gpu = _gpu(scheduler, "small-node")
    assert gpu.attribution == "blind"
    assert gpu.mem_used_mb == 22000
    assert gpu.external_mem_mb == 22000 - 8192  # not charged on top of the reservation
    assert gpu.free_mb(512) == 24564 - (22000 - 8192) - 8192 - 512
    # The process is still named, and attributed to the job that owns it.
    assert [process.job_id for process in gpu.processes] == [job_id]
    assert gpu.processes[0].owner == f"job {job_id} grid"


def test_whole_card_utilisation_is_recorded(scheduler, cluster):
    cluster["big-node"].gpus[0].util = 97
    cluster["big-node"].gpus[0].mem_util = 61
    cluster["big-node"].add_foreign_process(0, 36000)
    scheduler.tick(force=True)

    gpu = _gpu(scheduler)
    assert gpu.util_percent == 97
    assert gpu.mem_util_percent == 61
    assert gpu.mem_used_percent() == 49


def test_the_snapshot_carries_the_unmanaged_split(scheduler, cluster):
    cluster["big-node"].add_foreign_process(0, 69000, name="python sweep.py")
    scheduler.tick(force=True)

    node = next(
        item for item in scheduler.snapshot()["nodes"] if item["name"] == "big-node"
    )
    gpu = node["gpus"][0]
    assert gpu["external_mem_mb"] == 69000
    assert gpu["queue_mem_mb"] == 0
    assert gpu["mem_used_percent"] == 94
    assert gpu["processes"][0]["name"] == "python sweep.py"
    assert gpu["processes"][0]["managed"] is False


def test_process_rows_survive_the_database_round_trip(scheduler, cluster):
    cluster["big-node"].add_foreign_process(0, 512, name="Xorg", username="root")
    scheduler.tick(force=True)

    # A fresh reader (another client) sees the same detail.
    reopened = dbm.get_node(scheduler.conn, "big-node").gpus[0]
    assert [(p.name, p.username) for p in reopened.processes] == [("Xorg", "root")]


def test_gpu_process_describes_itself_for_compact_views():
    process = GpuProcess.from_dict(
        {"pid": 7, "mem_mb": 6242, "name": "python train.py", "user": "alice"}
    )
    assert process.describe() == "python train.py (alice, 6.1G)"
    assert GpuProcess(pid=9).describe() == "pid 9 (0M)"


# --- what the CLI shows ----------------------------------------------------

@pytest.fixture
def cli(tmp_path, monkeypatch, cluster):
    """The real CLI parser and handlers, wired to the fake cluster."""
    path = tmp_path / "cli-queue.db"
    monkeypatch.setenv("NVIDB_QUEUE_DB", str(path))
    monkeypatch.setattr("nvidb.config.load_config", lambda *a, **k: cluster.config())

    real_scheduler = Scheduler

    def _patched(conn, **kwargs):
        kwargs.setdefault("backend_factory", cluster.backend_factory)
        kwargs.setdefault(
            "settings",
            {
                "headroom_mb": 512,
                "max_jobs_per_gpu": 4,
                "probe_timeout": 5,
                "launch_timeout": 5,
                "tick_min_interval": 0,
                "lock_ttl": 60,
                "job_root": ".nvidb/jobs",
                "default_vram": "0",
                "placement": "spread",
                "include_local": False,
            },
        )
        return real_scheduler(conn, **kwargs)

    monkeypatch.setattr(sched_cli, "Scheduler", _patched)

    root = argparse.ArgumentParser(prog="nvidb")
    subparsers = root.add_subparsers(dest="command")
    sched_cli.register_parsers(subparsers)

    def run(argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = sched_cli.dispatch(root.parse_args(argv))
        return code, buffer.getvalue()

    return run


def test_nodes_names_the_unmanaged_work_by_default(cli, cluster):
    cluster["big-node"].add_foreign_process(
        0, 69000, name="python pretrain.py", username="alice"
    )
    code, output = cli(["queue", "nodes"])
    assert code == 0
    assert "UNMANAGED" in output
    assert "python pretrain.py (alice, 67.4G)" in output


def test_nodes_summarises_extra_unmanaged_processes(cli, cluster):
    for index in range(6):
        cluster["big-node"].add_foreign_process(0, 1000 + index, name=f"proc{index}")
    _, output = cli(["queue", "nodes"])
    assert "in 6 process(es)" in output
    assert "+3 more" in output


def test_nodes_procs_lists_every_process_with_its_owner(cli, cluster):
    cluster["small-node"].add_foreign_process(0, 6000, name="ollama", username="root")
    assert cli(["job", "submit", "--name", "train", "--vram", "4G",
                "--node", "small", "--", "python", "train.py"])[0] == 0
    cluster["small-node"].job_allocates(1, 0, 4000)

    _, output = cli(["queue", "nodes", "--procs", "--tick"])
    assert "PID" in output and "OWNER" in output
    assert "ollama" in output and "external" in output
    assert "job 1 train" in output


def test_nodes_json_carries_the_processes_and_the_split(cli, cluster):
    cluster["big-node"].add_foreign_process(0, 69000, name="python sweep.py")
    _, output = cli(["queue", "nodes", "--json"])
    payload = json.loads(output)
    gpu = next(
        node for node in payload["nodes"] if node["name"] == "big-node"
    )["gpus"][0]
    assert gpu["external_mem_mb"] == 69000
    assert gpu["mem_used_mb"] == 69000
    assert gpu["mem_used_percent"] == 94
    assert gpu["queue_mem_mb"] == 0
    assert gpu["processes"][0]["name"] == "python sweep.py"


def test_status_shows_unmanaged_usage_alongside_the_jobs(cli, cluster):
    cluster["big-node"].add_foreign_process(0, 40000, name="python other.py")
    _, output = cli(["queue", "status"])
    assert "UNMANAGED" in output
    assert "python other.py" in output


def test_a_blind_card_says_its_split_is_inferred(cli, cluster):
    wsl = cluster["small-node"]
    wsl.hide_processes = True
    assert cli(["job", "submit", "--vram", "8G", "--node", "small", "--", "sleep", "1"])[0] == 0
    wsl.job_allocates(1, 0, 8000)
    _, output = cli(["queue", "nodes", "--tick"])
    assert "reports no per-process memory" in output
    assert "(blind)" in output


def test_a_blind_card_still_names_the_process_it_can_see(cli, cluster):
    wsl = cluster["small-node"]
    wsl.hide_process_memory = True
    assert cli(["job", "submit", "--name", "grid", "--vram", "8G", "--node", "small",
                "--", "sleep", "1"])[0] == 0
    wsl.job_allocates(1, 0, 22000)
    _, output = cli(["queue", "nodes", "--tick"])
    assert "reports no per-process memory" in output
    assert "job 1 grid" in output


# --- what the TUI shows ----------------------------------------------------

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


@pytest.fixture(autouse=True)
def wide_terminal():
    previous = {key: os.environ.get(key) for key in ("COLUMNS", "LINES")}
    os.environ["COLUMNS"] = "150"
    os.environ["LINES"] = "42"
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _tui_state(processes, attribution="processes", external=6000, queue_mem=4000):
    return {
        "snapshot": {
            "generated_at": "2026-07-28T10:02:00+00:00",
            "db_path": "/tmp/queue.db",
            "last_tick_at": "2026-07-28T10:01:58+00:00",
            "tick_lock": None,
            "settings": {},
            "counts": {},
            "nodes": [
                {
                    "name": "small-node",
                    "hostname": "10.0.0.2",
                    "state": "up",
                    "enabled": True,
                    "last_seen": "2026-07-28T10:01:58+00:00",
                    "last_error": None,
                    "gpu_count": 1,
                    "gpus": [
                        {
                            "index": 0,
                            "name": "RTX 3090 Ti",
                            "mem_total_mb": 24564,
                            "mem_used_mb": 10000,
                            "mem_used_percent": 41,
                            "external_mem_mb": external,
                            "external_procs": len(
                                [p for p in processes if not p["managed"]]
                            ),
                            "queue_mem_mb": queue_mem,
                            "attribution": attribution,
                            "reserved_mb": 4096,
                            "queue_jobs": 1,
                            "free_mb": 13000,
                            "util_percent": 55,
                            "mem_util_percent": 20,
                            "temperature_c": 50,
                            "processes": processes,
                            "updated_at": None,
                        }
                    ],
                }
            ],
            "jobs": [],
            "recent": [],
            "alerts": [],
            "open_alerts": 0,
        },
        "notice": None,
        "log_text": "",
        "log_job": None,
        "busy": False,
        "auto_tick": True,
        "error": None,
    }


def _proc(pid, mem, name, user="alice", job_id=None, job_name=None):
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


@pytest.fixture
def tui():
    from nvidb.sched.tui import QueueTUI

    return QueueTUI()


def _render(tui, state):
    return ANSI.sub("", tui.render(state))


def test_the_tui_lists_unmanaged_processes_by_default(tui):
    state = _tui_state(
        [_proc(11, 6000, "ollama", "root"), _proc(12, 4000, "python train.py", job_id=3, job_name="train")]
    )
    output = _render(tui, state)
    assert "ollama" in output
    assert "unmanaged" in output
    # The queue's own job is already visible in the job table below.
    assert "job 3 train" not in output


def test_pressing_p_cycles_to_every_process_then_to_none(tui):
    state = _tui_state(
        [_proc(11, 6000, "ollama", "root"), _proc(12, 4000, "python train.py", job_id=3, job_name="train")]
    )
    tui.render(state)  # populates the node list the key handler reads

    tui.handle_key(_Key("p"))
    assert tui.proc_view == "all"
    output = _render(tui, state)
    assert "ollama" in output and "job 3 train" in output

    tui.handle_key(_Key("p"))
    assert tui.proc_view == "off"
    output = _render(tui, state)
    assert "ollama" not in output

    tui.handle_key(_Key("p"))
    assert tui.proc_view == "summary"


def test_the_tui_says_when_a_cards_split_is_inferred(tui):
    output = _render(tui, _tui_state([], attribution="blind", external=8000))
    assert "reports no per-process memory" in output


def test_the_tui_names_the_job_behind_blind_memory(tui):
    state = _tui_state(
        [_proc(12, 0, "/python3.11", job_id=32, job_name="sweep")],
        attribution="blind",
        external=8000,
    )
    output = _render(tui, state)
    assert "job 32 sweep" in output


def test_a_long_process_list_cannot_push_the_jobs_off_the_screen(tui):
    processes = [_proc(100 + i, 500 - i, f"proc{i}") for i in range(30)]
    tui.proc_view = "all"
    state = _tui_state(processes)
    output = _render(tui, state)
    assert "more line(s), press p" in output
    # The footer survives, which is what the cap protects.
    assert "q quit" in output


class _Key(str):
    """blessed hands the handler a str subclass carrying a `name` attribute."""

    name = None
