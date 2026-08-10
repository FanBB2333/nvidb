"""Tests for the nvidb job queue: storage, scheduling policy and job lifecycle."""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fake_cluster import FakeCluster, FakeGpu, FakeNode  # noqa: E402

from nvidb.sched import db as dbm  # noqa: E402
from nvidb.sched.executor import build_run_script, parse_probe_output  # noqa: E402
from nvidb.sched.model import (  # noqa: E402
    display_width,
    fit_display,
    format_duration,
    format_mb,
    pad_display,
    parse_size_mb,
)
from nvidb.sched.scheduler import Scheduler  # noqa: E402


# --- fixtures --------------------------------------------------------------

@pytest.fixture
def cluster():
    """Two single-GPU machines shaped like the real ones: 72G busy, 24G free."""
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


# --- units -----------------------------------------------------------------

def test_parse_size_accepts_the_forms_people_type():
    assert parse_size_mb("20G") == 20480
    assert parse_size_mb("20g") == 20480
    assert parse_size_mb("512M") == 512
    assert parse_size_mb("1.5G") == 1536
    assert parse_size_mb(2048) == 2048
    assert parse_size_mb(None) == 0
    with pytest.raises(ValueError):
        parse_size_mb("many")


def test_format_helpers():
    assert format_mb(512) == "512M"
    assert format_mb(20480) == "20.0G"
    assert format_duration(3725) == "01:02:05"
    assert format_duration(90061) == "1-01:01:01"


def test_wide_characters_are_measured_in_terminal_columns():
    """Notes and job names are often Chinese, where one character is two columns."""
    assert display_width("abc") == 3
    assert display_width("基线实验") == 8
    assert display_width("lr=1e-4 复现") == 12
    assert fit_display("基线实验对照", 6) == "基线…"
    assert fit_display("abc", 10) == "abc"
    assert display_width(pad_display("基线", 10)) == 10


def test_a_chinese_note_keeps_a_table_aligned():
    from nvidb.sched.cli import _table

    rows = [["1", "基线实验 A"], ["2", "ascii only"], ["3", "x"]]
    lines = _table(rows, ["ID", "NOTE"], indent="").splitlines()
    # The NOTE column must start at the same terminal offset on every line,
    # which is what a character-counting `ljust` gets wrong.
    offsets = {
        display_width(line.split(marker)[0])
        for line, marker in zip(lines, ("NOTE", "基线", "ascii", "x"))
    }
    assert len(offsets) == 1


def test_run_script_embeds_the_command_without_quoting_it():
    script = build_run_script(
        job_id=3,
        job_name="demo",
        command="python train.py --note \"it's fine\"",
        run_dir="/home/u/.nvidb/jobs/3",
        workdir="/data/work",
        env={"WANDB_MODE": "offline"},
        gpu_ids=[1, 2],
        node_name="gpu-a",
    )
    assert "python train.py --note \"it's fine\"" in script
    assert "export CUDA_VISIBLE_DEVICES=1,2" in script
    assert "export WANDB_MODE=offline" in script
    assert "cd /data/work" in script
    # The exit code is published atomically, so a probe never reads a partial file.
    assert 'mv "$NVIDB_JOB_DIR/exit_code.tmp" "$NVIDB_JOB_DIR/exit_code"' in script


def test_probe_output_parses_both_sections():
    probe = parse_probe_output(
        "NVIDB_PROBE_V1\n"
        "JOB|7|4321|4321||1\n"
        "JOB|8|5555|5550|0 1753612345|0\n"
        "JOB|9|6666|6660|3|0\n"
        "PSTABLE\n"
        " 4321  4321\n"
        " 4400  4321\n"
    )
    assert probe.jobs[7].alive is True
    assert probe.jobs[7].exit_code is None
    assert probe.jobs[8].exit_code == 0
    assert probe.jobs[8].alive is False
    assert probe.jobs[8].finished_epoch == 1753612345
    # A status without a timestamp still parses, so an interrupted write or a
    # job started by an older version is not misread as still running.
    assert probe.jobs[9].exit_code == 3
    assert probe.jobs[9].finished_epoch is None
    assert probe.process_groups[4400] == 4321


# --- storage ---------------------------------------------------------------

def test_lock_is_exclusive_until_its_lease_expires(tmp_path):
    conn = dbm.open_db(tmp_path / "queue.db")
    try:
        assert dbm.acquire_lock(conn, "scheduler", "client-a", 60) is True
        assert dbm.acquire_lock(conn, "scheduler", "client-b", 60) is False
        # The holder may renew its own lease.
        assert dbm.acquire_lock(conn, "scheduler", "client-a", 60) is True
        dbm.release_lock(conn, "scheduler", "client-a")
        assert dbm.acquire_lock(conn, "scheduler", "client-b", 60) is True
    finally:
        conn.close()


def test_expired_lease_can_be_stolen(tmp_path):
    conn = dbm.open_db(tmp_path / "queue.db")
    try:
        assert dbm.acquire_lock(conn, "scheduler", "crashed", -1) is True
        # A client that died mid-tick must not wedge the queue forever.
        assert dbm.acquire_lock(conn, "scheduler", "healthy", 60) is True
    finally:
        conn.close()


def test_two_connections_see_each_others_writes(tmp_path):
    path = tmp_path / "queue.db"
    first = dbm.open_db(path)
    second = dbm.open_db(path)
    try:
        job_id = dbm.insert_job(first, name="a", command="echo hi", vram_mb=1024)
        assert dbm.get_job(second, job_id).command == "echo hi"
        dbm.update_job(second, job_id, state="running")
        assert dbm.get_job(first, job_id).state == "running"
    finally:
        first.close()
        second.close()


def test_purge_only_removes_finished_jobs(tmp_path):
    conn = dbm.open_db(tmp_path / "queue.db")
    try:
        keep = dbm.insert_job(conn, command="a")
        drop = dbm.insert_job(conn, command="b")
        dbm.update_job(conn, drop, state="completed")
        assert dbm.purge_jobs(conn) == 1
        assert dbm.get_job(conn, keep) is not None
        assert dbm.get_job(conn, drop) is None
    finally:
        conn.close()


# --- capacity accounting ---------------------------------------------------

def test_external_processes_are_counted_against_capacity(scheduler, cluster):
    cluster["big-node"].add_foreign_process(0, 69000)
    cluster["big-node"].gpus[0].util = 97
    scheduler.tick(force=True)

    big = dbm.get_node(scheduler.conn, "big-node")
    gpu = big.gpus[0]
    assert gpu.external_mem_mb == 69000
    assert gpu.external_procs == 1
    # 73415 total - 69000 foreign - 512 headroom
    assert gpu.free_mb(512) == pytest.approx(3903, abs=2)


def test_a_full_gpu_pushes_work_to_the_other_node(scheduler, cluster):
    cluster["big-node"].add_foreign_process(0, 69000)
    scheduler.tick(force=True)

    job_id = scheduler.submit("python train.py", name="train", vram="20G")
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "running"
    assert job.node == "small-node"


def test_a_job_larger_than_every_gpu_stays_pending(scheduler, cluster):
    cluster["big-node"].add_foreign_process(0, 69000)
    scheduler.tick(force=True)

    job_id = scheduler.submit("python huge.py", vram="40G")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "pending"


def test_capacity_frees_up_when_foreign_work_ends(scheduler, cluster):
    cluster["big-node"].add_foreign_process(0, 69000)
    scheduler.tick(force=True)
    job_id = scheduler.submit("python big.py", vram="40G", node="big-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "pending"

    cluster["big-node"].clear_gpu(0)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


def test_reservations_stop_a_second_job_from_oversubscribing(scheduler, cluster):
    scheduler.tick(force=True)
    first = scheduler.submit("a", vram="20G", node="small-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, first).state == "running"

    # The first job has not allocated anything yet, but its reservation holds.
    second = scheduler.submit("b", vram="20G", node="small-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, second).state == "pending"


def test_a_job_growing_past_its_reservation_is_accounted_honestly(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("a", vram="2G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].job_allocates(job_id, 0, 20000)
    scheduler.tick(force=True)

    node = dbm.get_node(scheduler.conn, "small-node")
    # Charged for what it really uses, not the 2G it asked for.
    assert node.gpus[0].reserved_mb == 20000
    assert node.gpus[0].external_mem_mb == 0
    assert dbm.get_job(scheduler.conn, job_id).gpu_mem_mb == 20000


def test_a_node_without_a_process_list_still_credits_its_own_jobs(scheduler, cluster):
    """WSL reports memory in use but names no processes.

    Without special handling a job's own memory would be charged twice: once as
    foreign usage and again as its reservation.
    """
    wsl = cluster["small-node"]
    wsl.hide_processes = True
    scheduler.tick(force=True)

    job_id = scheduler.submit("a", vram="8G", node="small-node")
    scheduler.tick(force=True)
    wsl.job_allocates(job_id, 0, 8000)
    scheduler.tick(force=True)

    gpu = dbm.get_node(scheduler.conn, "small-node").gpus[0]
    assert gpu.attribution == "blind"
    assert gpu.mem_used_mb == 8000
    assert gpu.external_mem_mb == 0  # not double-counted
    assert gpu.reserved_mb == 8192
    assert gpu.free_mb(512) == 24564 - 8192 - 512


def test_a_blind_node_still_sees_foreign_work_beyond_its_reservations(scheduler, cluster):
    wsl = cluster["small-node"]
    wsl.hide_processes = True
    wsl.add_foreign_process(0, 20000)
    scheduler.tick(force=True)

    gpu = dbm.get_node(scheduler.conn, "small-node").gpus[0]
    assert gpu.attribution == "processes"  # no queue jobs here, so nothing to infer
    assert gpu.external_mem_mb == 20000

    job_id = scheduler.submit("a", vram="1G", node="small-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"

    # Capacity is recomputed on the refresh pass, so the job shows up in the
    # accounting on the tick after the one that started it.
    scheduler.tick(force=True)
    gpu = dbm.get_node(scheduler.conn, "small-node").gpus[0]
    assert gpu.attribution == "blind"
    # 20000 of foreign memory minus the 1G this job is allowed to claim.
    assert gpu.external_mem_mb == 20000 - 1024


def test_queue_jobs_are_not_mistaken_for_foreign_work(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("a", vram="4G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].job_allocates(job_id, 0, 4000)
    scheduler.tick(force=True)

    gpu = dbm.get_node(scheduler.conn, "small-node").gpus[0]
    assert gpu.external_procs == 0
    assert gpu.external_mem_mb == 0
    assert gpu.mem_used_mb == 4000


# --- job lifecycle ---------------------------------------------------------

def test_a_finished_job_records_its_exit_code_and_result(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("python eval.py", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=0, result={"accuracy": 0.93})
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "completed"
    assert job.exit_code == 0
    assert job.result == {"accuracy": 0.93}
    assert job.finished_at is not None


def test_a_nonzero_exit_marks_the_job_failed(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("false", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=2)
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "failed"
    assert job.exit_code == 2


def test_a_vanished_process_is_reported_lost(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("python train.py", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].vanish_job(job_id)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "lost"


def test_a_vanished_process_is_retried_when_retries_remain(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit(
        "python train.py", vram="1G", node="small-node", max_retries=1
    )
    scheduler.tick(force=True)
    cluster["small-node"].vanish_job(job_id)
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "running"
    assert job.attempt == 2


def test_cancel_kills_a_running_job(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("sleep 999", vram="1G", node="small-node")
    scheduler.tick(force=True)

    assert scheduler.cancel(job_id) is True
    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "cancelled"
    assert job_id in cluster["small-node"].killed
    # A cancelled job must not be resurrected as "lost" on the next pass.
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "cancelled"


def test_cancel_releases_the_gpu_budget(scheduler, cluster):
    scheduler.tick(force=True)
    first = scheduler.submit("a", vram="20G", node="small-node")
    scheduler.tick(force=True)
    second = scheduler.submit("b", vram="20G", node="small-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, second).state == "pending"

    scheduler.cancel(first)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, second).state == "running"


def test_requeue_runs_a_finished_job_again(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("flaky.sh", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=1)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "failed"

    assert scheduler.requeue(job_id) is True
    cluster["small-node"].jobs.pop(job_id)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


def test_timeout_kills_an_overrunning_job(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit(
        "sleep 999", vram="1G", node="small-node", max_runtime_s=1
    )
    scheduler.tick(force=True)
    time.sleep(1.2)
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "timeout"
    assert job_id in cluster["small-node"].killed


# --- notes and progress ----------------------------------------------------

def test_a_note_can_be_set_at_submit_time_and_edited_later(scheduler):
    job_id = scheduler.submit("python train.py", notes="baseline A, lr=1e-4")
    assert dbm.get_job(scheduler.conn, job_id).notes == "baseline A, lr=1e-4"

    scheduler.set_notes(job_id, "loss plateaued", append=True)
    assert dbm.get_job(scheduler.conn, job_id).notes == "baseline A, lr=1e-4 | loss plateaued"

    scheduler.set_notes(job_id, "replaced")
    assert dbm.get_job(scheduler.conn, job_id).notes == "replaced"

    scheduler.set_notes(job_id, None)
    assert dbm.get_job(scheduler.conn, job_id).notes is None


def test_a_note_outlives_the_job_it_describes(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("run.sh", vram="1G", node="small-node", notes="run #3")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=0)
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "completed"
    assert job.notes == "run #3"
    # A client can still record what it concluded from the result.
    scheduler.set_notes(job_id, "accepted as the new baseline", append=True)
    assert "accepted" in dbm.get_job(scheduler.conn, job_id).notes


def test_annotating_a_missing_job_is_an_error(scheduler):
    with pytest.raises(ValueError):
        scheduler.set_notes(999, "nope")


def test_a_job_reports_its_own_progress(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).progress is None

    cluster["small-node"].report_progress(job_id, "epoch 3/10 loss 0.42")
    scheduler.tick(force=True)
    job = dbm.get_job(scheduler.conn, job_id)
    assert job.progress == "epoch 3/10 loss 0.42"
    assert job.progress_at is not None

    cluster["small-node"].report_progress(job_id, "epoch 7/10 loss 0.21")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).progress == "epoch 7/10 loss 0.21"


def test_the_last_progress_survives_into_the_finished_record(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].report_progress(job_id, "epoch 10/10 done")
    cluster["small-node"].finish_job(job_id, exit_code=0)
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "completed"
    assert job.progress == "epoch 10/10 done"


def test_progress_explains_how_far_a_lost_job_got(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].report_progress(job_id, "epoch 4/10")
    cluster["small-node"].vanish_job(job_id)
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "lost"
    assert job.progress == "epoch 4/10"


def test_a_retry_starts_with_a_clean_progress_line(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit(
        "train.sh", vram="1G", node="small-node", max_retries=1, notes="keep me"
    )
    scheduler.tick(force=True)
    cluster["small-node"].report_progress(job_id, "epoch 4/10")
    cluster["small-node"].vanish_job(job_id)
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "running"
    assert job.progress is None  # stale status must not look current
    assert job.notes == "keep me"  # the annotation is not the job's to reset


def test_requeue_keeps_the_note_and_drops_the_progress(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node", notes="run #3")
    scheduler.tick(force=True)
    cluster["small-node"].report_progress(job_id, "epoch 9/10")
    cluster["small-node"].finish_job(job_id, exit_code=1)
    scheduler.tick(force=True)

    scheduler.requeue(job_id)
    job = dbm.get_job(scheduler.conn, job_id)
    assert job.notes == "run #3"
    assert job.progress is None


def test_a_status_line_containing_pipes_is_not_mangled():
    probe = parse_probe_output(
        "NVIDB_PROBE_V1\n"
        "JOB|7|4321|4321||1\n"
        "STAT|7|epoch 3/10 | loss 0.42 | lr 1e-4\n"
    )
    assert probe.jobs[7].progress == "epoch 3/10 | loss 0.42 | lr 1e-4"
    assert probe.jobs[7].alive is True


def test_notes_and_progress_are_in_the_machine_readable_view(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node", notes="run #3")
    scheduler.tick(force=True)
    cluster["small-node"].report_progress(job_id, "epoch 3/10")
    scheduler.tick(force=True)

    job = next(job for job in scheduler.snapshot()["jobs"] if job["id"] == job_id)
    assert job["notes"] == "run #3"
    assert job["progress"] == "epoch 3/10"
    assert job["progress_at"]


# --- ordering and dependencies ---------------------------------------------

def test_priority_wins_over_arrival_order(scheduler, cluster):
    scheduler.tick(force=True)
    low = scheduler.submit("low", vram="20G", node="small-node", priority=0)
    high = scheduler.submit("high", vram="20G", node="small-node", priority=10)
    scheduler.tick(force=True)

    assert dbm.get_job(scheduler.conn, high).state == "running"
    assert dbm.get_job(scheduler.conn, low).state == "pending"


def test_a_dependent_job_waits_for_its_dependency(scheduler, cluster):
    scheduler.tick(force=True)
    first = scheduler.submit("stage1", vram="1G", node="small-node")
    second = scheduler.submit("stage2", vram="1G", node="small-node", depends_on=[first])
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, second).state == "pending"

    cluster["small-node"].finish_job(first, exit_code=0)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, second).state == "running"


def test_a_failed_dependency_fails_the_dependent_job(scheduler, cluster):
    scheduler.tick(force=True)
    first = scheduler.submit("stage1", vram="1G", node="small-node")
    second = scheduler.submit("stage2", vram="1G", node="small-node", depends_on=[first])
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(first, exit_code=1)
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, second)
    assert job.state == "failed"
    assert "dependency" in (job.last_error or "")


def test_submitting_an_unknown_dependency_is_rejected(scheduler):
    with pytest.raises(ValueError):
        scheduler.submit("x", depends_on=[999])


def test_node_names_resolve_by_prefix(scheduler):
    scheduler.submit("x", node="small")
    job = dbm.list_jobs(scheduler.conn)[0]
    assert job.node_constraint == "small-node"


def test_an_unknown_node_is_rejected_at_submit_time(scheduler):
    with pytest.raises(ValueError):
        scheduler.submit("x", node="does-not-exist")


# --- node availability -----------------------------------------------------

def test_work_waits_for_an_offline_node_and_starts_when_it_returns(scheduler, cluster):
    cluster["small-node"].online = False
    scheduler.tick(force=True)
    assert dbm.get_node(scheduler.conn, "small-node").state == "down"

    job_id = scheduler.submit("python train.py", vram="1G", node="small-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "pending"

    # The node comes back; nothing else has to happen for the job to start.
    cluster["small-node"].online = True
    scheduler.tick(force=True)
    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "running"
    assert job.node == "small-node"

    kinds = [event["kind"] for event in dbm.list_events(scheduler.conn)]
    assert "node_down" in kinds and "node_up" in kinds


def test_an_offline_node_reports_no_capacity(scheduler, cluster):
    scheduler.tick(force=True)
    assert dbm.get_node(scheduler.conn, "big-node").gpus

    cluster["big-node"].online = False
    scheduler.tick(force=True)
    node = dbm.get_node(scheduler.conn, "big-node")
    assert node.state == "down"
    assert node.gpus == []


def test_a_drained_node_receives_no_new_work(scheduler, cluster):
    scheduler.tick(force=True)
    dbm.set_node_enabled(scheduler.conn, "small-node", False)
    dbm.set_node_enabled(scheduler.conn, "big-node", False)

    job_id = scheduler.submit("python train.py", vram="1G")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "pending"

    dbm.set_node_enabled(scheduler.conn, "small-node", True)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


def test_a_launch_failure_leaves_the_job_pending_for_a_retry(scheduler, cluster):
    scheduler.tick(force=True)
    cluster["small-node"].launch_error = "disk full"
    cluster["big-node"].launch_error = "disk full"
    job_id = scheduler.submit("python train.py", vram="1G")
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "pending"
    assert "disk full" in (job.last_error or "")

    cluster["small-node"].launch_error = None
    cluster["big-node"].launch_error = None
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


def test_one_broken_node_does_not_stop_the_other(scheduler, cluster):
    cluster["big-node"].online = False
    job_id = scheduler.submit("python train.py", vram="1G")
    summary = scheduler.tick(force=True)

    assert summary["nodes_up"] == 1
    assert summary["nodes_down"] == 1
    assert dbm.get_job(scheduler.conn, job_id).node == "small-node"


# --- placement -------------------------------------------------------------

def test_spread_placement_prefers_the_emptiest_gpu(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("python train.py", vram="2G")
    scheduler.tick(force=True)
    # Both fit, so the 72G card wins on free memory.
    assert dbm.get_job(scheduler.conn, job_id).node == "big-node"


def test_pack_placement_prefers_the_fullest_gpu_that_fits(scheduler, cluster):
    scheduler.settings["placement"] = "pack"
    scheduler.tick(force=True)
    job_id = scheduler.submit("python train.py", vram="2G")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).node == "small-node"


def test_a_cpu_only_job_needs_no_gpu_budget(scheduler, cluster):
    cluster["big-node"].add_foreign_process(0, 73000)
    cluster["small-node"].add_foreign_process(0, 24000)
    scheduler.tick(force=True)

    job_id = scheduler.submit("python prepare_data.py", gpus=0, vram=0)
    scheduler.tick(force=True)
    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "running"
    assert job.gpu_ids == []


def test_max_jobs_per_gpu_is_enforced(scheduler, cluster):
    scheduler.settings["max_jobs_per_gpu"] = 2
    scheduler.tick(force=True)
    ids = [
        scheduler.submit(f"job{index}", vram="1G", node="small-node")
        for index in range(3)
    ]
    scheduler.tick(force=True)
    states = [dbm.get_job(scheduler.conn, job_id).state for job_id in ids]
    assert states.count("running") == 2
    assert states.count("pending") == 1


# --- concurrency between clients -------------------------------------------

def test_a_second_client_skips_a_tick_that_is_already_running(tmp_path, cluster):
    conn_a = dbm.open_db(tmp_path / "queue.db")
    conn_b = dbm.open_db(tmp_path / "queue.db")
    settings = {
        **Scheduler(conn_a, cfg=cluster.config()).settings,
        "tick_min_interval": 0,
    }
    try:
        first = Scheduler(
            conn_a,
            cfg=cluster.config(),
            settings=settings,
            backend_factory=cluster.backend_factory,
            owner="client-a",
        )
        second = Scheduler(
            conn_b,
            cfg=cluster.config(),
            settings=settings,
            backend_factory=cluster.backend_factory,
            owner="client-b",
        )
        # Simulate client-a being mid-tick when client-b arrives.
        dbm.acquire_lock(conn_a, "scheduler", "client-a", 60)
        summary = second.tick(force=True)
        assert summary["skipped"] == "locked"
        assert summary["lock_owner"] == "client-a"

        dbm.release_lock(conn_a, "scheduler", "client-a")
        assert second.tick(force=True)["ran"] is True
        first.close()
        second.close()
    finally:
        conn_a.close()
        conn_b.close()


def test_jobs_submitted_by_separate_clients_share_one_queue(tmp_path, cluster):
    path = tmp_path / "queue.db"
    conn_a = dbm.open_db(path)
    conn_b = dbm.open_db(path)
    common = dict(
        cfg=cluster.config(),
        settings={**Scheduler(conn_a, cfg=cluster.config()).settings, "tick_min_interval": 0},
        backend_factory=cluster.backend_factory,
    )
    try:
        client_a = Scheduler(conn_a, owner="client-a", **common)
        client_b = Scheduler(conn_b, owner="client-b", **common)
        client_a.sync_nodes_from_config()

        job_a = client_a.submit("train", vram="1G", submitter="client-a")
        job_b = client_b.submit("eval", vram="1G", submitter="client-b")

        # Either client's tick moves both jobs forward.
        client_b.tick(force=True)
        assert dbm.get_job(conn_a, job_a).state == "running"
        assert dbm.get_job(conn_a, job_b).state == "running"
        assert dbm.get_job(conn_b, job_a).submitter == "client-a"
        client_a.close()
        client_b.close()
    finally:
        conn_a.close()
        conn_b.close()


def test_rate_limiting_keeps_chatty_clients_off_the_nodes(scheduler):
    scheduler.settings["tick_min_interval"] = 60
    assert scheduler.tick(force=True)["ran"] is True
    assert scheduler.tick()["skipped"] == "rate_limited"
    assert scheduler.tick(force=True)["ran"] is True


# --- the machine-readable view ---------------------------------------------

def test_snapshot_carries_everything_a_client_needs(scheduler, cluster):
    cluster["big-node"].add_foreign_process(0, 69000)
    scheduler.tick(force=True)
    job_id = scheduler.submit("python train.py", name="train", vram="20G")
    scheduler.tick(force=True)

    snapshot = scheduler.snapshot()
    assert snapshot["counts"]["running"] == 1
    assert {node["name"] for node in snapshot["nodes"]} == {"big-node", "small-node"}
    big = next(node for node in snapshot["nodes"] if node["name"] == "big-node")
    assert big["gpus"][0]["external_mem_mb"] == 69000
    assert big["gpus"][0]["free_mb"] < 5000

    job = next(job for job in snapshot["jobs"] if job["id"] == job_id)
    assert job["node"] == "small-node"
    assert job["vram_mb"] == 20480
    assert job["state"] == "running"

    import json

    json.dumps(snapshot)  # must stay serializable for other tools


def test_events_form_a_replayable_stream(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("python train.py", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=0)
    scheduler.tick(force=True)

    events = dbm.list_events(scheduler.conn, limit=100)
    kinds = [event["kind"] for event in events]
    assert kinds.index("job_submitted") < kinds.index("job_started")
    assert kinds.index("job_started") < kinds.index("job_finished")

    # A client that was away resumes from the last id it saw.
    midpoint = events[1]["id"]
    later = dbm.list_events(scheduler.conn, since_id=midpoint)
    assert all(event["id"] > midpoint for event in later)


# --- priority and manual ordering ------------------------------------------

def _pending_order(scheduler):
    return [job.id for job in dbm.pending_jobs(scheduler.conn)]


def test_priority_can_be_set_and_nudged(scheduler):
    job_id = scheduler.submit("a", vram="1G")
    assert scheduler.set_priority(job_id, 5) == 5
    assert dbm.get_job(scheduler.conn, job_id).priority == 5
    assert scheduler.adjust_priority(job_id, -2) == 3
    assert dbm.get_job(scheduler.conn, job_id).priority == 3

    kinds = [event["kind"] for event in dbm.list_events(scheduler.conn, limit=100)]
    assert kinds.count("job_priority") == 2

    with pytest.raises(ValueError):
        scheduler.set_priority(9999, 1)


def test_concurrent_priority_adjustments_are_not_lost(scheduler):
    job_id = scheduler.submit("a", vram="1G")
    db_path = dbm.connection_path(scheduler.conn)
    workers = 4
    increments = 20
    barrier = threading.Barrier(workers)
    errors = []

    def adjust_repeatedly():
        conn = dbm.open_db(db_path)
        concurrent = Scheduler(conn, cfg={"servers": []})
        try:
            for _ in range(increments):
                barrier.wait(timeout=10)
                concurrent.adjust_priority(job_id, 1)
        except Exception as error:
            errors.append(error)
        finally:
            concurrent.close()
            conn.close()

    threads = [threading.Thread(target=adjust_repeatedly) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert dbm.get_job(scheduler.conn, job_id).priority == workers * increments
    events = dbm.list_events(scheduler.conn, job_id=job_id, limit=100)
    assert [event["kind"] for event in events].count("job_priority") == (
        workers * increments
    )


def test_a_finished_job_refuses_a_priority(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("a", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=0)
    scheduler.tick(force=True)

    assert scheduler.set_priority(job_id, 5) is None
    assert scheduler.adjust_priority(job_id, 1) is None


def test_moving_a_pending_job_changes_the_dispatch_order(scheduler):
    first = scheduler.submit("a", vram="1G")
    second = scheduler.submit("b", vram="1G")
    third = scheduler.submit("c", vram="1G")
    assert _pending_order(scheduler) == [first, second, third]

    # One slot up: the moved job passes exactly one neighbour.
    assert scheduler.move_pending(third, -1) is True
    assert _pending_order(scheduler) == [first, third, second]

    # And back down again.
    assert scheduler.move_pending(third, 1) is True
    assert _pending_order(scheduler) == [first, second, third]

    # All the way to the front, across jobs with differing priorities.
    scheduler.set_priority(first, 7)
    assert scheduler.move_pending(third, -2) is True
    assert _pending_order(scheduler) == [third, first, second]


def test_moving_past_the_edge_or_a_non_pending_job_is_refused(scheduler, cluster):
    scheduler.tick(force=True)
    running = scheduler.submit("a", vram="1G", node="small-node")
    scheduler.tick(force=True)
    waiting = scheduler.submit("b", vram="200G")  # can never fit anywhere

    assert scheduler.move_pending(waiting, -1) is False  # already first
    assert scheduler.move_pending(running, 1) is False  # not pending
    assert scheduler.move_pending(waiting, 0) is False
