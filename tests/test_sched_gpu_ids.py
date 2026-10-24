"""`job submit --gpu-ids`: a job pinned to named cards of one node, in order."""
import json
import sqlite3

import pytest

from fake_cluster import FakeCluster, FakeGpu, FakeNode, make_scheduler

from nvidb.sched import cli as sched_cli
from nvidb.sched import db as dbm
from nvidb.sched.executor import build_run_script

from test_sched_migration import LEGACY_SCHEMA


@pytest.fixture
def cluster():
    # Three 24G cards and one 80G card, so "a card that fits" and "the card
    # that was named" can disagree.
    return FakeCluster(
        FakeNode(
            "lab1",
            [
                FakeGpu(0, "RTX 4090", 24564),
                FakeGpu(1, "RTX 4090", 24564),
                FakeGpu(2, "RTX 4090", 24564),
                FakeGpu(3, "A100 80GB", 81920),
            ],
        ),
        FakeNode("lab2", [FakeGpu(0, "RTX 3090", 24564)]),
    )


def _scheduler(tmp_path, cluster, *, allowlist=None):
    cfg = cluster.config()
    if allowlist is not None:
        cfg["servers"][0]["gpu_ids"] = allowlist
    conn = dbm.open_db(tmp_path / "queue.db")
    sched = make_scheduler(conn, cluster, cfg=cfg)
    sched.tick(force=True)  # probe, so the cards are known at submit
    return sched


@pytest.fixture
def scheduler(tmp_path, cluster):
    sched = _scheduler(tmp_path, cluster)
    yield sched
    sched.close()
    sched.conn.close()


def _job(sched, job_id):
    return dbm.get_job(sched.conn, job_id)


def _submit(sched, gpu_ids, *, gpus=None, vram="8G", node="lab1", **extra):
    return sched.submit(
        "python serve.py",
        gpus=len(gpu_ids) if gpus is None else gpus,
        vram=vram,
        node=node,
        gpu_ids=gpu_ids,
        **extra,
    )


# --- submit rules ----------------------------------------------------------

def test_the_pin_is_stored_in_request_order(scheduler):
    job = _job(scheduler, _submit(scheduler, [3, 1]))
    assert job.gpu_constraint == [3, 1]
    assert job.node_constraint == "lab1"
    assert job.to_dict()["gpu_constraint"] == [3, 1]


def test_a_gpus_count_that_disagrees_with_the_ids_is_rejected(scheduler):
    with pytest.raises(ValueError, match="--gpus 3 disagrees with --gpu-ids 0,1"):
        _submit(scheduler, [0, 1], gpus=3)


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"node": None}, "--gpu-ids needs --node"),
        ({"node": None, "pool": "exp"}, "cannot be combined with --pool or --lane"),
        ({"node": None, "lane": "lab1:0"}, "cannot be combined with --pool or --lane"),
    ],
)
def test_the_pin_needs_a_node_and_excludes_pools_and_lanes(scheduler, extra, message):
    with pytest.raises(ValueError, match=message):
        _submit(scheduler, [0], **extra)


@pytest.mark.parametrize(
    "gpu_ids, message",
    [
        ([1, 1], "repeats GPU 1"),
        ([-1], "cannot be negative: -1"),
        ([2, 7], "lab1 has no GPU 7; its GPUs are 0,1,2,3"),
    ],
)
def test_impossible_indices_are_rejected_at_submit(scheduler, gpu_ids, message):
    with pytest.raises(ValueError, match=message):
        _submit(scheduler, gpu_ids)
    assert dbm.list_jobs(scheduler.conn) == []


def test_indices_outside_the_allowlist_are_rejected_at_submit(tmp_path, cluster):
    sched = _scheduler(tmp_path, cluster, allowlist=[0, 1])
    try:
        with pytest.raises(ValueError, match="GPU 2 on lab1 is outside .* may use 0,1"):
            _submit(sched, [1, 2])
    finally:
        sched.close()
        sched.conn.close()


# --- placement -------------------------------------------------------------

def test_an_idle_node_gives_exactly_the_named_cards_in_order(scheduler, cluster):
    job_id = _submit(scheduler, [3, 1])
    scheduler.tick(force=True)
    job = _job(scheduler, job_id)
    assert job.state == "running"
    assert job.gpu_ids == [3, 1]
    assert cluster["lab1"].jobs[job_id]["gpu_ids"] == [3, 1]
    script = build_run_script(
        job_id=job_id, job_name="", command="true", run_dir="/r", workdir=None,
        gpu_ids=job.gpu_ids, node_name="lab1",
    )
    assert "export CUDA_VISIBLE_DEVICES=3,1" in script


def test_a_busy_pinned_card_makes_the_job_wait_not_move(scheduler, cluster):
    foreign = cluster["lab1"].add_foreign_process(3, 80000)
    job_id = _submit(scheduler, [3])
    for _ in range(2):
        scheduler.tick(force=True)
        job = _job(scheduler, job_id)
        # Cards 0-2 are idle and would fit, but they were not asked for.
        assert (job.state, job.gpu_ids, job.held_reason) == ("pending", [], None)

    cluster["lab1"].gpus[3].processes.pop(foreign)
    scheduler.tick(force=True)
    job = _job(scheduler, job_id)
    assert (job.state, job.gpu_ids) == ("running", [3])


# --- failures and holds ----------------------------------------------------

def test_a_pinned_card_too_small_for_the_request_is_unschedulable(scheduler):
    # 40G would fit on card 3, but the job named card 0.
    job_id = _submit(scheduler, [0], vram="40G")
    scheduler.tick(force=True)
    job = _job(scheduler, job_id)
    assert job.state == "failed"
    assert job.last_error == "reserves 40.0G; GPU 0 on lab1 holds 23.5G at most"


def test_a_pinned_card_that_disappears_is_unschedulable(scheduler, cluster):
    cluster["lab1"].add_foreign_process(2, 24000)
    job_id = _submit(scheduler, [2])
    scheduler.tick(force=True)
    cluster["lab1"].gpus.pop(2)
    scheduler.tick(force=True)
    job = _job(scheduler, job_id)
    assert job.state == "failed"
    assert "asks for GPU 2 on lab1, which has no such GPU (has 0,1,3)" in job.last_error


def test_an_allowlist_edit_holds_the_job_and_restoring_it_lets_it_run(scheduler, cluster):
    foreign = cluster["lab1"].add_foreign_process(1, 24000)
    job_id = _submit(scheduler, [1])
    scheduler.tick(force=True)
    assert _job(scheduler, job_id).state == "pending"

    scheduler.cfg["servers"][0]["gpu_ids"] = [0, 2, 3]
    scheduler.tick(force=True)
    job = _job(scheduler, job_id)
    assert job.state == "pending"
    assert job.held_reason == (
        "GPU 1 on lab1 is outside the node's gpu_ids allowlist; "
        "restore it or cancel the job"
    )

    del scheduler.cfg["servers"][0]["gpu_ids"]
    cluster["lab1"].gpus[1].processes.pop(foreign)
    scheduler.tick(force=True)
    job = _job(scheduler, job_id)
    assert job.held_reason is None
    assert (job.state, job.gpu_ids) == ("running", [1])


# --- lanes and requeue -----------------------------------------------------

def test_lane_assign_drops_the_pin(scheduler, cluster):
    cluster["lab1"].add_foreign_process(1, 24000)
    job_id = _submit(scheduler, [1])
    scheduler.tick(force=True)
    scheduler.lane_assign(job_id, "lab1:2")
    job = _job(scheduler, job_id)
    assert (job.lane, job.gpu_constraint) == ("lab1:2", [])


def test_requeue_keeps_the_pin(scheduler, cluster):
    job_id = _submit(scheduler, [3, 1])
    scheduler.tick(force=True)
    cluster["lab1"].finish_job(job_id, 0)
    scheduler.tick(force=True)
    assert _job(scheduler, job_id).state == "completed"

    assert scheduler.requeue(job_id)
    assert _job(scheduler, job_id).gpu_constraint == [3, 1]
    scheduler.tick(force=True)
    job = _job(scheduler, job_id)
    assert (job.state, job.gpu_ids) == ("running", [3, 1])


# --- CLI -------------------------------------------------------------------

@pytest.fixture
def parser():
    root = sched_cli.ArgumentParser(prog="nvidb")
    subparsers = root.add_subparsers(dest="command")
    sched_cli.register_parsers(subparsers)
    return root


@pytest.fixture
def run(parser, tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDB_QUEUE_DB", str(tmp_path / "queue.db"))
    monkeypatch.setattr(
        "nvidb.config.load_config",
        lambda *a, **k: {"servers": [{"hostname": "10.0.0.1", "nickname": "lab1"}]},
    )

    def run(*argv):
        # --no-tick throughout: the node is configured but never contacted.
        return sched_cli.dispatch(parser.parse_args(list(argv)))

    return run


def test_cli_submits_a_pinned_job_and_shows_where_it_waits(run, capsys):
    assert run(
        "job", "submit", "--no-tick", "--node", "lab1", "--gpu-ids", "4,5,6,7",
        "--vram", "23G", "--", "python", "serve.py",
    ) == 0
    assert "waiting for GPU 4,5,6,7 on lab1" in capsys.readouterr().out

    assert run("job", "ls", "--no-tick", "--json") == 0
    (job,) = json.loads(capsys.readouterr().out)["jobs"]
    assert (job["gpus"], job["gpu_constraint"]) == (4, [4, 5, 6, 7])

    assert run("job", "ls", "--no-tick") == 0
    assert "lab1:4,5,6,7" in capsys.readouterr().out

    assert run("job", "show", "--no-tick", "1") == 0
    assert "node=lab1 gpu_ids=4,5,6,7" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--gpu-ids", "4,x"], "--gpu-ids expects comma-separated GPU indices"),
        (["--node", "lab1", "--gpus", "3", "--gpu-ids", "4,5"], "--gpus 3 disagrees with --gpu-ids 4,5"),
        (["--gpu-ids", "0"], "--gpu-ids needs --node"),
    ],
)
def test_cli_rejects_bad_pins(run, capsys, argv, message):
    assert run("job", "submit", "--no-tick", "--json", "--vram", "8G", *argv, "--", "true") == 1
    assert message in json.loads(capsys.readouterr().out)["error"]


def test_cli_gpus_still_defaults_to_one(run, capsys):
    assert run("job", "submit", "--no-tick", "--json", "--vram", "8G", "--", "true") == 0
    assert json.loads(capsys.readouterr().out)["job"]["gpus"] == 1


# --- migration -------------------------------------------------------------

def test_an_old_database_gains_the_column_and_old_rows_read_unpinned(tmp_path):
    path = tmp_path / "old-queue.db"
    legacy = sqlite3.connect(str(path))
    legacy.executescript(LEGACY_SCHEMA)
    legacy.execute(
        "INSERT INTO jobs(name, command, state, created_at, updated_at) "
        "VALUES('legacy', 'train.sh', 'pending', '2026-01-01T00:00:00+00:00', "
        "'2026-01-01T00:00:00+00:00')",
    )
    legacy.commit()
    legacy.close()

    conn = dbm.open_db(path)
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        assert "gpu_constraint" in columns
        assert dbm.get_job(conn, 1).gpu_constraint == []
    finally:
        conn.close()
