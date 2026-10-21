"""Named GPU pools: a job may run on any member card, across nodes."""
import json
from pathlib import Path

import pytest
import yaml

from fake_cluster import FakeCluster, FakeGpu, FakeNode, make_scheduler

from nvidb.sched import cli as sched_cli
from nvidb.sched import db as dbm
from nvidb.sched.scheduler import gpu_pools

POOLS = {
    # One card on alpha, both on beta; gamma is outside the pool.
    "experiments": {
        "members": [
            {"node": "alpha", "gpus": [0]},
            {"node": "beta"},
        ]
    },
    "alpha-only": ["alpha"],
}


@pytest.fixture
def cluster():
    return FakeCluster(
        FakeNode("alpha", [FakeGpu(0, "RTX 4090", 24564), FakeGpu(1, "RTX 4090", 24564)]),
        FakeNode("beta", [FakeGpu(0, "RTX 3090", 24564), FakeGpu(1, "RTX 3090", 24564)]),
        FakeNode("gamma", [FakeGpu(0, "A100 80GB", 81920)]),
    )


def _scheduler(tmp_path, cluster, pools=None, **settings):
    cfg = cluster.config()
    cfg["queue"] = {"pools": POOLS if pools is None else pools}
    conn = dbm.open_db(tmp_path / "queue.db")
    return make_scheduler(conn, cluster, cfg=cfg, **settings)


@pytest.fixture
def scheduler(tmp_path, cluster):
    sched = _scheduler(tmp_path, cluster)
    yield sched
    sched.close()
    sched.conn.close()


def _job(sched, job_id):
    return dbm.get_job(sched.conn, job_id)


# --- configuration ---------------------------------------------------------

def test_pools_parse_members_shorthand_and_repeated_nodes():
    pools = gpu_pools({"queue": {"pools": {
        "a": {"members": [
            {"node": "n1", "gpus": [1]},
            {"node": "n1", "gpus": [0, 1]},
            "n2",
        ]},
        "b": [{"node": "n2", "gpus": [3]}, {"node": "n2"}],
    }}})
    assert pools == {"a": {"n1": [0, 1], "n2": None}, "b": {"n2": None}}
    assert gpu_pools({}) == {}
    assert gpu_pools({"queue": {}}) == {}


@pytest.mark.parametrize("pools, message", [
    (["a"], "must map"),
    ({"bad name": ["n1"]}, "pool names"),
    ({"a": {"members": []}}, "non-empty"),
    ({"a": [{"gpus": [0]}]}, "needs a `node`"),
    ({"a": [{"node": "n1", "gpus": []}]}, "names no card"),
    ({"a": [{"node": "n1", "gpus": [-1]}]}, "non-negative"),
    ({"a": [42]}, "a member is"),
])
def test_malformed_pools_are_rejected(pools, message):
    with pytest.raises(ValueError, match=message):
        gpu_pools({"queue": {"pools": pools}})


# --- submission ------------------------------------------------------------

def test_submit_resolves_a_pool_by_prefix_and_rejects_unknown_ones(scheduler):
    job_id = scheduler.submit("x", vram="8G", pool="exp")
    assert _job(scheduler, job_id).pool == "experiments"
    with pytest.raises(ValueError, match="Unknown pool"):
        scheduler.submit("x", pool="nope")


def test_a_pool_cannot_be_combined_with_a_node_or_a_lane(scheduler):
    with pytest.raises(ValueError, match="--pool"):
        scheduler.submit("x", pool="experiments", node="alpha")
    with pytest.raises(ValueError, match="--pool"):
        scheduler.submit("x", pool="experiments", lane="alpha:0")


# --- placement -------------------------------------------------------------

def test_pool_jobs_spread_over_member_cards_on_several_nodes(scheduler):
    jobs = [scheduler.submit(f"j{i}", vram="8G", pool="experiments") for i in range(4)]
    scheduler.tick(force=True)

    placed = {(_job(scheduler, j).node, tuple(_job(scheduler, j).gpu_ids)) for j in jobs[:3]}
    # Exclusive mode: one job per card, and only the three member cards.
    assert placed == {("alpha", (0,)), ("beta", (0,)), ("beta", (1,))}
    assert _job(scheduler, jobs[3]).state == "pending"


def test_a_pool_job_skips_busy_members_rather_than_using_outside_cards(scheduler, cluster):
    cluster["alpha"].add_foreign_process(0, 20000)
    cluster["beta"].add_foreign_process(0, 20000)
    cluster["beta"].add_foreign_process(1, 20000)
    job_id = scheduler.submit("x", vram="8G", pool="experiments")
    scheduler.tick(force=True)
    # alpha:1 and gamma:0 are free but outside the pool.
    assert _job(scheduler, job_id).state == "pending"

    cluster["beta"].clear_gpu(1)
    scheduler.tick(force=True)
    job = _job(scheduler, job_id)
    assert (job.state, job.node, job.gpu_ids) == ("running", "beta", [1])


def test_a_member_gpu_outside_the_node_allowlist_is_never_used(tmp_path, cluster):
    sched = _scheduler(tmp_path, cluster)
    sched.cfg["servers"][1]["gpu_ids"] = [0]  # beta hands the queue GPU 0 only
    sched.sync_nodes_from_config()
    try:
        jobs = [sched.submit(f"j{i}", vram="8G", pool="experiments") for i in range(3)]
        sched.tick(force=True)
        placed = sorted(
            (_job(sched, j).node, tuple(_job(sched, j).gpu_ids))
            for j in jobs if _job(sched, j).state == "running"
        )
        assert placed == [("alpha", (0,)), ("beta", (0,))]
    finally:
        sched.close()
        sched.conn.close()


def test_overlapping_pools_share_one_budget_per_card(scheduler):
    first = scheduler.submit("a", vram="8G", pool="alpha-only")
    second = scheduler.submit("b", vram="8G", pool="alpha-only")
    third = scheduler.submit("c", vram="8G", pool="alpha-only")
    scheduler.tick(force=True)
    assert {_job(scheduler, j).state for j in (first, second)} == {"running"}
    assert _job(scheduler, third).state == "pending"

    # alpha:0 belongs to both pools and is taken, so experiments goes to beta.
    other = scheduler.submit("d", vram="8G", pool="experiments")
    scheduler.tick(force=True)
    assert _job(scheduler, other).node == "beta"


def test_a_multi_gpu_job_needs_its_cards_on_one_node(scheduler):
    split = scheduler.submit("x", gpus=3, vram="8G", pool="experiments")
    pair = scheduler.submit("y", gpus=2, vram="8G", pool="experiments")
    scheduler.tick(force=True)

    failed = _job(scheduler, split)
    assert failed.state == "failed"
    assert "the largest node in pool experiments has 2" in failed.last_error
    job = _job(scheduler, pair)
    assert (job.state, job.node, sorted(job.gpu_ids)) == ("running", "beta", [0, 1])


def test_a_pool_job_is_judged_against_its_own_cards(scheduler):
    # gamma's 80G card could hold it, but gamma is not in the pool.
    job_id = scheduler.submit("x", vram="40G", pool="experiments")
    scheduler.tick(force=True)
    job = _job(scheduler, job_id)
    assert job.state == "failed"
    assert "largest schedulable GPU in pool experiments" in job.last_error


def test_a_pool_whose_cards_are_all_unseen_waits(tmp_path, cluster):
    cluster["alpha"].online = False
    cluster["beta"].online = False
    sched = _scheduler(tmp_path, cluster)
    try:
        job_id = sched.submit("x", vram="8G", pool="experiments")
        sched.tick(force=True)
        assert _job(sched, job_id).state == "pending"
        assert _job(sched, job_id).last_error is None
    finally:
        sched.close()
        sched.conn.close()


def test_cpu_only_pool_jobs_stay_on_member_nodes(scheduler):
    jobs = [scheduler.submit(f"c{i}", gpus=0, pool="alpha-only") for i in range(3)]
    scheduler.tick(force=True)
    assert {_job(scheduler, j).node for j in jobs} == {"alpha"}


# --- membership changes ----------------------------------------------------

def test_membership_changes_move_waiting_jobs_but_not_running_ones(scheduler):
    running = scheduler.submit("a", vram="8G", pool="alpha-only")
    scheduler.submit("b", vram="8G", pool="alpha-only")  # fills alpha's other card
    waiting = scheduler.submit("c", vram="8G", pool="alpha-only")
    scheduler.tick(force=True)
    assert _job(scheduler, running).node == "alpha"
    assert _job(scheduler, waiting).state == "pending"

    scheduler.cfg["queue"]["pools"] = {**POOLS, "alpha-only": ["gamma"]}
    scheduler.tick(force=True)

    assert (_job(scheduler, running).state, _job(scheduler, running).node) == ("running", "alpha")
    assert (_job(scheduler, waiting).state, _job(scheduler, waiting).node) == ("running", "gamma")


def test_a_removed_pool_holds_its_jobs_until_it_returns(scheduler):
    job_id = scheduler.submit("x", vram="8G", pool="alpha-only")
    scheduler.cfg["queue"]["pools"] = {"experiments": POOLS["experiments"]}
    for _ in range(3):
        scheduler.tick(force=True)

    job = _job(scheduler, job_id)
    assert job.state == "pending"
    assert "pool alpha-only is no longer configured" in job.held_reason
    held_events = [
        event for event in dbm.list_events(scheduler.conn, job_id=job_id)
        if event["kind"] == "job_held"
    ]
    assert len(held_events) == 1

    scheduler.cfg["queue"]["pools"] = dict(POOLS)
    scheduler.tick(force=True)
    job = _job(scheduler, job_id)
    assert (job.state, job.node, job.held_reason) == ("running", "alpha", None)


def test_assigning_a_pool_job_to_a_lane_takes_it_out_of_the_pool(scheduler):
    scheduler.tick(force=True)  # discover GPUs so lanes exist
    job_id = scheduler.submit("x", vram="8G", pool="experiments", priority=-1)
    scheduler.lane_assign(job_id, "gamma:0")
    job = _job(scheduler, job_id)
    assert (job.lane, job.node_constraint, job.pool) == ("gamma:0", "gamma", None)


# --- views -----------------------------------------------------------------

def test_pool_view_reports_member_cards_and_jobs(scheduler, cluster):
    cluster["beta"].online = False
    job_id = scheduler.submit("x", vram="8G", pool="experiments")
    queued = scheduler.submit("y", vram="8G", pool="experiments")
    scheduler.tick(force=True)

    view = next(p for p in scheduler.snapshot()["pools"] if p["name"] == "experiments")
    alpha, beta = view["members"]
    assert (alpha["node"], alpha["state"], alpha["gpu_ids"]) == ("alpha", "up", [0])
    assert [gpu["index"] for gpu in alpha["gpus"]] == [0]
    assert (beta["node"], beta["state"], beta["gpu_ids"], beta["gpus"]) == ("beta", "down", None, [])
    assert view["gpus_up"] == 1
    assert (view["running"], view["pending"]) == ([job_id], [queued])


@pytest.fixture
def parser():
    root = sched_cli.ArgumentParser(prog="nvidb")
    subparsers = root.add_subparsers(dest="command")
    sched_cli.register_parsers(subparsers)
    return root


def test_cli_submits_into_a_pool_and_lists_it(parser, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NVIDB_QUEUE_DB", str(tmp_path / "queue.db"))
    monkeypatch.setattr(
        "nvidb.config.load_config",
        lambda *a, **k: {"servers": [], "queue": {"pools": {"exp": ["lab1"]}}},
    )

    def run(*argv):
        return sched_cli.dispatch(parser.parse_args(list(argv)))

    assert run("job", "submit", "--pool", "exp", "--vram", "8G", "--", "python", "t.py") == 0
    assert "waiting in pool exp" in capsys.readouterr().out
    assert run("job", "submit", "--json", "--pool", "nope", "--", "true") == 1
    assert "Unknown pool" in json.loads(capsys.readouterr().out)["error"]

    assert run("queue", "pools", "--json") == 0
    (pool,) = json.loads(capsys.readouterr().out)["pools"]
    assert pool["name"] == "exp"
    assert pool["members"][0]["state"] == "unconfigured"
    assert pool["pending"] == [1]

    assert run("queue", "pools") == 0
    text = capsys.readouterr().out
    assert "POOL exp" in text and "lab1" in text and "unconfigured" in text

    assert run("job", "ls") == 0
    assert "pool:exp" in capsys.readouterr().out


def test_the_example_queue_config_defines_a_valid_pool():
    example = yaml.safe_load(
        (Path(__file__).parents[1] / "queue.example.yml").read_text()
    )
    assert gpu_pools(example) == {
        "experiments": {"gpu-node": [0], "queue-host": None}
    }
