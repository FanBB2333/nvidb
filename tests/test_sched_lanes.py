"""Tests for lanes: per-GPU serial queues whose order is stored and editable.

A lane exists so that "what this card runs next, and next after that" is a fact
in the database rather than something implied by priorities and dependency
chains. These tests are mostly about that promise: the printed order is the
order, editing it is cheap, and nothing silently runs out of turn.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fake_cluster import FakeCluster, FakeGpu, FakeNode  # noqa: E402

from nvidb.sched import db as dbm  # noqa: E402
from nvidb.sched.model import make_lane_name, parse_lane_name  # noqa: E402
from nvidb.sched.scheduler import Scheduler  # noqa: E402


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


def _queue_ids(scheduler, lane):
    return [job.id for job in dbm.lane_jobs(scheduler.conn, lane, states=["pending"])]


def _fill(scheduler, lane, count, **kwargs):
    return [
        scheduler.submit(f"step{index}", lane=lane, vram="1G", **kwargs)
        for index in range(count)
    ]


# --- naming ----------------------------------------------------------------

def test_a_lane_name_round_trips():
    name = make_lane_name("box406", [0])
    assert name == "box406:0"
    assert parse_lane_name(name) == ("box406", [0])


def test_a_node_named_with_a_port_still_parses():
    """A server with no nickname is called user@host:port, so the separator
    has to be the last colon rather than the first."""
    name = make_lane_name("tester@10.0.0.1:22", [1])
    assert parse_lane_name(name) == ("tester@10.0.0.1:22", [1])


@pytest.mark.parametrize("bad", ["", "no-colon", "node:", "node:x"])
def test_a_malformed_lane_name_is_rejected(bad):
    with pytest.raises(ValueError):
        parse_lane_name(bad)


# --- discovery -------------------------------------------------------------

def test_every_schedulable_gpu_gets_a_lane(scheduler):
    scheduler.tick(force=True)
    names = {lane.name for lane in dbm.get_lanes(scheduler.conn)}
    assert names == {"big-node:0", "small-node:0"}


def test_lanes_list_in_config_order_then_by_gpu(tmp_path):
    """The lanes listing mirrors the node listing: nodes as config.yml orders
    them, and one node's lanes numerically, so box:10 follows box:2."""
    cluster = FakeCluster(
        FakeNode(
            "zeta-node",
            [FakeGpu(index, "RTX 3090 Ti", 24564) for index in (0, 2, 10)],
        ),
        FakeNode("alpha-node", [FakeGpu(0, "RTX 3090 Ti", 24564)]),
    )
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
    try:
        sched.sync_nodes_from_config()
        sched.tick(force=True)
        assert [lane.name for lane in dbm.get_lanes(conn)] == [
            "zeta-node:0",
            "zeta-node:2",
            "zeta-node:10",
            "alpha-node:0",
        ]
    finally:
        sched.close()
        conn.close()


def test_discovery_does_not_disturb_a_lane_a_person_configured(scheduler):
    scheduler.tick(force=True)
    scheduler.set_lane_paused("small-node:0", True)
    scheduler.set_lane_concurrency("small-node:0", 3)
    scheduler.tick(force=True)

    lane = dbm.get_lane(scheduler.conn, "small-node:0")
    assert lane.paused is True
    assert lane.concurrency == 3


# --- serial execution ------------------------------------------------------

def test_a_lane_runs_one_job_at_a_time_in_order(scheduler, cluster):
    scheduler.tick(force=True)
    ids = _fill(scheduler, "small-node:0", 3)
    scheduler.tick(force=True)

    states = [dbm.get_job(scheduler.conn, job_id).state for job_id in ids]
    assert states == ["running", "pending", "pending"]

    cluster["small-node"].finish_job(ids[0], exit_code=0)
    scheduler.tick(force=True)
    states = [dbm.get_job(scheduler.conn, job_id).state for job_id in ids]
    assert states == ["completed", "running", "pending"]


def test_a_lane_job_lands_on_the_lane_s_own_gpu(scheduler):
    """Two cards on one machine, and a lane must not wander between them."""
    node = FakeNode(
        "two-card", [FakeGpu(0, "A", 24564), FakeGpu(1, "B", 24564)]
    )
    cluster = FakeCluster(node)
    scheduler.cfg = cluster.config()
    scheduler._backend_factory = cluster.backend_factory
    scheduler._configured_nodes = {"two-card"}
    scheduler._last_sync_signature = None
    scheduler.sync_nodes_from_config()
    scheduler.tick(force=True)

    job_id = scheduler.submit("x", lane="two-card:1", vram="1G")
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "running"
    assert job.gpu_ids == [1]


def test_a_paused_lane_starts_nothing_new(scheduler, cluster):
    scheduler.tick(force=True)
    ids = _fill(scheduler, "small-node:0", 2)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, ids[0]).state == "running"

    scheduler.set_lane_paused("small-node:0", True)
    cluster["small-node"].finish_job(ids[0], exit_code=0)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, ids[1]).state == "pending"

    scheduler.set_lane_paused("small-node:0", False)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, ids[1]).state == "running"


def test_pausing_a_lane_leaves_the_running_job_alone(scheduler, cluster):
    scheduler.tick(force=True)
    ids = _fill(scheduler, "small-node:0", 2)
    scheduler.tick(force=True)
    scheduler.set_lane_paused("small-node:0", True)
    scheduler.tick(force=True)

    assert dbm.get_job(scheduler.conn, ids[0]).state == "running"
    assert cluster["small-node"].killed == []


def test_a_held_head_blocks_its_lane_rather_than_being_stepped_over(scheduler, cluster):
    """The order is a promise. Quietly running the second job because the
    first is stuck would make the printed queue a lie."""
    scheduler.tick(force=True)
    upstream = scheduler.submit("upstream", vram="1G", node="big-node")
    blocked = scheduler.submit(
        "blocked", lane="small-node:0", vram="1G", depends_on=[upstream]
    )
    behind = scheduler.submit("behind", lane="small-node:0", vram="1G")
    scheduler.tick(force=True)
    scheduler.cancel(upstream)
    scheduler.tick(force=True)

    assert dbm.get_job(scheduler.conn, blocked).is_held
    assert dbm.get_job(scheduler.conn, behind).state == "pending"

    # Skipping past it is a decision someone makes, not something that happens.
    scheduler.lane_skip("small-node:0")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, behind).state == "running"


def test_a_lane_waits_when_foreign_work_fills_its_card(scheduler, cluster):
    scheduler.tick(force=True)
    cluster["small-node"].add_foreign_process(0, 24000)
    job_id = scheduler.submit("big", lane="small-node:0", vram="20G")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "pending"

    cluster["small-node"].clear_gpu(0)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


def test_a_lane_on_an_unreachable_node_just_waits(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("x", lane="small-node:0", vram="1G")
    cluster["small-node"].online = False
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "pending"

    cluster["small-node"].online = True
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


# --- editing the order -----------------------------------------------------

def test_moving_a_job_to_the_head_makes_it_run_next(scheduler, cluster):
    scheduler.tick(force=True)
    ids = _fill(scheduler, "small-node:0", 4)
    scheduler.tick(force=True)  # ids[0] starts; ids[1:] queue

    scheduler.lane_move(ids[3], to="head")
    assert _queue_ids(scheduler, "small-node:0") == [ids[3], ids[1], ids[2]]

    cluster["small-node"].finish_job(ids[0], exit_code=0)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, ids[3]).state == "running"


def test_reordering_needs_no_contact_with_the_node(scheduler, cluster):
    """The queue behind the running job is a local record, so it stays
    editable while the machine is unreachable."""
    scheduler.tick(force=True)
    ids = _fill(scheduler, "small-node:0", 3)
    scheduler.tick(force=True)
    cluster["small-node"].online = False

    scheduler.lane_move(ids[2], to="head")
    assert _queue_ids(scheduler, "small-node:0") == [ids[2], ids[1]]


def test_move_before_and_after_place_relative_to_another_job(scheduler):
    scheduler.tick(force=True)
    ids = _fill(scheduler, "big-node:0", 4)
    scheduler.tick(force=True)
    queued = _queue_ids(scheduler, "big-node:0")

    scheduler.lane_move(queued[2], before=queued[0])
    assert _queue_ids(scheduler, "big-node:0") == [queued[2], queued[0], queued[1]]

    scheduler.lane_move(queued[2], after=queued[1])
    assert _queue_ids(scheduler, "big-node:0") == [queued[0], queued[1], queued[2]]


def test_move_to_a_numbered_slot_counts_only_queued_jobs(scheduler):
    scheduler.tick(force=True)
    ids = _fill(scheduler, "big-node:0", 4)
    scheduler.tick(force=True)
    queued = _queue_ids(scheduler, "big-node:0")

    scheduler.lane_move(queued[-1], to=2)
    assert _queue_ids(scheduler, "big-node:0")[1] == queued[-1]


def test_a_slot_past_the_end_lands_at_the_end(scheduler):
    scheduler.tick(force=True)
    _fill(scheduler, "big-node:0", 3)
    scheduler.tick(force=True)
    queued = _queue_ids(scheduler, "big-node:0")

    scheduler.lane_move(queued[0], to=99)
    assert _queue_ids(scheduler, "big-node:0")[-1] == queued[0]


def test_swapping_exchanges_two_places(scheduler):
    scheduler.tick(force=True)
    _fill(scheduler, "big-node:0", 4)
    scheduler.tick(force=True)
    queued = _queue_ids(scheduler, "big-node:0")

    scheduler.lane_swap(queued[0], queued[2])
    assert _queue_ids(scheduler, "big-node:0") == [queued[2], queued[1], queued[0]]


def test_a_running_job_cannot_be_reordered(scheduler):
    scheduler.tick(force=True)
    ids = _fill(scheduler, "small-node:0", 2)
    scheduler.tick(force=True)
    with pytest.raises(ValueError, match="running"):
        scheduler.lane_move(ids[0], to="tail")


def test_a_free_job_cannot_be_moved_within_a_lane_it_is_not_in(scheduler):
    scheduler.tick(force=True)
    job_id = scheduler.submit("loose", vram="1G")
    with pytest.raises(ValueError, match="not in a lane"):
        scheduler.lane_move(job_id, to="head")


def test_submitting_at_the_head_jumps_the_queue(scheduler):
    scheduler.tick(force=True)
    _fill(scheduler, "big-node:0", 3)
    scheduler.tick(force=True)
    urgent = scheduler.submit("urgent", lane="big-node:0", vram="1G", at="head")
    assert _queue_ids(scheduler, "big-node:0")[0] == urgent


def test_a_new_submission_defaults_to_the_back(scheduler):
    scheduler.tick(force=True)
    _fill(scheduler, "big-node:0", 3)
    scheduler.tick(force=True)
    last = scheduler.submit("last", lane="big-node:0", vram="1G")
    assert _queue_ids(scheduler, "big-node:0")[-1] == last


def test_a_finished_job_does_not_free_its_slot_for_a_newcomer(scheduler, cluster):
    """New work goes behind what is already waiting, even after the jobs
    ahead of it have come and gone."""
    scheduler.tick(force=True)
    ids = _fill(scheduler, "small-node:0", 2)
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(ids[0], exit_code=0)
    scheduler.tick(force=True)

    newcomer = scheduler.submit("newcomer", lane="small-node:0", vram="1G")
    assert _queue_ids(scheduler, "small-node:0")[-1] == newcomer


# --- moving between lanes --------------------------------------------------

def test_a_job_can_be_moved_to_another_lane(scheduler):
    scheduler.tick(force=True)
    job_id = scheduler.submit("x", lane="big-node:0", vram="1G", at="tail")
    scheduler.lane_assign(job_id, "small-node:0")

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.lane == "small-node:0"
    # The lane names the machine, so pinning follows it rather than being
    # left pointing at the node the job used to be queued on.
    assert job.node_constraint == "small-node"


def test_a_job_can_leave_its_lane_for_the_free_pool(scheduler):
    scheduler.tick(force=True)
    scheduler.submit("head", lane="big-node:0", vram="1G")
    queued = scheduler.submit("queued", lane="big-node:0", vram="1G")
    scheduler.tick(force=True)  # the head starts, `queued` waits behind it

    scheduler.lane_assign(queued, None)
    job = dbm.get_job(scheduler.conn, queued)
    assert job.lane is None
    assert job.lane_seq is None

    # Back in the free pool it is placed by budget, so the other machine takes it.
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, queued).state == "running"


def test_a_lane_refuses_a_job_that_needs_more_gpus_than_it_has(scheduler):
    scheduler.tick(force=True)
    job_id = scheduler.submit("wide", vram="1G", gpus=2)
    with pytest.raises(ValueError, match="needs 2 GPUs"):
        scheduler.lane_assign(job_id, "small-node:0")


def test_a_lane_job_that_cannot_fit_its_own_card_is_failed_not_parked(scheduler):
    """Judged against the lane's card, not the biggest card on the machine."""
    node = FakeNode(
        "mixed", [FakeGpu(0, "small", 8000), FakeGpu(1, "large", 73415)]
    )
    cluster = FakeCluster(node)
    scheduler.cfg = cluster.config()
    scheduler._backend_factory = cluster.backend_factory
    scheduler._configured_nodes = {"mixed"}
    scheduler._last_sync_signature = None
    scheduler.sync_nodes_from_config()
    scheduler.tick(force=True)

    job_id = scheduler.submit("huge", lane="mixed:0", vram="40G")
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "failed"
    assert "reserves" in (job.last_error or "")


# --- coexistence with the free pool ---------------------------------------

def test_jobs_without_a_lane_are_scheduled_exactly_as_before(scheduler, cluster):
    scheduler.tick(force=True)
    free = scheduler.submit("free", vram="1G")
    scheduler.tick(force=True)
    job = dbm.get_job(scheduler.conn, free)
    assert job.state == "running"
    assert job.lane is None


def test_a_lane_does_not_stop_the_free_pool_using_the_other_node(scheduler):
    scheduler.tick(force=True)
    _fill(scheduler, "small-node:0", 2)
    free = scheduler.submit("free", vram="1G", node="big-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, free).state == "running"


def test_priority_moves_do_not_touch_lane_jobs(scheduler):
    scheduler.tick(force=True)
    laned = scheduler.submit("laned", lane="big-node:0", vram="1G")
    free_a = scheduler.submit("a", vram="1G", node="small-node")
    free_b = scheduler.submit("b", vram="1G", node="small-node")

    scheduler.move_pending(free_b, -1)
    assert dbm.get_job(scheduler.conn, laned).lane_seq is not None
    assert dbm.get_job(scheduler.conn, free_b).priority >= dbm.get_job(
        scheduler.conn, free_a
    ).priority


# --- concurrency -----------------------------------------------------------

def test_raising_concurrency_lets_a_lane_run_two_at_once(scheduler):
    scheduler.tick(force=True)
    ids = _fill(scheduler, "big-node:0", 3)
    scheduler.set_lane_concurrency("big-node:0", 2)
    scheduler.tick(force=True)

    states = [dbm.get_job(scheduler.conn, job_id).state for job_id in ids]
    assert states == ["running", "running", "pending"]


def test_concurrency_below_one_is_rejected(scheduler):
    scheduler.tick(force=True)
    with pytest.raises(ValueError):
        scheduler.set_lane_concurrency("big-node:0", 0)


# --- reporting -------------------------------------------------------------

def test_the_lane_view_explains_why_a_lane_is_stopped(scheduler, cluster):
    scheduler.tick(force=True)
    scheduler.submit("x", lane="small-node:0", vram="1G")
    scheduler.set_lane_paused("small-node:0", True)

    view = scheduler.lane_view(dbm.get_lane(scheduler.conn, "small-node:0"))
    assert view["blocked"] == "paused"
    assert [job["id"] for job in view["queued"]]


def test_the_snapshot_carries_the_lanes(scheduler):
    scheduler.tick(force=True)
    names = {lane["name"] for lane in scheduler.snapshot()["lanes"]}
    assert "small-node:0" in names
