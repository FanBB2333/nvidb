"""GPU admission across ordinary jobs, staged jobs, and mode changes."""
import sys

import pytest

from fake_cluster import FakeCluster, FakeGpu, FakeNode
from nvidb.sched import db, runner_agent
from nvidb.sched.scheduler import DEFAULT_SETTINGS, Scheduler, load_settings
from nvidb.sched.transport import TransportError


@pytest.fixture
def queue(tmp_path):
    cluster = FakeCluster(FakeNode("box", [FakeGpu(0, "test", 24576), FakeGpu(1, "test", 24576)]))
    conn = db.open_db(tmp_path / "queue.db")
    sched = Scheduler(conn, cfg=cluster.config(), settings=dict(DEFAULT_SETTINGS, tick_min_interval=0),
                      backend_factory=cluster.backend_factory)
    sched.sync_nodes_from_config()
    yield sched, cluster
    sched.close()
    conn.close()


def test_default_is_exclusive_and_invalid_mode_is_rejected():
    assert load_settings({})["gpu_mode"] == "exclusive"
    with pytest.raises(ValueError, match="gpu_mode"):
        load_settings({"queue": {"gpu_mode": "typo"}})


@pytest.mark.parametrize("lane_runner", [None, False, True])
def test_exclusive_gpu_slots_ignore_concurrency_settings(queue, lane_runner):
    sched, cluster = queue
    sched.settings["max_jobs_per_gpu"] = 10
    kwargs = {"gpus": 2} if lane_runner is None else {"lane": "box:0", "vram": "1G"}
    if lane_runner is not None:
        sched.set_lane_runner("box:0", lane_runner)
        sched.set_lane_concurrency("box:0", 4)
    ids = [sched.submit("train", **kwargs) for _ in range(3)]
    sched.tick(force=True)
    assert [db.get_job(sched.conn, i).state for i in ids] == ["running", "pending", "pending"]
    cluster["box"].finish_job(ids[0])
    sched.tick(force=True)
    assert db.get_job(sched.conn, ids[1]).state == "running"


@pytest.mark.parametrize("vram", [None, 0, "0", "0G", ""])
def test_shared_requires_explicit_positive_peak_even_with_default(queue, vram):
    sched, _ = queue
    sched.settings.update(gpu_mode="shared", default_vram="8G")
    with pytest.raises(ValueError, match="explicit positive --vram"):
        sched.submit("train", vram=vram)
    assert sched.submit("prepare", gpus=0, vram=vram)


@pytest.mark.parametrize("runner", [False, True])
def test_cpu_lane_jobs_do_not_receive_or_reserve_gpus(queue, runner):
    sched, _ = queue
    sched.settings["gpu_mode"] = "shared"
    sched.set_lane_runner("box:0", runner)
    job = sched.submit("prepare", gpus=0, lane="box:0")
    sched.tick(force=True)
    assert db.get_job(sched.conn, job).state == "running"
    assert db.get_job(sched.conn, job).gpu_ids == []
    assert sched._gpu_budgets(db.get_node(sched.conn, "box"))[0].jobs == 0
    started = next(e for e in db.list_events(sched.conn, job_id=job) if e["kind"] == "job_started")
    assert started["data"]["gpu_ids"] == []


@pytest.mark.parametrize("runner", [False, True])
def test_shared_lane_and_pool_use_one_budget(queue, runner):
    sched, _ = queue
    sched.settings.update(gpu_mode="shared", max_jobs_per_gpu=2)
    sched.set_lane_runner("box:0,1", runner)
    lane = sched.submit("lane", lane="box:0,1", gpus=2, vram="12G")
    too_big = sched.submit("too-big", gpus=2, vram="12G")
    fits = sched.submit("fits", gpus=2, vram="8G")
    no_slot = sched.submit("no-slot", gpus=2, vram="1G")
    sched.tick(force=True)
    assert [db.get_job(sched.conn, i).state for i in (lane, too_big, fits, no_slot)] == [
        "running", "pending", "running", "pending"]


def test_staged_runner_work_blocks_pool_and_overlapping_lane(queue, monkeypatch):
    sched, cluster = queue
    sched.settings.update(gpu_mode="shared", max_jobs_per_gpu=1)
    # Delay the runner: a reservation must count even before it claims a spec.
    runner = cluster.backend_factory(db.get_node(sched.conn, "box")).runner
    monkeypatch.setattr(runner, "_advance", lambda lane_name: None)
    lane = sched.submit("lane", lane="box:0,1", gpus=2, vram="1G")
    overlap = sched.submit("overlap", lane="box:1", vram="1G")
    pool = sched.submit("pool", gpus=2, vram="1G")
    sched.tick(force=True)
    assert db.get_job(sched.conn, lane).run_dir
    assert db.get_job(sched.conn, overlap).run_dir is None
    assert db.get_job(sched.conn, pool).state == "pending"
    for edit in (lambda: sched.edit_job(lane, vram="2G"), lambda: sched.lane_assign(lane, None)):
        with pytest.raises(ValueError, match="staged"):
            edit()
    sched.cancel(lane)
    sched.tick(force=True)
    assert db.get_job(sched.conn, overlap).run_dir


def test_runner_prefetch_reserves_memory_not_yet_allocated(queue, monkeypatch):
    sched, cluster = queue
    sched.settings.update(gpu_mode="shared", lane_lookahead=4)
    runner = cluster.backend_factory(db.get_node(sched.conn, "box")).runner
    monkeypatch.setattr(runner, "_advance", lambda lane_name: None)
    ids = [sched.submit("train", lane="box:0", vram="12G") for _ in range(2)]
    sched.tick(force=True)
    assert db.get_job(sched.conn, ids[0]).run_dir
    assert db.get_job(sched.conn, ids[1]).run_dir is None


@pytest.mark.parametrize("missing_ack", [False, True])
def test_uncertain_runner_sync_prevents_reusing_node_capacity(queue, monkeypatch, missing_ack):
    sched, cluster = queue
    sched.settings["gpu_mode"] = "shared"
    backend = cluster.backend_factory(db.get_node(sched.conn, "box"))
    sync = backend.lane_sync
    def uncertain(**kwargs):
        if missing_ack:
            return None
        raise TransportError("reply lost after staging")
    monkeypatch.setattr(backend, "lane_sync", uncertain)
    lane = sched.submit("lane", lane="box:0", vram="1G")
    pool = sched.submit("pool", gpus=2, vram="1G")
    result = sched.tick(force=True)
    assert result["nodes_down"] == 1
    assert db.get_job(sched.conn, pool).state == "pending"
    assert db.get_job(sched.conn, lane).run_dir
    monkeypatch.setattr(backend, "lane_sync", sync)
    sched.tick(force=True)
    assert db.get_job(sched.conn, lane).state == "running"
    assert db.get_job(sched.conn, pool).state == "running"


def test_switch_to_shared_preserves_running_exclusive_allocation(queue):
    sched, cluster = queue
    first = sched.submit("exclusive", gpus=2)
    sched.tick(force=True)
    sched.settings["gpu_mode"] = "shared"
    second = sched.submit("shared", gpus=2, vram="1G")
    sched.tick(force=True)
    assert db.get_job(sched.conn, second).state == "pending"
    cluster["box"].finish_job(first)
    sched.tick(force=True)
    assert db.get_job(sched.conn, second).state == "running"
    assert db.get_job(sched.conn, second).allocation_mode == "shared"


def test_switch_to_shared_preserves_staged_exclusive_without_estimate(queue, monkeypatch):
    sched, cluster = queue
    runner = cluster.backend_factory(db.get_node(sched.conn, "box")).runner
    monkeypatch.setattr(runner, "_advance", lambda lane_name: None)
    job = sched.submit("exclusive", lane="box:0,1", gpus=2)
    sched.tick(force=True)
    assert db.get_job(sched.conn, job).allocation_mode == "exclusive"
    with pytest.raises(ValueError, match="staged jobs"):
        sched.set_lane_runner("box:0,1", False)
    sched.settings["gpu_mode"] = "shared"
    pool = sched.submit("shared", gpus=2, vram="1G")
    sched.tick(force=True)
    assert db.get_job(sched.conn, pool).state == "pending"
    assert db.get_job(sched.conn, job).held_reason is None
    assert runner.lanes["box:0,1"]["queued"][0]["gpu_mode"] == "exclusive"


def test_switch_to_exclusive_waits_for_all_shared_jobs(queue):
    sched, cluster = queue
    sched.settings["gpu_mode"] = "shared"
    ids = [sched.submit("shared", gpus=2, vram="1G") for _ in range(2)]
    sched.tick(force=True)
    sched.settings["gpu_mode"] = "exclusive"
    exclusive = sched.submit("exclusive", gpus=2)
    cluster["box"].finish_job(ids[0])
    sched.tick(force=True)
    assert db.get_job(sched.conn, exclusive).state == "pending"
    cluster["box"].finish_job(ids[1])
    sched.tick(force=True)
    assert db.get_job(sched.conn, exclusive).state == "running"


def test_old_zero_estimate_is_held_and_can_be_corrected(queue):
    sched, _ = queue
    job = sched.submit("old", gpus=2)
    sched.settings["gpu_mode"] = "shared"
    sched.tick(force=True)
    assert "--vram" in db.get_job(sched.conn, job).held_reason
    with pytest.raises(ValueError, match="positive"):
        sched.edit_job(job, vram=0)
    sched.edit_job(job, vram="4G")
    sched.tick(force=True)
    assert db.get_job(sched.conn, job).state == "running"


@pytest.mark.parametrize("free,needed,starts", [(None, 1024, False), (1400, 1024, False),
                                               (1536, 1024, True), (24000, 0, False)])
def test_real_runner_requires_known_memory_headroom_and_positive_estimate(tmp_path, monkeypatch, free, needed, starts):
    monkeypatch.setattr(sys, "argv", ["runner", str(tmp_path), "box:0", "marker", "test", "1", "1"])
    ns = {"__name__": "test_runner"}
    exec(runner_agent.RUNNER_SCRIPT, ns)
    ns["read_json"] = lambda *args: {"gpu_mode": "shared", "concurrency": 1, "headroom_mb": 512}
    ns["gpu_free_mb"] = lambda indices: free
    instance = ns["Runner"]()
    spec = {"gpu_ids": [0], "gpu_mode": "shared", "vram_mb": needed}
    instance.pending_specs = lambda: [("job.json", spec)]
    class Job:
        def start(self):
            pass
    instance.claim_next = lambda: Job()
    instance._maybe_start_locked()
    assert bool(instance.current) is starts
