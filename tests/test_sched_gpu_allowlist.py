"""A server entry's `gpus:` key restricts which cards the queue may hand out."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fake_cluster import FakeCluster, FakeGpu, FakeNode  # noqa: E402

from nvidb import config  # noqa: E402
from nvidb.data_modules import ServerListInfo  # noqa: E402
from nvidb.sched import db as dbm  # noqa: E402
from nvidb.sched.scheduler import Scheduler, gpu_allowlists  # noqa: E402

SETTINGS = {
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
}


def _shared_cluster():
    return FakeCluster(
        FakeNode(
            "shared-box",
            [
                FakeGpu(0, "A100", 40000),
                FakeGpu(1, "A100", 40000),
                FakeGpu(2, "A100", 81920),
            ],
        )
    )


def _scheduler(conn, cluster, gpu_indices=None):
    cfg = cluster.config()
    if gpu_indices is not None:
        cfg["servers"][0]["gpus"] = gpu_indices
    return Scheduler(
        conn,
        cfg=cfg,
        settings=dict(SETTINGS),
        backend_factory=cluster.backend_factory,
        owner="test",
    )


def test_gpu_allowlists_reads_only_entries_with_the_key():
    cfg = {
        "servers": [
            {"nickname": "masked", "gpus": [0, 1, 1]},
            {"nickname": "open"},
        ]
    }
    assert gpu_allowlists(cfg) == {"masked": {0, 1}}
    assert gpu_allowlists(None) == {}


@pytest.mark.parametrize("value", ["0,1", 0, [True], [-1], [0, 1.5]])
def test_gpu_allowlists_reject_invalid_values(value):
    cfg = {"servers": [{"nickname": "masked", "gpus": value}]}
    with pytest.raises(ValueError, match="server 'masked' gpus"):
        gpu_allowlists(cfg)


def test_monitor_config_accepts_and_preserves_the_allowlist():
    raw = [
        {
            "hostname": "gpu.example.com",
            "port": 22,
            "username": "alice",
            "nickname": "shared-box",
            "gpus": [0, 1],
        }
    ]
    servers = ServerListInfo.from_dict(raw)

    assert servers[0].gpus == [0, 1]
    assert servers.to_dict()[0]["gpus"] == [0, 1]
    assert "    gpus: [0, 1]" in config.format_servers_yaml(servers.to_dict())


@pytest.fixture
def masked_scheduler(tmp_path):
    # GPU2 is the emptiest card, so unrestricted spread placement would pick
    # it; the allowlist must keep the job on GPU0/1 anyway.
    cluster = _shared_cluster()
    conn = dbm.open_db(tmp_path / "queue.db")
    sched = _scheduler(conn, cluster, [0, 1])
    sched.sync_nodes_from_config()
    yield sched, cluster
    sched.close()
    conn.close()


def test_dispatch_never_uses_an_excluded_gpu(masked_scheduler):
    scheduler, _cluster = masked_scheduler
    job_id = scheduler.submit("train", vram="1G")
    scheduler.tick(force=True)
    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "running"
    assert job.gpu_ids and set(job.gpu_ids) <= {0, 1}


def test_monitor_inventory_still_contains_excluded_gpus(masked_scheduler):
    scheduler, _cluster = masked_scheduler
    scheduler.tick(force=True)
    node = dbm.get_node(scheduler.conn, "shared-box")
    assert [gpu.index for gpu in node.gpus] == [0, 1, 2]


def test_jobs_that_cannot_fit_the_allowlist_fail_with_a_reason(masked_scheduler):
    scheduler, _cluster = masked_scheduler
    too_wide = scheduler.submit("wide", gpus=3, vram="1G")
    too_large = scheduler.submit("large", gpus=1, vram="60G")

    scheduler.tick(force=True)

    assert dbm.get_job(scheduler.conn, too_wide).state == "failed"
    assert "2 schedulable GPUs" in dbm.get_job(
        scheduler.conn, too_wide
    ).last_error
    assert dbm.get_job(scheduler.conn, too_large).state == "failed"
    assert "largest schedulable GPU" in dbm.get_job(
        scheduler.conn, too_large
    ).last_error


def test_existing_job_on_an_excluded_gpu_runs_to_completion(tmp_path):
    cluster = _shared_cluster()
    conn = dbm.open_db(tmp_path / "queue.db")
    unrestricted = _scheduler(conn, cluster)
    try:
        unrestricted.sync_nodes_from_config()
        job_id = unrestricted.submit("existing", vram="1G")
        unrestricted.tick(force=True)
        assert dbm.get_job(conn, job_id).gpu_ids == [2]
    finally:
        unrestricted.close()

    restricted = _scheduler(conn, cluster, [0, 1])
    try:
        restricted.tick(force=True)
        job = dbm.get_job(conn, job_id)
        assert job.state == "running"
        assert job.gpu_ids == [2]
        monitored = dbm.get_node(conn, "shared-box").gpus
        assert [gpu.index for gpu in monitored] == [0, 1, 2]
        assert next(gpu for gpu in monitored if gpu.index == 2).queue_jobs == 1

        new_job_id = restricted.submit("new", vram="1G")
        restricted.tick(force=True)
        new_job = dbm.get_job(conn, new_job_id)
        assert new_job.state == "running"
        assert new_job.gpu_ids and set(new_job.gpu_ids) <= {0, 1}
        assert dbm.get_job(conn, job_id).gpu_ids == [2]

        cluster["shared-box"].finish_job(job_id)
        restricted.tick(force=True)
        assert dbm.get_job(conn, job_id).state == "completed"
    finally:
        restricted.close()
        conn.close()


def test_daemon_style_scheduler_reloads_gpu_allowlists(tmp_path, monkeypatch):
    cluster = _shared_cluster()
    current = {"config": cluster.config()}
    monkeypatch.setattr(
        "nvidb.sched.scheduler.nvidb_config.load_queue_config",
        lambda: current["config"],
    )
    conn = dbm.open_db(tmp_path / "queue.db")
    scheduler = Scheduler(
        conn,
        settings=dict(SETTINGS),
        backend_factory=cluster.backend_factory,
        owner="test",
    )
    try:
        scheduler.sync_nodes_from_config()
        existing = scheduler.submit("existing", vram="1G")
        scheduler.tick(force=True)
        assert dbm.get_job(conn, existing).gpu_ids == [2]

        restricted = cluster.config()
        restricted["servers"][0]["gpus"] = [0, 1]
        current["config"] = restricted
        new = scheduler.submit("new", vram="1G")
        scheduler.tick(force=True)

        assert dbm.get_job(conn, existing).gpu_ids == [2]
        assert set(dbm.get_job(conn, new).gpu_ids) <= {0, 1}
    finally:
        scheduler.close()
        conn.close()


def test_empty_allowlist_still_permits_cpu_only_jobs(tmp_path):
    cluster = _shared_cluster()
    conn = dbm.open_db(tmp_path / "queue.db")
    scheduler = _scheduler(conn, cluster, [])
    try:
        scheduler.sync_nodes_from_config()
        gpu_job = scheduler.submit("gpu", vram="1G")
        cpu_job = scheduler.submit("cpu", gpus=0, vram=0)
        scheduler.tick(force=True)

        assert dbm.get_job(conn, gpu_job).state == "failed"
        assert dbm.get_job(conn, cpu_job).state == "running"
    finally:
        scheduler.close()
        conn.close()
