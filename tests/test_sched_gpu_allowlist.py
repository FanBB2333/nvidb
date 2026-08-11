"""A server entry's `gpus:` key restricts which cards the queue may hand out."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fake_cluster import FakeCluster, FakeGpu, FakeNode  # noqa: E402

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


def test_gpu_allowlists_reads_only_entries_with_the_key():
    cfg = {
        "servers": [
            {"nickname": "masked", "gpus": [0, 1]},
            {"nickname": "open"},
        ]
    }
    assert gpu_allowlists(cfg) == {"masked": {0, 1}}
    assert gpu_allowlists(None) == {}


@pytest.fixture
def masked_scheduler(tmp_path):
    # GPU2 is the emptiest card, so unrestricted spread placement would pick
    # it; the allowlist must keep the job on GPU0/1 anyway.
    node = FakeNode(
        "shared-box",
        [
            FakeGpu(0, "A100", 40000),
            FakeGpu(1, "A100", 40000),
            FakeGpu(2, "A100", 81920),
        ],
    )
    cluster = FakeCluster(node)
    cfg = cluster.config()
    cfg["servers"][0]["gpus"] = [0, 1]
    conn = dbm.open_db(tmp_path / "queue.db")
    sched = Scheduler(
        conn,
        cfg=cfg,
        settings=dict(SETTINGS),
        backend_factory=cluster.backend_factory,
        owner="test",
    )
    sched.sync_nodes_from_config()
    yield sched
    sched.close()
    conn.close()


def test_dispatch_never_uses_an_excluded_gpu(masked_scheduler):
    job_id = masked_scheduler.submit("train", vram="1G")
    masked_scheduler.tick(force=True)
    job = dbm.get_job(masked_scheduler.conn, job_id)
    assert job.state == "running"
    assert job.gpu_ids and set(job.gpu_ids) <= {0, 1}
