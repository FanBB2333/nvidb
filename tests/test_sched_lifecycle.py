"""Lifecycle edges that leave a client waiting on something that never happens.

Every case here was found by driving the scheduler through a situation the
happy path does not cover: a drained node, a cursor that falls behind, a job
nothing can host. They share a failure mode - the queue stays quietly wrong,
and whoever called `nvidb job wait` waits forever.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fake_cluster import FakeCluster, FakeGpu, FakeNode  # noqa: E402

from nvidb.sched import db as dbm  # noqa: E402
from nvidb.sched.scheduler import Scheduler  # noqa: E402

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
        settings=dict(SETTINGS),
        backend_factory=cluster.backend_factory,
        owner="test",
    )
    sched.sync_nodes_from_config()
    yield sched
    sched.close()
    conn.close()


# --- draining ---------------------------------------------------------------

def test_a_drained_node_still_settles_the_jobs_already_on_it(scheduler, cluster):
    """Draining stops dispatch, not observation.

    A node is usually drained so its work can finish; if the queue stopped
    probing it, that work would stay `running` forever and `job wait` with it.
    """
    scheduler.tick(force=True)
    job_id = scheduler.submit("train", name="t", vram="4G", node="small-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"

    dbm.set_node_enabled(scheduler.conn, "small-node", False)
    cluster["small-node"].finish_job(job_id, exit_code=0, result={"ok": True})
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "completed"
    assert job.result == {"ok": True}


def test_a_drained_node_takes_no_new_work(scheduler, cluster):
    scheduler.tick(force=True)
    dbm.set_node_enabled(scheduler.conn, "small-node", False)
    job_id = scheduler.submit("x", vram="4G", node="small-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "pending"

    dbm.set_node_enabled(scheduler.conn, "small-node", True)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


def test_draining_does_not_fake_a_node_coming_back_online(scheduler):
    """`state` means reachability; drain lives in `enabled`."""
    scheduler.tick(force=True)
    dbm.set_node_enabled(scheduler.conn, "small-node", False)
    before = len(dbm.list_events(scheduler.conn, limit=200))
    scheduler.tick(force=True)
    events = dbm.list_events(scheduler.conn, limit=200)[before:]
    assert not [event for event in events if event["kind"] == "node_up"]

    node = dbm.get_node(scheduler.conn, "small-node")
    assert node.enabled is False
    assert node.state == "up"
    assert node.is_schedulable is False


def test_a_drained_node_keeps_reporting_its_capacity(scheduler, cluster):
    scheduler.tick(force=True)
    dbm.set_node_enabled(scheduler.conn, "small-node", False)
    cluster["small-node"].add_foreign_process(0, 9000, name="python")
    scheduler.tick(force=True)

    gpu = dbm.get_node(scheduler.conn, "small-node").gpus[0]
    assert gpu.external_mem_mb == 9000


# --- the event cursor -------------------------------------------------------

def test_events_page_forward_from_a_cursor_without_skipping_any(scheduler):
    """`--since <id>` is how a client catches up on what it missed."""
    for index in range(30):
        dbm.add_event(scheduler.conn, "tick", message=f"event {index}")

    seen = []
    cursor = 0
    while True:
        batch = dbm.list_events(scheduler.conn, since_id=cursor, limit=10)
        if not batch:
            break
        seen.extend(event["message"] for event in batch)
        cursor = batch[-1]["id"]

    assert seen == [f"event {index}" for index in range(30)]


def test_events_without_a_cursor_still_show_the_newest(scheduler):
    for index in range(30):
        dbm.add_event(scheduler.conn, "tick", message=f"event {index}")
    tail = dbm.list_events(scheduler.conn, limit=5)
    assert [event["message"] for event in tail] == [
        f"event {index}" for index in range(25, 30)
    ]


# --- work nothing can host --------------------------------------------------

def test_a_job_no_gpu_could_ever_hold_fails_with_a_reason(scheduler):
    scheduler.tick(force=True)
    job_id = scheduler.submit("huge", name="huge", vram="500G")
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "failed"
    assert "500G" in job.last_error and "71.2G" in job.last_error
    alerts = dbm.list_alerts(scheduler.conn)
    assert [alert["kind"] for alert in alerts] == ["job_unschedulable"]


def test_a_job_wanting_more_gpus_than_exist_fails_with_a_reason(scheduler):
    scheduler.tick(force=True)
    job_id = scheduler.submit("wide", gpus=8, vram="1G")
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "failed"
    assert "needs 8 GPUs" in job.last_error


def test_the_pinned_node_is_what_a_pinned_job_is_judged_against(scheduler):
    """40G fits the big card, so this is only impossible because of `--node`."""
    scheduler.tick(force=True)
    pinned = scheduler.submit("a", vram="40G", node="small-node")
    free = scheduler.submit("b", vram="40G")
    scheduler.tick(force=True)

    assert dbm.get_job(scheduler.conn, pinned).state == "failed"
    assert "small-node" in dbm.get_job(scheduler.conn, pinned).last_error
    assert dbm.get_job(scheduler.conn, free).state == "running"


def test_a_job_that_merely_has_no_room_yet_keeps_waiting(scheduler, cluster):
    """The difference between "does not fit now" and "never fits" is the point."""
    cluster["big-node"].add_foreign_process(0, 69000)
    cluster["small-node"].add_foreign_process(0, 20000)
    scheduler.tick(force=True)

    job_id = scheduler.submit("later", vram="20G")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "pending"

    cluster["big-node"].clear_gpu(0)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


def test_an_unreachable_cluster_never_declares_a_job_impossible(scheduler, cluster):
    """A node that is down reports no GPUs; that is ignorance, not a verdict."""
    scheduler.tick(force=True)
    cluster["big-node"].online = False
    cluster["small-node"].online = False
    scheduler.tick(force=True)

    job_id = scheduler.submit("huge", vram="500G")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "pending"

    # ... and the verdict arrives as soon as the cluster can be seen again.
    cluster["big-node"].online = True
    cluster["small-node"].online = True
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "failed"


def test_cpu_only_work_is_never_called_unschedulable(scheduler):
    scheduler.tick(force=True)
    job_id = scheduler.submit("prep", gpus=0, vram="500G")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


# --- opening the database ---------------------------------------------------

def test_many_clients_can_create_the_queue_at_the_same_moment(tmp_path):
    """A burst of independent sessions is the normal way this queue gets used.

    `PRAGMA journal_mode=WAL` is refused outright rather than waiting on the
    busy timeout, so without a retry the losers of that race fail with
    "database is locked" on their very first command.
    """
    import threading

    path = tmp_path / "contended.db"
    errors = []
    ready = threading.Barrier(12)

    def open_and_write(index):
        try:
            ready.wait(timeout=10)
            conn = dbm.open_db(path)
            try:
                dbm.insert_job(conn, command=f"job {index}")
            finally:
                conn.close()
        except Exception as error:  # noqa: BLE001 - the assertion is the report
            errors.append(f"{type(error).__name__}: {error}")

    threads = [threading.Thread(target=open_and_write, args=(i,)) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    conn = dbm.open_db(path)
    try:
        assert len(dbm.list_jobs(conn)) == 12  # every client's write landed
    finally:
        conn.close()


def test_a_schema_upgrade_racing_another_client_is_not_an_error(tmp_path):
    """Two clients can both find a column missing and both try to add it."""
    path = tmp_path / "upgrade.db"
    first = dbm.open_db(path)
    try:
        first.execute("ALTER TABLE gpus DROP COLUMN processes_json")
    finally:
        first.close()

    # Re-opening runs the upgrade; doing it twice must be harmless either way.
    for _ in range(2):
        conn = dbm.open_db(path)
        try:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(gpus)")}
            assert "processes_json" in columns
        finally:
            conn.close()
