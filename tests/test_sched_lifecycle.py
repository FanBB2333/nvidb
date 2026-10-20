"""Lifecycle edges that leave a client waiting on something that never happens.

Every case here was found by driving the scheduler through a situation the
happy path does not cover: a drained node, a cursor that falls behind, a job
nothing can host. They share a failure mode - the queue stays quietly wrong,
and whoever called `nvidb job wait` waits forever.
"""
import time

import pytest

from fake_cluster import (
    SCHEDULER_SETTINGS,
    FakeCluster,
    FakeGpu,
    FakeNode,
    make_scheduler,
    two_node_cluster,
)

from nvidb.sched import db as dbm
from nvidb.sched.scheduler import Scheduler

SETTINGS = {**SCHEDULER_SETTINGS, "max_cpu_jobs_per_node": 4}


@pytest.fixture
def cluster():
    return two_node_cluster()


@pytest.fixture
def scheduler(tmp_path, cluster):
    conn = dbm.open_db(tmp_path / "queue.db")
    sched = make_scheduler(conn, cluster, **SETTINGS)
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


def test_a_removed_config_node_only_finishes_existing_work(scheduler, cluster):
    scheduler.tick(force=True)
    running_id = scheduler.submit("existing", vram="1G", node="small-node")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, running_id).state == "running"

    scheduler.cfg["servers"] = [
        server
        for server in scheduler.cfg["servers"]
        if server["nickname"] != "small-node"
    ]
    scheduler.tick(force=True)

    with pytest.raises(ValueError, match="no longer configured"):
        scheduler.submit("new-pinned", vram="1G", node="small-node")
    # The removed node remains in SQLite while its detached job is observed,
    # but ordinary snapshots mirror the current configured server list just as
    # the monitor TUI does.
    assert [node["name"] for node in scheduler.snapshot()["nodes"]] == [
        "big-node"
    ]
    assert any(
        job["id"] == running_id and job["node"] == "small-node"
        for job in scheduler.snapshot()["jobs"]
    )
    assert "small-node" in {
        node["name"]
        for node in scheduler.snapshot(include_ignored=True)["nodes"]
    }
    new_id = scheduler.submit("new-unpinned", vram="1G")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, new_id).node == "big-node"

    cluster["small-node"].finish_job(running_id)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, running_id).state == "completed"


# --- ignoring ---------------------------------------------------------------

def test_an_ignored_node_is_not_probed_scheduled_or_shown(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train", vram="4G", node="small-node")
    dbm.set_node_ignored(scheduler.conn, "small-node", True)
    cluster["small-node"].online = False

    summary = scheduler.tick(force=True)

    assert summary["nodes_ignored"] == 1
    assert summary["nodes_down"] == 0
    assert dbm.get_node(scheduler.conn, "small-node").state == "up"
    assert dbm.get_job(scheduler.conn, job_id).state == "pending"
    assert {node["name"] for node in scheduler.snapshot()["nodes"]} == {"big-node"}

    included = scheduler.snapshot(include_ignored=True)["nodes"]
    hidden = next(node for node in included if node["name"] == "small-node")
    assert hidden["ignored"] is True

    cluster["small-node"].online = True
    dbm.set_node_ignored(scheduler.conn, "small-node", False)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


def test_submit_refuses_an_explicitly_ignored_node(scheduler):
    dbm.set_node_ignored(scheduler.conn, "small-node", True)
    with pytest.raises(ValueError, match="is ignored"):
        scheduler.submit("train", vram="4G", node="small")


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


def test_a_multi_gpu_request_must_fit_on_the_same_node(tmp_path):
    cluster = FakeCluster(
        FakeNode(
            "mixed-node",
            [FakeGpu(0, "A100", 81920), FakeGpu(1, "small", 8192)],
        )
    )
    conn = dbm.open_db(tmp_path / "queue.db")
    scheduler = Scheduler(
        conn,
        cfg=cluster.config(),
        settings=dict(SETTINGS),
        backend_factory=cluster.backend_factory,
        owner="test",
    )
    try:
        scheduler.sync_nodes_from_config()
        scheduler.tick(force=True)
        job_id = scheduler.submit("wide", gpus=2, vram="40G")
        scheduler.tick(force=True)

        job = dbm.get_job(conn, job_id)
        assert job.state == "failed"
        assert "at most 1 matching GPUs" in job.last_error
    finally:
        scheduler.close()
        conn.close()


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


# --- processes left behind by a closed job ----------------------------------

def test_a_job_cancelled_during_an_outage_is_killed_when_the_node_returns(
    scheduler, cluster
):
    """Cancelling cannot reach an offline node, but the process is still there.

    Without this the GPU stays occupied for the length of the outage by work
    the user already told the queue to stop, and nothing is left to clean up.
    """
    scheduler.tick(force=True)
    job_id = scheduler.submit("train", vram="4G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].job_allocates(job_id, 0, 4000)

    cluster["small-node"].online = False
    assert scheduler.cancel(job_id) is True
    assert dbm.get_job(scheduler.conn, job_id).state == "cancelled"
    # The record is final, but the process is not: it is remembered instead.
    assert cluster["small-node"].jobs[job_id]["alive"] is True
    assert len(dbm.list_orphans(scheduler.conn)) == 1

    cluster["small-node"].online = True
    summary = scheduler.tick(force=True)

    assert cluster["small-node"].reaped == [job_id]
    assert cluster["small-node"].jobs[job_id]["alive"] is False
    assert dbm.list_orphans(scheduler.conn) == []
    assert summary["reaped"] == [
        {"job": job_id, "node": "small-node", "pid": cluster["small-node"].jobs[job_id]["pid"]}
    ]
    kinds = [event["kind"] for event in dbm.list_events(scheduler.conn, limit=100)]
    assert "job_reaped" in kinds

    # NVML was sampled before the kill, so the card reads free on the next pass.
    scheduler.tick(force=True)
    assert dbm.get_node(scheduler.conn, "small-node").gpus[0].external_mem_mb == 0


def test_an_unreachable_timeout_waits_for_evidence(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("slow", vram="1G", node="small-node", max_runtime_s=1)
    scheduler.tick(force=True)
    time.sleep(1.2)

    cluster["small-node"].online = False
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"
    assert cluster["small-node"].jobs[job_id]["alive"] is True

    cluster["small-node"].online = True
    scheduler.tick(force=True)
    assert cluster["small-node"].killed == [job_id]
    assert dbm.get_job(scheduler.conn, job_id).state == "timeout"


def test_a_process_that_died_on_its_own_is_simply_forgotten(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("x", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].online = False
    scheduler.cancel(job_id)

    # The machine rebooted, taking the process with it.
    cluster["small-node"].vanish_job(job_id)
    cluster["small-node"].online = True
    summary = scheduler.tick(force=True)

    assert dbm.list_orphans(scheduler.conn) == []
    assert cluster["small-node"].reaped == []
    assert summary.get("reaped") is None  # nothing to report


def test_the_cleanup_survives_purging_the_job_record(scheduler, cluster):
    """`job purge` deletes the row; the process it started still has to go."""
    scheduler.tick(force=True)
    job_id = scheduler.submit("x", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].online = False
    scheduler.cancel(job_id)
    assert dbm.purge_jobs(scheduler.conn) == 1
    assert dbm.get_job(scheduler.conn, job_id) is None

    cluster["small-node"].online = True
    scheduler.tick(force=True)
    assert cluster["small-node"].reaped == [job_id]


def test_an_unreachable_node_keeps_its_cleanup_queued(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("x", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].online = False
    scheduler.cancel(job_id)

    for _ in range(3):
        scheduler.tick(force=True)
    assert len(dbm.list_orphans(scheduler.conn)) == 1  # not dropped, not retried away
    assert scheduler.snapshot()["pending_reaps"] == 1


# --- CPU-only work ----------------------------------------------------------

def test_cpu_only_jobs_are_capped_per_node(scheduler):
    """They reserve no VRAM, so without a count nothing bounds them at all."""
    scheduler.tick(force=True)
    ids = [scheduler.submit(f"cpu{index}", gpus=0) for index in range(12)]
    scheduler.tick(force=True)

    states = [dbm.get_job(scheduler.conn, job_id).state for job_id in ids]
    # Two nodes, four slots each.
    assert states.count("running") == 8
    assert states.count("pending") == 4

    per_node: dict = {}
    for job in dbm.live_jobs(scheduler.conn):
        per_node[job.node] = per_node.get(job.node, 0) + 1
    assert sorted(per_node.values()) == [4, 4]


def test_a_finished_cpu_job_frees_its_slot(scheduler, cluster):
    scheduler.tick(force=True)
    ids = [scheduler.submit(f"cpu{index}", gpus=0, node="small-node") for index in range(6)]
    scheduler.tick(force=True)
    assert [dbm.get_job(scheduler.conn, i).state for i in ids].count("running") == 4

    for job_id in ids[:2]:
        cluster["small-node"].finish_job(job_id, exit_code=0)
    scheduler.tick(force=True)

    states = [dbm.get_job(scheduler.conn, job_id).state for job_id in ids]
    assert states.count("completed") == 2
    assert states.count("running") == 4  # the waiting two moved up


def test_the_cpu_cap_can_be_switched_off(tmp_path, cluster):
    conn = dbm.open_db(tmp_path / "uncapped.db")
    settings = dict(SETTINGS)
    settings["max_cpu_jobs_per_node"] = 0
    sched = Scheduler(
        conn, cfg=cluster.config(), settings=settings,
        backend_factory=cluster.backend_factory, owner="test",
    )
    try:
        sched.sync_nodes_from_config()
        sched.tick(force=True)
        ids = [sched.submit(f"cpu{index}", gpus=0) for index in range(10)]
        sched.tick(force=True)
        states = [dbm.get_job(conn, job_id).state for job_id in ids]
        assert states.count("running") == 10
    finally:
        sched.close()
        conn.close()


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
        first.execute("ALTER TABLE nodes DROP COLUMN ignored")
    finally:
        first.close()

    # Re-opening runs the upgrade; doing it twice must be harmless either way.
    for _ in range(2):
        conn = dbm.open_db(path)
        try:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(gpus)")}
            assert "processes_json" in columns
            node_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(nodes)")
            }
            assert "ignored" in node_columns
        finally:
            conn.close()


def _overrun(scheduler, job_id):
    from datetime import datetime, timedelta, timezone
    dbm.update_job(scheduler.conn, job_id,
                   started_at=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())


def test_offline_completion_never_becomes_timeout_or_holds_dependents(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("work", node="small-node", max_runtime_s=60)
    child = scheduler.submit("next", node="small-node", depends_on=[job_id])
    scheduler.tick(force=True)
    _overrun(scheduler, job_id)
    cluster["small-node"].online = False
    cluster["small-node"].finish_job(job_id, result={"accuracy": .9})
    scheduler.tick(force=True)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"
    assert not dbm.get_job(scheduler.conn, child).held_reason
    events = dbm.list_events(scheduler.conn, limit=100)
    assert sum(e["kind"] == "job_overdue" for e in events) == 1
    assert not any(e["kind"] == "job_timeout" for e in events)
    assert not dbm.list_orphans(scheduler.conn)
    cluster["small-node"].online = True
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "completed"
    assert dbm.get_job(scheduler.conn, job_id).result == {"accuracy": .9}
    assert dbm.get_job(scheduler.conn, child).state == "running"
    assert not cluster["small-node"].killed


def test_an_overdue_job_found_vanished_closes_its_overdue_alert(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("work", node="small-node", max_runtime_s=60)
    scheduler.tick(force=True)
    _overrun(scheduler, job_id)
    cluster["small-node"].online = False
    scheduler.tick(force=True)
    open_kinds = [a["kind"] for a in dbm.list_alerts(scheduler.conn, open_only=True)]
    assert "job_overdue" in open_kinds

    cluster["small-node"].vanish_job(job_id)
    cluster["small-node"].online = True
    scheduler.tick(force=True)

    assert dbm.get_job(scheduler.conn, job_id).state == "lost"
    open_kinds = [a["kind"] for a in dbm.list_alerts(scheduler.conn, open_only=True)]
    assert "job_overdue" not in open_kinds


@pytest.mark.parametrize("signal_delivered", [False, True])
def test_timeout_transport_failure_preserves_running_until_reconciled(scheduler, cluster, monkeypatch, signal_delivered):
    from nvidb.sched.transport import TransportError
    scheduler.tick(force=True)
    job_id = scheduler.submit("work", node="small-node", max_runtime_s=60)
    scheduler.tick(force=True)
    _overrun(scheduler, job_id)
    executor = cluster.backends["small-node"].executor
    original = executor.request_timeout

    def interrupted(**kwargs):
        if signal_delivered:
            original(**kwargs)
        raise TransportError("connection lost during termination")

    monkeypatch.setattr(executor, "request_timeout", interrupted)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"
    assert not dbm.list_orphans(scheduler.conn)
    monkeypatch.setattr(executor, "request_timeout", original)
    if not signal_delivered:
        cluster["small-node"].finish_job(job_id)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == ("timeout" if signal_delivered else "completed")


def test_previous_tick_observation_cannot_authorize_a_timeout(scheduler, cluster, monkeypatch):
    from nvidb.sched.executor import NodeProbe
    scheduler.tick(force=True)
    job_id = scheduler.submit("work", node="small-node", max_runtime_s=60)
    scheduler.tick(force=True)
    _overrun(scheduler, job_id)
    executor = cluster.backends["small-node"].executor
    monkeypatch.setattr(executor, "probe", lambda *a, **k: NodeProbe())
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"
    assert not cluster["small-node"].killed


@pytest.mark.parametrize("new_attempt,cancelled", [(False, False), (True, False), (False, True)])
def test_old_timeout_correction_is_scoped_to_its_attempt(scheduler, cluster, new_attempt, cancelled):
    from nvidb.sched.model import utcnow
    scheduler.tick(force=True)
    job_id = scheduler.submit("old work", node="small-node")
    scheduler.tick(force=True)
    old = dbm.get_job(scheduler.conn, job_id)
    scheduler._remember_orphan(old, reason="cancelled while unreachable" if cancelled else "timed out while unreachable")
    dbm.update_job(scheduler.conn, job_id, state="cancelled" if cancelled else "timeout",
                   finished_at=utcnow())
    cluster["small-node"].finish_job(job_id, result={"old": True})
    cluster["small-node"].jobs[job_id]["finished_epoch"] = int(time.time()) - 10
    if new_attempt:
        scheduler.requeue(job_id)
        scheduler.tick(force=True)
        current = dbm.get_job(scheduler.conn, job_id)
        assert current.attempt == old.attempt + 1
        assert current.state == "running"
    else:
        scheduler.tick(force=True)
        current = dbm.get_job(scheduler.conn, job_id)
        assert current.state == ("cancelled" if cancelled else "completed")
        if not cancelled:
            assert current.result == {"old": True}
    assert not dbm.list_orphans(scheduler.conn)


def test_requeue_unholds_jobs_beyond_the_lane_window_without_dropping_dependencies(scheduler):
    from nvidb.sched.model import utcnow
    scheduler.tick(force=True)
    dep = scheduler.submit("dependency", node="big-node")
    dbm.update_job(scheduler.conn, dep, state="failed", finished_at=utcnow())
    for i in range(3):
        scheduler.submit(f"earlier {i}", lane="small-node:0")
    child = scheduler.submit("waiting", lane="small-node:0", depends_on=[dep])
    job = dbm.get_job(scheduler.conn, child)
    scheduler._hold(job, f"dependency {dep} ended as failed", {})
    original_seq = job.lane_seq
    scheduler.requeue(dep)
    job = dbm.get_job(scheduler.conn, child)
    assert job.held_reason is None
    assert job.depends_on == [dep]
    assert job.lane_seq == original_seq
    assert all(a["acknowledged_at"] for a in dbm.list_alerts(scheduler.conn)
               if a["job_id"] == child and a["kind"] == "job_held")
    # Correction/recovery can also happen outside requeue().
    scheduler._hold(job, "stale dependency failure", {})
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, child).held_reason is None


def test_legacy_runner_timeout_is_not_corrected_to_failure(scheduler, cluster):
    from nvidb.sched.model import utcnow
    scheduler.tick(force=True)
    job_id = scheduler.submit("old runner", node="small-node", max_runtime_s=60)
    scheduler.tick(force=True)
    job = dbm.get_job(scheduler.conn, job_id)
    scheduler._remember_orphan(job, reason="timed out while unreachable")
    dbm.update_job(scheduler.conn, job_id, state="timeout", finished_at=utcnow())
    cluster["small-node"].finish_job(job_id, exit_code=143)
    cluster["small-node"].jobs[job_id]["finished_epoch"] = int(time.time()) - 10
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "timeout"
    assert not any(e["kind"] == "job_corrected" for e in dbm.list_events(scheduler.conn))
