"""Tests for how remote failures reach the local machine.

Covers the two halves separately: the scheduler recording an alert whenever
something goes wrong on a node, and the notifier pushing those alerts out
exactly once.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fake_cluster import FakeCluster, FakeGpu, FakeNode  # noqa: E402

from nvidb.sched import db as dbm  # noqa: E402
from nvidb.sched.notify import Notifier, format_alert, load_notify_settings  # noqa: E402
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


def _open_alerts(scheduler):
    return dbm.list_alerts(scheduler.conn, open_only=True)


def _kinds(scheduler):
    return [alert["kind"] for alert in _open_alerts(scheduler)]


# --- what raises an alert --------------------------------------------------

def test_a_failed_job_raises_an_alert_carrying_its_output(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node", name="train")
    scheduler.tick(force=True)

    node = cluster["small-node"]
    node.logs[node.jobs[job_id]["run_dir"]] = (
        "Traceback (most recent call last):\n"
        "  File \"train.py\", line 42\n"
        "torch.cuda.OutOfMemoryError: CUDA out of memory\n"
    )
    node.finish_job(job_id, exit_code=1)
    scheduler.tick(force=True)

    alerts = _open_alerts(scheduler)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["kind"] == "job_failed"
    assert alert["severity"] == "error"
    assert alert["job_id"] == job_id
    assert alert["node"] == "small-node"
    assert "exit code 1" in alert["title"]
    # The reason travels with the alert, so no second trip is needed to see it.
    assert "OutOfMemoryError" in alert["detail"]

    # The one-line summary is on the job itself.
    job = dbm.get_job(scheduler.conn, job_id)
    assert "CUDA out of memory" in job.last_error


def test_a_successful_job_raises_nothing(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=0)
    scheduler.tick(force=True)
    assert _open_alerts(scheduler) == []


def test_a_lost_job_raises_an_alert(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].vanish_job(job_id)
    scheduler.tick(force=True)

    alert = _open_alerts(scheduler)[0]
    assert alert["kind"] == "job_lost"
    assert alert["job_id"] == job_id


def test_a_retry_is_a_warning_not_an_error(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node", max_retries=1)
    scheduler.tick(force=True)
    cluster["small-node"].vanish_job(job_id)
    scheduler.tick(force=True)

    alert = _open_alerts(scheduler)[0]
    assert alert["kind"] == "job_retried"
    assert alert["severity"] == "warning"
    assert dbm.get_job(scheduler.conn, job_id).state == "running"


def test_a_timeout_raises_an_alert(scheduler, cluster):
    import time

    scheduler.tick(force=True)
    scheduler.submit("train.sh", vram="1G", node="small-node", max_runtime_s=1)
    scheduler.tick(force=True)
    time.sleep(1.2)
    scheduler.tick(force=True)

    alert = _open_alerts(scheduler)[0]
    assert alert["kind"] == "job_timeout"
    assert "limit 1s" in alert["title"]


def test_an_unreachable_node_raises_one_alert_not_one_per_tick(scheduler, cluster):
    scheduler.tick(force=True)
    cluster["small-node"].online = False
    scheduler.tick(force=True)
    scheduler.tick(force=True)
    scheduler.tick(force=True)

    alerts = [a for a in _open_alerts(scheduler) if a["kind"] == "node_down"]
    assert len(alerts) == 1
    assert alerts[0]["node"] == "small-node"


def test_a_node_alert_counts_the_jobs_it_stranded(scheduler, cluster):
    scheduler.tick(force=True)
    scheduler.submit("a", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].online = False
    scheduler.tick(force=True)

    alert = next(a for a in _open_alerts(scheduler) if a["kind"] == "node_down")
    assert "1 running job(s) cannot be checked" in alert["title"]


def test_a_node_coming_back_lets_the_next_outage_alert_again(scheduler, cluster):
    scheduler.tick(force=True)
    cluster["small-node"].online = False
    scheduler.tick(force=True)
    cluster["small-node"].online = True
    scheduler.tick(force=True)
    cluster["small-node"].online = False
    scheduler.tick(force=True)

    assert len([a for a in _open_alerts(scheduler) if a["kind"] == "node_down"]) == 2


def test_a_launch_failure_raises_a_warning(scheduler, cluster):
    scheduler.tick(force=True)
    cluster["small-node"].launch_error = "No space left on device"
    cluster["big-node"].launch_error = "No space left on device"
    scheduler.submit("train.sh", vram="1G")
    scheduler.tick(force=True)

    alert = _open_alerts(scheduler)[0]
    assert alert["kind"] == "launch_failed"
    assert alert["severity"] == "warning"
    assert "No space left" in alert["detail"]


def test_a_dead_dependency_raises_an_alert(scheduler, cluster):
    scheduler.tick(force=True)
    first = scheduler.submit("stage1", vram="1G", node="small-node")
    scheduler.submit("stage2", vram="1G", node="small-node", depends_on=[first])
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(first, exit_code=1)
    scheduler.tick(force=True)

    kinds = _kinds(scheduler)
    assert "job_failed" in kinds
    # The dependent is held for a decision, not failed along with it.
    assert "job_held" in kinds


def test_a_held_job_is_only_announced_once(scheduler, cluster):
    """A hold lasts until someone acts on it, and every tick in between
    must not re-raise it - the queue is polled every few seconds."""
    scheduler.tick(force=True)
    first = scheduler.submit("stage1", vram="1G", node="small-node")
    scheduler.submit("stage2", vram="1G", node="small-node", depends_on=[first])
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(first, exit_code=1)
    for _ in range(4):
        scheduler.tick(force=True)

    assert _kinds(scheduler).count("job_held") == 1


def test_a_cancelled_job_is_not_an_alert(scheduler, cluster):
    """Cancelling is a decision, not a failure."""
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node")
    scheduler.tick(force=True)
    scheduler.cancel(job_id)
    scheduler.tick(force=True)
    assert _open_alerts(scheduler) == []


def test_the_tick_summary_names_what_it_raised(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=2)
    summary = scheduler.tick(force=True)

    assert [item["kind"] for item in summary["alerts"]] == ["job_failed"]


# --- acknowledging ---------------------------------------------------------

def test_alerts_stay_until_acknowledged(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("train.sh", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=1)
    scheduler.tick(force=True)

    assert dbm.open_alert_count(scheduler.conn) == 1
    scheduler.tick(force=True)
    assert dbm.open_alert_count(scheduler.conn) == 1  # not re-raised, not dropped

    alert_id = _open_alerts(scheduler)[0]["id"]
    assert dbm.acknowledge_alerts(scheduler.conn, [alert_id]) == 1
    assert dbm.open_alert_count(scheduler.conn) == 0
    # Acknowledging twice is a no-op rather than an error.
    assert dbm.acknowledge_alerts(scheduler.conn, [alert_id]) == 0
    assert len(dbm.list_alerts(scheduler.conn, open_only=False)) == 1


def test_acknowledging_everything_clears_the_backlog(scheduler, cluster):
    scheduler.tick(force=True)
    for _ in range(3):
        job_id = scheduler.submit("x", vram="1G", node="small-node")
        scheduler.tick(force=True)
        cluster["small-node"].finish_job(job_id, exit_code=1)
        scheduler.tick(force=True)

    assert dbm.open_alert_count(scheduler.conn) == 3
    assert dbm.acknowledge_alerts(scheduler.conn, all_open=True) == 3
    assert dbm.open_alert_count(scheduler.conn) == 0


def test_the_snapshot_carries_open_alerts(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("x", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=1)
    scheduler.tick(force=True)

    snapshot = scheduler.snapshot()
    assert snapshot["open_alerts"] == 1
    assert snapshot["alerts"][0]["kind"] == "job_failed"
    json.dumps(snapshot)  # stays serializable for other tools


# --- delivery --------------------------------------------------------------

class RecordingNotifier:
    """Stands in for the real notifier, recording what it was asked to send."""

    def __init__(self, fail_on=None):
        self.sent = []
        self.fail_on = fail_on

    def deliver(self, alerts):
        handled = []
        for alert in alerts:
            if self.fail_on and alert["id"] == self.fail_on:
                continue
            self.sent.append(alert)
            handled.append(alert["id"])
        return handled


def _raise_failure(scheduler, cluster, exit_code=1):
    job_id = scheduler.submit("x", vram="1G", node="small-node")
    scheduler.tick(force=True)
    cluster["small-node"].finish_job(job_id, exit_code=exit_code)
    scheduler.tick(force=True)
    return job_id


def test_each_alert_is_delivered_once(scheduler, cluster):
    scheduler.tick(force=True)
    _raise_failure(scheduler, cluster)
    notifier = RecordingNotifier()

    assert scheduler.deliver_alerts(notifier) == 1
    assert len(notifier.sent) == 1
    # A second pass has nothing new to say.
    assert scheduler.deliver_alerts(notifier) == 0
    assert len(notifier.sent) == 1


def test_delivery_survives_a_restart_without_replaying(scheduler, cluster, tmp_path):
    scheduler.tick(force=True)
    _raise_failure(scheduler, cluster)
    scheduler.deliver_alerts(RecordingNotifier())

    reopened = dbm.open_db(tmp_path / "queue.db")
    try:
        assert dbm.undelivered_alerts(reopened) == []
    finally:
        reopened.close()


def test_an_undelivered_alert_is_retried_next_pass(scheduler, cluster):
    scheduler.tick(force=True)
    _raise_failure(scheduler, cluster)
    pending = dbm.undelivered_alerts(scheduler.conn)
    stubborn = RecordingNotifier(fail_on=pending[0]["id"])

    assert scheduler.deliver_alerts(stubborn) == 0
    assert dbm.undelivered_alerts(scheduler.conn)  # still queued

    assert scheduler.deliver_alerts(RecordingNotifier()) == 1


def test_acknowledging_does_not_suppress_delivery(scheduler, cluster):
    """Reading an alert in the TUI must not stop the daemon from reporting it."""
    scheduler.tick(force=True)
    _raise_failure(scheduler, cluster)
    dbm.acknowledge_alerts(scheduler.conn, all_open=True)

    notifier = RecordingNotifier()
    assert scheduler.deliver_alerts(notifier) == 1


# --- the notifier itself ---------------------------------------------------

def test_notify_settings_merge_over_the_defaults():
    assert load_notify_settings({})["enabled"] is True
    assert load_notify_settings({"notify": {"desktop": False}})["desktop"] is False
    assert load_notify_settings({"notify": False})["enabled"] is False
    assert load_notify_settings({"notify": {"command": "cat"}})["command"] == "cat"


def test_a_disabled_notifier_still_marks_alerts_handled():
    notifier = Notifier({"notify": {"enabled": False}})
    assert notifier.deliver([{"id": 5, "severity": "error"}]) == [5]


def test_severity_filtering_still_consumes_the_alert():
    """A filtered alert must not be reconsidered on every single pass."""
    notifier = Notifier({"notify": {"min_severity": "error", "desktop": False,
                                    "log": False}})
    warning = {"id": 1, "severity": "warning", "kind": "job_retried", "title": "x"}
    assert notifier.wants(warning) is False
    assert notifier.deliver([warning]) == [1]


def test_the_command_channel_receives_the_alert_as_json(tmp_path):
    target = tmp_path / "received.json"
    notifier = Notifier(
        {
            "notify": {
                "desktop": False,
                "log": False,
                "command": f"cat > {target}",
            }
        }
    )
    alert = {
        "id": 7,
        "kind": "job_failed",
        "severity": "error",
        "job_id": 12,
        "node": "big-node",
        "title": "train failed with exit code 1",
        "detail": "CUDA out of memory",
    }
    assert notifier.deliver([alert]) == [7]
    assert json.loads(target.read_text())["title"] == alert["title"]


def test_a_broken_command_channel_does_not_block_the_others(tmp_path, monkeypatch):
    log_path = tmp_path / "alerts.log"
    monkeypatch.setattr("nvidb.sched.notify.alert_log_path", lambda: log_path)
    notifier = Notifier(
        {"notify": {"desktop": False, "log": True, "command": "exit 1; false"}}
    )
    alert = {"id": 3, "kind": "job_lost", "severity": "error", "title": "gone",
             "job_id": None, "node": "n", "detail": None}

    assert notifier.deliver([alert]) == [3]
    assert "gone" in log_path.read_text()


def test_the_log_channel_appends_one_json_line_per_alert(tmp_path, monkeypatch):
    log_path = tmp_path / "alerts.log"
    monkeypatch.setattr("nvidb.sched.notify.alert_log_path", lambda: log_path)
    notifier = Notifier({"notify": {"desktop": False, "command": None}})
    notifier.deliver(
        [
            {"id": 1, "severity": "error", "kind": "job_failed", "title": "a",
             "job_id": 1, "node": "n", "detail": None},
            {"id": 2, "severity": "error", "kind": "job_lost", "title": "b",
             "job_id": 2, "node": "n", "detail": None},
        ]
    )
    lines = log_path.read_text().strip().splitlines()
    assert [json.loads(line)["title"] for line in lines] == ["a", "b"]


def test_the_alert_summary_names_the_job_and_the_node():
    assert format_alert(
        {"title": "train failed", "job_id": 12, "node": "big-node"}
    ) == "train failed [job 12 on big-node]"
    assert format_alert({"title": "node down", "node": "big-node"}) == (
        "node down [big-node]"
    )
    assert format_alert({"title": "something"}) == "something"
