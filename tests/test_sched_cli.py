"""Tests for the queue CLI's argument parsing and output.

These run the real parser and handlers against a temporary database with node
probing disabled, so they cover the command surface without touching a network.
"""
import json
import logging
import os
import socket

import pytest

from nvidb.sched import cli as sched_cli
from nvidb.sched import db as dbm
from nvidb.sched.scheduler import TICK_LOCK, UNSTAGED_CANCEL_OWNER, Scheduler
from nvidb.tui_theme import display_width


@pytest.fixture
def parser():
    root = sched_cli.ArgumentParser(prog="nvidb")
    subparsers = root.add_subparsers(dest="command")
    sched_cli.register_parsers(subparsers)
    return root


@pytest.fixture
def queue_db(tmp_path, monkeypatch):
    """An isolated queue database with no servers configured."""
    path = tmp_path / "queue.db"
    monkeypatch.setenv("NVIDB_QUEUE_DB", str(path))
    monkeypatch.setattr("nvidb.config.load_config", lambda *a, **k: {"servers": []})
    return path


def _run(parser, argv, queue_db):
    # The database comes from NVIDB_QUEUE_DB rather than --db-path, which would
    # land inside the job command whenever the argv contains `--`.
    # No servers are configured, so the scheduler passes these commands trigger
    # find nothing to do and never reach the network.
    return sched_cli.dispatch(parser.parse_args(argv))


def _submit(parser, queue_db, *extra):
    assert _run(parser, ["job", "submit", *extra], queue_db) == 0


# --- parsing -------------------------------------------------------------

def test_a_subcommand_positional_does_not_shadow_the_top_level_one(parser):
    """`submit` takes a positional command; it must not overwrite `command`."""
    args = parser.parse_args(["job", "submit", "--", "python", "train.py"])
    assert args.command == "job"
    assert args.job_command == "submit"
    assert args.command_parts == ["python", "train.py"]
    assert args.queue_cli is True


def test_the_queue_commands_are_all_recognised(parser):
    for argv in (
        ["queue"],
        ["queue", "status"],
        ["queue", "tick"],
        ["queue", "backup"],
        ["queue", "nodes"],
        ["queue", "events"],
        ["queue", "drain", "n1"],
        ["queue", "ignore", "n1"],
        ["queue", "unignore", "n1"],
        ["job", "ls"],
        ["job", "show", "1"],
        ["job", "wait", "1"],
        ["job", "note", "1"],
    ):
        args = parser.parse_args(argv)
        assert args.queue_cli is True
        assert callable(args.func)


def test_q_is_a_full_alias_for_the_queue_command_tree(parser):
    for suffix in ([], ["status"], ["tick"], ["lane", "box:0"]):
        canonical = parser.parse_args(["queue", *suffix])
        alias = parser.parse_args(["q", *suffix])
        assert alias.queue_cli is True
        assert alias.func is canonical.func


def test_paramiko_internal_errors_are_quiet_in_machine_readable_commands():
    loggers = [
        logging.getLogger(name)
        for name in ("paramiko", "paramiko.transport", "paramiko.transport.sftp")
    ]
    previous = [logger.level for logger in loggers]
    try:
        sched_cli.quiet_transport_logging()
        assert all(not logger.isEnabledFor(logging.ERROR) for logger in loggers)
    finally:
        for logger, level in zip(loggers, previous):
            logger.setLevel(level)


def test_note_accepts_each_of_its_forms_without_ambiguity(parser):
    read = parser.parse_args(["job", "note", "7"])
    assert read.text == [] and read.append is None and read.clear is False

    replace = parser.parse_args(["job", "note", "7", "new text"])
    assert replace.text == ["new text"]

    # A bare flag followed by a variable positional is what argparse gets
    # wrong, so --append carries its own value.
    append = parser.parse_args(["job", "note", "7", "--append", "more text"])
    assert append.append == "more text" and append.text == []

    clear = parser.parse_args(["job", "note", "7", "--clear"])
    assert clear.clear is True


# --- behaviour -----------------------------------------------------------

def test_submit_stores_the_note_and_the_request(parser, queue_db, capsys):
    _submit(parser, queue_db, "--name", "train", "--vram", "20G",
            "--note", "基线实验 A", "--", "python", "train.py")
    capsys.readouterr()

    conn = dbm.open_db(queue_db)
    try:
        job = dbm.list_jobs(conn)[0]
        assert job.name == "train"
        assert job.vram_mb == 20480
        assert job.notes == "基线实验 A"
        assert job.command == "python train.py"
    finally:
        conn.close()


def test_submit_without_a_command_is_refused(parser, queue_db, capsys):
    assert _run(parser, ["job", "submit", "--name", "empty"], queue_db) == 1
    assert "nothing to run" in capsys.readouterr().err


def test_submit_refuses_limits_that_would_silently_mean_nothing(parser, queue_db, capsys):
    """`--timeout 0` reads as "no limit" once stored, which is not what it says."""
    assert _run(parser, ["job", "submit", "--timeout", "0", "--", "true"], queue_db) == 1
    assert "--timeout must be a positive" in capsys.readouterr().err

    assert _run(parser, ["job", "submit", "--gpus", "-1", "--", "true"], queue_db) == 1
    assert "--gpus cannot be negative" in capsys.readouterr().err

    assert _run(parser, ["job", "submit", "--retries", "-2", "--", "true"], queue_db) == 1
    assert "--retries cannot be negative" in capsys.readouterr().err

    # A name no shell can export was silently dropped on the way to the node.
    argv = ["job", "submit", "--env", "MY-VAR=1", "--", "true"]
    assert _run(parser, argv, queue_db) == 1
    assert "not a valid shell variable name" in capsys.readouterr().err

    conn = dbm.open_db(queue_db)
    try:
        assert dbm.list_jobs(conn) == []  # nothing was queued by a rejected call
    finally:
        conn.close()


def test_edit_changes_a_pending_jobs_vram_reservation(parser, queue_db, capsys):
    _submit(
        parser,
        queue_db,
        "--vram",
        "40G",
        "--note",
        "preserve me",
        "--",
        "python",
        "train.py",
    )
    capsys.readouterr()

    assert _run(
        parser,
        ["job", "edit", "1", "--vram", "39.5G", "--no-tick", "--json"],
        queue_db,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job"]["vram_mb"] == 40448
    assert payload["job"]["notes"] == "preserve me"
    assert payload["tick"] == {"skipped": "disabled"}

    conn = dbm.open_db(queue_db)
    try:
        assert dbm.get_job(conn, 1).vram_mb == 40448
        event = next(
            item
            for item in dbm.list_events(conn, job_id=1)
            if item["kind"] == "job_vram"
        )
        assert event["data"] == {"before_mb": 40960, "after_mb": 40448}
    finally:
        conn.close()


def test_edit_rejects_an_invalid_vram_without_changing_the_job(
    parser, queue_db, capsys
):
    _submit(parser, queue_db, "--vram", "4G", "--", "true")
    capsys.readouterr()

    assert _run(
        parser,
        ["job", "edit", "1", "--vram", "many", "--no-tick", "--json"],
        queue_db,
    ) == 1
    assert "Invalid size" in json.loads(capsys.readouterr().out)["error"]

    conn = dbm.open_db(queue_db)
    try:
        assert dbm.get_job(conn, 1).vram_mb == 4096
    finally:
        conn.close()


def test_ignore_hides_a_node_until_it_is_explicitly_requested(
    parser, queue_db, capsys
):
    conn = dbm.open_db(queue_db)
    try:
        dbm.upsert_node(conn, "training-A100", hostname="10.0.0.42")
    finally:
        conn.close()

    assert _run(
        parser, ["queue", "ignore", "training", "--json"], queue_db
    ) == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "node": "training-A100",
        "ignored": True,
    }

    assert _run(
        parser, ["queue", "nodes", "--json", "--no-tick"], queue_db
    ) == 0
    assert json.loads(capsys.readouterr().out)["nodes"] == []

    assert _run(
        parser,
        ["queue", "nodes", "--json", "--no-tick", "--include-ignored"],
        queue_db,
    ) == 0
    shown = json.loads(capsys.readouterr().out)["nodes"]
    assert shown[0]["name"] == "training-A100"
    assert shown[0]["ignored"] is True

    assert _run(
        parser, ["queue", "unignore", "TRAINING-A100", "--json"], queue_db
    ) == 0
    assert json.loads(capsys.readouterr().out)["ignored"] is False


def test_ignore_refuses_to_abandon_a_running_job(parser, queue_db, capsys):
    conn = dbm.open_db(queue_db)
    try:
        dbm.upsert_node(conn, "busy-node", hostname="10.0.0.1")
        job_id = dbm.insert_job(conn, command="train")
        dbm.update_job(conn, job_id, state="running", node="busy-node")
    finally:
        conn.close()

    assert _run(
        parser, ["queue", "ignore", "busy", "--json"], queue_db
    ) == 1
    payload = json.loads(capsys.readouterr().out)
    assert "running job(s)" in payload["error"]

    conn = dbm.open_db(queue_db)
    try:
        assert dbm.get_node(conn, "busy-node").ignored is False
    finally:
        conn.close()


def test_ignore_waits_for_an_active_scheduler_tick(parser, queue_db, capsys):
    conn = dbm.open_db(queue_db)
    tick_owner = f"{socket.gethostname()}:{os.getpid()}"
    try:
        dbm.upsert_node(conn, "busy-node", hostname="10.0.0.1")
        # The command runs in the same process as this simulated tick. Its
        # administrative lease still needs a distinct owner to avoid treating
        # the scheduler's lease as reentrant.
        assert dbm.acquire_lock(conn, TICK_LOCK, tick_owner, 60) is True

        assert _run(
            parser, ["queue", "ignore", "busy", "--json"], queue_db
        ) == 1
        payload = json.loads(capsys.readouterr().out)
        assert "scheduler is busy" in payload["error"]
        assert dbm.get_node(conn, "busy-node").ignored is False
    finally:
        dbm.release_lock(conn, TICK_LOCK, tick_owner)
        conn.close()


def test_note_reads_writes_appends_and_clears(parser, queue_db, capsys):
    _submit(parser, queue_db, "--note", "first", "--", "true")
    capsys.readouterr()

    assert _run(parser, ["job", "note", "1"], queue_db) == 0
    assert capsys.readouterr().out.strip() == "first"

    assert _run(parser, ["job", "note", "1", "--append", "second"], queue_db) == 0
    assert "first | second" in capsys.readouterr().out

    assert _run(parser, ["job", "note", "1", "replaced"], queue_db) == 0
    assert "replaced" in capsys.readouterr().out

    assert _run(parser, ["job", "note", "1", "--clear"], queue_db) == 0
    assert "cleared" in capsys.readouterr().out


def test_note_refuses_contradictory_arguments(parser, queue_db, capsys):
    _submit(parser, queue_db, "--", "true")
    capsys.readouterr()
    assert _run(parser, ["job", "note", "1", "--clear", "--append", "x"], queue_db) == 1
    assert "pick one" in capsys.readouterr().err


def test_note_on_a_missing_job_reports_an_error(parser, queue_db, capsys):
    assert _run(parser, ["job", "note", "404", "hello"], queue_db) == 1
    assert "not found" in capsys.readouterr().err


def test_priority_reads_sets_adjusts_and_moves(parser, queue_db, capsys):
    _submit(parser, queue_db, "--", "true")
    _submit(parser, queue_db, "--", "true")
    capsys.readouterr()

    assert _run(parser, ["job", "priority", "1"], queue_db) == 0
    assert "priority: 0" in capsys.readouterr().out

    assert _run(parser, ["job", "priority", "1", "5"], queue_db) == 0
    assert "priority: 5" in capsys.readouterr().out

    assert _run(parser, ["job", "priority", "1", "-2"], queue_db) == 0
    assert "priority: 3" in capsys.readouterr().out

    assert _run(parser, ["job", "priority", "2", "--up", "1"], queue_db) == 0
    assert "moved up" in capsys.readouterr().out
    conn = dbm.open_db(queue_db)
    try:
        pending = [job.id for job in dbm.pending_jobs(conn)]
    finally:
        conn.close()
    assert pending == [2, 1]

    assert _run(parser, ["job", "priority", "404", "1"], queue_db) == 1
    assert "not found" in capsys.readouterr().err


def test_wait_reports_a_timeout_differently_from_a_failure(parser, queue_db, capsys):
    """"It failed" and "I stopped waiting" call for opposite next steps."""
    _submit(parser, queue_db, "--name", "slow", "--", "true")
    capsys.readouterr()

    # No nodes are configured, so the job cannot start: waiting must time out.
    assert _run(parser, ["job", "wait", "1", "--timeout", "1", "--json"], queue_db) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["timed_out"] is True
    assert payload["all_done"] is False

    conn = dbm.open_db(queue_db)
    try:
        dbm.update_job(conn, 1, state="failed", exit_code=3)
    finally:
        conn.close()
    assert _run(parser, ["job", "wait", "1", "--json"], queue_db) == 1
    assert json.loads(capsys.readouterr().out)["timed_out"] is False

    conn = dbm.open_db(queue_db)
    try:
        dbm.update_job(conn, 1, state="completed", exit_code=0)
    finally:
        conn.close()
    assert _run(parser, ["job", "wait", "1", "--json"], queue_db) == 0


def test_wait_fails_if_a_finished_job_is_purged_while_waiting(
    parser, queue_db, monkeypatch, capsys
):
    _submit(parser, queue_db, "--name", "failed", "--", "false")
    capsys.readouterr()
    conn = dbm.open_db(queue_db)
    try:
        dbm.update_job(conn, 1, state="failed", exit_code=1)
    finally:
        conn.close()

    def purge_during_tick(scheduler, *args, **kwargs):
        dbm.purge_jobs(scheduler.conn)
        return {"ran": True}

    monkeypatch.setattr(Scheduler, "tick", purge_during_tick)
    assert _run(parser, ["job", "wait", "1", "--json"], queue_db) == 1
    payload = json.loads(capsys.readouterr().out)
    assert "disappeared while waiting" in payload["error"]


def test_status_names_the_database_it_actually_read(parser, tmp_path, monkeypatch, capsys):
    """Clients decide whether they are looking at the same queue by this path."""
    monkeypatch.setattr("nvidb.config.load_config", lambda *a, **k: {"servers": []})
    monkeypatch.delenv("NVIDB_QUEUE_DB", raising=False)
    elsewhere = tmp_path / "elsewhere.db"
    argv = ["queue", "status", "--json", "--no-tick", "--db-path", str(elsewhere)]
    assert sched_cli.dispatch(parser.parse_args(argv)) == 0
    assert json.loads(capsys.readouterr().out)["db_path"] == str(elsewhere)


def test_alerts_exit_non_zero_for_any_open_alert_not_just_listed_ones(
    parser, queue_db, capsys
):
    """Agents use this exit code as a health check before reporting success."""
    conn = dbm.open_db(queue_db)
    try:
        old = dbm.add_alert(conn, "job_failed", "an old failure nobody handled")
        for index in range(5):
            dbm.acknowledge_alerts(
                conn, [dbm.add_alert(conn, "job_failed", f"handled {index}")]
            )
    finally:
        conn.close()

    # The five newest alerts are all acknowledged; the open one is off the page.
    assert _run(parser, ["queue", "alerts", "--all", "-n", "5", "--no-tick"], queue_db) == 1
    capsys.readouterr()

    conn = dbm.open_db(queue_db)
    try:
        dbm.acknowledge_alerts(conn, [old])
    finally:
        conn.close()
    assert _run(parser, ["queue", "alerts", "--all", "-n", "5", "--no-tick"], queue_db) == 0


def test_json_output_is_the_only_thing_on_stdout(parser, queue_db, capsys):
    _submit(parser, queue_db, "--name", "train", "--json", "--", "true")
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["job"]["name"] == "train"


def test_a_json_error_is_still_valid_json(parser, queue_db, capsys):
    assert _run(parser, ["job", "show", "404", "--json"], queue_db) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_the_job_table_only_shows_columns_that_carry_data(parser, queue_db, capsys):
    _submit(parser, queue_db, "--name", "plain", "--", "true")
    capsys.readouterr()
    assert _run(parser, ["job", "ls"], queue_db) == 0
    assert "PROGRESS" not in capsys.readouterr().out

    conn = dbm.open_db(queue_db)
    try:
        dbm.update_job(conn, 1, progress="epoch 3/10")
    finally:
        conn.close()
    assert _run(parser, ["job", "ls"], queue_db) == 0
    output = capsys.readouterr().out
    assert "PROGRESS" in output and "epoch 3/10" in output


def test_the_job_table_adapts_to_a_narrow_terminal(
    parser,
    queue_db,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sched_cli.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((40, 24)),
    )
    _submit(
        parser,
        queue_db,
        "--name",
        "中文训练任务",
        "--",
        "python",
        "train.py",
        "--epochs",
        "100",
    )
    capsys.readouterr()

    assert _run(parser, ["job", "ls"], queue_db) == 0
    output = capsys.readouterr().out
    header = output.splitlines()[0]

    assert all(display_width(line) <= 40 for line in output.splitlines())
    assert "ID" in header and "STATE" in header and "COMMAND" in header
    assert "USED" not in header and "NODE" not in header
    assert "中" in output and "python" in output


def test_events_replay_from_a_given_id(parser, queue_db, capsys):
    _submit(parser, queue_db, "--name", "a", "--", "true")
    _submit(parser, queue_db, "--name", "b", "--", "true")
    capsys.readouterr()

    assert _run(parser, ["queue", "events", "--json"], queue_db) == 0
    events = json.loads(capsys.readouterr().out)["events"]
    assert [event["kind"] for event in events] == ["job_submitted", "job_submitted"]

    assert _run(parser, ["queue", "events", "--json", "--since", str(events[0]["id"])],
                queue_db) == 0
    later = json.loads(capsys.readouterr().out)["events"]
    assert len(later) == 1


def test_result_can_be_written_and_read_back(parser, queue_db, capsys):
    _submit(parser, queue_db, "--", "true")
    capsys.readouterr()

    assert _run(parser, ["job", "result", "1", "--set", '{"accuracy": 0.93}'],
                queue_db) == 0
    capsys.readouterr()
    assert _run(parser, ["job", "result", "1", "--json"], queue_db) == 0
    assert json.loads(capsys.readouterr().out)["result"] == {"accuracy": 0.93}


def _raise_alert(queue_db, **fields):
    conn = dbm.open_db(queue_db)
    try:
        return dbm.add_alert(
            conn,
            fields.pop("kind", "job_failed"),
            fields.pop("title", "train failed with exit code 1"),
            **fields,
        )
    finally:
        conn.close()


def test_alerts_exit_non_zero_while_anything_is_unacknowledged(parser, queue_db, capsys):
    assert _run(parser, ["queue", "alerts"], queue_db) == 0
    assert "no open alerts" in capsys.readouterr().out

    _raise_alert(queue_db, job_id=1, node="big-node", detail="CUDA out of memory")
    # Non-zero is what lets a shell or an agent branch on "anything wrong?".
    assert _run(parser, ["queue", "alerts"], queue_db) == 1
    output = capsys.readouterr().out
    assert "job_failed" in output and "exit code 1" in output
    assert "CUDA out of memory" not in output  # only with --detail

    assert _run(parser, ["queue", "alerts", "--detail"], queue_db) == 1
    assert "CUDA out of memory" in capsys.readouterr().out


def test_acknowledging_clears_the_alert_exit_status(parser, queue_db, capsys):
    _raise_alert(queue_db)
    _raise_alert(queue_db, kind="node_lost", title="node down")
    capsys.readouterr()

    assert _run(parser, ["queue", "ack", "--all"], queue_db) == 0
    assert "acknowledged 2" in capsys.readouterr().out
    assert _run(parser, ["queue", "alerts"], queue_db) == 0
    capsys.readouterr()

    # Acknowledged alerts are kept, just not shown by default.
    assert _run(parser, ["queue", "alerts", "--all", "--json"], queue_db) == 0
    assert len(json.loads(capsys.readouterr().out)["alerts"]) == 2


def test_acknowledging_one_alert_leaves_the_others(parser, queue_db, capsys):
    first = _raise_alert(queue_db, title="first")
    _raise_alert(queue_db, title="second")
    capsys.readouterr()

    assert _run(parser, ["queue", "ack", str(first)], queue_db) == 0
    capsys.readouterr()
    assert _run(parser, ["queue", "alerts", "--json"], queue_db) == 1
    open_alerts = json.loads(capsys.readouterr().out)["alerts"]
    assert [alert["title"] for alert in open_alerts] == ["second"]


def test_status_leads_with_open_alerts(parser, queue_db, capsys):
    _raise_alert(queue_db, title="train failed with exit code 1")
    assert _run(parser, ["queue", "status"], queue_db) == 0
    output = capsys.readouterr().out
    assert "ALERTS  1 open" in output
    assert output.index("ALERTS") < output.index("NODES")


def test_the_daemon_ticks_and_delivers_in_one_pass(parser, queue_db, capsys, tmp_path):
    log_path = tmp_path / "alerts.log"
    import nvidb.sched.notify as notify_module

    original = notify_module.alert_log_path
    notify_module.alert_log_path = lambda: log_path
    try:
        _raise_alert(queue_db, title="train failed with exit code 1")
        assert _run(
            parser,
            ["queue", "daemon", "--once", "--interval", "2", "--json"],
            queue_db,
        ) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["notified"] == 1
        assert payload["tick"]["ran"] is True
        assert "train failed" in log_path.read_text()
    finally:
        notify_module.alert_log_path = original


def test_the_daemon_honours_queue_notify_settings(
    parser, queue_db, capsys, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "nvidb.config.load_config",
        lambda *a, **k: {
            "servers": [],
            "queue": {"notify": {"desktop": False, "log": False}},
        },
    )
    log_path = tmp_path / "disabled-alerts.log"
    monkeypatch.setattr("nvidb.sched.notify.alert_log_path", lambda: log_path)
    _raise_alert(queue_db, title="should not be logged")

    assert _run(parser, ["queue", "daemon", "--once", "--json"], queue_db) == 0
    assert json.loads(capsys.readouterr().out)["notified"] == 1
    assert not log_path.exists()


def test_the_daemon_can_tick_without_notifying(parser, queue_db, capsys):
    _raise_alert(queue_db)
    assert _run(
        parser, ["queue", "daemon", "--once", "--no-notify", "--json"], queue_db
    ) == 0
    assert json.loads(capsys.readouterr().out)["notified"] == 0

    conn = dbm.open_db(queue_db)
    try:
        assert dbm.undelivered_alerts(conn)  # still waiting for a real pass
    finally:
        conn.close()


def test_backup_command_creates_a_readable_snapshot(
    parser, queue_db, tmp_path, capsys
):
    _submit(parser, queue_db, "--name", "preserved", "--", "true")
    capsys.readouterr()
    destination = tmp_path / "queue-backup.db"

    assert _run(
        parser,
        ["queue", "backup", str(destination), "--json"],
        queue_db,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backup"]["verified"] is True

    backup_conn = dbm.open_db(destination)
    try:
        assert dbm.get_job(backup_conn, 1).name == "preserved"
    finally:
        backup_conn.close()

    # An existing destination is protected from accidental replacement.
    assert _run(
        parser,
        ["queue", "backup", str(destination), "--json"],
        queue_db,
    ) == 1
    assert "already exists" in json.loads(capsys.readouterr().out)["error"]


def test_daemon_creates_a_configured_periodic_backup(
    parser, queue_db, tmp_path, monkeypatch, capsys
):
    directory = tmp_path / "automatic-backups"
    monkeypatch.setattr(
        "nvidb.config.load_config",
        lambda *a, **k: {
            "servers": [],
            "queue": {
                "backup": {
                    "enabled": True,
                    "interval_hours": 24,
                    "directory": str(directory),
                    "keep": 2,
                }
            },
        },
    )

    assert _run(
        parser,
        ["queue", "daemon", "--once", "--no-notify", "--json"],
        queue_db,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backup"]["verified"] is True
    assert payload["backup_error"] is None
    assert len(list(directory.glob("queue-*.db"))) == 1


def test_submitting_against_an_unknown_node_is_refused(parser, queue_db, capsys):
    assert _run(parser, ["job", "submit", "--node", "nowhere", "--", "true"],
                queue_db) == 1
    assert "Unknown node" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["job", "cancel", "--json"],
    ["job", "show", "not-an-id", "--json"],
    ["job", "cancel", "123", "--json", "--unknown"],
])
def test_argument_errors_are_json(parser, argv, capsys):
    with pytest.raises(SystemExit) as error:
        parser.parse_args(argv)
    assert error.value.code == 2
    output = capsys.readouterr()
    assert json.loads(output.out)["ok"] is False
    assert output.err == ""


def test_cancel_accepts_no_tick_and_preserves_other_queued_jobs(parser, queue_db, capsys):
    _submit(parser, queue_db, "--no-tick", "--", "first")
    _submit(parser, queue_db, "--no-tick", "--", "second")
    capsys.readouterr()
    assert _run(parser, ["job", "cancel", "1", "--no-tick", "--json"], queue_db) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["cancelled"]
    conn = dbm.open_db(queue_db)
    assert dbm.get_job(conn, 2).state == "pending"
    conn.close()


def test_cancel_no_tick_batch_succeeds_while_a_guarded_tick_holds_the_lease(
    parser, queue_db, capsys, monkeypatch
):
    conn = dbm.open_db(queue_db)
    dbm.upsert_node(conn, "box", hostname="localhost")
    dbm.ensure_lane(conn, "box:0", node="box", gpu_ids=[0])
    for index in range(3):
        dbm.insert_job(conn, command="keep", lane="box:0", lane_seq=index + 1)
    dbm.acquire_lock(conn, TICK_LOCK, "guarded-tick", 60)
    dbm.set_meta(conn, UNSTAGED_CANCEL_OWNER, "guarded-tick")

    def no_tick_or_ssh(*args, **kwargs):
        pytest.fail("--no-tick unstaged cancellation must stay local")

    monkeypatch.setattr(Scheduler, "tick", no_tick_or_ssh)
    monkeypatch.setattr(Scheduler, "backend", no_tick_or_ssh)
    try:
        assert _run(parser, ["job", "cancel", "1", "2", "--no-tick", "--json"], queue_db) == 0
        assert json.loads(capsys.readouterr().out)["results"] == [
            {"id": 1, "cancelled": True, "holds": []},
            {"id": 2, "cancelled": True, "holds": []},
        ]
        assert dbm.get_job(conn, 3).state == "pending"
        assert dbm.lock_holder(conn, TICK_LOCK)["owner"] == "guarded-tick"
    finally:
        dbm.release_lock(conn, TICK_LOCK, "guarded-tick")
        conn.close()


def test_job_command_json_flag_does_not_change_parser_error_format(parser, capsys):
    with pytest.raises(SystemExit):
        parser.parse_args(["job", "submit", "--gpus", "bad", "--", "app", "--json"])
    output = capsys.readouterr()
    assert output.out == ""
    assert "error:" in output.err


def test_daemon_survives_a_lost_scheduler_lease(parser, monkeypatch, capsys):
    from nvidb.sched.scheduler import SchedulerLeaseLost
    class FakeScheduler:
        cfg = {}
        calls = 0
        closed = False

        def tick(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise SchedulerLeaseLost("test")
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

    scheduler = FakeScheduler()
    monkeypatch.setattr(sched_cli, "_open", lambda args: scheduler)
    monkeypatch.setattr(sched_cli.time, "sleep", lambda seconds: None)
    args = parser.parse_args(["queue", "daemon", "--no-notify", "--no-backup", "--json"])
    assert sched_cli.cmd_daemon(args) == 0
    assert scheduler.calls == 2
    assert scheduler.closed
    assert json.loads(capsys.readouterr().out)["tick"]["skipped"] == "lease_lost"


def test_flags_given_before_a_subcommand_are_kept(parser):
    """argparse copies a subcommand's defaults over what its parent parsed."""
    args = parser.parse_args(["queue", "--local", "--json", "--db-path", "/x", "status"])
    assert (args.local, args.json, args.db_path) == (True, True, "/x")
    args = parser.parse_args(["queue", "lane", "box:0", "--no-tick", "--json", "skip"])
    assert (args.no_tick, args.json) == (True, True)
    assert parser.parse_args(["queue", "--refresh", "1", "tui"]).refresh == 1.0
    # Every command still gets every shared key, with the usual defaults.
    for argv in (["job"], ["job", "ls"], ["queue", "tick"], ["queue", "tui"]):
        args = parser.parse_args(argv)
        assert (args.json, args.local, args.db_path, args.tick, args.no_tick) == (
            False, False, None, False, False,
        )


def test_an_unknown_node_filter_matches_nothing(parser, queue_db, capsys):
    conn = dbm.open_db(queue_db)
    try:
        for name in ("box406", "box407"):
            dbm.upsert_node(conn, name)
            dbm.ensure_lane(conn, f"{name}:0", node=name, gpu_ids=[0])
            job_id = dbm.insert_job(conn, command="sleep 1")
            dbm.update_job(conn, job_id, state="running", node=name)
    finally:
        conn.close()

    # "box" is ambiguous and "typo" unknown: neither may widen to every node.
    for query, expected in (("box406", ["box406"]), ("typo", []), ("box", [])):
        assert _run(parser, ["job", "ls", "--node", query, "--json", "--no-tick"], queue_db) == 0
        assert [job["node"] for job in json.loads(capsys.readouterr().out)["jobs"]] == expected
        assert _run(parser, ["queue", "lanes", "--node", query, "--json", "--no-tick"], queue_db) == 0
        assert [lane["node"] for lane in json.loads(capsys.readouterr().out)["lanes"]] == expected


def test_unseen_skips_what_a_sliding_log_window_already_showed():
    assert sched_cli._unseen("", "a\nb\n") == "a\nb\n"
    assert sched_cli._unseen("a\nb", "a\nbc\n") == "c\n"
    assert sched_cli._unseen("a\nb\nc\n", "b\nc\nd\n") == "d\n"
    assert sched_cli._unseen("a\nb\n", "x\ny\n") == "x\ny\n"


def test_follow_prints_each_log_line_once(parser, monkeypatch, capsys):
    """Past `-n` lines the tail slides; the lines written as the job ends still show."""
    from types import SimpleNamespace

    log = [f"line {i}\n" for i in range(1, 6)]
    state = {"done": False}

    class FakeScheduler:
        conn = None

        def job_logs(self, job_id, *, stream, lines):
            return "".join(log[-lines:])

        def tick(self, **kwargs):
            if len(log) == 5:
                log.extend(["line 6\n", "line 7\n"])
            else:
                # The job writes its last line and is marked finished by
                # another client before this follower reads again.
                log.append("line 8\n")
                state["done"] = True

        def close(self):
            pass

    monkeypatch.setattr(sched_cli, "_open", lambda args: FakeScheduler())
    monkeypatch.setattr(sched_cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sched_cli.dbm,
        "get_job",
        lambda conn, job_id: SimpleNamespace(id=job_id, is_terminal=state["done"]),
    )
    args = parser.parse_args(["job", "logs", "1", "-f", "-n", "3", "--no-tick"])
    assert sched_cli.cmd_logs(args) == 0
    assert capsys.readouterr().out == "".join(f"line {i}\n" for i in range(3, 9))
