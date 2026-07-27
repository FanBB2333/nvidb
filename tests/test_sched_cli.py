"""Tests for the queue CLI's argument parsing and output.

These run the real parser and handlers against a temporary database with node
probing disabled, so they cover the command surface without touching a network.
"""
import argparse
import json

import pytest

from nvidb.sched import cli as sched_cli
from nvidb.sched import db as dbm


@pytest.fixture
def parser():
    root = argparse.ArgumentParser(prog="nvidb")
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
        ["queue", "nodes"],
        ["queue", "events"],
        ["queue", "drain", "n1"],
        ["job", "ls"],
        ["job", "show", "1"],
        ["job", "wait", "1"],
        ["job", "note", "1"],
    ):
        args = parser.parse_args(argv)
        assert args.queue_cli is True
        assert callable(args.func)


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


def test_submitting_against_an_unknown_node_is_refused(parser, queue_db, capsys):
    assert _run(parser, ["job", "submit", "--node", "nowhere", "--", "true"],
                queue_db) == 1
    assert "Unknown node" in capsys.readouterr().err
