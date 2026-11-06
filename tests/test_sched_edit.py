"""Editing a command must preserve the queued job and never race a launch."""
import json
import sys

import pytest

from fake_cluster import FakeCluster, FakeGpu, FakeNode, make_scheduler

from nvidb.sched import cli, db, remote
from nvidb.sched.scheduler import Scheduler, TICK_LOCK, UNSTAGED_EDIT_OWNER


@pytest.fixture
def scheduler(tmp_path):
    conn = db.open_db(tmp_path / "queue.db")
    scheduler = Scheduler(conn, cfg={"servers": [{
        "nickname": "box", "hostname": "test.invalid", "username": "test",
    }]})
    scheduler.sync_nodes_from_config()
    yield scheduler
    scheduler.close()
    conn.close()


def stored_job(scheduler, job_id):
    return dict(scheduler.conn.execute(
        "SELECT * FROM jobs WHERE id = ?", (job_id,),
    ).fetchone())


def test_command_edit_preserves_definition_dependencies_and_lane_order(scheduler):
    upstream = scheduler.submit("train")
    predecessor = scheduler.submit("first", lane="box:0")
    job_id = scheduler.submit(
        "old evaluation", name="eval", lane="box:0", workdir="/repo with spaces",
        env={"LABEL": "评测"}, gpus=1, vram="4G", priority=3,
        depends_on=[upstream], depends_any=[predecessor], max_runtime_s=120,
        max_retries=2, submitter="review", tags=["eval"], notes="keep this",
    )
    follower = scheduler.submit("last", lane="box:0", depends_on=[job_id])
    db.update_job(scheduler.conn, job_id, held_reason="upstream failed")
    before = stored_job(scheduler, job_id)
    replacement = "\npython eval.py --note '中文 $HOME'\nprintf '%s\\n' done\n"

    edited = scheduler.edit_job(job_id, command=replacement)

    after = stored_job(scheduler, job_id)
    assert after.pop("command") == replacement
    before.pop("command")
    after.pop("updated_at")
    before.pop("updated_at")
    assert after == before
    assert edited.is_held
    assert [job.id for job in db.lane_jobs(scheduler.conn, "box:0")] == [
        predecessor, job_id, follower,
    ]
    assert db.get_job(scheduler.conn, follower).depends_on == [job_id]
    events = db.list_events(scheduler.conn, job_id=job_id)
    assert events[-1]["kind"] == "job_command"
    assert events[-1]["data"] == {"before": "old evaluation", "after": replacement}


def test_command_and_request_changes_are_one_transaction(scheduler):
    upstream = scheduler.submit("train")
    job_id = scheduler.submit("old", vram="1G")
    before = stored_job(scheduler, job_id)
    with pytest.raises(ValueError, match="does not exist"):
        scheduler.edit_job(job_id, command="new", vram="2G", add=[9999])
    assert stored_job(scheduler, job_id) == before
    assert [e["kind"] for e in db.list_events(scheduler.conn, job_id=job_id)] == [
        "job_submitted",
    ]
    edited = scheduler.edit_job(job_id, command="new", vram="2G", add=[upstream])
    assert (edited.command, edited.vram_mb, edited.depends_on) == ("new", 2048, [upstream])


@pytest.mark.parametrize("state", ["running", "completed"])
def test_command_edit_refuses_jobs_that_are_not_pending(scheduler, state):
    job_id = scheduler.submit("old")
    db.update_job(scheduler.conn, job_id, state=state)
    before = stored_job(scheduler, job_id)
    with pytest.raises(ValueError):
        scheduler.edit_job(job_id, command="new")
    assert stored_job(scheduler, job_id) == before


def test_command_edit_refuses_staged_jobs_even_if_database_says_pending(scheduler):
    job_id = scheduler.submit("old", lane="box:0")
    db.update_job(scheduler.conn, job_id, run_dir="/jobs/1")
    before = stored_job(scheduler, job_id)
    with pytest.raises(ValueError, match="already been staged"):
        scheduler.edit_job(job_id, command="new", vram="2G")
    assert stored_job(scheduler, job_id) == before


def test_command_edit_respects_a_concurrent_scheduler_lease(scheduler):
    job_id = scheduler.submit("old")
    db.acquire_lock(scheduler.conn, TICK_LOCK, "another-controller", 60)
    try:
        with pytest.raises(RuntimeError, match="scheduler is busy"):
            scheduler.edit_job(job_id, command="new")
        assert db.get_job(scheduler.conn, job_id).command == "old"
    finally:
        db.release_lock(scheduler.conn, TICK_LOCK, "another-controller")


def test_unchanged_command_does_not_write_an_edit_event(scheduler):
    job_id = scheduler.submit("same")
    before = stored_job(scheduler, job_id)
    scheduler.edit_job(job_id, command="same")
    assert stored_job(scheduler, job_id) == before
    assert len(db.list_events(scheduler.conn, job_id=job_id)) == 1


@pytest.fixture
def parser():
    parser = cli.ArgumentParser(prog="nvidb")
    cli.register_parsers(parser.add_subparsers(dest="command"))
    return parser


@pytest.fixture
def edit_cli(parser, scheduler, monkeypatch):
    monkeypatch.setattr(cli, "_open", lambda args: scheduler)

    def must_not_run(*args, **kwargs):
        pytest.fail("--no-tick command edits must not dispatch or contact nodes")

    monkeypatch.setattr(scheduler, "tick", must_not_run)
    monkeypatch.setattr(scheduler, "backend", must_not_run)

    def run(job_id, *options):
        args = parser.parse_args([
            "job", "edit", str(job_id), "--no-tick", "--json", *options,
        ])
        assert args.command == "job"
        return cli.dispatch(args)

    return run


@pytest.mark.parametrize("source", ["command", "script"])
def test_cli_replaces_the_exact_command_text(edit_cli, scheduler, tmp_path, capsys, source):
    job_id = scheduler.submit("old")
    replacement = "printf '%s\\n' \"中文 '$HOME'\"\npython eval.py --limit 20\n"
    value = replacement
    if source == "script":
        path = tmp_path / "evaluate.sh"
        path.write_text(replacement, encoding="utf-8")
        value = str(path)
    assert edit_cli(job_id, f"--{source}", value) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job"]["command"] == replacement
    assert payload["tick"] == {"skipped": "disabled"}


def test_cli_rejects_empty_or_unreadable_commands(edit_cli, scheduler, tmp_path, capsys):
    job_id = scheduler.submit("old")
    for options in [("--command", " \n"), ("--script", str(tmp_path / "missing.sh"))]:
        assert edit_cli(job_id, *options) == 1
        assert json.loads(capsys.readouterr().out)["ok"] is False
        assert db.get_job(scheduler.conn, job_id).command == "old"


def test_command_and_script_are_mutually_exclusive(parser, capsys):
    with pytest.raises(SystemExit) as error:
        parser.parse_args([
            "job", "edit", "1", "--command", "new", "--script", "eval.sh", "--json",
        ])
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_remote_edit_reads_the_local_script_before_forwarding(parser, tmp_path, monkeypatch):
    path = tmp_path / "local-only.sh"
    command = "printf '%s\\n' '评测 $HOME'\n"
    path.write_text(command, encoding="utf-8")
    argv = ["job", "edit", "42", "--script", str(path), "--no-tick", "--json"]
    monkeypatch.setattr(sys, "argv", ["nvidb", *argv])
    captured = {}

    def forward(target, arguments, **kwargs):
        captured.update(target=target, arguments=arguments, **kwargs)
        return 0

    monkeypatch.setattr(remote, "forward", forward)
    target = {"host": "queue-host", "nvidb": "nvidb", "keeper": "ensure"}
    assert cli._forward_to_queue_host(target, parser.parse_args(argv)) == 0
    assert captured["script_text"] == command
    shell = remote.build_command(target, argv, script_text=captured["script_text"])
    assert 'job edit 42 --script "$nvidb_script"' in shell
    assert str(path) not in shell
    assert "nvidb_keeper=" not in shell


@pytest.fixture
def runner_editors(tmp_path):
    cluster = FakeCluster(FakeNode("box", [FakeGpu(0, "test GPU", 24564)]))
    conn = db.open_db(tmp_path / "queue.db")
    scheduler = make_scheduler(conn, cluster)
    editor_conn = db.open_db(db.connection_path(conn))
    editor = make_scheduler(editor_conn, cluster)
    try:
        yield scheduler, editor, cluster
    finally:
        editor.close()
        editor_conn.close()
        scheduler.close()
        conn.close()


def test_failed_dependency_can_be_repaired_while_a_tick_probes(
    runner_editors, monkeypatch,
):
    scheduler, editor, cluster = runner_editors
    failed = scheduler.submit("failed export", gpus=0)
    replacement = scheduler.submit("replacement export", gpus=0)
    db.update_job(scheduler.conn, failed, state="failed", exit_code=1)
    db.update_job(scheduler.conn, replacement, state="completed", exit_code=0)
    jobs = [scheduler.submit(
        f"eval {i}", lane="box:0", vram="1G", depends_on=[failed], notes="keep",
    ) for i in range(2)]
    db.update_job(scheduler.conn, jobs[0], held_reason="export failed")
    positions = [db.get_job(scheduler.conn, i).lane_seq for i in jobs]
    backend = cluster.backend_factory(db.get_node(scheduler.conn, "box"))
    probe = backend.probe_gpus

    def repair_during_probe():
        assert db.lock_holder(editor.conn, TICK_LOCK)["owner"] == scheduler.owner
        for job_id in jobs:
            edited = editor.edit_job(job_id, drop=[failed], add=[replacement])
            assert edited.depends_on == [replacement]
            assert not edited.is_held
        assert db.lock_holder(editor.conn, TICK_LOCK)["owner"] == scheduler.owner
        return probe()

    monkeypatch.setattr(backend, "probe_gpus", repair_during_probe)
    monkeypatch.setattr(editor, "backend", lambda *a: pytest.fail("edit contacted a node"))
    assert scheduler.tick(force=True)["errors"] == []
    assert db.get_job(scheduler.conn, jobs[0]).state == "running"
    for job_id, position in zip(jobs, positions):
        job = db.get_job(scheduler.conn, job_id)
        assert (job.depends_on, job.lane_seq, job.notes) == ([replacement], position, "keep")
    assert db.get_job(scheduler.conn, failed).state == "failed"


@pytest.mark.parametrize("change", ["command", "vram", "add", "add_any"])
def test_edit_winning_before_reservation_never_publishes_the_stale_head(
    runner_editors, monkeypatch, change,
):
    scheduler, editor, cluster = runner_editors
    upstream = scheduler.submit("upstream", gpus=0)
    head = scheduler.submit("old", lane="box:0", vram="1G")
    tail = scheduler.submit("tail", lane="box:0", vram="1G")
    resolve = scheduler._lane_run_dir
    changes = {"command": "new", "vram": "2G", "add": [upstream], "add_any": [upstream]}

    def edit_before_reservation(lane, job, attempt):
        if job.id == head:
            editor.edit_job(head, **{change: changes[change]})
        return resolve(lane, job, attempt)

    monkeypatch.setattr(scheduler, "_lane_run_dir", edit_before_reservation)
    assert scheduler.tick(force=True)["errors"] == []
    for job_id in (head, tail):
        assert db.get_job(scheduler.conn, job_id).run_dir is None
        assert job_id not in cluster["box"].jobs

    monkeypatch.setattr(scheduler, "_lane_run_dir", resolve)
    if change in ("add", "add_any"):
        scheduler.tick(force=True)
        assert head not in cluster["box"].jobs
        cluster["box"].finish_job(upstream)
    assert scheduler.tick(force=True)["errors"] == []
    assert db.get_job(scheduler.conn, head).state == "running"
    current = db.get_job(scheduler.conn, head)
    if change == "command":
        assert current.command == "new"
    if change == "vram":
        assert current.vram_mb == 2048


@pytest.mark.parametrize("change", [{"command": "new"}, {"vram": "2G"}, {"drop": [1]}])
def test_a_staged_edit_cannot_bypass_the_tick_lease(runner_editors, monkeypatch, change):
    scheduler, editor, cluster = runner_editors
    job_id = scheduler.submit("old", lane="box:0", vram="1G")
    backend = cluster.backend_factory(db.get_node(scheduler.conn, "box"))
    sync = backend.lane_sync

    def edit_after_reservation(**kwargs):
        assert db.get_job(editor.conn, job_id).run_dir
        with pytest.raises(RuntimeError, match="scheduler is busy"):
            editor.edit_job(job_id, **change)
        return sync(**kwargs)

    monkeypatch.setattr(backend, "lane_sync", edit_after_reservation)
    assert scheduler.tick(force=True)["errors"] == []
    assert db.get_job(scheduler.conn, job_id).state == "running"
    assert db.get_job(scheduler.conn, job_id).command == "old"


@pytest.mark.parametrize("owner", ["older-tick", "controller:cancel"])
def test_old_controllers_and_administrative_operations_keep_exclusive_edits(
    runner_editors, owner,
):
    scheduler, editor, _cluster = runner_editors
    job_id = scheduler.submit("old", lane="box:0", vram="1G")
    db.set_meta(scheduler.conn, UNSTAGED_EDIT_OWNER, "different-tick")
    db.acquire_lock(scheduler.conn, TICK_LOCK, owner, 60)
    try:
        with pytest.raises(RuntimeError, match="scheduler is busy"):
            editor.edit_job(job_id, command="new")
        assert db.get_job(scheduler.conn, job_id).command == "old"
    finally:
        db.release_lock(scheduler.conn, TICK_LOCK, owner)


def test_no_tick_dependency_edit_succeeds_during_a_compatible_tick(
    edit_cli, scheduler, capsys,
):
    old = scheduler.submit("old export", gpus=0)
    new = scheduler.submit("new export", gpus=0)
    job_id = scheduler.submit("eval", lane="box:0", depends_on=[old])
    db.set_meta(scheduler.conn, UNSTAGED_EDIT_OWNER, "live-tick")
    db.acquire_lock(scheduler.conn, TICK_LOCK, "live-tick", 60)
    try:
        assert edit_cli(job_id, "--drop-after", str(old), "--add-after", str(new)) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["job"]["depends_on"] == [new]
        assert payload["tick"] == {"skipped": "disabled"}
    finally:
        db.release_lock(scheduler.conn, TICK_LOCK, "live-tick")
