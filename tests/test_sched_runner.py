"""Tests for the resident lane runner.

Two halves, tested separately because they fail in different ways:

* the shipped node-side program, which is executed here for real against a
  temporary spool - it runs on machines nobody administers, so "it parses" is
  not enough;
* the controller protocol around it: staging a window of work, adopting what
  the runner started, and cleaning up after a job that was called off too late.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fake_cluster import FakeCluster, FakeGpu, FakeNode  # noqa: E402

from nvidb.sched import db as dbm  # noqa: E402
from nvidb.sched import runner_agent  # noqa: E402
from nvidb.sched.scheduler import Scheduler  # noqa: E402


# --- the shipped program ---------------------------------------------------

@pytest.fixture
def spool(tmp_path):
    path = tmp_path / "lane-spool"
    (path / "queue").mkdir(parents=True)
    (path / "claimed").mkdir()
    return path


def _runner_file(tmp_path):
    script = tmp_path / "runner.py"
    script.write_text(runner_agent.RUNNER_SCRIPT, encoding="utf-8")
    return script


def _stage(spool, tmp_path, *, job_id, seq, command, **extra):
    """Write a spec the way the controller does, wrapping a real run.sh."""
    from nvidb.sched.executor import build_run_script
    import base64

    run_dir = tmp_path / f"job-{job_id}"
    script = build_run_script(
        job_id=job_id,
        job_name=f"job{job_id}",
        command=command,
        run_dir=str(run_dir),
        workdir=None,
        gpu_ids=[0],
        node_name="test",
    )
    spec = {
        "job_id": job_id,
        "lane": "test:0",
        "seq": seq,
        "run_dir": str(run_dir),
        "gpu_ids": [0],
        "vram_mb": 0,
        "max_runtime_s": None,
        "concurrency": 1,
        "script_b64": base64.b64encode(script.encode("utf-8")).decode("ascii"),
    }
    spec.update(extra)
    (spool / "queue" / f"{seq:012d}-{job_id}.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    return run_dir


def _run_runner(spool, tmp_path, *, idle_exit=2.0, interval=0.05, timeout=60):
    """Run the real runner until it exits on its idle timeout."""
    script = _runner_file(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(script),
            str(spool),
            "test:0",
            "NVIDB_LANE=test",
            "testversion",
            str(interval),
            str(idle_exit),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed


def _state(spool):
    return json.loads((spool / "state.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(os.name == "nt", reason="the runner targets POSIX nodes")
def test_the_runner_runs_a_staged_job(spool, tmp_path):
    run_dir = _stage(spool, tmp_path, job_id=1, seq=1000, command="echo hello")
    result = _run_runner(spool, tmp_path)

    assert result.returncode == 0, result.stderr
    assert (run_dir / "stdout.log").read_text().strip() == "hello"
    assert (run_dir / "exit_code").read_text().split()[0] == "0"


@pytest.mark.skipif(os.name == "nt", reason="the runner targets POSIX nodes")
def test_the_runner_runs_a_lane_strictly_in_order(spool, tmp_path):
    """Sequence numbers are the schedule, not the order the files landed in."""
    marker = tmp_path / "order.txt"
    for job_id, seq in ((7, 3000), (8, 1000), (9, 2000)):
        _stage(
            spool,
            tmp_path,
            job_id=job_id,
            seq=seq,
            command=f"echo {job_id} >> {marker}",
        )
    result = _run_runner(spool, tmp_path)

    assert result.returncode == 0, result.stderr
    assert marker.read_text().split() == ["8", "9", "7"]


@pytest.mark.skipif(os.name == "nt", reason="the runner targets POSIX nodes")
def test_a_failing_job_does_not_stop_the_lane(spool, tmp_path):
    _stage(spool, tmp_path, job_id=1, seq=1000, command="exit 3")
    after = _stage(spool, tmp_path, job_id=2, seq=2000, command="echo second")
    result = _run_runner(spool, tmp_path)

    assert result.returncode == 0, result.stderr
    assert (after / "stdout.log").read_text().strip() == "second"
    finished = {item["job_id"]: item["exit_code"] for item in _state(spool)["finished"]}
    assert finished == {1: 3, 2: 0}


@pytest.mark.skipif(os.name == "nt", reason="the runner targets POSIX nodes")
def test_the_runner_reports_what_it_is_running(spool, tmp_path):
    _stage(spool, tmp_path, job_id=5, seq=1000, command="sleep 0.4")
    _run_runner(spool, tmp_path)

    state = _state(spool)
    assert state["lane"] == "test:0"
    assert state["queued"] == []
    assert [item["job_id"] for item in state["finished"]] == [5]


@pytest.mark.skipif(os.name == "nt", reason="the runner targets POSIX nodes")
def test_the_runner_enforces_a_time_limit(spool, tmp_path):
    run_dir = _stage(
        spool,
        tmp_path,
        job_id=1,
        seq=1000,
        command="sleep 30",
        max_runtime_s=1,
    )
    started = time.time()
    result = _run_runner(spool, tmp_path, idle_exit=2.0, timeout=40)
    elapsed = time.time() - started

    assert result.returncode == 0, result.stderr
    assert elapsed < 30, "the runner should not have waited for the job"
    # run.sh traps TERM and reports it, so the outcome is recorded rather than
    # vanishing with the process.
    assert (run_dir / "exit_code").exists()


@pytest.mark.skipif(os.name == "nt", reason="the runner targets POSIX nodes")
def test_the_runner_exits_when_idle_and_leaves_no_pid_behind(spool, tmp_path):
    result = _run_runner(spool, tmp_path, idle_exit=0.3)
    assert result.returncode == 0, result.stderr
    assert not (spool / "runner.pid").exists()


@pytest.mark.skipif(os.name == "nt", reason="the runner targets POSIX nodes")
def test_a_withdrawn_spec_is_never_run(spool, tmp_path):
    """Withdrawal is how a cancelled job is taken back before it starts."""
    run_dir = _stage(spool, tmp_path, job_id=1, seq=1000, command="echo ran")
    for path in (spool / "queue").iterdir():
        path.unlink()
    result = _run_runner(spool, tmp_path, idle_exit=0.3)

    assert result.returncode == 0, result.stderr
    assert not run_dir.exists()


@pytest.mark.skipif(os.name == "nt", reason="the runner targets POSIX nodes")
def test_a_second_runner_adopts_the_job_the_first_one_started(spool, tmp_path):
    """Restarting or upgrading a runner must not take the work down with it."""
    _stage(spool, tmp_path, job_id=1, seq=1000, command="sleep 2; echo done")
    script = _runner_file(tmp_path)
    first = subprocess.Popen(
        [sys.executable, "-u", str(script), str(spool), "test:0",
         "NVIDB_LANE=test", "v1", "0.05", "60"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            if (spool / "state.json").exists() and _state(spool)["current"]:
                break
            time.sleep(0.05)
        assert _state(spool)["current"], "the first runner never started the job"
        first.terminate()
        first.wait(timeout=10)
    finally:
        if first.poll() is None:  # pragma: no cover - only on a hung runner
            first.kill()

    # The job outlived its runner; a fresh one picks it up and sees it through.
    result = _run_runner(spool, tmp_path, idle_exit=0.5, timeout=40)
    assert result.returncode == 0, result.stderr
    assert "adopted job 1" in result.stderr
    assert [item["job_id"] for item in _state(spool)["finished"]] == [1]


def _has_nvidia_smi() -> bool:
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(os.name == "nt", reason="the runner targets POSIX nodes")
def test_the_vram_gate_holds_a_job_back_only_when_it_can_measure(spool, tmp_path):
    """The gate is a last check before starting, and it fails open.

    On a node with a driver, a job asking for more memory than is free waits.
    Where nothing can answer - no driver, or a machine that reports no
    per-process memory - the job starts: the controller already decided the
    lane could hold it, and refusing to run on an unanswerable question would
    strand the lane permanently.
    """
    _stage(
        spool,
        tmp_path,
        job_id=1,
        seq=1000,
        command="echo ran",
        vram_mb=10 ** 9,  # more VRAM than any card has
        gpu_ids=[0],
    )
    result = _run_runner(spool, tmp_path, idle_exit=0.5)
    assert result.returncode == 0, result.stderr

    state = _state(spool)
    if _has_nvidia_smi():
        assert state["queued"] == [1], "an impossible request should have waited"
        assert state["finished"] == []
    else:
        assert [item["job_id"] for item in state["finished"]] == [1]


# --- the command the controller sends -------------------------------------

def test_the_sync_command_carries_the_specs_and_the_runner():
    command = runner_agent.build_sync_command(
        lane_name="box:0",
        spool_root=".nvidb/lanes",
        specs=[{"job_id": 12, "seq": 1000}],
        want_runner=True,
    )
    assert "base64 -d" in command
    assert runner_agent.RUNNER_VERSION in command
    assert runner_agent.runner_marker("box:0") in command
    assert runner_agent.STATE_MARKER in command
    # Started detached, or it would die with the SSH channel that launched it.
    assert "setsid" in command and "nohup" in command


def test_the_sync_command_withdraws_everything_when_nothing_is_staged():
    command = runner_agent.build_sync_command(
        lane_name="box:0", spool_root=".nvidb/lanes", specs=[], want_runner=False
    )
    assert 'rm -f "$d"/queue/*.json' in command
    assert ': > "$d/stop"' in command


def test_a_relative_spool_root_expands_against_the_remote_home():
    """`$HOME` inside single quotes is four literal characters, and the node
    ends up with a directory actually named `$HOME` next to the real one."""
    for command in (
        runner_agent.build_sync_command(
            lane_name="box:0", spool_root=".nvidb/lanes", specs=[], want_runner=True
        ),
        runner_agent.build_stop_command(lane_name="box:0", spool_root=".nvidb/lanes"),
    ):
        assert "'$HOME" not in command, command.splitlines()[0]
        assert command.startswith('d="$HOME"/.nvidb/lanes/')


def test_an_absolute_spool_root_is_used_as_given():
    command = runner_agent.build_stop_command(
        lane_name="box:0", spool_root="/scratch/nvidb/lanes"
    )
    assert "$HOME" not in command
    assert command.startswith("d=/scratch/nvidb/lanes/")


@pytest.mark.skipif(os.name == "nt", reason="the runner targets POSIX nodes")
def test_the_spool_command_really_lands_where_it_says(tmp_path):
    """Run the generated shell for real and check where the directory went."""
    home = tmp_path / "home"
    home.mkdir()
    command = runner_agent.build_sync_command(
        lane_name="box:0", spool_root=".nvidb/lanes", specs=[], want_runner=False
    )
    subprocess.run(
        ["bash", "-c", command],
        cwd=str(home),
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        check=True,
    )
    assert (home / ".nvidb" / "lanes").is_dir()
    assert not (home / "$HOME").exists()


def test_a_lane_slug_is_a_safe_directory_name_and_unique():
    a = runner_agent.lane_slug("user@10.0.0.1:22:0")
    b = runner_agent.lane_slug("user@10.0.0.1:22:1")
    assert a != b
    for slug in (a, b):
        assert "/" not in slug and ":" not in slug and "@" not in slug


def test_parsing_state_ignores_anything_before_the_marker():
    stdout = f"warning: some shell noise\n{runner_agent.STATE_MARKER}\n" + json.dumps(
        {"lane": "box:0", "current": []}
    )
    assert runner_agent.parse_state(stdout)["lane"] == "box:0"


@pytest.mark.parametrize("stdout", ["", "no marker at all", runner_agent.STATE_MARKER])
def test_parsing_state_returns_nothing_when_there_is_none(stdout):
    assert runner_agent.parse_state(stdout) is None


def test_a_truncated_state_document_is_not_believed():
    """A half-written read must not be mistaken for a runner that reported
    nothing running, which would restart jobs that are already going."""
    stdout = runner_agent.STATE_MARKER + '\n{"lane": "box:0", "current": [{'
    assert runner_agent.parse_state(stdout) is None


# --- the controller protocol ----------------------------------------------

@pytest.fixture
def cluster():
    return FakeCluster(FakeNode("box", [FakeGpu(0, "RTX 3090 Ti", 24564)]))


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
            "lane_lookahead": 1,
        },
        backend_factory=cluster.backend_factory,
        owner="test",
    )
    sched.sync_nodes_from_config()
    yield sched
    sched.close()
    conn.close()


def _runner(cluster):
    return cluster.backends["box"].runner


def test_only_the_head_of_the_lane_is_handed_over(scheduler, cluster):
    """Everything behind the window stays a local record, which is what makes
    reordering it free."""
    scheduler.tick(force=True)
    for index in range(4):
        scheduler.submit(f"step{index}", lane="box:0", vram="1G")
    scheduler.tick(force=True)

    staged = [ids for _lane, ids, _want in _runner(cluster).syncs if ids]
    assert all(len(ids) == 1 for ids in staged), staged


def test_a_job_the_runner_started_is_recorded_as_running(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("x", lane="box:0", vram="1G")
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "running"
    assert job.node == "box"
    assert job.gpu_ids == [0]
    assert job.remote_pid
    assert job.run_dir
    assert job.attempt == 1


def test_the_start_is_recorded_once_not_on_every_pass(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("x", lane="box:0", vram="1G")
    for _ in range(3):
        scheduler.tick(force=True)

    starts = [
        event
        for event in dbm.list_events(scheduler.conn, job_id=job_id)
        if event["kind"] == "job_started"
    ]
    assert len(starts) == 1
    assert dbm.get_job(scheduler.conn, job_id).attempt == 1


def test_the_runner_advances_the_lane_without_the_controller(scheduler, cluster):
    """The point of the runner: the next job starts when the previous one
    exits, not when a client next happens to look."""
    scheduler.tick(force=True)
    first = scheduler.submit("first", lane="box:0", vram="1G")
    second = scheduler.submit("second", lane="box:0", vram="1G")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, first).state == "running"
    # With the head running, the next tick hands the runner the job behind it.
    scheduler.tick(force=True)
    assert [spec["job_id"] for spec in _runner(cluster).lanes["box:0"]["queued"]] == [second]

    # No controller from here on: the runner's own loop notices the exit and
    # starts the next job by itself.
    cluster["box"].finish_job(first, exit_code=0)
    state = _runner(cluster).poll("box:0")
    assert [entry["job_id"] for entry in state["current"]] == [second]


def test_a_job_that_starts_and_ends_between_two_passes_is_not_run_again(
    scheduler, cluster
):
    """The runner's `current` is a window, not a record.

    A short job is started and retired by the runner without any pass ever
    seeing it there. Its run directory is durable though, so the exit code is
    still found - and finding it is what stops the job being staged, run, and
    staged again on every pass after that.
    """
    scheduler.tick(force=True)
    job_id = scheduler.submit("quick", lane="box:0", vram="1G")
    scheduler.tick(force=True)

    # It ran and was retired while nothing was looking.
    cluster["box"].finish_job(job_id, exit_code=0)
    _runner(cluster).poll("box:0")
    assert _runner(cluster).lanes["box:0"]["current"] == []

    scheduler.tick(force=True)
    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "completed"
    assert job.exit_code == 0
    assert job.attempt == 1

    starts = [
        event
        for event in dbm.list_events(scheduler.conn, job_id=job_id)
        if event["kind"] == "job_started"
    ]
    assert len(starts) <= 1, "the job should not have been started twice"

    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "completed"


def test_a_lane_moves_on_after_a_job_nobody_saw_running(scheduler, cluster):
    """The lane must not wedge on a job that finished unobserved."""
    scheduler.tick(force=True)
    first = scheduler.submit("first", lane="box:0", vram="1G")
    second = scheduler.submit("second", lane="box:0", vram="1G")
    scheduler.tick(force=True)

    cluster["box"].finish_job(first, exit_code=0)
    _runner(cluster).poll("box:0")
    scheduler.tick(force=True)

    assert dbm.get_job(scheduler.conn, first).state == "completed"
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, second).state in ("running", "completed")


def test_a_short_job_that_failed_unobserved_is_recorded_as_failed(scheduler, cluster):
    scheduler.tick(force=True)
    job_id = scheduler.submit("boom", lane="box:0", vram="1G")
    scheduler.tick(force=True)
    cluster["box"].finish_job(job_id, exit_code=7)
    _runner(cluster).poll("box:0")
    scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "failed"
    assert job.exit_code == 7
    kinds = [
        alert["kind"] for alert in dbm.list_alerts(scheduler.conn, open_only=True)
    ]
    assert "job_failed" in kinds


def test_a_staged_job_that_has_not_started_is_left_alone(scheduler, cluster):
    """No process and no exit code means "not started yet" here, where for a
    running job the same observation means it vanished."""
    scheduler.tick(force=True)
    scheduler.set_lane_paused("box:0", True)
    job_id = scheduler.submit("waiting", lane="box:0", vram="1G")
    for _ in range(3):
        scheduler.tick(force=True)

    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == "pending"
    assert job.exit_code is None


def test_a_paused_lane_is_stripped_of_its_staged_work(scheduler, cluster):
    scheduler.tick(force=True)
    scheduler.submit("first", lane="box:0", vram="1G")
    queued = scheduler.submit("queued", lane="box:0", vram="1G")
    scheduler.tick(force=True)
    scheduler.tick(force=True)
    assert [spec["job_id"] for spec in _runner(cluster).lanes["box:0"]["queued"]] == [queued]

    scheduler.set_lane_paused("box:0", True)
    scheduler.tick(force=True)
    assert _runner(cluster).lanes["box:0"]["queued"] == []


def test_cancelling_a_staged_job_takes_it_back_before_it_starts(scheduler, cluster):
    scheduler.tick(force=True)
    first = scheduler.submit("first", lane="box:0", vram="1G")
    scheduler.tick(force=True)
    queued = scheduler.submit("queued", lane="box:0", vram="1G")
    scheduler.tick(force=True)

    scheduler.cancel(queued)
    staged = [spec["job_id"] for spec in _runner(cluster).lanes["box:0"]["queued"]]
    assert queued not in staged
    assert dbm.get_job(scheduler.conn, first).state == "running"


def test_a_job_started_after_it_was_cancelled_is_stopped(scheduler, cluster):
    """The withdrawal can lose the race. What it must not do is leave work
    running that nothing is tracking."""
    scheduler.tick(force=True)
    job_id = scheduler.submit("x", lane="box:0", vram="1G")
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, job_id).state == "running"

    # Close the record behind the runner's back, as a cancel racing a claim does.
    dbm.update_job(scheduler.conn, job_id, state="cancelled")
    scheduler.tick(force=True)

    assert job_id in cluster["box"].reaped


def test_a_lane_with_no_work_gets_no_resident_process(scheduler, cluster):
    scheduler.tick(force=True)
    scheduler.tick(force=True)
    assert _runner(cluster).syncs == []


def test_turning_the_runner_off_leaves_the_lane_working(scheduler, cluster):
    """Without a runner the lane still runs in order; it just needs a client
    to advance it."""
    scheduler.set_lane_runner("box:0", False)
    scheduler.tick(force=True)
    first = scheduler.submit("first", lane="box:0", vram="1G")
    second = scheduler.submit("second", lane="box:0", vram="1G")
    scheduler.tick(force=True)

    assert dbm.get_job(scheduler.conn, first).state == "running"
    assert dbm.get_job(scheduler.conn, second).state == "pending"
    assert _runner(cluster).syncs == []

    cluster["box"].finish_job(first, exit_code=0)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, second).state == "running"


def test_an_unreachable_node_does_not_break_the_tick(scheduler, cluster):
    scheduler.tick(force=True)
    scheduler.submit("x", lane="box:0", vram="1G")
    cluster["box"].online = False
    summary = scheduler.tick(force=True)

    assert summary["ran"] is True
    assert summary["nodes_down"] == 1


def test_the_lane_view_reports_the_runner(scheduler, cluster):
    scheduler.tick(force=True)
    scheduler.submit("x", lane="box:0", vram="1G")
    scheduler.tick(force=True)

    view = scheduler.lane_view(dbm.get_lane(scheduler.conn, "box:0"))
    assert view["runner"] is True
    assert view["runner_up"] is True
    assert view["blocked"] is None
