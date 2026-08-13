"""Integration tests for the job executor against real local processes.

These exercise the mechanism the queue relies on everywhere: a detached process
group, a pid file, an atomically written exit code, captured output, and a
`result.json` handed back to whoever asked for the job.
"""
import time

import pytest

from nvidb.sched.executor import JobExecutor
from nvidb.sched.transport import LocalTransport


def _wait_for_exit(executor, job_id, run_dir, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = executor.probe([(job_id, run_dir)]).jobs.get(job_id)
        if probe is not None and probe.finished:
            return probe
        time.sleep(0.1)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


@pytest.fixture
def executor(tmp_path):
    return JobExecutor(LocalTransport("test-local"), job_root=str(tmp_path / "jobs"))


def test_a_job_runs_detached_and_reports_its_exit_code(executor):
    launched = executor.launch(
        job_id=1,
        job_name="hello",
        command="echo out-line; echo err-line >&2; exit 0",
        node_name="test-local",
    )
    assert launched.pid

    probe = _wait_for_exit(executor, 1, launched.run_dir)
    assert probe.exit_code == 0
    assert probe.alive is False
    assert "out-line" in executor.read_log(launched.run_dir)
    assert "err-line" in executor.read_log(launched.run_dir, stream="stderr")


def test_a_failing_command_propagates_its_status(executor):
    launched = executor.launch(
        job_id=2, job_name="boom", command="exit 42", node_name="test-local"
    )
    probe = _wait_for_exit(executor, 2, launched.run_dir)
    assert probe.exit_code == 42
    # The job stamps its own finish time, so elapsed does not depend on how
    # soon anyone happens to look.
    assert probe.finished_epoch is not None
    assert abs(probe.finished_epoch - time.time()) < 60


def test_a_job_can_hand_back_a_result_payload(executor):
    command = (
        'printf \'{"accuracy": 0.91, "note": "done"}\' > "$NVIDB_JOB_DIR/result.json"'
    )
    launched = executor.launch(
        job_id=3, job_name="eval", command=command, node_name="test-local"
    )
    _wait_for_exit(executor, 3, launched.run_dir)
    assert executor.read_result(launched.run_dir) == {"accuracy": 0.91, "note": "done"}


def test_a_job_publishes_progress_through_its_status_file(executor):
    command = "\n".join(
        [
            'echo "epoch 1/3" > "$NVIDB_STATUS_FILE"',
            "sleep 1",
            'echo "epoch 2/3 loss 0.42" > "$NVIDB_STATUS_FILE"',
            "sleep 4",
        ]
    )
    launched = executor.launch(
        job_id=11, job_name="trainer", command=command, node_name="test-local"
    )
    time.sleep(0.5)
    assert executor.probe([(11, launched.run_dir)]).jobs[11].progress == "epoch 1/3"

    time.sleep(1.2)
    probe = executor.probe([(11, launched.run_dir)]).jobs[11]
    assert probe.progress == "epoch 2/3 loss 0.42"
    assert probe.alive is True
    executor.kill(pid=launched.pid, pgid=launched.pgid, signal="KILL")


def test_only_the_last_status_line_is_reported(executor):
    command = 'printf "old line\\nnewest line\\n" > "$NVIDB_STATUS_FILE"'
    launched = executor.launch(
        job_id=12, job_name="appender", command=command, node_name="test-local"
    )
    _wait_for_exit(executor, 12, launched.run_dir)
    assert executor.probe([(12, launched.run_dir)]).jobs[12].progress == "newest line"


def test_a_job_that_reports_nothing_has_no_progress(executor):
    launched = executor.launch(
        job_id=13, job_name="quiet", command="echo hi", node_name="test-local"
    )
    _wait_for_exit(executor, 13, launched.run_dir)
    assert executor.probe([(13, launched.run_dir)]).jobs[13].progress is None


def test_the_job_environment_describes_the_placement(executor):
    command = 'env | grep -E "^(NVIDB_|CUDA_VISIBLE_DEVICES)" | sort'
    launched = executor.launch(
        job_id=4,
        job_name="envcheck",
        command=command,
        gpu_ids=[0, 1],
        env={"MY_SETTING": "value with spaces"},
        node_name="node-x",
    )
    _wait_for_exit(executor, 4, launched.run_dir)
    log = executor.read_log(launched.run_dir)
    assert "CUDA_VISIBLE_DEVICES=0,1" in log
    assert "NVIDB_JOB_ID=4" in log
    assert "NVIDB_NODE=node-x" in log
    # The status file is how a job reports progress, so it must be advertised.
    assert "NVIDB_STATUS_FILE=" in log


def test_a_command_with_awkward_quoting_survives_the_trip(executor):
    command = """python3 -c 'print("quotes \\" and $dollar and \\'\\''single\\'\\''")'"""
    launched = executor.launch(
        job_id=5, job_name="quoting", command=command, node_name="test-local"
    )
    probe = _wait_for_exit(executor, 5, launched.run_dir)
    assert probe.exit_code == 0
    assert 'quotes " and $dollar and' in executor.read_log(launched.run_dir)


def test_a_multi_line_command_runs_as_a_script(executor):
    command = "\n".join(
        [
            "total=0",
            "for i in 1 2 3; do",
            "  total=$((total + i))",
            "done",
            'echo "total=$total"',
        ]
    )
    launched = executor.launch(
        job_id=6, job_name="script", command=command, node_name="test-local"
    )
    _wait_for_exit(executor, 6, launched.run_dir)
    assert "total=6" in executor.read_log(launched.run_dir)


def test_relaunching_a_live_job_adopts_it_instead_of_running_it_twice(executor):
    """The transport retries a command whose channel died mid-flight.

    If that retry reached a node which had already started the job, the queue
    would end up with two copies of the same training run and a record of only
    one of them.
    """
    launched = executor.launch(
        job_id=20,
        job_name="long",
        command='echo started; sleep 30',
        node_name="test-local",
    )
    assert launched.adopted is False
    time.sleep(0.4)

    again = executor.launch(
        job_id=20, job_name="long", command="echo started; sleep 30", node_name="test-local"
    )
    assert again.adopted is True
    assert again.pid == launched.pid
    assert again.run_dir == launched.run_dir

    # One process, one "started" line: the second call started nothing.
    log = executor.read_log(launched.run_dir)
    assert log.count("started") == 1
    assert executor.probe([(20, launched.run_dir)]).jobs[20].alive is True
    executor.kill(pid=launched.pid, pgid=launched.pgid, signal="KILL")


def test_a_finished_job_can_still_be_launched_again(executor):
    """Adoption must not block a re-queue: that job's process is gone."""
    launched = executor.launch(
        job_id=21, job_name="short", command="echo run-one", node_name="test-local"
    )
    _wait_for_exit(executor, 21, launched.run_dir)

    again = executor.launch(
        job_id=21, job_name="short", command="echo run-two", node_name="test-local"
    )
    assert again.adopted is False
    assert again.pid != launched.pid
    _wait_for_exit(executor, 21, again.run_dir)
    log = executor.read_log(again.run_dir)
    assert "run-two" in log and "run-one" not in log  # the log was reset


def test_requeued_attempts_use_distinct_run_directories(executor):
    assert executor.run_dir(22, 1).endswith("/22")
    assert executor.run_dir(22, 2).endswith("/22-attempt-2")


def test_reaping_kills_a_leftover_process_but_only_the_right_one(executor):
    """The cleanup runs long after the fact, when the pid may belong to anyone."""
    launched = executor.launch(
        job_id=30, job_name="leftover", command="sleep 30", node_name="test-local"
    )
    time.sleep(0.3)

    # A pid that is alive but is not this job: an unrelated process, which the
    # node has every right to have started since.
    import subprocess

    stranger = subprocess.Popen(["sleep", "30"])
    try:
        assert executor.reap(run_dir=launched.run_dir, pid=stranger.pid) is False
        assert stranger.poll() is None  # untouched
    finally:
        stranger.kill()
        stranger.wait()

    assert executor.reap(
        run_dir=launched.run_dir, pid=launched.pid, pgid=launched.pgid
    ) is True
    probe = _wait_for_exit(executor, 30, launched.run_dir)
    assert probe.alive is False
    # Reaping something already gone is a no-op, not an error.
    assert executor.reap(run_dir=launched.run_dir, pid=launched.pid) is False


def test_probe_does_not_adopt_a_reused_pid(executor):
    import subprocess
    from pathlib import Path

    stranger = subprocess.Popen(["sleep", "30"])
    run_dir = Path(executor.run_dir(31))
    run_dir.mkdir(parents=True)
    (run_dir / "pid").write_text(str(stranger.pid))
    try:
        probe = executor.probe([(31, str(run_dir))])
        assert probe.jobs[31].alive is False
        assert stranger.poll() is None
    finally:
        stranger.kill()
        stranger.wait()


def test_a_missing_workdir_fails_the_job_instead_of_hanging(executor):
    launched = executor.launch(
        job_id=7,
        job_name="badcwd",
        command="echo never",
        workdir="/definitely/not/here",
        node_name="test-local",
    )
    assert _wait_for_exit(executor, 7, launched.run_dir).exit_code == 127


def test_a_running_job_can_be_killed(executor):
    launched = executor.launch(
        job_id=8, job_name="sleeper", command="sleep 120", node_name="test-local"
    )
    time.sleep(0.5)
    assert executor.probe([(8, launched.run_dir)]).jobs[8].alive is True

    executor.kill(pid=launched.pid, pgid=launched.pgid, signal="TERM")
    deadline = time.time() + 10
    while time.time() < deadline:
        if not executor.probe([(8, launched.run_dir)]).jobs[8].alive:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the job was still alive after SIGTERM")


def test_the_probe_reports_the_process_group_table(executor):
    launched = executor.launch(
        job_id=9, job_name="sleeper", command="sleep 5", node_name="test-local"
    )
    time.sleep(0.4)
    probe = executor.probe([(9, launched.run_dir)])
    assert probe.process_groups
    # The job's own pid must appear, which is what lets GPU processes be
    # attributed to the job that owns them.
    assert launched.pid in probe.process_groups
    executor.kill(pid=launched.pid, pgid=launched.pgid, signal="KILL")


def test_probing_an_unknown_job_reports_nothing_rather_than_failing(executor, tmp_path):
    probe = executor.probe([(99, str(tmp_path / "missing"))])
    assert probe.jobs[99].alive is False
    assert probe.jobs[99].exit_code is None


def test_the_run_directory_can_be_removed(executor):
    launched = executor.launch(
        job_id=10, job_name="tidy", command="echo bye", node_name="test-local"
    )
    _wait_for_exit(executor, 10, launched.run_dir)
    executor.remove_run_dir(launched.run_dir)
    assert executor.read_log(launched.run_dir) == ""
