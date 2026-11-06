"""Large queue scripts must survive both shells in an SSH command hop."""
import hashlib
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from nvidb.sched import remote, runner_agent
from nvidb.sched.executor import build_run_script
from nvidb.sched.transport import LocalTransport, SSHTransport


@pytest.mark.parametrize("kind", ["local", "ssh"])
def test_large_unicode_bootstraps_are_executed_without_argument_limits(kind, monkeypatch):
    command = "# 中文 $HOME `literal`\n" * 20000 + "printf '%s' 'large script complete'\n"
    if kind == "local":
        transport = LocalTransport()
    else:
        # Execute the SSH server's login-shell hop for real. The client must
        # send a short exec request and stream the large script to its stdin.
        def exec_command(command, timeout):
            assert command == "sh -s"
            process = subprocess.Popen(
                ["/bin/sh", "-c", command], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            channel = SimpleNamespace(recv_exit_status=process.wait)
            return (
                process.stdin,
                SimpleNamespace(read=process.stdout.read, channel=channel),
                SimpleNamespace(read=process.stderr.read),
            )

        transport = SSHTransport("test.invalid")
        monkeypatch.setattr(transport, "_ensure_client", lambda: SimpleNamespace(
            exec_command=exec_command,
        ))
    result = transport.run(command)
    assert result.exit_status == 0, result.stderr
    assert result.stdout == "large script complete"


def test_large_lane_payload_never_becomes_a_python_argument(tmp_path):
    marker = tmp_path / "should-not-exist"
    spec = {
        "job_id": 12, "seq": 1000, "run_dir": str(tmp_path / "job"),
        "script_b64": "abcd" * 400000,
        "note": f"中文 '$HOME' $(touch {marker})\nNVIDB_LANE_PAYLOAD",
    }
    command = runner_agent.build_sync_command(
        lane_name="box:0", spool_root=str(tmp_path), specs=[spec],
        want_runner=False, python_exe=sys.executable,
    )
    result = LocalTransport().run(command)
    assert result.exit_status == 0, result.stderr
    path = tmp_path / runner_agent.lane_slug("box:0") / "queue" / "000000001000-12.json"
    assert json.loads(path.read_text()) == spec
    assert not marker.exists()


def test_a_large_script_runs_through_the_real_lane_protocol(tmp_path):
    import base64

    run_dir = tmp_path / "large lane job"
    command = "# 中文 large script\n" * 40000 + "echo completed\n"
    script = build_run_script(
        job_id=12, job_name="large", command=command, run_dir=str(run_dir),
        workdir=None, gpu_ids=[], node_name="test",
    )
    spec = {
        "job_id": 12, "seq": 1000, "run_dir": str(run_dir), "gpu_ids": [],
        "vram_mb": 0, "script_b64": base64.b64encode(script.encode()).decode(),
    }
    bootstrap = runner_agent.build_sync_command(
        lane_name="test:0", spool_root=str(tmp_path / "spools"), specs=[spec],
        want_runner=True, python_exe=sys.executable, interval=.05, idle_exit=.2,
    )
    result = LocalTransport().run(bootstrap)
    assert result.exit_status == 0, result.stderr
    deadline = time.monotonic() + 10
    while not (run_dir / "exit_code").exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert (run_dir / "exit_code").read_text().split()[0] == "0"
    assert (run_dir / "stdout.log").read_text().strip() == "completed"


def test_remote_script_forwarding_streams_large_files_and_preserves_exit_status(
    tmp_path, monkeypatch, capfd,
):
    script = "# 中文 '$HOME' `literal`\n" * 20000 + "echo complete\n"
    fake_cli = tmp_path / "queue cli"
    fake_cli.write_text(
        f"#!{sys.executable}\n"
        "import hashlib,json,pathlib,sys\n"
        "body=pathlib.Path(sys.argv[sys.argv.index('--script')+1]).read_bytes()\n"
        "print(json.dumps({'sha256':hashlib.sha256(body).hexdigest()}))\n"
        "raise SystemExit(7)\n",
    )
    fake_cli.chmod(0o755)
    target = {"host": "test.invalid", "nvidb": str(fake_cli), "keeper": "none"}

    def login_shell(target, command, **kwargs):
        assert command == "sh -s"
        return ["/bin/sh", "-c", command]

    monkeypatch.setattr(remote, "ssh_argv", login_shell)
    status = remote.forward(
        target, ["job", "submit", "--script", "local-only.sh"], script_text=script,
    )
    assert status == 7
    assert json.loads(capfd.readouterr().out)["sha256"] == hashlib.sha256(script.encode()).hexdigest()
