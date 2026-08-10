"""Tests for queue configuration, process supervision, and remote forwarding."""
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nvidb import config as nvidb_config
from nvidb.sched import keeper as keeper_mod
from nvidb.sched import remote as remote_mod


@pytest.fixture
def nvidb_home(isolated_nvidb_config, tmp_path, monkeypatch):
    """An isolated working directory, as `NVIDB_HOME` would give.

    Takes the suite-wide isolation first, then undoes the two parts of it these
    tests are here to exercise: config discovery and the forwarding decision.
    """
    home = tmp_path / "nvidb"
    home.mkdir()
    monkeypatch.setattr(nvidb_config, "WORKING_DIR", home)
    monkeypatch.delenv("NVIDB_QUEUE_CONFIG", raising=False)
    monkeypatch.delenv(remote_mod.NO_REMOTE_ENV, raising=False)
    return home


def _write(path, text: str):
    path.write_text(text, encoding="utf-8")


# --- queue.yml ------------------------------------------------------------

def test_without_queue_yml_the_main_config_is_used_unchanged(nvidb_home, monkeypatch):
    _write(nvidb_home / "config.yml", "servers:\n  - hostname: a\n    nickname: n1\nqueue:\n  headroom_mb: 99\n")
    cfg = nvidb_config.load_queue_config()
    assert [server["nickname"] for server in cfg["servers"]] == ["n1"]
    assert cfg["queue"]["headroom_mb"] == 99


def test_invalid_queue_yml_fails_instead_of_using_monitor_nodes(nvidb_home):
    _write(
        nvidb_home / "config.yml",
        "servers:\n  - hostname: colleague\n    nickname: monitor-only\n",
    )
    _write(nvidb_home / "queue.yml", "servers: [\n")

    with pytest.raises(ValueError, match="queue[.]yml"):
        nvidb_config.load_queue_config()


def test_queue_yml_must_contain_a_mapping(nvidb_home):
    _write(nvidb_home / "queue.yml", "- hostname: mine\n")

    with pytest.raises(ValueError, match="queue[.]yml.*mapping"):
        nvidb_config.load_queue_config()


def test_queue_yml_replaces_the_server_list(nvidb_home):
    """The monitor watches machines the queue must not dispatch onto."""
    _write(
        nvidb_home / "config.yml",
        "servers:\n  - hostname: colleague\n    nickname: someone-elses-box\n",
    )
    _write(nvidb_home / "queue.yml", "servers:\n  - hostname: mine\n    nickname: gpu-node\n")
    cfg = nvidb_config.load_queue_config()
    assert [server["nickname"] for server in cfg["servers"]] == ["gpu-node"]


def test_queue_yml_may_call_the_list_nodes(nvidb_home):
    _write(nvidb_home / "queue.yml", "nodes:\n  - hostname: mine\n    nickname: gpu-node\n")
    cfg = nvidb_config.load_queue_config()
    assert [server["nickname"] for server in cfg["servers"]] == ["gpu-node"]
    assert "nodes" not in cfg


def test_queue_yml_without_servers_inherits_them(nvidb_home):
    _write(nvidb_home / "config.yml", "servers:\n  - hostname: a\n    nickname: n1\n")
    _write(nvidb_home / "queue.yml", "queue:\n  include_local: true\n")
    cfg = nvidb_config.load_queue_config()
    assert [server["nickname"] for server in cfg["servers"]] == ["n1"]
    assert cfg["queue"]["include_local"] is True


def test_queue_settings_merge_over_the_main_ones(nvidb_home):
    _write(
        nvidb_home / "config.yml",
        "queue:\n  headroom_mb: 512\n  notify:\n    command: 'push'\n    desktop: true\n",
    )
    _write(nvidb_home / "queue.yml", "queue:\n  headroom_mb: 1024\n  notify:\n    desktop: false\n")
    settings = nvidb_config.load_queue_config()["queue"]
    assert settings["headroom_mb"] == 1024
    # A file that only silences desktop pop-ups must not drop the push hook.
    assert settings["notify"] == {"command": "push", "desktop": False}


def test_the_scheduler_reads_the_queue_config(nvidb_home, tmp_path, monkeypatch):
    from nvidb.sched import db as dbm
    from nvidb.sched.scheduler import Scheduler

    _write(nvidb_home / "config.yml", "servers:\n  - hostname: a\n    nickname: monitor-only\n")
    _write(
        nvidb_home / "queue.yml",
        "servers:\n  - hostname: b\n    nickname: queue-node\n"
        "queue:\n  include_local: true\n  local_node_name: 'queue-host'\n",
    )
    conn = dbm.open_db(tmp_path / "queue.db")
    try:
        scheduler = Scheduler(conn)
        names = scheduler.sync_nodes_from_config()
        assert names == ["queue-node", "queue-host"]
    finally:
        conn.close()


# --- the keeper script ----------------------------------------------------

def test_install_bakes_in_an_absolute_nvidb_path(nvidb_home, tmp_path):
    fake = tmp_path / "bin" / "nvidb"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)

    info = keeper_mod.install(nvidb_bin=str(fake), interval=42)
    script = (nvidb_home / keeper_mod.SCRIPT_NAME).read_text(encoding="utf-8")

    assert info["nvidb"] == str(fake.resolve())
    assert str(fake.resolve()) in script
    assert "__NVIDB_BIN__" not in script and "__INTERVAL__" not in script
    assert "__MANAGER__" not in script and "__KEEPER_TOKEN__" not in script
    assert str(nvidb_home) in script
    assert 'INTERVAL="${NVIDB_QUEUE_INTERVAL:-42}"' in script
    assert info["manager"] == "shell"
    assert (nvidb_home / keeper_mod.MANAGER_NAME).read_text().strip() == "shell"
    assert os.stat(nvidb_home / keeper_mod.SCRIPT_NAME).st_mode & stat.S_IXUSR


def test_install_refuses_a_path_that_is_not_executable(nvidb_home, tmp_path):
    plain = tmp_path / "not-a-binary"
    plain.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        keeper_mod.install(nvidb_bin=str(plain))


def test_status_reports_a_dead_pid_as_stopped(nvidb_home, tmp_path):
    fake = tmp_path / "nvidb-bin"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    keeper_mod.install(nvidb_bin=str(fake))

    assert keeper_mod.status() == {
        "installed": True,
        "running": False,
        "pid": None,
        "manager": "shell",
        "script": str(nvidb_home / keeper_mod.SCRIPT_NAME),
        "unit": None,
        "log": str(nvidb_home / keeper_mod.LOG_NAME),
    }

    (nvidb_home / keeper_mod.PID_NAME).write_text("999999999", encoding="utf-8")
    assert keeper_mod.status()["running"] is False

    # A live but unrelated pid must not be accepted from a stale pid file.
    (nvidb_home / keeper_mod.PID_NAME).write_text(str(os.getpid()), encoding="utf-8")
    (nvidb_home / keeper_mod.TOKEN_NAME).write_text("stale-token", encoding="utf-8")
    assert keeper_mod.status()["running"] is False

    (nvidb_home / keeper_mod.PID_NAME).write_text("9" * 1000, encoding="utf-8")
    assert keeper_mod.status()["running"] is False

    (nvidb_home / keeper_mod.PID_NAME).write_text("-1", encoding="utf-8")
    assert keeper_mod.status()["running"] is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_the_generated_script_is_valid_shell_and_reports_status(nvidb_home, tmp_path):
    fake = tmp_path / "nvidb-bin"
    fake.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
    fake.chmod(0o755)
    script = str(keeper_mod.install(nvidb_bin=str(fake))["script"])

    # Parsed but not run: a syntax error here would only show up as a keeper
    # that silently never starts.
    assert subprocess.run(["sh", "-n", script]).returncode == 0

    stopped = subprocess.run(["sh", script, "status"], capture_output=True, text=True)
    assert stopped.returncode == 1 and "stopped" in stopped.stdout

    usage = subprocess.run(["sh", script, "nonsense"], capture_output=True, text=True)
    assert usage.returncode == 2 and "usage:" in usage.stderr


def _write_fake_daemon(path: Path, parent_log: Path):
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$PPID\" >> {str(parent_log)!r}\n"
        "trap 'exit 0' TERM INT\n"
        "while :; do sleep 1; done\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_reinstall_recognizes_and_restarts_a_tokenless_legacy_keeper(
    nvidb_home, tmp_path
):
    script = nvidb_home / keeper_mod.SCRIPT_NAME
    pid_file = nvidb_home / keeper_mod.PID_NAME
    script.write_text(
        "#!/bin/sh\n"
        f"PIDFILE={str(pid_file)!r}\n"
        'if [ "${1:-}" = "_loop" ]; then\n'
        '  echo $$ > "$PIDFILE"\n'
        "  trap 'rm -f \"$PIDFILE\"; exit 0' TERM INT\n"
        "  while :; do sleep 1; done\n"
        "fi\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    legacy = subprocess.Popen(
        ["sh", str(script), "_loop"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    fake = tmp_path / "nvidb-bin"
    parent_log = tmp_path / "daemon-parents"
    _write_fake_daemon(fake, parent_log)

    try:
        assert _wait_until(pid_file.exists)
        assert int(pid_file.read_text()) == legacy.pid

        keeper_mod.install(nvidb_bin=str(fake), interval=2)
        # The newly written controller must still identify the old tokenless
        # process, otherwise `install --start` would leave two supervisors.
        assert keeper_mod.run("status").returncode == 0
        assert keeper_mod.status()["pid"] == legacy.pid

        restarted = keeper_mod.run("restart")
        assert restarted.returncode == 0, restarted.stderr
        assert _wait_until(lambda: legacy.poll() is not None)
        assert keeper_mod.status()["running"] is True
        assert keeper_mod.status()["pid"] != legacy.pid
        assert _wait_until(parent_log.exists)
    finally:
        if keeper_mod.status()["running"]:
            keeper_mod.run("stop")
        if legacy.poll() is None:
            legacy.terminate()
        try:
            legacy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            legacy.kill()
            legacy.wait(timeout=5)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_keeper_really_detaches_and_concurrent_ensure_starts_one_loop(
    nvidb_home, tmp_path
):
    """Several clients may reconnect together; only one supervisor may start."""
    parent_log = tmp_path / "daemon-parents"
    fake = tmp_path / "nvidb-bin"
    _write_fake_daemon(fake, parent_log)
    script = str(keeper_mod.install(nvidb_bin=str(fake), interval=2)["script"])

    callers = [
        subprocess.Popen(
            ["sh", script, "ensure"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(6)
    ]
    try:
        results = [caller.communicate(timeout=12) for caller in callers]
        assert all(caller.returncode == 0 for caller in callers), results
        assert _wait_until(lambda: keeper_mod.status()["running"])
        assert _wait_until(parent_log.exists)
        parents = {
            line.strip()
            for line in parent_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert len(parents) == 1
        assert keeper_mod.status()["pid"] == int(next(iter(parents)))
    finally:
        keeper_mod.run("stop")
        assert _wait_until(lambda: not keeper_mod.status()["running"])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_a_live_start_lock_owner_is_never_removed(nvidb_home, tmp_path):
    fake = tmp_path / "nvidb-bin"
    parent_log = tmp_path / "daemon-parents"
    _write_fake_daemon(fake, parent_log)
    script = str(keeper_mod.install(nvidb_bin=str(fake), interval=2)["script"])
    lock_dir = nvidb_home / keeper_mod.LOCK_NAME
    lock_dir.mkdir()
    owner_file = lock_dir / "owner"
    owner = subprocess.Popen(
        [
            "sh",
            "-c",
            "trap 'exit 0' TERM INT; while :; do sleep 1; done",
            script,
            "ensure",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    owner_file.write_text(str(owner.pid), encoding="utf-8")

    try:
        result = subprocess.run(
            ["sh", script, "ensure"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert "could not acquire" in result.stderr
        assert owner_file.read_text().strip() == str(owner.pid)
        assert keeper_mod.status()["running"] is False
        assert not parent_log.exists()
    finally:
        owner.terminate()
        try:
            owner.wait(timeout=5)
        except subprocess.TimeoutExpired:
            owner.kill()
            owner.wait(timeout=5)
        if keeper_mod.status()["running"]:
            keeper_mod.run("stop")
        shutil.rmtree(lock_dir, ignore_errors=True)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
def test_a_reused_unrelated_start_lock_pid_is_reclaimed(nvidb_home, tmp_path):
    fake = tmp_path / "nvidb-bin"
    parent_log = tmp_path / "daemon-parents"
    _write_fake_daemon(fake, parent_log)
    script = str(keeper_mod.install(nvidb_bin=str(fake), interval=2)["script"])
    lock_dir = nvidb_home / keeper_mod.LOCK_NAME
    lock_dir.mkdir()
    (lock_dir / "owner").write_text(str(os.getpid()), encoding="utf-8")

    try:
        result = subprocess.run(
            ["sh", script, "ensure"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert _wait_until(lambda: keeper_mod.status()["running"])
        assert _wait_until(parent_log.exists)
    finally:
        if keeper_mod.status()["running"]:
            keeper_mod.run("stop")
        shutil.rmtree(lock_dir, ignore_errors=True)


def test_systemd_install_writes_a_user_unit(nvidb_home, tmp_path, monkeypatch):
    fake = tmp_path / "nvidb-bin"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    unit = tmp_path / "systemd" / keeper_mod.SYSTEMD_UNIT_NAME
    calls = []

    monkeypatch.setattr(keeper_mod, "systemd_unit_path", lambda: unit)
    monkeypatch.setattr(keeper_mod.shutil, "which", lambda name: f"/usr/bin/{name}")

    def systemctl(*args):
        calls.append(args)
        output = "4321\n" if args and args[0] == "show" else ""
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(keeper_mod, "_systemctl", systemctl)
    info = keeper_mod.install(
        nvidb_bin=str(fake),
        interval=17,
        manager_name="systemd",
    )

    service = unit.read_text(encoding="utf-8")
    script = (nvidb_home / keeper_mod.SCRIPT_NAME).read_text(encoding="utf-8")
    assert info["manager"] == "systemd"
    assert info["unit"] == str(unit)
    assert "ExecStart=" in service and "queue daemon --interval 17" in service
    assert "Restart=always" in service
    assert "network-online.target" not in service
    assert "MANAGER=systemd" in script
    assert calls == [("daemon-reload",)]

    state = keeper_mod.status()
    assert state["running"] is True
    assert state["pid"] == 4321
    assert state["manager"] == "systemd"


# --- remote forwarding ----------------------------------------------------

@pytest.fixture
def target():
    return {
        "host": "queue-host",
        "port": 2222,
        "username": "alice",
        "nvidb": "/home/alice/.local/bin/nvidb",
        "keeper": "ensure",
        "nvidb_home": None,
        "ssh_options": [],
    }


def test_no_target_without_configuration(nvidb_home):
    assert remote_mod.load_target() is None


def test_target_is_read_from_queue_yml(nvidb_home):
    _write(
        nvidb_home / "queue.yml",
        "remote:\n  host: queue-host.example.com\n"
        "  nvidb: /home/alice/.local/bin/nvidb\n",
    )
    assert remote_mod.load_target() == {
        "host": "queue-host.example.com",
        "port": None,
        "username": None,
        "nvidb": "/home/alice/.local/bin/nvidb",
        "keeper": "ensure",
        "nvidb_home": None,
        "ssh_options": [],
    }


def test_the_far_side_is_told_not_to_forward_again(nvidb_home):
    """A remote pointing at itself must fail as a command, not loop."""
    _write(nvidb_home / "queue.yml", "remote:\n  host: h\n")
    os.environ[remote_mod.NO_REMOTE_ENV] = "1"
    try:
        assert remote_mod.load_target() is None
    finally:
        del os.environ[remote_mod.NO_REMOTE_ENV]


def test_local_only_flags_are_stripped_but_the_job_command_is_not(target):
    command = remote_mod.build_command(
        target,
        ["job", "submit", "--db-path", "/tmp/x.db", "--local", "--vram", "20G",
         "--", "python", "train.py", "--db-path", "keep-me"],
    )
    assert "/tmp/x.db" not in command
    assert "--local" not in command
    assert "--vram 20G" in command
    # Everything past `--` belongs to the user's command.
    assert command.endswith("-- python train.py --db-path keep-me")


def test_the_keeper_is_ensured_in_the_same_round_trip(target):
    command = remote_mod.build_command(target, ["queue", "status"])
    assert 'nvidb_keeper="$HOME/.nvidb/queue-keeper.sh"' in command
    assert 'sh "$nvidb_keeper" ensure' in command
    assert "warning: queue keeper is not installed" in command
    assert command.splitlines()[-1] == (
        "NVIDB_QUEUE_NO_REMOTE=1 /home/alice/.local/bin/nvidb queue status"
    )


def test_keeper_lifecycle_commands_do_not_auto_start_it(target):
    command = remote_mod.build_command(target, ["queue", "keeper", "status"])
    assert "nvidb_keeper=" not in command
    assert command == (
        "NVIDB_QUEUE_NO_REMOTE=1 /home/alice/.local/bin/nvidb "
        "queue keeper status"
    )


def test_the_keeper_can_be_left_alone(target):
    target["keeper"] = "off"
    assert "queue-keeper.sh" not in remote_mod.build_command(target, ["queue", "status"])


def test_a_custom_home_reaches_both_the_keeper_and_the_command(target):
    target["nvidb_home"] = "/data/nvidb"
    command = remote_mod.build_command(target, ["queue", "status"])
    assert "nvidb_keeper=/data/nvidb/queue-keeper.sh" in command
    assert 'sh "$nvidb_keeper" ensure' in command
    assert "NVIDB_HOME=/data/nvidb" in command


def test_a_missing_remote_keeper_warns_without_corrupting_the_command(
    target, tmp_path
):
    target["nvidb"] = shutil.which("true")
    target["nvidb_home"] = str(tmp_path / "home with spaces")
    command = remote_mod.build_command(target, ["queue", "status", "--json"])
    result = subprocess.run(
        ["sh", "-c", command],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "warning: queue keeper is not installed" in result.stderr
    assert str(tmp_path / "home with spaces") in result.stderr


def test_a_remote_keeper_start_failure_is_reported(target, tmp_path):
    target["nvidb"] = shutil.which("true")
    target["nvidb_home"] = str(tmp_path)
    keeper = tmp_path / keeper_mod.SCRIPT_NAME
    keeper.write_text(
        "#!/bin/sh\necho 'systemd user bus unavailable' >&2\nexit 7\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["sh", "-c", remote_mod.build_command(target, ["queue", "status"])],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "warning: queue keeper could not be started" in result.stderr
    assert "systemd user bus unavailable" in result.stderr


def test_a_local_script_travels_with_the_command(target):
    body = "#!/bin/bash\necho '你好'  # quotes, non-ASCII\n"
    command = remote_mod.build_command(
        target, ["job", "submit", "--script", "/local/train.sh"], script_text=body
    )
    assert "/local/train.sh" not in command
    assert "nvidb_script=$(mktemp)" in command
    assert '--script "$nvidb_script"' in command
    assert 'rm -f "$nvidb_script"' in command
    # The command's status has to survive the cleanup.
    assert command.endswith("exit $nvidb_rc")


def test_ssh_arguments_carry_the_port_and_user(target):
    target["ssh_options"] = ["ConnectTimeout=8"]
    argv = remote_mod.ssh_argv(target, "true", tty=True)
    assert argv[:8] == [
        "ssh", "-t", "-p", "2222", "-o", "ConnectTimeout=8",
        "alice@queue-host", "true",
    ]
