"""Tests for the queue's own configuration file, the keeper, and remote forwarding.

The keeper is a shell script, so the parts worth testing here are the ones that
decide what ends up inside it and what the CLI does with it: nothing starts a
real process.
"""
import os
import stat
import subprocess
import sys

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
    assert str(nvidb_home) in script
    assert 'INTERVAL="${NVIDB_QUEUE_INTERVAL:-42}"' in script
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
        "script": str(nvidb_home / keeper_mod.SCRIPT_NAME),
        "log": str(nvidb_home / keeper_mod.LOG_NAME),
    }

    (nvidb_home / keeper_mod.PID_NAME).write_text("999999999", encoding="utf-8")
    assert keeper_mod.status()["running"] is False

    (nvidb_home / keeper_mod.PID_NAME).write_text(str(os.getpid()), encoding="utf-8")
    state = keeper_mod.status()
    assert state["running"] is True and state["pid"] == os.getpid()


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
        "remote:\n  host: queue-host.example.com\n  nvidb: /home/alice/.local/bin/nvidb\n",
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
    assert command.splitlines()[0].startswith("sh $HOME/.nvidb/queue-keeper.sh ensure")
    assert command.splitlines()[-1] == (
        "NVIDB_QUEUE_NO_REMOTE=1 /home/alice/.local/bin/nvidb queue status"
    )


def test_the_keeper_can_be_left_alone(target):
    target["keeper"] = "off"
    assert "queue-keeper.sh" not in remote_mod.build_command(target, ["queue", "status"])


def test_a_custom_home_reaches_both_the_keeper_and_the_command(target):
    target["nvidb_home"] = "/data/nvidb"
    command = remote_mod.build_command(target, ["queue", "status"])
    assert "sh /data/nvidb/queue-keeper.sh ensure" in command
    assert "NVIDB_HOME=/data/nvidb" in command


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
    assert argv[:8] == ["ssh", "-t", "-p", "2222", "-o", "ConnectTimeout=8", "alice@queue-host", "true"]
