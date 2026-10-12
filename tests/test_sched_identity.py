"""Submitter identity is resolved on the client from its global Git config."""
import os
import shlex
import subprocess

from nvidb.sched import identity, remote


def test_global_git_identity_ignores_repository_override(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDB_SUBMITTER", raising=False)
    config = tmp_path / "gitconfig"
    config.write_text('[user]\n name = Personal Name\n email = personal@example.test\n')
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.chdir(repo)
    subprocess.run(["git", "config", "user.name", "Repository Name"], check=True)
    assert identity.default_submitter() == "Personal Name <personal@example.test>"


def test_explicit_environment_takes_precedence(monkeypatch):
    monkeypatch.setenv("NVIDB_SUBMITTER", "agent:experiment")
    monkeypatch.setattr(identity.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no git lookup")))
    assert identity.default_submitter() == "agent:experiment"


def test_missing_git_falls_back_to_local_identity(monkeypatch):
    monkeypatch.delenv("NVIDB_SUBMITTER", raising=False)
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(identity.subprocess, "run", unavailable)
    monkeypatch.setattr(identity.getpass, "getuser", lambda: "client")
    monkeypatch.setattr(identity.socket, "gethostname", lambda: "laptop")
    assert identity.default_submitter() == f"client@laptop:{os.getpid()}"


def test_remote_submission_carries_client_identity_with_safe_quoting(monkeypatch):
    name = "User's $(echo unsafe) <user@example.test>"
    monkeypatch.setattr(remote, "default_submitter", lambda: name)
    target = {"host": "queue", "nvidb": "nvidb", "keeper": "off"}
    cmd = remote.build_command(target, ["job", "submit", "--vram", "4G", "--", "echo", "ok"])
    assert f"NVIDB_SUBMITTER={name}" in shlex.split(cmd)
    assert cmd.endswith("-- echo ok")
    assert "NVIDB_SUBMITTER" not in remote.build_command(target, ["job", "ls"])
