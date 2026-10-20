import subprocess
import sys
from types import SimpleNamespace

import pytest

from nvidb.cli import main
from nvidb.test.run import main as legacy_main


def test_public_cli_module_preserves_the_existing_main_function():
    assert main is legacy_main


def test_package_module_exposes_the_cli():
    completed = subprocess.run(
        [sys.executable, "-m", "nvidb", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.startswith("nvidb ")
    assert completed.stderr == ""


@pytest.mark.parametrize("flags, expected", [([], False), (["--debug"], True)])
def test_monitor_debug_is_an_explicit_cli_option(monkeypatch, flags, expected):
    from nvidb.test import run

    seen = []
    monkeypatch.setattr(sys, "argv", ["nvidb", "--no-remote", *flags])
    monkeypatch.setattr(run, "_load_config_yaml", lambda: {})

    def pool(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(print_refresh=lambda: None)

    monkeypatch.setattr(run, "NVClientPool", pool)
    run.main()
    assert seen[0]["debug"] is expected
    assert seen[0]["defer_connect"] is True
    assert seen[0]["dcgm"] is False


def test_dcgm_collection_requires_an_explicit_cli_option(monkeypatch):
    from nvidb.test import run

    seen = []
    monkeypatch.setattr(sys, "argv", ["nvidb", "--no-remote", "--dcgm"])
    monkeypatch.setattr(run, "_load_config_yaml", lambda: {})

    def pool(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(print_refresh=lambda: None)

    monkeypatch.setattr(run, "NVClientPool", pool)
    run.main()
    assert seen[0]["dcgm"] is True
    assert seen[0]["debug"] is False


def test_importing_nvidb_leaves_root_logging_alone():
    completed = subprocess.run(
        [sys.executable, "-c", "import logging, nvidb.config, nvidb.utils; print(logging.root.handlers)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "[]"


@pytest.mark.parametrize(
    "argv",
    [
        ["log", "--db-path", "/X", "ls"],
        ["log", "ls", "--db-path", "/X"],
        ["log", "--db-path", "/X", "info"],
    ],
)
def test_log_db_path_reaches_the_log_subcommands(monkeypatch, argv):
    from nvidb.test import run

    seen = []
    monkeypatch.setattr(sys, "argv", ["nvidb", *argv])
    monkeypatch.setattr(run, "_load_config_yaml", lambda: {})
    monkeypatch.setattr(run, "list_log_sessions", lambda db_path: seen.append(db_path))
    monkeypatch.setattr(run, "show_log_info", lambda session_id, db_path: seen.append(db_path))
    run.main()
    assert seen == ["/X"]


def test_log_info_reports_users_from_the_logged_process_summary(tmp_path, capsys):
    import sqlite3

    from nvidb.logger import NVLoggerSQLite
    from nvidb.test import run

    db_path = tmp_path / "gpu_log.db"
    NVLoggerSQLite(db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO log_sessions (start_time, end_time, status, interval_seconds)"
            " VALUES ('2026-10-20T10:00:00', '2026-10-20T10:00:10', 'completed', 5)"
        )
        for timestamp in ("2026-10-20T10:00:00", "2026-10-20T10:00:05"):
            conn.execute(
                "INSERT INTO gpu_logs (session_id, timestamp, node, gpu_id, processes)"
                " VALUES (1, ?, 'local', 0, 'alice(2048M) bob(4M)')",
                (timestamp,),
            )

    run.show_log_info(db_path=db_path)
    out = capsys.readouterr().out

    assert "No user process data available." not in out
    assert "alice@local                     10s" in out
    assert "alice@local                     2.0 GB" in out
    assert "bob@local                       4 MB" in out


def test_server_commands_accept_a_bare_servers_key(tmp_path, capsys, monkeypatch):
    from nvidb.test import run

    config_path = tmp_path / "config.yml"
    config_path.write_text("servers:\n")

    run.show_servers(config_path)
    assert "No servers configured yet." in capsys.readouterr().out

    answers = iter(["10.0.0.1", "", "alice", "", "", "", "", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    run.interactive_add_server(config_path)

    assert [server["hostname"] for server in run.init(config_path).to_dict()] == ["10.0.0.1"]


def test_remote_init_without_a_config_neither_fails_nor_creates_the_home(tmp_path):
    from nvidb.test import run

    home = tmp_path / "home"
    servers = run.init(home / "config.yml")

    assert servers.to_dict() == []
    assert not home.exists()
