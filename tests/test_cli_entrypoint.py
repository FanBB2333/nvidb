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
