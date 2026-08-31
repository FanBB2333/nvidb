import subprocess
import sys

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
