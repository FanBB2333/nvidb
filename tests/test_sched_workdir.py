"""The wrapper resolves working directories on the execution node."""
import os
import subprocess
from pathlib import Path

import pytest

from nvidb.sched.executor import build_run_script


@pytest.mark.parametrize("path_kind", ["home", "home_relative", "absolute", "relative"])
def test_workdir_expands_only_the_nodes_home_prefix(tmp_path, path_kind):
    # Shell metacharacters must stay part of the directory name.
    project = tmp_path / "project's $cache; $(touch injected)"
    project.mkdir()
    home = Path.home()
    workdir, expected = {
        "home": ("~", home),
        "home_relative": ("~/" + os.path.relpath(project, home), project),
        "absolute": (str(project), project),
        "relative": (project.name, project),
    }[path_kind]
    script = build_run_script(
        job_id=1, job_name="workdir", command="pwd -P",
        run_dir=str(tmp_path / "job"), workdir=workdir,
    )
    result = subprocess.run(
        ["bash"], input=script, cwd=tmp_path,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(expected.resolve())
    assert not (tmp_path / "injected").exists()
