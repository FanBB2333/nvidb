"""The nvidb-queue skill must only document commands that actually exist.

Agents follow the skill literally, so a stale flag in it turns into a failed
tool call rather than a documentation nit. This walks every `nvidb ...` command
in SKILL.md and checks the real parser accepts it.
"""
import argparse
import re
import shlex
from pathlib import Path

import pytest

from nvidb.sched import cli as sched_cli

SKILL = Path(__file__).resolve().parent.parent / "skills" / "nvidb-queue" / "SKILL.md"
FENCE = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)


def _documented_commands():
    text = SKILL.read_text(encoding="utf-8")
    commands = []
    for block in FENCE.findall(text):
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("nvidb "):
                continue
            # Drop the trailing explanatory comment, but not a `#` inside quotes.
            parts = shlex.split(line, comments=True)
            if "--help" in parts:
                continue
            commands.append((line, parts[1:]))
    return commands


@pytest.fixture(scope="module")
def parser():
    root = argparse.ArgumentParser(prog="nvidb")
    subparsers = root.add_subparsers(dest="command")
    sched_cli.register_parsers(subparsers)
    return root


def test_the_skill_documents_some_commands():
    assert len(_documented_commands()) >= 15


@pytest.mark.parametrize("line,argv", _documented_commands(), ids=lambda v: None)
def test_every_documented_command_parses(parser, line, argv):
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        pytest.fail(f"skill documents a command the CLI rejects: {line!r} ({error})")
    assert getattr(args, "queue_cli", False), line
    assert callable(getattr(args, "func", None)), line


def test_the_skill_names_the_environment_variables_jobs_actually_get():
    """A job's contract with the queue is what the skill tells agents to use."""
    from nvidb.sched.executor import build_run_script

    script = build_run_script(
        job_id=1,
        job_name="x",
        command="true",
        run_dir="/tmp/x",
        workdir=None,
        gpu_ids=[0],
        node_name="n",
    )
    text = SKILL.read_text(encoding="utf-8")
    for variable in (
        "NVIDB_STATUS_FILE",
        "NVIDB_JOB_ID",
        "NVIDB_JOB_NAME",
        "NVIDB_NODE",
        "NVIDB_JOB_DIR",
        "CUDA_VISIBLE_DEVICES",
    ):
        assert f"export {variable}=" in script, f"{variable} is not set for jobs"
        assert variable in text, f"the skill does not mention {variable}"


def test_the_skill_names_the_real_alert_kinds():
    """Agents branch on these strings, so they must match what is raised."""
    source = (
        Path(__file__).resolve().parent.parent / "nvidb" / "sched" / "scheduler.py"
    ).read_text(encoding="utf-8")
    text = SKILL.read_text(encoding="utf-8")
    for kind in (
        "job_failed",
        "job_lost",
        "job_timeout",
        "job_held",
        "launch_failed",
        "node_down",
        "job_retried",
    ):
        assert f'"{kind}"' in source, f"{kind} is never raised"
        assert kind in text, f"the skill does not mention {kind}"
