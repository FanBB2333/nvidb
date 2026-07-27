"""Launching, probing and killing queue jobs on a node.

Jobs run as detached process groups started from a generated `run.sh`. Nothing
is installed on the target machine: the script is delivered over the transport,
and every later interaction is a single shell round trip. A job therefore
survives the client that submitted it - which is exactly what is needed when
several short-lived clients take turns driving the same queue.
"""
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

from .transport import CommandResult, Transport, TransportError

PROBE_MARKER = "NVIDB_PROBE_V1"
PS_MARKER = "PSTABLE"
DEFAULT_JOB_ROOT = ".nvidb/jobs"


@dataclass
class LaunchResult:
    pid: Optional[int]
    pgid: Optional[int]
    run_dir: str
    session_isolated: bool = True
    stdout: str = ""
    stderr: str = ""


@dataclass
class JobProbe:
    """What one shell round trip learned about a single job."""

    job_id: int
    pid: Optional[int] = None
    pgid: Optional[int] = None
    exit_code: Optional[int] = None
    alive: bool = False
    finished_epoch: Optional[int] = None

    @property
    def finished(self) -> bool:
        return self.exit_code is not None


@dataclass
class NodeProbe:
    jobs: Dict[int, JobProbe] = field(default_factory=dict)
    process_groups: Dict[int, int] = field(default_factory=dict)


def build_run_script(
    *,
    job_id: int,
    job_name: str,
    command: str,
    run_dir: str,
    workdir: Optional[str],
    env: Optional[Dict[str, str]] = None,
    gpu_ids: Optional[Sequence[int]] = None,
    node_name: Optional[str] = None,
) -> str:
    """Generate the wrapper script that owns one job's lifetime on a node.

    The user command is embedded verbatim as script text, so it needs no shell
    quoting and multi-line commands work as written. The script records its own
    pid and process-group id before doing anything else, and publishes the exit
    code from an EXIT trap: that way a command containing its own `exit`, or one
    cut short by a signal, still leaves a status behind, and the file appears
    atomically so a probe never reads a half-written value.
    """
    quoted_dir = shlex.quote(run_dir)
    lines = [
        "#!/bin/bash",
        f"# nvidb queue job {job_id}",
        f"NVIDB_JOB_DIR={quoted_dir}",
        'mkdir -p "$NVIDB_JOB_DIR"',
        'echo $$ > "$NVIDB_JOB_DIR/pid"',
        'ps -o pgid= -p $$ 2>/dev/null | tr -d " \\n" > "$NVIDB_JOB_DIR/pgid" || true',
        "nvidb_finish() {",
        "  nvidb_rc=$?",
        # The wall-clock finish time is recorded here rather than inferred from
        # whenever a client next looks, so elapsed times stay honest even if
        # nobody polls for an hour.
        '  printf "%s %s" "$nvidb_rc" "$(date +%s)" > "$NVIDB_JOB_DIR/exit_code.tmp"',
        '  mv "$NVIDB_JOB_DIR/exit_code.tmp" "$NVIDB_JOB_DIR/exit_code"',
        "}",
        "trap nvidb_finish EXIT",
        # Handling the signals explicitly is what lets the EXIT trap run when a
        # job is cancelled, instead of the status vanishing with the process.
        "trap 'exit 143' TERM",
        "trap 'exit 130' INT",
        f"export NVIDB_JOB_ID={shlex.quote(str(job_id))}",
        f"export NVIDB_JOB_NAME={shlex.quote(job_name or '')}",
        'export NVIDB_JOB_DIR="$NVIDB_JOB_DIR"',
        f"export NVIDB_NODE={shlex.quote(node_name or '')}",
    ]

    visible = ",".join(str(index) for index in (gpu_ids or []))
    lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(visible)}")

    for key, value in (env or {}).items():
        if not key or not str(key).replace("_", "").isalnum():
            continue
        lines.append(f"export {key}={shlex.quote(str(value))}")

    if workdir:
        lines.append(f"cd {shlex.quote(workdir)} || exit 127")

    lines.extend(
        [
            "",
            "# --- job command ---",
            command.rstrip("\n"),
            "# --- end job command ---",
        ]
    )
    return "\n".join(lines) + "\n"


class JobExecutor:
    """Drives job processes on one node through a `Transport`."""

    def __init__(self, transport: Transport, *, job_root: str = DEFAULT_JOB_ROOT):
        self.transport = transport
        self.job_root = job_root
        self._home: Optional[str] = None

    # --- paths ------------------------------------------------------------

    def home(self) -> str:
        """Resolve and cache the remote home directory.

        Every stored path is absolute; a `~` would not survive the quoting that
        keeps arbitrary job commands safe.
        """
        if self._home is None:
            result = self.transport.run('printf "%s" "$HOME"', timeout=15)
            home = (result.stdout or "").strip()
            if not home:
                raise TransportError(f"{self.transport.name}: could not resolve $HOME")
            self._home = home.rstrip("/")
        return self._home

    def run_dir(self, job_id: int) -> str:
        root = self.job_root
        if not root.startswith("/"):
            root = f"{self.home()}/{root}"
        return f"{root.rstrip('/')}/{int(job_id)}"

    # --- lifecycle --------------------------------------------------------

    def launch(
        self,
        *,
        job_id: int,
        job_name: str,
        command: str,
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        gpu_ids: Optional[Sequence[int]] = None,
        node_name: Optional[str] = None,
        timeout: float = 60.0,
    ) -> LaunchResult:
        """Start a job detached and return the pid/pgid it reported."""
        run_dir = self.run_dir(job_id)
        script = build_run_script(
            job_id=job_id,
            job_name=job_name,
            command=command,
            run_dir=run_dir,
            workdir=workdir,
            env=env,
            gpu_ids=gpu_ids,
            node_name=node_name,
        )
        import base64

        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        quoted_dir = shlex.quote(run_dir)
        bootstrap = "\n".join(
            [
                f"d={quoted_dir}",
                'mkdir -p "$d" || exit 1',
                f"printf '%s' {shlex.quote(encoded)} | base64 -d > \"$d/run.sh\" || exit 1",
                'chmod +x "$d/run.sh"',
                'rm -f "$d/exit_code" "$d/pid" "$d/pgid" "$d/result.json"',
                ': > "$d/stdout.log"',
                ': > "$d/stderr.log"',
                # setsid puts the job in its own session, so it survives the
                # SSH channel closing and can later be signalled as a group.
                # macOS has no setsid; nohup still detaches, but the job then
                # shares a process group with the login shell and must never be
                # signalled by group id.
                'if command -v setsid >/dev/null 2>&1; then',
                '  ( setsid bash "$d/run.sh" >> "$d/stdout.log" 2>> "$d/stderr.log"'
                " < /dev/null & )",
                '  echo "NVIDB_SETSID=1"',
                "else",
                '  ( nohup bash "$d/run.sh" >> "$d/stdout.log" 2>> "$d/stderr.log"'
                " < /dev/null & )",
                '  echo "NVIDB_SETSID=0"',
                "fi",
                "i=0",
                'while [ "$i" -lt 60 ]; do',
                '  [ -s "$d/pid" ] && break',
                "  sleep 0.05",
                "  i=$((i+1))",
                "done",
                'echo "NVIDB_PID=$(cat "$d/pid" 2>/dev/null | tr -d " \\n\\r")"',
                'echo "NVIDB_PGID=$(cat "$d/pgid" 2>/dev/null | tr -d " \\n\\r")"',
            ]
        )
        result = self.transport.run(bootstrap, timeout=timeout)
        pid = _parse_marker_int(result.stdout, "NVIDB_PID=")
        pgid = _parse_marker_int(result.stdout, "NVIDB_PGID=")
        isolated = _parse_marker_int(result.stdout, "NVIDB_SETSID=") == 1
        if pid is None:
            raise TransportError(
                f"{self.transport.name}: job {job_id} did not report a pid: "
                f"{(result.stderr or result.stdout).strip()[:400]}"
            )
        return LaunchResult(
            pid=pid,
            # Without a private session the group belongs to someone else's
            # shell, so the caller only ever gets a pgid it is safe to kill.
            pgid=pgid if (isolated and pgid is not None) else None,
            run_dir=run_dir,
            session_isolated=isolated,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def probe(
        self,
        specs: Iterable,
        *,
        want_process_table: bool = True,
        timeout: float = 30.0,
    ) -> NodeProbe:
        """Check many jobs, plus the pid→pgid table, in one round trip.

        `specs` is an iterable of `(job_id, run_dir)`. The process table lets the
        caller decide which GPU processes belong to this queue and which are
        someone else's work.
        """
        specs = [(int(job_id), run_dir) for job_id, run_dir in specs if run_dir]
        if not specs and not want_process_table:
            return NodeProbe()

        lines = [
            "nvidb_probe() {",
            '  _id="$1"; _d="$2"',
            '  _pid=$(cat "$_d/pid" 2>/dev/null | tr -d " \\n\\r")',
            '  _pgid=$(cat "$_d/pgid" 2>/dev/null | tr -d " \\n\\r")',
            '  _ec=$(cat "$_d/exit_code" 2>/dev/null | tr -d "\\n\\r")',
            "  _alive=0",
            '  if [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null; then _alive=1; fi',
            '  echo "JOB|$_id|$_pid|$_pgid|$_ec|$_alive"',
            "}",
            f"echo {PROBE_MARKER}",
        ]
        for job_id, run_dir in specs:
            lines.append(f"nvidb_probe {job_id} {shlex.quote(run_dir)}")
        if want_process_table:
            lines.append(f"echo {PS_MARKER}")
            lines.append("ps -eo pid=,pgid= 2>/dev/null || true")

        result = self.transport.run("\n".join(lines), timeout=timeout)
        return parse_probe_output(result.stdout)

    def kill(self, *, pid: Optional[int], pgid: Optional[int], signal: str = "TERM") -> CommandResult:
        """Signal the job's whole process group, falling back to the bare pid."""
        parts = []
        if pgid:
            parts.append(f"kill -{signal} -{int(pgid)} 2>/dev/null")
        if pid:
            parts.append(f"kill -{signal} {int(pid)} 2>/dev/null")
            if not pgid:
                # No private process group to signal, so reach the job's direct
                # children explicitly instead of leaving them orphaned.
                parts.append(f"pkill -{signal} -P {int(pid)} 2>/dev/null")
        if not parts:
            return CommandResult(1, "", "no pid recorded")
        return self.transport.run(" ; ".join(parts) + " ; true", timeout=20)

    def read_log(
        self, run_dir: str, *, stream: str = "stdout", lines: int = 200
    ) -> str:
        name = "stderr.log" if stream == "stderr" else "stdout.log"
        return self.transport.read_file(f"{run_dir}/{name}", tail_lines=lines)

    def read_result(self, run_dir: str):
        """Read the optional `result.json` a job may leave for its consumers."""
        raw = self.transport.read_file(f"{run_dir}/result.json")
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return {"raw": raw[:4000]}

    def remove_run_dir(self, run_dir: str) -> None:
        if not run_dir or run_dir in ("/", "~"):
            return
        self.transport.run(f"rm -rf {shlex.quote(run_dir)}", timeout=20)


def _parse_marker_int(text: str, marker: str) -> Optional[int]:
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith(marker):
            value = line[len(marker):].strip()
            try:
                return int(value)
            except ValueError:
                return None
    return None


def parse_probe_output(text: str) -> NodeProbe:
    """Parse the two-section probe payload into structured state."""
    probe = NodeProbe()
    section = None
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == PROBE_MARKER:
            section = "jobs"
            continue
        if line == PS_MARKER:
            section = "ps"
            continue
        if section == "jobs" and line.startswith("JOB|"):
            parts = line.split("|")
            if len(parts) < 6:
                continue
            try:
                job_id = int(parts[1])
            except ValueError:
                continue
            # The exit-code field is "<status> <epoch seconds>"; older jobs and
            # partially written files may carry only the status.
            status_parts = parts[4].split()
            probe.jobs[job_id] = JobProbe(
                job_id=job_id,
                pid=_maybe_int(parts[2]),
                pgid=_maybe_int(parts[3]),
                exit_code=_maybe_int(status_parts[0]) if status_parts else None,
                alive=parts[5] == "1",
                finished_epoch=_maybe_int(status_parts[1]) if len(status_parts) > 1 else None,
            )
        elif section == "ps":
            parts = line.split()
            if len(parts) < 2:
                continue
            pid = _maybe_int(parts[0])
            pgid = _maybe_int(parts[1])
            if pid is not None and pgid is not None:
                probe.process_groups[pid] = pgid
    return probe


def _maybe_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None
