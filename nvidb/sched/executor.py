"""Launching, probing and killing queue jobs on a node.

Jobs run as detached process groups started from a generated `run.sh`. Nothing
is installed on the target machine: the script is delivered over the transport,
and every later interaction is a single shell round trip. A job therefore
survives the client that submitted it - which is exactly what is needed when
several short-lived clients take turns driving the same queue.
"""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

from .transport import CommandResult, Transport, TransportError

PROBE_MARKER = "NVIDB_PROBE_V1"
PS_MARKER = "PSTABLE"
DEFAULT_JOB_ROOT = ".nvidb/jobs"

# What a POSIX shell will accept on the left of `export NAME=value`.
VALID_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class LaunchResult:
    pid: Optional[int]
    pgid: Optional[int]
    run_dir: str
    session_isolated: bool = True
    # True when this launch found the job already running and took it over
    # rather than starting a second copy. See `JobExecutor.launch`.
    adopted: bool = False
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
    progress: Optional[str] = None

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
        # A job reports its own progress by writing this file; the queue picks
        # up the last line on every probe.
        'export NVIDB_STATUS_FILE="$NVIDB_JOB_DIR/status"',
    ]

    visible = ",".join(str(index) for index in (gpu_ids or []))
    lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(visible)}")

    for key, value in (env or {}).items():
        # Anything a shell would refuse is dropped rather than emitted as a line
        # that fails at runtime where nobody would see it. `nvidb job submit`
        # rejects these up front, so reaching here means a hand-written record.
        if not VALID_ENV_KEY.match(str(key or "")):
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
                # The transport retries a command whose channel died, which can
                # happen after the remote shell already started the job. Adopt
                # that process instead of starting a second copy: the pid must
                # still be alive *and* still be running this job's run.sh, so a
                # recycled pid cannot be mistaken for our own.
                'nvidb_pid=$(cat "$d/pid" 2>/dev/null | tr -d " \\n\\r")',
                'if [ -n "$nvidb_pid" ] && kill -0 "$nvidb_pid" 2>/dev/null && '
                'ps -o command= -p "$nvidb_pid" 2>/dev/null | grep -q "$d/run.sh"; then',
                '  echo "NVIDB_ADOPTED=1"',
                "else",
                f"  printf '%s' {shlex.quote(encoded)} | base64 -d > \"$d/run.sh\" || exit 1",
                '  chmod +x "$d/run.sh"',
                '  rm -f "$d/exit_code" "$d/pid" "$d/pgid" "$d/result.json"',
                '  : > "$d/stdout.log"',
                '  : > "$d/stderr.log"',
                # setsid puts the job in its own session, so it survives the
                # SSH channel closing and can later be signalled as a group.
                # macOS has no setsid; nohup still detaches, but the job then
                # shares a process group with the login shell and must never be
                # signalled by group id. Which one ran is recorded on the node,
                # so an adopting retry reports the same answer.
                '  if command -v setsid >/dev/null 2>&1; then',
                '    echo 1 > "$d/session"',
                '    ( setsid bash "$d/run.sh" >> "$d/stdout.log" 2>> "$d/stderr.log"'
                " < /dev/null & )",
                "  else",
                '    echo 0 > "$d/session"',
                '    ( nohup bash "$d/run.sh" >> "$d/stdout.log" 2>> "$d/stderr.log"'
                " < /dev/null & )",
                "  fi",
                "  i=0",
                '  while [ "$i" -lt 60 ]; do',
                '    [ -s "$d/pid" ] && break',
                "    sleep 0.05",
                "    i=$((i+1))",
                "  done",
                "fi",
                'echo "NVIDB_SETSID=$(cat "$d/session" 2>/dev/null | tr -d " \\n\\r")"',
                'echo "NVIDB_PID=$(cat "$d/pid" 2>/dev/null | tr -d " \\n\\r")"',
                'echo "NVIDB_PGID=$(cat "$d/pgid" 2>/dev/null | tr -d " \\n\\r")"',
            ]
        )
        result = self.transport.run(bootstrap, timeout=timeout)
        pid = _parse_marker_int(result.stdout, "NVIDB_PID=")
        pgid = _parse_marker_int(result.stdout, "NVIDB_PGID=")
        isolated = _parse_marker_int(result.stdout, "NVIDB_SETSID=") == 1
        adopted = _parse_marker_int(result.stdout, "NVIDB_ADOPTED=") == 1
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
            adopted=adopted,
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
            # The status line is free-form text, so it travels on its own line
            # where only the first two fields need splitting.
            '  _st=$(tail -n 1 "$_d/status" 2>/dev/null | tr -d "\\n\\r")',
            '  if [ -n "$_st" ]; then echo "STAT|$_id|$_st"; fi',
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
        if not pid and not pgid:
            return CommandResult(1, "", "no pid recorded")
        return self.transport.run(
            self._signal_command(pid=pid, pgid=pgid, signal=signal), timeout=20
        )

    def reap(
        self,
        *,
        run_dir: str,
        pid: Optional[int],
        pgid: Optional[int] = None,
        grace: int = 5,
    ) -> bool:
        """Kill a process left behind by a job whose record is already final.

        Returns True when something was found and signalled. The pid must still
        be running *this* job's run.sh: a job cancelled hours ago on a machine
        that was offline has a pid the node has long since handed to someone
        else, and killing that would be far worse than leaking a process.
        """
        if not pid or not run_dir:
            return False
        quoted_dir = shlex.quote(run_dir)
        script = "\n".join(
            [
                f"d={quoted_dir}",
                f"pid={int(pid)}",
                'if kill -0 "$pid" 2>/dev/null && '
                'ps -o command= -p "$pid" 2>/dev/null | grep -q "$d/run.sh"; then',
                f"  {self._signal_command(pid=pid, pgid=pgid, signal='TERM')}",
                # The wrapper only runs its EXIT trap once the work it is
                # waiting on returns, so the escalation has to reach the same
                # set of processes rather than just the wrapper again.
                f"  ( sleep {int(grace)}; "
                f"{self._signal_command(pid=pid, pgid=pgid, signal='KILL')} ) "
                ">/dev/null 2>&1 &",
                '  echo "NVIDB_REAPED=1"',
                "else",
                '  echo "NVIDB_REAPED=0"',
                "fi",
            ]
        )
        result = self.transport.run(script, timeout=20)
        return _parse_marker_int(result.stdout, "NVIDB_REAPED=") == 1

    @staticmethod
    def _signal_command(
        *, pid: Optional[int], pgid: Optional[int], signal: str = "TERM"
    ) -> str:
        """Shell to signal a job: its group when it has one, its children when not."""
        parts = []
        if pgid:
            parts.append(f"kill -{signal} -{int(pgid)} 2>/dev/null")
        if pid:
            parts.append(f"kill -{signal} {int(pid)} 2>/dev/null")
            if not pgid:
                # No private process group, so reach the job's own children
                # explicitly instead of leaving the real work running.
                parts.append(f"pkill -{signal} -P {int(pid)} 2>/dev/null")
        return " ; ".join(parts) + " ; true" if parts else "true"

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
        if section == "jobs" and line.startswith("STAT|"):
            # Only the id is delimited; the rest is the status text verbatim.
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            job_id = _maybe_int(parts[1])
            if job_id is None:
                continue
            existing = probe.jobs.get(job_id)
            if existing is None:
                existing = probe.jobs[job_id] = JobProbe(job_id=job_id)
            existing.progress = parts[2].strip() or None
        elif section == "jobs" and line.startswith("JOB|"):
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
            existing = probe.jobs.get(job_id)
            probe.jobs[job_id] = JobProbe(
                job_id=job_id,
                pid=_maybe_int(parts[2]),
                pgid=_maybe_int(parts[3]),
                exit_code=_maybe_int(status_parts[0]) if status_parts else None,
                alive=parts[5] == "1",
                finished_epoch=_maybe_int(status_parts[1]) if len(status_parts) > 1 else None,
                # Section order puts JOB before STAT, but do not lose a status
                # line if that ever changes.
                progress=existing.progress if existing else None,
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
