"""Data model for the nvidb job queue.

The queue is a slurm-like scheduler whose only shared state is a local SQLite
database. Several independent clients (typically Claude Code sessions) submit,
observe and coordinate work purely by reading and writing that file, so every
structure here is designed to round-trip through plain SQL rows and JSON.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# --- Job lifecycle ---------------------------------------------------------
# pending   waiting for a free GPU budget, an online node, or a dependency
# running   launched on a node, the remote process group is alive
# completed the remote command exited 0
# failed    the remote command exited non-zero, or the launch itself failed
# cancelled a client cancelled it before or during execution
# timeout   the job exceeded max_runtime_s and was killed
# lost      the process disappeared without leaving an exit code
JOB_STATES = (
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "timeout",
    "lost",
)
ACTIVE_JOB_STATES = frozenset({"pending", "running"})
TERMINAL_JOB_STATES = frozenset(
    {"completed", "failed", "cancelled", "timeout", "lost"}
)
SUCCESS_JOB_STATES = frozenset({"completed"})

NODE_STATES = ("up", "down", "drain", "unknown")

# Event kinds written to the append-only `events` table. Clients that were not
# running when something happened replay this stream to catch up.
EVENT_KINDS = (
    "job_submitted",
    "job_started",
    "job_finished",
    "job_cancelled",
    "job_requeued",
    # A person changed where the job sits in the dispatch order.
    "job_priority",
    "job_failed",
    "job_lost",
    "job_timeout",
    # A process that outlived the job record - killed once its node answered.
    "job_reaped",
    "node_up",
    "node_down",
    "tick",
)


def utcnow() -> str:
    """Current time as a timezone-aware ISO-8601 string (the storage format)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    """Parse a stored timestamp, tolerating naive values from older rows."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def age_seconds(value: Optional[str], now: Optional[datetime] = None) -> Optional[float]:
    """Seconds elapsed since a stored timestamp, or None when unparseable."""
    parsed = parse_ts(value)
    if parsed is None:
        return None
    reference = now or datetime.now(timezone.utc)
    return (reference - parsed).total_seconds()


_SIZE_PATTERN = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")
_SIZE_UNITS = {
    "": 1,
    "m": 1,
    "mb": 1,
    "mib": 1,
    "g": 1024,
    "gb": 1024,
    "gib": 1024,
    "t": 1024 * 1024,
    "tb": 1024 * 1024,
    "tib": 1024 * 1024,
}


def parse_size_mb(value: Any) -> int:
    """Parse a VRAM size such as `20G`, `20480`, or `512MiB` into MiB.

    Bare numbers are MiB, matching how nvidia-smi reports memory.
    """
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(round(float(value))))
    match = _SIZE_PATTERN.match(str(value))
    if not match:
        raise ValueError(f"Invalid size: {value!r} (try 20G, 512M, or 20480)")
    number, unit = match.groups()
    factor = _SIZE_UNITS.get(unit.lower())
    if factor is None:
        raise ValueError(f"Unknown size unit in {value!r}")
    return max(0, int(round(float(number) * factor)))


def format_mb(value: Optional[int]) -> str:
    """Render MiB compactly: 512M, 4.2G, 73.4G."""
    if value is None:
        return "-"
    value = int(value)
    if value < 1024:
        return f"{value}M"
    gib = value / 1024
    if gib >= 100:
        return f"{gib:.0f}G"
    return f"{gib:.1f}G"


def display_width(text: Optional[str]) -> int:
    """Terminal columns a string occupies.

    Job names and notes are routinely written in Chinese, where one character
    takes two columns; counting characters instead would shear every table that
    contains one.
    """
    if not text:
        return 0
    width = 0
    for char in str(text):
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def fit_display(text: Optional[str], limit: int, *, ellipsis: str = "…") -> str:
    """Truncate to at most `limit` terminal columns, marking any loss."""
    text = "" if text is None else str(text)
    if limit <= 0:
        return ""
    if display_width(text) <= limit:
        return text
    budget = limit - display_width(ellipsis)
    out = []
    used = 0
    for char in text:
        char_width = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        )
        if used + char_width > budget:
            break
        out.append(char)
        used += char_width
    return "".join(out) + ellipsis


def pad_display(text: Optional[str], width: int) -> str:
    """Left-align to `width` terminal columns."""
    text = "" if text is None else str(text)
    return text + " " * max(0, width - display_width(text))


def format_duration(seconds: Optional[float]) -> str:
    """Render a duration as HH:MM:SS (or D-HH:MM:SS past a day), slurm style."""
    if seconds is None:
        return "-"
    seconds = int(max(0, seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _json_loads(value: Optional[str], default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _csv_ints(value: Optional[str]) -> List[int]:
    if not value:
        return []
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


@dataclass
class Job:
    """One queued unit of work."""

    id: int
    name: str
    command: str
    state: str
    workdir: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    priority: int = 0
    gpus: int = 1
    vram_mb: int = 0
    node_constraint: Optional[str] = None
    depends_on: List[int] = field(default_factory=list)
    max_runtime_s: Optional[int] = None
    submitter: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    node: Optional[str] = None
    gpu_ids: List[int] = field(default_factory=list)
    remote_pid: Optional[int] = None
    remote_pgid: Optional[int] = None
    run_dir: Optional[str] = None
    exit_code: Optional[int] = None
    attempt: int = 0
    max_retries: int = 0
    last_error: Optional[str] = None
    heartbeat_at: Optional[str] = None
    gpu_mem_mb: Optional[int] = None
    result: Any = None
    # Free-form annotation written by whoever submitted or inspected the job.
    notes: Optional[str] = None
    # The job's own status line, published by writing $NVIDB_STATUS_FILE.
    progress: Optional[str] = None
    progress_at: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> "Job":
        data = dict(row)
        return cls(
            id=int(data["id"]),
            name=data.get("name") or "",
            command=data.get("command") or "",
            state=data.get("state") or "pending",
            workdir=data.get("workdir"),
            env=_json_loads(data.get("env"), {}) or {},
            priority=int(data.get("priority") or 0),
            gpus=int(data.get("gpus") or 0),
            vram_mb=int(data.get("vram_mb") or 0),
            node_constraint=data.get("node_constraint"),
            depends_on=_csv_ints(data.get("depends_on")),
            max_runtime_s=data.get("max_runtime_s"),
            submitter=data.get("submitter"),
            tags=_json_loads(data.get("tags"), []) or [],
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            node=data.get("node"),
            gpu_ids=_csv_ints(data.get("gpu_ids")),
            remote_pid=data.get("remote_pid"),
            remote_pgid=data.get("remote_pgid"),
            run_dir=data.get("run_dir"),
            exit_code=data.get("exit_code"),
            attempt=int(data.get("attempt") or 0),
            max_retries=int(data.get("max_retries") or 0),
            last_error=data.get("last_error"),
            heartbeat_at=data.get("heartbeat_at"),
            gpu_mem_mb=data.get("gpu_mem_mb"),
            result=_json_loads(data.get("result"), None),
            notes=data.get("notes"),
            progress=data.get("progress"),
            progress_at=data.get("progress_at"),
        )

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_JOB_STATES

    def elapsed_seconds(self, now: Optional[datetime] = None) -> Optional[float]:
        """Wall-clock runtime: since start for live jobs, start→finish once done."""
        started = parse_ts(self.started_at)
        if started is None:
            return None
        if self.finished_at:
            finished = parse_ts(self.finished_at)
            if finished is not None:
                return (finished - started).total_seconds()
        reference = now or datetime.now(timezone.utc)
        return (reference - started).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view; this is the schema other tools consume."""
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "command": self.command,
            "workdir": self.workdir,
            "env": self.env,
            "priority": self.priority,
            "gpus": self.gpus,
            "vram_mb": self.vram_mb,
            "node_constraint": self.node_constraint,
            "depends_on": self.depends_on,
            "max_runtime_s": self.max_runtime_s,
            "submitter": self.submitter,
            "tags": self.tags,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": self.elapsed_seconds(),
            "node": self.node,
            "gpu_ids": self.gpu_ids,
            "remote_pid": self.remote_pid,
            "run_dir": self.run_dir,
            "exit_code": self.exit_code,
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "last_error": self.last_error,
            "heartbeat_at": self.heartbeat_at,
            "gpu_mem_mb": self.gpu_mem_mb,
            "result": self.result,
            "notes": self.notes,
            "progress": self.progress,
            "progress_at": self.progress_at,
        }


@dataclass
class GpuProcess:
    """One process NVML reports on a card, whoever started it.

    The queue schedules around work it did not start, so a machine's real
    occupancy has to be visible: these rows carry the foreign processes too,
    with `job_id` left None when the queue is not the one running them.
    """

    pid: int
    mem_mb: int = 0
    name: Optional[str] = None
    username: Optional[str] = None
    # NVML's process type: "C" compute, "G" graphics, "C+G" both.
    kind: Optional[str] = None
    job_id: Optional[int] = None
    job_name: Optional[str] = None

    @property
    def managed(self) -> bool:
        """True when this queue started the process, so it is already budgeted."""
        return self.job_id is not None

    @property
    def owner(self) -> str:
        if self.job_id is None:
            return "external"
        return f"job {self.job_id}" + (f" {self.job_name}" if self.job_name else "")

    def describe(self) -> str:
        """A one-line `name (user, 6.1G)` summary for compact views."""
        label = self.name or f"pid {self.pid}"
        details = [detail for detail in (self.username, format_mb(self.mem_mb)) if detail]
        return f"{label} ({', '.join(details)})" if details else label

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GpuProcess":
        return cls(
            pid=int(data.get("pid") or 0),
            mem_mb=int(data.get("mem_mb") or 0),
            name=data.get("name"),
            username=data.get("user") or data.get("username"),
            kind=data.get("type") or data.get("kind"),
            job_id=data.get("job_id"),
            job_name=data.get("job_name"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "mem_mb": self.mem_mb,
            "name": self.name,
            "user": self.username,
            "type": self.kind,
            "job_id": self.job_id,
            "job_name": self.job_name,
            "managed": self.managed,
        }


@dataclass
class GpuState:
    """One GPU as of the last successful node probe."""

    node: str
    index: int
    name: Optional[str] = None
    mem_total_mb: int = 0
    mem_used_mb: int = 0
    util_percent: Optional[int] = None
    mem_util_percent: Optional[int] = None
    temperature_c: Optional[int] = None
    external_mem_mb: int = 0
    external_procs: int = 0
    queue_mem_mb: int = 0
    reserved_mb: int = 0
    queue_jobs: int = 0
    # "processes" when NVML named the processes on this card, "blind" when it
    # reported memory in use but no process list (WSL passes the GPU through
    # the Windows driver, so nothing on the Linux side is visible).
    attribution: str = "processes"
    # Every process on the card, queue-owned and foreign alike, biggest first.
    processes: List[GpuProcess] = field(default_factory=list)
    updated_at: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> "GpuState":
        data = dict(row)
        return cls(
            node=data["node"],
            index=int(data["gpu_index"]),
            name=data.get("name"),
            mem_total_mb=int(data.get("mem_total_mb") or 0),
            mem_used_mb=int(data.get("mem_used_mb") or 0),
            util_percent=data.get("util_percent"),
            mem_util_percent=data.get("mem_util_percent"),
            temperature_c=data.get("temperature_c"),
            external_mem_mb=int(data.get("external_mem_mb") or 0),
            external_procs=int(data.get("external_procs") or 0),
            queue_mem_mb=int(data.get("queue_mem_mb") or 0),
            reserved_mb=int(data.get("reserved_mb") or 0),
            queue_jobs=int(data.get("queue_jobs") or 0),
            attribution=data.get("attribution") or "processes",
            processes=[
                GpuProcess.from_dict(item)
                for item in _json_loads(data.get("processes_json"), []) or []
            ],
            updated_at=data.get("updated_at"),
        )

    @property
    def external_processes(self) -> List["GpuProcess"]:
        """The processes the queue does not manage - the ones it schedules around."""
        return [process for process in self.processes if not process.managed]

    def mem_used_percent(self) -> Optional[int]:
        if not self.mem_total_mb:
            return None
        return int(round(100.0 * self.mem_used_mb / self.mem_total_mb))

    def free_mb(self, headroom_mb: int = 0) -> int:
        """Schedulable VRAM: total minus foreign usage, our reservations, headroom.

        `external_mem_mb` covers everything on the card that this queue did not
        put there, so a machine busy with hand-started training still reports an
        honest budget instead of looking empty.
        """
        return max(
            0,
            self.mem_total_mb - self.external_mem_mb - self.reserved_mb - headroom_mb,
        )

    def to_dict(self, headroom_mb: int = 0) -> Dict[str, Any]:
        return {
            "node": self.node,
            "index": self.index,
            "name": self.name,
            "mem_total_mb": self.mem_total_mb,
            "mem_used_mb": self.mem_used_mb,
            "mem_used_percent": self.mem_used_percent(),
            # What is on the card that this queue did not put there, and what it
            # did: a reader can tell real occupancy from queue bookkeeping.
            "external_mem_mb": self.external_mem_mb,
            "external_procs": self.external_procs,
            "queue_mem_mb": self.queue_mem_mb,
            "attribution": self.attribution,
            "reserved_mb": self.reserved_mb,
            "queue_jobs": self.queue_jobs,
            "free_mb": self.free_mb(headroom_mb),
            "util_percent": self.util_percent,
            "mem_util_percent": self.mem_util_percent,
            "temperature_c": self.temperature_c,
            "processes": [process.to_dict() for process in self.processes],
            "updated_at": self.updated_at,
        }


@dataclass
class Node:
    """A machine the queue can dispatch to, mirrored from config.yml."""

    name: str
    hostname: Optional[str] = None
    port: int = 22
    username: Optional[str] = None
    state: str = "unknown"
    enabled: bool = True
    ignored: bool = False
    last_seen: Optional[str] = None
    last_error: Optional[str] = None
    gpu_count: int = 0
    probed_at: Optional[str] = None
    gpus: List[GpuState] = field(default_factory=list)

    @classmethod
    def from_row(cls, row) -> "Node":
        data = dict(row)
        return cls(
            name=data["name"],
            hostname=data.get("hostname"),
            port=int(data.get("port") or 22),
            username=data.get("username"),
            state=data.get("state") or "unknown",
            enabled=bool(data.get("enabled", 1)),
            ignored=bool(data.get("ignored", 0)),
            last_seen=data.get("last_seen"),
            last_error=data.get("last_error"),
            gpu_count=int(data.get("gpu_count") or 0),
            probed_at=data.get("probed_at"),
        )

    @property
    def is_schedulable(self) -> bool:
        return not self.ignored and self.enabled and self.state == "up"

    def to_dict(self, headroom_mb: int = 0) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hostname": self.hostname,
            "port": self.port,
            "username": self.username,
            "state": self.state,
            "enabled": self.enabled,
            "ignored": self.ignored,
            "last_seen": self.last_seen,
            "last_error": self.last_error,
            "gpu_count": self.gpu_count,
            "probed_at": self.probed_at,
            "gpus": [gpu.to_dict(headroom_mb) for gpu in self.gpus],
        }
