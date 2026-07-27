"""The nvidb queue scheduler.

`Scheduler.tick()` is the whole engine and it is deliberately reentrant: it
takes a lease on the database, brings the world up to date, and returns. Any
process may call it - a CLI command, the TUI's refresh thread, a cron entry -
and concurrent callers simply find the lease taken and move on. There is no
privileged daemon, so no single process can become a point of failure.

One tick performs three passes:

1. **refresh** - probe every node for GPU state and for the fate of the jobs it
   is running, attributing GPU memory to either this queue or to outside work.
2. **reconcile** - settle finished, lost and timed-out jobs.
3. **dispatch** - place pending jobs onto GPUs that have budget for them.
"""
from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import config as nvidb_config
from . import db as dbm
from .executor import JobExecutor, NodeProbe
from .model import (
    Job,
    Node,
    format_mb,
    parse_size_mb,
    utcnow,
)
from .transport import LocalTransport, SSHTransport, Transport, TransportError

TICK_LOCK = "scheduler"

DEFAULT_SETTINGS = {
    # VRAM left untouched on every GPU so a scheduled job never starves the
    # driver or a co-tenant that grows slightly.
    "headroom_mb": 512,
    "max_jobs_per_gpu": 4,
    "probe_timeout": 25,
    "launch_timeout": 60,
    "tick_min_interval": 3,
    "lock_ttl": 180,
    "job_root": ".nvidb/jobs",
    "default_vram": "0",
    # spread: prefer the emptiest GPU (best for a small heterogeneous cluster)
    # pack:   prefer the fullest GPU that still fits (keeps big cards free)
    "placement": "spread",
    "include_local": False,
}


def load_settings(cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Merge the `queue:` section of config.yml over the defaults."""
    if cfg is None:
        cfg = nvidb_config.load_config()
    raw = (cfg or {}).get("queue") or {}
    settings = dict(DEFAULT_SETTINGS)
    if isinstance(raw, dict):
        for key, default in DEFAULT_SETTINGS.items():
            if key not in raw:
                continue
            value = raw[key]
            if isinstance(default, bool):
                settings[key] = bool(value)
            elif isinstance(default, int) and not isinstance(default, bool):
                try:
                    settings[key] = int(value)
                except (TypeError, ValueError):
                    pass
            else:
                settings[key] = value
    return settings


def node_name_for_server(server: dict) -> str:
    """The stable display name of a configured server."""
    nickname = server.get("nickname") or server.get("description")
    if nickname:
        return str(nickname)
    host = server.get("hostname") or server.get("host") or "unknown"
    return f"{server.get('username', '')}@{host}:{server.get('port', 22)}"


class NodeBackend:
    """Everything the scheduler needs to talk to one node."""

    def __init__(self, name: str, transport: Transport, *, job_root: str, probe_timeout: float = 25.0):
        self.name = name
        self.transport = transport
        self.executor = JobExecutor(transport, job_root=job_root)
        self.probe_timeout = probe_timeout

    def probe_gpus(self) -> dict:
        """Run the bundled NVML agent once and return its JSON payload.

        The agent is shipped inline over the transport, so a node needs nothing
        beyond python3 and the NVIDIA driver.
        """
        from ..nvml import make_nvml_agent_command

        result = self.transport.run(
            make_nvml_agent_command(once=True), timeout=self.probe_timeout
        )
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                return payload
        message = (result.stderr or result.stdout or "").strip()
        return {"ok": False, "error": message[:500] or "NVML agent produced no output"}

    def close(self) -> None:
        try:
            self.transport.close()
        except Exception:
            pass


@dataclass
class GpuBudget:
    """A GPU's schedulable capacity while dispatch is planning placements."""

    node: str
    index: int
    free_mb: int
    jobs: int
    util_percent: int = 0

    def fits(self, vram_mb: int, max_jobs: int) -> bool:
        return self.free_mb >= vram_mb and self.jobs < max_jobs


class Scheduler:
    def __init__(
        self,
        conn,
        *,
        settings: Optional[Dict[str, Any]] = None,
        cfg: Optional[dict] = None,
        backend_factory=None,
        owner: Optional[str] = None,
    ):
        self.conn = conn
        self.cfg = cfg if cfg is not None else nvidb_config.load_config()
        self.settings = settings or load_settings(self.cfg)
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}"
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backends: Dict[str, NodeBackend] = {}
        self._lock = threading.RLock()

    # --- backends ---------------------------------------------------------

    def _default_backend_factory(self, node: Node) -> NodeBackend:
        servers = (self.cfg or {}).get("servers") or []
        server = None
        for candidate in servers:
            if node_name_for_server(candidate) == node.name:
                server = candidate
                break

        if node.hostname in (None, "localhost", "127.0.0.1") and server is None:
            transport: Transport = LocalTransport(name=node.name)
        else:
            server = server or {}
            transport = SSHTransport(
                hostname=node.hostname or server.get("hostname"),
                port=node.port or server.get("port", 22),
                username=node.username or server.get("username"),
                name=node.name,
                auth=server.get("auth", "auto"),
                identityfile=server.get("identityfile"),
                password=server.get("password"),
            )
        return NodeBackend(
            node.name,
            transport,
            job_root=str(self.settings["job_root"]),
            probe_timeout=float(self.settings["probe_timeout"]),
        )

    def backend(self, node: Node) -> NodeBackend:
        with self._lock:
            backend = self._backends.get(node.name)
            if backend is None:
                backend = self._backend_factory(node)
                self._backends[node.name] = backend
            return backend

    def close(self) -> None:
        with self._lock:
            backends = list(self._backends.values())
            self._backends.clear()
        for backend in backends:
            backend.close()

    # --- configuration sync ----------------------------------------------

    def sync_nodes_from_config(self) -> List[str]:
        """Mirror config.yml's server list into the `nodes` table."""
        names = []
        for server in (self.cfg or {}).get("servers") or []:
            name = node_name_for_server(server)
            dbm.upsert_node(
                self.conn,
                name,
                hostname=server.get("hostname") or server.get("host"),
                port=int(server.get("port") or 22),
                username=server.get("username"),
            )
            names.append(name)
        if self.settings.get("include_local"):
            dbm.upsert_node(
                self.conn, "local", hostname="localhost", port=0, username=None
            )
            names.append("local")
        return names

    # --- submission -------------------------------------------------------

    def submit(
        self,
        command: str,
        *,
        name: str = "",
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        gpus: int = 1,
        vram: Any = None,
        priority: int = 0,
        node: Optional[str] = None,
        depends_on: Optional[Sequence[int]] = None,
        max_runtime_s: Optional[int] = None,
        submitter: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        max_retries: int = 0,
        notes: Optional[str] = None,
    ) -> int:
        """Add a job to the queue and return its id."""
        if not command or not command.strip():
            raise ValueError("A job needs a command")
        vram_mb = parse_size_mb(
            vram if vram is not None else self.settings.get("default_vram", 0)
        )
        resolved_node = None
        if node:
            resolved_node = dbm.resolve_node_name(self.conn, node)
            if resolved_node is None:
                known = ", ".join(n.name for n in dbm.get_nodes(self.conn, with_gpus=False))
                raise ValueError(f"Unknown node {node!r}. Known nodes: {known or '(none)'}")

        depends_on = [int(x) for x in (depends_on or [])]
        for dep in depends_on:
            if dbm.get_job(self.conn, dep) is None:
                raise ValueError(f"Dependency job {dep} does not exist")

        job_id = dbm.insert_job(
            self.conn,
            name=name,
            command=command,
            workdir=workdir,
            env=env or {},
            gpus=int(gpus),
            vram_mb=vram_mb,
            priority=int(priority),
            node_constraint=resolved_node,
            depends_on=depends_on,
            max_runtime_s=max_runtime_s,
            submitter=submitter,
            tags=list(tags or []),
            max_retries=int(max_retries),
            notes=notes,
        )
        dbm.add_event(
            self.conn,
            "job_submitted",
            job_id=job_id,
            message=name or command[:80],
            data={
                "gpus": int(gpus),
                "vram_mb": vram_mb,
                "node": resolved_node,
                "submitter": submitter,
            },
        )
        return job_id

    def set_notes(
        self, job_id: int, text: Optional[str], *, append: bool = False
    ) -> Optional[str]:
        """Write, extend or clear a job's annotation; returns the new value.

        Notes stay editable for the whole life of a job, including after it
        finishes, so a client can record what it concluded from the result.
        """
        job = dbm.get_job(self.conn, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        if text is None:
            value = None
        elif append and job.notes:
            value = f"{job.notes} | {text}"
        else:
            value = text
        dbm.update_job(self.conn, job_id, notes=value)
        return value

    def cancel(self, job_id: int, *, reason: str = "cancelled by user") -> bool:
        """Cancel a job, killing its process group when it is already running."""
        job = dbm.get_job(self.conn, job_id)
        if job is None or job.is_terminal:
            return False
        if job.state == "running" and job.node:
            node = dbm.get_node(self.conn, job.node)
            if node is not None:
                try:
                    backend = self.backend(node)
                    self._terminate(backend, job)
                except TransportError:
                    pass  # The node is unreachable; the record still becomes final.
        dbm.update_job(
            self.conn,
            job_id,
            state="cancelled",
            finished_at=utcnow(),
            last_error=reason,
        )
        dbm.add_event(self.conn, "job_cancelled", job_id=job_id, message=reason)
        return True

    def requeue(self, job_id: int) -> bool:
        """Put a terminal job back in the queue as a fresh attempt."""
        job = dbm.get_job(self.conn, job_id)
        if job is None or not job.is_terminal:
            return False
        dbm.update_job(
            self.conn,
            job_id,
            state="pending",
            node=None,
            gpu_ids="",
            remote_pid=None,
            remote_pgid=None,
            run_dir=None,
            exit_code=None,
            started_at=None,
            finished_at=None,
            heartbeat_at=None,
            gpu_mem_mb=None,
            last_error=None,
            attempt=job.attempt,
            # The annotation survives a re-run; the previous attempt's status
            # line does not.
            progress=None,
            progress_at=None,
        )
        dbm.add_event(self.conn, "job_requeued", job_id=job_id, message="requeued")
        return True

    # --- the tick ---------------------------------------------------------

    def tick(self, *, force: bool = False, dispatch: bool = True) -> Dict[str, Any]:
        """Bring the queue up to date once. Safe to call from anywhere."""
        from .model import age_seconds

        summary: Dict[str, Any] = {
            "ran": False,
            "skipped": None,
            "nodes_up": 0,
            "nodes_down": 0,
            "finished": [],
            "dispatched": [],
            "errors": [],
        }

        if not force:
            last = dbm.get_meta(self.conn, "last_tick_at")
            age = age_seconds(last)
            if age is not None and age < float(self.settings["tick_min_interval"]):
                summary["skipped"] = "rate_limited"
                return summary

        if not dbm.acquire_lock(
            self.conn, TICK_LOCK, self.owner, int(self.settings["lock_ttl"])
        ):
            holder = dbm.lock_holder(self.conn, TICK_LOCK)
            summary["skipped"] = "locked"
            summary["lock_owner"] = holder["owner"] if holder else None
            return summary

        try:
            summary["ran"] = True
            self.sync_nodes_from_config()
            for node in dbm.get_nodes(self.conn, with_gpus=False):
                if not node.enabled:
                    continue
                try:
                    up = self._refresh_node(node, summary)
                except TransportError as error:
                    up = False
                    self._mark_node_down(node, str(error))
                except Exception as error:  # a broken node must not stall the queue
                    up = False
                    self._mark_node_down(node, f"{type(error).__name__}: {error}")
                    summary["errors"].append({"node": node.name, "error": str(error)})
                summary["nodes_up" if up else "nodes_down"] += 1

            self._enforce_timeouts(summary)
            if dispatch:
                self._dispatch(summary)
            dbm.set_meta(self.conn, "last_tick_at", utcnow())
        finally:
            dbm.release_lock(self.conn, TICK_LOCK, self.owner)
        return summary

    # --- pass 1: refresh --------------------------------------------------

    def _refresh_node(self, node: Node, summary: Dict[str, Any]) -> bool:
        backend = self.backend(node)
        running = dbm.live_jobs(self.conn, node.name)

        # An unreachable node raises out of here and is marked down by tick().
        # Getting past both calls proves the node answers commands, even when
        # NVML itself is unhappy (driver reload, WSL quirk, no GPU at all).
        payload = backend.probe_gpus()
        probe = backend.executor.probe(
            [(job.id, job.run_dir) for job in running if job.run_dir],
            want_process_table=True,
            timeout=float(self.settings["probe_timeout"]),
        )

        self._reconcile_jobs(node, backend, running, probe, summary)

        still_running = dbm.live_jobs(self.conn, node.name)
        gpus = self._build_gpu_states(node, payload, probe, still_running)
        dbm.replace_node_gpus(self.conn, node.name, gpus)

        error = None if payload.get("ok") else str(payload.get("error") or "")[:300]
        previous = dbm.set_node_state(
            self.conn, node.name, "up", error=error, gpu_count=len(gpus)
        )
        if previous != "up":
            dbm.add_event(
                self.conn,
                "node_up",
                node=node.name,
                message=f"{node.name} is online ({len(gpus)} GPU)",
            )
        return True

    def _mark_node_down(self, node: Node, error: str) -> None:
        previous = dbm.set_node_state(self.conn, node.name, "down", error=error[:300])
        # Stale capacity would let dispatch place jobs on a machine that is gone.
        dbm.replace_node_gpus(self.conn, node.name, [])
        if previous != "down":
            dbm.add_event(
                self.conn, "node_down", node=node.name, message=error[:200]
            )

    # --- pass 2: reconcile ------------------------------------------------

    def _reconcile_jobs(
        self,
        node: Node,
        backend: NodeBackend,
        running: List[Job],
        probe: NodeProbe,
        summary: Dict[str, Any],
    ) -> None:
        for job in running:
            observed = probe.jobs.get(job.id)
            if observed is None:
                continue  # nothing learned this round; leave the job alone

            # The job's own status line is picked up whatever else happened, so
            # the last thing a job said survives into its finished record.
            progress_updates: Dict[str, Any] = {}
            if observed.progress and observed.progress != job.progress:
                progress_updates = {
                    "progress": observed.progress,
                    "progress_at": utcnow(),
                }

            if observed.finished:
                exit_code = observed.exit_code
                state = "completed" if exit_code == 0 else "failed"
                result = None
                try:
                    result = backend.executor.read_result(job.run_dir)
                except TransportError:
                    pass
                dbm.update_job(
                    self.conn,
                    job.id,
                    state=state,
                    exit_code=exit_code,
                    finished_at=_epoch_to_iso(observed.finished_epoch) or utcnow(),
                    result=result,
                    last_error=None if state == "completed" else f"exit code {exit_code}",
                    **progress_updates,
                )
                dbm.add_event(
                    self.conn,
                    "job_finished",
                    job_id=job.id,
                    node=node.name,
                    message=f"{state} (exit {exit_code})",
                    data={"exit_code": exit_code, "state": state},
                )
                summary["finished"].append(
                    {"id": job.id, "state": state, "exit_code": exit_code}
                )
                continue

            if observed.alive:
                updates: Dict[str, Any] = {"heartbeat_at": utcnow(), **progress_updates}
                if observed.pid and observed.pid != job.remote_pid:
                    updates["remote_pid"] = observed.pid
                if observed.pgid and observed.pgid != job.remote_pgid:
                    updates["remote_pgid"] = observed.pgid
                dbm.update_job(self.conn, job.id, **updates)
                continue

            # Neither alive nor holding an exit code: the process vanished.
            if job.attempt <= job.max_retries and job.max_retries > 0:
                dbm.update_job(
                    self.conn,
                    job.id,
                    state="pending",
                    node=None,
                    gpu_ids="",
                    remote_pid=None,
                    remote_pgid=None,
                    run_dir=None,
                    started_at=None,
                    heartbeat_at=None,
                    last_error="process vanished; retrying",
                    # The next attempt starts fresh, so the old status line
                    # must not linger as if it were current.
                    progress=None,
                    progress_at=None,
                )
                dbm.add_event(
                    self.conn,
                    "job_requeued",
                    job_id=job.id,
                    node=node.name,
                    message="process vanished; retrying",
                )
            else:
                dbm.update_job(
                    self.conn,
                    job.id,
                    state="lost",
                    finished_at=utcnow(),
                    last_error="process vanished without an exit code",
                    # Whatever the job last reported is the best clue about how
                    # far it got, so it is kept on the record.
                    **progress_updates,
                )
                dbm.add_event(
                    self.conn,
                    "job_lost",
                    job_id=job.id,
                    node=node.name,
                    message="process vanished without an exit code",
                )
                summary["finished"].append({"id": job.id, "state": "lost"})

    def _enforce_timeouts(self, summary: Dict[str, Any]) -> None:
        for job in dbm.live_jobs(self.conn):
            if not job.max_runtime_s:
                continue
            elapsed = job.elapsed_seconds()
            if elapsed is None or elapsed <= job.max_runtime_s:
                continue
            node = dbm.get_node(self.conn, job.node) if job.node else None
            if node is not None:
                try:
                    self._terminate(self.backend(node), job)
                except TransportError:
                    pass
            dbm.update_job(
                self.conn,
                job.id,
                state="timeout",
                finished_at=utcnow(),
                last_error=f"exceeded max runtime of {job.max_runtime_s}s",
            )
            dbm.add_event(
                self.conn,
                "job_timeout",
                job_id=job.id,
                node=job.node,
                message=f"killed after {int(elapsed)}s",
            )
            summary["finished"].append({"id": job.id, "state": "timeout"})

    def _terminate(self, backend: NodeBackend, job: Job, grace: int = 5) -> None:
        """Ask the process group to stop, escalating to SIGKILL in the background."""
        backend.executor.kill(pid=job.remote_pid, pgid=job.remote_pgid, signal="TERM")
        if job.remote_pgid or job.remote_pid:
            target = f"-{job.remote_pgid}" if job.remote_pgid else str(job.remote_pid)
            backend.transport.run(
                f"( sleep {int(grace)}; kill -KILL {target} 2>/dev/null ) >/dev/null 2>&1 &",
                timeout=10,
            )

    # --- capacity accounting ----------------------------------------------

    def _build_gpu_states(
        self,
        node: Node,
        payload: dict,
        probe: NodeProbe,
        running: List[Job],
    ) -> List[dict]:
        """Turn an NVML snapshot into per-GPU budgets.

        Memory on a card is split into three buckets: what this queue's jobs
        actually use, what everything else uses, and what the queue has promised
        to jobs that have not grown into their reservation yet.
        """
        pgid_to_job = {job.remote_pgid: job.id for job in running if job.remote_pgid}
        pid_to_job = {job.remote_pid: job.id for job in running if job.remote_pid}

        measured: Dict[Tuple[int, int], int] = {}  # (job_id, gpu_index) -> MiB
        states: List[dict] = []

        for entry in payload.get("gpus") or []:
            index = int(entry.get("gpu_index") or 0)
            total_mb = _bytes_to_mb(entry.get("memory_total_bytes"))
            used_mb = _bytes_to_mb(entry.get("memory_used_bytes"))

            processes = entry.get("processes") or []
            ours_mb = 0
            external_procs = 0
            for process in processes:
                pid = process.get("pid")
                process_mb = _bytes_to_mb(process.get("used_gpu_memory_bytes"))
                # A job's GPU process is usually a child of its run.sh, so match
                # on the process group first and fall back to the pid itself.
                job_id = pid_to_job.get(pid)
                if job_id is None:
                    pgid = probe.process_groups.get(pid)
                    job_id = pgid_to_job.get(pgid) if pgid is not None else None
                if job_id is None:
                    external_procs += 1
                    continue
                ours_mb += process_mb
                key = (job_id, index)
                measured[key] = measured.get(key, 0) + process_mb

            on_gpu = [job for job in running if index in job.gpu_ids]
            reserved_mb = sum(
                max(job.vram_mb, measured.get((job.id, index), 0)) for job in on_gpu
            )

            if processes or not on_gpu:
                attribution = "processes"
                external_mb = max(0, used_mb - ours_mb)
            else:
                # NVML named no processes at all yet memory is in use and this
                # queue has jobs here - the WSL case, where the card is driven
                # from Windows. Credit our jobs up to their reservation rather
                # than charging them once as "external" and again as "reserved".
                attribution = "blind"
                external_mb = max(0, used_mb - reserved_mb)

            states.append(
                {
                    "index": index,
                    "name": entry.get("name"),
                    "mem_total_mb": total_mb,
                    "mem_used_mb": used_mb,
                    "util_percent": entry.get("gpu_util_percent"),
                    "temperature_c": entry.get("temperature_c"),
                    "external_mem_mb": external_mb,
                    "external_procs": external_procs,
                    "attribution": attribution,
                    "reserved_mb": reserved_mb,
                    "queue_jobs": len(on_gpu),
                }
            )

        # Record each job's real footprint so the TUI can show promised vs used.
        totals: Dict[int, int] = {}
        for (job_id, _index), value in measured.items():
            totals[job_id] = totals.get(job_id, 0) + value
        for job in running:
            new_value = totals.get(job.id)
            if new_value is not None and new_value != job.gpu_mem_mb:
                dbm.update_job(self.conn, job.id, gpu_mem_mb=new_value)

        return states

    # --- pass 3: dispatch -------------------------------------------------

    def _dispatch(self, summary: Dict[str, Any]) -> None:
        pending = dbm.pending_jobs(self.conn)
        if not pending:
            return

        nodes = [node for node in dbm.get_nodes(self.conn) if node.is_schedulable]
        if not nodes:
            return

        headroom = int(self.settings["headroom_mb"])
        max_jobs = int(self.settings["max_jobs_per_gpu"])
        budgets: Dict[str, List[GpuBudget]] = {}
        for node in nodes:
            budgets[node.name] = [
                GpuBudget(
                    node=node.name,
                    index=gpu.index,
                    free_mb=gpu.free_mb(headroom),
                    jobs=gpu.queue_jobs,
                    util_percent=int(gpu.util_percent or 0),
                )
                for gpu in node.gpus
            ]

        for job in pending:
            ready, problem = self._dependencies_ready(job)
            if problem:
                dbm.update_job(
                    self.conn,
                    job.id,
                    state="failed",
                    finished_at=utcnow(),
                    last_error=problem,
                )
                dbm.add_event(
                    self.conn, "job_failed", job_id=job.id, message=problem
                )
                continue
            if not ready:
                continue

            placement = self._find_placement(job, nodes, budgets, max_jobs)
            if placement is None:
                continue
            node_name, gpu_ids = placement

            node = next((n for n in nodes if n.name == node_name), None)
            if node is None:
                continue
            try:
                self._launch(node, job, gpu_ids)
            except TransportError as error:
                dbm.update_job(
                    self.conn, job.id, last_error=f"launch failed: {error}"[:500]
                )
                dbm.add_event(
                    self.conn,
                    "job_failed",
                    job_id=job.id,
                    node=node_name,
                    message=f"launch failed: {error}"[:200],
                )
                summary["errors"].append({"job": job.id, "error": str(error)})
                continue

            for budget in budgets[node_name]:
                if budget.index in gpu_ids:
                    budget.free_mb = max(0, budget.free_mb - job.vram_mb)
                    budget.jobs += 1
            summary["dispatched"].append(
                {"id": job.id, "node": node_name, "gpu_ids": gpu_ids}
            )

    def _dependencies_ready(self, job: Job) -> Tuple[bool, Optional[str]]:
        """Return (ready, fatal_problem). A failed dependency fails the job."""
        for dep_id in job.depends_on:
            dep = dbm.get_job(self.conn, dep_id)
            if dep is None:
                return False, f"dependency {dep_id} no longer exists"
            if dep.state == "completed":
                continue
            if dep.is_terminal:
                return False, f"dependency {dep_id} ended as {dep.state}"
            return False, None
        return True, None

    def _find_placement(
        self,
        job: Job,
        nodes: List[Node],
        budgets: Dict[str, List[GpuBudget]],
        max_jobs: int,
    ) -> Optional[Tuple[str, List[int]]]:
        """Pick a node and GPU set for one job, or None when nothing fits."""
        candidates = nodes
        if job.node_constraint:
            candidates = [node for node in nodes if node.name == job.node_constraint]
        if not candidates:
            return None

        if job.gpus <= 0:
            # CPU-only work: any live node, least loaded first.
            best = min(
                candidates,
                key=lambda node: sum(b.jobs for b in budgets.get(node.name, [])),
            )
            return best.name, []

        spread = str(self.settings.get("placement", "spread")).lower() != "pack"
        best_choice: Optional[Tuple[Any, str, List[int]]] = None

        for node in candidates:
            usable = [
                budget
                for budget in budgets.get(node.name, [])
                if budget.fits(job.vram_mb, max_jobs)
            ]
            if len(usable) < job.gpus:
                continue
            # Spread hands out the emptiest cards; pack hands out the fullest
            # card that still fits, keeping large GPUs free for large jobs.
            usable.sort(
                key=lambda b: (-b.free_mb if spread else b.free_mb, b.util_percent, b.index)
            )
            chosen = usable[: job.gpus]
            slack = sum(budget.free_mb for budget in chosen)
            rank = (
                (-slack, min(b.util_percent for b in chosen))
                if spread
                else (slack, min(b.util_percent for b in chosen))
            )
            if best_choice is None or rank < best_choice[0]:
                best_choice = (rank, node.name, [budget.index for budget in chosen])

        if best_choice is None:
            return None
        return best_choice[1], best_choice[2]

    def _launch(self, node: Node, job: Job, gpu_ids: List[int]) -> None:
        backend = self.backend(node)
        launched = backend.executor.launch(
            job_id=job.id,
            job_name=job.name,
            command=job.command,
            workdir=job.workdir,
            env=job.env,
            gpu_ids=gpu_ids,
            node_name=node.name,
            timeout=float(self.settings["launch_timeout"]),
        )
        now = utcnow()
        dbm.update_job(
            self.conn,
            job.id,
            state="running",
            node=node.name,
            gpu_ids=gpu_ids,
            remote_pid=launched.pid,
            remote_pgid=launched.pgid,
            run_dir=launched.run_dir,
            started_at=now,
            heartbeat_at=now,
            attempt=job.attempt + 1,
            last_error=None,
        )
        dbm.add_event(
            self.conn,
            "job_started",
            job_id=job.id,
            node=node.name,
            message=f"pid {launched.pid} on GPU {','.join(str(i) for i in gpu_ids) or '-'}",
            data={
                "pid": launched.pid,
                "gpu_ids": gpu_ids,
                "vram_mb": job.vram_mb,
                "vram": format_mb(job.vram_mb),
            },
        )

    # --- reads ------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """A single JSON-friendly view of the whole queue, for other tools."""
        headroom = int(self.settings["headroom_mb"])
        nodes = dbm.get_nodes(self.conn)
        jobs = dbm.list_jobs(self.conn, states=["pending", "running"])
        recent = dbm.list_jobs(self.conn, limit=20, newest_first=True)
        recent_terminal = [job for job in recent if job.is_terminal][:10]
        holder = dbm.lock_holder(self.conn, TICK_LOCK)
        return {
            "generated_at": utcnow(),
            "db_path": str(dbm.default_db_path()),
            "last_tick_at": dbm.get_meta(self.conn, "last_tick_at"),
            "tick_lock": dict(holder) if holder else None,
            "settings": {
                "headroom_mb": headroom,
                "max_jobs_per_gpu": int(self.settings["max_jobs_per_gpu"]),
                "placement": self.settings.get("placement"),
            },
            "counts": dbm.job_counts(self.conn),
            "nodes": [node.to_dict(headroom) for node in nodes],
            "jobs": [job.to_dict() for job in jobs],
            "recent": [job.to_dict() for job in recent_terminal],
        }

    def job_logs(self, job_id: int, *, stream: str = "stdout", lines: int = 200) -> str:
        job = dbm.get_job(self.conn, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        if not job.run_dir or not job.node:
            return ""
        node = dbm.get_node(self.conn, job.node)
        if node is None:
            return ""
        backend = self.backend(node)
        return backend.executor.read_log(job.run_dir, stream=stream, lines=lines)


def _bytes_to_mb(value) -> int:
    try:
        return int(int(value) / (1024 * 1024))
    except (TypeError, ValueError):
        return 0


def _epoch_to_iso(value) -> Optional[str]:
    """Convert a node-reported finish time into the queue's storage format.

    A node clock that is wildly out of step would produce nonsense elapsed
    times, so obviously bad values are discarded in favour of the local clock.
    """
    from datetime import datetime, timezone

    try:
        epoch = int(value)
    except (TypeError, ValueError):
        return None
    if epoch <= 0:
        return None
    moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
    drift = abs((datetime.now(timezone.utc) - moment).total_seconds())
    if drift > 86400:
        return None
    return moment.isoformat(timespec="seconds")
