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
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .. import config as nvidb_config
from . import db as dbm
from . import keeper as keeper_mod
from .executor import JobExecutor, LaunchRejected, NodeProbe
from .model import (
    GpuProcess,
    Job,
    Lane,
    Node,
    format_mb,
    make_lane_name,
    parse_lane_name,
    parse_size_mb,
    utcnow,
)
from .transport import LocalTransport, SSHTransport, Transport, TransportError

TICK_LOCK = "scheduler"


class SchedulerLeaseLost(RuntimeError):
    """The scheduler must stop because another owner acquired its lease."""

# How much of a failed job's stderr to keep locally. Enough to see a traceback's
# last frames without turning the queue database into a log store.
FAILURE_LOG_LINES = 25
FAILURE_DETAIL_CHARS = 4000

# GPU processes kept per card, largest first. Enough to explain who is using a
# busy GPU without storing every graphics context a desktop session opens.
MAX_RECORDED_GPU_PROCESSES = 16

DEFAULT_SETTINGS = {
    # VRAM left untouched on every GPU so a scheduled job never starves the
    # driver or a co-tenant that grows slightly.
    "headroom_mb": 512,
    "max_jobs_per_gpu": 4,
    # CPU-only jobs (`--gpus 0`) reserve no VRAM, so nothing else bounds how
    # many of them a node runs at once. This does.
    "max_cpu_jobs_per_node": 4,
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
    # What to call this machine when `include_local` puts it in the queue. Worth
    # setting once the scheduler runs on a node rather than on a laptop, where
    # "local" reads as "wherever I happen to be". Changing it later strands jobs
    # recorded against the old name, so it is a set-it-once value.
    "local_node_name": "local",
}


def load_settings(cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Merge the `queue:` section of the queue's configuration over the defaults."""
    if cfg is None:
        cfg = nvidb_config.load_queue_config()
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


def gpu_allowlists(cfg: Optional[dict]) -> Dict[str, Set[int]]:
    """Per-node GPU allowlists from the server config's optional `gpu_ids:` key.

    A server entry may hand the queue only some of its cards:

        servers:
          - nickname: "shared-box"
            gpu_ids: [0, 1]   # never dispatch queue jobs onto the other cards

    Nodes without the key are unrestricted. Jobs already running on an
    excluded GPU are untouched; the mask only affects new placements.
    """
    out: Dict[str, Set[int]] = {}
    for server in (cfg or {}).get("servers") or []:
        node_name = node_name_for_server(server)
        normalized = nvidb_config.server_gpu_ids(
            server, context=f"server {node_name!r}"
        )
        if normalized is None:
            continue
        out[node_name] = set(normalized)
    return out


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
        self._load_config_on_tick = cfg is None
        self._settings_from_config = settings is None
        self.cfg = cfg if cfg is not None else nvidb_config.load_queue_config()
        self.settings = settings or load_settings(self.cfg)
        self._gpu_allowlists = gpu_allowlists(self.cfg)
        self._configured_nodes = self._configured_node_names()
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        self._last_lease_renewal = 0.0
        self._last_sync_signature = None
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backends: Dict[str, NodeBackend] = {}
        self._lock = threading.RLock()

    # --- backends ---------------------------------------------------------

    def _default_backend_factory(self, node: Node) -> NodeBackend:
        servers = self.cfg.get("servers") or []
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
                proxyjump=server.get("proxyjump"),
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

    def _configured_node_names(self) -> Set[str]:
        names = {
            node_name_for_server(server)
            for server in self.cfg.get("servers") or []
        }
        if self.settings.get("include_local"):
            names.add(str(self.settings.get("local_node_name") or "local"))
        return names

    def _reload_config(self) -> None:
        """Refresh file-backed settings before a daemon scheduler pass."""
        if not self._load_config_on_tick:
            return
        fresh = nvidb_config.load_queue_config()
        if fresh == self.cfg:
            return
        self.cfg = fresh
        if self._settings_from_config:
            self.settings = load_settings(fresh)
        self._gpu_allowlists = gpu_allowlists(fresh)
        # Connection details may have changed alongside the allowlist. Closing
        # an SSH client does not touch detached jobs; the next probe reconnects.
        self.close()

    def sync_nodes_from_config(self) -> List[str]:
        """Mirror the queue configuration's server list into the `nodes` table.

        Runs once per CLI invocation and again at the head of every tick; when
        the configuration has not changed since the last mirror, the writes
        are skipped rather than repeated verbatim.
        """
        self._gpu_allowlists = gpu_allowlists(self.cfg)
        servers = self.cfg.get("servers") or []
        signature = (
            json.dumps(servers, sort_keys=True),
            bool(self.settings.get("include_local")),
            str(self.settings.get("local_node_name") or "local"),
        )
        if signature == self._last_sync_signature:
            return list(self._configured_nodes)
        names = []
        for server in servers:
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
            local_name = str(self.settings.get("local_node_name") or "local")
            dbm.upsert_node(
                self.conn, local_name, hostname="localhost", port=0, username=None
            )
            names.append(local_name)
        self._configured_nodes = set(names)
        self._last_sync_signature = signature
        return names

    # --- administration ---------------------------------------------------

    def _lease_ttl(self) -> int:
        """Lease long enough for one bounded remote operation."""
        return max(
            60,
            int(self.settings["lock_ttl"]),
            int(float(self.settings["probe_timeout"]) * 2 + 30),
            int(float(self.settings["launch_timeout"]) * 2 + 60),
        )

    def _renew_tick_lease(self) -> None:
        # Renewals sit at many points in one tick, most inside per-item loops.
        # One write per third of the TTL keeps the lease held with the same
        # margin while cutting the write traffic by an order of magnitude.
        now = time.monotonic()
        if now - self._last_lease_renewal < self._lease_ttl() / 3:
            return
        if not dbm.acquire_lock(
            self.conn, TICK_LOCK, self.owner, self._lease_ttl()
        ):
            raise SchedulerLeaseLost("scheduler lease was lost during the tick")
        self._last_lease_renewal = now

    @contextmanager
    def _operation_lease(self, operation: str):
        owner = f"{self.owner}:{operation}:{uuid.uuid4().hex}"
        if not dbm.acquire_lock(self.conn, TICK_LOCK, owner, self._lease_ttl()):
            raise RuntimeError(
                "scheduler is busy; retry after the current tick finishes"
            )
        try:
            yield
        finally:
            dbm.release_lock(self.conn, TICK_LOCK, owner)

    def set_node_ignored(self, name: str, ignored: bool) -> None:
        """Ignore or restore a node without racing an active scheduler tick."""
        with self._operation_lease("node-ignore"):
            with dbm.transaction(self.conn):
                if ignored:
                    running = dbm.live_jobs(self.conn, name)
                    if running:
                        ids = ", ".join(str(job.id) for job in running)
                        raise ValueError(
                            f"node {name} has running job(s) {ids}; drain it and "
                            "wait for them to finish before ignoring it"
                        )
                dbm.set_node_ignored(self.conn, name, ignored)

    def set_node_enabled(self, name: str, enabled: bool) -> None:
        """Drain or resume a node without racing a placement decision."""
        with self._operation_lease("node-enabled"):
            dbm.set_node_enabled(self.conn, name, enabled)

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
        depends_any: Optional[Sequence[int]] = None,
        max_runtime_s: Optional[int] = None,
        submitter: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        max_retries: int = 0,
        notes: Optional[str] = None,
        lane: Optional[str] = None,
        at: Any = "tail",
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
            target = dbm.get_node(self.conn, resolved_node)
            if resolved_node not in self._configured_nodes:
                raise ValueError(f"Node {resolved_node!r} is no longer configured")
            if target is not None and target.ignored:
                raise ValueError(
                    f"Node {resolved_node!r} is ignored; run "
                    f"`nvidb queue unignore {resolved_node}` before submitting to it"
                )

        lane_row = None
        if lane:
            lane_row = self.resolve_lane(lane)
            if resolved_node and resolved_node != lane_row.node:
                raise ValueError(
                    f"Lane {lane_row.name!r} is on {lane_row.node!r}, but "
                    f"--node asked for {resolved_node!r}"
                )
            # A lane already names its machine, so pinning follows from it and
            # the placement code needs no special case.
            resolved_node = lane_row.node

        depends_on = [int(x) for x in (depends_on or [])]
        depends_any = [int(x) for x in (depends_any or [])]
        with dbm.transaction(self.conn):
            for dep in depends_on + depends_any:
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
                depends_any=depends_any,
                max_runtime_s=max_runtime_s,
                submitter=submitter,
                tags=list(tags or []),
                max_retries=int(max_retries),
                notes=notes,
                lane=lane_row.name if lane_row else None,
                lane_seq=(
                    dbm.next_lane_seq(self.conn, lane_row.name) if lane_row else None
                ),
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
                    "lane": lane_row.name if lane_row else None,
                    "submitter": submitter,
                },
            )
            # Submitted at the back, then moved: one path decides order, so a
            # job placed by `--at head` sits exactly where a later `lane move`
            # would put it.
            if lane_row and str(at) not in ("tail", "", "None"):
                self._reorder_lane(lane_row.name, job_id, to=at)
        return job_id

    # --- lanes ------------------------------------------------------------

    def resolve_lane(self, query: str) -> Lane:
        """Find a lane by name or unambiguous abbreviation, or explain why not.

        A lane that names a real GPU but has no row yet is created here, so
        `--lane box406:1` works on a machine the queue has not probed since
        the card was added.
        """
        name = dbm.resolve_lane_name(self.conn, query)
        if name is not None:
            lane = dbm.get_lane(self.conn, name)
            if lane is not None:
                return lane

        try:
            node_query, gpu_ids = parse_lane_name(query)
        except ValueError as error:
            known = ", ".join(lane.name for lane in dbm.get_lanes(self.conn))
            raise ValueError(
                f"{error} Known lanes: {known or '(none yet; run a tick first)'}"
            ) from None

        node_name = dbm.resolve_node_name(self.conn, node_query)
        if node_name is None:
            raise ValueError(f"Unknown node {node_query!r} in lane {query!r}")
        name = make_lane_name(node_name, gpu_ids)
        dbm.ensure_lane(self.conn, name, node=node_name, gpu_ids=gpu_ids)
        lane = dbm.get_lane(self.conn, name)
        if lane is None:  # pragma: no cover - the insert above just created it
            raise ValueError(f"Could not create lane {name!r}")
        return lane

    def _reorder_lane(
        self,
        lane_name: str,
        job_id: int,
        *,
        to: Any = None,
        before: Optional[int] = None,
        after: Optional[int] = None,
    ) -> int:
        """Move one pending job within its lane and return its new position.

        Positions are 1-based and count only the pending jobs: a running job
        cannot be reordered, so counting it would make `--to 1` mean different
        things depending on whether the lane happened to be busy.
        """
        queue = [
            job
            for job in dbm.lane_jobs(self.conn, lane_name, states=["pending"])
        ]
        ids = [job.id for job in queue]
        if job_id not in ids:
            raise ValueError(
                f"Job {job_id} is not queued in lane {lane_name!r} "
                "(only pending jobs can be reordered)"
            )
        index = ids.index(job_id)

        if before is not None or after is not None:
            anchor = int(before if before is not None else after)
            if anchor == job_id:
                raise ValueError("A job cannot be moved relative to itself")
            if anchor not in ids:
                raise ValueError(
                    f"Job {anchor} is not queued in lane {lane_name!r}"
                )
            ids.pop(index)
            position = ids.index(anchor)
            target = position if before is not None else position + 1
        else:
            text = str(to).lower()
            if text == "head":
                target = 0
            elif text == "tail":
                target = len(ids) - 1
            else:
                try:
                    target = int(text) - 1
                except ValueError:
                    raise ValueError(
                        f"Invalid position {to!r}; use head, tail or a 1-based slot"
                    ) from None
                target = max(0, min(len(ids) - 1, target))
            ids.pop(index)

        ids.insert(target, job_id)
        dbm.renumber_lane(self.conn, lane_name, ids)
        return target + 1

    def lane_move(
        self,
        job_id: int,
        *,
        to: Any = None,
        before: Optional[int] = None,
        after: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Move a queued job within its lane."""
        given = [value for value in (to, before, after) if value is not None]
        if len(given) != 1:
            raise ValueError("Pass exactly one of --to, --before or --after")
        with self._operation_lease("lane-move"):
            job = dbm.get_job(self.conn, job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found")
            if not job.lane:
                raise ValueError(
                    f"Job {job_id} is not in a lane; assign it with "
                    f"`nvidb queue lane assign {job_id} <lane>`"
                )
            if job.state != "pending":
                raise ValueError(
                    f"Job {job_id} is {job.state}; only queued jobs can be reordered"
                )
            position = self._reorder_lane(
                job.lane, job_id, to=to, before=before, after=after
            )
            dbm.add_event(
                self.conn,
                "lane_reordered",
                job_id=job_id,
                message=f"{job.lane}: moved to slot {position}",
            )
            return {"lane": job.lane, "position": position}

    def lane_swap(self, first_id: int, second_id: int) -> Dict[str, Any]:
        """Exchange two queued jobs' places in their lane."""
        with self._operation_lease("lane-swap"):
            first = dbm.get_job(self.conn, first_id)
            second = dbm.get_job(self.conn, second_id)
            for job in (first, second):
                if job is None:
                    raise ValueError("Both jobs must exist")
                if job.state != "pending":
                    raise ValueError(
                        f"Job {job.id} is {job.state}; only queued jobs can be swapped"
                    )
            if not first.lane or first.lane != second.lane:
                raise ValueError("Both jobs must be queued in the same lane")

            ids = [job.id for job in dbm.lane_jobs(self.conn, first.lane, states=["pending"])]
            left, right = ids.index(first_id), ids.index(second_id)
            ids[left], ids[right] = ids[right], ids[left]
            dbm.renumber_lane(self.conn, first.lane, ids)
            dbm.add_event(
                self.conn,
                "lane_reordered",
                job_id=first_id,
                message=f"{first.lane}: swapped with job {second_id}",
            )
            return {"lane": first.lane, "order": ids}

    def lane_assign(
        self, job_id: int, lane: Optional[str], *, at: Any = "tail"
    ) -> Dict[str, Any]:
        """Put a queued job into a lane, move it between lanes, or take it out.

        Passing `lane=None` returns the job to the free pool, where the
        scheduler places it by budget as it did before lanes existed.
        """
        with self._operation_lease("lane-assign"):
            job = dbm.get_job(self.conn, job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found")
            if job.state != "pending":
                raise ValueError(
                    f"Job {job_id} is {job.state}; only queued jobs can be assigned"
                )
            previous = job.lane
            if lane is None:
                dbm.update_job(self.conn, job_id, lane=None, lane_seq=None)
                dbm.add_event(
                    self.conn,
                    "lane_assigned",
                    job_id=job_id,
                    message=f"removed from lane {previous or '-'}",
                )
                return {"lane": None, "previous": previous}

            target = self.resolve_lane(lane)
            if job.gpus > len(target.gpu_ids) and job.gpus > 0:
                raise ValueError(
                    f"Job {job_id} needs {job.gpus} GPUs; lane {target.name!r} "
                    f"has {len(target.gpu_ids)}"
                )
            dbm.update_job(
                self.conn,
                job_id,
                lane=target.name,
                lane_seq=dbm.next_lane_seq(self.conn, target.name),
                node_constraint=target.node,
            )
            position = None
            if str(at) not in ("tail", "", "None"):
                position = self._reorder_lane(target.name, job_id, to=at)
            dbm.add_event(
                self.conn,
                "lane_assigned",
                job_id=job_id,
                message=f"{previous or '-'} -> {target.name}",
            )
            return {"lane": target.name, "previous": previous, "position": position}

    def set_lane_paused(self, lane: str, paused: bool) -> Lane:
        """Stop or resume starting new jobs in a lane.

        Pausing never touches what is already running: the lane finishes its
        current job and then stops, which is what makes it safe to pause a
        lane in order to rearrange what comes after.
        """
        target = self.resolve_lane(lane)
        dbm.update_lane(self.conn, target.name, paused=paused)
        dbm.add_event(
            self.conn,
            "lane_paused" if paused else "lane_resumed",
            node=target.node,
            message=target.name,
        )
        return dbm.get_lane(self.conn, target.name)

    def set_lane_concurrency(self, lane: str, concurrency: int) -> Lane:
        if int(concurrency) < 1:
            raise ValueError("A lane must be allowed to run at least one job")
        target = self.resolve_lane(lane)
        dbm.update_lane(self.conn, target.name, concurrency=int(concurrency))
        return dbm.get_lane(self.conn, target.name)

    def lane_skip(self, lane: str) -> Dict[str, Any]:
        """Send a lane's next job to the back, so the one behind it runs first.

        The way past a head job that is waiting on something, without
        cancelling it or rewriting the rest of the order by hand.
        """
        with self._operation_lease("lane-skip"):
            target = self.resolve_lane(lane)
            ids = [
                job.id
                for job in dbm.lane_jobs(self.conn, target.name, states=["pending"])
            ]
            if len(ids) < 2:
                return {"lane": target.name, "skipped": None}
            head = ids.pop(0)
            ids.append(head)
            dbm.renumber_lane(self.conn, target.name, ids)
            dbm.add_event(
                self.conn,
                "lane_reordered",
                job_id=head,
                message=f"{target.name}: skipped to the back",
            )
            return {"lane": target.name, "skipped": head, "order": ids}

    def lane_view(self, lane: Lane) -> Dict[str, Any]:
        """One lane as reported to the UIs: its order and why it is or is not moving."""
        jobs = dbm.lane_jobs(self.conn, lane.name)
        running = [job for job in jobs if job.state == "running"]
        queued = [job for job in jobs if job.state == "pending"]
        node = dbm.get_node(self.conn, lane.node)

        blocked = None
        if lane.paused:
            blocked = "paused"
        elif node is None:
            blocked = f"node {lane.node} is not configured"
        elif node.ignored:
            blocked = f"node {lane.node} is ignored"
        elif not node.enabled:
            blocked = f"node {lane.node} is drained"
        elif node.state != "up":
            blocked = f"node {lane.node} is {node.state}"
        elif queued and queued[0].is_held:
            blocked = f"job {queued[0].id} is held: {queued[0].held_reason}"

        return {
            **lane.to_dict(),
            "node_state": node.state if node else "unknown",
            "running": [job.to_dict() for job in running],
            "queued": [job.to_dict() for job in queued],
            "blocked": blocked,
        }

    def lanes(self, *, node: Optional[str] = None) -> List[Dict[str, Any]]:
        return [self.lane_view(lane) for lane in dbm.get_lanes(self.conn, node)]

    # --- dependencies -----------------------------------------------------

    def _dead_dependencies(self, job: Job) -> List[Job]:
        """Prerequisites of `job` that ended in a way it can never wait out.

        Only `depends_on` can produce these: `depends_any` is satisfied by any
        outcome, so a failure there is simply the answer it was waiting for.
        """
        dead = []
        for dep_id in job.depends_on:
            dep = dbm.get_job(self.conn, dep_id)
            if dep is None or (dep.is_terminal and dep.state != "completed"):
                dead.append(dep or Job(id=dep_id, name="", command="", state="missing"))
        return dead

    def edit_dependencies(
        self,
        job_id: int,
        *,
        add: Optional[Sequence[int]] = None,
        drop: Optional[Sequence[int]] = None,
        add_any: Optional[Sequence[int]] = None,
        drop_any: Optional[Sequence[int]] = None,
    ) -> Job:
        """Rewire what a job waits for, so a dead chain can be reattached.

        Editing is what turns a held job back into a runnable one without
        resubmitting it, which is the whole point of holding rather than
        failing: the job, its notes and its queue position all survive.
        """
        with dbm.transaction(self.conn):
            job = dbm.get_job(self.conn, job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found")
            if job.is_terminal:
                raise ValueError(
                    f"Job {job_id} already finished as {job.state}; "
                    "its dependencies no longer decide anything"
                )

            wanted = {
                "depends_on": (list(job.depends_on), add or [], drop or []),
                "depends_any": (list(job.depends_any), add_any or [], drop_any or []),
            }
            resolved: Dict[str, List[int]] = {}
            for column, (current, additions, removals) in wanted.items():
                values = list(current)
                for dep in (int(x) for x in removals):
                    while dep in values:
                        values.remove(dep)
                for dep in (int(x) for x in additions):
                    if dep == job_id:
                        raise ValueError("A job cannot depend on itself")
                    if dbm.get_job(self.conn, dep) is None:
                        raise ValueError(f"Dependency job {dep} does not exist")
                    if self._would_cycle(job_id, dep):
                        raise ValueError(
                            f"Job {dep} already waits on job {job_id}; "
                            "that dependency would deadlock both"
                        )
                    if dep not in values:
                        values.append(dep)
                resolved[column] = values

            updates: Dict[str, Any] = {}
            for column, values in resolved.items():
                if values != getattr(job, column):
                    updates[column] = values
            if not updates:
                return job

            # Whatever the hold was about, the answer has changed. The next
            # dispatch decides afresh instead of inheriting a stale reason.
            if job.held_reason:
                updates["held_reason"] = None
            dbm.update_job(self.conn, job_id, **updates)
            dbm.add_event(
                self.conn,
                "job_dependencies",
                job_id=job_id,
                message=(
                    "after="
                    + (",".join(str(i) for i in resolved["depends_on"]) or "-")
                    + " after-any="
                    + (",".join(str(i) for i in resolved["depends_any"]) or "-")
                ),
            )
            return dbm.get_job(self.conn, job_id)

    def _would_cycle(self, job_id: int, new_dependency: int) -> bool:
        """True when `new_dependency` already waits, directly or not, on `job_id`."""
        seen: Set[int] = set()
        stack = [int(new_dependency)]
        while stack:
            current = stack.pop()
            if current == job_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            dep = dbm.get_job(self.conn, current)
            if dep is None:
                continue
            stack.extend(dep.depends_on)
            stack.extend(dep.depends_any)
        return False

    def release(self, job_id: int) -> Dict[str, Any]:
        """Let a held job run by forgetting the prerequisites that died.

        The dead entries are dropped rather than merely ignored, so the release
        survives the next dispatch instead of being re-decided every tick.
        """
        with dbm.transaction(self.conn):
            job = dbm.get_job(self.conn, job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found")
            if job.is_terminal:
                return {
                    "released": False,
                    "reason": f"job {job_id} already finished as {job.state}",
                    "dropped": [],
                }
            dead = self._dead_dependencies(job)
            dropped = [dep.id for dep in dead]
            if not dropped and not job.held_reason:
                return {
                    "released": False,
                    "reason": f"job {job_id} is not held",
                    "dropped": [],
                }
            updates: Dict[str, Any] = {"held_reason": None}
            if dropped:
                updates["depends_on"] = [
                    dep for dep in job.depends_on if dep not in dropped
                ]
            dbm.update_job(self.conn, job_id, **updates)
            message = (
                "released; dropped dependency "
                + ", ".join(str(dep) for dep in dropped)
                if dropped
                else "released"
            )
            dbm.add_event(self.conn, "job_released", job_id=job_id, message=message)
            return {"released": True, "reason": message, "dropped": dropped}

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

    def _change_priority(
        self, job_id: int, value: int, *, relative: bool
    ) -> Optional[int]:
        """Apply one priority change and its event in a single write transaction."""
        with dbm.transaction(self.conn):
            job = dbm.get_job(self.conn, job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found")
            if job.is_terminal:
                return None
            priority = job.priority + int(value) if relative else int(value)
            if priority != job.priority:
                dbm.update_job(self.conn, job_id, priority=priority)
                dbm.add_event(
                    self.conn,
                    "job_priority",
                    job_id=job_id,
                    message=f"priority {job.priority} -> {priority}",
                )
            return priority

    def set_priority(self, job_id: int, priority: int) -> Optional[int]:
        """Set a job's priority; returns the new value, or None when the job
        is finished (its place in the queue no longer exists to change)."""
        return self._change_priority(job_id, priority, relative=False)

    def adjust_priority(self, job_id: int, delta: int) -> Optional[int]:
        """Nudge a job's priority up or down; returns the new value."""
        return self._change_priority(job_id, delta, relative=True)

    def move_pending(self, job_id: int, delta: int) -> bool:
        """Move a pending job by `delta` slots in the dispatch order.

        Dispatch order is `priority DESC, id ASC`, so an arbitrary manual
        order can only be persisted through priorities. The new order is
        written back by walking the list once and lowering the priority of
        whichever job would otherwise sit out of place, so only the jobs
        from the insertion point onwards can change.
        """
        delta = int(delta)
        if not delta:
            return False
        with dbm.transaction(self.conn):
            # Lane jobs are ordered by their lane, not by priority. Letting
            # this rewrite their priorities would change nothing they obey
            # while quietly reshuffling the free pool around them.
            pending = [job for job in dbm.pending_jobs(self.conn) if not job.lane]
            ids = [job.id for job in pending]
            if job_id not in ids:
                return False
            index = ids.index(job_id)
            target = max(0, min(len(pending) - 1, index + delta))
            if target == index:
                return False
            moved = pending.pop(index)
            pending.insert(target, moved)
            previous = None
            for job in pending:
                if previous is not None:
                    ordered = job.priority < previous.priority or (
                        job.priority == previous.priority and job.id > previous.id
                    )
                    if not ordered:
                        job.priority = (
                            previous.priority
                            if job.id > previous.id
                            else previous.priority - 1
                        )
                        dbm.update_job(self.conn, job.id, priority=job.priority)
                previous = job
            dbm.add_event(
                self.conn,
                "job_priority",
                job_id=job_id,
                message=f"moved {'up' if delta < 0 else 'down'} to slot {target + 1}",
            )
        return True

    def cancel(self, job_id: int, *, reason: str = "cancelled by user") -> bool:
        """Cancel a job, killing its process group when it is already running."""
        with self._operation_lease("cancel"):
            job = dbm.get_job(self.conn, job_id)
            if job is None or job.is_terminal:
                return False
            orphan_reason = None
            if job.state == "running" and job.node:
                node = dbm.get_node(self.conn, job.node)
                if node is not None:
                    try:
                        self._terminate(self.backend(node), job)
                    except TransportError:
                        orphan_reason = "cancelled while unreachable"
            with dbm.transaction(self.conn):
                if orphan_reason:
                    self._remember_orphan(job, reason=orphan_reason)
                dbm.update_job(
                    self.conn,
                    job_id,
                    state="cancelled",
                    finished_at=utcnow(),
                    last_error=reason,
                )
                dbm.add_event(
                    self.conn, "job_cancelled", job_id=job_id, message=reason
                )
            return True

    def requeue(self, job_id: int) -> bool:
        """Put a terminal job back in the queue as a fresh attempt."""
        with self._operation_lease("requeue"):
            with dbm.transaction(self.conn):
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
                    held_reason=None,
                    attempt=job.attempt,
                    # The annotation survives a re-run; the previous attempt's
                    # status line does not.
                    progress=None,
                    progress_at=None,
                )
                dbm.add_event(
                    self.conn, "job_requeued", job_id=job_id, message="requeued"
                )
            return True

    # --- the tick ---------------------------------------------------------

    def tick(self, *, force: bool = False) -> Dict[str, Any]:
        """Bring the queue up to date once. Safe to call from anywhere."""
        from .model import age_seconds

        summary: Dict[str, Any] = {
            "ran": False,
            "skipped": None,
            "nodes_up": 0,
            "nodes_down": 0,
            "nodes_ignored": 0,
            "finished": [],
            "dispatched": [],
            "errors": [],
            "alerts": [],
        }

        if not force:
            last = dbm.get_meta(self.conn, "last_tick_at")
            age = age_seconds(last)
            if age is not None and age < float(self.settings["tick_min_interval"]):
                summary["skipped"] = "rate_limited"
                return summary

        if not dbm.acquire_lock(
            self.conn, TICK_LOCK, self.owner, self._lease_ttl()
        ):
            holder = dbm.lock_holder(self.conn, TICK_LOCK)
            summary["skipped"] = "locked"
            summary["lock_owner"] = holder["owner"] if holder else None
            return summary
        self._last_lease_renewal = time.monotonic()

        try:
            summary["ran"] = True
            self._reload_config()
            self._renew_tick_lease()
            self.sync_nodes_from_config()
            # Drained nodes are refreshed too. Draining stops *dispatch* onto a
            # node, and a node is usually drained precisely so its running jobs
            # can finish - which they can only be seen to do if it is probed.
            for node in dbm.get_nodes(self.conn, with_gpus=False):
                self._renew_tick_lease()
                # Ignoring is stronger than draining: it is an explicit request
                # to make no connection attempts until the node is unignored.
                if node.ignored:
                    summary["nodes_ignored"] += 1
                    continue
                # A removed server must not receive new work. Keep observing it
                # only while a running job or deferred cleanup still needs it.
                if (
                    node.name not in self._configured_nodes
                    and not dbm.live_jobs(self.conn, node.name)
                    and not dbm.list_orphans(self.conn, node.name)
                ):
                    continue
                try:
                    up = self._refresh_node(node, summary)
                except SchedulerLeaseLost:
                    raise
                except TransportError as error:
                    up = False
                    self._mark_node_down(node, str(error), summary)
                except Exception as error:  # a broken node must not stall the queue
                    up = False
                    self._mark_node_down(
                        node, f"{type(error).__name__}: {error}", summary
                    )
                    summary["errors"].append({"node": node.name, "error": str(error)})
                summary["nodes_up" if up else "nodes_down"] += 1

            self._renew_tick_lease()
            self._enforce_timeouts(summary)
            self._renew_tick_lease()
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
        self._renew_tick_lease()
        probe = backend.executor.probe(
            [(job.id, job.run_dir) for job in running if job.run_dir],
            timeout=float(self.settings["probe_timeout"]),
        )
        self._renew_tick_lease()

        self._reconcile_jobs(node, backend, running, probe, summary)
        self._reap_orphans(node, backend, summary)

        still_running = dbm.live_jobs(self.conn, node.name)
        gpus = self._build_gpu_states(node, payload, probe, still_running)
        dbm.replace_node_gpus(self.conn, node.name, gpus)
        self._ensure_lanes(node, gpus)

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

    def _ensure_lanes(self, node: Node, gpus: List[dict]) -> None:
        """Give every card this queue may schedule onto a lane of its own.

        Created empty and unused: a lane only decides anything for jobs that
        were submitted to it, so existing work is unaffected. Lanes are never
        removed automatically, because a card that disappears for one probe
        must not take a queue of pending jobs with it.
        """
        mask = self._gpu_allowlists.get(node.name)
        for gpu in gpus:
            index = int(gpu.get("index", 0))
            if mask is not None and index not in mask:
                continue
            dbm.ensure_lane(
                self.conn,
                make_lane_name(node.name, [index]),
                node=node.name,
                gpu_ids=[index],
            )

    def _mark_node_down(
        self, node: Node, error: str, summary: Optional[Dict[str, Any]] = None
    ) -> None:
        previous = dbm.set_node_state(self.conn, node.name, "down", error=error[:300])
        # Stale capacity would let dispatch place jobs on a machine that is gone.
        dbm.replace_node_gpus(self.conn, node.name, [])
        if previous == "down":
            return  # already reported; do not alert once per tick
        dbm.add_event(self.conn, "node_down", node=node.name, message=error[:200])
        stranded = dbm.live_jobs(self.conn, node.name)
        title = f"{node.name} is unreachable"
        if stranded:
            title += f"; {len(stranded)} running job(s) cannot be checked"
        self._raise_alert(
            "node_down",
            title,
            node=node.name,
            detail=error[:FAILURE_DETAIL_CHARS],
            summary=summary,
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
            self._renew_tick_lease()
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
                self._renew_tick_lease()

                detail = None
                last_error = None
                if state == "failed":
                    detail = self._failure_detail(backend, job)
                    # The first line of the reason belongs on the job row; the
                    # whole tail goes to the alert.
                    reason = (detail or "").strip().splitlines()
                    last_error = f"exit code {exit_code}"
                    if reason:
                        last_error += f": {reason[-1][:200]}"

                dbm.update_job(
                    self.conn,
                    job.id,
                    state=state,
                    exit_code=exit_code,
                    finished_at=_epoch_to_iso(observed.finished_epoch) or utcnow(),
                    result=result,
                    last_error=last_error,
                    **progress_updates,
                )
                if state == "failed":
                    self._raise_alert(
                        "job_failed",
                        f"{job.name or 'job'} failed with exit code {exit_code}",
                        job_id=job.id,
                        node=node.name,
                        detail=detail,
                        summary=summary,
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
                self._raise_alert(
                    "job_retried",
                    f"{job.name or 'job'} vanished on {node.name}; retrying "
                    f"(attempt {job.attempt + 1} of {job.max_retries + 1})",
                    severity="warning",
                    job_id=job.id,
                    node=node.name,
                    detail=self._failure_detail(backend, job),
                    summary=summary,
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
                self._raise_alert(
                    "job_lost",
                    f"{job.name or 'job'} vanished on {node.name} without an exit code",
                    job_id=job.id,
                    node=node.name,
                    detail=self._failure_detail(backend, job),
                    summary=summary,
                )
                summary["finished"].append({"id": job.id, "state": "lost"})

    def _remember_orphan(self, job: Job, *, reason: str) -> None:
        """Record a process this queue closed the books on but could not kill."""
        if not job.node:
            return
        dbm.add_orphan(
            self.conn,
            node=job.node,
            job_id=job.id,
            run_dir=job.run_dir,
            pid=job.remote_pid,
            pgid=job.remote_pgid,
            reason=reason,
        )

    def _reap_orphans(
        self, node: Node, backend: NodeBackend, summary: Dict[str, Any]
    ) -> None:
        """Kill processes belonging to jobs whose records were already closed.

        A job cancelled or timed out while its node was unreachable keeps its
        GPU until someone notices. This is the "someone": the first probe that
        gets through finishes the job the cancel could not.
        """
        for orphan in dbm.list_orphans(self.conn, node.name):
            self._renew_tick_lease()
            try:
                killed = backend.executor.reap(
                    run_dir=orphan["run_dir"],
                    pid=orphan["pid"],
                    pgid=orphan["pgid"],
                )
            except TransportError:
                continue  # still unreachable; the row keeps it on the list
            self._renew_tick_lease()
            dbm.drop_orphan(self.conn, orphan["id"])
            if not killed:
                continue  # already gone on its own, which is the usual case
            dbm.add_event(
                self.conn,
                "job_reaped",
                job_id=orphan["job_id"],
                node=node.name,
                message=f"killed a leftover process ({orphan['reason']})",
            )
            summary.setdefault("reaped", []).append(
                {"job": orphan["job_id"], "node": node.name, "pid": orphan["pid"]}
            )

    def _enforce_timeouts(self, summary: Dict[str, Any]) -> None:
        for job in dbm.live_jobs(self.conn):
            self._renew_tick_lease()
            if not job.max_runtime_s:
                continue
            elapsed = job.elapsed_seconds()
            if elapsed is None or elapsed <= job.max_runtime_s:
                continue
            node = dbm.get_node(self.conn, job.node) if job.node else None
            detail = None
            if node is None:
                self._remember_orphan(job, reason="timed out with its node gone")
            else:
                backend = self.backend(node)
                try:
                    detail = self._failure_detail(backend, job)
                    self._terminate(backend, job)
                except TransportError:
                    # Unreachable: the kill is retried on the reap pass.
                    self._remember_orphan(job, reason="timed out while unreachable")
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
            self._raise_alert(
                "job_timeout",
                f"{job.name or 'job'} killed after {int(elapsed)}s "
                f"(limit {job.max_runtime_s}s)",
                job_id=job.id,
                node=job.node,
                detail=detail,
                summary=summary,
            )
            summary["finished"].append({"id": job.id, "state": "timeout"})

    def _failure_detail(self, backend: NodeBackend, job: Job) -> Optional[str]:
        """Fetch why a job died, so the reason is readable without a second trip.

        stderr is where a crash normally lands; a job that logs everything to
        stdout still gets an explanation rather than an empty alert.
        """
        if not job.run_dir:
            return None
        for stream in ("stderr", "stdout"):
            self._renew_tick_lease()
            try:
                text = backend.executor.read_log(
                    job.run_dir, stream=stream, lines=FAILURE_LOG_LINES
                )
            except TransportError:
                return None
            text = (text or "").strip()
            if text:
                return text[-FAILURE_DETAIL_CHARS:]
        return None

    def _raise_alert(
        self,
        kind: str,
        title: str,
        *,
        severity: str = "error",
        job_id: Optional[int] = None,
        node: Optional[str] = None,
        detail: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        alert_id = dbm.add_alert(
            self.conn,
            kind,
            title,
            severity=severity,
            job_id=job_id,
            node=node,
            detail=detail,
        )
        if summary is not None:
            summary.setdefault("alerts", []).append(
                {"id": alert_id, "kind": kind, "title": title, "job_id": job_id}
            )

    def _terminate(self, backend: NodeBackend, job: Job) -> None:
        """Stop the process only if its pid still belongs to this job."""
        backend.executor.terminate(
            run_dir=job.run_dir,
            pid=job.remote_pid,
            pgid=job.remote_pgid,
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
        job_names = {job.id: job.name for job in running}

        measured: Dict[Tuple[int, int], int] = {}  # (job_id, gpu_index) -> MiB
        states: List[dict] = []

        for entry in payload.get("gpus") or []:
            index = int(entry.get("gpu_index") or 0)
            total_mb = _bytes_to_mb(entry.get("memory_total_bytes"))
            used_mb = _bytes_to_mb(entry.get("memory_used_bytes"))

            processes = entry.get("processes") or []
            ours_mb = 0
            external_procs = 0
            # How much of the card's usage NVML managed to attribute to anyone.
            # WSL lists the processes but reports no memory for them, which is a
            # different failure from listing nothing at all.
            reported_mb = 0
            # Foreign work is what the queue schedules around, so it is recorded
            # rather than merely counted: a card that looks full should be able
            # to say who filled it, queue job or not.
            process_rows: List[dict] = []
            for process in processes:
                pid = process.get("pid")
                process_mb = _bytes_to_mb(process.get("used_gpu_memory_bytes"))
                reported_mb += process_mb
                # A job's GPU process is usually a child of its run.sh, so match
                # on the process group first and fall back to the pid itself.
                job_id = pid_to_job.get(pid)
                if job_id is None:
                    pgid = probe.process_groups.get(pid)
                    job_id = pgid_to_job.get(pgid) if pgid is not None else None
                process_rows.append(
                    GpuProcess(
                        pid=int(pid or 0),
                        mem_mb=process_mb,
                        name=process.get("process_name"),
                        username=process.get("username"),
                        kind=process.get("type"),
                        job_id=job_id,
                        job_name=job_names.get(job_id) if job_id else None,
                    ).to_dict()
                )
                if job_id is None:
                    external_procs += 1
                    continue
                ours_mb += process_mb
                key = (job_id, index)
                measured[key] = measured.get(key, 0) + process_mb

            # Biggest consumer first, and bounded: a busy display server can list
            # dozens of tiny graphics contexts nobody needs stored.
            process_rows.sort(key=lambda row: row["mem_mb"], reverse=True)
            del process_rows[MAX_RECORDED_GPU_PROCESSES:]

            on_gpu = [job for job in running if index in job.gpu_ids]
            reserved_mb = sum(
                max(job.vram_mb, measured.get((job.id, index), 0)) for job in on_gpu
            )

            if reported_mb > 0 or not on_gpu:
                attribution = "processes"
                external_mb = max(0, used_mb - ours_mb)
            else:
                # Memory is in use, this queue has jobs here, and NVML attributed
                # none of it - the WSL case, where the card is driven from
                # Windows. It shows up two ways: no process list at all, or a
                # process list whose memory figures are all zero. Either way,
                # credit our jobs up to their reservation rather than charging
                # them once as "external" and again as "reserved".
                attribution = "blind"
                external_mb = max(0, used_mb - reserved_mb)

            states.append(
                {
                    "index": index,
                    "name": entry.get("name"),
                    "mem_total_mb": total_mb,
                    "mem_used_mb": used_mb,
                    "util_percent": entry.get("gpu_util_percent"),
                    "mem_util_percent": entry.get("memory_util_percent"),
                    "temperature_c": entry.get("temperature_c"),
                    "external_mem_mb": external_mb,
                    "external_procs": external_procs,
                    "queue_mem_mb": ours_mb,
                    "attribution": attribution,
                    "reserved_mb": reserved_mb,
                    "queue_jobs": len(on_gpu),
                    "processes": process_rows,
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

    def _schedulable_gpus(self, node: Node):
        """GPUs available to new queue placements on a node."""
        mask = self._gpu_allowlists.get(node.name)
        if mask is None:
            return list(node.gpus)
        return [gpu for gpu in node.gpus if gpu.index in mask]

    def _dispatch(self, summary: Dict[str, Any]) -> None:
        pending = dbm.pending_jobs(self.conn)
        if not pending:
            return

        all_nodes = [
            node
            for node in dbm.get_nodes(self.conn)
            if node.name in self._configured_nodes
        ]
        # Everything the queue knows a GPU about, drained or not. Used only to
        # tell "waiting for room" apart from "will never fit anywhere"; a node
        # that is down has no GPU rows, so it cannot make a job look doomed.
        inventory = [node for node in all_nodes if node.gpus]
        nodes = [node for node in all_nodes if node.is_schedulable]

        headroom = int(self.settings["headroom_mb"])
        max_jobs = int(self.settings["max_jobs_per_gpu"])
        # CPU-only jobs are held to a count per node, since they reserve no
        # memory for anything else to schedule around.
        max_cpu_jobs = int(self.settings.get("max_cpu_jobs_per_node") or 0)
        cpu_slots: Dict[str, int] = {}
        if max_cpu_jobs > 0:
            running_cpu: Dict[str, int] = {}
            for job in dbm.live_jobs(self.conn):
                if job.gpus <= 0 and job.node:
                    running_cpu[job.node] = running_cpu.get(job.node, 0) + 1
            for node in nodes:
                cpu_slots[node.name] = max(0, max_cpu_jobs - running_cpu.get(node.name, 0))

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
                for gpu in self._schedulable_gpus(node)
            ]

        # Lanes are an explicit order a person wrote down, so they are placed
        # before the opportunistic pool competes for the same budget.
        lane_pending = [job for job in pending if job.lane]
        free_pending = [job for job in pending if not job.lane]
        if lane_pending:
            self._dispatch_lanes(
                lane_pending, nodes, budgets, inventory, headroom, max_jobs, summary
            )

        for job in free_pending:
            self._renew_tick_lease()
            if self._gate(job, inventory, headroom, summary) is not None:
                continue

            excluded: Set[str] = set()
            launch_errors: List[Tuple[str, str]] = []
            launched = False
            while True:
                placement = self._find_placement(
                    job,
                    nodes,
                    budgets,
                    max_jobs,
                    cpu_slots if max_cpu_jobs > 0 else None,
                    excluded_nodes=excluded,
                )
                if placement is None:
                    break
                node_name, gpu_ids = placement
                node = next((n for n in nodes if n.name == node_name), None)
                if node is None:
                    break
                try:
                    self._launch(node, job, gpu_ids)
                except LaunchRejected as error:
                    launch_errors.append((node_name, str(error)))
                    summary["errors"].append({"job": job.id, "error": str(error)})
                    excluded.add(node_name)
                    self._renew_tick_lease()
                    continue
                except TransportError as error:
                    # A channel failure may happen after the remote process was
                    # started. Do not try a second node in that uncertain case.
                    launch_errors.append((node_name, str(error)))
                    summary["errors"].append({"job": job.id, "error": str(error)})
                    self._renew_tick_lease()
                    break

                launched = True
                self._renew_tick_lease()
                if job.gpus <= 0 and node_name in cpu_slots:
                    cpu_slots[node_name] = max(0, cpu_slots[node_name] - 1)
                for budget in budgets[node_name]:
                    if budget.index in gpu_ids:
                        budget.free_mb = max(0, budget.free_mb - job.vram_mb)
                        budget.jobs += 1
                summary["dispatched"].append(
                    {"id": job.id, "node": node_name, "gpu_ids": gpu_ids}
                )
                break

            if launched or not launch_errors:
                continue
            self._report_launch_failure(job, launch_errors, summary)

    def _gate(
        self,
        job: Job,
        inventory: List[Node],
        headroom: int,
        summary: Dict[str, Any],
    ) -> Optional[str]:
        """Decide whether a pending job may be placed at all.

        Returns None when it may, or why not: `impossible` (the job was just
        failed, nothing could ever host it), `held` (a dependency ended badly)
        or `waiting` (a dependency has not finished yet). Lanes and the free
        pool share this so a job means the same thing in both.
        """
        impossible = self._impossible_reason(job, inventory, headroom)
        if impossible:
            dbm.update_job(
                self.conn,
                job.id,
                state="failed",
                finished_at=utcnow(),
                last_error=impossible,
            )
            dbm.add_event(
                self.conn, "job_failed", job_id=job.id, message=impossible
            )
            self._raise_alert(
                "job_unschedulable",
                f"{job.name or 'job'} will never run: {impossible}",
                job_id=job.id,
                summary=summary,
            )
            return "impossible"

        ready, hold_reason = self._dependencies_ready(job)
        if hold_reason:
            self._hold(job, hold_reason, summary)
            return "held"
        self._unhold(job)
        return None if ready else "waiting"

    def _report_launch_failure(
        self,
        job: Job,
        launch_errors: List[Tuple[str, str]],
        summary: Dict[str, Any],
    ) -> None:
        detail = "; ".join(
            f"{node_name}: {error}" for node_name, error in launch_errors
        )[:FAILURE_DETAIL_CHARS]
        message = f"launch failed: {detail}"[:500]
        # A persistent node fault should produce one warning, not one on
        # every daemon pass.
        if job.last_error == message:
            return
        last_node = launch_errors[-1][0]
        with dbm.transaction(self.conn):
            dbm.update_job(self.conn, job.id, last_error=message)
            dbm.add_event(
                self.conn,
                "job_failed",
                job_id=job.id,
                node=last_node,
                message=message[:200],
            )
            self._raise_alert(
                "launch_failed",
                f"could not start {job.name or 'job'}",
                severity="warning",
                job_id=job.id,
                node=last_node,
                detail=detail,
                summary=summary,
            )

    # --- lane dispatch ----------------------------------------------------

    def _dispatch_lanes(
        self,
        lane_pending: List[Job],
        nodes: List[Node],
        budgets: Dict[str, List[GpuBudget]],
        inventory: List[Node],
        headroom: int,
        max_jobs: int,
        summary: Dict[str, Any],
    ) -> None:
        """Run each lane's queue strictly in order.

        Head-of-line blocking is the feature, not a limitation: a lane exists
        so that what runs next is knowable in advance, and quietly skipping
        past a stuck job would make the printed order a lie.
        """
        queues: Dict[str, List[Job]] = {}
        for job in lane_pending:
            queues.setdefault(job.lane, []).append(job)

        nodes_by_name = {node.name: node for node in nodes}
        running: Dict[str, int] = {}
        for job in dbm.live_jobs(self.conn):
            if job.lane:
                running[job.lane] = running.get(job.lane, 0) + 1

        for lane_name in sorted(queues):
            self._renew_tick_lease()
            lane = self._lane_for(lane_name)
            if lane is None or lane.paused:
                continue
            node = nodes_by_name.get(lane.node)
            if node is None:
                continue  # the node is down, drained or ignored; the lane waits

            for job in sorted(queues[lane_name], key=_lane_order):
                if running.get(lane_name, 0) >= max(1, lane.concurrency):
                    break
                blocked = self._gate(job, inventory, headroom, summary)
                if blocked == "impossible":
                    # It was just failed, so it has left the queue. Carrying on
                    # is right: a job nothing can host must not wedge the lane.
                    continue
                if blocked:
                    break
                if not self._lane_has_room(lane, node, budgets, job, max_jobs):
                    break

                launched, errors = self._attempt_launch(
                    job, node, lane.gpu_ids, summary
                )
                if not launched:
                    if errors:
                        self._report_launch_failure(job, errors, summary)
                    break

                running[lane_name] = running.get(lane_name, 0) + 1
                for budget in budgets.get(node.name, []):
                    if budget.index in lane.gpu_ids:
                        budget.free_mb = max(0, budget.free_mb - job.vram_mb)
                        budget.jobs += 1
                summary["dispatched"].append(
                    {
                        "id": job.id,
                        "node": node.name,
                        "gpu_ids": list(lane.gpu_ids),
                        "lane": lane_name,
                    }
                )

    def _lane_for(self, lane_name: str) -> Optional[Lane]:
        """The lane's row, recreated from its name if the row went missing.

        Jobs outlive lane rows - a lane can be deleted while work is still
        pinned to it - and stranding those jobs forever would be worse than
        restoring the row the name already fully describes.
        """
        lane = dbm.get_lane(self.conn, lane_name)
        if lane is not None:
            return lane
        try:
            node_name, gpu_ids = parse_lane_name(lane_name)
        except ValueError:
            return None
        dbm.ensure_lane(self.conn, lane_name, node=node_name, gpu_ids=gpu_ids)
        return dbm.get_lane(self.conn, lane_name)

    def _lane_has_room(
        self,
        lane: Lane,
        node: Node,
        budgets: Dict[str, List[GpuBudget]],
        job: Job,
        max_jobs: int,
    ) -> bool:
        """Whether the lane's own GPUs can take this job right now.

        A lane owns its cards but does not own the machine: work nobody
        submitted through the queue can fill them, and the lane then waits
        rather than starting a job that would run into an out-of-memory error.
        """
        if job.gpus <= 0:
            return True
        by_index = {budget.index: budget for budget in budgets.get(node.name, [])}
        for index in lane.gpu_ids:
            budget = by_index.get(index)
            if budget is None or not budget.fits(job.vram_mb, max_jobs):
                return False
        return True

    def _attempt_launch(
        self,
        job: Job,
        node: Node,
        gpu_ids: Sequence[int],
        summary: Dict[str, Any],
    ) -> Tuple[bool, List[Tuple[str, str]]]:
        """Start one job on one placement, reporting rather than raising."""
        try:
            self._launch(node, job, [int(index) for index in gpu_ids])
        except (LaunchRejected, TransportError) as error:
            summary["errors"].append({"job": job.id, "error": str(error)})
            self._renew_tick_lease()
            return False, [(node.name, str(error))]
        self._renew_tick_lease()
        return True, []

    def _impossible_reason(
        self, job: Job, inventory: List[Node], headroom_mb: int
    ) -> Optional[str]:
        """Why this job can never run, or None if it is merely waiting for room.

        A job nothing can host would otherwise sit `pending` forever with no
        explanation, and any client blocked in `job wait` would sit with it.
        The judgement is made only against GPUs the queue has actually seen, so
        a request that outgrows the machines which happen to be reachable right
        now keeps waiting instead of being declared doomed.
        """
        if job.gpus <= 0:
            return None  # CPU-only work needs no GPU to fit on
        candidates = inventory
        if job.node_constraint:
            candidates = [node for node in inventory if node.name == job.node_constraint]
        if not candidates:
            return None

        if job.lane:
            # A lane job is pinned to named cards, so it is judged against
            # those alone. Measuring it against the node's other GPUs would
            # let a job that cannot fit its own lane wait there forever.
            lane = dbm.get_lane(self.conn, job.lane)
            if lane is None:
                return None
            gpu_inventory = [
                (node, [gpu for gpu in node.gpus if gpu.index in set(lane.gpu_ids)])
                for node in candidates
                if node.name == lane.node
            ]
            if not gpu_inventory or not any(gpus for _node, gpus in gpu_inventory):
                return None  # the lane's cards are not in view yet
        else:
            gpu_inventory = [
                (node, self._schedulable_gpus(node)) for node in candidates
            ]
        widest = max((len(gpus) for _node, gpus in gpu_inventory), default=0)
        if job.gpus > widest:
            where = job.node_constraint or "the largest known node"
            requested_unit = "GPU" if job.gpus == 1 else "GPUs"
            available_unit = "GPU" if widest == 1 else "GPUs"
            return (
                f"needs {job.gpus} {requested_unit}; {where} has {widest} "
                f"schedulable {available_unit}"
            )

        biggest = max(
            gpu.mem_total_mb for _node, gpus in gpu_inventory for gpu in gpus
        )
        usable = max(0, biggest - headroom_mb)
        if job.vram_mb > usable:
            where = (
                f"the largest schedulable GPU on {job.node_constraint}"
                if job.node_constraint
                else "the largest schedulable GPU the queue knows of"
            )
            return (
                f"reserves {format_mb(job.vram_mb)}; {where} holds "
                f"{format_mb(usable)} at most"
            )

        fitting_width = max(
            (
                sum(
                    1
                    for gpu in gpus
                    if max(0, gpu.mem_total_mb - headroom_mb) >= job.vram_mb
                )
                for _node, gpus in gpu_inventory
            ),
            default=0,
        )
        if job.gpus > fitting_width:
            where = job.node_constraint or "any known node"
            return (
                f"needs {job.gpus} GPUs each reserving {format_mb(job.vram_mb)}; "
                f"{where} has at most {fitting_width} matching GPUs"
            )
        return None

    def _dependencies_ready(self, job: Job) -> Tuple[bool, Optional[str]]:
        """Return (ready, hold_reason).

        A dependency that ended badly *holds* the job rather than failing it.
        Cancelling one job used to fail everything queued behind it, which
        destroyed the queue position, the notes and the submission of every
        job in the chain over a decision that only concerned the first one.
        Holding keeps all of that and asks a person what they meant.
        """
        for dep_id in job.depends_on:
            dep = dbm.get_job(self.conn, dep_id)
            if dep is None:
                return False, f"dependency {dep_id} no longer exists"
            if dep.state == "completed":
                continue
            if dep.is_terminal:
                return False, f"dependency {dep_id} ended as {dep.state}"
            return False, None
        for dep_id in job.depends_any:
            dep = dbm.get_job(self.conn, dep_id)
            if dep is None:
                return False, f"dependency {dep_id} no longer exists"
            # This form waits for an outcome, not for a good one.
            if dep.is_terminal:
                continue
            return False, None
        return True, None

    def _hold(self, job: Job, reason: str, summary: Dict[str, Any]) -> None:
        """Park a pending job and say so once, not once per tick."""
        if job.held_reason == reason:
            return
        with dbm.transaction(self.conn):
            dbm.update_job(self.conn, job.id, held_reason=reason)
            dbm.add_event(
                self.conn, "job_held", job_id=job.id, message=reason
            )
            self._raise_alert(
                "job_held",
                f"{job.name or 'job'} is waiting for a decision: {reason}",
                severity="warning",
                job_id=job.id,
                detail=(
                    f"Job {job.id} stays queued and keeps its place.\n"
                    f"  nvidb job release {job.id}          run it without that dependency\n"
                    f"  nvidb job edit {job.id} --add-after NEW --drop-after OLD\n"
                    f"  nvidb job cancel {job.id}           drop it instead"
                ),
                summary=summary,
            )
        summary.setdefault("held", []).append({"id": job.id, "reason": reason})

    def _unhold(self, job: Job) -> None:
        """Clear a hold whose cause went away (a dependency was re-run and passed)."""
        if not job.held_reason:
            return
        with dbm.transaction(self.conn):
            dbm.update_job(self.conn, job.id, held_reason=None)
            dbm.add_event(
                self.conn,
                "job_unheld",
                job_id=job.id,
                message="dependencies are satisfiable again",
            )

    def _find_placement(
        self,
        job: Job,
        nodes: List[Node],
        budgets: Dict[str, List[GpuBudget]],
        max_jobs: int,
        cpu_slots: Optional[Dict[str, int]] = None,
        excluded_nodes: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, List[int]]]:
        """Pick a node and GPU set for one job, or None when nothing fits."""
        excluded_nodes = excluded_nodes or set()
        candidates = [node for node in nodes if node.name not in excluded_nodes]
        if job.node_constraint:
            candidates = [
                node for node in candidates if node.name == job.node_constraint
            ]
        if not candidates:
            return None

        if job.gpus <= 0:
            # CPU-only work: any live node with a free slot, least loaded first.
            if cpu_slots is not None:
                candidates = [
                    node for node in candidates if cpu_slots.get(node.name, 0) > 0
                ]
                if not candidates:
                    return None
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
            attempt=job.attempt + 1,
            timeout=float(self.settings["launch_timeout"]),
        )
        now = utcnow()
        with dbm.transaction(self.conn):
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
                message=(
                    f"pid {launched.pid} on GPU "
                    f"{','.join(str(i) for i in gpu_ids) or '-'}"
                ),
                data={
                    "pid": launched.pid,
                    "gpu_ids": gpu_ids,
                    "vram_mb": job.vram_mb,
                    "vram": format_mb(job.vram_mb),
                },
            )

    # --- reads ------------------------------------------------------------

    def _node_view(self, node: Node, headroom: int) -> Dict[str, Any]:
        """One node as reported to the UIs, masked by its GPU allowlist.

        A host may hand the queue only some of its cards; the rest are
        somebody else's business and would only be noise on screen. The
        inventory kept in the database stays complete either way - the
        mask is a view, not a deletion - and a card excluded after a job
        of ours landed on it stays visible until that job is gone.
        """
        view = node.to_dict(headroom)
        mask = self._gpu_allowlists.get(node.name)
        if mask is None:
            return view
        view["gpus"] = [
            gpu
            for gpu in view.get("gpus") or []
            if gpu.get("index") in mask or gpu.get("queue_jobs")
        ]
        return view

    def snapshot(self, *, include_ignored: bool = False) -> Dict[str, Any]:
        """A single JSON-friendly view of the whole queue, for other tools."""
        headroom = int(self.settings["headroom_mb"])
        nodes = dbm.get_nodes(self.conn, include_ignored=include_ignored)
        jobs = dbm.list_jobs(self.conn, states=["pending", "running"])
        recent = dbm.list_jobs(self.conn, limit=20, newest_first=True)
        recent_terminal = [job for job in recent if job.is_terminal][:10]
        holder = dbm.lock_holder(self.conn, TICK_LOCK)
        return {
            "generated_at": utcnow(),
            "db_path": dbm.connection_path(self.conn),
            "last_tick_at": dbm.get_meta(self.conn, "last_tick_at"),
            "last_backup_at": dbm.get_meta(self.conn, "last_backup_at"),
            "tick_lock": dict(holder) if holder else None,
            # Whether this machine has something keeping the queue moving while
            # no client is looking. Reported rather than acted on: a queue with
            # a stopped keeper is not broken, only idle.
            "keeper": keeper_mod.status(),
            "settings": {
                "headroom_mb": headroom,
                "max_jobs_per_gpu": int(self.settings["max_jobs_per_gpu"]),
                "placement": self.settings.get("placement"),
            },
            "counts": dbm.job_counts(self.conn),
            # Pending jobs that will not move until someone decides what they
            # should have waited for. They are counted as pending above, so
            # they need saying separately or they read as ordinary queueing.
            "held": [job.to_dict() for job in dbm.held_jobs(self.conn)],
            "nodes": [self._node_view(node, headroom) for node in nodes],
            "lanes": self.lanes(),
            "jobs": [job.to_dict() for job in jobs],
            "recent": [job.to_dict() for job in recent_terminal],
            "alerts": dbm.list_alerts(self.conn, open_only=True, limit=20),
            "open_alerts": dbm.open_alert_count(self.conn),
            # Processes still to be killed on nodes that were unreachable when
            # their job was cancelled or timed out.
            "pending_reaps": dbm.orphan_count(self.conn),
        }

    def deliver_alerts(self, notifier) -> int:
        """Push any undelivered alerts through a notifier. Used by the daemon.

        Alerts are stamped as delivered whether or not a channel accepted them,
        so a misconfigured hook cannot make the daemon replay the same failure
        on every pass.
        """
        pending = dbm.undelivered_alerts(self.conn)
        if not pending:
            return 0
        delivered = notifier.deliver(pending)
        dbm.mark_alerts_delivered(self.conn, delivered)
        return len(delivered)

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


def _lane_order(job: Job):
    """A lane's running order. A job with no place yet goes to the back."""
    return (job.lane_seq is None, job.lane_seq or 0, job.id)


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
