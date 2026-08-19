"""An in-memory stand-in for a GPU cluster, used by the queue tests.

`FakeNode` models what the scheduler can actually observe on a machine: NVML
memory figures, GPU processes with pids, and the pid/pgid table. Tests drive it
directly - start a foreign process, let a job exit, take the node offline - and
assert on what the scheduler decides.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from nvidb.sched.executor import JobProbe, LaunchRejected, LaunchResult, NodeProbe
from nvidb.sched.transport import CommandResult, TransportError

MB = 1024 * 1024


class FakeGpu:
    def __init__(self, index: int, name: str, total_mb: int):
        self.index = index
        self.name = name
        self.total_mb = total_mb
        # pid -> MiB, covering both foreign work and this queue's jobs
        self.processes: Dict[int, int] = {}
        # pid -> (process name, username), as NVML would report them
        self.identities: Dict[int, tuple] = {}
        self.util = 0
        self.mem_util = 0

    def used_mb(self) -> int:
        return sum(self.processes.values())


class FakeNode:
    """One machine: GPUs, running job processes, and reachability."""

    def __init__(self, name: str, gpus: List[FakeGpu]):
        self.name = name
        self.gpus = gpus
        self.online = True
        self.nvml_ok = True
        # WSL passes the GPU through the Windows driver: NVML reports memory in
        # use but cannot attribute it. That shows up two ways on real machines -
        # no process list at all, or a list whose memory figures are all zero.
        self.hide_processes = False
        self.hide_process_memory = False
        self.launch_error: Optional[str] = None
        self.jobs: Dict[int, dict] = {}
        self.results: Dict[str, object] = {}
        self.logs: Dict[str, str] = {}
        self.killed: List[int] = []
        # Jobs whose leftover process was cleaned up after the fact.
        self.reaped: List[int] = []
        self._next_pid = 1000

    # --- test-facing helpers ---------------------------------------------

    def add_foreign_process(
        self,
        gpu_index: int,
        mb: int,
        pid: Optional[int] = None,
        name: str = "python",
        username: str = "someone",
    ) -> int:
        """Simulate work this queue did not start (a hand-launched training run)."""
        pid = pid if pid is not None else self._allocate_pid()
        self.gpus[gpu_index].processes[pid] = mb
        self.gpus[gpu_index].identities[pid] = (name, username)
        return pid

    def clear_gpu(self, gpu_index: int) -> None:
        self.gpus[gpu_index].processes.clear()

    def job_allocates(self, job_id: int, gpu_index: int, mb: int) -> None:
        """Let a running job actually take VRAM on a card."""
        record = self.jobs[job_id]
        self.gpus[gpu_index].processes[record["gpu_pid"]] = mb
        self.gpus[gpu_index].identities[record["gpu_pid"]] = ("python", "tester")

    def report_progress(self, job_id: int, text: str) -> None:
        """Simulate the job writing its own status line to $NVIDB_STATUS_FILE."""
        self.jobs[job_id]["progress"] = text

    def finish_job(self, job_id: int, exit_code: int = 0, result=None) -> None:
        record = self.jobs[job_id]
        record["exit_code"] = exit_code
        record["alive"] = False
        self._release_job_memory(record)
        if result is not None:
            self.results[record["run_dir"]] = result

    def vanish_job(self, job_id: int) -> None:
        """The process disappears with no exit code (reboot, OOM killer, ...)."""
        record = self.jobs[job_id]
        record["alive"] = False
        self._release_job_memory(record)

    def _release_job_memory(self, record: dict) -> None:
        for gpu in self.gpus:
            gpu.processes.pop(record["gpu_pid"], None)

    def _allocate_pid(self) -> int:
        self._next_pid += 1
        return self._next_pid

    # --- backend-facing -----------------------------------------------------

    def nvml_payload(self) -> dict:
        if not self.nvml_ok:
            return {"ok": False, "error": "NVML init failed"}
        return {
            "ok": True,
            "backend": "fake",
            "gpus": [
                {
                    "gpu_index": gpu.index,
                    "name": gpu.name,
                    "memory_total_bytes": gpu.total_mb * MB,
                    "memory_used_bytes": gpu.used_mb() * MB,
                    "gpu_util_percent": gpu.util,
                    "memory_util_percent": gpu.mem_util,
                    "temperature_c": 45,
                    "processes": []
                    if self.hide_processes
                    else [
                        {
                            "pid": pid,
                            "used_gpu_memory_bytes": None
                            if self.hide_process_memory
                            else mb * MB,
                            "process_name": gpu.identities.get(pid, ("python", "someone"))[0],
                            "username": gpu.identities.get(pid, ("python", "someone"))[1],
                            "type": "C",
                        }
                        for pid, mb in sorted(gpu.processes.items())
                    ],
                }
                for gpu in self.gpus
            ],
        }

    def start_lane_job(self, spec: dict) -> dict:
        """Start a job the way the node's lane runner would: as its own child.

        Records it exactly as a scheduler-launched job, so everything else the
        fake cluster can do to a job - finish it, let it take VRAM, make it
        vanish - works the same whichever started it.
        """
        pid = self._allocate_pid()
        self.jobs[spec["job_id"]] = {
            "pid": pid,
            "pgid": pid,
            "gpu_pid": self._allocate_pid(),
            "exit_code": None,
            "alive": True,
            "run_dir": spec["run_dir"],
            "gpu_ids": list(spec.get("gpu_ids") or []),
            "command": spec.get("command", ""),
            "progress": None,
        }
        self.logs[spec["run_dir"]] = ""
        return {
            "job_id": spec["job_id"],
            "pid": pid,
            "run_dir": spec["run_dir"],
            "started_at": None,
        }

    def process_groups(self) -> Dict[int, int]:
        table: Dict[int, int] = {}
        for record in self.jobs.values():
            if not record["alive"]:
                continue
            table[record["pid"]] = record["pgid"]
            # The GPU process is a child of run.sh, sharing its process group.
            table[record["gpu_pid"]] = record["pgid"]
        return table


class FakeTransport:
    def __init__(self, node: FakeNode):
        self.node = node
        self.name = node.name
        self.commands: List[str] = []

    def run(self, command: str, timeout=None) -> CommandResult:
        if not self.node.online:
            raise TransportError(f"{self.name}: unreachable")
        self.commands.append(command)
        return CommandResult(0, "", "")

    def close(self) -> None:
        pass


class FakeExecutor:
    def __init__(self, node: FakeNode):
        self.node = node

    def run_dir(self, job_id, attempt: int = 1) -> str:
        if not self.node.online:
            raise TransportError(f"{self.node.name}: unreachable")
        suffix = str(job_id) if int(attempt) <= 1 else f"{job_id}-attempt-{attempt}"
        return f"/fake/{self.node.name}/jobs/{suffix}"

    def launch(
        self,
        *,
        job_id,
        job_name,
        command,
        workdir=None,
        env=None,
        gpu_ids=None,
        node_name=None,
        attempt=1,
        timeout=None,
    ) -> LaunchResult:
        if not self.node.online:
            raise TransportError(f"{self.node.name}: unreachable")
        if self.node.launch_error:
            raise LaunchRejected(self.node.launch_error)
        pid = self.node._allocate_pid()
        run_dir = self.run_dir(job_id, attempt)
        self.node.jobs[job_id] = {
            "pid": pid,
            "pgid": pid,
            "gpu_pid": self.node._allocate_pid(),
            "exit_code": None,
            "alive": True,
            "run_dir": run_dir,
            "gpu_ids": list(gpu_ids or []),
            "command": command,
            "progress": None,
        }
        self.node.logs[run_dir] = f"$ {command}\n"
        return LaunchResult(pid=pid, pgid=pid, run_dir=run_dir)

    def probe(self, specs, timeout=None) -> NodeProbe:
        if not self.node.online:
            raise TransportError(f"{self.node.name}: unreachable")
        probe = NodeProbe()
        for job_id, _run_dir in specs:
            record = self.node.jobs.get(job_id)
            if record is None:
                continue
            probe.jobs[job_id] = JobProbe(
                job_id=job_id,
                pid=record["pid"],
                pgid=record["pgid"],
                exit_code=record["exit_code"],
                alive=record["alive"],
                progress=record.get("progress"),
            )
        probe.process_groups = self.node.process_groups()
        return probe

    def kill(self, *, pid, pgid, signal="TERM"):
        # An offline node cannot be signalled - the scheduler has to cope with
        # closing a job's record while its process keeps running.
        if not self.node.online:
            raise TransportError(f"{self.node.name}: unreachable")
        for job_id, record in self.node.jobs.items():
            if record["pid"] == pid and record["alive"]:
                record["alive"] = False
                self.node._release_job_memory(record)
                self.node.killed.append(job_id)
        return CommandResult(0, "", "")

    def reap(self, *, run_dir, pid, pgid=None, grace=5):
        if not self.node.online:
            raise TransportError(f"{self.node.name}: unreachable")
        for job_id, record in self.node.jobs.items():
            if record["pid"] != pid or record["run_dir"] != run_dir:
                continue
            if not record["alive"]:
                return False
            record["alive"] = False
            self.node._release_job_memory(record)
            self.node.reaped.append(job_id)
            return True
        return False

    def terminate(self, *, run_dir, pid, pgid=None, grace=5):
        if not self.node.online:
            raise TransportError(f"{self.node.name}: unreachable")
        for job_id, record in self.node.jobs.items():
            if record["pid"] != pid or record["run_dir"] != run_dir:
                continue
            if not record["alive"]:
                return False
            record["alive"] = False
            self.node._release_job_memory(record)
            self.node.killed.append(job_id)
            return True
        return False

    def read_result(self, run_dir):
        return self.node.results.get(run_dir)

    def read_log(self, run_dir, stream="stdout", lines=200):
        return self.node.logs.get(run_dir, "")

    def remove_run_dir(self, run_dir):
        self.node.logs.pop(run_dir, None)


class FakeLaneRunner:
    """The resident lane runner, as a node sees it.

    Models the behaviour the scheduler depends on rather than the shell that
    delivers it: specs are staged and withdrawn, the head one is claimed and
    started as soon as there is room, and the runner reports what it did. The
    shell that carries this is covered separately by the command-builder tests.
    """

    def __init__(self, node: FakeNode):
        self.node = node
        self.lanes: Dict[str, dict] = {}
        # Every sync the controller performed, so tests can assert on what was
        # staged rather than only on the outcome.
        self.syncs: List[tuple] = []

    def _lane(self, lane_name: str) -> dict:
        return self.lanes.setdefault(
            lane_name, {"queued": [], "current": [], "finished": [], "up": False}
        )

    def sync(self, lane_name: str, specs, want_runner: bool):
        lane = self._lane(lane_name)
        self.syncs.append((lane_name, [spec["job_id"] for spec in specs], want_runner))

        # A claimed spec is gone from the staging area, so withdrawing it is
        # not possible - which is exactly how the real rename-based claim
        # keeps the controller and the runner from both acting on one job.
        claimed = {entry["job_id"] for entry in lane["current"]}
        lane["queued"] = [spec for spec in specs if spec["job_id"] not in claimed]

        if not want_runner:
            lane["queued"] = []
            if not lane["current"]:
                lane["up"] = False
                return None
        else:
            lane["up"] = True

        self._advance(lane_name)
        return self._state(lane_name)

    def poll(self, lane_name: str):
        """One turn of the runner's own loop, with no controller involved."""
        self._advance(lane_name)
        return self._state(lane_name)

    def stop(self, lane_name: str, hard: bool = False) -> None:
        lane = self._lane(lane_name)
        lane["queued"] = []
        if hard or not lane["current"]:
            lane["up"] = False

    def _free_mb(self, gpu_ids) -> int:
        free = []
        for index in gpu_ids:
            gpu = self.node.gpus[index]
            free.append(gpu.total_mb - gpu.used_mb())
        return min(free) if free else 0

    def _advance(self, lane_name: str) -> None:
        lane = self.lanes[lane_name]
        still = []
        for entry in lane["current"]:
            record = self.node.jobs.get(entry["job_id"])
            if record is not None and record["alive"]:
                still.append(entry)
            else:
                lane["finished"].append(
                    {
                        "job_id": entry["job_id"],
                        "exit_code": (record or {}).get("exit_code"),
                    }
                )
        lane["current"] = still

        while lane["queued"]:
            spec = lane["queued"][0]
            concurrency = max(1, int(spec.get("concurrency") or 1))
            if len(lane["current"]) >= concurrency:
                break
            needed = int(spec.get("vram_mb") or 0)
            if needed and self._free_mb(spec.get("gpu_ids") or []) < needed:
                break  # the card filled up with work the queue did not start
            lane["queued"].pop(0)
            lane["current"].append(self.node.start_lane_job(spec))

    def _state(self, lane_name: str) -> dict:
        lane = self.lanes[lane_name]
        return {
            "lane": lane_name,
            "pid": 99000,
            "current": list(lane["current"]),
            "queued": [spec["job_id"] for spec in lane["queued"]],
            "finished": list(lane["finished"]),
        }


class FakeBackend:
    def __init__(self, node: FakeNode):
        self.name = node.name
        self.node = node
        self.transport = FakeTransport(node)
        self.executor = FakeExecutor(node)
        self.runner = FakeLaneRunner(node)

    def probe_gpus(self) -> dict:
        if not self.node.online:
            raise TransportError(f"{self.name}: unreachable")
        return self.node.nvml_payload()

    def lane_sync(
        self,
        *,
        lane_name,
        specs,
        spool_root=None,
        interval=None,
        idle_exit=None,
        want_runner=True,
        timeout=None,
    ):
        if not self.node.online:
            raise TransportError(f"{self.name}: unreachable")
        return self.runner.sync(lane_name, list(specs), want_runner)

    def lane_stop(self, *, lane_name, spool_root=None, hard=False, timeout=None):
        if not self.node.online:
            raise TransportError(f"{self.name}: unreachable")
        self.runner.stop(lane_name, hard=hard)

    def close(self) -> None:
        pass


class FakeCluster:
    """Holds the nodes and hands the scheduler a backend factory."""

    def __init__(self, *nodes: FakeNode):
        self.nodes = {node.name: node for node in nodes}
        self.backends = {node.name: FakeBackend(node) for node in nodes}

    def config(self) -> dict:
        return {
            "servers": [
                {
                    "hostname": f"10.0.0.{index + 1}",
                    "port": 22,
                    "username": "tester",
                    "nickname": name,
                }
                for index, name in enumerate(self.nodes)
            ]
        }

    def backend_factory(self, node):
        return self.backends[node.name]

    def __getitem__(self, name: str) -> FakeNode:
        return self.nodes[name]
