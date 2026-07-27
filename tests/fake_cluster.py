"""An in-memory stand-in for a GPU cluster, used by the queue tests.

`FakeNode` models what the scheduler can actually observe on a machine: NVML
memory figures, GPU processes with pids, and the pid/pgid table. Tests drive it
directly - start a foreign process, let a job exit, take the node offline - and
assert on what the scheduler decides.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from nvidb.sched.executor import JobProbe, LaunchResult, NodeProbe
from nvidb.sched.transport import CommandResult, TransportError

MB = 1024 * 1024


class FakeGpu:
    def __init__(self, index: int, name: str, total_mb: int):
        self.index = index
        self.name = name
        self.total_mb = total_mb
        # pid -> MiB, covering both foreign work and this queue's jobs
        self.processes: Dict[int, int] = {}
        self.util = 0

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
        # use but names no processes at all.
        self.hide_processes = False
        self.launch_error: Optional[str] = None
        self.jobs: Dict[int, dict] = {}
        self.results: Dict[str, object] = {}
        self.logs: Dict[str, str] = {}
        self.killed: List[int] = []
        self._next_pid = 1000

    # --- test-facing helpers ---------------------------------------------

    def add_foreign_process(self, gpu_index: int, mb: int, pid: Optional[int] = None) -> int:
        """Simulate work this queue did not start (a hand-launched training run)."""
        pid = pid if pid is not None else self._allocate_pid()
        self.gpus[gpu_index].processes[pid] = mb
        return pid

    def clear_gpu(self, gpu_index: int) -> None:
        self.gpus[gpu_index].processes.clear()

    def job_allocates(self, job_id: int, gpu_index: int, mb: int) -> None:
        """Let a running job actually take VRAM on a card."""
        record = self.jobs[job_id]
        self.gpus[gpu_index].processes[record["gpu_pid"]] = mb

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
                    "temperature_c": 45,
                    "processes": []
                    if self.hide_processes
                    else [
                        {"pid": pid, "used_gpu_memory_bytes": mb * MB}
                        for pid, mb in sorted(gpu.processes.items())
                    ],
                }
                for gpu in self.gpus
            ],
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
        timeout=None,
    ) -> LaunchResult:
        if not self.node.online:
            raise TransportError(f"{self.node.name}: unreachable")
        if self.node.launch_error:
            raise TransportError(self.node.launch_error)
        pid = self.node._allocate_pid()
        run_dir = f"/fake/{self.node.name}/jobs/{job_id}"
        self.node.jobs[job_id] = {
            "pid": pid,
            "pgid": pid,
            "gpu_pid": self.node._allocate_pid(),
            "exit_code": None,
            "alive": True,
            "run_dir": run_dir,
            "gpu_ids": list(gpu_ids or []),
            "command": command,
        }
        self.node.logs[run_dir] = f"$ {command}\n"
        return LaunchResult(pid=pid, pgid=pid, run_dir=run_dir)

    def probe(self, specs, want_process_table=True, timeout=None) -> NodeProbe:
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
            )
        if want_process_table:
            probe.process_groups = self.node.process_groups()
        return probe

    def kill(self, *, pid, pgid, signal="TERM"):
        for job_id, record in self.node.jobs.items():
            if record["pid"] == pid and record["alive"]:
                record["alive"] = False
                self.node._release_job_memory(record)
                self.node.killed.append(job_id)
        return CommandResult(0, "", "")

    def read_result(self, run_dir):
        return self.node.results.get(run_dir)

    def read_log(self, run_dir, stream="stdout", lines=200):
        return self.node.logs.get(run_dir, "")

    def remove_run_dir(self, run_dir):
        self.node.logs.pop(run_dir, None)


class FakeBackend:
    def __init__(self, node: FakeNode):
        self.name = node.name
        self.node = node
        self.transport = FakeTransport(node)
        self.executor = FakeExecutor(node)

    def probe_gpus(self) -> dict:
        if not self.node.online:
            raise TransportError(f"{self.name}: unreachable")
        return self.node.nvml_payload()

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
