"""Run nvidb against a made-up cluster, for README screenshots.

usage: NVIDB_HOME=<dir> python scripts/screenshots/demo_cluster.py <nvidb args...>
       e.g. `--remote`, `log --remote --interval 2`, `web --port 8620`

SSH, NVML, `ps` and the local machine's identity are all replaced by the
nodes below, so a screenshot never shows a real host, user, address or GPU.
`write_config` puts matching servers into $NVIDB_HOME/config.yml.
"""
import getpass
import platform
import random
import socket
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MIB = 1024 * 1024
# name, architecture, VRAM MiB, power limit W, has a fan, PCIe (gen, width)
H100 = ("NVIDIA H100 80GB HBM3", "Hopper", 81559, 700, False, (5, 16))
B300 = ("NVIDIA B300 SXM6 AC", "Blackwell", 275040, 1100, False, (6, 16))
RTX5090 = ("NVIDIA GeForce RTX 5090", "Blackwell", 32607, 575, True, (5, 16))

COMMANDS = {
    "llama": "python train.py --model llama-3-8b --data /data/pile --lr 2e-5 --bf16 --grad-ckpt",
    "vit": "torchrun --nproc_per_node=2 pretrain.py --config configs/vit_l16.yaml --epochs 300",
    "vllm": "python -m vllm.entrypoints.openai.api_server --model Qwen3-32B --tensor-parallel-size 2",
    "sft": "accelerate launch sft.py --model qwen3-8b --dataset ultrachat --epochs 3",
    "eval": "python eval.py --ckpt runs/resnet50/last.pt --split val --batch-size 512",
    "sample": "python sample.py --model sdxl --prompts prompts.txt --steps 30",
    "xorg": "/usr/lib/xorg/Xorg vt1 -displayfd 3 -auth /run/user/128/gdm/Xauthority -nolisten tcp",
}

# nickname, address, CPU cores, CPU %, memory (used, total) GB, swap (used, total) GB, GPUs.
# A GPU is (spec, util %, used MiB, temp C, power W, memory util %, processes) and a
# process is (pid, user, MiB, command key, CPU %, elapsed).
NODES = [
    ("lab1", "192.168.3.27", 64, 46, (388, 1007), (0.4, 8), [
        (H100, 98, 71214, 71, 652, 61, [(48213, "alice", 71204, "llama", 101.2, "2-03:14:55")]),
        (H100, 93, 63930, 68, 611, 55, [(51877, "bob", 31962, "vit", 99.6, "17:42:08"),
                                        (51878, "bob", 31958, "vit", 99.1, "17:42:08")]),
    ]),
    ("lab2", "192.168.4.113", 128, 31, (846, 2015), (0.0, 0), [
        (B300, 87, 241388, 66, 1012, 48, [(72305, "carol", 241378, "vllm", 64.0, "5-11:20:31")]),
        (B300, 85, 240917, 65, 998, 47, [(72306, "carol", 240907, "vllm", 63.2, "5-11:20:31")]),
        (B300, 76, 189433, 61, 884, 39, [(80412, "alice", 189423, "sft", 98.7, "06:12:44")]),
        (B300, 0, 9, 34, 142, 0, []),
    ]),
    ("lab3", "192.168.5.64", 32, 12, (74, 251), (1.2, 16), [
        (RTX5090, 64, 18311, 62, 402, 41, [(9314, "dave", 18210, "eval", 100.4, "01:05:37"),
                                           (1840, "gdm", 91, "xorg", 0.0, "21-04:11:02")]),
        (RTX5090, 52, 22790, 58, 351, 37, [(9902, "dave", 22780, "sample", 97.9, "38:21")]),
    ] + [(RTX5090, 0, 3, 31 + i, 18 + i, 0, []) for i in range(6)]),
]
BY_ADDRESS = {node[1]: node for node in NODES}
_processes = {}  # (address, pid) -> the fields `ps` would have reported


def _jitter(value, spread, low=0, high=100):
    return max(low, min(high, value + random.uniform(-spread, spread))) if value else value


def _snapshot(address):
    gpus = []
    for index, (spec, util, used, temp, power, mem_util, procs) in enumerate(BY_ADDRESS[address][6]):
        name, architecture, total, limit, has_fan, (gen, width) = spec
        busy = util > 0
        processes = []
        for pid, user, mib, command, cpu, elapsed in procs:
            processes.append({"pid": pid, "process_name": COMMANDS[command].split()[0],
                              "type": "G" if user == "gdm" else "C", "used_gpu_memory_bytes": mib * MIB})
            _processes[(address, str(pid))] = {
                "username": user, "cpu_percent": _jitter(cpu, 2, 0, 400), "mem_percent": round(mib / 9000, 1),
                "rss_kb": 1.5e6 + mib * 40, "elapsed": elapsed, "state": "Sl" if user == "gdm" else "Rl",
                "threads": 4 if user == "gdm" else 48, "command": COMMANDS[command],
            }
        gpus.append({
            "gpu_index": index, "name": name, "architecture": architecture,
            "memory_total_bytes": total * MIB, "memory_used_bytes": used * MIB,
            "memory_free_bytes": (total - used) * MIB,
            "gpu_util_percent": round(_jitter(util, 4)), "memory_util_percent": round(_jitter(mem_util, 3)),
            "pcie_tx_kib_per_s": random.randint(9000, 26000) if busy else random.randint(0, 40),
            "pcie_rx_kib_per_s": random.randint(60000, 180000) if busy else random.randint(0, 40),
            "pcie_link_gen_current": gen if busy else 1, "pcie_link_width_current": width,
            "pcie_link_gen_max": gen, "pcie_link_width_max": width,
            "power_usage_mw": int(_jitter(power, power * 0.04, 0, limit) * 1000), "power_limit_mw": limit * 1000,
            "temperature_c": temp, "fan_percent": (30 + temp // 3 if busy else 30) if has_fan else None,
            "performance_state": 0 if busy else 8, "processes": processes,
        })
    return {"ok": True, "backend": "ctypes", "driver_version": "580.95.05", "cuda_version": "13.0", "gpus": gpus}


def _system_stats(address):
    _name, _address, cores, cpu, (mem_used, mem_total), (swap_used, swap_total), _gpus = BY_ADDRESS[address]
    return {"cpu_cores": cores, "cpu_percent": _jitter(cpu, 3), "mem_used_gb": mem_used,
            "mem_total_gb": mem_total, "swap_used_gb": swap_used, "swap_total_gb": swap_total}


def install():
    """Swap every real data source for the made-up cluster."""
    platform.system = lambda: "Linux"
    platform.release = lambda: "6.8.0-60-generic"
    platform.machine = lambda: "x86_64"
    platform.node = lambda: "workstation"
    socket.gethostname = lambda: "workstation"
    getpass.getuser = lambda: "alice"

    from nvidb import connection

    def connect(self, **_):
        self._set_connected()
        return True

    remote = connection.RemoteClient
    remote.connect = connect
    remote.ensure_connected = lambda self: True
    remote._transport_alive = lambda self: True
    remote.query_nvml_snapshot = lambda self: _snapshot(self.host)
    remote.get_system_stats = lambda self: _system_stats(self.host)
    remote.execute_command = lambda self, command: ""
    local = connection.LocalClient
    local.query_nvml_snapshot = lambda self: {"ok": False, "backend": "pynvml", "error": "NVML Shared Library Not Found"}
    local.get_system_stats = lambda self: {
        "cpu_cores": 16, "cpu_percent": _jitter(14, 3), "mem_used_gb": 21.4, "mem_total_gb": 64,
        "swap_used_gb": 0.3, "swap_total_gb": 8, "load_avg": (2.12, 1.87, 1.64)}
    local.execute_command = lambda self, command: ""
    connection.BaseClient._query_pid_details = lambda self, pids, fields, chunk_size: {
        str(pid): _processes[(self.host, str(pid))] for pid in pids if (self.host, str(pid)) in _processes}


def write_config(home, mode="unified", nodes=None):
    """Write $NVIDB_HOME/config.yml listing the made-up servers (or only `nodes`)."""
    config = {
        "basic": {"compact": False, "remote": True},
        "view": {"mode": mode, "detailed": False, "sort": "utilization", "filter": "all", "processes": True,
                 "trends": False, "group_by_node": True, "hide_unsupported": True, "mouse": True,
                 "theme": "classic"},
        "servers": [{"hostname": address, "port": 22, "username": "alice", "nickname": name, "auth": "key"}
                    for name, address, *_ in NODES if nodes is None or name in nodes],
    }
    Path(home).mkdir(parents=True, exist_ok=True)
    Path(home, "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    install()
    from nvidb.cli import main

    sys.argv = ["nvidb", *sys.argv[1:]]
    main()
