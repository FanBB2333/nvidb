"""Open the queue TUI on a made-up queue, for README screenshots.

usage: NVIDB_HOME=<empty dir> python scripts/screenshots/demo_queue.py [queue TUI args...]

Submits a handful of jobs, lets them start, finish or fail on the in-memory
cluster from tests/fake_cluster.py, then opens `nvidb queue` in this same
process so its ticks keep talking to that cluster. No real node is contacted.
"""
import contextlib
import io
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

import demo_cluster

ROOT = demo_cluster.ROOT
sys.path.insert(0, str(ROOT / "tests"))

from fake_cluster import FakeCluster, FakeGpu, FakeNode  # noqa: E402
from nvidb.cli import main  # noqa: E402
from nvidb.sched import db as dbm  # noqa: E402
from nvidb.sched.scheduler import Scheduler  # noqa: E402

HOME = Path(os.environ["NVIDB_HOME"])
WORKDIR = "/home/alice/projects/vision"
SUBMITTER = "alice <alice@example.com>"
QUEUE_NODES = ("lab1", "lab2")

# name, extra submit options, command
JOBS = [
    ("download-weights", ["--gpus", "0"], "python3 download.py"),
    ("smoke-test", ["--gpus", "0"], "python3 smoke_test.py"),
    ("train-resnet50", ["--lane", "lab1:0", "--vram", "6G", "--note", "baseline, lr=0.1"],
     "python3 train.py --model resnet50 --epochs 90"),
    ("eval-resnet50", ["--lane", "lab1:0", "--vram", "4G", "--after", "3"],
     "python3 eval.py --ckpt runs/resnet50/last.pt"),
    ("finetune-llama", ["--lane", "lab1:1", "--vram", "16G"], "python3 train.py --model llama-3-8b --epochs 3 --lora"),
    ("sweep-lr-1e-4", ["--lane", "lab1:1", "--vram", "16G"], "python3 train.py --model vit-b16 --epochs 30 --lr 1e-4"),
    ("sweep-lr-3e-4", ["--lane", "lab1:1", "--vram", "16G"], "python3 train.py --model vit-b16 --epochs 30 --lr 3e-4"),
    ("preprocess-shards", ["--gpus", "0"], "python3 preprocess.py --shards 256"),
    ("sft-qwen3-8b", ["--lane", "lab2:3", "--vram", "80G"], "accelerate launch sft.py --model qwen3-8b"),
]
# job id -> (exit code, or None while running; progress line; minutes since it started)
OUTCOMES = {1: (0, None, 41), 2: (1, None, 40), 3: (None, "epoch 34/90 loss 0.812", 7),
            5: (0, None, 9), 6: (None, "epoch 11/30 loss 1.204", 2), 8: (None, "shard 97/256", 7),
            9: (None, "step 1200/5000 loss 1.08", 26)}
# job id -> (GPU index, MiB it holds)
VRAM = {3: (0, 6100), 6: (1, 15800), 9: (3, 79000)}


def build_cluster():
    nodes = []
    for name, _address, *_rest, gpus in demo_cluster.NODES:
        if name not in QUEUE_NODES:
            continue
        node = FakeNode(name, [FakeGpu(index, spec[0], spec[2]) for index, (spec, *_gpu) in enumerate(gpus)])
        if name == "lab2":  # someone else's server holds two cards outside the queue
            for index in (0, 1):
                node.add_foreign_process(index, 241000, name="python", username="carol")
                node.gpus[index].util = 86
        nodes.append(node)
    return FakeCluster(*nodes)


def nvidb(*args):
    sys.argv = ["nvidb", *args]
    try:
        main()
    except SystemExit:
        pass


def ago(minutes):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")


def prepare(cluster):
    queue_config = HOME / "queue.yml"
    queue_config.write_text(yaml.safe_dump({"queue": {
        "gpu_mode": "shared", "include_local": False, "backup": {"enabled": False},
        "notify": {"desktop": False}}}))
    os.environ["NVIDB_QUEUE_CONFIG"] = str(queue_config)
    os.environ["NVIDB_QUEUE_NO_REMOTE"] = "1"
    demo_cluster.write_config(HOME, nodes=QUEUE_NODES)

    for name, options, command in JOBS:
        nvidb("job", "submit", "--local", "--no-tick", "-n", name, "--workdir", WORKDIR,
              "--submitter", SUBMITTER, *options, "--", *command.split())
    nvidb("queue", "tick", "--local")

    def node_of(job_id):
        return next(node for node in cluster.nodes.values() if job_id in node.jobs)

    for job_id in (1, 2, 5):
        node_of(job_id).finish_job(job_id, exit_code=OUTCOMES[job_id][0])
    nvidb("queue", "tick", "--local")  # settles 1, 2 and 5, and starts 6 behind 5
    for job_id, (_exit_code, progress, _minutes) in OUTCOMES.items():
        if progress:
            node_of(job_id).report_progress(job_id, progress)
    for job_id, (gpu, mib) in VRAM.items():
        node_of(job_id).job_allocates(job_id, gpu, mib)
    cluster["lab1"].gpus[0].util, cluster["lab1"].gpus[1].util, cluster["lab2"].gpus[3].util = 97, 100, 91
    nvidb("queue", "tick", "--local")

    conn = dbm.open_db(HOME / "queue.db")
    try:
        for job_id, (_exit_code, _progress, minutes) in OUTCOMES.items():
            dbm.update_job(conn, job_id, started_at=ago(minutes))
    finally:
        conn.close()


if __name__ == "__main__":
    tui_args = sys.argv[1:]
    demo_cluster.install()
    cluster = build_cluster()
    Scheduler._default_backend_factory = lambda self, node: cluster.backends[node.name]
    with contextlib.redirect_stdout(io.StringIO()):  # the ticks' own progress lines
        prepare(cluster)
    nvidb("queue", "--local", *tui_args)
