"""nvidb job queue: a small slurm-like scheduler backed by one SQLite file.

The database is the only shared state, so independent clients coordinate simply
by opening it. Typical use from Python:

    from nvidb.sched import open_queue

    with open_queue() as queue:
        job_id = queue.submit("python train.py", vram="20G", name="train")
        queue.tick()
"""
from .model import (  # noqa: F401
    ACTIVE_JOB_STATES,
    JOB_STATES,
    TERMINAL_JOB_STATES,
    GpuState,
    Job,
    Node,
    format_duration,
    format_mb,
    parse_size_mb,
)
from .scheduler import Scheduler, load_settings  # noqa: F401
from . import db  # noqa: F401


class Queue:
    """A `Scheduler` bound to an open connection, usable as a context manager."""

    def __init__(self, conn, scheduler: Scheduler):
        self.conn = conn
        self.scheduler = scheduler

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __getattr__(self, item):
        return getattr(self.scheduler, item)

    def close(self):
        try:
            self.scheduler.close()
        finally:
            self.conn.close()


def open_queue(db_path=None, *, cfg=None) -> Queue:
    """Open the queue database and return a ready-to-use scheduler."""
    conn = db.open_db(db_path)
    scheduler = Scheduler(conn, cfg=cfg)
    scheduler.sync_nodes_from_config()
    return Queue(conn, scheduler)
