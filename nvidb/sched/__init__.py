"""nvidb job queue: a small slurm-like scheduler backed by one SQLite file.

The database is the only shared state, so independent clients coordinate simply
by opening it. The CLI (`nvidb queue …`) is the front door; the Scheduler and
db modules below are the building blocks it assembles.
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
