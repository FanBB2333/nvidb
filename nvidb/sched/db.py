"""SQLite storage for the nvidb job queue.

This file *is* the coordination channel. Several unrelated processes open the
same database concurrently, so every mutation runs inside an explicit
`BEGIN IMMEDIATE` transaction and the connection is opened in WAL mode with a
generous busy timeout. Readers never block writers and a writer that loses the
race retries instead of failing.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .. import config
from .model import TERMINAL_JOB_STATES, GpuState, Job, Lane, Node, utcnow

SCHEMA_VERSION = 2
DEFAULT_BUSY_TIMEOUT_MS = 15000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS nodes (
    name       TEXT PRIMARY KEY,
    hostname   TEXT,
    port       INTEGER DEFAULT 22,
    username   TEXT,
    state      TEXT NOT NULL DEFAULT 'unknown',
    enabled    INTEGER NOT NULL DEFAULT 1,
    ignored    INTEGER NOT NULL DEFAULT 0,
    last_seen  TEXT,
    last_error TEXT,
    gpu_count  INTEGER DEFAULT 0,
    probed_at  TEXT
);

CREATE TABLE IF NOT EXISTS gpus (
    node            TEXT NOT NULL,
    gpu_index       INTEGER NOT NULL,
    name            TEXT,
    mem_total_mb    INTEGER DEFAULT 0,
    mem_used_mb     INTEGER DEFAULT 0,
    util_percent    INTEGER,
    mem_util_percent INTEGER,
    temperature_c   INTEGER,
    external_mem_mb INTEGER DEFAULT 0,
    external_procs  INTEGER DEFAULT 0,
    queue_mem_mb    INTEGER DEFAULT 0,
    reserved_mb     INTEGER DEFAULT 0,
    queue_jobs      INTEGER DEFAULT 0,
    attribution     TEXT DEFAULT 'processes',
    processes_json  TEXT,
    updated_at      TEXT,
    PRIMARY KEY (node, gpu_index)
);

CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    command         TEXT NOT NULL,
    workdir         TEXT,
    env             TEXT,
    state           TEXT NOT NULL DEFAULT 'pending',
    priority        INTEGER NOT NULL DEFAULT 0,
    gpus            INTEGER NOT NULL DEFAULT 1,
    vram_mb         INTEGER NOT NULL DEFAULT 0,
    node_constraint TEXT,
    depends_on      TEXT,
    -- Dependencies that only have to finish, whatever they finished as.
    depends_any     TEXT,
    max_runtime_s   INTEGER,
    submitter       TEXT,
    tags            TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    node            TEXT,
    gpu_ids         TEXT,
    -- The serial queue this job belongs to, and its place in it.
    lane            TEXT,
    lane_seq        INTEGER,
    remote_pid      INTEGER,
    remote_pgid     INTEGER,
    run_dir         TEXT,
    exit_code       INTEGER,
    attempt         INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    heartbeat_at    TEXT,
    gpu_mem_mb      INTEGER,
    result          TEXT,
    -- `notes` is what a person or a client says about the job; `progress` is
    -- what the job says about itself. Keeping them apart means a job's own
    -- status line can never overwrite a human's annotation.
    notes           TEXT,
    progress        TEXT,
    progress_at     TEXT,
    -- Set while a dependency that ended badly parks this job. It stays pending
    -- and keeps its place; a person decides whether to release or drop it.
    held_reason     TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_node ON jobs(node);

-- One serial queue, normally one GPU. A lane holds the answer to "what does
-- this card run next", which priorities and dependency chains could only
-- imply. Rows are created on demand and outlive the jobs that filled them.
CREATE TABLE IF NOT EXISTS lanes (
    name        TEXT PRIMARY KEY,
    node        TEXT NOT NULL,
    gpu_ids     TEXT NOT NULL DEFAULT '',
    paused      INTEGER NOT NULL DEFAULT 0,
    concurrency INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lanes_node ON lanes(node);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    kind    TEXT NOT NULL,
    job_id  INTEGER,
    node    TEXT,
    message TEXT,
    data    TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id);

CREATE TABLE IF NOT EXISTS locks (
    name        TEXT PRIMARY KEY,
    owner       TEXT,
    acquired_at TEXT,
    expires_at  TEXT
);

-- Something went wrong on a node and a person should know. Alerts outlive the
-- process that noticed them: whichever client happens to run the scheduler pass
-- records the alert, and it stays until someone acknowledges it.
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    kind            TEXT NOT NULL,
    severity        TEXT NOT NULL DEFAULT 'error',
    job_id          INTEGER,
    node            TEXT,
    title           TEXT NOT NULL,
    detail          TEXT,
    acknowledged_at TEXT,
    notified_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_open ON alerts(acknowledged_at);

-- A job's record was closed without proof its remote process died: cancelled or
-- timed out while the node was unreachable. The pid is kept here rather than on
-- the job row because it has to outlive both `job requeue` (which clears the
-- placement) and `job purge` (which deletes the row entirely) - otherwise the
-- process keeps its GPU with nothing left that knows about it.
CREATE TABLE IF NOT EXISTS orphans (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     INTEGER,
    node       TEXT NOT NULL,
    run_dir    TEXT,
    pid        INTEGER,
    pgid       INTEGER,
    reason     TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orphans_node ON orphans(node);
"""

_local = threading.local()


def default_db_path() -> Path:
    """Queue database path, overridable with NVIDB_QUEUE_DB for tests/sandboxes."""
    override = os.environ.get("NVIDB_QUEUE_DB")
    if override:
        return Path(override).expanduser()
    return Path(config.WORKING_DIR).expanduser() / "queue.db"


def connection_path(conn: sqlite3.Connection) -> str:
    """The file a connection is actually attached to.

    Reads that say which database they came from have to ask the connection:
    a caller may have opened one explicitly with `--db-path`, in which case the
    default location is the wrong answer.
    """
    try:
        for row in conn.execute("PRAGMA database_list"):
            if row["name"] == "main" and row["file"]:
                return str(row["file"])
    except sqlite3.Error:
        pass
    return str(default_db_path())


def _retry_while_locked(action, *, attempts: int = 60, delay: float = 0.05):
    """Run `action`, retrying while SQLite reports the database as locked.

    The busy timeout covers ordinary statements, but not all of them: SQLite
    refuses `PRAGMA journal_mode=WAL` immediately rather than consulting the
    busy handler. Several clients opening a brand-new queue at the same moment
    therefore all try to convert the journal and every loser fails outright -
    which is exactly the moment this queue is designed for, since a burst of
    independent agent sessions is how it gets used.
    """
    import random
    import time

    for attempt in range(attempts):
        try:
            return action()
        except sqlite3.OperationalError as error:
            if "locked" not in str(error) and "busy" not in str(error):
                raise
            if attempt == attempts - 1:
                raise
            # Jittered, so a herd of clients does not retry in lockstep.
            time.sleep(delay * (1 + random.random()))


def open_db(path=None) -> sqlite3.Connection:
    """Open (and if needed create) the queue database."""
    db_path = Path(path or default_db_path()).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        timeout=DEFAULT_BUSY_TIMEOUT_MS / 1000.0,
        isolation_level=None,  # explicit transactions only
    )
    conn.row_factory = sqlite3.Row
    # Set the busy timeout before anything that can contend, so every statement
    # after this point waits its turn instead of failing.
    conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
    _retry_while_locked(lambda: conn.execute("PRAGMA journal_mode=WAL"))
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _retry_while_locked(lambda: init_schema(conn))
    return conn


# Columns added after a table first shipped. `CREATE TABLE IF NOT EXISTS` will
# not add them to a database that already exists, so they are applied here.
_ADDED_COLUMNS = {
    "nodes": {
        "ignored": "INTEGER NOT NULL DEFAULT 0",
    },
    "gpus": {
        "attribution": "TEXT DEFAULT 'processes'",
        "mem_util_percent": "INTEGER",
        "queue_mem_mb": "INTEGER DEFAULT 0",
        "processes_json": "TEXT",
    },
    "jobs": {
        "notes": "TEXT",
        "progress": "TEXT",
        "progress_at": "TEXT",
        "depends_any": "TEXT",
        "held_reason": "TEXT",
        "lane": "TEXT",
        "lane_seq": "INTEGER",
    },
}


# Indexes over columns that were added after their table first shipped. These
# cannot live in `_SCHEMA`: it runs before the columns are added, so on an
# existing database the index would name a column that is not there yet.
_ADDED_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_jobs_lane ON jobs(lane, lane_seq)",
)


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables when missing, add new columns, and stamp the version."""
    # executescript() commits whatever is open before it runs, so it must stay
    # outside the transaction helper.
    conn.executescript(_SCHEMA)
    for table, columns in _ADDED_COLUMNS.items():
        existing = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
        }
        for name, definition in columns.items():
            if name in existing:
                continue
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            except sqlite3.OperationalError as error:
                # Another client added the column between the read above and
                # this write. Its work is ours; anything else is a real fault.
                if "duplicate column" not in str(error).lower():
                    raise
    for statement in _ADDED_INDEXES:
        conn.execute(statement)
    with transaction(conn):
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Run a block inside `BEGIN IMMEDIATE`, so writers serialize cleanly.

    Nesting is tracked per connection and per thread: an inner `with` block
    joins the transaction the outermost caller already owns, which lets the
    scheduler compose the small single-purpose helpers below into one atomic
    step without any of them knowing about the others.
    """
    depths = getattr(_local, "depths", None)
    if depths is None:
        depths = _local.depths = {}
    key = id(conn)

    if depths.get(key):
        depths[key] += 1
        try:
            yield conn
        finally:
            depths[key] -= 1
        return

    conn.execute("BEGIN IMMEDIATE")
    depths[key] = 1
    try:
        yield conn
    except Exception:
        depths[key] = 0
        conn.execute("ROLLBACK")
        raise
    else:
        depths[key] = 0
        conn.execute("COMMIT")


# --- meta ------------------------------------------------------------------

def get_meta(conn: sqlite3.Connection, key: str):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )


# --- locks -----------------------------------------------------------------

def acquire_lock(conn: sqlite3.Connection, name: str, owner: str, ttl_seconds: int) -> bool:
    """Take a lease-style lock, stealing it only once the holder's TTL expired.

    A crashed client cannot wedge the queue: its lease simply ages out.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    expires = (now + timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds")
    now_text = now.isoformat(timespec="seconds")
    with transaction(conn):
        row = conn.execute("SELECT * FROM locks WHERE name = ?", (name,)).fetchone()
        if row is not None and row["owner"] and row["owner"] != owner:
            if (row["expires_at"] or "") > now_text:
                return False
        conn.execute(
            "INSERT INTO locks(name, owner, acquired_at, expires_at) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET owner=excluded.owner, "
            "acquired_at=excluded.acquired_at, expires_at=excluded.expires_at",
            (name, owner, now_text, expires),
        )
        return True


def release_lock(conn: sqlite3.Connection, name: str, owner: str) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE locks SET owner=NULL, expires_at=NULL WHERE name = ? AND owner = ?",
            (name, owner),
        )


def lock_holder(conn: sqlite3.Connection, name: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM locks WHERE name = ?", (name,)).fetchone()


# --- events ----------------------------------------------------------------

def add_event(
    conn: sqlite3.Connection,
    kind: str,
    *,
    job_id: Optional[int] = None,
    node: Optional[str] = None,
    message: Optional[str] = None,
    data: Optional[dict] = None,
) -> int:
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO events(ts, kind, job_id, node, message, data) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (
                utcnow(),
                kind,
                job_id,
                node,
                message,
                json.dumps(data, ensure_ascii=False) if data else None,
            ),
        )
        return int(cursor.lastrowid)


def list_events(
    conn: sqlite3.Connection,
    *,
    since_id: Optional[int] = None,
    job_id: Optional[int] = None,
    limit: int = 100,
) -> List[dict]:
    clauses = []
    params: List[Any] = []
    if since_id is not None:
        clauses.append("id > ?")
        params.append(int(since_id))
    if job_id is not None:
        clauses.append("job_id = ?")
        params.append(int(job_id))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    # Paging forward from a cursor must return the *oldest* unseen rows: a
    # client that catches up with `--since <last id>` would otherwise jump to
    # the newest few and silently skip everything in between. Without a cursor
    # the newest rows are the interesting ones, but callers want them
    # oldest-first either way.
    if since_id is not None:
        rows = conn.execute(
            f"SELECT * FROM events {where} ORDER BY id ASC LIMIT ?",
            (*params, int(limit)),
        ).fetchall()
    else:
        rows = list(
            reversed(
                conn.execute(
                    f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ?",
                    (*params, int(limit)),
                ).fetchall()
            )
        )
    out = []
    for row in rows:
        item = dict(row)
        if item.get("data"):
            try:
                item["data"] = json.loads(item["data"])
            except ValueError:
                pass
        out.append(item)
    return out


# --- alerts ----------------------------------------------------------------

def add_alert(
    conn: sqlite3.Connection,
    kind: str,
    title: str,
    *,
    severity: str = "error",
    job_id: Optional[int] = None,
    node: Optional[str] = None,
    detail: Optional[str] = None,
) -> int:
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO alerts(ts, kind, severity, job_id, node, title, detail) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (utcnow(), kind, severity, job_id, node, title, detail),
        )
        return int(cursor.lastrowid)


def list_alerts(
    conn: sqlite3.Connection,
    *,
    open_only: bool = True,
    limit: int = 100,
) -> List[dict]:
    where = "WHERE acknowledged_at IS NULL" if open_only else ""
    rows = conn.execute(
        f"SELECT * FROM alerts {where} ORDER BY id DESC LIMIT ?", (int(limit),)
    ).fetchall()
    return [dict(row) for row in reversed(rows)]


def open_alert_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM alerts WHERE acknowledged_at IS NULL"
    ).fetchone()
    return int(row["n"]) if row else 0


def acknowledge_alerts(
    conn: sqlite3.Connection,
    ids: Optional[Sequence[int]] = None,
    *,
    all_open: bool = False,
) -> int:
    now = utcnow()
    with transaction(conn):
        if all_open:
            cursor = conn.execute(
                "UPDATE alerts SET acknowledged_at = ? WHERE acknowledged_at IS NULL",
                (now,),
            )
        elif ids:
            placeholders = ", ".join("?" for _ in ids)
            cursor = conn.execute(
                f"UPDATE alerts SET acknowledged_at = ? WHERE acknowledged_at IS NULL "
                f"AND id IN ({placeholders})",
                (now, *[int(i) for i in ids]),
            )
        else:
            return 0
        return int(cursor.rowcount or 0)


def undelivered_alerts(conn: sqlite3.Connection, limit: int = 50) -> List[dict]:
    """Alerts nobody has pushed to a notification channel yet."""
    rows = conn.execute(
        "SELECT * FROM alerts WHERE notified_at IS NULL ORDER BY id LIMIT ?",
        (int(limit),),
    ).fetchall()
    return [dict(row) for row in rows]


def mark_alerts_delivered(conn: sqlite3.Connection, ids: Sequence[int]) -> None:
    if not ids:
        return
    placeholders = ", ".join("?" for _ in ids)
    with transaction(conn):
        conn.execute(
            f"UPDATE alerts SET notified_at = ? WHERE id IN ({placeholders})",
            (utcnow(), *[int(i) for i in ids]),
        )


def upsert_node(
    conn: sqlite3.Connection,
    name: str,
    *,
    hostname: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
) -> None:
    """Register a node from config.yml without disturbing its runtime state."""
    with transaction(conn):
        conn.execute(
            "INSERT INTO nodes(name, hostname, port, username) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET hostname=excluded.hostname, "
            "port=excluded.port, username=excluded.username",
            (name, hostname, port or 22, username),
        )


def set_node_state(
    conn: sqlite3.Connection,
    name: str,
    state: str,
    *,
    error: Optional[str] = None,
    gpu_count: Optional[int] = None,
) -> Optional[str]:
    """Update a node's reachability and return the state it had before."""
    with transaction(conn):
        row = conn.execute("SELECT state FROM nodes WHERE name = ?", (name,)).fetchone()
        previous = row["state"] if row else None
        now = utcnow()
        fields = ["state = ?", "probed_at = ?", "last_error = ?"]
        params: List[Any] = [state, now, error]
        if state == "up":
            fields.append("last_seen = ?")
            params.append(now)
        if gpu_count is not None:
            fields.append("gpu_count = ?")
            params.append(int(gpu_count))
        params.append(name)
        conn.execute(f"UPDATE nodes SET {', '.join(fields)} WHERE name = ?", params)
        return previous


def set_node_enabled(conn: sqlite3.Connection, name: str, enabled: bool) -> None:
    """Drain or resume a node.

    Only `enabled` moves: `state` means reachability, and a drained node is
    still probed so its running jobs can be seen to finish. Writing 'drain' into
    `state` here would also fake a state change, making the next probe look like
    the node had just come back online.
    """
    with transaction(conn):
        conn.execute(
            "UPDATE nodes SET enabled = ? WHERE name = ?",
            (1 if enabled else 0, name),
        )


def set_node_ignored(conn: sqlite3.Connection, name: str, ignored: bool) -> None:
    """Hide a node from normal views and suspend all contact with it.

    This is deliberately separate from draining. A drained node keeps being
    probed so work already running there can finish; an ignored node is treated
    as administratively absent until it is explicitly unignored.
    """
    with transaction(conn):
        conn.execute(
            "UPDATE nodes SET ignored = ? WHERE name = ?",
            (1 if ignored else 0, name),
        )


def replace_node_gpus(conn: sqlite3.Connection, node: str, gpus: Sequence[dict]) -> None:
    """Overwrite the cached GPU rows for one node."""
    now = utcnow()
    with transaction(conn):
        conn.execute("DELETE FROM gpus WHERE node = ?", (node,))
        conn.executemany(
            "INSERT INTO gpus(node, gpu_index, name, mem_total_mb, mem_used_mb, "
            "util_percent, mem_util_percent, temperature_c, external_mem_mb, "
            "external_procs, queue_mem_mb, reserved_mb, queue_jobs, attribution, "
            "processes_json, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    node,
                    int(gpu.get("index", 0)),
                    gpu.get("name"),
                    int(gpu.get("mem_total_mb") or 0),
                    int(gpu.get("mem_used_mb") or 0),
                    gpu.get("util_percent"),
                    gpu.get("mem_util_percent"),
                    gpu.get("temperature_c"),
                    int(gpu.get("external_mem_mb") or 0),
                    int(gpu.get("external_procs") or 0),
                    int(gpu.get("queue_mem_mb") or 0),
                    int(gpu.get("reserved_mb") or 0),
                    int(gpu.get("queue_jobs") or 0),
                    gpu.get("attribution") or "processes",
                    json.dumps(gpu.get("processes") or [], ensure_ascii=False),
                    now,
                )
                for gpu in gpus
            ],
        )


def get_nodes(
    conn: sqlite3.Connection,
    *,
    with_gpus: bool = True,
    include_ignored: bool = True,
) -> List[Node]:
    where = "" if include_ignored else "WHERE ignored = 0"
    nodes = [Node.from_row(row) for row in conn.execute(
        f"SELECT * FROM nodes {where} ORDER BY name"
    ).fetchall()]
    if with_gpus:
        by_name = {node.name: node for node in nodes}
        for row in conn.execute("SELECT * FROM gpus ORDER BY node, gpu_index"):
            node = by_name.get(row["node"])
            if node is not None:
                node.gpus.append(GpuState.from_row(row))
    return nodes


def get_node(conn: sqlite3.Connection, name: str) -> Optional[Node]:
    row = conn.execute("SELECT * FROM nodes WHERE name = ?", (name,)).fetchone()
    if row is None:
        return None
    node = Node.from_row(row)
    node.gpus = [
        GpuState.from_row(gpu)
        for gpu in conn.execute(
            "SELECT * FROM gpus WHERE node = ? ORDER BY gpu_index", (name,)
        )
    ]
    return node


def resolve_node_name(conn: sqlite3.Connection, query: str) -> Optional[str]:
    """Match a node by exact name, then case-insensitively, then by substring.

    Lets a client say `--node lab2` instead of `lab2-workstation-tailscale`.
    """
    if not query:
        return None
    names = [row["name"] for row in conn.execute("SELECT name FROM nodes ORDER BY name")]
    if query in names:
        return query
    lowered = query.lower()
    exact = [name for name in names if name.lower() == lowered]
    if len(exact) == 1:
        return exact[0]
    partial = [name for name in names if lowered in name.lower()]
    if len(partial) == 1:
        return partial[0]
    return None


# --- lanes -----------------------------------------------------------------

# The gap left between neighbouring jobs when a lane is renumbered. Wide enough
# that a reorder is a handful of writes rather than a rewrite of the lane.
LANE_STEP = 1000


def ensure_lane(
    conn: sqlite3.Connection,
    name: str,
    *,
    node: str,
    gpu_ids: Sequence[int],
) -> None:
    """Register a lane if it is new, leaving an existing one's settings alone.

    Called for every schedulable GPU on every refresh, so it must never
    overwrite a lane a person has paused or reconfigured.
    """
    now = utcnow()
    with transaction(conn):
        conn.execute(
            "INSERT INTO lanes(name, node, gpu_ids, created_at, updated_at) "
            "VALUES(?, ?, ?, ?, ?) ON CONFLICT(name) DO NOTHING",
            (name, node, ",".join(str(int(i)) for i in gpu_ids), now, now),
        )


def get_lane(conn: sqlite3.Connection, name: str) -> Optional[Lane]:
    row = conn.execute("SELECT * FROM lanes WHERE name = ?", (name,)).fetchone()
    return Lane.from_row(row) if row else None


def get_lanes(conn: sqlite3.Connection, node: Optional[str] = None) -> List[Lane]:
    if node:
        rows = conn.execute(
            "SELECT * FROM lanes WHERE node = ? ORDER BY name", (node,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM lanes ORDER BY name").fetchall()
    return [Lane.from_row(row) for row in rows]


def update_lane(conn: sqlite3.Connection, name: str, **fields) -> None:
    if not fields:
        return
    if "gpu_ids" in fields and isinstance(fields["gpu_ids"], (list, tuple)):
        fields["gpu_ids"] = ",".join(str(int(i)) for i in fields["gpu_ids"])
    if "paused" in fields:
        fields["paused"] = 1 if fields["paused"] else 0
    fields["updated_at"] = utcnow()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with transaction(conn):
        conn.execute(
            f"UPDATE lanes SET {assignments} WHERE name = ?",
            (*fields.values(), name),
        )


def delete_lane(conn: sqlite3.Connection, name: str) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM lanes WHERE name = ?", (name,))


def resolve_lane_name(conn: sqlite3.Connection, query: str) -> Optional[str]:
    """Match a lane by exact name, then case-insensitively, then by substring.

    Lets a person say `lane 406` instead of `406-tailscale:0` when only one
    lane could be meant.
    """
    if not query:
        return None
    names = [row["name"] for row in conn.execute("SELECT name FROM lanes ORDER BY name")]
    if query in names:
        return query
    lowered = query.lower()
    exact = [name for name in names if name.lower() == lowered]
    if len(exact) == 1:
        return exact[0]
    partial = [name for name in names if lowered in name.lower()]
    if len(partial) == 1:
        return partial[0]
    return None


def lane_jobs(
    conn: sqlite3.Connection, lane: str, *, states: Optional[Iterable[str]] = None
) -> List[Job]:
    """A lane's jobs in the order it will run them.

    `lane_seq` is the schedule, so it orders the result; the id only breaks
    ties between rows that were never given distinct places.
    """
    states = list(states) if states else ["pending", "running"]
    placeholders = ", ".join("?" for _ in states)
    rows = conn.execute(
        f"SELECT * FROM jobs WHERE lane = ? AND state IN ({placeholders}) "
        "ORDER BY lane_seq IS NULL, lane_seq ASC, id ASC",
        (lane, *states),
    ).fetchall()
    return [Job.from_row(row) for row in rows]


def next_lane_seq(conn: sqlite3.Connection, lane: str) -> int:
    """The place at the end of a lane, past everything already queued there.

    Terminal jobs count: reusing a finished job's slot would silently put a new
    submission ahead of work that is already waiting.
    """
    row = conn.execute(
        "SELECT MAX(lane_seq) AS top FROM jobs WHERE lane = ?", (lane,)
    ).fetchone()
    top = row["top"] if row and row["top"] is not None else 0
    return int(top) + LANE_STEP


def renumber_lane(conn: sqlite3.Connection, lane: str, ordered_ids: Sequence[int]) -> None:
    """Write an explicit order over a lane's pending jobs.

    Rewriting all of them costs one small write each and removes every way the
    numbering can run out of room, which matters more here than the writes: a
    reorder that silently failed would be indistinguishable from one that
    worked until the wrong job started.
    """
    with transaction(conn):
        row = conn.execute(
            "SELECT MAX(lane_seq) AS top FROM jobs "
            "WHERE lane = ? AND state = 'running'",
            (lane,),
        ).fetchone()
        base = int(row["top"]) if row and row["top"] is not None else 0
        for position, job_id in enumerate(ordered_ids, start=1):
            conn.execute(
                "UPDATE jobs SET lane_seq = ?, updated_at = ? WHERE id = ?",
                (base + position * LANE_STEP, utcnow(), int(job_id)),
            )


# --- jobs ------------------------------------------------------------------

def insert_job(conn: sqlite3.Connection, **fields) -> int:
    now = utcnow()
    payload = {
        "name": fields.get("name") or "",
        "command": fields["command"],
        "workdir": fields.get("workdir"),
        "env": json.dumps(fields.get("env") or {}, ensure_ascii=False),
        "state": "pending",
        "priority": int(fields.get("priority") or 0),
        "gpus": int(fields.get("gpus", 1)),
        "vram_mb": int(fields.get("vram_mb") or 0),
        "node_constraint": fields.get("node_constraint"),
        "depends_on": ",".join(str(x) for x in (fields.get("depends_on") or [])) or None,
        "depends_any": ",".join(str(x) for x in (fields.get("depends_any") or [])) or None,
        "max_runtime_s": fields.get("max_runtime_s"),
        "submitter": fields.get("submitter"),
        "tags": json.dumps(fields.get("tags") or [], ensure_ascii=False),
        "created_at": now,
        "updated_at": now,
        "max_retries": int(fields.get("max_retries") or 0),
        "notes": fields.get("notes") or None,
        "lane": fields.get("lane") or None,
        "lane_seq": fields.get("lane_seq"),
    }
    columns = ", ".join(payload)
    placeholders = ", ".join("?" for _ in payload)
    with transaction(conn):
        cursor = conn.execute(
            f"INSERT INTO jobs({columns}) VALUES({placeholders})",
            tuple(payload.values()),
        )
        return int(cursor.lastrowid)


def update_job(conn: sqlite3.Connection, job_id: int, **fields) -> None:
    if not fields:
        return
    if "gpu_ids" in fields and isinstance(fields["gpu_ids"], (list, tuple)):
        fields["gpu_ids"] = ",".join(str(x) for x in fields["gpu_ids"])
    for key in ("depends_on", "depends_any"):
        if key in fields and isinstance(fields[key], (list, tuple)):
            fields[key] = ",".join(str(x) for x in fields[key]) or None
    if "result" in fields and not isinstance(fields["result"], (str, type(None))):
        fields["result"] = json.dumps(fields["result"], ensure_ascii=False)
    if "env" in fields and isinstance(fields["env"], dict):
        fields["env"] = json.dumps(fields["env"], ensure_ascii=False)
    fields["updated_at"] = utcnow()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with transaction(conn):
        conn.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?",
            (*fields.values(), int(job_id)),
        )


def get_job(conn: sqlite3.Connection, job_id: int) -> Optional[Job]:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (int(job_id),)).fetchone()
    return Job.from_row(row) if row else None


def list_jobs(
    conn: sqlite3.Connection,
    *,
    states: Optional[Iterable[str]] = None,
    node: Optional[str] = None,
    limit: Optional[int] = None,
    newest_first: bool = False,
) -> List[Job]:
    clauses = []
    params: List[Any] = []
    if states:
        states = list(states)
        clauses.append(f"state IN ({', '.join('?' for _ in states)})")
        params.extend(states)
    if node:
        clauses.append("node = ?")
        params.append(node)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order = "DESC" if newest_first else "ASC"
    sql = f"SELECT * FROM jobs {where} ORDER BY id {order}"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [Job.from_row(row) for row in conn.execute(sql, params)]


def pending_jobs(conn: sqlite3.Connection) -> List[Job]:
    """Pending jobs in dispatch order: highest priority first, then FIFO."""
    rows = conn.execute(
        "SELECT * FROM jobs WHERE state = 'pending' ORDER BY priority DESC, id ASC"
    ).fetchall()
    return [Job.from_row(row) for row in rows]


def dependents_of(conn: sqlite3.Connection, job_id: int) -> List[Job]:
    """Unfinished jobs that are waiting on this one, either way of waiting.

    Used to say what a cancellation is about to park, before it happens.
    """
    job_id = int(job_id)
    out = []
    for row in conn.execute(
        "SELECT * FROM jobs WHERE state IN ('pending', 'running') "
        "AND (depends_on IS NOT NULL OR depends_any IS NOT NULL) ORDER BY id"
    ):
        job = Job.from_row(row)
        if job_id in job.depends_on or job_id in job.depends_any:
            out.append(job)
    return out


def held_jobs(conn: sqlite3.Connection) -> List[Job]:
    rows = conn.execute(
        "SELECT * FROM jobs WHERE state = 'pending' AND held_reason IS NOT NULL "
        "ORDER BY id"
    ).fetchall()
    return [Job.from_row(row) for row in rows]


# --- orphaned processes ----------------------------------------------------

def add_orphan(
    conn: sqlite3.Connection,
    *,
    node: str,
    job_id: Optional[int] = None,
    run_dir: Optional[str] = None,
    pid: Optional[int] = None,
    pgid: Optional[int] = None,
    reason: Optional[str] = None,
) -> Optional[int]:
    """Remember a process that must be killed once its node answers again."""
    if not pid or not run_dir:
        return None  # nothing identifiable enough to kill safely later
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO orphans(job_id, node, run_dir, pid, pgid, reason, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (job_id, node, run_dir, int(pid), pgid, reason, utcnow()),
        )
        return int(cursor.lastrowid)


def list_orphans(
    conn: sqlite3.Connection, node: Optional[str] = None
) -> List[Dict[str, Any]]:
    if node:
        rows = conn.execute(
            "SELECT * FROM orphans WHERE node = ? ORDER BY id", (node,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM orphans ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def drop_orphan(conn: sqlite3.Connection, orphan_id: int) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM orphans WHERE id = ?", (int(orphan_id),))


def orphan_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM orphans").fetchone()
    return int(row["n"]) if row else 0


def live_jobs(conn: sqlite3.Connection, node: Optional[str] = None) -> List[Job]:
    """Jobs currently occupying resources (running)."""
    if node:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE state = 'running' AND node = ? ORDER BY id",
            (node,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE state = 'running' ORDER BY id"
        ).fetchall()
    return [Job.from_row(row) for row in rows]


def job_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    rows = conn.execute("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")
    return {row["state"]: int(row["n"]) for row in rows}


def purge_jobs(
    conn: sqlite3.Connection,
    *,
    states: Optional[Iterable[str]] = None,
) -> int:
    """Delete unreferenced terminal jobs. Active jobs are never removed."""
    states = list(states) if states else sorted(TERMINAL_JOB_STATES)
    states = [state for state in states if state in TERMINAL_JOB_STATES]
    if not states:
        return 0
    clauses = [f"state IN ({', '.join('?' for _ in states)})"]
    params: List[Any] = list(states)
    with transaction(conn):
        # A pending/running job still needs its completed prerequisites to
        # exist so dependency checks remain meaningful. Keep those rows until
        # the dependent itself becomes terminal.
        protected = set()
        for row in conn.execute(
            "SELECT depends_on, depends_any FROM jobs "
            "WHERE state IN ('pending', 'running') "
            "AND (depends_on IS NOT NULL OR depends_any IS NOT NULL)"
        ):
            for column in ("depends_on", "depends_any"):
                protected.update(
                    int(value)
                    for value in (row[column] or "").split(",")
                    if value.strip()
                )
        if protected:
            placeholders = ", ".join("?" for _ in protected)
            clauses.append(f"id NOT IN ({placeholders})")
            params.extend(sorted(protected))
        where = " AND ".join(clauses)
        cursor = conn.execute(f"DELETE FROM jobs WHERE {where}", params)
        return int(cursor.rowcount or 0)
