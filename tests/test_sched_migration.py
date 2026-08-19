"""Opening a queue database that predates the current schema must upgrade it.

Every test elsewhere starts from an empty file, so the shipped `CREATE TABLE`
statements always match what the code expects and no upgrade is ever exercised.
A real queue is years of rows that cannot be recreated, and it is the only place
these paths run. So this builds the old shapes on purpose.
"""
import sqlite3

import pytest

from nvidb.sched import db as dbm

# The `jobs` table as it shipped before lanes and held dependencies existed.
# Kept verbatim rather than derived, so a future edit to the live schema cannot
# quietly change what "old" means here.
LEGACY_SCHEMA = """
CREATE TABLE jobs (
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
    max_runtime_s   INTEGER,
    submitter       TEXT,
    tags            TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    node            TEXT,
    gpu_ids         TEXT,
    remote_pid      INTEGER,
    remote_pgid     INTEGER,
    run_dir         TEXT,
    exit_code       INTEGER,
    attempt         INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    heartbeat_at    TEXT,
    gpu_mem_mb      INTEGER,
    result          TEXT
);
CREATE INDEX idx_jobs_state ON jobs(state);
CREATE TABLE nodes (
    name       TEXT PRIMARY KEY,
    hostname   TEXT,
    port       INTEGER DEFAULT 22,
    username   TEXT,
    state      TEXT NOT NULL DEFAULT 'unknown',
    enabled    INTEGER NOT NULL DEFAULT 1,
    last_seen  TEXT,
    last_error TEXT,
    gpu_count  INTEGER DEFAULT 0,
    probed_at  TEXT
);
"""


@pytest.fixture
def legacy_db(tmp_path):
    """A database holding one job, written by a version that predates lanes."""
    path = tmp_path / "old-queue.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO jobs(name, command, state, created_at, updated_at, node) "
        "VALUES('legacy', 'train.sh', 'running', '2026-01-01T00:00:00+00:00', "
        "'2026-01-01T00:00:00+00:00', 'box')",
    )
    conn.commit()
    conn.close()
    return path


def test_an_old_database_gains_the_new_columns(legacy_db):
    conn = dbm.open_db(legacy_db)
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        assert {"lane", "lane_seq", "depends_any", "held_reason"} <= columns
    finally:
        conn.close()


def test_the_lane_index_is_created_after_its_columns_exist(legacy_db):
    """The index names columns the old table does not have, so creating it
    with the rest of the schema would fail on every existing queue."""
    conn = dbm.open_db(legacy_db)
    try:
        indexes = {
            row["name"]
            for row in conn.execute("PRAGMA index_list(jobs)")
        }
        assert "idx_jobs_lane" in indexes
    finally:
        conn.close()


def test_rows_written_by_the_old_version_still_read_back(legacy_db):
    conn = dbm.open_db(legacy_db)
    try:
        job = dbm.get_job(conn, 1)
        assert job.name == "legacy"
        assert job.state == "running"
        # Fields the old writer knew nothing about read as empty, not as errors.
        assert job.lane is None
        assert job.depends_any == []
        assert job.held_reason is None
        assert job.is_held is False
    finally:
        conn.close()


def test_the_lane_queries_work_against_an_upgraded_database(legacy_db):
    conn = dbm.open_db(legacy_db)
    try:
        assert dbm.lane_jobs(conn, "box:0") == []
        assert dbm.get_lanes(conn) == []
        dbm.ensure_lane(conn, "box:0", node="box", gpu_ids=[0])
        assert [lane.name for lane in dbm.get_lanes(conn)] == ["box:0"]
        assert dbm.next_lane_seq(conn, "box:0") == dbm.LANE_STEP
    finally:
        conn.close()


def test_opening_twice_is_harmless(legacy_db):
    """Every CLI call re-runs the upgrade, so it has to be idempotent."""
    for _ in range(3):
        conn = dbm.open_db(legacy_db)
        conn.close()
    conn = dbm.open_db(legacy_db)
    try:
        assert dbm.get_job(conn, 1).name == "legacy"
    finally:
        conn.close()


def test_a_node_row_written_before_ignoring_existed_still_loads(legacy_db):
    conn = dbm.open_db(legacy_db)
    try:
        conn.execute(
            "INSERT INTO nodes(name, hostname) VALUES('box', '10.0.0.1')"
        )
        conn.commit()
        node = dbm.get_node(conn, "box")
        assert node.ignored is False
        assert node.enabled is True
    finally:
        conn.close()


def test_nodes_without_a_stored_position_still_list_by_name(legacy_db):
    """An upgraded database has no sort_order yet; until a config sync fills
    it in, listings fall back to the old alphabetical order."""
    conn = dbm.open_db(legacy_db)
    try:
        conn.execute("INSERT INTO nodes(name) VALUES('zeta')")
        conn.execute("INSERT INTO nodes(name) VALUES('alpha')")
        conn.commit()
        assert [node.name for node in dbm.get_nodes(conn)] == ["alpha", "zeta"]
    finally:
        conn.close()
