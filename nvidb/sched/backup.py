"""Consistent, optionally scheduled backups of the queue database.

The queue uses SQLite in WAL mode, so copying ``queue.db`` with a filesystem
command can miss committed pages that still live in ``queue.db-wal``. SQLite's
online backup API takes a coherent snapshot while the daemon and clients keep
using the source database.
"""
from __future__ import annotations

import os
import socket
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .. import config as nvidb_config
from . import db as dbm

LAST_BACKUP_META = "last_backup_at"
BACKUP_LOCK = "database_backup"
DEFAULT_SETTINGS = {
    "enabled": False,
    "interval_hours": 24.0,
    "keep": 7,
    "directory": None,
}


def default_directory() -> Path:
    return Path(nvidb_config.WORKING_DIR).expanduser() / "backups"


def load_settings(queue_settings: Optional[dict] = None) -> Dict[str, Any]:
    """Normalize ``queue.backup`` without making malformed values fatal."""
    raw = queue_settings.get("backup") if isinstance(queue_settings, dict) else None
    settings = dict(DEFAULT_SETTINGS)
    if raw is False:
        return settings
    if raw is True:
        settings["enabled"] = True
        return settings
    if not isinstance(raw, dict):
        return settings

    settings["enabled"] = bool(raw.get("enabled", settings["enabled"]))
    try:
        settings["interval_hours"] = max(0.0, float(raw.get("interval_hours", 24)))
    except (TypeError, ValueError):
        pass
    try:
        settings["keep"] = max(0, int(raw.get("keep", 7)))
    except (TypeError, ValueError):
        pass
    directory = raw.get("directory")
    if directory:
        path = Path(str(directory)).expanduser()
        if not path.is_absolute():
            path = Path(nvidb_config.WORKING_DIR).expanduser() / path
        settings["directory"] = str(path)
    return settings


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _generated_path(directory: Path, now: datetime) -> Path:
    stem = now.astimezone(timezone.utc).strftime("queue-%Y%m%dT%H%M%S.%fZ")
    candidate = directory / f"{stem}.db"
    suffix = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{suffix}.db"
        suffix += 1
    return candidate


def _same_path(first: Path, second: Path) -> bool:
    return os.path.abspath(str(first)) == os.path.abspath(str(second))


def _publish(
    temporary: Path,
    target: Path,
    *,
    generated: bool,
    created: datetime,
) -> Path:
    """Publish a complete file without ever replacing an existing path.

    A hard link is atomic and fails when ``target`` already exists. Because the
    temporary database is in the same directory, this also keeps publication on
    one filesystem. Generated names retry with a suffix if another process won
    the same timestamp; an explicit destination remains strictly no-clobber.
    """
    while True:
        try:
            os.link(str(temporary), str(target))
            return target
        except FileExistsError:
            if not generated:
                raise
            target = _generated_path(target.parent, created)


def _prune(directory: Path, keep: int) -> list:
    if keep <= 0:
        return []
    backups = sorted(
        (
            path
            for path in directory.glob("queue-*.db")
            if path.is_file()
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    removed = []
    for path in backups[keep:]:
        path.unlink()
        removed.append(str(path))
    return removed


def create(
    conn: sqlite3.Connection,
    destination: Optional[str] = None,
    *,
    keep: int = 0,
    directory: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Create and verify an atomic SQLite backup.

    An explicit destination is never overwritten. Generated backups use a
    timestamped name and may be rotated with ``keep``.
    """
    created = now or _utc_now()
    created_at = _iso(created)
    source = Path(dbm.connection_path(conn)).expanduser()
    backup_dir = Path(directory).expanduser() if directory else default_directory()
    generated = destination is None

    if destination:
        target = Path(destination).expanduser()
        if _same_path(source, target):
            raise ValueError("backup destination is the queue database itself")
        if target.exists() and target.is_dir():
            backup_dir = target
            target = _generated_path(backup_dir, created)
            generated = True
        elif target.exists():
            raise FileExistsError(f"backup already exists: {target}")
        else:
            backup_dir = target.parent
    else:
        target = _generated_path(backup_dir, created)

    if _same_path(source, target):
        raise ValueError("backup destination is the queue database itself")

    backup_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        suffix=".db",
        dir=str(backup_dir),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    target_conn = None
    try:
        target_conn = sqlite3.connect(str(temporary))
        conn.backup(target_conn)
        # Leases describe processes using the source database, not persistent
        # queue state. None of them may survive into a restored snapshot.
        target_conn.execute(
            "UPDATE locks SET owner=NULL, acquired_at=NULL, expires_at=NULL"
        )
        # Record the snapshot's own creation time inside the copy as well as in
        # the live source. A restored database then reports the backup it came
        # from and does not immediately create a duplicate periodic snapshot.
        target_conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (LAST_BACKUP_META, created_at),
        )
        target_conn.commit()
        check = target_conn.execute("PRAGMA quick_check").fetchone()
        if not check or check[0] != "ok":
            raise sqlite3.DatabaseError(
                f"backup verification failed: {check[0] if check else 'no result'}"
            )
        target_conn.close()
        target_conn = None
        temporary.chmod(0o600)
        target = _publish(
            temporary,
            target,
            generated=generated,
            created=created,
        )
    finally:
        if target_conn is not None:
            target_conn.close()
        try:
            temporary.unlink()
        except OSError:
            pass

    dbm.set_meta(conn, LAST_BACKUP_META, created_at)
    removed = _prune(backup_dir, int(keep))
    return {
        "path": str(target),
        "source": str(source),
        "created_at": created_at,
        "bytes": target.stat().st_size,
        "verified": True,
        "pruned": removed,
    }


def maybe_create(
    conn: sqlite3.Connection,
    queue_settings: Optional[dict],
    *,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Create a configured periodic backup when its interval has elapsed."""
    settings = load_settings(queue_settings)
    if not settings["enabled"]:
        return None

    owner = f"{socket.gethostname()}:{os.getpid()}"
    if not dbm.acquire_lock(conn, BACKUP_LOCK, owner, 300):
        return None

    try:
        # Recheck after acquiring the lease: another daemon may have completed
        # the due backup while this process was waiting for the scheduler.
        current = now or _utc_now()
        previous = _parse_iso(dbm.get_meta(conn, LAST_BACKUP_META))
        interval_seconds = float(settings["interval_hours"]) * 3600
        if (
            previous is not None
            and (current - previous).total_seconds() < interval_seconds
        ):
            return None

        return create(
            conn,
            keep=int(settings["keep"]),
            directory=settings["directory"],
            now=current,
        )
    finally:
        dbm.release_lock(conn, BACKUP_LOCK, owner)
