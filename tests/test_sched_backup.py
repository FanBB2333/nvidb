"""Queue database backups stay coherent while the WAL-backed queue is live."""
from datetime import datetime, timedelta, timezone

import pytest

from nvidb.sched import backup as backup_mod
from nvidb.sched import db as dbm


def _seed(conn, name="train"):
    return dbm.insert_job(
        conn,
        name=name,
        command="python train.py",
        vram_mb=1024,
    )


def test_backup_is_verified_and_contains_committed_queue_state(tmp_path):
    source = tmp_path / "queue.db"
    destination = tmp_path / "snapshot.db"
    conn = dbm.open_db(source)
    try:
        job_id = _seed(conn)
        info = backup_mod.create(conn, str(destination))
    finally:
        conn.close()

    assert info["verified"] is True
    assert info["path"] == str(destination)
    assert info["bytes"] > 0

    restored = dbm.open_db(destination)
    try:
        assert dbm.get_job(restored, job_id).name == "train"
        assert restored.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        restored.close()


def test_backup_never_overwrites_an_existing_file_or_the_source(tmp_path):
    source = tmp_path / "queue.db"
    conn = dbm.open_db(source)
    try:
        _seed(conn)
        existing = tmp_path / "existing.db"
        existing.write_text("keep me", encoding="utf-8")
        with pytest.raises(FileExistsError):
            backup_mod.create(conn, str(existing))
        assert existing.read_text(encoding="utf-8") == "keep me"

        with pytest.raises(ValueError):
            backup_mod.create(conn, str(source))
    finally:
        conn.close()


def test_generated_backups_are_rotated_without_touching_other_files(tmp_path):
    source = tmp_path / "queue.db"
    directory = tmp_path / "backups"
    conn = dbm.open_db(source)
    try:
        _seed(conn)
        start = datetime(2026, 7, 30, tzinfo=timezone.utc)
        for offset in range(4):
            backup_mod.create(
                conn,
                keep=2,
                directory=str(directory),
                now=start + timedelta(seconds=offset),
            )
        unrelated = directory / "notes.txt"
        unrelated.write_text("keep", encoding="utf-8")
        backup_mod.create(
            conn,
            keep=2,
            directory=str(directory),
            now=start + timedelta(seconds=5),
        )
    finally:
        conn.close()

    assert len(list(directory.glob("queue-*.db"))) == 2
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_periodic_backup_waits_for_its_interval(tmp_path):
    source = tmp_path / "queue.db"
    directory = tmp_path / "backups"
    settings = {
        "backup": {
            "enabled": True,
            "interval_hours": 24,
            "keep": 3,
            "directory": str(directory),
        }
    }
    start = datetime(2026, 7, 30, tzinfo=timezone.utc)
    conn = dbm.open_db(source)
    try:
        _seed(conn)
        first = backup_mod.maybe_create(conn, settings, now=start)
        early = backup_mod.maybe_create(
            conn, settings, now=start + timedelta(hours=23)
        )
        second = backup_mod.maybe_create(
            conn, settings, now=start + timedelta(hours=24)
        )
    finally:
        conn.close()

    assert first is not None
    assert early is None
    assert second is not None
    assert len(list(directory.glob("queue-*.db"))) == 2


def test_periodic_backup_respects_a_concurrent_backup_lease(tmp_path):
    source = tmp_path / "queue.db"
    directory = tmp_path / "backups"
    settings = {
        "backup": {
            "enabled": True,
            "directory": str(directory),
        }
    }
    first = dbm.open_db(source)
    second = dbm.open_db(source)
    try:
        _seed(first)
        assert dbm.acquire_lock(
            first, backup_mod.BACKUP_LOCK, "other-daemon", 60
        )
        assert backup_mod.maybe_create(second, settings) is None
        assert not directory.exists()

        dbm.release_lock(first, backup_mod.BACKUP_LOCK, "other-daemon")
        assert backup_mod.maybe_create(second, settings) is not None
    finally:
        first.close()
        second.close()
