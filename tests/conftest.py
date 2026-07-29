"""Test-wide isolation from whatever `~/.nvidb` the developer actually has.

The queue CLI reads `queue.yml` on every invocation to decide whether it owns
the queue or should forward the command to the machine that does. A developer
whose own laptop forwards to a queue host would otherwise have the test suite
open SSH connections to it, so the tests are pinned to a configuration file that
does not exist and told never to forward.
"""
import pytest

from nvidb.sched import remote as remote_mod


@pytest.fixture(autouse=True)
def isolated_nvidb_config(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDB_QUEUE_CONFIG", str(tmp_path / "absent-queue.yml"))
    monkeypatch.setenv(remote_mod.NO_REMOTE_ENV, "1")
