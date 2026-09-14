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


@pytest.fixture(autouse=True)
def fresh_colour_decision():
    """Let each test decide for itself whether termcolor may emit colour.

    termcolor 3 remembers its first answer to "can this process colour?" for
    the life of the interpreter. Under pytest that first answer is taken with
    stdout captured - no colour - and a later test that sets `FORCE_COLOR`
    would be stuck with it. Clearing the memo restores the pre-3.0 behaviour
    the colour assertions were written against.
    """
    try:
        from termcolor.termcolor import can_colorize
    except ImportError:  # termcolor < 3 keeps no memo
        can_colorize = None
    if can_colorize is not None and hasattr(can_colorize, "cache_clear"):
        can_colorize.cache_clear()
    yield
    if can_colorize is not None and hasattr(can_colorize, "cache_clear"):
        can_colorize.cache_clear()
