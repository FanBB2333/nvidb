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


@pytest.fixture(autouse=True, scope="session")
def colour_decided_per_call():
    """Make termcolor decide afresh, on every call, whether it may colour.

    termcolor 3 memoises its first answer to "can this process colour?" for
    the life of the interpreter. Under pytest that first answer is taken with
    stdout captured - no colour - and a later test that sets `FORCE_COLOR`
    would be stuck with it. Clearing the memo around each test turned out not
    to be enough: on CI the colour tests still failed on some jobs and not
    others, so something can fill it again before a test sets the variable.
    Swapping in the undecorated function restores the pre-3.0 behaviour the
    colour assertions were written against, whatever the timing.
    """
    try:
        from termcolor import termcolor as impl
    except ImportError:
        yield
        return
    # The memoised predicate was renamed part-way through the 3.x line:
    # `_can_do_colour` up to 3.1, `can_colorize` from 3.2 on. termcolor < 3
    # keeps no memo and has nothing to swap.
    swapped = {}
    for name in ("can_colorize", "_can_do_colour"):
        cached = getattr(impl, name, None)
        plain = getattr(cached, "__wrapped__", None)
        if plain is not None:
            swapped[name] = cached
            setattr(impl, name, plain)
    try:
        yield
    finally:
        for name, cached in swapped.items():
            setattr(impl, name, cached)
