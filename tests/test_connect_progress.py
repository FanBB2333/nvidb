"""Starting up shows one progress line instead of a wall of SSH logs."""
import io
import logging
import re

from nvidb import connection


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class FakeClient:
    def __init__(self, description, *, ok=True, boom=None):
        self.description = description
        self.ok = ok
        self.boom = boom
        self.connect_kwargs = None
        self.connect_error = None

    def connect(self, *, allow_prompt=True, announce=True):
        self.connect_kwargs = {"allow_prompt": allow_prompt, "announce": announce}
        if self.boom:
            raise RuntimeError(self.boom)
        return self.ok

    def _set_connect_error(self, message, error_type=None):
        self.connect_error = (message, error_type)


class FakePool:
    """The slice of NVClientPool connect_all touches."""

    def __init__(self, clients):
        self.pool = clients

    connect_all = connection.NVClientPool.connect_all


def _visible(text):
    """Strip the styling so assertions read the line, not the escapes."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def test_a_terminal_run_reports_progress_on_one_self_rewriting_line():
    stream = FakeTTY()
    progress = connection._ConnectProgress(3, stream=stream)
    assert progress.enabled

    for index, name in enumerate(["alpha", "beta", "gamma"]):
        progress.show(name)
        progress.mark(index != 1)
    progress.finish()

    written = stream.getvalue()
    # One line, rewritten in place: every repaint returns to column 0 and
    # wipes to the end, and nothing ever emits a newline.
    assert "\n" not in written
    assert written.count("\r") == 4  # three repaints plus the final wipe
    assert written.endswith("\r\x1b[K")
    assert "1/3" in _visible(written) and "2/3" in _visible(written)
    assert progress.done == 3 and progress.failed == 1


def test_a_piped_run_keeps_its_plain_log_output():
    stream = io.StringIO()  # not a tty
    progress = connection._ConnectProgress(3, stream=stream)

    assert not progress.enabled
    progress.show("alpha")
    progress.mark(True)
    progress.finish()
    assert stream.getvalue() == ""


def test_an_empty_pool_draws_nothing():
    assert not connection._ConnectProgress(0, stream=FakeTTY()).enabled


def test_a_broken_stdout_disables_the_meter_instead_of_raising():
    class Broken(FakeTTY):
        def write(self, text):
            raise OSError("stdout is gone")

    progress = connection._ConnectProgress(2, stream=Broken())
    progress.show("alpha")  # must not raise
    assert not progress.enabled


def test_connect_all_visits_every_client_and_records_failures():
    clients = [
        FakeClient("alpha"),
        FakeClient("beta", ok=False),
        FakeClient("gamma", boom="host unreachable"),
    ]
    FakePool(clients).connect_all()

    for client in clients:
        # `announce=False` is what keeps each client from logging its own
        # "Connected to ..." line over the meter.
        assert client.connect_kwargs == {"allow_prompt": True, "announce": False}
    assert clients[2].connect_error == ("host unreachable", "connect")


def test_connect_all_leaves_logging_and_the_pause_hook_as_it_found_them(monkeypatch):
    monkeypatch.setattr(connection.sys, "stdout", FakeTTY())
    seen = []

    class Prompting(FakeClient):
        def connect(self, *, allow_prompt=True, announce=True):
            # A password prompt from deep inside a client must be able to
            # take the line back before it asks.
            seen.append(connection._active_connect_progress)
            connection.pause_connect_progress()
            return True

    FakePool([Prompting("alpha")]).connect_all()

    assert seen and seen[0] is not None and seen[0].enabled
    assert connection._active_connect_progress is None
    assert logging.root.manager.disable == 0


def test_a_failing_client_still_restores_logging(monkeypatch):
    monkeypatch.setattr(connection.sys, "stdout", FakeTTY())

    class Exploding(FakeClient):
        def connect(self, *, allow_prompt=True, announce=True):
            raise KeyboardInterrupt

    try:
        FakePool([Exploding("alpha")]).connect_all()
    except KeyboardInterrupt:
        pass
    assert logging.root.manager.disable == 0
    assert connection._active_connect_progress is None
