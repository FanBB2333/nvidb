"""Exercise queue keyboard input through a real terminal, without a scheduler."""

import codecs
import os
import sys
import time
from types import SimpleNamespace

import pytest
from blessed import Terminal

from nvidb.sched import tui as tui_module
from nvidb.tui_theme import DiffScreen
from test_sched_tui import _state


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="requires a POSIX pseudo-terminal")
def test_queue_run_accepts_real_keyboard_input_with_mouse_reporting(monkeypatch):
    try:
        master, slave = os.openpty()
    except OSError as error:
        pytest.skip(f"pseudo-terminal unavailable: {error}")
    try:
        monkeypatch.setenv("COLUMNS", "100")
        monkeypatch.setenv("LINES", "30")
        term = Terminal(kind="xterm-256color", force_styling=True)
        # pytest captures stdin; connect Blessed's real decoder to our PTY.
        monkeypatch.setattr(term, "_keyboard_fd", slave)
        monkeypatch.setattr(term, "_keyboard_decoder", codecs.getincrementaldecoder("utf-8")(), raising=False)
        lifecycle = []
        state = _state()
        worker = SimpleNamespace(
            start=lambda: lifecycle.append("start"),
            read_state=lambda: state,
            stop=lambda: lifecycle.append("stop"),
            join=lambda timeout: lifecycle.append("join"),
        )
        monkeypatch.setattr(tui_module, "_Worker", lambda **kwargs: worker)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(tui_module, "DiffScreen", lambda terminal: DiffScreen(terminal, writer=lambda text: None))
        tui = tui_module.QueueTUI()
        tui.term = term

        # Each packet follows the previous dispatched key. In particular, a
        # bare Escape cannot accidentally become an Alt+q chord in the test.
        packets = iter([
            b"j", b"k", b"\x1b[B", b"\x1b[A", b"\r", b"\r",
            b"\t", b"\x1b[B", b"\x1b[A", b"\t", b"m", b"\x1b[B",
            b"\x1b", b"\x1b[<35;80;29Mj", b"q",
        ])
        observed = []
        real_handle = tui.handle_key

        def send_next():
            packet = next(packets, None)
            if packet is not None:
                os.write(master, packet)

        def handle_key(key):
            running = real_handle(key)
            observed.append((str(key), key.name, tui.focus, tui.job_index,
                             tui.node_index, tui.show_detail,
                             tui._menu.selected if tui._menu else None, running))
            send_next()
            return running

        monkeypatch.setattr(tui, "handle_key", handle_key)
        real_frame = tui._frame_lines
        first_frame = True

        def frame(state):
            nonlocal first_frame
            lines = real_frame(state)
            if first_frame:
                first_frame = False
                send_next()  # run() has entered cbreak and enabled mouse modes.
            return lines

        monkeypatch.setattr(tui, "_frame_lines", frame)
        real_inkey = term.inkey
        deadline = time.monotonic() + 5

        def inkey(timeout):
            assert time.monotonic() < deadline, f"keyboard input stalled: {observed!r}"
            return real_inkey(timeout=timeout)

        monkeypatch.setattr(term, "inkey", inkey)
        assert tui.run() == 0
        assert lifecycle == ["start", "stop", "join"]
        # A normal terminal may translate Return to newline in cbreak mode.
        assert [entry[0].replace("\r", "\n") for entry in observed] == [
            "j", "k", "\x1b[B", "\x1b[A", "\n", "\n", "\t",
            "\x1b[B", "\x1b[A", "\t", "m", "\x1b[B", "\x1b", "j", "q",
        ]
        assert [entry[3] for entry in observed[:4]] == [1, 0, 1, 0]
        assert [entry[5] for entry in observed[4:6]] == [True, False]
        assert [entry[2] for entry in observed[6:10]] == ["nodes", "nodes", "nodes", "jobs"]
        assert [entry[4] for entry in observed[6:10]] == [0, 1, 0, 0]
        assert [entry[6] for entry in observed[10:13]] == [0, 1, None]
        assert observed[13][3] == 1  # A preceding mouse report cannot swallow j.
        assert observed[-1][-1] is False
    finally:
        os.close(master)
        os.close(slave)
