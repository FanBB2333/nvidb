from types import SimpleNamespace

import pytest
from blessed import Terminal
from blessed.keyboard import resolve_sequence

from nvidb.mouse import (
    DISABLE_SEQUENCE,
    ENABLE_SEQUENCE,
    MouseSequenceParser,
    set_mouse_reporting,
)


def _keystrokes(text):
    """Split a byte stream the way blessed's inkey() would hand it over."""
    term = Terminal(kind="xterm-256color", force_styling=True)
    keymap, keycodes = term._keymap, term._keycodes
    while text:
        keystroke = resolve_sequence(text, keymap, keycodes)
        text = text[len(keystroke) :]
        yield keystroke


def _drive(stream):
    parser = MouseSequenceParser()
    events, keys = [], []
    for keystroke in _keystrokes(stream):
        new_events, new_keys = parser.feed(keystroke)
        events.extend(new_events)
        keys.extend(new_keys)
    keys.extend(parser.flush())
    return events, keys


def test_sgr_reports_become_mouse_events():
    events, keys = _drive("\x1b[<0;42;7M\x1b[<0;42;7m\x1b[<64;10;3M")

    assert keys == []
    assert [(e.button, e.column, e.row, e.pressed) for e in events] == [
        (0, 42, 7, True),
        (0, 42, 7, False),
        (64, 10, 3, True),
    ]
    assert events[0].is_left_press
    assert events[1].is_left_press is False
    assert events[2].is_wheel_up
    assert not any(event.is_motion for event in events)


def test_motion_reports_are_distinct_from_clicks_and_scrolls():
    # 35 is movement with no button held; 32/33/34 are left/middle/right drags.
    # Modifier bits may also accompany motion reports.
    buttons = [35, 32, 33, 34, 35 | 4, 32 | 16]
    events, keys = _drive("".join(f"\x1b[<{button};42;7M" for button in buttons))

    assert keys == []
    assert [event.button for event in events] == buttons
    for event in events:
        assert event.is_motion
        assert not event.is_left_press
        assert not event.is_wheel_up
        assert not event.is_wheel_down


def test_complete_report_in_one_keystroke_is_decoded():
    parser = MouseSequenceParser()

    events, keys = parser.feed("\x1b[<35;42;7M")

    assert keys == []
    assert len(events) == 1
    assert events[0].is_motion
    assert (events[0].column, events[0].row) == (42, 7)
    assert parser.flush() == []


def test_clicks_and_scrolls_survive_interleaved_motion():
    events, keys = _drive(
        "\x1b[<35;42;7M\x1b[<0;42;7M\x1b[<32;42;8M"
        "\x1b[<0;42;8m\x1b[<64;42;8M\x1b[<65;42;8M"
    )

    assert keys == []
    assert len(events) == 6
    assert events[0].is_motion
    assert events[1].is_left_press
    assert events[2].is_motion
    assert not events[3].pressed
    assert not events[3].is_motion
    assert events[4].is_wheel_up
    assert events[5].is_wheel_down


def test_mouse_reporting_enables_motion_and_disables_every_mode():
    assert "\x1b[?1003h" in ENABLE_SEQUENCE
    for mode in (1000, 1003, 1006):
        assert f"\x1b[?{mode}h" in ENABLE_SEQUENCE
        assert f"\x1b[?{mode}l" in DISABLE_SEQUENCE


@pytest.mark.parametrize("cache", [None, "", {}])
def test_setting_mouse_reporting_supports_old_and_new_blessed(cache, capsys):
    term = SimpleNamespace()
    if cache is not None:
        term._dec_mode_cache = cache

    set_mouse_reporting(term, True)
    assert capsys.readouterr().out == ENABLE_SEQUENCE
    if isinstance(cache, dict):
        assert cache == {1000: 1, 1003: 1, 1006: 1}

    set_mouse_reporting(term, False)
    assert capsys.readouterr().out == DISABLE_SEQUENCE
    if isinstance(cache, dict):
        assert cache == {1000: 2, 1003: 2, 1006: 2}
    elif cache is None:
        assert not hasattr(term, "_dec_mode_cache")
    else:
        assert term._dec_mode_cache == cache


def test_complete_mouse_reports_do_not_wait_for_escape(monkeypatch):
    term = Terminal(kind="xterm-256color", force_styling=True)
    if not isinstance(getattr(term, "_dec_mode_cache", None), dict):
        pytest.skip("This Blessed version uses the fragmented-report parser")
    set_mouse_reporting(term, True)
    waits = []
    monkeypatch.setattr(term, "kbhit", lambda timeout=None: waits.append(timeout) or False)
    reports = ["\x1b[<35;42;7M", "\x1b[<35;43;7M", "\x1b[<0;43;7M"]
    term.ungetch("".join(reports))
    parser = MouseSequenceParser()

    for report in reports:
        key = term.inkey(timeout=0.1)
        assert str(key) == report  # One inkey call per complete report.
        events, keys = parser.feed(key)
        assert len(events) == 1
        assert keys == []
    assert waits and all(timeout == 0 for timeout in waits)


def test_native_mouse_decoding_preserves_real_escape_wait(monkeypatch):
    term = Terminal(kind="xterm-256color", force_styling=True)
    set_mouse_reporting(term, True)
    waits = []
    monkeypatch.setattr(term, "kbhit", lambda timeout=None: waits.append(timeout) or False)
    term.ungetch("\x1b")

    key = term.inkey(timeout=0.1)

    assert key.name == "KEY_ESCAPE"
    assert any(timeout > 0 for timeout in waits)
    parser = MouseSequenceParser()
    assert parser.feed(key) == ([], [])
    assert parser.flush() == [key]


def test_native_mouse_decoding_preserves_split_arrow_keys(monkeypatch):
    term = Terminal(kind="xterm-256color", force_styling=True)
    set_mouse_reporting(term, True)
    remaining = list("[D")
    waits = []

    def kbhit(timeout=None):
        waits.append(timeout)
        # The continuation arrives only after the initial Escape byte, while
        # inkey waits for it. A global esc_delay=0 would break this arrow.
        return bool(remaining) and timeout is not None and timeout > 0

    monkeypatch.setattr(term, "kbhit", kbhit)
    monkeypatch.setattr(term, "getch", lambda **kwargs: remaining.pop(0))
    term.ungetch("\x1b")

    key = term.inkey(timeout=0.1)

    assert key.name == "KEY_LEFT"
    assert str(key) == "\x1b[D"
    assert remaining == []
    assert any(timeout > 0 for timeout in waits)
    assert MouseSequenceParser().feed(key) == ([], [key])


def test_regular_keys_pass_through_with_their_names():
    events, keys = _drive("\x1b[<0;5;5Mj\x1b[Dq")

    assert len(events) == 1
    assert [(str(key), key.name) for key in keys] == [
        ("j", None),
        ("\x1b[D", "KEY_LEFT"),
        ("q", None),
    ]


def test_a_real_escape_key_is_released_not_swallowed():
    parser = MouseSequenceParser()
    keystrokes = list(_keystrokes("\x1b"))

    events, keys = parser.feed(keystrokes[0])
    assert (events, keys) == ([], [])

    released = parser.flush()
    assert [str(key) for key in released] == ["\x1b"]
    assert parser.flush() == []


def test_escape_followed_by_a_normal_key_replays_both():
    events, keys = _drive("\x1bZ")

    assert events == []
    # Older blessed hands this over as ESC then Z; newer blessed resolves it to
    # one alt-shift-Z keystroke. Either way nothing may be swallowed.
    assert "".join(str(key) for key in keys) == "\x1bZ"


def test_truncated_report_does_not_wedge_the_parser():
    events, keys = _drive("\x1b[<0;42" + "9" * 30 + "j")

    assert events == []
    assert "j" in [str(key) for key in keys]
