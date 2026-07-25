from blessed import Terminal
from blessed.keyboard import resolve_sequence

from nvidb.mouse import MouseSequenceParser


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
    assert [str(key) for key in keys] == ["\x1b", "Z"]


def test_truncated_report_does_not_wedge_the_parser():
    events, keys = _drive("\x1b[<0;42" + "9" * 30 + "j")

    assert events == []
    assert "j" in [str(key) for key in keys]
