"""SGR mouse reporting helpers for the nvidb TUI.

blessed resolves only the escape sequences in the terminfo database, so an SGR
mouse report (``ESC [ < button ; column ; row M|m``) arrives from ``inkey()`` as
a ``KEY_ESCAPE`` keystroke followed by its individual characters. The parser
here reassembles those keystrokes into mouse events and hands anything that is
not a mouse report back to the caller untouched.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# 1000: report button press/release. 1006: SGR coordinates, which lift the
# 223-column limit of the original protocol.
ENABLE_SEQUENCE = "\x1b[?1000h\x1b[?1006h"
DISABLE_SEQUENCE = "\x1b[?1006l\x1b[?1000l"

_REPORT = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)([Mm])")
_PARTIAL = re.compile(r"^\x1b(\[(<\d*(;\d*){0,2})?)?$")
# A report never exceeds "ESC [ < 3 ; 5 ; 5 M"-ish lengths; bail out well past it
# so a stray escape key cannot wedge the buffer.
_MAX_BUFFER = 24

BUTTON_LEFT = 0
BUTTON_WHEEL_UP = 64
BUTTON_WHEEL_DOWN = 65


class MouseEvent(NamedTuple):
    button: int
    column: int  # 1-based
    row: int  # 1-based
    pressed: bool

    @property
    def is_wheel_up(self) -> bool:
        return self.button == BUTTON_WHEEL_UP

    @property
    def is_wheel_down(self) -> bool:
        return self.button == BUTTON_WHEEL_DOWN

    @property
    def is_left_press(self) -> bool:
        return self.pressed and self.button == BUTTON_LEFT


class MouseSequenceParser:
    """Turn a keystroke stream into mouse events plus pass-through keys."""

    def __init__(self):
        self._pending = []

    def _text(self) -> str:
        return "".join(text for _key, text in self._pending)

    def feed(self, key):
        """Consume one keystroke.

        Returns ``(events, keys)``: mouse events decoded from the stream, and
        the keystrokes the caller should still handle as keyboard input. The
        original keystroke objects are preserved so named keys (``KEY_LEFT`` and
        friends) survive a pass-through.
        """
        text = str(key)
        if not self._pending:
            if text != "\x1b":
                return [], [key]
            self._pending.append((key, text))
            return [], []

        self._pending.append((key, text))
        buffered = self._text()

        match = _REPORT.match(buffered)
        if match:
            button, column, row, action = match.groups()
            self._pending = []
            event = MouseEvent(
                button=int(button),
                column=int(column),
                row=int(row),
                pressed=action == "M",
            )
            return [event], []

        if len(buffered) <= _MAX_BUFFER and _PARTIAL.match(buffered):
            return [], []

        keys = [key for key, _text in self._pending]
        self._pending = []
        return [], keys

    def flush(self):
        """Release anything held back, e.g. when the input stream goes idle."""
        keys = [key for key, _text in self._pending]
        self._pending = []
        return keys
