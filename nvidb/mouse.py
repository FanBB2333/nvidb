"""SGR mouse reporting helpers for the nvidb TUI.

Depending on the Blessed version, an SGR mouse report
(``ESC [ < button ; column ; row M|m``) arrives from ``inkey()`` as a complete
keystroke or in pieces, beginning with ``KEY_ESCAPE`` or ``CSI``. The parser
accepts both forms and hands other keystrokes back to the caller untouched.
Reporting setup also keeps newer Blessed decoders aware of the enabled modes,
so complete mouse reports do not wait for Escape-key disambiguation.
"""

from __future__ import annotations

import re
import sys
from typing import NamedTuple

# 1000: report button press/release. 1003: report all pointer movement, including
# movement with no button held. 1006: SGR coordinates, which lift the 223-column
# limit of the original protocol. Restore every enabled mode on exit.
ENABLE_SEQUENCE = "\x1b[?1000h\x1b[?1003h\x1b[?1006h"
DISABLE_SEQUENCE = "\x1b[?1006l\x1b[?1003l\x1b[?1000l"

_REPORT = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)([Mm])")
_PARTIAL = re.compile(r"^\x1b(\[(<\d*(;\d*){0,2})?)?$")
# A report never exceeds "ESC [ < 3 ; 5 ; 5 M"-ish lengths; bail out well past it
# so a stray escape key cannot wedge the buffer.
_MAX_BUFFER = 24

BUTTON_LEFT = 0
MOTION_FLAG = 32
BUTTON_WHEEL_UP = 64
BUTTON_WHEEL_DOWN = 65


def set_mouse_reporting(term, enabled: bool) -> None:
    """Write mouse modes and keep Blessed's input decoder in sync."""
    sys.stdout.write(ENABLE_SEQUENCE if enabled else DISABLE_SEQUENCE)
    sys.stdout.flush()

    # Recent Blessed versions only decode SGR mouse reports when their DEC
    # mode cache says 1006 is enabled. Without this, even a complete report
    # waits through the normal Escape-key delay (350 ms) before being split.
    # We already know the modes we just wrote; querying them would add another
    # terminal round trip and does not work in every terminal. Older Blessed
    # versions have no cache and continue using our fragmented-report parser.
    cache = getattr(term, "_dec_mode_cache", None)
    if isinstance(cache, dict):
        for mode in (1000, 1003, 1006):
            cache[mode] = 1 if enabled else 2  # DEC SET / RESET states.


class MouseEvent(NamedTuple):
    button: int
    column: int  # 1-based
    row: int  # 1-based
    pressed: bool

    @property
    def is_motion(self) -> bool:
        return bool(self.button & MOTION_FLAG)

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
            # Whatever blessed made of the report's first bytes - a bare
            # escape, the whole `ESC [`, or a complete report - it must still
            # match the mouse protocol. Other resolved keys pass straight back.
            if not _PARTIAL.match(text) and not _REPORT.fullmatch(text):
                return [], [key]

        self._pending.append((key, text))
        buffered = self._text()

        match = _REPORT.fullmatch(buffered)
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
