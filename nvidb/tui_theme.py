"""Shared design tokens and drawing primitives for the terminal UIs.

The look is the one tokscale made popular: chrome recedes and content
floats. Borders are rounded hairlines in a grey close to the terminal
background rather than bright ASCII art, interior areas stay borderless
and are separated by dim rules and whitespace, and frames are painted
line-by-line so a refresh never flickers.

Everything here is width-aware (a Chinese note still cannot push a line
past the right edge) and returns plain text: styling stays with each
caller, because the queue TUI colours through blessed while the monitor
TUI colours through termcolor.
"""
from __future__ import annotations

import sys
import unicodedata
from typing import Callable, List, Optional, Sequence, Tuple

MUTED = "bright_black"

# Rounded frame corners; the frame stays continuous through in-frame
# separators via the T junctions.
CORNER_TL, CORNER_TR, CORNER_BL, CORNER_BR = "╭", "╮", "╰", "╯"
TEE_L, TEE_R = "├", "┤"
HLINE = "─"

# Smooth bar glyphs: one continuous line whose filled part is the heavy
# rule, so a load reads as a line, not a stack of blocks.
BAR_FILL = "━"
BAR_EMPTY = "─"


def display_width(text: Optional[str]) -> int:
    """Terminal columns a string occupies, counting wide characters."""
    if not text:
        return 0
    width = 0
    for char in str(text):
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def fit(text: str, width: int) -> str:
    """Truncate to terminal columns, marking the cut with an ellipsis."""
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    kept = []
    used = 0
    for char in text:
        char_width = display_width(char)
        if used + char_width > width - 1:
            break
        kept.append(char)
        used += char_width
    return "".join(kept) + "…"


def frame_top(width: int, *, title: str = "") -> str:
    """`╭─ title ─────╮`, the top edge of a rounded panel."""
    inner = max(0, width - 2)
    left = f" {title} " if title else ""
    room = inner - display_width(left)
    if room < 0:
        left = fit(left, inner).rstrip() or " "
        room = inner - display_width(left)
    return CORNER_TL + left + HLINE * max(0, room) + CORNER_TR


def frame_bottom(width: int) -> str:
    return CORNER_BL + HLINE * max(0, width - 2) + CORNER_BR


def frame_separator(width: int) -> str:
    """`├──────────┤`: keeps the frame continuous under a header row."""
    return TEE_L + HLINE * max(0, width - 2) + TEE_R


def smooth_bar(
    width: int,
    fractions: Sequence[Tuple[float, str]],
    *,
    empty: str = BAR_EMPTY,
) -> List[Tuple[str, str]]:
    """A bar as one smooth line, split into named-style segments.

    `fractions` is `(share_of_bar, style)` pairs in draw order - a GPU
    memory bar passes foreign processes and queue reservations, and the
    untouched remainder of the line stays the dim empty rule. Shares are
    absolute (0.25 means a quarter of the bar), not relative to each
    other, and rounding is clamped so partial cells never split a glyph
    or overrun the width.
    """
    width = max(0, width)
    cells = [0] * len(fractions)
    budget = width
    for index, (fraction, _style) in enumerate(fractions):
        share = max(0.0, min(1.0, fraction))
        cells[index] = min(budget, int(round(width * share)))
        budget -= cells[index]
    segments: List[Tuple[str, str]] = []
    for (fraction, style), count in zip(fractions, cells):
        if count > 0:
            segments.append((BAR_FILL * count, style))
    used = sum(cells)
    if width - used > 0:
        segments.append((empty * (width - used), MUTED))
    return segments


class DiffScreen:
    """Paints only the rows that changed since the previous frame.

    Clearing and rewriting the whole screen every refresh is what makes
    a TUI flicker, especially over SSH. Ratatui earns its "zero flicker"
    reputation by diffing frames cell by cell; diffing ours line by line
    gets the same visible result with blessed alone.
    """

    def __init__(self, term, writer: Optional[Callable[[str], None]] = None):
        self._term = term
        self._write = writer or sys.stdout.write
        self._previous: List[str] = []
        self._previous_width = 0

    def paint(self, lines: Sequence[str]) -> None:
        term = self._term
        width = term.width or 80
        # The bottom row stays unwritten so painting can never scroll
        # the screen and shear the frame.
        height = max(0, (term.height or 24) - 1)
        lines = list(lines[:height])

        out = []
        force = width != self._previous_width
        if force:
            out.append(term.home + term.clear)
        for row, line in enumerate(lines):
            if force or row >= len(self._previous) or line != self._previous[row]:
                out.append(term.move_xy(0, row) + term.clear_eol + line)
        if len(lines) < len(self._previous):
            out.append(term.move_xy(0, len(lines)) + term.clear_eos)
        self._previous = lines
        self._previous_width = width
        if out:
            self._write("".join(out))
            sys.stdout.flush()

    def reset(self) -> None:
        self._previous = []
        self._previous_width = 0
