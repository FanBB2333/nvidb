"""Mouse and keyboard controls shared by the Blessed terminal views."""
from __future__ import annotations

import re

from .tui_theme import display_width, fit_display, frame_bottom, frame_top, pad_display


# CSI, OSC, and charset escapes occupy no terminal columns. Keeping them intact
# lets hover coexist with coloured metrics, selection bands, and wide glyphs.
_TERMINAL_TOKEN = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x1b\x07]*(?:\x07|\x1b\\)|[()][0-2A-Z0-9]|[@-_])|.",
    re.DOTALL,
)
_HOVER_ON = "\x1b[7m"
_RESET = "\x1b[0m"

# Blessed names the paging keys differently across terminfo entries.
PAGE_DOWN_KEYS = ("KEY_PGDOWN", "KEY_NPAGE", "KEY_PAGEDOWN")
PAGE_UP_KEYS = ("KEY_PGUP", "KEY_PPAGE", "KEY_PAGEUP")


def _highlight_span(line, start, end):
    """Inverse video over an inclusive cell span, preserving existing styles."""
    parts, styles = [], []
    column = 0
    active = False
    for token in _TERMINAL_TOKEN.findall(line):
        if token.startswith("\x1b"):
            parts.append(token)
            if token.startswith("\x1b[") and token.endswith("m"):
                styles.append(token)
                if active:
                    parts.append(_HOVER_ON)
            continue
        cells = display_width(token)
        inside = (column <= end and column + cells > start) or (not cells and active)
        if active and not inside:
            parts.append(_RESET + "".join(styles))
            active = False
        if inside and not active:
            parts.append(_HOVER_ON)
            active = True
        parts.append(token)
        column += cells
    if active:
        parts.append(_RESET + "".join(styles))
    return "".join(parts)


class HoverState:
    """Pointer feedback independent of keyboard focus and selected objects."""

    def __init__(self):
        self.position = None
        self.region = None

    def _hit(self, regions, row_targets, width, row_kinds):
        if self.position is None:
            return None
        row, column = self.position
        if row < 0 or not 0 <= column < width:
            return None
        for region_row, start, end, target in regions:
            if region_row == row and start <= column <= end:
                return row, start, min(end, width - 1), target
        target = row_targets.get(row)
        if target is not None and (row_kinds is None or target[0] in row_kinds):
            return row, 0, width - 1, target
        return None

    def move(self, event, regions, row_targets, width, *, row_kinds=None):
        self.position = (event.row - 1, event.column - 1)
        region = self._hit(regions, row_targets, width, row_kinds)
        changed = region != self.region
        self.region = region
        return changed

    def clear(self):
        changed = self.region is not None
        self.position = self.region = None
        return changed

    def paint(self, lines, regions, row_targets, width, *, enabled=True, row_kinds=None):
        # Resolve again after refresh/resize: hit boxes can move while the
        # pointer stays still. Never cache decorated rows across frames.
        self.region = self._hit(regions, row_targets, width, row_kinds)
        if not enabled or self.region is None:
            return lines
        row, start, end, _ = self.region
        if not 0 <= row < len(lines):
            self.region = None
            return lines
        lines = list(lines)
        lines[row] = _highlight_span(lines[row], start, end)
        return lines


def action_bar(controls, width, style, *, max_rows=1):
    """Lay out whole buttons, keeping overflow reachable through More.

    Controls are (label, action, value, style) tuples. Regions use inclusive,
    zero-based terminal coordinates, measured before adding ANSI styles.
    """
    width = max(1, width)
    max_rows = max(1, max_rows)

    def pack(items):
        rows = [[]]
        used = 0
        for control in items:
            size = display_width(control[0]) + 2
            if size > width:
                return None
            if used and used + 1 + size > width:
                rows.append([])
                used = 0
            if len(rows) > max_rows:
                return None
            rows[-1].append(control)
            used += (1 if used else 0) + size
        return rows

    controls = list(controls)
    rows = pack(controls)
    hidden = []
    if rows is None:
        more = ("More…" if width >= 7 else "…", "menu", "actions", "cyan")
        for count in range(len(controls) - 1, -1, -1):
            rows = pack(controls[:count] + [more])
            if rows is not None:
                hidden = controls[count:]
                break
        else:
            return [fit_display("…", width)], [], controls

    lines, regions = [], []
    for row, items in enumerate(rows):
        parts, column = [], 0
        for label, action, value, color in items:
            if parts:
                parts.append(" ")
                column += 1
            token = f"[{label}]"
            size = display_width(token)
            parts.append(style(token, color))
            regions.append((row, column, column + size - 1, (action, value)))
            column += size
        lines.append("".join(parts))
    return lines, regions, hidden


class ActionMenu:
    """A modal, scrollable choice list; all input stays in the open menu."""

    def __init__(self, title, entries, *, selected=0):
        self.title = title
        self.entries = list(entries)  # (label, (action, value))
        self.selected = max(0, min(selected, len(self.entries) - 1))
        self.page_size = 1
        self.regions = []
        self.hover = HoverState()

    def render(self, width, height, style, *, hover_enabled=True):
        width, height = max(1, width), max(1, height)
        self.page_size = max(1, height - 4)
        start = self.selected // self.page_size * self.page_size
        end = min(len(self.entries), start + self.page_size)
        self.regions = []
        lines = [style(fit_display(frame_top(width, title=self.title), width), "cyan")]
        inner = max(1, width - 4)
        for index in range(start, end):
            label = ("› " if index == self.selected else "  ") + self.entries[index][0]
            line = "│ " + pad_display(fit_display(label, inner), inner) + " │"
            lines.append(style(fit_display(line, width), "cyan" if index == self.selected else None))
            self.regions.append((len(lines) - 1, 0, width - 1, ("choose", index)))
        lines.append(style(fit_display(frame_bottom(width), width), "bright_black"))
        controls = [("Close", "close", None, "cyan")]
        if start:
            controls.append(("↑ Prev", "page", -1, "cyan"))
        if end < len(self.entries):
            controls.append(("↓ Next", "page", 1, "cyan"))
        column, parts = 0, []
        for label, action, value, color in controls:
            token = f"[{label}]"
            size = display_width(token)
            if column + size > width:
                continue
            self.regions.append((len(lines), column, column + size - 1, (action, value)))
            parts.append(style(token, color) + " ")
            column += size + 1
        lines.append("".join(parts).rstrip())
        lines.append(style(fit_display(
            f"{start + 1 if self.entries else 0}-{end}/{len(self.entries)} · Click to choose · ↑/↓ Enter · Esc back", width
        ), "bright_black"))
        self.regions = [region for region in self.regions if region[0] < height]
        return self.hover.paint(lines[:height], self.regions, {}, width, enabled=hover_enabled)

    def move(self, delta):
        self.hover.clear()
        self.selected = max(0, min(self.selected + delta, len(self.entries) - 1))

    def handle_key(self, text, name=""):
        """Return (closed, chosen action); unknown keys are consumed."""
        self.hover.clear()
        if text in ("q", "?", "\x1b") or name == "KEY_ESCAPE":
            return True, None
        if text in ("\r", "\n", " ") or name == "KEY_ENTER":
            return True, self.entries[self.selected][1] if self.entries else None
        if text == "j" or name == "KEY_DOWN":
            self.move(1)
        elif text == "k" or name == "KEY_UP":
            self.move(-1)
        elif name in PAGE_DOWN_KEYS:
            self.move(self.page_size)
        elif name in PAGE_UP_KEYS:
            self.move(-self.page_size)
        elif text == "g" or name == "KEY_HOME":
            self.selected = 0
        elif text == "G" or name == "KEY_END":
            self.selected = max(0, len(self.entries) - 1)
        return False, None

    def handle_mouse(self, event):
        if event.is_motion:
            width = max((end + 1 for _, _, end, _ in self.regions), default=0)
            self.hover.move(event, self.regions, {}, width)
            return False, None
        if event.is_wheel_up or event.is_wheel_down:
            self.move(-1 if event.is_wheel_up else 1)
        elif event.is_left_press:
            for row, start, end, (action, value) in self.regions:
                if row == event.row - 1 and start <= event.column - 1 <= end:
                    if action == "choose":
                        return True, self.entries[value][1]
                    if action == "page":
                        self.move(value * self.page_size)
                        return False, None
                    return True, None
            # Dismiss without forwarding the click to anything underneath.
            return True, None
        return False, None
