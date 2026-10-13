"""Width-aware drawing for the monitor's process detail panel."""
from .tui_theme import display_width, fit_display


class ProcessPanel:
    """Draw panel rows without accessing clients or changing selection state."""

    def __init__(self, width, *, focused, colorize):
        self.width = width
        self.inner_width = max(1, width - 4)
        self.focused = focused
        self.colorize = colorize
        self.horizontal = "─" if focused else "╌"
        self.vertical = "│" if focused else "┊"
        self.label_color = "cyan" if focused else "dark_grey"

    @staticmethod
    def trim(text, limit):
        return fit_display(str(text), limit, ellipsis="" if limit <= 2 else "..")

    def heading(self, title):
        label = self.trim(f"{self.horizontal} {title} ", max(0, self.width - 2))
        tail = self.horizontal * max(0, self.width - 2 - display_width(label)) + "╮"
        return (
            self.colorize("╭", "dark_grey")
            + self.colorize(label, self.label_color, attrs=["bold"] if self.focused else None)
            + self.colorize(tail, "dark_grey")
        )

    def rule(self, label, *, bottom=False):
        left, right = ("╰", "╯") if bottom else ("├", "┤")
        label = self.trim(f"{self.horizontal} {label} ", max(0, self.width - 2)) if label else ""
        plain = left + label + self.horizontal * max(0, self.width - 2 - display_width(label)) + right
        return self.colorize(plain, "dark_grey")

    def line(self, plain, styled=None):
        original = str(plain)
        plain = self.trim(original, self.inner_width)
        if styled is None or plain != original:
            styled = plain
        padding = " " * max(0, self.inner_width - display_width(plain))
        return (
            self.colorize(f"{self.vertical} ", "dark_grey")
            + styled + padding
            + self.colorize(f" {self.vertical}", "dark_grey")
        )

    def fields(self, label, fields):
        """Pack colored detail fields, indenting continuation rows."""
        prefix = f"{label:<9}"
        continuation = " " * display_width(prefix)
        packed = []
        current = []
        current_length = display_width(prefix)
        for text, color in fields:
            separator_length = 3 if current else 0
            if current and current_length + separator_length + display_width(text) > self.inner_width:
                packed.append((prefix, current))
                prefix = continuation
                current = []
                current_length = display_width(prefix)
                separator_length = 0
            current.append((text, color))
            current_length += separator_length + display_width(text)
        packed.append((prefix, current))

        rows = []
        for row_prefix, row_fields in packed:
            plain = row_prefix + " | ".join(text for text, _ in row_fields)
            styled = self.colorize(
                row_prefix, self.label_color,
                attrs=["bold"] if row_prefix.strip() else None,
            ) + " | ".join(self.colorize(text, color) if color else text for text, color in row_fields)
            rows.append((plain, styled))
        return rows
