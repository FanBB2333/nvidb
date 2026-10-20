"""Geometry and input isolation for the shared terminal controls."""
import pytest

from nvidb.mouse import MouseEvent
from nvidb.tui_controls import ActionMenu, action_bar
from nvidb.tui_theme import display_width


def plain(text, style):
    return text


@pytest.mark.parametrize("width", [12, 36, 80, 120])
@pytest.mark.parametrize("rows", [1, 2])
def test_overflow_keeps_every_action_reachable_and_whole(width, rows):
    controls = [(f"选项 {index}", "select", index, "cyan") for index in range(15)]
    lines, regions, hidden = action_bar(controls, width, plain, max_rows=rows)
    assert len(lines) <= rows
    assert all(display_width(line) <= width for line in lines)
    targets = [target for _, _, _, target in regions]
    assert (("menu", "actions") in targets) == bool(hidden)
    visible = {value for action, value in targets if action == "select"}
    assert visible.isdisjoint({value for _, _, value, _ in hidden})
    assert visible | {value for _, _, value, _ in hidden} == set(range(15))
    for row, start, end, _ in regions:
        assert 0 <= start <= end < display_width(lines[row])
    assert all(line.endswith("]") for line in lines)


def test_menu_pages_resizes_and_never_passes_input_to_underlying_view():
    menu = ActionMenu("Actions", [(f"Action {i}", ("do", i)) for i in range(20)])
    menu.render(36, 9, plain)
    assert menu.handle_key("K") == (False, None)  # no process/queue action
    menu.handle_mouse(MouseEvent(65, 1, 2, True))
    assert menu.selected == 1
    menu.handle_key("", "KEY_PGDOWN")
    assert menu.selected == 6
    lines = menu.render(20, 7, plain)
    assert len(lines) <= 7
    assert all(display_width(line) <= 20 for line in lines)
    assert "Action 6" in lines[1]
    assert menu.handle_mouse(MouseEvent(0, 2, 2, False)) == (False, None)
    assert menu.handle_mouse(MouseEvent(0, 2, 2, True)) == (True, ("do", 6))
    assert menu.handle_mouse(MouseEvent(0, 30, 20, True)) == (True, None)
    assert menu.handle_key("\x1b") == (True, None)


def test_menu_page_buttons_can_reach_last_action_using_only_mouse():
    menu = ActionMenu("More", [(f"Action {i}", ("do", i)) for i in range(19)])
    for _ in range(10):
        menu.render(36, 9, plain)
        next_page = next((r for r in menu.regions if r[3] == ("page", 1)), None)
        if next_page is None:
            break
        row, start, _, _ = next_page
        assert menu.handle_mouse(MouseEvent(0, start + 1, row + 1, True)) == (False, None)
    row, start, _, _ = next(r for r in menu.regions if r[3] == ("choose", 18))
    assert menu.handle_mouse(MouseEvent(0, start + 1, row + 1, True)) == (True, ("do", 18))


def test_hover_preserves_wide_text_and_ansi_styles_without_touching_neighbours():
    from nvidb.tui_controls import HoverState

    hover = HoverState()
    line = "前 " + "\x1b[31m[危险\x1b[0m操作]" + " 后"
    regions = [(0, 3, 12, ("cancel", 7))]
    baseline = [line, "unchanged"]
    assert hover.move(MouseEvent(35, 6, 1, True), regions, {}, 30)
    painted = hover.paint(baseline, regions, {}, 30)
    assert painted[0] != line
    assert "\x1b[0m\x1b[7m" in painted[0]  # reset inside the button retains hover
    assert painted[0].startswith("前 \x1b[31m")
    assert painted[1] == baseline[1]
    assert baseline[0] == line  # overlay never mutates the next frame's source
    import re
    strip = lambda text: re.sub(r"\x1b\[[0-9;]*m", "", text)
    assert strip(painted[0]) == strip(line)
    assert display_width(strip(painted[0])) == display_width(strip(line))
    assert not hover.move(MouseEvent(35, 8, 1, True), regions, {}, 30)
    assert hover.move(MouseEvent(35, 25, 1, True), regions, {}, 30)
    assert hover.paint(baseline, regions, {}, 30) == baseline


def test_hover_gives_buttons_priority_and_rechecks_geometry_on_resize():
    from nvidb.tui_controls import HoverState

    hover = HoverState()
    regions = [(1, 3, 8, ("signal", "TERM"))]
    rows = {1: ("process", 0), 2: ("pane", "jobs")}
    lines = ["top", "xx [TERM] rest", "blank pane"]
    hover.move(MouseEvent(35, 5, 2, True), regions, rows, 20, row_kinds={"process"})
    painted = hover.paint(lines, regions, rows, 20, row_kinds={"process"})
    assert painted[1].startswith("xx \x1b[7m[TERM]")
    assert hover.paint(lines, [], {}, 12) == lines  # control disappeared
    hover.move(MouseEvent(35, 5, 3, True), [], rows, 20, row_kinds={"process"})
    assert hover.paint(lines, [], rows, 20, row_kinds={"process"}) == lines
    hover.move(MouseEvent(35, 5, 2, True), regions, rows, 20)
    assert hover.paint(lines, regions, rows, 20, enabled=False) == lines
    assert hover.clear()
    assert hover.position is None


def test_menu_hover_highlights_choices_and_close_without_selecting_or_activating():
    menu = ActionMenu("Filter", [("All", ("filter", "all")), ("Busy", ("filter", "busy"))])
    baseline = menu.render(40, 12, plain)
    assert menu.handle_mouse(MouseEvent(35, 4, 3, True)) == (False, None)
    painted = menu.render(40, 12, plain)
    assert menu.selected == 0
    assert "\x1b[7m" in painted[2]
    assert painted[1] == baseline[1]
    assert menu.handle_key("", "KEY_DOWN") == (False, None)
    assert menu.hover.position is None
    assert menu.selected == 1
    row, start, _, _ = next(r for r in menu.regions if r[3][0] == "close")
    assert menu.handle_mouse(MouseEvent(35, start + 1, row + 1, True)) == (False, None)
    assert "\x1b[7m" in menu.render(40, 12, plain)[row]
    assert "\x1b[7m" not in menu.render(40, 12, plain, hover_enabled=False)[row]
    menu.handle_mouse(MouseEvent(35, 40, 12, True))
    assert all("\x1b[7m" not in line for line in menu.render(40, 12, plain))
