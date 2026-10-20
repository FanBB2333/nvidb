"""Queue/monitor visual consistency without any worker or remote connection."""

import pytest
from blessed import Terminal
from blessed.keyboard import Keystroke

from nvidb import config
from nvidb.mouse import MouseEvent
from nvidb.tui_theme import display_width
from test_sched_tui import _job, _mouse_at, _plain, _snapshot, _state, _tui


@pytest.fixture
def queue_view(monkeypatch):
    # Freeze relative timestamps so pointer-only paints can be compared exactly.
    monkeypatch.setattr("nvidb.sched.tui.age_seconds", lambda value: 125)
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("LINES", "32")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    def create(width=120, height=32):
        tui = _tui(width=width, height=height)
        tui.term = Terminal(kind="xterm-256color", force_styling=True)
        return tui

    return create


@pytest.mark.parametrize("width,height", [(120, 32), (80, 24), (72, 14)])
def test_queue_matches_monitor_continuous_rounded_frame(queue_view, width, height):
    tui = queue_view(width, height)
    lines = _plain(tui.render(_state())).splitlines()

    assert "nvidb queue" in lines[0]
    assert f"v{config.VERSION}" in lines[0]
    assert lines[1].startswith("╭") and lines[1].endswith("╮")
    assert lines[-1].startswith("╰") and lines[-1].endswith("╯")
    assert len(lines) <= height - 1  # Never trigger terminal scrolling.
    assert all(display_width(line) == width for line in lines[1:])
    assert all(line[0] in "│├" and line[-1] in "│┤" for line in lines[2:-1])
    assert any(line.startswith("├") and "JOBS" in line for line in lines)
    assert "q quit" in lines[-2] or "· q " in lines[-2]


@pytest.mark.parametrize("width,height", [(71, 24), (60, 24), (36, 14), (120, 13), (40, 10)])
def test_compact_queue_drops_outer_frame_and_stays_within_terminal(queue_view, width, height):
    tui = queue_view(width, height)
    lines = _plain(tui.render(_state())).splitlines()

    assert not any(line.startswith(("╭", "╰", "│")) for line in lines)
    assert len(lines) <= height - 1
    assert all(display_width(line) <= width for line in lines)
    assert any("JOBS" in line for line in lines)
    assert any(target == ("menu", "actions") for *_, target in tui._click_regions)


def test_framed_controls_activate_at_their_visible_positions(queue_view):
    tui = queue_view()
    state = _state()
    output = _plain(tui.render(state))

    assert tui.handle_mouse(_mouse_at(output, "[Waiting"))
    assert tui.filter == "pending"
    output = _plain(tui.render(state))
    assert len(tui.jobs) == 1 and tui.selected_job()["id"] == 2

    assert tui.handle_mouse(_mouse_at(output, "STATUS"))
    assert tui.sort_key == "state"
    output = _plain(tui.render(state))
    assert tui.handle_mouse(_mouse_at(output, "[Enter details]"))
    output = _plain(tui.render(state))
    assert tui.show_detail and "JOB 2 DETAIL" in output
    assert tui.worker.actions.empty()


def test_framed_unicode_node_and_job_rows_keep_correct_click_targets(queue_view):
    tui = queue_view()
    snapshot = _snapshot(jobs=[
        _job(1, name="训练任务", node="服务器乙"),
        _job(2, name="评估作业", node="服务器乙"),
    ])
    snapshot["nodes"][1]["name"] = "服务器乙"
    state = _state(snapshot)
    output = _plain(tui.render(state))

    assert all(display_width(line) <= 120 for line in output.splitlines())
    assert tui.handle_mouse(_mouse_at(output, "评估作业"))
    output = _plain(tui.render(state))
    assert tui.selected_job()["id"] == 2 and tui.focus == "jobs"
    assert tui.handle_mouse(_mouse_at(output, "服务器乙"))
    output = _plain(tui.render(state))
    assert tui.job_scope_node == "服务器乙" and tui.focus == "jobs"
    assert tui.selected_job()["id"] == 1  # Applying a server filter starts its list.
    assert all(display_width(line) <= 120 for line in output.splitlines())
    assert tui.worker.actions.empty()


@pytest.mark.parametrize("column", [1, 2, 119, 120])
def test_frame_border_and_padding_do_not_hover_or_activate_rows(queue_view, column):
    tui = queue_view()
    state = _state(_snapshot(jobs=[_job(1), _job(2)]))
    original = tui._frame_lines(state)
    row = next(row for row, target in tui._row_targets.items() if target == ("job", 1))

    assert tui.handle_mouse(MouseEvent(35, column, row + 1, True))
    assert tui._paint_hover() == original
    assert tui._hover.region is None
    assert tui.handle_mouse(MouseEvent(0, column, row + 1, True))
    assert tui.selected_job()["id"] == 1 and not tui.show_detail
    assert tui.focus == "jobs" and tui.worker.actions.empty()


def test_framed_hover_repaints_only_inner_row_without_rebuilding_data(queue_view, monkeypatch):
    tui = queue_view()
    state = _state(_snapshot(jobs=[_job(1), _job(2)]))
    original = tui._frame_lines(state)
    row = next(row for row, target in tui._row_targets.items() if target == ("job", 1))
    selected = tui.selected_job()["id"]

    def unexpected_refresh(*args, **kwargs):
        pytest.fail("Pointer feedback must reuse the cached base frame")

    monkeypatch.setattr(tui.worker, "read_state", unexpected_refresh)
    monkeypatch.setattr(tui, "_overview_job_lines", unexpected_refresh)
    assert tui.handle_mouse(MouseEvent(35, 4, row + 1, True))
    hovered = tui._paint_hover()
    assert [index for index, pair in enumerate(zip(original, hovered)) if pair[0] != pair[1]] == [row]
    assert [_plain(line) for line in hovered] == [_plain(line) for line in original]
    assert tui._hover.region[1:3] == (2, 117)
    assert hovered[row].index("\x1b[7m") > hovered[row].index("│")
    assert tui._base_frame == original
    assert tui.selected_job()["id"] == selected and tui.focus == "jobs"
    assert tui.worker.actions.empty()

    assert tui.handle_mouse(MouseEvent(35, 1, row + 1, True))
    assert tui._paint_hover() == original


def test_framed_mouse_and_keyboard_navigation_keep_job_focus(queue_view):
    tui = queue_view(80, 24)
    state = _state(_snapshot(jobs=[_job(1), _job(2), _job(3)]))
    output = _plain(tui.render(state))
    assert tui.handle_mouse(_mouse_at(output, "#2 train"))
    tui.render(state)
    assert tui.selected_job()["id"] == 2

    # Pointer feedback over a server must not redirect the keyboard cursor.
    output = _plain(tui.render(state))
    assert tui.handle_mouse(_mouse_at(output, "big-node", button=35))
    for text, name, expected in [
        ("\x1b[B", "KEY_DOWN", 3), ("k", None, 2),
        ("\x1b[A", "KEY_UP", 1), ("j", None, 2),
    ]:
        assert tui.handle_key(Keystroke(text, name=name))
        tui.render(state)
        assert tui.selected_job()["id"] == expected and tui.focus == "jobs"
    assert tui.handle_key(Keystroke("\r", name="KEY_ENTER"))
    tui.render(state)
    assert tui.show_detail
    assert tui.handle_key(Keystroke("\t", name="KEY_TAB"))
    tui.render(state)
    assert tui.resource_view == "servers" and tui.focus == "nodes"
    assert tui.handle_key(Keystroke("\t", name="KEY_TAB"))
    tui.render(state)
    assert tui.focus == "jobs" and tui.selected_job()["id"] == 2
    assert tui.worker.actions.empty()
    assert tui.handle_key(Keystroke("q")) is False


def test_modal_menu_uses_its_own_frame_and_unshifted_click_targets(queue_view):
    tui = queue_view()
    state = _state()
    output = _plain(tui.render(state))
    assert tui.handle_mouse(_mouse_at(output, "[View:"))
    menu = tui._menu
    output = _plain(tui.render(state))

    assert output.splitlines()[0].startswith("╭")
    assert sum(line.startswith("╭") for line in output.splitlines()) == 1
    assert "nvidb queue" not in output
    assert menu is not None
    assert tui.handle_mouse(_mouse_at(output, "Server capacity"))
    tui.render(state)
    assert tui._menu is None and tui.resource_view == "servers"
    assert tui.worker.actions.empty()


@pytest.mark.parametrize("theme,accent", [("classic", "\x1b[36m"), ("muted", "\x1b[34m")])
def test_queue_table_uses_monitor_ansi_palette_and_selection(queue_view, theme, accent):
    tui = queue_view()
    tui.theme = theme
    lines = tui._frame_lines(_state())
    header = next(line for line in lines if "PROGRESS / WAIT" in line)
    selected = next(row for row, target in tui._row_targets.items() if target == ("job", 0))

    assert "\x1b[1m" in header and accent in header
    assert "\x1b[100m" in lines[selected]
    assert "\x1b[48;2;" not in lines[selected]
    assert "\x1b[38;2;" not in lines[selected]


def test_resize_removes_frame_and_rebuilds_visible_hitboxes(queue_view, monkeypatch):
    tui = queue_view()
    state = _state(_snapshot(jobs=[_job(1), _job(2, name="评估作业")]))
    output = _plain(tui.render(state))
    assert tui.handle_mouse(_mouse_at(output, "评估作业", button=35))
    assert tui._paint_hover() != tui._base_frame

    monkeypatch.setenv("COLUMNS", "60")
    assert tui._paint_hover() is None
    output = _plain(tui.render(state))
    assert not any(line.startswith("╭") for line in output.splitlines())
    assert all(display_width(line) <= 60 for line in output.splitlines())
    assert tui.handle_mouse(_mouse_at(output, "评估作业"))
    tui.render(state)
    assert tui.selected_job()["id"] == 2 and tui.focus == "jobs"
