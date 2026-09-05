"""Interactive terminal UI for the nvidb job queue.

The screen is three stacked panes - task flow or node capacity, the job table,
and a detail or log view for the selected job - drawn with `blessed`.

All database and SSH work happens on one worker thread that owns the scheduler;
the render thread only reads an immutable snapshot and posts actions onto a
queue. That keeps SQLite on a single thread and means a slow or unreachable
node can never freeze the interface.
"""
from __future__ import annotations

import queue as queue_module
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from blessed import Terminal

from .. import config as nvidb_config
from ..mouse import (
    DISABLE_SEQUENCE as MOUSE_DISABLE_SEQUENCE,
    ENABLE_SEQUENCE as MOUSE_ENABLE_SEQUENCE,
    MouseSequenceParser,
)
from ..tui_theme import (
    DiffScreen,
    display_width,
    fit_display,
    pad_display,
    smooth_bar,
    wrap_display,
)
from . import db as dbm
from .model import (
    GpuProcess,
    age_seconds,
    format_duration,
    format_mb,
)
from .scheduler import Scheduler

FILTERS = ("active", "all", "running", "pending", "finished")
# summary: only the processes the queue does not manage, which is what explains
# a card being full; all: those plus this queue's own jobs; off: neither.
PROC_VIEWS = ("summary", "all", "off")
PROC_SUMMARY_LIMIT = 2
# The flow view answers "what is every card doing, and what runs next?".  The
# existing server view remains available for capacity/process inspection.
RESOURCE_VIEWS = ("flow", "servers")
FILTER_STATES = {
    "active": ("pending", "running"),
    "running": ("running",),
    "pending": ("pending",),
    "finished": ("completed", "failed", "cancelled", "timeout", "lost"),
}

# Call sites use semantic ANSI names; a theme maps them to concrete tones and
# blessed degrades each to the nearest colour the terminal actually has.
# "classic" is the plain high-contrast palette the nvidb monitor TUI uses and
# is the default; "muted" keeps the softer tokscale tones this TUI shipped
# with, where saturated colour is reserved to mean "look at this".
THEMES = {
    "classic": {},
    "muted": {
        "green": "mediumseagreen",
        "yellow": "darkkhaki",
        "red": "indianred",
        "cyan": "cadetblue",
        "magenta": "rosybrown",
        "bright_blue": "steelblue",
    },
}
THEME_ORDER = tuple(THEMES)


@dataclass(frozen=True)
class LayoutProfile:
    """Space policy for a terminal size.

    The renderer still owns the exact row counts.  A profile only states how
    much of the body the resource overview may claim, which keeps the job
    table and its controls usable as the terminal shrinks.
    """

    name: str
    resource_fraction: float
    resource_limit: int
    footer_rows: int = 2

    @classmethod
    def for_terminal(cls, width: int, height: int) -> "LayoutProfile":
        if width < 72 or height < 24:
            return cls("compact", 0.30, 10)
        if width < 120 or height < 32:
            return cls("standard", 0.34, 14)
        return cls("wide", 0.50, height)


@dataclass(frozen=True)
class Command:
    """One keyboard command shared by dispatch, help, and the action bar."""

    name: str
    action: str
    text_keys: Tuple[str, ...] = ()
    key_names: Tuple[str, ...] = ()
    contexts: Tuple[str, ...] = ("jobs", "nodes")
    value: Any = None
    help_key: Optional[str] = None
    description: Optional[str] = None
    footer_order: Optional[int] = None

    def matches(self, text: str, key_name: str, context: str) -> bool:
        return (
            context in self.contexts
            and (text in self.text_keys or key_name in self.key_names)
        )


COMMANDS = (
    Command(
        "cursor_down",
        "move_cursor",
        ("j",),
        ("KEY_DOWN",),
        value=1,
        help_key="j / k / ↑ / ↓",
        description="Move the selection in the focused pane",
    ),
    Command("cursor_up", "move_cursor", ("k",), ("KEY_UP",), value=-1),
    Command(
        "gpu_next", "gpu_cursor", key_names=("KEY_RIGHT",),
        contexts=("nodes",), value=1, help_key="← / →",
        description="Select a GPU on this node; Enter scopes its jobs",
    ),
    Command(
        "gpu_previous", "gpu_cursor", key_names=("KEY_LEFT",),
        contexts=("nodes",), value=-1,
    ),
    Command(
        "page_down",
        "move_cursor",
        key_names=("KEY_PGDOWN", "KEY_NPAGE", "KEY_PAGEDOWN"),
        value=10,
        help_key="PgUp / PgDn",
        description="Move a page at a time",
    ),
    Command(
        "page_up",
        "move_cursor",
        key_names=("KEY_PGUP", "KEY_PPAGE", "KEY_PAGEUP"),
        value=-10,
    ),
    Command(
        "first",
        "move_edge",
        ("g",),
        value=-1,
        help_key="g / G",
        description="Select the first / last item",
    ),
    Command("last", "move_edge", ("G",), value=1),
    Command(
        "switch_pane",
        "switch_pane",
        key_names=("KEY_TAB",),
        help_key="Tab",
        description="Switch focus between nodes and jobs",
        footer_order=20,
    ),
    Command(
        "job_detail",
        "detail",
        ("\n", "\r"),
        ("KEY_ENTER", "KEY_RETURN"),
        ("jobs",),
        help_key="Enter",
        description="Show or hide job detail",
        footer_order=10,
    ),
    Command(
        "node_scope",
        "scope_node",
        ("\n", "\r"),
        ("KEY_ENTER", "KEY_RETURN"),
        ("nodes",),
        help_key="Enter",
        description="Show jobs running on the selected node",
    ),
    Command(
        "detail_previous",
        "scroll_detail",
        ("[",),
        contexts=("jobs",),
        value=-1,
        help_key="[ / ]",
        description="Page through detail or log text",
    ),
    Command(
        "detail_next",
        "scroll_detail",
        ("]",),
        contexts=("jobs",),
        value=1,
    ),
    Command(
        "log",
        "log",
        ("L",),
        contexts=("jobs",),
        help_key="L",
        description="Toggle the selected job's log tail",
        footer_order=30,
    ),
    Command(
        "cancel",
        "cancel",
        ("c",),
        contexts=("jobs",),
        help_key="c",
        description="Cancel the selected job (press twice)",
        footer_order=11,
    ),
    Command(
        "requeue",
        "requeue",
        ("r",),
        contexts=("jobs",),
        help_key="r",
        description="Re-queue the selected finished job",
        footer_order=11,
    ),
    Command(
        "priority_up",
        "priority",
        ("+", "="),
        contexts=("jobs",),
        value=1,
        help_key="+ / -",
        description="Raise or lower the selected job's priority",
        footer_order=31,
    ),
    Command(
        "priority_down",
        "priority",
        ("-",),
        contexts=("jobs",),
        value=-1,
        footer_order=32,
    ),
    Command(
        "queue_up",
        "move",
        text_keys=("K",),
        key_names=("KEY_SR",),
        contexts=("jobs",),
        value=-1,
        help_key="K / J",
        description="Move a pending job in dispatch order",
        footer_order=33,
    ),
    Command(
        "queue_down",
        "move",
        text_keys=("J",),
        key_names=("KEY_SF",),
        contexts=("jobs",),
        value=1,
        footer_order=34,
    ),
    Command(
        "filter",
        "filter",
        ("f",),
        help_key="f",
        description="Cycle the job filter",
        footer_order=40,
    ),
    Command(
        "sort",
        "sort",
        ("s",),
        help_key="s / S",
        description="Cycle the sort column / flip direction",
        footer_order=41,
    ),
    Command("sort_reverse", "reverse_sort", ("S",)),
    Command(
        "resource_view",
        "resource_view",
        ("v",),
        help_key="v",
        description="Switch task flow / server capacity view",
        footer_order=42,
    ),
    Command(
        "process_view",
        "procs",
        ("p",),
        help_key="p",
        description="Open capacity view; cycle process detail",
        footer_order=43,
    ),
    Command(
        "tick",
        "tick",
        ("t",),
        help_key="t",
        description="Force a scheduler tick now",
        footer_order=44,
    ),
    Command(
        "auto_tick",
        "auto",
        ("a",),
        help_key="a",
        description="Toggle automatic ticking",
        footer_order=45,
    ),
    Command(
        "theme",
        "theme",
        ("T",),
        help_key="T",
        description="Switch colour theme (classic / muted)",
        footer_order=46,
    ),
    Command(
        "node_toggle",
        "node_toggle",
        ("d",),
        contexts=("nodes",),
        help_key="d",
        description="Drain or resume the selected node",
        footer_order=11,
    ),
    Command(
        "scope_all",
        "scope_all",
        ("x",),
        help_key="x / Esc",
        description="Clear a server or GPU job scope",
        footer_order=12,
    ),
    Command("escape", "escape", key_names=("KEY_ESCAPE",)),
    Command(
        "acknowledge",
        "ack",
        ("A",),
        help_key="A",
        description="Acknowledge every open alert",
        footer_order=47,
    ),
    Command(
        "help",
        "help",
        ("?",),
        help_key="?",
        description="Show this context-sensitive help",
        footer_order=21,
    ),
    Command(
        "quit",
        "quit",
        ("q",),
        help_key="q",
        description="Quit",
        footer_order=22,
    ),
)

# The selection band: a grey clearly lighter than a dark terminal background,
# so the cursor is findable at a glance yet still reads as a tint, not a bar.
SELECTION_BG = "gray27"

STATE_STYLE = {
    "running": "green",
    "pending": "yellow",
    # Held jobs are pending in the database but waiting on a person, so they
    # get the colour of something that needs attention rather than of a queue.
    "held": "magenta",
    "completed": "cyan",
    "failed": "red",
    "timeout": "red",
    "lost": "red",
    "cancelled": "bright_black",
}

# A glyph plus a four-letter word reads at a glance and costs six columns
# where the full state name cost nine.
STATE_BADGE = {
    "running": "● run",
    "pending": "· pend",
    "held": "⏸ held",
    "completed": "✓ done",
    "failed": "✗ fail",
    "cancelled": "⊘ canc",
    "timeout": "! tout",
    "lost": "? lost",
}


def display_state(job: Dict[str, Any]) -> str:
    """What to call a job on screen. A held job is pending but stuck, and
    drawing it as plain `pending` is how someone waits for a job that will
    never start until they act."""
    return "held" if job.get("held") else job["state"]


# Runtime is the opening view: the TIME column reads naturally from the
# longest-running job at the top to the shortest at the bottom.  Queue order
# remains one keypress away and is restored automatically before a pending job
# is moved, because that is the order in which K/J make sense.
SORT_KEYS = (
    "time",
    "queue",
    "id",
    "state",
    "pri",
    "name",
    "node",
    "vram",
    "used",
)

# (title, width, sort key or None when the column is not worth sorting by).
JOB_COLUMNS = (
    ("ID", 4, "id"),
    ("ST", 6, "state"),
    ("PRI", 3, "pri"),
    ("NAME", 16, "name"),
    ("NODE", 14, "node"),
    ("GPU", 4, None),
    ("VRAM", 6, "vram"),
    ("USED", 6, "used"),
    ("TIME", 9, "time"),
    ("RC", 3, None),
)
RIGHT_ALIGNED_COLUMNS = frozenset({"ID", "PRI", "VRAM", "USED", "TIME", "RC"})

CONFIRM_SECONDS = 5.0


class _Worker(threading.Thread):
    """Owns the scheduler: ticks, refreshes the snapshot, runs user actions."""

    def __init__(self, db_path=None, refresh: float = 3.0):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.refresh = max(1.0, float(refresh))
        self.actions: "queue_module.Queue[Tuple]" = queue_module.Queue()
        self.state_lock = threading.Lock()
        self.snapshot: Optional[Dict[str, Any]] = None
        self.notice: Optional[Tuple[str, str]] = None
        self.log_request: Optional[Tuple[int, str]] = None
        self.log_text: str = ""
        self.log_job: Optional[int] = None
        self.busy = False
        self.auto_tick = True
        self.error: Optional[str] = None
        # Thread.join() calls Thread._stop() internally. Keep our event under a
        # distinct name so quitting the TUI cannot replace that method.
        self._stop_event = threading.Event()
        self._scheduler: Optional[Scheduler] = None

    # --- public API (render thread) --------------------------------------

    def post(self, *action) -> None:
        self.actions.put(action)

    def stop(self) -> None:
        self._stop_event.set()
        self.actions.put(("quit",))

    def set_notice(self, message: str, style: str = "yellow") -> None:
        with self.state_lock:
            self.notice = (message, style)

    def set_log_request(self, request: Optional[Tuple[int, str]]) -> None:
        with self.state_lock:
            if request != self.log_request:
                self.log_request = request
                self.log_text = ""
                self.log_job = None

    def read_state(self) -> Dict[str, Any]:
        with self.state_lock:
            return {
                "snapshot": self.snapshot,
                "notice": self.notice,
                "log_text": self.log_text,
                "log_job": self.log_job,
                "busy": self.busy,
                "auto_tick": self.auto_tick,
                "error": self.error,
            }

    # --- worker thread ----------------------------------------------------

    def run(self) -> None:
        try:
            conn = dbm.open_db(self.db_path)
        except Exception as error:
            with self.state_lock:
                self.error = f"cannot open queue database: {error}"
            return
        self._scheduler = Scheduler(conn)
        try:
            self._scheduler.sync_nodes_from_config()
        except Exception as error:
            with self.state_lock:
                self.error = f"cannot sync nodes from config: {error}"

        next_refresh = 0.0
        while not self._stop_event.is_set():
            try:
                action = self.actions.get(timeout=0.2)
            except queue_module.Empty:
                action = None

            if action is not None:
                if action[0] == "quit":
                    break
                self._run_action(action)
                next_refresh = 0.0

            if time.time() >= next_refresh:
                self._refresh()
                next_refresh = time.time() + self.refresh

        try:
            self._scheduler.close()
            conn.close()
        except Exception:
            pass

    def _set_busy(self, value: bool) -> None:
        with self.state_lock:
            self.busy = value

    def _refresh(self) -> None:
        scheduler = self._scheduler
        if scheduler is None:
            return
        self._set_busy(True)
        try:
            if self.auto_tick:
                scheduler.tick()
            snapshot = scheduler.snapshot()
            log_text, log_job = self._fetch_log(scheduler)
            with self.state_lock:
                self.snapshot = snapshot
                self.error = None
                if log_job is not None:
                    self.log_text = log_text
                    self.log_job = log_job
        except Exception as error:
            with self.state_lock:
                self.error = f"{type(error).__name__}: {error}"
        finally:
            self._set_busy(False)

    def _fetch_log(self, scheduler: Scheduler) -> Tuple[str, Optional[int]]:
        with self.state_lock:
            request = self.log_request
        if not request:
            return "", None
        job_id, stream = request
        try:
            return scheduler.job_logs(job_id, stream=stream, lines=200), job_id
        except Exception as error:
            return f"(log unavailable: {error})", job_id

    def _run_action(self, action: Tuple) -> None:
        scheduler = self._scheduler
        if scheduler is None:
            return
        name = action[0]
        self._set_busy(True)
        try:
            if name == "tick":
                summary = scheduler.tick(force=True)
                started = len(summary.get("dispatched") or [])
                done = len(summary.get("finished") or [])
                self.set_notice(
                    f"tick: {started} started, {done} finished, "
                    f"{summary.get('nodes_up', 0)} node(s) up",
                    "cyan",
                )
            elif name == "cancel":
                ok = scheduler.cancel(action[1])
                self.set_notice(
                    f"job {action[1]} cancelled" if ok else f"job {action[1]} is not cancellable",
                    "green" if ok else "yellow",
                )
            elif name == "requeue":
                ok = scheduler.requeue(action[1])
                if ok:
                    scheduler.tick(force=True)
                self.set_notice(
                    f"job {action[1]} requeued" if ok else f"job {action[1]} is not requeueable",
                    "green" if ok else "yellow",
                )
            elif name == "drain":
                scheduler.set_node_enabled(action[1], False)
                self.set_notice(f"{action[1]} drained", "yellow")
            elif name == "resume":
                scheduler.set_node_enabled(action[1], True)
                scheduler.tick(force=True)
                self.set_notice(f"{action[1]} resumed", "green")
            elif name == "ack":
                count = dbm.acknowledge_alerts(scheduler.conn, all_open=True)
                self.set_notice(
                    f"acknowledged {count} alert(s)" if count else "no open alerts",
                    "green" if count else "bright_black",
                )
            elif name == "auto":
                self.auto_tick = bool(action[1])
                self.set_notice(
                    f"auto tick {'on' if self.auto_tick else 'off'}", "cyan"
                )
            elif name == "priority":
                value = scheduler.adjust_priority(action[1], action[2])
                self.set_notice(
                    f"job {action[1]} priority → {value}"
                    if value is not None
                    else f"job {action[1]} is finished; priority no longer matters",
                    "cyan" if value is not None else "yellow",
                )
            elif name == "move":
                job = dbm.get_job(scheduler.conn, action[1])
                if job is None or job.state != "pending":
                    self.set_notice(f"job {action[1]} is no longer pending", "yellow")
                    return
                if job.lane:
                    result = scheduler.lane_move(job.id, delta=int(action[2]))
                    self.set_notice(
                        f"job {job.id} → {result['lane']} slot {result['position']}"
                        if result["moved"] else f"job {job.id} is already at that end of lane {job.lane}",
                        "cyan" if result["moved"] else "yellow",
                    )
                    return
                ok = scheduler.move_pending(job.id, action[2])
                self.set_notice(
                    f"job {action[1]} moved {'up' if action[2] < 0 else 'down'}"
                    if ok
                    else f"job {action[1]} is already at that end of the queue",
                    "cyan" if ok else "yellow",
                )
        except Exception as error:
            self.set_notice(f"{type(error).__name__}: {error}", "red")
        finally:
            self._set_busy(False)


class QueueTUI:
    def __init__(
        self,
        db_path=None,
        refresh: float = 3.0,
        *,
        mouse_enabled: bool = True,
        theme: str = "classic",
    ):
        self.term = Terminal()
        self.worker = _Worker(db_path=db_path, refresh=refresh)
        self.mouse_enabled = bool(mouse_enabled)
        self.theme = theme if theme in THEMES else "classic"
        self._mouse_reporting = False
        self.focus = "jobs"  # jobs | nodes
        self.job_index = 0
        self.node_index = 0
        self._resource_scroll: Optional[int] = None
        self._resource_start = 0
        self._resource_content_size = 0
        self._resource_page_size = 1
        self._cursor_gpu: Optional[int] = None
        self.filter = "active"
        self.sort_key = "time"
        self.sort_reverse = False
        # A node/GPU click scopes the lower table to work that is running on
        # that resource.  Keep this separate from node_index: the latter is a
        # keyboard cursor and must not silently hide jobs just because focus
        # moved into the server pane.
        self.job_scope_node: Optional[str] = None
        self.job_scope_gpu: Optional[int] = None
        self.job_scope_pool: Optional[str] = None
        self.resource_view = "flow"
        # The cursor follows a job, not a row number: a refresh or a priority
        # change may reorder the table under the selection.
        self._selected_job_id: Optional[int] = None
        self.proc_view = "summary"
        # Start with the table as the primary view.  Detail is one Enter away
        # and no longer consumes most of a short terminal before it is asked
        # for.
        self.show_detail = False
        self.show_log = False
        self.show_help = False
        self.help_offset = 0
        self._help_page_size = 1
        self._help_max_offset = 0
        self.detail_page = 0
        self.detail_pages = 1
        self._detail_job_id: Optional[int] = None
        self.log_offset = 0
        self._last_log_request: Optional[Tuple[int, str]] = None
        self._log_max_offset = 0
        self._log_page_height = 1
        self.pending_confirm: Optional[Tuple[str, int, float]] = None
        self.jobs: List[Dict[str, Any]] = []
        self.nodes: List[Dict[str, Any]] = []
        self._snapshot: Optional[Dict[str, Any]] = None
        # Mouse coordinates are stored as zero-based screen rows/columns. A
        # region wins over a whole-row target so buttons embedded in the footer
        # remain clickable without changing how wheel routing works.
        self._click_regions: List[Tuple[int, int, int, Tuple[str, Any]]] = []
        self._row_targets: Dict[int, Tuple[str, Any]] = {}
        self._header_regions: List[Tuple[int, int, int, Tuple[str, Any]]] = []
        self._alert_targets: Dict[int, Tuple[str, Any]] = {}
        self._node_line_targets: Dict[int, int] = {}
        # (row offset within the server pane, start col, end col, target).
        # Regions take precedence over the containing server card, allowing a
        # click on a GPU cell to drill down one level further.
        self._node_gpu_regions: List[Tuple[int, int, int, Tuple[str, Any]]] = []
        self._node_header_regions: List[Tuple[int, int, int, Tuple[str, Any]]] = []
        self._job_line_targets: Dict[int, int] = {}
        # (row offset within the job pane, start col, end col, target).
        self._job_header_regions: List[Tuple[int, int, int, Tuple[str, Any]]] = []
        self._footer_regions: List[Tuple[int, int, int, Tuple[str, Any]]] = []

    # --- data -------------------------------------------------------------

    def _visible_jobs(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        pool = list(snapshot["jobs"]) + list(snapshot["recent"])
        if self.job_scope_pool is not None:
            jobs = [job for job in pool if job.get("state") == "pending"]
            if self.job_scope_pool == "shared":
                jobs = [job for job in jobs if not job.get("lane") and int(job.get("gpus") or 0) > 0]
            elif self.job_scope_pool == "cpu":
                jobs = [job for job in jobs if int(job.get("gpus") or 0) <= 0]
            else:
                jobs = [job for job in jobs if job.get("lane") == self.job_scope_pool]
        elif self.job_scope_node is not None:
            jobs = [
                job
                for job in pool
                if job.get("state") == "running"
                and job.get("node") == self.job_scope_node
                and (
                    self.job_scope_gpu is None
                    or self.job_scope_gpu in (job.get("gpu_ids") or [])
                )
            ]
        elif self.filter == "all":
            jobs = pool
        else:
            wanted = FILTER_STATES[self.filter]
            jobs = [job for job in pool if job["state"] in wanted]
        self._sort_jobs(jobs)
        return jobs

    def _sort_jobs(self, jobs: List[Dict[str, Any]]) -> None:
        lane_order = {
            lane["name"]: index
            for index, lane in enumerate((self._snapshot or {}).get("lanes") or [])
        }

        def queue_key(job):
            # Each lane has its own schedule. Priorities order only the free
            # pool; never use them to reshuffle a lane in the displayed order.
            rank = {"running": 0, "pending": 1}.get(job["state"], 2)
            pri = -(job.get("priority") or 0) if job["state"] == "pending" else 0
            lane = job.get("lane")
            if lane and job["state"] == "pending":
                return (rank, 1, lane_order.get(lane, len(lane_order)), lane,
                        job.get("lane_seq") is None, job.get("lane_seq") or 0, job["id"])
            return (rank, 0, 0, "", False, pri, job["id"])

        def time_key(job):
            elapsed = job.get("elapsed_s")
            # Jobs without a runtime (normally pending jobs) follow every
            # measured job.  Durations then descend, matching the TIME column's
            # opening arrow and keeping the longest-running work at the top.
            return (elapsed is None, -(elapsed or 0.0), job["id"])

        key_funcs = {
            "queue": queue_key,
            "id": lambda job: job["id"],
            "state": lambda job: (job["state"], job["id"]),
            "pri": lambda job: (-(job.get("priority") or 0), job["id"]),
            "name": lambda job: ((job.get("name") or "").lower(), job["id"]),
            "node": lambda job: (
                job.get("node") or job.get("node_constraint") or "~",
                job["id"],
            ),
            # Sizes and durations start biggest-first; S flips them.
            "vram": lambda job: (-(job.get("vram_mb") or 0), job["id"]),
            "used": lambda job: (-(job.get("gpu_mem_mb") or 0), job["id"]),
            "time": time_key,
        }
        jobs.sort(key=key_funcs.get(self.sort_key, queue_key))
        if self.sort_reverse:
            jobs.reverse()

    def _reanchor_selection(self) -> None:
        """Put the cursor back on the job it was on before rows moved."""
        if not self.jobs:
            self.job_index = 0
            self._selected_job_id = None
            return
        if self._selected_job_id is not None:
            for position, job in enumerate(self.jobs):
                if job["id"] == self._selected_job_id:
                    self.job_index = position
                    return
        self.job_index = max(0, min(self.job_index, len(self.jobs) - 1))
        self._selected_job_id = self.jobs[self.job_index]["id"]

    def _set_job_index(self, position: int) -> None:
        if not self.jobs:
            self.job_index = 0
            self._selected_job_id = None
            return
        self.job_index = max(0, min(position, len(self.jobs) - 1))
        self._selected_job_id = self.jobs[self.job_index]["id"]

    def selected_job(self) -> Optional[Dict[str, Any]]:
        if not self.jobs:
            return None
        self.job_index = max(0, min(self.job_index, len(self.jobs) - 1))
        return self.jobs[self.job_index]

    def selected_node(self) -> Optional[Dict[str, Any]]:
        if not self.nodes:
            return None
        self.node_index = max(0, min(self.node_index, len(self.nodes) - 1))
        return self.nodes[self.node_index]

    def _reanchor_node_scope(self) -> None:
        """Keep a resource scope attached to its node across live refreshes."""
        if not self.nodes:
            self.node_index = 0
            self.job_scope_node = None
            self.job_scope_gpu = None
            return
        if self.job_scope_node is not None:
            for position, node in enumerate(self.nodes):
                if node.get("name") == self.job_scope_node:
                    self.node_index = position
                    gpu_ids = {
                        gpu.get("index") for gpu in (node.get("gpus") or [])
                    }
                    if (
                        self.job_scope_gpu is not None
                        and self.job_scope_gpu not in gpu_ids
                    ):
                        self.job_scope_gpu = None
                    return
            self.job_scope_node = None
            self.job_scope_gpu = None
        self.node_index = max(0, min(self.node_index, len(self.nodes) - 1))

    def _scope_label(self) -> Optional[str]:
        if self.job_scope_pool is not None:
            return {"shared": "ANY GPU", "cpu": "CPU"}.get(
                self.job_scope_pool, f"lane {self.job_scope_pool}"
            )
        if self.job_scope_node is None:
            return None
        if self.job_scope_gpu is None:
            return self.job_scope_node
        return f"{self.job_scope_node}/GPU{self.job_scope_gpu}"

    def _scope_description(self) -> str:
        return f"{'queued in' if self.job_scope_pool is not None else 'running on'} {self._scope_label()}"

    def _gpu_selected(self, node_name: str, gpu_index: int) -> bool:
        if self.focus == "nodes" and self._cursor_gpu is not None:
            return (
                node_name == (self.selected_node() or {}).get("name")
                and gpu_index == self._cursor_gpu
            )
        return node_name == self.job_scope_node and gpu_index == self.job_scope_gpu

    def _set_pool_scope(self, pool: str) -> None:
        self.job_scope_node = None
        self.job_scope_gpu = None
        self.job_scope_pool = pool
        self.sort_key = "queue"
        self.sort_reverse = False
        self.focus = "jobs"
        self.show_detail = False
        self._reset_job_selection_for_view()

    def _reset_job_selection_for_view(self) -> None:
        self.job_index = 0
        self._selected_job_id = None
        self.detail_page = 0
        self.log_offset = 0
        self.pending_confirm = None
        if self._snapshot is not None:
            self.jobs = self._visible_jobs(self._snapshot)
            self._reanchor_selection()
        if self.show_log:
            job = self.selected_job()
            self._request_log((job["id"], "stdout") if job is not None else None)

    def _set_job_scope(self, node_name: str, gpu_index: Optional[int] = None) -> None:
        node_name = str(node_name)
        gpu_index = None if gpu_index is None else int(gpu_index)
        if (
            node_name == self.job_scope_node
            and gpu_index == self.job_scope_gpu
        ):
            return
        self.job_scope_node = node_name
        self.job_scope_gpu = gpu_index
        self.job_scope_pool = None
        self._reset_job_selection_for_view()

    def _clear_job_scope(self) -> bool:
        if self._scope_label() is None:
            return False
        self.job_scope_node = None
        self.job_scope_gpu = None
        self.job_scope_pool = None
        self._reset_job_selection_for_view()
        return True

    # --- rendering --------------------------------------------------------

    @property
    def _palette(self) -> Dict[str, str]:
        return THEMES.get(self.theme, THEMES["classic"])

    def _style(self, text: str, style: Optional[str]) -> str:
        if not style:
            return text
        formatter = getattr(self.term, self._palette.get(style, style), None)
        return formatter(text) if callable(formatter) else text

    @staticmethod
    def _wrap_plain(text: Any, width: int) -> List[str]:
        return wrap_display(text, width)

    def _field_lines(
        self,
        label: str,
        value: Any,
        width: int,
        *,
        style: Optional[str] = None,
    ) -> List[str]:
        """Render a labeled detail field without truncating its value."""
        prefix = f"  {label:<5}"
        continuation = " " * display_width(prefix)
        available = max(1, width - display_width(prefix))
        values = self._wrap_plain(value, available)
        return [
            self._compose(
                [
                    (prefix if index == 0 else continuation, "bright_black"),
                    (line, style),
                ],
                width,
            )
            for index, line in enumerate(values)
        ]

    def _compose(self, segments, width: int, *, highlight: bool = False) -> str:
        """Join `(text, style)` pairs, truncating on plain text before styling.

        Truncating first is what keeps escape sequences from being counted as
        visible columns, and every measurement is in terminal columns so a
        Chinese note cannot push the line past the right edge.
        """
        plain_parts = []
        remaining = width
        for text, _style in segments:
            if remaining <= 0:
                plain_parts.append("")
                continue
            plain_parts.append(fit_display(text, remaining))
            remaining -= display_width(plain_parts[-1])
        if highlight:
            return self._highlight_row(plain_parts, segments, width)
        return "".join(
            self._style(part, style)
            for part, (_text, style) in zip(plain_parts, segments)
        )

    def _highlight_row(self, plain_parts, segments, width: int) -> str:
        """The selection cursor: a dim band under the row.

        Reverse video would paint a solid block and discard every column's
        colour; laying a near-background grey underneath keeps the row
        readable as data while still being findable as the cursor.
        """
        out = []
        for part, (_text, style) in zip(plain_parts, segments):
            if not part:
                continue
            fg = self._palette.get(style, style) if style else None
            attr = f"{fg}_on_{SELECTION_BG}" if fg else f"on_{SELECTION_BG}"
            formatter = getattr(self.term, attr, None)
            out.append(formatter(part) if callable(formatter) else part)
        used = sum(display_width(part) for part in plain_parts)
        pad = " " * max(0, width - used)
        if pad:
            formatter = getattr(self.term, f"on_{SELECTION_BG}", None)
            out.append(formatter(pad) if callable(formatter) else pad)
        return "".join(out)

    def _header_lines(self, state: Dict[str, Any], width: int) -> List[str]:
        snapshot = state["snapshot"] or {}
        counts = snapshot.get("counts") or {}
        self._header_regions = [(0, 0, len(" nvidb queue ") - 1, ("help", None))]
        segments = [(" ", None)]
        count_column = 1
        for label, style in (
            ("running", "green"),
            ("pending", "yellow"),
            ("completed", "cyan"),
            ("failed", "red"),
            ("cancelled", "bright_black"),
            ("timeout", "red"),
            ("lost", "red"),
        ):
            if counts.get(label):
                if len(segments) > 1:
                    segments.append((" · ", "bright_black"))
                    count_column += 3
                count_text = f"{label} {counts[label]}"
                filter_name = label if label in ("running", "pending") else "finished"
                self._header_regions.append(
                    (
                        1,
                        count_column,
                        count_column + display_width(count_text) - 1,
                        ("filter", filter_name),
                    )
                )
                segments.append((count_text, style))
                count_column += display_width(count_text)
        if len(segments) == 1:
            segments.append(("queue empty", "bright_black"))

        tick_age = age_seconds(snapshot.get("last_tick_at"))
        tick_text = "never" if tick_age is None else f"{int(tick_age)}s ago"
        mode = "auto" if state["auto_tick"] else "manual"
        status = "working" if state["busy"] else "idle"
        # Only where a keeper is meant to exist: its absence is the difference
        # between "the queue is idle" and "nothing will start until I look".
        keeper = snapshot.get("keeper") or {}
        keeper_text = ""
        if keeper.get("installed") or keeper.get("running"):
            keeper_text = f"keeper {'up' if keeper.get('running') else 'DOWN'} · "
        # A coloured wordmark, tokscale-style, instead of a reverse-video block.
        title = " nvidb queue "
        scope = self._scope_label()
        job_view = self._scope_description() if scope else f"filter {self.filter}"
        meta = (
            f"tick {tick_text} · {keeper_text}{mode} · {status} · {job_view} "
            f"· view {self.resource_view} "
        )
        gap = max(1, width - len(title) - len(meta))
        return [
            self._compose(
                [(title, "cyan"), (" " * gap, None), (meta, "bright_black")], width
            ),
            self._compose(segments, width),
        ]

    def _alert_lines(self, snapshot: Dict[str, Any], width: int) -> List[str]:
        """A banner for failures nobody has acknowledged yet.

        Alerts sit above everything else because they are the one thing on this
        screen that needs a decision rather than a glance.
        """
        self._alert_targets = {}
        alerts = snapshot.get("alerts") or []
        open_alerts = [alert for alert in alerts if not alert.get("acknowledged_at")]
        if not open_alerts:
            return []
        shown = open_alerts[-3:]
        lines = [
            self._compose(
                [
                    (f" ⚠ {len(open_alerts)} alert(s) ", "red"),
                    ("— click a row to inspect; A acknowledges all", "bright_black"),
                ],
                width,
            )
        ]
        for alert in shown:
            style = "red" if alert.get("severity") == "error" else "yellow"
            line_index = len(lines)
            lines.append(
                self._compose(
                    [
                        (f"   {alert['id']:>4} ", "bright_black"),
                        (f"{alert['kind']:<18}", style),
                        (" ".join(str(alert["title"]).split()), None),
                    ],
                    width,
                )
            )
            if alert.get("job_id") is not None:
                self._alert_targets[line_index] = ("job_log", alert["job_id"])
            elif alert.get("node"):
                self._alert_targets[line_index] = ("node_name", alert["node"])
        if len(open_alerts) > len(shown):
            lines.append(
                self._style(
                    f"   … {len(open_alerts) - len(shown)} more", "bright_black"
                )
            )
        return lines

    @staticmethod
    def _node_health_dot(node: Dict[str, Any]) -> Tuple[str, str]:
        """One glanceable dot per node: can this node take a job right now?

        Green = up with free VRAM somewhere, yellow = up but full (or
        drained), red = down, hollow = no GPU data yet.
        """
        if node["state"] == "down":
            return ("●", "red")
        if not node["enabled"] or node["state"] == "drain":
            return ("●", "yellow")
        gpus = node.get("gpus") or []
        if not gpus:
            return ("◌", "bright_black")
        if any(gpu.get("free_mb", 0) >= 1024 for gpu in gpus):
            return ("●", "green")
        return ("●", "yellow")

    @staticmethod
    def _flow_segment_width(segments) -> int:
        return sum(display_width(segment[0]) for segment in segments)

    def _flow_line(self, segments, width: int, row: int) -> str:
        """Render styled flow segments and retain hit boxes for job tokens."""
        rendered = []
        column = 0
        for text, style, target in segments:
            if column >= width:
                break
            shown = fit_display(str(text), width - column)
            shown_width = display_width(shown)
            if shown_width and target is not None:
                self._node_gpu_regions.append(
                    (row, column, column + shown_width - 1, target)
                )
            rendered.append(self._style(shown, style))
            column += shown_width
        return "".join(rendered)

    @staticmethod
    def _flow_job_map(snapshot: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """All jobs reachable from a snapshot, including lane-only entries."""
        jobs: Dict[int, Dict[str, Any]] = {}
        for job in (list(snapshot.get("dependencies") or [])
                    + list(snapshot.get("jobs") or []) + list(snapshot.get("recent") or [])):
            if job.get("id") is not None:
                jobs[int(job["id"])] = job
        for lane in snapshot.get("lanes") or []:
            for job in list(lane.get("running") or []) + list(lane.get("queued") or []):
                if job.get("id") is not None:
                    jobs[int(job["id"])] = job
        return jobs

    @staticmethod
    def _flow_priority(job: Dict[str, Any]) -> str:
        priority = int(job.get("priority") or 0)
        return f"P{priority:+d}" if priority else "P0"

    @staticmethod
    def _sort_jobs_in_lane(jobs) -> None:
        jobs.sort(key=lambda job: (
            job.get("lane_seq") is None, job.get("lane_seq") or 0, job["id"]
        ))

    def _flow_job_token(
        self,
        job: Dict[str, Any],
        *,
        position: Optional[int] = None,
        name_width: int = 12,
    ) -> str:
        name = fit_display(job.get("name") or "-", name_width)
        prefix = f"{position}:" if position is not None else ""
        markers = []
        if job.get("held") or job.get("held_reason"):
            markers.append("HOLD")
        dependencies = list(job.get("depends_on") or [])
        any_dependencies = list(job.get("depends_any") or [])
        if dependencies:
            markers.append(f"←✓#{dependencies[0]}")
            if len(dependencies) > 1:
                markers[-1] += f"+{len(dependencies) - 1}"
        if any_dependencies:
            markers.append(f"←◇#{any_dependencies[0]}")
            if len(any_dependencies) > 1:
                markers[-1] += f"+{len(any_dependencies) - 1}"
        marker_text = f" {' '.join(markers)}" if markers else ""
        node = ""
        if job.get("lane") is None and job.get("node_constraint"):
            node = f" @{fit_display(job['node_constraint'], 8)}"
        return (
            f"[{prefix}#{job['id']} {self._flow_priority(job)} "
            f"{name}{node}{marker_text}]"
        )

    def _flow_running_token(self, jobs: List[Dict[str, Any]], width: int) -> str:
        if not jobs:
            return "[ IDLE ]"
        job = jobs[0]
        more = f" +{len(jobs) - 1}" if len(jobs) > 1 else ""
        if width < 70:
            return (
                f"[ RUN #{job['id']} "
                f"{fit_display(job.get('name') or '-', 6)}{more} ]"
            )
        runtime = (
            format_duration(job.get("elapsed_s"))
            if job.get("elapsed_s") is not None
            else "-"
        )
        name_width = 12 if width >= 120 else 8
        return (
            f"[ RUN #{job['id']} {fit_display(job.get('name') or '-', name_width)} "
            f"{runtime}{more} ]"
        )

    def _flow_gpu_activity(self, node, gpu, running, width):
        if node.get("state") not in ("up", "drain"):
            return "[ UNKNOWN ]", "yellow"
        if running:
            return self._flow_running_token(running, width), "green"
        if (gpu.get("external_procs") or gpu.get("external_mem_mb") or any(
            not process.get("managed") for process in gpu.get("processes") or []
        )):
            return "[ EXTERNAL ]", "yellow"
        if gpu.get("util_percent") or gpu.get("mem_used_mb"):
            return "[ BUSY ]", "yellow"
        if gpu.get("util_percent") is None or gpu.get("mem_used_mb") is None:
            return "[ UNKNOWN ]", "bright_black"
        return "[ IDLE ]", "bright_black"

    @staticmethod
    def _unique_flow_jobs(jobs) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for job in jobs:
            job_id = job.get("id")
            if job_id in seen:
                continue
            seen.add(job_id)
            unique.append(job)
        return unique

    def _append_flow_queue_tokens(
        self,
        segments,
        queued: List[Dict[str, Any]],
        width: int,
        pool: Optional[str] = None,
    ) -> None:
        """Append as many ordered queue tokens as fit, then an overflow count."""
        if not queued:
            return
        name_width = 12 if width >= 130 else 8
        shown = 0
        for position, job in enumerate(queued, start=1):
            token = self._flow_job_token(
                job,
                position=position,
                name_width=name_width,
            )
            addition = [
                (" ──▶ ", "bright_black", None),
                (
                    token,
                    "magenta"
                    if job.get("held") or job.get("held_reason")
                    else "yellow",
                    ("flow_job", int(job["id"])),
                ),
            ]
            remaining_after = len(queued) - position
            reserve = display_width(f"  +{remaining_after}") if remaining_after else 0
            if self._flow_segment_width(segments + addition) + reserve > width:
                break
            segments.extend(addition)
            shown += 1
        hidden = len(queued) - shown
        if hidden:
            segments.append((f"  +{hidden}", "bright_black", ("pool", pool) if pool else None))

    def _flow_dependency_lines(
        self,
        jobs: Dict[int, Dict[str, Any]],
        width: int,
        start_row: int,
    ) -> List[str]:
        edges = []
        for job in sorted(jobs.values(), key=lambda item: int(item.get("id") or 0)):
            if job.get("state") not in ("pending", "running"):
                continue
            for dependency in job.get("depends_on") or []:
                edges.append((int(dependency), int(job["id"]), "✓"))
            for dependency in job.get("depends_any") or []:
                edges.append((int(dependency), int(job["id"]), "◇"))
        if not edges:
            return []

        lines: List[str] = []
        edge_index = 0
        while edge_index < len(edges) and len(lines) < 2:
            segments = [
                (
                    "  DEPS " if not lines else "       ",
                    "magenta" if not lines else "bright_black",
                    None,
                )
            ]
            while edge_index < len(edges):
                dependency, dependent, kind = edges[edge_index]
                upstream = jobs.get(dependency) or {}
                state = upstream.get("state")
                style = (
                    "green"
                    if state == "completed"
                    else "red"
                    if state in ("failed", "cancelled", "timeout", "lost")
                    else "cyan"
                    if state == "running"
                    else "yellow"
                    if state == "pending"
                    else "bright_black"
                )
                separator = " · " if len(segments) > 1 else ""
                edge = f"#{dependency} ─{kind}▶ #{dependent}"
                addition = [
                    (separator, "bright_black", None),
                    (edge, style, ("flow_job", dependent)),
                ]
                remaining = len(edges) - edge_index - 1
                reserve = display_width(f"  +{remaining}") if remaining else 0
                if self._flow_segment_width(segments + addition) + reserve > width:
                    break
                segments.extend(addition)
                edge_index += 1
            if len(segments) == 1:
                break
            lines.append(self._flow_line(segments, width, start_row + len(lines)))
        hidden = len(edges) - edge_index
        if hidden and lines:
            suffix = f"  +{hidden}"
            if display_width(self.term.strip_seqs(lines[-1])) + display_width(suffix) <= width:
                lines[-1] += self._style(suffix, "bright_black")
        return lines

    def _flow_lines(self, width: int) -> List[str]:
        """Visual task pools: node → GPU → running job → ordered lane queue."""
        self._node_line_targets = {}
        self._node_gpu_regions = []
        self._node_header_regions = []
        snapshot = self._snapshot or {}
        active = list(snapshot.get("jobs") or [])
        job_map = self._flow_job_map(snapshot)
        lanes = list(snapshot.get("lanes") or [])
        gpu_count = sum(len(node.get("gpus") or []) for node in self.nodes)
        running_count = len({job["id"] for job in active if job.get("state") == "running"})
        pending_count = len({job["id"] for job in active if job.get("state") == "pending"})

        title = (
            f"─ TASK FLOW · {len(self.nodes)}N {gpu_count}G · "
            f"R{running_count} Q{pending_count} "
        )
        scope = self._scope_label()
        if scope:
            scope_text = f"· {self._scope_description()} "
            clear_text = "[all jobs]"
            clear_start = display_width(title) + display_width(scope_text)
            self._node_header_regions.append(
                (
                    0,
                    clear_start,
                    clear_start + display_width(clear_text) - 1,
                    ("scope_all", None),
                )
            )
            title += scope_text + clear_text + " "
        else:
            title += "· ✓=success ◇=any-result "
        lines = [
            self._compose(
                [
                    (title, "cyan" if scope else "bright_black"),
                    ("─" * max(0, width - display_width(title)), "bright_black"),
                ],
                width,
            )
        ]

        lanes_by_gpu: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for lane in lanes:
            for gpu_index in lane.get("gpu_ids") or []:
                lanes_by_gpu.setdefault((str(lane.get("node")), int(gpu_index)), lane)
        running_by_gpu: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for job in active:
            if job.get("state") != "running" or not job.get("node"):
                continue
            for gpu_index in job.get("gpu_ids") or []:
                running_by_gpu.setdefault(
                    (str(job["node"]), int(gpu_index)), []
                ).append(job)

        index_width = len(str(max(1, len(self.nodes))))
        for position, node in enumerate(self.nodes):
            node_name = str(node.get("name"))
            node_lanes = [lane for lane in lanes if lane.get("node") == node_name]
            node_queued = {
                job["id"]
                for lane in node_lanes
                for job in (lane.get("queued") or [])
            }
            node_running = {
                job["id"]
                for job in active
                if job.get("state") == "running" and job.get("node") == node_name
            }
            selected = node_name == self.job_scope_node or (
                self.job_scope_node is None
                and self.focus == "nodes"
                and position == self.node_index
            )
            marker = "❯" if selected else " "
            dot, dot_style = self._node_health_dot(node)
            state = "drain" if not node.get("enabled") else node.get("state") or "unknown"
            header = [
                (f"{marker} ▾ [{position + 1:{index_width}d}] ", "cyan" if selected else "bright_black", None),
                (f"{dot} ", dot_style, None),
                (node_name, "bold", None),
                (
                    f"  {len(node.get('gpus') or [])} GPU · R{len(node_running)} · Q{len(node_queued)}",
                    "bright_black",
                    None,
                ),
                (f"  {state}", "red" if state == "down" else "yellow" if state == "drain" else "bright_black", None),
            ]
            if state == "down" and node.get("last_error"):
                header.append(
                    (
                        f" · ! {fit_display(node['last_error'], 28)}",
                        "red",
                        None,
                    )
                )
            header_row = len(lines)
            lines.append(self._flow_line(header, width, header_row))
            self._node_line_targets[header_row] = position

            for gpu in node.get("gpus") or []:
                gpu_index = int(gpu.get("index") or 0)
                lane = lanes_by_gpu.get((node_name, gpu_index))
                lane_running = list((lane or {}).get("running") or [])
                observed_running = running_by_gpu.get((node_name, gpu_index), [])
                running = self._unique_flow_jobs(lane_running + observed_running)
                queued = list((lane or {}).get("queued") or [])
                activity, activity_style = self._flow_gpu_activity(node, gpu, running, width)
                util = gpu.get("util_percent")
                util_text = f"{int(util):>3}%" if util is not None else "  -%"
                if width < 70:
                    gpu_status = (
                        f"{util_text} {format_mb(gpu.get('free_mb') or 0):>6}  "
                    )
                    flow_arrow = " → "
                else:
                    gpu_status = (
                        f"{util_text} · "
                        f"{format_mb(gpu.get('free_mb') or 0):>6} free  "
                    )
                    flow_arrow = " ──▶ "
                gpu_selected = self._gpu_selected(node_name, gpu_index)
                segments = [
                    (f"{'  ❯ ' if gpu_selected else '    '}G{gpu_index:<2} ", "cyan" if gpu_selected else "bright_black", None),
                    (
                        gpu_status,
                        "bright_black",
                        None,
                    ),
                    (
                        activity,
                        activity_style,
                        ("flow_job", int(running[0]["id"])) if running else None,
                    ),
                    (flow_arrow, "bright_black", None),
                    (
                        f"{{ lane Q{len(queued)} }}",
                        "yellow" if queued else "bright_black",
                        ("pool", lane["name"]) if lane else None,
                    ),
                ]
                self._append_flow_queue_tokens(segments, queued, width, (lane or {}).get("name"))
                if lane:
                    blocked = lane.get("blocked")
                    lane_state = (
                        "PAUSED"
                        if lane.get("paused")
                        else fit_display(blocked, 22)
                        if blocked
                        else ""
                    )
                    if lane_state:
                        segments.append((f"  ! {lane_state}", "magenta", None))
                gpu_row = len(lines)
                lines.append(self._flow_line(segments, width, gpu_row))
                # Job tokens were registered first and therefore win over this
                # whole-row GPU target when their regions overlap.
                self._node_gpu_regions.append(
                    (gpu_row, 0, width - 1, ("gpu", (position, gpu_index)))
                )
                self._node_line_targets[gpu_row] = position

        shared = sorted(
            [
                job
                for job in active
                if job.get("state") == "pending"
                and not job.get("lane")
                and int(job.get("gpus") or 0) > 0
            ],
            key=lambda job: (-(int(job.get("priority") or 0)), int(job["id"])),
        )
        if shared:
            segments = [
                ("  ANY GPU  ", "cyan", None),
                ("──▶ ", "bright_black", None),
                (f"{{ Q{len(shared)} · priority }}", "yellow", ("pool", "shared")),
            ]
            self._append_flow_queue_tokens(segments, shared, width, "shared")
            lines.append(self._flow_line(segments, width, len(lines)))

        cpu = [
            job
            for job in active
            if int(job.get("gpus") or 0) <= 0
            and job.get("state") in ("pending", "running")
        ]
        if cpu:
            cpu_running = [job for job in cpu if job.get("state") == "running"]
            cpu_queued = sorted(
                [job for job in cpu if job.get("state") == "pending"],
                key=lambda job: (-(int(job.get("priority") or 0)), int(job["id"])),
            )
            segments = [
                ("  CPU      ", "cyan", None),
                (self._flow_running_token(cpu_running, width), "green" if cpu_running else "bright_black", ("flow_job", int(cpu_running[0]["id"])) if cpu_running else None),
                (" ──▶ ", "bright_black", None),
                (f"{{ Q{len(cpu_queued)} }}", "yellow" if cpu_queued else "bright_black", ("pool", "cpu")),
            ]
            self._append_flow_queue_tokens(segments, cpu_queued, width, "cpu")
            lines.append(self._flow_line(segments, width, len(lines)))

        lines.extend(self._flow_dependency_lines(job_map, width, len(lines)))
        if not self.nodes and not shared and not cpu:
            lines.append(self._style("  (no active resources or jobs)", "bright_black"))
        return lines

    def _node_lines(self, width: int) -> List[str]:
        self._node_line_targets = {}
        self._node_gpu_regions = []
        self._node_header_regions = []
        title = f"─ SERVERS ({len(self.nodes)}) "
        scope = self._scope_label()
        if scope:
            scope_text = f"{self._scope_description()} "
            clear_text = "[all jobs]"
            clear_start = display_width(title) + display_width(scope_text)
            self._node_header_regions.append(
                (
                    0,
                    clear_start,
                    clear_start + display_width(clear_text) - 1,
                    ("scope_all", None),
                )
            )
            heading = title + scope_text + clear_text + " "
        else:
            heading = title
        lines = [
            self._compose(
                [
                    (heading, "cyan" if scope else "bright_black"),
                    ("─" * max(0, width - display_width(heading)), "bright_black"),
                ],
                width,
            )
        ]
        index_width = len(str(max(1, len(self.nodes))))
        for position, node in enumerate(self.nodes):
            if position:
                # Node rows are single lines now, so a thin rule is what keeps
                # neighbouring nodes from reading as one list.
                lines.append(self._style("┄" * width, "bright_black"))
            node_start = len(lines)
            selected = node.get("name") == self.job_scope_node or (
                self.job_scope_node is None
                and self.focus == "nodes"
                and position == self.node_index
            )
            block = self._node_block(
                node,
                width,
                selected,
                position=position,
                index_width=index_width,
            )
            lines.extend(block)
            for line_index in range(node_start, len(lines)):
                self._node_line_targets[line_index] = position
            self._node_gpu_regions.extend(
                self._gpu_click_regions(node, block, node_start, position, width)
            )
        return lines

    @staticmethod
    def _segments_width(segments) -> int:
        return sum(display_width(text) for text, _style in segments)

    def _node_one_line(
        self, head, middle_groups, tail, width: int, *, highlight: bool
    ) -> Optional[str]:
        """Everything about one node on a single line, or None if the width
        cannot take it — the caller then falls back to stacked lines."""
        segments = list(head)
        for group in middle_groups:
            segments.append(("  ", None))
            segments.extend(group)
        gap = width - self._segments_width(segments) - self._segments_width(tail)
        if gap < 2:
            return None
        segments.append((" " * gap, None))
        segments.extend(tail)
        return self._compose(segments, width, highlight=highlight)

    def _node_header_line(self, head, tail, width: int, *, highlight: bool) -> str:
        """The stacked layout's first line: name left, hostname and state
        right, truncating rather than refusing when the width is tight."""
        gap = max(
            1, width - self._segments_width(head) - self._segments_width(tail)
        )
        return self._compose(
            [*head, (" " * gap, None), *tail], width, highlight=highlight
        )

    def _node_block(
        self,
        node: Dict[str, Any],
        width: int,
        selected: bool,
        *,
        position: int = 0,
        index_width: int = 1,
    ) -> List[str]:
        state = node["state"]
        # An up node is the normal case and stays quiet; only trouble
        # (down, drain) earns colour.
        state_style = {"down": "red", "drain": "yellow"}.get(state, "bright_black")
        if not node["enabled"]:
            state = "drain"
            state_style = "yellow"
        dot, dot_style = self._node_health_dot(node)
        # The GPU model lives on the node line: nodes are almost always
        # homogeneous, so saying it once frees the grid cells to show nothing
        # but occupancy.
        models = []
        for gpu in node.get("gpus") or []:
            name = gpu.get("name") or "-"
            if name not in models:
                models.append(name)
        model_text = (
            f" · {len(node.get('gpus') or [])}× {'/'.join(models)}"
            if models
            else ""
        )
        marker = "❯" if selected else " "
        expand_icon = "▾" if node.get("gpus") else "▸"
        head = [
            (f"{marker} ", "cyan"),
            (f"{expand_icon} [{position + 1:{index_width}d}] ", "bright_black"),
            (f"{dot} ", dot_style),
            (node["name"], "bold"),
            (model_text, "bright_black"),
        ]
        tail = [
            (f"{node['hostname'] or ''}  ", "bright_black"),
            (state, state_style),
        ]

        if state == "down" and node.get("last_error"):
            error = [(f"! {node['last_error']}", "red")]
            line = self._node_one_line(
                head, [error], tail, width, highlight=selected
            )
            if line is not None:
                return [line]
            return [
                self._node_header_line(head, tail, width, highlight=selected),
                self._style(fit_display(f"    ! {node['last_error']}", width), "red"),
            ]

        if self.proc_view == "all":
            # The full drill-down: one line per GPU, one per process.
            block = [self._node_header_line(head, tail, width, highlight=selected)]
            for gpu in node["gpus"]:
                block.append(
                    self._compose(
                        self._gpu_segments(
                            gpu,
                            width,
                            selected=self._gpu_selected(node["name"], gpu["index"]),
                        ),
                        width,
                    )
                )
                block.extend(self._gpu_process_lines(gpu, width))
            return block

        running = self._running_job_ids_by_gpu(node)
        cells = [
            self._gpu_cell_segments(
                gpu,
                running.get(gpu["index"], []),
                fixed_width=False,
                selected=self._gpu_selected(node["name"], gpu["index"]),
            )
            for gpu in node.get("gpus") or []
        ]
        ext = (
            self._external_summary_segments(node)
            if self.proc_view == "summary"
            else []
        )
        # Squeeze the whole node onto one line, shedding parts only as the
        # width forces it: first the foreign-process summary drops to its own
        # line, then the GPU cells fall back to the stacked grid.
        if ext:
            line = self._node_one_line(
                head, cells + [ext], tail, width, highlight=selected
            )
            if line is not None:
                return [line]
        line = self._node_one_line(head, cells, tail, width, highlight=selected)
        if line is not None:
            block = [line]
            if ext:
                block.append(self._compose([("    ", None), *ext], width))
            return block
        block = [self._node_header_line(head, tail, width, highlight=selected)]
        block.extend(self._gpu_grid_lines(node, width))
        if ext:
            block.append(self._compose([("    ", None), *ext], width))
        return block

    def _gpu_click_regions(
        self,
        node: Dict[str, Any],
        block: List[str],
        row_offset: int,
        node_position: int,
        width: int,
    ) -> List[Tuple[int, int, int, Tuple[str, Any]]]:
        """Locate each rendered GPU cell in terminal display coordinates.

        A node may fit inline, fall back to a fixed grid, or use the full
        per-process row.  Deriving hit boxes from the finished lines keeps the
        mouse map aligned with all three layouts, including wide node names.
        """
        regions: List[Tuple[int, int, int, Tuple[str, Any]]] = []
        running = self._running_job_ids_by_gpu(node)
        for gpu in node.get("gpus") or []:
            gpu_index = int(gpu["index"])
            label = f"GPU{gpu_index}" if self.proc_view == "all" else f"G{gpu_index}"
            for relative_row, line in enumerate(block):
                plain = self.term.strip_seqs(line)
                if self.proc_view == "all":
                    prefix = f"    {label} "
                    character_start = 4 if plain.startswith(prefix) else -1
                else:
                    character_start = plain.find(label)
                    while character_start >= 0:
                        after = character_start + len(label)
                        index_width = max(3, len(label))
                        bar_start = character_start + index_width
                        if (
                            (after >= len(plain) or not plain[after].isdigit())
                            and bar_start < len(plain)
                            and plain[bar_start] in "━─"
                        ):
                            break
                        character_start = plain.find(label, after)
                if character_start < 0:
                    continue
                start = display_width(plain[:character_start])
                if self.proc_view == "all":
                    cell_width = max(1, display_width(plain) - start)
                else:
                    cell_width = self._segments_width(
                        self._gpu_cell_segments(
                            gpu,
                            running.get(gpu_index, []),
                            fixed_width=relative_row != 0,
                            selected=False,
                        )
                    )
                end = min(width - 1, start + max(1, cell_width) - 1)
                regions.append(
                    (
                        row_offset + relative_row,
                        start,
                        end,
                        ("gpu", (node_position, gpu_index)),
                    )
                )
                break
        return regions

    # Plain columns one grid cell occupies: "G0 " + 10-column bar + " " +
    # 6-column used + "/" + 6-column total + " " + 5-column job ids.
    _GPU_CELL_WIDTH = 3 + 10 + 1 + 6 + 1 + 6 + 1 + 5
    _GPU_CELL_GAP = 3

    def _running_job_ids_by_gpu(self, node: Dict[str, Any]) -> Dict[int, List[int]]:
        """Which queue jobs are running on each of this node's GPUs."""
        running: Dict[int, List[int]] = {}
        jobs = (self._snapshot or {}).get("jobs") or []
        for job in jobs:
            if job.get("state") != "running" or job.get("node") != node["name"]:
                continue
            for gpu_index in job.get("gpu_ids") or []:
                running.setdefault(gpu_index, []).append(job["id"])
        return running

    def _gpu_cell_segments(
        self,
        gpu: Dict[str, Any],
        running_ids: List[int],
        *,
        fixed_width: bool = True,
        selected: bool = False,
    ) -> List[Tuple[str, Optional[str]]]:
        """One GPU as a grid cell: index, bar, used/total, jobs.

        The bar keeps the pane's one legend - foreign memory (yellow),
        queue reservations (cyan), free (dim). The used amount carries the
        scarcity colour, a `~` before it means the split is inferred (a
        blind driver), and the tail names the queue jobs running here.

        Fixed-width cells line up under each other in the stacked grid;
        inline on a single node row there is nothing to line up with, so
        `fixed_width=False` drops the padding instead of carrying it.
        """
        total = max(1, gpu["mem_total_mb"])
        external = max(0, gpu["external_mem_mb"])
        reserved = max(0, gpu["reserved_mb"])
        used_style = (
            None if gpu["free_mb"] >= 4096
            else "yellow" if gpu["free_mb"] >= 1024
            else "red"
        )
        used_text = format_mb(gpu["mem_used_mb"])
        if gpu.get("attribution") == "blind":
            used_text = "~" + used_text
            used_style = used_style or "yellow"
        ids_text = ""
        if running_ids:
            ids_text = "#" + ",".join(str(job_id) for job_id in running_ids)
            if display_width(ids_text) > 5:
                ids_text = f"#{running_ids[0]}+{len(running_ids) - 1}"
        index_label = f"G{gpu['index']}"
        segments: List[Tuple[str, Optional[str]]] = [
            (f"{index_label:<3}", "cyan" if selected else "bright_black"),
        ]
        segments.extend(
            smooth_bar(
                10,
                (
                    (min(1.0, external / total), "yellow"),
                    (min(1.0, reserved / total), "cyan"),
                ),
            )
        )
        segments.append((" ", None))
        if fixed_width:
            segments.append((f"{used_text:>6}", used_style))
            segments.append(("/", "bright_black"))
            segments.append((f"{format_mb(gpu['mem_total_mb']):<6}", "bright_black"))
            segments.append((" ", None))
            segments.append((f"{fit_display(ids_text, 5):<5}", "cyan"))
        else:
            segments.append((used_text, used_style))
            segments.append(("/", "bright_black"))
            segments.append((format_mb(gpu["mem_total_mb"]), "bright_black"))
            if ids_text:
                segments.append((" ", None))
                segments.append((ids_text, "cyan"))
        return segments

    def _gpu_grid_lines(self, node: Dict[str, Any], width: int) -> List[str]:
        """Lay the node's GPUs out several to a line instead of one each."""
        gpus = node.get("gpus") or []
        if not gpus:
            return []
        running = self._running_job_ids_by_gpu(node)
        indent = 4
        per_row = max(
            1,
            (width - indent + self._GPU_CELL_GAP)
            // (self._GPU_CELL_WIDTH + self._GPU_CELL_GAP),
        )
        lines = []
        for start in range(0, len(gpus), per_row):
            segments: List[Tuple[str, Optional[str]]] = [(" " * indent, None)]
            for offset, gpu in enumerate(gpus[start : start + per_row]):
                if offset:
                    segments.append((" " * self._GPU_CELL_GAP, None))
                segments.extend(
                    self._gpu_cell_segments(
                        gpu,
                        running.get(gpu["index"], []),
                        selected=self._gpu_selected(node["name"], gpu["index"]),
                    )
                )
            lines.append(self._compose(segments, width))
        return lines

    def _external_summary_segments(
        self, node: Dict[str, Any]
    ) -> List[Tuple[str, Optional[str]]]:
        """The card's foreign occupants as one run of segments, biggest first.

        These processes are why a queued job is waiting, so they stay
        visible by default - inline on the node's line when it fits, on a
        single summary line below it otherwise. Empty when there is nothing
        foreign to report.
        """
        entries = []
        multi_gpu = len(node.get("gpus") or []) > 1
        for gpu in node.get("gpus") or []:
            if gpu.get("attribution") == "blind":
                continue
            for process in gpu.get("processes") or []:
                entry = GpuProcess.from_dict(process)
                if entry.managed:
                    continue
                entries.append((entry.mem_mb or 0, gpu["index"], entry))
        if not entries:
            return []
        entries.sort(key=lambda item: (-item[0], item[1]))
        segments: List[Tuple[str, Optional[str]]] = [("ext ", "bright_black")]
        shown = entries[:PROC_SUMMARY_LIMIT + 1]
        for index, (mem_mb, gpu_index, entry) in enumerate(shown):
            if index:
                segments.append((" · ", "bright_black"))
            if multi_gpu:
                segments.append((f"G{gpu_index} ", "bright_black"))
            segments.append((f"{entry.username or '-'} ", None))
            segments.append((format_mb(mem_mb), "yellow"))
            segments.append(
                (f" {fit_display(entry.name or str(entry.pid), 18)}", "bright_black")
            )
        if len(entries) > len(shown):
            segments.append((f" · +{len(entries) - len(shown)} more", "bright_black"))
        return segments

    def _gpu_segments(
        self, gpu: Dict[str, Any], width: int, *, selected: bool = False
    ) -> List[Tuple[str, Optional[str]]]:
        """One GPU as a btop-style line: temp, util, and a memory bar whose
        segments say *whose* memory it is - foreign processes (yellow), this
        queue's reservations (cyan), free (dim)."""
        total = max(1, gpu["mem_total_mb"])
        bar_width = 20 if width >= 110 else 10
        external = max(0, gpu["external_mem_mb"])
        reserved = max(0, gpu["reserved_mb"])

        # Quiet by default: a healthy number is grey, colour appears only once
        # a value is worth a second look.
        util = gpu["util_percent"]
        util_text = f"{util:>3}%" if util is not None else "  -%"
        util_style = (
            "red" if util is not None and util >= 90
            else "yellow" if util is not None and util >= 50
            else "bright_black"
        )
        temp = gpu.get("temperature_c")
        temp_text = f"{temp:>3}°" if temp is not None else "  -°"
        temp_style = (
            "red" if temp is not None and temp >= 85
            else "yellow" if temp is not None and temp >= 70
            else "bright_black"
        )
        free_style = (
            None if gpu["free_mb"] >= 4096
            else "yellow" if gpu["free_mb"] >= 1024
            else "red"
        )
        source = (
            "blind"
            if gpu.get("attribution") == "blind"
            else f"{gpu['external_procs']}p"
        )
        name_width = 20 if width >= 100 else 14
        segments: List[Tuple[str, Optional[str]]] = [
            (f"    GPU{gpu['index']} ", "cyan" if selected else "bright_black"),
            (pad_display(fit_display(gpu["name"] or "-", name_width), name_width + 1), None),
            (temp_text + " ", temp_style),
            (util_text + " ", util_style),
        ]
        # One continuous line - heavy where memory is spoken for, dim where
        # it is free - instead of a stack of block glyphs.
        segments.extend(
            smooth_bar(
                bar_width,
                (
                    (min(1.0, external / total), "yellow"),
                    (min(1.0, reserved / total), "cyan"),
                ),
            )
        )
        segments.extend(
            (
                (" ", "bright_black"),
                (f"free {format_mb(gpu['free_mb']):>6}", free_style),
            )
        )
        # Whole-card occupancy: what is actually on the GPU matters even when
        # none of it came from this queue.
        if width >= 120:
            segments.append(
                (
                    f"  use {format_mb(gpu['mem_used_mb'])}/{format_mb(gpu['mem_total_mb'])}",
                    "bright_black",
                )
            )
        segments.append(
            (
                f"  ext {format_mb(external)}·{source}"
                f"  res {format_mb(reserved)}"
                f"  jobs {gpu['queue_jobs']}",
                "bright_black",
            )
        )
        return segments

    def _gpu_process_lines(self, gpu: Dict[str, Any], width: int) -> List[str]:
        """Name what is on the card, so a full GPU explains itself.

        Unmanaged processes are the interesting ones - they are why a job is
        waiting - so the compact view lists only those.
        """
        if self.proc_view == "off":
            return []
        processes = gpu.get("processes") or []
        if gpu.get("attribution") == "blind":
            lines = [
                self._style(
                    fit_display(
                        f"        · ~{format_mb(gpu['external_mem_mb'])} of "
                        f"{format_mb(gpu['mem_used_mb'])} in use, this driver "
                        "reports no per-process memory",
                        width,
                    ),
                    "yellow",
                )
            ]
            for process in processes:
                entry = GpuProcess.from_dict(process)
                lines.append(
                    self._compose(
                        [
                            (f"        · pid {entry.pid}  ", "bright_black"),
                            (f"{fit_display(entry.name or '-', 26):<26}  ", None),
                            (entry.owner if entry.managed else "unmanaged",
                             "cyan" if entry.managed else "yellow"),
                        ],
                        width,
                    )
                )
            return lines
        if self.proc_view == "summary":
            processes = [item for item in processes if not item.get("managed")]
            processes = processes[:PROC_SUMMARY_LIMIT]
        lines = []
        for process in processes:
            entry = GpuProcess.from_dict(process)
            owner = entry.owner if entry.managed else "unmanaged"
            lines.append(
                self._compose(
                    [
                        (f"        · {format_mb(entry.mem_mb):>6}  ", "bright_black"),
                        (f"{fit_display(entry.name or str(entry.pid), 26):<26}  ", None),
                        (f"{fit_display(entry.username or '-', 10):<10}  ", "bright_black"),
                        (owner, "cyan" if entry.managed else "yellow"),
                    ],
                    width,
                )
            )
        return lines

    @staticmethod
    def _cell(text: str, size: int, *, right: bool) -> str:
        """One fixed-width column cell plus its separator space."""
        text = fit_display(text, size)
        pad = " " * (size - display_width(text))
        return (pad + text if right else text + pad) + " "

    def _job_cell_style(self, job: Dict[str, Any], column: str) -> Optional[str]:
        """tokscale-style colouring: each column keeps one colour, and rows
        that are already finished fade out so the live ones carry the eye."""
        if column == "ST":
            return STATE_STYLE.get(display_state(job))
        if job["state"] not in ("pending", "running"):
            if column == "RC" and job.get("exit_code"):
                return "red"
            return "bright_black"
        return {
            "PRI": "yellow" if job.get("priority") else "bright_black",
            "NODE": "bright_black",
            "GPU": "bright_black",
            "VRAM": "green",
            "USED": "magenta",
            "TIME": "bright_black",
            "RC": "bright_black",
        }.get(column)

    def _job_lines(self, width: int, height: int) -> List[str]:
        self._job_line_targets = {}
        self._job_header_regions = []
        order = "▲" if self.sort_reverse else "▼"
        scope = self._scope_label()
        scope_text = f" · {self._scope_description()}" if scope else ""
        title = f"─ JOBS ({len(self.jobs)}){scope_text} ── "
        sort_label = f"sort {self.sort_key}{order}"
        self._job_header_regions.append(
            (
                0,
                display_width(title),
                display_width(title) + display_width(sort_label) - 1,
                ("sort", None),
            )
        )
        rule = title + sort_label + " "
        lines = [
            self._compose(
                [
                    (title, "bright_black"),
                    (sort_label, "cyan"),
                    (" " + "─" * max(0, width - display_width(rule) - 1), "bright_black"),
                ],
                width,
            )
        ]
        columns = list(JOB_COLUMNS)
        if width < 100:
            columns = [column for column in columns if column[0] not in ("USED", "NODE")]
        used = sum(size + 1 for _, size, _ in columns)
        command_width = max(10, width - used - 2)
        # A job that reports its own progress is being watched for exactly that,
        # so the last column names whichever of the two it will show.
        any_progress = any(job.get("progress") for job in self.jobs)
        last_column = "PROGRESS / COMMAND" if any_progress else "COMMAND"
        header_segments: List[Tuple[str, Optional[str]]] = [(" ", None)]
        column_start = 1
        for name, size, sort_key in columns:
            label = f"{name}{order}" if sort_key and sort_key == self.sort_key else name
            if sort_key:
                self._job_header_regions.append(
                    (1, column_start, column_start + size, ("sort", sort_key))
                )
            header_segments.append(
                (
                    self._cell(label, size, right=name in RIGHT_ALIGNED_COLUMNS),
                    "cyan" if sort_key == self.sort_key else "bright_black",
                )
            )
            column_start += size + 1
        header_segments.append((last_column, "bright_black"))
        lines.append(self._compose(header_segments, width))

        if not self.jobs:
            empty = (
                f"  (no jobs {self._scope_description()})"
                if scope
                else "  (no jobs match this filter)"
            )
            lines.append(self._style(fit_display(empty, width), "bright_black"))
            return lines

        # Reserve a row for the hidden-count indicator when the table cannot
        # show every job. This keeps the pane within its assigned height.
        row_budget = max(1, height - 2)
        rows = max(1, height - 3) if len(self.jobs) > row_budget else row_budget
        start = max(0, min(self.job_index - rows // 2, len(self.jobs) - rows))
        start = max(0, start)
        for position in range(start, min(len(self.jobs), start + rows)):
            job = self.jobs[position]
            values = {
                "ID": str(job["id"]),
                "ST": STATE_BADGE.get(display_state(job), display_state(job)),
                "PRI": str(job.get("priority") or 0),
                "NAME": job["name"] or "-",
                "NODE": job["node"] or job["node_constraint"] or "-",
                "GPU": ",".join(str(i) for i in job["gpu_ids"]) or "-",
                "VRAM": format_mb(job["vram_mb"]) if job["vram_mb"] else "-",
                "USED": format_mb(job["gpu_mem_mb"]) if job.get("gpu_mem_mb") else "-",
                "TIME": format_duration(job["elapsed_s"])
                if job.get("elapsed_s") is not None
                else "-",
                "RC": "-" if job["exit_code"] is None else str(job["exit_code"]),
            }
            # What the job says about itself beats the command line it was
            # started with; `▸` marks which one is on screen.
            progress = job.get("progress")
            if progress:
                tail, tail_style = f"▸ {' '.join(progress.split())}", "cyan"
            else:
                tail = " ".join(job["command"].split())
                tail_style = (
                    "bright_black"
                    if job["state"] not in ("pending", "running")
                    else None
                )
            current = position == self.job_index
            selected = self.focus == "jobs" and current
            # Full-size triangles in the accent colour: the small ▸/▾ pair is
            # indistinguishable at ordinary font sizes.
            marker = ("▼" if self.show_detail else "▶") if current else " "
            segments: List[Tuple[str, Optional[str]]] = [(marker, "cyan")]
            segments.extend(
                (
                    self._cell(values[name], size, right=name in RIGHT_ALIGNED_COLUMNS),
                    self._job_cell_style(job, name),
                )
                for name, size, _ in columns
            )
            segments.append((fit_display(tail, command_width), tail_style))
            lines.append(self._compose(segments, width, highlight=selected))
            self._job_line_targets[len(lines) - 1] = position
        if start + rows < len(self.jobs):
            lines.append(
                self._style(f"  … {len(self.jobs) - start - rows} more", "bright_black")
            )
        return lines

    def _sync_detail_selection(self) -> None:
        job = self.selected_job()
        job_id = job["id"] if job else None
        if job_id != self._detail_job_id:
            self._detail_job_id = job_id
            self.detail_page = 0
            self.detail_pages = 1
            self.log_offset = 0

    def _collapsed_detail_line(self, width: int) -> Optional[str]:
        job = self.selected_job()
        if job is None:
            return None
        title = f"─ ▶ JOB {job['id']} DETAIL HIDDEN · Enter to show "
        return self._style(
            title + "─" * max(0, width - display_width(title)),
            "bright_black",
        )

    def _detail_lines(self, state: Dict[str, Any], width: int, height: int) -> List[str]:
        job = self.selected_job()
        if job is None or height < 3:
            return []
        if self.show_log:
            self.detail_page = 0
            self.detail_pages = 1
            text = state["log_text"] if state["log_job"] == job["id"] else ""
            if not text:
                self.log_offset = 0
                self._log_max_offset = 0
                title = f"─ LOG job {job['id']} [tail] "
                lines = [
                    self._style(
                        title + "─" * max(0, width - len(title)),
                        "bright_black",
                    )
                ]
                lines.append(self._style("  (fetching…)", "bright_black"))
                return lines
            log_lines = text.rstrip("\n").splitlines()
            self._log_page_height = max(1, height - 1)
            self._log_max_offset = max(0, len(log_lines) - self._log_page_height)
            self.log_offset = max(0, min(self.log_offset, self._log_max_offset))
            title = f"─ LOG job {job['id']}"
            if self.log_offset:
                title += f" [{self.log_offset} line(s) above tail]"
            else:
                title += " [tail]"
            title += " "
            lines = [
                self._style(
                    title + "─" * max(0, width - len(title)),
                    "bright_black",
                )
            ]
            end = len(log_lines) - self.log_offset
            start = max(0, end - self._log_page_height)
            body = log_lines[start:end]
            lines.extend("  " + fit_display(line, width - 2) for line in body)
            return lines

        content: List[str] = []
        pieces = [
            f"name {job['name'] or '-'}",
            f"state {display_state(job)}",
            f"request gpus={job['gpus']} vram={format_mb(job['vram_mb'])}",
            f"node {job['node'] or job['node_constraint'] or 'any'}",
            f"pid {job['remote_pid'] or '-'}",
            f"submitter {job['submitter'] or '-'}",
        ]
        content.extend(self._wrap_plain("  " + "   ".join(pieces), width))
        content.extend(self._field_lines("cmd", job["command"], width))
        if job.get("workdir"):
            content.extend(self._field_lines("cwd", job["workdir"], width))
        if job.get("lane"):
            queued = [item for item in self._flow_job_map(self._snapshot or {}).values()
                      if item.get("lane") == job["lane"] and item.get("state") == "pending"]
            self._sort_jobs_in_lane(queued)
            ids = [item["id"] for item in queued]
            position = f" · queued position {ids.index(job['id']) + 1}/{len(ids)}" if job["id"] in ids else ""
            content.extend(self._field_lines("lane", job["lane"] + position, width))
        dependencies = self._flow_job_map(self._snapshot or {})
        for field, label in (("depends_on", "after"), ("depends_any", "any")):
            for dep_id in job.get(field) or []:
                upstream = dependencies.get(int(dep_id))
                dep_state = display_state(upstream) if upstream else "not in snapshot"
                requirement = "success" if field == "depends_on" else "any terminal result"
                content.extend(self._field_lines(label, f"#{dep_id} · {dep_state} · requires {requirement}", width))
        if job.get("progress"):
            content.extend(
                self._field_lines(
                    "live",
                    " ".join(job["progress"].split()),
                    width,
                    style="cyan",
                )
            )
        if job.get("held_reason"):
            content.extend(
                self._field_lines(
                    "held",
                    f"{job['held_reason']} — nvidb job release {job['id']}",
                    width,
                    style="magenta",
                )
            )
        if job.get("notes"):
            content.extend(self._field_lines("note", job["notes"], width))
        if job.get("last_error"):
            content.extend(
                self._field_lines("err", job["last_error"], width, style="red")
            )
        if job.get("result") is not None:
            import json as _json

            content.extend(
                self._field_lines(
                    "out",
                    _json.dumps(job["result"], ensure_ascii=False),
                    width,
                )
            )

        page_height = max(1, height - 1)
        self.detail_pages = max(1, (len(content) + page_height - 1) // page_height)
        self.detail_page = max(0, min(self.detail_page, self.detail_pages - 1))
        page = self.detail_page + 1
        title = f"─ ▼ JOB {job['id']} DETAIL"
        if self.detail_pages > 1:
            title += f" [{page}/{self.detail_pages} · [/] page]"
        title += " "
        lines = [
            self._style(
                title + "─" * max(0, width - display_width(title)),
                "bright_black",
            )
        ]
        start = self.detail_page * page_height
        lines.extend(content[start : start + page_height])
        return lines

    def _register_job_header_regions(
        self, job_start: int, shown: int, width: int
    ) -> None:
        """Make the job pane's sort label and column headers clickable."""
        for row_offset, start, end, target in self._job_header_regions:
            if row_offset < shown and start < width:
                self._click_regions.append(
                    (job_start + row_offset, start, min(end, width - 1), target)
                )

    def _control_lines(
        self,
        controls,
        width: int,
        *,
        max_lines: Optional[int] = None,
    ):
        """Render a wrapping action bar and retain each button's hit box."""
        lines: List[str] = []
        regions: List[Tuple[int, int, int, Tuple[str, Any]]] = []
        segments = []
        column = 0

        def finish_line() -> None:
            nonlocal segments, column
            if segments:
                lines.append(self._compose(segments, width))
            segments = []
            column = 0

        for label, action, value, style in controls:
            token = f"[{label}]"
            token_width = display_width(token)
            # Buttons sit on one line separated by a muted dot, the way
            # tokscale separates its footer hints.
            separator = " · " if segments else " "
            if segments and column + display_width(separator) + token_width > width:
                finish_line()
                if max_lines is not None and len(lines) >= max_lines:
                    break
                separator = " "
            start = column + display_width(separator)
            available = max(1, width - start)
            shown = fit_display(token, available)
            shown_width = display_width(shown)
            segments.extend(((separator, "bright_black"), (shown, style)))
            if shown_width:
                regions.append(
                    (
                        len(lines),
                        start,
                        start + shown_width - 1,
                        (action, value),
                    )
                )
            column = start + shown_width
        if max_lines is None or len(lines) < max_lines:
            finish_line()
        return lines, regions

    def _footer_control(self, command: Command, state: Dict[str, Any]):
        """Return the current label and target for one registered command."""
        if command.footer_order is None or self.focus not in command.contexts:
            return None
        job = self.selected_job()
        node = self.selected_node()
        name = command.name
        label = None
        style = "bright_black"
        value = command.value

        if name == "job_detail" and job is not None:
            label = f"Enter {'hide' if self.show_detail else 'show'} detail"
            style = "cyan" if self.show_detail else "bright_black"
        elif name == "cancel" and job is not None and job["state"] in (
            "pending",
            "running",
        ):
            confirming = bool(
                self.pending_confirm
                and self.pending_confirm[0] == "cancel"
                and self.pending_confirm[1] == job["id"]
            )
            label = "c confirm cancel" if confirming else "c cancel"
            style = "red" if confirming else "yellow"
        elif name == "requeue" and job is not None and job["state"] in (
            "completed",
            "failed",
            "cancelled",
            "timeout",
            "lost",
        ):
            label, style = "r requeue", "yellow"
        elif name == "log" and job is not None:
            label = f"L log:{'on' if self.show_log else 'off'}"
            style = "cyan" if self.show_log else "bright_black"
        elif name == "priority_up" and job is not None and job["state"] in (
            "pending",
            "running",
        ):
            label, style = f"+ pri:{job.get('priority') or 0}", "cyan"
        elif name == "priority_down" and job is not None and job["state"] in (
            "pending",
            "running",
        ):
            label, style = "-", "cyan"
        elif name in ("queue_up", "queue_down") and job is not None and job[
            "state"
        ] == "pending":
            label, style = (
                ("K ▲queue", "cyan") if value < 0 else ("J ▼", "cyan")
            )
        elif name == "node_toggle" and node is not None:
            node_name = fit_display(node["name"], 12)
            label = f"d {'resume' if not node['enabled'] else 'drain'}:{node_name}"
            style = "green" if not node["enabled"] else "yellow"
        elif name == "scope_all" and self._scope_label() is not None:
            label, style = "x all jobs", "cyan"
        elif name == "switch_pane":
            label, style = (
                f"Tab {'nodes' if self.focus == 'jobs' else 'jobs'}",
                "cyan",
            )
        elif name == "help":
            label = "? help"
        elif name == "quit":
            label = "q quit"
        elif name == "filter":
            label = f"f filter:{self.filter}"
        elif name == "sort":
            label = f"s sort:{self.sort_key}{'▲' if self.sort_reverse else '▼'}"
        elif name == "resource_view":
            label = f"v view:{self.resource_view}"
            style = "cyan" if self.resource_view == "flow" else "bright_black"
        elif name == "process_view":
            label = f"p procs:{self.proc_view}"
        elif name == "tick":
            label = "t tick"
        elif name == "auto_tick":
            label = f"a auto:{'on' if state['auto_tick'] else 'off'}"
            value = not state["auto_tick"]
            style = "cyan" if state["auto_tick"] else "yellow"
        elif name == "theme":
            label = f"T theme:{self.theme}"
        elif name == "acknowledge":
            alerts = (state.get("snapshot") or {}).get("alerts") or []
            if alerts:
                label, style = f"A ack:{len(alerts)}", "red"

        if label is None:
            return None
        return (label, command.action, value, style)

    def _footer_lines(
        self,
        state: Dict[str, Any],
        width: int,
        *,
        max_rows: int = 2,
    ) -> List[str]:
        """Render only actions that apply to the focused pane.

        The bar is capped so controls cannot push the selected data off a
        short screen.  Less common actions remain documented in ``?`` help.
        """
        lines: List[str] = []
        self._footer_regions = []
        if self.pending_confirm:
            action, job_id, _ = self.pending_confirm
            message = (
                f" press {action[0]} again or click confirm to {action} job "
                f"{job_id} (Esc cancels)"
            )
            lines.append(self._style(fit_display(message, width), "yellow"))
        elif state.get("error"):
            lines.append(
                self._style(fit_display(f" ! {state['error']}", width), "red")
            )
        elif state.get("notice"):
            message, style = state["notice"]
            lines.append(self._style(fit_display(f" {message}", width), style))

        remaining_rows = max(0, max_rows - len(lines))
        if not remaining_rows:
            return lines[:max_rows]
        controls = [
            control
            for command in sorted(
                COMMANDS,
                key=lambda item: item.footer_order or 10_000,
            )
            if (control := self._footer_control(command, state)) is not None
        ]
        control_offset = len(lines)
        control_lines, control_regions = self._control_lines(
            controls,
            width,
            max_lines=remaining_rows,
        )
        lines.extend(control_lines)
        self._footer_regions = [
            (line + control_offset, start, end, target)
            for line, start, end, target in control_regions
        ]
        return lines

    def _help_lines(self, width: int, height: Optional[int] = None) -> List[str]:
        rows = [
            (command.help_key, command.description)
            for command in COMMANDS
            if self.focus in command.contexts
            and command.help_key is not None
            and command.description is not None
        ]
        if self.focus == "jobs":
            rows.append(("Mouse job", "Select a job; click it again for detail"))
        else:
            rows.extend(
                [
                    ("Mouse server", "Show jobs running on that server"),
                    ("Mouse GPU", "Show jobs running on that GPU"),
                    ("Mouse pool/+N", "Open the complete queued task list"),
                    ("GPU bar", "amber: others · teal: queue · dim: free"),
                ]
            )

        title = self._style("─ HELP " + "─" * max(0, width - 7), "bright_black")
        body = [line for key, description in rows
                for line in wrap_display(f"  {key:<16}{description}", width)]
        self._help_page_size = max(1, (height - 2) if height is not None else len(body))
        self._help_max_offset = max(0, len(body) - self._help_page_size)
        self.help_offset = max(0, min(self.help_offset, self._help_max_offset))
        end = min(len(body), self.help_offset + self._help_page_size)
        close = self._style(fit_display(
            f" {self.help_offset + 1}-{end}/{len(body)} · j/k, PgUp/PgDn scroll · ?/Esc close", width
        ), "bright_black")
        return [title, *body[self.help_offset:end], close][:height]

    def _scroll_help(self, delta: int) -> None:
        self.help_offset = max(0, min(self.help_offset + delta, self._help_max_offset))

    def _resource_viewport(
        self,
        lines: List[str],
        budget: int,
        width: int,
    ) -> List[str]:
        """Scroll every resource row, or follow the selected node/GPU."""
        budget = max(0, budget)
        self._resource_content_size = max(0, len(lines) - 1)
        self._resource_page_size = max(1, budget - 2)
        if len(lines) <= budget:
            self._resource_start = 0
            self._resource_scroll = None
            return lines
        if budget == 0:
            self._node_line_targets = {}
            self._node_gpu_regions = []
            self._node_header_regions = []
            return []
        if budget == 1:
            self._node_line_targets = {}
            self._node_gpu_regions = []
            self._node_header_regions = [
                region for region in self._node_header_regions if region[0] == 0
            ]
            return lines[:1]

        content_size = len(lines) - 1
        selected_rows = [
            row
            for row, position in self._node_line_targets.items()
            if position == self.node_index and row > 0
        ]
        gpu_rows = [
            row for row, _start, _end, target in self._node_gpu_regions
            if target == ("gpu", (self.node_index, self._cursor_gpu))
        ]
        if gpu_rows:
            selected_rows = gpu_rows
        anchor = (min(selected_rows) - 1) if selected_rows else 0
        data_rows = self._resource_page_size
        requested = (
            anchor - data_rows // 2
            if self._resource_scroll is None else self._resource_scroll
        )
        start = max(0, min(requested, content_size - data_rows))
        self._resource_start = start
        end = min(content_size, start + data_rows)

        visible: List[str] = [lines[0]]
        row_map = {0: 0}
        for content_index in range(start, end):
            old_row = content_index + 1
            row_map[old_row] = len(visible)
            visible.append(lines[old_row])
        if budget >= 3:
            hidden = content_size - end
            hint = (
                "press p to collapse; PgUp/PgDn scroll"
                if self.resource_view == "servers" and self.proc_view == "all"
                else "PgUp/PgDn or wheel"
            )
            visible.append(
                self._style(
                    fit_display(
                        f"  ↑ {start} above · ↓ {hidden} more line(s), {hint}", width
                    ),
                    "bright_black",
                )
            )

        self._node_line_targets = {
            row_map[row]: position
            for row, position in self._node_line_targets.items()
            if row in row_map
        }
        self._node_gpu_regions = [
            (row_map[row], start_col, end_col, target)
            for row, start_col, end_col, target in self._node_gpu_regions
            if row in row_map
        ]
        self._node_header_regions = [
            (row_map[row], start_col, end_col, target)
            for row, start_col, end_col, target in self._node_header_regions
            if row in row_map
        ]
        return visible

    def render(self, state: Dict[str, Any]) -> str:
        """The whole frame as one string; tests read this, the loop paints
        only what changed between frames."""
        return "\n".join(self._frame_lines(state))

    def _frame_lines(self, state: Dict[str, Any]) -> List[str]:
        width = max(1, self.term.width or 100)
        height = max(2, self.term.height or 30)
        layout = LayoutProfile.for_terminal(width, height)
        self._click_regions = []
        self._row_targets = {}
        self._alert_targets = {}
        snapshot = state["snapshot"]
        self._snapshot = snapshot
        if snapshot is None:
            message = state.get("error") or "connecting to nodes…"
            return [fit_display(f"  {message}", width)]

        self.nodes = snapshot["nodes"]
        self._reanchor_node_scope()
        if self._cursor_gpu not in {
            gpu["index"] for gpu in (self.selected_node() or {}).get("gpus", [])
        }:
            self._cursor_gpu = None
        self.jobs = self._visible_jobs(snapshot)
        self._reanchor_selection()
        self._sync_detail_selection()

        # Leave the bottom row free so writing the last line cannot scroll the
        # screen and shear the frame.
        usable = height - 1
        header_lines = self._header_lines(state, width)
        lines: List[str] = list(header_lines)
        for row, start, end, target in self._header_regions:
            if row < len(header_lines) and start < width:
                self._click_regions.append(
                    (row, start, min(end, width - 1), target)
                )

        alert_start = len(lines)
        alert_lines = self._alert_lines(snapshot, width)
        lines.extend(alert_lines)
        for relative_row, target in self._alert_targets.items():
            if relative_row < len(alert_lines):
                self._row_targets[alert_start + relative_row] = target

        if self.show_help:
            help_start = len(lines)
            help_lines = self._help_lines(width, max(1, usable - help_start))
            lines.extend(help_lines)
            lines = lines[:usable]
            if help_start < len(lines):
                help_end = len(lines) - 1
                for row in range(help_start, help_end + 1):
                    self._row_targets[row] = ("close_help", None)
        else:
            footer = self._footer_lines(
                state,
                width,
                max_rows=layout.footer_rows,
            )
            body_height = max(0, usable - len(footer))
            pane_rows = max(0, body_height - len(lines))

            node_lines = (
                self._flow_lines(width)
                if self.resource_view == "flow"
                else self._node_lines(width)
            )
            # A useful job table needs a title, a header, and at least one row;
            # its collapsed detail summary takes one more.  Opening detail in
            # compact mode temporarily gives those rows back to the job pane.
            job_reserve = 4 if self.jobs else 3
            if self.show_detail and self.jobs:
                job_reserve = 6
            resource_room = max(0, pane_rows - job_reserve)
            desired_resource_rows = min(
                layout.resource_limit,
                max(2, int(pane_rows * layout.resource_fraction)),
            )
            if layout.name == "compact" and self.show_detail:
                desired_resource_rows = 0
            node_budget = min(resource_room, desired_resource_rows)
            node_lines = self._resource_viewport(node_lines, node_budget, width)
            node_start = len(lines)
            lines.extend(node_lines)
            if node_lines:
                node_end = node_start + len(node_lines) - 1
                for row in range(node_start, node_end + 1):
                    self._row_targets[row] = ("pane", "nodes")
                for relative_row, position in self._node_line_targets.items():
                    if relative_row < len(node_lines):
                        self._row_targets[node_start + relative_row] = (
                            "node",
                            position,
                        )
                for relative_row, start, end, target in self._node_gpu_regions:
                    if relative_row < len(node_lines) and start < width:
                        self._click_regions.append(
                            (
                                node_start + relative_row,
                                start,
                                min(end, width - 1),
                                target,
                            )
                        )
                for relative_row, start, end, target in self._node_header_regions:
                    if relative_row < len(node_lines) and start < width:
                        self._click_regions.append(
                            (
                                node_start + relative_row,
                                start,
                                min(end, width - 1),
                                target,
                            )
                        )
            available = max(0, body_height - len(lines))
            if self.show_detail and self.jobs:
                minimum_detail = min(8, max(3, available - 4))
                job_height = min(
                    max(4, len(self.jobs) + 2),
                    max(4, available - minimum_detail),
                )
                job_lines = self._job_lines(width, job_height)
                # At least three rows are needed for a useful detail pane.
                job_lines = job_lines[: max(0, available - 3)]
                job_start = len(lines)
                lines.extend(job_lines)
                if job_lines:
                    job_end = job_start + len(job_lines) - 1
                    for row in range(job_start, job_end + 1):
                        self._row_targets[row] = ("pane", "jobs")
                    for relative_row, position in self._job_line_targets.items():
                        if relative_row < len(job_lines):
                            self._row_targets[job_start + relative_row] = (
                                "job",
                                position,
                            )
                    self._register_job_header_regions(job_start, len(job_lines), width)
                detail_height = max(0, body_height - len(lines))
                detail_start = len(lines)
                detail_lines = self._detail_lines(state, width, detail_height)
                lines.extend(detail_lines)
                if detail_lines:
                    detail_end = detail_start + len(detail_lines) - 1
                    for row in range(detail_start, detail_end + 1):
                        self._row_targets[row] = ("detail_scroll", None)
                    self._click_regions.append(
                        (detail_start, 0, width - 1, ("detail", None))
                    )
            else:
                collapsed_height = 1 if self.jobs else 0
                job_height = max(4, available - collapsed_height)
                job_lines = self._job_lines(width, job_height)
                job_lines = job_lines[: max(0, available - collapsed_height)]
                job_start = len(lines)
                lines.extend(job_lines)
                if job_lines:
                    job_end = job_start + len(job_lines) - 1
                    for row in range(job_start, job_end + 1):
                        self._row_targets[row] = ("pane", "jobs")
                    for relative_row, position in self._job_line_targets.items():
                        if relative_row < len(job_lines):
                            self._row_targets[job_start + relative_row] = (
                                "job",
                                position,
                            )
                    self._register_job_header_regions(job_start, len(job_lines), width)
                collapsed = self._collapsed_detail_line(width)
                if collapsed:
                    collapsed_row = len(lines)
                    lines.append(collapsed)
                    self._row_targets[collapsed_row] = ("detail", None)
            # The footer holds the keybindings, so it is reserved rather than
            # left to whatever space happens to remain.
            lines = lines[:body_height]
            lines.extend([""] * (body_height - len(lines)))
            footer_start = len(lines)
            lines.extend(footer)
            for relative_row, start, end, target in self._footer_regions:
                if relative_row < len(footer) and start < width:
                    self._click_regions.append(
                        (
                            footer_start + relative_row,
                            start,
                            min(end, width - 1),
                            target,
                        )
                    )

        self._click_regions = [
            region for region in self._click_regions if region[0] < len(lines)
        ]
        self._row_targets = {
            row: target for row, target in self._row_targets.items() if row < len(lines)
        }
        return lines

    def _request_log(self, request: Optional[Tuple[int, str]]) -> None:
        """Forward a log request to the worker, skipping unchanged ones.

        The render loop re-asserts the current selection every frame; without
        this memo each frame would take the worker's state lock to say
        nothing. The worker keeps its own `!=` guard for other callers.
        """
        if request == self._last_log_request:
            return
        self._last_log_request = request
        self.worker.set_log_request(request)

    # --- input ------------------------------------------------------------

    def _scroll_resources(self, delta: int) -> None:
        self._resource_scroll = max(
            0, min(
                self._resource_start + delta,
                self._resource_content_size - self._resource_page_size,
            ),
        )

    def _move_gpu_cursor(self, delta: int) -> None:
        node = self.selected_node()
        ids = [int(gpu["index"]) for gpu in (node or {}).get("gpus", [])]
        if not ids:
            self._cursor_gpu = None
            return
        current = ids.index(self._cursor_gpu) if self._cursor_gpu in ids else (
            -1 if delta > 0 else len(ids)
        )
        self._cursor_gpu = ids[max(0, min(current + delta, len(ids) - 1))]
        self._resource_scroll = None

    def _move(self, delta: int) -> None:
        if self.focus == "nodes":
            self._resource_scroll = None
            self._cursor_gpu = None
            if self.nodes:
                previous = self.node_index
                self.node_index = max(
                    0,
                    min(self.node_index + delta, len(self.nodes) - 1),
                )
                # Once a resource scope is active, keyboard/wheel navigation
                # carries it to the newly selected server.  Before that, the
                # node cursor remains harmless until Enter or a click applies
                # a scope.
                if (
                    self.node_index != previous
                    and self.job_scope_node is not None
                ):
                    self._set_job_scope(self.nodes[self.node_index]["name"])
        elif self.jobs:
            previous = self.job_index
            self._set_job_index(self.job_index + delta)
            if self.job_index != previous:
                self.detail_page = 0
                self.log_offset = 0
                self.pending_confirm = None
                if self.show_log:
                    job = self.selected_job()
                    self._request_log(
                        (job["id"], "stdout") if job is not None else None
                    )

    def _toggle_detail(self) -> None:
        if self.selected_job() is None:
            return
        self.show_detail = not self.show_detail
        if self.show_detail:
            self.detail_page = 0
            self.log_offset = 0
        elif self.show_log:
            self._request_log(None)

    def _toggle_log(self) -> None:
        job = self.selected_job()
        if job is None:
            return
        self.show_log = not self.show_log
        self.show_detail = True
        self.detail_page = 0
        self.log_offset = 0
        self._request_log(
            (job["id"], "stdout") if self.show_log else None
        )

    def _scroll_detail(self, delta: int, *, page: bool = False) -> bool:
        if not self.show_detail or not delta:
            return False
        if self.show_log:
            amount = self._log_page_height if page else 3
            previous = self.log_offset
            if delta < 0:
                self.log_offset = min(self._log_max_offset, self.log_offset + amount)
            else:
                self.log_offset = max(0, self.log_offset - amount)
            return self.log_offset != previous
        previous = self.detail_page
        self.detail_page = max(
            0,
            min(self.detail_pages - 1, self.detail_page + delta),
        )
        return self.detail_page != previous

    def _select_job_position(self, position: int, *, toggle_current: bool) -> None:
        if not (0 <= position < len(self.jobs)):
            return
        current = self.focus == "jobs" and position == self.job_index
        changed = position != self.job_index
        self.focus = "jobs"
        self._set_job_index(position)
        if changed:
            self.detail_page = 0
            self.log_offset = 0
            self.pending_confirm = None
            if self.show_log:
                job = self.selected_job()
                self._request_log(
                    (job["id"], "stdout") if job is not None else None
                )
        elif current and toggle_current:
            self._toggle_detail()

    def _select_node_position(self, position: int) -> None:
        if not (0 <= position < len(self.nodes)):
            return
        self.focus = "nodes"
        self.node_index = position
        self._cursor_gpu = None
        self._resource_scroll = None
        self._set_job_scope(self.nodes[position]["name"])

    def _select_gpu(self, position: int, gpu_index: int) -> None:
        if not (0 <= position < len(self.nodes)):
            return
        node = self.nodes[position]
        if not any(
            int(gpu.get("index")) == int(gpu_index)
            for gpu in (node.get("gpus") or [])
        ):
            return
        self.focus = "nodes"
        self.node_index = position
        self._cursor_gpu = gpu_index
        self._resource_scroll = None
        self._set_job_scope(node["name"], int(gpu_index))

    def _select_job_id(self, job_id: int, *, show_log: bool = False) -> bool:
        snapshot = self._snapshot or {}
        pool = list(snapshot.get("jobs") or []) + list(snapshot.get("recent") or [])
        if not any(job.get("id") == job_id for job in pool):
            self.worker.set_notice(f"job {job_id} is no longer in the snapshot", "yellow")
            return False
        self.job_scope_node = None
        self.job_scope_gpu = None
        self.job_scope_pool = None
        self.filter = "all"
        self.jobs = self._visible_jobs(snapshot)
        for position, job in enumerate(self.jobs):
            if job.get("id") == job_id:
                self._select_job_position(position, toggle_current=False)
                if show_log:
                    self.show_log = True
                    self.show_detail = True
                    self.log_offset = 0
                    self._request_log((job_id, "stdout"))
                return True
        return False

    def _select_node_name(self, name: str) -> bool:
        for position, node in enumerate(self.nodes):
            if node.get("name") == name:
                self._select_node_position(position)
                return True
        self.worker.set_notice(f"node {name} is no longer in the snapshot", "yellow")
        return False

    def _confirm(self, action: str, job_id: int) -> bool:
        """Two-step confirmation: the same key twice within a few seconds."""
        now = time.time()
        pending = self.pending_confirm
        if (
            pending
            and pending[0] == action
            and pending[1] == job_id
            and now - pending[2] < CONFIRM_SECONDS
        ):
            self.pending_confirm = None
            return True
        self.pending_confirm = (action, job_id, now)
        return False

    def _activate(self, action: str, value=None) -> bool:
        """Run one named UI action. False means the caller should quit."""
        if action == "quit":
            return False
        if action in {"detail", "log", "cancel", "requeue", "priority", "move"}:
            if self.focus != "jobs":
                return True
        if action == "node_toggle" and self.focus != "nodes":
            return True
        if action == "help":
            self.show_help = True
            self.help_offset = 0
        elif action == "close_help":
            self.show_help = False
        elif action == "switch_pane":
            self.focus = "nodes" if self.focus == "jobs" else "jobs"
        elif action == "detail":
            self._toggle_detail()
        elif action == "log":
            self._toggle_log()
        elif action == "filter":
            self.job_scope_node = None
            self.job_scope_gpu = None
            self.job_scope_pool = None
            if value in FILTERS:
                self.filter = value
            else:
                self.filter = FILTERS[(FILTERS.index(self.filter) + 1) % len(FILTERS)]
            self.job_index = 0
            self._selected_job_id = None
            self.detail_page = 0
            self.log_offset = 0
            self.pending_confirm = None
            if self._snapshot is not None:
                self.jobs = self._visible_jobs(self._snapshot)
            if self.show_log:
                job = self.selected_job()
                self._request_log(
                    (job["id"], "stdout") if job is not None else None
                )
        elif action == "sort":
            if value in SORT_KEYS:
                if value == self.sort_key:
                    self.sort_reverse = not self.sort_reverse
                else:
                    self.sort_key = value
                    self.sort_reverse = False
            else:
                self.sort_key = SORT_KEYS[
                    (SORT_KEYS.index(self.sort_key) + 1) % len(SORT_KEYS)
                ]
                self.sort_reverse = False
            if self._snapshot is not None:
                self.jobs = self._visible_jobs(self._snapshot)
                self._reanchor_selection()
        elif action == "priority":
            job = self.selected_job()
            if job is not None:
                if job["state"] in ("pending", "running"):
                    self.worker.post("priority", job["id"], int(value or 0))
                else:
                    self.worker.set_notice(
                        "finished jobs have no priority to change", "yellow"
                    )
        elif action == "move":
            job = self.selected_job()
            if job is not None:
                if job["state"] == "pending":
                    # Reordering only reads sensibly in dispatch order, so
                    # moving a job snaps the table back to it.
                    if self.sort_key != "queue" or self.sort_reverse:
                        self.sort_key = "queue"
                        self.sort_reverse = False
                    self.worker.post("move", job["id"], int(value or 0))
                else:
                    self.worker.set_notice(
                        "only pending jobs can be reordered", "yellow"
                    )
        elif action == "procs":
            self._resource_scroll = None
            self.proc_view = PROC_VIEWS[
                (PROC_VIEWS.index(self.proc_view) + 1) % len(PROC_VIEWS)
            ]
            # Process rows belong to the capacity view.  Switching here makes
            # `p` useful from the default flow view instead of changing hidden
            # state with no visible result.
            self.resource_view = "servers"
        elif action == "resource_view":
            self._resource_scroll = None
            self.resource_view = RESOURCE_VIEWS[
                (RESOURCE_VIEWS.index(self.resource_view) + 1)
                % len(RESOURCE_VIEWS)
            ]
        elif action == "theme":
            self.theme = THEME_ORDER[
                (THEME_ORDER.index(self.theme) + 1) % len(THEME_ORDER)
            ]
            # Persist over what is stored so the monitor TUI's own view
            # settings survive the write.
            try:
                settings = nvidb_config.load_view_settings()
                settings["theme"] = self.theme
                nvidb_config.save_view_settings(settings)
            except Exception:
                pass
        elif action == "tick":
            self.worker.post("tick")
        elif action == "auto":
            enabled = not self.worker.auto_tick if value is None else bool(value)
            self.worker.post("auto", enabled)
        elif action == "cancel":
            job = self.selected_job()
            if job and self._confirm("cancel", job["id"]):
                self.worker.post("cancel", job["id"])
        elif action == "requeue":
            job = self.selected_job()
            if job:
                self.worker.post("requeue", job["id"])
        elif action == "node_toggle":
            node = self.selected_node()
            if node:
                self.worker.post(
                    "resume" if not node["enabled"] else "drain",
                    node["name"],
                )
        elif action == "ack":
            self.worker.post("ack")
        elif action == "job_log" and value is not None:
            self._select_job_id(int(value), show_log=True)
        elif action == "node_name" and value is not None:
            self._select_node_name(str(value))
        elif action == "flow_job" and value is not None:
            job_id = int(value)
            current = self.selected_job()
            if (
                self.focus == "jobs"
                and current is not None
                and int(current["id"]) == job_id
            ):
                self._toggle_detail()
            else:
                self._select_job_id(job_id)
                self.show_detail = True
        elif action == "scope_all":
            self._clear_job_scope()
        elif action == "pool" and value is not None:
            self._set_pool_scope(str(value))
        elif action == "pane" and value in ("jobs", "nodes"):
            self.focus = value
        return True

    def handle_mouse(self, event) -> bool:
        """Handle one decoded SGR mouse event. False requests TUI exit."""
        if not self.mouse_enabled:
            return True
        row = event.row - 1
        column = event.column - 1
        target = None
        for region_row, start, end, region_target in self._click_regions:
            if region_row == row and start <= column <= end:
                target = region_target
                break
        if target is None:
            target = self._row_targets.get(row)

        if self.show_help:
            if event.is_wheel_up or event.is_wheel_down:
                self._scroll_help(-1 if event.is_wheel_up else 1)
            if event.is_left_press and target == ("close_help", None):
                self.show_help = False
            return True

        if event.is_wheel_up or event.is_wheel_down:
            delta = -1 if event.is_wheel_up else 1
            kind, value = target if target is not None else (None, None)
            if kind in ("node", "gpu", "flow_job", "pool") or (
                kind == "pane" and value == "nodes"
            ):
                self.focus = "nodes"
                self._scroll_resources(delta)
            elif kind in ("job", "sort") or (kind == "pane" and value == "jobs"):
                self.focus = "jobs"
                self._move(delta)
            elif kind in ("detail", "detail_scroll"):
                self._scroll_detail(delta)
            return True

        if not event.is_left_press or target is None:
            return True
        kind, value = target
        if kind == "job":
            self._select_job_position(int(value), toggle_current=True)
            return True
        if kind == "node":
            self._select_node_position(int(value))
            return True
        if kind == "gpu":
            position, gpu_index = value
            self._select_gpu(int(position), int(gpu_index))
            return True
        if kind == "flow_job":
            return self._activate("flow_job", value)
        if kind in {"detail", "log", "cancel", "requeue", "priority", "move"}:
            self.focus = "jobs"
        return self._activate(kind, value)

    def _dispatch_command(self, command: Command) -> bool:
        if command.action == "move_cursor":
            if self.focus == "nodes" and command.name in ("page_down", "page_up"):
                self._scroll_resources(
                    self._resource_page_size * (1 if command.value > 0 else -1)
                )
            else:
                self._move(int(command.value))
            return True
        if command.action == "gpu_cursor":
            self._move_gpu_cursor(int(command.value))
            return True
        if command.action == "move_edge":
            distance = max(len(self.nodes), len(self.jobs))
            self._move(distance if command.value > 0 else -distance)
            if self.focus == "jobs":
                self.detail_page = 0
            return True
        if command.action == "scope_node":
            if self._cursor_gpu is None:
                self._select_node_position(self.node_index)
            else:
                self._select_gpu(self.node_index, self._cursor_gpu)
            return True
        if command.action == "scroll_detail":
            if self.show_detail:
                self._scroll_detail(int(command.value), page=True)
            return True
        if command.action == "reverse_sort":
            self.sort_reverse = not self.sort_reverse
            if self._snapshot is not None:
                self.jobs = self._visible_jobs(self._snapshot)
                self._reanchor_selection()
            return True
        if command.action == "escape":
            if not self._clear_job_scope():
                self.pending_confirm = None
            return True
        return self._activate(command.action, command.value)

    def handle_key(self, key) -> bool:
        """Dispatch one key through the command registry."""
        name = key.name or ""
        text = str(key)

        if self.show_help:
            if text in ("?", "q") or name == "KEY_ESCAPE":
                self.show_help = False
            elif text == "j" or name == "KEY_DOWN":
                self._scroll_help(1)
            elif text == "k" or name == "KEY_UP":
                self._scroll_help(-1)
            elif name in ("KEY_PGDOWN", "KEY_NPAGE", "KEY_PAGEDOWN"):
                self._scroll_help(self._help_page_size)
            elif name in ("KEY_PGUP", "KEY_PPAGE", "KEY_PAGEUP"):
                self._scroll_help(-self._help_page_size)
            elif text in ("g", "G"):
                self.help_offset = 0 if text == "g" else self._help_max_offset
            return True

        command = next(
            (
                candidate
                for candidate in COMMANDS
                if candidate.matches(text, name, self.focus)
            ),
            None,
        )
        return self._dispatch_command(command) if command is not None else True

    # --- main loop --------------------------------------------------------

    def _start_mouse_reporting(self) -> None:
        self._mouse_reporting = False
        if not self.mouse_enabled or not sys.stdout.isatty():
            return
        try:
            sys.stdout.write(MOUSE_ENABLE_SEQUENCE)
            sys.stdout.flush()
            self._mouse_reporting = True
        except Exception:
            self._mouse_reporting = False

    def _stop_mouse_reporting(self) -> None:
        if not self._mouse_reporting:
            return
        self._mouse_reporting = False
        try:
            sys.stdout.write(MOUSE_DISABLE_SEQUENCE)
            sys.stdout.flush()
        except Exception:
            pass

    def run(self) -> int:
        term = self.term
        parser = MouseSequenceParser()
        screen = DiffScreen(term)
        self.worker.start()
        try:
            with term.fullscreen(), term.cbreak(), term.hidden_cursor():
                self._start_mouse_reporting()
                try:
                    running = True
                    while running:
                        state = self.worker.read_state()
                        if self.show_log and self.show_detail:
                            job = self.selected_job()
                            if job:
                                self._request_log((job["id"], "stdout"))
                        screen.paint(self._frame_lines(state))
                        key = term.inkey(timeout=0.4)
                        events, keys = parser.feed(key) if key else ([], parser.flush())
                        for event in events:
                            if not self.handle_mouse(event):
                                running = False
                                break
                        if running:
                            for pending in keys:
                                if not self.handle_key(pending):
                                    running = False
                                    break
                        if (
                            self.pending_confirm
                            and time.time() - self.pending_confirm[2] > CONFIRM_SECONDS
                        ):
                            self.pending_confirm = None
                finally:
                    self._stop_mouse_reporting()
        except KeyboardInterrupt:
            pass
        finally:
            self.worker.stop()
            self.worker.join(timeout=3)
        return 0


def run_tui(db_path=None, refresh: float = 3.0) -> int:
    from .cli import quiet_transport_logging

    # Log records written straight to the terminal would scribble over the UI.
    quiet_transport_logging()
    view_settings = nvidb_config.load_view_settings()
    return QueueTUI(
        db_path=db_path,
        refresh=refresh,
        mouse_enabled=view_settings["mouse"],
        theme=view_settings.get("theme", "classic"),
    ).run()
