"""Interactive terminal UI for the nvidb job queue.

The screen is three stacked panes - node capacity, the job table, and a detail
or log view for the selected job - drawn with `blessed`.

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
from typing import Any, Dict, List, Optional, Tuple

from blessed import Terminal

from .. import config as nvidb_config
from ..mouse import (
    DISABLE_SEQUENCE as MOUSE_DISABLE_SEQUENCE,
    ENABLE_SEQUENCE as MOUSE_ENABLE_SEQUENCE,
    MouseSequenceParser,
)
from . import db as dbm
from .model import (
    GpuProcess,
    age_seconds,
    display_width,
    fit_display,
    format_duration,
    format_mb,
    pad_display,
)
from .scheduler import Scheduler

FILTERS = ("active", "all", "running", "pending", "finished")
# summary: only the processes the queue does not manage, which is what explains
# a card being full; all: those plus this queue's own jobs; off: neither.
PROC_VIEWS = ("summary", "all", "off")
PROC_SUMMARY_LIMIT = 2
FILTER_STATES = {
    "active": ("pending", "running"),
    "running": ("running",),
    "pending": ("pending",),
    "finished": ("completed", "failed", "cancelled", "timeout", "lost"),
}

# Muted tokscale-style palette. Call sites keep semantic ANSI names; this maps
# them to softer absolute tones so ordinary data stays calm and saturated
# colour is left meaning "look at this". blessed degrades each name to the
# nearest colour the terminal actually has.
PALETTE = {
    "green": "mediumseagreen",
    "yellow": "darkkhaki",
    "red": "indianred",
    "cyan": "cadetblue",
    "magenta": "rosybrown",
    "bright_blue": "steelblue",
}

# The selection band: a grey clearly lighter than a dark terminal background,
# so the cursor is findable at a glance yet still reads as a tint, not a bar.
SELECTION_BG = "gray27"

STATE_STYLE = {
    "running": "green",
    "pending": "yellow",
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
    "completed": "✓ done",
    "failed": "✗ fail",
    "cancelled": "⊘ canc",
    "timeout": "! tout",
    "lost": "? lost",
}

# "queue" is the order the scheduler will actually dispatch in - running
# first, then pending by (priority DESC, id ASC) - which is what makes the
# reorder keys make visual sense.
SORT_KEYS = ("queue", "id", "state", "pri", "name", "node", "vram", "used", "time")

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
        except Exception:
            pass

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
                dbm.set_node_enabled(scheduler.conn, action[1], False)
                self.set_notice(f"{action[1]} drained", "yellow")
            elif name == "resume":
                dbm.set_node_enabled(scheduler.conn, action[1], True)
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
                ok = scheduler.move_pending(action[1], action[2])
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
    ):
        self.term = Terminal()
        self.worker = _Worker(db_path=db_path, refresh=refresh)
        self.mouse_enabled = bool(mouse_enabled)
        self._mouse_reporting = False
        self.focus = "jobs"  # jobs | nodes
        self.job_index = 0
        self.node_index = 0
        self.filter = "active"
        self.sort_key = "queue"
        self.sort_reverse = False
        # The cursor follows a job, not a row number: a refresh or a priority
        # change may reorder the table under the selection.
        self._selected_job_id: Optional[int] = None
        self.proc_view = "summary"
        self.show_detail = True
        self.show_log = False
        self.show_help = False
        self.detail_page = 0
        self.detail_pages = 1
        self._detail_job_id: Optional[int] = None
        self.log_offset = 0
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
        self._job_line_targets: Dict[int, int] = {}
        # (row offset within the job pane, start col, end col, target).
        self._job_header_regions: List[Tuple[int, int, int, Tuple[str, Any]]] = []
        self._footer_regions: List[Tuple[int, int, int, Tuple[str, Any]]] = []

    # --- data -------------------------------------------------------------

    def _visible_jobs(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.filter == "all":
            jobs = list(snapshot["jobs"]) + list(snapshot["recent"])
        else:
            wanted = FILTER_STATES[self.filter]
            pool = list(snapshot["jobs"]) + list(snapshot["recent"])
            jobs = [job for job in pool if job["state"] in wanted]
        self._sort_jobs(jobs)
        return jobs

    def _sort_jobs(self, jobs: List[Dict[str, Any]]) -> None:
        def queue_key(job):
            # Running jobs head the table (they hold GPUs now), pending follow
            # in the exact order the scheduler would dispatch them, finished
            # jobs sink to the bottom.
            rank = {"running": 0, "pending": 1}.get(job["state"], 2)
            pri = -(job.get("priority") or 0) if job["state"] == "pending" else 0
            return (rank, pri, job["id"])

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
            "time": lambda job: (-(job.get("elapsed_s") or 0.0), job["id"]),
        }
        jobs.sort(key=key_funcs.get(self.sort_key, queue_key))
        if self.sort_reverse:
            jobs.reverse()

    def _reanchor_selection(self) -> None:
        """Put the cursor back on the job it was on before rows moved."""
        if not self.jobs:
            self.job_index = 0
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

    # --- rendering --------------------------------------------------------

    def _style(self, text: str, style: Optional[str]) -> str:
        if not style:
            return text
        formatter = getattr(self.term, PALETTE.get(style, style), None)
        return formatter(text) if callable(formatter) else text

    def _fit(self, text: str, width: int) -> str:
        return fit_display(text, width)

    @staticmethod
    def _wrap_plain(text: Any, width: int) -> List[str]:
        """Wrap plain text by terminal columns, including wide characters."""
        width = max(1, int(width))
        wrapped: List[str] = []
        logical_lines = str(text).splitlines() or [""]
        for logical_line in logical_lines:
            remaining = " ".join(logical_line.split())
            if not remaining:
                wrapped.append("")
                continue
            while display_width(remaining) > width:
                used = 0
                split_at = 0
                for index, char in enumerate(remaining):
                    char_width = display_width(char)
                    if used + char_width > width:
                        break
                    used += char_width
                    split_at = index + 1
                if split_at == 0:
                    split_at = 1
                candidate = remaining[:split_at]
                word_break = candidate.rfind(" ")
                if word_break > 0:
                    candidate = candidate[:word_break]
                    split_at = word_break + 1
                wrapped.append(candidate.rstrip())
                remaining = remaining[split_at:].lstrip()
            wrapped.append(remaining)
        return wrapped

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
            fg = PALETTE.get(style, style) if style else None
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
        meta = (
            f"tick {tick_text} · {keeper_text}{mode} · {status} · filter {self.filter} "
            f"· procs {self.proc_view} "
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

    def _node_lines(self, width: int) -> List[str]:
        self._node_line_targets = {}
        lines = [self._style("─ NODES " + "─" * max(0, width - 8), "bright_black")]
        for position, node in enumerate(self.nodes):
            node_start = len(lines)
            selected = self.focus == "nodes" and position == self.node_index
            marker = "›" if selected else " "
            state = node["state"]
            # An up node is the normal case and stays quiet; only trouble
            # (down, drain) earns colour.
            state_style = {"down": "red", "drain": "yellow"}.get(
                state, "bright_black"
            )
            label = node["name"]
            detail = f"{node['hostname'] or ''}"
            if not node["enabled"]:
                state = "drain"
                state_style = "yellow"
            head = f"{marker} {label}"
            tail = f"{detail}  "
            gap = max(
                1,
                width - display_width(head) - display_width(tail) - len(state) - 1,
            )
            lines.append(
                self._compose(
                    [
                        (head, "bold"),
                        (" " * gap, None),
                        (tail, "bright_black"),
                        (state, state_style),
                    ],
                    width,
                    highlight=selected,
                )
            )

            if state == "down" and node.get("last_error"):
                lines.append(
                    self._style(self._fit(f"    ! {node['last_error']}", width), "red")
                )
                for line_index in range(node_start, len(lines)):
                    self._node_line_targets[line_index] = position
                continue

            for gpu in node["gpus"]:
                lines.append(self._compose(self._gpu_segments(gpu, width), width))
                lines.extend(self._gpu_process_lines(gpu, width))
            for line_index in range(node_start, len(lines)):
                self._node_line_targets[line_index] = position
        return lines

    def _gpu_segments(
        self, gpu: Dict[str, Any], width: int
    ) -> List[Tuple[str, Optional[str]]]:
        """One GPU as a btop-style line: temp, util, and a memory bar whose
        segments say *whose* memory it is - foreign processes (yellow), this
        queue's reservations (cyan), free (dim)."""
        total = max(1, gpu["mem_total_mb"])
        bar_width = 20 if width >= 110 else 10
        external = max(0, gpu["external_mem_mb"])
        reserved = max(0, gpu["reserved_mb"])
        external_cells = int(round(bar_width * min(1.0, external / total)))
        reserved_cells = (
            int(round(bar_width * min(1.0, (external + reserved) / total)))
            - external_cells
        )
        reserved_cells = max(0, reserved_cells)
        free_cells = max(0, bar_width - external_cells - reserved_cells)

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
            (f"    GPU{gpu['index']} ", "bright_black"),
            (pad_display(fit_display(gpu["name"] or "-", name_width), name_width + 1), None),
            (temp_text + " ", temp_style),
            (util_text + " ", util_style),
            ("▕", "bright_black"),
            ("█" * external_cells, "yellow"),
            ("█" * reserved_cells, "cyan"),
            ("░" * free_cells, "bright_black"),
            ("▏ ", "bright_black"),
            (f"free {format_mb(gpu['free_mb']):>6}", free_style),
        ]
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
                    self._fit(
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
            return STATE_STYLE.get(job["state"])
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
        title = f"─ JOBS ({len(self.jobs)}) ── "
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
            lines.append(self._style("  (no jobs match this filter)", "bright_black"))
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
                "ST": STATE_BADGE.get(job["state"], job["state"]),
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
            segments.append((self._fit(tail, command_width), tail_style))
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
        title = f"─ ▶ JOB {job['id']} DETAIL HIDDEN — Enter to show "
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
            lines.extend("  " + self._fit(line, width - 2) for line in body)
            return lines

        content: List[str] = []
        pieces = [
            f"name {job['name'] or '-'}",
            f"state {job['state']}",
            f"request gpus={job['gpus']} vram={format_mb(job['vram_mb'])}",
            f"node {job['node'] or job['node_constraint'] or 'any'}",
            f"pid {job['remote_pid'] or '-'}",
            f"submitter {job['submitter'] or '-'}",
        ]
        content.append(self._fit("  " + "   ".join(pieces), width))
        content.append(self._fit(f"  cmd  {' '.join(job['command'].split())}", width))
        if job.get("workdir"):
            content.append(self._fit(f"  cwd  {job['workdir']}", width))
        if job.get("progress"):
            content.extend(
                self._field_lines(
                    "live",
                    " ".join(job["progress"].split()),
                    width,
                    style="cyan",
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

    def _control_lines(self, controls, width: int):
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
            needed = 1 + token_width
            if segments and column + needed > width:
                finish_line()
            prefix = " "
            start = column + display_width(prefix)
            available = max(1, width - start)
            shown = fit_display(token, available)
            shown_width = display_width(shown)
            segments.extend(((prefix, None), (shown, style)))
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
        finish_line()
        return lines, regions

    def _footer_lines(self, state: Dict[str, Any], width: int) -> List[str]:
        notice = state.get("notice")
        error = state.get("error")
        lines = []
        self._footer_regions = []
        if error:
            lines.append(self._style(self._fit(f" ! {error}", width), "red"))
        elif notice:
            message, style = notice
            lines.append(self._style(self._fit(f" {message}", width), style))
        if self.pending_confirm:
            action, job_id, _ = self.pending_confirm
            lines.append(
                self._style(
                    self._fit(
                        f" press {action[0]} again or click confirm to {action} job "
                        f"{job_id} (Esc cancels)",
                        width,
                    ),
                    "yellow",
                )
            )

        controls = []
        job = self.selected_job()
        node = self.selected_node()
        if job is not None:
            controls.extend(
                [
                    (
                        f"Enter {'hide' if self.show_detail else 'show'} detail",
                        "detail",
                        None,
                        "cyan" if self.show_detail else "bright_black",
                    ),
                    (
                        f"L log:{'on' if self.show_log else 'off'}",
                        "log",
                        None,
                        "cyan" if self.show_log else "bright_black",
                    ),
                ]
            )
            if job["state"] in ("pending", "running"):
                confirming = bool(
                    self.pending_confirm
                    and self.pending_confirm[0] == "cancel"
                    and self.pending_confirm[1] == job["id"]
                )
                controls.append(
                    (
                        "c confirm cancel" if confirming else "c cancel",
                        "cancel",
                        None,
                        "red" if confirming else "yellow",
                    )
                )
                controls.extend(
                    [
                        (f"+ pri:{job.get('priority') or 0}", "priority", 1, "cyan"),
                        ("-", "priority", -1, "cyan"),
                    ]
                )
                if job["state"] == "pending":
                    controls.extend(
                        [
                            ("K ▲queue", "move", -1, "cyan"),
                            ("J ▼", "move", 1, "cyan"),
                        ]
                    )
            elif job["state"] in ("completed", "failed", "cancelled", "timeout", "lost"):
                controls.append(("r requeue", "requeue", None, "yellow"))
        if node is not None:
            node_name = fit_display(node["name"], 12)
            controls.append(
                (
                    f"d {'resume' if not node['enabled'] else 'drain'}:{node_name}",
                    "node_toggle",
                    None,
                    "green" if not node["enabled"] else "yellow",
                )
            )

        controls.extend(
            [
                (
                    f"Tab {'nodes' if self.focus == 'jobs' else 'jobs'}",
                    "switch_pane",
                    None,
                    "cyan",
                ),
                ("t tick", "tick", None, "bright_black"),
                (
                    f"a auto:{'on' if state['auto_tick'] else 'off'}",
                    "auto",
                    not state["auto_tick"],
                    "cyan" if state["auto_tick"] else "yellow",
                ),
                (
                    f"f filter:{self.filter}",
                    "filter",
                    None,
                    "bright_black",
                ),
                (
                    f"s sort:{self.sort_key}{'▲' if self.sort_reverse else '▼'}",
                    "sort",
                    None,
                    "bright_black",
                ),
                (
                    f"p procs:{self.proc_view}",
                    "procs",
                    None,
                    "bright_black",
                ),
            ]
        )
        alerts = (state.get("snapshot") or {}).get("alerts") or []
        if alerts:
            controls.append((f"A ack:{len(alerts)}", "ack", None, "red"))
        controls.extend(
            [
                ("? help", "help", None, "bright_black"),
                ("q quit", "quit", None, "bright_black"),
            ]
        )

        control_offset = len(lines)
        control_lines, control_regions = self._control_lines(controls, width)
        lines.extend(control_lines)
        self._footer_regions = [
            (line + control_offset, start, end, target)
            for line, start, end, target in control_regions
        ]
        return lines

    def _help_lines(self, width: int) -> List[str]:
        rows = [
            ("j / k / ↑ / ↓", "Move the selection in the focused pane"),
            ("PgUp / PgDn", "Move a page at a time"),
            ("Tab", "Switch focus between the node and job panes"),
            ("Enter", "Show or hide the selected job's detail pane"),
            ("[ / ]", "Page through wrapped detail or log text"),
            ("L", "Toggle the log tail for the selected job"),
            ("c", "Cancel the selected job (press twice)"),
            ("r", "Re-queue the selected finished job"),
            ("+ / -", "Raise or lower the selected job's priority"),
            ("K / J", "Move a pending job up / down the dispatch order"),
            ("s / S", "Cycle the sort column / flip its direction"),
            ("t", "Force a scheduler tick now"),
            ("a", "Toggle automatic ticking"),
            ("f", "Cycle the job filter"),
            ("p", "GPU processes: unmanaged only / all / none"),
            ("d", "Drain or resume the selected node"),
            ("A", "Acknowledge every open alert"),
            ("Mouse click", "Select rows, activate [buttons], sort by a column header"),
            ("Mouse wheel", "Move in nodes/jobs; page detail or log text"),
            ("GPU bar", "amber: others' memory · teal: queue reservations · dim: free"),
            ("q", "Quit"),
        ]
        lines = [self._style("─ HELP " + "─" * max(0, width - 7), "bright_black")]
        for key, description in rows:
            lines.append(self._fit(f"  {key:<16}{description}", width))
        lines.append(self._style("  press ? or Esc to close", "bright_black"))
        return lines

    def render(self, state: Dict[str, Any]) -> str:
        width = max(60, self.term.width or 100)
        height = max(20, self.term.height or 30)
        self._click_regions = []
        self._row_targets = {}
        self._alert_targets = {}
        snapshot = state["snapshot"]
        self._snapshot = snapshot
        if snapshot is None:
            message = state.get("error") or "connecting to nodes…"
            return self.term.home + self.term.clear + f"\n  {message}\n"

        self.nodes = snapshot["nodes"]
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
            help_lines = self._help_lines(width)
            lines.extend(help_lines)
            lines = lines[:usable]
            if help_start < len(lines):
                help_end = len(lines) - 1
                for row in range(help_start, help_end + 1):
                    self._row_targets[row] = ("close_help", None)
        else:
            # Process listings can make the node pane arbitrarily tall, and the
            # job table is what this screen is for, so cap it at half the height.
            node_lines = self._node_lines(width)
            node_budget = max(6, usable // 2)
            if len(node_lines) > node_budget:
                hidden = len(node_lines) - node_budget + 1
                node_lines = node_lines[: node_budget - 1]
                node_lines.append(
                    self._style(f"    … {hidden} more line(s), press p", "bright_black")
                )
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
            footer = self._footer_lines(state, width)
            body_height = usable - len(footer)
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

        output = [self.term.home + self.term.clear]
        for line in lines:
            output.append(self.term.clear_eol + line + "\n")
        return "".join(output)

    # --- input ------------------------------------------------------------

    def _move(self, delta: int) -> None:
        if self.focus == "nodes":
            if self.nodes:
                self.node_index = max(0, min(self.node_index + delta, len(self.nodes) - 1))
        elif self.jobs:
            previous = self.job_index
            self._set_job_index(self.job_index + delta)
            if self.job_index != previous:
                self.detail_page = 0
                self.log_offset = 0
                self.pending_confirm = None
                if self.show_log:
                    job = self.selected_job()
                    self.worker.set_log_request(
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
            self.worker.set_log_request(None)

    def _toggle_log(self) -> None:
        job = self.selected_job()
        if job is None:
            return
        self.show_log = not self.show_log
        self.show_detail = True
        self.detail_page = 0
        self.log_offset = 0
        self.worker.set_log_request(
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
                self.worker.set_log_request(
                    (job["id"], "stdout") if job is not None else None
                )
        elif current and toggle_current:
            self._toggle_detail()

    def _select_node_position(self, position: int) -> None:
        if not (0 <= position < len(self.nodes)):
            return
        self.focus = "nodes"
        self.node_index = position

    def _select_job_id(self, job_id: int, *, show_log: bool = False) -> bool:
        snapshot = self._snapshot or {}
        pool = list(snapshot.get("jobs") or []) + list(snapshot.get("recent") or [])
        if not any(job.get("id") == job_id for job in pool):
            self.worker.set_notice(f"job {job_id} is no longer in the snapshot", "yellow")
            return False
        self.filter = "all"
        self.jobs = self._visible_jobs(snapshot)
        for position, job in enumerate(self.jobs):
            if job.get("id") == job_id:
                self._select_job_position(position, toggle_current=False)
                if show_log:
                    self.show_log = True
                    self.show_detail = True
                    self.log_offset = 0
                    self.worker.set_log_request((job_id, "stdout"))
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
        if action == "help":
            self.show_help = True
        elif action == "close_help":
            self.show_help = False
        elif action == "switch_pane":
            self.focus = "nodes" if self.focus == "jobs" else "jobs"
        elif action == "detail":
            self._toggle_detail()
        elif action == "log":
            self._toggle_log()
        elif action == "filter":
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
                self.worker.set_log_request(
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
            self.proc_view = PROC_VIEWS[
                (PROC_VIEWS.index(self.proc_view) + 1) % len(PROC_VIEWS)
            ]
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
            if event.is_left_press and target == ("close_help", None):
                self.show_help = False
            return True

        if event.is_wheel_up or event.is_wheel_down:
            delta = -1 if event.is_wheel_up else 1
            kind, value = target if target is not None else (None, None)
            if kind == "node" or (kind == "pane" and value == "nodes"):
                self.focus = "nodes"
                self._move(delta)
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
        return self._activate(kind, value)

    def handle_key(self, key) -> bool:
        """Return False to quit."""
        name = key.name or ""
        text = str(key)

        if self.show_help:
            if text in ("?", "q") or name == "KEY_ESCAPE":
                self.show_help = False
            return True

        if name == "KEY_ESCAPE":
            self.pending_confirm = None
            return True
        if text == "q":
            return self._activate("quit")
        if text == "?":
            return self._activate("help")
        if text in ("j",) or name == "KEY_DOWN":
            self._move(1)
        elif text in ("k",) or name == "KEY_UP":
            self._move(-1)
        elif name in ("KEY_PGDOWN", "KEY_NPAGE", "KEY_PAGEDOWN"):
            self._move(10)
        elif name in ("KEY_PGUP", "KEY_PPAGE", "KEY_PAGEUP"):
            self._move(-10)
        elif text == "g":
            self._set_job_index(0)
            self.detail_page = 0
        elif text == "G":
            self._set_job_index(len(self.jobs) - 1)
            self.detail_page = 0
        elif name == "KEY_TAB":
            self._activate("switch_pane")
        elif name in ("KEY_ENTER", "KEY_RETURN") or text in ("\n", "\r"):
            self._activate("detail")
        elif text == "[" and self.show_detail:
            self._scroll_detail(-1, page=True)
        elif text == "]" and self.show_detail:
            self._scroll_detail(1, page=True)
        elif text == "L":
            self._activate("log")
        elif text == "f":
            self._activate("filter")
        elif text == "s":
            self._activate("sort")
        elif text == "S":
            self.sort_reverse = not self.sort_reverse
            if self._snapshot is not None:
                self.jobs = self._visible_jobs(self._snapshot)
                self._reanchor_selection()
        elif text in ("+", "="):
            self._activate("priority", 1)
        elif text == "-":
            self._activate("priority", -1)
        elif text == "K" or name == "KEY_SR":
            self._activate("move", -1)
        elif text == "J" or name == "KEY_SF":
            self._activate("move", 1)
        elif text == "p":
            self._activate("procs")
        elif text == "t":
            self._activate("tick")
        elif text == "a":
            self._activate("auto")
        elif text == "c":
            self._activate("cancel")
        elif text == "r":
            self._activate("requeue")
        elif text == "d":
            self._activate("node_toggle")
        elif text == "A":
            self._activate("ack")
        return True

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
                                self.worker.set_log_request((job["id"], "stdout"))
                        print(self.render(state), end="", flush=True)
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
    mouse_enabled = nvidb_config.load_view_settings()["mouse"]
    return QueueTUI(
        db_path=db_path,
        refresh=refresh,
        mouse_enabled=mouse_enabled,
    ).run()
