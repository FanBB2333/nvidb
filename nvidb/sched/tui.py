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
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from blessed import Terminal

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

STATE_STYLE = {
    "running": "green",
    "pending": "yellow",
    "completed": "cyan",
    "failed": "red",
    "timeout": "red",
    "lost": "red",
    "cancelled": "bright_black",
}

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
        self._stop = threading.Event()
        self._scheduler: Optional[Scheduler] = None

    # --- public API (render thread) --------------------------------------

    def post(self, *action) -> None:
        self.actions.put(action)

    def stop(self) -> None:
        self._stop.set()
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
        while not self._stop.is_set():
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
        except Exception as error:
            self.set_notice(f"{type(error).__name__}: {error}", "red")
        finally:
            self._set_busy(False)


class QueueTUI:
    def __init__(self, db_path=None, refresh: float = 3.0):
        self.term = Terminal()
        self.worker = _Worker(db_path=db_path, refresh=refresh)
        self.focus = "jobs"  # jobs | nodes
        self.job_index = 0
        self.node_index = 0
        self.filter = "active"
        self.proc_view = "summary"
        self.show_detail = True
        self.show_log = False
        self.show_help = False
        self.pending_confirm: Optional[Tuple[str, int, float]] = None
        self.jobs: List[Dict[str, Any]] = []
        self.nodes: List[Dict[str, Any]] = []

    # --- data -------------------------------------------------------------

    def _visible_jobs(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.filter == "all":
            jobs = list(snapshot["jobs"]) + list(snapshot["recent"])
        else:
            wanted = FILTER_STATES[self.filter]
            pool = list(snapshot["jobs"]) + list(snapshot["recent"])
            jobs = [job for job in pool if job["state"] in wanted]
        jobs.sort(key=lambda job: job["id"])
        return jobs

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
        formatter = getattr(self.term, style, None)
        return formatter(text) if callable(formatter) else text

    def _fit(self, text: str, width: int) -> str:
        return fit_display(text, width)

    def _compose(self, segments, width: int, *, reverse: bool = False) -> str:
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
        if reverse:
            return self.term.reverse(pad_display("".join(plain_parts), width))
        return "".join(
            self._style(part, style)
            for part, (_text, style) in zip(plain_parts, segments)
        )

    @staticmethod
    def _bar(fraction: float, width: int) -> str:
        width = max(0, width)
        filled = int(round(max(0.0, min(1.0, fraction)) * width))
        return "█" * filled + "░" * (width - filled)

    def _header_lines(self, state: Dict[str, Any], width: int) -> List[str]:
        snapshot = state["snapshot"] or {}
        counts = snapshot.get("counts") or {}
        segments = [(" ", None)]
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
                segments.append((f"{label} {counts[label]}", style))
        if len(segments) == 1:
            segments.append(("queue empty", "bright_black"))

        tick_age = age_seconds(snapshot.get("last_tick_at"))
        tick_text = "never" if tick_age is None else f"{int(tick_age)}s ago"
        mode = "auto" if state["auto_tick"] else "manual"
        status = "working" if state["busy"] else "idle"
        title = " nvidb queue "
        meta = (
            f"tick {tick_text} · {mode} · {status} · filter {self.filter} "
            f"· procs {self.proc_view} "
        )
        gap = max(1, width - len(title) - len(meta))
        return [
            self._compose(
                [(title, "reverse"), (" " * gap, None), (meta, "bright_black")], width
            ),
            self._compose(segments, width),
        ]

    def _alert_lines(self, snapshot: Dict[str, Any], width: int) -> List[str]:
        """A banner for failures nobody has acknowledged yet.

        Alerts sit above everything else because they are the one thing on this
        screen that needs a decision rather than a glance.
        """
        alerts = snapshot.get("alerts") or []
        open_alerts = [alert for alert in alerts if not alert.get("acknowledged_at")]
        if not open_alerts:
            return []
        shown = open_alerts[-3:]
        lines = [
            self._compose(
                [
                    (f" ⚠ {len(open_alerts)} alert(s) ", "red"),
                    ("— press A to acknowledge, L to read the job's log", "bright_black"),
                ],
                width,
            )
        ]
        for alert in shown:
            style = "red" if alert.get("severity") == "error" else "yellow"
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
        if len(open_alerts) > len(shown):
            lines.append(
                self._style(
                    f"   … {len(open_alerts) - len(shown)} more", "bright_black"
                )
            )
        return lines

    def _node_lines(self, width: int) -> List[str]:
        lines = [self._style("─ NODES " + "─" * max(0, width - 8), "bright_black")]
        for position, node in enumerate(self.nodes):
            selected = self.focus == "nodes" and position == self.node_index
            marker = "›" if selected else " "
            state = node["state"]
            state_style = {"up": "green", "down": "red", "drain": "yellow"}.get(
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
                    reverse=selected,
                )
            )

            if state == "down" and node.get("last_error"):
                lines.append(
                    self._style(self._fit(f"    ! {node['last_error']}", width), "red")
                )
                continue

            for gpu in node["gpus"]:
                total = max(1, gpu["mem_total_mb"])
                # The bar shows how much of the card is already spoken for:
                # foreign processes plus this queue's reservations.
                committed = (gpu["external_mem_mb"] + gpu["reserved_mb"]) / total
                bar_width = 18 if width >= 100 else 10
                util = gpu["util_percent"]
                util_text = f"{util:>3}%" if util is not None else "  -%"
                free_style = (
                    "green" if gpu["free_mb"] >= 4096
                    else "yellow" if gpu["free_mb"] >= 1024
                    else "red"
                )
                name_width = 26 if width >= 100 else 16
                left = (
                    f"    GPU{gpu['index']} "
                    f"{self._fit(gpu['name'] or '-', name_width):<{name_width}} "
                    f"util {util_text}  [{self._bar(committed, bar_width)}] "
                )
                free = f"free {format_mb(gpu['free_mb']):>6}"
                source = (
                    "blind"
                    if gpu.get("attribution") == "blind"
                    else f"{gpu['external_procs']}p"
                )
                # Whole-card occupancy first: what is actually on the GPU matters
                # even when none of it came from this queue.
                occupancy = (
                    f"  mem {format_mb(gpu['mem_used_mb'])}/{format_mb(gpu['mem_total_mb'])}"
                    if width >= 120
                    else ""
                )
                rest = (
                    f"{occupancy}"
                    f"  other {format_mb(gpu['external_mem_mb']):>6} ({source})"
                    f"  res {format_mb(gpu['reserved_mb']):>6}  jobs {gpu['queue_jobs']}"
                )
                lines.append(
                    self._compose(
                        [(left, "bright_black"), (free, free_style), (rest, "bright_black")],
                        width,
                    )
                )
                lines.extend(self._gpu_process_lines(gpu, width))
        return lines

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

    def _job_lines(self, width: int, height: int) -> List[str]:
        title = f"─ JOBS ({len(self.jobs)}) "
        lines = [self._style(title + "─" * max(0, width - len(title)), "bright_black")]
        columns = [
            ("ID", 5),
            ("STATE", 9),
            ("NAME", 16),
            ("NODE", 20),
            ("GPU", 4),
            ("VRAM", 6),
            ("USED", 6),
            ("ELAPSED", 9),
            ("RC", 3),
        ]
        if width < 100:
            columns = [column for column in columns if column[0] not in ("USED", "NODE")]
        used = sum(size + 1 for _, size in columns)
        command_width = max(10, width - used - 2)
        # A job that reports its own progress is being watched for exactly that,
        # so the last column names whichever of the two it will show.
        any_progress = any(job.get("progress") for job in self.jobs)
        last_column = "PROGRESS / COMMAND" if any_progress else "COMMAND"
        header = " " + "".join(name.ljust(size + 1) for name, size in columns) + last_column
        lines.append(self._style(self._fit(header, width), "bright_black"))

        if not self.jobs:
            lines.append(self._style("  (no jobs match this filter)", "bright_black"))
            return lines

        rows = max(1, height - 2)
        start = max(0, min(self.job_index - rows // 2, len(self.jobs) - rows))
        start = max(0, start)
        for position in range(start, min(len(self.jobs), start + rows)):
            job = self.jobs[position]
            values = {
                "ID": str(job["id"]),
                "STATE": job["state"],
                "NAME": job["name"] or "-",
                "NODE": job["node"] or job["node_constraint"] or "-",
                "GPU": ",".join(str(i) for i in job["gpu_ids"]) or "-",
                "VRAM": format_mb(job["vram_mb"]) if job["vram_mb"] else "-",
                "USED": format_mb(job["gpu_mem_mb"]) if job.get("gpu_mem_mb") else "-",
                "ELAPSED": format_duration(job["elapsed_s"])
                if job.get("elapsed_s") is not None
                else "-",
                "RC": "-" if job["exit_code"] is None else str(job["exit_code"]),
            }
            body = "".join(
                pad_display(fit_display(values[name], size), size + 1)
                for name, size in columns
            )
            # What the job says about itself beats the command line it was
            # started with; `▸` marks which one is on screen.
            progress = job.get("progress")
            if progress:
                tail, tail_style = f"▸ {' '.join(progress.split())}", "cyan"
            else:
                tail, tail_style = " ".join(job["command"].split()), None
            selected = self.focus == "jobs" and position == self.job_index
            marker = "›" if selected else " "
            lines.append(
                self._compose(
                    [
                        (marker, "bright_black"),
                        (body, STATE_STYLE.get(job["state"])),
                        (self._fit(tail, command_width), tail_style),
                    ],
                    width,
                    reverse=selected,
                )
            )
        if start + rows < len(self.jobs):
            lines.append(
                self._style(f"  … {len(self.jobs) - start - rows} more", "bright_black")
            )
        return lines

    def _detail_lines(self, state: Dict[str, Any], width: int, height: int) -> List[str]:
        job = self.selected_job()
        if job is None or height < 3:
            return []
        if self.show_log:
            title = f"─ LOG job {job['id']} "
            lines = [self._style(title + "─" * max(0, width - len(title)), "bright_black")]
            text = state["log_text"] if state["log_job"] == job["id"] else ""
            if not text:
                lines.append(self._style("  (fetching…)", "bright_black"))
                return lines
            body = text.rstrip("\n").splitlines()[-(height - 1):]
            lines.extend("  " + self._fit(line, width - 2) for line in body)
            return lines

        title = f"─ JOB {job['id']} "
        lines = [self._style(title + "─" * max(0, width - len(title)), "bright_black")]
        pieces = [
            f"name {job['name'] or '-'}",
            f"state {job['state']}",
            f"request gpus={job['gpus']} vram={format_mb(job['vram_mb'])}",
            f"node {job['node'] or job['node_constraint'] or 'any'}",
            f"pid {job['remote_pid'] or '-'}",
            f"submitter {job['submitter'] or '-'}",
        ]
        lines.append(self._fit("  " + "   ".join(pieces), width))
        lines.append(self._fit(f"  cmd  {' '.join(job['command'].split())}", width))
        if job.get("workdir"):
            lines.append(self._fit(f"  cwd  {job['workdir']}", width))
        if job.get("progress"):
            lines.append(
                self._compose(
                    [("  live ", "bright_black"), (" ".join(job["progress"].split()), "cyan")],
                    width,
                )
            )
        if job.get("notes"):
            lines.append(self._fit(f"  note {job['notes']}", width))
        if job.get("last_error"):
            lines.append(self._style(self._fit(f"  err  {job['last_error']}", width), "red"))
        if job.get("result") is not None:
            import json as _json

            lines.append(
                self._fit(
                    f"  out  {_json.dumps(job['result'], ensure_ascii=False)}", width
                )
            )
        return lines[:height]

    def _footer_lines(self, state: Dict[str, Any], width: int) -> List[str]:
        notice = state.get("notice")
        error = state.get("error")
        lines = []
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
                        f" press {action[0]} again to {action} job {job_id} (Esc cancels)",
                        width,
                    ),
                    "yellow",
                )
            )
        keys = (
            "j/k move  Tab pane  Enter detail  L log  c cancel  r requeue  "
            "A ack  t tick  a auto  f filter  p procs  d drain  ? help  q quit"
        )
        lines.append(self._style(self._fit(" " + keys, width), "bright_black"))
        return lines

    def _help_lines(self, width: int) -> List[str]:
        rows = [
            ("j / k / ↑ / ↓", "Move the selection in the focused pane"),
            ("PgUp / PgDn", "Move a page at a time"),
            ("Tab", "Switch focus between the node and job panes"),
            ("Enter", "Show or hide the detail pane"),
            ("L", "Toggle the log tail for the selected job"),
            ("c", "Cancel the selected job (press twice)"),
            ("r", "Re-queue the selected finished job"),
            ("t", "Force a scheduler tick now"),
            ("a", "Toggle automatic ticking"),
            ("f", "Cycle the job filter"),
            ("p", "GPU processes: unmanaged only / all / none"),
            ("d", "Drain or resume the selected node"),
            ("A", "Acknowledge every open alert"),
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
        snapshot = state["snapshot"]
        if snapshot is None:
            message = state.get("error") or "connecting to nodes…"
            return self.term.home + self.term.clear + f"\n  {message}\n"

        self.nodes = snapshot["nodes"]
        self.jobs = self._visible_jobs(snapshot)

        # Leave the bottom row free so writing the last line cannot scroll the
        # screen and shear the frame.
        usable = height - 1
        lines: List[str] = list(self._header_lines(state, width))
        lines.extend(self._alert_lines(snapshot, width))

        if self.show_help:
            lines.extend(self._help_lines(width))
            lines = lines[:usable]
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
            lines.extend(node_lines)
            footer = self._footer_lines(state, width)
            detail_height = 0
            if self.show_detail and self.jobs:
                detail_height = 8
            body_height = usable - len(footer)
            job_height = body_height - len(lines) - detail_height
            lines.extend(self._job_lines(width, max(4, job_height)))
            if detail_height:
                lines.extend(self._detail_lines(state, width, detail_height))
            # The footer holds the keybindings, so it is reserved rather than
            # left to whatever space happens to remain.
            lines = lines[:body_height]
            lines.extend([""] * (body_height - len(lines)))
            lines.extend(footer)

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
            self.job_index = max(0, min(self.job_index + delta, len(self.jobs) - 1))

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
            return False
        if text == "?":
            self.show_help = True
            return True
        if text in ("j",) or name == "KEY_DOWN":
            self._move(1)
        elif text in ("k",) or name == "KEY_UP":
            self._move(-1)
        elif name == "KEY_PGDOWN":
            self._move(10)
        elif name == "KEY_PGUP":
            self._move(-10)
        elif text == "g":
            self.job_index = 0
        elif text == "G":
            self.job_index = max(0, len(self.jobs) - 1)
        elif name == "KEY_TAB":
            self.focus = "nodes" if self.focus == "jobs" else "jobs"
        elif name == "KEY_ENTER" or text in ("\n", "\r"):
            self.show_detail = not self.show_detail
        elif text == "L":
            job = self.selected_job()
            self.show_log = not self.show_log
            self.show_detail = True
            self.worker.set_log_request(
                (job["id"], "stdout") if (self.show_log and job) else None
            )
        elif text == "f":
            self.filter = FILTERS[(FILTERS.index(self.filter) + 1) % len(FILTERS)]
            self.job_index = 0
        elif text == "p":
            self.proc_view = PROC_VIEWS[
                (PROC_VIEWS.index(self.proc_view) + 1) % len(PROC_VIEWS)
            ]
        elif text == "t":
            self.worker.post("tick")
        elif text == "a":
            self.worker.post("auto", not self.worker.auto_tick)
        elif text == "c":
            job = self.selected_job()
            if job and self._confirm("cancel", job["id"]):
                self.worker.post("cancel", job["id"])
        elif text == "r":
            job = self.selected_job()
            if job:
                self.worker.post("requeue", job["id"])
        elif text == "d":
            node = self.selected_node()
            if node:
                self.worker.post(
                    "resume" if not node["enabled"] else "drain", node["name"]
                )
        elif text == "A":
            self.worker.post("ack")
        return True

    # --- main loop --------------------------------------------------------

    def run(self) -> int:
        term = self.term
        self.worker.start()
        try:
            with term.fullscreen(), term.cbreak(), term.hidden_cursor():
                while True:
                    state = self.worker.read_state()
                    if self.show_log:
                        job = self.selected_job()
                        if job:
                            self.worker.set_log_request((job["id"], "stdout"))
                    print(self.render(state), end="", flush=True)
                    key = term.inkey(timeout=0.4)
                    if key and not self.handle_key(key):
                        break
                    if (
                        self.pending_confirm
                        and time.time() - self.pending_confirm[2] > CONFIRM_SECONDS
                    ):
                        self.pending_confirm = None
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
    return QueueTUI(db_path=db_path, refresh=refresh).run()
