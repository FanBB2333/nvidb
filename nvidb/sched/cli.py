"""Command-line surface for the nvidb job queue.

Every read command prints a stable table by default and the same data as JSON
with `--json`, because the queue's main consumers are other programs. Commands
that change the world (submit, cancel, requeue) drive a scheduler tick on the
way out, which is what keeps the queue moving without a daemon.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import socket
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import backup as backup_mod
from . import db as dbm
from . import keeper as keeper_mod
from . import remote as remote_mod
from .executor import VALID_ENV_KEY
from .model import (
    JOB_STATES,
    TERMINAL_JOB_STATES,
    GpuProcess,
    Job,
    age_seconds,
    display_width,
    fit_display,
    format_duration,
    format_mb,
    pad_display,
)
from .scheduler import Scheduler

STATE_ORDER = ["running", "pending"] + [
    state for state in JOB_STATES if state in TERMINAL_JOB_STATES
]


# --- helpers ---------------------------------------------------------------

def _default_submitter() -> str:
    explicit = os.environ.get("NVIDB_SUBMITTER")
    if explicit:
        return explicit
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    return f"{user}@{socket.gethostname()}:{os.getpid()}"


def _print_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


def _error(message: str, *, as_json: bool = False) -> int:
    if as_json:
        _print_json({"ok": False, "error": message})
    else:
        print(f"error: {message}", file=sys.stderr)
    return 1


def _table(rows: List[Sequence[str]], headers: Sequence[str], indent: str = "  ") -> str:
    """Render a fixed-width table; column widths follow the widest cell.

    Widths are measured in terminal columns rather than characters, so a
    Chinese job name or note does not shear the alignment.
    """
    if not rows:
        return f"{indent}(none)"
    columns = len(headers)
    widths = [display_width(header) for header in headers]
    for row in rows:
        for index in range(columns):
            widths[index] = max(widths[index], display_width(row[index]))
    lines = [
        indent + "  ".join(pad_display(headers[i], widths[i]) for i in range(columns)).rstrip()
    ]
    for row in rows:
        lines.append(
            indent + "  ".join(pad_display(row[i], widths[i]) for i in range(columns)).rstrip()
        )
    return "\n".join(lines)


def _short(text: Optional[str], limit: int = 60) -> str:
    text = " ".join((text or "").split())
    return fit_display(text, limit) if text else "-"


def _age(value: Optional[str]) -> str:
    seconds = age_seconds(value)
    if seconds is None:
        return "never"
    return f"{format_duration(seconds)} ago"


def _open(args) -> Scheduler:
    conn = dbm.open_db(getattr(args, "db_path", None))
    scheduler = Scheduler(conn)
    scheduler.sync_nodes_from_config()
    return scheduler


def _maybe_tick(scheduler: Scheduler, args, *, force: bool = False) -> Dict[str, Any]:
    if getattr(args, "no_tick", False):
        return {"skipped": "disabled"}
    return scheduler.tick(force=force)


def _refresh(scheduler: Scheduler, args) -> Dict[str, Any]:
    """Bring state up to date before a read command prints it.

    Without a daemon nothing else advances the queue, so a plain `job show`
    would report whatever the last client happened to observe. The tick is
    rate-limited, so a client polling in a loop still only reaches the nodes
    every few seconds; `--tick` forces it and `--no-tick` skips it entirely.
    """
    if getattr(args, "no_tick", False):
        return {"skipped": "disabled"}
    return scheduler.tick(force=bool(getattr(args, "tick", False)))


def _resolve_ids(scheduler: Scheduler, raw_ids: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for value in raw_ids:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            ids.append(int(part))
    return ids


# --- rendering -------------------------------------------------------------

SUMMARY_PROCESSES = 3


def _gpu_process_lines(gpu: Dict[str, Any], *, detail: bool) -> List[str]:
    """Name the work sitting on a card, whether or not the queue started it.

    Capacity numbers alone cannot answer "why is this GPU full"; the answer is
    usually a process nobody submitted through the queue.
    """
    processes = gpu.get("processes") or []
    external = [process for process in processes if not process.get("managed")]

    if gpu.get("attribution") == "blind":
        # The driver accounts for none of the memory in use (WSL), so say that
        # rather than implying the card is idle or that the split is measured.
        lines = [
            f"      unmanaged  ~{format_mb(gpu['external_mem_mb'])} of "
            f"{format_mb(gpu['mem_used_mb'])} in use; this driver reports no "
            "per-process memory"
        ]
        shown = processes if detail else processes[:SUMMARY_PROCESSES]
        for process in shown:
            entry = GpuProcess.from_dict(process)
            lines.append(
                f"        pid {entry.pid}  {_short(entry.name, 40)}  "
                f"{entry.username or '-'}  {entry.owner}"
            )
        return lines
    if not processes:
        return []

    if not detail:
        if not external:
            return []
        shown = external[:SUMMARY_PROCESSES]
        text = ", ".join(GpuProcess.from_dict(process).describe() for process in shown)
        if len(external) > len(shown):
            text += f", +{len(external) - len(shown)} more"
        return [
            f"      unmanaged  {format_mb(gpu['external_mem_mb'])} "
            f"in {len(external)} process(es): {text}"
        ]

    rows = [
        [
            str(process.get("pid") or "-"),
            _short(process.get("user"), 12),
            format_mb(process.get("mem_mb")),
            process.get("type") or "-",
            GpuProcess.from_dict(process).owner,
            _short(process.get("name"), 40),
        ]
        for process in processes
    ]
    table = _table(rows, ["PID", "USER", "MEM", "T", "OWNER", "PROCESS"], indent="        ")
    return [f"      GPU{gpu['index']} processes"] + table.splitlines()


def _render_nodes(snapshot: Dict[str, Any], *, procs: bool = False) -> str:
    lines = ["NODES"]
    for node in snapshot["nodes"]:
        flags = [node["state"]]
        if not node["enabled"]:
            flags.append("disabled")
        if node.get("ignored"):
            flags.append("ignored")
        header = f"  {node['name']}  [{'/'.join(flags)}]  {node['hostname'] or ''}"
        if node["state"] != "up" and node.get("last_error"):
            header += f"  ! {_short(node['last_error'], 70)}"
        elif node.get("last_seen"):
            header += f"  seen {_age(node['last_seen'])}"
        lines.append(header)
        rows = []
        for gpu in node["gpus"]:
            # "blind" means the driver named no processes, so the split between
            # foreign work and our own is inferred rather than measured.
            source = (
                "blind"
                if gpu.get("attribution") == "blind"
                else f"{gpu['external_procs']}p"
            )
            used = f"{format_mb(gpu['mem_used_mb'])}/{format_mb(gpu['mem_total_mb'])}"
            if gpu.get("mem_used_percent") is not None:
                used += f" {gpu['mem_used_percent']}%"
            rows.append(
                [
                    f"GPU{gpu['index']}",
                    _short(gpu["name"], 26),
                    f"{gpu['util_percent'] if gpu['util_percent'] is not None else '-'}%",
                    used,
                    f"{format_mb(gpu['external_mem_mb'])} ({source})",
                    format_mb(gpu.get("queue_mem_mb") or 0),
                    format_mb(gpu["reserved_mb"]),
                    f"{gpu['queue_jobs']}",
                    format_mb(gpu["free_mb"]),
                ]
            )
        lines.append(
            _table(
                rows,
                # UNMANAGED is everything on the card the queue did not start;
                # QUEUE is what its own jobs actually hold right now.
                [
                    "GPU",
                    "MODEL",
                    "UTIL",
                    "MEM",
                    "UNMANAGED",
                    "QUEUE",
                    "RESERVED",
                    "JOBS",
                    "FREE",
                ],
                indent="    ",
            )
        )
        for gpu in node["gpus"]:
            lines.extend(_gpu_process_lines(gpu, detail=procs))
    return "\n".join(lines)


JOB_HEADERS = [
    "ID",
    "STATE",
    "NAME",
    "NODE",
    "GPU",
    "VRAM",
    "USED",
    "ELAPSED",
    "RC",
    "COMMAND",
]


def _job_row(job: Dict[str, Any], *, extra: Sequence[str] = ()) -> List[str]:
    gpu = ",".join(str(index) for index in job["gpu_ids"]) or "-"
    used = format_mb(job["gpu_mem_mb"]) if job.get("gpu_mem_mb") else "-"
    row = [
        str(job["id"]),
        # A held job is pending in the database but stuck in practice, and
        # calling it "pending" is how someone waits for a job that will never
        # start on its own.
        "held" if job.get("held") else job["state"],
        _short(job["name"] or "-", 20),
        job["node"] or (job["node_constraint"] or "-"),
        gpu,
        format_mb(job["vram_mb"]) if job["vram_mb"] else "-",
        used,
        format_duration(job["elapsed_s"]) if job.get("elapsed_s") is not None else "-",
        "-" if job["exit_code"] is None else str(job["exit_code"]),
        _short(job["command"], 44),
    ]
    row.extend(_short(job.get(key), 40) for key in extra)
    return row


def _job_table(jobs: Sequence[Dict[str, Any]], indent: str = "  ") -> str:
    """Render jobs, adding PROGRESS and NOTE columns only when they carry data.

    Most rows have neither, and always reserving the width would squeeze the
    command out of a normal terminal.
    """
    optional = {"held_reason": "HELD BY", "progress": "PROGRESS", "notes": "NOTE"}
    extra = [key for key in optional if any(job.get(key) for job in jobs)]
    headers = JOB_HEADERS + [optional[key] for key in extra]
    return _table([_job_row(job, extra=extra) for job in jobs], headers, indent=indent)


def _render_status(snapshot: Dict[str, Any], *, procs: bool = False) -> str:
    counts = dict(snapshot["counts"])
    held = snapshot.get("held") or []
    if held:
        # Held jobs are stored as pending; counting them in both places would
        # overstate what the queue is actually going to get to.
        counts["pending"] = max(0, counts.get("pending", 0) - len(held))
        counts["held"] = len(held)
    summary = " · ".join(
        f"{state} {counts[state]}"
        for state in ["running", "pending", "held"] + [
            state for state in STATE_ORDER if state in TERMINAL_JOB_STATES
        ]
        if counts.get(state)
    ) or "empty"
    lock = snapshot.get("tick_lock") or {}
    header = (
        f"QUEUE  db={snapshot['db_path']}  last tick {_age(snapshot['last_tick_at'])}"
    )
    if lock.get("owner"):
        header += f"  (tick held by {lock['owner']})"
    if snapshot.get("last_backup_at"):
        header += f"  backup {_age(snapshot['last_backup_at'])}"
    keeper_state = snapshot.get("keeper") or {}
    # Only worth a word where a keeper is meant to exist: on a laptop driving a
    # remote queue there is nothing here to report.
    if keeper_state.get("installed") or keeper_state.get("running"):
        header += f"  keeper {'up' if keeper_state.get('running') else 'DOWN'}"
    parts = [header]
    if snapshot.get("open_alerts"):
        # The first thing to say about a queue with failures in it is that it
        # has failures in it.
        parts.append("")
        parts.append(f"ALERTS  {snapshot['open_alerts']} open (nvidb queue alerts)")
        parts.append(_render_alerts(snapshot.get("alerts") or []))
    parts += [
        "",
        _render_nodes(snapshot, procs=procs),
        "",
        f"JOBS  {summary}",
    ]
    parts.append(_job_table(snapshot["jobs"]))
    if snapshot["recent"]:
        parts.append("")
        parts.append("RECENTLY FINISHED")
        parts.append(_job_table(snapshot["recent"]))
    return "\n".join(parts)


def _render_job_detail(job: Job, *, logs: Optional[str] = None) -> str:
    data = job.to_dict()
    lines = [
        f"JOB {job.id}  {job.display_state}"
        + (f"  (exit {job.exit_code})" if job.exit_code is not None else ""),
        f"  name        {job.name or '-'}",
        f"  command     {job.command}",
        f"  workdir     {job.workdir or '-'}",
        f"  request     gpus={job.gpus} vram={format_mb(job.vram_mb)}"
        + (f" node={job.node_constraint}" if job.node_constraint else "")
        + (f" priority={job.priority}" if job.priority else ""),
        f"  placement   node={job.node or '-'} gpu={','.join(str(i) for i in job.gpu_ids) or '-'} pid={job.remote_pid or '-'}"
        + (f" lane={job.lane}" if job.lane else ""),
        f"  vram used   {format_mb(job.gpu_mem_mb) if job.gpu_mem_mb else '-'}",
        f"  timing      created={job.created_at or '-'} started={job.started_at or '-'} finished={job.finished_at or '-'}",
        f"  elapsed     {format_duration(data['elapsed_s']) if data['elapsed_s'] is not None else '-'}",
        f"  attempt     {job.attempt}/{job.max_retries + 1}",
        f"  submitter   {job.submitter or '-'}",
        f"  run_dir     {job.run_dir or '-'}",
    ]
    if job.progress:
        lines.append(f"  progress    {job.progress}   (reported {_age(job.progress_at)})")
    if job.notes:
        lines.append(f"  note        {job.notes}")
    if job.held_reason:
        lines.append(f"  held        {job.held_reason}")
        lines.append(
            f"              release it with `nvidb job release {job.id}`, or "
            f"repoint it with `nvidb job edit {job.id} --add-after ID`"
        )
    if job.depends_on:
        lines.append(f"  depends_on  {','.join(str(i) for i in job.depends_on)}")
    if job.depends_any:
        lines.append(
            f"  after_any   {','.join(str(i) for i in job.depends_any)}"
            "   (any outcome counts)"
        )
    if job.tags:
        lines.append(f"  tags        {','.join(job.tags)}")
    if job.env:
        lines.append(f"  env         {json.dumps(job.env, ensure_ascii=False)}")
    if job.last_error:
        lines.append(f"  error       {job.last_error}")
    if job.result is not None:
        lines.append(f"  result      {json.dumps(job.result, ensure_ascii=False)}")
    if logs:
        lines.append("")
        lines.append("LOG (stdout tail)")
        lines.extend(f"  {line}" for line in logs.rstrip("\n").splitlines())
    return "\n".join(lines)


# --- command handlers ------------------------------------------------------

def cmd_status(args) -> int:
    scheduler = _open(args)
    try:
        _refresh(scheduler, args)
        snapshot = scheduler.snapshot(
            include_ignored=bool(getattr(args, "include_ignored", False))
        )
        if args.json:
            _print_json(snapshot)
        else:
            print(_render_status(snapshot, procs=bool(getattr(args, "procs", False))))
        return 0
    finally:
        scheduler.close()


def cmd_tick(args) -> int:
    scheduler = _open(args)
    try:
        while True:
            summary = scheduler.tick(force=True)
            if args.json:
                _print_json(summary)
            else:
                bits = [
                    f"nodes up={summary['nodes_up']} down={summary['nodes_down']}",
                    f"dispatched={len(summary['dispatched'])}",
                    f"finished={len(summary['finished'])}",
                ]
                if summary.get("nodes_ignored"):
                    bits.insert(1, f"ignored={summary['nodes_ignored']}")
                if summary["skipped"]:
                    bits.insert(0, f"skipped={summary['skipped']}")
                print("tick: " + "  ".join(bits))
                for item in summary["dispatched"]:
                    print(f"  -> job {item['id']} started on {item['node']} GPU {item['gpu_ids']}")
                for item in summary["finished"]:
                    print(f"  <- job {item['id']} {item['state']}")
                for item in summary["errors"]:
                    print(f"  !! {item}")
            if not args.watch:
                return 0
            time.sleep(max(1, int(args.watch)))
    except KeyboardInterrupt:
        return 0
    finally:
        scheduler.close()


def _render_alerts(alerts: Sequence[Dict[str, Any]], *, detail: bool = False) -> str:
    if not alerts:
        return "no open alerts"
    rows = [
        [
            str(alert["id"]),
            (alert["ts"] or "")[:19].replace("T", " "),
            alert["severity"],
            alert["kind"],
            str(alert["job_id"] or "-"),
            alert["node"] or "-",
            _short(alert["title"], 60),
        ]
        for alert in alerts
    ]
    out = [_table(rows, ["ID", "TIME", "LEVEL", "KIND", "JOB", "NODE", "WHAT"], indent="")]
    if detail:
        for alert in alerts:
            if not alert.get("detail"):
                continue
            out.append("")
            out.append(f"--- alert {alert['id']}: {alert['title']}")
            out.extend(f"    {line}" for line in str(alert["detail"]).splitlines())
    return "\n".join(out)


def cmd_alerts(args) -> int:
    scheduler = _open(args)
    try:
        _refresh(scheduler, args)
        alerts = dbm.list_alerts(
            scheduler.conn, open_only=not args.all, limit=args.number
        )
        open_count = dbm.open_alert_count(scheduler.conn)
        if args.json:
            _print_json({"alerts": alerts, "open": open_count})
        else:
            print(_render_alerts(alerts, detail=args.detail))
        # A non-zero status lets a shell or agent branch on "anything wrong?".
        # It counts every open alert, not just the ones this listing happened to
        # show: `--all -n 5` must not report a clean queue because the five
        # newest alerts were acknowledged.
        return 1 if open_count else 0
    finally:
        scheduler.close()


def cmd_ack(args) -> int:
    scheduler = _open(args)
    try:
        ids = _resolve_ids(scheduler, args.ids) if args.ids else None
        count = dbm.acknowledge_alerts(scheduler.conn, ids, all_open=args.all)
        if args.json:
            _print_json({"ok": True, "acknowledged": count})
        else:
            print(f"acknowledged {count} alert(s)")
        return 0
    finally:
        scheduler.close()


def cmd_backup(args) -> int:
    """Write a verified SQLite snapshot without stopping the queue."""
    scheduler = _open(args)
    try:
        if args.keep < 0:
            return _error("--keep cannot be negative", as_json=args.json)
        try:
            info = backup_mod.create(
                scheduler.conn,
                args.path,
                keep=args.keep,
            )
        except (OSError, ValueError, sqlite3.Error) as error:
            return _error(f"could not back up the queue: {error}", as_json=args.json)
        if args.json:
            _print_json({"ok": True, "backup": info})
        else:
            print(f"backed up {info['source']} to {info['path']}")
            print("  verified  yes")
            print(f"  size      {info['bytes']} bytes")
            if info["pruned"]:
                print(f"  pruned    {len(info['pruned'])} older backup(s)")
        return 0
    finally:
        scheduler.close()


def cmd_daemon(args) -> int:
    """Keep the queue moving and push failures out as they happen."""
    from .notify import Notifier

    scheduler = _open(args)
    queue_settings = ((scheduler.cfg or {}).get("queue") or {})
    if not isinstance(queue_settings, dict):
        queue_settings = {}
    notifier = Notifier(queue_settings)
    interval = max(2, int(args.interval))
    if not args.json:
        channels = [
            name
            for name, on in (
                ("desktop", notifier.settings["desktop"] and notifier._desktop_command),
                ("log", notifier.settings["log"]),
                ("command", bool(notifier.settings["command"])),
            )
            if on
        ]
        print(
            f"nvidb queue daemon: every {interval}s, "
            f"notifying via {', '.join(channels) or 'nothing (disabled)'}",
            file=sys.stderr,
        )
        print("Press Ctrl+C to stop.", file=sys.stderr)
    try:
        while True:
            summary = scheduler.tick(force=True)
            sent = 0 if args.no_notify else scheduler.deliver_alerts(notifier)
            backup = None
            backup_error = None
            if not args.no_backup:
                try:
                    backup = backup_mod.maybe_create(
                        scheduler.conn, queue_settings
                    )
                except (OSError, ValueError, sqlite3.Error) as error:
                    # A backup failure needs attention, but it must not stop
                    # scheduling and timeout enforcement.
                    backup_error = str(error)
            if args.json:
                _print_json(
                    {
                        "tick": summary,
                        "notified": sent,
                        "backup": backup,
                        "backup_error": backup_error,
                    }
                )
                sys.stdout.flush()
            else:
                for item in summary.get("alerts") or []:
                    print(f"  ! {item['kind']}: {item['title']}", file=sys.stderr)
                if backup:
                    print(f"  backup: {backup['path']}", file=sys.stderr)
                if backup_error:
                    print(f"  ! backup failed: {backup_error}", file=sys.stderr)
            if args.once:
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    finally:
        scheduler.close()


def _render_keeper(state: Dict[str, Any]) -> str:
    if state.get("running"):
        pid = f", pid {state['pid']}" if state.get("pid") else ""
        return f"running ({state.get('manager', 'shell')}{pid})"
    if state.get("installed"):
        return f"stopped ({state.get('manager', 'shell')})"
    return "not installed"


def cmd_keeper(args) -> int:
    """Install and drive the shell script that keeps this queue moving."""
    action = getattr(args, "keeper_action", None) or "status"

    if action == "install":
        manager_name = "systemd" if args.systemd else "shell"
        try:
            info = keeper_mod.install(
                nvidb_bin=args.nvidb,
                interval=args.interval,
                manager_name=manager_name,
            )
        except (ValueError, RuntimeError) as error:
            return _error(str(error), as_json=args.json)
        except OSError as error:
            return _error(f"could not write the keeper: {error}", as_json=args.json)
        started = None
        start_error = None
        if args.start:
            result = keeper_mod.run("restart" if info["replaced"] else "start")
            started = result.returncode == 0
            if not started:
                start_error = (
                    result.stderr or result.stdout or f"exit {result.returncode}"
                ).strip()
        if args.json:
            _print_json(
                {
                    "ok": started is not False,
                    **info,
                    "started": started,
                    "start_error": start_error,
                    "state": keeper_mod.status(),
                }
            )
        else:
            verb = "reinstalled" if info["replaced"] else "installed"
            print(f"{verb} {info['script']}")
            print(f"  manager {info['manager']}")
            print(f"  runs   {info['nvidb']} queue daemon --interval {info['interval']}")
            print(f"  log    {info['log']}")
            if info.get("unit"):
                print(f"  unit   {info['unit']}")
            if started is None:
                print("  start it with: nvidb queue keeper start")
            elif start_error:
                print(f"error: keeper did not start: {start_error}", file=sys.stderr)
            else:
                print(f"  keeper {_render_keeper(keeper_mod.status())}")
        return 0 if started is not False else 1

    if action == "status":
        state = keeper_mod.status()
        if args.json:
            _print_json(state)
        else:
            print(f"keeper {_render_keeper(state)}")
            print(f"  script {state['script']}")
            print(f"  log    {state['log']}")
            if state.get("unit"):
                print(f"  unit   {state['unit']}")
        # Non-zero while nothing is keeping the queue moving, so a shell or an
        # agent can branch on it the way it does on `queue alerts`.
        return 0 if state["running"] else 1

    extra = [args.number] if action == "logs" else []
    try:
        result = keeper_mod.run(action, *extra)
    except FileNotFoundError as error:
        return _error(
            f"no keeper installed at {error} (run `nvidb queue keeper install`)",
            as_json=args.json,
        )
    output = (result.stdout or "").rstrip("\n")
    if args.json:
        _print_json(
            {
                "ok": result.returncode == 0,
                "action": action,
                "output": output,
                "state": keeper_mod.status(),
            }
        )
    else:
        if output:
            print(output)
        if result.stderr.strip():
            print(result.stderr.rstrip("\n"), file=sys.stderr)
    return result.returncode


def cmd_events(args) -> int:
    scheduler = _open(args)
    try:
        events = dbm.list_events(
            scheduler.conn,
            since_id=args.since,
            job_id=args.job,
            limit=args.number,
        )
        if args.json:
            _print_json({"events": events, "last_id": events[-1]["id"] if events else args.since})
        else:
            rows = [
                [
                    str(event["id"]),
                    (event["ts"] or "")[:19].replace("T", " "),
                    event["kind"],
                    str(event["job_id"] or "-"),
                    event["node"] or "-",
                    _short(event["message"], 60),
                ]
                for event in events
            ]
            print(_table(rows, ["ID", "TIME", "KIND", "JOB", "NODE", "MESSAGE"], indent=""))
        return 0
    finally:
        scheduler.close()


def cmd_nodes(args) -> int:
    scheduler = _open(args)
    try:
        _refresh(scheduler, args)
        headroom = int(scheduler.settings["headroom_mb"])
        nodes = [
            node.to_dict(headroom)
            for node in dbm.get_nodes(
                scheduler.conn,
                include_ignored=bool(getattr(args, "include_ignored", False)),
            )
        ]
        if args.json:
            _print_json({"nodes": nodes})
        else:
            print(_render_nodes({"nodes": nodes}, procs=bool(args.procs)))
        return 0
    finally:
        scheduler.close()


def _render_lanes(lanes: Sequence[Dict[str, Any]]) -> str:
    """One row per lane: what it is running, what is next, and what stops it."""
    if not lanes:
        return "no lanes yet (run `nvidb queue tick` to discover the GPUs)"
    rows = []
    for lane in lanes:
        running = lane["running"]
        queued = lane["queued"]
        state = "paused" if lane["paused"] else lane["node_state"]
        head = queued[0] if queued else None
        if not lane["runner"]:
            runner = "off"
        elif lane["runner_up"]:
            runner = "up"
        elif (running or queued) and lane["node_state"] == "up":
            # Only worth calling out where a runner could be up: on a node that
            # is down it is the node that needs attention, and the BLOCKED
            # column already says so.
            runner = "DOWN"
        else:
            runner = "-"
        rows.append(
            [
                lane["name"],
                state,
                runner,
                str(len(running)),
                str(len(queued)),
                _short(running[0]["name"] or str(running[0]["id"]), 22)
                if running
                else "-",
                _short(f"{head['id']} {head['name'] or ''}", 22) if head else "-",
                _short(lane["blocked"], 40) if lane["blocked"] else "-",
            ]
        )
    return _table(
        rows,
        ["LANE", "STATE", "RUNNER", "RUN", "QUEUED", "NOW", "NEXT", "BLOCKED"],
        indent="",
    )


def _render_lane_detail(lane: Dict[str, Any]) -> str:
    if not lane["runner"]:
        runner = "runner off"
    elif lane["runner_up"]:
        runner = "runner up"
    else:
        runner = "no runner"
    lines = [
        f"LANE {lane['name']}  [{'paused' if lane['paused'] else lane['node_state']}]"
        f"  node={lane['node']} gpu={','.join(str(i) for i in lane['gpu_ids'])}"
        f"  {runner}"
        + (f"  concurrency={lane['concurrency']}" if lane["concurrency"] != 1 else "")
    ]
    if lane["blocked"]:
        lines.append(f"  blocked: {lane['blocked']}")
    lines.append("")
    if lane["running"]:
        lines.append("RUNNING")
        lines.append(_job_table(lane["running"]))
        lines.append("")
    # The queue is printed in the order it will run, numbered, because the
    # number is what `lane move --to N` takes.
    lines.append("QUEUED (in order)")
    if not lane["queued"]:
        lines.append("  (none)")
    else:
        rows = [
            [
                str(position),
                str(job["id"]),
                "held" if job.get("held") else "pending",
                _short(job["name"] or "-", 24),
                format_mb(job["vram_mb"]) if job["vram_mb"] else "-",
                _short(job.get("held_reason") or job["command"], 50),
            ]
            for position, job in enumerate(lane["queued"], start=1)
        ]
        lines.append(_table(rows, ["#", "ID", "STATE", "NAME", "VRAM", "WHAT"]))
    return "\n".join(lines)


def cmd_lanes(args) -> int:
    scheduler = _open(args)
    try:
        _refresh(scheduler, args)
        node = dbm.resolve_node_name(scheduler.conn, args.node) if args.node else None
        lanes = scheduler.lanes(node=node)
        if args.json:
            _print_json({"lanes": lanes})
        else:
            print(_render_lanes(lanes))
        return 0
    finally:
        scheduler.close()


def cmd_lane_show(args) -> int:
    scheduler = _open(args)
    try:
        _refresh(scheduler, args)
        try:
            lane = scheduler.resolve_lane(args.lane)
        except ValueError as error:
            return _error(str(error), as_json=args.json)
        view = scheduler.lane_view(lane)
        if args.json:
            _print_json(view)
        else:
            print(_render_lane_detail(view))
        return 0
    finally:
        scheduler.close()


def cmd_lane_move(args) -> int:
    scheduler = _open(args)
    try:
        try:
            outcome = scheduler.lane_move(
                args.id, to=args.to, before=args.before, after=args.after
            )
        except (ValueError, RuntimeError) as error:
            return _error(str(error), as_json=args.json)
        if args.json:
            _print_json({"ok": True, "id": args.id, **outcome})
        else:
            print(
                f"job {args.id} is now #{outcome['position']} in {outcome['lane']}"
            )
        return 0
    finally:
        scheduler.close()


def cmd_lane_swap(args) -> int:
    scheduler = _open(args)
    try:
        try:
            outcome = scheduler.lane_swap(args.first, args.second)
        except (ValueError, RuntimeError) as error:
            return _error(str(error), as_json=args.json)
        if args.json:
            _print_json({"ok": True, **outcome})
        else:
            print(f"swapped jobs {args.first} and {args.second} in {outcome['lane']}")
        return 0
    finally:
        scheduler.close()


def cmd_lane_assign(args) -> int:
    scheduler = _open(args)
    try:
        try:
            outcome = scheduler.lane_assign(
                args.id, None if args.none else args.lane, at=args.at
            )
        except (ValueError, RuntimeError) as error:
            return _error(str(error), as_json=args.json)
        summary = _maybe_tick(scheduler, args, force=True)
        if args.json:
            _print_json({"ok": True, "id": args.id, **outcome, "tick": summary})
        elif outcome["lane"] is None:
            print(f"job {args.id} left lane {outcome['previous'] or '-'}")
        else:
            where = f" at #{outcome['position']}" if outcome.get("position") else ""
            print(f"job {args.id} queued in {outcome['lane']}{where}")
        return 0
    finally:
        scheduler.close()


def cmd_lane_pause(args) -> int:
    scheduler = _open(args)
    try:
        paused = args.action == "pause"
        try:
            lane = scheduler.set_lane_paused(args.lane, paused)
        except (ValueError, RuntimeError) as error:
            return _error(str(error), as_json=args.json)
        summary = {} if paused else _maybe_tick(scheduler, args, force=True)
        if args.json:
            _print_json({"ok": True, "lane": lane.to_dict(), "tick": summary})
        else:
            print(
                f"lane {lane.name} {'paused' if paused else 'resumed'}"
                + (
                    "; jobs already running are left alone"
                    if paused
                    else ""
                )
            )
        return 0
    finally:
        scheduler.close()


def cmd_lane_skip(args) -> int:
    scheduler = _open(args)
    try:
        try:
            outcome = scheduler.lane_skip(args.lane)
        except (ValueError, RuntimeError) as error:
            return _error(str(error), as_json=args.json)
        summary = _maybe_tick(scheduler, args, force=True)
        if args.json:
            _print_json({"ok": True, **outcome, "tick": summary})
        elif outcome["skipped"] is None:
            print(f"lane {outcome['lane']} has nothing to skip past")
        else:
            print(f"job {outcome['skipped']} moved to the back of {outcome['lane']}")
        return 0
    finally:
        scheduler.close()


def cmd_lane_runner(args) -> int:
    """Inspect or control the resident process that drives one lane."""
    scheduler = _open(args)
    try:
        action = getattr(args, "runner_action", None) or "status"
        try:
            if action == "status":
                _refresh(scheduler, args)
                lane = scheduler.resolve_lane(args.lane)
                view = scheduler.lane_view(lane)
                if args.json:
                    _print_json(
                        {
                            "lane": lane.name,
                            "enabled": lane.runner,
                            "up": lane.runner_up(),
                            "seen": lane.runner_seen_at,
                            "state": lane.runner_state,
                            "blocked": view["blocked"],
                        }
                    )
                else:
                    if not lane.runner:
                        print(f"lane {lane.name}: runner disabled")
                    elif lane.runner_up():
                        state = lane.runner_state or {}
                        current = ", ".join(
                            str(item.get("job_id")) for item in state.get("current") or []
                        )
                        print(
                            f"lane {lane.name}: runner up (pid {state.get('pid', '?')}, "
                            f"seen {_age(lane.runner_seen_at)})"
                        )
                        print(f"  running  {current or '-'}")
                        print(
                            "  staged   "
                            + (
                                ", ".join(str(i) for i in state.get("queued") or [])
                                or "-"
                            )
                        )
                    else:
                        print(f"lane {lane.name}: no runner (it starts when work is queued)")
                return 0 if lane.runner_up() or not lane.runner else 1

            if action in ("on", "off"):
                lane = scheduler.set_lane_runner(args.lane, action == "on")
                summary = _maybe_tick(scheduler, args, force=True) if action == "on" else {}
                if args.json:
                    _print_json({"ok": True, "lane": lane.to_dict(), "tick": summary})
                else:
                    print(
                        f"lane {lane.name}: runner "
                        + ("enabled" if lane.runner else "disabled")
                    )
                return 0

            outcome = scheduler.stop_lane_runner(args.lane, hard=args.hard)
            if action == "restart":
                summary = _maybe_tick(scheduler, args, force=True)
                if args.json:
                    _print_json({"ok": True, **outcome, "tick": summary})
                else:
                    print(f"lane {outcome['lane']}: runner restarted")
                return 0
            if args.json:
                _print_json({"ok": True, **outcome})
            else:
                print(
                    f"lane {outcome['lane']}: runner asked to stop"
                    + ("" if args.hard else " once its current job finishes")
                )
            return 0
        except (ValueError, RuntimeError) as error:
            return _error(str(error), as_json=args.json)
        except Exception as error:  # a node that will not answer is not a crash
            return _error(f"{type(error).__name__}: {error}", as_json=args.json)
    finally:
        scheduler.close()


def cmd_lane_concurrency(args) -> int:
    scheduler = _open(args)
    try:
        try:
            lane = scheduler.set_lane_concurrency(args.lane, args.value)
        except (ValueError, RuntimeError) as error:
            return _error(str(error), as_json=args.json)
        if args.json:
            _print_json({"ok": True, "lane": lane.to_dict()})
        else:
            print(f"lane {lane.name} runs up to {lane.concurrency} job(s) at once")
        return 0
    finally:
        scheduler.close()


def cmd_node_set(args) -> int:
    scheduler = _open(args)
    try:
        name = dbm.resolve_node_name(scheduler.conn, args.name)
        if name is None:
            return _error(f"unknown node {args.name!r}", as_json=args.json)
        enabled = args.action == "resume"
        try:
            scheduler.set_node_enabled(name, enabled)
        except RuntimeError as error:
            return _error(str(error), as_json=args.json)
        payload = {"ok": True, "node": name, "enabled": enabled}
        if args.json:
            _print_json(payload)
        else:
            print(f"node {name} {'resumed' if enabled else 'drained'}")
        return 0
    finally:
        scheduler.close()


def cmd_node_ignore(args) -> int:
    scheduler = _open(args)
    try:
        name = dbm.resolve_node_name(scheduler.conn, args.name)
        if name is None:
            return _error(f"unknown node {args.name!r}", as_json=args.json)
        ignored = args.action == "ignore"
        try:
            scheduler.set_node_ignored(name, ignored)
        except (RuntimeError, ValueError) as error:
            return _error(str(error), as_json=args.json)
        payload = {"ok": True, "node": name, "ignored": ignored}
        if args.json:
            _print_json(payload)
        else:
            print(f"node {name} {'ignored' if ignored else 'unignored'}")
        return 0
    finally:
        scheduler.close()


def cmd_submit(args) -> int:
    scheduler = _open(args)
    try:
        if args.script:
            script_path = Path(args.script).expanduser()
            if not script_path.exists():
                return _error(f"script not found: {script_path}", as_json=args.json)
            command = script_path.read_text(encoding="utf-8")
        elif len(args.command_parts) == 1:
            command = args.command_parts[0]
        else:
            command = shlex.join(args.command_parts) if args.command_parts else ""
        if not command.strip():
            return _error("nothing to run (pass a command, or --script FILE)", as_json=args.json)

        env = {}
        for item in args.env or []:
            if "=" not in item:
                return _error(f"--env expects KEY=VALUE, got {item!r}", as_json=args.json)
            key, value = item.split("=", 1)
            key = key.strip()
            # A name no shell can export would otherwise be dropped on the way
            # to the node, and the job would run without it saying anything.
            if not VALID_ENV_KEY.match(key):
                return _error(
                    f"--env name {key!r} is not a valid shell variable name "
                    "(letters, digits and underscore; not starting with a digit)",
                    as_json=args.json,
                )
            env[key] = value

        # A zero or negative limit would be stored as "no limit at all", which is
        # the opposite of what someone typing `--timeout 0` is asking for.
        if args.timeout is not None and args.timeout <= 0:
            return _error("--timeout must be a positive number of seconds", as_json=args.json)
        if args.gpus < 0:
            return _error("--gpus cannot be negative (use 0 for CPU-only work)", as_json=args.json)
        if args.retries < 0:
            return _error("--retries cannot be negative", as_json=args.json)

        try:
            job_id = scheduler.submit(
                command,
                name=args.name or "",
                workdir=str(Path(args.workdir).expanduser()) if args.workdir else None,
                env=env,
                gpus=args.gpus,
                vram=args.vram,
                priority=args.priority,
                node=args.node,
                depends_on=_resolve_ids(scheduler, args.after or []),
                depends_any=_resolve_ids(scheduler, args.after_any or []),
                max_runtime_s=args.timeout,
                submitter=args.submitter or _default_submitter(),
                tags=args.tag or [],
                max_retries=args.retries,
                notes=args.note,
                lane=args.lane,
                at=args.at,
            )
        except ValueError as error:
            return _error(str(error), as_json=args.json)

        summary = _maybe_tick(scheduler, args, force=True)
        job = dbm.get_job(scheduler.conn, job_id)

        if args.wait:
            job = _wait_for(scheduler, [job_id], timeout=args.wait_timeout)[0]

        if args.json:
            _print_json({"ok": True, "job": job.to_dict(), "tick": summary})
        else:
            print(f"submitted job {job_id} ({job.display_state})")
            if job.node and job.state == "running":
                print(f"  running on {job.node} GPU {','.join(str(i) for i in job.gpu_ids) or '-'}")
            elif job.lane:
                queued = [
                    item.id
                    for item in dbm.lane_jobs(scheduler.conn, job.lane, states=["pending"])
                ]
                position = queued.index(job_id) + 1 if job_id in queued else len(queued)
                print(f"  queued in lane {job.lane} at #{position} of {len(queued)}")
            elif job.state == "pending":
                print("  waiting for capacity; it starts on the next tick that fits it")
        if args.wait:
            # Same three-way status as `job wait`, for the same reason.
            if not job.is_terminal:
                return 2
            return 0 if job.state == "completed" else 1
        return 0 if job.state != "failed" else 1
    finally:
        scheduler.close()


def cmd_list(args) -> int:
    scheduler = _open(args)
    try:
        _refresh(scheduler, args)
        states = args.state or None
        if args.active:
            states = ["pending", "running"]
        if getattr(args, "held", False):
            jobs = dbm.held_jobs(scheduler.conn)
        else:
            jobs = dbm.list_jobs(
                scheduler.conn,
                states=states,
                node=dbm.resolve_node_name(scheduler.conn, args.node) if args.node else None,
                limit=args.number,
                newest_first=bool(args.number),
            )
            if args.number:
                jobs = list(reversed(jobs))
        if args.json:
            _print_json({"jobs": [job.to_dict() for job in jobs]})
        else:
            print(_job_table([job.to_dict() for job in jobs], indent=""))
        return 0
    finally:
        scheduler.close()


def cmd_show(args) -> int:
    scheduler = _open(args)
    try:
        _refresh(scheduler, args)
        job = dbm.get_job(scheduler.conn, args.id)
        if job is None:
            return _error(f"job {args.id} not found", as_json=args.json)
        logs = None
        if args.logs:
            try:
                logs = scheduler.job_logs(job.id, lines=args.logs)
            except Exception as error:
                logs = f"(log unavailable: {error})"
        if args.json:
            payload = job.to_dict()
            payload["events"] = dbm.list_events(scheduler.conn, job_id=job.id, limit=50)
            if logs is not None:
                payload["log_tail"] = logs
            _print_json(payload)
        else:
            print(_render_job_detail(job, logs=logs))
        return 0
    finally:
        scheduler.close()


def cmd_logs(args) -> int:
    scheduler = _open(args)
    try:
        _refresh(scheduler, args)
        job = dbm.get_job(scheduler.conn, args.id)
        if job is None:
            return _error(f"job {args.id} not found", as_json=args.json)
        stream = "stderr" if args.stderr else "stdout"
        if not args.follow:
            text = scheduler.job_logs(job.id, stream=stream, lines=args.number)
            if args.json:
                _print_json({"id": job.id, "stream": stream, "log": text})
            else:
                sys.stdout.write(text)
                if text and not text.endswith("\n"):
                    sys.stdout.write("\n")
            return 0

        seen = ""
        try:
            while True:
                text = scheduler.job_logs(job.id, stream=stream, lines=args.number)
                if text.startswith(seen):
                    sys.stdout.write(text[len(seen):])
                else:
                    sys.stdout.write(text)
                sys.stdout.flush()
                seen = text
                job = dbm.get_job(scheduler.conn, job.id)
                if job is None or job.is_terminal:
                    break
                time.sleep(2)
                scheduler.tick()
        except KeyboardInterrupt:
            pass
        return 0
    finally:
        scheduler.close()


def _wait_for(scheduler: Scheduler, job_ids: Sequence[int], *, timeout: Optional[int]) -> List[Job]:
    """Poll the database until every job reaches a terminal state."""
    deadline = None if not timeout else time.time() + timeout
    while True:
        scheduler.tick()
        jobs = [dbm.get_job(scheduler.conn, job_id) for job_id in job_ids]
        missing = [job_id for job_id, job in zip(job_ids, jobs) if job is None]
        if missing:
            raise ValueError(f"job ids disappeared while waiting: {missing}")
        if all(job.is_terminal for job in jobs):
            return jobs
        if deadline is not None and time.time() >= deadline:
            return jobs
        time.sleep(2)


def cmd_wait(args) -> int:
    scheduler = _open(args)
    try:
        ids = _resolve_ids(scheduler, args.ids)
        missing = [job_id for job_id in ids if dbm.get_job(scheduler.conn, job_id) is None]
        if missing:
            return _error(f"unknown job ids: {missing}", as_json=args.json)
        try:
            jobs = _wait_for(scheduler, ids, timeout=args.timeout)
        except KeyboardInterrupt:
            return 130
        except ValueError as error:
            return _error(str(error), as_json=args.json)
        payload = {"jobs": []}
        for job in jobs:
            entry = job.to_dict()
            if args.logs:
                try:
                    entry["log_tail"] = scheduler.job_logs(job.id, lines=args.logs)
                except Exception:
                    entry["log_tail"] = None
            payload["jobs"].append(entry)
        payload["all_done"] = all(job.is_terminal for job in jobs)
        payload["all_succeeded"] = all(job.state == "completed" for job in jobs)
        payload["timed_out"] = not payload["all_done"]
        if args.json:
            _print_json(payload)
        else:
            print(_job_table(payload["jobs"], indent=""))
            if payload["timed_out"]:
                print(
                    "wait timed out; the job(s) above are still going",
                    file=sys.stderr,
                )
        # 0 succeeded, 1 finished badly, 2 still running. Giving the timeout its
        # own status is what lets a caller tell "this failed" from "I stopped
        # waiting", which are opposite things to do next.
        if payload["timed_out"]:
            return 2
        return 0 if payload["all_succeeded"] else 1
    finally:
        scheduler.close()


def cmd_cancel(args) -> int:
    scheduler = _open(args)
    try:
        results = []
        try:
            for job_id in _resolve_ids(scheduler, args.ids):
                # Read the dependents first: once the job is cancelled its
                # dependents are held, and the point of saying so is to name
                # what this command is about to stop.
                waiting = [
                    dependent.id
                    for dependent in dbm.dependents_of(scheduler.conn, job_id)
                    if job_id in dependent.depends_on
                ]
                ok = scheduler.cancel(job_id)
                results.append(
                    {"id": job_id, "cancelled": ok, "holds": waiting if ok else []}
                )
        except RuntimeError as error:
            return _error(str(error), as_json=args.json)
        if args.json:
            _print_json({"results": results})
        else:
            for item in results:
                print(
                    f"job {item['id']}: "
                    + ("cancelled" if item["cancelled"] else "not cancellable")
                )
                if item["holds"]:
                    ids = ", ".join(str(i) for i in item["holds"])
                    print(
                        f"  job(s) {ids} were waiting for it and are now held. "
                        "They keep their place in the queue; release or repoint "
                        f"them with `nvidb job release {item['holds'][0]}` or "
                        f"`nvidb job edit {item['holds'][0]} --add-after ID`."
                    )
        return 0
    finally:
        scheduler.close()


def cmd_release(args) -> int:
    """Let held jobs run by forgetting the prerequisites that died."""
    scheduler = _open(args)
    try:
        results = []
        try:
            for job_id in _resolve_ids(scheduler, args.ids):
                results.append({"id": job_id, **scheduler.release(job_id)})
        except (ValueError, RuntimeError) as error:
            return _error(str(error), as_json=args.json)
        summary = _maybe_tick(scheduler, args, force=True)
        if args.json:
            _print_json({"results": results, "tick": summary})
        else:
            for item in results:
                print(f"job {item['id']}: {item['reason']}")
        return 0
    finally:
        scheduler.close()


def cmd_edit(args) -> int:
    """Rewire a job's dependencies without resubmitting it."""
    scheduler = _open(args)
    try:
        changes = {
            "add": _resolve_ids(scheduler, args.add_after or []),
            "drop": _resolve_ids(scheduler, args.drop_after or []),
            "add_any": _resolve_ids(scheduler, args.add_after_any or []),
            "drop_any": _resolve_ids(scheduler, args.drop_after_any or []),
        }
        if not any(changes.values()):
            return _error(
                "nothing to change (pass --add-after, --drop-after, "
                "--add-after-any or --drop-after-any)",
                as_json=args.json,
            )
        try:
            job = scheduler.edit_dependencies(args.id, **changes)
        except ValueError as error:
            return _error(str(error), as_json=args.json)
        summary = _maybe_tick(scheduler, args, force=True)
        job = dbm.get_job(scheduler.conn, args.id) or job
        if args.json:
            _print_json({"ok": True, "job": job.to_dict(), "tick": summary})
        else:
            after = ",".join(str(i) for i in job.depends_on) or "-"
            after_any = ",".join(str(i) for i in job.depends_any) or "-"
            print(f"job {job.id} ({job.display_state}): after={after} after-any={after_any}")
            if job.held_reason:
                print(f"  still held: {job.held_reason}")
        return 0
    finally:
        scheduler.close()


def cmd_requeue(args) -> int:
    scheduler = _open(args)
    try:
        results = []
        try:
            for job_id in _resolve_ids(scheduler, args.ids):
                results.append({"id": job_id, "requeued": scheduler.requeue(job_id)})
        except RuntimeError as error:
            return _error(str(error), as_json=args.json)
        summary = _maybe_tick(scheduler, args, force=True)
        if args.json:
            _print_json({"results": results, "tick": summary})
        else:
            for item in results:
                print(
                    f"job {item['id']}: "
                    + ("requeued" if item["requeued"] else "not requeueable")
                )
        return 0
    finally:
        scheduler.close()


def cmd_result(args) -> int:
    scheduler = _open(args)
    try:
        job = dbm.get_job(scheduler.conn, args.id)
        if job is None:
            return _error(f"job {args.id} not found", as_json=args.json)
        if args.set is not None:
            try:
                value = json.loads(args.set)
            except ValueError:
                value = args.set
            dbm.update_job(scheduler.conn, job.id, result=value)
            job = dbm.get_job(scheduler.conn, job.id)
        if args.json:
            _print_json({"id": job.id, "state": job.state, "result": job.result})
        else:
            print(json.dumps(job.result, ensure_ascii=False, indent=2) if job.result is not None else "(no result)")
        return 0
    finally:
        scheduler.close()


def cmd_note(args) -> int:
    scheduler = _open(args)
    try:
        replacement = " ".join(args.text) if args.text else None
        given = [
            name
            for name, value in (
                ("text", replacement),
                ("--append", args.append),
                ("--clear", args.clear or None),
            )
            if value
        ]
        if len(given) > 1:
            return _error(
                f"pick one of {', '.join(given)}", as_json=args.json
            )

        if not given:
            # No argument at all is a read, not an accidental erase.
            job = dbm.get_job(scheduler.conn, args.id)
            if job is None:
                return _error(f"job {args.id} not found", as_json=args.json)
            if args.json:
                _print_json({"id": job.id, "notes": job.notes, "progress": job.progress})
            else:
                print(job.notes or "(no note)")
            return 0

        text = None if args.clear else (args.append or replacement)
        try:
            value = scheduler.set_notes(args.id, text, append=bool(args.append))
        except ValueError as error:
            return _error(str(error), as_json=args.json)
        if args.json:
            _print_json({"ok": True, "id": args.id, "notes": value})
        else:
            print(f"job {args.id} note: {value or '(cleared)'}")
        return 0
    finally:
        scheduler.close()


def cmd_priority(args) -> int:
    scheduler = _open(args)
    try:
        if args.up or args.down:
            slots = args.up or args.down
            moved = scheduler.move_pending(args.id, -slots if args.up else slots)
            job = dbm.get_job(scheduler.conn, args.id)
            if args.json:
                _print_json(
                    {
                        "id": args.id,
                        "moved": moved,
                        "priority": job.priority if job else None,
                    }
                )
            else:
                direction = "up" if args.up else "down"
                print(
                    f"job {args.id}: "
                    + (f"moved {direction}" if moved else "not moved (not pending?)")
                )
            return 0

        if args.value is None:
            job = dbm.get_job(scheduler.conn, args.id)
            if job is None:
                return _error(f"job {args.id} not found", as_json=args.json)
            if args.json:
                _print_json({"id": job.id, "priority": job.priority, "state": job.state})
            else:
                print(f"job {job.id} priority: {job.priority}")
            return 0

        text = str(args.value)
        try:
            relative = text.startswith(("+", "-")) and text not in ("+", "-")
            amount = int(text)
        except ValueError:
            return _error(
                f"invalid priority {text!r} (use 5, +1, or -2)", as_json=args.json
            )
        try:
            if relative:
                value = scheduler.adjust_priority(args.id, amount)
            else:
                value = scheduler.set_priority(args.id, amount)
        except ValueError as error:
            return _error(str(error), as_json=args.json)
        if value is None:
            return _error(
                f"job {args.id} is finished; its priority no longer matters",
                as_json=args.json,
            )
        if args.json:
            _print_json({"ok": True, "id": args.id, "priority": value})
        else:
            print(f"job {args.id} priority: {value}")
        return 0
    finally:
        scheduler.close()


def cmd_purge(args) -> int:
    scheduler = _open(args)
    try:
        removed = dbm.purge_jobs(scheduler.conn, states=args.state or None)
        if args.json:
            _print_json({"removed": removed})
        else:
            print(f"removed {removed} finished job record(s)")
        return 0
    finally:
        scheduler.close()


def cmd_tui(args) -> int:
    from .tui import run_tui

    return run_tui(db_path=getattr(args, "db_path", None), refresh=args.refresh)


# --- parser wiring ---------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser, *, json_flag: bool = True) -> None:
    if json_flag:
        parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--db-path", default=None, help="Queue database path")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use this machine's own queue even when a remote queue host is configured",
    )


def _add_refresh_flags(parser: argparse.ArgumentParser) -> None:
    """Reads refresh on their own; these flags force or suppress that."""
    parser.add_argument(
        "--tick", action="store_true", help="Force a refresh even if one just ran"
    )
    parser.add_argument(
        "--no-tick", action="store_true", help="Read the database without contacting nodes"
    )


def _add_procs_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--procs",
        action="store_true",
        help="List every process on each GPU, queue-managed or not",
    )


def _add_ignored_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Include nodes that were administratively ignored",
    )


def register_parsers(subparsers) -> None:
    """Attach the `queue` and `job` command trees to the main nvidb parser."""
    queue = subparsers.add_parser(
        "queue", help="Cluster job queue: status, TUI, scheduler tick"
    )
    _add_common(queue)
    queue.add_argument(
        "--refresh", type=float, default=3.0, help="TUI refresh interval in seconds"
    )
    # `queue_cli` is how the main entry point recognises these commands, rather
    # than matching on a subcommand name that a positional could shadow.
    queue.set_defaults(func=cmd_tui, queue_cli=True)
    queue_sub = queue.add_subparsers(dest="queue_command")

    status = queue_sub.add_parser("status", help="Show nodes, capacity and jobs")
    _add_common(status)
    _add_refresh_flags(status)
    _add_procs_flag(status)
    _add_ignored_flag(status)
    status.set_defaults(func=cmd_status)

    tick = queue_sub.add_parser("tick", help="Run one scheduler pass")
    _add_common(tick)
    tick.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                      help="Keep ticking every N seconds until interrupted")
    tick.set_defaults(func=cmd_tick)

    alerts = queue_sub.add_parser(
        "alerts",
        help="Show failures that still need attention",
        description=(
            "Lists what went wrong on the nodes. Exits non-zero while any alert "
            "is unacknowledged, so a shell or an agent can branch on it."
        ),
    )
    _add_common(alerts)
    _add_refresh_flags(alerts)
    alerts.add_argument("--all", action="store_true", help="Include acknowledged alerts")
    alerts.add_argument("--detail", action="store_true", help="Show the captured output")
    alerts.add_argument("-n", "--number", type=int, default=50)
    alerts.set_defaults(func=cmd_alerts)

    ack = queue_sub.add_parser("ack", help="Acknowledge alerts")
    _add_common(ack)
    ack.add_argument("ids", nargs="*", help="Alert ids to acknowledge")
    ack.add_argument("--all", action="store_true", help="Acknowledge every open alert")
    ack.set_defaults(func=cmd_ack)

    daemon = queue_sub.add_parser(
        "daemon",
        help="Run the scheduler continuously and deliver alerts",
        description=(
            "Optional: the queue works without it, because every command runs a "
            "scheduler pass. Run the daemon when you want failures pushed to you "
            "instead of waiting to be asked."
        ),
    )
    _add_common(daemon)
    daemon.add_argument("--interval", type=int, default=15, metavar="SECONDS")
    daemon.add_argument("--no-notify", action="store_true", help="Tick but stay quiet")
    daemon.add_argument(
        "--no-backup", action="store_true", help="Skip configured periodic backups"
    )
    daemon.add_argument("--once", action="store_true", help="Run a single pass and exit")
    daemon.set_defaults(func=cmd_daemon)

    backup = queue_sub.add_parser(
        "backup",
        help="Create and verify a consistent snapshot of the queue database",
    )
    _add_common(backup)
    backup.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Destination file or existing directory (default: $NVIDB_HOME/backups)",
    )
    backup.add_argument(
        "--keep",
        type=int,
        default=0,
        metavar="N",
        help="Keep the newest N generated backups (default: 0, keep all)",
    )
    backup.set_defaults(func=cmd_backup)

    keeper = queue_sub.add_parser(
        "keeper",
        help="Install and control the supervisor that keeps this queue moving",
        description=(
            "Uses a portable detached shell supervisor by default. Pass "
            "--systemd while installing to create a user service instead; "
            "--start enables it so the daemon can return after a host reboot."
        ),
    )
    _add_common(keeper)
    keeper.add_argument(
        "keeper_action",
        nargs="?",
        default="status",
        choices=["install", "start", "ensure", "stop", "restart", "status", "logs"],
    )
    keeper.add_argument(
        "--nvidb",
        default=None,
        metavar="PATH",
        help="The nvidb executable to run (default: the one on PATH, resolved to an absolute path)",
    )
    keeper.add_argument(
        "--interval",
        type=int,
        default=keeper_mod.DEFAULT_INTERVAL,
        metavar="SECONDS",
        help="Seconds between scheduler passes",
    )
    keeper.add_argument(
        "--start", action="store_true", help="Start it straight after installing"
    )
    keeper.add_argument(
        "--systemd",
        action="store_true",
        help="Install a systemd user service instead of the detached shell supervisor",
    )
    keeper.add_argument("-n", "--number", type=int, default=50, help="Log lines to show")
    keeper.set_defaults(func=cmd_keeper)

    events = queue_sub.add_parser("events", help="Replay the queue event log")
    _add_common(events)
    events.add_argument("--since", type=int, default=None, help="Only events after this id")
    events.add_argument("--job", type=int, default=None, help="Only events for this job")
    events.add_argument("-n", "--number", type=int, default=50, help="Maximum events")
    events.set_defaults(func=cmd_events)

    nodes = queue_sub.add_parser(
        "nodes",
        help="Show node capacity, including GPU work the queue does not manage",
    )
    _add_common(nodes)
    _add_procs_flag(nodes)
    _add_refresh_flags(nodes)
    _add_ignored_flag(nodes)
    nodes.set_defaults(func=cmd_nodes)

    lanes = queue_sub.add_parser(
        "lanes",
        help="Show every GPU's serial queue: what it runs now and what is next",
    )
    _add_common(lanes)
    _add_refresh_flags(lanes)
    lanes.add_argument("--node", default=None, help="Only lanes on this node")
    lanes.set_defaults(func=cmd_lanes)

    lane = queue_sub.add_parser(
        "lane",
        help="Inspect and reorder one GPU's serial queue",
        description=(
            "A lane is one GPU and the order it runs its jobs in. With no "
            "subcommand the lane's queue is printed in running order; the "
            "subcommands edit that order."
        ),
    )
    _add_common(lane)
    _add_refresh_flags(lane)
    lane.add_argument("lane", help="Lane name or an unambiguous part of one")
    lane.set_defaults(func=cmd_lane_show)
    lane_sub = lane.add_subparsers(dest="lane_command")

    lane_move = lane_sub.add_parser(
        "move",
        help="Move a queued job within its lane",
        description=(
            "Positions are 1-based over the queued jobs, so `--to 1` means "
            "next to run. A running job cannot be moved."
        ),
    )
    _add_common(lane_move)
    lane_move.add_argument("id", type=int)
    lane_move.add_argument("--to", default=None, metavar="head|tail|N")
    lane_move.add_argument("--before", type=int, default=None, metavar="ID")
    lane_move.add_argument("--after", type=int, default=None, metavar="ID")
    lane_move.set_defaults(func=cmd_lane_move)

    lane_swap = lane_sub.add_parser("swap", help="Exchange two queued jobs' places")
    _add_common(lane_swap)
    lane_swap.add_argument("first", type=int)
    lane_swap.add_argument("second", type=int)
    lane_swap.set_defaults(func=cmd_lane_swap)

    lane_assign = lane_sub.add_parser(
        "assign", help="Put a queued job into this lane, or take it out"
    )
    _add_common(lane_assign)
    lane_assign.add_argument("id", type=int)
    lane_assign.add_argument("--at", default="tail", metavar="head|tail|N")
    lane_assign.add_argument(
        "--none",
        action="store_true",
        help="Remove the job from its lane and let the scheduler place it",
    )
    lane_assign.add_argument("--no-tick", action="store_true")
    lane_assign.set_defaults(func=cmd_lane_assign)

    lane_pause = lane_sub.add_parser(
        "pause",
        help="Stop starting new jobs here; what is running is left alone",
    )
    _add_common(lane_pause)
    lane_pause.set_defaults(func=cmd_lane_pause, action="pause")

    lane_resume = lane_sub.add_parser("resume", help="Start running this lane again")
    _add_common(lane_resume)
    lane_resume.add_argument("--no-tick", action="store_true")
    lane_resume.set_defaults(func=cmd_lane_pause, action="resume")

    lane_skip = lane_sub.add_parser(
        "skip", help="Send the next job to the back of the lane"
    )
    _add_common(lane_skip)
    lane_skip.add_argument("--no-tick", action="store_true")
    lane_skip.set_defaults(func=cmd_lane_skip)

    lane_runner = lane_sub.add_parser(
        "runner",
        help="The resident process on the node that starts this lane's jobs",
        description=(
            "With a runner, the lane starts its next job the moment the "
            "previous one exits, with no client involved. Stopping or "
            "restarting the runner never touches the job it started."
        ),
    )
    _add_common(lane_runner)
    _add_refresh_flags(lane_runner)
    lane_runner.add_argument(
        "runner_action",
        nargs="?",
        default="status",
        choices=["status", "on", "off", "stop", "restart"],
    )
    lane_runner.add_argument(
        "--hard",
        action="store_true",
        help="Stop the runner now instead of after its current job",
    )
    lane_runner.set_defaults(func=cmd_lane_runner)

    lane_concurrency = lane_sub.add_parser(
        "concurrency",
        help="How many of this lane's jobs may run at once (default 1)",
    )
    _add_common(lane_concurrency)
    lane_concurrency.add_argument("value", type=int)
    lane_concurrency.set_defaults(func=cmd_lane_concurrency)

    drain = queue_sub.add_parser("drain", help="Stop scheduling new jobs onto a node")
    _add_common(drain)
    drain.add_argument("name")
    drain.set_defaults(func=cmd_node_set, action="drain")

    resume = queue_sub.add_parser("resume", help="Allow scheduling onto a node again")
    _add_common(resume)
    resume.add_argument("name")
    resume.set_defaults(func=cmd_node_set, action="resume")

    ignore = queue_sub.add_parser(
        "ignore",
        help="Stop probing, scheduling, and normally displaying a node",
    )
    _add_common(ignore)
    ignore.add_argument("name")
    ignore.set_defaults(func=cmd_node_ignore, action="ignore")

    unignore = queue_sub.add_parser(
        "unignore",
        help="Restore a previously ignored node",
    )
    _add_common(unignore)
    unignore.add_argument("name")
    unignore.set_defaults(func=cmd_node_ignore, action="unignore")

    tui = queue_sub.add_parser("tui", help="Open the interactive queue TUI")
    _add_common(tui, json_flag=False)
    tui.add_argument("--refresh", type=float, default=3.0)
    tui.set_defaults(func=cmd_tui)

    job = subparsers.add_parser("job", help="Submit and manage queue jobs")
    job_sub = job.add_subparsers(dest="job_command")
    job.set_defaults(
        queue_cli=True,
        func=lambda args: _error("pick a job subcommand (submit, ls, show, wait, ...)"),
    )

    submit = job_sub.add_parser(
        "submit",
        help="Queue a command",
        description=(
            "Queue a command. Put the command last, after `--`, or pass it as a "
            "single quoted string."
        ),
    )
    _add_common(submit)
    submit.add_argument("-n", "--name", default="", help="Human-readable job name")
    submit.add_argument("--vram", default=None, help="VRAM to reserve, e.g. 20G")
    submit.add_argument("--gpus", type=int, default=1, help="GPUs required (0 for CPU-only)")
    submit.add_argument("--node", default=None, help="Pin to one node (prefix match)")
    submit.add_argument(
        "--lane",
        default=None,
        metavar="NODE:GPU",
        help="Queue on one GPU's serial lane, e.g. --lane box406:0",
    )
    submit.add_argument(
        "--at",
        default="tail",
        metavar="head|tail|N",
        help="Where in the lane to queue it (default: tail)",
    )
    submit.add_argument("--workdir", default=None, help="Working directory on the node")
    submit.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    submit.add_argument("--priority", type=int, default=0, help="Higher runs first")
    submit.add_argument("--after", action="append", default=[], metavar="ID",
                        help="Run only after these jobs complete successfully")
    submit.add_argument("--after-any", action="append", default=[], metavar="ID",
                        help="Run once these jobs finish, whatever the outcome")
    submit.add_argument("--timeout", type=int, default=None, metavar="SECONDS",
                        help="Kill the job after this long")
    submit.add_argument("--retries", type=int, default=0, help="Retries if the process vanishes")
    submit.add_argument("--tag", action="append", default=[], help="Free-form tag")
    submit.add_argument("--submitter", default=None, help="Identify the submitting client")
    submit.add_argument("--note", default=None, help="Free-form annotation for this job")
    submit.add_argument("--script", default=None, metavar="FILE",
                        help="Use a local file's contents as the command body")
    submit.add_argument("--wait", action="store_true", help="Block until the job finishes")
    submit.add_argument("--wait-timeout", type=int, default=None, metavar="SECONDS")
    submit.add_argument("--no-tick", action="store_true", help="Do not schedule right away")
    # `dest` must not be "command": the top-level parser already stores the
    # chosen subcommand there, and a positional of the same name would clobber it.
    submit.add_argument(
        "command_parts", nargs="*", metavar="COMMAND", help="Command to run"
    )
    submit.set_defaults(func=cmd_submit)

    listing = job_sub.add_parser("ls", help="List jobs")
    _add_common(listing)
    listing.add_argument("--state", action="append", choices=list(JOB_STATES), default=[])
    listing.add_argument("--active", action="store_true", help="Only pending and running")
    listing.add_argument(
        "--held",
        action="store_true",
        help="Only jobs parked by a dependency that ended badly",
    )
    listing.add_argument("--node", default=None)
    listing.add_argument("-n", "--number", type=int, default=None, help="Show the newest N")
    _add_refresh_flags(listing)
    listing.set_defaults(func=cmd_list)

    show = job_sub.add_parser("show", help="Show one job in detail")
    _add_common(show)
    show.add_argument("id", type=int)
    show.add_argument("--logs", type=int, nargs="?", const=40, default=0,
                      metavar="LINES", help="Include the tail of stdout")
    _add_refresh_flags(show)
    show.set_defaults(func=cmd_show)

    logs = job_sub.add_parser("logs", help="Read a job's output")
    _add_common(logs)
    logs.add_argument("id", type=int)
    logs.add_argument("-n", "--number", type=int, default=200, help="Lines to show")
    logs.add_argument("--stderr", action="store_true", help="Read stderr instead")
    logs.add_argument("-f", "--follow", action="store_true", help="Keep streaming")
    _add_refresh_flags(logs)
    logs.set_defaults(func=cmd_logs)

    wait = job_sub.add_parser("wait", help="Block until jobs finish")
    _add_common(wait)
    wait.add_argument("ids", nargs="+")
    wait.add_argument("--timeout", type=int, default=None, metavar="SECONDS")
    wait.add_argument("--logs", type=int, nargs="?", const=40, default=0, metavar="LINES")
    wait.set_defaults(func=cmd_wait)

    cancel = job_sub.add_parser("cancel", help="Cancel pending or running jobs")
    _add_common(cancel)
    cancel.add_argument("ids", nargs="+")
    cancel.set_defaults(func=cmd_cancel)

    release = job_sub.add_parser(
        "release",
        help="Run a held job without the dependency that died",
        description=(
            "A job whose prerequisite was cancelled or failed is held, not "
            "failed: it keeps its place and its notes. Releasing it drops the "
            "dead prerequisites so it can be scheduled."
        ),
    )
    _add_common(release)
    release.add_argument("ids", nargs="+")
    release.add_argument("--no-tick", action="store_true")
    release.set_defaults(func=cmd_release)

    edit = job_sub.add_parser(
        "edit",
        help="Change what a job waits for",
        description=(
            "Repoint a queued job at different prerequisites, which is how a "
            "held job is reattached to a re-run of the chain it was waiting on."
        ),
    )
    _add_common(edit)
    edit.add_argument("id", type=int)
    edit.add_argument("--add-after", action="append", default=[], metavar="ID",
                      help="Also wait for this job to complete successfully")
    edit.add_argument("--drop-after", action="append", default=[], metavar="ID",
                      help="Stop waiting for this job")
    edit.add_argument("--add-after-any", action="append", default=[], metavar="ID",
                      help="Also wait for this job to finish, whatever the outcome")
    edit.add_argument("--drop-after-any", action="append", default=[], metavar="ID")
    edit.add_argument("--no-tick", action="store_true")
    edit.set_defaults(func=cmd_edit)

    requeue = job_sub.add_parser("requeue", help="Re-queue finished jobs")
    _add_common(requeue)
    requeue.add_argument("ids", nargs="+")
    requeue.add_argument("--no-tick", action="store_true")
    requeue.set_defaults(func=cmd_requeue)

    result = job_sub.add_parser(
        "result", help="Read or write a job's result payload (a channel between clients)"
    )
    _add_common(result)
    result.add_argument("id", type=int)
    result.add_argument("--set", default=None, metavar="JSON", help="Store this payload")
    result.set_defaults(func=cmd_result)

    note = job_sub.add_parser(
        "note",
        help="Read or write a job's annotation",
        description=(
            "With no text the current note is printed. Notes stay editable "
            "after a job finishes, so a client can record what it concluded."
        ),
    )
    _add_common(note)
    note.add_argument("id", type=int)
    note.add_argument("text", nargs="*", help="Replace the note with this text")
    # `--append` carries its own value: argparse cannot reliably split a bare
    # flag from a following variable-length positional.
    note.add_argument("--append", default=None, metavar="TEXT",
                      help="Add this to the existing note")
    note.add_argument("--clear", action="store_true", help="Remove the note")
    note.set_defaults(func=cmd_note)

    priority = job_sub.add_parser(
        "priority",
        help="Read or change a job's priority / queue position",
        description=(
            "With no value the current priority is printed. A bare number "
            "sets it, +N/-N adjusts it, and --up/--down move a pending job "
            "that many slots in the dispatch order (rewriting the "
            "priorities of the jobs it passes)."
        ),
    )
    _add_common(priority)
    priority.add_argument("id", type=int)
    priority.add_argument("value", nargs="?", default=None,
                          help="New priority: 5 sets, +1/-2 adjust")
    priority.add_argument("--up", type=int, default=0, metavar="N",
                          help="Move a pending job N slots earlier")
    priority.add_argument("--down", type=int, default=0, metavar="N",
                          help="Move a pending job N slots later")
    priority.set_defaults(func=cmd_priority)

    purge = job_sub.add_parser("purge", help="Delete finished job records")
    _add_common(purge)
    purge.add_argument("--state", action="append", choices=sorted(TERMINAL_JOB_STATES), default=[])
    purge.set_defaults(func=cmd_purge)


def quiet_transport_logging() -> None:
    """Keep Paramiko's internal connection tracebacks out of queue output.

    Queue output is parsed by other programs and drawn over by the TUI, so a
    library-generated line is a real problem here. Transport failures are
    already raised to the scheduler as ``TransportError`` and recorded with the
    affected node name, so Paramiko's background-thread ERROR traceback is both
    redundant and missing the context a user needs.
    """
    import logging

    for name in ("paramiko", "paramiko.transport", "paramiko.transport.sftp"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _forward_to_queue_host(target: Dict[str, Any], args) -> int:
    """Hand this invocation to the machine that owns the queue."""
    script_text = None
    if getattr(args, "func", None) is cmd_submit and getattr(args, "script", None):
        # `--script` names a file on *this* machine, so it is read here and the
        # contents travel with the command.
        script_path = Path(args.script).expanduser()
        try:
            script_text = script_path.read_text(encoding="utf-8")
        except OSError as error:
            return _error(f"could not read {script_path}: {error}", as_json=getattr(args, "json", False))
    return remote_mod.forward(
        target,
        sys.argv[1:],
        # The TUI needs a terminal on the far side; everything else is a pipe
        # and must not be given one, or its output would grow carriage returns.
        tty=getattr(args, "func", None) is cmd_tui and sys.stdin.isatty(),
        script_text=script_text,
    )


def dispatch(args) -> int:
    """Run the handler argparse selected, or report a usable error."""
    quiet_transport_logging()
    handler = getattr(args, "func", None)
    if handler is None:
        return _error("no queue subcommand selected")
    if not getattr(args, "local", False):
        target = remote_mod.load_target()
        if target is not None:
            try:
                return _forward_to_queue_host(target, args)
            except KeyboardInterrupt:
                return 130
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # keep JSON consumers from seeing a traceback
        return _error(f"{type(error).__name__}: {error}", as_json=getattr(args, "json", False))
