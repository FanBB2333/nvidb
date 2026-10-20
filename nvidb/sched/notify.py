"""Delivery of queue alerts to wherever the user will actually see them.

Recording an alert and delivering it are deliberately separate. Any client that
runs a scheduler pass records what went wrong, so nothing is lost when nobody is
watching; delivery happens in `nvidb queue daemon`, because a background CLI
call popping up a desktop notification would be a surprise.

Delivery is at-most-once per alert: the daemon claims a batch, pushes it, and
stamps `notified_at`, so restarting it does not replay old failures.
"""
from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import config as nvidb_config

LOGGER = logging.getLogger("nvidb.queue.notify")

DEFAULT_NOTIFY_SETTINGS = {
    "enabled": True,
    "desktop": True,
    "log": True,
    # A shell command receiving the alert as JSON on stdin. This is the hook for
    # anything else - chat, pager, webhook - without nvidb needing to know about it.
    "command": None,
    "min_severity": "warning",  # warning | error
}

SEVERITY_ORDER = {"warning": 0, "error": 1}


def load_notify_settings(queue_settings: Optional[dict] = None) -> Dict[str, Any]:
    raw = (queue_settings or {}).get("notify")
    if isinstance(raw, dict):
        return nvidb_config.merge_settings(DEFAULT_NOTIFY_SETTINGS, raw)

    settings = dict(DEFAULT_NOTIFY_SETTINGS)
    if raw is not None:
        settings["enabled"] = bool(raw)
    return settings


def alert_log_path() -> Path:
    return Path(nvidb_config.WORKING_DIR).expanduser() / "alerts.log"


_ALERT_LABELS = {
    "node_down": "Node unreachable",
    "node_disk_low": "Disk space low",
    "job_lost": "Job process lost",
    "job_retried": "Retrying job",
    "job_unschedulable": "Job cannot be scheduled",
    "job_held": "Job needs attention",
    "job_overdue": "Job overdue",
    "job_timeout": "Job timed out",
    "launch_failed": "Job could not start",
}


def desktop_content(alert: Dict[str, Any]) -> Tuple[str, str, str]:
    """Keep the event, its location, and the explanation on separate lines."""
    kind = str(alert["kind"])
    title = f"nvidb · {_ALERT_LABELS.get(kind, kind.replace('_', ' ').capitalize())}"
    body = str(alert["title"])
    node = str(alert.get("node") or "")
    location = []
    if alert.get("job_id"):
        location.append(f"Job #{alert['job_id']}")
    if kind == "node_down" and node:
        match = re.fullmatch(
            re.escape(node) + r" is unreachable(?:; (\d+) running job\(s\) cannot be checked)?",
            body,
        )
        if match:
            count = match.group(1)
            if count:
                noun = "job" if count == "1" else "jobs"
                body = f"Cannot check {count} running {noun}."
            else:
                body = "Connection to this node failed."
    # Some existing alert titles already include the node; do not repeat it.
    if node and not re.search(r"(?<![\w.-])" + re.escape(node) + r"(?![\w.-])", body):
        location.append(node)
    return title, " · ".join(location), body


class Notifier:
    """Pushes alerts to the configured channels."""

    def __init__(self, settings: Optional[dict] = None):
        self.settings = load_notify_settings(settings)
        self._desktop_command = _find_desktop_command() if self.settings["desktop"] else None

    @property
    def enabled(self) -> bool:
        return self.settings["enabled"]

    def wants(self, alert: Dict[str, Any]) -> bool:
        minimum = SEVERITY_ORDER.get(str(self.settings["min_severity"]), 0)
        return SEVERITY_ORDER.get(str(alert.get("severity", "error")), 1) >= minimum

    def deliver(self, alerts: Sequence[Dict[str, Any]]) -> List[int]:
        """Send each alert to every channel; returns the ids that were handled.

        A channel that fails is logged and skipped rather than allowed to stop
        the others or to wedge the daemon's loop.
        """
        if not self.enabled:
            return [int(alert["id"]) for alert in alerts]

        handled: List[int] = []
        for alert in alerts:
            # Filtered-out alerts still count as handled, so they are not
            # reconsidered on every pass.
            if self.wants(alert):
                for channel in (self._to_log, self._to_desktop, self._to_command):
                    try:
                        channel(alert)
                    except Exception as error:
                        LOGGER.warning(
                            "alert %s: %s channel failed: %s",
                            alert.get("id"),
                            channel.__name__.lstrip("_"),
                            error,
                        )
            handled.append(int(alert["id"]))
        return handled

    # --- channels ---------------------------------------------------------

    def _to_log(self, alert: Dict[str, Any]) -> None:
        if not self.settings["log"]:
            return
        path = alert_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert, ensure_ascii=False, default=str) + "\n")

    def _to_desktop(self, alert: Dict[str, Any]) -> None:
        if not self._desktop_command:
            return
        title, subtitle, body = desktop_content(alert)
        subprocess.run(
            self._desktop_command(title, subtitle, body),
            check=True,
            capture_output=True,
            timeout=10,
        )

    def _to_command(self, alert: Dict[str, Any]) -> None:
        command = self.settings["command"]
        if not command:
            return
        subprocess.run(
            str(command),
            shell=True,
            input=json.dumps(alert, ensure_ascii=False, default=str),
            text=True,
            check=False,
            capture_output=True,
            timeout=30,
        )


def _find_desktop_command():
    """Return a builder for the platform's notification command, or None."""
    system = platform.system()
    if system == "Darwin":
        from .macos_notify import notification_command

        fallback = None
        if shutil.which("terminal-notifier"):
            def fallback(title, subtitle, body):
                return [
                    "terminal-notifier", "-title", title, "-subtitle", subtitle, "-message", body
                ]
        elif shutil.which("osascript"):
            def fallback(title, subtitle, body):
                return [
                    "osascript",
                    "-e",
                    # display notification takes AppleScript string literals.
                    f'display notification {_applescript_string(body)} '
                    f'with title {_applescript_string(title)} '
                    f'subtitle {_applescript_string(subtitle)}',
                ]
        if not shutil.which("xcrun") or not shutil.which("codesign"):
            return fallback

        build_failed = False

        def macos_command(title, subtitle, body):
            nonlocal build_failed
            if not build_failed:
                try:
                    return notification_command(title, subtitle, body)
                except (OSError, subprocess.SubprocessError) as error:
                    if fallback is None:
                        raise
                    build_failed = True
                    LOGGER.warning("could not build nvidb notification app; using fallback: %s", error)
            return fallback(title, subtitle, body)

        return macos_command
    if shutil.which("notify-send"):
        return lambda title, subtitle, body: [
            "notify-send", title, "\n".join(part for part in (subtitle, body) if part)
        ]
    return None


def _applescript_string(text: str) -> str:
    """Quote a Python string as an AppleScript literal."""
    escaped = str(text).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
