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
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
    settings = dict(DEFAULT_NOTIFY_SETTINGS)
    if isinstance(raw, dict):
        for key, default in DEFAULT_NOTIFY_SETTINGS.items():
            if key not in raw:
                continue
            value = raw[key]
            settings[key] = bool(value) if isinstance(default, bool) else value
    elif raw is not None:
        settings["enabled"] = bool(raw)
    return settings


def alert_log_path() -> Path:
    return Path(nvidb_config.WORKING_DIR).expanduser() / "alerts.log"


def format_alert(alert: Dict[str, Any]) -> str:
    """One human-readable line: what broke, and where."""
    where = []
    if alert.get("job_id"):
        where.append(f"job {alert['job_id']}")
    if alert.get("node"):
        where.append(alert["node"])
    location = f" [{' on '.join(where)}]" if where else ""
    return f"{alert['title']}{location}"


class Notifier:
    """Pushes alerts to the configured channels."""

    def __init__(self, settings: Optional[dict] = None):
        self.settings = load_notify_settings(settings)
        self._desktop_command = _find_desktop_command() if self.settings["desktop"] else None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.get("enabled"))

    def wants(self, alert: Dict[str, Any]) -> bool:
        minimum = SEVERITY_ORDER.get(str(self.settings.get("min_severity", "warning")), 0)
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
        if not self.settings.get("log"):
            return
        path = alert_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert, ensure_ascii=False, default=str) + "\n")

    def _to_desktop(self, alert: Dict[str, Any]) -> None:
        if not self._desktop_command:
            return
        title = f"nvidb: {alert['kind'].replace('_', ' ')}"
        body = format_alert(alert)
        subprocess.run(
            self._desktop_command(title, body),
            check=False,
            capture_output=True,
            timeout=10,
        )

    def _to_command(self, alert: Dict[str, Any]) -> None:
        command = self.settings.get("command")
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
        if shutil.which("terminal-notifier"):
            return lambda title, body: [
                "terminal-notifier", "-title", title, "-message", body
            ]
        if shutil.which("osascript"):
            return lambda title, body: [
                "osascript",
                "-e",
                # display notification takes AppleScript string literals.
                f'display notification {_applescript_string(body)} '
                f'with title {_applescript_string(title)}',
            ]
        return None
    if shutil.which("notify-send"):
        return lambda title, body: ["notify-send", title, body]
    return None


def _applescript_string(text: str) -> str:
    """Quote a Python string as an AppleScript literal."""
    escaped = str(text).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
