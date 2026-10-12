"""Resolve job attribution on the submitting computer, before SSH forwarding."""
from __future__ import annotations

import getpass
import os
import socket
import subprocess


def default_submitter() -> str:
    explicit = os.environ.get("NVIDB_SUBMITTER")
    if explicit:
        return explicit
    identity = {}
    for key in ("name", "email"):
        try:
            result = subprocess.run(
                ["git", "config", "--global", "--get", f"user.{key}"],
                capture_output=True, text=True, timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired, UnicodeError):
            continue
        if result.returncode == 0 and result.stdout.strip():
            identity[key] = result.stdout.strip()
    name, email = identity.get("name"), identity.get("email")
    if name and email:
        return f"{name} <{email}>"
    if name or email:
        return name or email
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    return f"{user}@{socket.gethostname()}:{os.getpid()}"
