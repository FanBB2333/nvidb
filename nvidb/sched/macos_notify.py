"""Build a branded UserNotifications app using Apple's Command Line Tools."""
from __future__ import annotations

import hashlib
import plistlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from .. import config as nvidb_config

BUNDLE_ID = "com.nvidb.desktop"
ASSETS = Path(__file__).parent


def ensure_app(home: Optional[Path] = None) -> Path:
    """Build on first use, reusing the bundle until its resources change."""
    import fcntl

    home = home or Path(nvidb_config.WORKING_DIR).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    source = ASSETS / "macos_notify.m"
    icon = ASSETS / "nvidb.icns"
    digest = hashlib.sha256(
        b"nvidb-notifications-v6" + BUNDLE_ID.encode() + source.read_bytes() + icon.read_bytes()
    ).hexdigest()
    app = home / "nvidb.app"
    stamp = app / "Contents" / "Resources" / "nvidb-version"
    with (home / "notification-app.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if stamp.exists() and stamp.read_text() == digest:
            return app
        with tempfile.TemporaryDirectory(prefix=".nvidb-notify-", dir=home) as stage:
            built = Path(stage) / "nvidb.app"
            executable = built / "Contents" / "MacOS" / "nvidb-notify"
            resources = built / "Contents" / "Resources"
            executable.parent.mkdir(parents=True)
            resources.mkdir(parents=True)
            # Check first so machines without CLT fall back without opening the
            # interactive developer-tools installation dialog.
            subprocess.run(
                ["/usr/bin/xcode-select", "-p"],
                check=True, capture_output=True, timeout=5,
            )
            subprocess.run(
                ["/usr/bin/xcrun", "clang", "-fobjc-arc", "-mmacosx-version-min=11.0",
                 "-framework", "AppKit", "-framework", "CoreServices",
                 "-framework", "UserNotifications",
                 "-o", str(executable), str(source)],
                check=True, capture_output=True, timeout=30,
            )
            plist_path = built / "Contents" / "Info.plist"
            info = {
                "CFBundleIdentifier": BUNDLE_ID,
                "CFBundleName": "nvidb",
                "CFBundleDisplayName": "nvidb",
                "CFBundleExecutable": "nvidb-notify",
                "CFBundlePackageType": "APPL",
                "CFBundleInfoDictionaryVersion": "6.0",
                "CFBundleIconFile": "nvidb.icns",
                "CFBundleVersion": "6",
                "CFBundleShortVersionString": "1.0",
                "LSMinimumSystemVersion": "11.0",
                "LSUIElement": True,
            }
            with plist_path.open("wb") as handle:
                plistlib.dump(info, handle)
            shutil.copyfile(icon, resources / "nvidb.icns")
            (resources / "nvidb-version").write_text(digest)
            subprocess.run(
                ["/usr/bin/codesign", "--force", "--sign", "-", str(built)],
                check=True, capture_output=True, timeout=30,
            )
            # Keep the previous working helper until the replacement is built.
            previous = Path(stage) / "previous.app"
            if app.exists():
                app.rename(previous)
            try:
                built.rename(app)
            except OSError:
                if previous.exists():
                    previous.rename(app)
                raise
    return app


def notification_command(title: str, subtitle: str, body: str) -> List[str]:
    app = ensure_app()
    return [str(app / "Contents" / "MacOS" / "nvidb-notify"), "--notify", title, subtitle, body]
