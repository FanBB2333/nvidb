"""Package the selected nvidb logo as ICNS (development only; requires Pillow).

Run from any directory with ``python resources/generate_notification_icon.py``.
The canonical PNG preserves candidate D exactly; only the ICNS sizes are derived.
"""
from pathlib import Path

from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    with Image.open(root / "resources" / "nvidb-icon.png") as source:
        image = source.convert("RGBA").resize((1024, 1024), Image.Resampling.LANCZOS)
        image.save(root / "nvidb" / "sched" / "nvidb.icns")


if __name__ == "__main__":
    main()
