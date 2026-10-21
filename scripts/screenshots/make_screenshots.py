"""Regenerate the README screenshots from made-up data.

usage: python scripts/screenshots/make_screenshots.py [--out resources] [--only monitor,queue,web]

Each terminal view runs the real TUI in a pseudo-terminal against the fake
cluster (demo_cluster.py / demo_queue.py), is emulated with pyte, drawn as
HTML and photographed by headless Chrome; the web views are photographed from
`nvidb web` serving the same fake cluster. Needs pyte, selenium, dash and Chrome.
"""
import argparse
import fcntl
import html
import os
import pty
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path

import pyte
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

import demo_cluster

HERE = Path(__file__).resolve().parent
ROOT = demo_cluster.ROOT
PYTHON = sys.executable

PALETTE = {
    "default": "#d4d7e0", "black": "#1b1e28", "red": "#f07178", "green": "#7fd88f",
    "brown": "#e6c07b", "yellow": "#e6c07b", "blue": "#61afef", "magenta": "#c792ea",
    "cyan": "#56d4dd", "white": "#d4d7e0", "brightblack": "#6b7089", "brightred": "#ff8b92",
    "brightgreen": "#a5e7a3", "brightyellow": "#ffd580", "brightblue": "#82c4ff",
    "brightmagenta": "#ddb6f2", "brightcyan": "#8be9fd", "brightwhite": "#ffffff",
}
BACKGROUND = "#11131a"
# Backgrounds differ from foregrounds only where the TUI draws its selection band.
BACKGROUNDS = {**{k: v for k, v in PALETTE.items() if k != "default"}, "brightblack": "#2c3142"}

# name, script, NVIDB_HOME setup, columns, rows, keys sent after the first frame
TERMINAL_SHOTS = {
    "monitor": [
        ("monitor-gpus", "demo_cluster.py", "unified", 170, 48, [(2, "j")]),
        ("monitor-nodes", "demo_cluster.py", "nodes", 170, 46, []),
    ],
    "queue": [
        ("queue-overview", "demo_queue.py", None, 170, 30, [(3, "\r")]),
        ("queue-flow", "demo_queue.py", None, 170, 34, [(3, "v")]),
    ],
}


def capture(script, home, columns, rows, keys, wait=12.0):
    """Run `script` in a pty sized columns x rows and return the final screen."""
    screen = pyte.Screen(columns, rows)
    stream = pyte.ByteStream(screen)
    env = dict(os.environ, NVIDB_HOME=str(home), TERM="xterm-256color", COLUMNS=str(columns), LINES=str(rows))
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(ROOT)
        os.execve(PYTHON, [PYTHON, str(HERE / script)], env)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))

    def pump(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not select.select([fd], [], [], 0.1)[0]:
                continue
            try:
                data = os.read(fd, 65536)
            except OSError:
                return
            if not data:
                return
            stream.feed(data)
            if b"\x1b[6n" in data:  # the TUI asks where the cursor is
                os.write(fd, f"\x1b[{screen.cursor.y + 1};{screen.cursor.x + 1}R".encode())

    pump(wait)
    for delay, text in [*keys, (3, "")]:
        os.write(fd, text.encode())
        pump(delay)
    cells = [[screen.buffer[y][x] for x in range(columns)] for y in range(rows)]
    try:
        os.write(fd, b"q")
        pump(1)
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    os.waitpid(pid, 0)
    while cells and all(cell.data in (" ", "") and cell.bg == "default" for cell in cells[-1]):
        cells.pop()
    return cells


def _colour(value, palette, default):
    if value == "default":
        return default
    if value in palette:
        return palette[value]
    return "#" + value if len(value) == 6 else default


def to_html(cells):
    rows = []
    for row in cells:
        spans = []
        for cell in row:
            if cell.data == "":
                continue  # the right half of a wide glyph
            fg = _colour(cell.fg, PALETTE, PALETTE["default"])
            bg = _colour(cell.bg, BACKGROUNDS, BACKGROUND)
            if cell.reverse:
                fg, bg = bg, fg
            style = f"color:{fg};" + (f"background:{bg};" if bg != BACKGROUND else "")
            style += "font-weight:700;" if cell.bold else ""
            style += "text-decoration:underline;" if cell.underscore else ""
            spans.append(f'<span style="{style}">{html.escape(cell.data)}</span>')
        rows.append(f'<div class="row">{"".join(spans)}</div>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin:0; background:#0b0d12; }}
.term {{ display:inline-block; background:{BACKGROUND}; padding:18px 22px; border-radius:12px;
  font-family:"JetBrains Mono","SF Mono",Menlo,monospace; font-size:14px; line-height:1.32;
  white-space:pre; font-variant-ligatures:none; }}
.row {{ height:1.32em; }}
</style></head><body><div class="term" id="term">{"".join(rows)}</div></body></html>"""


def browser(width, height):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument(f"--window-size={width},{height}")
    options.add_argument("--force-device-scale-factor=2")
    return webdriver.Chrome(options=options)


def terminal_shots(groups, work, out):
    pages = []
    for group in groups:
        for name, script, mode, columns, rows, keys in TERMINAL_SHOTS.get(group, []):
            home = work / name
            if mode:
                demo_cluster.write_config(home, mode=mode)
            else:
                home.mkdir(parents=True)
            page = work / f"{name}.html"
            page.write_text(to_html(capture(script, home, columns, rows, keys)), encoding="utf-8")
            pages.append((name, page))
            print("captured", name)
    if not pages:
        return
    driver = browser(2400, 1600)
    try:
        for name, page in pages:
            driver.get(page.as_uri())
            driver.find_element(By.ID, "term").screenshot(str(out / f"{name}.png"))
            print("wrote", out / f"{name}.png")
    finally:
        driver.quit()


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def web_shots(work, out, record_seconds=45, history_seconds=30):
    home = work / "web"
    demo_cluster.write_config(home)
    env = dict(os.environ, NVIDB_HOME=str(home))
    script = str(HERE / "demo_cluster.py")
    logger = subprocess.Popen([PYTHON, script, "log", "--remote", "--interval", "2"], env=env, cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(record_seconds)
    logger.send_signal(signal.SIGINT)
    logger.wait(30)

    port = _free_port()
    server = subprocess.Popen([PYTHON, script, "web", "--port", str(port)], env=env, cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    driver = browser(1600, 1000)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                driver.get(url)
                break
            except Exception:
                time.sleep(1)
        wait = WebDriverWait(driver, 60)
        wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "#live-remote input"))
        include_remote = driver.find_element(By.CSS_SELECTOR, "#live-remote input")
        if not include_remote.is_selected():
            include_remote.click()
        wait.until(lambda d: "lab1" in d.find_element(By.TAG_NAME, "body").text)
        time.sleep(history_seconds)  # let the rolling charts gather a history
        driver.save_screenshot(str(out / "web-live.png"))
        print("wrote", out / "web-live.png")

        driver.find_element(By.XPATH, "//*[@id='tabs']//*[contains(text(), 'Logs')]").click()
        wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "#session-table td"))
        driver.find_elements(By.CSS_SELECTOR, "#session-table td")[0].click()
        wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "#logs-body .js-plotly-plot"))
        time.sleep(3)
        driver.save_screenshot(str(out / "web-logs.png"))
        print("wrote", out / "web-logs.png")
    finally:
        driver.quit()
        server.terminate()
        server.wait(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "resources")
    parser.add_argument("--only", default="monitor,queue,web", help="comma-separated: monitor, queue, web")
    args = parser.parse_args()
    groups = [group.strip() for group in args.only.split(",") if group.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nvidb-shots-") as work:
        terminal_shots(groups, Path(work), args.out.resolve())
        if "web" in groups:
            web_shots(Path(work), args.out.resolve())


if __name__ == "__main__":
    main()
