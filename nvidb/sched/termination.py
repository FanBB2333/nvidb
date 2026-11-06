"""Node-side cancellation of a verified job and its surviving children."""
import shlex


TERMINATION_SCRIPT = r'''
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def process_table():
    output = subprocess.check_output(
        ["ps", "-ww", "-eo", "pid=,ppid=,pgid=,stat=,lstart=,command="],
        text=True, errors="replace", env=dict(os.environ, LC_ALL="C"),
    )
    table = {}
    for line in output.splitlines():
        fields = line.split(None, 9)
        if len(fields) != 10 or fields[3].startswith("Z"):
            continue
        table[int(fields[0])] = {
            "parent": int(fields[1]), "group": int(fields[2]),
            "started": " ".join(fields[4:9]), "command": fields[9],
        }
    return table


def identity(pid, record):
    if sys.platform.startswith("linux"):
        try:
            # Linux start ticks distinguish recycled PIDs even within a second.
            stat = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            return None
        if int(stat[2]) != record["group"]:
            return None
        started = stat[19]
    else:
        started = record["started"]
    return (record["group"], started)


def members(table, roots, group):
    if group:
        return {pid for pid, record in table.items() if record["group"] == group}
    found = set(roots)
    while True:
        children = {pid for pid, record in table.items() if record["parent"] in found}
        expanded = found | children
        if expanded == found:
            return found
        found = expanded


def snapshot(table, pids):
    return {pid: stamp for pid in pids
            if pid in table and (stamp := identity(pid, table[pid])) is not None}


def send(pids, group, sig):
    targets = [-group] if group else pids
    for target in targets:
        try:
            os.kill(target, sig)
        except ProcessLookupError:
            pass


def escalate(known, group, grace):
    deadline = time.monotonic() + grace
    while True:
        table = process_table()
        verified = {pid for pid, stamp in known.items()
                    if pid in table and identity(pid, table[pid]) == stamp}
        if not verified:
            if group and any(record["group"] == group for record in table.values()):
                raise RuntimeError("process group identity is no longer verifiable; refusing to signal it")
            return
        # Retain identity through reparenting and include new children while a
        # known member still proves ownership of this group or descendant tree.
        known = snapshot(table, members(table, verified, group))
        if time.monotonic() >= deadline:
            send(known, group, signal.SIGKILL)
            return
        time.sleep(min(.1, max(0, deadline - time.monotonic())))


def terminate(directory, pid, group, grace, timed_out, seal):
    directory = Path(directory)
    if seal:
        (directory / "cancelled").touch()
    if not pid:
        path = directory / "pid"
        if not path.exists():
            return False
        pid = int(path.read_text().strip())
        session = directory / "session"
        if session.exists() and session.read_text().strip() == "1":
            group = int((directory / "pgid").read_text().strip())

    table = process_table()
    wrapper = table.get(pid)
    if wrapper is None or str(directory / "run.sh") not in wrapper["command"]:
        return False
    if group and (group != pid or wrapper["group"] != group):
        raise ValueError("job does not own the requested process group")
    known = snapshot(table, members(table, [pid], group))
    if pid not in known:
        return False
    if timed_out:
        if (directory / "exit_code").exists():
            return False
        marker = directory / "timeout"
        if not marker.exists():
            temporary = directory / ("timeout.tmp-%d" % os.getpid())
            temporary.write_text(str(time.time()))
            temporary.replace(marker)

    # Start the watcher before signalling. It owns a snapshot of verified
    # identities and survives both the wrapper's exit and the SSH connection.
    with open(directory / "cleanup.log", "ab", buffering=0) as log:
        watcher = os.fork()
        if watcher == 0:
            os.setsid()
            with open(os.devnull, "rb") as source:
                os.dup2(source.fileno(), 0)
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            # Keep only stdio. Inherited environment variables cannot prove
            # which descriptors are still open in this cancellation process.
            os.closerange(3, os.sysconf("SC_OPEN_MAX"))
            try:
                escalate(known, group, grace)
            except Exception:
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
    # Signal the wrapper first in child-only mode so its EXIT trap records TERM.
    send([pid] + [child for child in known if child != pid], group, signal.SIGTERM)
    return True


if __name__ == "__main__":
    directory, pid, group, grace, timed_out, seal = sys.argv[1:]
    stopped = terminate(directory, int(pid) or None, int(group) or None,
                        float(grace), bool(int(timed_out)), bool(int(seal)))
    print("NVIDB_REAPED=%d" % stopped)
'''


def build_termination_command(run_dir, *, pid, pgid, grace, timed_out=False, seal=False):
    args = [run_dir, str(pid or 0), str(pgid or 0), str(grace), str(int(timed_out)), str(int(seal))]
    return "python3 -c " + shlex.quote(TERMINATION_SCRIPT) + " " + shlex.join(args)
