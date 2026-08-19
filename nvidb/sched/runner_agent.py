"""The lane runner: nvidb's own scheduler process, resident on a GPU node.

Without it, every job is started by whichever short-lived client happened to
run a scheduler pass, so a lane only advances when somebody looks: a job that
finishes at 03:00 leaves its card idle until the next tick, and the queue
behind it exists only as rows on a laptop that may be shut.

The runner closes that gap by being the parent. It owns one lane's spool
directory on the node, starts each job as its own child, notices the moment
that child exits, and starts the next one. The controller's job shrinks to
maintaining a small window of staged work and reading back what happened,
which is why reordering everything behind that window is a purely local edit.

The program below is shipped inline over the transport, the way the NVML agent
is, so a node needs nothing installed beyond python3. It is plain stdlib and
deliberately conservative: it must keep running on machines nobody administers.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from typing import Optional, Sequence

# --- the node-side program -------------------------------------------------
#
# Kept as a string rather than a module so it travels over an SSH channel with
# no install step. Edited here, versioned by its own hash, and rolled out by
# the controller when that hash changes.

RUNNER_SCRIPT = r'''
"""nvidb lane runner - runs one lane's jobs, in order, one after another."""
import base64
import errno
import json
import os
import signal
import subprocess
import sys
import time

SPOOL = sys.argv[1]
LANE = sys.argv[2]
MARKER = sys.argv[3]        # identifies this runner in the process table
VERSION = sys.argv[4]
INTERVAL = float(sys.argv[5])
IDLE_EXIT = float(sys.argv[6])

QUEUE = os.path.join(SPOOL, "queue")
CLAIMED = os.path.join(SPOOL, "claimed")
STATE = os.path.join(SPOOL, "state.json")
PIDFILE = os.path.join(SPOOL, "runner.pid")
STOPFILE = os.path.join(SPOOL, "stop")

# Jobs kept in the state file after they finish. Enough for a controller that
# missed a few passes to see what happened without reading every run directory.
FINISHED_MEMORY = 20


def log(message):
    sys.stderr.write("[%s] %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), message))
    sys.stderr.flush()


def write_atomic(path, text):
    temporary = path + ".tmp-%d" % os.getpid()
    with open(temporary, "w") as handle:
        handle.write(text)
    os.rename(temporary, path)


def read_json(path, default=None):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (IOError, OSError, ValueError):
        return default


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError as error:
        return error.errno == errno.EPERM
    return True


def pid_runs(pid, run_dir):
    """True when `pid` is alive *and* still running this job's run.sh.

    A pid on its own proves nothing: the node recycles them, and killing or
    crediting an unrelated process would be far worse than losing track of
    one of ours.
    """
    if not pid_alive(pid):
        return False
    try:
        result = subprocess.Popen(
            ["ps", "-o", "command=", "-p", str(pid)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, _ = result.communicate()
    except (OSError, ValueError):
        return False
    return os.path.join(run_dir, "run.sh") in out.decode("utf-8", "replace")


def gpu_free_mb(indices):
    """Free VRAM on the lane's cards, or None when nothing can say.

    Used as a last gate before starting: the controller staged this job some
    time ago, and the card may have filled with work nobody submitted through
    the queue in between.
    """
    if not indices:
        return None
    try:
        process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, _ = process.communicate()
        if process.returncode != 0:
            return None
    except (OSError, ValueError):
        return None
    free = {}
    for line in out.decode("utf-8", "replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            free[int(parts[0])] = int(float(parts[1]))
        except ValueError:
            continue
    values = [free[index] for index in indices if index in free]
    return min(values) if values else None


class Job(object):
    """One claimed spec and the process running it."""

    def __init__(self, spec, path):
        self.spec = spec
        self.path = path
        self.proc = None
        self.pid = None
        self.started_at = None

    @property
    def job_id(self):
        return self.spec.get("job_id")

    @property
    def run_dir(self):
        return self.spec.get("run_dir") or ""

    def alive(self):
        if self.proc is not None:
            # Our own child: poll() also reaps it, so a finished job cannot
            # linger as a zombie that still answers `kill -0`.
            return self.proc.poll() is None
        return pid_runs(self.pid, self.run_dir)

    def exit_code(self):
        raw = ""
        try:
            with open(os.path.join(self.run_dir, "exit_code")) as handle:
                raw = handle.read().strip()
        except (IOError, OSError):
            pass
        parts = raw.split()
        if parts:
            try:
                return int(parts[0])
            except ValueError:
                pass
        if self.proc is not None and self.proc.returncode is not None:
            return int(self.proc.returncode)
        return None

    def start(self):
        run_dir = self.run_dir
        try:
            os.makedirs(run_dir)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
        script = base64.b64decode(self.spec["script_b64"]).decode("utf-8")
        write_atomic(os.path.join(run_dir, "run.sh"), script)
        os.chmod(os.path.join(run_dir, "run.sh"), 0o755)
        for name in ("exit_code", "pid", "pgid", "result.json", "status"):
            try:
                os.unlink(os.path.join(run_dir, name))
            except OSError:
                pass
        out = open(os.path.join(run_dir, "stdout.log"), "wb")
        err = open(os.path.join(run_dir, "stderr.log"), "wb")
        try:
            # Its own session, so the job outlives this runner: a runner that
            # is restarted or upgraded must never take the work down with it.
            self.proc = subprocess.Popen(
                ["bash", os.path.join(run_dir, "run.sh")],
                stdout=out,
                stderr=err,
                stdin=open(os.devnull, "rb"),
                preexec_fn=os.setsid,
            )
        finally:
            out.close()
            err.close()
        self.pid = self.proc.pid
        self.started_at = time.time()
        log("job %s started as pid %s" % (self.job_id, self.pid))

    def adopt(self):
        """Take over a job started by a previous runner, if it is still going."""
        pid = None
        try:
            with open(os.path.join(self.run_dir, "pid")) as handle:
                pid = int(handle.read().strip())
        except (IOError, OSError, ValueError):
            return False
        if not pid_runs(pid, self.run_dir):
            return False
        self.pid = pid
        self.proc = None
        try:
            self.started_at = os.path.getmtime(os.path.join(self.run_dir, "pid"))
        except OSError:
            self.started_at = time.time()
        log("adopted job %s (pid %s) from a previous runner" % (self.job_id, pid))
        return True

    def kill(self, sig=signal.SIGTERM):
        if not self.pid:
            return
        try:
            os.killpg(os.getpgid(self.pid), sig)
        except OSError:
            try:
                os.kill(self.pid, sig)
            except OSError:
                pass


class Runner(object):
    def __init__(self):
        self.current = []
        self.finished = []
        self.idle_since = time.time()
        self.stopping = False

    # --- spool ---------------------------------------------------------

    def ensure_dirs(self):
        for path in (QUEUE, CLAIMED):
            try:
                os.makedirs(path)
            except OSError as error:
                if error.errno != errno.EEXIST:
                    raise

    def concurrency(self):
        """How many jobs this lane may run at once, as last staged."""
        for job in self.current:
            value = job.spec.get("concurrency")
            if value:
                return max(1, int(value))
        for spec in self.pending_specs():
            value = spec[1].get("concurrency")
            if value:
                return max(1, int(value))
        return 1

    def pending_specs(self):
        """Staged specs not yet claimed, in the order the lane runs them."""
        out = []
        try:
            names = os.listdir(QUEUE)
        except OSError:
            return out
        for name in names:
            if not name.endswith(".json") or name.startswith("."):
                continue
            spec = read_json(os.path.join(QUEUE, name))
            if isinstance(spec, dict):
                out.append((name, spec))
        out.sort(key=lambda item: (item[1].get("seq") or 0, item[1].get("job_id") or 0))
        return out

    def recover(self):
        """Re-establish what was running before this runner existed."""
        try:
            names = sorted(os.listdir(CLAIMED))
        except OSError:
            return
        for name in names:
            path = os.path.join(CLAIMED, name)
            spec = read_json(path)
            if not isinstance(spec, dict):
                os.unlink(path)
                continue
            job = Job(spec, path)
            if job.adopt():
                self.current.append(job)
            else:
                # It finished while nothing was watching; record it and move on.
                self.retire(job)

    def claim_next(self):
        """Take the lane's next spec, or None.

        The claim is a rename, so it either happens or it does not: a
        controller withdrawing the same spec at the same moment finds it gone
        rather than both of them acting on it.
        """
        for name, spec in self.pending_specs():
            source = os.path.join(QUEUE, name)
            target = os.path.join(CLAIMED, name)
            try:
                os.rename(source, target)
            except OSError:
                continue  # withdrawn underneath us
            return Job(spec, target)
        return None

    def retire(self, job):
        code = job.exit_code()
        self.finished.append(
            {
                "job_id": job.job_id,
                "exit_code": code,
                "finished_at": time.time(),
                "run_dir": job.run_dir,
            }
        )
        del self.finished[:-FINISHED_MEMORY]
        try:
            os.unlink(job.path)
        except OSError:
            pass
        log("job %s finished with exit code %s" % (job.job_id, code))

    # --- the loop ------------------------------------------------------

    def enforce_timeouts(self):
        now = time.time()
        for job in self.current:
            limit = job.spec.get("max_runtime_s")
            if not limit or not job.started_at:
                continue
            if now - job.started_at <= float(limit):
                continue
            log("job %s exceeded %ss; stopping it" % (job.job_id, limit))
            job.kill(signal.SIGTERM)
            job.spec["max_runtime_s"] = None  # do not signal it every pass

    def maybe_start(self):
        while len(self.current) < self.concurrency():
            specs = self.pending_specs()
            if not specs:
                return
            head = specs[0][1]
            needed = int(head.get("vram_mb") or 0)
            if needed:
                free = gpu_free_mb([int(i) for i in head.get("gpu_ids") or []])
                if free is not None and free < needed:
                    # The card filled up with work this queue did not start.
                    # Waiting is right: the lane's order is still the order.
                    return
            job = self.claim_next()
            if job is None:
                return
            try:
                job.start()
            except Exception as error:  # a bad spec must not kill the runner
                log("job %s failed to start: %s" % (job.job_id, error))
                try:
                    with open(os.path.join(job.run_dir, "stderr.log"), "a") as handle:
                        handle.write("nvidb runner could not start this job: %s\n" % error)
                except (IOError, OSError):
                    pass
                self.retire(job)
                continue
            self.current.append(job)

    def publish(self):
        write_atomic(
            STATE,
            json.dumps(
                {
                    "lane": LANE,
                    "version": VERSION,
                    "pid": os.getpid(),
                    "at": time.time(),
                    "stopping": self.stopping,
                    "current": [
                        {
                            "job_id": job.job_id,
                            "pid": job.pid,
                            "run_dir": job.run_dir,
                            "started_at": job.started_at,
                        }
                        for job in self.current
                    ],
                    "queued": [spec.get("job_id") for _name, spec in self.pending_specs()],
                    "finished": self.finished,
                }
            ),
        )

    def run(self):
        self.ensure_dirs()
        write_atomic(PIDFILE, "%d\n" % os.getpid())
        self.recover()
        log("lane runner %s up (version %s, pid %d)" % (LANE, VERSION, os.getpid()))

        while True:
            if os.path.exists(STOPFILE):
                self.stopping = True

            still = []
            for job in self.current:
                if job.alive():
                    still.append(job)
                else:
                    self.retire(job)
            self.current = still

            self.enforce_timeouts()
            if not self.stopping:
                self.maybe_start()
            self.publish()

            if self.current or self.pending_specs():
                self.idle_since = time.time()
            elif self.stopping:
                log("stopping: nothing left to run")
                break
            elif IDLE_EXIT and time.time() - self.idle_since > IDLE_EXIT:
                # Nothing to do for a long time. Exiting keeps idle machines
                # free of processes; the controller starts a runner again the
                # moment the lane has work.
                log("idle for %.0fs; exiting" % (time.time() - self.idle_since))
                break

            time.sleep(INTERVAL)

        try:
            os.unlink(PIDFILE)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        Runner().run()
    except KeyboardInterrupt:
        pass
'''


RUNNER_VERSION = hashlib.sha256(RUNNER_SCRIPT.encode("utf-8")).hexdigest()[:12]

DEFAULT_SPOOL_ROOT = ".nvidb/lanes"
DEFAULT_INTERVAL = 3.0
# How long a runner stays resident with nothing to do. Long enough that a lane
# being refilled does not restart it every time, short enough that a machine
# nobody is using ends up with no nvidb processes on it.
DEFAULT_IDLE_EXIT = 900.0

STATE_MARKER = "NVIDB_LANE_STATE"

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def lane_slug(lane_name: str) -> str:
    """A directory name for a lane that is readable and cannot collide.

    Lane names contain characters that are awkward in paths (`:` always, `@`
    for a node with no nickname), so they are flattened - and a hash of the
    original is appended, because flattening alone can map two different lanes
    onto one directory.
    """
    flattened = _SLUG_UNSAFE.sub("_", lane_name).strip("_") or "lane"
    digest = hashlib.sha1(lane_name.encode("utf-8")).hexdigest()[:8]
    return f"{flattened[:48]}-{digest}"


def runner_marker(lane_name: str) -> str:
    """What identifies this lane's runner in the node's process table."""
    return f"NVIDB_LANE={lane_slug(lane_name)}"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def spool_path_expr(spool_root: str, slug: str) -> str:
    """Shell for a lane's spool directory, given a possibly relative root.

    `$HOME` must be left outside the quoting or it is passed through as four
    literal characters and the node grows a directory actually named `$HOME`.
    Only the part that could contain anything surprising is quoted.
    """
    if spool_root.startswith("/"):
        return shlex.quote(f"{spool_root.rstrip('/')}/{slug}")
    return '"$HOME"/' + shlex.quote(f"{spool_root.strip('/')}/{slug}")


def _write_file(path_expr: str, text: str) -> str:
    """Shell that writes a file atomically from an inline base64 payload."""
    return (
        f"printf '%s' {shlex.quote(_b64(text))} | base64 -d > {path_expr}.tmp && "
        f"mv {path_expr}.tmp {path_expr}"
    )


def build_sync_command(
    *,
    lane_name: str,
    spool_root: str,
    specs: Sequence[dict],
    python_exe: str = "python3",
    interval: float = DEFAULT_INTERVAL,
    idle_exit: float = DEFAULT_IDLE_EXIT,
    want_runner: bool = True,
) -> str:
    """One round trip that stages work, keeps the runner up, and reads state.

    Staging, withdrawal, starting the runner and reading back what it is doing
    are all the same conversation on purpose: a lane that needed three round
    trips per tick would cost more than the scheduling it replaces.
    """
    slug = lane_slug(lane_name)
    marker = runner_marker(lane_name)

    lines = [
        f"d={spool_path_expr(spool_root, slug)}",
        'mkdir -p "$d/queue" "$d/claimed" || exit 1',
    ]

    # 1. Stage the specs the controller wants queued, newest content winning.
    keep = []
    for spec in specs:
        name = f"{int(spec['seq']):012d}-{int(spec['job_id'])}.json"
        keep.append(name)
        payload = json.dumps(spec, ensure_ascii=False)
        lines.append(_write_file(f'"$d/queue/{name}"', payload))

    # 2. Withdraw anything staged earlier that is no longer wanted. A spec the
    #    runner has already claimed is not here to remove, which is the point:
    #    withdrawal and claiming cannot both win.
    if keep:
        pattern = "|".join(keep)
        lines.append(
            'for f in "$d"/queue/*.json; do [ -e "$f" ] || continue; '
            f'case "$(basename "$f")" in {pattern}) ;; *) rm -f "$f";; esac; done'
        )
    else:
        lines.append('rm -f "$d"/queue/*.json 2>/dev/null || true')

    if want_runner:
        lines.append('rm -f "$d/stop"')
        lines.extend(
            [
                'p=$(cat "$d/runner.pid" 2>/dev/null | tr -d " \\n\\r")',
                "alive=0",
                'if [ -n "$p" ] && kill -0 "$p" 2>/dev/null && '
                'ps -o command= -p "$p" 2>/dev/null | '
                f'grep -F -q -- {shlex.quote(marker)}; then alive=1; fi',
                'v=$(cat "$d/runner.version" 2>/dev/null | tr -d " \\n\\r")',
                # A runner whose code is out of date is replaced rather than
                # left alone: its jobs are in their own sessions, so they
                # survive the swap and the new runner adopts them.
                f'if [ "$alive" = 1 ] && [ "$v" != {shlex.quote(RUNNER_VERSION)} ]; then',
                '  kill -TERM "$p" 2>/dev/null || true',
                "  alive=0",
                "fi",
                'if [ "$alive" = 0 ]; then',
                "  " + _write_file('"$d/runner.py"', RUNNER_SCRIPT),
                f'  echo {shlex.quote(RUNNER_VERSION)} > "$d/runner.version"',
                "  if command -v setsid >/dev/null 2>&1; then",
                f'    ( setsid {python_exe} -u "$d/runner.py" "$d" {shlex.quote(lane_name)} '
                f"{shlex.quote(marker)} {shlex.quote(RUNNER_VERSION)} {float(interval)} "
                f'{float(idle_exit)} >> "$d/runner.log" 2>&1 < /dev/null & )',
                "  else",
                f'    ( nohup {python_exe} -u "$d/runner.py" "$d" {shlex.quote(lane_name)} '
                f"{shlex.quote(marker)} {shlex.quote(RUNNER_VERSION)} {float(interval)} "
                f'{float(idle_exit)} >> "$d/runner.log" 2>&1 < /dev/null & )',
                "  fi",
                "  i=0",
                '  while [ "$i" -lt 40 ]; do',
                '    [ -s "$d/state.json" ] && break',
                "    sleep 0.05",
                "    i=$((i+1))",
                "  done",
                "fi",
            ]
        )
    else:
        # Ask the runner to finish what it has and go, without touching the
        # job it is running.
        lines.append('[ -f "$d/runner.pid" ] && : > "$d/stop" || true')

    lines.append(f"echo {STATE_MARKER}")
    lines.append('cat "$d/state.json" 2>/dev/null || true')
    return "\n".join(lines)


def build_stop_command(*, lane_name: str, spool_root: str, hard: bool = False) -> str:
    """Ask a lane's runner to stop; `hard` does not wait for the current job.

    Neither form signals the job itself. Stopping the scheduler and stopping
    the work are separate decisions, and conflating them would make upgrading
    the runner a reason to lose a training run.
    """
    lines = [
        f"d={spool_path_expr(spool_root, lane_slug(lane_name))}",
        '[ -d "$d" ] || exit 0',
        ': > "$d/stop"',
    ]
    if hard:
        lines.extend(
            [
                'p=$(cat "$d/runner.pid" 2>/dev/null | tr -d " \\n\\r")',
                '[ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true',
            ]
        )
    lines.append("echo NVIDB_LANE_STOPPED")
    return "\n".join(lines)


def parse_state(stdout: str) -> Optional[dict]:
    """Pull the runner's state document out of a sync command's output."""
    if not stdout:
        return None
    _, marker, tail = stdout.partition(STATE_MARKER)
    if not marker:
        return None
    text = tail.strip()
    if not text:
        return None
    try:
        state = json.loads(text)
    except ValueError:
        return None
    return state if isinstance(state, dict) else None
