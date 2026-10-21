# nvidb cluster job queue

`nvidb queue` adds a small slurm-like scheduler on top of the servers already in
`config.yml`. Jobs are submitted from any machine that can SSH to the nodes, wait
until a GPU has room for them, run detached on the node, and report back.

`nvidb q` is a short alias for `nvidb queue`; every queue subcommand accepts
either spelling.

The queue's only shared state is one SQLite file (`~/.nvidb/queue.db`). Several
independent clients — several Claude Code sessions, a script, an open TUI —
coordinate purely by reading and writing that file, so none of them has to stay
running for the queue to work.

## How it schedules

There is no daemon. Any command that touches the queue performs a **tick**:
probe every node, settle finished jobs, then place whatever now fits. Ticks are
rate-limited and guarded by a lease, so a burst of concurrent clients results in
one round of SSH traffic rather than one per client. The open TUI ticks on a
timer, and `nvidb queue tick --watch 10` gives you a background ticker if you
want one.

GPUs default to **exclusive** allocation: each card runs at most one queue
job, regardless of `max_jobs_per_gpu` or lane concurrency. Cards with detected
external GPU processes are not offered to new exclusive jobs. This is queue
coordination, not OS-level isolation from processes started outside nvidb.

For your own workloads that benefit from sharing, configure the queue host:

```yaml
queue:
  gpu_mode: shared       # default: exclusive
  max_jobs_per_gpu: 4
  headroom_mb: 512
```

In shared mode every GPU submission **must explicitly supply a positive
`--vram`**, estimating the maximum memory it will use **on each GPU**:

```bash
nvidb job submit --gpus 2 --vram 20G -- python train.py
```

This reserves 20 GiB on each of two cards. Missing or zero estimates are
rejected, even if `default_vram` is configured. CPU-only jobs (`--gpus 0`)
are exempt. Existing pending jobs without a positive estimate wait until
their request is corrected. Running jobs are not stopped by a mode change.
Already issued allocations retain their mode until that attempt finishes;
in particular, switching to shared does not take away an existing exclusive
allocation. `allocation_mode` in job JSON shows the mode of that allocation.

Shared GPUs are allocated by **VRAM budget**:

```
free = total − memory used by non-queue processes − reservations of queue jobs − headroom
```

Counting foreign processes is what lets the queue share machines with work you
started by hand: a card that someone else has filled simply reports no capacity,
and jobs go elsewhere until it frees up. A job is charged the larger of what it
reserved and what it actually uses. An estimate is a scheduling reservation,
not a hard memory limit: underestimating a later peak can still cause OOM.
CPU-only jobs (`--gpus 0`) reserve nothing, so they are bounded by a
count instead: `max_cpu_jobs_per_node`, four by default.

Jobs staged for a lane runner reserve capacity before they start, so the
ordinary queue cannot reuse that capacity. An exclusive lane may prefetch its
next job while retaining ownership of its cards; the runner executes that
sequence one job at a time. Shared-mode prefetch reserves each staged job's
memory and a per-GPU slot, and is limited by available resources.
Runners pause new GPU starts when memory probing fails. A staged job's memory
request cannot be edited because it may already have started; cancel and
resubmit it instead.

Job attribution defaults to the submitting computer's global Git `user.name`
and `user.email` (`Name <email>`). Remote submissions carry this identity to
the queue host. `--submitter` and `NVIDB_SUBMITTER` override it; when Git identity
is unavailable nvidb falls back to the local user, hostname and process ID.

A job cancelled or timed out while its node was unreachable keeps its record
final but its process alive. The pid is remembered separately from the job — it
has to outlive both `job requeue` and `job purge` — and the first probe that
reaches that node again kills it, provided the pid is still running that job's
`run.sh`. `nvidb queue status --json` reports how many such cleanups are
outstanding as `pending_reaps`.

On nodes where the driver accounts for none of the memory in use — WSL, where
the GPU is driven from Windows, and which either names no processes at all or
names them without memory figures — the split between foreign work and the
queue's own jobs cannot be measured. Those GPUs are marked `blind` and their own
jobs are credited up to their reservation, which keeps the queue from charging a
job twice.

## What is on the GPUs

The queue's own bookkeeping is only half the picture: these are workstations,
and most of what runs on them was started by hand. `nvidb queue nodes` reports
the whole card. Like `queue status`, it leaves out cards outside a server's
`gpu_ids`.

```bash
nvidb queue nodes               # capacity, plus the unmanaged work behind it
nvidb queue nodes --procs       # every process on every GPU, with its owner
nvidb queue nodes --json        # the same as structured data
```

```
  workstation  [up]  10.0.0.2  seen 00:00:00 ago
    GPU   MODEL                       UTIL  MEM              UNMANAGED      QUEUE  RESERVED  JOBS  FREE
    GPU0  NVIDIA GeForce RTX 3090 Ti  10%   23.2G/24.0G 97%  15.2G (blind)  0M     8.0G      1     316M
      unmanaged  ~15.2G of 23.2G in use; this driver reports no per-process memory
        pid 1608997  /python3.11  alice  job 12 sweep
```

`UTIL` and `MEM` describe the card itself, whoever is using it. `UNMANAGED` is
what the queue did not start — `(2p)` counts the processes behind it, `(blind)`
means the figure is inferred rather than measured. `QUEUE` is what the queue's
own jobs hold right now, against the `RESERVED` they were promised.

In JSON each GPU carries `mem_used_mb`, `mem_total_mb`, `mem_used_percent`,
`util_percent`, `external_mem_mb`, `queue_mem_mb`, `reserved_mb`, `free_mb`, and
a `processes` list naming every process with `managed` and `job_id`.

## Submitting and watching jobs

```bash
# Put the command last, after `--`, or pass it as one quoted string
nvidb job submit --name train --vram 20G -- python train.py --epochs 10
nvidb job submit --node gpu-node --vram 8G --workdir ~/proj -- python eval.py
nvidb job submit --pool experiments --vram 16G -- python train.py  # any card in a pool
nvidb job submit --gpus 0 -- python prepare_data.py        # CPU-only
nvidb job submit --after 12 --vram 4G -- python report.py  # runs after job 12
nvidb job submit --script run.sh --vram 12G                # a local file's contents

nvidb job ls                    # everything, newest state
nvidb job ls --active           # only pending and running
nvidb job show 12 --logs 40     # detail plus the tail of stdout
nvidb job logs 12 -f            # follow the output
nvidb job wait 12 13            # block until both finish (0 ok, 1 failed, 2 timed out)
nvidb job edit 12 --vram 39G    # change a pending job's reservation in place
nvidb job edit 12 --command 'python evaluate.py --limit 200' --no-tick
nvidb job edit 12 --script evaluate.sh --no-tick  # replace with a local file's contents
nvidb job cancel 12             # kill the remote process group
nvidb job requeue 12            # run a finished job again
nvidb job priority 12 5         # set the priority (bare number sets, +1/-2 adjust)
nvidb job priority 12 --up 2    # move a pending job two slots earlier instead
nvidb job purge                 # forget finished job records

nvidb queue status              # nodes, capacity and jobs in one view
nvidb queue status --procs      # the same, with every GPU process listed
nvidb queue pools               # each named pool's cards, their state, and its jobs
nvidb queue events --since 42   # replay what happened while you were away
nvidb queue drain gpu-node        # stop scheduling onto a node (`resume` undoes it);
                                  # jobs already on it keep running and still report back
nvidb queue ignore offline-node   # stop probing/scheduling and hide a disconnected node
nvidb queue unignore offline-node # restore it; `nodes --include-ignored` lists hidden nodes
nvidb queue tick                # force one scheduler pass
```

Useful `submit` options: `--priority N` (higher goes first), `--timeout SECONDS`
(kill an overrunning job), `--retries N` (restart if the process vanishes),
`--env KEY=VALUE`, `--tag`, `--note`, and `--wait` to block until the job
finishes.

**GPU pools** name a set of cards that may span several nodes. A job submitted
with `--pool NAME` waits in the ordinary queue and is placed on whichever member
card fits it when it starts, using the same budget, exclusive/shared rules and
`placement` policy as any other job. Pools are defined on the queue host:

```yaml
queue:
  pools:
    experiments:
      members:
        - node: lab1
          gpus: [0]
        - node: lab2            # no `gpus`: every queue GPU on the node
    large-memory: [lab3]  # a bare list of members also works
```

- A member never widens the node's `gpu_ids` allowlist.
- Pools may overlap; a card in two pools still has one budget, not two.
- A multi-GPU job still needs all of its cards on one node. A job too wide or
  too large for every member card the queue has seen fails at once; a pool
  none of whose cards has been seen yet keeps its jobs waiting.
- Membership is read on every pass: editing it moves jobs that are still
  waiting, while running and staged jobs keep their card. A pending job whose
  pool is removed is held, not failed, and continues when the pool returns.
- `--pool` cannot be combined with `--node` or `--lane`; `lane assign` takes a
  job out of its pool.

Job listings show a waiting pool job's place as `pool:NAME`; JSON carries `pool`.

`job edit --vram` changes the reservation of a `pending` job without replacing
its record, so its lane/queue position, note and submitter are preserved. The
command runs a scheduler pass afterward unless `--no-tick` is given. A running
job's reservation cannot be changed because its placement is already active.

`job edit --command '...'` replaces a pending job's command in place;
`--script FILE` reads the replacement from a local UTF-8 file, including when
the queue host is remote. Use one of these options at a time. The job keeps
its ID, dependencies, lane position, working directory, environment and notes.
A command-only edit also preserves dependency holds, and a `job_command` event
records the previous and replacement text. Add `--no-tick` to edit without
starting a scheduler pass.

Commands can only change before the job has been staged on its node. A pending
job with a `run_dir` may already have started in the runner even though the
database has not observed it yet, so that edit is refused. Running and finished
jobs also refuse command edits.

Read commands refresh the queue themselves before printing, so `nvidb job show`
never reports stale state. Pass `--no-tick` for a pure database read, or
`--tick` to force a refresh.

Every job runs with `CUDA_VISIBLE_DEVICES` set to its allocation, plus
`NVIDB_JOB_ID`, `NVIDB_JOB_NAME`, `NVIDB_NODE`, `NVIDB_JOB_DIR`, and
`NVIDB_STATUS_FILE`.

## Notes and live progress

A job carries two independent pieces of free text, kept apart because they have
different authors and neither should be able to overwrite the other:

**`note`** — what you or a client says *about* the job. Set it at submit time and
edit it whenever, including long after the job has finished:

```bash
nvidb job submit --note "baseline A, lr=1e-4" -- python train.py
nvidb job note 12                              # read it
nvidb job note 12 --append "loss plateaued at epoch 30"
nvidb job note 12 "superseded by job 19"       # replace
nvidb job note 12 --clear
```

**`progress`** — what the job says about *itself*. The job writes one line to
`$NVIDB_STATUS_FILE` and the scheduler collects it on every probe, so you can
watch a long run without tailing its log:

```python
# inside a training script
import os
with open(os.environ["NVIDB_STATUS_FILE"], "w") as handle:
    handle.write(f"epoch {epoch}/{total} loss {loss:.3f}")
```

```bash
echo "epoch $i/$n loss $loss" > "$NVIDB_STATUS_FILE"   # or from shell
```

Only the last line of the file is read, so overwriting is the normal pattern.
Both fields appear in `job ls`, `queue status`, `job show`, the TUI and every
`--json` payload. The last thing a job reported is kept on its finished record,
which is often the quickest explanation of how far a lost or failed job got. A
retry clears the stale status line but keeps your note.

## Failures, alerts and the optional daemon

When something goes wrong on a node, the queue records an **alert** locally,
classified by what actually happened:

| Kind                | Raised when                                          | Severity |
| ------------------- | ---------------------------------------------------- | -------- |
| `job_failed`        | the command exited non-zero                          | error    |
| `job_lost`          | the process vanished with no exit status             | error    |
| `job_timeout`       | the job passed `--timeout` and was killed            | error    |
| `dependency_failed` | a job can never run because a dependency failed      | error    |
| `job_unschedulable` | no GPU in the cluster is big enough to ever hold it  | error    |
| `job_retried`       | the process vanished but a retry remains             | warning  |
| `launch_failed`     | the job could not be started (disk full, permissions)| warning  |
| `node_down`         | a node stopped answering                             | error    |

For a failed job the queue pulls the tail of its stderr (falling back to stdout)
onto the alert, so the reason is readable locally without another round trip:

```bash
nvidb queue alerts            # what needs attention
nvidb queue alerts --detail   # ... with the captured output
nvidb queue ack 3             # acknowledge one
nvidb queue ack --all         # acknowledge everything
```

`nvidb queue alerts` exits non-zero while anything is unacknowledged, so a shell
or an agent can branch on it. Alerts stay until acknowledged; they are never
re-raised for the same failure, and `nvidb queue status` and the TUI both lead
with them.

Recording an alert and delivering it are separate. Recording happens on whatever
scheduler pass notices the failure, so nothing is lost when nobody is watching.
Delivery is the daemon's job:

```bash
nvidb queue daemon                      # tick every 15s and push failures
nvidb queue daemon --interval 5
nvidb queue daemon --once --json        # a single pass, for cron
```

The daemon is **optional** — the queue works exactly as before without it, since
every command runs a scheduler pass. Run it when you want failures pushed to you
rather than waiting to be asked, and for prompt timeout enforcement: without it,
a job that overruns is only killed the next time some command happens to tick.

Each alert is delivered once, whether or not the daemon restarts. Channels are
configured under `queue.notify`: a desktop notification, a JSON-lines file at
`$NVIDB_HOME/alerts.log`, and a `command` hook that receives the alert as JSON
on stdin — which is how you route failures to anything else.

On macOS, alerts use native notifications with a dedicated **nvidb** app name
and icon. On macOS 11 or later, the small notification helper is built with
Apple's Command Line Tools (`xcode-select --install`) and cached at
`$NVIDB_HOME/nvidb.app`. Allow notifications for nvidb when macOS
first asks; its banner/alert style and other options are managed in
System Settings → Notifications → nvidb. Notifications retain the system's
dismiss button, Focus behavior and Notification Center history. The daemon
must run in a logged-in graphical session. Restart the daemon or keeper after
updating nvidb to load the new notification code.
If the helper cannot be built, nvidb logs the error and falls back to
`terminal-notifier` or `osascript`, which uses that tool's icon instead.

## Driving the queue from other programs

Every command accepts `--json` and prints one JSON document on stdout, which is
the intended way for tools — Claude Code sessions in particular — to use the
queue:

```bash
nvidb queue status --json     # nodes, per-GPU budgets, job table, counts
nvidb job submit --json --vram 20G -- python train.py
nvidb job wait 12 --json --logs 40
nvidb queue events --json --since 42
```

Three things make the database usable as a coordination channel between
processes that never talk directly:

- **Dependencies.** `--after 12,13` records the ordering in the queue, so a
  client can lay out a pipeline and exit; the jobs still run in order.
- **Results.** A job that writes `$NVIDB_JOB_DIR/result.json` has that payload
  collected on completion and served by `nvidb job result <id>`. Clients can
  also write one directly with `nvidb job result <id> --set '{"note":"..."}'`,
  which passes structured data between them without a shared filesystem.
- **Events.** `nvidb queue events --since <id>` replays every state change, so a
  client that was not running can catch up on exactly what it missed.

### Agent skill

[`skills/nvidb-queue/`](../skills/nvidb-queue/SKILL.md) is an Agent Skill that
teaches Claude Code, Codex and other skill-aware tools to route GPU work through
the queue instead of starting it over raw SSH. Install it by symlinking, so both
agents track the repository:

```bash
ln -s "$PWD/skills/nvidb-queue" ~/.claude/skills/nvidb-queue
ln -s "$PWD/skills/nvidb-queue" ~/.codex/skills/nvidb-queue
```

## The queue TUI

```bash
nvidb queue                 # or: nvidb queue tui
```

The screen opens in **Overview**, with a short resource summary and one list
of jobs. Each job shows its name and ID, full status, placement, and progress
or a known waiting reason. Pending jobs show holds, paused queues, dependencies,
or their actual lane position when those facts are available; otherwise they
say they are waiting for scheduling. Narrow terminals use two lines per job.
Commands, detailed memory readings and other metadata remain in job detail.

The queue shares the monitor TUI's rounded grey frame, cyan table headers,
selection band and `classic` / `muted` colour themes. The active filter has a
`●` marker, and the version stays at the top right. Below 72 columns or 14 rows,
the outer frame disappears to leave more room for jobs and keyboard hints.

Click **Active**, **Running**, **Waiting**, or **Finished** to filter the list.
The bottom bar keeps **Detail**, **Log**, **Cancel**, and **More** close at hand;
**View** appears beside the filters or in the action bar. Cancellation still
requires a second confirmation. **More** contains
sorting, priority, reordering, scheduler controls, and other actions.
The navigation hint stays visible: `↑`/`↓` or `j`/`k` selects a job, `Enter`
opens its details, `m` opens More, `?` shows help, and `q` quits. Scrolling the
server summary does not move keyboard focus away from jobs; returning to
Overview also restores job focus.

Use **View** to choose Overview, Task flow, or Server capacity; `v` cycles
through them in that order. `Tab` from Overview opens Server capacity when it
moves focus to the nodes, so resource selection is always visible. Resource
views follow the active server configuration order (the same list as the
monitor TUI unless `queue.yml` deliberately overrides `servers`); removed
servers do not linger in the normal view.

In **Task flow**, every GPU gets a row that reads from left to right: live
utilisation and free VRAM, the running job, the number of jobs in that card's
lane, and as many ordered job cards as fit:

```text
G0  30% · 23.5G free  [ RUN #10 train 01:00:00 ] ──▶ { lane Q3 } ──▶ [1:#11 P+5 eval ←✓#10] ──▶ [2:#12 P0 report ←◇#11]  +1
ANY GPU ──▶ { Q2 · priority } ──▶ [1:#20 P+8 urgent]
DEPS #10 ─✓▶ #11 · #11 ─◇▶ #12
```

`lane Q3` is the queue attached to that exact card; the numbered positions are
its real execution order. `P+5` is the job priority (it controls dispatch in the
shared `ANY GPU` pool and is informational once a job has a fixed lane).
`✓` dependencies require successful completion, while `◇` dependencies accept
any terminal result. Held jobs and paused or blocked lanes are called out on
their GPU row. Pending GPU jobs without a lane are deliberately kept in the
`ANY GPU` pool because the scheduler has not assigned them to a card yet.
Jobs submitted with `--pool NAME` wait there too, marked `∈NAME`.

The GPU activity badge distinguishes queue-managed `RUN`, `EXTERNAL` work,
unattributed `BUSY` activity, confirmed `IDLE`, and `UNKNOWN` readings (including
unreachable nodes). Clicking `lane QN`, the shared pool, or a `+N` overflow count
opens the complete pending list for that pool. Job detail includes the lane
position and each dependency's state, even when the upstream job is older than
the recent-job list. Commands and working directories wrap across detail pages.

Choose **Server capacity** for memory bars and process details.
All SSH work happens on a worker thread, so an
unreachable node slows the numbers down but never freezes the interface.

| Key                | Action                                        |
| ------------------ | --------------------------------------------- |
| `j` / `k` / ↑ / ↓ | Move the node or job selection                |
| `←` / `→`         | Select a GPU on the focused node              |
| `PgUp` / `PgDn`    | Scroll resource rows, or page job selection   |
| `Tab`              | Switch focus between the node and job panes   |
| `Enter`            | Scope to selected node/GPU, or toggle detail  |
| `[` / `]`          | Page through wrapped detail or log text       |
| `L`                | Toggle a live tail of the selected job's log  |
| `c`                | Cancel the selected job (press twice)         |
| `r`                | Re-queue the selected finished job            |
| `+` / `-`          | Raise or lower the selected job's priority    |
| `K` / `J`          | Move a pending job up/down the dispatch order |
| `s` / `S`          | Cycle the sort column / flip its direction    |
| `t`                | Force a scheduler tick now                    |
| `a`                | Toggle automatic ticking                      |
| `f`                | Cycle the job filter                          |
| `x` / `Esc`        | Clear the current server/GPU/task-pool scope  |
| `v`                | Cycle Overview / Task flow / Server capacity  |
| `p`                | Open capacity view; cycle GPU process detail  |
| `d`                | Drain or resume the selected node             |
| `A`                | Acknowledge every open alert                  |
| `?`                | Help                                          |
| `m` / More         | Open actions hidden by the terminal width     |
| `q`                | Quit                                          |
| Mouse click        | Scope server/GPU jobs, select, sort or act    |
| Mouse wheel        | Scroll resources/help, jobs, or detail/log    |

The job list opens in **queue order**: running jobs first, followed by pending
jobs in their queue order. Runtime sorting, longest-running first, remains
available through **More → Sort** or `s`. Using `K`/`J` restores queue order
before moving a pending job.
For lane jobs, queue sorting follows lane position and `K`/`J` moves the job
within its lane without changing priorities. Lanes are grouped independently;
there is no single dispatch order across GPUs. Free-pool reordering adjusts
priorities; the detailed tables in Task flow and Server capacity retain the
`PRI` column. Clicking a column header sorts
by that column (a second click flips it). In the server-capacity view, each GPU
line draws one memory bar whose segments distinguish foreign memory (amber)
from this queue's reservations (teal) and free space (dim). Colours are
deliberately muted: healthy values render grey or plain, and saturation is
reserved for states that need attention.

Mouse reporting is enabled by default. Clicking a server limits the lower table
to jobs running on that server; clicking any GPU cell drills down to jobs whose
allocation includes that card. Clicking a running or queued task card in the
flow opens that job directly; clicking it again hides its detail. `[all jobs]`,
`x`, or `Esc` returns to the global job view. Click a job-table row to select it;
clicking the selected row again shows or hides its detail pane. The wheel acts
on the pane under the pointer. Status counts select their matching global job
filter, and clicking a job alert opens that job's log. The bottom action bar
exposes the actions that apply to the current selection, including the second
confirmation required for cancellation. **Filter** and **Sort** in **More** open direct
choice menus; choosing the current sort field reverses it. **More** keeps
overflow actions accessible with the mouse even in narrow windows. Menus
support the wheel, clickable page controls, arrow keys, Enter and Esc. Buttons,
job/node/GPU rows and menu choices highlight on hover without changing focus
or submitting an action.
Most terminals reserve normal text
selection for `Shift`-drag or `Option`-drag while mouse reporting is active. Set
`mouse: false` under `view` in `~/.nvidb/config.yml` to disable it in both nvidb
TUIs.

The resource pane scrolls independently, so GPU rows and the shared/dependency
pools remain reachable even when a single node is taller than the pane. In
help, use `j`/`k`, PgUp/PgDn, or the wheel to read every command; `g`/`G` jumps
to the first/last help page.

## What runs on the nodes

Nothing is installed. A generated `run.sh` is delivered over SSH and started
with `setsid`, so the job outlives the client that launched it. Each job keeps a
directory on its node (`~/.nvidb/jobs/<id>/` by default) holding the script,
`stdout.log`, `stderr.log`, its pid, and the exit status with the time it
finished. Every later interaction — checking liveness, reading output, killing a
job — is a single shell round trip.

Tuning lives under `queue:` in `config.yml`; see
[config.example.yml](../config.example.yml) for the full set of keys.

## Keeping the queue on a machine that stays on

Everything above assumes some client eventually runs a scheduler pass. That
stops being enough when the point is to submit work and walk away: a job queued
just before a laptop is closed stays `pending` until something looks again.
Jobs already **running** are unaffected — they were detached from the SSH
session that started them — so the only thing missing is somebody to dispatch.

Move the queue onto a machine that stays on, and the laptop becomes a client of
it. The queue's own configuration lives in its own file, so one `~/.nvidb` can
hold both a monitor watching many machines and a queue scheduling onto a few:

```yaml
# ~/.nvidb/queue.yml on the queue host — see queue.example.yml
servers:
  - hostname: "gpu-node.example.com"
    port: 2222
    username: "user"
    nickname: "gpu-node"
    gpu_ids: [0, 1]  # optional: only new queue jobs are restricted
queue:
  include_local: true
  local_node_name: "queue-host"
```

The **keeper** keeps that machine scheduling. A new installation selects launchd
on macOS, systemd where a Linux user session supports it, or the portable shell
supervisor otherwise. Reinstalling preserves the existing manager unless an
explicit manager flag is supplied:

```bash
nvidb queue keeper install --start   # write ~/.nvidb/queue-keeper.sh and run it
nvidb queue keeper status            # non-zero while nothing is keeping the queue moving
nvidb queue keeper logs -n 50
nvidb queue keeper stop
```

On a Linux host with a working user systemd session, install a user service
instead. `--start` enables and starts `nvidb-queue.service`; systemd then
restarts the daemon after failures and starts it with the user's next session:

```bash
nvidb queue keeper install --systemd --start
systemctl --user status nvidb-queue.service
```

Starting the user service during boot, before that user logs in, additionally
requires lingering where the host permits it:

```bash
loginctl enable-linger "$USER"
```

On macOS, install a LaunchAgent with
`nvidb queue keeper install --launchd --start`. It runs while the user is logged
in and does not keep a sleeping computer awake. Use `--shell` to request the
portable supervisor explicitly.

The shell supervisor does not survive a reboot. Ordinary local and remote
commands can recover an installed keeper after an unexpected exit.
`keeper stop` persists an explicit stop: automatic ensure respects it until
`keeper start` or `keeper restart`. Commands using `--no-tick` skip automatic
ensure as well. A client configured with `remote:` opens no database of its own and
forwards the whole command line to the queue host, running the keeper's `ensure`
in the same round trip. A missing or failed keeper is reported on stderr rather
than silently leaving the queue idle:

```yaml
# ~/.nvidb/queue.yml on the laptop
remote:
  host: "queue-host.example.com"
  nvidb: "/home/user/.local/bin/nvidb"   # absolute: a non-interactive shell has a thin PATH
```

`nvidb job submit`, `job wait`, `logs --follow`, `queue status` and the TUI then
behave exactly as they do locally — it is the same CLI, running over there — and
the exit codes come back unchanged. `--local` runs one command against this
machine's own queue instead, and `nvidb queue status` reports `keeper up` or
`keeper DOWN` wherever a keeper is installed.

Back up the queue with SQLite's online backup API, which includes committed WAL
state without pausing the daemon:

```bash
nvidb queue backup                         # ~/.nvidb/backups/queue-<time>.db
nvidb queue backup /secure/queue-copy.db  # explicit destination, never overwritten
```

The daemon can create and rotate backups automatically:

```yaml
queue:
  backup:
    enabled: true
    interval_hours: 24
    keep: 7
    directory: null        # defaults to $NVIDB_HOME/backups
```

Each snapshot is checked with `PRAGMA quick_check` before its complete temporary
database is atomically published without replacing an existing path.
`nvidb queue status --json` includes `last_backup_at`.

---
## Upgrade notes

### Upgrading an existing queue to v1.9.3

Existing job IDs, attempts, dependencies, lane positions, concurrency values and
`lane_lookahead` settings are retained. The default lookahead remains 1; this
release does not enable `all`, `auto`, ready-first dispatch or candidate lanes.

- An unreachable job that exceeds its runtime limit stays `running` and raises
  one `job_overdue` warning per attempt. Reconnection settles the job from the
  node's result. A timeout request is recorded on that attempt before signalling;
  the controller waits for evidence that the process has stopped.
- Dependency holds are re-evaluated outside the staging window. Requeueing a
  prerequisite clears obsolete holds without removing dependencies or changing
  queue positions.
- Lane concurrency and pause settings live in `lane.json`. Lowering concurrency
  lets existing work finish and limits subsequent starts. A runner upgrade or
  `runner restart` waits for a safe handover and adopts the same job processes.
  Replayed syncs do not recreate claimed or completed attempts.
- Parse errors requested with `--json` are JSON with exit status 2.
  `job cancel --no-tick --json` is accepted; `--no-tick` skips automatic
  scheduling, not cancellation of the named jobs.

Before changing an installed service, create a queue backup. Restart the
controller to load the new code; running jobs retain their own process sessions.
This release adds only an optional `overdue_at` column. Automatic historical
correction is limited to a matching timeout orphan for the same attempt and
run directory, with a recorded earlier node completion and no node timeout
marker. It never rewrites a requeued attempt or a user cancellation.

### Cancelling queued lane jobs in v1.9.4

`nvidb job cancel ID... --no-tick --json` can immediately cancel pending runner
lane jobs that have not been staged, even while another tick is waiting on SSH.
This path changes only the named jobs and makes no node connections. Reserving
a run directory and cancelling both check the current record in a short database
transaction, so a stale tick cannot publish a successfully cancelled job.

Staged jobs, running jobs, free-pool jobs and lanes without a runner retain the
exclusive cancellation path and can still report `scheduler is busy`.
An active older controller also retains that protection: restart or let it exit
before relying on concurrent cancellation. The patch changes neither the runner
script nor lane settings, and requires no schema migration.
