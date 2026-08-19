---
name: nvidb-queue
description: Run GPU work on the user's own cluster through the nvidb job queue instead of starting it by hand. Use whenever a task needs a GPU, needs to run on a remote machine, is long-running, or must survive this session ending - training, evaluation, inference sweeps, dataset preprocessing, benchmarks. Also use to check what is running, read a job's output or result, wait for one to finish, or find out why one failed. Do NOT use for short local commands that need no GPU.
license: MIT
compatibility: Designed for Claude Code, Codex, and any Agent Skills-compatible tool.
metadata:
  author: L1ght
  version: "1.0"
  requires:
    bins: ["nvidb"]
  cliHelp: "nvidb queue --help"
---

# nvidb queue

`nvidb` schedules GPU jobs across the user's own machines. Its state lives in one
SQLite file (`~/.nvidb/queue.db`), which is the point: several agent sessions
coordinate through that file alone, without talking to each other and without
any of them needing to stay alive.

That file may be on another machine. When `~/.nvidb/queue.yml` has a `remote:`
section, every command below is forwarded to the queue host and behaves
identically — same output, same exit codes — so nothing here changes. It does
mean jobs keep being dispatched after this session ends, not merely the ones
already running.

**Never start GPU work with a bare `ssh ... python train.py &`.** A job started
that way is invisible to every other session, holds no reservation, and is lost
when its SSH connection drops. Submit it to the queue instead.

## The rule

| Situation | What to do |
| --- | --- |
| Needs a GPU | `nvidb job submit` |
| Runs on a remote machine | `nvidb job submit` |
| Takes more than a minute or two | `nvidb job submit` |
| Must outlive this session | `nvidb job submit` |
| Short local command, no GPU | run it directly |

## Always use `--json`

Every command prints one JSON document with `--json`. Parse that; do not scrape
the human tables, which adapt their columns to the data.

## Submitting

Put the command last, after `--`, or pass it as one quoted string:

```bash
nvidb job submit --json --name train --vram 20G -- python train.py --epochs 10
```

Key options:

- `--vram 20G` — **always state this.** It is the reservation the scheduler
  places against; it is not a limit on the job. Estimate honestly: too low and
  the GPU gets oversubscribed, too high and the job waits for room it will not
  use. Omitting it means "reserve nothing", which is only right for CPU work.
- `--gpus N` — GPUs required, default 1. Use `--gpus 0` for CPU-only work.
- `--node <name>` — pin to one machine; prefix matches, so `--node gpu-node` works.
  Leave it off unless the job genuinely needs that machine (its data is there,
  it needs that GPU's memory); the scheduler places it better than you can.
  A server-level `gpus: [0, 1]` allowlist is applied automatically to new
  placements and must not be treated as hiding the node's other cards.
- `--note "..."` — **write one.** Say what the job is for and what would make it
  a success. Another session, or the user in a week, reads this to understand
  why the job exists. `--name` is a label; the note is the explanation.
- `--after 12,13` — run only after those jobs complete **successfully**. This is
  how to build a pipeline and then exit: the ordering lives in the database, not
  in your session. If a dependency ends any other way — failed, cancelled, timed
  out — the dependent is **held**, not failed: it stays pending, keeps its place
  and its note, and waits for a person. Nothing cascades.
- `--after-any 12` — run once job 12 *finishes*, whatever the outcome. Use this
  when the next step should look at the wreckage rather than skip it.
- `--timeout SECONDS` — kill a job that overruns.
- `--retries N` — restart if the process vanishes (node reboot, OOM killer).
  Do not use it for jobs that are not safe to run twice.
- `--workdir DIR`, `--env KEY=VALUE`, `--priority N`, `--tag`.
- `--script FILE` — use a local file's contents as the command body. Best for
  anything multi-line; write the script, then submit it.
- `--wait` — block until the job finishes. Only when you truly cannot proceed
  without the result. It ties up your session for the job's whole runtime.

Identify yourself so the user can tell whose jobs are whose:

```bash
export NVIDB_SUBMITTER="claude-code:refactor-dataloader"
```

## Watching

```bash
nvidb queue status --json     # nodes, per-GPU budgets, jobs, open alerts
nvidb job ls --json --active  # pending and running only
nvidb job show 12 --json      # one job in full, including its events
nvidb job logs 12 -n 200      # its output
nvidb job wait 12 13 --json   # block until both finish (see the exit codes below)
```

Read commands refresh from the nodes themselves (rate-limited), so a plain
`nvidb job show` is current. Add `--no-tick` for a pure database read when
polling in a tight loop.

`nvidb job wait` exits **0** when everything completed, **1** when a job ended
badly, and **2** when `--timeout` ran out with work still going. Two is not a
failure: the job is still running and can be waited on again. Do not report a
timeout to the user as a failed job.

## Lanes: running a sequence on one GPU

A **lane** is one GPU and the order it runs its jobs in, stored in the database.
Use a lane whenever several jobs must run one after another on the same card —
a sweep, a set of arms, anything where the second job would otherwise fight the
first for memory. It is named `<node>:<gpu>`, and prefix matching works.

```bash
nvidb queue lanes --json                    # every GPU: what it runs now, what is next
nvidb queue lane box406:0 --json            # one lane's queue, in running order
nvidb job submit --lane box406:0 --vram 20G -n arm-a "bash run.sh"
```

Submitting to a lane appends to it. The lane runs its jobs strictly in order,
one at a time, starting the next the moment the previous one exits.

**Prefer a lane over an `--after` chain for same-GPU sequencing.** A chain says
"B must succeed after A"; a lane says "this card runs A then B". They differ when
something goes wrong: cancelling a job in a lane just lets the next one start,
while cancelling a link in a chain holds everything behind it for a decision.
Keep `--after` for genuine data dependencies, especially across machines.

Reordering is a local database write, so it is instant and works while the node
is unreachable or busy:

```bash
nvidb queue lane box406:0 move 1419 --to head     # run this one next
nvidb queue lane box406:0 move 1419 --before 1417
nvidb queue lane box406:0 swap 1417 1419
nvidb queue lane box406:0 skip                    # send the next job to the back
nvidb queue lane box406:0 assign 1421 --at head   # move a job into this lane
nvidb queue lane box406:0 pause                   # stop starting new jobs here
nvidb queue lane box406:0 resume
```

Positions are 1-based over the *queued* jobs, so `--to 1` means "next to run".
A running job cannot be reordered — cancel it if it must stop.

`pause` never touches what is already running: the lane finishes its current job
and then stops. That is what makes it safe to pause a lane in order to rearrange
what comes after it.

A lane stops rather than stepping over a problem. If its next job is held, or
its card is full of work the queue did not start, the whole lane waits and
`blocked` in `nvidb queue lanes --json` says which it is. That is deliberate:
the printed order is a promise about what runs next.

## Reporting progress from inside a job

A job can publish a status line by writing `$NVIDB_STATUS_FILE`; only the last
line is read, so overwrite it. This is what makes a long run watchable without
tailing its log, and it is worth adding to any script that loops:

```python
import os
with open(os.environ["NVIDB_STATUS_FILE"], "w") as handle:
    handle.write(f"epoch {epoch}/{total} loss {loss:.3f}")
```

```bash
echo "step $i/$n" > "$NVIDB_STATUS_FILE"
```

Every job also gets `CUDA_VISIBLE_DEVICES` set to its allocation, plus
`NVIDB_JOB_ID`, `NVIDB_JOB_NAME`, `NVIDB_NODE` and `NVIDB_JOB_DIR`.

## Passing results between sessions

A job that writes `$NVIDB_JOB_DIR/result.json` has that payload collected on
completion. Any later session reads it back:

```bash
nvidb job result 12 --json
nvidb job result 12 --set '{"chosen_lr": 1e-4}'   # or write one directly
```

Use this for anything a later step needs — a metric, a checkpoint path, a
decision. It survives your session ending, which a variable in your context does
not.

## When something fails

Failures become **alerts** in the queue, classified by what happened:
`job_failed` (non-zero exit), `job_lost` (process vanished), `job_timeout`,
`job_held` (a prerequisite ended badly, so this job is parked for a decision),
`job_unschedulable` (no GPU in the cluster could ever hold it), `launch_failed`,
`node_down`, `job_retried`.

```bash
nvidb queue alerts --json            # what needs attention
nvidb queue alerts --detail          # with the captured stderr
nvidb queue ack 3                    # once it is understood
```

`nvidb queue alerts` exits non-zero while anything is unacknowledged, so it is a
cheap health check to run before reporting to the user.

A failed job carries the tail of its stderr, so `nvidb job show 12 --json` gives
you the traceback without another SSH round trip. Diagnose from that first;
only fetch more with `nvidb job logs` if you need earlier output.

### Held jobs

A job whose prerequisite was cancelled, failed or timed out is **held**: still
pending, still in place, going nowhere until someone decides what it should have
waited for. `nvidb job ls --held` lists them and `nvidb job show` explains each
one. Three ways out, and the right one depends on what the user meant:

```bash
nvidb job release 1418                              # run it without the dead prerequisite
nvidb job edit 1418 --drop-after 1404 --add-after 1415   # reattach it to the re-run
nvidb job cancel 1418                               # it is genuinely moot now
```

Prefer `edit` over resubmitting. Resubmitting loses the note, the queue position
and the original submitter; editing keeps all three. Never release a job whose
prerequisite produced the input it reads — releasing it makes it run against
missing or stale data. Check what it consumes first, and when it is not obvious,
ask the user rather than guessing.

**Do not acknowledge an alert the user has not seen.** Acknowledging is the
user's signal that they have dealt with it. Report the failure, and only ack
when they say so or when you have actually fixed the cause.

## Capacity, and why a job is waiting

A GPU's schedulable memory is:

```
free = total − memory used by processes the queue did not start − reservations − headroom
```

Work the user started by hand counts fully against capacity, so the queue shares
machines rather than fighting them. A job stuck in `pending` is almost always
waiting for VRAM — check `free_mb` per GPU before assuming anything is broken.

A request no GPU in the cluster could ever hold — more GPUs than any node has,
or more VRAM than the largest card — fails immediately with `job_unschedulable`
rather than waiting forever. If that happens, the reservation was wrong, not the
cluster.

## What is actually on the GPUs

These are the user's own workstations and they run plenty of work nobody
submitted through the queue. `nvidb queue nodes` reports the whole card, not
just the queue's share:

```bash
nvidb queue nodes --json      # per-GPU memory, utilisation and every process
nvidb queue nodes --procs     # same, as a table with a PID/USER/MEM/OWNER row each
```

Per GPU: `mem_used_mb` / `mem_total_mb` / `mem_used_percent` and `util_percent`
describe the card itself; `external_mem_mb` (+ `external_procs`) is what the
queue does not manage, `queue_mem_mb` is what its own jobs hold right now, and
`reserved_mb` is what it has promised them. `processes[]` names them all, each
with `managed` and, when the queue started it, `job_id`.

**Read `util_percent` and `external_mem_mb` before telling the user a machine is
free.** A GPU at 99% utilisation with no queue jobs on it is busy with the
user's own work; scheduling onto it is legal but slow, and worth mentioning.

Some GPUs report `"attribution": "blind"`: their driver (WSL) accounts for none
of the memory in use, either naming no processes or naming them without memory
figures. The foreign/queue split there is inferred from reservations, not
measured, so treat it as approximate.

## Managing

```bash
nvidb job cancel 12          # kill the remote process group
nvidb job requeue 12         # run a finished job again
nvidb job note 12 --append "loss plateaued at epoch 30"
nvidb job priority 12 5      # dispatch earlier (bare number sets, +1/-2 adjust)
nvidb job priority 12 --up 2 # move a pending job two slots earlier instead
nvidb queue drain gpu-node        # stop scheduling onto a node (`resume` undoes it)
nvidb queue ignore offline-node   # do not probe, schedule, or normally display it
nvidb queue unignore offline-node # restore it (`nodes --include-ignored` finds hidden nodes)
```

Cancelling is destructive and discards work in progress. **Confirm with the user
before cancelling a job you did not submit** — another session may be waiting on
it.

Draining and ignoring are different. A drained node is still probed so jobs
already running there can finish. Ignore an unavailable node only when it has no
running queue jobs; it then disappears from normal status/TUI views and makes no
SSH connections until it is unignored.

## Things to avoid

- Starting GPU work over raw SSH. It is invisible to the scheduler and to every
  other session.
- Submitting without `--vram`. The job reserves nothing and can oversubscribe a
  GPU that looked free.
- Pinning with `--node` out of habit. It defeats the scheduler and leaves jobs
  queued behind a busy machine while another sits idle.
- Polling `nvidb job ls` in a tight loop. Use `nvidb job wait`, or poll with
  `--no-tick` and a sleep.
- Acknowledging alerts to make the output look clean.

## More

`nvidb queue --help`, `nvidb job submit --help`, and section 3 of the nvidb
README cover the rest: the TUI (`nvidb queue`), the optional notification daemon
(`nvidb queue daemon`), the event log (`nvidb queue events --since <id>`, for
catching up on what happened while you were not running), the keeper that keeps
a queue host scheduling (`nvidb queue keeper status`), consistent database
snapshots (`nvidb queue backup --json`), and the `queue:` settings in
`~/.nvidb/config.yml` or `~/.nvidb/queue.yml`.
