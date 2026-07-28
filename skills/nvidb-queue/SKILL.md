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
- `--node <name>` — pin to one machine; prefix matches, so `--node gem12` works.
  Leave it off unless the job genuinely needs that machine (its data is there,
  it needs that GPU's memory); the scheduler places it better than you can.
- `--note "..."` — **write one.** Say what the job is for and what would make it
  a success. Another session, or the user in a week, reads this to understand
  why the job exists. `--name` is a label; the note is the explanation.
- `--after 12,13` — run only after those jobs complete. This is how to build a
  pipeline and then exit: the ordering lives in the database, not in your
  session. If a dependency fails, the dependent job fails rather than hanging.
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
nvidb job wait 12 13 --json   # block until both finish; non-zero if any failed
```

Read commands refresh from the nodes themselves (rate-limited), so a plain
`nvidb job show` is current. Add `--no-tick` for a pure database read when
polling in a tight loop.

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
`dependency_failed`, `launch_failed`, `node_down`, `job_retried`.

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
nvidb queue drain gem12      # stop scheduling onto a node (`resume` undoes it)
```

Cancelling is destructive and discards work in progress. **Confirm with the user
before cancelling a job you did not submit** — another session may be waiting on
it.

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
catching up on what happened while you were not running), and the `queue:`
settings in `~/.nvidb/config.yml`.
