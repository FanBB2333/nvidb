# Proposal: pin a job to physical GPUs (`job submit --gpu-ids`)

Status: implemented in 88220ed. Written against `main` at 4abd7da.

## Problem

A multi-GPU job can only be pinned to a node; the scheduler picks the cards.
`_find_placement` sorts the usable cards by free memory, then utilisation, then
index, and takes the first `--gpus` of them. On an idle 8-GPU node, a 4-GPU job
asked to use cards 4-7 was given `2,0,1,3`. That order became
`CUDA_VISIBLE_DEVICES`, so `cuda:0` inside the job was physical card 2.

The submitter needs a way to say "exactly these cards, in this order".

## Behaviour

```bash
nvidb job submit --node lab1 --gpu-ids 4,5,6,7 --vram 23G -- python serve.py
```

- **Meaning.** `--gpu-ids` takes physical indices on the `--node`, numbered as
  `nvidia-smi` numbers them. The job runs on exactly those cards, and their order
  becomes `CUDA_VISIBLE_DEVICES`: `4,5,6,7` means `cuda:0` is card 4.
- **Requires `--node`.** Indices mean nothing without a machine.
- **`--gpus` defaults to the number of ids.** An explicit `--gpus` that disagrees
  with the ids is rejected.
- **Incompatible options.** `--gpu-ids` cannot be combined with `--pool`, which
  chooses the card, or `--lane`, which already names its cards.
- **Waiting.** The job waits in the ordinary (free) queue until every named card
  fits its `--vram` under the usual exclusive or shared rules. It never falls
  back to other cards.
- **Rejected at submit** (when the node's cards are known):
  - indices that are negative or repeated;
  - an index the node does not have;
  - an index outside the node's `gpu_ids` allowlist.
- **After submit.**
  - If a named card disappears from a node in view, or the card is too small for
    `--vram`, the job fails as unschedulable, as a too-wide job does today.
  - If an allowlist edit excludes a named card later, the job is held, not
    failed, like a job whose pool was removed. It continues when the card
    returns.
- **Lanes and requeue.** `lane assign` clears the pin, because the lane names the
  cards. `requeue` keeps it.

## Changes, file by file

### `nvidb/sched/db.py`

- `_SCHEMA`, table `jobs`: add `gpu_constraint TEXT` after `node_constraint`,
  with a comment. It holds comma-separated physical indices in request order,
  and is NULL when the scheduler may choose.
- `_ADDED_COLUMNS["jobs"]`: add `"gpu_constraint": "TEXT"` so existing databases
  get the column. `SCHEMA_VERSION` can stay at 2, as it did for `pool`.
- `insert_job`: store `",".join(str(i) for i in fields.get("gpu_constraint") or []) or None`.
- `update_job`: csv-encode a list or tuple `gpu_constraint`, as it already does
  for `gpu_ids`.

### `nvidb/sched/model.py`

- `Job`: add `gpu_constraint: List[int] = field(default_factory=list)` next to
  `node_constraint`, with a comment.
- `Job.from_row`: `gpu_constraint=_csv_ints(data.get("gpu_constraint"))`.
- `Job.to_dict`: add `"gpu_constraint"`. This is the JSON field other tools read.
- `requested_place`: when both are set, return `f"{node_constraint}:{csv}"`, so
  `job ls` and status show `lab1:4,5,6,7` for a waiting pinned job.

### `nvidb/sched/scheduler.py`

- **`Scheduler.submit`.** Add the parameter `gpu_ids: Optional[Sequence[int]] = None`.
  - Reject the pin together with `pool` or `lane`, and require `node`.
  - Normalise to a list of ints, keeping the order. Reject negatives and
    duplicates, and reject `int(gpus) != len(gpu_ids)`.
  - After `resolved_node` is known: if `dbm.get_node(...)` has GPU rows, every
    index must be among them. If `self._gpu_allowlists.get(resolved_node)` is
    set, every index must be in it. Error messages name the indices that are
    available.
  - Pass `gpu_constraint=` to `dbm.insert_job` and add `"gpu_ids"` to the
    `job_submitted` event data.
- **`Scheduler._find_placement`.** Inside the per-node loop, after `usable` is
  built:
  - if `job.gpu_constraint` is set, map `usable` by index;
  - skip the node if any pinned index is missing;
  - otherwise set `chosen = [by_index[i] for i in job.gpu_constraint]`, keeping the
    request order and not sorting.

  The rank and slack logic can stay; with `node_constraint` set there is one
  candidate node anyway.
- **`Scheduler._impossible_reason`.** Add a branch for pinned jobs before the
  generic width and VRAM checks, judged against those cards alone, as the lane
  branch is:
  - if the node is not in `inventory`, return None (wait);
  - an index not among the node's GPUs gives the reason
    `asks for GPU 9 on lab1, which has no such GPU (has 0-7)`;
  - a pinned card with `mem_total_mb - headroom < vram_mb` gives
    `reserves 23 GiB; GPU 4 on lab1 holds 22.5 GiB at most`.
  - Allowlist exclusion is left to the hold below.
- **`Scheduler._request_hold_reason`.** If any pinned index is outside the node's
  current allowlist, return a hold such as `GPU 5 on lab1 is outside the node's
  gpu_ids allowlist; restore it or cancel the job`. This mirrors the
  removed-pool case. `_refresh_holds` then lifts the hold on its own.
- **`Scheduler.lane_assign`.** When moving a job into a lane, also set
  `gpu_constraint=None` next to the existing `pool=None`.

### `nvidb/sched/cli.py`

- **`_register_submit_parser`.**
  - Add `--gpu-ids` with `metavar="IDS"` and help text along the lines of "Run
    on exactly these GPUs of --node, e.g. 4,5,6,7 (nvidia-smi indices; this order
    becomes CUDA_VISIBLE_DEVICES)".
  - Change `--gpus` to `default=None`, with help "GPUs required (default 1, or
    the number of --gpu-ids; 0 for CPU-only)".
- **`cmd_submit`.**
  - Parse `--gpu-ids` as comma-separated ints, with an `_error` for bad items.
  - Resolve `gpus = args.gpus if args.gpus is not None else (len(ids) if ids else 1)`.
  - Run the existing negative-`--gpus` check on the resolved value, then pass
    `gpus=` and `gpu_ids=` to `scheduler.submit`.
  - For a pending pinned job, print `waiting for GPU 4,5,6,7 on lab1`.
- **`_render_job_detail`.** On the `request` line, add `gpu_ids=4,5,6,7` when it is
  set.

### `nvidb/sched/tui.py`

- `_flow_job_token`, near the `@node_constraint` marker: append `:4,5,6,7` when
  `gpu_constraint` is set, so the ANY GPU pool shows where the job is waiting to
  go.

### Docs

- `docs/queue.md`, "Submitting and watching jobs":
  - add the example above;
  - add a short "Pinning GPUs" paragraph covering the rules in "Behaviour";
  - extend the `--pool` bullet: it cannot be combined with `--gpu-ids` either.
- `skills/nvidb-queue/SKILL.md`, "Key options":
  - add a `--gpu-ids` bullet: when to use it (the user names cards, or the work
    must not share a PCIe switch), and that it needs `--node`;
  - under "Things to avoid", say not to use it out of habit, for the same reason
    as `--node`.

## Tests (new `tests/test_sched_gpu_ids.py`, using `fake_cluster`)

- Submit rules:
  - stores ids in order;
  - `--gpus` is inferred from the ids, and a mismatch is rejected;
  - rejected without `--node`, with `--pool`, and with `--lane`;
  - rejects duplicates, negatives, unknown indices and indices outside the
    allowlist.
- Placement:
  - on an idle 4-GPU node, `--gpu-ids 3,1` gets exactly `[3, 1]`, and the
    executor's `CUDA_VISIBLE_DEVICES` is `3,1`;
  - with a foreign process filling card 3, the job stays pending even though
    cards 0 and 2 are free, then starts once card 3 frees up.
- Failures and holds:
  - unschedulable when a pinned card is smaller than `--vram`;
  - held, not failed, when an allowlist edit drops a pinned card;
  - unheld when the card is restored.
- Lanes and requeue: `lane assign` clears the pin; `requeue` keeps it.
- CLI:
  - `--json` output carries `gpu_constraint`;
  - `job ls` shows `node:ids` for a waiting pinned job;
  - `job show` shows `gpu_ids=` on the request line.
- Migration: an old database without the column opens, gains it, and old rows
  read `gpu_constraint == []`.

## Rollout

The queue host must run the new version before anyone uses the flag. A client
that forwards to an older queue host gets `unrecognized arguments: --gpu-ids`.
The column is added on first open (`ALTER TABLE ... ADD COLUMN`), and older
clients ignore it. For a local editable install, the change is live as soon as
the files are saved, including for a scheduler pass that is running, so land it
with the tests green.

## Related, separate change

Unpinned multi-GPU jobs still get their cards in spread order (for example
`2,0,1,3`). Sorting `chosen` by index in `_find_placement` would make
`cuda:0` the lowest card on every placement without changing which cards are
picked. This changes current behaviour, so it should land as its own commit.

## Possible workaround until this lands (from reading the code, not tried)

Lanes already accept a fixed group of cards. `resolve_lane` creates a lane named
`lab1:4,5,6,7` on demand, and `_lane_job_gpus` hands a lane job all of the lane's
cards in that order:

```bash
nvidb job submit --lane lab1:4,5,6,7 --gpus 4 --vram 23G -- python serve.py
```

The job then has lane semantics: it runs strictly in order with that lane's other
jobs, and a group lane can collide with the single-card lanes `lab1:4`
through `lab1:7`.
