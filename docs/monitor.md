# nvidb monitor, logging and web dashboard

## Installation

### Install using `pip`

You can install `nvidb` using pip. First, clone the repository:

```bash
git clone https://github.com/FanBB2333/nvidb.git
cd nvidb
pip install .
```

Or install directly from PyPI:

```bash
pip install nvidb
# If the specified version is unavailable in your custom repository, use pypi.org as the source:
pip install nvidb -i https://pypi.org/simple
```

---

### Configuration

#### Option A: Interactive Setup (Recommended)

Use the interactive command to add servers:

```bash
nvidb add
```

This will guide you through adding a new server with prompts for host, port, username, authentication method, etc.

#### Option B: Manual Configuration

To manually configure remote servers, create or edit the configuration file at `~/.nvidb/config.yml`:

```bash
mkdir -p ~/.nvidb/
cp config.example.yml ~/.nvidb/config.yml
# Edit the file with your server details
```

Configuration file [template](../config.example.yml):

```yaml
servers:
  - hostname: "example1.com"       # Server hostname or IP address
    port: 22                       # SSH port number
    username: "user1"              # SSH username for authentication
    nickname: "Production GPU"     # Human-readable nickname for display
    auth: "auto"                   # Authentication method: auto | key | password
    identityfile: "~/.ssh/id_ed25519"  # Optional, used only when auth is auto/key
    proxyjump: "login.example.com" # Optional OpenSSH ProxyJump host or alias
    gpu_ids: [0, 1]                 # Optional: the only GPUs nvidb uses here
```

**Configuration Options:**
- `hostname`: Server hostname or IP address (required)
- `port`: SSH port, default is 22 (required)
- `username`: SSH username (required)
- `nickname`: Human-readable server nickname (optional)
- `auth`: Authentication method - `auto`, `key`, or `password` (optional, default: `auto`)
- `identityfile`: SSH private key path (optional, only effective when `auth` is `auto` or `key`)
- `password`: SSH password (optional, will prompt if needed)
- `proxyjump`: OpenSSH jump host, alias, or comma-separated chain (optional; for
  example `login` or `login-a,alice@login-b:2222`)
- `gpu_ids`: the GPU indices nvidb uses on this server (optional). The monitor,
  `nvidb log` and `nvidb queue status`/`nodes` show only these cards (under
  their own indices) and new queue placements use only them; running jobs
  outside the list continue normally, and their card stays listed in the queue
  views until they end. An empty list disables GPU placement on the node while
  still permitting CPU-only queue jobs.

The server-level `gpu_ids` list is a placement policy. The job option `--gpus N`
still means "request N cards"; it does not name individual indices.

> **Warning**: Storing passwords in plaintext in the configuration file is **NOT RECOMMENDED** for security reasons. Consider using SSH key-based authentication (`auth: key`) instead.

`proxyjump` uses the local OpenSSH client, so aliases and credentials configured
in `~/.ssh/config` are reused. Connect to the jump host once with `ssh login` to
accept its host key and verify key/agent authentication before running the
non-interactive job queue. `nvidb import` copies `ProxyJump` from OpenSSH config:

```sshconfig
Host login
    HostName login.example.com
    User jump-user

Host training-a100
    HostName 10.0.0.42
    User gpu-user
    ProxyJump login
```

```bash
nvidb import
nvidb --remote
```

The same file also holds a `view` section that nvidb maintains itself. Persistent
layout keys (`v`, `d`, `s`, `f`, `g`, `u`, `t`, and `p` in single-line mode)
write the new state back so the next run opens with the same view. Pane focus and
the temporary Detailed-mode process collapse are not persisted. See
[Interactive TUI Navigation](#interactive-tui-navigation) for what each
setting does.

#### Environment Variables

You can customize the working directory by setting `NVIDB_HOME`:

```bash
export NVIDB_HOME=/path/to/custom/nvidb
```

Default working directory is `~/.nvidb/`.

---

## Usage

### Basic Commands

```bash
nvidb                  # Monitor local GPU only (interactive TUI)
nvidb --remote         # Monitor local and remote servers
nvidb --once           # Print GPU stats once and exit
nvidb --once --remote  # Print all servers once and exit
nvidb --version        # Show version
```

> **Tip**: Set `remote: true` under the `basic` section of `~/.nvidb/config.yml` to make plain `nvidb` include remote servers by default (same for `nvidb log`). Pass `--no-remote` for a one-off local-only run.

GPU status is collected directly from NVML. Local collection uses `nvidia-ml-py`;
remote collection keeps a standard-library Python agent open over SSH and calls
`libnvidia-ml.so.1` directly, so no Python package needs to be installed on the
remote host. `nvidia-smi -q -x` is retained only as a compatibility fallback when
NVML cannot be initialized.

### Server Management

```bash
nvidb add              # Interactively add a new server
nvidb import [path]    # Import servers from SSH config (default: ~/.ssh/config)
nvidb info             # Show configuration info and server list
```

### GPU Logging

Continuously log GPU statistics to an SQLite database:

```bash
nvidb log                          # Log local GPU with default settings
nvidb log --remote                 # Log local and remote GPUs
nvidb log --interval 10            # Set logging interval to 10 seconds
nvidb log --db-path /path/to/db    # Specify custom database path
```

Rows are written to the database every interval. Press `Ctrl+C` to stop logging.

### Web Dashboard

Open a [Dash](https://github.com/plotly/dash)-based interactive web dashboard to view live GPU info and browse log sessions:

```bash
pip install dash
nvidb web                 # Web dashboard (Live + Logs)
nvidb web --db-path /path/to/db
nvidb web --port 8502
```

After the server starts (http://localhost:8501 by default):
- **Live**: per-server GPU tables plus rolling utilization / VRAM charts; toggle `include remote`, pick the refresh interval, or pause auto-refresh. (`basic.remote: true` or `nvidb --remote web` enables remote by default.)
- **Logs**: pick a session in the left table, then filter by node / metric / time range. Charts support zoom, pan and legend isolation; **click any chart point to inspect that snapshot**. The raw table supports filtering, sorting and CSV export.

`nvidb log web` is deprecated; use `nvidb web` instead.

### Cleanup

Remove server configurations or delete log data:

```bash
nvidb clean              # Interactive cleanup menu
nvidb clean all          # Delete all data (requires double confirmation)
```

### Interactive TUI Navigation

The monitor opens before connecting to remote nodes. Each node connects and
refreshes independently, so an unreachable or slow SSH server does not delay
healthy nodes. GPU rows appear first; CPU/memory and process details follow.
The live monitor does not query DCGM or show its advanced tables by default.
Use `nvidb --dcgm` (or `nvidb --remote --dcgm`) to explicitly enable DCGM
profiling; its metrics load after the base GPU rows. `--debug` does not enable
DCGM, and `d` only changes the GPU row layout.

By default, refresh progress is represented only by the
top **Updated** timestamp; per-node telemetry, loading/collecting phases, sample
ages and timing banners are hidden to keep the layout stable. Real connection
and collection errors remain visible.

Use `nvidb --debug` (or `nvidb --remote --debug`) to show collection phases,
stale-sample banners and per-node SSH/GPU/details timings (`v` switches to
per-node panels). Debug display is off on every normal invocation and is not
saved in the view settings. The `d` key still switches unified GPU row detail;
it does not toggle debugging.

The normal refresh interval is one second after each node finishes its own
sample; failed connections use increasing retry delays, including the first
failure. Password/passphrase prompts are still supported and temporarily pause
screen updates while other nodes continue collecting. `q` exits without waiting
for pending SSH calls (use Ctrl+C to exit a credential prompt).
ProxyJump helpers run in batch mode in the live monitor; jump hosts must already
have non-interactive authentication configured (for example, an unlocked key in
`ssh-agent`). Destination password/passphrase prompts remain available.

`--once`, logging and web consumers retain their complete-snapshot behavior.

When viewing GPU stats, use these keyboard shortcuts:

| Key               | Action                        |
| ----------------- | ----------------------------- |
| `v`               | Switch unified/per-node view  |
| `d`               | Toggle unified row detail     |
| `s`               | Cycle unified GPU sorting     |
| `f`               | Cycle unified GPU filters     |
| `g`               | Toggle unified per-node grouping |
| `u`               | Show/hide nodes without GPU support |
| `t`               | Toggle GPU and selected-process history |
| `Enter` / `Space` / `l` / `→` | Show and enter the selected GPU's process pane |
| `h` / `←`         | Return focus to the GPU/node pane |
| `Tab`             | Switch between visible GPU and process panes |
| `p`               | Show/hide the selected GPU's process pane |
| `j` / `↓`         | Move the active-pane selection down |
| `k` / `↑`         | Move the active-pane selection up |
| `PgUp` / `PgDn`   | Move the active-pane selection by a page |
| `[` / `]`         | Page through a long wrapped command |
| `/`               | Edit a live process filter |
| `o` / `F6`        | Cycle process sorting |
| `O`               | Reverse the current process sort |
| `+` / `-`         | Show more/fewer process rows |
| `i` / `T` / `K`   | Arm SIGINT / SIGTERM / SIGKILL for the selected process |
| `Esc`             | Cancel a signal or clear the process filter |
| `?`               | Open context-sensitive help |
| `m` / More        | Open actions that do not fit in the toolbar |
| Mouse             | Select rows, sort headers, click actions, or scroll either pane |
| `Enter` (signal armed) | Confirm the pending process signal |
| `Enter` / `Space` (per-node view) | Toggle the selected server details |
| `a`               | Expand all servers            |
| `c`               | Collapse all servers          |
| `q`               | Quit                          |

The default per-node view keeps each server's summary and expandable detail
table. The unified view places GPUs from every node in one table. By default,
single-line rows are grouped into per-node blocks: a band names the node in
bold cyan followed by its hostname/IP, GPU count, free GPUs, average
utilization, and VRAM, with a dim rule filling the rest of the line. The rows
below it drop the redundant `Node` / `Hostname/IP` columns so more width goes
to the GPU metrics. Press `g` to turn grouping off (or sort by anything other
than node order) and the flat table with `Node` and `Hostname/IP` columns comes
back.
Detailed cards include node identity themselves and omit the redundant band.
Columns adapt to the terminal width, with core identity, utilization, model, and
VRAM fields kept ahead of secondary metrics.
In the unified view, press `d` to switch between the single-line table and
Detailed cards. Each card is divided into four labelled rows: `GPU` identity,
`LOAD`, `MEM/TEMP`, and `I/O`. Utilization and VRAM gain block bars on terminals
at least 100 columns wide. The palette stays deliberately quiet: grey for
structure and secondary values, cyan for the active focus and normal metrics,
and yellow/red for warnings. Green is reserved for a running/healthy state.
The node name comes before the GPU index so the machine is the first thing you
read.
PCIe throughput is scored against the link the card is actually running on. The
`PCIe` column names that link (`4.0x16`, `4.0x4`), and the `RX`/`TX` cells fill
up with their share of it — green under 40%, yellow past 40%, red past 70%, at
which point the interconnect is shaping transfers. Detailed cards spell the same
thing out as `Link 4.0x4 7.34GB/s of 4.0x16` plus a percentage per direction,
where the trailing `of ...` is the widest link the card itself supports. A card
wired through a narrower slot or riser reports the narrower link, and a card
sitting idle reports the reduced generation it drops to for power saving; both
are the real ceiling for the traffic measured alongside them. SXM boards are
scored on their host PCIe link, which is what these counters measure — NVLink
traffic between GPUs is not included.
The capacity line summarizes available and busy GPUs, average utilization, and
used/total/free VRAM. Press `s` to cycle between node order, available GPUs
first, and highest utilization first. A GPU is considered available when its
utilization is below 5% and its VRAM usage is below 10%.
Unified pages are sized from the current terminal height. The title shows the
current focus and visible GPU range, and `›` marks the active selection.
Press `f` to cycle through all, available, busy, and error-only views. Busy
means at least 50% GPU utilization. Error-only mode hides GPU rows and lists
nodes whose latest refresh failed. Because the filter is restored from the
config on the next run, a warning line above the table spells out how many GPUs
it is hiding.
Node failures also appear in a fixed top-of-body alert, ahead of GPU and
process details. Click that alert to open the error-only list; `j`/`k`,
PgUp/PgDn, and the wheel can reach every failed node on a short terminal.
Detailed mode places the selected GPU's process pane directly below the cards;
there is no separate drill-down screen. `Enter`, `Space`, `→`, or `l` shows the
pane when necessary and moves focus into its task list. Repeating one of these
keys keeps process focus instead of collapsing the pane. Press `p` to hide or
show it explicitly; `←` or `h` returns to GPU/node selection without hiding it.
The same navigation applies in single-line mode. The process table shows PID,
user, process VRAM, percentage of total GPU VRAM, CPU%, host MEM%, RSS, elapsed
time, state, and command. Narrow terminals discard secondary columns first. The
selected process gets an htop-style low-contrast grey background, and its block
below the table spells out both percentage of the whole GPU and percentage of
currently used GPU memory, threads, state, elapsed time, and the complete
wrapped command line.
Press `/` to filter the process list as you type. The query matches PID, user,
type, state, process name, and the full command. `Enter` keeps the current
filter and leaves edit mode; `Esc` clears it. Press `o` or `F6` to cycle through
VRAM, CPU, host memory, RSS, elapsed time, PID, and command sorting. Uppercase
`O` reverses the current order. The visible table headers are also clickable:
click a field once to sort by it and again to reverse it.
Use `+` and `-`, or the row controls in the action bar, to change how many
processes stay visible. The request is capped at 12 rows and reduced
automatically when the selected-process details need the space.
When a wrapped command would push the action bar off-screen, it automatically
uses height-aware pages (at most five command lines on terminals shorter than
28 rows). Use `[`/`]`, the clickable command buttons, or the wheel over the
command to see every line.

Use `Enter`, `→`, or `l` to enter process focus, `←` or `h` to return to
GPU/node focus, and `Tab` to switch between visible panes. Then use `j`/`k` or
the arrow keys to move the highlighted row. The active pane uses solid borders;
the inactive pane uses dashed borders. Only the active pane displays a
selected-row background, so process rows are not highlighted while GPU/node
selection has focus. Mouse reporting is on by default. The top toolbar has
clickable **GPUs** and **Nodes** tabs, **Filter** and **Sort** menus, and view
controls. Hover over buttons, clickable rows or menu choices to highlight
them; moving away restores their normal appearance. Hover does not change the
selection or activate an action. Choose a filter or sort directly from its menu; the existing `f` and
`s` keys still cycle them. Click the GPU table's **GPU/Node** or **Util** header
to select node order or highest utilization first. Controls that do not fit
move into **More**, including history, grouping, theme, paging, help and quit.
Menus support clicking, the wheel, arrow keys, Enter and Esc; clicking outside
closes a menu without activating the view underneath.

Click a GPU row or card to select it and reveal its processes; click the
selected GPU again to enter process focus. Clicking a closed node opens its
details immediately. The process pane offers **Find**, **Clear** and **Hide**
buttons (`/`, `x`, `p` in short windows), and **Apply** while editing a search.
Process sorting is also available as a menu, including columns hidden in
narrow windows; choosing the current field reverses it. Use the wheel over
either pane to scroll that pane. Most terminals need `Shift` (or
`Option`) held down for their own drag-to-select while mouse reporting is
active; set `mouse: false` under `view` to disable TUI mouse handling.

The process action bar exposes `SIGINT`, `SIGTERM`, and `SIGKILL` as `i`, `T`,
and `K`. Every signal requires a second identical click/key press or `Enter`
within five seconds; explicit **Confirm** and **Cancel** buttons are also
available, and `Esc` cancels it. Signals run on the node that owns the
selected GPU and report permission or command failures in the pane. If
GPU/node selection has focus, the first signal key only transfers focus to the
process pane; press it again to arm the action.

Press `?` for a help panel tailored to the current per-node, GPU, or process
focus. Click anywhere inside the panel, or press `?`, `Esc`, or `q`, to close
it. Closing help with `q` does not quit the monitor.

Press `t` (or click History) to show utilization, VRAM, and temperature for the
selected GPU plus CPU, GPU VRAM share, host memory, and RSS for the selected
process. Both histories retain the latest 60 successful refresh samples in
memory and do not add remote requests. On terminals shorter than 36 lines, the
history rows temporarily replace the selected process's command block so the
action buttons remain visible; toggle History off to restore the command.

The local machine is not one more node to manage: it reports through a
one-line strip at the very top of the page (in both views) with the system
description, CPU utilization, load average, and memory usage. It only keeps a
node section of its own while it has NVIDIA GPUs to show, or when it is the
only machine in the pool.

Remote machines without NVIDIA GPUs (a CPU-only host) are not expanded by
default in the per-node view and are collapsed into a single "hidden" line in
the unified node status. Press `u` to show them.
Layout toggles are written back to the `view` section of `~/.nvidb/config.yml`,
so the next `nvidb` run starts with the same layout. Process filters, sorting,
row counts, and help visibility are session-only.

### GPU Monitor Decorator

Use the `@nvidb.monitor` decorator to track GPU usage during function execution:

```python
import nvidb

@nvidb.monitor
def train_model():
    # Your training code here
    pass

# With custom options
@nvidb.monitor(sample_interval=0.05, gpu_indices=[0, 1])
def multi_gpu_training(epochs: int = 100):
    pass

# Async function support
@nvidb.monitor
async def async_training():
    pass
```

After function execution, it outputs:
```
======================================================================
[nvidb.monitor] Function completed: train_model
  Signature: train_model()
  Location: /path/to/file.py:14
----------------------------------------------------------------------
  Duration: 125.3s
----------------------------------------------------------------------
  GPU 0: NVIDIA GeForce RTX 3090 Ti
    Memory:
      Peak:    8192.00 MiB / 24.00 GiB
      Delta:   +6144.00 MiB
    Utilization:
      Avg:     85.0%
    Temperature:
      Peak:    72C
    Power:
      Peak:    320.5W
======================================================================
```

**Decorator Options:**
- `sample_interval`: Sampling interval in seconds (default: 0.1)
- `gpu_indices`: List of GPU indices to monitor (default: all GPUs)
- `enabled`: Enable/disable monitoring (default: True)

---

## System Requirements

- NVIDIA driver with NVML (`libnvidia-ml.so.1`)
- Python 3.9+
- Python 3.9+ and SSH access on remote servers
- Local OpenSSH client when `proxyjump` is configured
- `nvidia-smi` is optional and used only as an NVML failure fallback

## Tips

- The live header shows `Source: nvml` during normal collection and
  `Source: nvidia-smi` if the compatibility fallback was needed
- Database files are stored in `~/.nvidb/gpu_log.db` by default
- Configuration and logs are stored in `~/.nvidb/` directory
