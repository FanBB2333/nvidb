# nvidb

A package that provides an aggregated view of the NVIDIA GPU information on several hosts.

## 1. Installation

### 1.1 Install using `pip`

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

### 1.2 Configuration

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

Configuration file [template](config.example.yml):

```yaml
servers:
  - hostname: "example1.com"       # Server hostname or IP address
    port: 22                       # SSH port number
    username: "user1"              # SSH username for authentication
    nickname: "Production GPU"     # Human-readable nickname for display
    auth: "auto"                   # Authentication method: auto | key | password
    identityfile: "~/.ssh/id_ed25519"  # Optional, used only when auth is auto/key
```

**Configuration Options:**
- `hostname`: Server hostname or IP address (required)
- `port`: SSH port, default is 22 (required)
- `username`: SSH username (required)
- `nickname`: Human-readable server nickname (optional)
- `auth`: Authentication method - `auto`, `key`, or `password` (optional, default: `auto`)
- `identityfile`: SSH private key path (optional, only effective when `auth` is `auto` or `key`)
- `password`: SSH password (optional, will prompt if needed)

> **Warning**: Storing passwords in plaintext in the configuration file is **NOT RECOMMENDED** for security reasons. Consider using SSH key-based authentication (`auth: key`) instead.

The same file also holds a `view` section that nvidb maintains itself: every TUI
layout key (`v`, `d`, `s`, `f`, `g`, `u`, `t`, `Enter`) writes the new state back
so the next run opens with the same view. See
[2.6 Interactive TUI Navigation](#26-interactive-tui-navigation) for what each
setting does.

#### Environment Variables

You can customize the working directory by setting `NVIDB_HOME`:

```bash
export NVIDB_HOME=/path/to/custom/nvidb
```

Default working directory is `~/.nvidb/`.

---

## 2. Usage

### 2.1 Basic Commands

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

### 2.2 Server Management

```bash
nvidb add              # Interactively add a new server
nvidb import [path]    # Import servers from SSH config (default: ~/.ssh/config)
nvidb info             # Show configuration info and server list
```

### 2.3 GPU Logging

Continuously log GPU statistics to an SQLite database:

```bash
nvidb log                          # Log local GPU with default settings
nvidb log --remote                 # Log local and remote GPUs
nvidb log --interval 10            # Set logging interval to 10 seconds
nvidb log --db-path /path/to/db    # Specify custom database path
```

Press `Ctrl+C` to stop logging and save data.

### 2.4 Web Dashboard

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

### 2.5 Cleanup

Remove server configurations or delete log data:

```bash
nvidb clean              # Interactive cleanup menu
nvidb clean all          # Delete all data (requires double confirmation)
```

### 2.6 Interactive TUI Navigation

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
| `Tab` / `h` / `l` / `←` / `→` | Switch between GPU and process panes |
| `j` / `↓`         | Move the active-pane selection down |
| `k` / `↑`         | Move the active-pane selection up |
| `PgUp` / `PgDn`   | Move the active-pane selection by a page |
| `[` / `]`         | Page through a long wrapped command |
| `i` / `T` / `K`   | Arm SIGINT / SIGTERM / SIGKILL for the selected process |
| `Esc`             | Cancel an armed process signal |
| Mouse             | Select GPUs/processes, click actions, or scroll either pane |
| `Enter` / `Space` | Toggle server/process details or confirm a signal |
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
Detailed mode always places the selected GPU's process pane directly below the
cards; there is no separate drill-down screen. In single-line mode, `Enter` or
`Space` toggles the same pane. Its table shows PID, user, process VRAM, percentage
of total GPU VRAM, CPU%, host MEM%, RSS, elapsed time, state, and command. Narrow
terminals discard secondary columns first. The selected process gets an
htop-style low-contrast grey background, and its block below the table spells
out both percentage of the whole GPU and percentage of currently used GPU
memory, threads, state, elapsed time, and the complete wrapped command line.
When a wrapped command would push the action bar off-screen, it automatically
uses height-aware pages (at most five command lines on terminals shorter than
28 rows). Use `[`/`]`, the clickable command buttons, or the wheel over the
command to see every line.

Use `Tab`, `←`/`→`, or `h`/`l` to switch pane focus, then `j`/`k` or the arrow
keys to move the highlighted row. The active pane uses solid borders; the
inactive pane uses dashed borders. Only the active pane displays a selected-row
background, so process rows are not highlighted while GPU/node selection has
focus. Mouse reporting is on by default: click any GPU card or process row to
select it, use the wheel over either pane to scroll that pane, and click the
action buttons directly. Most terminals need `Shift` (or `Option`) held down
for their own drag-to-select while mouse reporting is active; set
`mouse: false` under `view` to disable TUI mouse handling.

The process action bar exposes `SIGINT`, `SIGTERM`, and `SIGKILL` as `i`, `T`,
and `K`. Every signal requires a second identical click/key press or `Enter`
within five seconds; `Esc` cancels it. Signals run on the node that owns the
selected GPU and report permission or command failures in the pane. If
GPU/node selection has focus, the first signal key only transfers focus to the
process pane; press it again to arm the action.

Press `t` (or click History) to show utilization, VRAM, and temperature for the
selected GPU plus CPU, GPU VRAM share, host memory, and RSS for the selected
process. Both histories retain the latest 60 successful refresh samples in
memory and do not add remote requests. On terminals shorter than 36 lines, the
history rows temporarily replace the selected process's command block so the
action buttons remain visible; toggle History off to restore the command.

Machines without NVIDIA GPUs (a macOS laptop, a CPU-only host) are not expanded
by default in the per-node view and are collapsed into a single "hidden" line in
the unified node status. Press `u` to show them.
Layout toggles are written back to the `view` section of `~/.nvidb/config.yml`,
so the next `nvidb` run starts with the same layout.

### 2.7 GPU Monitor Decorator

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

## 4. System Requirements

- NVIDIA driver with NVML (`libnvidia-ml.so.1`)
- Python 3.8+
- Python 3.8+ and SSH access on remote servers
- `nvidia-smi` is optional and used only as an NVML failure fallback

## 5. Tips

- The live header shows `Source: nvml` during normal collection and
  `Source: nvidia-smi` if the compatibility fallback was needed
- Database files are stored in `~/.nvidb/gpu_log.db` by default
- Configuration and logs are stored in `~/.nvidb/` directory

## 6. Show me the screenshots

- Monitor local info with `nvidb`:

![nvidb local](resources/nvidb_local.png)

- Monitor remote info with `nvidb --remote`:

![nvidb remote](resources/nvidb_remote.png)

- Monitor on web panel with `nvidb web`:

Local info:

![nvidb web local](resources/nvidb_web_local.png)

Remote info:

![nvidb web remote](resources/nvidb_web_remote.png)

---

## 7. Acknowledgements

- Thanks to NVIDIA for providing NVML and
  [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py), used for direct GPU
  telemetry collection.
- Thanks to [nvitop](https://github.com/XuehaiPan/nvitop) for demonstrating
  efficient direct NVML polling and metric caching patterns.
- Thanks to NVIDIA for providing `nvidia-smi`, retained as a compatibility
  fallback.
- Thanks to [Paramiko](https://github.com/paramiko/paramiko) for powering SSH connections for remote monitoring.
- Thanks to [PyYAML](https://github.com/yaml/pyyaml) for YAML-based configuration loading and saving.
- Thanks to [pandas](https://github.com/pandas-dev/pandas) for parsing and processing GPU stats and log data.
- Thanks to [blessed](https://github.com/jquast/blessed) for building the interactive terminal UI.
- Thanks to [termcolor](https://github.com/termcolor/termcolor) for colored terminal output.
- Thanks to [Streamlit](https://github.com/streamlit/streamlit) for providing the web dashboard framework.
