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
| `j` / `↓`         | Move server/GPU selection down |
| `k` / `↑`         | Move server/GPU selection up  |
| `PgUp` / `PgDn`   | Change unified GPU page       |
| `Enter` / `Space` | Toggle server/GPU details      |
| `a`               | Expand all servers            |
| `c`               | Collapse all servers          |
| `q`               | Quit                          |

The default per-node view keeps each server's summary and expandable detail
table. The unified view places GPUs from every node in one table and adds
`Node` and `Hostname/IP` columns so nicknames and configured hostnames/IP
addresses remain visible. Columns adapt to the terminal width, with core
identity, utilization, model, and VRAM fields kept ahead of secondary metrics.
In the unified view, press `d` to switch between the single-line table and
Detailed cards. Detailed mode uses three lines per GPU: GPU status and identity,
core utilization metrics, then fan, PCIe RX/TX, and processes. The status badge
and utilization use cyan, green, yellow, or red to distinguish idle, active,
busy, and high utilization while preserving fixed-width alignment.
The capacity line summarizes available and busy GPUs, average utilization, and
used/total/free VRAM. Press `s` to cycle between node order, available GPUs
first, and highest utilization first. A GPU is considered available when its
utilization is below 5% and its VRAM usage is below 10%.
Unified pages are sized from the current terminal height. The title shows the
visible GPU range, and `>` marks the selected GPU.
Press `f` to cycle through all, available, busy, and error-only views. Busy
means at least 50% GPU utilization. Error-only mode hides GPU rows and lists
nodes whose latest refresh failed.
Press `Enter` or `Space` on a unified GPU to show its cached process details:
PID, user, VRAM, process type, and command. The panel uses the existing NVML
snapshot and does not issue another remote command.

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
