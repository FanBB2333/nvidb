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

Configuration file structure:

```yaml
servers:
  - host: "example1.com"
    port: 22
    username: "user1"
    description: "Description of the first server"
    auth: "auto"  # auto | key | password
    
  - host: "example2.com"
    port: 22
    username: "user2"
    password: "password2"  # Optional, prompted if not set
    description: "Description of the second server"
    auth: "auto"
```

**Configuration Options:**
- `host`: Server hostname or IP address (required)
- `port`: SSH port, default is 22 (required)
- `username`: SSH username (required)
- `description`: Human-readable server description (optional)
- `auth`: Authentication method - `auto`, `key`, or `password` (optional, default: `auto`)
- `password`: SSH password (optional, will prompt if needed)

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
nvidb                  # Monitor local GPU only
nvidb --remote         # Monitor local and remote servers
nvidb --version        # Show version
```

### 2.2 Server Management

```bash
nvidb add              # Interactively add a new server
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

### 2.4 Interactive TUI Navigation

When viewing GPU stats, use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| `j` / `↓` | Move selection down |
| `k` / `↑` | Move selection up |
| `Enter` / `Space` | Toggle expand/collapse server |
| `a` | Expand all servers |
| `c` | Collapse all servers |
| `q` | Quit |

---

## 3. Sample Output

```bash
Time: 09:41:00 | Servers: 2 | [j/k] Navigate [Enter] Toggle [a] Expand All [c] Collapse All [q] Quit
--------------------------------------------------------------------------------
* v [1] Local Machine (l1ght@localhost)  1 GPUs | 1 idle | 0% avg | 0GB/24GB

Local Machine (l1ght@localhost)
Driver: 570.169 | CUDA: 12.8 | GPUs: 1
GPU  |    name     |   fan   |  util   | mem_util |  temp   |    rx    |    tx    |      power       | memory[used/total] |   processes   
-----+-------------+---------+---------+----------+---------+----------+----------+------------------+--------------------+---------------
 0   | RTX 3090 Ti |   0 %   |   0 %   |   0 %    |  39 C   | 350KB/s  | 500KB/s  | P8 32.72/450.00  |      41/24564      |    gdm(17M)   

  > [2] Server 1  8 GPUs | 0 idle | 78% avg | 156GB/192GB
```

---

## 4. System Requirements

- NVIDIA driver installed with `nvidia-smi` available in terminal
- Python 3.8+
- SSH access to remote servers (for remote monitoring)

## 5. Tips

- Use `nvidia-smi --help-query-gpu` to see available query options
- Database files are stored in `~/.nvidb/gpu_log.db` by default
- Configuration and logs are stored in `~/.nvidb/` directory

## 6. Acknowledgements

Thanks to NVIDIA for providing the `nvidia-smi` tool, which is used to query GPU information.