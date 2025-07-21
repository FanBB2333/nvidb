# nvidb
A package that provides an aggregated view of the NVIDIA GPU information on several hosts.
## 1.Installation
### 1.1 Install using `pip`
You can install `nvidb` using pip. First, clone the repository:
```bash
git clone https://github.com/FanBB2333/nvidb.git
cd nvidb
pip install .
```

Or using pip directly:
```bash
pip install nvidb
# If the specified version is unavailable in your custom repository, use pypi.org as the source:
pip install nvidb -i https://pypi.org/simple
```

---

### 1.2 \[Optional] Manually Add a Configuration File

To monitor the status of remote servers, a configuration file is required. `nvidb` will look for the `config.yml` file in the `~/.nvidb/` directory.

To create the configuration file, follow these steps:

```bash
mkdir -p ~/.nvidb/
cd ~/.nvidb/
touch config.yml
```

Then, edit the `config.yml` file with the following structure:

```yaml
servers:
  - host: "example1.com"
    port: 8080
    username: "user1"
    description: "Description of the first server"
  - host: "example2.com"
    port: 9090
    username: "user2"
    password: "password2" # Optional, if password-based authentication is required
    description: "Description of the second server"
```
- The `password` field is optional, omit the field if the server can be accessed with the public key (By default, the program will read the key located in `~/.ssh`). If your key is not accessed or the filled password is incorrect, the program will prompt you to enter the password.


## 2.Usage
After installation, the command `nvidb` will be available in the terminal. Run the command to get the aggregated view of the NVIDIA GPU information on several hosts.
```bash
nvidb # for local machine only
nvidb --remote # for local and remote servers
```

The output format will be like:
```bash
[Local Machine Info]
[Remote Server0 GPU Info]
[Remote Server1 GPU Info]
...

```

One sample output for a remote server might look like:
```bash
⏰ Time: 09:41:00

Local Machine (l1ght@localhost)
Driver: 570.169 | CUDA: 12.8 | GPUs: 1
GPU  |         name         |   fan    |   util   | mem_util |   temp   |     rx     |     tx     |       power        |   memory[used/total]   |      processes      
-----+----------------------+----------+----------+----------+----------+------------+------------+--------------------+------------------------+---------------------
 0   |     RTX 3090 Ti      |   0 %    |   0 %    |   0 %    |   39 C   |  350KB/s   |  500KB/s   |  P8 32.72/450.00   |        41/24564        |       gdm(17M)      

Server 1
Driver: 575.57.08 | CUDA: 12.9 | GPUs: 8
GPU  |         name         |   fan    |   util   | mem_util |   temp   |     rx     |     tx     |       power        |   memory[used/total]   |      processes      
-----+----------------------+----------+----------+----------+----------+------------+------------+--------------------+------------------------+---------------------
 0   |       RTX 3090       |   39 %   |   6 %    |   4 %    |   50 C   |  350KB/s   |  350KB/s   |  P2 150.35/350.00  |      16147/24576       | user1(16124M) gdm(4M) 
 1   |       RTX 3090       |   57 %   |   13 %   |   11 %   |   62 C   |  350KB/s   |  400KB/s   |  P2 174.41/350.00  |      17581/24576       | user1(17558M) gdm(4M) 
 2   |       RTX 3090       |   89 %   |  100 %   |   37 %   |   80 C   |  13.9MB/s  |  4.8MB/s   |  P2 314.48/350.00  |      21415/24576       | user1(21392M) gdm(4M) 
 3   |       RTX 3090       |  100 %   |  100 %   |   32 %   |   71 C   |  27.6MB/s  |  7.7MB/s   |  P2 260.81/350.00  |      21035/24576       | user1(21012M) gdm(4M) 
 4   |       RTX 3090       |   79 %   |  100 %   |   31 %   |   75 C   |  13.1MB/s  |  7.4MB/s   |  P2 321.92/350.00  |      20975/24576       | user1(20952M) gdm(4M) 
 5   |       RTX 3090       |   90 %   |  100 %   |   28 %   |   84 C   |  35.0MB/s  |  9.8MB/s   |  P2 283.55/350.00  |      21035/24576       | user1(21012M) gdm(4M) 
 6   |       RTX 3090       |   78 %   |  100 %   |   56 %   |   75 C   |  28.8MB/s  |  8.3MB/s   |  P2 349.30/350.00  |      21135/24576       | user1(21112M) gdm(4M) 
 7   |       RTX 3090       |   84 %   |  100 %   |   82 %   |   80 C   |  13.9MB/s  |  4.0MB/s   |  P2 362.74/350.00  |      21235/24576       | user1(21212M) gdm(4M) 


Server 2
Driver: 575.57.08 | CUDA: 12.9 | GPUs: 7
GPU  |         name         |   fan    |   util   | mem_util |   temp   |     rx     |     tx     |       power        |   memory[used/total]   |      processes      
-----+----------------------+----------+----------+----------+----------+------------+------------+--------------------+------------------------+---------------------
 0   |       RTX 3090       |   41 %   |   0 %    |   0 %    |   30 C   |  400KB/s   |  500KB/s   |  P8 22.33/350.00   |        18/24576        |       gdm(4M)       
 1   |       RTX 3090       |   30 %   |   0 %    |   0 %    |   33 C   |  400KB/s   |  450KB/s   |  P8 15.31/350.00   |        18/24576        |       gdm(4M)       
 2   |       RTX 3090       |   30 %   |   0 %    |   0 %    |   29 C   |  450KB/s   |  500KB/s   |   P8 7.20/350.00   |        18/24576        |       gdm(4M)       
 3   |       RTX 3090       |   30 %   |   0 %    |   0 %    |   29 C   |  500KB/s   |  500KB/s   |   P8 4.42/350.00   |        18/24576        |       gdm(4M)       
 4   |       RTX 3090       |   30 %   |   0 %    |   0 %    |   26 C   |  800KB/s   |  950KB/s   |   P8 5.04/350.00   |        18/24576        |       gdm(4M)       
 5   |       RTX 3090       |   30 %   |   0 %    |   0 %    |   24 C   |  500KB/s   |  550KB/s   |   P8 5.13/350.00   |        18/24576        |       gdm(4M)       
 6   |       RTX 3090       |   30 %   |   0 %    |   0 %    |   25 C   |  450KB/s   |  550KB/s   |   P8 8.03/350.00   |        18/24576        |       gdm(4M)       
```

## 3.System Requirements
The hosts should install the NVIDIA driver and be able to use `nvidia-smi` in terminal.

## 4.Tips
`nvidia-smi` query options: use `nvidia-smi --help-query-gpu` to get the query options.


## 5.Acknowledgements
Thanks to NVIDIA for providing the `nvidia-smi` tool, which is used to query GPU information.