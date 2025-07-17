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
pip install nvidb==1.0.0
# If the specified version is unavailable in your custom repository, use pypi.org as the source:
pip install nvidb==1.0.0 -i https://pypi.org/simple
```

Here's an optimized version of the instructions for the project documentation:

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
        name         |   fan    |  util  |  mem   |   temp   |     rx     |     tx     |      power      |    memory[used/total]   
---------------------+----------+--------+--------+----------+------------+------------+-----------------+-------------------------
RTX 3090 Ti          |      0 % |    0 % |    1 % |     42 C |     50KB/s |     50KB/s | P8 34.99/450.00 | 2/24564                 

Server 0
        name         |   fan    |  util  |  mem   |   temp   |     rx     |     tx     |      power      |    memory[used/total]   
---------------------+----------+--------+--------+----------+------------+------------+-----------------+-------------------------
RTX 3090             |     62 % |  100 % |   55 % |     64 C |   14.0MB/s |    4.2MB/s | P2 329.05/350.. | 20795/24576             
RTX 3090             |     82 % |  100 % |   62 % |     74 C |   13.6MB/s |    4.0MB/s | P2 349.71/350.. | 20773/24576             
RTX 3090             |     95 % |  100 % |   29 % |     81 C |   15.3MB/s |    6.4MB/s | P2 315.79/350.. | 20813/24576             
RTX 3090             |    100 % |   99 % |   52 % |     70 C |   12.6MB/s |    5.1MB/s | P2 256.12/350.. | 21013/24576             
RTX 3090             |     75 % |    1 % |    0 % |     63 C |  331.0MB/s |    400KB/s | P2 154.99/350.. | 20973/24576             
RTX 3090             |     93 % |  100 % |   46 % |     83 C |   24.7MB/s |    4.5MB/s | P2 345.66/350.. | 21235/24576             
RTX 3090             |     50 % |    0 % |    0 % |     52 C |    350KB/s |    350KB/s | P2 110.68/350.. | 23673/24576             
RTX 3090             |     53 % |    0 % |    0 % |     56 C |    400KB/s |    350KB/s | P2 117.88/350.. | 14559/24576             

Server 1
        name         |   fan    |  util  |  mem   |   temp   |     rx     |     tx     |      power      |    memory[used/total]   
---------------------+----------+--------+--------+----------+------------+------------+-----------------+-------------------------
RTX 3090             |     41 % |    0 % |    0 % |     29 C |    450KB/s |    450KB/s | P8 30.58/350.00 | 18/24576                
RTX 3090             |     30 % |    0 % |    0 % |     32 C |    400KB/s |    500KB/s | P8 21.97/350.00 | 18/24576                
RTX 3090             |     30 % |    0 % |    0 % |     29 C |    500KB/s |    500KB/s | P8 17.17/350.00 | 18/24576                
RTX 3090             |     30 % |    0 % |    0 % |     29 C |    500KB/s |    600KB/s | P8 8.32/350.00  | 18/24576                
RTX 3090             |     79 % |   99 % |   37 % |     70 C |   31.8MB/s |    5.5MB/s | P2 355.76/350.. | 23105/24576             
RTX 3090             |     67 % |   97 % |   17 % |     64 C |   30.6MB/s |    6.8MB/s | P2 342.09/350.. | 22137/24576             
RTX 3090             |     82 % |   99 % |   37 % |     71 C |   34.2MB/s |    7.1MB/s | P2 344.00/350.. | 23887/24576             

Server 2
        name         |   fan    |  util  |  mem   |   temp   |     rx     |     tx     |      power      |    memory[used/total]   
---------------------+----------+--------+--------+----------+------------+------------+-----------------+-------------------------
RTX 2080 Ti          |     41 % |    0 % |    0 % |     55 C |    300KB/s |    300KB/s | P2 63.93/250.00 | 10811/11264             
RTX 2080 Ti          |     38 % |    0 % |    0 % |     53 C |    300KB/s |    300KB/s | P2 73.18/250.00 | 10801/11264             
RTX 2080 Ti          |     43 % |    0 % |    0 % |     56 C |    300KB/s |    300KB/s | P2 69.35/250.00 | 10801/11264             
RTX 2080 Ti          |     42 % |    0 % |    0 % |     54 C |    300KB/s |    350KB/s | P2 70.95/250.00 | 10801/11264             
RTX 2080 Ti          |     45 % |    0 % |    0 % |     55 C |    300KB/s |    300KB/s | P2 63.08/250.00 | 10805/11264             
RTX 2080 Ti          |     44 % |    0 % |    0 % |     54 C |    300KB/s |    350KB/s | P2 64.53/250.00 | 10803/11264             
RTX 2080 Ti          |     46 % |    0 % |    0 % |     54 C |    300KB/s |    300KB/s | P2 76.54/250.00 | 10803/11264             
RTX 2080 Ti          |     54 % |    0 % |    0 % |     54 C |    300KB/s |    300KB/s | P2 47.10/250.00 | 10803/11264             
RTX 2080 Ti          |     52 % |    0 % |    0 % |     53 C |    350KB/s |    300KB/s | P8 59.44/250.00 | 10625/11264             
RTX 2080 Ti          |     56 % |    0 % |    0 % |     54 C |    300KB/s |    350KB/s | P8 5.40/250.00  | 10661/11264             

Server 3
        name         |   fan    |  util  |  mem   |   temp   |     rx     |     tx     |      power      |    memory[used/total]   
---------------------+----------+--------+--------+----------+------------+------------+-----------------+-------------------------
RTX 2080 Ti          |     31 % |    0 % |    0 % |     29 C |    350KB/s |    400KB/s | P8 24.05/250.00 | 9/11264                 
RTX 2080 Ti          |     31 % |    0 % |    0 % |     29 C |    350KB/s |    400KB/s | P8 4.41/250.00  | 9/11264                 
RTX 2080 Ti          |     31 % |    0 % |    0 % |     29 C |    350KB/s |    350KB/s | P8 9.53/250.00  | 9/11264                 
RTX 2080 Ti          |     31 % |    0 % |    0 % |     28 C |    350KB/s |    350KB/s | P8 3.35/250.00  | 9/11264                 
RTX 2080 Ti          |     31 % |    0 % |    0 % |     29 C |    750KB/s |    350KB/s | P8 23.32/250.00 | 9/11264                 
RTX 2080 Ti          |     31 % |    0 % |    0 % |     30 C |    350KB/s |    350KB/s | P8 21.49/250.00 | 9/11264                 
RTX 2080 Ti          |     31 % |    0 % |    0 % |     29 C |    350KB/s |    350KB/s | P8 8.79/250.00  | 9/11264                 
RTX 2080 Ti          |     31 % |    0 % |    0 % |     28 C |    350KB/s |    350KB/s | P8 19.25/250.00 | 9/11264                 
RTX 2080 Ti          |      0 % |    0 % |    0 % |     30 C |    350KB/s |    300KB/s | P8 14.51/250.00 | 9/11264                 
```

## 3.System Requirements
The hosts should install the NVIDIA driver and be able to use `nvidia-smi` in terminal.

## 4.Tips
`nvidia-smi` query options: use `nvidia-smi --help-query-gpu` to get the query options.


## 5.Acknowledgements
Thanks to NVIDIA for providing the `nvidia-smi` tool, which is used to query GPU information.