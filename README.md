# nvidb
A package that provides an aggregated view of the NVIDIA GPU information on several hosts.
## 1.Installation
### 1.1 Install from source
```bash
pip install .
```

### 1.2 Add configuration file manually
First, create a configuration file in the home directory:
```bash
mkdir -p ~/.nvidb/
cd ~/.nvidb/
touch config.yml
```

Then edit the `config.yml` file in the following format:
```yaml
servers:
  - host: "example1.com"
    port: 8080
    username: "user1"
    description: "First server description"
  - host: "example2.com"
    port: 9090
    username: "user2"
    password: "password2" # Optional, if use password to login
    description: "Second server description"
```
- The `password` field is optional, omit the field if the server can be accessed with the public key (By default, the program will read the key located in `~/.ssh`). If your key is not accessed or the filled password is incorrect, the program will prompt you to enter the password.

## 2.Usage
After installation, the command `nvidb` will be available in the terminal. Run the command to get the aggregated view of the NVIDIA GPU information on several hosts.
```bash
nvidb
```

The output will be like:
```bash
[Server0 Description]
[Info column names]
[GPU0 Information]
[GPU1 Information]
...

[Server1 Description]
[Info column names]
[GPU0 Information]
[GPU1 Information]
...


```

## 3.System Requirements
The hosts should install the NVIDIA driver and be able to use `nvidia-smi` in terminal.

## 4.Tips
`nvidia-smi` query options: use `nvidia-smi --help-query-gpu` to get the query options.