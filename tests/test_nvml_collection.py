import pandas as pd

from nvidb.connection import BaseClient
from nvidb.nvml import REMOTE_NVML_AGENT_SCRIPT, make_nvml_agent_command


NVIDIA_SMI_XML = """<?xml version="1.0" ?>
<nvidia_smi_log>
  <driver_version>550.54.15</driver_version>
  <cuda_version>12.4</cuda_version>
  <attached_gpus>1</attached_gpus>
  <gpu>
    <product_name>NVIDIA Test GPU</product_name>
    <product_architecture>Ampere</product_architecture>
    <performance_state>P8</performance_state>
    <fan_speed>30 %</fan_speed>
    <pci>
      <pci_gpu_link_info>
        <pcie_gen>
          <max_link_gen>4</max_link_gen>
          <current_link_gen>1</current_link_gen>
          <max_device_link_gen>5</max_device_link_gen>
          <max_host_link_gen>4</max_host_link_gen>
        </pcie_gen>
        <link_widths>
          <max_link_width>16x</max_link_width>
          <current_link_width>8x</current_link_width>
        </link_widths>
      </pci_gpu_link_info>
      <tx_util>1 KB/s</tx_util>
      <rx_util>2 KB/s</rx_util>
    </pci>
    <fb_memory_usage>
      <total>8192 MiB</total>
      <used>1024 MiB</used>
      <free>7168 MiB</free>
    </fb_memory_usage>
    <utilization>
      <gpu_util>10 %</gpu_util>
      <memory_util>20 %</memory_util>
    </utilization>
    <temperature>
      <gpu_temp>40 C</gpu_temp>
    </temperature>
    <gpu_power_readings>
      <power_state>N/A</power_state>
      <power_draw>50.00 W</power_draw>
      <current_power_limit>200.00 W</current_power_limit>
    </gpu_power_readings>
    <processes />
  </gpu>
</nvidia_smi_log>
"""


class FakeClient(BaseClient):
    def __init__(self, payload, *, nvidia_smi_xml="", ps_output=""):
        super().__init__()
        self.connected = True
        self.payload = payload
        self.nvidia_smi_xml = nvidia_smi_xml
        self.ps_output = ps_output
        self.commands = []

    def connect(self):
        self.connected = True
        return True

    def query_nvml_snapshot(self):
        return self.payload

    def execute_command(self, command):
        self.commands.append(command)
        if command == "nvidia-smi -q -x":
            return self.nvidia_smi_xml
        if command.startswith("ps "):
            return self.ps_output
        return '{"ok":false,"error":"dcgm unavailable"}'


def test_full_gpu_info_uses_nvml_payload_and_native_processes():
    payload = {
        "ok": True,
        "backend": "ctypes",
        "driver_version": "570.1",
        "cuda_version": "12.8",
        "gpus": [
            {
                "gpu_index": 3,
                "name": "NVIDIA Example GPU",
                "architecture": "Blackwell",
                "memory_total_bytes": 8 * 1024 * 1024,
                "memory_used_bytes": 3 * 1024 * 1024,
                "memory_free_bytes": 5 * 1024 * 1024,
                "gpu_util_percent": 42,
                "memory_util_percent": 7,
                "pcie_tx_kib_per_s": 100,
                "pcie_rx_kib_per_s": 200,
                "pcie_link_gen_current": 4,
                "pcie_link_width_current": 4,
                "pcie_link_gen_max": 4,
                "pcie_link_width_max": 16,
                "fan_percent": 30,
                "temperature_c": 51,
                "performance_state": 2,
                "power_usage_mw": 125500,
                "power_limit_mw": 300000,
                "processes": [
                    {
                        "pid": 123,
                        "type": "C",
                        "process_name": "python",
                        "username": "alice",
                        "used_gpu_memory_bytes": 2 * 1024 * 1024,
                        "gpu_instance_id": 0,
                        "compute_instance_id": 1,
                    }
                ],
            }
        ],
    }
    client = FakeClient(payload)

    stats, system_info = client.get_full_gpu_info()

    assert isinstance(stats, pd.DataFrame)
    assert system_info["data_source"] == "nvml"
    assert system_info["data_source_detail"] == "libnvidia-ml.so.1"
    assert system_info["attached_gpus"] == "1"
    assert stats.iloc[0]["gpu_index"] == 3
    assert stats.iloc[0]["used"] == "3 MiB"
    assert stats.iloc[0]["gpu_util"] == "42 %"
    assert stats.iloc[0]["power_draw"] == "125.50 W"
    # The link the card runs on now, and the widest one it could ever use.
    assert stats.iloc[0]["pcie_link_gen_current"] == 4
    assert stats.iloc[0]["pcie_link_width_current"] == 4
    assert stats.iloc[0]["pcie_link_width_max"] == 16

    processes, user_summary = client.get_process_summary(stats)
    assert processes == [
        {
            "gpu_instance_id": 0,
            "compute_instance_id": 1,
            "pid": 123,
            "type": "C",
            "process_name": "python",
            "used_memory": "2 MiB",
            "username": "alice",
            "gpu_index": 3,
            "command": "python",
            "cpu_percent": None,
            "mem_percent": None,
            "rss_kb": None,
            "elapsed": None,
            "state": None,
            "threads": None,
        }
    ]
    assert user_summary == {"alice": 2}
    assert not any(command.startswith("ps ") for command in client.commands)
    assert "nvidia-smi -q -x" not in client.commands


def test_detailed_process_summary_adds_htop_style_fields():
    payload = {
        "ok": True,
        "backend": "ctypes",
        "driver_version": "570.1",
        "cuda_version": "12.8",
        "gpus": [
            {
                "gpu_index": 0,
                "name": "NVIDIA RTX 6000 Ada",
                "memory_total_bytes": 49140 * 1024 * 1024,
                "memory_used_bytes": 2 * 1024 * 1024,
                "processes": [
                    {
                        "pid": 4242,
                        "type": "C",
                        "process_name": "python",
                        "used_gpu_memory_bytes": 2 * 1024 * 1024,
                        "username": None,
                    }
                ],
            }
        ],
    }
    client = FakeClient(payload)
    client.ps_output = (
        "4242 alice 91.5 12.3 8388608 03:21:07 Rl 24 "
        "/usr/bin/python3 train.py --config configs/big.yaml\n"
    )

    stats, _system_info = client.get_full_gpu_info()
    processes, user_summary = client.get_process_summary(stats, detailed=True)

    assert user_summary == {"alice": 2}
    process = processes[0]
    assert process["username"] == "alice"
    assert process["cpu_percent"] == 91.5
    assert process["mem_percent"] == 12.3
    assert process["rss_kb"] == 8388608
    assert process["elapsed"] == "03:21:07"
    assert process["state"] == "Rl"
    assert process["threads"] == 24
    assert process["command"] == (
        "/usr/bin/python3 train.py --config configs/big.yaml"
    )
    ps_commands = [command for command in client.commands if command.startswith("ps ")]
    assert len(ps_commands) == 1
    assert ps_commands[0].startswith(
        "ps -o pid=,user=,pcpu=,pmem=,rss=,etime=,stat=,nlwp=,args= -p"
    )


def test_full_gpu_info_falls_back_to_nvidia_smi_when_nvml_fails():
    client = FakeClient(
        {
            "ok": False,
            "backend": "ctypes",
            "error": "NVMLError: driver unavailable",
        },
        nvidia_smi_xml=NVIDIA_SMI_XML,
    )

    stats, system_info = client.get_full_gpu_info()

    assert len(stats) == 1
    assert system_info["data_source"] == "nvidia-smi"
    assert system_info["fallback_reason"] == "NVMLError: driver unavailable"
    assert system_info["driver_version"] == "550.54.15"
    # `nvidia-smi -q -x` nests the link under <pci> and suffixes widths with x.
    assert stats.iloc[0]["pcie_link_gen_current"] == "1"
    assert stats.iloc[0]["pcie_link_width_current"] == "8x"
    assert stats.iloc[0]["pcie_link_gen_max"] == "4"
    assert stats.iloc[0]["pcie_link_width_max"] == "16x"
    assert client.commands.count("nvidia-smi -q -x") == 1


def test_empty_nvml_gpu_list_does_not_trigger_fallback():
    client = FakeClient(
        {
            "ok": True,
            "backend": "ctypes",
            "driver_version": "570.1",
            "cuda_version": "12.8",
            "gpus": [],
        },
        nvidia_smi_xml=NVIDIA_SMI_XML,
    )

    stats, system_info = client.get_full_gpu_info()

    assert stats.empty
    assert system_info["data_source"] == "nvml"
    assert system_info["attached_gpus"] == "0"
    assert "nvidia-smi -q -x" not in client.commands


def test_remote_agent_uses_nvml_without_nvidia_smi():
    command = make_nvml_agent_command(once=True)

    assert "nvidia-smi" not in REMOTE_NVML_AGENT_SCRIPT
    assert command.startswith("python3 -S -u -c ")
    assert command.endswith(" once")
