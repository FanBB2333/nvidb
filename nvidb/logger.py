import pandas as pd
import logging
from pathlib import Path

class NVLogger:
    def __init__(self, filename="nvidia_stats.csv"):
        self.filename = filename
        # columns: node, GPU  |     name     |   fan    |   util   | mem_util |   temp   |     rx     |     tx     |       power        | memory[used/total] |      processes 
        self.data = pd.DataFrame(columns=[
            'node', 'gpu_id', 'name', 'fan_speed', 'util_gpu', 'util_mem', 'temperature', 'rx', 'tx', 'power', 'memory_used', 'memory_total', 'processes'
        ])

    def log(self, node, gpu_id, name, fan_speed, util_gpu, util_mem, temperature, rx, tx, power, memory_used, memory_total, processes):
        timestamp = pd.Timestamp.now()
        new_entry = {
            'timestamp': timestamp,
            'node': node,
            'gpu_id': gpu_id,
            'name': name,
            'fan_speed': fan_speed,
            'util_gpu': util_gpu,
            'util_mem': util_mem,
            'temperature': temperature,
            'rx': rx,
            'tx': tx,
            'power': power,
            'memory_used': memory_used,
            'memory_total': memory_total,
            'processes': processes,
        }
        self.data = self.data.append(new_entry, ignore_index=True)
        self.data.to_csv(self.filename, index=False)


if __name__ == "__main__":
    # logging the data info into specified csv file
    pass