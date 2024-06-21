from typing import Dict
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(name=__name__)


# memory.total [MiB], memory.used [MiB], memory.free [MiB]
# 11264 MiB, 6 MiB, 11005 MiB

def get_gpu_memory(device=0) -> Dict[str, int]:
    response: list[str] = os.popen(cmd='nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv')
    # use pandas to parse the response
    stats: pd.DataFrame = pd.read_csv(filepath_or_buffer=response, header=0)
    
    return stats.loc[device].to_dict()



if __name__ == '__main__':
    pass