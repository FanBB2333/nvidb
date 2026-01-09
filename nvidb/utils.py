from typing import Dict, Any
import subprocess
import logging
import os
import re
import pandas as pd
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(name=__name__)


# memory.total [MiB], memory.used [MiB], memory.free [MiB]
# 11264 MiB, 6 MiB, 11005 MiB

def get_gpu_memory(device=0) -> Dict[str, int]:
    try:
        response: list[str] = os.popen(cmd='nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv')
        # use pandas to parse the response
        stats: pd.DataFrame = pd.read_csv(filepath_or_buffer=response, header=0)
        if stats.empty:
            return {}
        return stats.loc[device].to_dict()
    except Exception:
        return {}



def get_gpu_stats_query():
    try:
        result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True)
        if result.returncode != 0:
            return
        response = result.stdout
        if not response or not response.strip().startswith('<?xml'):
            return
        root = ET.fromstring(response)
        gpus = root.findall('gpu')
        for gpu in gpus:
            product_name = gpu.find('product_name').text
            product_architecture = gpu.find('product_architecture').text

            pci = gpu.find('pci')
            tx_util = pci.find('tx_util').text
            rx_util = pci.find('rx_util').text
            fan_speed = gpu.find('fan_speed').text

            fb_memory_usage = gpu.find('fb_memory_usage')
            total = fb_memory_usage.find('total').text
            used = fb_memory_usage.find('used').text
            free = fb_memory_usage.find('free').text

            utilization = gpu.find('utilization')
            gpu_util = utilization.find('gpu_util').text
            memory_util = utilization.find('memory_util').text

            temperature = gpu.find('temperature')
            gpu_temp = temperature.find('gpu_temp').text

            gpu_power_readings = gpu.find('gpu_power_readings')
            power_state = gpu_power_readings.find('power_state').text
            power_draw = gpu_power_readings.find('power_draw').text
            current_power_limit = gpu_power_readings.find('current_power_limit').text

            processes = gpu.find('processes')

            logging.info(msg=f"Product Name: {product_name}")
            logging.info(msg=f"Product Architecture: {product_architecture}, TX Util: {tx_util}, RX Util: {rx_util}, Fan Speed: {fan_speed}")
            logging.info(msg=f"Total Memory: {total}, Used Memory: {used}, Free Memory: {free}")
            logging.info(msg=f"GPU Util: {gpu_util}, Memory Util: {memory_util}")
            logging.info(msg=f"GPU Temp: {gpu_temp}")
            logging.info(msg=f"Power Draw: {power_draw}, Power Limit: {current_power_limit}, Power State: {power_state}")
    except Exception:
        pass
        

def num_from_str(s: str, type: Any = float) -> int:
    return type(''.join(filter(str.isdigit, s)))

def units_from_str(s: str) -> str:
    return ''.join(filter(str.isalpha, s))

def extract_numbers(s):
    # 捕获整数和小数部分，包括前导零和小数点
    return re.findall(r'\d+\.?\d*', s)

def extract_value_and_unit(s: str) -> tuple[str, str]:
    """从字符串中提取数值和单位，例如 '1024 KB/s' -> ('1024', 'KB/s')"""
    if not s or s.strip() == 'N/A':
        return ('0', '')
    
    # 匹配数字(包括小数)和单位
    match = re.match(r'(\d+\.?\d*)\s*(.*)$', s.strip())
    if match:
        value, unit = match.groups()
        return (value, unit.strip())
    return ('0', '')

def get_utilization_color(value_str: str) -> str:
    """根据利用率获取对应的颜色
    
    Args:
        value_str: 利用率字符串，如 "50%", "75", "N/A"
    
    Returns:
        颜色名称字符串: 'red', 'yellow', 或 None (无颜色)
    """
    if not value_str or value_str.strip() in ['N/A', '0', '0%']:
        return None
    
    try:
        # 提取数值，移除百分号
        numeric_value = float(value_str.replace('%', '').replace(' ', '').strip())
        
        if numeric_value >= 80:
            return 'red'      # 高利用率 (>=80%)
        elif numeric_value >= 50:
            return 'yellow'   # 中等利用率 (50%-80%)
        elif numeric_value >= 5:
            return 'green'    # 低但非 idle (5%-50%)
        else:
            return None       # idle (<5%)
            
    except (ValueError, AttributeError):
        return None

def get_memory_color(value_str: str) -> str:
    """根据内存利用率获取对应的颜色
    
    Args:
        value_str: 内存利用率字符串，如 "50%", "75", "N/A"
    
    Returns:
        颜色名称字符串: 'red', 'yellow', 或 None (无颜色)
    """
    if not value_str or value_str.strip() in ['N/A', '0', '0%']:
        return None
    
    try:
        # 提取数值，移除百分号
        numeric_value = float(value_str.replace('%', '').strip())
        
        if numeric_value >= 60:
            return 'red'      # 高内存使用率 (>=60%)
        elif numeric_value >= 10:
            return 'yellow'   # 中等内存使用率 (10%-60%)
        else:
            return None       # 低内存使用率 (<10%)
            
    except (ValueError, AttributeError):
        return None

def get_memory_ratio_color(used_str: str, total_str: str) -> str:
    """根据内存使用比例获取对应的颜色
    
    Args:
        used_str: 已使用内存字符串
        total_str: 总内存字符串
    
    Returns:
        颜色名称字符串: 'red', 'yellow', 或 None (无颜色)
    """
    try:
        used_numbers = extract_numbers(used_str)
        total_numbers = extract_numbers(total_str)
        
        if used_numbers and total_numbers:
            used_val = float(used_numbers[0])
            total_val = float(total_numbers[0])
            
            if total_val > 0:
                usage_ratio = (used_val / total_val) * 100
                
                if usage_ratio >= 60:
                    return 'red'      # 高内存使用率 (>=60%)
                elif usage_ratio >= 10:
                    return 'yellow'   # 中等内存使用率 (10%-60%)
                else:
                    return None       # 低内存使用率 (<10%)
    except (ValueError, AttributeError, IndexError):
        pass
    
    return None

def format_bandwidth(value: str, unit: str) -> str:
    """格式化带宽显示，优化单位"""
    if not value or value == '0':
        return '0'
    
    try:
        val = float(value)
        
        # 如果值为0，返回简洁的0
        if val == 0:
            return '0'
        
        # 如果单位包含 /s，说明是带宽，进行单位转换
        if '/s' in unit.lower():
            if 'kb/s' in unit.lower():
                if val >= 1024 * 1024:  # >= 1GB/s
                    return f"{val/(1024*1024):.2f}GB/s"
                elif val >= 1024:  # >= 1MB/s
                    return f"{val/1024:.1f}MB/s"
                else:
                    return f"{val:.0f}KB/s"
            elif 'mb/s' in unit.lower():
                if val >= 1024:
                    return f"{val/1024:.2f}GB/s"
                else:
                    return f"{val:.1f}MB/s"
            elif 'gb/s' in unit.lower():
                return f"{val:.2f}GB/s"
        
        # 如果没有单位，直接返回数值
        if not unit:
            return f"{val:.0f}" if val == int(val) else f"{val:.1f}"
            
        # 其他情况保持原样
        return f"{value}{unit}"
        
    except ValueError:
        return f"{value}{unit}" if unit else value

def xml_to_dict(root):
    # root = ET.fromstring(xml_string)
    child_to_dict = {} 
    for child in root:
        child_tag = child.tag
        child_text = child.text

        if len(child) > 0: # child nodes available
            child_to_dict[child_tag] = xml_to_dict(child) 
        else:
            child_to_dict[child_tag] = child_text
    return child_to_dict


if __name__ == '__main__':
    pass
