import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(name=__name__)


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


# Usable throughput of one PCIe lane in one direction, in bytes per second,
# with the link encoding already discounted (8b/10b up to gen2, 128b/130b for
# gen3-5, PAM4 FLIT for gen6+). PCIe is full duplex, so RX and TX each get the
# whole figure rather than sharing it.
PCIE_LANE_BYTES_PER_SECOND = {
    1: 250_000_000,
    2: 500_000_000,
    3: 984_600_000,
    4: 1_969_200_000,
    5: 3_938_500_000,
    6: 7_563_000_000,
    7: 15_125_000_000,
}


def parse_link_number(value):
    """Read a PCIe generation or lane count, as NVML or nvidia-smi reports it.

    nvidia-smi spells widths as "16x", NVML returns plain integers, and a
    driver that cannot answer returns nothing at all.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        return number if number > 0 else None
    numbers = extract_numbers(str(value))
    if not numbers:
        return None
    try:
        number = int(float(numbers[0]))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def pcie_link_capacity_kib_per_second(generation, width):
    """Return one direction's PCIe ceiling in KiB/s, NVML's throughput unit.

    Returns None when the generation or width is unknown, which keeps callers
    from inventing a load percentage out of a guessed link speed.
    """
    generation = parse_link_number(generation)
    width = parse_link_number(width)
    if generation is None or width is None:
        return None
    per_lane = PCIE_LANE_BYTES_PER_SECOND.get(generation)
    if per_lane is None:
        return None
    return per_lane * width / 1024


def format_pcie_link(generation, width) -> str:
    """Render a link mode the way lspci and nvidia-smi do, e.g. "4.0x16"."""
    generation = parse_link_number(generation)
    width = parse_link_number(width)
    if generation is None or width is None:
        return "N/A"
    return f"{generation}.0x{width}"


def get_pcie_load_color(percent) -> str:
    """Colour a PCIe direction by how close it runs to the link ceiling.

    A saturated link stalls transfers, so the warning starts lower than for
    VRAM: past ~40% the interconnect is already shaping throughput.
    """
    if percent is None:
        return None
    try:
        value = float(percent)
    except (TypeError, ValueError):
        return None
    if value >= 70:
        return 'red'
    if value >= 40:
        return 'yellow'
    if value >= 5:
        return 'green'
    return None


def xml_to_dict(root):
    child_to_dict = {}
    for child in root:
        child_tag = child.tag
        child_text = child.text

        if len(child) > 0: # child nodes available
            child_to_dict[child_tag] = xml_to_dict(child)
        else:
            child_to_dict[child_tag] = child_text
    return child_to_dict
