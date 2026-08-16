"""RX/TX only mean something next to the link they have to fit through.

A bare "16.3MB/s" says nothing about whether PCIe is the bottleneck; the same
figure is a rounding error on a gen5 x16 link and a third of the budget on the
gen4 x4 riser one of these machines actually runs. So each direction is scored
against the link the card is on right now, and the mode is shown next to it.
"""
import pandas as pd
import pytest

from nvidb.metrics import HIDDEN_COLUMN_PREFIX, visible_columns
from nvidb.utils import format_pcie_link, pcie_link_capacity_kib_per_second

from test_tui_views import _gpu_row, _pool, _without_ansi


def _stats(**overrides):
    row = {
        "GPU": 0,
        "rx": "1.0GB/s",
        "tx": "500MB/s",
        "pcie_link_gen_current": 4,
        "pcie_link_width_current": 4,
        "pcie_link_gen_max": 4,
        "pcie_link_width_max": 16,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _with_link(pool, stats):
    pool._add_pcie_link_columns(stats)
    return stats.iloc[0]


@pytest.mark.parametrize(
    "generation, width, expected_gb_per_second",
    [
        (3, 16, 15.75),
        (4, 4, 7.88),
        (4, 16, 31.51),
        (5, 16, 63.02),
    ],
)
def test_link_capacity_matches_the_published_pcie_figures(
    generation, width, expected_gb_per_second
):
    capacity = pcie_link_capacity_kib_per_second(generation, width)

    assert capacity * 1024 / 1e9 == pytest.approx(expected_gb_per_second, abs=0.02)


def test_an_unknown_link_has_no_capacity_and_no_load():
    pool = _pool()
    assert pcie_link_capacity_kib_per_second(None, 16) is None
    assert pcie_link_capacity_kib_per_second(4, None) is None
    assert format_pcie_link(None, None) == "N/A"

    row = _with_link(
        pool,
        _stats(
            pcie_link_gen_current=None,
            pcie_link_width_current=None,
            pcie_link_gen_max=None,
            pcie_link_width_max=None,
        ),
    )

    assert row["link"] == "N/A"
    assert row[f"{HIDDEN_COLUMN_PREFIX}rx_percent"] is None
    assert row[f"{HIDDEN_COLUMN_PREFIX}link_capacity"] == "N/A"


def test_nvidia_smi_link_text_is_parsed():
    """`nvidia-smi -q -x` spells widths as "16x" where NVML returns integers."""
    pool = _pool()

    row = _with_link(
        pool,
        _stats(
            pcie_link_gen_current="4",
            pcie_link_width_current="16x",
            pcie_link_gen_max="4",
            pcie_link_width_max="16x",
        ),
    )

    assert row["link"] == "4.0x16"
    assert row[f"{HIDDEN_COLUMN_PREFIX}link_max"] == ""


def test_load_is_scored_against_the_live_link_not_the_cards_maximum():
    """A 3090 Ti behind an x4 riser reports a maximum of x16 all the same.

    Dividing by that maximum would report a saturated link as a quarter busy.
    """
    pool = _pool()

    row = _with_link(pool, _stats())

    assert row["link"] == "4.0x4"
    assert row[f"{HIDDEN_COLUMN_PREFIX}link_max"] == "4.0x16"
    # 1.0 GiB/s of a gen4 x4 direction (7.88 GB/s), not of gen4 x16.
    assert row[f"{HIDDEN_COLUMN_PREFIX}rx_percent"] == pytest.approx(13.6, abs=0.2)
    assert row[f"{HIDDEN_COLUMN_PREFIX}tx_percent"] == pytest.approx(6.7, abs=0.2)


def test_an_idle_downshifted_link_falls_back_to_the_maximum_when_unreported():
    pool = _pool()

    row = _with_link(
        pool,
        _stats(pcie_link_gen_current=None, pcie_link_width_current=None),
    )

    assert row["link"] == "4.0x16"
    assert row[f"{HIDDEN_COLUMN_PREFIX}link_max"] == ""


def test_a_saturated_direction_is_red_and_a_quiet_one_is_not(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    pool = _pool()
    pool.unified_group_by_node = False
    busy = _gpu_row(0, "RTX 3090 Ti", "99 %", "22817/24564")
    busy.update(
        {
            "Node": "gem12",
            "Hostname": "100.109.8.69",
            "link": "4.0x4",
            "rx": "7.0GB/s",
            "tx": "20MB/s",
            f"{HIDDEN_COLUMN_PREFIX}rx_percent": 95.4,
            f"{HIDDEN_COLUMN_PREFIX}tx_percent": 0.3,
        }
    )
    monkeypatch.setattr(
        "nvidb.connection.os.get_terminal_size",
        lambda: __import__("os").terminal_size((200, 40)),
    )

    rendered = pool._format_fixed_width_table(
        pd.DataFrame([busy]),
        column_labels=pool.UNIFIED_TABLE_LABELS,
    )

    row_line = [line for line in rendered.splitlines() if "7.0GB/s" in line][0]
    cells = row_line.split("│")

    def _cell(marker):
        cell = [cell for cell in cells if marker in cell][0]
        # The hairline separators carry their own dim-grey codes; they are
        # chrome, not cell content, so they do not count as cell colour.
        return cell.replace("\x1b[90m", "").replace("\x1b[0m", "")

    rx_cell = _cell("7.0GB/s")
    tx_cell = _cell("20MB/s")

    assert "\x1b[41m" in rx_cell  # RX fills its cell in red at 95% of the link
    # The quiet direction stays plain instead of borrowing RX's colour.
    assert "\x1b[" not in tx_cell


def test_helper_columns_never_become_table_columns(monkeypatch):
    pool = _pool()
    row = _gpu_row(0, "RTX 4090", "10 %", "1024/24576")
    row.update(
        {
            f"{HIDDEN_COLUMN_PREFIX}rx_percent": 42.0,
            f"{HIDDEN_COLUMN_PREFIX}link_capacity": "7.34GB/s",
        }
    )
    monkeypatch.setattr(
        "nvidb.connection.os.get_terminal_size",
        lambda: __import__("os").terminal_size((200, 40)),
    )

    rendered = pool._format_fixed_width_table(pd.DataFrame([row]))

    assert HIDDEN_COLUMN_PREFIX not in rendered
    assert "7.34GB/s" not in rendered
    assert visible_columns(["GPU", f"{HIDDEN_COLUMN_PREFIX}rx_percent"]) == ["GPU"]


def test_the_unified_table_carries_the_helper_columns_across_nodes():
    pool = _pool()
    row = _gpu_row(0, "RTX 3090 Ti", "99 %", "22817/24564")
    row[f"{HIDDEN_COLUMN_PREFIX}rx_percent"] = 95.4

    table = pool._build_unified_gpu_table({1: (pd.DataFrame([row]), {})})

    assert table.loc[0, "link"] == "4.0x16"
    assert table.loc[0, f"{HIDDEN_COLUMN_PREFIX}rx_percent"] == 95.4


def test_the_detailed_card_names_the_link_and_scores_each_direction(monkeypatch):
    pool = _pool()
    pool.unified_detailed = True
    row = _gpu_row(0, "RTX 3090 Ti", "99 %", "22817/24564")
    row.update(
        {
            "Node": "gem12",
            "Hostname": "100.109.8.69",
            "link": "4.0x4",
            "rx": "7.0GB/s",
            "tx": "20MB/s",
            f"{HIDDEN_COLUMN_PREFIX}link_max": "4.0x16",
            f"{HIDDEN_COLUMN_PREFIX}link_capacity": "7.34GB/s",
            f"{HIDDEN_COLUMN_PREFIX}rx_percent": 95.4,
            f"{HIDDEN_COLUMN_PREFIX}tx_percent": 0.26,
        }
    )
    monkeypatch.setattr(
        "nvidb.connection.os.get_terminal_size",
        lambda: __import__("os").terminal_size((160, 40)),
    )

    rendered = _without_ansi(
        pool._format_unified_detailed_table(pd.DataFrame([row]))
    )

    assert "Link 4.0x4 7.34GB/s" in rendered
    assert "of 4.0x16" in rendered
    assert "RX 7.0GB/s 95%" in rendered
    # A trickle must not round down to a flat 0%.
    assert "TX 20MB/s <1%" in rendered


def test_a_link_with_no_traffic_says_so_without_a_percentage(monkeypatch):
    pool = _pool()
    pool.unified_detailed = True
    row = _gpu_row(0, "RTX 3090 Ti", "0 %", "0/24564")
    row.update(
        {
            "Node": "gem12",
            "Hostname": "100.109.8.69",
            "rx": "0",
            "tx": "0",
            f"{HIDDEN_COLUMN_PREFIX}rx_percent": 0.0,
            f"{HIDDEN_COLUMN_PREFIX}tx_percent": 0.0,
        }
    )
    monkeypatch.setattr(
        "nvidb.connection.os.get_terminal_size",
        lambda: __import__("os").terminal_size((160, 40)),
    )

    rendered = _without_ansi(
        pool._format_unified_detailed_table(pd.DataFrame([row]))
    )

    io_line = [line for line in rendered.splitlines() if "I/O" in line][0]
    assert "RX 0  " in io_line
    assert "%" not in io_line
