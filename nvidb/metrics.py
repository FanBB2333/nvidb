from __future__ import annotations

from typing import Iterable, Sequence

# Columns under this prefix ride along with a GPU table so renderers can read
# values they do not draw - a PCIe load percentage next to its RX/TX cell, for
# example. Every view filters them out before listing columns to the user.
HIDDEN_COLUMN_PREFIX = "_nvidb_"


def visible_columns(df_columns: Iterable[str]) -> list[str]:
    """Drop the renderer-only columns from a GPU table's column list."""
    return [
        column
        for column in df_columns
        if not str(column).startswith(HIDDEN_COLUMN_PREFIX)
    ]


# Advanced profiling metrics (DCGM)
ADVANCED_METRIC_LABELS = {
    "GPU": "GPU",
    "sm_active": "SM Active",
    "sm_occupancy": "SM Occupancy",
    "instruction_throughput": "Instruction Throughput",
    "sol": "SOL",
    "tensor_active": "Tensor Active",
    "fp64_active": "FP64 Active",
    "fp32_active": "FP32 Active",
    "fp16_active": "FP16 Active",
}

# Group definitions used by both CLI and Web for consistent display.
ADVANCED_METRIC_GROUPS: Sequence[tuple[str, Sequence[str]]] = (
    ("SM", ("GPU", "sm_active", "sm_occupancy")),
    (
        "Pipelines",
        ("GPU", "tensor_active", "fp64_active", "fp32_active", "fp16_active"),
    ),
    ("Derived", ("GPU", "instruction_throughput", "sol")),
)


def present_columns(df_columns: Iterable[str], desired: Sequence[str]) -> list[str]:
    cols = set(df_columns)
    return [c for c in desired if c in cols]
