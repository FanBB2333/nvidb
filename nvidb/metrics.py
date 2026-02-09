from __future__ import annotations

from typing import Iterable, Sequence

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
    if df_columns is None:
        cols = set()
    else:
        cols = set(list(df_columns))
    return [c for c in desired if c in cols]
