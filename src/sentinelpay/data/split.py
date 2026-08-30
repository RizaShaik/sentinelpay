"""Chronological partitioning protocol: train / embargo_1 / validation /
embargo_2 / holdout.

Partition boundaries live in configs/split.yaml as fixed day-index ranges
(day = TransactionDT // seconds_per_day). They are NOT derived by searching
for a boundary that optimizes fraud rate, drift, or any label-based
objective -- `load_split_config` only parses and structurally validates the
configured ranges; it never looks at transaction data.

`holdout` is reserved for Phase H. Nothing in this module or in
eda/run_phase_b.py computes a content statistic (fraud rate, drift,
correlation, missingness, entity/proxy stats) on holdout rows -- only its
row count and chronological position are ever checked.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sentinelpay.config import PROJECT_ROOT

DEFAULT_SPLIT_CONFIG_PATH = PROJECT_ROOT / "configs" / "split.yaml"

PARTITION_ORDER = ["train", "embargo_1", "validation", "embargo_2", "holdout"]
DEVELOPMENT_PARTITIONS = ["train", "embargo_1", "validation", "embargo_2"]  # excludes holdout


@dataclass(frozen=True)
class PartitionRange:
    name: str
    start_day: int
    end_day: int  # inclusive


@dataclass(frozen=True)
class SplitConfig:
    partitions: dict[str, PartitionRange]


def load_split_config(path: Path | str = DEFAULT_SPLIT_CONFIG_PATH) -> SplitConfig:
    """Parse configs/split.yaml. Raises ValueError on a malformed config
    (missing partition, non-integer day, etc.) -- this is a schema check,
    not a data-driven validity check (see `validate_split` for that)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    raw_partitions = raw.get("partitions", {})
    missing = [name for name in PARTITION_ORDER if name not in raw_partitions]
    if missing:
        raise ValueError(f"configs/split.yaml is missing required partition(s): {missing}")

    partitions: dict[str, PartitionRange] = {}
    for name in PARTITION_ORDER:
        entry = raw_partitions[name]
        try:
            start_day = int(entry["start_day"])
            end_day = int(entry["end_day"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"configs/split.yaml partition '{name}' has invalid start_day/end_day") from exc
        partitions[name] = PartitionRange(name=name, start_day=start_day, end_day=end_day)

    return SplitConfig(partitions=partitions)


def assign_partition(
    df: pd.DataFrame, split_config: SplitConfig, day_col: str = "_day"
) -> pd.DataFrame:
    """Add a `partition` categorical column, one label per row, based on
    `day_col` (expected to already be TransactionDT // seconds_per_day).
    Rows outside every configured range get NaN -- surfaced by
    `validate_split` as n_unassigned_rows, never silently dropped."""
    if day_col not in df.columns:
        raise ValueError(f"assign_partition requires column '{day_col}' (run add_day_index first)")

    df = df.copy()
    days = df[day_col]
    partition = pd.Series(pd.NA, index=df.index, dtype="object")
    for name in PARTITION_ORDER:
        pr = split_config.partitions[name]
        mask = (days >= pr.start_day) & (days <= pr.end_day) & partition.isna()
        partition = partition.where(~mask, name)
    df["partition"] = pd.Categorical(partition, categories=PARTITION_ORDER)
    return df


@dataclass
class SplitValidationResult:
    is_valid: bool
    config_ranges_valid: bool
    config_errors: list[str]
    partition_row_counts: dict[str, int]
    n_unassigned_rows: int
    empty_partitions: list[str]
    n_transaction_ids_in_multiple_partitions: int
    chronological_order_ok: bool
    chronological_bounds_dt: dict[str, tuple[float, float]]
    holdout_strictly_after_validation: bool
    embargoes_isolated: bool

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "config_ranges_valid": self.config_ranges_valid,
            "config_errors": self.config_errors,
            "partition_row_counts": self.partition_row_counts,
            "n_unassigned_rows": self.n_unassigned_rows,
            "empty_partitions": self.empty_partitions,
            "n_transaction_ids_in_multiple_partitions": self.n_transaction_ids_in_multiple_partitions,
            "chronological_order_ok": self.chronological_order_ok,
            "chronological_bounds_dt": self.chronological_bounds_dt,
            "holdout_strictly_after_validation": self.holdout_strictly_after_validation,
            "embargoes_isolated": self.embargoes_isolated,
        }


def _invalid_config_result(config_errors: list[str]) -> SplitValidationResult:
    return SplitValidationResult(
        is_valid=False,
        config_ranges_valid=False,
        config_errors=config_errors,
        partition_row_counts={},
        n_unassigned_rows=-1,
        empty_partitions=list(PARTITION_ORDER),
        n_transaction_ids_in_multiple_partitions=-1,
        chronological_order_ok=False,
        chronological_bounds_dt={},
        holdout_strictly_after_validation=False,
        embargoes_isolated=False,
    )


def _check_config_ranges(split_config: SplitConfig) -> list[str]:
    errors: list[str] = []
    ranges = [(name, split_config.partitions[name]) for name in PARTITION_ORDER]
    for name, pr in ranges:
        if pr.start_day > pr.end_day:
            errors.append(f"{name}: start_day {pr.start_day} > end_day {pr.end_day}")
    for (name_a, pr_a), (name_b, pr_b) in zip(ranges, ranges[1:]):
        if pr_b.start_day <= pr_a.end_day:
            errors.append(f"{name_a} (ends day {pr_a.end_day}) overlaps/is not before {name_b} (starts day {pr_b.start_day})")
    return errors


def validate_split(
    df: pd.DataFrame,
    split_config: SplitConfig,
    dt_col: str = "TransactionDT",
    day_col: str = "_day",
    id_col: str = "TransactionID",
) -> SplitValidationResult:
    """Structural validation of a partitioning, run every time it's produced.

    Raises ValueError only for missing required columns (a caller bug).
    Substantive validity problems (overlapping config, empty partition,
    ordering violations, an ID split across partitions) are reported as
    fields on the returned result, not exceptions -- so this is safe to call
    speculatively and inspect, including in tests that deliberately feed it
    a broken split.
    """
    for col in (dt_col, day_col, id_col):
        if col not in df.columns:
            raise ValueError(f"validate_split requires column '{col}'")

    config_errors = _check_config_ranges(split_config)
    if config_errors:
        return _invalid_config_result(config_errors)

    assigned = assign_partition(df, split_config, day_col=day_col)

    counts = assigned["partition"].value_counts(dropna=False)
    partition_row_counts = {name: int(counts.get(name, 0)) for name in PARTITION_ORDER}
    n_unassigned = int(assigned["partition"].isna().sum())
    empty_partitions = [name for name in PARTITION_ORDER if partition_row_counts[name] == 0]

    labeled = assigned.dropna(subset=["partition"])
    ids_per_partition_count = labeled.groupby(id_col, observed=True)["partition"].nunique()
    n_id_multi = int((ids_per_partition_count > 1).sum())

    bounds: dict[str, tuple[float, float]] = {}
    order_ok = True
    prev_max = None
    for name in PARTITION_ORDER:
        sub = labeled[labeled["partition"] == name]
        if sub.empty:
            order_ok = False
            continue
        lo, hi = float(sub[dt_col].min()), float(sub[dt_col].max())
        bounds[name] = (lo, hi)
        if prev_max is not None and lo <= prev_max:
            order_ok = False
        prev_max = hi

    holdout_after_val = (
        "validation" in bounds and "holdout" in bounds and bounds["holdout"][0] > bounds["validation"][1]
    )
    embargoes_isolated = "embargo_1" in bounds and "embargo_2" in bounds

    is_valid = (
        not config_errors
        and n_unassigned == 0
        and not empty_partitions
        and n_id_multi == 0
        and order_ok
        and holdout_after_val
        and embargoes_isolated
    )

    return SplitValidationResult(
        is_valid=is_valid,
        config_ranges_valid=True,
        config_errors=[],
        partition_row_counts=partition_row_counts,
        n_unassigned_rows=n_unassigned,
        empty_partitions=empty_partitions,
        n_transaction_ids_in_multiple_partitions=n_id_multi,
        chronological_order_ok=order_ok,
        chronological_bounds_dt=bounds,
        holdout_strictly_after_validation=holdout_after_val,
        embargoes_isolated=embargoes_isolated,
    )
