import pandas as pd
import pytest

from sentinelpay.data.split import (
    PartitionRange,
    SplitConfig,
    assign_partition,
    validate_split,
)


def _good_split_config() -> SplitConfig:
    # Small synthetic analogue of configs/split.yaml: 20 "days" total.
    return SplitConfig(
        partitions={
            "train": PartitionRange("train", 1, 10),
            "embargo_1": PartitionRange("embargo_1", 11, 12),
            "validation": PartitionRange("validation", 13, 16),
            "embargo_2": PartitionRange("embargo_2", 17, 18),
            "holdout": PartitionRange("holdout", 19, 20),
        }
    )


def _synthetic_df(days: list[int]) -> pd.DataFrame:
    # One row per day value given, deterministic TransactionDT from day.
    return pd.DataFrame(
        {
            "TransactionID": range(1, len(days) + 1),
            "TransactionDT": [d * 86400 for d in days],
            "_day": days,
        }
    )


def test_assign_partition_labels_every_row_correctly():
    df = _synthetic_df([1, 5, 10, 11, 12, 13, 16, 17, 18, 19, 20])
    config = _good_split_config()
    out = assign_partition(df, config, day_col="_day")
    expected = [
        "train", "train", "train", "embargo_1", "embargo_1", "validation",
        "validation", "embargo_2", "embargo_2", "holdout", "holdout",
    ]
    assert list(out["partition"]) == expected


def test_assign_partition_leaves_out_of_range_rows_unassigned():
    df = _synthetic_df([0, 21])  # outside every configured range
    config = _good_split_config()
    out = assign_partition(df, config, day_col="_day")
    assert out["partition"].isna().all()


def test_partitions_are_exhaustive_and_mutually_exclusive_for_in_range_days():
    all_days = list(range(1, 21))
    df = _synthetic_df(all_days)
    config = _good_split_config()
    out = assign_partition(df, config, day_col="_day")
    assert out["partition"].isna().sum() == 0
    # every row has exactly one partition value (categorical Series is single-valued per row by construction)
    assert out["partition"].notna().all()


def test_validate_split_passes_on_well_formed_split():
    df = _synthetic_df(list(range(1, 21)))
    config = _good_split_config()
    result = validate_split(df, config, dt_col="TransactionDT", day_col="_day", id_col="TransactionID")

    assert result.is_valid
    assert result.config_ranges_valid
    assert result.n_unassigned_rows == 0
    assert result.empty_partitions == []
    assert result.n_transaction_ids_in_multiple_partitions == 0
    assert result.chronological_order_ok
    assert result.holdout_strictly_after_validation
    assert result.embargoes_isolated
    assert sum(result.partition_row_counts.values()) == 20


def test_validate_split_detects_overlapping_configured_ranges():
    bad_config = SplitConfig(
        partitions={
            "train": PartitionRange("train", 1, 10),
            "embargo_1": PartitionRange("embargo_1", 9, 12),  # overlaps train
            "validation": PartitionRange("validation", 13, 16),
            "embargo_2": PartitionRange("embargo_2", 17, 18),
            "holdout": PartitionRange("holdout", 19, 20),
        }
    )
    df = _synthetic_df(list(range(1, 21)))
    result = validate_split(df, bad_config)

    assert not result.is_valid
    assert not result.config_ranges_valid
    assert result.config_errors  # non-empty, describes the overlap


def test_validate_split_detects_empty_required_partition():
    df = _synthetic_df(list(range(1, 11)))  # only days 1-10: train only, everything else empty
    config = _good_split_config()
    result = validate_split(df, config)

    assert not result.is_valid
    assert result.config_ranges_valid  # config itself is fine
    assert set(result.empty_partitions) == {"embargo_1", "validation", "embargo_2", "holdout"}


def test_validate_split_detects_transaction_id_crossing_partitions():
    df = _synthetic_df(list(range(1, 21)))
    # Force one TransactionID to appear in two different day rows (two partitions).
    df.loc[df["_day"] == 15, "TransactionID"] = df.loc[df["_day"] == 5, "TransactionID"].iloc[0]
    config = _good_split_config()
    result = validate_split(df, config)

    assert not result.is_valid
    assert result.n_transaction_ids_in_multiple_partitions >= 1


def test_validate_split_checks_chronological_order_from_dt_not_row_order():
    # Shuffle row order; TransactionDT/day values still define the correct order.
    df = _synthetic_df(list(range(1, 21))).sample(frac=1.0, random_state=0).reset_index(drop=True)
    config = _good_split_config()
    result = validate_split(df, config)
    assert result.is_valid
    assert result.chronological_order_ok


def test_validate_split_requires_expected_columns():
    df = pd.DataFrame({"TransactionID": [1], "TransactionDT": [86400]})  # missing _day
    config = _good_split_config()
    with pytest.raises(ValueError):
        validate_split(df, config, day_col="_day")


def test_embargo_rows_excluded_when_filtering_to_train_or_validation():
    df = _synthetic_df(list(range(1, 21)))
    config = _good_split_config()
    out = assign_partition(df, config, day_col="_day")

    train_and_val = out[out["partition"].isin(["train", "validation"])]
    assert set(train_and_val["_day"]).isdisjoint(set(range(11, 13)))  # embargo_1
    assert set(train_and_val["_day"]).isdisjoint(set(range(17, 19)))  # embargo_2
