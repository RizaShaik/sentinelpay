import pandas as pd
import pytest

from sentinelpay.data.temporal import (
    add_day_index,
    categorical_drift,
    daily_amount_stats_by_group,
    daily_count_by_group,
    daily_fraud_rate,
    daily_volume,
    numeric_drift,
)


def _synthetic_tx():
    # 2 days x several rows each, deliberately skewed so drift tests have
    # something obvious to detect. Day 0: seconds < 86400. Day 1: >= 86400.
    return pd.DataFrame(
        {
            "TransactionDT": [0, 1000, 50000, 86400, 90000, 150000, 170000],
            "isFraud": [0, 0, 1, 0, 0, 1, 1],
            "TransactionAmt": [10.0, 12.0, 11.0, 100.0, 105.0, 98.0, 110.0],
            "ProductCD": ["W", "W", "W", "C", "C", "C", "C"],
        }
    )


def test_add_day_index_buckets_correctly():
    df = add_day_index(_synthetic_tx(), dt_col="TransactionDT", seconds_per_day=86400)
    assert list(df["_day"]) == [0, 0, 0, 1, 1, 1, 1]
    assert df["_day"].dtype == "int32"


def test_add_day_index_does_not_mutate_input():
    original = _synthetic_tx()
    before_cols = list(original.columns)
    add_day_index(original, dt_col="TransactionDT", seconds_per_day=86400)
    assert list(original.columns) == before_cols  # add_day_index returns a copy


def test_day_index_is_chronologically_ordered_with_dt():
    df = add_day_index(_synthetic_tx(), dt_col="TransactionDT", seconds_per_day=86400)
    df_sorted_by_dt = df.sort_values("TransactionDT")
    assert df_sorted_by_dt["_day"].is_monotonic_increasing


def test_daily_volume_counts():
    df = add_day_index(_synthetic_tx(), dt_col="TransactionDT", seconds_per_day=86400)
    vol = daily_volume(df, day_col="_day")
    assert vol.set_index("_day")["n_transactions"].to_dict() == {0: 3, 1: 4}


def test_daily_fraud_rate_calculation():
    df = add_day_index(_synthetic_tx(), dt_col="TransactionDT", seconds_per_day=86400)
    fraud = daily_fraud_rate(df, day_col="_day", target_col="isFraud")
    row0 = fraud[fraud["_day"] == 0].iloc[0]
    row1 = fraud[fraud["_day"] == 1].iloc[0]
    assert row0["n_transactions"] == 3
    assert row0["n_fraud"] == 1
    assert row0["fraud_rate"] == 1 / 3
    assert row1["n_transactions"] == 4
    assert row1["n_fraud"] == 2
    assert row1["fraud_rate"] == 0.5


def test_numeric_drift_detects_obvious_mean_shift():
    df = add_day_index(_synthetic_tx(), dt_col="TransactionDT", seconds_per_day=86400)
    # TransactionAmt jumps from ~10-12 (day 0) to ~98-110 (day 1) -- an obvious shift.
    result = numeric_drift(df, ["TransactionAmt"], day_col="_day", split_day=1)
    row = result.iloc[0]
    assert row["column"] == "TransactionAmt"
    assert row["ks_statistic"] == 1.0  # fully separated distributions
    assert row["mean_early"] < row["mean_late"]
    assert row["pct_missing_early"] == 0.0
    assert row["pct_missing_late"] == 0.0


def test_numeric_drift_reports_missingness_per_period():
    df = _synthetic_tx()
    df.loc[0, "TransactionAmt"] = None
    df = add_day_index(df, dt_col="TransactionDT", seconds_per_day=86400)
    result = numeric_drift(df, ["TransactionAmt"], day_col="_day", split_day=1)
    row = result.iloc[0]
    assert row["pct_missing_early"] == pytest.approx(100 / 3)
    assert row["pct_missing_late"] == 0.0


def test_categorical_drift_detects_obvious_category_shift():
    df = add_day_index(_synthetic_tx(), dt_col="TransactionDT", seconds_per_day=86400)
    # ProductCD is entirely "W" on day 0 and entirely "C" on day 1 -- maximal drift.
    result = categorical_drift(df, ["ProductCD"], day_col="_day", split_day=1)
    row = result.iloc[0]
    assert row["column"] == "ProductCD"
    assert row["chi2_statistic"] > 0
    assert row["n_categories_observed"] == 2


def test_categorical_drift_handles_missing_column_gracefully():
    df = add_day_index(_synthetic_tx(), dt_col="TransactionDT", seconds_per_day=86400)
    result = categorical_drift(df, ["NoSuchColumn"], day_col="_day", split_day=1)
    assert result.empty


def _grouped_synthetic():
    return pd.DataFrame(
        {
            "_day": [0, 0, 0, 1, 1, 1, 1],
            "group": ["A", "A", "B", "A", "B", "B", "B"],
            "TransactionAmt": [10.0, 20.0, 5.0, 30.0, 1.0, 3.0, 5.0],
        }
    )


def test_daily_count_by_group_counts_correctly():
    out = daily_count_by_group(_grouped_synthetic(), group_col="group", day_col="_day")
    lookup = {(row["group"], row["_day"]): row["n_transactions"] for _, row in out.iterrows()}
    assert lookup[("A", 0)] == 2
    assert lookup[("B", 0)] == 1
    assert lookup[("A", 1)] == 1
    assert lookup[("B", 1)] == 3


def test_daily_count_by_group_requires_columns():
    df = _grouped_synthetic()
    with pytest.raises(ValueError):
        daily_count_by_group(df, group_col="no_such_group_col", day_col="_day")
    with pytest.raises(ValueError):
        daily_count_by_group(df, group_col="group", day_col="no_such_day_col")


def test_daily_amount_stats_by_group_computes_expected_aggregates():
    out = daily_amount_stats_by_group(_grouped_synthetic(), group_col="group", amount_col="TransactionAmt", day_col="_day")
    row = out[(out["group"] == "A") & (out["_day"] == 0)].iloc[0]
    assert row["n_transactions"] == 2
    assert row["amount_sum"] == 30.0
    assert row["amount_mean"] == 15.0

    row_b1 = out[(out["group"] == "B") & (out["_day"] == 1)].iloc[0]
    assert row_b1["n_transactions"] == 3
    assert row_b1["amount_sum"] == pytest.approx(9.0)
    assert row_b1["amount_mean"] == pytest.approx(3.0)


def test_daily_amount_stats_by_group_requires_columns():
    df = _grouped_synthetic()
    with pytest.raises(ValueError):
        daily_amount_stats_by_group(df, group_col="group", amount_col="NoSuchAmt", day_col="_day")
    with pytest.raises(ValueError):
        daily_amount_stats_by_group(df, group_col="NoSuchGroup", amount_col="TransactionAmt", day_col="_day")
