import pandas as pd
import pytest

from sentinelpay.data.history import (
    prior_group_count,
    prior_group_amount_stats,
    time_since_last_group_event,
)


def _synthetic():
    # Deliberately synthetic group key -- never a real proxy key (payment_proxy_key /
    # device_proxy_key / ProductCD). Two rows (idx 2, 3) share group "A" and the
    # same TransactionDT (200) to exercise tie handling.
    return pd.DataFrame(
        {
            "synthetic_group": ["A", "A", "A", "A", "B", "A"],
            "TransactionDT": [100, 150, 200, 200, 100, 300],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 5.0, 50.0],
        }
    )


def test_prior_group_count_basic():
    df = _synthetic()
    out = prior_group_count(df, group_col="synthetic_group", dt_col="TransactionDT")
    # group A rows sorted by dt: 100, 150, 200, 200, 300 (idx 0,1,2,3,5)
    assert out.loc[0] == 0  # dt=100, nothing earlier
    assert out.loc[1] == 1  # dt=150, one earlier (100)
    assert out.loc[2] == 2  # dt=200, two earlier (100,150)
    assert out.loc[3] == 2  # dt=200 (tie with idx 2) -- same prior count, does not count idx 2
    assert out.loc[5] == 4  # dt=300, four earlier in group A
    assert out.loc[4] == 0  # group B, only row


def test_prior_group_count_requires_columns():
    df = _synthetic()
    with pytest.raises(ValueError):
        prior_group_count(df, group_col="no_such_col", dt_col="TransactionDT")
    with pytest.raises(ValueError):
        prior_group_count(df, group_col="synthetic_group", dt_col="no_such_col")


def test_prior_group_amount_stats_basic():
    df = _synthetic()
    out = prior_group_amount_stats(df, group_col="synthetic_group", amount_col="TransactionAmt", dt_col="TransactionDT")
    # group A: dt=100 (amt 10), dt=150 (amt 20), dt=200 tie (amt 30 + amt 40), dt=300 (amt 50)
    assert out.loc[0, "prior_sum"] == 0.0
    assert out.loc[0, "prior_count"] == 0
    assert out.loc[1, "prior_sum"] == 10.0
    assert out.loc[1, "prior_count"] == 1
    # both tied rows (idx 2, 3) see only the strictly-earlier bucket (100, 150) -> sum 30
    assert out.loc[2, "prior_sum"] == 30.0
    assert out.loc[2, "prior_count"] == 2
    assert out.loc[3, "prior_sum"] == 30.0
    assert out.loc[3, "prior_count"] == 2
    # neither tied row counts the other's amount (30 or 40 excluded from each other)
    assert out.loc[2, "prior_mean"] == pytest.approx(15.0)
    # last row (dt=300) sees all four earlier group-A rows: 10+20+30+40=100
    assert out.loc[5, "prior_sum"] == 100.0
    assert out.loc[5, "prior_count"] == 4


def test_time_since_last_group_event_basic():
    df = _synthetic()
    out = time_since_last_group_event(df, group_col="synthetic_group", dt_col="TransactionDT")
    assert pd.isna(out.loc[0])  # first in group A
    assert out.loc[1] == 50  # 150 - 100
    assert out.loc[2] == 50  # 200 - 150
    assert out.loc[3] == 50  # tie: same "previous distinct dt" (150) as idx 2
    assert out.loc[5] == 100  # 300 - 200
    assert pd.isna(out.loc[4])  # only row in group B


def test_tied_timestamps_never_see_each_other():
    # Two rows, same group, same TransactionDT -- neither may contribute to the other.
    df = pd.DataFrame(
        {
            "synthetic_group": ["A", "A"],
            "TransactionDT": [500, 500],
            "TransactionAmt": [1000.0, 2000.0],
        }
    )
    counts = prior_group_count(df, group_col="synthetic_group", dt_col="TransactionDT")
    stats = prior_group_amount_stats(df, group_col="synthetic_group", amount_col="TransactionAmt", dt_col="TransactionDT")
    recency = time_since_last_group_event(df, group_col="synthetic_group", dt_col="TransactionDT")
    assert (counts == 0).all()
    assert (stats["prior_count"] == 0).all()
    assert (stats["prior_sum"] == 0.0).all()
    assert recency.isna().all()


def test_adversarial_perturbing_future_row_amount_does_not_change_past_rows():
    df = _synthetic()
    stats_before = prior_group_amount_stats(df, group_col="synthetic_group", amount_col="TransactionAmt", dt_col="TransactionDT")

    df_mutated = df.copy()
    df_mutated.loc[5, "TransactionAmt"] = 999999.0  # idx 5 is the chronologically-last row in group A
    stats_after = prior_group_amount_stats(df_mutated, group_col="synthetic_group", amount_col="TransactionAmt", dt_col="TransactionDT")

    for idx in [0, 1, 2, 3]:
        assert stats_before.loc[idx, "prior_sum"] == stats_after.loc[idx, "prior_sum"]
        assert stats_before.loc[idx, "prior_count"] == stats_after.loc[idx, "prior_count"]


def test_adversarial_appending_future_row_does_not_change_existing_rows():
    df = _synthetic()
    counts_before = prior_group_count(df, group_col="synthetic_group", dt_col="TransactionDT")
    recency_before = time_since_last_group_event(df, group_col="synthetic_group", dt_col="TransactionDT")

    new_row = pd.DataFrame({"synthetic_group": ["A"], "TransactionDT": [10_000], "TransactionAmt": [1.0]})
    df_extended = pd.concat([df, new_row], ignore_index=True)
    counts_after = prior_group_count(df_extended, group_col="synthetic_group", dt_col="TransactionDT")
    recency_after = time_since_last_group_event(df_extended, group_col="synthetic_group", dt_col="TransactionDT")

    for idx in df.index:
        assert counts_before.loc[idx] == counts_after.loc[idx]
        left, right = recency_before.loc[idx], recency_after.loc[idx]
        assert (pd.isna(left) and pd.isna(right)) or left == right


def test_adversarial_moving_a_row_further_into_the_future_does_not_change_earlier_rows():
    df = _synthetic()
    counts_before = prior_group_count(df, group_col="synthetic_group", dt_col="TransactionDT")

    df_shifted = df.copy()
    df_shifted.loc[5, "TransactionDT"] = 999_999  # was already the latest; push it further out
    counts_after = prior_group_count(df_shifted, group_col="synthetic_group", dt_col="TransactionDT")

    for idx in [0, 1, 2, 3]:
        assert counts_before.loc[idx] == counts_after.loc[idx]


def test_row_order_independence():
    df = _synthetic()
    df_shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    counts_orig = prior_group_count(df, group_col="synthetic_group", dt_col="TransactionDT")
    counts_shuf = prior_group_count(df_shuffled, group_col="synthetic_group", dt_col="TransactionDT")

    lookup_orig = {(g, dt): c for g, dt, c in zip(df["synthetic_group"], df["TransactionDT"], counts_orig)}
    for g, dt, c in zip(df_shuffled["synthetic_group"], df_shuffled["TransactionDT"], counts_shuf):
        assert lookup_orig[(g, dt)] == c


def test_no_target_dependency():
    # isFraud present (with arbitrary/garbage values) vs. absent entirely -> identical output.
    # None of the three functions accepts or reads a target column at all.
    df = _synthetic()
    df_with_target = df.copy()
    df_with_target["isFraud"] = [1, 0, 1, 0, 1, 0]

    a1 = prior_group_count(df, group_col="synthetic_group", dt_col="TransactionDT")
    a2 = prior_group_count(df_with_target, group_col="synthetic_group", dt_col="TransactionDT")
    pd.testing.assert_series_equal(a1, a2, check_names=False)

    b1 = prior_group_amount_stats(df, group_col="synthetic_group", amount_col="TransactionAmt", dt_col="TransactionDT")
    b2 = prior_group_amount_stats(df_with_target, group_col="synthetic_group", amount_col="TransactionAmt", dt_col="TransactionDT")
    pd.testing.assert_frame_equal(b1, b2)

    c1 = time_since_last_group_event(df, group_col="synthetic_group", dt_col="TransactionDT")
    c2 = time_since_last_group_event(df_with_target, group_col="synthetic_group", dt_col="TransactionDT")
    pd.testing.assert_series_equal(c1, c2, check_names=False)
