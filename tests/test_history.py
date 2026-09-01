import pandas as pd
import pytest

from sentinelpay.data.history import (
    prior_group_count,
    prior_group_amount_stats,
    prior_group_windowed_robust_stats,
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


def _windowed_synthetic_no_ties():
    return pd.DataFrame(
        {
            "synthetic_group": ["A"] * 5,
            "TransactionDT": [100, 150, 200, 250, 300],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


def test_prior_group_windowed_robust_stats_basic_no_ties():
    # window_size_events=2, no ties: once >=2 prior rows exist the window is
    # exactly the 2 most recent.
    df = _windowed_synthetic_no_ties()
    out = prior_group_windowed_robust_stats(
        df, group_col="synthetic_group", amount_col="TransactionAmt", dt_col="TransactionDT", window_size_events=2
    )
    assert out.loc[0, "prior_count_in_window"] == 0  # dt=100, nothing earlier
    assert pd.isna(out.loc[0, "prior_median"])
    assert pd.isna(out.loc[0, "prior_mad"])

    assert out.loc[1, "prior_count_in_window"] == 1  # dt=150, only dt=100 available (< window_size_events)
    assert out.loc[1, "prior_median"] == pytest.approx(10.0)
    assert out.loc[1, "prior_mad"] == pytest.approx(0.0)

    assert out.loc[3, "prior_count_in_window"] == 2  # dt=250, window = dt=150,200 -> [20,30]
    assert out.loc[3, "prior_median"] == pytest.approx(25.0)
    assert out.loc[3, "prior_mad"] == pytest.approx(5.0)

    assert out.loc[4, "prior_count_in_window"] == 2  # dt=300, window = dt=200,250 -> [30,40]
    assert out.loc[4, "prior_median"] == pytest.approx(35.0)
    assert out.loc[4, "prior_mad"] == pytest.approx(5.0)


def test_prior_group_windowed_robust_stats_tied_boundary_bucket_can_exceed_window_size():
    # Hand-computed: window_size_events=3, but the boundary bucket that
    # crosses the threshold has 3 tied rows (dt=1), so the realized window
    # for the dt=3 row is 4 events (1 from dt=2 + 3 from dt=1), not exactly
    # 3 -- the whole-bucket-inclusion rule never splits a tied bucket to hit
    # the configured size exactly.
    df = pd.DataFrame(
        {
            "synthetic_group": ["A"] * 5,
            "TransactionDT": [1, 1, 1, 2, 3],
            "TransactionAmt": [10.0, 20.0, 30.0, 100.0, 999.0],
        }
    )
    out = prior_group_windowed_robust_stats(
        df, group_col="synthetic_group", amount_col="TransactionAmt", dt_col="TransactionDT", window_size_events=3
    )
    target_idx = df.index[df["TransactionDT"] == 3][0]
    assert out.loc[target_idx, "prior_count_in_window"] == 4  # 1 (dt=2) + 3 (tied dt=1) -- exceeds window_size_events=3
    assert out.loc[target_idx, "prior_median"] == pytest.approx(25.0)  # median([10,20,30,100])
    assert out.loc[target_idx, "prior_mad"] == pytest.approx(10.0)  # median(|x-25| for x in [10,20,30,100]) = median([15,5,5,75])


def test_prior_group_windowed_robust_stats_tied_timestamps_never_see_each_other():
    df = pd.DataFrame(
        {
            "synthetic_group": ["A", "A"],
            "TransactionDT": [500, 500],
            "TransactionAmt": [1000.0, 2000.0],
        }
    )
    out = prior_group_windowed_robust_stats(
        df, group_col="synthetic_group", amount_col="TransactionAmt", dt_col="TransactionDT", window_size_events=5
    )
    assert (out["prior_count_in_window"] == 0).all()
    assert out["prior_median"].isna().all()
    assert out["prior_mad"].isna().all()


def test_prior_group_windowed_robust_stats_row_order_independence():
    df = _windowed_synthetic_no_ties()
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)

    out_orig = prior_group_windowed_robust_stats(
        df, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2
    )
    out_shuf = prior_group_windowed_robust_stats(
        shuffled, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2
    )

    lookup = {
        dt: (med, mad, cnt)
        for dt, med, mad, cnt in zip(
            df["TransactionDT"], out_orig["prior_median"], out_orig["prior_mad"], out_orig["prior_count_in_window"]
        )
    }
    for dt, med, mad, cnt in zip(
        shuffled["TransactionDT"], out_shuf["prior_median"], out_shuf["prior_mad"], out_shuf["prior_count_in_window"]
    ):
        exp_med, exp_mad, exp_cnt = lookup[dt]
        if pd.isna(exp_med):
            assert pd.isna(med) and pd.isna(mad)
        else:
            assert med == pytest.approx(exp_med)
            assert mad == pytest.approx(exp_mad)
        assert cnt == exp_cnt


def test_prior_group_windowed_robust_stats_future_perturbation_does_not_change_earlier_rows():
    df = _windowed_synthetic_no_ties()
    before = prior_group_windowed_robust_stats(
        df, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2
    )

    df_mutated = df.copy()
    last_idx = df_mutated["TransactionDT"].idxmax()
    df_mutated.loc[last_idx, "TransactionAmt"] = 999_999.0
    df_mutated.loc[last_idx, "TransactionDT"] = 999_999
    after = prior_group_windowed_robust_stats(
        df_mutated, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2
    )

    for idx in [i for i in df.index if i != last_idx]:
        b_med, b_mad, b_cnt = (
            before.loc[idx, "prior_median"],
            before.loc[idx, "prior_mad"],
            before.loc[idx, "prior_count_in_window"],
        )
        a_med, a_mad, a_cnt = (
            after.loc[idx, "prior_median"],
            after.loc[idx, "prior_mad"],
            after.loc[idx, "prior_count_in_window"],
        )
        assert (pd.isna(b_med) and pd.isna(a_med)) or b_med == pytest.approx(a_med)
        assert (pd.isna(b_mad) and pd.isna(a_mad)) or b_mad == pytest.approx(a_mad)
        assert b_cnt == a_cnt


def test_prior_group_windowed_robust_stats_requires_columns():
    df = _windowed_synthetic_no_ties()
    with pytest.raises(ValueError):
        prior_group_windowed_robust_stats(df, "no_such_col", "TransactionAmt", "TransactionDT", window_size_events=2)
    with pytest.raises(ValueError):
        prior_group_windowed_robust_stats(df, "synthetic_group", "no_such_col", "TransactionDT", window_size_events=2)
    with pytest.raises(ValueError):
        prior_group_windowed_robust_stats(df, "synthetic_group", "TransactionAmt", "no_such_col", window_size_events=2)


def test_prior_group_windowed_robust_stats_requires_positive_window():
    df = _windowed_synthetic_no_ties()
    with pytest.raises(ValueError):
        prior_group_windowed_robust_stats(df, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=0)
    with pytest.raises(ValueError):
        prior_group_windowed_robust_stats(df, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=-1)


def test_prior_group_windowed_robust_stats_no_target_dependency():
    import inspect

    assert "isFraud" not in inspect.signature(prior_group_windowed_robust_stats).parameters
    assert "target" not in inspect.signature(prior_group_windowed_robust_stats).parameters

    df = _windowed_synthetic_no_ties()
    df_with_target = df.copy()
    df_with_target["isFraud"] = [1, 0, 1, 0, 1]
    df_shuffled_target = df.copy()
    df_shuffled_target["isFraud"] = [0, 1, 0, 1, 0]

    a = prior_group_windowed_robust_stats(df, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2)
    b = prior_group_windowed_robust_stats(
        df_with_target, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2
    )
    c = prior_group_windowed_robust_stats(
        df_shuffled_target, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2
    )
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)


def test_prior_group_windowed_robust_stats_non_default_and_duplicate_index_alignment():
    # Real callers (e.g. sentinelpay.detection via build_group_key's
    # dropna(...).copy()) hand this function a non-contiguous, non-default
    # index. The final merge is done via an explicit `_pos` positional
    # column specifically so alignment back to `df.index` is correct
    # regardless of index values -- this test exercises that directly,
    # rather than relying on the default RangeIndex every other test uses.
    df = _windowed_synthetic_no_ties()
    baseline = prior_group_windowed_robust_stats(
        df, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2
    )
    expected_by_dt = {
        dt: (med, mad, cnt)
        for dt, med, mad, cnt in zip(
            df["TransactionDT"], baseline["prior_median"], baseline["prior_mad"], baseline["prior_count_in_window"]
        )
    }

    def _assert_matches_expected(out, frame):
        assert list(out.index) == list(frame.index)  # output index is exactly the input index
        for idx in frame.index:
            dt = frame.loc[idx, "TransactionDT"]
            exp_med, exp_mad, exp_cnt = expected_by_dt[dt]
            row = out.loc[idx]
            if pd.isna(exp_med):
                assert pd.isna(row["prior_median"]) and pd.isna(row["prior_mad"])
            else:
                assert row["prior_median"] == pytest.approx(exp_med)
                assert row["prior_mad"] == pytest.approx(exp_mad)
            assert row["prior_count_in_window"] == exp_cnt

    # Non-default, non-contiguous index (in original row order).
    df_custom_index = df.copy()
    df_custom_index.index = [100, 205, 7, 999, 3]
    out_custom = prior_group_windowed_robust_stats(
        df_custom_index, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2
    )
    _assert_matches_expected(out_custom, df_custom_index)

    # Non-default index AND shuffled row order together.
    df_shuffled = df_custom_index.sample(frac=1.0, random_state=3)
    out_shuffled = prior_group_windowed_robust_stats(
        df_shuffled, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2
    )
    _assert_matches_expected(out_shuffled, df_shuffled)

    # Duplicate index labels.
    df_dup_index = df.copy()
    df_dup_index.index = [0, 0, 1, 1, 2]
    out_dup = prior_group_windowed_robust_stats(
        df_dup_index, "synthetic_group", "TransactionAmt", "TransactionDT", window_size_events=2
    )
    assert list(out_dup.index) == list(df_dup_index.index)
    for pos in range(len(df)):
        exp_med, exp_mad, exp_cnt = expected_by_dt[df["TransactionDT"].iloc[pos]]
        row = out_dup.iloc[pos]
        if pd.isna(exp_med):
            assert pd.isna(row["prior_median"]) and pd.isna(row["prior_mad"])
        else:
            assert row["prior_median"] == pytest.approx(exp_med)
            assert row["prior_mad"] == pytest.approx(exp_mad)
        assert row["prior_count_in_window"] == exp_cnt


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
