import numpy as np
import pandas as pd
import pytest

from sentinelpay.data.history import (
    prior_group_count,
    prior_group_amount_stats,
    prior_group_distinct_other_count,
    prior_group_windowed_distinct_other_count,
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


# ---------------------------------------------------------------------------
# prior_group_distinct_other_count / prior_group_windowed_distinct_other_count
# (Phase E.1 link-sufficiency primitives)
# ---------------------------------------------------------------------------


def _synthetic_with_other():
    # group A sorted by dt: 100(X), 150(Y), 200(X)/200(Z) tie, 300(Y).
    # group B: 100(P), only row.
    return pd.DataFrame(
        {
            "synthetic_group": ["A", "A", "A", "A", "B", "A"],
            "TransactionDT": [100, 150, 200, 200, 100, 300],
            "synthetic_other": ["X", "Y", "X", "Z", "P", "Y"],
        }
    )


def test_prior_group_distinct_other_count_basic():
    df = _synthetic_with_other()
    out = prior_group_distinct_other_count(df, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")
    assert out.loc[0] == 0  # dt=100, nothing earlier
    assert out.loc[1] == 1  # dt=150, prior partners {X}
    assert out.loc[2] == 2  # dt=200, prior partners {X,Y} -- does not include idx3's own-bucket Z
    assert out.loc[3] == 2  # dt=200 (tie with idx 2) -- same prior set, does not count idx2's X a second time nor see it as new
    assert out.loc[5] == 3  # dt=300, prior partners {X,Y,Z} (X,Y,X,Z all strictly before)
    assert out.loc[4] == 0  # group B, only row


def test_prior_group_distinct_other_count_requires_columns():
    df = _synthetic_with_other()
    with pytest.raises(ValueError):
        prior_group_distinct_other_count(df, group_col="no_such_col", other_col="synthetic_other", dt_col="TransactionDT")
    with pytest.raises(ValueError):
        prior_group_distinct_other_count(df, group_col="synthetic_group", other_col="no_such_col", dt_col="TransactionDT")
    with pytest.raises(ValueError):
        prior_group_distinct_other_count(df, group_col="synthetic_group", other_col="synthetic_other", dt_col="no_such_col")


def test_prior_group_distinct_other_count_tied_timestamps_never_see_each_other():
    df = pd.DataFrame(
        {
            "synthetic_group": ["A", "A"],
            "TransactionDT": [500, 500],
            "synthetic_other": ["X", "Y"],
        }
    )
    out = prior_group_distinct_other_count(df, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")
    assert (out == 0).all()


def test_prior_group_distinct_other_count_null_other_contributes_no_partner():
    df = pd.DataFrame(
        {
            "synthetic_group": ["A", "A", "A"],
            "TransactionDT": [100, 200, 300],
            "synthetic_other": ["X", None, "Y"],
        }
    )
    out = prior_group_distinct_other_count(df, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")
    assert out.loc[0] == 0
    assert out.loc[1] == 1  # {X} -- the null row itself contributes nothing when it's the "other" side
    assert out.loc[2] == 1  # still just {X}; the null-other row at dt=200 added no partner


def test_prior_group_distinct_other_count_adversarial_future_perturbation():
    df = _synthetic_with_other()
    before = prior_group_distinct_other_count(df, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")

    df_mutated = df.copy()
    df_mutated.loc[5, "synthetic_other"] = "BRAND_NEW_VALUE"  # idx 5 is the chronologically-last group-A row
    after_mutation = prior_group_distinct_other_count(df_mutated, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")
    for idx in [0, 1, 2, 3]:
        assert before.loc[idx] == after_mutation.loc[idx]

    new_row = pd.DataFrame({"synthetic_group": ["A"], "TransactionDT": [10_000], "synthetic_other": ["ANOTHER_NEW_VALUE"]})
    df_extended = pd.concat([df, new_row], ignore_index=True)
    after_append = prior_group_distinct_other_count(df_extended, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")
    for idx in df.index:
        assert before.loc[idx] == after_append.loc[idx]


def test_prior_group_distinct_other_count_row_order_independence():
    df = _synthetic_with_other()
    shuffled = df.sample(frac=1.0, random_state=13).reset_index(drop=True)

    counts_orig = prior_group_distinct_other_count(df, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")
    counts_shuf = prior_group_distinct_other_count(shuffled, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")

    lookup = {(g, dt): c for g, dt, c in zip(df["synthetic_group"], df["TransactionDT"], counts_orig)}
    for g, dt, c in zip(shuffled["synthetic_group"], shuffled["TransactionDT"], counts_shuf):
        assert lookup[(g, dt)] == c


def test_prior_group_distinct_other_count_no_target_dependency():
    import inspect

    assert "isFraud" not in inspect.signature(prior_group_distinct_other_count).parameters
    assert "target" not in inspect.signature(prior_group_distinct_other_count).parameters

    df = _synthetic_with_other()
    df_with_target = df.copy()
    df_with_target["isFraud"] = [1, 0, 1, 0, 1, 0]
    df_shuffled_target = df.copy()
    df_shuffled_target["isFraud"] = [0, 1, 0, 1, 0, 1]

    a = prior_group_distinct_other_count(df, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")
    b = prior_group_distinct_other_count(df_with_target, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")
    c = prior_group_distinct_other_count(df_shuffled_target, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")
    pd.testing.assert_series_equal(a, b)
    pd.testing.assert_series_equal(a, c)


def _brute_force_prior_distinct_other_count(df, group_col, other_col, dt_col):
    out = []
    for i in df.index:
        gi, dti = df.loc[i, group_col], df.loc[i, dt_col]
        prior = df[(df[group_col] == gi) & (df[dt_col] < dti) & df[other_col].notna()]
        out.append(prior[other_col].nunique())
    return pd.Series(out, index=df.index)


def test_prior_group_distinct_other_count_matches_brute_force_on_random_data():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "synthetic_group": rng.choice(["A", "B", "C"], size=n),
            "TransactionDT": rng.integers(0, 15, size=n),  # small range forces many ties
            "synthetic_other": rng.choice(["p", "q", "r", "s", None], size=n),
        }
    )
    fast = prior_group_distinct_other_count(df, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT")
    slow = _brute_force_prior_distinct_other_count(df, "synthetic_group", "synthetic_other", "TransactionDT")
    pd.testing.assert_series_equal(fast, slow.astype("int64"), check_names=False)


def _windowed_synthetic_with_other_no_ties():
    return pd.DataFrame(
        {
            "synthetic_group": ["A"] * 5,
            "TransactionDT": [100, 150, 200, 250, 300],
            "synthetic_other": ["X", "Y", "X", "Z", "Y"],
        }
    )


def test_prior_group_windowed_distinct_other_count_basic_no_ties():
    df = _windowed_synthetic_with_other_no_ties()
    out = prior_group_windowed_distinct_other_count(
        df, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT", window_size_events=2
    )
    assert out.loc[0, "prior_count_in_window"] == 0
    assert out.loc[0, "prior_distinct_other_count_in_window"] == 0

    assert out.loc[1, "prior_count_in_window"] == 1  # only dt=100 (X) available
    assert out.loc[1, "prior_distinct_other_count_in_window"] == 1

    assert out.loc[2, "prior_count_in_window"] == 2  # window = dt=100(X), dt=150(Y)
    assert out.loc[2, "prior_distinct_other_count_in_window"] == 2

    assert out.loc[3, "prior_count_in_window"] == 2  # window = dt=150(Y), dt=200(X)
    assert out.loc[3, "prior_distinct_other_count_in_window"] == 2

    assert out.loc[4, "prior_count_in_window"] == 2  # window = dt=200(X), dt=250(Z)
    assert out.loc[4, "prior_distinct_other_count_in_window"] == 2


def test_prior_group_windowed_distinct_other_count_tied_boundary_bucket_distinguishes_raw_from_distinct():
    # window_size_events=3; boundary bucket at dt=1 has 3 tied rows whose
    # other_col values are P, P, Q (a duplicate WITHIN the bucket itself --
    # raw row count 3, distinct count only 2); dt=2 bucket has a single row
    # valued "T". Raw prior_count_in_window must be 4 (whole boundary bucket
    # included, never split, counting rows not distinct values) while the
    # DISTINCT count must be 3 ({P, Q, T}) -- this specifically exercises
    # _bucket_count being a raw row count, not accidentally the size of the
    # deduped value set (a bug that would coincidentally pass if every
    # bucket's own values happened to already be distinct).
    df = pd.DataFrame(
        {
            "synthetic_group": ["A"] * 5,
            "TransactionDT": [1, 1, 1, 2, 3],
            "synthetic_other": ["P", "P", "Q", "T", "Z"],
        }
    )
    out = prior_group_windowed_distinct_other_count(
        df, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT", window_size_events=3
    )
    target_idx = df.index[df["TransactionDT"] == 3][0]
    assert out.loc[target_idx, "prior_count_in_window"] == 4
    assert out.loc[target_idx, "prior_distinct_other_count_in_window"] == 3


def test_prior_group_windowed_distinct_other_count_tied_timestamps_never_see_each_other():
    df = pd.DataFrame(
        {
            "synthetic_group": ["A", "A"],
            "TransactionDT": [500, 500],
            "synthetic_other": ["X", "Y"],
        }
    )
    out = prior_group_windowed_distinct_other_count(
        df, group_col="synthetic_group", other_col="synthetic_other", dt_col="TransactionDT", window_size_events=5
    )
    assert (out["prior_count_in_window"] == 0).all()
    assert (out["prior_distinct_other_count_in_window"] == 0).all()


def test_prior_group_windowed_distinct_other_count_row_order_independence():
    df = _windowed_synthetic_with_other_no_ties()
    shuffled = df.sample(frac=1.0, random_state=17).reset_index(drop=True)

    out_orig = prior_group_windowed_distinct_other_count(
        df, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=2
    )
    out_shuf = prior_group_windowed_distinct_other_count(
        shuffled, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=2
    )
    lookup = {
        dt: (dcnt, rcnt)
        for dt, dcnt, rcnt in zip(
            df["TransactionDT"], out_orig["prior_distinct_other_count_in_window"], out_orig["prior_count_in_window"]
        )
    }
    for dt, dcnt, rcnt in zip(
        shuffled["TransactionDT"], out_shuf["prior_distinct_other_count_in_window"], out_shuf["prior_count_in_window"]
    ):
        exp_dcnt, exp_rcnt = lookup[dt]
        assert dcnt == exp_dcnt
        assert rcnt == exp_rcnt


def test_prior_group_windowed_distinct_other_count_future_perturbation_does_not_change_earlier_rows():
    df = _windowed_synthetic_with_other_no_ties()
    before = prior_group_windowed_distinct_other_count(
        df, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=2
    )

    df_mutated = df.copy()
    last_idx = df_mutated["TransactionDT"].idxmax()
    df_mutated.loc[last_idx, "synthetic_other"] = "BRAND_NEW_VALUE"
    df_mutated.loc[last_idx, "TransactionDT"] = 999_999
    after = prior_group_windowed_distinct_other_count(
        df_mutated, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=2
    )

    for idx in [i for i in df.index if i != last_idx]:
        assert before.loc[idx, "prior_distinct_other_count_in_window"] == after.loc[idx, "prior_distinct_other_count_in_window"]
        assert before.loc[idx, "prior_count_in_window"] == after.loc[idx, "prior_count_in_window"]


def test_prior_group_windowed_distinct_other_count_requires_columns():
    df = _windowed_synthetic_with_other_no_ties()
    with pytest.raises(ValueError):
        prior_group_windowed_distinct_other_count(df, "no_such_col", "synthetic_other", "TransactionDT", window_size_events=2)
    with pytest.raises(ValueError):
        prior_group_windowed_distinct_other_count(df, "synthetic_group", "no_such_col", "TransactionDT", window_size_events=2)
    with pytest.raises(ValueError):
        prior_group_windowed_distinct_other_count(df, "synthetic_group", "synthetic_other", "no_such_col", window_size_events=2)


def test_prior_group_windowed_distinct_other_count_requires_positive_window():
    df = _windowed_synthetic_with_other_no_ties()
    with pytest.raises(ValueError):
        prior_group_windowed_distinct_other_count(df, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=0)
    with pytest.raises(ValueError):
        prior_group_windowed_distinct_other_count(df, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=-1)


def test_prior_group_windowed_distinct_other_count_no_target_dependency():
    import inspect

    assert "isFraud" not in inspect.signature(prior_group_windowed_distinct_other_count).parameters
    assert "target" not in inspect.signature(prior_group_windowed_distinct_other_count).parameters

    df = _windowed_synthetic_with_other_no_ties()
    df_with_target = df.copy()
    df_with_target["isFraud"] = [1, 0, 1, 0, 1]
    df_shuffled_target = df.copy()
    df_shuffled_target["isFraud"] = [0, 1, 0, 1, 0]

    a = prior_group_windowed_distinct_other_count(df, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=2)
    b = prior_group_windowed_distinct_other_count(df_with_target, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=2)
    c = prior_group_windowed_distinct_other_count(df_shuffled_target, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=2)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)


def test_prior_group_windowed_distinct_other_count_non_default_and_duplicate_index_alignment():
    df = _windowed_synthetic_with_other_no_ties()
    baseline = prior_group_windowed_distinct_other_count(
        df, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=2
    )
    expected_by_dt = {
        dt: (dcnt, rcnt)
        for dt, dcnt, rcnt in zip(
            df["TransactionDT"], baseline["prior_distinct_other_count_in_window"], baseline["prior_count_in_window"]
        )
    }

    df_custom_index = df.copy()
    df_custom_index.index = [100, 205, 7, 999, 3]
    out_custom = prior_group_windowed_distinct_other_count(
        df_custom_index, "synthetic_group", "synthetic_other", "TransactionDT", window_size_events=2
    )
    assert list(out_custom.index) == list(df_custom_index.index)
    for idx in df_custom_index.index:
        dt = df_custom_index.loc[idx, "TransactionDT"]
        exp_dcnt, exp_rcnt = expected_by_dt[dt]
        assert out_custom.loc[idx, "prior_distinct_other_count_in_window"] == exp_dcnt
        assert out_custom.loc[idx, "prior_count_in_window"] == exp_rcnt
