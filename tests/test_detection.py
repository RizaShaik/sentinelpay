import inspect

import numpy as np
import pandas as pd
import pytest

from sentinelpay.config import DetectionConfig
from sentinelpay.detection import (
    FLAG_INSUFFICIENT_HISTORY,
    FLAG_SCORED_NORMAL,
    FLAG_SCORED_OUTLIER,
    FLAG_ZERO_MAD,
    compute_behavioral_change_score,
)


def _detection_config(**overrides) -> DetectionConfig:
    defaults = dict(
        min_history_for_score=2,
        window_size_events=3,
        modified_zscore_scale_constant=0.6745,
        modified_zscore_threshold=3.5,
        zero_mad_epsilon=1e-9,
    )
    defaults.update(overrides)
    return DetectionConfig(**defaults)


def _synthetic_tx(group, dt, amt):
    # Deliberately synthetic group key -- never a real proxy key.
    return pd.DataFrame({"_payment_group_key": group, "TransactionDT": dt, "TransactionAmt": amt})


def test_cold_start_insufficient_history_flag_and_nan_score():
    df = _synthetic_tx(["A"] * 5, [10, 20, 30, 40, 50], [10.0, 20.0, 30.0, 40.0, 50.0])
    cfg = _detection_config(min_history_for_score=2, window_size_events=3)
    out = compute_behavioral_change_score(df, cfg)

    assert out.loc[0, "flag"] == FLAG_INSUFFICIENT_HISTORY  # 0 prior events
    assert pd.isna(out.loc[0, "modified_zscore"])
    assert out.loc[1, "flag"] == FLAG_INSUFFICIENT_HISTORY  # 1 prior event, < min_history_for_score=2
    assert pd.isna(out.loc[1, "modified_zscore"])
    assert out.loc[2, "flag"] != FLAG_INSUFFICIENT_HISTORY  # 2 prior events -> meets the bar
    assert not pd.isna(out.loc[2, "modified_zscore"])


def test_zero_mad_flag_and_nan_score_when_prior_amounts_identical():
    df = _synthetic_tx(["A"] * 3, [1, 2, 3], [5.0, 5.0, 5.0])
    cfg = _detection_config(min_history_for_score=1, window_size_events=3)
    out = compute_behavioral_change_score(df, cfg)

    assert out.loc[2, "prior_mad"] == pytest.approx(0.0)
    assert out.loc[2, "flag"] == FLAG_ZERO_MAD
    assert pd.isna(out.loc[2, "modified_zscore"])


def test_modified_zscore_formula_matches_independently_computed_window():
    # window_size_events=3, min_history_for_score=3: the dt=4 row's window
    # is exactly the 3 prior rows (dt=1,2,3).
    df = _synthetic_tx(["A"] * 4, [1, 2, 3, 4], [10.0, 12.0, 8.0, 50.0])
    cfg = _detection_config(min_history_for_score=3, window_size_events=3)
    out = compute_behavioral_change_score(df, cfg)

    prior_log1p = np.log1p(np.array([10.0, 12.0, 8.0]))
    expected_median = np.median(prior_log1p)
    expected_mad = np.median(np.abs(prior_log1p - expected_median))
    target_log1p = np.log1p(50.0)
    expected_score = cfg.modified_zscore_scale_constant * (target_log1p - expected_median) / expected_mad

    assert out.loc[3, "prior_median"] == pytest.approx(expected_median)
    assert out.loc[3, "prior_mad"] == pytest.approx(expected_mad)
    assert out.loc[3, "modified_zscore"] == pytest.approx(expected_score)
    assert abs(expected_score) >= cfg.modified_zscore_threshold
    assert out.loc[3, "flag"] == FLAG_SCORED_OUTLIER


def test_threshold_boundary_is_inclusive():
    cfg = _detection_config(min_history_for_score=3, window_size_events=3)
    # Prior window log1p values [0, 1, 2] -> median=1.0, mad=median(|0-1|,|1-1|,|2-1|)=1.0.
    prior_log1p_values = np.array([0.0, 1.0, 2.0])
    prior_amts = np.expm1(prior_log1p_values)
    # Construct the target so modified_zscore lands EXACTLY on the threshold.
    target_log1p = 1.0 + 1.0 * cfg.modified_zscore_threshold / cfg.modified_zscore_scale_constant
    target_amt = np.expm1(target_log1p)

    df = _synthetic_tx(["A"] * 4, [1, 2, 3, 4], list(prior_amts) + [target_amt])
    out = compute_behavioral_change_score(df, cfg)

    assert out.loc[3, "modified_zscore"] == pytest.approx(cfg.modified_zscore_threshold)
    assert out.loc[3, "flag"] == FLAG_SCORED_OUTLIER  # inclusive >=


def test_flag_is_scored_normal_for_small_deviation():
    df = _synthetic_tx(["A"] * 4, [1, 2, 3, 4], [10.0, 12.0, 8.0, 10.5])
    cfg = _detection_config(min_history_for_score=3, window_size_events=3)
    out = compute_behavioral_change_score(df, cfg)
    assert out.loc[3, "flag"] == FLAG_SCORED_NORMAL
    assert abs(out.loc[3, "modified_zscore"]) < cfg.modified_zscore_threshold


def test_requires_columns():
    df = _synthetic_tx(["A"] * 3, [1, 2, 3], [1.0, 2.0, 3.0])
    cfg = _detection_config()
    with pytest.raises(ValueError):
        compute_behavioral_change_score(df, cfg, group_col="no_such_col")
    with pytest.raises(ValueError):
        compute_behavioral_change_score(df, cfg, amount_col="no_such_col")
    with pytest.raises(ValueError):
        compute_behavioral_change_score(df, cfg, dt_col="no_such_col")


def test_no_isfraud_dependency_behavioral_and_signature():
    assert "isFraud" not in inspect.signature(compute_behavioral_change_score).parameters
    assert "target" not in inspect.signature(compute_behavioral_change_score).parameters

    df = _synthetic_tx(["A"] * 5, [10, 20, 30, 40, 50], [10.0, 12.0, 8.0, 50.0, 9.0])
    cfg = _detection_config(min_history_for_score=2, window_size_events=3)

    df_with_target = df.copy()
    df_with_target["isFraud"] = [1, 0, 1, 0, 1]
    df_shuffled_target = df.copy()
    df_shuffled_target["isFraud"] = [0, 1, 0, 1, 0]

    a = compute_behavioral_change_score(df, cfg)
    b = compute_behavioral_change_score(df_with_target, cfg)
    c = compute_behavioral_change_score(df_shuffled_target, cfg)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)


def test_row_order_independence():
    df = _synthetic_tx(["A"] * 5, [10, 20, 30, 40, 50], [10.0, 12.0, 8.0, 50.0, 9.0])
    cfg = _detection_config(min_history_for_score=2, window_size_events=3)

    shuffled = df.sample(frac=1.0, random_state=5).reset_index(drop=True)

    out_orig = compute_behavioral_change_score(df, cfg)
    out_shuf = compute_behavioral_change_score(shuffled, cfg)

    lookup = {
        dt: (row["modified_zscore"], row["flag"])
        for dt, (_, row) in zip(df["TransactionDT"], out_orig.iterrows())
    }
    for dt, (_, row) in zip(shuffled["TransactionDT"], out_shuf.iterrows()):
        exp_score, exp_flag = lookup[dt]
        if pd.isna(exp_score):
            assert pd.isna(row["modified_zscore"])
        else:
            assert row["modified_zscore"] == pytest.approx(exp_score)
        assert row["flag"] == exp_flag


def test_holdout_rows_never_influence_scores_when_excluded_before_the_call():
    df = _synthetic_tx(["A"] * 5, [10, 20, 30, 40, 50], [10.0, 12.0, 8.0, 50.0, 9.0])
    df["partition"] = ["train", "train", "train", "validation", "validation"]
    cfg = _detection_config(min_history_for_score=2, window_size_events=3)

    result_without_holdout_ever_present = compute_behavioral_change_score(df, cfg)

    holdout_rows = _synthetic_tx(["A"] * 3, [1000, 1001, 1002], [1.0, 500.0, 2.0])
    holdout_rows["partition"] = "holdout"
    combined = pd.concat([df, holdout_rows], ignore_index=True)
    development_only = combined[combined["partition"] != "holdout"].copy()

    result_after_filtering_holdout_out = compute_behavioral_change_score(development_only, cfg)

    pd.testing.assert_frame_equal(result_without_holdout_ever_present, result_after_filtering_holdout_out)
