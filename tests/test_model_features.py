import numpy as np
import pandas as pd
import pytest

from sentinelpay.config import DetectionConfig
from sentinelpay.model_features import (
    B1_COLUMNS,
    B2_COLUMNS,
    F1_COLUMNS,
    F2_COLUMNS,
    IMPUTE_FIXED_VALUE,
    LADDER_FEATURE_COLUMNS,
    NON_FEATURE_COLUMNS,
    assemble_ladder_matrix,
    get_ladder_matrix,
)


def _detection_config(min_history_for_score=1):
    return DetectionConfig(
        min_history_for_score=min_history_for_score,
        window_size_events=5,
        modified_zscore_scale_constant=0.6745,
        modified_zscore_threshold=3.5,
        zero_mad_epsilon=1e-9,
    )


def _embargo_asymmetry_scenario():
    # t1 (train, dt=10, K1, amt=100, fraud=1)      -- must feed BOTH Phase D and Phase F for v2
    # e1 (embargo_1, dt=50, K1, amt=200, fraud=1)   -- must feed Phase D but NOT Phase F for v2
    # v1 (validation, dt=60, K1, amt=150, fraud=1)  -- must feed BOTH for v2
    # v2 (validation, dt=70, K1, amt=120, fraud=0)  -- row under test
    # A second, unrelated key K2 row so build_feature_frame/detection have >1 group to work with.
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "TransactionDT": [10, 50, 60, 70, 5],
            "TransactionAmt": [100.0, 200.0, 150.0, 120.0, 50.0],
            "isFraud": [1, 1, 1, 0, 0],
            "partition": ["train", "embargo_1", "validation", "validation", "train"],
            "_payment_group_key": ["K1", "K1", "K1", "K1", "K2"],
        }
    )


def test_phase_d_sees_embargo_1_but_phase_f_does_not():
    df = _embargo_asymmetry_scenario()
    assembled = assemble_ladder_matrix(df, identity_ids=[], detection_config=_detection_config())

    v2 = assembled[assembled["TransactionID"] == 4].iloc[0]

    # Phase D (non-target): window_size_events=5 easily covers all 3 prior
    # same-key events (t1, e1, v1) -- embargo_1 (e1) MUST be counted.
    assert v2["prior_count_in_window"] == 3

    # Phase F (target-derived): only t1 and v1 are eligible sources for a
    # validation row -- embargo_1's fraud label (e1) MUST NOT be counted,
    # even though it is chronologically prior and even though Phase D
    # (above) DID use it for this exact same row.
    assert v2["payment_proxy_prior_fraud_count"] == 2
    assert v2["payment_proxy_prior_event_count"] == 2


def test_holdout_rows_never_reach_assembly():
    df = _embargo_asymmetry_scenario()
    holdout_row = pd.DataFrame(
        {
            "TransactionID": [999],
            "TransactionDT": [1_000_000],
            "TransactionAmt": [10.0],
            "isFraud": [0],
            "partition": ["holdout"],
            "_payment_group_key": ["K1"],
        }
    )
    df_with_holdout = pd.concat([df, holdout_row], ignore_index=True)
    assembled = assemble_ladder_matrix(df_with_holdout, identity_ids=[], detection_config=_detection_config())

    assert 999 not in assembled["TransactionID"].values
    assert set(assembled["partition"].unique()) <= {"train", "validation"}
    # Also embargo_1 (present as a source row upstream) never appears as an
    # assembled OUTPUT row -- only train/validation are recipients.
    assert "embargo_1" not in assembled["partition"].unique()


def test_future_perturbation_does_not_change_earlier_rows():
    df = _embargo_asymmetry_scenario()
    before = assemble_ladder_matrix(df, identity_ids=[], detection_config=_detection_config())

    df_mutated = df.copy()
    last_idx = df_mutated["TransactionDT"].idxmax()  # v2, dt=70
    df_mutated.loc[last_idx, "isFraud"] = 1
    df_mutated.loc[last_idx, "TransactionAmt"] = 999_999.0
    after = assemble_ladder_matrix(df_mutated, identity_ids=[], detection_config=_detection_config())

    # e1 (id=2, embargo_1) is intentionally excluded: it is a history
    # SOURCE, never an assembled OUTPUT row (embargo_1 is never a
    # recipient) -- only recipient rows (train/validation) are checked here.
    earlier_ids = [1, 5, 3]  # t1 (train), K2's train row, v1 (validation) -- all strictly before v2's dt=70
    for tid in earlier_ids:
        row_before = before[before["TransactionID"] == tid].iloc[0]
        row_after = after[after["TransactionID"] == tid].iloc[0]
        for col in F2_COLUMNS + ["prior_count_in_window"]:
            b, a = row_before[col], row_after[col]
            assert (pd.isna(b) and pd.isna(a)) or b == a


def test_non_feature_columns_never_in_any_ladder_step():
    for step, cols in LADDER_FEATURE_COLUMNS.items():
        for banned in NON_FEATURE_COLUMNS:
            assert banned not in cols, f"{banned} leaked into {step}"


def test_ladder_is_strictly_incremental_and_ordered():
    assert LADDER_FEATURE_COLUMNS["B1"] == B1_COLUMNS
    assert LADDER_FEATURE_COLUMNS["B2"][: len(B1_COLUMNS)] == B1_COLUMNS
    assert LADDER_FEATURE_COLUMNS["B2"] == B2_COLUMNS
    assert LADDER_FEATURE_COLUMNS["F1"][: len(B2_COLUMNS)] == B2_COLUMNS
    assert LADDER_FEATURE_COLUMNS["F1"] == F1_COLUMNS
    assert LADDER_FEATURE_COLUMNS["F2"][: len(F1_COLUMNS)] == F1_COLUMNS
    assert LADDER_FEATURE_COLUMNS["F2"] == F2_COLUMNS
    # amt_log1p appears exactly once across the whole ladder (Phase D's own
    # copy is deliberately excluded -- see module docstring).
    assert F2_COLUMNS.count("amt_log1p") == 1


def test_schema_and_order_identical_between_train_and_validation():
    df = _embargo_asymmetry_scenario()
    assembled = assemble_ladder_matrix(df, identity_ids=[], detection_config=_detection_config())
    train_rows = assembled[assembled["partition"] == "train"]
    validation_rows = assembled[assembled["partition"] == "validation"]
    for step in LADDER_FEATURE_COLUMNS:
        X_train = get_ladder_matrix(train_rows, step)
        X_validation = get_ladder_matrix(validation_rows, step)
        assert list(X_train.columns) == list(X_validation.columns) == LADDER_FEATURE_COLUMNS[step]


def test_get_ladder_matrix_no_nan_and_float64():
    df = _embargo_asymmetry_scenario()
    assembled = assemble_ladder_matrix(df, identity_ids=[], detection_config=_detection_config())
    for step in LADDER_FEATURE_COLUMNS:
        X = get_ladder_matrix(assembled, step)
        assert X.dtypes.eq("float64").all()
        assert not X.isna().any().any()


def test_get_ladder_matrix_rejects_unknown_step():
    df = _embargo_asymmetry_scenario()
    assembled = assemble_ladder_matrix(df, identity_ids=[], detection_config=_detection_config())
    with pytest.raises(ValueError):
        get_ladder_matrix(assembled, "B0")
    with pytest.raises(ValueError):
        get_ladder_matrix(assembled, "nonsense")


def test_no_duplicate_columns_in_assembled_frame():
    df = _embargo_asymmetry_scenario()
    assembled = assemble_ladder_matrix(df, identity_ids=[], detection_config=_detection_config())
    assert not assembled.columns.duplicated().any()


def test_true_cold_start_row_imputed_with_fixed_constant_and_indicator_set():
    # A single-row, single-key, train-only frame: the very first row this
    # pool has ever seen -- both Phase D (prior_count_in_window==0) and
    # Phase F (global_cold_start) must hit their respective cold-start
    # cases, and both must be imputed to IMPUTE_FIXED_VALUE with their
    # companion indicator set, never left as an undefined/NaN value.
    df = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [100],
            "TransactionAmt": [50.0],
            "isFraud": [0],
            "partition": ["train"],
            "_payment_group_key": ["ONLY_KEY"],
        }
    )
    assembled = assemble_ladder_matrix(df, identity_ids=[], detection_config=_detection_config())
    row = assembled.iloc[0]

    assert row["prior_median"] == IMPUTE_FIXED_VALUE
    assert row["prior_mad"] == IMPUTE_FIXED_VALUE
    assert row["modified_zscore"] == IMPUTE_FIXED_VALUE
    assert row["flag_insufficient_history"] == 1

    assert row["payment_proxy_prior_fraud_rate_smoothed"] == IMPUTE_FIXED_VALUE
    assert row["global_cold_start"] == 1
    # global_prior_fraud_rate itself stays undefined -- never used as a
    # fallback value (see module docstring "Phase F imputation -- corrected").
    assert pd.isna(row["global_prior_fraud_rate"])


def test_requires_columns():
    df = _embargo_asymmetry_scenario().drop(columns=["isFraud"])
    with pytest.raises(ValueError):
        assemble_ladder_matrix(df, identity_ids=[], detection_config=_detection_config())
