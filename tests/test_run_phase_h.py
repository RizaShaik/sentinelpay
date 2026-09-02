import pandas as pd
import pytest

from sentinelpay.config import DetectionConfig, load_config
from sentinelpay.eda.run_phase_h import (
    HOLDOUT_SOURCE_PARTITIONS,
    LADDER_STEPS_EVALUATED,
    assemble_train_and_holdout_matrices,
    assert_holdout_not_yet_evaluated,
    build_full_frame,
    build_holdout_eligible_pool,
    compute_phase_f_holdout_features,
)
from sentinelpay.model_features import LADDER_FEATURE_COLUMNS, get_ladder_matrix


def _detection_config():
    return DetectionConfig(
        min_history_for_score=1,
        window_size_events=10,
        modified_zscore_scale_constant=0.6745,
        modified_zscore_threshold=3.5,
        zero_mad_epsilon=1e-9,
    )


def _holdout_causal_scenario():
    # Group K1, chronological order: t1(train) -> e1(embargo_1) -> v1(validation)
    # -> e2(embargo_2) -> h1(holdout) -> h_tie_a/h_tie_b(holdout, TIED) -> h3(holdout,
    # row under test) -> h_future(holdout, strictly later than h3).
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "TransactionDT": [10, 50, 60, 210, 300, 310, 310, 320, 400],
            "TransactionAmt": [100.0, 200.0, 150.0, 80.0, 90.0, 95.0, 85.0, 60.0, 50.0],
            "isFraud": [1, 1, 1, 1, 1, 1, 1, 0, 1],
            "partition": ["train", "embargo_1", "validation", "embargo_2", "holdout", "holdout", "holdout", "holdout", "holdout"],
            "_payment_group_key": ["K1"] * 9,
        }
    )


def _assembled(df):
    return assemble_train_and_holdout_matrices(df, identity_ids=[], detection_config=_detection_config())


# ---------------------------------------------------------------------------
# 1-5: the core causal-eligibility properties (Option A, online continuation)
# ---------------------------------------------------------------------------


def test_full_causal_scenario_all_properties_at_once():
    df = _holdout_causal_scenario()
    train_assembled, holdout_assembled = _assembled(df)

    def row(tid):
        return holdout_assembled[holdout_assembled["TransactionID"] == tid].iloc[0]

    # h1 (id=5, dt=300): prior = {t1, v1} only -- e1 (embargo_1) excluded.
    h1 = row(5)
    assert h1["payment_proxy_prior_fraud_count"] == 2
    assert h1["payment_proxy_prior_event_count"] == 2

    # h_tie_a and h_tie_b (id=6,7, both dt=310): prior = {t1, v1, h1} = 3 --
    # neither sees the other (same-TransactionDT tie), and e2 (embargo_2) is
    # excluded.
    h_tie_a = row(6)
    h_tie_b = row(7)
    assert h_tie_a["payment_proxy_prior_fraud_count"] == 3
    assert h_tie_a["payment_proxy_prior_event_count"] == 3
    assert h_tie_b["payment_proxy_prior_fraud_count"] == 3
    assert h_tie_b["payment_proxy_prior_event_count"] == 3

    # h3 (id=8, dt=320): prior = {t1, v1, h1, h_tie_a, h_tie_b} = 5 -- earlier
    # HOLDOUT rows (h1, both ties) DO contribute (property 1); e1/e2 excluded
    # (property 4); train/validation DO contribute (property 5);
    # h_future (dt=400, strictly later) must NOT contribute (property 2).
    h3 = row(8)
    assert h3["payment_proxy_prior_fraud_count"] == 5
    assert h3["payment_proxy_prior_event_count"] == 5

    # Phase D (non-target, unrestricted) vs Phase F (target-derived,
    # embargo-excluded) asymmetry for holdout specifically (property 6):
    # h3's Phase D window sees ALL 7 strictly-prior events (t1,e1,v1,e2,h1,
    # h_tie_a,h_tie_b), while its Phase F count is only 5 (embargo_1/2
    # excluded).
    assert h3["prior_count_in_window"] == 7
    assert h3["payment_proxy_prior_event_count"] == 5

    # h3's own isFraud (0) is irrelevant here, but confirm no row ever counts
    # itself: h1's own fraud=1 is not in its own count (property 7, general
    # form -- see the dedicated isolated-row test below for the sharpest case).
    assert h1["payment_proxy_prior_fraud_count"] < 3  # would be 3 if h1 counted itself


def test_later_holdout_fraud_does_not_affect_earlier_row_future_perturbation():
    df = _holdout_causal_scenario()
    _, holdout_before = _assembled(df)
    h3_before = holdout_before[holdout_before["TransactionID"] == 8].iloc[0]

    df_mutated = df.copy()
    last_idx = df_mutated["TransactionDT"].idxmax()  # h_future, id=9
    df_mutated.loc[last_idx, "isFraud"] = 0
    df_mutated.loc[last_idx, "TransactionAmt"] = 999_999.0
    _, holdout_after = _assembled(df_mutated)
    h3_after = holdout_after[holdout_after["TransactionID"] == 8].iloc[0]

    for col in ["payment_proxy_prior_fraud_count", "payment_proxy_prior_event_count", "payment_proxy_prior_fraud_rate_smoothed"]:
        assert h3_before[col] == h3_after[col]


# ---------------------------------------------------------------------------
# 7: a holdout row's own isFraud never contributes to its own feature vector
# (sharpest case: an isolated row with NO other history at all)
# ---------------------------------------------------------------------------


def test_isolated_holdout_row_own_label_never_self_contributes():
    df = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [500],
            "TransactionAmt": [42.0],
            "isFraud": [1],
            "partition": ["holdout"],
            "_payment_group_key": ["LONELY_KEY"],
        }
    )
    _, holdout_assembled = _assembled(df)
    row = holdout_assembled.iloc[0]
    assert row["payment_proxy_prior_fraud_count"] == 0
    assert row["payment_proxy_prior_event_count"] == 0
    assert row["global_cold_start"] == 1


# ---------------------------------------------------------------------------
# 8: y_train never contains a holdout label; no feature column is isFraud
# ---------------------------------------------------------------------------


def test_train_holdout_row_separation_and_no_isfraud_feature_column():
    df = _holdout_causal_scenario()
    train_assembled, holdout_assembled = _assembled(df)

    assert set(train_assembled["partition"].unique()) == {"train"}
    assert set(holdout_assembled["partition"].unique()) == {"holdout"}
    # No row from one recipient ever appears in the other's output.
    assert set(train_assembled["TransactionID"]).isdisjoint(set(holdout_assembled["TransactionID"]))

    for step in LADDER_STEPS_EVALUATED:
        assert "isFraud" not in LADDER_FEATURE_COLUMNS[step]


# ---------------------------------------------------------------------------
# 9: X_holdout columns match the frozen B2/F1/F2 schema exactly, same order
# as X_train
# ---------------------------------------------------------------------------


def test_schema_identical_between_train_and_holdout():
    df = _holdout_causal_scenario()
    train_assembled, holdout_assembled = _assembled(df)
    for step in LADDER_STEPS_EVALUATED:
        X_train = get_ladder_matrix(train_assembled, step)
        X_holdout = get_ladder_matrix(holdout_assembled, step)
        assert list(X_train.columns) == list(X_holdout.columns) == LADDER_FEATURE_COLUMNS[step]
        assert X_train.dtypes.eq("float64").all()
        assert X_holdout.dtypes.eq("float64").all()
        assert not X_holdout.isna().any().any()


# ---------------------------------------------------------------------------
# 10: one-shot guard -- tmp_path only, never the real path
# ---------------------------------------------------------------------------


def test_one_shot_guard_raises_when_results_file_exists(tmp_path):
    results_path = tmp_path / "phase_h_results.json"
    results_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        assert_holdout_not_yet_evaluated(results_path)


def test_one_shot_guard_does_not_raise_when_absent(tmp_path):
    results_path = tmp_path / "phase_h_results.json"
    assert not results_path.exists()
    assert_holdout_not_yet_evaluated(results_path)  # must not raise


# ---------------------------------------------------------------------------
# 11: row-count/partition-composition cross-check against split.yaml's real
# day ranges, before any modeling -- and confirms holdout is NOT excluded
# (unlike every prior phase's build_development_frame)
# ---------------------------------------------------------------------------


def test_build_full_frame_includes_holdout_using_real_day_boundaries(monkeypatch):
    import sentinelpay.eda.run_phase_h as run_phase_h_module
    from sentinelpay.data.split import load_split_config

    config = load_config()
    split_config = load_split_config()
    seconds_per_day = config.seconds_per_day
    days = {"train": 5, "embargo_1": 133, "validation": 150, "embargo_2": 165, "holdout": 175}
    rows = []
    for partition, day in days.items():
        for i in range(2):
            rows.append(
                {
                    "TransactionID": len(rows) + 1,
                    "TransactionDT": day * seconds_per_day + i,
                    "TransactionAmt": 10.0 + i,
                    "isFraud": i % 2,
                    **{col: 1 for col in config.payment_proxy_key_columns},
                }
            )
    synthetic = pd.DataFrame(rows)

    def fake_load_transaction_columns(split, columns, config=None):
        return synthetic[columns].copy()

    monkeypatch.setattr(run_phase_h_module, "load_transaction_columns", fake_load_transaction_columns)

    full, n_rows_total = build_full_frame(config, split_config)
    assert n_rows_total == len(synthetic)
    assert len(full) == len(synthetic)  # NOTHING excluded -- holdout included
    counts = full["partition"].value_counts()
    for partition in days:
        assert counts.get(partition, 0) == 2
    assert "holdout" in full["partition"].unique()


# ---------------------------------------------------------------------------
# Supporting structural tests
# ---------------------------------------------------------------------------


def test_holdout_source_partitions_excludes_embargoes():
    assert HOLDOUT_SOURCE_PARTITIONS == {"train", "validation", "holdout"}
    assert "embargo_1" not in HOLDOUT_SOURCE_PARTITIONS
    assert "embargo_2" not in HOLDOUT_SOURCE_PARTITIONS


def test_build_holdout_eligible_pool_excludes_embargoes():
    df = _holdout_causal_scenario()
    pool = build_holdout_eligible_pool(df)
    assert set(pool["partition"].unique()) <= {"train", "validation", "holdout"}
    assert "embargo_1" not in pool["partition"].unique()
    assert "embargo_2" not in pool["partition"].unique()


def test_build_holdout_eligible_pool_requires_partition_column():
    df = _holdout_causal_scenario().drop(columns=["partition"])
    with pytest.raises(ValueError):
        build_holdout_eligible_pool(df)


def test_compute_phase_f_holdout_features_returns_holdout_rows_only():
    df = _holdout_causal_scenario()
    out = compute_phase_f_holdout_features(df)
    original_holdout_ids = set(df[df["partition"] == "holdout"]["TransactionID"])
    matched_ids = set(df.loc[out.index, "TransactionID"])
    assert matched_ids == original_holdout_ids


def test_assemble_requires_columns():
    df = _holdout_causal_scenario().drop(columns=["isFraud"])
    with pytest.raises(ValueError):
        assemble_train_and_holdout_matrices(df, identity_ids=[], detection_config=_detection_config())
