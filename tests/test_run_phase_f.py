import math

import pandas as pd
import pytest

import sentinelpay.eda.run_phase_f as run_phase_f_module
from sentinelpay.config import load_config
from sentinelpay.data.split import DEVELOPMENT_PARTITIONS, load_split_config
from sentinelpay.eda.run_phase_f import build_development_frame, evaluate_target_derived_validation_only


def _synthetic_transaction_frame(config):
    """One small, hand-placed row per partition (via day boundaries from the
    real configs/split.yaml), plus holdout rows -- proves
    build_development_frame excludes holdout without touching real CSVs."""
    seconds_per_day = config.seconds_per_day
    days = {"train": 5, "embargo_1": 133, "validation": 150, "embargo_2": 165, "holdout": 175}
    rows = []
    for partition, day in days.items():
        n_rows = 3 if partition == "holdout" else 2
        for i in range(n_rows):
            rows.append(
                {
                    "TransactionID": len(rows) + 1,
                    "TransactionDT": day * seconds_per_day + i,
                    "isFraud": i % 2,
                    **{col: 1 for col in config.payment_proxy_key_columns},
                }
            )
    return pd.DataFrame(rows), 3


def test_build_development_frame_excludes_holdout_rows(monkeypatch):
    config = load_config()
    split_config = load_split_config()
    synthetic, n_holdout_in_fixture = _synthetic_transaction_frame(config)

    def fake_load_transaction_columns(split, columns, config=None):
        return synthetic[columns].copy()

    monkeypatch.setattr(run_phase_f_module, "load_transaction_columns", fake_load_transaction_columns)

    development, n_rows_total, n_holdout_excluded = build_development_frame(config, split_config)

    assert n_rows_total == len(synthetic)
    assert n_holdout_excluded == n_holdout_in_fixture
    assert len(development) == len(synthetic) - n_holdout_in_fixture
    assert set(development["partition"].unique()) == set(DEVELOPMENT_PARTITIONS)
    assert "holdout" not in development["partition"].unique()
    assert "isFraud" in development.columns


def test_evaluate_target_derived_validation_only_rejects_non_validation_rows():
    bad_df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "partition": ["train", "validation"],
            "isFraud": [0, 1],
            "payment_proxy_prior_fraud_rate_smoothed": [0.02, 0.05],
            "payment_proxy_prior_fraud_rate_raw": [0.02, float("nan")],
            "sufficient_target_history": [True, False],
            "global_cold_start": [False, False],
        }
    )
    with pytest.raises(ValueError):
        evaluate_target_derived_validation_only(bad_df, fraud_rate_bucket_edges=[0.01, 0.02, 0.03, 0.04], fraud_rate_bucket_labels=["a", "b", "c", "d", "e"])


def _hand_scored_validation():
    # 6 validation rows with controlled smoothed_rate/isFraud pairs chosen
    # for a hand-verifiable AUC and clean bucket placement against fixed
    # edges [0.01, 0.02, 0.03, 0.04].
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "partition": ["validation"] * 6,
            "isFraud": [0, 0, 1, 0, 1, 1],
            "payment_proxy_prior_fraud_rate_smoothed": [0.005, 0.015, 0.025, 0.035, 0.045, 0.045],
            "payment_proxy_prior_fraud_rate_raw": [float("nan"), 0.1, float("nan"), 0.2, 0.3, float("nan")],
            "sufficient_target_history": [False, True, False, True, True, False],
            "global_cold_start": [False, False, False, False, False, False],
        }
    )


def test_evaluate_target_derived_validation_only_hand_computed():
    df = _hand_scored_validation()
    edges = [0.01, 0.02, 0.03, 0.04]
    labels = ["rate_lt_p25", "rate_p25_to_p50", "rate_p50_to_p75", "rate_p75_to_p90", "rate_ge_p90"]

    result = evaluate_target_derived_validation_only(df, edges, labels)

    assert result["n_validation_rows"] == 6
    assert result["fraud_rate_overall"] == pytest.approx(3.0 / 6.0)
    assert result["fraud_rate_bucket_edges"] == edges
    assert result["fraud_rate_bucket_labels"] == labels

    by_bucket = {b["bucket"]: b for b in result["fraud_rate_by_smoothed_rate_bucket"]}
    assert by_bucket["rate_lt_p25"]["n_rows"] == 1  # 0.005
    assert by_bucket["rate_lt_p25"]["fraud_rate"] == pytest.approx(0.0)
    assert by_bucket["rate_p25_to_p50"]["n_rows"] == 1  # 0.015
    assert by_bucket["rate_p50_to_p75"]["n_rows"] == 1  # 0.025
    assert by_bucket["rate_p75_to_p90"]["n_rows"] == 1  # 0.035
    assert by_bucket["rate_ge_p90"]["n_rows"] == 2  # 0.045, 0.045
    assert by_bucket["rate_ge_p90"]["fraud_rate"] == pytest.approx(1.0)  # rows 5,6 both fraud=1

    # ROC-AUC of smoothed rate vs isFraud: scores strictly increasing with
    # index, isFraud = [0,0,1,0,1,1] -> should rank well above chance.
    auc = result["roc_auc_smoothed_rate_vs_isFraud"]
    assert 0.5 < auc <= 1.0

    # raw rate is defined for rows 2,4,5 only (index 1,3,4): raw=[0.1,0.2,0.3], isFraud=[0,0,1]
    assert result["roc_auc_raw_rate_n_rows"] == 3

    sufficient = {row["sufficient_target_history"]: row for row in result["fraud_rate_by_sufficient_target_history"]}
    assert sufficient[True]["n_rows"] == 3
    assert sufficient[False]["n_rows"] == 3

    assert result["sufficient_target_history_coverage"]["n_true"] == 3
    assert result["global_cold_start_coverage"]["n_true"] == 0


def test_evaluate_target_derived_validation_only_does_not_mutate_input():
    df = _hand_scored_validation()
    original = df.copy()
    evaluate_target_derived_validation_only(df, [0.01, 0.02, 0.03, 0.04], ["a", "b", "c", "d", "e"])
    pd.testing.assert_frame_equal(df, original)


def test_evaluate_target_derived_validation_only_empty_raw_subset():
    df = _hand_scored_validation()
    df["payment_proxy_prior_fraud_rate_raw"] = float("nan")  # every row cold-start on own key
    result = evaluate_target_derived_validation_only(df, [0.01, 0.02, 0.03, 0.04], ["a", "b", "c", "d", "e"])
    assert result["roc_auc_raw_rate_n_rows"] == 0
    assert math.isnan(result["roc_auc_raw_rate_vs_isFraud"])
