import math

import pandas as pd
import pytest

import sentinelpay.eda.run_phase_e2 as run_phase_e2_module
from sentinelpay.config import load_config
from sentinelpay.eda.run_phase_e2 import (
    COMPONENT_SIZE_BIN_LABELS,
    FANOUT_STRATUM_LABELS,
    _is_monotonic_nondecreasing,
    _summarize_fanout_stratified_conclusion,
    evaluate_fanout_stratified,
)


def test_evaluate_fanout_stratified_rejects_non_validation_rows():
    config = load_config()
    bad_df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "partition": ["train", "validation"],
            "merged_component_size_total": [1, 2],
            "endpoints_same_component": [True, False],
            "_device_to_payment_fanout": [10, 20],
        }
    )
    with pytest.raises(ValueError):
        evaluate_fanout_stratified(bad_df, config)


def _hand_crafted_scored_validation():
    # 2 rows per fan-out stratum (100 -> low, 500 -> mid_low, 1500 ->
    # mid_high, 3000 -> high), merged_component_size_total values chosen to
    # land in distinct fixed size buckets, isFraud values chosen to produce
    # hand-verifiable AUC (1.0, undefined/nan x2, 0.0) per stratum -- see the
    # accompanying PR/commit message for the full hand computation.
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6, 7, 8],
            "partition": ["validation"] * 8,
            "_device_to_payment_fanout": [100, 100, 500, 500, 1500, 1500, 3000, 3000],
            "merged_component_size_total": [5000, 18500, 9000, 13000, 16500, 18500, 5000, 18500],
            "endpoints_same_component": [False, True, False, True, False, True, False, True],
        }
    )


def _fake_isfraud():
    return pd.DataFrame({"TransactionID": [1, 2, 3, 4, 5, 6, 7, 8], "isFraud": [0, 1, 0, 0, 1, 1, 1, 0]})


def test_evaluate_fanout_stratified_hand_computed(monkeypatch):
    config = load_config()
    scored_validation = _hand_crafted_scored_validation()
    fake_isfraud = _fake_isfraud()

    def fake_load_transaction_columns(split, columns, config=None):
        return fake_isfraud[columns].copy()

    monkeypatch.setattr(run_phase_e2_module, "load_transaction_columns", fake_load_transaction_columns)

    result = evaluate_fanout_stratified(scored_validation, config)

    assert result["n_validation_rows"] == 8
    assert result["fanout_stratum_labels"] == FANOUT_STRATUM_LABELS
    assert result["component_size_bin_labels"] == COMPONENT_SIZE_BIN_LABELS

    by_stratum = {s["stratum"]: s for s in result["fanout_strata"]}

    low = by_stratum["low_fanout_lt_p25"]
    assert low["n_rows"] == 2
    assert low["fraud_rate"] == pytest.approx(0.5)
    assert low["roc_auc_merged_component_size_total_vs_isFraud"] == pytest.approx(1.0)
    low_buckets = {b["bucket"]: b for b in low["merged_component_size_total_fraud_rate_by_bucket"]}
    assert low_buckets["merged_size_lt_p25"]["n_rows"] == 1
    assert low_buckets["merged_size_lt_p25"]["fraud_rate"] == pytest.approx(0.0)
    assert low_buckets["merged_size_ge_p90"]["n_rows"] == 1
    assert low_buckets["merged_size_ge_p90"]["fraud_rate"] == pytest.approx(1.0)

    mid_low = by_stratum["mid_fanout_p25_to_p50"]
    assert mid_low["n_rows"] == 2
    assert mid_low["fraud_rate"] == pytest.approx(0.0)
    assert math.isnan(mid_low["roc_auc_merged_component_size_total_vs_isFraud"])  # n_pos == 0

    mid_high = by_stratum["mid_fanout_p50_to_p75"]
    assert mid_high["n_rows"] == 2
    assert mid_high["fraud_rate"] == pytest.approx(1.0)
    assert math.isnan(mid_high["roc_auc_merged_component_size_total_vs_isFraud"])  # n_neg == 0

    high = by_stratum["high_fanout_ge_p75"]
    assert high["n_rows"] == 2
    assert high["fraud_rate"] == pytest.approx(0.5)
    assert high["roc_auc_merged_component_size_total_vs_isFraud"] == pytest.approx(0.0)

    unstratified = result["unstratified"]
    assert unstratified["n_rows"] == 8
    assert unstratified["fraud_rate"] == pytest.approx(0.5)
    assert unstratified["roc_auc_merged_component_size_total_vs_isFraud"] == pytest.approx(0.65625)

    same_by_val = {row["endpoints_same_component"]: row for row in unstratified["endpoints_same_component_fraud_rate"]}
    assert same_by_val[True]["n_rows"] == 4
    assert same_by_val[False]["n_rows"] == 4

    assert "conclusion" in result
    assert isinstance(result["conclusion"], str) and len(result["conclusion"]) > 0


def test_evaluate_fanout_stratified_does_not_mutate_input(monkeypatch):
    config = load_config()
    scored_validation = _hand_crafted_scored_validation()
    original = scored_validation.copy()
    fake_isfraud = _fake_isfraud()

    def fake_load_transaction_columns(split, columns, config=None):
        return fake_isfraud[columns].copy()

    monkeypatch.setattr(run_phase_e2_module, "load_transaction_columns", fake_load_transaction_columns)
    evaluate_fanout_stratified(scored_validation, config)

    pd.testing.assert_frame_equal(scored_validation, original)


# ---------------------------------------------------------------------------
# Conclusion-synthesis helpers (pure functions, no isFraud loading)
# ---------------------------------------------------------------------------


def test_is_monotonic_nondecreasing():
    assert _is_monotonic_nondecreasing([0.0, 0.1, 0.5, 0.5, 0.9])
    assert not _is_monotonic_nondecreasing([0.5, 0.1, 0.9])
    assert not _is_monotonic_nondecreasing([0.5])  # fewer than 2 real values
    assert _is_monotonic_nondecreasing([float("nan"), 0.1, 0.5])  # NaNs dropped, remaining is monotonic


def test_summarize_conclusion_all_strata_confirm_signal():
    fanout_strata = [
        {
            "stratum": label,
            "n_rows": 10,
            "roc_auc_merged_component_size_total_vs_isFraud": 0.8,
            "merged_component_size_total_fraud_rate_by_bucket": [
                {"bucket": "b1", "fraud_rate": 0.1},
                {"bucket": "b2", "fraud_rate": 0.2},
                {"bucket": "b3", "fraud_rate": 0.3},
            ],
        }
        for label in FANOUT_STRATUM_LABELS
    ]
    text = _summarize_fanout_stratified_conclusion(fanout_strata)
    assert "CONCLUSION" in text
    assert "retains a fraud-rate gradient" in text


def test_summarize_conclusion_no_evidence_of_signal_beyond_fanout():
    fanout_strata = [
        {
            "stratum": label,
            "n_rows": 10,
            "roc_auc_merged_component_size_total_vs_isFraud": 0.3,
            "merged_component_size_total_fraud_rate_by_bucket": [
                {"bucket": "b1", "fraud_rate": 0.3},
                {"bucket": "b2", "fraud_rate": 0.1},
                {"bucket": "b3", "fraud_rate": 0.2},
            ],
        }
        for label in FANOUT_STRATUM_LABELS
    ]
    text = _summarize_fanout_stratified_conclusion(fanout_strata)
    assert "CONCLUSION" in text
    assert "does not clearly persist" in text


def test_summarize_conclusion_no_computable_auc_anywhere():
    fanout_strata = [
        {
            "stratum": label,
            "n_rows": 0,
            "roc_auc_merged_component_size_total_vs_isFraud": float("nan"),
            "merged_component_size_total_fraud_rate_by_bucket": [],
        }
        for label in FANOUT_STRATUM_LABELS
    ]
    text = _summarize_fanout_stratified_conclusion(fanout_strata)
    assert "No discriminatory-value conclusion can be drawn" in text
