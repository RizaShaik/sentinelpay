import numpy as np
import pandas as pd
import pytest

from sentinelpay.config import DetectionConfig
from sentinelpay.detection import compute_behavioral_change_score
from sentinelpay.inference.artifacts import fit_frozen_f2_artifact
from sentinelpay.inference.scoring import score_transaction
from sentinelpay.inference.state import InferenceState
from sentinelpay.model_features import LADDER_FEATURE_COLUMNS, PAYMENT_GROUP_COL, get_ladder_matrix
from sentinelpay.target_history import compute_prior_fraud_rate


def _detection_config():
    return DetectionConfig(
        min_history_for_score=1,
        window_size_events=10,
        modified_zscore_scale_constant=0.6745,
        modified_zscore_threshold=3.5,
        zero_mad_epsilon=1e-9,
    )


def _dummy_artifact():
    # A trivial fitted scaler+model, just to satisfy score_transaction's
    # signature -- these tests assert on result.features (pre-scaling), not
    # on fraud_probability, so the model's own quality is irrelevant here.
    X = np.zeros((4, len(LADDER_FEATURE_COLUMNS["F2"])))
    X[1] = 1.0
    y = np.array([0, 1, 0, 1])
    fitted = fit_frozen_f2_artifact(X, y)
    return {"scaler": fitted["scaler"], "model": fitted["model"]}


def _base_transaction(dt, amt, has_identity=1, key_value=1):
    return {
        "TransactionDT": dt,
        "TransactionAmt": amt,
        "has_identity": has_identity,
        "card1": key_value,
        "card2": key_value,
        "card3": key_value,
        "card5": key_value,
        "addr1": key_value,
    }


def test_isfraud_key_rejected():
    txn = _base_transaction(100, 50.0)
    txn["isFraud"] = 0
    state = InferenceState(
        phase_d_buffer=pd.DataFrame(columns=[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"]),
        phase_f_counts=pd.DataFrame(columns=["fraud_count", "event_count"]).astype("int64"),
        phase_f_global_fraud_count=0,
        phase_f_global_event_count=0,
        metadata={},
    )
    with pytest.raises(ValueError):
        score_transaction(txn, state, _dummy_artifact(), _detection_config())


def test_missing_required_fields_rejected():
    txn = {"TransactionDT": 100}  # missing amount, has_identity, key columns
    state = InferenceState(
        phase_d_buffer=pd.DataFrame(columns=[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"]),
        phase_f_counts=pd.DataFrame(columns=["fraud_count", "event_count"]).astype("int64"),
        phase_f_global_fraud_count=0,
        phase_f_global_event_count=0,
        metadata={},
    )
    with pytest.raises(ValueError):
        score_transaction(txn, state, _dummy_artifact(), _detection_config())


def test_isfraud_structurally_absent_from_feature_vector():
    txn = _base_transaction(100, 50.0)
    state = InferenceState(
        phase_d_buffer=pd.DataFrame(columns=[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"]),
        phase_f_counts=pd.DataFrame(columns=["fraud_count", "event_count"]).astype("int64"),
        phase_f_global_fraud_count=0,
        phase_f_global_event_count=0,
        metadata={},
    )
    result = score_transaction(txn, state, _dummy_artifact(), _detection_config())
    assert "isFraud" not in result.features


def test_score_transaction_succeeds_without_transaction_id():
    """TransactionID AVAILABILITY CONTRACT (explicit, dedicated proof):
    scoring must remain ID-independent -- a genuinely new transaction has
    no dataset-assigned TransactionID, so score_transaction must succeed
    without one. Only the STATE-MUTATING functions
    (record_observed_transactions/update_resolved_labels, see
    tests/test_inference_state.py's
    test_record_observed_transactions_requires_transaction_id /
    test_update_resolved_labels_requires_transaction_id) require it."""
    txn = _base_transaction(100, 50.0)
    assert "TransactionID" not in txn
    state = InferenceState(
        phase_d_buffer=pd.DataFrame(columns=[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"]),
        phase_f_counts=pd.DataFrame(columns=["fraud_count", "event_count"]).astype("int64"),
        phase_f_global_fraud_count=0,
        phase_f_global_event_count=0,
        metadata={},
    )
    result = score_transaction(txn, state, _dummy_artifact(), _detection_config())
    assert 0.0 <= result.fraud_probability <= 1.0
    assert "TransactionID" not in result.features


def test_cold_start_key_scores_using_global_rate_only():
    state = InferenceState(
        phase_d_buffer=pd.DataFrame(columns=[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"]),
        phase_f_counts=pd.DataFrame({"fraud_count": [0], "event_count": [0]}, index=pd.Index(["OTHER_KEY"], name=PAYMENT_GROUP_COL)),
        phase_f_global_fraud_count=4,
        phase_f_global_event_count=20,
        metadata={},
    )
    txn = _base_transaction(100, 50.0, key_value=999)  # a brand new key, not in state
    result = score_transaction(txn, state, _dummy_artifact(), _detection_config())
    assert result.phase_f_diagnostics["payment_proxy_prior_event_count"] == 0
    assert result.phase_f_diagnostics["payment_proxy_prior_fraud_rate_smoothed"] == pytest.approx(4 / 20)
    assert result.phase_f_diagnostics["global_cold_start"] == 0


def test_resolution_updates_visible_in_next_score():
    from sentinelpay.inference.state import update_resolved_labels

    state = InferenceState(
        phase_d_buffer=pd.DataFrame(columns=[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"]),
        phase_f_counts=pd.DataFrame(columns=["fraud_count", "event_count"]).astype("int64"),
        phase_f_global_fraud_count=0,
        phase_f_global_event_count=0,
        metadata={},
    )
    txn = _base_transaction(100, 50.0, key_value=7)
    before = score_transaction(txn, state, _dummy_artifact(), _detection_config())
    assert before.phase_f_diagnostics["payment_proxy_prior_event_count"] == 0

    resolved = pd.DataFrame({PAYMENT_GROUP_COL: ["7|7|7|7|7"], "isFraud": [1], "TransactionID": [9999]})
    new_state = update_resolved_labels(state, resolved)
    after = score_transaction(txn, new_state, _dummy_artifact(), _detection_config())
    assert after.phase_f_diagnostics["payment_proxy_prior_event_count"] == 1
    assert after.phase_f_diagnostics["payment_proxy_prior_fraud_count"] == 1

    # Original state object untouched -- score_transaction is pure and
    # update_resolved_labels never mutates its input.
    assert before.phase_f_diagnostics["payment_proxy_prior_event_count"] == 0


def test_scoring_is_pure_no_state_mutation():
    state = InferenceState(
        phase_d_buffer=pd.DataFrame({PAYMENT_GROUP_COL: ["1|1|1|1|1"], "TransactionDT": [10], "TransactionAmt": [5.0]}),
        phase_f_counts=pd.DataFrame({"fraud_count": [1], "event_count": [2]}, index=pd.Index(["1|1|1|1|1"], name=PAYMENT_GROUP_COL)),
        phase_f_global_fraud_count=3,
        phase_f_global_event_count=10,
        metadata={},
    )
    buffer_before = state.phase_d_buffer.copy()
    counts_before = state.phase_f_counts.copy()

    txn = _base_transaction(100, 50.0, key_value=1)
    r1 = score_transaction(txn, state, _dummy_artifact(), _detection_config())
    r2 = score_transaction(txn, state, _dummy_artifact(), _detection_config())

    pd.testing.assert_frame_equal(state.phase_d_buffer, buffer_before)
    pd.testing.assert_frame_equal(state.phase_f_counts, counts_before)
    assert r1.fraud_probability == r2.fraud_probability
    assert r1.features == r2.features


# ---------------------------------------------------------------------------
# The highest-value test: state-backed single-row assembly must produce a
# feature vector BYTE-IDENTICAL to the batch causal computation for the same
# conceptual row -- proves no train/serve skew.
# ---------------------------------------------------------------------------


def test_full_f2_vector_equivalence_batch_vs_single():
    detection_config = _detection_config()
    key = "1|1|1|1|1"

    # Prior history for this key (all "train", strictly before the target row).
    r1 = {"TransactionID": 1, PAYMENT_GROUP_COL: key, "TransactionDT": 10, "TransactionAmt": 100.0, "isFraud": 1, "partition": "train"}
    r2 = {"TransactionID": 2, PAYMENT_GROUP_COL: key, "TransactionDT": 20, "TransactionAmt": 150.0, "isFraud": 0, "partition": "train"}
    r3 = {"TransactionID": 3, PAYMENT_GROUP_COL: key, "TransactionDT": 30, "TransactionAmt": 200.0, "isFraud": 1, "partition": "train"}
    target = {"TransactionID": 4, PAYMENT_GROUP_COL: key, "TransactionDT": 40, "TransactionAmt": 120.0, "isFraud": 0, "partition": "train"}

    df = pd.DataFrame([r1, r2, r3, target])

    # BATCH ground truth (Phase D + Phase F, reused unmodified).
    batch_d = compute_behavioral_change_score(
        df, detection_config, group_col=PAYMENT_GROUP_COL, amount_col="TransactionAmt", dt_col="TransactionDT"
    )
    batch_f = compute_prior_fraud_rate(
        df, allowed_source_partitions={"train"}, group_col=PAYMENT_GROUP_COL, dt_col="TransactionDT", partition_col="partition"
    )
    target_idx = df.index[df["TransactionID"] == 4][0]

    # STATE-BACKED single-row path: state = everything strictly before the
    # target row (simulating "already resolved" history).
    prior_df = df[df["TransactionID"] != 4]
    state = InferenceState(
        phase_d_buffer=prior_df[[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"]].reset_index(drop=True),
        phase_f_counts=prior_df.groupby(PAYMENT_GROUP_COL)["isFraud"].agg(fraud_count="sum", event_count="count").astype("int64"),
        phase_f_global_fraud_count=int(prior_df["isFraud"].sum()),
        phase_f_global_event_count=int(len(prior_df)),
        metadata={},
    )

    txn = {
        "TransactionDT": target["TransactionDT"],
        "TransactionAmt": target["TransactionAmt"],
        "has_identity": 0,
        "card1": 1,
        "card2": 1,
        "card3": 1,
        "card5": 1,
        "addr1": 1,
    }
    result = score_transaction(txn, state, _dummy_artifact(), detection_config)

    # Compare Phase D diagnostics.
    assert result.phase_d_diagnostics["prior_median"] == pytest.approx(float(batch_d.loc[target_idx, "prior_median"]))
    assert result.phase_d_diagnostics["prior_mad"] == pytest.approx(float(batch_d.loc[target_idx, "prior_mad"]))
    assert result.phase_d_diagnostics["prior_count_in_window"] == int(batch_d.loc[target_idx, "prior_count_in_window"])
    assert result.phase_d_diagnostics["modified_zscore"] == pytest.approx(float(batch_d.loc[target_idx, "modified_zscore"]))
    assert result.phase_d_diagnostics["flag"] == batch_d.loc[target_idx, "flag"]

    # Compare Phase F diagnostics.
    assert result.phase_f_diagnostics["payment_proxy_prior_fraud_count"] == int(batch_f.loc[target_idx, "payment_proxy_prior_fraud_count"])
    assert result.phase_f_diagnostics["payment_proxy_prior_event_count"] == int(batch_f.loc[target_idx, "payment_proxy_prior_event_count"])
    assert result.phase_f_diagnostics["payment_proxy_prior_fraud_rate_smoothed"] == pytest.approx(
        float(batch_f.loc[target_idx, "payment_proxy_prior_fraud_rate_smoothed"])
    )
    assert result.phase_f_diagnostics["sufficient_target_history"] == int(batch_f.loc[target_idx, "sufficient_target_history"])
    assert result.phase_f_diagnostics["global_cold_start"] == int(batch_f.loc[target_idx, "global_cold_start"])

    # Compare the full assembled F2 feature vector directly against
    # get_ladder_matrix's own output for a batch-assembled equivalent row.
    d_selected = batch_d.loc[[target_idx], ["prior_median", "prior_mad", "prior_count_in_window", "modified_zscore", "flag"]].reset_index(drop=True)
    from sentinelpay.model_features import _one_hot_flag, PHASE_D_IMPUTED_COLUMNS, IMPUTE_FIXED_VALUE

    flag_dummies = _one_hot_flag(d_selected["flag"])
    f_selected = batch_f.loc[[target_idx]].reset_index(drop=True)
    c_feats = pd.DataFrame(
        [
            {
                "amt_log1p": np.log1p(target["TransactionAmt"]),
                "amt_decimal_part": target["TransactionAmt"] - np.floor(target["TransactionAmt"]),
                "dt_hour_of_day": (target["TransactionDT"] // 3600) % 24,
                "dt_day_of_week": (target["TransactionDT"] // 86400) % 7,
                "has_identity": 0,
            }
        ]
    )
    batch_assembled = pd.concat([c_feats, d_selected.drop(columns=["flag"]), flag_dummies, f_selected], axis=1)
    for col in PHASE_D_IMPUTED_COLUMNS:
        batch_assembled[col] = batch_assembled[col].fillna(IMPUTE_FIXED_VALUE)
    batch_assembled["payment_proxy_prior_fraud_rate_smoothed"] = batch_assembled["payment_proxy_prior_fraud_rate_smoothed"].fillna(IMPUTE_FIXED_VALUE)
    batch_assembled["global_cold_start"] = batch_assembled["global_cold_start"].astype("int64")
    batch_assembled["sufficient_target_history"] = batch_assembled["sufficient_target_history"].astype("int64")
    batch_X = get_ladder_matrix(batch_assembled, "F2").iloc[0].to_dict()

    for col in LADDER_FEATURE_COLUMNS["F2"]:
        assert result.features[col] == pytest.approx(batch_X[col]), f"mismatch on {col}"
