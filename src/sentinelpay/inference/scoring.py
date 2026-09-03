"""Phase I: state-backed scoring for a new transaction whose own `isFraud`
is unavailable.

Reuses Phase C (`sentinelpay.features`), Phase D
(`sentinelpay.detection.compute_behavioral_change_score`), and Phase F
constants/helpers (`sentinelpay.target_history`, `sentinelpay.model_features`)
COMPLETELY UNMODIFIED. See `sentinelpay.inference.state` for the state
source contract and `sentinelpay.inference.artifacts` for the model source
contract.

===========================================================================
INVARIANT: a transaction's own hypothetical isFraud can never affect its
own score.
===========================================================================
Enforced structurally: `score_transaction`'s input schema has no `isFraud`
field, and a caller-supplied `'isFraud'` key RAISES rather than being
silently ignored. Enforced causally: Phase F features are read from STATE
ONLY (built from strictly-resolved history via
`sentinelpay.inference.state.update_resolved_labels`, never from the
transaction being scored); Phase D features are computed from the key's
buffer plus this one transaction via `compute_behavioral_change_score`,
which never reads `isFraud` at all, by construction.

`score_transaction` is a PURE function -- it never mutates `state`. A
scored transaction is NOT automatically added to Phase D's buffer or
Phase F's counters; growing state is a separate, explicit action -- see
`sentinelpay.inference.state.record_observed_transactions` for Phase D
(occurrence-only, no label needed, idempotent by `TransactionID`) and
`sentinelpay.inference.state.update_resolved_labels` for Phase F
(resolution-only, also idempotent by `TransactionID`). The required call
order is: score, THEN record -- never record a transaction before its own
score has been computed against the pre-recording state. `score_transaction`
itself does NOT require `TransactionID` -- it never de-duplicates anything
(it is pure and stateless per call), so it has nothing to key a replay
check on; `TransactionID` is only required by the two state-mutating
functions above.

`has_identity` is supplied DIRECTLY by the caller (was device/identity
data collected for this transaction) -- NOT derived from a `TransactionID`
lookup against `train_identity.csv`, since a genuinely new transaction has
no such training-file entry. This is a necessary, explicitly documented
adaptation of Phase C's `has_identity` definition for live scoring.

NO THRESHOLD IS APPLIED HERE. `score_transaction` returns a raw
probability and diagnostic features/state only -- no threshold has been
reviewed or approved, so none is invented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from sentinelpay.config import DataConfig, DetectionConfig, load_config
from sentinelpay.detection import compute_behavioral_change_score
from sentinelpay.eda.grouping_key_sufficiency import build_group_key
from sentinelpay.features import add_amt_decimal_part, add_amt_log1p, add_dt_day_of_week, add_dt_hour_of_day
from sentinelpay.inference.state import InferenceState
from sentinelpay.model_features import (
    IMPUTE_FIXED_VALUE,
    LADDER_FEATURE_COLUMNS,
    PAYMENT_GROUP_COL,
    PHASE_D_IMPUTED_COLUMNS,
    PHASE_D_NUMERIC_COLUMNS,
    _one_hot_flag,
    get_ladder_matrix,
)
from sentinelpay.target_history import SMOOTHING_K, SUFFICIENT_HISTORY_THRESHOLD

GRADUATED_STEP = "F2"
REQUIRED_BASE_FIELDS = {"TransactionDT", "TransactionAmt", "has_identity"}


@dataclass
class ScoreResult:
    payment_proxy_key: str
    fraud_probability: float
    features: dict
    phase_d_diagnostics: dict
    phase_f_diagnostics: dict


def _nan_to_none(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def _phase_c_features_single(txn_df: pd.DataFrame) -> pd.DataFrame:
    df = add_amt_log1p(txn_df)
    df = add_amt_decimal_part(df)
    df = add_dt_hour_of_day(df)
    df = add_dt_day_of_week(df)
    df["has_identity"] = txn_df["has_identity"].astype("int8")
    return df[["amt_log1p", "amt_decimal_part", "dt_hour_of_day", "dt_day_of_week", "has_identity"]]


def _phase_d_features_single(state: InferenceState, key, txn_row: pd.DataFrame, detection_config: DetectionConfig) -> pd.DataFrame:
    """Reuses `compute_behavioral_change_score` UNMODIFIED on (this key's
    full buffer + the one new row) -- guarantees byte-for-byte fidelity to
    Phase D's proven algorithm rather than reimplementing its windowing
    logic. Output row order is proven to match input row order by that
    function's own `_pos`-based realignment, so the new row's own output is
    reliably the LAST row here (the new row is concatenated last)."""
    buffer_slice = state.phase_d_buffer[state.phase_d_buffer[PAYMENT_GROUP_COL] == key]
    # Avoid pandas' empty-DataFrame concat dtype-inference FutureWarning for
    # a brand-new key with no buffer history yet -- functionally equivalent
    # either way, but this keeps the cold-start path warning-free.
    combined = pd.concat([buffer_slice, txn_row], ignore_index=True) if len(buffer_slice) else txn_row.reset_index(drop=True)
    scored = compute_behavioral_change_score(
        combined, detection_config, group_col=PAYMENT_GROUP_COL, amount_col="TransactionAmt", dt_col="TransactionDT"
    )
    return scored.iloc[[-1]].reset_index(drop=True)


def _phase_f_features_single(state: InferenceState, key) -> dict:
    """Restates Phase F's smoothing formula
    (`(fraud_count + k*global_rate) / (event_count + k)`) inline against
    STATE lookups only -- verified against `compute_prior_fraud_rate`'s own
    output by `tests/test_inference_scoring.py`'s batch-vs-single
    equivalence test, not merely assumed identical."""
    if key in state.phase_f_counts.index:
        fraud_count = int(state.phase_f_counts.loc[key, "fraud_count"])
        event_count = int(state.phase_f_counts.loc[key, "event_count"])
    else:
        fraud_count, event_count = 0, 0

    global_event_count = state.phase_f_global_event_count
    global_fraud_count = state.phase_f_global_fraud_count
    global_cold_start = global_event_count == 0
    global_rate = float("nan") if global_cold_start else global_fraud_count / global_event_count

    smoothed_raw = (fraud_count + SMOOTHING_K * global_rate) / (event_count + SMOOTHING_K)
    smoothed = IMPUTE_FIXED_VALUE if (isinstance(smoothed_raw, float) and math.isnan(smoothed_raw)) else smoothed_raw
    sufficient_target_history = int(event_count >= SUFFICIENT_HISTORY_THRESHOLD)

    return {
        "payment_proxy_prior_fraud_count": fraud_count,
        "payment_proxy_prior_event_count": event_count,
        "global_prior_fraud_rate": global_rate,
        "payment_proxy_prior_fraud_rate_raw": (fraud_count / event_count) if event_count > 0 else float("nan"),
        "payment_proxy_prior_fraud_rate_smoothed": smoothed,
        "sufficient_target_history": sufficient_target_history,
        "global_cold_start": int(global_cold_start),
    }


def score_transaction(
    transaction: dict,
    state: InferenceState,
    artifact: dict,
    detection_config: DetectionConfig,
    config: DataConfig | None = None,
) -> ScoreResult:
    if "isFraud" in transaction:
        raise ValueError(
            "score_transaction: 'isFraud' must not be present in a scoring request -- a transaction's own "
            "label is never an input to its own score (see module docstring INVARIANT)."
        )
    config = config or load_config()
    required = REQUIRED_BASE_FIELDS | set(config.payment_proxy_key_columns)
    missing = required - set(transaction)
    if missing:
        raise ValueError(f"score_transaction requires fields {sorted(missing)}")

    txn_df = pd.DataFrame([transaction])
    key_df = build_group_key(txn_df, config.payment_proxy_key_columns, key_name=PAYMENT_GROUP_COL)
    if len(key_df) == 0:
        raise ValueError("score_transaction: payment_proxy_key components incomplete for this transaction")
    key = key_df[PAYMENT_GROUP_COL].iloc[0]

    c_feats = _phase_c_features_single(txn_df)
    d_feats = _phase_d_features_single(
        state, key, key_df[[PAYMENT_GROUP_COL, "TransactionDT", "TransactionAmt"]], detection_config
    )
    f_feats_dict = _phase_f_features_single(state, key)

    d_selected = d_feats[PHASE_D_NUMERIC_COLUMNS + ["flag"]].reset_index(drop=True)
    flag_dummies = _one_hot_flag(d_selected["flag"])
    f_feats = pd.DataFrame([f_feats_dict])

    assembled = pd.concat(
        [c_feats.reset_index(drop=True), d_selected.drop(columns=["flag"]), flag_dummies, f_feats], axis=1
    )
    for col in PHASE_D_IMPUTED_COLUMNS:
        assembled[col] = assembled[col].fillna(IMPUTE_FIXED_VALUE)
    assembled["payment_proxy_prior_fraud_rate_smoothed"] = assembled["payment_proxy_prior_fraud_rate_smoothed"].fillna(
        IMPUTE_FIXED_VALUE
    )
    assembled["global_cold_start"] = assembled["global_cold_start"].astype("int64")
    assembled["sufficient_target_history"] = assembled["sufficient_target_history"].astype("int64")

    X = get_ladder_matrix(assembled, GRADUATED_STEP).to_numpy()
    X_scaled = artifact["scaler"].transform(X)
    proba = float(artifact["model"].predict_proba(X_scaled)[0, 1])

    return ScoreResult(
        payment_proxy_key=str(key),
        fraud_probability=proba,
        features=dict(zip(LADDER_FEATURE_COLUMNS[GRADUATED_STEP], X[0].tolist())),
        phase_d_diagnostics=_nan_to_none(
            {
                "prior_median": float(d_feats["prior_median"].iloc[0]),
                "prior_mad": float(d_feats["prior_mad"].iloc[0]),
                "prior_count_in_window": int(d_feats["prior_count_in_window"].iloc[0]),
                "modified_zscore": float(d_feats["modified_zscore"].iloc[0]),
                "flag": str(d_feats["flag"].iloc[0]),
            }
        ),
        phase_f_diagnostics=_nan_to_none(f_feats_dict),
    )
