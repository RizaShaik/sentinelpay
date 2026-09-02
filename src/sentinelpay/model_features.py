"""Phase G feature-matrix assembly: the B0-F2 baseline ladder, built
strictly by joining Phase C (`sentinelpay.features`), Phase D
(`sentinelpay.detection`), and Phase F (`sentinelpay.target_history`) --
all three consumed UNMODIFIED and VERBATIM. No new causal-history logic is
written here; this module only selects, joins, and (for the small number of
NaN cells each already-approved phase produces at its own documented edge
cases) imputes with a fixed constant.

===========================================================================
EXPLICIT PER-ROW LEAKAGE CONTRACT (read before touching this file)
===========================================================================

Every row in the assembled matrix belongs to exactly one of two recipients,
`train` or `validation` -- the same two partitions Phase F already
restricted itself to. `embargo_1`/`embargo_2`/`holdout` NEVER appear as
rows in the assembled matrix. For every row, by feature block:

**Phase C (row-local, no history)**: identical treatment for `train` and
`validation` rows -- no partition dependency at all. `amt_log1p`/
`amt_decimal_part`/`dt_hour_of_day`/`dt_day_of_week` are pure functions of
that row's own `TransactionAmt`/`TransactionDT`; `has_identity` is a pure
structural join against the identity file. Cannot leak by construction.

**Phase D (non-target causal history -- `prior_median`/`prior_mad`/
`prior_count_in_window`/`modified_zscore`/`flag`)**: computed by
`sentinelpay.detection.compute_behavioral_change_score` over ALL FOUR
development partitions concatenated (`train` + `embargo_1` + `validation` +
`embargo_2`), EXACTLY as `sentinelpay.eda.run_phase_d` already does and
already had approved. Phase D's own embargo semantics (documented in
`sentinelpay/data/history.py` and `reports/eda/phase_d_report.md`) hold that
`embargo_1` content LEGITIMATELY contributes to later rows' NON-TARGET
history, since the embargo exists to guard a label-based boundary decision,
not to blank out real antecedent transaction content:

    TRAIN row      <- strictly-TransactionDT-prior same-payment_proxy_key
                       rows within `train` ONLY (no partition precedes
                       train).
    VALIDATION row  <- strictly-prior same-payment_proxy_key rows drawn
                       from {train, embargo_1, validation} -- embargo_1
                       IS a legitimate source here.

**Phase F (target-derived causal history --
`payment_proxy_prior_fraud_rate_smoothed`/`global_cold_start`/
`sufficient_target_history`)**: computed by
`sentinelpay.target_history.build_eligible_pools` +
`compute_prior_fraud_rate`, called exactly as `sentinelpay.eda.run_phase_f`
already does, UNMODIFIED. Because this feature reads `isFraud`, Phase F's
OWN (already-approved, frozen) narrower contract applies -- `embargo_1`/
`embargo_2` are NEVER a label source, regardless of chronological order:

    TRAIN row      <- strictly-prior same-payment_proxy_key TRAIN rows
                       ONLY.
    VALIDATION row  <- strictly-prior same-payment_proxy_key rows drawn
                       from {train, validation} ONLY -- embargo_1 is
                       EXCLUDED, even though it is chronologically prior
                       and even though the Phase D block above DOES use it
                       for this exact same row.

**THIS ASYMMETRY IS INTENTIONAL, not a bug.** Phase D's non-target embargo
semantics and Phase F's target-derived embargo semantics were reviewed and
approved separately, in different phases, for different reasons. Phase G
does not change either -- it assembles both, verbatim, onto the same rows.
See `tests/test_model_features.py::test_phase_d_sees_embargo_1_but_phase_f_does_not`
for the integration-level proof against the actual assembled matrix (not
just the underlying functions in isolation).

No row, in either block, ever depends on its own `isFraud` value or on any
strictly-later-in-time row's `isFraud` value. `holdout` rows are never
loaded by `sentinelpay.eda.run_phase_g` at all -- not filtered out
downstream, never read from disk in the first place, matching every prior
phase.
===========================================================================

**Phase D column audit -- why each Phase D output is/isn't a B2 feature**
(required before any Phase D column enters the model):

- `amt_log1p`: **EXCLUDED**. `compute_behavioral_change_score` recomputes
  this internally (`np.log1p(TransactionAmt)`) as a row-local intermediate
  -- it is IDENTICAL to Phase C's own `amt_log1p` (same formula, same input
  column), already present in B1. Including it again would duplicate an
  existing column, not add information.
- `prior_median` / `prior_mad`: legitimate causal group-level features --
  the row's `payment_proxy_key` group's robust prior amount statistics.
  `NaN` exactly when `prior_count_in_window == 0` (true cold start);
  imputed to the fixed constant `IMPUTE_FIXED_VALUE` (0.0), with `flag`
  already carrying why (`flag == insufficient_history` covers this `NaN`
  population, since `prior_count_in_window == 0` implies
  `prior_count_in_window < min_history_for_score`).
- `prior_count_in_window`: legitimate causal feature, never `NaN` (0 is
  its minimum).
- `modified_zscore`: NOT redundant with `prior_median`/`prior_mad`/
  `amt_log1p` despite being a deterministic function of them -- it is a
  RATIO (`(amt_log1p - prior_median) / prior_mad`), a nonlinear transform a
  LINEAR model over the three raw inputs cannot reconstruct on its own.
  Included as a genuinely different functional feature for a linear model.
  `NaN` whenever `insufficient_history` OR `zero_mad`; imputed to the same
  fixed constant, with `flag` as its companion indicator (the same
  impute-plus-indicator pattern used for Phase F's `global_cold_start`
  below).
- `flag`: categorical, one-hot encoded as `k-1` dummy columns
  (`flag_insufficient_history`, `flag_zero_mad`, `flag_scored_outlier`;
  `scored_normal` -- the majority class -- is the dropped reference
  category, avoiding the dummy-variable trap against the model's
  intercept). This is ALSO the companion indicator for BOTH
  `prior_median`/`prior_mad`'s and `modified_zscore`'s imputation -- no
  separate indicator column is needed for those two.

**Phase F imputation -- corrected.** `global_prior_fraud_rate` is NEVER
imputed and is NOT used as a fallback value for
`payment_proxy_prior_fraud_rate_smoothed` -- at the one row where
`global_cold_start` is `True`, `global_prior_fraud_rate` is ITSELF
undefined (that is exactly why `smoothed_rate` is `NaN` there in the first
place; see `sentinelpay.target_history`'s own formula). Using an undefined
value as a fallback would be incoherent. Instead: `smoothed_rate`'s `NaN`
at `global_cold_start` is imputed to the SAME fixed constant
`IMPUTE_FIXED_VALUE` (0.0, train-only-declared, never data-derived), and
`global_cold_start` is retained as an explicit 0/1 indicator feature
wherever `smoothed_rate` is used (both F1 and F2 -- `global_cold_start` is
the necessary companion indicator for `smoothed_rate`'s own imputation,
exactly parallel to `flag` accompanying Phase D's imputed columns; it is
not itself the substantive ladder step being tested -- `sufficient_target_history`
is, and is what distinguishes F2 from F1). In the real Phase F run this
affects exactly 1 row (of 386,407) in `train` and 0 rows in `validation`.
"""
from __future__ import annotations

import pandas as pd

from sentinelpay.config import DetectionConfig
from sentinelpay.detection import (
    FLAG_INSUFFICIENT_HISTORY,
    FLAG_SCORED_NORMAL,
    FLAG_SCORED_OUTLIER,
    FLAG_ZERO_MAD,
    compute_behavioral_change_score,
)
from sentinelpay.features import build_feature_frame
from sentinelpay.target_history import (
    TRAIN_RECIPIENT_PARTITIONS,
    VALIDATION_SOURCE_PARTITIONS,
    build_eligible_pools,
    compute_prior_fraud_rate,
)

PAYMENT_GROUP_COL = "_payment_group_key"

# Fixed, pre-declared imputation constant -- NOT data-derived, matching
# every other fixed constant in this project. Used for prior_median/
# prior_mad/modified_zscore (Phase D, NaN at insufficient history/zero MAD)
# and for payment_proxy_prior_fraud_rate_smoothed at the single
# global_cold_start edge case (Phase F). See module docstring
# "Phase F imputation -- corrected" for why global_prior_fraud_rate itself
# is never used as the fallback.
IMPUTE_FIXED_VALUE = 0.0

PHASE_D_NUMERIC_COLUMNS = ["prior_median", "prior_mad", "prior_count_in_window", "modified_zscore"]
PHASE_D_IMPUTED_COLUMNS = ["prior_median", "prior_mad", "modified_zscore"]
PHASE_D_FLAG_REFERENCE = FLAG_SCORED_NORMAL  # dropped reference category (majority class)
# Raw flag values (not just their dummy-column names) kept as a parallel
# list so _one_hot_flag never has to reverse-engineer a value from a
# "flag_<value>" column-name string.
PHASE_D_FLAG_VALUES = [FLAG_INSUFFICIENT_HISTORY, FLAG_ZERO_MAD, FLAG_SCORED_OUTLIER]
PHASE_D_FLAG_DUMMY_COLUMNS = [f"flag_{v}" for v in PHASE_D_FLAG_VALUES]

PHASE_F_MODEL_COLUMNS_F1 = ["payment_proxy_prior_fraud_rate_smoothed", "global_cold_start"]
PHASE_F_MODEL_COLUMNS_F2_ADDITION = ["sufficient_target_history"]

B1_COLUMNS = ["amt_log1p", "amt_decimal_part", "dt_hour_of_day", "dt_day_of_week", "has_identity"]
B2_COLUMNS = B1_COLUMNS + PHASE_D_NUMERIC_COLUMNS + PHASE_D_FLAG_DUMMY_COLUMNS
F1_COLUMNS = B2_COLUMNS + PHASE_F_MODEL_COLUMNS_F1
F2_COLUMNS = F1_COLUMNS + PHASE_F_MODEL_COLUMNS_F2_ADDITION

# Fixed, single source of truth for every ladder step's feature schema and
# ORDER -- sentinelpay.eda.run_phase_g and every test both read from this
# dict rather than each re-deriving a column list. isFraud, partition
# labels, TransactionID, and every diagnostic-only column
# (payment_proxy_prior_fraud_count/event_count/global_prior_fraud_rate/
# payment_proxy_prior_fraud_rate_raw, and the raw categorical `flag`) are
# deliberately absent from every list below.
LADDER_FEATURE_COLUMNS: dict[str, list[str]] = {
    "B1": list(B1_COLUMNS),
    "B2": list(B2_COLUMNS),
    "F1": list(F1_COLUMNS),
    "F2": list(F2_COLUMNS),
}

NON_FEATURE_COLUMNS = {"TransactionID", "TransactionDT", "isFraud", "partition", PAYMENT_GROUP_COL, "flag"}


def compute_phase_c_features(
    df: pd.DataFrame, identity_ids, amount_col: str = "TransactionAmt", dt_col: str = "TransactionDT", id_col: str = "TransactionID"
) -> pd.DataFrame:
    """Thin call-through to `sentinelpay.features.build_feature_frame`,
    unmodified. Row-local -- `df` may be any subset of rows; no partition
    dependency."""
    feats, _ = build_feature_frame(df, identity_ids, amount_col=amount_col, dt_col=dt_col, id_col=id_col)
    return feats


def compute_phase_d_features(
    df_all_development_partitions: pd.DataFrame,
    detection_config: DetectionConfig,
    group_col: str = PAYMENT_GROUP_COL,
    amount_col: str = "TransactionAmt",
    dt_col: str = "TransactionDT",
) -> pd.DataFrame:
    """Thin call-through to
    `sentinelpay.detection.compute_behavioral_change_score`, unmodified.

    `df_all_development_partitions` MUST already contain all four
    development partitions (`train`/`embargo_1`/`validation`/`embargo_2`)
    -- matching Phase D's own approved non-target embargo semantics (see
    module docstring's leakage contract). Passing a narrower frame would
    silently change Phase D's own already-reviewed behavior for
    `validation` rows.
    """
    return compute_behavioral_change_score(
        df_all_development_partitions, detection_config, group_col=group_col, amount_col=amount_col, dt_col=dt_col
    )


def compute_phase_f_features(
    valid_df: pd.DataFrame, dt_col: str = "TransactionDT", group_col: str = PAYMENT_GROUP_COL, partition_col: str = "partition"
) -> pd.DataFrame:
    """Thin call-through to `sentinelpay.target_history.build_eligible_pools`
    + `compute_prior_fraud_rate`, unmodified, in the exact same shape
    `sentinelpay.eda.run_phase_f` already uses (this module does not import
    from `sentinelpay.eda.run_phase_f` itself -- Phase F's frozen artifact
    is `sentinelpay.target_history`, not its own orchestration script; that
    script is left untouched).

    `valid_df` MUST already contain all four development partitions --
    `build_eligible_pools` performs Phase F's own train/validation
    eligible-pool split internally, discarding `embargo_1`/`embargo_2` by
    construction (see `sentinelpay.target_history.build_eligible_pools`).
    Returns a DataFrame covering ONLY `train`+`validation` rows (Phase F's
    own two approved recipients), aligned to those rows' original index.
    """
    train_pool, validation_pool = build_eligible_pools(valid_df, partition_col=partition_col)
    train_features = compute_prior_fraud_rate(
        train_pool, allowed_source_partitions=TRAIN_RECIPIENT_PARTITIONS, group_col=group_col, dt_col=dt_col, partition_col=partition_col
    )
    validation_pool_features = compute_prior_fraud_rate(
        validation_pool, allowed_source_partitions=VALIDATION_SOURCE_PARTITIONS, group_col=group_col, dt_col=dt_col, partition_col=partition_col
    )
    validation_mask = validation_pool[partition_col] == "validation"
    validation_features = validation_pool_features[validation_mask]
    return pd.concat([train_features, validation_features])


def _one_hot_flag(
    flag_series: pd.Series,
    flag_values: list[str] = PHASE_D_FLAG_VALUES,
    dummy_columns: list[str] = PHASE_D_FLAG_DUMMY_COLUMNS,
) -> pd.DataFrame:
    out = pd.DataFrame(index=flag_series.index)
    for value, col in zip(flag_values, dummy_columns):
        out[col] = (flag_series == value).astype("int64")
    return out


def assemble_ladder_matrix(
    valid_df: pd.DataFrame,
    identity_ids,
    detection_config: DetectionConfig,
    dt_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    group_col: str = PAYMENT_GROUP_COL,
    partition_col: str = "partition",
    id_col: str = "TransactionID",
) -> pd.DataFrame:
    """Assemble the full B0-F2 feature superset (every ladder column, plus
    `TransactionID`/`TransactionDT`/`partition`/`isFraud`/diagnostic-only
    columns) for `train`+`validation` rows of `valid_df`. `valid_df` must
    already be an already-holdout-sealed, already-`payment_proxy_key`-valid
    frame covering all four development partitions -- see this module's
    docstring for the full per-row leakage contract. Callers slice
    `LADDER_FEATURE_COLUMNS[step]` (via `get_ladder_matrix`) to get any one
    ladder step's model-ready matrix.
    """
    required = {dt_col, amount_col, group_col, partition_col, id_col, "isFraud"}
    for col in required:
        if col not in valid_df.columns:
            raise ValueError(f"assemble_ladder_matrix requires column '{col}'")

    c_all = compute_phase_c_features(valid_df, identity_ids, amount_col=amount_col, dt_col=dt_col, id_col=id_col)
    d_all = compute_phase_d_features(valid_df, detection_config, group_col=group_col, amount_col=amount_col, dt_col=dt_col)
    f_train_val = compute_phase_f_features(valid_df, dt_col=dt_col, group_col=group_col, partition_col=partition_col)

    model_rows_mask = valid_df[partition_col].isin(["train", "validation"])
    base = valid_df.loc[model_rows_mask, [id_col, dt_col, partition_col, "isFraud"]].copy()

    c_aligned = c_all.loc[base.index]
    d_selected = d_all.loc[base.index, PHASE_D_NUMERIC_COLUMNS + ["flag"]]
    flag_dummies = _one_hot_flag(d_selected["flag"])
    f_aligned = f_train_val.loc[base.index]

    out = pd.concat([base, c_aligned, d_selected, flag_dummies, f_aligned], axis=1)

    # Fixed, pre-declared imputation (never data-derived) -- see module
    # docstring's "Phase D column audit" and "Phase F imputation --
    # corrected" sections.
    for col in PHASE_D_IMPUTED_COLUMNS:
        out[col] = out[col].fillna(IMPUTE_FIXED_VALUE)
    out["payment_proxy_prior_fraud_rate_smoothed"] = out["payment_proxy_prior_fraud_rate_smoothed"].fillna(IMPUTE_FIXED_VALUE)

    out["global_cold_start"] = out["global_cold_start"].astype("int64")
    out["sufficient_target_history"] = out["sufficient_target_history"].astype("int64")

    return out


def get_ladder_matrix(assembled: pd.DataFrame, step: str) -> pd.DataFrame:
    """Select one ladder step's fixed, ordered feature columns from an
    already-`assemble_ladder_matrix`-built frame, cast to `float64` (sklearn
    does not accept pandas nullable `Int64`/extension dtypes directly).
    `step` must be one of `LADDER_FEATURE_COLUMNS`'s keys ("B0" has no
    feature matrix -- it is a constant predictor, handled entirely in
    `sentinelpay.model_evaluation`)."""
    if step not in LADDER_FEATURE_COLUMNS:
        raise ValueError(f"get_ladder_matrix: unknown step '{step}' -- must be one of {sorted(LADDER_FEATURE_COLUMNS)}")
    cols = LADDER_FEATURE_COLUMNS[step]
    return assembled[cols].astype("float64")
