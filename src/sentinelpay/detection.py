"""Phase D: non-target, causal, per-`payment_proxy_key` behavioral-change
score.

Hard non-target boundary: this module never imports, accepts, or reads
`isFraud` (or any target column) in any function signature or computation.
Every score, flag, cold-start decision, and zero-MAD decision below is
computed strictly from `sentinelpay.data.history.prior_group_windowed_robust_stats`
(median/MAD over `amt_log1p`, itself computed by
`sentinelpay.features.add_amt_log1p` -- row-local, non-target) and the fixed
hyperparameters in `sentinelpay.config.DetectionConfig`/`configs/detection.yaml`.
`isFraud` is read in exactly one place in this project:
`sentinelpay.eda.run_phase_d`'s validation-only evaluation step, which runs
strictly after every score/flag here is already final and never writes
anything back into this module's output.

Scoring formula: the standard Iglewicz & Hoaglin (1993) modified z-score,

    modified_zscore = modified_zscore_scale_constant * (amt_log1p - prior_median) / prior_mad

flagged `scored_outlier` when `abs(modified_zscore) >= modified_zscore_threshold`,
else `scored_normal`. Rows with fewer than `min_history_for_score`
strictly-prior same-key events get flag `insufficient_history` and
`score = NaN` (cold start -- not imputed). Rows with sufficient history but
`prior_mad <= zero_mad_epsilon` get flag `zero_mad` and `score = NaN`
(avoids division blow-up; `sentinelpay.data.history` itself returns the true
computed MAD, including a legitimate 0.0 -- the epsilon guard lives only
here, where the division happens).

All five hyperparameters (`min_history_for_score`, `window_size_events`,
`modified_zscore_scale_constant`, `modified_zscore_threshold`,
`zero_mad_epsilon`) come from the `DetectionConfig` passed in by the caller
(loaded from `configs/detection.yaml`) -- none is hard-coded here. None may
be selected, tuned, or changed based on validation `isFraud` results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinelpay.config import DetectionConfig
from sentinelpay.data.history import prior_group_windowed_robust_stats
from sentinelpay.features import add_amt_log1p

FLAG_INSUFFICIENT_HISTORY = "insufficient_history"
FLAG_ZERO_MAD = "zero_mad"
FLAG_SCORED_OUTLIER = "scored_outlier"
FLAG_SCORED_NORMAL = "scored_normal"

DEFAULT_AMOUNT_COL = "TransactionAmt"
DEFAULT_DT_COL = "TransactionDT"
DEFAULT_GROUP_COL = "_payment_group_key"

OUTPUT_COLUMNS = [
    "amt_log1p",
    "prior_median",
    "prior_mad",
    "prior_count_in_window",
    "modified_zscore",
    "flag",
]


def compute_behavioral_change_score(
    df: pd.DataFrame,
    detection_config: DetectionConfig,
    group_col: str = DEFAULT_GROUP_COL,
    amount_col: str = DEFAULT_AMOUNT_COL,
    dt_col: str = DEFAULT_DT_COL,
) -> pd.DataFrame:
    """Non-target behavioral-change score per row, aligned to `df.index`.

    Does not accept or read `isFraud` (or any target column) -- pass any
    frame with `group_col`/`amount_col`/`dt_col`; an `isFraud` column, if
    present in `df`, is simply never selected or referenced anywhere in this
    function.

    Returns a DataFrame with columns `amt_log1p`, `prior_median`,
    `prior_mad`, `prior_count_in_window`, `modified_zscore`, `flag`.
    """
    for col in (group_col, amount_col, dt_col):
        if col not in df.columns:
            raise ValueError(f"compute_behavioral_change_score requires column '{col}'")

    amt_log1p = add_amt_log1p(df, amount_col=amount_col)["amt_log1p"]

    working = df[[group_col, dt_col]].copy()
    working["amt_log1p"] = amt_log1p

    robust = prior_group_windowed_robust_stats(
        working,
        group_col=group_col,
        amount_col="amt_log1p",
        dt_col=dt_col,
        window_size_events=detection_config.window_size_events,
    )

    out = pd.DataFrame(index=df.index)
    out["amt_log1p"] = amt_log1p
    out["prior_median"] = robust["prior_median"]
    out["prior_mad"] = robust["prior_mad"]
    out["prior_count_in_window"] = robust["prior_count_in_window"]

    insufficient = out["prior_count_in_window"] < detection_config.min_history_for_score
    zero_mad = (~insufficient) & (out["prior_mad"] <= detection_config.zero_mad_epsilon)
    scoreable = ~insufficient & ~zero_mad

    modified_zscore = pd.Series(np.nan, index=df.index, dtype="float64")
    modified_zscore.loc[scoreable] = (
        detection_config.modified_zscore_scale_constant
        * (out.loc[scoreable, "amt_log1p"] - out.loc[scoreable, "prior_median"])
        / out.loc[scoreable, "prior_mad"]
    )
    out["modified_zscore"] = modified_zscore

    flag = pd.Series(FLAG_SCORED_NORMAL, index=df.index, dtype="object")
    flag.loc[insufficient] = FLAG_INSUFFICIENT_HISTORY
    flag.loc[zero_mad] = FLAG_ZERO_MAD
    is_outlier = scoreable & (modified_zscore.abs() >= detection_config.modified_zscore_threshold)
    flag.loc[is_outlier] = FLAG_SCORED_OUTLIER
    out["flag"] = flag

    return out[OUTPUT_COLUMNS]
