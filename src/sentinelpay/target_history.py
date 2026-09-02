"""Phase F: causal, fold-safe, target-derived historical fraud-rate feature
for `payment_proxy_key`.

This is the FIRST module in this project where `isFraud` is a legitimate
INPUT to feature computation, not merely a downstream diagnostic --
`sentinelpay.data.history`, `sentinelpay.detection`, and every
`sentinelpay.eda.*_analysis`/`link_sufficiency`/`component_analysis` module
are and remain strictly non-target. This module is the one deliberate,
explicitly-isolated exception; nothing else in this project imports or reads
`isFraud`.

Approved scope (Phase F proposal review): `payment_proxy_key` only. No
`device_proxy_key`, no E.1/E.2 relationship-node keys, no generic
multi-key framework -- this module is intentionally narrow, not a reusable
target-encoding utility.

**No new causal primitive.** `sentinelpay.data.history.prior_group_amount_stats`
is reused UNMODIFIED -- it is already generic over any numeric `amount_col`
(its own tests assert identical output whether or not an `isFraud`-named
column even exists in the frame; it never special-cases a column name or
meaning). Calling it with `amount_col="isFraud"` does not make
`sentinelpay.data.history` target-aware -- the target dependency is
introduced entirely here, by this module choosing to pass `isFraud` as data.
`sentinelpay.data.history` is not modified, and no duplicate causal-history
implementation is written.

**Explicit source/recipient eligibility -- not incidental filtering.** This
module's `compute_prior_fraud_rate` REQUIRES the caller to declare
`allowed_source_partitions` and RAISES if `pool_df` contains any other
partition value. It does not filter partitions itself and does not infer
eligibility from `TransactionDT` ordering -- the caller
(`sentinelpay.eda.run_phase_f`) must construct a separate, explicitly-scoped
`pool_df` for each recipient partition:

    train row       <- pool_df built from `train` rows ONLY
                        (allowed_source_partitions={"train"})
    validation row   <- pool_df built from `train` + `validation` rows ONLY
                        (allowed_source_partitions={"train", "validation"})

`embargo_1`, `embargo_2`, and `holdout` are NEVER label sources for anyone,
regardless of chronological order -- this is the one deliberate exception to
"strictly-earlier-in-time is always eligible" that every non-target feature
in this project (C, D, D.1, E.1, E.2) otherwise relies on. Those phases'
reasoning (embargo content safely feeds later partitions) does NOT extend to
target labels -- this is exactly the case the embargo exists to protect
against (see `sentinelpay/data/history.py`'s and `sentinelpay/features.py`'s
own docstrings). `embargo_1`/`embargo_2`/`holdout` rows never receive a
computed feature value in this phase either.

**Smoothing.** Raw per-key fraud rate is not used directly (rejected in
proposal review -- most `payment_proxy_key` groups are small; see
`reports/eda/phase_d1_report.md`). Additive (credibility-weighted) smoothing
toward a causal, strictly-prior GLOBAL rate:

    smoothed_rate = (prior_fraud_count + k * global_prior_fraud_rate)
                     / (prior_event_count + k)

`global_prior_fraud_rate` is computed over the EXACT SAME eligible pool as
the per-key counts (same `pool_df`, same `allowed_source_partitions`
contract), just pooled across all keys instead of grouped -- so the
numerator/denominator and the smoothing prior are internally consistent, not
two different notions of "prior." `k` (`SMOOTHING_K`) is a fixed first-pass
constant declared before any validation-only evaluation and never tuned to
one -- see its own docstring for why it must NOT be read as a derived
statistical equivalence to any other constant in this project.

Cold start falls directly out of the formula, no special branch needed for
the ordinary case: `prior_event_count == 0` -> `smoothed_rate ==
global_prior_fraud_rate` exactly. The one genuine edge case is
`global_prior_event_count == 0` (only possible at the very start of `train`,
before this pool has seen ANY row at all) -- `global_prior_fraud_rate` is
`NaN` there by construction, which propagates `smoothed_rate` to `NaN`
automatically; `global_cold_start` flags exactly this case explicitly.

Tie semantics: two rows sharing `(group_col, dt_col)` never see each other's
`isFraud` value -- inherited unchanged from `prior_group_amount_stats`'s own
proven bucket-collapse-then-exclusive-cumulative-sum construction (see
`tests/test_history.py`), re-asserted at this module's own integration level
in `tests/test_target_history.py` rather than only trusted transitively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinelpay.data.history import prior_group_amount_stats

# Fixed first-pass smoothing pseudo-count strength (the "k" in the additive-
# smoothing formula above). This VALUE is the same as Phase D's
# `window_size_events` (configs/detection.yaml) -- reused for
# PROJECT-CONSISTENCY ONLY. This is explicitly NOT a claim that the two
# constants are mathematically equivalent: `window_size_events` bounds an
# EVENT-COUNT WINDOW for a robust median/MAD statistic; `SMOOTHING_K` is a
# CREDIBILITY WEIGHT (a pseudo-count of "how many observations of the global
# rate to pretend we already had") in a completely different formula. They
# measure different things and merely share a round, already-reviewed value.
# Fixed here before this module's output is ever compared to `isFraud` in
# the validation-only evaluation (`sentinelpay.eda.run_phase_f`) -- never
# tuned to that comparison's outcome.
SMOOTHING_K = 20.0

# Diagnostic-only "enough own-key causal history to be more data than prior"
# bar -- reuses the same VALUE as D.1's own `DECISION_THRESHOLD`
# (`sentinelpay.eda.grouping_key_sufficiency.DECISION_THRESHOLD`), for the
# identical project-consistency reasoning `SMOOTHING_K` documents above, not
# a derived equivalence. Redeclared as an independent constant here (rather
# than imported) to avoid a top-level-module -> eda-module dependency --
# mirrors `sentinelpay.detection`'s own layering, which never imports from
# `sentinelpay.eda`. Purely diagnostic: does NOT gate `smoothed_rate` itself
# (the formula already handles low counts via smoothing), only labels rows
# for the validation-only evaluation's stratification.
SUFFICIENT_HISTORY_THRESHOLD = 5

DEFAULT_GROUP_COL = "_payment_group_key"
DEFAULT_DT_COL = "TransactionDT"
DEFAULT_TARGET_COL = "isFraud"
DEFAULT_PARTITION_COL = "partition"

_GLOBAL_POOL_COL = "_global_pool"
_GLOBAL_POOL_VALUE = "GLOBAL"

OUTPUT_COLUMNS = [
    "payment_proxy_prior_fraud_count",
    "payment_proxy_prior_event_count",
    "global_prior_fraud_rate",
    "payment_proxy_prior_fraud_rate_raw",
    "payment_proxy_prior_fraud_rate_smoothed",
    "sufficient_target_history",
    "global_cold_start",
]


def compute_prior_fraud_rate(
    pool_df: pd.DataFrame,
    allowed_source_partitions: set[str],
    group_col: str = DEFAULT_GROUP_COL,
    dt_col: str = DEFAULT_DT_COL,
    target_col: str = DEFAULT_TARGET_COL,
    partition_col: str = DEFAULT_PARTITION_COL,
    k: float = SMOOTHING_K,
    min_history_threshold: int = SUFFICIENT_HISTORY_THRESHOLD,
) -> pd.DataFrame:
    """Causal, fold-safe prior fraud-rate metrics for every row in `pool_df`,
    aligned to `pool_df.index`. See module docstring for the full contract.

    `pool_df` must already BE the exact eligible label-source pool for the
    rows being scored (the caller's responsibility -- see
    `sentinelpay.eda.run_phase_f` for how `train`'s and `validation`'s pools
    are built explicitly, as two separate calls). This function does not
    filter partitions itself; it enforces the contract instead: raises
    `ValueError` if `pool_df[partition_col]` contains any value outside
    `allowed_source_partitions`, so a caller cannot silently pass a mixed or
    over-broad frame and rely on incidental filtering.

    For every row: `payment_proxy_prior_fraud_count` / `_prior_event_count`
    are the strictly-prior same-`group_col` `target_col` sum/count within
    `pool_df` (via `prior_group_amount_stats`, unmodified); `global_prior_fraud_rate`
    is the same quantity pooled across ALL rows in `pool_df` (same eligible
    pool, ungrouped); `payment_proxy_prior_fraud_rate_raw` is the unsmoothed
    per-key rate (`NaN` when `prior_event_count == 0`);
    `payment_proxy_prior_fraud_rate_smoothed` is the additive-smoothed rate
    (see module docstring for the formula and both cold-start cases);
    `sufficient_target_history` and `global_cold_start` are diagnostic flags.
    """
    for col in (group_col, dt_col, target_col, partition_col):
        if col not in pool_df.columns:
            raise ValueError(f"compute_prior_fraud_rate requires column '{col}'")

    allowed = set(allowed_source_partitions)
    actual_partitions = set(pool_df[partition_col].unique())
    if not actual_partitions <= allowed:
        raise ValueError(
            f"compute_prior_fraud_rate requires every row's '{partition_col}' to be one of "
            f"{sorted(allowed)} -- got {sorted(actual_partitions)}. Build the eligible "
            "label-source pool explicitly before calling this function; it does not filter "
            "partitions itself, by design (no incidental filtering)."
        )

    working = pool_df[[group_col, dt_col, target_col]].copy()
    # prior_group_amount_stats's own merge does not defend against the
    # Windows int32-vs-int64 merge-key factorizer mismatch its SIBLING
    # functions in sentinelpay.data.history explicitly document and guard
    # against (numpy's platform-default integer dtype is 32-bit there, and
    # groupby-then-reset_index commonly promotes only an index level, not a
    # value column, to int64). Rather than modify sentinelpay.data.history
    # (out of scope -- see module docstring), the defense is applied here,
    # at the call site, on data this module's own real-CSV-loaded callers
    # actually hit this on.
    working[dt_col] = working[dt_col].astype("int64")
    working[_GLOBAL_POOL_COL] = _GLOBAL_POOL_VALUE

    per_key = prior_group_amount_stats(working, group_col=group_col, amount_col=target_col, dt_col=dt_col)
    global_ = prior_group_amount_stats(working, group_col=_GLOBAL_POOL_COL, amount_col=target_col, dt_col=dt_col)

    prior_fraud_count = per_key["prior_sum"]
    prior_event_count = per_key["prior_count"]
    global_prior_fraud_count = global_["prior_sum"]
    global_prior_event_count = global_["prior_count"]

    global_cold_start = global_prior_event_count == 0
    global_prior_fraud_rate = pd.Series(np.nan, index=pool_df.index, dtype="float64")
    has_global_history = ~global_cold_start
    global_prior_fraud_rate.loc[has_global_history] = (
        global_prior_fraud_count.loc[has_global_history] / global_prior_event_count.loc[has_global_history]
    )

    has_own_history = prior_event_count > 0
    prior_fraud_rate_raw = pd.Series(np.nan, index=pool_df.index, dtype="float64")
    prior_fraud_rate_raw.loc[has_own_history] = (
        prior_fraud_count.loc[has_own_history] / prior_event_count.loc[has_own_history]
    )

    # prior_event_count == 0 with global history present -> reduces exactly
    # to global_prior_fraud_rate (0 + k*rate)/(0+k) == rate. When
    # global_prior_event_count == 0 too, global_prior_fraud_rate is NaN
    # above, which propagates smoothed to NaN here automatically -- no
    # separate branch needed for either cold-start case.
    smoothed = (prior_fraud_count + k * global_prior_fraud_rate) / (prior_event_count + k)

    sufficient_target_history = prior_event_count >= min_history_threshold

    out = pd.DataFrame(index=pool_df.index)
    out["payment_proxy_prior_fraud_count"] = prior_fraud_count.astype("int64")
    out["payment_proxy_prior_event_count"] = prior_event_count.astype("int64")
    out["global_prior_fraud_rate"] = global_prior_fraud_rate
    out["payment_proxy_prior_fraud_rate_raw"] = prior_fraud_rate_raw
    out["payment_proxy_prior_fraud_rate_smoothed"] = smoothed
    out["sufficient_target_history"] = sufficient_target_history
    out["global_cold_start"] = global_cold_start
    return out[OUTPUT_COLUMNS]


# Phase F's two approved feature recipients (proposal review) -- `embargo_1`
# and `embargo_2` are deliberately absent from both: neither is ever a label
# source for anyone, and neither receives a computed feature value in this
# phase. `holdout` is assumed already excluded from `valid_df` before it
# reaches this function (see `sentinelpay.eda.run_phase_f`'s own holdout
# sealing, matching every prior phase) -- this module has no holdout concept
# of its own to enforce, the same way `sentinelpay.data.history` doesn't.
TRAIN_RECIPIENT_PARTITIONS = {"train"}
VALIDATION_SOURCE_PARTITIONS = {"train", "validation"}


def build_eligible_pools(
    valid_df: pd.DataFrame, partition_col: str = DEFAULT_PARTITION_COL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explicit eligible-pool construction for Phase F's two approved
    feature recipients -- `train` and `validation` -- from an already
    holdout-sealed, already-key-valid `valid_df` covering every development
    partition. Returns `(train_pool, validation_pool)`:

        train_pool      = rows with partition == "train" ONLY
        validation_pool = rows with partition in {"train", "validation"} ONLY

    This is the single place the source/recipient contract from the module
    docstring is expressed as code -- `sentinelpay.eda.run_phase_f` and this
    module's own tests both call this function rather than each
    re-deriving the partition filter inline (no incidental filtering,
    duplicated or otherwise). `embargo_1`/`embargo_2` rows are excluded from
    BOTH pools by construction, not filtered out after the fact.
    """
    if partition_col not in valid_df.columns:
        raise ValueError(f"build_eligible_pools requires column '{partition_col}'")
    train_pool = valid_df[valid_df[partition_col] == "train"].copy()
    validation_pool = valid_df[valid_df[partition_col].isin(["train", "validation"])].copy()
    return train_pool, validation_pool
