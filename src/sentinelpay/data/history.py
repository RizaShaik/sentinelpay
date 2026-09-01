"""Generic, target-agnostic, strictly-causal historical aggregation utilities.

Every function here guarantees: for a given row i, only same-`group_col` rows
with `dt_col` STRICTLY LESS than row i's `dt_col` value ever contribute to row
i's output. Two rows that share the same group and the same `dt_col` value
never see each other -- neither counts, sums, or otherwise contributes to the
other's result. Row order in the input DataFrame never matters (results are
computed from `(group_col, dt_col)` values, not row position), and
`TransactionID` (or any identifier) is never used as a tie-breaker anywhere in
this module.

Fully generic over `group_col`: this module does not choose, name, or endorse
a production grouping key. `payment_proxy_key`/`device_proxy_key`/`ProductCD`
(see sentinelpay.eda.entity, configs/data.yaml) are NOT used here or in this
module's tests -- only synthetic group columns. Real grouping-key selection is
a Phase D/E decision.

No target dependency: none of these functions accepts or reads `isFraud` (or
any target column). They compute non-target aggregates only (counts, sums,
means, recency of `amount_col`/`dt_col`).

Embargo note (applies wherever these utilities are eventually applied to
train/embargo_1/validation/embargo_2 data): these are non-target aggregates,
not fraud-rate statistics. `embargo_1` rows chronologically precede
`validation` rows, so they DO contribute to a validation row's historical
aggregate -- the 7-day embargo in configs/split.yaml exists to guard a
*label-based* boundary decision against undocumented D-column lookback
windows (see reports/eda/phase_b_report.md section 3), not to blank out real
antecedent transaction content from non-target feature computation. Excluding
embargo rows here would only make early-validation historical features
artificially sparse/wrong relative to how the same features would be computed
at real scoring time. This reasoning does NOT extend to any future
target-derived historical feature (e.g. "group's historical fraud rate") --
that is exactly the case the embargo protects against, and no such feature is
implemented anywhere in this project yet. Holdout rows are never fed to these
functions by Phase C (see sentinelpay.eda.run_phase_c).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prior_group_count(df: pd.DataFrame, group_col: str, dt_col: str) -> pd.Series:
    """Count of same-`group_col` rows with `dt_col` strictly less than this
    row's `dt_col`, aligned to `df.index`.

    Implemented via `rank(method="min") - 1`: tied `dt_col` values within a
    group receive the same (minimum) rank, so this is exactly "count of
    strictly-smaller same-group timestamps" -- tied rows get identical
    results and never count each other. Rank depends only on values, never on
    row position, so this is automatically row-order-independent.
    """
    for col in (group_col, dt_col):
        if col not in df.columns:
            raise ValueError(f"prior_group_count requires column '{col}'")
    rank = df.groupby(group_col, observed=True)[dt_col].rank(method="min")
    return (rank - 1).astype("int64").rename("prior_count")


def prior_group_amount_stats(
    df: pd.DataFrame, group_col: str, amount_col: str, dt_col: str
) -> pd.DataFrame:
    """Sum/count/mean of `amount_col` over strictly-earlier same-group rows,
    aligned to `df.index`.

    Collapses to one row per `(group_col, dt_col)` bucket first (sum, count
    of `amount_col` at that exact timestamp), sorts buckets by `dt_col`
    within each group, then takes an EXCLUSIVE cumulative sum/count
    (`cumsum() - own_bucket_value`) before merging back onto `df` by
    `(group_col, dt_col)` value. Because the merge key is the value pair, not
    row position or `TransactionID`, tied rows share the same prior
    aggregate and a future row's value can never reach a past row.
    """
    for col in (group_col, amount_col, dt_col):
        if col not in df.columns:
            raise ValueError(f"prior_group_amount_stats requires column '{col}'")

    bucket = (
        df.groupby([group_col, dt_col], observed=True)[amount_col]
        .agg(_bucket_sum="sum", _bucket_count="count")
        .reset_index()
        .sort_values([group_col, dt_col], ignore_index=True)
    )
    grp = bucket.groupby(group_col, observed=True)
    cum_sum_incl = grp["_bucket_sum"].cumsum()
    cum_count_incl = grp["_bucket_count"].cumsum()
    bucket["prior_sum"] = cum_sum_incl - bucket["_bucket_sum"]
    bucket["prior_count"] = cum_count_incl - bucket["_bucket_count"]
    bucket["prior_mean"] = bucket["prior_sum"] / bucket["prior_count"].where(bucket["prior_count"] > 0)

    merged = df.merge(
        bucket[[group_col, dt_col, "prior_sum", "prior_count", "prior_mean"]],
        on=[group_col, dt_col],
        how="left",
    )
    merged.index = df.index
    return merged[["prior_sum", "prior_count", "prior_mean"]]


def prior_group_windowed_robust_stats(
    df: pd.DataFrame,
    group_col: str,
    amount_col: str,
    dt_col: str,
    window_size_events: int,
) -> pd.DataFrame:
    """Median/MAD of `amount_col` over an event-count window of the most
    recent strictly-prior same-`group_col` rows, aligned to `df.index`.

    Returns columns `prior_median`, `prior_mad`, `prior_count_in_window`.
    This function has no target dependency (never accepts or reads
    `isFraud`) and no threshold/cold-start concept of its own -- it always
    reports whatever real prior median/MAD/count exist (including
    `prior_count_in_window == 0` -> `NaN` median/MAD when there is no
    strictly-earlier same-group row at all). Deciding what counts as "enough"
    history is a caller concern (see `sentinelpay.detection`).

    Causal contract (mirrors `prior_group_count`/`prior_group_amount_stats`,
    extended for windowing):
    - Strictly-earlier only: for row i, only same-group rows with `dt_col`
      strictly less than row i's `dt_col` are ever eligible.
    - Ties never see each other: two rows sharing `(group_col, dt_col)` are
      never in each other's eligible set -- rows are first collapsed into
      `(group_col, dt_col)` buckets, and only whole buckets are ever
      considered.
    - Window boundary + ties: eligible buckets (strictly before row i,
      sorted by `dt_col` within group) are accumulated from most-recent to
      least-recent until their combined row count is >= `window_size_events`;
      the bucket that crosses the threshold is included WHOLE, never split.
      So the realized window size is `window_size_events` rows or slightly
      more when the boundary bucket ties multiple rows together at one
      `dt_col` value -- never fewer (except when total prior history itself
      is smaller than `window_size_events`, in which case all of it is
      used). This rule is defined purely by `(group, dt)` bucket values, so
      it can never depend on row order or on which specific tied row is
      "row i."
    - Row-order independence: implemented via a groupby/sort over
      `(group_col, dt_col)` values only, never row position or any
      identifier column -- shuffling the input rows does not change any
      row's result.
    - Future perturbations: row i's window is built only from buckets with
      `dt_col` strictly less than row i's -- mutating or appending a row at
      or after row i's `dt_col` can only change buckets at or after row i,
      never row i's own window, median, or MAD.

    Implementation note: median/MAD cannot be decomposed into a cumulative
    sum the way `prior_group_amount_stats` computes prior sum/count/mean, so
    this function walks each group's buckets once with a two-pointer sliding
    window (the minimal-window start index is provably non-decreasing as the
    right edge advances), amortized O(bucket count) per group plus the cost
    of re-concatenating each window's values.
    """
    for col in (group_col, amount_col, dt_col):
        if col not in df.columns:
            raise ValueError(f"prior_group_windowed_robust_stats requires column '{col}'")
    if window_size_events <= 0:
        raise ValueError("window_size_events must be a positive integer")

    bucket = (
        df.groupby([group_col, dt_col], observed=True)[amount_col]
        .apply(lambda s: np.sort(s.to_numpy(dtype="float64")))
        .rename("_bucket_values")
        .reset_index()
    )
    bucket["_bucket_count"] = bucket["_bucket_values"].apply(len).astype("int64")
    # groupby-then-reset_index commonly promotes an integer index level to
    # int64 regardless of the source column's dtype (e.g. a downcast int32
    # TransactionDT) -- realign before the final merge so pandas' key
    # factorizer never sees a dtype mismatch between df[dt_col] and
    # bucket[dt_col].
    bucket[dt_col] = bucket[dt_col].astype(df[dt_col].dtype)
    bucket = bucket.sort_values([group_col, dt_col], ignore_index=True)

    n_buckets = len(bucket)
    prior_median = np.full(n_buckets, np.nan)
    prior_mad = np.full(n_buckets, np.nan)
    prior_count_in_window = np.zeros(n_buckets, dtype="int64")

    for _, sub in bucket.groupby(group_col, observed=True, sort=False):
        idx = sub.index.to_numpy()
        counts = sub["_bucket_count"].to_numpy()
        values = sub["_bucket_values"].to_numpy()
        n = len(idx)
        lo = 0
        running_sum = 0
        for i in range(n):
            # Shrink from the left while still satisfying the threshold --
            # the minimal (most-recent) window achieving >= window_size_events.
            while lo < i and (running_sum - counts[lo]) >= window_size_events:
                running_sum -= counts[lo]
                lo += 1
            pos = idx[i]
            prior_count_in_window[pos] = running_sum
            if running_sum > 0:
                arr = values[lo] if i - lo == 1 else np.concatenate(list(values[lo:i]))
                med = np.median(arr)
                prior_median[pos] = med
                prior_mad[pos] = np.median(np.abs(arr - med))
            running_sum += counts[i]

    bucket["prior_median"] = prior_median
    bucket["prior_mad"] = prior_mad
    bucket["prior_count_in_window"] = prior_count_in_window

    # pandas' merge key factorizer hard-requires int64 buffers; on Windows
    # numpy's platform-default integer dtype is 32-bit, so even two columns
    # that report the "same" dtype (e.g. both int32) can trigger a Cython
    # buffer-dtype mismatch. Force both merge-key integer columns to int64
    # explicitly, in copies, rather than relying on dtype-matching alone.
    left_key = df[[group_col, dt_col]].copy()
    left_key[dt_col] = left_key[dt_col].astype("int64")
    left_key["_pos"] = np.arange(len(df))

    right_key = bucket[[group_col, dt_col, "prior_median", "prior_mad", "prior_count_in_window"]].copy()
    right_key[dt_col] = right_key[dt_col].astype("int64")

    merged = left_key.merge(right_key, on=[group_col, dt_col], how="left").sort_values("_pos", ignore_index=True)
    result = merged[["prior_median", "prior_mad", "prior_count_in_window"]]
    result.index = df.index
    return result


def time_since_last_group_event(df: pd.DataFrame, group_col: str, dt_col: str) -> pd.Series:
    """`dt_col` minus the most recent strictly-earlier same-group `dt_col`
    value, aligned to `df.index`. NaN when there is no earlier same-group
    timestamp (first bucket in the group).

    Built from the distinct `(group_col, dt_col)` pairs, sorted by `dt_col`
    within group, with a `shift(1)` -- ties (rows sharing a bucket) merge
    back onto the same "previous distinct dt" value, so they get an
    identical result and never see each other.
    """
    for col in (group_col, dt_col):
        if col not in df.columns:
            raise ValueError(f"time_since_last_group_event requires column '{col}'")

    bucket = df[[group_col, dt_col]].drop_duplicates().sort_values([group_col, dt_col], ignore_index=True)
    bucket["_prev_dt"] = bucket.groupby(group_col, observed=True)[dt_col].shift(1)

    merged = df.merge(bucket, on=[group_col, dt_col], how="left")
    merged.index = df.index
    return (merged[dt_col] - merged["_prev_dt"]).rename("time_since_last")
