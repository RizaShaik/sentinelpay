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
