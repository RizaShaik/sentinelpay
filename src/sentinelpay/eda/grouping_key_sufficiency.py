"""Phase D.1: non-target grouping-key sufficiency analysis.

Purpose: establish whether `payment_proxy_key` or `device_proxy_key` has
enough strictly-causal historical density to support a future Phase D
per-entity behavioral-change detector. This module answers that question
only -- it does not implement rolling median/MAD, EWMA, CUSUM, target
encoding, fraud-rate evaluation, or any detector.

Completely non-target: nothing here accepts, reads, or depends on `isFraud`
(or any target column). Every function operates on key columns, a
timestamp column, and (for partition breakdowns) a partition label -- never
a label column. Causal correctness is not re-derived here: strictly-prior
counting is delegated to `sentinelpay.data.history.prior_group_count`,
which is already tested (tie handling, no self-count, no future leakage,
row-order independence) in tests/test_history.py.

Holdout sealing: every function here assumes its caller has already
excluded holdout rows (see sentinelpay.eda.run_phase_d1) -- no function in
this module knows about partitions except as an opaque label column used
purely for grouping the report, never for filtering rows itself.
"""
from __future__ import annotations

import pandas as pd

from sentinelpay.data.history import prior_group_count
from sentinelpay.eda.entity import group_size_distribution, group_size_summary_stats

DEFAULT_THRESHOLDS = [1, 3, 5, 10, 20]
DEFAULT_TOP_K_LIST = [1, 3, 5, 10]

# This phase's fixed decision criteria for section "recommendation" below.
# DECISION_THRESHOLD, MIN_OVERALL_SUFFICIENT_PCT, MIN_PARTITION_RELATIVE_PCT,
# and MIN_DOMINANT_ADJUSTED_RETENTION_PCT were fixed before D.1 was run
# against real data (like configs/split.yaml's boundaries).
#
# MIN_ROW_COVERAGE_PCT was added after that first real run: the initial
# criteria checked density only among rows that already have the key, so a
# key present on a small minority of transactions could still score as
# "suitable" -- a methodology gap, not a result to accept. The 50% bar
# (a majority of transactions must carry the key at all) is a round,
# domain-driven number chosen for that reason, not fit to either
# candidate's observed row-coverage figure (87.05% / 20.30%), and it has
# not been adjusted since. It is fixed for every run from here forward.
DECISION_THRESHOLD = 5  # "enough causal history" bar for this phase's verdict
MIN_ROW_COVERAGE_PCT = 50.0  # a key present on a minority of transactions cannot serve as a population-wide per-entity key, however dense its history is among the rows that do have it
MIN_OVERALL_SUFFICIENT_PCT = 50.0
MIN_PARTITION_RELATIVE_PCT = 50.0
MIN_DOMINANT_ADJUSTED_RETENTION_PCT = 50.0


def build_group_key(df: pd.DataFrame, key_columns: list[str], key_name: str = "_group_key") -> pd.DataFrame:
    """Rows of `df` where every `key_columns` component is non-null, with an
    added `key_name` column combining the components into one hashable
    group id (string-joined per row). Fully generic over `key_columns` --
    does not choose, name, or endorse any specific proxy key.

    String-joining is safe here even for numeric columns: two rows are
    compared component-by-component against the SAME column's dtype (fixed
    per column after dropna), so int-vs-float formatting never causes a
    false mismatch or false match across rows.
    """
    for col in key_columns:
        if col not in df.columns:
            raise ValueError(f"build_group_key requires column '{col}'")
    valid = df.dropna(subset=key_columns).copy()
    valid[key_name] = valid[key_columns].astype(str).agg("|".join, axis=1)
    return valid


def causal_prior_counts(df: pd.DataFrame, group_col: str, dt_col: str) -> pd.Series:
    """Strictly-causal count of same-group rows with `dt_col` strictly
    earlier, aligned to `df.index`. Thin wrapper over
    `sentinelpay.data.history.prior_group_count` -- see that function's
    docstring and tests/test_history.py for the causal-correctness
    guarantees (no self-count, tied timestamps never see each other, no
    future leakage, row-order independence)."""
    return prior_group_count(df, group_col=group_col, dt_col=dt_col)


def sufficiency_at_thresholds(prior_counts: pd.Series, thresholds: list[int] = DEFAULT_THRESHOLDS) -> dict[int, float]:
    """% of rows in `prior_counts` with a strictly-prior-event count >= each
    threshold. Denominator is `len(prior_counts)` -- pass exactly the rows
    you want the percentage taken over (e.g. a single partition's subset)."""
    n = len(prior_counts)
    if n == 0:
        return {t: float("nan") for t in thresholds}
    return {t: round(100.0 * int((prior_counts >= t).sum()) / n, 4) for t in thresholds}


def prior_count_distribution(prior_counts: pd.Series) -> dict:
    """Scalar distribution summary of strictly-prior event counts."""
    if len(prior_counts) == 0:
        return {}
    q = prior_counts.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "min": int(prior_counts.min()),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.9]),
        "p99": float(q.loc[0.99]),
        "max": int(prior_counts.max()),
        "mean": float(prior_counts.mean()),
    }


def sufficiency_by_partition(
    valid_df: pd.DataFrame,
    partition_col: str,
    prior_counts: pd.Series,
    partitions: list[str],
    thresholds: list[int] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    """Row count and per-threshold sufficiency %, one row per partition in
    `partitions` -- lets a caller see whether coverage is concentrated in
    `train` or holds up in `validation`/`embargo_2` too."""
    if partition_col not in valid_df.columns:
        raise ValueError(f"sufficiency_by_partition requires column '{partition_col}'")
    rows = []
    for name in partitions:
        mask = (valid_df[partition_col] == name).to_numpy()
        sub_counts = prior_counts[mask]
        row = {"partition": name, "n_valid_rows": int(mask.sum())}
        row.update({f"pct_sufficient_ge_{t}": v for t, v in sufficiency_at_thresholds(sub_counts, thresholds).items()})
        rows.append(row)
    return rows


def dominant_group_exclusion_sensitivity(
    valid_df: pd.DataFrame,
    group_col: str,
    prior_counts: pd.Series,
    top_k_list: list[int] = DEFAULT_TOP_K_LIST,
    n_development_total: int | None = None,
    thresholds: list[int] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    """For each `top_k` in `top_k_list`: exclude the `top_k` largest groups
    (by row count) from `valid_df`, then report the remaining valid row
    count, remaining valid-row coverage (of the original valid rows, and of
    `n_development_total` if given), and sufficiency percentages RECOMPUTED
    on the remaining rows -- so a denominator change from removing dominant
    groups is visible rather than hidden behind an unchanged-looking
    percentage.

    Quantifies (without any target statistic) whether a key's apparent
    sufficiency is concentrated in a few oversized/generic groups -- see
    Phase A's device_proxy_key finding (reports/eda/phase_a_report.md
    section 9): one generic fingerprint group held 16,406 of 118,367 valid
    rows.
    """
    if group_col not in valid_df.columns:
        raise ValueError(f"dominant_group_exclusion_sensitivity requires column '{group_col}'")
    n_valid_original = len(valid_df)
    group_sizes = valid_df[group_col].value_counts()

    rows = []
    for top_k in top_k_list:
        dominant = group_sizes.nlargest(top_k)
        keep_mask = (~valid_df[group_col].isin(set(dominant.index))).to_numpy()
        n_remaining = int(keep_mask.sum())
        remaining_counts = prior_counts[keep_mask]
        row = {
            "top_k_excluded": top_k,
            "excluded_group_sizes": [int(s) for s in dominant.tolist()],
            "n_valid_rows_remaining": n_remaining,
            "pct_valid_rows_remaining_of_original_valid": (
                round(100.0 * n_remaining / n_valid_original, 4) if n_valid_original else float("nan")
            ),
        }
        if n_development_total:
            row["pct_valid_rows_remaining_of_development_total"] = round(100.0 * n_remaining / n_development_total, 4)
        row.update(
            {f"pct_sufficient_ge_{t}": v for t, v in sufficiency_at_thresholds(remaining_counts, thresholds).items()}
        )
        rows.append(row)
    return rows


def analyze_grouping_key(
    df: pd.DataFrame,
    key_columns: list[str],
    dt_col: str,
    partition_col: str,
    partitions: list[str],
    thresholds: list[int] = DEFAULT_THRESHOLDS,
    top_k_list: list[int] = DEFAULT_TOP_K_LIST,
    key_name: str = "_group_key",
) -> dict:
    """Full D.1 sufficiency analysis for one candidate grouping key over an
    already partition-filtered (non-holdout) `df`. Never reads or requires
    `isFraud` -- only `key_columns`, `dt_col`, and `partition_col`.
    """
    n_total = len(df)
    valid = build_group_key(df, key_columns, key_name=key_name)
    n_valid = len(valid)

    prior_counts = causal_prior_counts(valid, group_col=key_name, dt_col=dt_col)
    size_stats = group_size_summary_stats(valid, proxy_key_columns=[key_name])
    size_distribution = group_size_distribution(valid, proxy_key_columns=[key_name])

    return {
        "key_columns": key_columns,
        "n_rows_development": n_total,
        "n_rows_valid": n_valid,
        "pct_rows_valid": round(100.0 * n_valid / n_total, 4) if n_total else float("nan"),
        "n_groups": size_stats["n_groups"],
        "n_singleton_groups": size_stats["n_singleton_groups"],
        "singleton_fraction_of_valid_rows": (
            round(100.0 * size_stats["n_singleton_groups"] / n_valid, 4) if n_valid else float("nan")
        ),
        "max_group_size": size_stats["max_group_size"],
        "median_group_size": size_stats["median_group_size"],
        "group_size_distribution": size_distribution.to_dict(orient="records"),
        "prior_count_distribution": prior_count_distribution(prior_counts),
        "sufficiency_overall": sufficiency_at_thresholds(prior_counts, thresholds),
        "sufficiency_by_partition": sufficiency_by_partition(valid, partition_col, prior_counts, partitions, thresholds),
        "dominant_group_exclusion_sensitivity": dominant_group_exclusion_sensitivity(
            valid, key_name, prior_counts, top_k_list, n_development_total=n_total, thresholds=thresholds
        ),
    }


def evaluate_key_sufficiency(result: dict, threshold: int = DECISION_THRESHOLD) -> dict:
    """Apply this phase's fixed, pre-declared pass/fail criteria (module
    constants above) to one `analyze_grouping_key` result. Returns the
    individual checks plus an overall `is_suitable` verdict -- a pure
    function of the measured result, not a hardcoded key choice.
    """
    key_str = f"pct_sufficient_ge_{threshold}"

    row_coverage_pct = result.get("pct_rows_valid", 100.0)
    row_coverage_ok = row_coverage_pct >= MIN_ROW_COVERAGE_PCT

    overall_pct = result["sufficiency_overall"][threshold]
    density_ok = overall_pct >= MIN_OVERALL_SUFFICIENT_PCT

    partition_pcts = {row["partition"]: row[key_str] for row in result["sufficiency_by_partition"]}
    partition_ok = all(
        (overall_pct == 0) or (pct >= overall_pct * MIN_PARTITION_RELATIVE_PCT / 100.0)
        for pct in partition_pcts.values()
    )

    dominant_rows = {row["top_k_excluded"]: row[key_str] for row in result["dominant_group_exclusion_sensitivity"]}
    worst_case_after_exclusion = min(dominant_rows.values()) if dominant_rows else overall_pct
    dominant_ok = (overall_pct == 0) or (worst_case_after_exclusion >= overall_pct * MIN_DOMINANT_ADJUSTED_RETENTION_PCT / 100.0)

    return {
        "threshold": threshold,
        "row_coverage_pct": row_coverage_pct,
        "row_coverage_ok": row_coverage_ok,
        "overall_sufficiency_pct": overall_pct,
        "coverage_ok": density_ok,
        "partition_stability_ok": partition_ok,
        "partition_sufficiency_pct": partition_pcts,
        "dominant_group_robustness_ok": dominant_ok,
        "worst_case_sufficiency_pct_after_exclusion": worst_case_after_exclusion,
        "is_suitable": bool(row_coverage_ok and density_ok and partition_ok and dominant_ok),
    }


def recommend_grouping_key(payment_eval: dict, device_eval: dict) -> dict:
    """Evidence-based recommendation among `payment_proxy_key`,
    `device_proxy_key`, `neither`, or `different_strategy` -- a pure
    function of the two evaluate_key_sufficiency() results. Does not pick a
    winner in advance: with both, neither, or only one candidate passing,
    the branch taken is entirely determined by the two `is_suitable`
    verdicts and, as a tiebreaker only, the measured overall sufficiency.
    """
    payment_ok = payment_eval["is_suitable"]
    device_ok = device_eval["is_suitable"]

    if payment_ok and device_ok:
        winner = (
            "payment_proxy_key"
            if payment_eval["overall_sufficiency_pct"] >= device_eval["overall_sufficiency_pct"]
            else "device_proxy_key"
        )
        reason = "Both candidates meet this phase's sufficiency, partition-stability, and dominant-group-robustness criteria; recommending the one with higher overall sufficiency at the decision threshold."
        return {"recommendation": winner, "reason": reason}
    if payment_ok:
        return {"recommendation": "payment_proxy_key", "reason": "Only payment_proxy_key meets this phase's criteria."}
    if device_ok:
        return {"recommendation": "device_proxy_key", "reason": "Only device_proxy_key meets this phase's criteria."}

    # Neither key is suitable as-is. Distinguish a structural coverage
    # problem (raw row coverage itself is too low -- a different key/join
    # is needed) from a density problem (coverage is fine but causal event
    # counts are too thin/unstable/dominant-group-driven -- "neither" as
    # currently defined works, but a wider embargo/history window or a
    # composite key might).
    low_coverage = (payment_eval["overall_sufficiency_pct"] == 0) or (device_eval["overall_sufficiency_pct"] == 0)
    if low_coverage:
        return {
            "recommendation": "different_strategy",
            "reason": "Neither candidate meets this phase's criteria, and at least one has zero measured sufficiency -- a different grouping strategy (e.g. a composite key) should be investigated before Phase D, not just a lower threshold.",
        }
    return {
        "recommendation": "neither",
        "reason": "Neither payment_proxy_key nor device_proxy_key meets this phase's sufficiency, partition-stability, and dominant-group-robustness criteria as currently defined.",
    }
