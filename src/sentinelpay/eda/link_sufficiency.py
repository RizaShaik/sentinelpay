"""Phase E.1: non-target link/relationship sufficiency and causal cross-key
fan-out measurement.

Purpose: for each candidate directional relationship between two node types
(e.g. "does payment_node have unusually wide device_node fan-out"), measure
whether there is enough strictly-causal cross-key structure to be worth
building a future Phase E.2 ring/fraud-network mechanism on. This module
answers that question with measurement only:

- **No Union-Find, no connected components, no multi-hop/transitive graph
  traversal, no persistent graph state.** The only things here that look
  beyond a single anchor are `overlap_diagnostic` (M5) and
  `frequency_adjusted_overlap_diagnostic` (M5b), both bounded and non-recursive, computing
  at most one pairwise set-intersection per pair of high-fan-out anchors --
  see their docstrings for the exact causal construction.
- **No `isFraud` of any kind.** Nothing in this module accepts, reads, or is
  tested against a target column. There is no diagnostic-evaluation step at
  all (stronger than Phase D, which reads `isFraud` exactly once, downstream
  of every score) -- matching Phase D.1's own precedent exactly.
- **No scoring, no flags, no persistence.** `recommend_relationships` is a
  pure function of measured results, not a preference decided in advance.

**M5 vs. M5b** (added after a critical review of M5's real-data results):
`overlap_diagnostic` (M5) reports RAW partner-set overlap among high-fan-out
anchors -- retained unchanged, for descriptive/contextual reporting only.
Real-data review showed raw overlap is dominated by population-generic
partner values (the most common browser/OS strings, the most common email
domains) and does not by itself evidence coordinated multi-hop structure.
`frequency_adjusted_overlap_diagnostic` (M5b) is the diagnostic that DRIVES
`recommend_union_find_for_e2`: it compares observed overlap against a
population-prevalence null baseline and reports the excess (`lift_ratio`)
beyond what population-wide popularity alone would predict. M5b also fixes a
second, independent problem M5 has for small anchor populations: M5's
percentile-VALUE anchor selection degenerates to a single anchor whenever
`n_anchors_total` is small (e.g. `P_emaildomain`'s 59 values) regardless of
the underlying distribution shape; M5b selects anchors by a rank-based target
COUNT with a predeclared floor instead. See `frequency_adjusted_overlap_diagnostic`'s
docstring for both fixes in full.

**Offline measurement, not an online feature** (documentation correction from
the M5b review): every partner set here -- both M5's and M5b's -- is strictly
causal RELATIVE TO EACH ANCHOR'S OWN CUTOFF (that anchor's own last `dt_col`
within the measured development window); this per-anchor guarantee is
unchanged and still fully tested. `frequency_adjusted_overlap_diagnostic`'s population
PREVALENCE baseline, however, is an AGGREGATE over every anchor in the
relationship, each with its own different cutoff -- it is a fixed snapshot of
this run's development window, not a claim that recomputing it after some
change elsewhere in the dataset (e.g. a new anchor's future transactions)
would leave it unchanged, and not a claim that it could be recomputed
incrementally as new rows arrive. E.1 as a whole is an offline sufficiency
measurement over a fixed development window, not an online/streaming feature
computation -- this distinction applies throughout this phase, not only to
the new prevalence baseline.

Causal correctness is not re-derived here: strictly-prior counting and
distinct-partner fan-out are delegated to
`sentinelpay.data.history.prior_group_distinct_other_count` /
`prior_group_windowed_distinct_other_count` (tested in tests/test_history.py)
and reused unchanged. Row-level sufficiency-at-threshold, prior-count
distribution summaries, per-partition breakdowns, and anchor-side
dominant-group exclusion sensitivity are reused unchanged from
`sentinelpay.eda.grouping_key_sufficiency` (D.1's own module) -- this module
does not re-implement any of that math, only extends it to a second column.

Holdout sealing: every function here assumes its caller (`run_phase_e1`) has
already excluded holdout rows -- no function in this module knows about
partitions except as an opaque label column used purely for grouping the
report, never for filtering rows itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinelpay.data.history import (
    prior_group_distinct_other_count,
    prior_group_windowed_distinct_other_count,
)
from sentinelpay.eda.entity import group_size_distribution, group_size_summary_stats
from sentinelpay.eda.grouping_key_sufficiency import (
    DEFAULT_TOP_K_LIST,
    dominant_group_exclusion_sensitivity,
    prior_count_distribution,
    sufficiency_at_thresholds,
    sufficiency_by_partition,
)

# This phase's fixed, pre-declared decision criteria (module constants, not a
# new YAML -- matching D.1's own precedent of no dedicated config file for a
# pure sufficiency phase). Reasoned by analogy to D.1's constants but NOT
# inherited by default where the underlying question differs -- see the
# proposal review for the justification of each value below.
FANOUT_THRESHOLDS = [1, 2, 3, 5, 10]
WINDOW_CANDIDATES = [5, 10, 20, 50]

# The minimal bar for "this anchor has ever shown more than one partner" --
# the necessary precondition for any ring-like structure. Lower than D.1's
# raw-event bar of 5 because distinct-partner counts are inherently smaller
# than raw event counts.
FANOUT_DECISION_THRESHOLD = 2

# D.1's 50% row-coverage bar answers "can this key stand alone as a
# population-wide per-entity key" -- a different question. A relationship
# doesn't need population-wide coverage to be worth building on; it needs a
# non-trivial population to measure fan-out over.
MIN_RELATIONSHIP_ROW_COVERAGE_PCT = 10.0

# D.1's own numbers show most payment_proxy_key groups are small (14,786 of
# 37,149 groups are singletons); expecting a majority of anchors to show
# >= 2 distinct partners is an unreasonably high bar for a rarer phenomenon
# than "has >= 5 raw prior events." A minority showing real fan-out is
# enough population to justify a scorer.
MIN_FANOUT_SUFFICIENT_PCT = 10.0

# Relative-consistency checks (reused from D.1 unchanged -- their meaning,
# "is this subset's rate at least half the reference rate," transfers as-is).
MIN_PARTITION_RELATIVE_PCT = 50.0
MIN_DOMINANT_ADJUSTED_RETENTION_PCT = 50.0

# M5 overlap diagnostic constants (unchanged; M5 is retained for descriptive
# reporting only -- see module docstring "M5 vs. M5b").
OVERLAP_HIGH_FANOUT_PERCENTILE = 99.0  # top 1% of a relationship's own per-anchor fan-out distribution
OVERLAP_MULTI_HOP_SIGNAL_PCT = 20.0  # if >= 20% of high-fan-out anchors' partners overlap another high-fan-out anchor's, structure plausibly extends beyond one hop

# M5b frequency-adjusted overlap diagnostic constants -- this is the diagnostic that
# drives recommend_union_find_for_e2 (see recommend_relationships).
#
# HIGH_FANOUT_PERCENTILE/HIGH_FANOUT_MIN_ANCHOR_COUNT fix M5's percentile-
# VALUE selection degeneracy: "top 1% of anchors" is a target COUNT here,
# not a value threshold, so it no longer collapses to 1 anchor whenever
# n_anchors_total is small (P_emaildomain's 59 values, in this project's
# real data). 10 is reused from DEFAULT_TOP_K_LIST's own largest value
# (consistency, not a new arbitrary number) and is the smallest count that
# makes an H x H pairwise-overlap statistic minimally meaningful (45
# unordered pairs at H=10; none at H=1). See select_high_fanout_anchors.
HIGH_FANOUT_PERCENTILE = 99.0
HIGH_FANOUT_MIN_ANCHOR_COUNT = 10

# LIFT_EPSILON guards the observed/expected division when the population-
# prevalence null predicts near-zero expected overlap (a genuinely rare
# partner value) -- same numerical-safety-floor role as detection.py's
# zero_mad_epsilon, not a sensitivity knob.
LIFT_EPSILON = 0.5

# The bar for "genuine excess beyond population-popularity expectation":
# observed overlap must be at least double what population-wide reuse of
# these particular partner values alone would predict. A round, predeclared
# starting value reasoned the same way OVERLAP_MULTI_HOP_SIGNAL_PCT
# originally was -- fixed before looking at what this run's real lift values
# turn out to be, not fit to any particular direction's outcome.
LIFT_SIGNAL_THRESHOLD = 2.0


def build_node_key_column(df: pd.DataFrame, key_columns: list[str]) -> pd.Series:
    """Row-PRESERVING join of `key_columns` into one hashable node-id string
    per row: NaN wherever any component is missing, else the joined id.

    Unlike `grouping_key_sufficiency.build_group_key`, this never drops
    rows: a row missing `payment_node`'s components must remain available
    (as NaN in that column) for `device_node`/`email_purchaser_node`
    relationships that don't need it. Each relationship direction determines
    its own valid subset independently, via `build_relationship_frame`.
    """
    for col in key_columns:
        if col not in df.columns:
            raise ValueError(f"build_node_key_column requires column '{col}'")
    valid_mask = df[key_columns].notna().all(axis=1)
    joined = df[key_columns].astype(str).agg("|".join, axis=1)
    return joined.where(valid_mask)


def build_relationship_frame(df: pd.DataFrame, anchor_col: str, other_col: str) -> pd.DataFrame:
    """Rows of `df` where both `anchor_col` and `other_col` are non-null.

    Symmetric in `anchor_col`/`other_col` -- the same valid frame is correct
    for both directions of a relationship pair (e.g. payment_node ->
    device_node and device_node -> payment_node share the exact same set of
    valid rows; only which column is the "anchor" differs).
    """
    for col in (anchor_col, other_col):
        if col not in df.columns:
            raise ValueError(f"build_relationship_frame requires column '{col}'")
    return df.dropna(subset=[anchor_col, other_col]).copy()


def other_side_top_k_concentration(
    valid_df: pd.DataFrame, other_col: str, top_k_list: list[int] = DEFAULT_TOP_K_LIST
) -> list[dict]:
    """For each `top_k` in `top_k_list`: the % of (anchor, other) pairs
    (i.e. valid rows) whose `other_col` value is one of the `top_k` most
    frequent `other_col` values in `valid_df`.

    This is the reverse-direction hub check M4 needs that D.1 never did --
    D.1 only ever had one column per key. Exposes e.g. a device_node value
    so generic that its huge "distinct payment_nodes" count is population
    noise, not ring evidence (the failure Phase A already flagged for
    device_proxy_key's largest group).
    """
    if other_col not in valid_df.columns:
        raise ValueError(f"other_side_top_k_concentration requires column '{other_col}'")
    n_valid = len(valid_df)
    value_counts = valid_df[other_col].value_counts()
    rows = []
    for top_k in top_k_list:
        top_values = set(value_counts.nlargest(top_k).index)
        n_touching = int(valid_df[other_col].isin(top_values).sum())
        rows.append(
            {
                "top_k": top_k,
                "top_other_values_row_count": int(value_counts.nlargest(top_k).sum()),
                "pct_pairs_touching_top_k_other_values": round(100.0 * n_touching / n_valid, 4) if n_valid else float("nan"),
            }
        )
    return rows


def analyze_relationship_direction(
    valid_df: pd.DataFrame,
    anchor_col: str,
    other_col: str,
    dt_col: str,
    partition_col: str,
    partitions: list[str],
    thresholds: list[int] = FANOUT_THRESHOLDS,
    window_candidates: list[int] = WINDOW_CANDIDATES,
    top_k_list: list[int] = DEFAULT_TOP_K_LIST,
) -> dict:
    """M1/M2/M4 for one directional relationship (`anchor_col` -> `other_col`)
    over an already-holdout-sealed, already-both-columns-valid `valid_df`.

    Never reads or requires `isFraud`. `valid_df` must already be the output
    of `build_relationship_frame` (or equivalent) -- this function does not
    filter for missing `anchor_col`/`other_col` itself, since M3 (relationship
    row coverage) needs to be computed once against the full development
    frame, not against the already-filtered subset this function receives.
    """
    n_valid = len(valid_df)

    unbounded = prior_group_distinct_other_count(valid_df, group_col=anchor_col, other_col=other_col, dt_col=dt_col)

    windowed_results: dict[int, dict] = {}
    for w in window_candidates:
        windowed = prior_group_windowed_distinct_other_count(
            valid_df, group_col=anchor_col, other_col=other_col, dt_col=dt_col, window_size_events=w
        )
        w_counts = windowed["prior_distinct_other_count_in_window"]
        windowed_results[w] = {
            "prior_count_distribution": prior_count_distribution(w_counts),
            "sufficiency_overall": sufficiency_at_thresholds(w_counts, thresholds),
            "sufficiency_by_partition": sufficiency_by_partition(valid_df, partition_col, w_counts, partitions, thresholds),
        }

    anchor_size_stats = group_size_summary_stats(valid_df, proxy_key_columns=[anchor_col])
    anchor_size_distribution = group_size_distribution(valid_df, proxy_key_columns=[anchor_col])

    return {
        "anchor_col": anchor_col,
        "other_col": other_col,
        "n_rows_valid": n_valid,
        "n_anchor_groups": anchor_size_stats["n_groups"],
        "n_anchor_singleton_groups": anchor_size_stats["n_singleton_groups"],
        "max_anchor_group_size": anchor_size_stats["max_group_size"],
        "median_anchor_group_size": anchor_size_stats["median_group_size"],
        "anchor_group_size_distribution": anchor_size_distribution.to_dict(orient="records"),
        "unbounded": {
            "prior_count_distribution": prior_count_distribution(unbounded),
            "sufficiency_overall": sufficiency_at_thresholds(unbounded, thresholds),
            "sufficiency_by_partition": sufficiency_by_partition(valid_df, partition_col, unbounded, partitions, thresholds),
            "dominant_anchor_exclusion_sensitivity": dominant_group_exclusion_sensitivity(
                valid_df, anchor_col, unbounded, top_k_list, n_development_total=n_valid, thresholds=thresholds
            ),
        },
        "windowed": windowed_results,
        "other_side_dominant_concentration": {
            "other_group_size_summary_stats": group_size_summary_stats(valid_df, proxy_key_columns=[other_col]),
            "top_k_concentration": other_side_top_k_concentration(valid_df, other_col, top_k_list),
        },
    }


def relationship_row_coverage(valid_df: pd.DataFrame, n_development_total: int, partition_col: str, partitions: list[str]) -> dict:
    """M3: % of development rows with both relationship columns non-null,
    overall and by partition. `valid_df` must already be the output of
    `build_relationship_frame` (both columns present on every row)."""
    n_valid = len(valid_df)
    by_partition = []
    for name in partitions:
        n_part = int((valid_df[partition_col] == name).sum())
        by_partition.append({"partition": name, "n_valid_rows": n_part})
    return {
        "n_rows_valid": n_valid,
        "n_rows_development": n_development_total,
        "pct_rows_valid": round(100.0 * n_valid / n_development_total, 4) if n_development_total else float("nan"),
        "by_partition": by_partition,
    }


def overlap_diagnostic(
    valid_df: pd.DataFrame,
    anchor_col: str,
    other_col: str,
    dt_col: str,
    high_fanout_percentile: float = OVERLAP_HIGH_FANOUT_PERCENTILE,
) -> dict:
    """M5: bounded, non-recursive one-hop overlap diagnostic -- NOT a graph
    traversal, builds no persistent structure, never chains through a third
    anchor.

    Causal construction (see conversation review for the full justification):

    1. Per-row causal fan-out is `prior_group_distinct_other_count` (M1's
       primitive) -- for row i, the count of distinct `other_col` values
       among strictly-prior (ties collapsed, no future leakage) same-anchor
       rows.
    2. Each anchor `a`'s fan-out summary is the MAXIMUM of its own rows'
       values from step 1, which (since that quantity is non-decreasing
       along an anchor's chronologically-sorted rows) equals the value at
       `a`'s own chronologically last row -- a fact about `a`'s own history
       only, never dependent on any other anchor.
    3. `a`'s partner set is the set of distinct `other_col` values seen at
       any of `a`'s rows with `dt_col` strictly less than `a`'s own last
       `dt_col` -- its size equals step 2's fan-out value by construction.
    4. High-fan-out anchors are those at or above `high_fanout_percentile`
       of the PER-ANCHOR (one value per anchor, not per-row) fan-out
       distribution within this relationship.
    5. For each unordered pair of distinct high-fan-out anchors, at most one
       set intersection of their partner sets is computed -- never chained
       through a third anchor.
    6. Tie/future guarantees are inherited directly from
       `prior_group_distinct_other_count`'s own invariants, since every
       partner set is built from its output: no anchor's set ever uses a row
       dated at or after THAT anchor's own cutoff, and holdout rows never
       reach this function at all (excluded by the caller before
       `build_relationship_frame`).

    Returns the actual percentile cutoff value and selected-anchor count
    alongside the overlap statistic, so a degenerate case (e.g. too few
    distinct anchor values to form a meaningful top percentile) is visible
    in the report rather than silently hidden.
    """
    for col in (anchor_col, other_col, dt_col):
        if col not in valid_df.columns:
            raise ValueError(f"overlap_diagnostic requires column '{col}'")

    prior_distinct = prior_group_distinct_other_count(valid_df, group_col=anchor_col, other_col=other_col, dt_col=dt_col)

    working = valid_df[[anchor_col, dt_col, other_col]].copy()
    working["_prior_distinct"] = prior_distinct.to_numpy()

    # Per-anchor fan-out summary (step 2): the max prior-distinct value seen
    # across each anchor's own rows.
    anchor_fanout = working.groupby(anchor_col, observed=True)["_prior_distinct"].max()
    n_anchors_total = int(len(anchor_fanout))

    if n_anchors_total == 0:
        return {
            "n_anchors_total": 0,
            "n_high_fanout_anchors": 0,
            "high_fanout_percentile": high_fanout_percentile,
            "high_fanout_threshold_value": float("nan"),
            "overlap_fraction_mean_pct": float("nan"),
            "overlap_fraction_distribution": {},
            "clears_multi_hop_signal": False,
        }

    threshold_value = float(np.percentile(anchor_fanout.to_numpy(), high_fanout_percentile))
    high_fanout_anchors = anchor_fanout[anchor_fanout >= threshold_value].index.tolist()
    n_high_fanout = len(high_fanout_anchors)

    # Partner set per anchor (step 3): distinct other_col values among the
    # anchor's own rows strictly before the anchor's own last dt_col.
    last_dt = working.groupby(anchor_col, observed=True)[dt_col].max()
    partner_sets: dict = {}
    for a in high_fanout_anchors:
        sub = working[working[anchor_col] == a]
        cutoff = last_dt.loc[a]
        prior_rows = sub[sub[dt_col] < cutoff]
        partner_sets[a] = set(prior_rows[other_col].dropna().tolist())

    overlap_fractions = []
    for a1 in high_fanout_anchors:
        p1 = partner_sets[a1]
        if not p1:
            continue
        overlapping = set()
        for a2 in high_fanout_anchors:
            if a2 == a1:
                continue
            overlapping |= p1 & partner_sets[a2]
        overlap_fractions.append(len(overlapping) / len(p1))

    if overlap_fractions:
        arr = np.array(overlap_fractions) * 100.0
        distribution = {
            "min": float(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }
        overlap_mean_pct = float(arr.mean())
    else:
        distribution = {}
        overlap_mean_pct = float("nan")

    return {
        "n_anchors_total": n_anchors_total,
        "n_high_fanout_anchors": n_high_fanout,
        "high_fanout_percentile": high_fanout_percentile,
        "high_fanout_threshold_value": threshold_value,
        "overlap_fraction_mean_pct": overlap_mean_pct,
        "overlap_fraction_distribution": distribution,
        "clears_multi_hop_signal": bool(overlap_mean_pct >= OVERLAP_MULTI_HOP_SIGNAL_PCT) if overlap_fractions else False,
    }


def select_high_fanout_anchors(
    anchor_fanout: pd.Series,
    percentile: float = HIGH_FANOUT_PERCENTILE,
    min_anchor_count: int = HIGH_FANOUT_MIN_ANCHOR_COUNT,
) -> list:
    """Rank-COUNT high-fan-out anchor selection (M5b's fix for M5's
    percentile-VALUE degeneracy on small anchor populations).

    `n_target = max(min_anchor_count, ceil((100-percentile)/100 * n_anchors_total))`,
    clamped to the number of anchors with nonzero fan-out (never pads the
    selection with zero-information anchors), with every anchor tied at the
    selection-boundary rank included -- the same "never split a tie to hit an
    exact target size" rule `prior_group_windowed_robust_stats` already uses
    for its window-boundary bucket. Unlike M5's `high_fanout_percentile`
    argument (a VALUE threshold via `np.percentile`), this selects a target
    anchor COUNT directly, so "top 1%" no longer collapses to a single anchor
    whenever `n_anchors_total` is small (e.g. `P_emaildomain`'s 59 values) --
    that collapse is a property of small-n percentile arithmetic in general,
    not specific to any one relationship's distribution shape.

    Returns anchor labels only (not a DataFrame/Series) -- callers already
    hold `anchor_fanout` and any per-anchor data they need.
    """
    n_anchors_total = len(anchor_fanout)
    if n_anchors_total == 0:
        return []
    n_nonzero = int((anchor_fanout > 0).sum())
    if n_nonzero == 0:
        return []
    n_target = max(min_anchor_count, int(np.ceil((100.0 - percentile) / 100.0 * n_anchors_total)))
    n_target = min(n_target, n_nonzero, n_anchors_total)

    sorted_desc = anchor_fanout.sort_values(ascending=False)
    cutoff_value = sorted_desc.iloc[n_target - 1]
    selected = anchor_fanout[(anchor_fanout >= cutoff_value) & (anchor_fanout > 0)].index.tolist()
    return selected


def compute_partner_prevalence(valid_df: pd.DataFrame, anchor_col: str, other_col: str, dt_col: str) -> tuple[pd.Series, int]:
    """Full-population anchor-level partner prevalence: for every value `v`
    of `other_col`, the fraction of ALL anchors in this relationship (not
    just a high-fan-out subset) whose partner set contains `v`.

    Uses the EXACT SAME per-anchor partner-set construction as
    `overlap_diagnostic`'s step 3 (distinct `other_col` values seen at any of
    an anchor's rows with `dt_col` strictly less than that anchor's own last
    `dt_col`) -- extended here to every anchor rather than just a high-fan-out
    subset. An anchor's contribution to `prevalence(v)` depends only on that
    anchor's own rows strictly before that anchor's own last `dt_col`, never
    on any other anchor's rows and never on a row at or after its own
    cutoff -- the identical per-anchor causal guarantee `overlap_diagnostic`
    already carries and tests/test_history.py already proves for the
    underlying primitive.

    This is an aggregate, whole-of-development-window population statistic,
    not an online/streaming feature: it is not computed leave-one-out (a
    high-fan-out anchor being tested also contributes to its own partners'
    prevalence -- documented, bounded approximation; negligible for large
    anchor populations, up to ~1/n_anchors_total per anchor for small ones),
    and it is not claimed to be invariant to changes on other anchors -- see
    the module docstring's "Offline measurement, not an online feature" note.

    Returns `(prevalence, n_anchors_total)` where `prevalence` is a Series
    indexed by `other_col` value.
    """
    for col in (anchor_col, other_col, dt_col):
        if col not in valid_df.columns:
            raise ValueError(f"compute_partner_prevalence requires column '{col}'")

    working = valid_df[[anchor_col, dt_col, other_col]].dropna(subset=[other_col]).copy()
    n_anchors_total = int(valid_df[anchor_col].dropna().nunique())
    if n_anchors_total == 0 or working.empty:
        return pd.Series(dtype="float64"), n_anchors_total

    last_dt = working.groupby(anchor_col, observed=True)[dt_col].max()
    # Series.map() on a categorical-dtype anchor_col (e.g. P_emaildomain)
    # preserves categorical structure even though the mapped VALUES here are
    # numeric dt values -- the result would be an unordered Categorical
    # column, which then fails on the `<` comparison below ("Unordered
    # Categoricals can only compare equality or not"). Casting anchor_col to
    # object before mapping avoids that categorical-preserving special case;
    # the lookup itself is equality-based and unaffected by the cast.
    working["_anchor_last_dt"] = working[anchor_col].astype(object).map(last_dt)
    prior_rows = working[working[dt_col] < working["_anchor_last_dt"]]
    pairs = prior_rows[[anchor_col, other_col]].drop_duplicates()

    counts = pairs[other_col].value_counts()
    prevalence = counts / n_anchors_total
    return prevalence, n_anchors_total


def _empty_frequency_adjusted_overlap_result(percentile: float, min_anchor_count: int) -> dict:
    return {
        "n_anchors_total": 0,
        "n_high_fanout_anchors": 0,
        "n_anchors_with_nonempty_partner_set": 0,
        "percentile": percentile,
        "min_anchor_count": min_anchor_count,
        "mean_observed_overlap_pct": float("nan"),
        "mean_expected_overlap_pct": float("nan"),
        "excess_pct_points": float("nan"),
        "lift_ratio": float("nan"),
        "clears_lift_signal": False,
    }


def frequency_adjusted_overlap_diagnostic(
    valid_df: pd.DataFrame,
    anchor_col: str,
    other_col: str,
    dt_col: str,
    percentile: float = HIGH_FANOUT_PERCENTILE,
    min_anchor_count: int = HIGH_FANOUT_MIN_ANCHOR_COUNT,
) -> dict:
    """M5b: frequency-adjusted overlap diagnostic. This is the diagnostic that DRIVES
    `recommend_union_find_for_e2` -- `overlap_diagnostic` (M5) is retained
    unchanged, for descriptive/contextual reporting only (see module
    docstring "M5 vs. M5b").

    Question: "do high-fan-out anchors share partners more often than the
    population-wide popularity of those partner values alone would predict?"
    -- not "do high-fan-out anchors' partner sets overlap at all," which raw
    M5 answers but which real-data review showed is dominated by
    population-generic partner values (the most common browser/OS strings,
    the most common email domains) rather than evidence of coordinated
    structure.

    Fixes, relative to M5:
    1. Anchor selection is rank-COUNT based with a predeclared floor
       (`select_high_fanout_anchors`), not percentile-VALUE based -- avoids
       M5's degeneracy to a single anchor for small `n_anchors_total`.
    2. Observed overlap is compared against a population-prevalence null
       baseline (`compute_partner_prevalence`) instead of reported as a raw,
       reference-free percentage.

    Causal construction: identical partner-set definition as M5 (see that
    function's docstring, steps 1-3) for both the selected high-fan-out
    anchors AND the full-population prevalence baseline. Partner sets are
    strictly causal relative to EACH ANCHOR'S OWN cutoff; the prevalence
    baseline is a whole-of-development-window aggregate over many anchors'
    (each different) cutoffs -- see module docstring "Offline measurement,
    not an online feature" for the precise, deliberately narrower claim this
    aggregate does and does not make.

    Null model: each of an anchor's `H-1` other high-fan-out peers is treated
    as an independent Bernoulli trial with success probability
    `prevalence(v)` for each of that anchor's own actual partner values `v`
    -- a standard association-strength approximation, not an exact
    combinatorial (hypergeometric) or permutation null (a permutation test
    was considered and rejected on computational-cost grounds during the
    M5b proposal review). Expected overlap for anchor `a1`:

        E[overlap_fraction(a1)] = mean over v in partner_set(a1) of
                                   (1 - (1 - prevalence(v)) ** (H - 1))

    Aggregation is ratio-of-means, not mean-of-ratios: `mean_observed_overlap_pct`
    and `mean_expected_overlap_pct` are each averaged across the H anchors
    first, then divided -- avoids a handful of anchors with near-zero
    expected overlap producing unstable, dominating individual lift values.

    Bounded and non-recursive, exactly like M5: at most one pairwise set
    intersection per pair of high-fan-out anchors, no persistent structure,
    no chaining through a third anchor. Never reads or requires `isFraud`.

    Returns `mean_observed_overlap_pct`, `mean_expected_overlap_pct`,
    `excess_pct_points`, `lift_ratio`, and `clears_lift_signal`
    (`lift_ratio >= LIFT_SIGNAL_THRESHOLD`) alongside the selection
    diagnostics (`n_anchors_total`, `n_high_fanout_anchors`,
    `n_anchors_with_nonempty_partner_set`) so a degenerate case is visible
    rather than silently hidden, matching M5's own transparency precedent.
    """
    for col in (anchor_col, other_col, dt_col):
        if col not in valid_df.columns:
            raise ValueError(f"frequency_adjusted_overlap_diagnostic requires column '{col}'")

    prior_distinct = prior_group_distinct_other_count(valid_df, group_col=anchor_col, other_col=other_col, dt_col=dt_col)
    working = valid_df[[anchor_col, dt_col, other_col]].copy()
    working["_prior_distinct"] = prior_distinct.to_numpy()
    anchor_fanout = working.groupby(anchor_col, observed=True)["_prior_distinct"].max()
    n_anchors_total = int(len(anchor_fanout))

    if n_anchors_total == 0:
        return _empty_frequency_adjusted_overlap_result(percentile, min_anchor_count)

    high_fanout_anchors = select_high_fanout_anchors(anchor_fanout, percentile=percentile, min_anchor_count=min_anchor_count)
    n_high_fanout = len(high_fanout_anchors)

    if n_high_fanout < 2:
        result = _empty_frequency_adjusted_overlap_result(percentile, min_anchor_count)
        result["n_anchors_total"] = n_anchors_total
        result["n_high_fanout_anchors"] = n_high_fanout
        return result

    last_dt = working.groupby(anchor_col, observed=True)[dt_col].max()
    partner_sets: dict = {}
    for a in high_fanout_anchors:
        sub = working[working[anchor_col] == a]
        cutoff = last_dt.loc[a]
        prior_rows = sub[sub[dt_col] < cutoff]
        partner_sets[a] = set(prior_rows[other_col].dropna().tolist())

    prevalence, _ = compute_partner_prevalence(valid_df, anchor_col=anchor_col, other_col=other_col, dt_col=dt_col)

    n_peers = n_high_fanout - 1
    observed_fractions = []
    expected_fractions = []
    for a1 in high_fanout_anchors:
        p1 = partner_sets[a1]
        if not p1:
            continue
        overlapping = set()
        for a2 in high_fanout_anchors:
            if a2 == a1:
                continue
            overlapping |= p1 & partner_sets[a2]
        observed_fractions.append(len(overlapping) / len(p1))

        expected_sum = 0.0
        for v in p1:
            prev_v = float(prevalence.get(v, 0.0))
            expected_sum += 1.0 - (1.0 - prev_v) ** n_peers
        expected_fractions.append(expected_sum / len(p1))

    if not observed_fractions:
        result = _empty_frequency_adjusted_overlap_result(percentile, min_anchor_count)
        result["n_anchors_total"] = n_anchors_total
        result["n_high_fanout_anchors"] = n_high_fanout
        return result

    mean_observed_pct = 100.0 * float(np.mean(observed_fractions))
    mean_expected_pct = 100.0 * float(np.mean(expected_fractions))
    excess_pct_points = mean_observed_pct - mean_expected_pct
    lift_ratio = mean_observed_pct / max(mean_expected_pct, LIFT_EPSILON)

    return {
        "n_anchors_total": n_anchors_total,
        "n_high_fanout_anchors": n_high_fanout,
        "n_anchors_with_nonempty_partner_set": len(observed_fractions),
        "percentile": percentile,
        "min_anchor_count": min_anchor_count,
        "mean_observed_overlap_pct": mean_observed_pct,
        "mean_expected_overlap_pct": mean_expected_pct,
        "excess_pct_points": excess_pct_points,
        "lift_ratio": lift_ratio,
        "clears_lift_signal": bool(lift_ratio >= LIFT_SIGNAL_THRESHOLD),
    }


def recommend_window_size(
    direction_result: dict,
    thresholds: list[int] = FANOUT_THRESHOLDS,
    decision_threshold: int = FANOUT_DECISION_THRESHOLD,
    window_candidates: list[int] = WINDOW_CANDIDATES,
    min_partition_relative_pct: float = MIN_PARTITION_RELATIVE_PCT,
) -> int | None:
    """Largest candidate window in `window_candidates` whose windowed
    sufficiency-at-`decision_threshold` stays within `min_partition_relative_pct`
    of the unbounded sufficiency -- the same "largest pre-declared bucket
    that's still majority-stable" logic D.1's own reasoning used to justify
    `window_size_events=20` for Phase D. Returns `None` ("unbounded only")
    if no candidate window clears the bar.

    Scope simplification, documented explicitly rather than silently
    diverging from the reviewed proposal: this checks stability against the
    unbounded sufficiency only, not a full per-window recomputation of M4's
    dominant-anchor-exclusion sensitivity (that would require re-running the
    O(top_k) exclusion sweep for every candidate window on every direction --
    a materially larger, separately-reviewable unit of work). The unbounded
    direction's own `dominant_anchor_exclusion_sensitivity` (M4) already
    gates `is_suitable` in `evaluate_relationship_sufficiency`, so an
    anchor-concentration problem still blocks a direction from reaching this
    function at all.
    """
    unbounded_pct = direction_result["unbounded"]["sufficiency_overall"][decision_threshold]
    if unbounded_pct == 0:
        return None
    best: int | None = None
    for w in sorted(window_candidates):
        windowed_pct = direction_result["windowed"][w]["sufficiency_overall"][decision_threshold]
        if windowed_pct >= unbounded_pct * min_partition_relative_pct / 100.0:
            best = w
    return best


def evaluate_relationship_sufficiency(
    direction_result: dict,
    coverage_result: dict,
    decision_threshold: int = FANOUT_DECISION_THRESHOLD,
    min_row_coverage_pct: float = MIN_RELATIONSHIP_ROW_COVERAGE_PCT,
    min_fanout_sufficient_pct: float = MIN_FANOUT_SUFFICIENT_PCT,
    min_partition_relative_pct: float = MIN_PARTITION_RELATIVE_PCT,
    min_dominant_adjusted_retention_pct: float = MIN_DOMINANT_ADJUSTED_RETENTION_PCT,
) -> dict:
    """Apply this phase's fixed, pre-declared pass/fail criteria to one
    `analyze_relationship_direction` result -- mirrors D.1's
    `evaluate_key_sufficiency` shape exactly (AND-of-checks), applied per
    direction: payment_node -> device_node and device_node -> payment_node
    get independent verdicts, since M4 can fail for different reasons on
    each side."""
    key_str = f"pct_sufficient_ge_{decision_threshold}"

    row_coverage_pct = coverage_result["pct_rows_valid"]
    row_coverage_ok = row_coverage_pct >= min_row_coverage_pct

    overall_pct = direction_result["unbounded"]["sufficiency_overall"][decision_threshold]
    fanout_density_ok = overall_pct >= min_fanout_sufficient_pct

    partition_pcts = {
        row["partition"]: row[key_str] for row in direction_result["unbounded"]["sufficiency_by_partition"]
    }
    partition_ok = all(
        (overall_pct == 0) or (pct >= overall_pct * min_partition_relative_pct / 100.0)
        for pct in partition_pcts.values()
    )

    dominant_rows = {
        row["top_k_excluded"]: row[key_str]
        for row in direction_result["unbounded"]["dominant_anchor_exclusion_sensitivity"]
    }
    worst_case_after_exclusion = min(dominant_rows.values()) if dominant_rows else overall_pct
    dominant_ok = (overall_pct == 0) or (
        worst_case_after_exclusion >= overall_pct * min_dominant_adjusted_retention_pct / 100.0
    )

    return {
        "decision_threshold": decision_threshold,
        "row_coverage_pct": row_coverage_pct,
        "row_coverage_ok": row_coverage_ok,
        "overall_fanout_sufficiency_pct": overall_pct,
        "fanout_density_ok": fanout_density_ok,
        "partition_stability_ok": partition_ok,
        "partition_sufficiency_pct": partition_pcts,
        "dominant_anchor_robustness_ok": dominant_ok,
        "worst_case_sufficiency_pct_after_exclusion": worst_case_after_exclusion,
        "is_suitable": bool(row_coverage_ok and fanout_density_ok and partition_ok and dominant_ok),
    }


def recommend_relationships(all_directions: dict[str, dict]) -> dict:
    """Pure function over every evaluated direction's
    {"evaluation": ..., "overlap": ..., "frequency_adjusted_overlap": ...} results
    (mirrors `grouping_key_sufficiency.recommend_grouping_key`'s discipline:
    a recommendation that's a function of measured results, not a preference
    decided in advance).

    `all_directions` maps a direction key (e.g. "payment_to_device") to a
    dict with keys "evaluation" (an `evaluate_relationship_sufficiency`
    result), "overlap" (an `overlap_diagnostic`/M5 result -- DESCRIPTIVE
    ONLY, does not drive the recommendation below), "frequency_adjusted_overlap" (an
    `frequency_adjusted_overlap_diagnostic`/M5b result -- the DECISION-DRIVING
    diagnostic), and "recommended_window" (a `recommend_window_size` result,
    `None` meaning "unbounded only").

    Returns per-direction suitability + window recommendation (plus M5's raw
    overlap and M5b's lift_ratio, both for reference), and one aggregate
    boolean `recommend_union_find_for_e2`: True only if at least one USABLE
    (is_suitable) direction's M5b frequency-adjusted overlap statistic clears
    `LIFT_SIGNAL_THRESHOLD` -- i.e. "observed overlap is at least
    LIFT_SIGNAL_THRESHOLD times what population-wide popularity of these
    partner values alone would predict, so investigating Union-Find's added
    complexity for a future E.2 is evidenced," never "simple per-anchor
    fan-out is therefore invalid." Raw M5 overlap alone (however high) does
    NOT drive this boolean -- a critical review of E.1's real-data results
    showed raw overlap is dominated by population-generic partner values and
    does not by itself evidence coordinated structure; see module docstring
    "M5 vs. M5b".
    """
    per_direction = {}
    any_signal = False
    signal_directions = []
    for name, entry in all_directions.items():
        ev = entry["evaluation"]
        overlap = entry["overlap"]
        frequency_adjusted_overlap = entry["frequency_adjusted_overlap"]
        per_direction[name] = {
            "is_suitable": ev["is_suitable"],
            "recommended_window": entry.get("recommended_window"),
            "raw_overlap_pct_descriptive_only": overlap.get("overlap_fraction_mean_pct"),
            "lift_ratio": frequency_adjusted_overlap.get("lift_ratio"),
        }
        if ev["is_suitable"] and frequency_adjusted_overlap.get("clears_lift_signal"):
            any_signal = True
            signal_directions.append(name)

    if any_signal:
        reason = (
            "At least one usable relationship's M5b frequency-adjusted overlap statistic (lift_ratio) clears "
            f"LIFT_SIGNAL_THRESHOLD ({LIFT_SIGNAL_THRESHOLD}x population-popularity expectation): "
            f"{', '.join(signal_directions)}. This means high-fan-out anchors share partners "
            "measurably more often than the population-wide popularity of those partner values "
            "alone would predict -- investigating Union-Find/component structure for a future "
            "Phase E.2 is evidenced, not merely generically appealing. It does NOT mean "
            "per-anchor fan-out counts are invalid or unusable on their own, and it is NOT driven "
            "by M5's raw overlap percentage, which a critical review found to be dominated by "
            "population-generic partner values (see module docstring)."
        )
    else:
        reason = (
            "No usable relationship's M5b frequency-adjusted overlap statistic clears LIFT_SIGNAL_THRESHOLD "
            f"({LIFT_SIGNAL_THRESHOLD}x population-popularity expectation). Any high M5 raw-overlap "
            "numbers present in this run's results are population-popularity artifacts, not evidence "
            "of chaining into larger structure within one hop; on the evidence gathered here, "
            "Union-Find's added complexity is not yet justified for a future Phase E.2."
        )

    return {
        "per_direction": per_direction,
        "recommend_union_find_for_e2": any_signal,
        "reason": reason,
    }
