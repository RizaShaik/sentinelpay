"""Phase E.2 orchestration (measurement pass only): per-transaction
Union-Find / connected-component metrics for the device_node <->
payment_node relationship.

Scope: device_node <-> payment_node ONLY -- the one relationship Phase E.1's
M5b frequency-adjusted overlap diagnostic found clearing LIFT_SIGNAL_THRESHOLD
(lift_ratio 11.06x; see reports/eda/phase_e1_report.md section 5). No other
relationship, no scoring model, no persistence, no E.3 work.

This script computes and reports the four per-transaction component metrics
(`sentinelpay.eda.component_analysis.compute_component_metrics`) and their
purely descriptive distribution summary first -- every non-target quantity
(component metrics AND the device_to_payment fan-out stratification variable)
is fully computed before `isFraud` is read anywhere.

`evaluate_fanout_stratified` is then the ONE place in this script that reads
`isFraud` -- strictly downstream, `validation`-partition rows only (never
`train`/`embargo_1`/`embargo_2`, matching Phase D's `evaluate_validation_only`
precedent exactly), and never used to select, adjust, or tune
`FANOUT_STRATUM_EDGES`/`COMPONENT_SIZE_BIN_EDGES` -- both are fixed from
already-published non-target percentiles (E.1's and this same run's own
overall summary, respectively) before this function ever runs. This is a
one-time EDA evaluation only: no production feature, score, threshold,
config, or E.3 work is added by this script regardless of its result.

Holdout sealing: TransactionID/TransactionDT/payment+device columns are read
the same way `sentinelpay.eda.run_phase_e1` reads them. Rows are assigned a
partition and filtered to `DEVELOPMENT_PARTITIONS` BEFORE
`build_relationship_frame`/`compute_component_metrics` are ever called --
holdout rows never reach any component computation.

configs/split.yaml is read, never modified. No new configs/*.yaml is
introduced.

Run with:
    .venv\\Scripts\\python.exe -m sentinelpay.eda.run_phase_e2
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sentinelpay.config import DataConfig, load_config
from sentinelpay.data.history import prior_group_distinct_other_count
from sentinelpay.data.loader import load_identity, load_transaction_columns
from sentinelpay.data.split import DEVELOPMENT_PARTITIONS, assign_partition, load_split_config
from sentinelpay.data.temporal import add_day_index
from sentinelpay.eda.component_analysis import (
    DEVICE_NODE_COL,
    PAYMENT_NODE_COL,
    compute_component_metrics,
    component_metrics_summary,
    component_metrics_summary_by_partition,
)
from sentinelpay.eda.generate_report import render_phase_e2_report
from sentinelpay.eda.link_sufficiency import build_node_key_column, build_relationship_frame, relationship_row_coverage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase_e2")

FANOUT_COL = "_device_to_payment_fanout"

# Fixed stratum boundaries for the fan-out-stratified diagnostic evaluation
# below: the p25/p50/p75 of device_to_payment's own UNBOUNDED
# prior-distinct-partner-count distribution (Phase E.1's M1 quantity for
# this exact direction, over this exact valid-row population) as already
# published in reports/eda/phase_e1_report.md section 4.2. Declared here
# BEFORE evaluate_fanout_stratified reads isFraud -- these numbers come from
# E.1's own non-target measurement, not from any fraud-rate outcome this
# script's diagnostic evaluation produces.
FANOUT_STRATUM_EDGES = [274.0, 982.0, 2199.0]
FANOUT_STRATUM_LABELS = ["low_fanout_lt_p25", "mid_fanout_p25_to_p50", "mid_fanout_p50_to_p75", "high_fanout_ge_p75"]

# Fixed bucket boundaries for merged_component_size_total: this run's own
# ALREADY-PUBLISHED overall (non-target) p25/p50/p75/p90 -- see section 4 of
# reports/eda/phase_e2_report.md, generated earlier in this same run before
# this evaluation function existed. Declared here for the identical reason
# FANOUT_STRATUM_EDGES is: fixed from a non-target measurement, not fit to
# any fraud-rate outcome. Used identically across every fan-out stratum
# (never recomputed per stratum) so the same size scale is comparable
# stratum to stratum.
COMPONENT_SIZE_BIN_EDGES = [7506.0, 12241.0, 15893.0, 17925.0]
COMPONENT_SIZE_BIN_LABELS = [
    "merged_size_lt_p25",
    "merged_size_p25_to_p50",
    "merged_size_p50_to_p75",
    "merged_size_p75_to_p90",
    "merged_size_ge_p90",
]


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _roc_auc(score: pd.Series, target: pd.Series) -> float:
    """Rank-based (Mann-Whitney U) ROC-AUC -- the same formula as
    `sentinelpay.eda.run_phase_d._roc_auc`, duplicated rather than imported
    so this script's evaluation stays self-contained (run_phase_e1.py
    likewise does not import from run_phase_d.py)."""
    y = target.to_numpy()
    s = score.to_numpy()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = scipy_stats.rankdata(s)
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _component_size_bucket_fraud_rate(
    sub: pd.DataFrame,
    value_col: str,
    edges: list[float] = COMPONENT_SIZE_BIN_EDGES,
    labels: list[str] = COMPONENT_SIZE_BIN_LABELS,
) -> list[dict]:
    """Fraud rate by FIXED `value_col` bucket (see `COMPONENT_SIZE_BIN_EDGES`
    docstring) -- deliberately not `pd.qcut`: fixed, pre-declared bucket
    edges (identical across every fan-out stratum) make strata directly
    comparable on the same size scale, and guarantee the buckets were never
    adjusted after seeing this stratum's own fraud-rate outcome."""
    n = len(sub)
    bucket = pd.cut(sub[value_col], bins=[-np.inf] + edges + [np.inf], labels=labels, right=False)
    rows = []
    for label in labels:
        g = sub[bucket == label]
        rows.append(
            {
                "bucket": label,
                "n_rows": int(len(g)),
                "fraud_rate": float(g["isFraud"].mean()) if len(g) else float("nan"),
            }
        )
    return rows


def _same_component_fraud_rate(sub: pd.DataFrame) -> list[dict]:
    rows = []
    for val in [True, False]:
        g = sub[sub["endpoints_same_component"] == val]
        rows.append(
            {
                "endpoints_same_component": val,
                "n_rows": int(len(g)),
                "fraud_rate": float(g["isFraud"].mean()) if len(g) else float("nan"),
            }
        )
    return rows


def _is_monotonic_nondecreasing(rates: list[float]) -> bool:
    vals = [r for r in rates if not (isinstance(r, float) and np.isnan(r))]
    if len(vals) < 2:
        return False
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def _summarize_fanout_stratified_conclusion(fanout_strata: list[dict]) -> str:
    """Deterministic, evidence-based conclusion synthesis over
    `fanout_strata` (no new judgment call at report-render time -- the report
    only displays this string). "Gradient" is defined objectively as a
    monotonic non-decreasing `merged_component_size_total`-bucket fraud rate
    across the fixed buckets in `COMPONENT_SIZE_BIN_LABELS` order;
    "discriminatory value" as ROC-AUC > 0.5. No significance test is
    declared or implied -- this is a plain count of how many of the 4 fixed
    fan-out strata each pattern held in, on this one validation-partition
    sample."""
    n_total = len(fanout_strata)
    n_with_auc = 0
    n_auc_above_half = 0
    n_monotonic = 0
    detail_lines = []
    for s in fanout_strata:
        auc = s["roc_auc_merged_component_size_total_vs_isFraud"]
        has_auc = not (isinstance(auc, float) and np.isnan(auc))
        rates = [b["fraud_rate"] for b in s["merged_component_size_total_fraud_rate_by_bucket"]]
        mono = _is_monotonic_nondecreasing(rates)
        if has_auc:
            n_with_auc += 1
            if auc > 0.5:
                n_auc_above_half += 1
        if mono:
            n_monotonic += 1
        detail_lines.append(
            f"{s['stratum']} (n={s['n_rows']}): AUC="
            f"{'undefined (single class or 0 rows)' if not has_auc else f'{auc:.4f}'}, "
            f"bucket fraud-rate monotonic non-decreasing={mono}"
        )

    if n_with_auc == 0:
        headline = (
            "No fan-out stratum had both classes present -- ROC-AUC is undefined everywhere in this "
            "validation-partition sample. No discriminatory-value conclusion can be drawn."
        )
    else:
        headline = (
            f"ROC-AUC for merged_component_size_total vs. isFraud was computable in {n_with_auc}/{n_total} "
            f"fan-out strata; it was above 0.5 (better than chance ranking) in {n_auc_above_half}/{n_with_auc} "
            f"of those. A monotonic non-decreasing fraud-rate gradient across the fixed "
            f"merged_component_size_total buckets held in {n_monotonic}/{n_total} strata."
        )
        if n_with_auc == n_total and n_auc_above_half == n_with_auc and n_monotonic >= n_total - 1:
            headline += (
                " CONCLUSION: component structure retains a fraud-rate gradient and discriminatory value "
                "WITHIN every (or all but one) fan-out stratum -- evidence that the signal is not solely a "
                "re-detection of E.1's own device_to_payment fan-out."
            )
        elif n_auc_above_half <= n_with_auc // 2 and n_monotonic <= n_total // 2:
            headline += (
                " CONCLUSION: the pattern does not clearly persist once fan-out is controlled for -- "
                "consistent with the same population-generic-value/hub-domination confound Phase E.1's "
                "M5-vs-M5b correction found for raw overlap. Most of the unstratified signal, if any, is "
                "not shown to be beyond fan-out by this evaluation."
            )
        else:
            headline += (
                " CONCLUSION: mixed evidence across strata -- neither a clean confirmation nor a clean "
                "rejection of a beyond-fan-out component-structure signal in this validation-partition "
                "sample."
            )
    return headline + "\n\nPer-stratum detail:\n" + "\n".join(f"- {d}" for d in detail_lines)


def evaluate_fanout_stratified(scored_validation: pd.DataFrame, config: DataConfig) -> dict:
    """The pre-declared fan-out-stratified diagnostic evaluation for Phase
    E.2. Read-only, `isFraud`-reading, strictly downstream of every
    non-target quantity used here (`merged_component_size_total`,
    `endpoints_same_component`, and `_device_to_payment_fanout` are all
    already final) -- mirrors `sentinelpay.eda.run_phase_d.evaluate_validation_only`'s
    precedent exactly: requires every row already `partition == "validation"`,
    reads `isFraud` in exactly one place, never mutates a metric, never
    selects a stratum edge or any other constant from this function's own
    output.

    Question: does `merged_component_size_total`/`endpoints_same_component`
    carry fraud-rate signal BEYOND what device_to_payment's own prior fan-out
    (Phase E.1's M1 quantity) would already predict on its own -- or is it
    just re-detecting "this device/payment already has a huge generic
    fan-out," the same population-generic-value confound Phase E.1's M5-vs-M5b
    correction found for raw partner-set overlap? Section 4/5 of the E.2
    report already shows heavy hub-domination (components spanning nearly the
    whole valid population by `embargo_2`), so this confound is a live
    possibility here, not a hypothetical one.

    Stratifies validation rows into `FANOUT_STRATUM_LABELS` by `FANOUT_COL`
    using the fixed `FANOUT_STRATUM_EDGES` (E.1's own already-published
    p25/p50/p75 for this exact direction -- see module-level comment). Within
    each stratum (and once unstratified, for reference): fraud rate by
    `merged_component_size_total` decile, fraud rate by
    `endpoints_same_component`, and rank-based ROC-AUC of
    `merged_component_size_total` vs. `isFraud`. If the AUC/fraud-rate lift
    seen unstratified collapses within every stratum, that is evidence the
    unstratified signal was fan-out alone; if it persists within strata, that
    is evidence of a genuine component-structure signal beyond fan-out.
    """
    if not (scored_validation["partition"] == "validation").all():
        raise ValueError("evaluate_fanout_stratified requires every row to already be partition == 'validation'")

    isfraud = load_transaction_columns("train", columns=["TransactionID", "isFraud"], config=config)
    merged = scored_validation.merge(isfraud, on="TransactionID", how="left")
    n_validation = len(merged)

    merged["_fanout_stratum"] = pd.cut(
        merged[FANOUT_COL],
        bins=[-np.inf] + FANOUT_STRATUM_EDGES + [np.inf],
        labels=FANOUT_STRATUM_LABELS,
        right=False,
    )

    unstratified = {
        "n_rows": n_validation,
        "fraud_rate": float(merged["isFraud"].mean()) if n_validation else float("nan"),
        "merged_component_size_total_fraud_rate_by_bucket": _component_size_bucket_fraud_rate(
            merged, "merged_component_size_total"
        ),
        "endpoints_same_component_fraud_rate": _same_component_fraud_rate(merged),
        "roc_auc_merged_component_size_total_vs_isFraud": (
            _roc_auc(merged["merged_component_size_total"], merged["isFraud"]) if n_validation else float("nan")
        ),
    }

    fanout_strata = []
    for label in FANOUT_STRATUM_LABELS:
        sub = merged[merged["_fanout_stratum"] == label]
        n_sub = len(sub)
        fanout_strata.append(
            {
                "stratum": label,
                "n_rows": n_sub,
                "fraud_rate": float(sub["isFraud"].mean()) if n_sub else float("nan"),
                "merged_component_size_total_fraud_rate_by_bucket": _component_size_bucket_fraud_rate(
                    sub, "merged_component_size_total"
                ),
                "endpoints_same_component_fraud_rate": _same_component_fraud_rate(sub),
                "roc_auc_merged_component_size_total_vs_isFraud": (
                    _roc_auc(sub["merged_component_size_total"], sub["isFraud"]) if n_sub else float("nan")
                ),
            }
        )

    return {
        "fanout_stratum_edges": FANOUT_STRATUM_EDGES,
        "fanout_stratum_labels": FANOUT_STRATUM_LABELS,
        "component_size_bin_edges": COMPONENT_SIZE_BIN_EDGES,
        "component_size_bin_labels": COMPONENT_SIZE_BIN_LABELS,
        "n_validation_rows": n_validation,
        "unstratified": unstratified,
        "fanout_strata": fanout_strata,
        "conclusion": _summarize_fanout_stratified_conclusion(fanout_strata),
    }


def main() -> None:
    t0 = time.time()
    config = load_config()
    split_config = load_split_config()
    out_dir = config.reports_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading TransactionID/TransactionDT/%s from train_transaction.csv -- isFraud is never read in "
        "this measurement pass",
        config.payment_proxy_key_columns,
    )
    base = load_transaction_columns(
        "train",
        columns=["TransactionID", "TransactionDT"] + config.payment_proxy_key_columns,
        config=config,
    )

    logger.info("Loading %s from train_identity.csv, left-joining on %s", config.device_proxy_key_columns, config.join_key)
    identity = load_identity("train", config=config, usecols=[config.join_key] + config.device_proxy_key_columns)
    base = base.merge(identity, on=config.join_key, how="left")

    base = add_day_index(base, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    base = assign_partition(base, split_config, day_col="_day")

    non_holdout = base[base["partition"].isin(DEVELOPMENT_PARTITIONS)].copy()
    n_holdout_excluded = int((base["partition"] == "holdout").sum())
    logger.info(
        "Restricting to development partitions before any component computation: %d/%d rows "
        "(%d holdout rows loaded then excluded, never reaching build_relationship_frame)",
        len(non_holdout),
        len(base),
        n_holdout_excluded,
    )

    non_holdout[DEVICE_NODE_COL] = build_node_key_column(non_holdout, config.device_proxy_key_columns)
    non_holdout[PAYMENT_NODE_COL] = build_node_key_column(non_holdout, config.payment_proxy_key_columns)

    n_development = len(non_holdout)
    valid = build_relationship_frame(non_holdout, DEVICE_NODE_COL, PAYMENT_NODE_COL)
    coverage = relationship_row_coverage(
        valid, n_development_total=n_development, partition_col="partition", partitions=DEVELOPMENT_PARTITIONS
    )
    logger.info(
        "device_node <-> payment_node relationship row coverage: %d/%d (%.2f%%)",
        len(valid),
        n_development,
        coverage["pct_rows_valid"],
    )

    logger.info("Computing per-transaction Union-Find component metrics (bucket-at-a-time by TransactionDT)...")
    metrics = compute_component_metrics(valid, dt_col=config.dt_column)

    summary_overall = component_metrics_summary(metrics)
    summary_by_partition = component_metrics_summary_by_partition(
        metrics, valid, partition_col="partition", partitions=DEVELOPMENT_PARTITIONS
    )

    results: dict = {
        "split_config": {
            name: {"start_day": pr.start_day, "end_day": pr.end_day} for name, pr in split_config.partitions.items()
        },
        "n_rows_total": int(len(base)),
        "n_rows_development": n_development,
        "n_rows_holdout_excluded": n_holdout_excluded,
        "relationship_row_coverage": coverage,
        "component_metrics_summary_overall": summary_overall,
        "component_metrics_summary_by_partition": summary_by_partition,
    }

    logger.info(
        "Computing device_to_payment non-target prior-distinct-partner fan-out (E.1's own M1 quantity) "
        "for the fan-out-stratified diagnostic evaluation's stratification variable -- isFraud still not "
        "read yet"
    )
    fanout = prior_group_distinct_other_count(
        valid, group_col=DEVICE_NODE_COL, other_col=PAYMENT_NODE_COL, dt_col=config.dt_column
    )
    scored = valid[["TransactionID", "partition"]].copy()
    scored = pd.concat([scored, metrics], axis=1)
    scored[FANOUT_COL] = fanout

    scored_validation = scored[scored["partition"] == "validation"].copy()
    logger.info(
        "Running the pre-declared fan-out-stratified diagnostic evaluation on validation-partition rows "
        "only (%d rows) -- isFraud read here, and only here, in this script. Fixed bins: "
        "fanout_stratum_edges=%s, component_size_bin_edges=%s",
        len(scored_validation),
        FANOUT_STRATUM_EDGES,
        COMPONENT_SIZE_BIN_EDGES,
    )
    diagnostic_evaluation = evaluate_fanout_stratified(scored_validation, config)
    results["fanout_stratified_diagnostic_evaluation"] = diagnostic_evaluation

    results_path = out_dir / "phase_e2_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    report_path = out_dir / "phase_e2_report.md"
    render_phase_e2_report(results, report_path)

    elapsed = time.time() - t0
    logger.info("Phase E.2 measurement pass complete in %.1fs. Results: %s Report: %s", elapsed, results_path, report_path)


if __name__ == "__main__":
    main()
