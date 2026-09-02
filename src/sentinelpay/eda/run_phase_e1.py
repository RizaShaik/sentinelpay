"""Phase E.1 orchestration: non-target link/relationship sufficiency and
causal cross-key fan-out measurement.

Reduced first-pass scope, approved before this script was written: measure
only the 6 directions across the 3 relationships anchored on node types this
project already has strong prior evidence for --

    payment_node <-> device_node        (both directions)
    payment_node <-> email_purchaser_node (both directions)
    device_node  <-> email_purchaser_node (both directions)

`email_recipient_node` (R_emaildomain) and `addr_only_node` (addr1 alone)
are NOT measured in this pass -- they are cheap, mechanical follow-ups once
this module and its tests exist (just another `build_relationship_frame` +
`analyze_relationship_direction` call), deferred so this first reviewable
unit stays small. `payment_node`/`device_node` are the same
`payment_proxy_key`/`device_proxy_key` proxy groupings D.1 already measured;
`email_purchaser_node` is `P_emaildomain` used directly (single column, no
join needed).

This is NOT Phase E.2: no Union-Find, no connected-component detection, no
persistent graph structure, no scoring, no flags. `isFraud` is never read
anywhere in this script -- there is no diagnostic-evaluation step at all
(stronger than Phase D, matching D.1's own precedent exactly).

M5 vs. M5b: both `overlap_diagnostic` (M5, raw partner-set overlap) and
`frequency_adjusted_overlap_diagnostic` (M5b, overlap compared against a
population-prevalence null baseline) are computed for every direction. M5 is
retained for descriptive/contextual reporting only -- a critical review of
this phase's first real-data results found raw overlap dominated by
population-generic partner values, not evidence of coordinated structure.
M5b's `lift_ratio` is the sole driver of `recommend_union_find_for_e2` (see
`sentinelpay.eda.link_sufficiency.recommend_relationships`). Same 6-direction
scope as before this correction -- unchanged.

Holdout sealing: TransactionID/TransactionDT/payment+device+email columns
are read the same way run_phase_d1.py reads payment/device columns. Rows are
assigned a partition and filtered to DEVELOPMENT_PARTITIONS BEFORE
`build_relationship_frame`/`analyze_relationship_direction`/
`overlap_diagnostic` are ever called -- holdout rows never reach any
relationship-content computation.

configs/split.yaml is read, never modified. No new configs/*.yaml is
introduced -- matches D.1's precedent of module constants, not a config
file, for a pure sufficiency phase.

Run with:
    .venv\\Scripts\\python.exe -m sentinelpay.eda.run_phase_e1
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from sentinelpay.config import load_config
from sentinelpay.data.loader import load_identity, load_transaction_columns
from sentinelpay.data.split import DEVELOPMENT_PARTITIONS, assign_partition, load_split_config
from sentinelpay.data.temporal import add_day_index
from sentinelpay.eda.generate_report import render_phase_e1_report
from sentinelpay.eda.link_sufficiency import (
    analyze_relationship_direction,
    build_node_key_column,
    build_relationship_frame,
    evaluate_relationship_sufficiency,
    frequency_adjusted_overlap_diagnostic,
    overlap_diagnostic,
    recommend_relationships,
    recommend_window_size,
    relationship_row_coverage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase_e1")

EMAIL_PURCHASER_COL = "P_emaildomain"

# (direction_name, anchor_col, other_col) for the 6 approved directions.
# _payment_node/_device_node are built once below via build_node_key_column;
# EMAIL_PURCHASER_COL is used directly (already a single column).
DIRECTIONS = [
    ("payment_to_device", "_payment_node", "_device_node"),
    ("device_to_payment", "_device_node", "_payment_node"),
    ("payment_to_email_purchaser", "_payment_node", EMAIL_PURCHASER_COL),
    ("email_purchaser_to_payment", EMAIL_PURCHASER_COL, "_payment_node"),
    ("device_to_email_purchaser", "_device_node", EMAIL_PURCHASER_COL),
    ("email_purchaser_to_device", EMAIL_PURCHASER_COL, "_device_node"),
]

# Each direction belongs to one of these underlying (anchor,other)-unordered
# relationship pairs; M3 row coverage is computed once per pair (it is
# symmetric) and shared by that pair's two directions.
RELATIONSHIP_PAIRS = {
    frozenset(("_payment_node", "_device_node")): "payment_device",
    frozenset(("_payment_node", EMAIL_PURCHASER_COL)): "payment_email_purchaser",
    frozenset(("_device_node", EMAIL_PURCHASER_COL)): "device_email_purchaser",
}


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def main() -> None:
    t0 = time.time()
    config = load_config()
    split_config = load_split_config()
    out_dir = config.reports_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading TransactionID/TransactionDT/%s/%s from train_transaction.csv -- isFraud is never read in Phase E.1",
        config.payment_proxy_key_columns,
        EMAIL_PURCHASER_COL,
    )
    base = load_transaction_columns(
        "train",
        columns=["TransactionID", "TransactionDT"] + config.payment_proxy_key_columns + [EMAIL_PURCHASER_COL],
        config=config,
    )

    logger.info("Loading %s from train_identity.csv, left-joining on %s", config.device_proxy_key_columns, config.join_key)
    identity = load_identity(
        "train", config=config, usecols=[config.join_key] + config.device_proxy_key_columns
    )
    base = base.merge(identity, on=config.join_key, how="left")

    base = add_day_index(base, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    base = assign_partition(base, split_config, day_col="_day")

    non_holdout = base[base["partition"].isin(DEVELOPMENT_PARTITIONS)].copy()
    n_holdout_excluded = int((base["partition"] == "holdout").sum())
    logger.info(
        "Restricting to development partitions before any relationship content analysis: %d/%d rows "
        "(%d holdout rows loaded then excluded, never reaching build_relationship_frame)",
        len(non_holdout),
        len(base),
        n_holdout_excluded,
    )

    # Row-preserving node-key columns (see build_node_key_column docstring:
    # unlike D.1's build_group_key, missing components become NaN in place,
    # never dropping a row that another relationship might still need).
    non_holdout["_payment_node"] = build_node_key_column(non_holdout, config.payment_proxy_key_columns)
    non_holdout["_device_node"] = build_node_key_column(non_holdout, config.device_proxy_key_columns)

    n_development = len(non_holdout)

    logger.info("Building the 3 relationship-pair valid frames (M3 row coverage)...")
    valid_frames: dict[str, pd.DataFrame] = {}
    coverage_results: dict[str, dict] = {}
    for cols_frozenset, pair_name in RELATIONSHIP_PAIRS.items():
        col_a, col_b = tuple(cols_frozenset)
        valid = build_relationship_frame(non_holdout, col_a, col_b)
        valid_frames[pair_name] = valid
        coverage_results[pair_name] = relationship_row_coverage(
            valid, n_development_total=n_development, partition_col="partition", partitions=DEVELOPMENT_PARTITIONS
        )
        logger.info("  %s: %d/%d valid rows (%.2f%%)", pair_name, len(valid), n_development, coverage_results[pair_name]["pct_rows_valid"])

    def _pair_name_for(anchor_col: str, other_col: str) -> str:
        return RELATIONSHIP_PAIRS[frozenset((anchor_col, other_col))]

    all_directions: dict[str, dict] = {}
    for direction_name, anchor_col, other_col in DIRECTIONS:
        pair_name = _pair_name_for(anchor_col, other_col)
        valid_df = valid_frames[pair_name]
        logger.info("Analyzing direction %s (anchor=%s, other=%s, %d valid rows)...", direction_name, anchor_col, other_col, len(valid_df))

        analysis = analyze_relationship_direction(
            valid_df,
            anchor_col=anchor_col,
            other_col=other_col,
            dt_col=config.dt_column,
            partition_col="partition",
            partitions=DEVELOPMENT_PARTITIONS,
        )
        coverage = coverage_results[pair_name]
        evaluation = evaluate_relationship_sufficiency(analysis, coverage)
        # M5 -- retained, descriptive/contextual reporting only (does not
        # drive recommend_union_find_for_e2; see link_sufficiency module
        # docstring "M5 vs. M5b").
        overlap = overlap_diagnostic(valid_df, anchor_col=anchor_col, other_col=other_col, dt_col=config.dt_column)
        # M5b -- the decision-driving diagnostic.
        frequency_adjusted_overlap = frequency_adjusted_overlap_diagnostic(valid_df, anchor_col=anchor_col, other_col=other_col, dt_col=config.dt_column)
        recommended_window = recommend_window_size(analysis) if evaluation["is_suitable"] else None

        all_directions[direction_name] = {
            "anchor_col": anchor_col,
            "other_col": other_col,
            "relationship_pair": pair_name,
            "analysis": analysis,
            "coverage": coverage,
            "evaluation": evaluation,
            "overlap": overlap,
            "frequency_adjusted_overlap": frequency_adjusted_overlap,
            "recommended_window": recommended_window,
        }

    recommendation = recommend_relationships(all_directions)

    results: dict = {
        "split_config": {
            name: {"start_day": pr.start_day, "end_day": pr.end_day} for name, pr in split_config.partitions.items()
        },
        "n_rows_total": int(len(base)),
        "n_rows_development": n_development,
        "n_rows_holdout_excluded": n_holdout_excluded,
        "relationship_pairs_row_coverage": coverage_results,
        "directions": all_directions,
        "recommendation": recommendation,
        "not_yet_analyzed": [
            "email_recipient_node (R_emaildomain) -- columns not loaded by this run; measured columns available, not yet analyzed.",
            "addr_only_node (addr1 alone) -- columns not loaded by this run; measured columns available, not yet analyzed.",
        ],
    }

    results_path = out_dir / "phase_e1_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    report_path = out_dir / "phase_e1_report.md"
    render_phase_e1_report(results, report_path)

    elapsed = time.time() - t0
    logger.info("Phase E.1 link sufficiency analysis complete in %.1fs. Results: %s Report: %s", elapsed, results_path, report_path)


if __name__ == "__main__":
    main()
