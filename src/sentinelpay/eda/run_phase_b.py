"""Phase B orchestration: build and validate the chronological
train/embargo_1/validation/embargo_2/holdout partitioning, measure drift on
train vs. validation only, and demonstrate the generic temporal rollup
utilities. Writes reports/eda/phase_b_results.json and, deterministically
from that dict, reports/eda/phase_b_report.md + figures/split_boundaries.png.

`holdout` is reserved for Phase H: this script only ever checks its row
count and chronological position, never a content statistic. See
sentinelpay.data.split for the enforced structural-validation rules.

Run with:
    .venv\\Scripts\\python.exe -m sentinelpay.eda.run_phase_b
"""
from __future__ import annotations

import json
import logging
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import psutil

from sentinelpay.config import load_config
from sentinelpay.data.loader import load_identity_ids, load_transaction_columns, load_transaction_full
from sentinelpay.data.split import PARTITION_ORDER, assign_partition, load_split_config, validate_split
from sentinelpay.data.temporal import (
    add_day_index,
    categorical_drift,
    daily_amount_stats_by_group,
    daily_count_by_group,
    daily_fraud_rate,
    daily_volume,
    numeric_drift,
)
from sentinelpay.eda.generate_report import render_phase_b_report
from sentinelpay.eda.leakage import curated_target_correlation, v_block_target_correlation
from sentinelpay.utils.memory import log_memory_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase_b")

DEVELOPMENT_ONLY = ["train", "embargo_1", "validation", "embargo_2"]  # excludes holdout, everywhere


def _process_rss_mb() -> float:
    return round(psutil.Process().memory_info().rss / (1024**2), 2)


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _d_column_summary_by_partition(df: pd.DataFrame, d_columns: list[str]) -> list[dict]:
    """Mean and missing-rate of each D-column, split out by partition --
    restricted by the caller to train/embargo_1/validation rows only, to
    look for boundary artifacts near embargo_1 without touching holdout."""
    rows = []
    for name in ["train", "embargo_1", "validation"]:
        sub = df[df["partition"] == name]
        for col in d_columns:
            if col not in sub.columns:
                continue
            series = sub[col]
            rows.append(
                {
                    "partition": name,
                    "column": col,
                    "mean": float(series.mean()) if series.notna().any() else float("nan"),
                    "pct_missing": round(100.0 * series.isna().mean(), 4) if len(series) else float("nan"),
                    "n_rows": int(len(sub)),
                }
            )
    return rows


def main() -> None:
    t0 = time.time()
    config = load_config()
    split_config = load_split_config()
    out_dir = config.reports_dir / "eda"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "split_config": {
            name: {"start_day": pr.start_day, "end_day": pr.end_day} for name, pr in split_config.partitions.items()
        }
    }
    memory: dict = {"baseline_process_rss_mb": _process_rss_mb()}

    # ---- Step 1: minimal columns for ALL rows -> day index -> partition -> structural validation ----
    logger.info("Loading minimal columns (TransactionID, TransactionDT, isFraud) for full-file partitioning")
    base = load_transaction_columns("train", columns=["TransactionID", "TransactionDT", "isFraud"], config=config)
    base = add_day_index(base, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)

    validation_result = validate_split(base, split_config, dt_col=config.dt_column, day_col="_day", id_col="TransactionID")
    results["validation_result"] = validation_result.to_dict()
    if not validation_result.is_valid:
        logger.warning("Split validation FAILED: %s", validation_result.to_dict())

    base_assigned = assign_partition(base, split_config, day_col="_day")

    # ---- Step 2: split-boundary figure (holdout region carries NO plotted data) ----
    logger.info("Rendering split-boundary figure (non-holdout data only)")
    non_holdout = base_assigned[base_assigned["partition"] != "holdout"]
    vol = daily_volume(non_holdout, day_col="_day")
    fraud = daily_fraud_rate(non_holdout, day_col="_day", target_col="isFraud")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    band_colors = {
        "train": "#dbeafe",
        "embargo_1": "#fee2e2",
        "validation": "#dcfce7",
        "embargo_2": "#fee2e2",
        "holdout": "#e5e7eb",
    }
    for name in PARTITION_ORDER:
        pr = split_config.partitions[name]
        for ax in axes:
            ax.axvspan(pr.start_day, pr.end_day + 1, color=band_colors[name], alpha=0.5)
    axes[0].plot(vol["_day"], vol["n_transactions"], color="#2563eb", linewidth=1.5)
    axes[0].set_ylabel("Transactions / day")
    axes[0].set_title("Daily transaction volume (train/embargo_1/validation/embargo_2 only)")
    axes[1].plot(fraud["_day"], fraud["fraud_rate"], color="#dc2626", linewidth=1.5)
    axes[1].set_ylabel("Fraud rate")
    axes[1].set_xlabel("Day index (TransactionDT // 86400)")
    axes[1].set_title("Daily fraud rate (train/embargo_1/validation/embargo_2 only)")
    holdout_mid = (split_config.partitions["holdout"].start_day + split_config.partitions["holdout"].end_day) / 2
    axes[0].text(holdout_mid, axes[0].get_ylim()[1] * 0.5, "Holdout\n(sealed for Phase H)", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "split_boundaries.png", dpi=110)
    plt.close(fig)

    # ---- Step 3 (B2): embargo boundary sensitivity, train/embargo_1/validation only ----
    logger.info("Analyzing D-column boundary sensitivity (train/embargo_1/validation only)")
    d_columns = [f"D{i}" for i in range(1, 16)]
    d_df = load_transaction_columns("train", columns=["TransactionID", "TransactionDT"] + d_columns, config=config)
    d_df = add_day_index(d_df, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    d_df = assign_partition(d_df, split_config, day_col="_day")
    results["b2_embargo_sensitivity"] = {
        "note": (
            "Descriptive only. A 7-day embargo is a conservative engineering default, not a proof "
            "that boundary leakage from undocumented D-column semantics is eliminated -- several "
            "D-columns have means well beyond 7 days (Phase A: D1 ~86-104, D4 ~131-149), so their "
            "true lookback window relative to this embargo is unknown. Boundaries are NOT changed "
            "based on this result in Phase B."
        ),
        "d_column_summary_by_partition": _d_column_summary_by_partition(d_df, d_columns),
    }
    del d_df

    # ---- Step 4 (B3): full load, drift train vs. validation only ----
    logger.info("load_transaction_full: explicit high-memory load for train-vs-validation drift analysis")
    tx = load_transaction_full("train", config=config)
    mem_report = log_memory_report(tx, "train_transaction_full (Phase B)")
    memory["train_transaction_full_df_mb"] = mem_report["dataframe_mb"]
    memory["after_full_load_process_rss_mb"] = _process_rss_mb()

    tx = add_day_index(tx, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    tx = assign_partition(tx, split_config, day_col="_day")
    train_val_only = tx[tx["partition"].isin(["train", "validation"])].copy()
    del tx  # free the full 394-column frame; only the train+validation subset is needed below

    val_start_day = split_config.partitions["validation"].start_day
    numeric_probe_cols = [c for c in config.correlation_curated_columns if c in train_val_only.columns]
    drift_numeric = numeric_drift(train_val_only, numeric_probe_cols, day_col="_day", split_day=val_start_day)

    cat_probe_cols = [
        c for c in ["ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M5", "M6", "P_emaildomain"]
        if c in train_val_only.columns
    ]
    drift_categorical = categorical_drift(train_val_only, cat_probe_cols, day_col="_day", split_day=val_start_day)

    train_only = train_val_only[train_val_only["partition"] == "train"]
    val_only = train_val_only[train_val_only["partition"] == "validation"]

    curated_train = curated_target_correlation(train_only, "isFraud", config.correlation_curated_columns, top_n=15)
    curated_val = curated_target_correlation(val_only, "isFraud", config.correlation_curated_columns, top_n=15)

    v_cols = [c for c in train_val_only.columns if re.match(r"^V\d+$", c)]
    v_block_train = v_block_target_correlation(train_only, "isFraud", v_cols, top_n_blocks=10)
    v_block_val = v_block_target_correlation(val_only, "isFraud", v_cols, top_n_blocks=10)

    results["drift_train_vs_validation"] = {
        "numeric": drift_numeric.to_dict(orient="records"),
        "categorical": drift_categorical.to_dict(orient="records"),
        "curated_target_correlation_train": curated_train.to_dict(orient="records"),
        "curated_target_correlation_validation": curated_val.to_dict(orient="records"),
        "v_block_target_correlation_train": v_block_train.to_dict(orient="records"),
        "v_block_target_correlation_validation": v_block_val.to_dict(orient="records"),
    }
    memory["after_correlation_process_rss_mb"] = _process_rss_mb()
    del train_val_only, train_only, val_only

    # ---- Step 5 (B4): identity coverage by partition (holdout excluded entirely) ----
    logger.info("Checking identity coverage by partition (train/embargo_1/validation/embargo_2 only)")
    identity_ids = set(load_identity_ids("train", config=config).tolist())
    identity_cov: dict = {}
    for name in DEVELOPMENT_ONLY:
        sub_ids = base_assigned.loc[base_assigned["partition"] == name, "TransactionID"]
        n_total = len(sub_ids)
        n_with_identity = int(sub_ids.isin(identity_ids).sum())
        identity_cov[name] = {
            "n_transactions": n_total,
            "n_with_identity": n_with_identity,
            "pct_with_identity": round(100.0 * n_with_identity / n_total, 4) if n_total else 0.0,
        }
    identity_cov["holdout"] = {"note": "not computed -- sealed for Phase H"}
    results["identity_coverage_by_partition"] = identity_cov

    # ---- Step 6 (B5): rollup utility demo, illustrative only, train+validation only ----
    logger.info("Demonstrating generic rollup utilities (placeholder grouping column, illustrative only)")
    placeholder_group_col = "ProductCD"
    rollup_df = load_transaction_columns(
        "train", columns=["TransactionID", "TransactionDT", "TransactionAmt", placeholder_group_col], config=config
    )
    rollup_df = add_day_index(rollup_df, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    rollup_df = assign_partition(rollup_df, split_config, day_col="_day")
    rollup_df = rollup_df[rollup_df["partition"].isin(["train", "validation"])]

    daily_count = daily_count_by_group(rollup_df, group_col=placeholder_group_col, day_col="_day")
    daily_amount = daily_amount_stats_by_group(
        rollup_df, group_col=placeholder_group_col, amount_col="TransactionAmt", day_col="_day"
    )
    results["rollup_utility_demo"] = {
        "placeholder_group_column": placeholder_group_col,
        "note": (
            "Illustrative only -- does not establish a production entity/merchant definition. "
            "The actual grouping key is a Phase D decision."
        ),
        "daily_count_sample": daily_count.head(10).to_dict(orient="records"),
        "daily_amount_stats_sample": daily_amount.head(10).to_dict(orient="records"),
    }

    memory["final_process_rss_mb"] = _process_rss_mb()
    results["memory"] = memory

    # ---- write outputs ----
    results_path = out_dir / "phase_b_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    report_path = out_dir / "phase_b_report.md"
    render_phase_b_report(results, report_path)

    elapsed = time.time() - t0
    logger.info(
        "Phase B analysis complete in %.1fs. is_valid=%s Results: %s Report: %s",
        elapsed,
        validation_result.is_valid,
        results_path,
        report_path,
    )


if __name__ == "__main__":
    main()
