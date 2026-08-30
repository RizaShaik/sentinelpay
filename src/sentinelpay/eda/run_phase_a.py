"""Phase A orchestration: run all EDA / validation checks against the real
IEEE-CIS data, dump results as JSON (numbers) + PNG (figures), then generate
reports/eda/phase_a_report.md deterministically from that JSON so the report
and the numbers can never silently diverge (see generate_report.py).

Run with:
    .venv\\Scripts\\python.exe -m sentinelpay.eda.run_phase_a
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
from sentinelpay.data.loader import (
    load_identity,
    load_identity_ids,
    load_transaction_full,
    load_transaction_ids,
)
from sentinelpay.data.temporal import add_day_index, categorical_drift, daily_fraud_rate, daily_volume, numeric_drift
from sentinelpay.data.validation import (
    duplicate_row_report,
    missingness_report,
    validate_schema_consistency,
    validate_transaction_identity_join,
)
from sentinelpay.eda.entity import group_size_distribution, group_size_summary_stats, shared_key_fraud_summary
from sentinelpay.eda.generate_report import render_phase_a_report
from sentinelpay.eda.leakage import (
    curated_target_correlation,
    full_target_correlation,
    id_time_monotonicity,
    near_duplicate_row_rate,
    v_block_target_correlation,
)
from sentinelpay.utils.memory import log_memory_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase_a")


def _process_rss_mb() -> float:
    return round(psutil.Process().memory_info().rss / (1024**2), 2)


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def main() -> None:
    t0 = time.time()
    config = load_config()
    out_dir = config.reports_dir / "eda"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"correlation_mode": config.correlation_mode}
    memory: dict = {"baseline_process_rss_mb": _process_rss_mb()}

    # ---- 1. Schema consistency: train vs test transaction/identity ----
    logger.info("Loading column headers only for schema checks")
    train_tx_cols = load_transaction_full("train", config=config, nrows=0).columns.tolist()
    test_tx_cols = load_transaction_full("test", config=config, nrows=0).columns.tolist()
    train_id_cols = load_identity("train", config=config, nrows=0).columns.tolist()
    test_id_cols = load_identity("test", config=config, nrows=0).columns.tolist()

    tx_schema = validate_schema_consistency(train_tx_cols, test_tx_cols, ignore={"isFraud"})
    id_schema = validate_schema_consistency(train_id_cols, test_id_cols)
    results["schema"] = {
        "train_transaction_n_cols": len(train_tx_cols),
        "test_transaction_n_cols": len(test_tx_cols),
        "train_identity_n_cols": len(train_id_cols),
        "test_identity_n_cols": len(test_id_cols),
        "isFraud_absent_from_test_transaction": "isFraud" not in test_tx_cols,
        "transaction_schema_consistent_ignoring_isFraud": tx_schema.is_consistent(),
        "transaction_schema_only_in_train": tx_schema.only_in_a,
        "transaction_schema_only_in_test": tx_schema.only_in_b,
        "identity_schema_consistent_after_normalization": id_schema.is_consistent(),
        "identity_schema_only_in_train": id_schema.only_in_a,
        "identity_schema_only_in_test": id_schema.only_in_b,
    }

    # ---- 2. Join validation: train transaction <-> train identity ----
    logger.info("Validating train transaction<->identity join")
    train_tx_ids = load_transaction_ids("train", config=config)
    train_id_ids = load_identity_ids("train", config=config)
    join_result = validate_transaction_identity_join(train_tx_ids, train_id_ids)
    results["join_validation_train"] = vars(join_result) | {"is_clean": join_result.is_clean()}

    test_tx_ids = load_transaction_ids("test", config=config)
    test_id_ids = load_identity_ids("test", config=config)
    join_result_test = validate_transaction_identity_join(test_tx_ids, test_id_ids)
    results["join_validation_test"] = vars(join_result_test) | {"is_clean": join_result_test.is_clean()}

    # ---- 3. Full train_transaction load (explicit, high-memory, needed for
    # a whole-file missingness/duplicate scan and drift/entity probes below) ----
    logger.info("load_transaction_full: explicit high-memory load for comprehensive Phase A scan")
    tx = load_transaction_full("train", config=config)
    mem_report = log_memory_report(tx, "train_transaction_full (downcast)")
    memory["train_transaction_full_df_mb"] = mem_report["dataframe_mb"]
    memory["after_full_load_process_rss_mb"] = _process_rss_mb()

    # ---- 4. Duplicate rows & missingness ----
    results["duplicate_full_rows"] = duplicate_row_report(tx)
    miss = missingness_report(tx)
    results["top_20_missing_columns"] = miss.head(20).to_dict(orient="records")
    results["pct_columns_over_50pct_missing"] = round(100.0 * (miss["pct_missing"] > 50).mean(), 2)

    # ---- 5. Temporal structure ----
    logger.info("Analyzing temporal structure")
    tx = add_day_index(tx, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    vol = daily_volume(tx)
    fraud = daily_fraud_rate(tx)
    results["temporal"] = {
        "min_day": int(tx["_day"].min()),
        "max_day": int(tx["_day"].max()),
        "n_days_observed": int(tx["_day"].nunique()),
        "overall_fraud_rate": float(tx["isFraud"].mean()),
        "daily_volume_min": int(vol["n_transactions"].min()),
        "daily_volume_max": int(vol["n_transactions"].max()),
        "daily_volume_mean": float(vol["n_transactions"].mean()),
        "daily_fraud_rate_min": float(fraud["fraud_rate"].min()),
        "daily_fraud_rate_max": float(fraud["fraud_rate"].max()),
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(vol["_day"], vol["n_transactions"], color="#2563eb")
    axes[0].set_ylabel("Transactions / day")
    axes[0].set_title("Daily transaction volume (train)")
    axes[1].plot(fraud["_day"], fraud["fraud_rate"], color="#dc2626")
    axes[1].set_ylabel("Fraud rate")
    axes[1].set_xlabel("Day index (TransactionDT // 86400)")
    axes[1].set_title("Daily fraud rate (train)")
    fig.tight_layout()
    fig.savefig(fig_dir / "temporal_volume_fraud_rate.png", dpi=110)
    plt.close(fig)

    # ---- 6. Distribution drift: early half vs late half of the observed span ----
    split_day = int(tx["_day"].min() + (tx["_day"].max() - tx["_day"].min()) / 2)
    results["drift_split_day"] = split_day

    numeric_probe_cols = [c for c in config.correlation_curated_columns if c in tx.columns]
    drift_num = numeric_drift(tx, numeric_probe_cols, day_col="_day", split_day=split_day)
    results["numeric_drift_top10"] = drift_num.head(10).to_dict(orient="records")

    cat_probe_cols = [c for c in ["ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M5", "M6", "P_emaildomain"] if c in tx.columns]
    drift_cat = categorical_drift(tx, cat_probe_cols, day_col="_day", split_day=split_day)
    results["categorical_drift"] = drift_cat.to_dict(orient="records")

    v_cols = [c for c in tx.columns if re.match(r"^V\d+$", c)]
    v_missing_rate = tx[v_cols].isna().mean()
    v_sample = list(v_missing_rate[v_missing_rate < 0.5].index[:25])  # cheap representative, low-missing sample
    drift_v = numeric_drift(tx, v_sample, day_col="_day", split_day=split_day)
    results["v_column_drift_sample_top10"] = drift_v.head(10).to_dict(orient="records")

    # ---- 7. Leakage audit ----
    logger.info("Running leakage audit (correlation_mode=%s)", config.correlation_mode)
    results["leakage_id_time_monotonicity"] = id_time_monotonicity(tx, id_col="TransactionID", dt_col=config.dt_column)

    curated_corr = curated_target_correlation(tx, "isFraud", config.correlation_curated_columns, top_n=15)
    results["leakage_curated_target_correlation_top15"] = curated_corr.to_dict(orient="records")

    v_block_corr = v_block_target_correlation(tx, "isFraud", v_cols, top_n_blocks=10)
    results["leakage_v_block_target_correlation_top10"] = v_block_corr.to_dict(orient="records")

    if config.correlation_mode == "full":
        all_numeric = [
            c for c in tx.columns if pd.api.types.is_numeric_dtype(tx[c]) and c not in ("isFraud", "_day", "TransactionID")
        ]
        full_corr = full_target_correlation(tx, "isFraud", all_numeric, top_n=15)
        results["leakage_full_target_correlation_top15"] = full_corr.to_dict(orient="records")

    memory["after_correlation_process_rss_mb"] = _process_rss_mb()

    results["leakage_near_duplicate_card_amt_dt"] = near_duplicate_row_rate(
        tx, subset=["card1", "card2", "addr1", "TransactionAmt", "TransactionDT"]
    )

    # ---- 8. Candidate proxy-key relationships (exploratory evidence only) ----
    logger.info("Investigating candidate proxy-key relationships")
    payment_proxy_key = [c for c in config.payment_proxy_key_columns if c in tx.columns]
    results["entity_payment_proxy_key_summary_stats"] = group_size_summary_stats(tx, payment_proxy_key)
    results["entity_payment_proxy_key_fraud_buckets"] = group_size_distribution(
        tx, payment_proxy_key, target_col="isFraud"
    ).to_dict(orient="records")
    results["entity_payment_proxy_key_fraud_summary"] = shared_key_fraud_summary(tx, payment_proxy_key, target_col="isFraud")

    # device/browser proxy requires identity merge
    ident = load_identity("train", config=config, usecols=["TransactionID"] + config.device_proxy_key_columns)
    merged = tx[["TransactionID", "isFraud"]].merge(ident, on="TransactionID", how="inner")
    device_proxy_key = [c for c in config.device_proxy_key_columns if c in merged.columns]
    if device_proxy_key:
        results["entity_device_proxy_key_summary_stats"] = group_size_summary_stats(merged, device_proxy_key)
        results["entity_device_proxy_key_fraud_buckets"] = group_size_distribution(
            merged, device_proxy_key, target_col="isFraud"
        ).to_dict(orient="records")
        results["entity_device_proxy_key_fraud_summary"] = shared_key_fraud_summary(
            merged, device_proxy_key, target_col="isFraud"
        )

    memory["final_process_rss_mb"] = _process_rss_mb()
    results["memory"] = memory

    # ---- write outputs ----
    results_path = out_dir / "phase_a_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    report_path = out_dir / "phase_a_report.md"
    render_phase_a_report(results, report_path)

    elapsed = time.time() - t0
    logger.info("Phase A analysis complete in %.1fs. Results: %s Report: %s", elapsed, results_path, report_path)


if __name__ == "__main__":
    main()
