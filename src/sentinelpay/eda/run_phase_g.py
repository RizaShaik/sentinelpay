"""Phase G orchestration: development-only model integration and ablation
testing of the B0-F2 baseline ladder, headline comparison B2 vs. F2.

Phase F remains frozen and is consumed verbatim (`sentinelpay.target_history`
is not modified; `sentinelpay.eda.run_phase_f` is not touched either --
`sentinelpay.model_features` calls Phase F's underlying functions directly,
in the same shape). Phase D and Phase C are likewise consumed unmodified.
See `sentinelpay.model_features`'s module docstring for the full, explicit
per-row leakage contract every feature block in this phase satisfies.

No new causal-history logic, no new target-history mechanism, no feature-
matrix persistence (`data/processed/*.parquet` is NOT written -- matrices
are built in memory, once per run). No hyperparameter tuning: Logistic
Regression only, library defaults (`sentinelpay.model_evaluation.LOGREG_MAX_ITER`
is a solver-convergence budget, not a modeling choice), train-only
`StandardScaler` fitting, validation-only evaluation.

Holdout sealing: identical to every prior phase -- `holdout` rows are
excluded via `DEVELOPMENT_PARTITIONS` filtering before `build_group_key`/
`assemble_ladder_matrix` are ever called; `holdout` is never loaded from
disk by this script at all.

Run with:
    .venv\\Scripts\\python.exe -m sentinelpay.eda.run_phase_g
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from sentinelpay.config import DataConfig, load_config, load_detection_config
from sentinelpay.data.loader import load_identity_ids, load_transaction_columns
from sentinelpay.data.split import DEVELOPMENT_PARTITIONS, assign_partition, load_split_config
from sentinelpay.data.temporal import add_day_index
from sentinelpay.eda.generate_report import render_phase_g_report
from sentinelpay.eda.grouping_key_sufficiency import build_group_key
from sentinelpay.model_evaluation import (
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    LOGREG_MAX_ITER,
    bootstrap_pr_auc_delta_ci,
    constant_prevalence_scores,
    fit_and_score,
    score_metrics,
)
from sentinelpay.model_features import LADDER_FEATURE_COLUMNS, PAYMENT_GROUP_COL, assemble_ladder_matrix, get_ladder_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase_g")

LADDER_STEPS_FITTED = ["B1", "B2", "F1", "F2"]  # B0 is a constant predictor, handled separately
GRADUATION_RELATIVE_LIFT_THRESHOLD = 1.10  # F2 PR-AUC must be >= B2 PR-AUC * 1.10


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def build_development_frame(config: DataConfig, split_config) -> tuple[pd.DataFrame, int, int]:
    """Load TransactionID/TransactionDT/TransactionAmt/isFraud/payment_proxy_key
    columns, assign partitions, and filter to `DEVELOPMENT_PARTITIONS` before
    any key-building or feature assembly. Factored out for testability with
    a monkeypatched loader (mirrors `run_phase_d.py`/`run_phase_f.py`).

    Returns `(development_df, n_rows_total, n_holdout_excluded)`.
    """
    base = load_transaction_columns(
        "train",
        columns=["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"] + config.payment_proxy_key_columns,
        config=config,
    )
    base = add_day_index(base, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    base = assign_partition(base, split_config, day_col="_day")

    development = base[base["partition"].isin(DEVELOPMENT_PARTITIONS)].copy()
    n_holdout_excluded = int((base["partition"] == "holdout").sum())
    return development, int(len(base)), n_holdout_excluded


def main() -> None:
    t0 = time.time()
    config = load_config()
    split_config = load_split_config()
    detection_config = load_detection_config()
    out_dir = config.reports_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading TransactionID/TransactionDT/TransactionAmt/isFraud/%s from train_transaction.csv, plus "
        "identity ids",
        config.payment_proxy_key_columns,
    )
    identity_ids = load_identity_ids("train", config=config)
    development, n_rows_total, n_holdout_excluded = build_development_frame(config, split_config)
    logger.info(
        "Restricting to development partitions before any key-building or feature assembly: %d/%d rows "
        "(%d holdout rows loaded then excluded, never reaching build_group_key/assemble_ladder_matrix)",
        len(development),
        n_rows_total,
        n_holdout_excluded,
    )

    valid = build_group_key(development, config.payment_proxy_key_columns, key_name=PAYMENT_GROUP_COL)
    n_rows_missing_key = len(development) - len(valid)
    logger.info(
        "payment_proxy_key present on %d/%d development rows (%d excluded, missing a key component)",
        len(valid),
        len(development),
        n_rows_missing_key,
    )

    logger.info("Assembling the B0-F2 ladder feature matrix (Phase C + Phase D + Phase F, all verbatim)...")
    assembled = assemble_ladder_matrix(
        valid,
        identity_ids,
        detection_config,
        dt_col=config.dt_column,
        amount_col="TransactionAmt",
        group_col=PAYMENT_GROUP_COL,
        partition_col="partition",
        id_col="TransactionID",
    )
    train_rows = assembled[assembled["partition"] == "train"]
    validation_rows = assembled[assembled["partition"] == "validation"]
    y_train = train_rows["isFraud"].to_numpy()
    y_validation = validation_rows["isFraud"].to_numpy()
    logger.info("Model rows: train=%d, validation=%d", len(train_rows), len(validation_rows))

    ladder_results: dict = {}
    proba_by_step: dict = {}

    proba_b0 = constant_prevalence_scores(y_train, len(validation_rows))
    roc_b0, pr_b0 = score_metrics(y_validation, proba_b0)
    ladder_results["B0"] = {"roc_auc": roc_b0, "pr_auc": pr_b0, "n_features": 0, "converged": True, "n_iter": 0}
    proba_by_step["B0"] = proba_b0
    logger.info("B0 (constant prevalence): roc_auc=%.4f pr_auc=%.4f", roc_b0, pr_b0)

    for step in LADDER_STEPS_FITTED:
        X_train = get_ladder_matrix(train_rows, step).to_numpy()
        X_validation = get_ladder_matrix(validation_rows, step).to_numpy()
        result = fit_and_score(X_train, y_train, X_validation, y_validation)
        ladder_results[step] = {
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
            "n_features": result["n_features"],
            "converged": result["converged"],
            "n_iter": result["n_iter"],
        }
        proba_by_step[step] = result["proba_validation"]
        logger.info(
            "%s: roc_auc=%.4f pr_auc=%.4f n_features=%d converged=%s (n_iter=%d, max_iter=%d)",
            step,
            result["roc_auc"],
            result["pr_auc"],
            result["n_features"],
            result["converged"],
            result["n_iter"],
            LOGREG_MAX_ITER,
        )

    pr_b2 = ladder_results["B2"]["pr_auc"]
    pr_f1 = ladder_results["F1"]["pr_auc"]
    pr_f2 = ladder_results["F2"]["pr_auc"]
    roc_b2 = ladder_results["B2"]["roc_auc"]
    roc_f2 = ladder_results["F2"]["roc_auc"]

    logger.info("Running the fixed-seed %d-resample bootstrap PR-AUC delta CI (B2 -> F2)...", BOOTSTRAP_N_RESAMPLES)
    bootstrap = bootstrap_pr_auc_delta_ci(y_validation, proba_by_step["B2"], proba_by_step["F2"])

    gate1_relative_lift = bool(pr_f2 >= pr_b2 * GRADUATION_RELATIVE_LIFT_THRESHOLD)
    gate2_roc_non_degrading = bool(roc_f2 >= roc_b2)
    gate3_bootstrap_ci = bool(bootstrap["ci_lower"] > 0.0)
    gate4_f1_improves = bool(pr_f1 > pr_b2)
    monotonic_b2_f1_f2 = bool(pr_b2 <= pr_f1 <= pr_f2)
    all_gates_pass = gate1_relative_lift and gate2_roc_non_degrading and gate3_bootstrap_ci and gate4_f1_improves

    graduation = {
        "relative_lift_threshold": GRADUATION_RELATIVE_LIFT_THRESHOLD,
        "gate1_relative_pr_auc_lift_f2_over_b2": gate1_relative_lift,
        "gate2_roc_auc_f2_ge_b2": gate2_roc_non_degrading,
        "gate3_bootstrap_ci_lower_gt_zero": gate3_bootstrap_ci,
        "gate4_pr_auc_f1_gt_b2": gate4_f1_improves,
        "monotonic_b2_le_f1_le_f2_pr_auc": monotonic_b2_f1_f2,
        "all_gates_pass": all_gates_pass,
        "pr_auc_relative_lift_f2_over_b2": (pr_f2 / pr_b2) if pr_b2 else float("nan"),
        "bootstrap_pr_auc_delta_f2_minus_b2": bootstrap,
    }
    logger.info(
        "Graduation gates: 1(lift>=%.0f%%)=%s 2(ROC non-degrading)=%s 3(bootstrap CI lower>0)=%s "
        "4(F1>B2)=%s -> all_gates_pass=%s",
        (GRADUATION_RELATIVE_LIFT_THRESHOLD - 1.0) * 100,
        gate1_relative_lift,
        gate2_roc_non_degrading,
        gate3_bootstrap_ci,
        gate4_f1_improves,
        all_gates_pass,
    )

    results: dict = {
        "split_config": {
            name: {"start_day": pr.start_day, "end_day": pr.end_day} for name, pr in split_config.partitions.items()
        },
        "n_rows_total": n_rows_total,
        "n_rows_development": int(len(development)),
        "n_rows_holdout_excluded": n_holdout_excluded,
        "n_rows_missing_payment_proxy_key": int(n_rows_missing_key),
        "n_rows_valid_key": int(len(valid)),
        "n_rows_train": int(len(train_rows)),
        "n_rows_validation": int(len(validation_rows)),
        "ladder_feature_columns": LADDER_FEATURE_COLUMNS,
        "ladder_results": ladder_results,
        "graduation": graduation,
        "logreg_max_iter": LOGREG_MAX_ITER,
        "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    results_path = out_dir / "phase_g_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    report_path = out_dir / "phase_g_report.md"
    render_phase_g_report(results, report_path)

    elapsed = time.time() - t0
    logger.info("Phase G complete in %.1fs. Results: %s Report: %s", elapsed, results_path, report_path)


if __name__ == "__main__":
    main()
