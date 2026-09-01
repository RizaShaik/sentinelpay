"""Phase D orchestration: non-target per-`payment_proxy_key` behavioral-change
detector, plus a strictly downstream, `validation`-only target-association
evaluation.

Hard non-target boundary: `sentinelpay.data.history` and
`sentinelpay.detection` never import, accept, or read `isFraud` anywhere.
This script reads `isFraud` in exactly one place -- `evaluate_validation_only`
-- which runs after every score/flag already exists and only for
`validation`-partition rows. Nothing computed there is written back into a
score, a flag, or `configs/detection.yaml`.

Holdout sealing: `TransactionID`/`TransactionDT`/`TransactionAmt`/
`payment_proxy_key` columns are read from `train_transaction.csv`. Rows are
assigned a partition and filtered to `DEVELOPMENT_PARTITIONS`
(`build_development_frame`) BEFORE `build_group_key`/
`compute_behavioral_change_score` are ever called -- holdout rows never
reach group-key, history, or score computation.

`configs/split.yaml` and `configs/detection.yaml` are read, never modified.

Run with:
    .venv\\Scripts\\python.exe -m sentinelpay.eda.run_phase_d
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd
from scipy import stats as scipy_stats

from sentinelpay.config import DataConfig, load_config, load_detection_config
from sentinelpay.data.loader import load_transaction_columns
from sentinelpay.data.split import DEVELOPMENT_PARTITIONS, SplitConfig, assign_partition, load_split_config
from sentinelpay.data.temporal import add_day_index
from sentinelpay.detection import (
    FLAG_INSUFFICIENT_HISTORY,
    FLAG_SCORED_NORMAL,
    FLAG_SCORED_OUTLIER,
    FLAG_ZERO_MAD,
    compute_behavioral_change_score,
)
from sentinelpay.eda.generate_report import render_phase_d_report
from sentinelpay.eda.grouping_key_sufficiency import build_group_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase_d")

GROUP_KEY_NAME = "_payment_group_key"
ALL_FLAGS = [FLAG_INSUFFICIENT_HISTORY, FLAG_ZERO_MAD, FLAG_SCORED_NORMAL, FLAG_SCORED_OUTLIER]


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def build_development_frame(config: DataConfig, split_config: SplitConfig) -> tuple[pd.DataFrame, int, int]:
    """Load TransactionID/TransactionDT/TransactionAmt/payment_proxy_key
    columns from train_transaction.csv, assign partitions, and filter to
    `DEVELOPMENT_PARTITIONS` before any group-key, history, or score
    computation. `isFraud` is never read here.

    Factored out of `main()` so it is directly testable (with a monkeypatched
    loader) without touching real CSVs -- see tests/test_run_phase_d.py.

    Returns `(development_df, n_rows_total, n_holdout_excluded)`.
    """
    base = load_transaction_columns(
        "train",
        columns=["TransactionID", "TransactionDT", "TransactionAmt"] + config.payment_proxy_key_columns,
        config=config,
    )
    base = add_day_index(base, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    base = assign_partition(base, split_config, day_col="_day")

    development = base[base["partition"].isin(DEVELOPMENT_PARTITIONS)].copy()
    n_holdout_excluded = int((base["partition"] == "holdout").sum())
    return development, int(len(base)), n_holdout_excluded


def _roc_auc(score: pd.Series, target: pd.Series) -> float:
    """Rank-based (Mann-Whitney U) ROC-AUC. Avoids adding scikit-learn as a
    new project dependency -- `scipy.stats.rankdata` (already a dependency)
    is sufficient. AUC = (sum of positive-class ranks - n_pos*(n_pos+1)/2) /
    (n_pos * n_neg); `rankdata`'s default handles ties via average rank."""
    y = target.to_numpy()
    s = score.to_numpy()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = scipy_stats.rankdata(s)
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_validation_only(scored_validation: pd.DataFrame, config: DataConfig) -> dict:
    """Read-only, `isFraud`-reading diagnostic evaluation.

    Requires every row in `scored_validation` to already be
    `partition == "validation"` -- raises `ValueError` otherwise (rows from
    other partitions are rejected, not silently included). Runs strictly
    after scoring: never mutates `modified_zscore`/`flag`, never re-selects a
    hyperparameter. `isFraud` is read here and nowhere else in this project's
    detector construction path.
    """
    if not (scored_validation["partition"] == "validation").all():
        raise ValueError("evaluate_validation_only requires every row to already be partition == 'validation'")

    isfraud = load_transaction_columns("train", columns=["TransactionID", "isFraud"], config=config)
    merged = scored_validation.merge(isfraud, on="TransactionID", how="left")

    n_validation = len(merged)
    flag_counts = merged["flag"].value_counts()
    flag_counts_dict = {flag: int(flag_counts.get(flag, 0)) for flag in ALL_FLAGS}
    coverage = {
        "n_validation_rows": n_validation,
        "flag_counts": flag_counts_dict,
        "flag_pct": {
            flag: (round(100.0 * n / n_validation, 4) if n_validation else float("nan"))
            for flag, n in flag_counts_dict.items()
        },
    }

    scored = merged[merged["modified_zscore"].notna()].copy()
    n_scored = len(scored)

    decile_rows: list[dict] = []
    if n_scored > 0:
        try:
            scored["_decile"] = pd.qcut(scored["modified_zscore"], 10, duplicates="drop")
        except ValueError:
            scored["_decile"] = pd.qcut(scored["modified_zscore"].rank(method="first"), min(10, n_scored))
        for decile, sub in scored.groupby("_decile", observed=True):
            decile_rows.append(
                {
                    "score_decile": str(decile),
                    "n_rows": int(len(sub)),
                    "fraud_rate": float(sub["isFraud"].mean()) if len(sub) else float("nan"),
                }
            )

    outlier_vs_normal: list[dict] = []
    for flag_name in [FLAG_SCORED_OUTLIER, FLAG_SCORED_NORMAL]:
        sub = scored[scored["flag"] == flag_name]
        outlier_vs_normal.append(
            {
                "flag": flag_name,
                "n_rows": int(len(sub)),
                "fraud_rate": float(sub["isFraud"].mean()) if len(sub) else float("nan"),
            }
        )

    auc = _roc_auc(scored["modified_zscore"].abs(), scored["isFraud"]) if n_scored > 0 else float("nan")

    return {
        "coverage": coverage,
        "n_scored_rows": n_scored,
        "fraud_rate_by_score_decile": decile_rows,
        "fraud_rate_outlier_vs_normal": outlier_vs_normal,
        "roc_auc_abs_modified_zscore_vs_isFraud": auc,
        "roc_auc_n_rows": n_scored,
    }


def main() -> None:
    t0 = time.time()
    config = load_config()
    split_config = load_split_config()
    detection_config = load_detection_config()
    out_dir = config.reports_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading TransactionID/TransactionDT/TransactionAmt/%s from train_transaction.csv -- isFraud is "
        "never read while building the detector, only by the validation-only evaluation step below",
        config.payment_proxy_key_columns,
    )
    development, n_rows_total, n_holdout_excluded = build_development_frame(config, split_config)
    logger.info(
        "Restricting to development partitions before any group-key/history/score computation: %d/%d rows "
        "(%d holdout rows loaded then excluded, never reaching build_group_key/compute_behavioral_change_score)",
        len(development),
        n_rows_total,
        n_holdout_excluded,
    )

    valid = build_group_key(development, config.payment_proxy_key_columns, key_name=GROUP_KEY_NAME)
    n_rows_missing_key = len(development) - len(valid)
    logger.info(
        "payment_proxy_key present on %d/%d development rows (%d excluded, missing a key component)",
        len(valid),
        len(development),
        n_rows_missing_key,
    )

    logger.info(
        "Scoring with fixed detection_config (configs/detection.yaml): min_history_for_score=%d, "
        "window_size_events=%d, modified_zscore_scale_constant=%s, modified_zscore_threshold=%s, "
        "zero_mad_epsilon=%s",
        detection_config.min_history_for_score,
        detection_config.window_size_events,
        detection_config.modified_zscore_scale_constant,
        detection_config.modified_zscore_threshold,
        detection_config.zero_mad_epsilon,
    )
    scores = compute_behavioral_change_score(
        valid,
        detection_config,
        group_col=GROUP_KEY_NAME,
        amount_col="TransactionAmt",
        dt_col=config.dt_column,
    )
    scored = pd.concat([valid[["TransactionID", "partition"]], scores], axis=1)

    coverage_by_partition = []
    for name in DEVELOPMENT_PARTITIONS:
        sub = scored[scored["partition"] == name]
        counts = sub["flag"].value_counts()
        n = len(sub)
        row = {"partition": name, "n_rows": int(n)}
        row.update(
            {
                f"pct_{flag}": (round(100.0 * int(counts.get(flag, 0)) / n, 4) if n else float("nan"))
                for flag in ALL_FLAGS
            }
        )
        coverage_by_partition.append(row)

    scored_only = scored[scored["modified_zscore"].notna()]
    score_distribution: dict = {}
    if len(scored_only):
        q = scored_only["modified_zscore"].abs().quantile([0.5, 0.75, 0.9, 0.99])
        score_distribution = {
            "n_scored_rows": int(len(scored_only)),
            "abs_modified_zscore_p50": float(q.loc[0.5]),
            "abs_modified_zscore_p75": float(q.loc[0.75]),
            "abs_modified_zscore_p90": float(q.loc[0.9]),
            "abs_modified_zscore_p99": float(q.loc[0.99]),
        }

    logger.info("Running validation-only target-association evaluation (isFraud read here only)...")
    scored_validation = scored[scored["partition"] == "validation"].copy()
    evaluation = evaluate_validation_only(scored_validation, config)

    results: dict = {
        "split_config": {
            name: {"start_day": pr.start_day, "end_day": pr.end_day} for name, pr in split_config.partitions.items()
        },
        "detection_config": {
            "min_history_for_score": detection_config.min_history_for_score,
            "window_size_events": detection_config.window_size_events,
            "modified_zscore_scale_constant": detection_config.modified_zscore_scale_constant,
            "modified_zscore_threshold": detection_config.modified_zscore_threshold,
            "zero_mad_epsilon": detection_config.zero_mad_epsilon,
        },
        "n_rows_total": n_rows_total,
        "n_rows_development": int(len(development)),
        "n_rows_holdout_excluded": n_holdout_excluded,
        "n_rows_missing_payment_proxy_key": int(n_rows_missing_key),
        "n_rows_valid_key": int(len(valid)),
        "coverage_by_partition": coverage_by_partition,
        "score_distribution": score_distribution,
        "validation_evaluation": evaluation,
    }

    results_path = out_dir / "phase_d_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    report_path = out_dir / "phase_d_report.md"
    render_phase_d_report(results, report_path)

    elapsed = time.time() - t0
    logger.info("Phase D detector run complete in %.1fs. Results: %s Report: %s", elapsed, results_path, report_path)


if __name__ == "__main__":
    main()
