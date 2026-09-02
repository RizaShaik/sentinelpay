"""Phase F orchestration: causal, fold-safe, target-derived historical
fraud-rate feature for `payment_proxy_key`, plus its strict
validation-only evaluation.

Scope (approved proposal review): `payment_proxy_key` ONLY. No
`device_proxy_key`, no E.1/E.2 relationship-node keys, no generic
multi-key framework. Two feature recipients only: `train` and `validation`
-- `embargo_1`, `embargo_2`, and `holdout` never receive a computed feature
value and are never a label source for anyone (see
`sentinelpay.target_history` module docstring for the full source/recipient
contract).

**This is the first phase where `isFraud` is a legitimate INPUT to feature
computation**, not merely a downstream diagnostic -- loaded once, up front,
alongside `TransactionID`/`TransactionDT`/`payment_proxy_key` columns. The
safety guarantee this project has enforced everywhere else ("isFraud not
loaded until the evaluation step") does not apply verbatim here; what still
applies, unchanged: no row's feature value may depend on that row's own or
any later/embargoed row's label (enforced by `sentinelpay.target_history`'s
strictly-causal, explicit-pool construction), and no hyperparameter
(`SMOOTHING_K`, the fixed fraud-rate bucket edges below) is selected, tuned,
or changed based on the validation-only evaluation's results -- every one is
fixed before that evaluation runs, exactly like Phase D's
`configs/detection.yaml` hyperparameters.

Holdout sealing: identical to every prior phase -- `holdout` rows are
excluded via `DEVELOPMENT_PARTITIONS` filtering before `build_group_key`/
`build_eligible_pools`/`compute_prior_fraud_rate` are ever called.

Run with:
    .venv\\Scripts\\python.exe -m sentinelpay.eda.run_phase_f
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sentinelpay.config import DataConfig, load_config
from sentinelpay.data.loader import load_transaction_columns
from sentinelpay.data.split import DEVELOPMENT_PARTITIONS, assign_partition, load_split_config
from sentinelpay.data.temporal import add_day_index
from sentinelpay.eda.generate_report import render_phase_f_report
from sentinelpay.eda.grouping_key_sufficiency import build_group_key
from sentinelpay.target_history import (
    OUTPUT_COLUMNS,
    SMOOTHING_K,
    SUFFICIENT_HISTORY_THRESHOLD,
    TRAIN_RECIPIENT_PARTITIONS,
    VALIDATION_SOURCE_PARTITIONS,
    build_eligible_pools,
    compute_prior_fraud_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase_f")

GROUP_KEY_NAME = "_payment_group_key"


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _rate_distribution(series: pd.Series) -> dict:
    """Float-safe percentile summary (unlike
    `sentinelpay.eda.grouping_key_sufficiency.prior_count_distribution`,
    which casts min/max to `int` -- not valid for a [0, 1] rate). Computed
    over non-NaN values only; `n_nan` is reported separately so cold-start
    rows are visible rather than silently dropped."""
    non_nan = series.dropna()
    n_nan = int(series.isna().sum())
    if len(non_nan) == 0:
        return {"n_rows": int(len(series)), "n_nan": n_nan}
    q = non_nan.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "n_rows": int(len(series)),
        "n_nan": n_nan,
        "min": float(non_nan.min()),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.9]),
        "p99": float(q.loc[0.99]),
        "max": float(non_nan.max()),
        "mean": float(non_nan.mean()),
    }


def _bool_coverage(series: pd.Series) -> dict:
    n = len(series)
    n_true = int(series.sum())
    return {"n_rows": n, "n_true": n_true, "n_false": n - n_true, "pct_true": round(100.0 * n_true / n, 4) if n else float("nan")}


def build_development_frame(config: DataConfig, split_config) -> tuple[pd.DataFrame, int, int]:
    """Load TransactionID/TransactionDT/payment_proxy_key/isFraud columns,
    assign partitions, and filter to `DEVELOPMENT_PARTITIONS` before any
    key-building or history computation. Factored out for testability with a
    monkeypatched loader (mirrors `run_phase_d.build_development_frame`).

    Returns `(development_df, n_rows_total, n_holdout_excluded)`.
    """
    base = load_transaction_columns(
        "train",
        columns=["TransactionID", "TransactionDT", "isFraud"] + config.payment_proxy_key_columns,
        config=config,
    )
    base = add_day_index(base, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    base = assign_partition(base, split_config, day_col="_day")

    development = base[base["partition"].isin(DEVELOPMENT_PARTITIONS)].copy()
    n_holdout_excluded = int((base["partition"] == "holdout").sum())
    return development, int(len(base)), n_holdout_excluded


def _roc_auc(score: pd.Series, target: pd.Series) -> float:
    """Rank-based (Mann-Whitney U) ROC-AUC -- same formula as
    `sentinelpay.eda.run_phase_d._roc_auc` / `run_phase_e2._roc_auc`,
    duplicated rather than imported so this script's evaluation stays
    self-contained, matching this project's existing per-phase-script
    convention."""
    y = target.to_numpy()
    s = score.to_numpy()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = scipy_stats.rankdata(s)
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fraud_rate_by_fixed_bucket(sub: pd.DataFrame, value_col: str, edges: list[float], labels: list[str]) -> list[dict]:
    bucket = pd.cut(sub[value_col], bins=[-np.inf] + edges + [np.inf], labels=labels, right=False)
    rows = []
    for label in labels:
        g = sub[bucket == label]
        rows.append(
            {"bucket": label, "n_rows": int(len(g)), "fraud_rate": float(g["isFraud"].mean()) if len(g) else float("nan")}
        )
    return rows


def _fraud_rate_by_bool(sub: pd.DataFrame, bool_col: str) -> list[dict]:
    rows = []
    for val in [True, False]:
        g = sub[sub[bool_col] == val]
        rows.append({bool_col: val, "n_rows": int(len(g)), "fraud_rate": float(g["isFraud"].mean()) if len(g) else float("nan")})
    return rows


def evaluate_target_derived_validation_only(
    scored_validation: pd.DataFrame, fraud_rate_bucket_edges: list[float], fraud_rate_bucket_labels: list[str]
) -> dict:
    """Strict validation-only evaluation of the already-final,
    already-computed target-derived feature. Requires every row already
    `partition == "validation"`; reuses `isFraud` already loaded in
    `scored_validation` (no second load -- unlike Phase D/E.2, `isFraud` was
    already a legitimate input to feature computation in this phase, not
    newly introduced here). Never mutates a feature value, never selects
    `SMOOTHING_K` or `fraud_rate_bucket_edges` from what it finds here --
    both are fixed by the caller before this function runs.

    Reports, all validation-only: ROC-AUC of the smoothed rate vs.
    `isFraud`; ROC-AUC of the raw (unsmoothed) rate vs. `isFraud` (over its
    own naturally-defined, smaller, non-cold-start subset -- reported with
    its own `n_rows` so the comparison is transparent about using a
    different population, not silently treated as directly comparable);
    fraud rate by fixed `smoothed_rate` bucket; fraud rate by
    `sufficient_target_history`; and cold-start/global-cold-start coverage.
    Descriptive evidence only -- not a selection or tuning step.
    """
    if not (scored_validation["partition"] == "validation").all():
        raise ValueError("evaluate_target_derived_validation_only requires every row to already be partition == 'validation'")

    n_validation = len(scored_validation)
    has_raw = scored_validation["payment_proxy_prior_fraud_rate_raw"].notna()
    raw_subset = scored_validation[has_raw]

    return {
        "n_validation_rows": n_validation,
        "fraud_rate_overall": float(scored_validation["isFraud"].mean()) if n_validation else float("nan"),
        "fraud_rate_bucket_edges": fraud_rate_bucket_edges,
        "fraud_rate_bucket_labels": fraud_rate_bucket_labels,
        "roc_auc_smoothed_rate_vs_isFraud": (
            _roc_auc(scored_validation["payment_proxy_prior_fraud_rate_smoothed"], scored_validation["isFraud"])
            if n_validation
            else float("nan")
        ),
        "roc_auc_raw_rate_vs_isFraud": (
            _roc_auc(raw_subset["payment_proxy_prior_fraud_rate_raw"], raw_subset["isFraud"]) if len(raw_subset) else float("nan")
        ),
        "roc_auc_raw_rate_n_rows": int(len(raw_subset)),
        "fraud_rate_by_smoothed_rate_bucket": _fraud_rate_by_fixed_bucket(
            scored_validation, "payment_proxy_prior_fraud_rate_smoothed", fraud_rate_bucket_edges, fraud_rate_bucket_labels
        ),
        "fraud_rate_by_sufficient_target_history": _fraud_rate_by_bool(scored_validation, "sufficient_target_history"),
        "sufficient_target_history_coverage": _bool_coverage(scored_validation["sufficient_target_history"]),
        "global_cold_start_coverage": _bool_coverage(scored_validation["global_cold_start"]),
    }


def main() -> None:
    t0 = time.time()
    config = load_config()
    split_config = load_split_config()
    out_dir = config.reports_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading TransactionID/TransactionDT/isFraud/%s from train_transaction.csv -- isFraud IS a legitimate "
        "input to feature computation in this phase (the first phase where that's true), never used out of "
        "strictly-causal order",
        config.payment_proxy_key_columns,
    )
    development, n_rows_total, n_holdout_excluded = build_development_frame(config, split_config)
    logger.info(
        "Restricting to development partitions before any key-building or history computation: %d/%d rows "
        "(%d holdout rows loaded then excluded, never reaching build_group_key/build_eligible_pools)",
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

    train_pool, validation_pool = build_eligible_pools(valid, partition_col="partition")
    logger.info(
        "Eligible pools built explicitly: train_pool=%d rows (partitions=%s), validation_pool=%d rows "
        "(partitions=%s) -- embargo_1/embargo_2/holdout are absent from both by construction",
        len(train_pool),
        sorted(train_pool["partition"].unique().tolist()),
        len(validation_pool),
        sorted(validation_pool["partition"].unique().tolist()),
    )

    logger.info("Computing payment_proxy_key prior fraud-rate feature (SMOOTHING_K=%.1f, fixed, not tuned)...", SMOOTHING_K)
    train_features = compute_prior_fraud_rate(
        train_pool,
        allowed_source_partitions=TRAIN_RECIPIENT_PARTITIONS,
        group_col=GROUP_KEY_NAME,
        dt_col=config.dt_column,
        target_col="isFraud",
        partition_col="partition",
    )
    validation_pool_features = compute_prior_fraud_rate(
        validation_pool,
        allowed_source_partitions=VALIDATION_SOURCE_PARTITIONS,
        group_col=GROUP_KEY_NAME,
        dt_col=config.dt_column,
        target_col="isFraud",
        partition_col="partition",
    )
    validation_mask = validation_pool["partition"] == "validation"
    validation_features = validation_pool_features[validation_mask]

    train_scored = pd.concat([train_pool[["TransactionID", "partition", "isFraud"]], train_features], axis=1)
    validation_scored = pd.concat(
        [validation_pool.loc[validation_features.index, ["TransactionID", "partition", "isFraud"]], validation_features],
        axis=1,
    )

    feature_summary = {
        "train": {
            "n_rows": int(len(train_scored)),
            "payment_proxy_prior_fraud_rate_smoothed": _rate_distribution(train_scored["payment_proxy_prior_fraud_rate_smoothed"]),
            "payment_proxy_prior_fraud_rate_raw": _rate_distribution(train_scored["payment_proxy_prior_fraud_rate_raw"]),
            "sufficient_target_history": _bool_coverage(train_scored["sufficient_target_history"]),
            "global_cold_start": _bool_coverage(train_scored["global_cold_start"]),
        },
        "validation": {
            "n_rows": int(len(validation_scored)),
            "payment_proxy_prior_fraud_rate_smoothed": _rate_distribution(
                validation_scored["payment_proxy_prior_fraud_rate_smoothed"]
            ),
            "payment_proxy_prior_fraud_rate_raw": _rate_distribution(validation_scored["payment_proxy_prior_fraud_rate_raw"]),
            "sufficient_target_history": _bool_coverage(validation_scored["sufficient_target_history"]),
            "global_cold_start": _bool_coverage(validation_scored["global_cold_start"]),
        },
    }
    logger.info(
        "Feature computed. train n=%d, validation n=%d. train smoothed-rate p50=%.6f, validation smoothed-rate p50=%.6f",
        feature_summary["train"]["n_rows"],
        feature_summary["validation"]["n_rows"],
        feature_summary["train"]["payment_proxy_prior_fraud_rate_smoothed"].get("p50", float("nan")),
        feature_summary["validation"]["payment_proxy_prior_fraud_rate_smoothed"].get("p50", float("nan")),
    )

    # Fixed fraud-rate bucket edges for the validation-only evaluation below,
    # declared from TRAIN's OWN already-computed smoothed-rate percentiles
    # (never validation's) -- fixed BEFORE the validation-only evaluation
    # runs, matching this project's E.2 precedent of using already-published
    # non-target/non-validation percentiles as fixed bin edges.
    train_dist = feature_summary["train"]["payment_proxy_prior_fraud_rate_smoothed"]
    fraud_rate_bucket_edges = [train_dist["p25"], train_dist["p50"], train_dist["p75"], train_dist["p90"]]
    fraud_rate_bucket_labels = ["rate_lt_p25", "rate_p25_to_p50", "rate_p50_to_p75", "rate_p75_to_p90", "rate_ge_p90"]
    logger.info(
        "Fixed fraud_rate_bucket_edges declared from train's own smoothed-rate p25/p50/p75/p90 (BEFORE "
        "validation-only evaluation runs): %s",
        fraud_rate_bucket_edges,
    )

    logger.info("Running the strict validation-only evaluation (isFraud already loaded, reused here)...")
    evaluation = evaluate_target_derived_validation_only(validation_scored, fraud_rate_bucket_edges, fraud_rate_bucket_labels)

    results: dict = {
        "split_config": {
            name: {"start_day": pr.start_day, "end_day": pr.end_day} for name, pr in split_config.partitions.items()
        },
        "smoothing_k": SMOOTHING_K,
        "sufficient_history_threshold": SUFFICIENT_HISTORY_THRESHOLD,
        "n_rows_total": n_rows_total,
        "n_rows_development": int(len(development)),
        "n_rows_holdout_excluded": n_holdout_excluded,
        "n_rows_missing_payment_proxy_key": int(n_rows_missing_key),
        "n_rows_valid_key": int(len(valid)),
        "eligible_pools": {
            "train_pool_n_rows": int(len(train_pool)),
            "train_pool_partitions": sorted(train_pool["partition"].unique().tolist()),
            "validation_pool_n_rows": int(len(validation_pool)),
            "validation_pool_partitions": sorted(validation_pool["partition"].unique().tolist()),
        },
        "feature_summary": feature_summary,
        "validation_evaluation": evaluation,
    }

    results_path = out_dir / "phase_f_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    report_path = out_dir / "phase_f_report.md"
    render_phase_f_report(results, report_path)

    elapsed = time.time() - t0
    logger.info("Phase F complete in %.1fs. Results: %s Report: %s", elapsed, results_path, report_path)


if __name__ == "__main__":
    main()
