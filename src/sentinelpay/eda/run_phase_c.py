"""Phase C orchestration: build the non-target, chronology-safe row-wise
feature foundation (sentinelpay.features.build_feature_frame) over
train/embargo_1/validation/embargo_2 only, validate it, and write
reports/eda/phase_c_results.json + reports/eda/phase_c_report.md.

Holdout sealing: only TransactionID/TransactionDT/TransactionAmt are loaded
(isFraud is never read anywhere in this script). Rows are assigned a
partition and filtered down to DEVELOPMENT_PARTITIONS (train/embargo_1/
validation/embargo_2) BEFORE build_feature_frame is called -- holdout rows
never reach feature computation. configs/split.yaml boundaries are read,
never modified.

No parquet is written -- features are built in memory, validated, and
reported only. sentinelpay.data.history's generic historical utilities are
NOT invoked here (no approved production grouping key yet); see
tests/test_history.py for their correctness evidence instead.

Run with:
    .venv\\Scripts\\python.exe -m sentinelpay.eda.run_phase_c
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from sentinelpay.config import load_config
from sentinelpay.data.loader import load_identity_ids, load_transaction_columns
from sentinelpay.data.split import DEVELOPMENT_PARTITIONS, assign_partition, load_split_config
from sentinelpay.data.temporal import add_day_index
from sentinelpay.eda.generate_report import render_phase_c_report
from sentinelpay.features import build_feature_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase_c")


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _feature_summary_by_partition(feature_df: pd.DataFrame, partitions: pd.Series, feature_names: list[str]) -> list[dict]:
    """Mean/std/pct_missing/n_rows per feature per partition. No fraud rate
    -- isFraud is never loaded by this script."""
    rows = []
    for name in DEVELOPMENT_PARTITIONS:
        mask = partitions == name
        n_rows = int(mask.sum())
        for col in feature_names:
            series = feature_df.loc[mask, col]
            rows.append(
                {
                    "partition": name,
                    "feature": col,
                    "mean": float(series.mean()) if series.notna().any() else float("nan"),
                    "std": float(series.std()) if series.notna().sum() > 1 else float("nan"),
                    "pct_missing": round(100.0 * series.isna().mean(), 4) if n_rows else float("nan"),
                    "n_rows": n_rows,
                }
            )
    return rows


def main() -> None:
    t0 = time.time()
    config = load_config()
    split_config = load_split_config()
    out_dir = config.reports_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading TransactionID/TransactionDT/TransactionAmt only -- isFraud is never read in Phase C")
    base = load_transaction_columns(
        "train", columns=["TransactionID", "TransactionDT", "TransactionAmt"], config=config
    )
    base = add_day_index(base, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    base = assign_partition(base, split_config, day_col="_day")

    non_holdout = base[base["partition"].isin(DEVELOPMENT_PARTITIONS)].copy()
    n_holdout_excluded = int((base["partition"] == "holdout").sum())
    logger.info(
        "Restricting to development partitions before feature computation: %d/%d rows (%d holdout rows loaded then excluded, never reaching build_feature_frame)",
        len(non_holdout),
        len(base),
        n_holdout_excluded,
    )

    identity_ids = set(load_identity_ids("train", config=config).tolist())
    feature_df, registry = build_feature_frame(non_holdout, identity_ids)

    feature_names = [entry["feature"] for entry in registry]
    summary = _feature_summary_by_partition(feature_df, non_holdout["partition"], feature_names)

    results: dict = {
        "split_config": {
            name: {"start_day": pr.start_day, "end_day": pr.end_day} for name, pr in split_config.partitions.items()
        },
        "n_rows_total": int(len(base)),
        "n_rows_development": int(len(non_holdout)),
        "n_rows_holdout_excluded": n_holdout_excluded,
        "feature_registry": registry,
        "feature_summary_by_partition": summary,
    }

    results_path = out_dir / "phase_c_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    report_path = out_dir / "phase_c_report.md"
    render_phase_c_report(results, report_path)

    elapsed = time.time() - t0
    logger.info("Phase C feature foundation complete in %.1fs. Results: %s Report: %s", elapsed, results_path, report_path)


if __name__ == "__main__":
    main()
