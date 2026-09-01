"""Phase D.1 orchestration: non-target grouping-key sufficiency analysis.

Determines whether `payment_proxy_key` (card1/card2/card3/card5/addr1) or
`device_proxy_key` (DeviceInfo/id_31) has enough strictly-causal historical
density to support a future Phase D per-entity behavioral-change detector.
This is NOT Phase D itself: no detector, rolling median/MAD, EWMA, CUSUM,
target encoding, fraud-rate evaluation, or parquet persistence is
implemented here or anywhere D.1 touches.

Holdout sealing: TransactionID/TransactionDT/payment_proxy_key columns are
read from train_transaction.csv (all partitions are in that one file);
DeviceInfo/id_31 are read from train_identity.csv and left-joined on
TransactionID. `isFraud` is never read anywhere in this script. Rows are
assigned a partition and filtered to DEVELOPMENT_PARTITIONS BEFORE
sentinelpay.eda.grouping_key_sufficiency.analyze_grouping_key is ever
called -- holdout rows never reach any group-size, history, or
event-frequency computation.

configs/split.yaml is read, never modified.

Run with:
    .venv\\Scripts\\python.exe -m sentinelpay.eda.run_phase_d1
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
from sentinelpay.eda.generate_report import render_phase_d1_report
from sentinelpay.eda.grouping_key_sufficiency import (
    analyze_grouping_key,
    evaluate_key_sufficiency,
    recommend_grouping_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase_d1")


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
        "Loading TransactionID/TransactionDT/%s from train_transaction.csv -- isFraud is never read in Phase D.1",
        config.payment_proxy_key_columns,
    )
    base = load_transaction_columns(
        "train",
        columns=["TransactionID", "TransactionDT"] + config.payment_proxy_key_columns,
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
        "Restricting to development partitions before any grouping-key content analysis: %d/%d rows "
        "(%d holdout rows loaded then excluded, never reaching analyze_grouping_key)",
        len(non_holdout),
        len(base),
        n_holdout_excluded,
    )

    logger.info("Analyzing payment_proxy_key...")
    payment_result = analyze_grouping_key(
        non_holdout,
        key_columns=config.payment_proxy_key_columns,
        dt_col=config.dt_column,
        partition_col="partition",
        partitions=DEVELOPMENT_PARTITIONS,
        key_name="_payment_group_key",
    )

    logger.info("Analyzing device_proxy_key...")
    device_result = analyze_grouping_key(
        non_holdout,
        key_columns=config.device_proxy_key_columns,
        dt_col=config.dt_column,
        partition_col="partition",
        partitions=DEVELOPMENT_PARTITIONS,
        key_name="_device_group_key",
    )

    payment_eval = evaluate_key_sufficiency(payment_result)
    device_eval = evaluate_key_sufficiency(device_result)
    recommendation = recommend_grouping_key(payment_eval, device_eval)

    results: dict = {
        "split_config": {
            name: {"start_day": pr.start_day, "end_day": pr.end_day} for name, pr in split_config.partitions.items()
        },
        "n_rows_total": int(len(base)),
        "n_rows_development": int(len(non_holdout)),
        "n_rows_holdout_excluded": n_holdout_excluded,
        "payment_proxy_key": payment_result,
        "device_proxy_key": device_result,
        "payment_proxy_key_evaluation": payment_eval,
        "device_proxy_key_evaluation": device_eval,
        "recommendation": recommendation,
    }

    results_path = out_dir / "phase_d1_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    report_path = out_dir / "phase_d1_report.md"
    render_phase_d1_report(results, report_path)

    elapsed = time.time() - t0
    logger.info("Phase D.1 sufficiency analysis complete in %.1fs. Results: %s Report: %s", elapsed, results_path, report_path)


if __name__ == "__main__":
    main()
