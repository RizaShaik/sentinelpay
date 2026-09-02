"""Phase H orchestration: the ONE final sealed-holdout evaluation of the
frozen Phase F/G design. `holdout` is reserved for this phase alone (see
`configs/split.yaml`'s own docstring) -- this is the one, single,
authorized place in this entire project that `holdout` rows are ever
loaded from disk.

`sentinelpay.target_history`, `sentinelpay.model_features`, and
`sentinelpay.model_evaluation` are all consumed COMPLETELY UNMODIFIED --
every function this script calls from those three modules is imported and
used exactly as committed in Phases F and G. No new causal-history logic is
written here beyond the one small, explicitly-scoped extension in section
"Phase F holdout target-history" below.

===========================================================================
FROZEN TRAINING REGIME
===========================================================================
`train` -> fit (only). `validation` -> the selection Phase G already made
(F2 graduated over B2; validation's own numbers are read verbatim from
`reports/eda/phase_g_results.json`, never recomputed here). `holdout` -> one
final evaluation, read exactly once. Only `B2`/`F1`/`F2` are refit
(deterministically, on `train` only -- `StandardScaler`/`LogisticRegression`
with lbfgs are deterministic given fixed code and data) and evaluated on
holdout. `B0`/`B1` are NOT touched by this script at all.

===========================================================================
PHASE F HOLDOUT TARGET-HISTORY -- Option A, online causal continuation
(explicitly settled protocol; see proposal review)
===========================================================================

    train row       <- strictly-earlier train labels ONLY (unchanged from
                        Phase F).
    validation row   <- train + strictly-earlier validation labels
                        (unchanged from Phase F).
    holdout row      <- train + validation + strictly-earlier holdout
                        labels sharing the same payment_proxy_key.

`embargo_1`/`embargo_2` are NEVER a label source for ANY recipient,
including `holdout` -- unchanged, direct extension of Phase F's own already-
approved contract (an embargo partition is never a label source for its
downstream neighbor; embargo_2 plays the identical structural role for the
validation/holdout boundary that embargo_1 plays for the train/validation
boundary). Future holdout rows and same-TransactionDT holdout rows are
excluded by the same strictly-earlier, tie-collapsing causal contract
`sentinelpay.target_history.compute_prior_fraud_rate` already proves
generically -- reused UNMODIFIED here, called with a THIRD
`allowed_source_partitions` value ({"train", "validation", "holdout"}) that
was already a fully generic, caller-supplied parameter. No change to
`target_history.py` was needed or made.

**EXPLICIT HOLDOUT-LABEL INVARIANT**: a holdout row's `isFraud` may be used
ONLY as a Phase F target-history source for strictly LATER holdout rows
sharing its `payment_proxy_key`. It must never affect: (a) its own feature
vector, (b) an earlier holdout row's feature vector, (c) a same-
TransactionDT holdout row's feature vector. It must never enter model
training (`y_train` is `train`-partition `isFraud` only -- no holdout row's
label is ever part of any training target) or any model feature column (no
`LADDER_FEATURE_COLUMNS` entry is `isFraud` itself, and a holdout row's own
`isFraud` is never one of its own `X_holdout` values). See
`tests/test_run_phase_h.py` for the integration-level proof of every clause
of this invariant.

**Phase D contrast, unchanged**: Phase D's non-target history for `holdout`
is computed over ALL FIVE partitions concatenated (`train`+`embargo_1`+
`validation`+`embargo_2`+`holdout`), exactly matching its existing,
already-approved, unrestricted contract (`sentinelpay.model_features.compute_phase_d_features`,
reused unmodified) -- Phase D was never embargo-restricted; that restriction
is specific to Phase F's target-derived reasoning. This gives the SAME
intentional Phase D-vs-Phase F asymmetry Phase G already proved for
`validation`, now also proved for `holdout`.

===========================================================================
ONE-SHOT GUARD
===========================================================================
`assert_holdout_not_yet_evaluated(results_path)` is called as the FIRST
executable step of `main()` -- before `load_split_config`, before any
`load_transaction_columns` call, before anything else -- and raises if
`reports/eda/phase_h_results.json` already exists. Any subsequent edit to
`target_history.py`/`model_features.py`/`model_evaluation.py` after a
holdout run permanently invalidates that read for this design; a fresh
holdout would be required, and none exists. Tests call this function with
`tmp_path`-derived paths ONLY -- never the real path (see
`tests/test_run_phase_h.py`).

Run with (ONE TIME ONLY, once explicitly authorized):
    .venv\\Scripts\\python.exe -m sentinelpay.eda.run_phase_h
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from sentinelpay.config import DataConfig, DetectionConfig, load_config, load_detection_config
from sentinelpay.data.loader import load_identity_ids, load_transaction_columns
from sentinelpay.data.split import PARTITION_ORDER, assign_partition, load_split_config
from sentinelpay.data.temporal import add_day_index
from sentinelpay.eda.generate_report import render_phase_h_report
from sentinelpay.eda.grouping_key_sufficiency import build_group_key
from sentinelpay.eda.run_phase_g import GRADUATION_RELATIVE_LIFT_THRESHOLD
from sentinelpay.model_evaluation import (
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    LOGREG_MAX_ITER,
    bootstrap_pr_auc_delta_ci,
    fit_and_score,
)
from sentinelpay.model_features import (
    IMPUTE_FIXED_VALUE,
    LADDER_FEATURE_COLUMNS,
    PAYMENT_GROUP_COL,
    PHASE_D_IMPUTED_COLUMNS,
    PHASE_D_NUMERIC_COLUMNS,
    _one_hot_flag,
    compute_phase_c_features,
    compute_phase_d_features,
    compute_phase_f_features,
    get_ladder_matrix,
)
from sentinelpay.target_history import compute_prior_fraud_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_phase_h")

LADDER_STEPS_EVALUATED = ["B2", "F1", "F2"]  # B0/B1 are not touched by Phase H

# The one new, explicitly-scoped extension of Phase F's contract for this
# phase: holdout draws on train + validation + its own strictly-earlier
# rows. embargo_1/embargo_2 remain excluded, exactly as for every other
# recipient.
HOLDOUT_SOURCE_PARTITIONS = {"train", "validation", "holdout"}


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def assert_holdout_not_yet_evaluated(results_path: Path) -> None:
    """Raises `FileExistsError` if `results_path` already exists. Called as
    the FIRST executable step of `main()`, before any data is loaded.
    Callers (tests included) must pass an explicit path -- there is no
    hardcoded default, specifically so tests can never accidentally target
    the real sealed-evaluation artifact.
    """
    if results_path.exists():
        raise FileExistsError(
            f"{results_path} already exists -- Phase H is a ONE-TIME sealed holdout evaluation. "
            "Refusing to run again. Delete this file only if you are certain a fresh holdout read "
            "is explicitly authorized (see sentinelpay.eda.run_phase_h module docstring's ONE-SHOT "
            "GUARD section) -- any code change to target_history.py/model_features.py/"
            "model_evaluation.py since the last run permanently invalidates this design's prior "
            "holdout evidence regardless."
        )


def build_full_frame(config: DataConfig, split_config) -> tuple[pd.DataFrame, int]:
    """Load TransactionID/TransactionDT/TransactionAmt/isFraud/payment_proxy_key
    columns and assign partitions over the ENTIRE raw file -- unlike every
    prior phase, this does NOT filter out `holdout`; Phase H is the one
    authorized place that happens. Factored out for testability with a
    monkeypatched loader.

    Returns `(full_df, n_rows_total)`.
    """
    base = load_transaction_columns(
        "train",
        columns=["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"] + config.payment_proxy_key_columns,
        config=config,
    )
    base = add_day_index(base, dt_col=config.dt_column, seconds_per_day=config.seconds_per_day)
    base = assign_partition(base, split_config, day_col="_day")
    return base, int(len(base))


def build_holdout_eligible_pool(valid_df: pd.DataFrame, partition_col: str = "partition") -> pd.DataFrame:
    """Explicit eligible-pool construction for Phase H's one new recipient,
    `holdout` -- direct extension of `sentinelpay.target_history.build_eligible_pools`'s
    own pattern (not added there, to keep that module unmodified). Returns
    rows with partition in {"train", "validation", "holdout"} ONLY --
    `embargo_1`/`embargo_2` excluded by construction, matching Phase F's own
    contract exactly."""
    if partition_col not in valid_df.columns:
        raise ValueError(f"build_holdout_eligible_pool requires column '{partition_col}'")
    return valid_df[valid_df[partition_col].isin(sorted(HOLDOUT_SOURCE_PARTITIONS))].copy()


def compute_phase_f_holdout_features(
    valid_df: pd.DataFrame, dt_col: str = "TransactionDT", group_col: str = PAYMENT_GROUP_COL, partition_col: str = "partition"
) -> pd.DataFrame:
    """Holdout's Phase F target-history features -- Option A, online causal
    continuation (see module docstring). Reuses
    `sentinelpay.target_history.compute_prior_fraud_rate` COMPLETELY
    UNMODIFIED, called with `allowed_source_partitions=HOLDOUT_SOURCE_PARTITIONS`
    (a parameter that primitive already exposed generically). Returns only
    `holdout`-partition rows of the output, aligned to their original index
    -- `train`/`validation` rows recomputed as part of this same pool call
    are discarded here (their OWN canonical Phase F features come from
    `sentinelpay.model_features.compute_phase_f_features` instead, matching
    Phase F's own two-recipient definition exactly, unchanged).
    """
    pool = build_holdout_eligible_pool(valid_df, partition_col=partition_col)
    features = compute_prior_fraud_rate(
        pool, allowed_source_partitions=HOLDOUT_SOURCE_PARTITIONS, group_col=group_col, dt_col=dt_col, partition_col=partition_col
    )
    holdout_mask = pool[partition_col] == "holdout"
    return features[holdout_mask]


def _assemble_feature_block(row_index: pd.Index, c_all: pd.DataFrame, d_all: pd.DataFrame, f_features: pd.DataFrame) -> pd.DataFrame:
    """Join Phase C + Phase D + Phase F feature blocks for exactly
    `row_index`, with the SAME fixed imputation
    `sentinelpay.model_features.assemble_ladder_matrix` already applies --
    duplicated here (not imported, since it is a few lines of `pd.concat`/
    `fillna` glue, not causal logic) ONLY because `assemble_ladder_matrix`
    itself is hardcoded to `train`+`validation` recipients and must not be
    modified. Every constant/helper it uses (`PHASE_D_NUMERIC_COLUMNS`,
    `PHASE_D_IMPUTED_COLUMNS`, `IMPUTE_FIXED_VALUE`, `_one_hot_flag`) is
    imported from `sentinelpay.model_features`, not redefined.
    """
    c_aligned = c_all.loc[row_index]
    d_selected = d_all.loc[row_index, PHASE_D_NUMERIC_COLUMNS + ["flag"]]
    flag_dummies = _one_hot_flag(d_selected["flag"])
    f_aligned = f_features.loc[row_index]

    out = pd.concat([c_aligned, d_selected, flag_dummies, f_aligned], axis=1)

    for col in PHASE_D_IMPUTED_COLUMNS:
        out[col] = out[col].fillna(IMPUTE_FIXED_VALUE)
    out["payment_proxy_prior_fraud_rate_smoothed"] = out["payment_proxy_prior_fraud_rate_smoothed"].fillna(IMPUTE_FIXED_VALUE)
    out["global_cold_start"] = out["global_cold_start"].astype("int64")
    out["sufficient_target_history"] = out["sufficient_target_history"].astype("int64")
    return out


def assemble_train_and_holdout_matrices(
    valid_df: pd.DataFrame,
    identity_ids,
    detection_config: DetectionConfig,
    dt_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    group_col: str = PAYMENT_GROUP_COL,
    partition_col: str = "partition",
    id_col: str = "TransactionID",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the full feature superset for `train` rows (for fitting) and
    `holdout` rows (for the one sealed evaluation) of `valid_df`. `valid_df`
    must already be an already-`payment_proxy_key`-valid frame covering ALL
    FIVE partitions (`train`/`embargo_1`/`validation`/`embargo_2`/`holdout`)
    -- Phase D's unrestricted history needs every partition present; Phase
    F's contract (see module docstring) determines which of them are
    actually used as a label SOURCE, independent of which are present as
    rows in `valid_df`.

    Returns `(train_assembled, holdout_assembled)`, each with
    `TransactionID`/`TransactionDT`/`partition`/`isFraud` plus every
    `LADDER_FEATURE_COLUMNS["F2"]` column -- callers slice via
    `sentinelpay.model_features.get_ladder_matrix`.
    """
    required = {dt_col, amount_col, group_col, partition_col, id_col, "isFraud"}
    for col in required:
        if col not in valid_df.columns:
            raise ValueError(f"assemble_train_and_holdout_matrices requires column '{col}'")

    c_all = compute_phase_c_features(valid_df, identity_ids, amount_col=amount_col, dt_col=dt_col, id_col=id_col)
    d_all = compute_phase_d_features(valid_df, detection_config, group_col=group_col, amount_col=amount_col, dt_col=dt_col)
    f_train_val = compute_phase_f_features(valid_df, dt_col=dt_col, group_col=group_col, partition_col=partition_col)
    f_holdout = compute_phase_f_holdout_features(valid_df, dt_col=dt_col, group_col=group_col, partition_col=partition_col)

    train_index = valid_df.index[valid_df[partition_col] == "train"]
    holdout_index = valid_df.index[valid_df[partition_col] == "holdout"]

    base_cols = [id_col, dt_col, partition_col, "isFraud"]
    train_base = valid_df.loc[train_index, base_cols]
    holdout_base = valid_df.loc[holdout_index, base_cols]

    train_features = _assemble_feature_block(train_index, c_all, d_all, f_train_val)
    holdout_features = _assemble_feature_block(holdout_index, c_all, d_all, f_holdout)

    train_assembled = pd.concat([train_base, train_features], axis=1)
    holdout_assembled = pd.concat([holdout_base, holdout_features], axis=1)
    return train_assembled, holdout_assembled


def main() -> None:
    t0 = time.time()
    config = load_config()
    out_dir = config.reports_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "phase_h_results.json"

    # ONE-SHOT GUARD -- the absolute first thing this script does, before
    # any data (holdout or otherwise) is loaded.
    assert_holdout_not_yet_evaluated(results_path)

    split_config = load_split_config()
    detection_config = load_detection_config()

    logger.info(
        "One-shot guard passed (%s does not yet exist). Loading TransactionID/TransactionDT/TransactionAmt/"
        "isFraud/%s from train_transaction.csv, plus identity ids -- this IS the one authorized load of "
        "holdout rows in this project.",
        results_path,
        config.payment_proxy_key_columns,
    )
    identity_ids = load_identity_ids("train", config=config)
    full, n_rows_total = build_full_frame(config, split_config)
    partition_counts = full["partition"].value_counts().reindex(PARTITION_ORDER).to_dict()
    logger.info("Rows per partition (all five): %s", partition_counts)

    valid = build_group_key(full, config.payment_proxy_key_columns, key_name=PAYMENT_GROUP_COL)
    n_rows_missing_key = len(full) - len(valid)
    logger.info(
        "payment_proxy_key present on %d/%d total rows (%d excluded, missing a key component)",
        len(valid),
        len(full),
        n_rows_missing_key,
    )

    logger.info(
        "Assembling train (fit) and holdout (final evaluation) feature matrices -- Phase F holdout "
        "history = train + validation + strictly-earlier holdout, embargo_1/embargo_2 excluded; "
        "Phase D unrestricted across all five partitions..."
    )
    train_assembled, holdout_assembled = assemble_train_and_holdout_matrices(
        valid,
        identity_ids,
        detection_config,
        dt_col=config.dt_column,
        amount_col="TransactionAmt",
        group_col=PAYMENT_GROUP_COL,
        partition_col="partition",
        id_col="TransactionID",
    )
    y_train = train_assembled["isFraud"].to_numpy()
    y_holdout = holdout_assembled["isFraud"].to_numpy()
    logger.info("Model rows: train=%d, holdout=%d", len(train_assembled), len(holdout_assembled))

    ladder_results: dict = {}
    proba_by_step: dict = {}
    for step in LADDER_STEPS_EVALUATED:
        X_train = get_ladder_matrix(train_assembled, step).to_numpy()
        X_holdout = get_ladder_matrix(holdout_assembled, step).to_numpy()
        result = fit_and_score(X_train, y_train, X_holdout, y_holdout)
        ladder_results[step] = {
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
            "n_features": result["n_features"],
            "converged": result["converged"],
            "n_iter": result["n_iter"],
        }
        proba_by_step[step] = result["proba_validation"]
        logger.info(
            "%s (holdout): roc_auc=%.4f pr_auc=%.4f n_features=%d converged=%s (n_iter=%d, max_iter=%d)",
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

    logger.info("Running the fixed-seed %d-resample bootstrap PR-AUC delta CI (B2 -> F2) on holdout...", BOOTSTRAP_N_RESAMPLES)
    bootstrap = bootstrap_pr_auc_delta_ci(y_holdout, proba_by_step["B2"], proba_by_step["F2"])

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
        "Holdout graduation gates: 1(lift>=%.0f%%)=%s 2(ROC non-degrading)=%s 3(bootstrap CI lower>0)=%s "
        "4(F1>B2)=%s -> all_gates_pass=%s",
        (GRADUATION_RELATIVE_LIFT_THRESHOLD - 1.0) * 100,
        gate1_relative_lift,
        gate2_roc_non_degrading,
        gate3_bootstrap_ci,
        gate4_f1_improves,
        all_gates_pass,
    )

    phase_g_results_path = out_dir / "phase_g_results.json"
    with open(phase_g_results_path, "r", encoding="utf-8") as f:
        phase_g_results = json.load(f)
    phase_g_reference = {
        "ladder_results": phase_g_results["ladder_results"],
        "graduation": phase_g_results["graduation"],
    }

    results: dict = {
        "n_rows_total": n_rows_total,
        "partition_counts": {k: int(v) for k, v in partition_counts.items()},
        "n_rows_missing_payment_proxy_key": int(n_rows_missing_key),
        "n_rows_valid_key": int(len(valid)),
        "n_rows_train": int(len(train_assembled)),
        "n_rows_holdout": int(len(holdout_assembled)),
        "ladder_feature_columns": {step: LADDER_FEATURE_COLUMNS[step] for step in LADDER_STEPS_EVALUATED},
        "ladder_results_holdout": ladder_results,
        "graduation_holdout": graduation,
        "phase_g_validation_reference": phase_g_reference,
        "logreg_max_iter": LOGREG_MAX_ITER,
        "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    report_path = out_dir / "phase_h_report.md"
    render_phase_h_report(results, report_path)

    elapsed = time.time() - t0
    logger.info("Phase H complete in %.1fs. Results: %s Report: %s", elapsed, results_path, report_path)


if __name__ == "__main__":
    main()
