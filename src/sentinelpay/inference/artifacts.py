"""Phase I: persisting the frozen F2 model, fit on the SAME train-only
regime `sentinelpay.eda.run_phase_g`/`run_phase_h` already used.

`sentinelpay.model_evaluation.fit_and_score` does not expose the fitted
`StandardScaler`/`LogisticRegression` objects (only scores) -- so this
module restates its exact fit sequence (`StandardScaler` ->
`LogisticRegression(max_iter=LOGREG_MAX_ITER)`, library defaults otherwise)
to obtain persistable objects. This is verified byte-for-byte identical to
`fit_and_score`'s own output by `tests/test_inference_artifacts.py`, not
merely assumed identical by inspection.

`sentinelpay.model_evaluation`, `sentinelpay.model_features`, and
`sentinelpay.eda.run_phase_g` are imported from, NEVER modified.

F2 is the frozen, graduated design (Phase G validation + Phase H sealed
holdout, all gates passed) -- this module does not select a step; it always
fits `GRADUATED_STEP = "F2"`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sentinelpay.config import DataConfig, DetectionConfig, load_config, load_detection_config
from sentinelpay.data.loader import load_identity_ids
from sentinelpay.data.split import SplitConfig, load_split_config
from sentinelpay.eda.grouping_key_sufficiency import build_group_key
from sentinelpay.eda.run_phase_g import build_development_frame
from sentinelpay.model_evaluation import LOGREG_MAX_ITER
from sentinelpay.model_features import LADDER_FEATURE_COLUMNS, PAYMENT_GROUP_COL, assemble_ladder_matrix, get_ladder_matrix

GRADUATED_STEP = "F2"


def fit_frozen_f2_artifact(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Restates `sentinelpay.model_evaluation.fit_and_score`'s exact fit
    sequence, returning the FITTED objects (`fit_and_score` itself discards
    them). See module docstring for the equivalence guarantee."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=LOGREG_MAX_ITER)
    model.fit(X_scaled, y_train)
    return {"scaler": scaler, "model": model}


def build_f2_artifact(
    config: DataConfig | None = None,
    split_config: SplitConfig | None = None,
    detection_config: DetectionConfig | None = None,
) -> dict:
    """Fits F2 on `train`-partition, `payment_proxy_key`-valid rows ONLY --
    the identical population and features Phase G/H fit on. Never loads
    `holdout` (development partitions only, via
    `sentinelpay.eda.run_phase_g.build_development_frame`, unmodified).
    Returns `{"scaler", "model", "metadata"}` -- metadata makes the fit
    source and cutoff explicit (never inferred from context elsewhere).
    """
    config = config or load_config()
    detection_config = detection_config or load_detection_config()
    split_config = split_config or load_split_config()

    identity_ids = load_identity_ids("train", config=config)
    development, n_rows_total, n_holdout_excluded = build_development_frame(config, split_config)
    valid = build_group_key(development, config.payment_proxy_key_columns, key_name=PAYMENT_GROUP_COL)
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
    X_train = get_ladder_matrix(train_rows, GRADUATED_STEP).to_numpy()
    y_train = train_rows["isFraud"].to_numpy()

    fitted = fit_frozen_f2_artifact(X_train, y_train)

    metadata = {
        "graduated_step": GRADUATED_STEP,
        "model_fit_source": "train partition ONLY (Phase G/H regime) -- never validation, never holdout",
        "feature_columns": LADDER_FEATURE_COLUMNS[GRADUATED_STEP],
        "logreg_max_iter": LOGREG_MAX_ITER,
        "fit_row_count": int(len(X_train)),
        "fit_n_features": int(X_train.shape[1]),
        "holdout_used_for_fitting": False,
        "holdout_used_for_selection": False,
        "holdout_used_for_thresholding": False,
        "n_rows_total_loaded": n_rows_total,
        "n_rows_holdout_excluded": n_holdout_excluded,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {"scaler": fitted["scaler"], "model": fitted["model"], "metadata": metadata}


def save_artifact(artifact: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> dict:
    return joblib.load(Path(path))
