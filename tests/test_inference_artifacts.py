import numpy as np
import pandas as pd
import pytest

from sentinelpay.inference.artifacts import (
    GRADUATED_STEP,
    build_f2_artifact,
    fit_frozen_f2_artifact,
    load_artifact,
    save_artifact,
)
from sentinelpay.model_evaluation import fit_and_score


def test_fit_frozen_f2_artifact_matches_fit_and_score_exactly():
    # Equivalence test: the persisted-object fit path must produce a
    # BYTE-IDENTICAL result to model_evaluation.fit_and_score's own
    # (discarded) internal fit, on the same data.
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 5))
    y = (rng.random(n) < 0.2).astype(int)

    reference = fit_and_score(X, y, X, y)  # validation=train here, just to exercise the SAME fit on X,y

    fitted = fit_frozen_f2_artifact(X, y)
    proba_from_persisted = fitted["model"].predict_proba(fitted["scaler"].transform(X))[:, 1]

    np.testing.assert_array_equal(proba_from_persisted, reference["proba_validation"])


def test_save_load_artifact_roundtrip(tmp_path):
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [2.0, 2.0]])
    y = np.array([0, 1, 0, 1])
    fitted = fit_frozen_f2_artifact(X, y)
    artifact = {"scaler": fitted["scaler"], "model": fitted["model"], "metadata": {"note": "roundtrip"}}

    path = tmp_path / "artifacts" / "model.joblib"
    save_artifact(artifact, path)
    assert path.exists()

    loaded = load_artifact(path)
    proba_before = artifact["model"].predict_proba(artifact["scaler"].transform(X))[:, 1]
    proba_after = loaded["model"].predict_proba(loaded["scaler"].transform(X))[:, 1]
    np.testing.assert_array_equal(proba_before, proba_after)
    assert loaded["metadata"]["note"] == "roundtrip"


def _synthetic_dev_frame(config):
    seconds_per_day = config.seconds_per_day
    days = {"train": 5, "embargo_1": 133, "validation": 150, "embargo_2": 165}
    rng = np.random.default_rng(1)
    rows = []
    for partition, day in days.items():
        for i in range(20):
            rows.append(
                {
                    "TransactionID": len(rows) + 1,
                    "TransactionDT": day * seconds_per_day + i,
                    "TransactionAmt": float(10 + i),
                    "isFraud": int(rng.random() < 0.2),
                    **{col: int(i % 4) for col in config.payment_proxy_key_columns},
                }
            )
    return pd.DataFrame(rows)


def test_build_f2_artifact_train_only_and_metadata(monkeypatch):
    import sentinelpay.eda.run_phase_g as run_phase_g_module
    from sentinelpay.config import load_config

    config = load_config()
    synthetic = _synthetic_dev_frame(config)

    def fake_load_transaction_columns(split, columns, config=None):
        return synthetic[columns].copy()

    def fake_load_identity_ids(split, config=None):
        return pd.Series([], dtype="int64")

    monkeypatch.setattr(run_phase_g_module, "load_transaction_columns", fake_load_transaction_columns)
    import sentinelpay.inference.artifacts as artifacts_module

    monkeypatch.setattr(artifacts_module, "load_identity_ids", fake_load_identity_ids)

    artifact = build_f2_artifact(config=config)

    assert artifact["metadata"]["graduated_step"] == GRADUATED_STEP
    assert artifact["metadata"]["model_fit_source"] == "train partition ONLY (Phase G/H regime) -- never validation, never holdout"
    assert artifact["metadata"]["holdout_used_for_fitting"] is False
    assert artifact["metadata"]["holdout_used_for_selection"] is False
    assert artifact["metadata"]["fit_row_count"] == 20  # only the "train" partition's 20 rows
    assert artifact["metadata"]["fit_n_features"] == 15  # len(LADDER_FEATURE_COLUMNS["F2"])
    assert "scaler" in artifact and "model" in artifact
