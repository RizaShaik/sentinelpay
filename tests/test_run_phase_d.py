import pandas as pd
import pytest

import sentinelpay.eda.run_phase_d as run_phase_d_module
from sentinelpay.config import load_config
from sentinelpay.data.split import DEVELOPMENT_PARTITIONS, load_split_config
from sentinelpay.eda.run_phase_d import build_development_frame, evaluate_validation_only


def _synthetic_transaction_frame(config):
    """One small, hand-placed row per partition (via day boundaries from the
    real configs/split.yaml), plus a couple of holdout rows -- enough to
    prove build_development_frame excludes holdout without touching real
    CSVs (the loader is monkeypatched)."""
    seconds_per_day = config.seconds_per_day
    days = {"train": 5, "embargo_1": 133, "validation": 150, "embargo_2": 165, "holdout": 175}
    rows = []
    for partition, day in days.items():
        n_rows = 3 if partition == "holdout" else 2
        for i in range(n_rows):
            rows.append(
                {
                    "TransactionID": len(rows) + 1,
                    "TransactionDT": day * seconds_per_day + i,
                    "TransactionAmt": 10.0 + i,
                    **{col: 1 for col in config.payment_proxy_key_columns},
                }
            )
    return pd.DataFrame(rows), 3  # (frame, n_holdout_rows_in_fixture)


def test_build_development_frame_excludes_holdout_rows(monkeypatch):
    config = load_config()
    split_config = load_split_config()
    synthetic, n_holdout_in_fixture = _synthetic_transaction_frame(config)

    def fake_load_transaction_columns(split, columns, config=None):
        return synthetic[columns].copy()

    monkeypatch.setattr(run_phase_d_module, "load_transaction_columns", fake_load_transaction_columns)

    development, n_rows_total, n_holdout_excluded = build_development_frame(config, split_config)

    assert n_rows_total == len(synthetic)
    assert n_holdout_excluded == n_holdout_in_fixture
    assert len(development) == len(synthetic) - n_holdout_in_fixture
    assert set(development["partition"].unique()) == set(DEVELOPMENT_PARTITIONS)
    assert "holdout" not in development["partition"].unique()
    # Exactly the 2 fixture rows per non-holdout partition survive, no more/fewer.
    for partition in DEVELOPMENT_PARTITIONS:
        assert (development["partition"] == partition).sum() == 2


def test_evaluate_validation_only_rejects_non_validation_rows():
    config = load_config()
    bad_df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "partition": ["train", "validation"],
            "modified_zscore": [0.1, 0.2],
            "flag": ["scored_normal", "scored_normal"],
        }
    )
    with pytest.raises(ValueError):
        evaluate_validation_only(bad_df, config)


def test_evaluate_validation_only_does_not_mutate_input(monkeypatch):
    config = load_config()
    scored_validation = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "partition": ["validation", "validation", "validation"],
            "modified_zscore": [0.1, float("nan"), 4.0],
            "flag": ["scored_normal", "insufficient_history", "scored_outlier"],
        }
    )
    original = scored_validation.copy()

    fake_isfraud = pd.DataFrame({"TransactionID": [1, 2, 3], "isFraud": [0, 0, 1]})

    def fake_load_transaction_columns(split, columns, config=None):
        return fake_isfraud[columns].copy()

    monkeypatch.setattr(run_phase_d_module, "load_transaction_columns", fake_load_transaction_columns)

    result = evaluate_validation_only(scored_validation, config)

    pd.testing.assert_frame_equal(scored_validation, original)
    assert result["coverage"]["n_validation_rows"] == 3
    assert result["n_scored_rows"] == 2  # excludes the NaN-score (insufficient_history) row
