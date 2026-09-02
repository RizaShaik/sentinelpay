import pandas as pd

import sentinelpay.eda.run_phase_g as run_phase_g_module
from sentinelpay.config import load_config
from sentinelpay.data.split import DEVELOPMENT_PARTITIONS, load_split_config
from sentinelpay.eda.run_phase_g import build_development_frame


def _synthetic_transaction_frame(config):
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
                    "isFraud": i % 2,
                    **{col: 1 for col in config.payment_proxy_key_columns},
                }
            )
    return pd.DataFrame(rows), 3


def test_build_development_frame_excludes_holdout_rows(monkeypatch):
    config = load_config()
    split_config = load_split_config()
    synthetic, n_holdout_in_fixture = _synthetic_transaction_frame(config)

    def fake_load_transaction_columns(split, columns, config=None):
        return synthetic[columns].copy()

    monkeypatch.setattr(run_phase_g_module, "load_transaction_columns", fake_load_transaction_columns)

    development, n_rows_total, n_holdout_excluded = build_development_frame(config, split_config)

    assert n_rows_total == len(synthetic)
    assert n_holdout_excluded == n_holdout_in_fixture
    assert len(development) == len(synthetic) - n_holdout_in_fixture
    assert set(development["partition"].unique()) == set(DEVELOPMENT_PARTITIONS)
    assert "holdout" not in development["partition"].unique()
    assert "isFraud" in development.columns
    assert "TransactionAmt" in development.columns
