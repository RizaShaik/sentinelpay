import numpy as np
import pandas as pd
import pytest

from sentinelpay.features import FEATURE_REGISTRY, build_feature_frame


def _synthetic_tx():
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [0, 3600 * 5, 86400 * 2 + 3600 * 10, 86400 * 9],
            "TransactionAmt": [10.0, 12.5, 100.0, 99.99],
        }
    )


def test_amt_log1p_and_decimal_part_values():
    df = _synthetic_tx()
    out, _ = build_feature_frame(df, identity_ids=set())
    assert out["amt_log1p"].tolist() == pytest.approx(np.log1p(df["TransactionAmt"]).tolist())
    assert out["amt_decimal_part"].tolist() == pytest.approx([0.0, 0.5, 0.0, 0.99], abs=1e-6)


def test_dt_hour_and_day_of_week_values():
    df = _synthetic_tx()
    out, _ = build_feature_frame(df, identity_ids=set())
    assert out["dt_hour_of_day"].tolist() == [0, 5, 10, 0]
    assert out["dt_day_of_week"].tolist() == [0, 0, 2, 2]  # day index 0,0,2,9 -> %7 = 0,0,2,2


def test_has_identity_matches_join():
    df = _synthetic_tx()
    out, _ = build_feature_frame(df, identity_ids={2, 4})
    assert out["has_identity"].tolist() == [0, 1, 0, 1]


def test_output_has_no_isfraud_dependency():
    df = _synthetic_tx()
    df_with_target = df.copy()
    df_with_target["isFraud"] = [1, 0, 1, 0]
    df_shuffled_target = df.copy()
    df_shuffled_target["isFraud"] = [0, 1, 0, 1]

    out_a, _ = build_feature_frame(df, identity_ids={2})
    out_b, _ = build_feature_frame(df_with_target, identity_ids={2})
    out_c, _ = build_feature_frame(df_shuffled_target, identity_ids={2})

    pd.testing.assert_frame_equal(out_a, out_b)
    pd.testing.assert_frame_equal(out_a, out_c)

    import inspect

    assert "isFraud" not in inspect.signature(build_feature_frame).parameters
    assert "target" not in inspect.signature(build_feature_frame).parameters


def test_registry_matches_output_columns():
    df = _synthetic_tx()
    out, registry = build_feature_frame(df, identity_ids=set())
    registry_names = {entry["feature"] for entry in registry}
    assert registry_names == set(out.columns)
    assert registry_names == {e["feature"] for e in FEATURE_REGISTRY}
    for entry in registry:
        assert entry["uses_target"] is False


def test_missing_amount_propagates_to_nan_not_imputed():
    df = _synthetic_tx()
    df.loc[1, "TransactionAmt"] = np.nan
    out, _ = build_feature_frame(df, identity_ids=set())
    assert pd.isna(out.loc[1, "amt_log1p"])
    assert pd.isna(out.loc[1, "amt_decimal_part"])
    # unaffected rows are untouched
    assert not pd.isna(out.loc[0, "amt_log1p"])
