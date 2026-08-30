import pandas as pd
import pytest

from sentinelpay.config import DataConfig
from sentinelpay.data.loader import load_identity, load_transaction_columns, load_transaction_full, normalize_identity_columns
from sentinelpay.utils.memory import categorize_object_columns, downcast_numeric


def _make_config(tmp_path, files: dict[str, str]) -> DataConfig:
    return DataConfig(
        raw_dir=tmp_path,
        interim_dir=tmp_path,
        processed_dir=tmp_path,
        reports_dir=tmp_path,
        files=files,
        join_key="TransactionID",
        dt_column="TransactionDT",
        seconds_per_day=86400,
        total_span_days=182,
        known_categorical_transaction=["ProductCD"],
        correlation_mode="curated",
        correlation_curated_columns=[],
        payment_proxy_key_columns=[],
        device_proxy_key_columns=[],
    )


def test_normalize_identity_columns_renames_hyphenated():
    df = pd.DataFrame(columns=["TransactionID", "id-01", "id-02", "id-38", "DeviceType"])
    out = normalize_identity_columns(df)
    assert list(out.columns) == ["TransactionID", "id_01", "id_02", "id_38", "DeviceType"]


def test_normalize_identity_columns_is_noop_on_underscored():
    df = pd.DataFrame(columns=["TransactionID", "id_01", "id_02", "DeviceType"])
    out = normalize_identity_columns(df)
    assert list(out.columns) == list(df.columns)


def test_normalize_identity_columns_does_not_touch_unrelated_columns():
    df = pd.DataFrame(columns=["TransactionID", "card1", "V1", "id-05"])
    out = normalize_identity_columns(df)
    assert "card1" in out.columns and "V1" in out.columns
    assert "id_05" in out.columns and "id-05" not in out.columns


def test_downcast_numeric_shrinks_dtypes():
    df = pd.DataFrame({"a": pd.array([1.0, 2.0, 3.0], dtype="float64"), "b": pd.array([1, 2, 3], dtype="int64")})
    out = downcast_numeric(df)
    assert out["a"].dtype == "float32"
    assert str(out["b"].dtype) in {"int8", "int16", "int32"}


def test_downcast_numeric_preserves_values():
    df = pd.DataFrame({"a": [1.5, float("nan"), 3.25]})
    out = downcast_numeric(df.copy())
    pd.testing.assert_series_equal(out["a"].astype("float64"), df["a"], check_names=False)


def test_categorize_object_columns_respects_max_cardinality():
    df = pd.DataFrame({"low_card": ["A", "B", "A", "B"], "high_card": [str(i) for i in range(4)]})
    out = categorize_object_columns(df, max_cardinality=2)
    assert out["low_card"].dtype.name == "category"
    assert out["high_card"].dtype == "object"


def test_load_identity_normalizes_without_modifying_source_file(tmp_path):
    raw_path = tmp_path / "test_identity.csv"
    raw_bytes = b"TransactionID,id-01,id-02,DeviceType\n1,0.0,100.0,mobile\n2,-5.0,50.0,desktop\n"
    raw_path.write_bytes(raw_bytes)
    config = _make_config(tmp_path, {"test_identity": "test_identity.csv"})

    df = load_identity("test", config=config)

    assert list(df.columns) == ["TransactionID", "id_01", "id_02", "DeviceType"]
    assert raw_path.read_bytes() == raw_bytes  # source file byte-for-byte unchanged


def test_load_identity_train_style_underscored_is_also_untouched(tmp_path):
    raw_path = tmp_path / "train_identity.csv"
    raw_bytes = b"TransactionID,id_01,id_02,DeviceType\n1,0.0,100.0,mobile\n"
    raw_path.write_bytes(raw_bytes)
    config = _make_config(tmp_path, {"train_identity": "train_identity.csv"})

    df = load_identity("train", config=config)

    assert list(df.columns) == ["TransactionID", "id_01", "id_02", "DeviceType"]
    assert raw_path.read_bytes() == raw_bytes


def test_load_transaction_full_reflects_isFraud_absent_from_test_schema(tmp_path):
    train_path = tmp_path / "train_transaction.csv"
    test_path = tmp_path / "test_transaction.csv"
    train_path.write_text("TransactionID,isFraud,TransactionDT,TransactionAmt,ProductCD\n1,0,86400,10.0,W\n")
    test_path.write_text("TransactionID,TransactionDT,TransactionAmt,ProductCD\n2,90000,20.0,C\n")
    config = _make_config(tmp_path, {"train_transaction": "train_transaction.csv", "test_transaction": "test_transaction.csv"})

    train_df = load_transaction_full("train", config=config)
    test_df = load_transaction_full("test", config=config)

    assert "isFraud" in train_df.columns
    assert "isFraud" not in test_df.columns


def test_load_transaction_columns_requires_explicit_columns(tmp_path):
    path = tmp_path / "train_transaction.csv"
    path.write_text("TransactionID,isFraud,TransactionDT\n1,0,86400\n")
    config = _make_config(tmp_path, {"train_transaction": "train_transaction.csv"})

    with pytest.raises(ValueError):
        load_transaction_columns("train", columns=[], config=config)
