import pandas as pd
import pytest

from sentinelpay.data.validation import (
    duplicate_row_report,
    missingness_report,
    validate_schema_consistency,
    validate_transaction_identity_join,
)


def test_validate_join_clean_case():
    tx_ids = pd.Series([1, 2, 3, 4, 5])
    id_ids = pd.Series([1, 3, 5])  # subset of tx_ids, no dupes
    result = validate_transaction_identity_join(tx_ids, id_ids)

    assert result.n_transaction_rows == 5
    assert result.n_identity_rows == 3
    assert result.n_matched == 3
    assert result.n_identity_orphans == 0
    assert result.n_transaction_duplicate_ids == 0
    assert result.n_identity_duplicate_ids == 0
    assert result.is_clean()
    assert result.pct_transactions_with_identity == pytest.approx(60.0)


def test_validate_join_detects_orphan_identity_rows():
    tx_ids = pd.Series([1, 2, 3])
    id_ids = pd.Series([1, 2, 999])  # 999 has no matching transaction
    result = validate_transaction_identity_join(tx_ids, id_ids)

    assert result.n_identity_orphans == 1
    assert 999 in result.orphan_identity_ids
    assert not result.is_clean()


def test_validate_join_detects_duplicate_ids():
    tx_ids = pd.Series([1, 1, 2, 3])
    id_ids = pd.Series([1, 2, 2])
    result = validate_transaction_identity_join(tx_ids, id_ids)

    assert result.n_transaction_duplicate_ids == 1
    assert result.n_identity_duplicate_ids == 1
    assert not result.is_clean()


def test_validate_schema_consistency_matching():
    result = validate_schema_consistency(["a", "b", "c"], ["a", "b", "c"])
    assert result.is_consistent()


def test_validate_schema_consistency_with_ignore():
    result = validate_schema_consistency(["a", "b", "isFraud"], ["a", "b"], ignore={"isFraud"})
    assert result.is_consistent()


def test_validate_schema_consistency_detects_mismatch():
    result = validate_schema_consistency(["a", "b", "extra_a"], ["a", "b", "extra_b"])
    assert not result.is_consistent()
    assert result.only_in_a == ["extra_a"]
    assert result.only_in_b == ["extra_b"]


def test_missingness_report_counts_and_sorts_descending():
    df = pd.DataFrame({"full": [1, 2, 3], "half": [1, None, 3], "empty": [None, None, None]})
    report = missingness_report(df)
    assert report.iloc[0]["column"] == "empty"
    assert report.iloc[0]["pct_missing"] == 100.0
    assert report[report["column"] == "half"]["pct_missing"].iloc[0] == pytest.approx(100 / 3)


def test_duplicate_row_report():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    assert duplicate_row_report(df) == 1
    assert duplicate_row_report(df, subset=["a"]) == 1
