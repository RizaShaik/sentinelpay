"""Schema and join validation for the raw IEEE-CIS files."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class JoinValidationResult:
    n_transaction_rows: int
    n_identity_rows: int
    n_transaction_duplicate_ids: int
    n_identity_duplicate_ids: int
    n_matched: int
    pct_transactions_with_identity: float
    n_identity_orphans: int
    orphan_identity_ids: list = field(default_factory=list)

    def is_clean(self) -> bool:
        return (
            self.n_transaction_duplicate_ids == 0
            and self.n_identity_duplicate_ids == 0
            and self.n_identity_orphans == 0
        )


def validate_transaction_identity_join(
    transaction_ids: pd.Series, identity_ids: pd.Series
) -> JoinValidationResult:
    """Validate the TransactionID relationship between the two files.

    identity_ids is expected to be a subset of transaction_ids (every
    identity row should belong to a real transaction). Any identity
    TransactionID with no matching transaction row is an "orphan" and is a
    data-integrity red flag, not an expected outcome.
    """
    tx_ids = pd.Index(transaction_ids)
    id_ids = pd.Index(identity_ids)

    n_tx_dupes = int(len(tx_ids) - tx_ids.nunique())
    n_id_dupes = int(len(id_ids) - id_ids.nunique())

    tx_id_set = set(tx_ids.unique())
    id_id_set = set(id_ids.unique())

    matched = id_id_set & tx_id_set
    orphans = id_id_set - tx_id_set

    n_tx = len(tx_ids)
    pct_with_identity = (len(matched) / len(tx_id_set) * 100.0) if tx_id_set else 0.0

    return JoinValidationResult(
        n_transaction_rows=n_tx,
        n_identity_rows=len(id_ids),
        n_transaction_duplicate_ids=n_tx_dupes,
        n_identity_duplicate_ids=n_id_dupes,
        n_matched=len(matched),
        pct_transactions_with_identity=round(pct_with_identity, 4),
        n_identity_orphans=len(orphans),
        orphan_identity_ids=sorted(orphans)[:20],
    )


@dataclass
class SchemaValidationResult:
    only_in_a: list
    only_in_b: list

    def is_consistent(self) -> bool:
        return not self.only_in_a and not self.only_in_b


def validate_schema_consistency(
    columns_a: list[str], columns_b: list[str], ignore: set[str] | None = None
) -> SchemaValidationResult:
    """Compare two column sets (e.g. train vs test transaction schema)."""
    ignore = ignore or set()
    set_a = set(columns_a) - ignore
    set_b = set(columns_b) - ignore
    return SchemaValidationResult(
        only_in_a=sorted(set_a - set_b),
        only_in_b=sorted(set_b - set_a),
    )


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value count and rate, sorted descending."""
    n = len(df)
    missing = df.isna().sum()
    report = pd.DataFrame(
        {
            "column": missing.index,
            "n_missing": missing.values,
            "pct_missing": (missing.values / n * 100.0) if n else 0.0,
        }
    ).sort_values("pct_missing", ascending=False, ignore_index=True)
    return report


def duplicate_row_report(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    """Count of fully (or subset-) duplicated rows."""
    return int(df.duplicated(subset=subset).sum())
