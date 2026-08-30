"""Memory-efficiency helpers for large IEEE-CIS frames.

These act only on in-memory DataFrames / interim artifacts -- never on the
raw CSVs in data/raw, which are treated as immutable source-of-truth.
"""
from __future__ import annotations

import logging

import pandas as pd
import psutil

logger = logging.getLogger(__name__)


def downcast_numeric(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Downcast float64->smallest float, int64->smallest int, in place.

    Uses pandas' own downcast logic (pd.to_numeric(..., downcast=...)), which
    only narrows a column when doing so is lossless for the values present.
    """
    cols = columns if columns is not None else df.columns
    float_cols = [c for c in cols if c in df.columns and pd.api.types.is_float_dtype(df[c])]
    int_cols = [c for c in cols if c in df.columns and pd.api.types.is_integer_dtype(df[c])]

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def categorize_object_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    max_cardinality: int = 2000,
) -> pd.DataFrame:
    """Convert low-cardinality object columns to pandas 'category' dtype."""
    cols = columns if columns is not None else list(df.select_dtypes(include="object").columns)
    for col in cols:
        if col not in df.columns:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique <= max_cardinality:
            df[col] = df[col].astype("category")
    return df


def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024**2))


def log_memory_report(df: pd.DataFrame, label: str = "") -> dict:
    """Log and return a small dict describing DataFrame + process memory."""
    df_mb = memory_usage_mb(df)
    process_mb = psutil.Process().memory_info().rss / (1024**2)
    report = {
        "label": label,
        "rows": len(df),
        "cols": df.shape[1],
        "dataframe_mb": round(df_mb, 2),
        "process_rss_mb": round(process_mb, 2),
    }
    logger.info(
        "[%s] rows=%d cols=%d df=%.2fMB process_rss=%.2fMB",
        label,
        report["rows"],
        report["cols"],
        report["dataframe_mb"],
        report["process_rss_mb"],
    )
    return report
