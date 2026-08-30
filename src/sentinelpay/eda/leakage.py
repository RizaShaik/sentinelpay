"""Leakage-audit helpers.

These checks look for structural properties of the raw data that could let a
naive model or naive split leak future information into training, or leak
the label into a feature. They report evidence; interpretation (whether a
given correlation is a real leak vs. a legitimate signal) is left to the EDA
report.

EDA-only / no target encoding here: every function below that touches
`isFraud` (or any target column) computes a *global, whole-dataset*
statistic for inspection purposes. None of these numbers may be used
directly as a modeling feature. Any future target encoding or
historical-fraud-rate feature must be computed using only information
available strictly before the transaction being scored -- chronological,
fold-safe logic -- not the global aggregate these EDA scans compute. Phase
A/B implements no target encoding; that decision is deferred.
"""
from __future__ import annotations

import pandas as pd


def id_time_monotonicity(df: pd.DataFrame, id_col: str = "TransactionID", dt_col: str = "TransactionDT") -> dict:
    """Check whether TransactionID order agrees with TransactionDT order.

    If TransactionID is (nearly) monotonic in time, any model or feature
    that conditions on ID ordering (e.g. row position, ID as a feature) can
    leak temporal information equivalent to a timestamp.
    """
    id_rank = df[id_col].rank(method="first")
    dt_rank = df[dt_col].rank(method="first")
    corr = float(id_rank.corr(dt_rank, method="spearman"))
    is_sorted_by_dt = bool(df[dt_col].is_monotonic_increasing)
    is_sorted_by_id = bool(df[id_col].is_monotonic_increasing)
    return {
        "spearman_corr_id_vs_dt_rank": corr,
        "rows_sorted_by_dt_in_file": is_sorted_by_dt,
        "rows_sorted_by_id_in_file": is_sorted_by_id,
    }


_ZERO_VARIANCE_EPS = 1e-12


def _drop_zero_variance(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Filter out columns with (near-)zero standard deviation.

    A constant column makes `corrwith` divide by a ~0 stddev internally,
    which is mathematically undefined (correlation with a constant is
    undefined, not just unknown) and previously surfaced as a numpy
    `RuntimeWarning: invalid value encountered in divide`. Dropping these
    columns up front avoids the warning; the result is the same as before
    (such a column always ended up NaN and was dropped by `.dropna()`
    downstream) -- this just avoids computing the undefined division at all.
    """
    if not columns:
        return columns
    std = df[columns].std()
    return [c for c in columns if pd.notna(std.get(c)) and std.get(c) > _ZERO_VARIANCE_EPS]


def _corrwith_target(df: pd.DataFrame, target_col: str, columns: list[str], top_n: int) -> pd.DataFrame:
    """Column-vs-target correlation via `corrwith`: O(k*n), not the O(k^2*n)
    a full `df[cols].corr()` matrix costs for k unneeded pairs. This is the
    fix for the ~5.4GB/2.5min full-matrix cost observed when this scan
    previously used `.corr()` over the entire numeric column set."""
    if df[target_col].std() <= _ZERO_VARIANCE_EPS:
        return pd.DataFrame(columns=["column", "abs_corr_with_target", "corr_with_target"])
    present = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    present = _drop_zero_variance(df, present)
    if not present:
        return pd.DataFrame(columns=["column", "abs_corr_with_target", "corr_with_target"])
    corrs = df[present].corrwith(df[target_col])
    out = corrs.dropna().abs().sort_values(ascending=False).head(top_n)
    return pd.DataFrame(
        {
            "column": out.index,
            "abs_corr_with_target": out.values,
            "corr_with_target": corrs.loc[out.index].values,
        }
    )


def curated_target_correlation(
    df: pd.DataFrame, target_col: str, curated_columns: list[str], top_n: int = 20
) -> pd.DataFrame:
    """EDA-only correlation scan over a curated, non-V numeric column set
    (see configs/data.yaml `correlation.curated_columns`). Default
    correlation mode -- cheap and avoids blindly mixing V1-V339 into a flat
    ranking with everything else (see `v_block_target_correlation`)."""
    return _corrwith_target(df, target_col, curated_columns, top_n)


def v_column_missingness_blocks(df: pd.DataFrame, v_columns: list[str]) -> dict[float, list[str]]:
    """Group V1-V339 by identical (rounded) missing-rate.

    V columns visibly co-vary in missingness (confirmed in Phase A: ~14
    columns share the exact same 86.12% missing rate) -- grouping by that
    signature treats them as the structural blocks they appear to be,
    instead of 339 independent columns.
    """
    present = [c for c in v_columns if c in df.columns]
    if not present:
        return {}
    missing_pct = (df[present].isna().mean() * 100).round(4)
    blocks: dict[float, list[str]] = {}
    for col, pct in missing_pct.items():
        blocks.setdefault(float(pct), []).append(col)
    return blocks


def v_block_target_correlation(
    df: pd.DataFrame, target_col: str, v_columns: list[str], top_n_blocks: int = 10
) -> pd.DataFrame:
    """EDA-only: per-V-column corrwith (cheap, O(k*n)) aggregated to
    missingness-block level for reporting, so V1-V339 are never presented as
    339 independent flat findings."""
    empty_result = pd.DataFrame(columns=["pct_missing", "n_columns", "max_abs_corr", "mean_abs_corr", "top_column"])
    if df[target_col].std() <= _ZERO_VARIANCE_EPS:
        return empty_result

    blocks = v_column_missingness_blocks(df, v_columns)
    if not blocks:
        return empty_result

    all_cols = [c for cols in blocks.values() for c in cols]
    all_cols = _drop_zero_variance(df, all_cols)
    if not all_cols:
        return empty_result
    corrs = df[all_cols].corrwith(df[target_col]).abs()

    rows = []
    for pct_missing, cols in blocks.items():
        block_corrs = corrs.loc[[c for c in cols if c in corrs.index]].dropna()
        if block_corrs.empty:
            continue
        rows.append(
            {
                "pct_missing": pct_missing,
                "n_columns": len(cols),
                "max_abs_corr": float(block_corrs.max()),
                "mean_abs_corr": float(block_corrs.mean()),
                "top_column": block_corrs.idxmax(),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("max_abs_corr", ascending=False, ignore_index=True).head(top_n_blocks)


def full_target_correlation(
    df: pd.DataFrame, target_col: str, columns: list[str], top_n: int = 20
) -> pd.DataFrame:
    """Opt-in only (configs/data.yaml `correlation.mode: full`). Still
    O(k*n) via `corrwith`, not a full k-by-k matrix -- the original
    ~5.4GB/2.5min cost came from an accidental full `.corr()` matrix call,
    not from V-columns being inherently expensive. Kept behind the config
    flag anyway, per instruction that full correlation must be a deliberate
    opt-in, not a default."""
    return _corrwith_target(df, target_col, columns, top_n)


def near_duplicate_row_rate(df: pd.DataFrame, subset: list[str]) -> dict:
    present = [c for c in subset if c in df.columns]
    n = len(df)
    n_dupe = int(df.duplicated(subset=present).sum())
    return {
        "subset": present,
        "n_rows": n,
        "n_duplicated": n_dupe,
        "pct_duplicated": round(100.0 * n_dupe / n, 4) if n else 0.0,
    }
