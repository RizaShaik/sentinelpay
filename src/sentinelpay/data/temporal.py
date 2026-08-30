"""Temporal-drift inspection utilities.

TransactionDT is a seconds-based timedelta from an undisclosed reference
point (not a real timestamp) -- see reports/eda for citations. Everything
here is descriptive/inspective: it exists to inform where a chronological
train/validation boundary should later be drawn, not to draw it.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats


def add_day_index(df: pd.DataFrame, dt_col: str = "TransactionDT", seconds_per_day: int = 86400) -> pd.DataFrame:
    df = df.copy()
    df["_day"] = (df[dt_col] // seconds_per_day).astype("int32")
    return df


def daily_volume(df: pd.DataFrame, day_col: str = "_day") -> pd.DataFrame:
    out = df.groupby(day_col).size().rename("n_transactions").reset_index()
    return out


def daily_fraud_rate(df: pd.DataFrame, day_col: str = "_day", target_col: str = "isFraud") -> pd.DataFrame:
    grp = df.groupby(day_col)[target_col]
    out = grp.agg(n_transactions="count", n_fraud="sum").reset_index()
    out["fraud_rate"] = out["n_fraud"] / out["n_transactions"]
    return out


def daily_count_by_group(df: pd.DataFrame, group_col: str, day_col: str = "_day") -> pd.DataFrame:
    """Per-(group, day) transaction counts. Fully generic over `group_col`
    -- this is a building block for a future per-entity temporal detector
    (Phase D), not a claim about what the grouping key should be."""
    for col in (group_col, day_col):
        if col not in df.columns:
            raise ValueError(f"daily_count_by_group requires column '{col}'")
    out = df.groupby([group_col, day_col], observed=True).size().rename("n_transactions").reset_index()
    return out.sort_values([group_col, day_col], ignore_index=True)


def daily_amount_stats_by_group(
    df: pd.DataFrame, group_col: str, amount_col: str, day_col: str = "_day"
) -> pd.DataFrame:
    """Per-(group, day) count/sum/mean/std of `amount_col`. Fully generic
    over `group_col` -- see `daily_count_by_group`."""
    for col in (group_col, amount_col, day_col):
        if col not in df.columns:
            raise ValueError(f"daily_amount_stats_by_group requires column '{col}'")
    grp = df.groupby([group_col, day_col], observed=True)[amount_col]
    out = grp.agg(n_transactions="count", amount_sum="sum", amount_mean="mean", amount_std="std").reset_index()
    return out.sort_values([group_col, day_col], ignore_index=True)


def numeric_drift(
    df: pd.DataFrame,
    columns: list[str],
    day_col: str,
    split_day: int,
) -> pd.DataFrame:
    """Two-sample Kolmogorov-Smirnov drift test per numeric column.

    Compares the empirical distribution of each column before vs. at-or-after
    `split_day`. Rows with NaN in a column are dropped for that column's test
    (missingness-rate drift is reported separately, since silent NaN-drop
    would otherwise hide it).
    """
    early = df[df[day_col] < split_day]
    late = df[df[day_col] >= split_day]

    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        a = early[col].dropna()
        b = late[col].dropna()
        pct_missing_early = 100.0 * (1 - len(a) / len(early)) if len(early) else float("nan")
        pct_missing_late = 100.0 * (1 - len(b) / len(late)) if len(late) else float("nan")
        if len(a) < 2 or len(b) < 2:
            ks_stat, p_value = float("nan"), float("nan")
        else:
            result = stats.ks_2samp(a, b)
            ks_stat, p_value = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "column": col,
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "mean_early": float(a.mean()) if len(a) else float("nan"),
                "mean_late": float(b.mean()) if len(b) else float("nan"),
                "pct_missing_early": pct_missing_early,
                "pct_missing_late": pct_missing_late,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("ks_statistic", ascending=False, ignore_index=True)


def categorical_drift(
    df: pd.DataFrame,
    columns: list[str],
    day_col: str,
    split_day: int,
    max_categories: int = 20,
) -> pd.DataFrame:
    """Chi-square test of independence between period (early/late) and
    category, per categorical column."""
    period = (df[day_col] >= split_day).map({False: "early", True: "late"})

    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].astype("object").fillna("__missing__")
        top_categories = s.value_counts().nlargest(max_categories).index
        s_capped = s.where(s.isin(top_categories), other="__other__")
        table = pd.crosstab(period, s_capped)
        if table.shape[0] < 2 or table.shape[1] < 2 or (table.values == 0).all():
            chi2, p_value = float("nan"), float("nan")
        else:
            chi2, p_value, _, _ = stats.chi2_contingency(table)
        rows.append(
            {
                "column": col,
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "n_categories_observed": int(s.nunique()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("chi2_statistic", ascending=False, ignore_index=True)
