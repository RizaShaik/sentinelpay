"""Phase C feature foundation: a small, justified set of row-wise features
plus a `has_identity` structural indicator, assembled by one unified
`build_feature_frame` pipeline.

Every feature here is row-local or a pure structural join -- none has any
cross-row temporal dependency, so none can leak future information and none
needs sentinelpay.data.history. No feature reads or depends on `isFraud` (or
any target column); `build_feature_frame` does not accept a target column at
all. No V1-V339 handling, no target encoding, no parquet persistence, and no
raw-column passthrough -- broader feature groups (categorical encoding,
D/C/V-column treatment, target-derived historical features) are deferred to
later phases. See reports/eda/phase_c_report.md for the full rationale.

Missing-value rule: nothing here fills or imputes a missing source value --
a NaN in `TransactionAmt` propagates to NaN in `amt_log1p`/`amt_decimal_part`
rather than being silently replaced, so real missingness stays visible to
whoever reports `pct_missing` downstream (sentinelpay.eda.run_phase_c does).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_AMOUNT_COL = "TransactionAmt"
DEFAULT_DT_COL = "TransactionDT"
DEFAULT_ID_COL = "TransactionID"


def add_amt_log1p(df: pd.DataFrame, amount_col: str = DEFAULT_AMOUNT_COL) -> pd.DataFrame:
    df = df.copy()
    df["amt_log1p"] = np.log1p(df[amount_col])
    return df


def add_amt_decimal_part(df: pd.DataFrame, amount_col: str = DEFAULT_AMOUNT_COL) -> pd.DataFrame:
    df = df.copy()
    df["amt_decimal_part"] = df[amount_col] - np.floor(df[amount_col])
    return df


def add_dt_hour_of_day(df: pd.DataFrame, dt_col: str = DEFAULT_DT_COL) -> pd.DataFrame:
    df = df.copy()
    df["dt_hour_of_day"] = ((df[dt_col] // 3600) % 24).astype("Int64")
    return df


def add_dt_day_of_week(df: pd.DataFrame, dt_col: str = DEFAULT_DT_COL) -> pd.DataFrame:
    """Period-7 bucket of `dt_col`. NOT claimed to be an actual calendar
    weekday -- TransactionDT's reference epoch is undisclosed (Phase A) --
    only a structural period-7 cycle."""
    df = df.copy()
    df["dt_day_of_week"] = ((df[dt_col] // 86400) % 7).astype("Int64")
    return df


def add_has_identity(df: pd.DataFrame, identity_ids, id_col: str = DEFAULT_ID_COL) -> pd.DataFrame:
    """Structural indicator: does `id_col` appear in the identity file.

    `identity_ids` is a collection of TransactionIDs (e.g. from
    sentinelpay.data.loader.load_identity_ids) -- this does NOT pull any
    value out of the identity table itself, only join existence.
    """
    df = df.copy()
    id_set = set(identity_ids)
    df["has_identity"] = df[id_col].isin(id_set).astype("int8")
    return df


FEATURE_REGISTRY: list[dict] = [
    {
        "feature": "amt_log1p",
        "source_columns": [DEFAULT_AMOUNT_COL],
        "uses_target": False,
        "temporal_dependency": "row-local",
        "description": "log1p(TransactionAmt) -- variance-stabilizing transform of a documented, always-populated field.",
    },
    {
        "feature": "amt_decimal_part",
        "source_columns": [DEFAULT_AMOUNT_COL],
        "uses_target": False,
        "temporal_dependency": "row-local",
        "description": "TransactionAmt - floor(TransactionAmt); distinguishes round-dollar vs. exact-cent amounts.",
    },
    {
        "feature": "dt_hour_of_day",
        "source_columns": [DEFAULT_DT_COL],
        "uses_target": False,
        "temporal_dependency": "row-local",
        "description": "(TransactionDT // 3600) % 24 -- cyclical time-of-day bucket from the row's own timestamp only.",
    },
    {
        "feature": "dt_day_of_week",
        "source_columns": [DEFAULT_DT_COL],
        "uses_target": False,
        "temporal_dependency": "row-local",
        "description": "(TransactionDT // 86400) % 7 -- cyclical period-7 bucket; not a claimed calendar weekday (epoch undisclosed).",
    },
    {
        "feature": "has_identity",
        "source_columns": [DEFAULT_ID_COL],
        "uses_target": False,
        "temporal_dependency": "structural-join",
        "description": "Whether TransactionID has a matching identity-file row; existence only, no identity-table value used.",
    },
]


def build_feature_frame(
    df: pd.DataFrame,
    identity_ids,
    amount_col: str = DEFAULT_AMOUNT_COL,
    dt_col: str = DEFAULT_DT_COL,
    id_col: str = DEFAULT_ID_COL,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build the Phase C feature frame: the 5 features in FEATURE_REGISTRY,
    aligned to `df.index`. Returns (feature_df, registry) so a caller (e.g.
    sentinelpay.eda.run_phase_c) can report exactly what was computed without
    the registry and the code drifting apart.

    Does not accept or read `isFraud` (or any target column) -- pass any
    frame that has the source columns above; an `isFraud` column, if present,
    is simply ignored.
    """
    out = df[[id_col]].copy()
    tmp = add_amt_log1p(df, amount_col=amount_col)
    out["amt_log1p"] = tmp["amt_log1p"]
    tmp = add_amt_decimal_part(df, amount_col=amount_col)
    out["amt_decimal_part"] = tmp["amt_decimal_part"]
    tmp = add_dt_hour_of_day(df, dt_col=dt_col)
    out["dt_hour_of_day"] = tmp["dt_hour_of_day"]
    tmp = add_dt_day_of_week(df, dt_col=dt_col)
    out["dt_day_of_week"] = tmp["dt_day_of_week"]
    tmp = add_has_identity(df, identity_ids, id_col=id_col)
    out["has_identity"] = tmp["has_identity"]

    out = out.drop(columns=[id_col])
    return out, [dict(entry) for entry in FEATURE_REGISTRY]
