"""Candidate entity-relationship investigation (EDA-level only).

None of the groupings below are documented entity keys -- IEEE-CIS never
publishes a "customer ID" or "device ID". They are reasonable PROXY
groupings widely used in public analyses of this dataset (e.g. card
attributes + address as a `payment_proxy_key`, or a shared device/browser
fingerprint as a `device_proxy_key`). Every function here is investigative:
it reports how much shared structure exists and whether it correlates with
fraud, so later phases can decide whether a proxy is worth building
coordinated-ring features on. It does NOT assert that these proxies
identify real-world entities, and it is NOT the Phase E coordinated-ring
detection system -- this module produces exploratory evidence only.

EDA-only / no target encoding: `shared_key_fraud_summary` computes a
whole-dataset fraud rate per proxy-key group for inspection. That number
must never be used directly as a modeling feature. Any future
proxy-key-derived feature (e.g. "this payment_proxy_key's historical fraud
rate") must be computed using only information available strictly before
the transaction being scored -- chronological, fold-safe logic -- not the
global aggregate this function returns.
"""
from __future__ import annotations

import pandas as pd


def group_size_distribution(
    df: pd.DataFrame, proxy_key_columns: list[str], target_col: str | None = None
) -> pd.DataFrame:
    """Distribution of group sizes for a candidate proxy key, and (if a
    target column is given) the fraud rate by group-size bucket."""
    present = [c for c in proxy_key_columns if c in df.columns]
    if not present:
        raise ValueError(f"None of {proxy_key_columns} present in dataframe")

    valid = df.dropna(subset=present)
    sizes = valid.groupby(present, observed=True).size().rename("group_size")

    out = sizes.value_counts().rename("n_groups").sort_index().reset_index()
    out.columns = ["group_size", "n_groups"]
    out["n_rows_covered"] = out["group_size"] * out["n_groups"]

    if target_col and target_col in df.columns:
        merged = valid.merge(sizes.rename("group_size"), left_on=present, right_index=True)
        bucket_edges = [1, 2, 3, 6, 11, 26, 101, float("inf")]
        bucket_labels = ["1", "2", "3-5", "6-10", "11-25", "26-100", "100+"]
        merged["_bucket"] = pd.cut(merged["group_size"], bins=bucket_edges, labels=bucket_labels, right=False)
        fraud_by_bucket = merged.groupby("_bucket", observed=True)[target_col].agg(
            n_rows="count", fraud_rate="mean"
        ).reset_index()
        fraud_by_bucket.columns = ["group_size_bucket", "n_rows", "fraud_rate"]
        return fraud_by_bucket

    return out


def group_size_summary_stats(df: pd.DataFrame, proxy_key_columns: list[str]) -> dict:
    """Cheap scalar summary of proxy-key group sizes (no per-size table),
    for compact reporting: total groups, singleton count, largest group."""
    present = [c for c in proxy_key_columns if c in df.columns]
    valid = df.dropna(subset=present)
    sizes = valid.groupby(present, observed=True).size()
    return {
        "proxy_key_columns": present,
        "n_rows_valid": int(len(valid)),
        "n_groups": int(len(sizes)),
        "n_singleton_groups": int((sizes == 1).sum()),
        "max_group_size": int(sizes.max()) if len(sizes) else 0,
        "median_group_size": float(sizes.median()) if len(sizes) else float("nan"),
    }


def shared_key_fraud_summary(
    df: pd.DataFrame, proxy_key_columns: list[str], target_col: str = "isFraud", min_group_size: int = 2
) -> dict:
    """EDA-only: compare fraud rate for rows whose proxy key is shared by
    >= min_group_size rows vs. rows whose key is unique (singleton). Do not
    feed this result directly into a feature pipeline -- see module
    docstring."""
    present = [c for c in proxy_key_columns if c in df.columns]
    valid = df.dropna(subset=present)
    sizes = valid.groupby(present, observed=True).size().rename("group_size")
    merged = valid.merge(sizes, left_on=present, right_index=True)

    shared = merged[merged["group_size"] >= min_group_size]
    singleton = merged[merged["group_size"] < min_group_size]

    return {
        "proxy_key_columns": present,
        "n_rows_valid": len(valid),
        "n_rows_shared": len(shared),
        "n_rows_singleton": len(singleton),
        "fraud_rate_shared": float(shared[target_col].mean()) if len(shared) else float("nan"),
        "fraud_rate_singleton": float(singleton[target_col].mean()) if len(singleton) else float("nan"),
        "overall_fraud_rate": float(merged[target_col].mean()) if len(merged) else float("nan"),
    }
