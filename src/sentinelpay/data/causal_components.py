"""Generic, target-agnostic, strictly-causal bipartite connected-component
edge metrics -- the Phase E.2 counterpart of `sentinelpay.data.history`'s
causal aggregation family, built on `sentinelpay.data.union_find.UnionFind`.

`causal_bipartite_component_metrics` answers, for every edge (row) `(a, b)`
in a bipartite graph read in `dt_col` order: "how big were `a`'s and `b`'s
components, and what would merging them right now produce," using ONLY
component structure built from strictly-earlier edges. Same causal contract
as `sentinelpay.data.history`, extended from per-row aggregation to
whole-graph structure:

- **Strictly-earlier bucket-mutation only.** Rows are grouped into buckets by
  distinct `dt_col` value. For every row in a bucket, `node_a_component_size_total`/
  `node_b_component_size_total`/`endpoints_same_component`/
  `merged_component_size_total` are computed from the Union-Find structure
  exactly as it stood after every STRICTLY EARLIER bucket's edges were
  unioned in -- and BEFORE any edge belonging to the row's own bucket is
  unioned. Only after every row in a bucket has had its metrics read does
  that bucket's edges actually get unioned into the structure (one call per
  edge; a bucket with duplicate edges or edges that transitively connect
  through each other is fully resolved by ordinary `union` semantics at that
  point).
- **Ties never see each other.** Two edges sharing a `dt_col` value are
  computed from IDENTICAL pre-bucket structure -- neither can affect the
  other's `node_a_component_size_total`/`node_b_component_size_total`/
  `endpoints_same_component`/`merged_component_size_total`, even though both
  edges are unioned into the same structure once the bucket's reads are done.
- **Row-order independence.** Bucketing is by `dt_col` VALUE only (via
  `groupby`), never by row position -- shuffling the input rows does not
  change any row's result (each row's own array position, not its former
  DataFrame row order, drives where its output is written).
- **Future perturbation invariance.** A row's metrics depend only on buckets
  strictly before its own `dt_col`; mutating or appending a row at or after a
  given row's `dt_col` cannot change that row's already-determined metrics.

`merged_component_size_total` is the row's hypothetical CURRENT-EDGE merge
quantity: if `node_a` and `node_b` are already in the same component (from
strictly-earlier edges), it equals that shared component's size; otherwise it
equals the sum of the two (still-separate) prior component sizes. This is a
read-time quantity about what unioning this one edge right now would produce
-- it is not itself unioned into the structure until the row's whole bucket
has been read (see above).

`node_a_col`/`node_b_col`/`dt_col` are fully generic -- this module does not
choose, name, or endorse device_node/payment_node or any other production
node typing; that is `sentinelpay.eda.component_analysis`'s job (Phase E.2's
own domain application), the same division of responsibility
`sentinelpay.data.history` already has with `sentinelpay.eda.link_sufficiency`.

Both node columns are required non-null on every row -- unlike
`prior_group_distinct_other_count` (where a null `other_col` value simply
contributes no partner), a bipartite edge with a missing endpoint isn't a
degenerate edge, it's not an edge at all; callers filter with
`sentinelpay.eda.link_sufficiency.build_relationship_frame` (or equivalent)
before calling this function, exactly as `sentinelpay.eda.run_phase_e1` already
does for the same `_device_node`/`_payment_node` columns.

No target dependency: this function does not accept or read `isFraud` (or any
target column).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinelpay.data.union_find import UnionFind

OUTPUT_COLUMNS = [
    "node_a_component_size_total",
    "node_b_component_size_total",
    "endpoints_same_component",
    "merged_component_size_total",
]


def causal_bipartite_component_metrics(
    df: pd.DataFrame, node_a_col: str, node_b_col: str, dt_col: str
) -> pd.DataFrame:
    """Strictly-causal, bucket-at-a-time bipartite component metrics, aligned
    to `df.index`. See module docstring for the full causal contract.

    Raises `ValueError` if `node_a_col`/`node_b_col`/`dt_col` is missing, or
    if either node column contains a null value anywhere in `df` (a bipartite
    edge requires both endpoints; this function does not impute or silently
    drop such rows -- filter with `build_relationship_frame` first).
    """
    for col in (node_a_col, node_b_col, dt_col):
        if col not in df.columns:
            raise ValueError(f"causal_bipartite_component_metrics requires column '{col}'")
    if df[node_a_col].isna().any() or df[node_b_col].isna().any():
        raise ValueError(
            "causal_bipartite_component_metrics requires non-null node_a_col/node_b_col on every row "
            "-- filter with build_relationship_frame (or equivalent) first"
        )

    n = len(df)
    working = df[[node_a_col, node_b_col, dt_col]].copy()
    # Namespaced node identities: node_a and node_b are two different node
    # TYPES, so a value that happens to be equal as a raw string between the
    # two columns (e.g. a coincidental collision) must never be treated as
    # the same graph node.
    working["_key_a"] = list(zip(["a"] * n, working[node_a_col]))
    working["_key_b"] = list(zip(["b"] * n, working[node_b_col]))
    working["_pos"] = np.arange(n)

    uf = UnionFind()
    a_size = np.zeros(n, dtype="int64")
    b_size = np.zeros(n, dtype="int64")
    same = np.zeros(n, dtype=bool)
    merged = np.zeros(n, dtype="int64")

    for _dt_value, bucket in working.groupby(dt_col, sort=True, observed=True):
        positions = bucket["_pos"].to_numpy()
        keys_a = bucket["_key_a"].to_numpy()
        keys_b = bucket["_key_b"].to_numpy()

        # Phase 1 (read): every row in this bucket is measured against
        # Union-Find state built from strictly-earlier buckets only -- no
        # edge from this bucket has been unioned in yet.
        for j in range(len(positions)):
            ka, kb = keys_a[j], keys_b[j]
            ra, rb = uf.find(ka), uf.find(kb)
            sa, sb = uf.size(ra), uf.size(rb)
            is_same = ra == rb
            pos = positions[j]
            a_size[pos] = sa
            b_size[pos] = sb
            same[pos] = is_same
            merged[pos] = sa if is_same else sa + sb

        # Phase 2 (mutate): only now, after every row in the bucket has been
        # read, union this bucket's edges into the structure.
        for j in range(len(positions)):
            uf.union(keys_a[j], keys_b[j])

    result = pd.DataFrame(
        {
            "node_a_component_size_total": a_size,
            "node_b_component_size_total": b_size,
            "endpoints_same_component": same,
            "merged_component_size_total": merged,
        },
        index=working.index,
    )
    return result[OUTPUT_COLUMNS]
