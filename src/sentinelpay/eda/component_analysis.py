"""Phase E.2: minimal per-transaction Union-Find / connected-component
metrics for the device_node <-> payment_node relationship.

Scope, approved from Phase E.1's evidence (reports/eda/phase_e1_report.md
section 5): E.1's M5b frequency-adjusted overlap diagnostic found
device_to_payment's `lift_ratio` (11.06x) the only relationship clearing
`LIFT_SIGNAL_THRESHOLD`, evidencing that investigating Union-Find/component
structure for a Phase E.2 mechanism is justified for THIS relationship only
-- not payment_email_purchaser or device_email_purchaser (both stayed below
the threshold; see `sentinelpay.eda.link_sufficiency.recommend_relationships`'s
`reason` field). E.2 is therefore scoped to device_node <-> payment_node
only: no other relationship, no scoring model, no persistence, no E.3 work.

Correction made to the E.2 proposal before this module was written: a
current `(device_node, payment_node)` edge's two endpoints may already
belong to two DIFFERENT causal components -- reporting only the component
containing one endpoint would silently discard the other endpoint's size and
the fact that this edge might be a first-time bridge between two previously
separate components. Every row therefore gets FOUR per-transaction metrics,
computed at read time strictly from Union-Find state built from earlier
TransactionDT buckets only (the full causal contract, including the
bucket-at-a-time read/mutate deferral, lives in
`sentinelpay.data.causal_components` and is not re-derived here):

    device_component_size_total   -- size of the device_node's own component
    payment_component_size_total  -- size of the payment_node's own component
    endpoints_same_component      -- whether they were already the same component
    merged_component_size_total   -- the hypothetical size if this edge were
                                      unioned right now: the shared size if
                                      endpoints_same_component, else the sum
                                      of the two component sizes above.

This module is a pure measurement pass: no scoring, no flags, no
`configs/detection.yaml`-style persistence, and no target of any kind read
while computing these metrics or their descriptive summary below (mirrors
Phase E.1's own non-target discipline exactly). Any target-reading
diagnostic evaluation is a separate, strictly-downstream step (see
`sentinelpay.eda.run_phase_e2`), matching Phase D's `evaluate_validation_only`
precedent of reading `isFraud` only after every non-target quantity is
already final.

`_device_node`/`_payment_node` are built and filtered exactly as
`sentinelpay.eda.run_phase_e1` already does for the same relationship pair
(`build_node_key_column` + `build_relationship_frame`) -- reused unchanged,
not re-implemented here.
"""
from __future__ import annotations

import pandas as pd

from sentinelpay.data.causal_components import causal_bipartite_component_metrics
from sentinelpay.eda.grouping_key_sufficiency import prior_count_distribution

DEVICE_NODE_COL = "_device_node"
PAYMENT_NODE_COL = "_payment_node"

OUTPUT_COLUMNS = [
    "device_component_size_total",
    "payment_component_size_total",
    "endpoints_same_component",
    "merged_component_size_total",
]

_RENAME = {
    "node_a_component_size_total": "device_component_size_total",
    "node_b_component_size_total": "payment_component_size_total",
}


def compute_component_metrics(valid_df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """Per-transaction E.2 component metrics for the device_node <->
    payment_node relationship, aligned to `valid_df.index`.

    `valid_df` must already be the output of
    `sentinelpay.eda.link_sufficiency.build_relationship_frame(df,
    DEVICE_NODE_COL, PAYMENT_NODE_COL)` (or equivalent) -- both node columns
    non-null on every row; `causal_bipartite_component_metrics` enforces this
    itself and raises otherwise. Never reads or requires `isFraud`.
    """
    raw = causal_bipartite_component_metrics(
        valid_df, node_a_col=DEVICE_NODE_COL, node_b_col=PAYMENT_NODE_COL, dt_col=dt_col
    )
    return raw.rename(columns=_RENAME)[OUTPUT_COLUMNS]


def component_metrics_summary(metrics: pd.DataFrame) -> dict:
    """Descriptive-only distribution summary of the four per-transaction
    metrics -- no evaluation criteria, no `isFraud`. Reuses
    `sentinelpay.eda.grouping_key_sufficiency.prior_count_distribution`
    unchanged (it is generic over any non-negative integer Series, not
    specific to D.1's own prior-count quantity)."""
    n = len(metrics)
    same_count = int(metrics["endpoints_same_component"].sum())
    return {
        "n_rows": n,
        "device_component_size_total": prior_count_distribution(metrics["device_component_size_total"]),
        "payment_component_size_total": prior_count_distribution(metrics["payment_component_size_total"]),
        "merged_component_size_total": prior_count_distribution(metrics["merged_component_size_total"]),
        "endpoints_same_component": {
            "n_true": same_count,
            "n_false": n - same_count,
            "pct_true": round(100.0 * same_count / n, 4) if n else float("nan"),
        },
    }


def component_metrics_summary_by_partition(
    metrics: pd.DataFrame, valid_df: pd.DataFrame, partition_col: str, partitions: list[str]
) -> list[dict]:
    """Same descriptive summary as `component_metrics_summary`, broken out
    per partition -- lets a reader see whether component growth is
    concentrated in `train` (expected, since it spans the most days) or
    holds up qualitatively in `validation`/`embargo_2` too. Purely
    descriptive, still no `isFraud`."""
    if partition_col not in valid_df.columns:
        raise ValueError(f"component_metrics_summary_by_partition requires column '{partition_col}'")
    rows = []
    for name in partitions:
        mask = (valid_df[partition_col] == name).to_numpy()
        sub = metrics[mask]
        summary = component_metrics_summary(sub)
        rows.append({"partition": name, **summary})
    return rows
