import inspect

import pandas as pd
import pytest

from sentinelpay.eda.component_analysis import (
    DEVICE_NODE_COL,
    OUTPUT_COLUMNS,
    PAYMENT_NODE_COL,
    compute_component_metrics,
    component_metrics_summary,
    component_metrics_summary_by_partition,
)


def _valid_frame():
    # Same tie-deferral scenario as tests/test_causal_components.py's
    # _synthetic_bucket_scenario, expressed with the real E.2 column names.
    return pd.DataFrame(
        {
            DEVICE_NODE_COL: ["D1", "D1", "D2", "D1"],
            PAYMENT_NODE_COL: ["P1", "P2", "P1", "P1"],
            "TransactionDT": [100, 200, 200, 300],
            "partition": ["train", "train", "validation", "validation"],
        }
    )


def test_compute_component_metrics_renames_and_matches_generic_primitive():
    df = _valid_frame()
    out = compute_component_metrics(df, dt_col="TransactionDT")
    assert list(out.columns) == OUTPUT_COLUMNS
    # Same hand-verified values as test_causal_components' bucket scenario.
    assert out.loc[0, "device_component_size_total"] == 1
    assert out.loc[0, "payment_component_size_total"] == 1
    assert out.loc[0, "merged_component_size_total"] == 2
    assert out.loc[3, "device_component_size_total"] == 4
    assert out.loc[3, "payment_component_size_total"] == 4
    assert out.loc[3, "endpoints_same_component"] == True  # noqa: E712
    assert out.loc[3, "merged_component_size_total"] == 4


def test_compute_component_metrics_no_target_dependency():
    assert "isFraud" not in inspect.signature(compute_component_metrics).parameters
    assert "target" not in inspect.signature(compute_component_metrics).parameters

    df = _valid_frame()
    df_with_target = df.copy()
    df_with_target["isFraud"] = [1, 0, 1, 0]
    a = compute_component_metrics(df, dt_col="TransactionDT")
    b = compute_component_metrics(df_with_target, dt_col="TransactionDT")
    pd.testing.assert_frame_equal(a, b)


def test_component_metrics_summary_shape_and_counts():
    df = _valid_frame()
    metrics = compute_component_metrics(df, dt_col="TransactionDT")
    summary = component_metrics_summary(metrics)

    assert summary["n_rows"] == 4
    assert summary["endpoints_same_component"]["n_true"] == 1
    assert summary["endpoints_same_component"]["n_false"] == 3
    assert summary["endpoints_same_component"]["pct_true"] == pytest.approx(25.0)
    for key in ("device_component_size_total", "payment_component_size_total", "merged_component_size_total"):
        assert set(summary[key].keys()) == {"min", "p25", "p50", "p75", "p90", "p99", "max", "mean"}


def test_component_metrics_summary_empty_input():
    metrics = pd.DataFrame(columns=OUTPUT_COLUMNS)
    summary = component_metrics_summary(metrics)
    assert summary["n_rows"] == 0
    assert summary["endpoints_same_component"]["n_true"] == 0
    assert summary["endpoints_same_component"]["n_false"] == 0


def test_component_metrics_summary_by_partition():
    df = _valid_frame()
    metrics = compute_component_metrics(df, dt_col="TransactionDT")
    rows = component_metrics_summary_by_partition(
        metrics, df, partition_col="partition", partitions=["train", "validation"]
    )
    by_name = {r["partition"]: r for r in rows}
    assert by_name["train"]["n_rows"] == 2  # idx 0, 1
    assert by_name["validation"]["n_rows"] == 2  # idx 2, 3


def test_component_metrics_summary_by_partition_requires_column():
    df = _valid_frame()
    metrics = compute_component_metrics(df, dt_col="TransactionDT")
    with pytest.raises(ValueError):
        component_metrics_summary_by_partition(metrics, df, partition_col="no_such_col", partitions=["train"])
