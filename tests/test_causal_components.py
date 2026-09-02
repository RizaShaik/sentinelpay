import numpy as np
import pandas as pd
import pytest

from sentinelpay.data.causal_components import (
    OUTPUT_COLUMNS,
    causal_bipartite_component_metrics,
)
from sentinelpay.data.union_find import UnionFind


def _synthetic_bucket_scenario():
    # Deliberately synthetic node columns/values (never _device_node/
    # _payment_node) -- mirrors sentinelpay.data.history's own convention of
    # testing generic primitives against synthetic column names only.
    #
    # dt=100: one edge D1-P1 (first ever seen).
    # dt=200: TWO edges tied at the same dt -- D1-P2 and D2-P1 -- exercising
    #   "ties never see each other": each must be measured using ONLY the
    #   dt=100 structure, neither aware that D1 (resp. P1) is about to gain a
    #   new bucket-mate edge at the same timestamp.
    # dt=300: one edge D1-P1 again, now already in the same component --
    #   exercises merged_component_size_total == shared size, not a sum.
    return pd.DataFrame(
        {
            "node_a": ["D1", "D1", "D2", "D1"],
            "node_b": ["P1", "P2", "P1", "P1"],
            "TransactionDT": [100, 200, 200, 300],
        }
    )


def test_hand_computed_bucket_deferral_and_merged_size():
    df = _synthetic_bucket_scenario()
    out = causal_bipartite_component_metrics(df, "node_a", "node_b", "TransactionDT")

    # idx0: dt=100, D1-P1 -- both brand new, isolated singletons.
    assert out.loc[0, "node_a_component_size_total"] == 1
    assert out.loc[0, "node_b_component_size_total"] == 1
    assert out.loc[0, "endpoints_same_component"] == False  # noqa: E712
    assert out.loc[0, "merged_component_size_total"] == 2
    # After dt=100 is fully read, {D1, P1} is unioned -> component size 2.

    # idx1: dt=200, D1-P2. Measured BEFORE any dt=200 union: D1's component
    # is {D1, P1} (size 2, from dt=100). P2 is brand new (size 1).
    assert out.loc[1, "node_a_component_size_total"] == 2
    assert out.loc[1, "node_b_component_size_total"] == 1
    assert out.loc[1, "endpoints_same_component"] == False  # noqa: E712
    assert out.loc[1, "merged_component_size_total"] == 3

    # idx2: dt=200 (tied with idx1), D2-P1. Measured from the SAME pre-bucket
    # structure as idx1 -- P1's component is still size 2 ({D1, P1}), NOT 3,
    # even though idx1's D1-P2 edge belongs to the same bucket: idx1's edge
    # has not been unioned in yet when idx2 is measured. D2 is brand new.
    assert out.loc[2, "node_a_component_size_total"] == 1
    assert out.loc[2, "node_b_component_size_total"] == 2
    assert out.loc[2, "endpoints_same_component"] == False  # noqa: E712
    assert out.loc[2, "merged_component_size_total"] == 3
    # After dt=200 is fully read, both its edges are unioned: D1-P2 merges
    # {D1,P1} with {P2} -> size 3; D2-P1 then merges D2 in -> final
    # component {D1, D2, P1, P2}, size 4.

    # idx3: dt=300, D1-P1 -- both endpoints are now in the SAME size-4
    # component (built from dt=100 and dt=200). merged_component_size_total
    # must equal that shared size (4), not a sum (4+4=8).
    assert out.loc[3, "node_a_component_size_total"] == 4
    assert out.loc[3, "node_b_component_size_total"] == 4
    assert out.loc[3, "endpoints_same_component"] == True  # noqa: E712
    assert out.loc[3, "merged_component_size_total"] == 4


def test_output_columns_and_index_alignment():
    df = _synthetic_bucket_scenario()
    out = causal_bipartite_component_metrics(df, "node_a", "node_b", "TransactionDT")
    assert list(out.columns) == OUTPUT_COLUMNS
    assert list(out.index) == list(df.index)


def test_requires_columns():
    df = _synthetic_bucket_scenario()
    with pytest.raises(ValueError):
        causal_bipartite_component_metrics(df, "no_such_col", "node_b", "TransactionDT")
    with pytest.raises(ValueError):
        causal_bipartite_component_metrics(df, "node_a", "no_such_col", "TransactionDT")
    with pytest.raises(ValueError):
        causal_bipartite_component_metrics(df, "node_a", "node_b", "no_such_col")


def test_requires_non_null_node_columns():
    df = _synthetic_bucket_scenario()
    df_with_null_a = df.copy()
    df_with_null_a.loc[1, "node_a"] = None
    with pytest.raises(ValueError):
        causal_bipartite_component_metrics(df_with_null_a, "node_a", "node_b", "TransactionDT")

    df_with_null_b = df.copy()
    df_with_null_b.loc[2, "node_b"] = None
    with pytest.raises(ValueError):
        causal_bipartite_component_metrics(df_with_null_b, "node_a", "node_b", "TransactionDT")


def test_node_types_never_collide_across_columns():
    # A node value that happens to appear identically in both columns (e.g.
    # "X") must never be treated as the same graph node.
    df = pd.DataFrame({"node_a": ["X"], "node_b": ["X"], "TransactionDT": [1]})
    out = causal_bipartite_component_metrics(df, "node_a", "node_b", "TransactionDT")
    assert out.loc[0, "node_a_component_size_total"] == 1
    assert out.loc[0, "node_b_component_size_total"] == 1
    assert out.loc[0, "endpoints_same_component"] == False  # noqa: E712
    assert out.loc[0, "merged_component_size_total"] == 2


def test_tied_timestamps_never_see_each_other_minimal():
    # Two edges, same dt, sharing node_b -- neither may observe the other's
    # union even though both eventually merge into the same component.
    df = pd.DataFrame(
        {
            "node_a": ["D1", "D2"],
            "node_b": ["P1", "P1"],
            "TransactionDT": [500, 500],
        }
    )
    out = causal_bipartite_component_metrics(df, "node_a", "node_b", "TransactionDT")
    assert (out["node_a_component_size_total"] == 1).all()
    assert (out["node_b_component_size_total"] == 1).all()
    assert (out["endpoints_same_component"] == False).all()  # noqa: E712
    assert (out["merged_component_size_total"] == 2).all()


def test_row_order_independence():
    df = _synthetic_bucket_scenario()
    shuffled = df.sample(frac=1.0, random_state=5).reset_index(drop=True)

    out_orig = causal_bipartite_component_metrics(df, "node_a", "node_b", "TransactionDT")
    out_shuf = causal_bipartite_component_metrics(shuffled, "node_a", "node_b", "TransactionDT")

    # Align by (node_a, node_b, TransactionDT) triple since dt=200 has two
    # distinct edges (not literal duplicates) that must each keep their own
    # metrics regardless of row order.
    def _key(frame, i):
        return (frame.loc[i, "node_a"], frame.loc[i, "node_b"], frame.loc[i, "TransactionDT"])

    lookup = {_key(df, i): tuple(out_orig.loc[i]) for i in df.index}
    for i in shuffled.index:
        assert lookup[_key(shuffled, i)] == tuple(out_shuf.loc[i])


def test_adversarial_future_row_does_not_change_earlier_rows():
    df = _synthetic_bucket_scenario()
    before = causal_bipartite_component_metrics(df, "node_a", "node_b", "TransactionDT")

    new_row = pd.DataFrame({"node_a": ["D9"], "node_b": ["P9"], "TransactionDT": [10_000]})
    df_extended = pd.concat([df, new_row], ignore_index=True)
    after = causal_bipartite_component_metrics(df_extended, "node_a", "node_b", "TransactionDT")

    for idx in df.index:
        for col in OUTPUT_COLUMNS:
            assert before.loc[idx, col] == after.loc[idx, col]

    # Mutating the LAST row's endpoints must also never reach earlier rows.
    df_mutated = df.copy()
    df_mutated.loc[3, "node_a"] = "BRAND_NEW"
    df_mutated.loc[3, "node_b"] = "ALSO_NEW"
    after_mutation = causal_bipartite_component_metrics(df_mutated, "node_a", "node_b", "TransactionDT")
    for idx in [0, 1, 2]:
        for col in OUTPUT_COLUMNS:
            assert before.loc[idx, col] == after_mutation.loc[idx, col]


def test_no_target_dependency():
    import inspect

    assert "isFraud" not in inspect.signature(causal_bipartite_component_metrics).parameters
    assert "target" not in inspect.signature(causal_bipartite_component_metrics).parameters

    df = _synthetic_bucket_scenario()
    df_with_target = df.copy()
    df_with_target["isFraud"] = [1, 0, 1, 0]

    a = causal_bipartite_component_metrics(df, "node_a", "node_b", "TransactionDT")
    b = causal_bipartite_component_metrics(df_with_target, "node_a", "node_b", "TransactionDT")
    pd.testing.assert_frame_equal(a, b)


def test_custom_and_duplicate_index_alignment():
    df = _synthetic_bucket_scenario()
    baseline = causal_bipartite_component_metrics(df, "node_a", "node_b", "TransactionDT")

    df_custom_index = df.copy()
    df_custom_index.index = [100, 205, 7, 999]
    out_custom = causal_bipartite_component_metrics(df_custom_index, "node_a", "node_b", "TransactionDT")
    assert list(out_custom.index) == list(df_custom_index.index)
    for pos, idx in enumerate(df_custom_index.index):
        for col in OUTPUT_COLUMNS:
            assert out_custom.loc[idx, col] == baseline.iloc[pos][col]

    df_dup_index = df.copy()
    df_dup_index.index = [0, 0, 1, 1]
    out_dup = causal_bipartite_component_metrics(df_dup_index, "node_a", "node_b", "TransactionDT")
    assert list(out_dup.index) == list(df_dup_index.index)
    for pos in range(len(df)):
        for col in OUTPUT_COLUMNS:
            assert out_dup.iloc[pos][col] == baseline.iloc[pos][col]


# ---------------------------------------------------------------------------
# Brute-force oracle: a from-scratch, strictly-earlier-edge connected-
# components implementation, independent of causal_bipartite_component_metrics'
# own bucketing code path.
# ---------------------------------------------------------------------------


def _brute_force_component_metrics(df: pd.DataFrame, node_a_col: str, node_b_col: str, dt_col: str) -> pd.DataFrame:
    a_size, b_size, same, merged = [], [], [], []
    for i in df.index:
        dt_i = df.loc[i, dt_col]
        prior = df[df[dt_col] < dt_i]  # STRICTLY earlier -- excludes i's own bucket entirely, including ties.
        uf = UnionFind()
        for _, row in prior.iterrows():
            uf.union(("a", row[node_a_col]), ("b", row[node_b_col]))
        ka = ("a", df.loc[i, node_a_col])
        kb = ("b", df.loc[i, node_b_col])
        ra, rb = uf.find(ka), uf.find(kb)
        sa, sb = uf.size(ra), uf.size(rb)
        is_same = ra == rb
        a_size.append(sa)
        b_size.append(sb)
        same.append(is_same)
        merged.append(sa if is_same else sa + sb)
    return pd.DataFrame(
        {
            "node_a_component_size_total": a_size,
            "node_b_component_size_total": b_size,
            "endpoints_same_component": same,
            "merged_component_size_total": merged,
        },
        index=df.index,
    )


def test_matches_brute_force_on_hand_scenario():
    # Sanity-check the oracle itself against the hand-computed scenario above
    # before trusting it on random data.
    df = _synthetic_bucket_scenario()
    fast = causal_bipartite_component_metrics(df, "node_a", "node_b", "TransactionDT")
    slow = _brute_force_component_metrics(df, "node_a", "node_b", "TransactionDT")
    pd.testing.assert_frame_equal(fast.astype({"endpoints_same_component": bool}), slow)


def test_matches_brute_force_on_random_data_with_many_ties():
    rng = np.random.default_rng(0)
    n = 50
    # Small node universes and a small dt range deliberately force both
    # heavy tie-bucketing and repeated cross-bucket merges.
    df = pd.DataFrame(
        {
            "node_a": rng.choice(["D1", "D2", "D3", "D4"], size=n),
            "node_b": rng.choice(["P1", "P2", "P3"], size=n),
            "TransactionDT": rng.integers(0, 10, size=n),
        }
    )
    fast = causal_bipartite_component_metrics(df, "node_a", "node_b", "TransactionDT")
    slow = _brute_force_component_metrics(df, "node_a", "node_b", "TransactionDT")
    pd.testing.assert_frame_equal(fast.astype({"endpoints_same_component": bool}), slow)


def test_matches_brute_force_row_order_shuffled():
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame(
        {
            "node_a": rng.choice(["D1", "D2", "D3"], size=n),
            "node_b": rng.choice(["P1", "P2"], size=n),
            "TransactionDT": rng.integers(0, 8, size=n),
        }
    )
    shuffled = df.sample(frac=1.0, random_state=42)  # keep original index -- exercises non-default index too
    fast = causal_bipartite_component_metrics(shuffled, "node_a", "node_b", "TransactionDT")
    slow = _brute_force_component_metrics(shuffled, "node_a", "node_b", "TransactionDT")
    pd.testing.assert_frame_equal(fast.astype({"endpoints_same_component": bool}), slow)
