import pytest

from sentinelpay.data.union_find import UnionFind


def test_lazy_singleton_defaults():
    uf = UnionFind()
    assert uf.size("x") == 1
    assert uf.find("x") == "x"
    assert not uf.connected("x", "y")
    assert len(uf) == 2  # "x" and "y" both got created by the calls above


def test_union_merges_and_sizes():
    uf = UnionFind()
    uf.union("a", "b")
    assert uf.connected("a", "b")
    assert uf.size("a") == 2
    assert uf.size("b") == 2

    uf.union("b", "c")
    assert uf.connected("a", "c")
    assert uf.size("a") == 3
    assert uf.size("b") == 3
    assert uf.size("c") == 3


def test_union_is_idempotent_and_does_not_double_count():
    uf = UnionFind()
    uf.union("a", "b")
    root_before = uf.find("a")
    size_before = uf.size("a")
    uf.union("a", "b")  # already unioned -- must be a no-op
    uf.union("b", "a")  # reversed order -- still a no-op
    assert uf.find("a") == root_before
    assert uf.size("a") == size_before == 2


def test_disjoint_components_stay_separate():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("x", "y")
    assert uf.connected("a", "b")
    assert uf.connected("x", "y")
    assert not uf.connected("a", "x")
    assert uf.size("a") == 2
    assert uf.size("x") == 2


def test_union_by_size_path_compression_large_chain():
    uf = UnionFind()
    n = 200
    for i in range(1, n):
        uf.union(i - 1, i)
    root = uf.find(0)
    assert uf.size(0) == n
    for i in range(n):
        assert uf.find(i) == root
        assert uf.size(i) == n


def test_len_counts_distinct_nodes_not_components():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("c", "d")
    assert len(uf) == 4  # 2 components, 4 nodes


def test_contains():
    uf = UnionFind()
    assert "a" not in uf
    uf.find("a")
    assert "a" in uf
    assert "never_touched" not in uf


def test_typed_keys_never_collide():
    # Mirrors sentinelpay.data.causal_components' ("a", value)/("b", value)
    # namespacing -- two different-typed keys sharing a raw value must never
    # be treated as the same node.
    uf = UnionFind()
    uf.union(("device", "V1"), ("payment", "P1"))
    assert not uf.connected(("device", "V1"), ("payment", "V1"))
    assert uf.size(("device", "V1")) == 2
    assert uf.size(("payment", "V1")) == 1
