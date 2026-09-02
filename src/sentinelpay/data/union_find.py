"""Generic disjoint-set (Union-Find) data structure.

Plain, non-causal, non-domain-specific: no knowledge of `TransactionDT`,
device/payment nodes, or any project-specific concept -- mirrors
`sentinelpay.data.history`'s own split between a generic primitive and its
Phase-specific domain application in `sentinelpay.eda` (see
`sentinelpay.data.causal_components` for the strictly-causal, bucket-at-a-time
edge-processing algorithm built on top of this class).

Path compression + union by size. Node creation is lazy: any hashable key is
treated as its own singleton component the first time it is seen by `find`,
`union`, `size`, or `connected` -- there is no separate "add node" step.
"""
from __future__ import annotations

from typing import Hashable


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[Hashable, Hashable] = {}
        self._size: dict[Hashable, int] = {}

    def _ensure(self, x: Hashable) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1

    def find(self, x: Hashable) -> Hashable:
        """Root of `x`'s component (creating `x` as a fresh singleton first,
        if unseen). Compresses the path from `x` to the root in place."""
        self._ensure(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            next_x = self._parent[x]
            self._parent[x] = root
            x = next_x
        return root

    def size(self, x: Hashable) -> int:
        """Size (total node count) of `x`'s component."""
        return self._size[self.find(x)]

    def connected(self, x: Hashable, y: Hashable) -> bool:
        return self.find(x) == self.find(y)

    def union(self, x: Hashable, y: Hashable) -> Hashable:
        """Merge `x`'s and `y`'s components (union by size). Returns the
        resulting root. A no-op (returns the shared root) if `x` and `y` are
        already in the same component -- never double-counts size."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if self._size[rx] < self._size[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        self._size[rx] += self._size[ry]
        del self._size[ry]
        return rx

    def __len__(self) -> int:
        """Number of distinct nodes ever seen (not the number of components)."""
        return len(self._parent)

    def __contains__(self, x: Hashable) -> bool:
        return x in self._parent
