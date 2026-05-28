# SPDX-License-Identifier: Apache-2.0
"""GNN gene — message passing op の 5 パラメータ低次元遺伝子表現.

設計 (5 parameters, scalar 同次元 hidden_dim=4 適用):
    aggregation: agg(h_v) = α_sum * Σh_u + α_mean * mean(h_u) + α_max * max(h_u)
        where (α_sum, α_mean, α_max) ∈ Δ^2 (simplex, 正規化)
    update:      h_v_new = tanh(W * h_v + U * agg(h_v))

clip 範囲:
    α_sum, α_mean, α_max >= 0 + Σ α_* = 1 (simplex 投影)
    W ∈ [-1, 1]
    U ∈ [-1, 1]

graph: N=8 node の 1D ring topology (固定構造、ChangeOp は別 PoC).

honest 留保:
- α_sum と α_mean は構造的に scale (Σ h_u = N * mean(h_u)) で関連するが、
  per-node hidden_dim 同次元 (W/U scalar) で適用するため "係数空間" としては別軸として残す
- ring topology + N=8 固定。ChangeOp (node 追加/edge 削除) は本 PoC scope 外、
  別 PoC で扱う (llcore changeop module 流用候補).
- aggregation は permutation-equivariant op (sum/mean/max) の凸結合 → 構造的に
  permutation-equivariance が成立 (gnn_verifier.verify_equivariance_structure で
  symbolic に確認).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Gene
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GnnGene:
    """GNN message passing 低次元遺伝子 (5 scalar params).

    Attributes
    ----------
    alpha_sum : float
        sum-aggregation 重み (simplex 制約: alpha_sum + alpha_mean + alpha_max = 1).
    alpha_mean : float
        mean-aggregation 重み.
    alpha_max : float
        max-aggregation 重み.
    W : float
        self-update scalar (per-dim 同係数). h_v 経路.
    U : float
        message-update scalar (per-dim 同係数). agg 経路.
    """

    alpha_sum: float
    alpha_mean: float
    alpha_max: float
    W: float
    U: float

    def clipped(self) -> "GnnGene":
        """clip + simplex 投影で正規化."""
        a_sum = max(0.0, float(self.alpha_sum))
        a_mean = max(0.0, float(self.alpha_mean))
        a_max = max(0.0, float(self.alpha_max))
        s = a_sum + a_mean + a_max
        if s <= 1e-12:
            # 退化 = uniform に均す
            a_sum = a_mean = a_max = 1.0 / 3.0
        else:
            a_sum, a_mean, a_max = a_sum / s, a_mean / s, a_max / s
        W = float(np.clip(self.W, -1.0, 1.0))
        U = float(np.clip(self.U, -1.0, 1.0))
        return GnnGene(a_sum, a_mean, a_max, W, U)

    def as_array(self) -> np.ndarray:
        """5-vector (alpha_sum, alpha_mean, alpha_max, W, U)."""
        return np.array(
            [self.alpha_sum, self.alpha_mean, self.alpha_max, self.W, self.U],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "GnnGene":
        if len(arr) != 5:
            raise ValueError(f"GnnGene expects 5 params, got {len(arr)}")
        return cls(
            alpha_sum=float(arr[0]),
            alpha_mean=float(arr[1]),
            alpha_max=float(arr[2]),
            W=float(arr[3]),
            U=float(arr[4]),
        )


# ---------------------------------------------------------------------------
# Forward (numpy reference impl)
# ---------------------------------------------------------------------------


def aggregate(gene: GnnGene, h_neighbors: np.ndarray) -> np.ndarray:
    """近傍 hidden の凸結合 aggregation.

    Parameters
    ----------
    gene : GnnGene
        clipped 推奨 (内部で clipped 適用).
    h_neighbors : np.ndarray
        shape (num_neighbors, hidden_dim) の近傍 hidden states.

    Returns
    -------
    np.ndarray
        shape (hidden_dim,) の aggregated message.
    """
    g = gene.clipped()
    if h_neighbors.size == 0:
        # 孤立 node (この PoC では発生しない、ring なので必ず 2 近傍)
        return np.zeros(h_neighbors.shape[1] if h_neighbors.ndim == 2 else 0)
    sum_part = h_neighbors.sum(axis=0)
    mean_part = h_neighbors.mean(axis=0)
    max_part = h_neighbors.max(axis=0)
    return g.alpha_sum * sum_part + g.alpha_mean * mean_part + g.alpha_max * max_part


def update_node(gene: GnnGene, h_v: np.ndarray, agg: np.ndarray) -> np.ndarray:
    """update: h_v_new = tanh(W * h_v + U * agg)."""
    g = gene.clipped()
    return np.tanh(g.W * h_v + g.U * agg)


def _ring_adjacency(n_nodes: int) -> list[list[int]]:
    """1D ring graph (周期境界) の adjacency list."""
    return [
        [(v - 1) % n_nodes, (v + 1) % n_nodes] for v in range(n_nodes)
    ]


def forward_layer(
    gene: GnnGene,
    h: np.ndarray,
    adjacency: list[list[int]] | None = None,
) -> np.ndarray:
    """GNN 1 層 forward.

    Parameters
    ----------
    gene : GnnGene
    h : np.ndarray
        shape (n_nodes, hidden_dim).
    adjacency : list[list[int]] | None
        各 node の近傍 index list. None なら ring topology を生成.

    Returns
    -------
    np.ndarray
        次層の hidden states, shape (n_nodes, hidden_dim).
    """
    n_nodes = h.shape[0]
    adj = adjacency if adjacency is not None else _ring_adjacency(n_nodes)
    h_new = np.empty_like(h)
    for v in range(n_nodes):
        neigh_idx = adj[v]
        h_neighbors = h[neigh_idx] if neigh_idx else np.empty((0, h.shape[1]))
        agg = aggregate(gene, h_neighbors)
        h_new[v] = update_node(gene, h[v], agg)
    return h_new


def forward_stack(
    gene: GnnGene,
    h0: np.ndarray,
    n_layers: int,
    adjacency: list[list[int]] | None = None,
) -> np.ndarray:
    """GNN を n_layers 適用."""
    h = h0
    for _ in range(n_layers):
        h = forward_layer(gene, h, adjacency)
    return h


def variance_across_nodes(h: np.ndarray) -> float:
    """over-smoothing 計測用: ノード間 hidden state の variance 平均 (axis=0).

    over-smoothing = 全 node が同じ表現に収束 → variance → 0.
    """
    if h.shape[0] < 2:
        return 0.0
    # 各 hidden dim ごとの分散を平均
    return float(h.var(axis=0).mean())


__all__ = [
    "GnnGene",
    "aggregate",
    "update_node",
    "forward_layer",
    "forward_stack",
    "variance_across_nodes",
]
