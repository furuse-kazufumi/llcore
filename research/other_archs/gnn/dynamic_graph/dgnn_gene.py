# SPDX-License-Identifier: Apache-2.0
"""Dynamic GNN gene — graph 構造を ChangeOp 列で変化させる動的 PoC.

設計 (Stage 2 = 構造変化 ChangeOp 本格実証):
    - aggregation: agg(h_v) = α_sum * Σh_u + α_mean * mean(h_u) + α_max * max(h_u)
      (固定 ring PoC と同じ simplex)
    - update:      h_v_new = tanh(W * h_v + U * agg(h_v))   (固定 ring PoC と同じ)
    - graph:       N ∈ [N_MIN, N_MAX] の **動的 node 数** ring-like topology
    - ChangeOp:    add_node / remove_node / add_edge / remove_edge の 4 種

ChangeOp の構造変化保証 (permutation equivariance 不変):
    aggregation が sum/mean/max convex combination + nodewise 同じ W/U の限り、
    ChangeOp が op 自体を変えず graph topology のみを変えるため、
    permutation-equivariance は破られない (構造的保証)。

honest 留保:
- 真の 「構造変化 ChangeOp の sound refinement」 は ``dgnn_verifier.py`` で
  llcore.verifier.changeop / refinement と接続して検査する (Stage 2 の核)。
- node 数 N が動的に変わるため、Z3 invariant の閾値 (shrink_upper threshold) も
  N に依存。``dgnn_verifier.py`` で N-aware に再導出。
- add_node 時の新規 node の初期 hidden state は 0 で固定 (sound 上界保証)。
- remove_node 時、近傍 graph 連結性は維持するため自動的に edge 補修 (cycle 化)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

# llcore.* import 用 path (research/other_archs/gnn/dynamic_graph から src へ)
_PROJ_ROOT = Path(__file__).resolve().parents[4]
_SRC = _PROJ_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MIN = 6
N_MAX = 12
N_INIT = 8  # ring 初期 node 数
HIDDEN_DIM = 4
MAX_SEQ_LEN = 15  # MCC curriculum 上限

# ChangeOp op_type literal
GRAPH_OP_TYPES = ("add_node", "remove_node", "add_edge", "remove_edge")
GraphOpType = Literal["add_node", "remove_node", "add_edge", "remove_edge"]


# ---------------------------------------------------------------------------
# GraphChangeOp (独自 — llcore.verifier.changeop.ChangeOp とは op_type 値域が違う)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphChangeOp:
    """動的 graph 上の atomic 構造変更 op (4 種).

    llcore.verifier.changeop.ChangeOp は state_update gene 用 (decay_shift 等) で
    op_type 値域が固定 (OP_TYPES) のため、graph 構造変化用に **別 dataclass** を
    定義する (Codex F2 「llcore 本流 src/ への構造破綻リスク」を回避)。

    Attributes
    ----------
    op_type : str
        変更種別 (add_node / remove_node / add_edge / remove_edge).
    target : tuple[int, ...]
        対象 node/edge index. 解釈は op_type 依存:
        - add_node: target=() (新 node は max(existing_ids)+1 を採番)
        - remove_node: target=(node_id,) の 1-tuple
        - add_edge: target=(u, v) の 2-tuple
        - remove_edge: target=(u, v) の 2-tuple

    Notes
    -----
    sound 拡張 refinement relation の epsilon は op 種別ごとに以下を割当:
        add_node:    eps = 0.10  (新 node は h=0 で sound, 構造拡張のみ)
        remove_node: eps = 0.20  (近傍 hidden が消える, smoothing 寄りの影響大)
        add_edge:    eps = 0.05  (aggregation 範囲拡張、smoothing 微増)
        remove_edge: eps = 0.05  (aggregation 範囲縮小、信号伝達減)
    """

    op_type: str
    target: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.op_type not in GRAPH_OP_TYPES:
            raise ValueError(
                f"unknown op_type {self.op_type!r}; expected one of {GRAPH_OP_TYPES}"
            )
        if self.op_type == "remove_node" and len(self.target) != 1:
            raise ValueError(
                f"remove_node requires target=(node_id,), got {self.target}"
            )
        if self.op_type in ("add_edge", "remove_edge") and len(self.target) != 2:
            raise ValueError(
                f"{self.op_type} requires target=(u, v), got {self.target}"
            )
        if self.op_type == "add_node" and len(self.target) != 0:
            raise ValueError(
                f"add_node requires target=(), got {self.target}"
            )

    def magnitude(self) -> float:
        """変更の "大きさ" (refinement epsilon の基礎量)."""
        return {
            "add_node": 0.10,
            "remove_node": 0.20,
            "add_edge": 0.05,
            "remove_edge": 0.05,
        }[self.op_type]


@dataclass(frozen=True)
class GraphChangeOpSequence:
    """GraphChangeOp 列 (composition + total magnitude)."""

    ops: tuple[GraphChangeOp, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.ops)

    def total_magnitude(self) -> float:
        return float(sum(op.magnitude() for op in self.ops))

    def op_type_counts(self) -> dict[str, int]:
        """op_type 別出現数 (G8 diversity 用)."""
        counts = {t: 0 for t in GRAPH_OP_TYPES}
        for op in self.ops:
            counts[op.op_type] += 1
        return counts


# ---------------------------------------------------------------------------
# DynamicGraph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicGraph:
    """動的 graph (可変 node 数 + adjacency list).

    Attributes
    ----------
    n_nodes : int
        現在の node 数. N_MIN <= n_nodes <= N_MAX.
    adjacency : tuple[tuple[int, ...], ...]
        各 node の近傍 index tuple. adjacency[v] = (u1, u2, ...).
        無向 graph: u in adjacency[v] ⟺ v in adjacency[u].
    """

    n_nodes: int
    adjacency: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.adjacency) != self.n_nodes:
            raise ValueError(
                f"adjacency length {len(self.adjacency)} != n_nodes {self.n_nodes}"
            )

    def mean_degree(self) -> float:
        if self.n_nodes == 0:
            return 0.0
        return float(sum(len(adj) for adj in self.adjacency)) / self.n_nodes

    def max_degree(self) -> int:
        if self.n_nodes == 0:
            return 0
        return max(len(adj) for adj in self.adjacency)

    def n_edges(self) -> int:
        # 無向 graph (各 edge は両端で 1 回ずつ数えるため /2)
        return sum(len(adj) for adj in self.adjacency) // 2


def make_ring(n_nodes: int) -> DynamicGraph:
    """N node の 1D ring graph (周期境界) を構築."""
    if n_nodes < 3:
        raise ValueError(f"ring needs n_nodes >= 3, got {n_nodes}")
    adj = tuple(
        ((v - 1) % n_nodes, (v + 1) % n_nodes) for v in range(n_nodes)
    )
    return DynamicGraph(n_nodes=n_nodes, adjacency=adj)


# ---------------------------------------------------------------------------
# apply_changeop (graph 構造変化)
# ---------------------------------------------------------------------------


def apply_changeop(graph: DynamicGraph, op: GraphChangeOp) -> DynamicGraph:
    """GraphChangeOp を graph に適用した新 graph を返す (純関数, side effect なし).

    範囲外 op (N_MIN, N_MAX bound 違反 / 存在しない node/edge 操作) は
    入力 graph をそのまま返す (= no-op, sound 保守側).
    refinement 上の sound 性は保たれる (ε 上界は magnitude で決まり no-op でも有効).

    Parameters
    ----------
    graph : DynamicGraph
        変更前 graph.
    op : GraphChangeOp
        適用する変更.

    Returns
    -------
    DynamicGraph
        変更後 graph.
    """
    if op.op_type == "add_node":
        if graph.n_nodes >= N_MAX:
            return graph  # no-op (bound 違反)
        new_id = graph.n_nodes  # 新 node id = 現 n_nodes
        # 新 node を 2 つの既存 node に接続 (cycle 維持)
        # 末尾と先頭の中間に接続: 既存 last (n-1) と first (0)
        last_id = graph.n_nodes - 1
        first_id = 0
        # 既存 last - first edge があれば削除して間に挿入 (ring の自然拡張)
        new_adj_list: list[list[int]] = [list(adj) for adj in graph.adjacency]
        # last の近傍から first を削除 (あれば) して new_id を追加
        if first_id in new_adj_list[last_id]:
            new_adj_list[last_id] = [u for u in new_adj_list[last_id] if u != first_id]
        new_adj_list[last_id].append(new_id)
        # first の近傍から last を削除 (あれば) して new_id を追加
        if last_id in new_adj_list[first_id]:
            new_adj_list[first_id] = [u for u in new_adj_list[first_id] if u != last_id]
        new_adj_list[first_id].append(new_id)
        # 新 node の近傍 = (last, first)
        new_adj_list.append([last_id, first_id])
        adj = tuple(tuple(sorted(set(a))) for a in new_adj_list)
        return DynamicGraph(n_nodes=graph.n_nodes + 1, adjacency=adj)

    if op.op_type == "remove_node":
        if graph.n_nodes <= N_MIN:
            return graph
        (node_id,) = op.target
        if node_id < 0 or node_id >= graph.n_nodes:
            return graph
        # node_id を削除し、その近傍 2 つを互いに接続して連結性維持
        neighbors_of_removed = list(graph.adjacency[node_id])
        new_adj_list: list[list[int]] = []
        for v in range(graph.n_nodes):
            if v == node_id:
                continue
            adj_v = [u for u in graph.adjacency[v] if u != node_id]
            new_adj_list.append(adj_v)
        # neighbor 補修: 削除 node の 2 近傍を互いに接続 (cycle 維持)
        if len(neighbors_of_removed) >= 2:
            u, w = neighbors_of_removed[0], neighbors_of_removed[1]
            # node_id 削除後の id 再マッピング
            def remap(idx: int) -> int:
                return idx if idx < node_id else idx - 1
            u_new, w_new = remap(u), remap(w)
            if u_new != w_new and w_new not in new_adj_list[u_new]:
                new_adj_list[u_new].append(w_new)
                new_adj_list[w_new].append(u_new)
        # 全 index を新 id 系に変換
        def remap2(idx: int) -> int:
            return idx if idx < node_id else idx - 1
        adj = tuple(
            tuple(sorted({remap2(u) for u in a}))
            for a in new_adj_list
        )
        return DynamicGraph(n_nodes=graph.n_nodes - 1, adjacency=adj)

    if op.op_type == "add_edge":
        u, v = op.target
        if u == v or u < 0 or u >= graph.n_nodes or v < 0 or v >= graph.n_nodes:
            return graph
        if v in graph.adjacency[u]:
            return graph  # already exists
        new_adj_list: list[list[int]] = [list(a) for a in graph.adjacency]
        new_adj_list[u].append(v)
        new_adj_list[v].append(u)
        adj = tuple(tuple(sorted(set(a))) for a in new_adj_list)
        return DynamicGraph(n_nodes=graph.n_nodes, adjacency=adj)

    if op.op_type == "remove_edge":
        u, v = op.target
        if u == v or u < 0 or u >= graph.n_nodes or v < 0 or v >= graph.n_nodes:
            return graph
        if v not in graph.adjacency[u]:
            return graph  # doesn't exist
        # 連結性保護: degree=1 の node を孤立させない
        if len(graph.adjacency[u]) <= 1 or len(graph.adjacency[v]) <= 1:
            return graph
        new_adj_list: list[list[int]] = [list(a) for a in graph.adjacency]
        new_adj_list[u] = [w for w in new_adj_list[u] if w != v]
        new_adj_list[v] = [w for w in new_adj_list[v] if w != u]
        adj = tuple(tuple(sorted(set(a))) for a in new_adj_list)
        return DynamicGraph(n_nodes=graph.n_nodes, adjacency=adj)

    raise ValueError(f"unknown op_type {op.op_type}")


def apply_sequence(
    graph: DynamicGraph, seq: GraphChangeOpSequence
) -> DynamicGraph:
    """GraphChangeOp 列を順次適用し最終 graph を返す."""
    g = graph
    for op in seq.ops:
        g = apply_changeop(g, op)
    return g


# ---------------------------------------------------------------------------
# DynamicGnnGene (固定 ring PoC の GnnGene + ChangeOp 列)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicGnnGene:
    """動的 GNN 個体 = (message passing gene) + (ChangeOp seq).

    Attributes
    ----------
    alpha_sum, alpha_mean, alpha_max : float
        aggregation simplex (固定 ring PoC と同じ).
    W, U : float
        update 係数 (固定 ring PoC と同じ).
    changeop_seq : GraphChangeOpSequence
        本 individual が graph に適用する ChangeOp 列.
        進化対象: gene 5 軸 + ChangeOp 列の選択.
    """

    alpha_sum: float
    alpha_mean: float
    alpha_max: float
    W: float
    U: float
    changeop_seq: GraphChangeOpSequence = field(default_factory=GraphChangeOpSequence)

    def clipped(self) -> "DynamicGnnGene":
        """simplex 投影 + W/U clip (changeop_seq は不変)."""
        a_sum = max(0.0, float(self.alpha_sum))
        a_mean = max(0.0, float(self.alpha_mean))
        a_max = max(0.0, float(self.alpha_max))
        s = a_sum + a_mean + a_max
        if s <= 1e-12:
            a_sum = a_mean = a_max = 1.0 / 3.0
        else:
            a_sum, a_mean, a_max = a_sum / s, a_mean / s, a_max / s
        W = float(np.clip(self.W, -1.0, 1.0))
        U = float(np.clip(self.U, -1.0, 1.0))
        return DynamicGnnGene(a_sum, a_mean, a_max, W, U, self.changeop_seq)

    def gene_array(self) -> np.ndarray:
        return np.array(
            [self.alpha_sum, self.alpha_mean, self.alpha_max, self.W, self.U],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Forward (動的 graph 対応)
# ---------------------------------------------------------------------------


def aggregate(gene: DynamicGnnGene, h_neighbors: np.ndarray) -> np.ndarray:
    """近傍 hidden の凸結合 aggregation (動的 N でも N に依存しない凸結合)."""
    g = gene.clipped()
    if h_neighbors.size == 0:
        return np.zeros(h_neighbors.shape[1] if h_neighbors.ndim == 2 else HIDDEN_DIM)
    sum_part = h_neighbors.sum(axis=0)
    mean_part = h_neighbors.mean(axis=0)
    max_part = h_neighbors.max(axis=0)
    return g.alpha_sum * sum_part + g.alpha_mean * mean_part + g.alpha_max * max_part


def update_node(gene: DynamicGnnGene, h_v: np.ndarray, agg: np.ndarray) -> np.ndarray:
    g = gene.clipped()
    return np.tanh(g.W * h_v + g.U * agg)


def forward_layer(
    gene: DynamicGnnGene,
    h: np.ndarray,
    graph: DynamicGraph,
) -> np.ndarray:
    """動的 graph 上の GNN 1 層 forward.

    h.shape[0] と graph.n_nodes が一致しない場合は ValueError.
    """
    if h.shape[0] != graph.n_nodes:
        raise ValueError(
            f"h.shape[0]={h.shape[0]} != graph.n_nodes={graph.n_nodes}"
        )
    h_new = np.empty_like(h)
    for v in range(graph.n_nodes):
        neigh_idx = list(graph.adjacency[v])
        if neigh_idx:
            h_neighbors = h[neigh_idx]
        else:
            h_neighbors = np.empty((0, h.shape[1]))
        agg = aggregate(gene, h_neighbors)
        h_new[v] = update_node(gene, h[v], agg)
    return h_new


def forward_stack(
    gene: DynamicGnnGene,
    h0: np.ndarray,
    graph: DynamicGraph,
    n_layers: int,
) -> np.ndarray:
    h = h0
    for _ in range(n_layers):
        h = forward_layer(gene, h, graph)
    return h


def variance_across_nodes(h: np.ndarray) -> float:
    """ノード間 hidden state の variance 平均 (axis=0)."""
    if h.shape[0] < 2:
        return 0.0
    return float(h.var(axis=0).mean())


# ---------------------------------------------------------------------------
# Hidden state 拡張/縮約 (ChangeOp 適用時の h dimension 整合)
# ---------------------------------------------------------------------------


def resize_hidden(h: np.ndarray, op: GraphChangeOp, removed_node_id: int = -1) -> np.ndarray:
    """ChangeOp 適用後の graph に合わせて hidden h を再整形.

    add_node: 新 node の hidden = 0 (sound 上界、|h|=0 から開始)
    remove_node: 該当 node の hidden を削除
    add_edge / remove_edge: h は不変 (n_nodes 不変)
    """
    if op.op_type == "add_node":
        new_h = np.zeros((h.shape[0] + 1, h.shape[1]))
        new_h[: h.shape[0]] = h
        return new_h
    if op.op_type == "remove_node":
        node_id = op.target[0] if removed_node_id < 0 else removed_node_id
        if node_id < 0 or node_id >= h.shape[0]:
            return h
        return np.delete(h, node_id, axis=0)
    # add/remove edge: h 不変
    return h


__all__ = [
    "GRAPH_OP_TYPES",
    "N_MIN",
    "N_MAX",
    "N_INIT",
    "HIDDEN_DIM",
    "MAX_SEQ_LEN",
    "GraphChangeOp",
    "GraphChangeOpSequence",
    "DynamicGraph",
    "DynamicGnnGene",
    "make_ring",
    "apply_changeop",
    "apply_sequence",
    "aggregate",
    "update_node",
    "forward_layer",
    "forward_stack",
    "variance_across_nodes",
    "resize_hidden",
]
