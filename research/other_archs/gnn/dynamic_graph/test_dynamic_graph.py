# SPDX-License-Identifier: Apache-2.0
"""Dynamic GNN (Stage 2) pytest battery (10+ tests).

固定 ring PoC の test_gnn.py と独立。研究 PoC 用 minimal test。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# import path 整備
_PROJ_ROOT = Path(__file__).resolve().parents[4]
_SRC = _PROJ_ROOT / "src"
_RESEARCH = _PROJ_ROOT / "research" / "other_archs"
for p in (_SRC, _RESEARCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gnn.dynamic_graph.dgnn_gene import (
    GRAPH_OP_TYPES,
    HIDDEN_DIM,
    N_INIT,
    N_MAX,
    N_MIN,
    DynamicGnnGene,
    DynamicGraph,
    GraphChangeOp,
    GraphChangeOpSequence,
    apply_changeop,
    apply_sequence,
    forward_layer,
    forward_stack,
    make_ring,
    resize_hidden,
    variance_across_nodes,
)
from gnn.dynamic_graph.dgnn_verifier import (
    E_GRAPH,
    K_GRAPH,
    epsilon_for_graph_op,
    epsilon_for_seq,
    is_z3_available,
    shrink_upper_numeric,
    verify_equivariance_dynamic,
    verify_oversmoothing_dynamic,
    verify_refinement_single_graph_op,
    verify_seq_refinement_chain,
)


# ---------------------------------------------------------------------------
# DynamicGraph / make_ring basics
# ---------------------------------------------------------------------------


def test_make_ring_topology():
    g = make_ring(8)
    assert g.n_nodes == 8
    assert g.n_edges() == 8
    assert g.max_degree() == 2
    assert g.mean_degree() == 2.0
    for v in range(8):
        assert (v - 1) % 8 in g.adjacency[v]
        assert (v + 1) % 8 in g.adjacency[v]


def test_make_ring_rejects_too_small():
    with pytest.raises(ValueError):
        make_ring(2)


def test_dynamic_graph_adjacency_size_mismatch_rejected():
    with pytest.raises(ValueError):
        DynamicGraph(n_nodes=4, adjacency=((1, 2),))  # 1-tuple but 4 nodes


# ---------------------------------------------------------------------------
# GraphChangeOp validation
# ---------------------------------------------------------------------------


def test_graph_changeop_validates_op_type():
    with pytest.raises(ValueError):
        GraphChangeOp(op_type="invalid_op", target=())


def test_graph_changeop_validates_target_arity():
    # remove_node needs 1 target
    with pytest.raises(ValueError):
        GraphChangeOp(op_type="remove_node", target=())
    with pytest.raises(ValueError):
        GraphChangeOp(op_type="remove_node", target=(0, 1))
    # add_edge needs 2 targets
    with pytest.raises(ValueError):
        GraphChangeOp(op_type="add_edge", target=(0,))
    # add_node needs no target
    with pytest.raises(ValueError):
        GraphChangeOp(op_type="add_node", target=(0,))


def test_graph_changeop_magnitude_per_type():
    assert GraphChangeOp("add_node", ()).magnitude() == pytest.approx(0.10)
    assert GraphChangeOp("remove_node", (0,)).magnitude() == pytest.approx(0.20)
    assert GraphChangeOp("add_edge", (0, 1)).magnitude() == pytest.approx(0.05)
    assert GraphChangeOp("remove_edge", (0, 1)).magnitude() == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# apply_changeop / apply_sequence
# ---------------------------------------------------------------------------


def test_apply_add_node_increases_n_nodes():
    g = make_ring(8)
    g2 = apply_changeop(g, GraphChangeOp("add_node", ()))
    assert g2.n_nodes == 9
    # 新 node は末尾 (id=8)、最後と最初を接続
    assert 8 in g2.adjacency[7]
    assert 8 in g2.adjacency[0]
    # 新 node の近傍 = (0, 7)
    assert set(g2.adjacency[8]) == {0, 7}


def test_apply_add_node_respects_N_MAX():
    g = make_ring(N_MAX)
    g2 = apply_changeop(g, GraphChangeOp("add_node", ()))
    assert g2.n_nodes == N_MAX  # no-op (bound 違反)


def test_apply_remove_node_decreases_and_maintains_connectivity():
    g = make_ring(8)
    g2 = apply_changeop(g, GraphChangeOp("remove_node", (3,)))
    assert g2.n_nodes == 7
    # 連結性維持: 削除前 3 の近傍 (2, 4) は新 id (2, 3) で接続
    # remap: 3 削除 → 4 → 3
    # 2 と 3 (元 4) は隣接 になるはず (cycle 補修)
    assert 3 in g2.adjacency[2] or 2 in g2.adjacency[3]


def test_apply_remove_node_respects_N_MIN():
    g = make_ring(N_MIN)
    g2 = apply_changeop(g, GraphChangeOp("remove_node", (0,)))
    assert g2.n_nodes == N_MIN  # no-op (bound 違反)


def test_apply_add_edge_creates_new_edge():
    g = make_ring(8)
    # ring で 0-4 は隣接していない
    assert 4 not in g.adjacency[0]
    g2 = apply_changeop(g, GraphChangeOp("add_edge", (0, 4)))
    assert 4 in g2.adjacency[0]
    assert 0 in g2.adjacency[4]
    # n_nodes 不変
    assert g2.n_nodes == 8


def test_apply_remove_edge_removes_edge():
    g = make_ring(8)
    # 0-1 隣接、両者 degree=2 (ring) → remove で degree=1 だが連結性は維持される
    assert 1 in g.adjacency[0]
    g2 = apply_changeop(g, GraphChangeOp("remove_edge", (0, 1)))
    # ring の隣接 edge を抜くと 0 と 1 の degree=1 になるが本実装の保守ガードは
    # **削除前** に degree <= 1 だけ拒否する (= ring (deg=2) なら通る)。
    # よって edge は実際に削除される。
    assert 1 not in g2.adjacency[0]
    assert 0 not in g2.adjacency[1]


def test_apply_remove_edge_protects_degree_one_node():
    """degree=1 の node を孤立化させる remove は no-op."""
    g = make_ring(8)
    # 0-1 を remove (degree=2→1) → さらに 0-7 を remove しようとすると 0 が degree=0 になる
    g2 = apply_changeop(g, GraphChangeOp("remove_edge", (0, 1)))
    # 0 の degree=1 (近傍は 7 のみ)
    assert len(g2.adjacency[0]) == 1
    # remove (0, 7) は 0 を孤立させるため no-op
    g3 = apply_changeop(g2, GraphChangeOp("remove_edge", (0, 7)))
    assert 7 in g3.adjacency[0]  # 保護で残る


def test_apply_remove_edge_succeeds_when_safe():
    # 0 と 4 を add_edge してから remove_edge: 安全 (degree が下がっても孤立しない)
    g = make_ring(8)
    g2 = apply_changeop(g, GraphChangeOp("add_edge", (0, 4)))
    # 0 と 4 の degree=3
    g3 = apply_changeop(g2, GraphChangeOp("remove_edge", (0, 4)))
    assert 4 not in g3.adjacency[0]
    assert 0 not in g3.adjacency[4]


def test_apply_sequence_composes():
    g = make_ring(8)
    seq = GraphChangeOpSequence(ops=(
        GraphChangeOp("add_node", ()),
        GraphChangeOp("add_node", ()),
        GraphChangeOp("add_edge", (0, 4)),
        GraphChangeOp("remove_node", (1,)),
    ))
    g_final = apply_sequence(g, seq)
    # 2 add + 1 remove = +1
    assert g_final.n_nodes == 9
    # N range 維持
    assert N_MIN <= g_final.n_nodes <= N_MAX


# ---------------------------------------------------------------------------
# DynamicGnnGene
# ---------------------------------------------------------------------------


def test_gene_clipped_normalizes_alpha():
    gene = DynamicGnnGene(2.0, 2.0, 2.0, 0.5, 0.5).clipped()
    assert abs(gene.alpha_sum + gene.alpha_mean + gene.alpha_max - 1.0) < 1e-9


def test_gene_clipped_clips_W_U():
    gene = DynamicGnnGene(0.5, 0.3, 0.2, 5.0, -5.0).clipped()
    assert gene.W == 1.0
    assert gene.U == -1.0


def test_gene_keeps_changeop_seq_through_clip():
    seq = GraphChangeOpSequence(ops=(GraphChangeOp("add_node", ()),))
    gene = DynamicGnnGene(0.5, 0.3, 0.2, 0.5, 0.5, seq).clipped()
    assert gene.changeop_seq is seq or gene.changeop_seq.ops == seq.ops


# ---------------------------------------------------------------------------
# Forward (dynamic graph 対応)
# ---------------------------------------------------------------------------


def test_forward_layer_consistent_with_graph_size():
    g = make_ring(N_MIN)
    gene = DynamicGnnGene(0.5, 0.3, 0.2, 0.5, 0.5).clipped()
    h = np.zeros((N_MIN, HIDDEN_DIM))
    h[0] = 1.0
    h_new = forward_layer(gene, h, g)
    assert h_new.shape == (N_MIN, HIDDEN_DIM)
    assert np.all(np.abs(h_new) <= 1.0 + 1e-9)


def test_forward_layer_rejects_dimension_mismatch():
    g = make_ring(8)
    gene = DynamicGnnGene(0.5, 0.3, 0.2, 0.5, 0.5).clipped()
    h_wrong = np.zeros((5, HIDDEN_DIM))  # wrong n_nodes
    with pytest.raises(ValueError):
        forward_layer(gene, h_wrong, g)


def test_forward_stack_after_apply_sequence():
    """ChangeOp seq 適用後 graph で forward が正常動作."""
    g = make_ring(N_INIT)
    seq = GraphChangeOpSequence(ops=(
        GraphChangeOp("add_node", ()),
        GraphChangeOp("add_edge", (0, 4)),
    ))
    g_final = apply_sequence(g, seq)
    gene = DynamicGnnGene(0.4, 0.3, 0.3, 0.4, 0.3).clipped()
    h0 = np.zeros((g_final.n_nodes, HIDDEN_DIM))
    h0[0] = np.array([1.0, 0, 0, 0])
    hL = forward_stack(gene, h0, g_final, n_layers=4)
    assert hL.shape == (g_final.n_nodes, HIDDEN_DIM)
    # 信号が広がっている: variance > 0
    assert variance_across_nodes(hL) > 0


def test_resize_hidden_add_node_pads_zero():
    h = np.ones((8, HIDDEN_DIM))
    op = GraphChangeOp("add_node", ())
    h_new = resize_hidden(h, op)
    assert h_new.shape == (9, HIDDEN_DIM)
    assert np.all(h_new[8] == 0.0)


def test_resize_hidden_remove_node_drops_row():
    h = np.arange(8 * HIDDEN_DIM, dtype=np.float64).reshape(8, HIDDEN_DIM)
    op = GraphChangeOp("remove_node", (3,))
    h_new = resize_hidden(h, op)
    assert h_new.shape == (7, HIDDEN_DIM)
    # row 3 が消えている
    assert not np.allclose(h_new[3], h[3])


# ---------------------------------------------------------------------------
# Verifier — over-smoothing dynamic
# ---------------------------------------------------------------------------


def test_z3_available_or_fallback():
    flag = is_z3_available()
    assert isinstance(flag, bool)


def test_oversmoothing_good_gene_passes_at_multiple_N():
    gene = DynamicGnnGene(0.5, 0.3, 0.2, 0.7, 0.5).clipped()
    for N in (N_MIN, N_INIT, N_MAX):
        graph = make_ring(N)
        r = verify_oversmoothing_dynamic(gene, graph)
        assert r.ok, f"good gene should pass at N={N}: {r.reason}"


def test_oversmoothing_bad_gene_fails():
    gene = DynamicGnnGene(0.0, 1.0, 0.0, 0.1, 0.1).clipped()
    graph = make_ring(N_INIT)
    r = verify_oversmoothing_dynamic(gene, graph)
    assert not r.ok, f"smoothing-dominant gene should fail: {r.reason}"


def test_shrink_upper_numeric_matches_formula():
    gene = DynamicGnnGene(0.4, 0.3, 0.3, 0.5, 0.5).clipped()
    # K_max=2 (ring)
    agg = 0.4 * 2 + 0.3 + 0.3  # = 1.4
    expected = (abs(0.5) + abs(0.5) * 1.4) ** 2  # = 1.44
    assert shrink_upper_numeric(gene, 2) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Verifier — refinement chain
# ---------------------------------------------------------------------------


def test_epsilon_additivity():
    op1 = GraphChangeOp("add_node", ())  # mag 0.1
    op2 = GraphChangeOp("add_edge", (0, 1))  # mag 0.05
    e1 = epsilon_for_graph_op(op1)
    e2 = epsilon_for_graph_op(op2)
    seq = GraphChangeOpSequence(ops=(op1, op2))
    assert epsilon_for_seq(seq) == pytest.approx(e1 + e2, abs=1e-9)


def test_refinement_single_low_amp_gene_admits():
    """低 amplification gene は refinement bound 内 → admit."""
    gene = DynamicGnnGene(0.3, 0.3, 0.4, 0.3, 0.3).clipped()
    g = make_ring(N_INIT)
    op = GraphChangeOp("add_node", ())
    r = verify_refinement_single_graph_op(gene, g, op)
    assert r.ok, f"low-amp gene should admit add_node: {r.reason}"


def test_refinement_single_high_amp_gene_rejected():
    """高 amplification gene は refinement bound 外 → reject."""
    gene = DynamicGnnGene(0.8, 0.1, 0.1, 1.0, 1.0).clipped()
    g = make_ring(N_INIT)
    op = GraphChangeOp("add_node", ())
    r = verify_refinement_single_graph_op(gene, g, op)
    assert not r.ok, f"high-amp gene should fail: {r.reason}"
    if r.used_z3:
        assert r.counterexample is not None


def test_refinement_chain_admits_compatible_gene_seq():
    """良 gene + 10 step seq は全 step admit."""
    gene = DynamicGnnGene(0.3, 0.3, 0.4, 0.3, 0.3).clipped()
    g = make_ring(N_INIT)
    seq = GraphChangeOpSequence(ops=tuple(
        [GraphChangeOp("add_node", ()) if i % 4 == 0
         else GraphChangeOp("remove_node", (1,)) if i % 4 == 1
         else GraphChangeOp("add_edge", (0, 2)) if i % 4 == 2
         else GraphChangeOp("remove_edge", (0, 2))
         for i in range(10)]
    ))
    r = verify_seq_refinement_chain(gene, g, seq)
    assert r.ok
    assert r.passed_steps == 10
    assert r.epsilon_total == pytest.approx(epsilon_for_seq(seq), abs=1e-9)


def test_refinement_chain_rejects_amplifying_gene():
    """高 amp gene は chain 途中で reject."""
    gene = DynamicGnnGene(0.7, 0.15, 0.15, 1.0, 1.0).clipped()
    g = make_ring(N_INIT)
    seq = GraphChangeOpSequence(ops=(GraphChangeOp("add_node", ()),) * 5)
    r = verify_seq_refinement_chain(gene, g, seq)
    assert not r.ok
    assert r.first_failure is not None


# ---------------------------------------------------------------------------
# Verifier — equivariance
# ---------------------------------------------------------------------------


def test_equivariance_clipped_gene_passes():
    gene = DynamicGnnGene(0.5, 0.3, 0.2, 0.5, 0.5).clipped()
    r = verify_equivariance_dynamic(gene)
    assert r.ok


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def test_z3_latency_under_threshold():
    import time
    rng = np.random.default_rng(11)
    lats = []
    for _ in range(30):
        arr = np.array([
            rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1),
            rng.uniform(-1, 1), rng.uniform(-1, 1),
        ])
        gene = DynamicGnnGene(*arr).clipped()
        g = make_ring(int(rng.integers(N_MIN, N_MAX + 1)))
        t0 = time.perf_counter()
        verify_oversmoothing_dynamic(gene, g, timeout_ms=200)
        lats.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(np.mean(lats))
    assert mean_ms < 15.0, f"mean latency {mean_ms:.2f}ms exceeds 15ms"


# ---------------------------------------------------------------------------
# Integration smoke
# ---------------------------------------------------------------------------


def test_evolution_smoke_short_run():
    """4 個体 × 3 世代 minimal smoke run."""
    from gnn.dynamic_graph.poc import run_evolution

    rng = np.random.default_rng(99)
    trace = run_evolution(
        n_lineages=4,
        pop_per_lineage=1,
        n_generations=3,
        rng=rng,
        mutation_sigma=0.1,
        crossover_rate=0.5,
        floor_percentile=30.0,
        use_verifier_gate=True,
    )
    assert len(trace.populations) == 4  # 初期 + 3 世代
    assert len(trace.best_fitness_curve) == 4
    assert all(not np.isnan(f) for f in trace.best_fitness_curve)


def test_evolution_short_fitness_monotonic_ratchet():
    """3 世代 smoke で best fitness が ratchet で単調非減少."""
    from gnn.dynamic_graph.poc import run_evolution

    rng = np.random.default_rng(101)
    trace = run_evolution(
        n_lineages=4,
        pop_per_lineage=2,
        n_generations=3,
        rng=rng,
    )
    curve = trace.best_fitness_curve
    for i in range(len(curve) - 1):
        assert curve[i + 1] >= curve[i] - 1e-9


# ---------------------------------------------------------------------------
# ChangeOp seq op_type counts (G8 多様性ベース)
# ---------------------------------------------------------------------------


def test_op_type_counts_signature():
    seq = GraphChangeOpSequence(ops=(
        GraphChangeOp("add_node", ()),
        GraphChangeOp("add_node", ()),
        GraphChangeOp("remove_node", (0,)),
        GraphChangeOp("add_edge", (0, 1)),
        GraphChangeOp("remove_edge", (0, 1)),
    ))
    counts = seq.op_type_counts()
    assert counts["add_node"] == 2
    assert counts["remove_node"] == 1
    assert counts["add_edge"] == 1
    assert counts["remove_edge"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
