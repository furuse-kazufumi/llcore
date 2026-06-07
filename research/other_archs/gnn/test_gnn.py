# SPDX-License-Identifier: Apache-2.0
"""GNN PoC pytest battery (10+ tests).

研究 PoC 用 minimal test。falsifiable G1-G8 の数値的足場を確認する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# import path 整備 (pytest 実行時に research/other_archs を解決可能に)
_PROJ_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJ_ROOT / "src"
_RESEARCH = _PROJ_ROOT / "research" / "other_archs"
for p in (_SRC, _RESEARCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gnn.gnn_gene import (
    GnnGene,
    aggregate,
    forward_layer,
    forward_stack,
    update_node,
    variance_across_nodes,
)
from gnn.gnn_verifier import (
    is_z3_available,
    verify_equivariance_structure,
    verify_oversmoothing_lower_bound,
)


# ---------------------------------------------------------------------------
# Gene basics
# ---------------------------------------------------------------------------


def test_gene_simplex_projection_normalizes_alpha():
    """alpha 正規化が simplex に投影される."""
    g = GnnGene(alpha_sum=2.0, alpha_mean=2.0, alpha_max=2.0, W=0.5, U=0.5).clipped()
    assert abs(g.alpha_sum + g.alpha_mean + g.alpha_max - 1.0) < 1e-9
    assert g.alpha_sum > 0 and g.alpha_mean > 0 and g.alpha_max > 0


def test_gene_simplex_handles_negative_alpha():
    """負 alpha は 0 に clip され simplex 正規化される."""
    g = GnnGene(alpha_sum=-0.5, alpha_mean=0.5, alpha_max=0.5, W=0.0, U=0.0).clipped()
    assert g.alpha_sum == 0.0
    assert abs(g.alpha_sum + g.alpha_mean + g.alpha_max - 1.0) < 1e-9


def test_gene_W_U_clipped_to_unit_interval():
    """W, U が [-1, 1] に clip される."""
    g = GnnGene(alpha_sum=0.5, alpha_mean=0.3, alpha_max=0.2, W=5.0, U=-5.0).clipped()
    assert g.W == 1.0
    assert g.U == -1.0


def test_gene_array_roundtrip():
    """as_array / from_array の roundtrip 整合性."""
    g0 = GnnGene(0.3, 0.3, 0.4, 0.5, -0.5).clipped()
    g1 = GnnGene.from_array(g0.as_array()).clipped()
    np.testing.assert_allclose(g0.as_array(), g1.as_array(), atol=1e-9)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def test_aggregate_convex_combination():
    """aggregate が 3 op の凸結合になっている."""
    g = GnnGene(0.5, 0.3, 0.2, 0.0, 1.0)
    h = np.array([[1.0, 0.0], [3.0, 0.0]])  # 2 近傍
    agg = aggregate(g, h)
    expected = 0.5 * np.array([4.0, 0.0]) + 0.3 * np.array([2.0, 0.0]) + 0.2 * np.array([3.0, 0.0])
    np.testing.assert_allclose(agg, expected, atol=1e-9)


def test_forward_layer_permutation_equivariant():
    """forward_layer は ring topology の permutation で equivariant.

    ring topology 全体を回転 (cyclic shift) しても出力も同じ shift で対応する.
    """
    rng = np.random.default_rng(42)
    g = GnnGene(0.3, 0.3, 0.4, 0.5, 0.5).clipped()
    h = rng.normal(0, 1, size=(8, 4))
    out1 = forward_layer(g, h)
    # 1 つ右にシフト (ring rotation = permutation)
    shifted = np.roll(h, 1, axis=0)
    out2 = forward_layer(g, shifted)
    # equivariance: forward(shifted) == shift(forward(input))
    np.testing.assert_allclose(out2, np.roll(out1, 1, axis=0), atol=1e-9)


def test_forward_layer_output_bounded_by_tanh():
    """forward_layer 出力は |h| <= 1 (tanh)."""
    rng = np.random.default_rng(0)
    g = GnnGene(0.4, 0.3, 0.3, 1.0, 1.0).clipped()
    h = rng.normal(0, 10, size=(8, 4))  # 大きい入力
    out = forward_layer(g, h)
    assert np.all(np.abs(out) <= 1.0 + 1e-9)


def test_forward_stack_smoothing_kills_variance():
    """強い smoothing (mean dominant + small W) で variance が深層で激減する."""
    rng = np.random.default_rng(7)
    g = GnnGene(0.0, 1.0, 0.0, 0.1, 0.1).clipped()  # mean dominant, small magnitudes
    h0 = rng.normal(0, 1, size=(8, 4))
    var0 = variance_across_nodes(h0)
    h_deep = forward_stack(g, h0, n_layers=16)
    var_deep = variance_across_nodes(h_deep)
    assert var_deep < var0 * 0.5


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_z3_available_or_fallback():
    """Z3 が available または mock fallback で確認できる."""
    flag = is_z3_available()
    assert isinstance(flag, bool)


def test_verify_oversmoothing_good_gene_pass():
    """good gene は over-smoothing invariant pass."""
    g = GnnGene(alpha_sum=0.5, alpha_mean=0.3, alpha_max=0.2, W=0.9, U=0.7)
    r = verify_oversmoothing_lower_bound(g)
    assert r.ok is True


def test_verify_oversmoothing_bad_gene_fail():
    """smoothing 強すぎる gene は invariant 違反 (反例 sat)."""
    g = GnnGene(alpha_sum=0.0, alpha_mean=1.0, alpha_max=0.0, W=0.1, U=0.1)
    r = verify_oversmoothing_lower_bound(g)
    assert r.ok is False
    if r.used_z3:
        assert r.counterexample is not None


def test_verify_equivariance_clipped_gene_pass():
    """clipped gene は simplex 内 → equivariance 構造保証."""
    g = GnnGene(alpha_sum=0.5, alpha_mean=0.3, alpha_max=0.2, W=0.5, U=0.5).clipped()
    r = verify_equivariance_structure(g)
    assert r.ok is True


def test_verify_oversmoothing_sat_unsat_separation():
    """G1 と同じ: 複数 good/bad gene を投入して結果が分離する."""
    good = [
        GnnGene(0.4, 0.3, 0.3, 0.8, 0.8),
        GnnGene(0.5, 0.25, 0.25, 0.9, 0.7),
    ]
    bad = [
        GnnGene(0.0, 1.0, 0.0, 0.1, 0.1),
        GnnGene(0.0, 0.5, 0.5, 0.0, 0.2),
    ]
    for g in good:
        assert verify_oversmoothing_lower_bound(g).ok, f"good {g} should pass"
    for g in bad:
        assert not verify_oversmoothing_lower_bound(g).ok, f"bad {g} should fail"


def test_verify_latency_under_threshold():
    """Z3 verify latency が 10ms 未満 (mean of 50 calls)."""
    import time
    rng = np.random.default_rng(11)
    lats = []
    for _ in range(50):
        # ランダム gene
        arr = np.array([
            rng.uniform(0, 1),
            rng.uniform(0, 1),
            rng.uniform(0, 1),
            rng.uniform(-1, 1),
            rng.uniform(-1, 1),
        ])
        g = GnnGene.from_array(arr).clipped()
        t0 = time.perf_counter()
        verify_oversmoothing_lower_bound(g, timeout_ms=200)
        lats.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(np.mean(lats))
    assert mean_ms < 10.0, f"mean latency {mean_ms:.2f}ms exceeds 10ms"


# ---------------------------------------------------------------------------
# Integration smoke
# ---------------------------------------------------------------------------


def test_evolution_smoke_short_run():
    """進化器 minimal smoke: 4 個体 × 5 世代で破綻なく完走."""
    from gnn.poc import run_evolution

    rng = np.random.default_rng(99)
    trace = run_evolution(
        n_lineages=4,
        pop_per_lineage=1,  # 4 lineage × 1 = 4 個体
        n_generations=5,
        rng=rng,
        mutation_sigma=0.1,
        crossover_rate=0.5,
        floor_percentile=30.0,
        use_verifier_gate=True,
    )
    assert len(trace.populations) == 6  # 初期 + 5 世代
    assert len(trace.best_fitness_curve) == 6
    # best fitness が NaN でない
    assert all(not np.isnan(f) for f in trace.best_fitness_curve)


def test_evolution_short_fitness_monotonic_ratchet():
    """5 世代 smoke で best fitness が ratchet で単調非減少."""
    from gnn.poc import run_evolution

    rng = np.random.default_rng(101)
    trace = run_evolution(
        n_lineages=4,
        pop_per_lineage=2,
        n_generations=5,
        rng=rng,
    )
    curve = trace.best_fitness_curve
    for i in range(len(curve) - 1):
        assert curve[i + 1] >= curve[i] - 1e-9, (
            f"ratchet broken at gen {i}: {curve[i]:.4f} -> {curve[i+1]:.4f}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
