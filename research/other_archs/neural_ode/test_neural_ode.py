# SPDX-License-Identifier: Apache-2.0
"""pytest tests for Neural ODE PoC (gene API + verifier API + integration smoke + each gate).

Run from llcore project root::

    py -3.11 -m pytest research/other_archs/neural_ode/test_neural_ode.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# llcore.src を path に
_PROJ_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJ_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from research.other_archs.neural_ode.ode_gene import (  # noqa: E402
    A_HIGH,
    A_LOW,
    B_HIGH,
    B_LOW,
    DEFAULT_DIM,
    NeuralODEGene,
    W_HIGH,
    W_LOW,
    empirical_lipschitz,
    forward_euler,
    vector_field,
)
from research.other_archs.neural_ode.ode_verifier import (  # noqa: E402
    is_z3_available,
    verify_gene_hurwitz,
    verify_gene_lipschitz,
    verify_gene_ode_safe,
    verify_hurwitz_universal,
    verify_lipschitz_bound,
)
from research.other_archs.neural_ode.poc import (  # noqa: E402
    fitness_stability,
    gate_g1_lipschitz_invariant_universal,
    gate_g2_hurwitz_per_gene,
    gate_g3_best_fitness_monotonic,
    gate_g4_lineage_diversity,
    gate_g5_a_new_active,
    gate_g6_lipschitz_improves,
    gate_g7_verifier_latency,
    gate_g8_euler_vs_analytic_lipschitz,
    run_neural_ode_evolution,
)


# ---------------------------------------------------------------------------
# Gene API
# ---------------------------------------------------------------------------


class TestNeuralODEGene:
    def test_creation_and_clipping(self):
        # 範囲外でも clipped で clip 範囲に収まる
        g = NeuralODEGene(A=10.0, W=-5.0, b=100.0)
        c = g.clipped()
        assert A_LOW <= c.A <= A_HIGH
        assert W_LOW <= c.W <= W_HIGH
        assert B_LOW <= c.b <= B_HIGH

    def test_array_roundtrip(self):
        g = NeuralODEGene(A=-1.0, W=0.5, b=1.2)
        arr = g.as_array()
        assert arr.shape == (3,)
        g2 = NeuralODEGene.from_array(arr)
        assert g2.A == g.A and g2.W == g.W and g2.b == g.b

    def test_from_array_invalid_shape(self):
        with pytest.raises(ValueError):
            NeuralODEGene.from_array([1.0, 2.0])

    def test_analytic_lipschitz_upper(self):
        # |A|+|W||b|
        g = NeuralODEGene(A=-1.0, W=0.5, b=1.0)
        assert g.analytic_lipschitz_upper() == pytest.approx(1.0 + 0.5 * 1.0)
        # clip 範囲最大: |A|=2, |W|=1, |b|=2 → 4
        g_max = NeuralODEGene(A=-2.0, W=1.0, b=-2.0)
        assert g_max.analytic_lipschitz_upper() == pytest.approx(4.0)

    def test_hurwitz_test_sign(self):
        # stable: A + W*b < 0
        assert NeuralODEGene(A=-1.0, W=0.5, b=1.0).hurwitz_test() == pytest.approx(-0.5)
        # unstable: A + W*b > 0
        assert NeuralODEGene(A=-0.1, W=1.0, b=2.0).hurwitz_test() == pytest.approx(1.9)


class TestVectorFieldAndIntegration:
    def test_vector_field_origin_is_zero(self):
        # f(0) = A*0 + W*tanh(0) = 0 (平衡点 x=0)
        g = NeuralODEGene(A=-1.0, W=0.5, b=1.0)
        x = np.zeros(DEFAULT_DIM)
        f = vector_field(g, x)
        assert np.allclose(f, 0.0)

    def test_forward_euler_shape(self):
        g = NeuralODEGene(A=-1.0, W=0.3, b=1.0)
        x0 = np.array([0.5, -0.3, 0.2, 0.1])
        traj = forward_euler(g, x0, T=2.0, N=200)
        assert traj.shape == (201, 4)
        assert np.allclose(traj[0], x0)
        assert np.all(np.isfinite(traj))

    def test_forward_euler_stable_converges_near_zero(self):
        # Hurwitz 安定 + 強い decay → final norm が小さい
        g = NeuralODEGene(A=-2.0, W=0.0, b=0.0)
        x0 = np.array([1.0, -0.5, 0.3, 0.1])
        traj = forward_euler(g, x0, T=4.0, N=400)
        final_norm = float(np.linalg.norm(traj[-1]))
        # exp(-2*4) = 3.4e-4 で十分小さい
        assert final_norm < 0.01

    def test_forward_euler_unstable_grows(self):
        # A=0, W>0, b>0 → 不安定 (A+Wb=2>0 で x=0 から離れる)
        g = NeuralODEGene(A=0.0, W=1.0, b=2.0)
        x0 = np.array([0.01, 0.01, 0.01, 0.01])
        traj = forward_euler(g, x0, T=2.0, N=200)
        assert np.all(np.isfinite(traj))
        # norm が増える方向 (unstable)
        assert float(np.linalg.norm(traj[-1])) > float(np.linalg.norm(traj[0]))

    def test_empirical_lipschitz_under_analytic(self):
        # empirical <= analytic (sound 上界)
        g = NeuralODEGene(A=-1.0, W=0.5, b=1.0)
        emp = empirical_lipschitz(g, n_samples=32, rng=np.random.default_rng(1))
        ana = g.analytic_lipschitz_upper()
        assert emp <= ana + 1e-6


# ---------------------------------------------------------------------------
# Verifier API
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3 not installed")
class TestVerifier:
    def test_lipschitz_universal_L4_unsat(self):
        # |A|+|W||b| <= 4 は clip 範囲全域で成立 (universal upper)
        r = verify_lipschitz_bound(L=4.0)
        assert r.ok is True
        assert r.used_z3 is True

    def test_lipschitz_universal_L2_sat(self):
        # |A|+|W||b| <= 2 は反例あり (universal でない)
        r = verify_lipschitz_bound(L=2.0)
        assert r.ok is False
        assert r.counterexample is not None
        ce = r.counterexample
        # 反例の (A,W,b) は clip 範囲内 + |A|+|W||b| > 2
        assert A_LOW - 1e-6 <= ce["A"] <= A_HIGH + 1e-6
        assert W_LOW - 1e-6 <= ce["W"] <= W_HIGH + 1e-6
        assert B_LOW - 1e-6 <= ce["b"] <= B_HIGH + 1e-6
        assert ce["lipschitz_value"] > 2.0 - 1e-6

    def test_verify_gene_lipschitz_admit(self):
        # 安全 gene: L=4 以下 admit
        g = NeuralODEGene(A=-1.0, W=0.5, b=1.0)
        r = verify_gene_lipschitz(g, L=4.0)
        assert r.ok is True

    def test_verify_gene_lipschitz_reject_tighter_bound(self):
        # max gene: |A|+|W||b| = 4. L=1 では reject
        g_max = NeuralODEGene(A=-2.0, W=1.0, b=-2.0)
        r = verify_gene_lipschitz(g_max, L=1.0)
        assert r.ok is False

    def test_verify_gene_hurwitz_stable(self):
        # A + W*b = -0.5 < 0
        g = NeuralODEGene(A=-1.0, W=0.5, b=1.0)
        r = verify_gene_hurwitz(g)
        assert r.ok is True

    def test_verify_gene_hurwitz_unstable(self):
        # A + W*b = 1.9 > 0 → reject
        g = NeuralODEGene(A=-0.1, W=1.0, b=2.0)
        r = verify_gene_hurwitz(g)
        assert r.ok is False

    def test_verify_gene_ode_safe_AND_gate(self):
        # Lipschitz pass ∧ Hurwitz pass = admit
        g = NeuralODEGene(A=-1.5, W=0.3, b=0.5)
        r = verify_gene_ode_safe(g, L=4.0)
        assert r.ok is True

    def test_verify_hurwitz_universal_finds_unstable(self):
        # clip 範囲は不安定 gene を含む → sat (ok=False = "not universally stable")
        r = verify_hurwitz_universal()
        # sat 期待
        assert r.ok is False
        assert r.counterexample is not None


# ---------------------------------------------------------------------------
# Fitness & evolution loop smoke
# ---------------------------------------------------------------------------


class TestFitnessAndEvolution:
    def test_fitness_stability_in_unit_range(self):
        g = NeuralODEGene(A=-1.0, W=0.3, b=1.0)
        rng = np.random.default_rng(42)
        fit = fitness_stability(g, rng=rng)
        assert 0.0 <= fit <= 1.0

    def test_fitness_stability_stable_higher_than_unstable(self):
        # stable + low Lipschitz の方が high fitness
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        g_stable_low = NeuralODEGene(A=-1.0, W=0.0, b=0.0)  # L=1, A+Wb=-1
        g_unstable_high = NeuralODEGene(A=-0.1, W=1.0, b=2.0)  # L=2.1, A+Wb=1.9 不安定
        f_stable = fitness_stability(g_stable_low, rng=rng_a)
        f_unstable = fitness_stability(g_unstable_high, rng=rng_b)
        assert f_stable > f_unstable

    def test_evolution_smoke_small(self):
        # 小スケールで進化が動作することを smoke test
        rng = np.random.default_rng(123)
        trace = run_neural_ode_evolution(
            pop_size=16,
            n_lineages=4,
            n_generations=5,
            L_bound=4.0,
            mutation_sigma=0.1,
            rng=rng,
        )
        # 6 世代 (初期 + 5)
        assert len(trace.populations) == 6
        assert len(trace.best_fitness_curve) == 6
        assert all(len(p) == 16 for p in trace.populations)


# ---------------------------------------------------------------------------
# Gate G1-G8 individual integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evolution_trace():
    """共有 fixture: 64×50 進化を 1 度だけ走らせて全ゲートで使用 (cost 削減)."""
    rng = np.random.default_rng(20260529)
    trace = run_neural_ode_evolution(
        pop_size=64,
        n_lineages=8,
        n_generations=50,
        L_bound=4.0,
        mutation_sigma=0.1,
        crossover_rate=0.5,
        floor_percentile=30.0,
        rng=rng,
    )
    return trace


@pytest.mark.skipif(not is_z3_available(), reason="z3 not installed")
class TestGates:
    def test_g1_lipschitz_invariant_universal(self):
        ok, detail = gate_g1_lipschitz_invariant_universal()
        assert ok, f"G1 failed: {detail}"

    def test_g2_hurwitz_per_gene(self):
        ok, detail = gate_g2_hurwitz_per_gene()
        assert ok, f"G2 failed: {detail}"

    def test_g3_best_fitness_monotonic(self, evolution_trace):
        ok, detail = gate_g3_best_fitness_monotonic(evolution_trace)
        assert ok, f"G3 failed: {detail}"

    def test_g4_lineage_diversity(self, evolution_trace):
        ok, detail = gate_g4_lineage_diversity(evolution_trace, min_lineages=6)
        assert ok, f"G4 failed: {detail}"

    def test_g5_a_new_active(self, evolution_trace):
        ok, detail = gate_g5_a_new_active(evolution_trace)
        assert ok, f"G5 failed: {detail}"

    def test_g6_lipschitz_improves(self, evolution_trace):
        ok, detail = gate_g6_lipschitz_improves(evolution_trace)
        assert ok, f"G6 failed: {detail}"

    def test_g7_verifier_latency(self, evolution_trace):
        ok, detail = gate_g7_verifier_latency(evolution_trace, threshold_ms=10.0)
        assert ok, f"G7 failed: {detail}"

    def test_g8_euler_vs_analytic_lipschitz(self):
        ok, detail = gate_g8_euler_vs_analytic_lipschitz()
        assert ok, f"G8 failed: {detail}"
