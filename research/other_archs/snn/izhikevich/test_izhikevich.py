# SPDX-License-Identifier: Apache-2.0
"""Izhikevich SNN PoC pytest battery (Stage 2.3).

llcore approach の Izhikevich 移植 (IzhikevichGene + Z3 invariant + 進化) の
単体・統合テスト. LIF (Stage 2.2a) と同 pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# path setup (script 直叩き + pytest 両対応)
_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parents[3]
_SRC = _PROJ_ROOT / "src"
_RESEARCH = _PROJ_ROOT / "research"
for p in (_SRC, _RESEARCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from other_archs.snn.izhikevich.izh_gene import (  # noqa: E402
    A_MAX,
    A_MIN,
    B_MAX,
    B_MIN,
    C_MAX,
    C_MIN,
    D_MAX,
    D_MIN,
    DT,
    I_MAX_ABS,
    U_INIT,
    U_MAX,
    U_MIN,
    V_INIT,
    V_PEAK,
    V_PRE_MAX,
    V_PRE_MIN,
    IzhikevichGene,
    firing_rate_hz,
    make_constant_input,
    simulate_izh,
)
from other_archs.snn.izhikevich.izh_verifier import (  # noqa: E402
    is_z3_available,
    verify_firing_rate_per_gene,
    verify_v_bounded_global,
    verify_v_bounded_per_gene,
)


# ---------------------------------------------------------------------------
# Gene basic
# ---------------------------------------------------------------------------


class TestIzhikevichGene:
    def test_clipped_in_range(self):
        """範囲外 gene が clip 後に範囲内に収まる."""
        g = IzhikevichGene(a=10.0, b=5.0, c=-100.0, d=100.0).clipped()
        assert A_MIN <= g.a <= A_MAX
        assert B_MIN <= g.b <= B_MAX
        assert C_MIN <= g.c <= C_MAX
        assert D_MIN <= g.d <= D_MAX

    def test_clipped_below_threshold(self):
        """clip 後も c < V_PEAK - 5 (= 25) が成立."""
        g = IzhikevichGene(a=0.02, b=0.2, c=-50.0, d=8.0).clipped()
        assert g.c < V_PEAK - 5.0 + 1e-9  # c=-50 < 25

    def test_as_array_shape(self):
        g = IzhikevichGene(0.02, 0.2, -65.0, 8.0)
        arr = g.as_array()
        assert arr.shape == (4,)
        assert arr.dtype == np.float64

    def test_firing_type_guess_canonical(self):
        """Izhikevich 2003 canonical 4 type の type guess が正しい."""
        assert IzhikevichGene(0.02, 0.2, -65.0, 8.0).firing_type_guess() == "RS"
        assert IzhikevichGene(0.02, 0.2, -55.0, 4.0).firing_type_guess() == "IB"
        assert IzhikevichGene(0.02, 0.2, -50.0, 2.0).firing_type_guess() == "CH"
        assert IzhikevichGene(0.10, 0.2, -65.0, 2.0).firing_type_guess() == "FS"


# ---------------------------------------------------------------------------
# Simulator basic
# ---------------------------------------------------------------------------


class TestSimulateIzh:
    def test_no_input_stays_silent(self):
        """ゼロ入力なら発火なし (静止電位付近で停留)."""
        g = IzhikevichGene(0.02, 0.2, -65.0, 8.0)
        I = np.zeros(800)
        V, u, spikes = simulate_izh(g, I, T=200.0)
        assert len(spikes) == 0
        assert np.all(np.isfinite(V))
        assert np.all(np.isfinite(u))
        # V は V_INIT (-70) 近傍に収まる
        assert abs(V[-1] - V_INIT) < 5.0

    def test_constant_input_induces_spikes_rs(self):
        """RS で I=10 で発火が発生 (Izhikevich 2003 Fig. 1 で 約 30 Hz)."""
        g = IzhikevichGene(0.02, 0.2, -65.0, 8.0)
        I = make_constant_input(T=200.0, I_value=10.0)
        V, u, spikes = simulate_izh(g, I, T=200.0)
        assert len(spikes) > 0
        rate = firing_rate_hz(spikes, T=200.0)
        # RS で I=10 は 20-40 Hz 範囲
        assert 5.0 <= rate <= 60.0, f"RS rate {rate} out of expected range"

    def test_fs_faster_than_rs(self):
        """FS gene は RS より高 rate (同じ I で比較)."""
        rs = IzhikevichGene(0.02, 0.2, -65.0, 8.0)
        fs = IzhikevichGene(0.10, 0.2, -65.0, 2.0)
        I = make_constant_input(T=300.0, I_value=10.0)
        _, _, rs_spikes = simulate_izh(rs, I, T=300.0)
        _, _, fs_spikes = simulate_izh(fs, I, T=300.0)
        rs_rate = firing_rate_hz(rs_spikes, T=300.0)
        fs_rate = firing_rate_hz(fs_spikes, T=300.0)
        # FS は recovery が速いので RS より高 rate (Izhikevich 2003 Fig. 1)
        assert fs_rate > rs_rate, f"FS({fs_rate}) should be > RS({rs_rate})"

    def test_membrane_bounded_during_simulation(self):
        """V が pre-spike clip 範囲内 + spike 時 V_PEAK 近傍 + 直後 c へ reset.

        Izhikevich は spike 後 u += d で recovery 増加 → 次 step で hyperpolarize.
        d=8 (RS) で 1-step 後の v は c から数 mV 低下可能 (c=-65 → -74 程度まで).
        V_PRE_MIN=-80 が pre-spike 下界の clip 範囲, それを下回らないことを検査.
        """
        g = IzhikevichGene(0.02, 0.2, -65.0, 8.0)
        I = make_constant_input(T=200.0, I_value=10.0)
        V, u, spikes = simulate_izh(g, I, T=200.0)
        # V_trace は V_PRE_MIN を下回らない (Z3 invariant の下界整合)
        assert V.min() >= V_PRE_MIN - 1.0, (
            f"V.min()={V.min()} below V_PRE_MIN={V_PRE_MIN}"
        )
        # V_PEAK 越えは内部のみ、記録される V_trace は <= V_PEAK
        assert V.max() <= V_PEAK + 1e-6
        assert np.all(np.isfinite(V))
        assert np.all(np.isfinite(u))


# ---------------------------------------------------------------------------
# firing_rate_hz
# ---------------------------------------------------------------------------


def test_firing_rate_hz_basic():
    spikes = [10.0, 20.0, 30.0, 40.0, 50.0]
    rate = firing_rate_hz(spikes, T=100.0)
    assert abs(rate - 50.0) < 1e-9


def test_firing_rate_hz_empty():
    assert firing_rate_hz([], 100.0) == 0.0


# ---------------------------------------------------------------------------
# Verifier (Z3-dependent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
class TestIzhVerifier:
    def test_v_bounded_global_safe_contract_admit(self):
        """[Stage 2.3 G1] safe contract (I_max=5, margin=100) で admit (unsat)."""
        r = verify_v_bounded_global(safety_margin=100.0, I_max=5.0, timeout_ms=5000)
        assert r.ok is True, f"safe contract should admit: {r.reason}"
        assert r.used_z3 is True

    def test_v_bounded_global_loose_contract_reject(self):
        """[Stage 2.3 G1] loose contract (I_max=50, margin=5) で reject (sat)."""
        r = verify_v_bounded_global(safety_margin=5.0, I_max=50.0, timeout_ms=5000)
        assert r.ok is False, f"loose contract should reject: {r.reason}"
        assert r.counterexample is not None

    def test_v_bounded_per_gene_rs_admit(self):
        """RS canonical で margin=100, I_max=10 → admit."""
        g = IzhikevichGene(0.02, 0.2, -65.0, 8.0)
        r = verify_v_bounded_per_gene(
            g, safety_margin=100.0, I_max=I_MAX_ABS, timeout_ms=2000
        )
        assert r.ok is True, f"RS canonical should admit: {r.reason}"

    def test_v_bounded_per_gene_tight_margin_reject(self):
        """RS でも margin=1 だと overshoot CE 検出 (sat)."""
        g = IzhikevichGene(0.02, 0.2, -65.0, 8.0)
        r = verify_v_bounded_per_gene(
            g, safety_margin=1.0, I_max=I_MAX_ABS, timeout_ms=2000
        )
        assert r.ok is False
        assert r.counterexample is not None

    def test_v_bounded_per_gene_invalid_raises(self):
        """I_max <= 0 / safety_margin <= 0 は ValueError."""
        g = IzhikevichGene(0.02, 0.2, -65.0, 8.0)
        with pytest.raises(ValueError, match="I_max must be positive"):
            verify_v_bounded_per_gene(g, safety_margin=100.0, I_max=0.0)
        with pytest.raises(ValueError, match="safety_margin must be positive"):
            verify_v_bounded_per_gene(g, safety_margin=0.0, I_max=10.0)

    def test_firing_rate_bound_per_gene_unsat(self):
        """dt-discretization 上界が unsat (構造保証)."""
        g = IzhikevichGene(0.02, 0.2, -65.0, 8.0)
        r = verify_firing_rate_per_gene(g, n_spikes=10, T_window_ms=100.0)
        assert r.ok is True
        assert r.used_z3 is True

    def test_firing_rate_bound_per_gene_boundary(self):
        """dt=0.25, T=100 で n_max = 1 + 100/0.25 = 401 spike が boundary admit.

        違反条件 `(n-1) * dt > T` を考慮:
            n=401: (400)*0.25 = 100 NOT> 100 → admit
        """
        g = IzhikevichGene(0.02, 0.2, -65.0, 8.0)
        r = verify_firing_rate_per_gene(g, n_spikes=401, T_window_ms=100.0)
        assert r.ok is True, f"boundary n=401 should admit: {r.reason}"


def test_verifier_unavailable_path():
    """z3 import 失敗時の fallback path テスト (環境制約で skip 可)."""
    if is_z3_available():
        pytest.skip("z3 is available in this env; cannot test fallback path here")


# ---------------------------------------------------------------------------
# Evolution integration smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
def test_evolution_smoke_short():
    """短世代 (5 gen) で進化ループ smoke test."""
    from other_archs.snn.izhikevich.poc import run_izh_evolution

    rng = np.random.default_rng(42)
    trace = run_izh_evolution(
        pop_per_type=4,
        n_types=4,
        n_generations=5,
        target_rate_hz_start=20.0,
        target_rate_hz_end=40.0,
        I_value_curriculum=(5.0, 10.0),
        rng=rng,
        use_verifier=True,
    )
    assert len(trace.populations) == 6  # initial + 5 gens
    assert all(len(p) == 16 for p in trace.populations)
    assert len(trace.best_fitness_curve) == 6
    assert len(trace.verifier_latencies_ms) > 0


@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
def test_evolution_lineage_diversity_smoke():
    """smoke: 5 gen でも 4 firing-type lineage は維持される (LineageReservoir 効果)."""
    from other_archs.snn.izhikevich.poc import run_izh_evolution

    rng = np.random.default_rng(123)
    trace = run_izh_evolution(
        pop_per_type=4,
        n_types=4,
        n_generations=5,
        target_rate_hz_start=20.0,
        target_rate_hz_end=40.0,
        I_value_curriculum=(5.0, 10.0),
        rng=rng,
        use_verifier=True,
    )
    final_types = {ind.firing_type for ind in trace.populations[-1]}
    # Lineage reservoir で全 4 type を維持期待
    assert final_types == {0, 1, 2, 3}, f"missing types: {set(range(4)) - final_types}"


# ---------------------------------------------------------------------------
# Constants sanity (regression for accidental changes)
# ---------------------------------------------------------------------------


class TestConstantsSanity:
    def test_clip_ranges_nonempty(self):
        assert A_MIN < A_MAX
        assert B_MIN < B_MAX
        assert C_MIN < C_MAX
        assert D_MIN < D_MAX
        assert V_PRE_MIN < V_PRE_MAX
        assert U_MIN < U_MAX

    def test_v_peak_above_pre_max(self):
        assert V_PEAK >= V_PRE_MAX

    def test_canonical_4_types_in_clip(self):
        """Izhikevich 2003 canonical 4 type が clip 範囲内に収まる."""
        canonicals = [
            (0.02, 0.2, -65.0, 8.0),  # RS
            (0.02, 0.2, -55.0, 4.0),  # IB
            (0.02, 0.2, -50.0, 2.0),  # CH
            (0.10, 0.2, -65.0, 2.0),  # FS
        ]
        for a, b, c, d in canonicals:
            g = IzhikevichGene(a, b, c, d)
            g_c = g.clipped()
            # clip しても元 gene と等価 (範囲内)
            assert abs(g_c.a - a) < 1e-9
            assert abs(g_c.b - b) < 1e-9
            assert abs(g_c.c - c) < 1e-9
            assert abs(g_c.d - d) < 1e-9
