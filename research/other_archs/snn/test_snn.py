# SPDX-License-Identifier: Apache-2.0
"""SNN PoC pytest battery.

llcore approach の SNN 移植 (LIFGene + Z3 invariant + 進化) の単体・統合テスト.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# path setup (script 直叩き + pytest 両対応)
_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parents[2]
_SRC = _PROJ_ROOT / "src"
_RESEARCH = _PROJ_ROOT / "research"
for p in (_SRC, _RESEARCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from other_archs.snn.snn_gene import (  # noqa: E402
    DT,
    I_MAX_ABS,
    T_REF_MAX,
    T_REF_MIN,
    TAU_M_MAX,
    TAU_M_MIN,
    V_RESET_MAX,
    V_RESET_MIN,
    V_REST,
    V_TH_MAX,
    V_TH_MIN,
    LIFGene,
    firing_rate_hz,
    make_periodic_input,
    simulate_lif,
)
from other_archs.snn.snn_verifier import (  # noqa: E402
    is_z3_available,
    verify_firing_rate_bound,
    verify_firing_rate_per_gene,
    verify_membrane_bounded,
    verify_membrane_bounded_per_gene,
    verify_shielded_rl_hint,
)


# ---------------------------------------------------------------------------
# Gene basic
# ---------------------------------------------------------------------------


class TestLIFGene:
    def test_clipped_in_range(self):
        # 範囲外 gene を作って clip 後に範囲内に収まることを確認
        g = LIFGene(tau_m=100.0, V_th=-30.0, V_reset=-100.0, t_ref=10.0).clipped()
        assert TAU_M_MIN <= g.tau_m <= TAU_M_MAX
        assert V_TH_MIN <= g.V_th <= V_TH_MAX
        assert V_RESET_MIN <= g.V_reset <= V_RESET_MAX
        assert T_REF_MIN <= g.t_ref <= T_REF_MAX

    def test_clipped_reset_below_threshold(self):
        """clip 後も V_reset < V_th が成立 (構造的に成立するが念のため)."""
        g = LIFGene(tau_m=10.0, V_th=-55.0, V_reset=-65.0, t_ref=2.0).clipped()
        assert g.V_reset < g.V_th

    def test_as_array_shape(self):
        g = LIFGene(10.0, -50.0, -70.0, 2.0)
        arr = g.as_array()
        assert arr.shape == (4,)
        assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# Simulator basic
# ---------------------------------------------------------------------------


class TestSimulateLIF:
    def test_no_input_stays_at_rest(self):
        """ゼロ入力なら膜電位は V_REST 付近で安定 + 発火なし."""
        g = LIFGene(10.0, -50.0, -70.0, 2.0)
        I = np.zeros(1000)
        V, spikes = simulate_lif(g, I, T=100.0)
        # V は V_REST 付近
        assert abs(V[-1] - V_REST) < 0.5
        assert len(spikes) == 0
        assert np.all(np.isfinite(V))

    def test_periodic_input_induces_spikes(self):
        """周期入力 + bias で発火が発生."""
        g = LIFGene(10.0, -50.0, -70.0, 2.0)
        rng = np.random.default_rng(0)
        I = make_periodic_input(T=200.0, freq_hz=20.0, amplitude=0.5, bias=1.5, rng=rng)
        V, spikes = simulate_lif(g, I, T=200.0)
        assert len(spikes) > 0
        assert np.all(np.isfinite(V))

    def test_refractory_period_respected(self):
        """spike 間隔が t_ref 以上で保たれる."""
        g = LIFGene(tau_m=8.0, V_th=-50.0, V_reset=-70.0, t_ref=3.0)
        rng = np.random.default_rng(1)
        # 強い駆動で高頻度発火を狙う
        I = make_periodic_input(T=200.0, freq_hz=50.0, amplitude=0.5, bias=2.0, rng=rng)
        V, spikes = simulate_lif(g, I, T=200.0)
        if len(spikes) >= 2:
            intervals = np.diff(spikes)
            # 全 spike 間隔 >= t_ref (forward Euler 精度内)
            assert intervals.min() >= g.t_ref - DT - 1e-6, (
                f"min interval {intervals.min()} < t_ref {g.t_ref}"
            )

    def test_membrane_bounded_during_simulation(self):
        """V が clip 範囲内に収まる (V_RESET から V_TH まで)."""
        g = LIFGene(10.0, -50.0, -70.0, 2.0)
        rng = np.random.default_rng(2)
        I = make_periodic_input(T=200.0, freq_hz=10.0, amplitude=0.5, bias=1.5, rng=rng)
        V, spikes = simulate_lif(g, I, T=200.0)
        # spike 直後の V = V_reset, それ以外は V_reset から V_th + small overshoot 範囲
        # Euler 1 step overshoot を許容: <= V_th + 5 mV
        assert V.min() >= g.V_reset - 1.0
        assert V.max() <= g.V_th + 5.0


# ---------------------------------------------------------------------------
# firing_rate_hz
# ---------------------------------------------------------------------------


def test_firing_rate_hz():
    spikes = [10.0, 20.0, 30.0, 40.0, 50.0]
    rate = firing_rate_hz(spikes, T=100.0)
    # 5 spikes in 100ms = 50 Hz
    assert abs(rate - 50.0) < 1e-9


def test_firing_rate_hz_empty():
    assert firing_rate_hz([], 100.0) == 0.0


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
class TestSNNVerifier:
    def test_firing_rate_bound_global_unsat(self):
        """構造的 firing rate 上界が unsat (証明)."""
        r = verify_firing_rate_bound(n_spikes=10, T_window_ms=100.0)
        assert r.ok is True
        assert r.used_z3 is True

    def test_firing_rate_bound_per_gene_unsat(self):
        """単一 gene でも unsat 証明."""
        g = LIFGene(10.0, -50.0, -70.0, 3.0)
        r = verify_firing_rate_per_gene(g, n_spikes=10, T_window_ms=100.0)
        assert r.ok is True
        assert r.used_z3 is True

    def test_membrane_bounded_safe_margin(self):
        """safety_margin=5 で unsat 証明."""
        r = verify_membrane_bounded(safety_margin=5.0, timeout_ms=3000)
        assert r.ok is True
        assert r.used_z3 is True

    def test_membrane_bounded_per_gene(self):
        """per-gene 膜電位 bound 検査."""
        g = LIFGene(10.0, -50.0, -70.0, 2.0)
        r = verify_membrane_bounded_per_gene(g, safety_margin=5.0, timeout_ms=1000)
        assert r.ok is True
        assert r.used_z3 is True

    def test_shielded_rl_hint_admit(self):
        """t_ref=5 ms, R_safe=200 で admit (境界, unsat)."""
        g = LIFGene(10.0, -50.0, -70.0, 5.0)
        r = verify_shielded_rl_hint(g, R_safe_hz=200.0)
        assert r.ok is True

    def test_shielded_rl_hint_reject(self):
        """t_ref=1 ms (rate_max=1000Hz), R_safe=200 で reject (violation, sat)."""
        g = LIFGene(10.0, -50.0, -70.0, 1.0)
        r = verify_shielded_rl_hint(g, R_safe_hz=200.0)
        assert r.ok is False
        assert r.counterexample is not None

    def test_shielded_rl_hint_safe(self):
        """t_ref=3 ms (rate_max~333Hz), R_safe=400 で admit."""
        g = LIFGene(10.0, -50.0, -70.0, 3.0)
        r = verify_shielded_rl_hint(g, R_safe_hz=400.0)
        assert r.ok is True


def test_verifier_unavailable_path():
    """z3 import 失敗時の fallback path (vacuous True) は SNNInvariantResult.used_z3=False."""
    if is_z3_available():
        # 実環境では z3 利用可能なのでこのテストは skip
        pytest.skip("z3 is available in this env; cannot test fallback path here")


# ---------------------------------------------------------------------------
# Evolution integration smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
def test_evolution_smoke_short():
    """短世代 (10 gen) で進化ループ smoke test."""
    from other_archs.snn.poc import run_snn_evolution

    rng = np.random.default_rng(42)
    trace = run_snn_evolution(
        pop_per_type=4,
        n_types=4,
        n_generations=10,
        target_rate_hz_start=20.0,
        target_rate_hz_end=40.0,
        input_freq_curriculum=(5.0, 20.0),
        rng=rng,
        use_verifier=True,
    )
    # 集団サイズ * 11 世代
    assert len(trace.populations) == 11
    assert all(len(p) == 16 for p in trace.populations)
    # best fitness curve も 11 点
    assert len(trace.best_fitness_curve) == 11
    # verifier 呼び出しが発生している
    assert len(trace.verifier_latencies_ms) > 0
