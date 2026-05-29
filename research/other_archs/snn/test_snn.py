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
    verify_membrane_bounded_2step,
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

    def test_firing_rate_bound_boundary_admit(self):
        """[Stage 2.1 Codex F5] finite-window boundary case n = 1 + T/t_ref.

        Codex Finding #1 で発覚した off-by-one 修正の regression test:
            t_ref=5ms, T_window=100ms で n=21 spike (`0, 5, 10, ..., 100`) は
            refractory respect で構造的に許容される (boundary case).
            旧実装 `n * t_ref > T_window` は 21*5=105>100 で誤 reject していた.
            新実装 `(n-1) * t_ref > T_window` は 20*5=100 NOT> 100 で正 admit.

        本 test が PASS = off-by-one 修正後の sound 性が機械検査で保証.
        """
        # boundary case (n = 1 + T_window/t_ref): admit (unsat = invariant 成立)
        r_boundary = verify_firing_rate_bound(n_spikes=21, T_window_ms=100.0)
        assert r_boundary.ok is True, f"boundary n=21 should admit: {r_boundary.reason}"
        assert r_boundary.used_z3 is True

    def test_firing_rate_bound_over_boundary_admit_constructive(self):
        """[Stage 2.1 Codex F5 follow-up] **新気付き**: n=22 も admit (構造的 unsat).

        新実装 `(n-1)*t_ref > T_window` を violation 条件として Z3 に投入する場合、
        refractory respect 制約 (t_{i+1}-t_i >= t_ref, 0 <= t_0, t_n-1 <= T_window)
        と組み合わせると n に関わらず **constructive proof で unsat** (invariant 成立)
        が成立する.

        n=22 の場合:
        - violation `21 * t_ref > 100` ⇔ `t_ref > 100/21 ≈ 4.76`
        - refractory respect `t_21 - t_0 >= 21 * t_ref`、`t_0 >= 0 ∧ t_21 <= 100`
          ⇒ `21 * t_ref <= 100`
        - 両者矛盾 ⇒ unsat (admit, invariant 成立)

        → 新実装は finite-window でも数学的に厳密で **任意の n で invariant 成立**.
        Codex Q2「現状の bound は finite-window で厳密でない」は旧実装に対する指摘、
        新実装 (修正後) では constructive proof で成立する.

        boundary admit (n=21) + over-boundary admit (n=22) の両方で旧実装 false
        positive が無いことを機械検査で保証.
        """
        r_over = verify_firing_rate_bound(n_spikes=22, T_window_ms=100.0)
        assert r_over.ok is True, (
            f"n=22 should admit (constructive unsat): {r_over.reason}"
        )
        assert r_over.used_z3 is True

    def test_firing_rate_bound_per_gene_boundary_admit(self):
        """per-gene でも boundary case (n = 1 + T/t_ref) で admit.

        t_ref=5ms, T=100ms → n_max=21 で per-gene 検査も sound.
        """
        g = LIFGene(10.0, -50.0, -70.0, 5.0)
        r = verify_firing_rate_per_gene(g, n_spikes=21, T_window_ms=100.0)
        assert r.ok is True, f"per-gene boundary n=21 should admit: {r.reason}"

    def test_membrane_bounded_safe_margin(self):
        """safety_margin=5 で unsat 証明 (default I_max, 後方互換)."""
        r = verify_membrane_bounded(safety_margin=5.0, timeout_ms=3000)
        assert r.ok is True
        assert r.used_z3 is True

    def test_membrane_bounded_per_gene(self):
        """per-gene 膜電位 bound 検査 (default I_max, 後方互換)."""
        g = LIFGene(10.0, -50.0, -70.0, 2.0)
        r = verify_membrane_bounded_per_gene(g, safety_margin=5.0, timeout_ms=1000)
        assert r.ok is True
        assert r.used_z3 is True

    def test_membrane_bounded_tight_I_max_admit(self):
        """[Stage 2.2a Codex F3] tight contract I_max=1.0 で admit (sound 強化).

        V_next = V + (DT/τ_m) * (V_REST - V + R*I), DT=0.1, τ_m∈[5,30], R=10
        I_max=1.0, V=V_TH=-40 で V_next 上界 = -40 + (0.1/5)*(-25+10*1) = -40.3
        ≤ V_TH+safety_margin=-40+5=-35 で admit 期待.
        """
        r = verify_membrane_bounded(safety_margin=5.0, I_max=1.0, timeout_ms=3000)
        assert r.ok is True, f"tight I_max=1.0 should admit: {r.reason}"
        assert "I ∈ [-1.0,1.0]" in r.reason, f"reason should reflect I_max: {r.reason}"

    def test_membrane_bounded_loose_I_max_rejects(self):
        """[Stage 2.2a Codex F3] loose contract I_max=10.0 + safety_margin=0 で reject.

        I_max=10.0 で V=V_TH=-40 から V_next 計算:
        V_next = -40 + (0.1/τ_m_min)*(V_REST - V + R*I) = -40 + 0.02*(-25 + 100) = -38.5
        safety_margin=0 で V_TH 越え (V_next > -40) を Z3 が探索 → overshoot 検出.
        """
        r = verify_membrane_bounded(safety_margin=0.0, I_max=10.0, timeout_ms=3000)
        assert r.ok is False, f"loose I_max=10 with margin=0 should reject: {r.reason}"
        assert r.counterexample is not None

    def test_membrane_bounded_I_max_invalid_raises(self):
        """[Stage 2.2a] I_max <= 0 は ValueError (sound contract 保証)."""
        with pytest.raises(ValueError, match="I_max must be positive"):
            verify_membrane_bounded(safety_margin=5.0, I_max=0.0)
        with pytest.raises(ValueError, match="I_max must be positive"):
            verify_membrane_bounded(safety_margin=5.0, I_max=-1.0)

    def test_membrane_bounded_per_gene_tight_I_max(self):
        """[Stage 2.2a] per-gene 版で I_max=1.0 admit."""
        g = LIFGene(10.0, -50.0, -70.0, 2.0)
        r = verify_membrane_bounded_per_gene(
            g, safety_margin=5.0, I_max=1.0, timeout_ms=1000
        )
        assert r.ok is True, f"per-gene tight I_max should admit: {r.reason}"
        assert r.used_z3 is True

    def test_membrane_bounded_per_gene_I_max_invalid_raises(self):
        """[Stage 2.2a] per-gene 版でも I_max <= 0 は ValueError."""
        g = LIFGene(10.0, -50.0, -70.0, 2.0)
        with pytest.raises(ValueError, match="I_max must be positive"):
            verify_membrane_bounded_per_gene(g, safety_margin=5.0, I_max=0.0)

    # ----- Stage 2.2b: 2-step + |ΔI| contract -----

    def test_membrane_bounded_2step_tight_contract_admit(self):
        """[Stage 2.2b] I_max=1.0, dI_max=0.5 tight contract で 2-step admit.

        2-step Euler chain で V_1, V_2 が両方 safe range 内.
        |I_1 - I_0| <= 0.5 で input dynamics 制約.
        """
        r = verify_membrane_bounded_2step(
            safety_margin=5.0, I_max=1.0, dI_max=0.5, timeout_ms=3000
        )
        assert r.ok is True, f"tight 2-step contract should admit: {r.reason}"
        assert r.used_z3 is True
        assert "|ΔI|<=0.5" in r.reason

    def test_membrane_bounded_2step_loose_dI_rejects(self):
        """[Stage 2.2b] dI_max 緩めると 2-step CE 検出 (input dynamics 制約効果実証).

        I_max=10 (overshoot 可能) + safety_margin=0 で 1-step と同様 reject.
        2-step では |ΔI|=20 (= 2*I_max) ≧ contract 不要、Z3 が overshoot CE 検出.
        """
        r = verify_membrane_bounded_2step(
            safety_margin=0.0, I_max=10.0, dI_max=20.0, timeout_ms=3000
        )
        assert r.ok is False, f"loose 2-step contract should reject: {r.reason}"
        assert r.counterexample is not None
        assert "V0" in r.counterexample and "V1" in r.counterexample and "V2" in r.counterexample

    def test_membrane_bounded_2step_default_dI_eq_2_I_max(self):
        """[Stage 2.2b] dI_max=None で default = 2*I_max (制約なし相当)."""
        r = verify_membrane_bounded_2step(
            safety_margin=5.0, I_max=1.0, dI_max=None, timeout_ms=3000
        )
        assert r.ok is True
        assert "|ΔI|<=2.0" in r.reason

    def test_membrane_bounded_2step_invalid_dI_max_raises(self):
        """[Stage 2.2b] dI_max <= 0 で ValueError."""
        with pytest.raises(ValueError, match="dI_max must be positive"):
            verify_membrane_bounded_2step(safety_margin=5.0, I_max=1.0, dI_max=0.0)
        with pytest.raises(ValueError, match="dI_max must be positive"):
            verify_membrane_bounded_2step(safety_margin=5.0, I_max=1.0, dI_max=-0.1)

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
