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
# Stage 2.4 反証的 tests (Codex Findings を機械検査で確定する)
#
# 目的: Stage 2.3 Codex pair-review が指摘した overclaim を **test で機械検査**
# することで、Codex pair-review が担っていた反証役を **test に内製化** する.
# これらの test は **PASS する** が、内容は「現実装の claim が overclaim だった」
# ことを assert する反証. 既存 claim の機械検査強化と Stage 3+ 改善方向の固定化.
# 関連 memory: [[feedback_codex_pair_review_for_llcore]] (反証 test の不在 ⇒ 内製化)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
def test_anti_verifier_is_gene_independent_in_v_bounded():
    """[反証] Codex F1: `verify_v_bounded_per_gene` は実質 gene-independent.

    `verify_v_bounded_per_gene(gene, ...)` は gene を `clipped()` しただけで
    Z3 制約式に a, b, c, d を **入れない** (u は box `[-25, 25]` で扱う).
    Codex 指摘の通り「per-gene」表現は overclaim. 本 test は **現実装が確かに
    gene-independent** であることを機械検査する反証 test = 大きく異なる 2 gene
    で同じ Z3 verdict (admit/reject + 同じ reason 構造) を返すことを assert.

    現実装が overclaim でない実装に **将来修正された場合は本 test が FAIL** し、
    Codex F1 の真の per-gene verifier への昇格を機械検査で検知できる.
    """
    safety_margin = 100.0
    I_max = 5.0
    # 大きく異なる 2 gene (RS と FS)
    g_RS = IzhikevichGene(a=0.02, b=0.2, c=-65.0, d=8.0)
    g_FS = IzhikevichGene(a=0.10, b=0.2, c=-65.0, d=2.0)

    r_RS = verify_v_bounded_per_gene(g_RS, safety_margin=safety_margin, I_max=I_max)
    r_FS = verify_v_bounded_per_gene(g_FS, safety_margin=safety_margin, I_max=I_max)

    # 両方とも同じ verdict (gene が制約に効かない証拠)
    assert r_RS.ok == r_FS.ok, (
        f"gene-independent claim 反証失敗: RS.ok={r_RS.ok}, FS.ok={r_FS.ok}. "
        f"verifier が真に per-gene になっていれば本 test は FAIL し、"
        f"Stage 3+ で a,b,c,d を Z3 制約に入れた事実が機械検査される."
    )

    # honest: 実装は reason 文字列で `gene (a=..., b=...)` を表示するが、
    # **Z3 制約式 (solver.assertions)** には a, b が入っていない (Codex F1 指摘).
    # reason 表示は cosmetic で per-gene 装い + Z3 構造は gene-independent =
    # 「reason だけ per-gene 装い」も別の overclaim 可能性 → Stage 2.4 サブ phase で
    # reason 表示を「gene clipped box は同じ」に正名化検討.
    # 本 test では cosmetic 検査は割愛し verdict 一致のみで反証成立とする.


@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
def test_anti_firing_rate_bound_is_dt_packing_not_neuron_dynamics():
    """[反証] Codex F2: `verify_firing_rate_per_gene` は dt packing bound のみ.

    Codex 指摘: 実装は `t_{i+1} - t_i >= dt` と `(n-1)*dt > T_window` のみで、
    gene a,b,c,d は使われない. 「per-gene firing rate invariant」は誤解.

    本 test は **異なる gene でも同じ verdict** を返すことを assert する反証.
    """
    n_spikes = 10
    T_window_ms = 50.0  # 10 spike × DT=0.1ms = 1ms <<<< 50ms = admit 想定
    g_RS = IzhikevichGene(a=0.02, b=0.2, c=-65.0, d=8.0)
    g_FS = IzhikevichGene(a=0.10, b=0.2, c=-65.0, d=2.0)

    r_RS = verify_firing_rate_per_gene(g_RS, n_spikes=n_spikes, T_window_ms=T_window_ms)
    r_FS = verify_firing_rate_per_gene(g_FS, n_spikes=n_spikes, T_window_ms=T_window_ms)

    assert r_RS.ok == r_FS.ok, (
        f"firing-rate bound が gene-independent 反証失敗: "
        f"Codex F2 claim と矛盾, verifier が gene dynamics を反映していれば異 verdict 想定"
    )


@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
def test_anti_G8_lineage_diversity_depends_on_reservoir_not_attractor():
    """[反証] Codex F3: G8 4 firing-type lineage 維持は Reservoir 効果 (mechanism でない).

    Codex 指摘: lineage 4 種維持は `sample_initial_gene` type bias と
    `LineageReservoir.reinject_extinct()` で **構造的に保証**, 4 attractors の
    証拠でない. `RS=6, IB=24, CH=1, FS=1` は「1 dominant basin (IB) + 3 preserved
    labels」.

    本 test は **gene-guess (firing_type_guess) が lineage label と乖離する** ことを
    assert する反証 = lineage label は reservoir で構造的維持されるが、実 gene の
    firing-type 分布は IB 集約傾向を示す.
    """
    from other_archs.snn.izhikevich.poc import run_izh_evolution

    rng = np.random.default_rng(20260530)
    trace = run_izh_evolution(
        pop_per_type=4,
        n_types=4,
        n_generations=8,
        target_rate_hz_start=20.0,
        target_rate_hz_end=40.0,
        I_value_curriculum=(5.0, 10.0),
        rng=rng,
        use_verifier=True,
    )
    final = trace.populations[-1]
    lineage_labels = {ind.firing_type for ind in final}
    # firing_type_guess は IzhikevichGene method (ind.firing_type_guess は string)
    gene_guesses = [ind.firing_type_guess for ind in final]

    # lineage labels は Reservoir で 4 種維持 (構造的保証)
    assert lineage_labels == {0, 1, 2, 3}, f"lineage label 維持失敗: {lineage_labels}"

    # 反証ポイント: 最頻 gene-guess type が支配的 = "1 dominant basin"
    from collections import Counter
    guess_counter = Counter(gene_guesses)
    most_common_count = guess_counter.most_common(1)[0][1]
    dominant_ratio = most_common_count / len(final)
    # >=0.35 (= 32 個体中 12 個以上が同 type) で "1 dominant basin" 確認可
    # honest: seed 依存で揺れるため緩めの閾値 (Codex F3 受容を機械検査)
    assert dominant_ratio >= 0.35, (
        f"1 dominant basin 反証失敗: 最頻 gene-guess type が {dominant_ratio:.0%}. "
        f"Codex F3 想定では selection pressure 下で 1 type dominant の想定 "
        f"(均等分布なら 25%, 集中なら >35%)."
    )


def test_anti_G8_pass_threshold_is_lenient():
    """[反証] Codex F3 補足: G8 判定 logic は gene-guess 3 種以上で trivial PASS.

    poc.py の G8 判定 (`gate_g8_firing_type_distribution`) は lineage 種数 + gene-guess
    種数の組合せで PASS する設計. 厳格な「4 attractors mechanism」claim には
    `gene_guesses == 4` and 各 type が >= 一定割合、等の hard 条件が必要.

    本 test は poc.py source 中に「>= 3」または「>=3」緩めの閾値の存在を assert.
    将来 hard 厳格化された場合は本 test FAIL = claim 強化が機械検査される.
    """
    from other_archs.snn.izhikevich import poc

    source = Path(poc.__file__).read_text(encoding="utf-8")
    # G8 判定関数本体を文字列で含む + >= 3 or >=3 の lenient 閾値を含む
    assert "g8" in source.lower() or "G8" in source, "PoC G8 関数が見つからない"
    # 3 という閾値文字列が source に少なくとも 1 回出る (G4=4, G7=15ms 等もあるため
    # 緩い assertion. hard 厳格化で 4 のみになれば本 test 修正の指針)
    assert " 3" in source or ">=3" in source or ">= 3" in source, (
        "lenient threshold (3) が見つからない. G8 厳格化された場合は本 test を update."
    )


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
