# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PoC 3a — Marabou Incremental sound 拡張 refinement + MCC curriculum."""
from __future__ import annotations

import random

import pytest

from llcore.state_update import StateUpdateGene
from llcore.verifier import (
    E_BASE,
    KERNEL_SWAP_EXTRA,
    ChangeOp,
    ChangeOpSequence,
    apply_changeop,
    apply_sequence,
    decay_shift,
    epsilon_for,
    evolve_one_generation,
    gate_shift,
    get_bridge_status,
    is_marabou_available,
    is_saturated,
    is_z3_available,
    kernel_swap_mock,
    mix_shift,
    run_curriculum,
    sequence_from_iter,
    verify_composition,
    verify_refinement_single,
    verify_sequence_tolerance,
)
from llcore.verifier.curriculum import CurriculumState


SAFE_GENE = StateUpdateGene(decay=0.7, mix=0.3, gate_str=0.4)


# ---------------------------------------------------------------------------
# ChangeOp dataclass basics
# ---------------------------------------------------------------------------


def test_changeop_invalid_type_raises() -> None:
    """未知 op_type は ValueError."""
    with pytest.raises(ValueError):
        ChangeOp(op_type="unknown", delta=0.1)


def test_changeop_kernel_swap_requires_0_or_1() -> None:
    """kernel_swap_mock は delta ∈ {0.0, 1.0} のみ許容."""
    # 正常: swap=True/False は delta=1.0/0.0
    assert kernel_swap_mock(swap=True).delta == 1.0
    assert kernel_swap_mock(swap=False).delta == 0.0
    # 異常: 中間 delta は ValueError
    with pytest.raises(ValueError):
        ChangeOp(op_type="kernel_swap_mock", delta=0.5)


def test_changeop_magnitude_is_abs_delta() -> None:
    assert decay_shift(-0.3).magnitude() == pytest.approx(0.3)
    assert mix_shift(0.0).magnitude() == pytest.approx(0.0)
    assert kernel_swap_mock(swap=True).magnitude() == pytest.approx(1.0)


def test_changeop_sequence_composition_extends_length() -> None:
    """compose は順序保持の単純連結."""
    s1 = ChangeOpSequence(ops=(decay_shift(0.1), mix_shift(0.2)))
    s2 = ChangeOpSequence(ops=(gate_shift(0.05),))
    s12 = s1.compose(s2)
    assert len(s12) == 3
    assert s12.ops[0].op_type == "decay_shift"
    assert s12.ops[2].op_type == "gate_shift"


def test_apply_changeop_decay_shift() -> None:
    g = StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)
    g2 = apply_changeop(g, decay_shift(0.1))
    assert g2.decay == pytest.approx(0.6)
    assert g2.mix == pytest.approx(0.3)
    assert g2.gate_str == pytest.approx(0.4)


def test_apply_sequence_left_to_right() -> None:
    g = StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)
    seq = sequence_from_iter([decay_shift(0.1), mix_shift(0.2)])
    g2 = apply_sequence(g, seq)
    assert g2.decay == pytest.approx(0.6)
    assert g2.mix == pytest.approx(0.5)


def test_apply_kernel_swap_mock_flips_gate_str() -> None:
    g = StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)
    g2 = apply_changeop(g, kernel_swap_mock(swap=True))
    assert g2.gate_str == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# epsilon linearity / additivity (合成性の数式根拠)
# ---------------------------------------------------------------------------


def test_epsilon_linear_in_magnitude() -> None:
    eps_small = epsilon_for(decay_shift(0.1))
    eps_big = epsilon_for(decay_shift(0.5))
    assert eps_small == pytest.approx(E_BASE * 0.1)
    assert eps_big == pytest.approx(E_BASE * 0.5)


def test_epsilon_kernel_swap_has_extra_penalty() -> None:
    """kernel_swap_mock(True) は extra penalty を持つ (K=1 inheritance を超える discrete 変更)."""
    eps_swap = epsilon_for(kernel_swap_mock(swap=True))
    eps_noop = epsilon_for(kernel_swap_mock(swap=False))
    assert eps_swap == pytest.approx(E_BASE * 1.0 + KERNEL_SWAP_EXTRA)
    assert eps_noop == pytest.approx(0.0)


def test_epsilon_additive_on_sequence() -> None:
    """合成性 ε(c1 ∘ c2) = ε(c1) + ε(c2)."""
    c1 = decay_shift(0.2)
    c2 = mix_shift(0.3)
    seq = sequence_from_iter([c1, c2])
    assert epsilon_for(seq) == pytest.approx(epsilon_for(c1) + epsilon_for(c2))


# ---------------------------------------------------------------------------
# G1: single ChangeOp judged
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g1_single_changeop_safe_admitted() -> None:
    r = verify_refinement_single(SAFE_GENE, decay_shift(0.05), timeout_ms=500)
    assert r.ok
    assert r.used_z3
    assert r.epsilon == pytest.approx(E_BASE * 0.05)


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g1_pathological_delta_rejected_despite_large_epsilon() -> None:
    """病的 ChangeOp (decay+5.0 → post=5.7 unstable) は ε 巨大でも sound 反例検出.

    これが G4 と並ぶ "verifier が真に sound" の証拠 — ε が線形に大きくなっても
    structurally unstable な変換は反例として正しく検出される。
    """
    r = verify_refinement_single(SAFE_GENE, decay_shift(5.0), timeout_ms=500)
    assert not r.ok
    assert r.counterexample is not None
    assert r.epsilon == pytest.approx(E_BASE * 5.0)


# ---------------------------------------------------------------------------
# G2: composition (合成性)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g2_composition_sound() -> None:
    c1 = decay_shift(0.05)
    c2 = mix_shift(0.1)
    r = verify_composition(SAFE_GENE, c1, c2, state_bound=1.0, timeout_ms=1500)
    assert r.ok
    assert r.epsilon == pytest.approx(epsilon_for(c1) + epsilon_for(c2))


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g2_composition_with_kernel_swap() -> None:
    """kernel_swap (extra penalty) を含めても合成性が崩れない."""
    c1 = decay_shift(0.05)
    c2 = kernel_swap_mock(swap=True)
    r = verify_composition(SAFE_GENE, c1, c2, state_bound=1.0, timeout_ms=1500)
    assert r.ok


# ---------------------------------------------------------------------------
# G3: 100-step sequence bound holds
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g3_100step_bound_holds() -> None:
    rng = random.Random(20260529)
    ops = []
    for _ in range(100):
        kind = rng.choice(("decay", "mix", "gate"))
        delta = rng.uniform(-0.005, 0.005)
        if kind == "decay":
            ops.append(decay_shift(delta))
        elif kind == "mix":
            ops.append(mix_shift(delta))
        else:
            ops.append(gate_shift(delta))
    seq = ChangeOpSequence(ops=tuple(ops))
    r = verify_sequence_tolerance(SAFE_GENE, seq, per_step_timeout_ms=200)
    assert r.ok
    assert r.passed_steps == 100
    # 累積 ε は Σ magnitude * E_BASE 程度 (微小)
    assert r.epsilon_total < 1.0


# ---------------------------------------------------------------------------
# G4: pathological ChangeOp counterexample
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g4_pathological_decay_unstable_detected() -> None:
    """decay > 1 (illegal) を Z3 で直接組んだら sat 反例が出る."""
    import z3
    solver = z3.Solver()
    solver.set("timeout", 500)
    s = z3.Real("s")
    tanh_v = z3.Real("tanh_v")
    solver.add(s == 1.0)
    solver.add(tanh_v >= -1, tanh_v <= 1)
    decay = z3.RealVal(2.0)
    s_next = decay * s + (1 - decay) * tanh_v
    solver.add(z3.Or(s_next > 1.5, s_next < -1.5))
    assert solver.check() == z3.sat


# ---------------------------------------------------------------------------
# G5: Marabou containment sketch (docs)
# ---------------------------------------------------------------------------


def test_g5_paper_sketch_exists() -> None:
    """論文 sketch が docs/papers に存在し主要キーワードを含む."""
    from pathlib import Path
    paper = Path(__file__).resolve().parents[2] / "docs/papers/marabou_sound_extension_sketch.md"
    assert paper.exists()
    text = paper.read_text(encoding="utf-8")
    for key in ("Marabou", "refinement", "異構造", "ChangeOp", "包含"):
        assert key in text, f"key {key!r} missing in paper sketch"


# ---------------------------------------------------------------------------
# G6/G8: curriculum
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g6_curriculum_records_pass_rate_each_generation() -> None:
    state = run_curriculum(
        SAFE_GENE,
        n_generations=4,
        pop_size=12,
        per_changeop_timeout_ms=120,
        seed=20260529,
    )
    assert len(state.generations) == 4
    for gen in state.generations:
        assert 0.0 <= gen.pass_rate <= 1.0


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g6b_curriculum_not_saturated() -> None:
    state = run_curriculum(
        SAFE_GENE,
        n_generations=6,
        pop_size=16,
        initial_max_mag=0.1,
        magnitude_cap=0.6,
        epsilon_floor_quantile=0.5,
        per_changeop_timeout_ms=120,
        seed=20260530,
    )
    # is_saturated は plateau_eps 既定 1e-4 以下の slope を saturation とみなす
    # mutation + refill により完全 plateau は起きない設計
    assert not is_saturated(state) or state.frontier_slope > 0


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g8_curriculum_frontier_progresses() -> None:
    state = run_curriculum(
        SAFE_GENE,
        n_generations=8,
        pop_size=20,
        initial_max_mag=0.05,
        magnitude_cap=0.8,
        epsilon_floor_quantile=0.6,
        mutation_sigma=0.08,
        per_changeop_timeout_ms=120,
        seed=20260601,
    )
    initial_frontier = state.generations[0].epsilon_frontier
    last_frontier = state.last_frontier
    # frontier が漸増 (厳密でなくとも noise 越えで Δ>0 期待; mutation_sigma 0.08 で十分)
    assert last_frontier >= initial_frontier


# ---------------------------------------------------------------------------
# G7: timeout per step
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g7_per_step_timeout_under_100ms() -> None:
    seq = ChangeOpSequence(ops=tuple(decay_shift(0.001) for _ in range(50)))
    r = verify_sequence_tolerance(SAFE_GENE, seq, per_step_timeout_ms=100)
    assert r.ok
    assert max(r.per_step_ms) < 100.0


# ---------------------------------------------------------------------------
# G9: Marabou absent mock runs
# ---------------------------------------------------------------------------


def test_g9_marabou_absent_or_hybrid() -> None:
    """Marabou 不在 → bridge_mode=z3_mock、存在 → hybrid のどちらかになる."""
    status = get_bridge_status()
    assert status.bridge_mode in ("z3_mock", "hybrid")
    assert status.marabou_available == is_marabou_available()


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_g9_z3_path_runs_regardless_of_marabou() -> None:
    """Marabou があってもなくても Z3 検査は動く."""
    r = verify_refinement_single(SAFE_GENE, decay_shift(0.05), timeout_ms=300)
    assert r.used_z3
    assert r.ok


# ---------------------------------------------------------------------------
# z3 fallback path
# ---------------------------------------------------------------------------


def test_refinement_z3_unavailable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Z3 が無い場合も RefinementResult を返し、ε は線形計算でわかる."""
    import llcore.verifier.refinement as refinement_mod
    monkeypatch.setattr(refinement_mod, "_HAS_Z3", False)
    r = refinement_mod.verify_refinement_single(SAFE_GENE, decay_shift(0.1))
    assert r.ok
    assert not r.used_z3
    assert r.epsilon == pytest.approx(E_BASE * 0.1)


def test_composition_z3_unavailable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import llcore.verifier.refinement as refinement_mod
    monkeypatch.setattr(refinement_mod, "_HAS_Z3", False)
    r = refinement_mod.verify_composition(SAFE_GENE, decay_shift(0.1), mix_shift(0.2))
    assert r.ok
    assert not r.used_z3
    assert r.epsilon == pytest.approx(E_BASE * (0.1 + 0.2))


# ---------------------------------------------------------------------------
# evolve_one_generation internal behavior
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_z3_available(), reason="z3 required")
def test_evolve_one_generation_keeps_pop_size() -> None:
    """1 世代回しても pop size が維持される (refill が働く)."""
    state = CurriculumState(rng=random.Random(123), magnitude_cap=0.5)
    pop = tuple(decay_shift(0.05) for _ in range(10))
    new_pop = evolve_one_generation(SAFE_GENE, pop, state, timeout_ms=120)
    assert len(new_pop) == 10
    assert len(state.generations) == 1
