# SPDX-License-Identifier: Apache-2.0
"""Phase 2a — trajectory_tube gate PoC のユニットテスト.

検証範囲:
- 外乱チェッカ (research/verified_memory_poc/disturbance_checker.py):
  - (P1) contraction-certified gene で実測追従誤差 ≤ certified tube (0 違反)。
  - 非契約 gene では tube=∞ かつ tube_holds=False (vacuous guarantee を弾く)。
  - rollout port が scalar run_sequence 基板で動く (CoupledNDGene から再ポイント)。
- bridge anchor (P3): named-slot write 写像が eval_step と数値恒等 (bridge が fork でない)。

設計 doc: docs/research/phase2a_verified_memory_evolution_design_2026_06_06.md §4.5-2/3。

実行::

    pytest tests/unit/test_verified_memory_poc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from llcore.state_update import StateUpdateGene, eval_step, run_sequence
from llcore.verifier import tracking_tube, verify_lipschitz_contraction


def _load_disturbance_checker():
    """research/verified_memory_poc/disturbance_checker.py を import."""
    research_dir = Path(__file__).resolve().parents[2] / "research" / "verified_memory_poc"
    if str(research_dir) not in sys.path:
        sys.path.insert(0, str(research_dir))
    import disturbance_checker  # noqa: E402

    return disturbance_checker


# ---------------------------------------------------------------------------
# 外乱チェッカ (tube 定理の cross-check)
# ---------------------------------------------------------------------------


# 代表的な contracting gene (L<1, 多様な tube 半径)。
_CONTRACTING_GENES = [
    StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.0),   # L=0.9, G=0.05, tube small
    StateUpdateGene(decay=0.5, mix=1.0, gate_str=0.0),   # L=0.5, G=0.5,  tube mid
    StateUpdateGene(decay=0.7, mix=0.3, gate_str=0.2),   # coupling あり
    StateUpdateGene(decay=0.5, mix=0.0, gate_str=0.0),   # fallback (G=0 → tube=0)
    StateUpdateGene(decay=0.8, mix=-0.6, gate_str=-0.3),  # 負 mix / 抑制 recurrent
]


@pytest.mark.parametrize("gene", _CONTRACTING_GENES)
def test_p1_certified_tube_holds_for_contracting(gene: StateUpdateGene) -> None:
    """(P1) contraction-certified gene で実測追従誤差 ≤ certified tube (0 違反)."""
    dc = _load_disturbance_checker()
    # 前提: closed-form L<1 (tube gate の contraction_ok)。
    assert tracking_tube(gene, w_bar=0.1).contraction_ok is True
    res = dc.tube_cross_check(gene, w_bar=0.1, seq_len=256, dim=8, n_seeds=64)
    assert res.contraction_ok is True
    assert np.isfinite(res.certified_tube)
    assert res.tube_holds is True, (
        f"実測 {res.empirical_max_err_steady} > certified {res.certified_tube} for {gene}"
    )
    # 実測 ≤ certified (P1 不等式 そのもの)。
    assert res.empirical_max_err_steady <= res.certified_tube + 1e-9


def test_p1_noncontract_tube_infinite_and_not_holds() -> None:
    """非契約 gene は tube=∞ かつ tube_holds=False (vacuous guarantee を弾く)."""
    dc = _load_disturbance_checker()
    g = StateUpdateGene(decay=0.0, mix=1.0, gate_str=2.0)  # L>=1 (非契約)
    assert tracking_tube(g, w_bar=0.05).contraction_ok is False
    res = dc.tube_cross_check(g, w_bar=0.05, seq_len=128, dim=8, n_seeds=16)
    assert res.contraction_ok is False
    assert res.certified_tube == float("inf")
    assert res.tube_holds is False  # tube=∞ で vacuous に True にしない


def test_rollout_port_runs_on_scalar_substrate() -> None:
    """rollout_with_disturbance が scalar run_sequence 基板で動く (port 健全性)."""
    dc = _load_disturbance_checker()
    g = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.1)
    x_ref = np.random.default_rng(0).uniform(-1, 1, size=(32, 8))
    traj = dc.rollout_with_disturbance(g, x_ref, w_bar=0.1, seed=42)
    assert traj.shape == (33, 8)  # L+1 step
    assert np.all(np.isfinite(traj))
    # w_bar=0 (外乱なし) なら参照軌道と完全一致 (port が同一基板を使う証拠)。
    traj0 = dc.rollout_with_disturbance(g, x_ref, w_bar=0.0, seed=42)
    s_ref = run_sequence(x_ref, g)
    np.testing.assert_allclose(traj0, s_ref, atol=0.0)


def test_rollout_negative_wbar_raises() -> None:
    """rollout は w_bar<0 で fail-loud."""
    dc = _load_disturbance_checker()
    g = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.1)
    x_ref = np.zeros((4, 8))
    with pytest.raises(ValueError, match="non-negative"):
        dc.rollout_with_disturbance(g, x_ref, w_bar=-0.1, seed=0)


# ---------------------------------------------------------------------------
# (P3) bridge anchor: named-slot write == eval_step (数値恒等)
# ---------------------------------------------------------------------------


def named_slot_write(slot_value: np.ndarray, write_delta: np.ndarray, gene: StateUpdateGene) -> np.ndarray:
    """named-slot write 写像 (bridge anchor; 設計 doc §4.2 補助不変量).

    relabel: slot value = state s, write delta = input x。これは eval_step と
    **literal に同一**であるべき (bridge が proven object = 実際の memory write step で
    あることの唯一正当な anchor)。「bank」「external」「cross-slot」「retrieval」の語は
    使わない (設計 doc §3.2 / §4.6-1)。
    """
    return eval_step(slot_value, write_delta, gene)


@pytest.mark.parametrize(
    "gene",
    [
        StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.0),
        StateUpdateGene(decay=0.5, mix=1.0, gate_str=0.3),
        StateUpdateGene(decay=0.0, mix=-0.7, gate_str=1.5),
    ],
)
def test_p3_named_slot_write_identical_to_eval_step(gene: StateUpdateGene) -> None:
    """(P3) named-slot write が eval_step と数値恒等 (bridge が虚構/fork でない証拠)."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        s = rng.uniform(-1, 1, size=8)
        x = rng.uniform(-1, 1, size=8)
        got = named_slot_write(s, x, gene)
        expected = eval_step(s, x, gene)
        # literal 同一 (bit-for-bit; 同じ関数を呼ぶので atol=0)。
        np.testing.assert_array_equal(got, expected)


def test_p3_bridge_not_a_fork_of_run_sequence() -> None:
    """(P3) named-slot write を逐次適用すると run_sequence と一致 (bridge が独立 fork でない)."""
    gene = StateUpdateGene(decay=0.6, mix=0.4, gate_str=0.2)
    rng = np.random.default_rng(11)
    inputs = rng.uniform(-1, 1, size=(16, 8))
    # named-slot write を逐次適用 (slot = state)。
    s = np.zeros(8)
    manual = [s.copy()]
    for t in range(inputs.shape[0]):
        s = named_slot_write(s, inputs[t], gene)
        manual.append(s.copy())
    manual_arr = np.array(manual)
    np.testing.assert_array_equal(manual_arr, run_sequence(inputs, gene))
