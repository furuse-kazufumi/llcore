# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PoC 1b — 状態方向 Lipschitz contraction invariant verifier.

検証範囲:
- (i)   既知 contractive gene (decay 大・gate_str 小) が certified (L<1)
- (ii)  既知 non-contractive gene (decay 小・gate_str=2) が reject
- (iii) z3 / no-z3 両 path (fail-safe)
- (iv)  empirical_lipschitz <= Z3 解析上界 (L_upper_bound) — 健全性のクロスチェック
- (v)   contraction certified ⟹ verify_state_norm / verify_gene_safe も ok (整合性)
"""
from __future__ import annotations

import numpy as np
import pytest

from llcore.state_update import StateUpdateGene, run_sequence
from llcore.verifier import (
    LipschitzResult,
    empirical_lipschitz,
    is_z3_available,
    verify_gene_safe,
    verify_lipschitz_contraction,
    verify_state_norm_invariant,
)


# ---------------------------------------------------------------------------
# (i) contractive gene → certified
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gene",
    [
        StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.0),
        StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5),
        StateUpdateGene(decay=0.95, mix=0.3, gate_str=0.1),
        StateUpdateGene(decay=0.8, mix=0.2, gate_str=0.2),
    ],
)
def test_contractive_gene_certified(gene: StateUpdateGene) -> None:
    """[i] 既知 contractive gene が L<1 certified (Z3 unsat)."""
    r = verify_lipschitz_contraction(gene)
    assert r.contraction is True, f"{gene} expected certified: {r.reason}"
    assert r.used_z3
    assert r.solver_status == "unsat"
    assert r.L_upper_bound is not None and r.L_upper_bound < 1.0


# ---------------------------------------------------------------------------
# (ii) non-contractive gene → reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gene",
    [
        # decay 小 + gate_str=2 → J=2t, sup=2 で reject
        StateUpdateGene(decay=0.0, mix=1.0, gate_str=2.0),
        # decay 大でも gate_str=2 で L=decay+(1-decay)*2 > 1
        StateUpdateGene(decay=0.9, mix=1.0, gate_str=2.0),
        # state_norm は通すが contraction を破る DESIGN の代表例
        StateUpdateGene(decay=0.0, mix=0.0, gate_str=2.0),
    ],
)
def test_non_contractive_gene_reject(gene: StateUpdateGene) -> None:
    """[ii] 既知 non-contractive gene が reject (Z3 sat)."""
    r = verify_lipschitz_contraction(gene)
    assert r.contraction is False, f"{gene} expected reject: {r.reason}"
    assert r.used_z3
    assert r.solver_status == "sat"
    assert r.L_upper_bound is not None and r.L_upper_bound >= 1.0


def test_marginal_decay_one_rejected() -> None:
    """[ii-boundary] decay=1 (純記憶, gate=0) は L=1 ちょうど → strict L<1 で reject (honest)."""
    g = StateUpdateGene(decay=1.0, mix=0.0, gate_str=0.0)
    r = verify_lipschitz_contraction(g)
    # L=1 で |J|>=1 が sat (端点 t で J=1) → 保守的 reject
    assert r.contraction is False
    assert r.L_upper_bound == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# (iii) z3 / no-z3 path
# ---------------------------------------------------------------------------


def test_z3_path_uses_z3() -> None:
    """[iii] z3 利用可なら used_z3=True で実際に解く."""
    if not is_z3_available():
        pytest.skip("z3 not installed in this environment")
    r = verify_lipschitz_contraction(StateUpdateGene(0.9, 0.5, 0.0))
    assert r.used_z3
    assert r.solver_status in ("sat", "unsat")


def test_no_z3_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """[iii] z3 不在では fail-safe: used_z3=False, contraction=None (未判定/assumed)."""
    import llcore.verifier.invariants as inv_mod

    monkeypatch.setattr(inv_mod, "_HAS_Z3", False)
    r = inv_mod.verify_lipschitz_contraction(StateUpdateGene(0.9, 0.5, 0.0))
    assert not r.used_z3
    assert r.contraction is None  # 未検証 = fail-closed 側で扱う
    assert r.L_upper_bound is not None  # 閉形式上界は z3 不在でも計算できる
    assert "z3 not installed" in r.reason


def test_no_z3_upper_bound_still_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    """[iii] z3 不在でも L_upper_bound は閉形式で正しく計算される."""
    import llcore.verifier.invariants as inv_mod

    monkeypatch.setattr(inv_mod, "_HAS_Z3", False)
    # d=0.5, g=0.5: L = max(0.5, 0.5+0.5*0.5)=max(0.5,0.75)=0.75
    r = inv_mod.verify_lipschitz_contraction(StateUpdateGene(0.5, 0.5, 0.5))
    assert r.L_upper_bound == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# (iv) empirical_lipschitz <= Z3 上界 (健全性クロスチェック)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gene",
    [
        StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.0),
        StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5),
        StateUpdateGene(decay=0.0, mix=1.0, gate_str=2.0),
        StateUpdateGene(decay=0.9, mix=1.0, gate_str=2.0),
        StateUpdateGene(decay=0.3, mix=-0.7, gate_str=-1.5),
        StateUpdateGene(decay=0.7, mix=1.0, gate_str=1.0),
    ],
)
def test_empirical_le_z3_upper_bound(gene: StateUpdateGene) -> None:
    """[iv] 経験 Lipschitz <= Z3 解析上界 (over-approx の健全性)."""
    r = verify_lipschitz_contraction(gene)
    emp = empirical_lipschitz(gene, n_samples=5000, seed=0)
    assert r.L_upper_bound is not None
    # 経験値は解析上界を (数値誤差以内で) 超えてはならない
    assert emp <= r.L_upper_bound + 1e-6, (
        f"{gene}: empirical L={emp:.6f} > Z3 upper bound {r.L_upper_bound:.6f}"
    )


def test_empirical_lipschitz_deterministic() -> None:
    """[iv] empirical_lipschitz は seed 固定で決定論的."""
    g = StateUpdateGene(0.5, 0.5, 0.5)
    a = empirical_lipschitz(g, seed=3)
    b = empirical_lipschitz(g, seed=3)
    assert a == b


# ---------------------------------------------------------------------------
# (v) contraction certified ⟹ state_norm / verify_gene_safe も ok
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gene",
    [
        StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.0),
        StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5),
        StateUpdateGene(decay=0.95, mix=0.3, gate_str=0.1),
        StateUpdateGene(decay=0.8, mix=-0.4, gate_str=0.2),
    ],
)
def test_contraction_implies_state_norm(gene: StateUpdateGene) -> None:
    """[v] L<1 certified ⟹ verify_gene_safe (state_norm) も ok (Banach + convex comb)."""
    lr = verify_lipschitz_contraction(gene)
    assert lr.contraction is True, f"precondition: {gene} should be certified"
    sn = verify_gene_safe(gene)
    assert sn.ok, f"contraction certified but state_norm rejected: {sn.reason}"


def test_contraction_implies_bounded_simulation() -> None:
    """[v] certified gene は長系列シミュレーションでも |state| <= 1+eps."""
    g = StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.1)
    assert verify_lipschitz_contraction(g).contraction is True
    rng = np.random.default_rng(11)
    inputs = rng.uniform(-1.0, 1.0, size=(500, 8))
    states = run_sequence(inputs, g, initial_state=rng.uniform(-1.0, 1.0, size=8))
    assert float(np.max(np.abs(states))) <= 1.0 + 1e-6


def test_state_norm_does_not_imply_contraction() -> None:
    """[distinctness] state_norm は通すが contraction は破る gene が存在する.

    DESIGN の代表例: gene=(decay=0, gate_str=2)。state_norm は |tanh(2s)|<=1 で admit、
    しかし L=2 で contraction reject = 質的に強い不変量である証拠。
    """
    g = StateUpdateGene(decay=0.0, mix=0.0, gate_str=2.0)
    sn = verify_gene_safe(g)
    lr = verify_lipschitz_contraction(g)
    assert sn.ok, "state_norm should admit this gene"
    assert lr.contraction is False, "contraction should reject this gene"


# ---------------------------------------------------------------------------
# dataclass / API 形状
# ---------------------------------------------------------------------------


def test_lipschitz_result_frozen() -> None:
    """LipschitzResult は frozen dataclass."""
    r = LipschitzResult(
        contraction=True,
        L_upper_bound=0.5,
        used_z3=True,
        solver_status="unsat",
        reason="x",
    )
    with pytest.raises(Exception):
        r.contraction = False  # type: ignore[misc]


def test_existing_state_norm_unaffected() -> None:
    """[regression] 既存 verify_state_norm_invariant が無改変で従来通り unsat を返す."""
    r = verify_state_norm_invariant(max_input_abs=1.0, state_bound=1.0, timeout_ms=2000)
    assert r.ok
    assert r.used_z3
    assert "unsat" in r.reason


def test_timeout_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """[fail-closed] z3 unknown/timeout は contraction=False (reject 側)."""
    if not is_z3_available():
        pytest.skip("z3 not installed")
    import llcore.verifier.invariants as inv_mod

    real_solver = inv_mod.z3.Solver

    class _UnknownSolver:
        def __init__(self) -> None:
            self._s = real_solver()

        def set(self, *a, **k):  # noqa: ANN002, ANN003
            return self._s.set(*a, **k)

        def add(self, *a, **k):  # noqa: ANN002, ANN003
            return self._s.add(*a, **k)

        def check(self):  # noqa: ANN201
            return inv_mod.z3.unknown

    monkeypatch.setattr(inv_mod.z3, "Solver", _UnknownSolver)
    r = inv_mod.verify_lipschitz_contraction(StateUpdateGene(0.9, 0.5, 0.0))
    assert r.contraction is False
    assert r.solver_status == "unknown"
    assert r.used_z3
