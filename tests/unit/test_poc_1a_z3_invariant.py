# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PoC 1a Z3 state-norm invariant verifier."""
from __future__ import annotations

import numpy as np
import pytest

from llcore.state_update import StateUpdateGene
from llcore.verifier import (
    InvariantResult,
    is_z3_available,
    verify_gene_safe,
    verify_state_norm_invariant,
)


def test_g1_z3_available() -> None:
    """[G1] z3-solver が import 可能 (optional dep 入っている前提)."""
    assert is_z3_available()


def test_g2_clip_range_invariant_unsat() -> None:
    """[G2] clip 範囲下で |state|<=1 が証明される (unsat = invariant 成立)."""
    r = verify_state_norm_invariant(max_input_abs=1.0, state_bound=1.0, timeout_ms=2000)
    assert r.ok
    assert r.used_z3
    assert "unsat" in r.reason


def test_g3_illegal_range_finds_counterexample() -> None:
    """[G3] decay=2 (illegal) で Z3 が反例を検出 (sound)."""
    import z3

    solver = z3.Solver()
    solver.set("timeout", 2000)
    decay = z3.Real("decay")
    s = z3.Real("s")
    tanh_val = z3.Real("tanh_val")
    solver.add(decay == 2)
    solver.add(s == 1)
    solver.add(tanh_val >= -1, tanh_val <= 1)
    s_next = decay * s + (1 - decay) * tanh_val
    solver.add(z3.Or(s_next > 1, s_next < -1))
    assert solver.check() == z3.sat


def test_g4_verify_gene_safe_admits_clipped() -> None:
    """[G4] clipped gene 5 個が全て admit される."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        r = verify_gene_safe(gene)
        assert r.ok, f"gene {gene} rejected: {r.reason}"


def test_g5_online_gate_reject_rate_zero() -> None:
    """[G5] 20 random clipped gene 全部 admit (理論的 reject 率 = 0)."""
    rng = np.random.default_rng(43)
    rejects = 0
    for _ in range(20):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        r = verify_gene_safe(gene)
        if not r.ok:
            rejects += 1
    assert rejects == 0


def test_g6_practical_timeout() -> None:
    """[G6] verify_state_norm_invariant が 1 sec 以内."""
    import time as _t

    start = _t.perf_counter()
    r = verify_state_norm_invariant()
    elapsed = _t.perf_counter() - start
    assert elapsed < 1.0
    assert r.ok


def test_g7_determinism() -> None:
    """[G7] 同 query 2 回で同結果."""
    r1 = verify_state_norm_invariant()
    r2 = verify_state_norm_invariant()
    assert r1.ok == r2.ok
    assert r1.used_z3 == r2.used_z3


def test_invariant_result_frozen() -> None:
    """InvariantResult は frozen."""
    r = InvariantResult(ok=True, used_z3=False, reason="x")
    with pytest.raises(Exception):
        r.ok = False  # type: ignore[misc]


def test_state_bound_strict() -> None:
    """state_bound=0.5 で convex combination が 0.5 を超える反例があるはず."""
    # state_bound=0.5 で |s| <= 0.5 ⇒ |s_next| <= decay*0.5 + (1-decay)*1 = 1 - 0.5*decay
    # 1 - 0.5*decay > 0.5 ⇔ decay < 1 で大きく違反する可能性
    r = verify_state_norm_invariant(max_input_abs=1.0, state_bound=0.5)
    # state_bound=0.5 では tanh の bound 1 が大きすぎるので反例出る (sat = ok=False)
    assert not r.ok
    assert r.counterexample is not None


def test_z3_unavailable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Z3 が無い場合 fallback で ok=True, used_z3=False を返す."""
    import llcore.verifier.invariants as inv_mod

    monkeypatch.setattr(inv_mod, "_HAS_Z3", False)
    r = inv_mod.verify_state_norm_invariant()
    assert r.ok
    assert not r.used_z3
    assert "not installed" in r.reason or "skip" in r.reason.lower()
