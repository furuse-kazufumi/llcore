# SPDX-License-Identifier: Apache-2.0
"""Unit tests for T1 Phase 1 (b) — tracking tube reporter (additive, read-only).

検証範囲:
- (i)   PoC 結果 JSON の case A/B/C (cert_inf PASS) の L/G/tube が golden 値と一致。
- (ii)  case D (非契約, L>=1) で contraction_ok=False かつ tube_radius=+∞ (保証なし)。
- (iii) scalar StateUpdateGene (n=1) でも L/G/tube が正しく計算され、cert_inf 整合。
- (iv)  ゲート判定 admits: contraction_ok ∧ tube<=r_max (fail-closed)。
- (v)   API 形状 (frozen dataclass) + 入力ガード (w_bar<0 で fail-loud)。
- (vi)  read-only 性: 既存 certifies() / verify_gene_safe を一切変えない (regression)。

golden 値は research/target_trajectory_poc/poc_target_trajectory_results.json より。

実行::

    pytest tests/unit/test_tracking_tube.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from llcore.state_update import StateUpdateGene
from llcore.verifier import (
    InfNormBackend,
    TrackingTubeResult,
    input_gain_inf,
    state_lipschitz_inf,
    tracking_tube,
    verify_gene_safe,
)


# ---------------------------------------------------------------------------
# 小さな coupled gene stub (PoC CoupledNDGene と同 duck-typed interface)。
# backends._coupled_arrays は decay / W / 任意 V を読む。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CoupledGene:
    decay: np.ndarray
    W: np.ndarray
    V: np.ndarray

    @staticmethod
    def make(decay, W, V=None) -> "_CoupledGene":
        decay = np.asarray(decay, dtype=np.float64).reshape(-1)
        n = decay.shape[0]
        W = np.asarray(W, dtype=np.float64).reshape(n, n)
        V = np.eye(n) if V is None else np.asarray(V, dtype=np.float64).reshape(n, n)
        return _CoupledGene(decay=decay, W=W, V=V)


# PoC golden cases (poc_target_trajectory_results.json より厳密値)。
_CASE_A = _CoupledGene.make([0.9, 0.85], [[0.1, 0.05], [0.0, 0.12]])
_CASE_B = _CoupledGene.make(
    [0.7, 0.75, 0.6], [[0.2, 0.1, 0.0], [0.05, 0.15, 0.1], [0.0, 0.1, 0.2]]
)
_CASE_C = _CoupledGene.make([0.4, 0.4], [[0.3, 0.2], [0.2, 0.3]])
_CASE_D = _CoupledGene.make([0.15, 0.15], [[1.2, 0.6], [0.6, 1.2]])


# ---------------------------------------------------------------------------
# (i) golden 値一致 (cert_inf PASS の A/B/C)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gene, w_bar, L_gold, G_gold, tube_gold",
    [
        # name, w_bar, L, G, theoretical_tube_radius (all from PoC results JSON)
        (_CASE_A, 0.05, 0.915, 0.15000000000000002, 0.0882352941176471),
        (_CASE_B, 0.05, 0.825, 0.4, 0.11428571428571428),
        (_CASE_C, 0.05, 0.7000000000000001, 0.6, 0.10000000000000002),
    ],
)
def test_golden_tube_values(gene, w_bar, L_gold, G_gold, tube_gold) -> None:
    """[i] cert_inf PASS gene の L/G/tube が PoC golden 値と一致 (案 B 移植の忠実性)."""
    r = tracking_tube(gene, w_bar=w_bar)
    assert r.contraction_ok is True
    assert r.L_state == pytest.approx(L_gold, abs=1e-12)
    assert r.G_input == pytest.approx(G_gold, abs=1e-12)
    assert r.tube_radius == pytest.approx(tube_gold, abs=1e-12)
    assert np.isfinite(r.tube_radius)


def test_golden_helpers_match_case_a() -> None:
    """[i] state_lipschitz_inf / input_gain_inf 単体も golden 値を返す."""
    assert state_lipschitz_inf(_CASE_A) == pytest.approx(0.915, abs=1e-12)
    assert input_gain_inf(_CASE_A) == pytest.approx(0.15, abs=1e-12)


# ---------------------------------------------------------------------------
# (ii) non-contract (case D) → tube=inf, no guarantee
# ---------------------------------------------------------------------------


def test_noncontract_tube_is_infinite() -> None:
    """[ii] 非契約 gene (L>=1) は contraction_ok=False かつ tube=+∞ (保証なし)."""
    r = tracking_tube(_CASE_D, w_bar=0.2)
    assert r.contraction_ok is False
    assert r.L_state == pytest.approx(1.68, abs=1e-12)
    assert r.G_input == pytest.approx(0.85, abs=1e-12)
    assert r.tube_radius == float("inf")
    assert r.admits is False  # tube=inf は never admit (fail-closed)


def test_noncontract_admits_false_even_without_rmax() -> None:
    """[ii] r_max 未指定でも非契約は admit されない (tube 非有限)."""
    r = tracking_tube(_CASE_D, w_bar=0.2, r_max=None)
    assert r.admits is False


# ---------------------------------------------------------------------------
# (iii) scalar StateUpdateGene (n=1)
# ---------------------------------------------------------------------------


def test_scalar_gene_tube_matches_closed_form() -> None:
    """[iii] scalar StateUpdateGene でも L/G/tube が閉形式で正しい.

    scalar 更新 s' = decay·s + (1−decay)·tanh(gate_str·s + mix·x):
      n=1 持ち上げ W=[[gate_str]], V=[[mix]]。
      g=(decay=0.9, mix=0.5, gate_str=0.0): L=|0.9 + 0.1·t·0|=0.9 (t-independent),
      G=(1−0.9)·|0.5|=0.05。tube = G·w̄/(1−L) = 0.05·0.1/0.1 = 0.05。
    """
    g = StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.0)
    r = tracking_tube(g, w_bar=0.1)
    assert r.contraction_ok is True
    assert r.L_state == pytest.approx(0.9, abs=1e-9)
    assert r.G_input == pytest.approx(0.05, abs=1e-9)
    assert r.tube_radius == pytest.approx(0.05, abs=1e-9)


def test_scalar_noncontract_gene_infinite_tube() -> None:
    """[iii] scalar 非契約 gene (gate_str=2) は tube=+∞."""
    g = StateUpdateGene(decay=0.0, mix=1.0, gate_str=2.0)
    r = tracking_tube(g, w_bar=0.05)
    assert r.contraction_ok is False
    assert r.tube_radius == float("inf")


def test_scalar_contraction_ok_iff_L_lt_one() -> None:
    """[iii] tube の contraction_ok は cert_inf 基準 L<1 と同値 (scalar gene)."""
    for g in [
        StateUpdateGene(0.9, 0.5, 0.0),
        StateUpdateGene(0.5, 0.5, 0.5),
        StateUpdateGene(0.0, 1.0, 2.0),
        StateUpdateGene(0.95, 0.3, 0.1),
    ]:
        r = tracking_tube(g, w_bar=0.05)
        assert isinstance(r.contraction_ok, bool)
        assert r.contraction_ok == (r.L_state < 1.0)


def test_coupled_contraction_ok_matches_infnorm_backend() -> None:
    """[iii] coupled gene では contraction_ok が InfNormBackend.certifies と一致.

    coupled gene は ``W`` を持つため backend が正しく読める。tube reporter の
    cert_inf 基準 (L<1) は InfNormBackend の判定基準と厳密同一であることを確認する。
    """
    backend = InfNormBackend()
    for g in (_CASE_A, _CASE_B, _CASE_C, _CASE_D):
        r = tracking_tube(g, w_bar=0.05)
        assert r.contraction_ok == bool(backend.certifies(g))


# ---------------------------------------------------------------------------
# (iv) admits gate semantics (fail-closed)
# ---------------------------------------------------------------------------


def test_admits_within_rmax() -> None:
    """[iv] tube<=r_max なら admit、超えたら reject (fail-closed AND)."""
    # case A tube = 0.0882...
    r_in = tracking_tube(_CASE_A, w_bar=0.05, r_max=0.1)
    assert r_in.admits is True
    r_out = tracking_tube(_CASE_A, w_bar=0.05, r_max=0.05)
    assert r_out.admits is False  # 0.088 > 0.05
    assert r_out.contraction_ok is True  # 契約自体は PASS (tube が大きいだけ)


def test_admits_true_without_rmax_when_contracting() -> None:
    """[iv] r_max=None かつ契約済みなら admit (tube 有限)."""
    r = tracking_tube(_CASE_A, w_bar=0.05)
    assert r.admits is True


# ---------------------------------------------------------------------------
# (v) API 形状 + 入力ガード
# ---------------------------------------------------------------------------


def test_result_is_frozen() -> None:
    """[v] TrackingTubeResult は frozen dataclass."""
    r = tracking_tube(_CASE_A, w_bar=0.05)
    assert isinstance(r, TrackingTubeResult)
    with pytest.raises(Exception):
        r.tube_radius = 0.0  # type: ignore[misc]


def test_negative_w_bar_raises() -> None:
    """[v] w_bar<0 は fail-loud (ValueError)."""
    with pytest.raises(ValueError, match="non-negative"):
        tracking_tube(_CASE_A, w_bar=-0.01)


def test_zero_w_bar_gives_zero_tube() -> None:
    """[v] w_bar=0 (外乱なし) なら tube=0 (完全追従)."""
    r = tracking_tube(_CASE_A, w_bar=0.0)
    assert r.tube_radius == pytest.approx(0.0, abs=1e-15)
    assert r.contraction_ok is True


def test_tube_scales_linearly_with_w_bar() -> None:
    """[v] tube は w̄ に線形 (r = G·w̄/(1−L))."""
    r1 = tracking_tube(_CASE_A, w_bar=0.05)
    r2 = tracking_tube(_CASE_A, w_bar=0.10)
    assert r2.tube_radius == pytest.approx(2.0 * r1.tube_radius, rel=1e-12)


# ---------------------------------------------------------------------------
# (vi) read-only / regression: 既存 API を一切変えない
# ---------------------------------------------------------------------------


def test_reporter_does_not_mutate_existing_api() -> None:
    """[vi] tracking_tube 呼び出し後も verify_gene_safe / InfNormBackend が無改変で動く."""
    g = StateUpdateGene(0.9, 0.5, 0.0)
    before = verify_gene_safe(g).ok
    _ = tracking_tube(g, w_bar=0.05)
    after = verify_gene_safe(g).ok
    assert before == after is True

    backend = InfNormBackend()
    # coupled gene の certifies は tube reporter 前後で不変。
    v1 = backend.certifies(_CASE_A)
    _ = tracking_tube(_CASE_A, w_bar=0.05)
    v2 = backend.certifies(_CASE_A)
    assert v1 == v2 is True
