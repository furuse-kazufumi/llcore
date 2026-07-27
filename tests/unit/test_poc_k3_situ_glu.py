# SPDX-License-Identifier: Apache-2.0
"""PoC K3-2: SiTU-GLU (有界 GLU) の unit tests.

Kimi K3 技術レポート §2.3.2 より::

    SiTU-GLU(x) = [β1 tanh(Wg x / β1) ⊙ Sigmoid(Wg x)] ⊙ [β2 tanh(Wu x / β2)]

SwiGLU は gate 側 ``x·σ(x)`` も up 側も非有界なので、両方が同時に大きくなると
外れ値が出て低精度演算で overflow しやすい。SiTU-GLU は ``softcap(x,β)=β·tanh(x/β)``
を両枝に掛けて **出力を |f| <= β1·β2 に有界化**しつつ、原点近傍では SwiGLU に
一致する応答を保つ (K3 は β1=4, β2=25 なので上界 100)。

llcore にとっての意味: RWKV の channel-mix は ``squared-ReLU`` = 非有界かつ二次で、
低ビット化の際に最も危険な形。有界版は形式検証とも相性が良い。
"""
from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from llcore.lm.activations import situ_glu, softcap


# ── G1: softcap プリミティブ ───────────────────────────────────────────
def test_g1_softcap_is_bounded_by_beta() -> None:
    """[G1] softcap(x, β) は |出力| <= β を超えない."""
    x = torch.tensor([-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6])
    y = softcap(x, beta=4.0)

    assert torch.all(y.abs() <= 4.0), f"β を超えた: {y}"


def test_g1b_softcap_is_near_identity_at_origin() -> None:
    """[G1b] 原点近傍では softcap(x, β) ≈ x (局所応答を壊さない)."""
    x = torch.tensor([-0.05, -0.01, 0.0, 0.01, 0.05])
    y = softcap(x, beta=25.0)

    assert torch.allclose(y, x, atol=1e-5), f"原点近傍で恒等から外れた: {y - x}"


def test_g1c_softcap_rejects_nonpositive_beta() -> None:
    """[G1c] β <= 0 は cap にならないので拒否する."""
    with pytest.raises(ValueError, match="beta"):
        softcap(torch.zeros(3), beta=0.0)


# ── G2: SiTU-GLU の有界性 (本命) ───────────────────────────────────────
def test_g2_situ_glu_output_is_bounded_by_beta_product() -> None:
    """[G2] 両枝が極端でも |SiTU-GLU| <= β1*β2 を超えない."""
    beta_gate, beta_up = 4.0, 25.0
    grid = torch.tensor([-1e6, -1e3, -10.0, 0.0, 10.0, 1e3, 1e6])
    gate_pre, up_pre = torch.meshgrid(grid, grid, indexing="ij")

    y = situ_glu(gate_pre, up_pre, beta_gate=beta_gate, beta_up=beta_up)

    assert torch.all(torch.isfinite(y)), "非有限値が出た"
    assert torch.all(y.abs() <= beta_gate * beta_up), f"上界 {beta_gate * beta_up} を超えた: {y.abs().max()}"


def test_g2b_swiglu_is_unbounded_where_situ_glu_is_not() -> None:
    """[G2b] 対照: 同じ入力で SwiGLU は上界 β1*β2 を大きく超える (動機の実証)."""
    beta_gate, beta_up = 4.0, 25.0
    gate_pre = torch.tensor([1e3])
    up_pre = torch.tensor([1e3])

    swiglu = F.silu(gate_pre) * up_pre
    situ = situ_glu(gate_pre, up_pre, beta_gate=beta_gate, beta_up=beta_up)

    assert float(swiglu.abs()) > beta_gate * beta_up * 100, "SwiGLU が有界に見える (対照不成立)"
    assert float(situ.abs()) <= beta_gate * beta_up


def test_g2c_situ_glu_tracks_swiglu_near_origin() -> None:
    """[G2c] 原点近傍では SwiGLU とほぼ一致する (局所応答の保存)."""
    beta_gate, beta_up = 4.0, 25.0
    pre = torch.tensor([-0.2, -0.1, 0.0, 0.1, 0.2])

    swiglu = F.silu(pre) * pre
    situ = situ_glu(pre, pre, beta_gate=beta_gate, beta_up=beta_up)

    assert torch.allclose(situ, swiglu, atol=1e-3), f"原点近傍で乖離: {situ - swiglu}"


def test_g2d_situ_glu_attains_the_bound_only_in_the_limit() -> None:
    """[G2d] 上界は極限でのみ到達する (float 飽和で等号になり得る点を明示)."""
    beta_gate, beta_up = 4.0, 25.0
    y = situ_glu(
        torch.tensor([1e6]), torch.tensor([1e6]), beta_gate=beta_gate, beta_up=beta_up
    )

    assert float(y) == pytest.approx(beta_gate * beta_up, rel=1e-5)


def test_g2e_situ_glu_rejects_nonpositive_beta() -> None:
    """[G2e] β <= 0 は拒否する."""
    with pytest.raises(ValueError, match="beta"):
        situ_glu(torch.zeros(2), torch.zeros(2), beta_gate=4.0, beta_up=-1.0)


# ── G3: 解析上界と実測の整合 ───────────────────────────────────────────
def test_g3_analytic_bound_matches_empirical_max() -> None:
    """[G3] ランダム入力での実測 max が解析上界を超えず、桁として妥当."""
    torch.manual_seed(0)
    beta_gate, beta_up = 4.0, 25.0
    pre = torch.randn(4096) * 50.0

    y = situ_glu(pre, pre, beta_gate=beta_gate, beta_up=beta_up)

    assert float(y.abs().max()) <= beta_gate * beta_up
    assert float(y.abs().max()) > 0.0, "全部ゼロは異常"
    assert math.isfinite(float(y.sum()))


# ── G4: Z3 による出力有界の証明 ───────────────────────────────────────
def test_g4_z3_proves_situ_glu_output_bound() -> None:
    """[G4] |SiTU-GLU| <= β_gate*β_up を Z3 が証明する (unsat = 反例なし)."""
    from llcore.verifier import is_z3_available, verify_activation_bound

    if not is_z3_available():
        pytest.skip("z3-solver not installed")

    r = verify_activation_bound(beta_gate=4.0, beta_up=25.0)

    assert r.ok
    assert r.used_z3
    assert r.solver_status == "unsat"


def test_g4b_z3_rejects_a_bound_tighter_than_the_product() -> None:
    """[G4b] soundness: 積より小さい上界を主張すると Z3 が反例を出す."""
    from llcore.verifier import is_z3_available, verify_activation_bound

    if not is_z3_available():
        pytest.skip("z3-solver not installed")

    r = verify_activation_bound(beta_gate=4.0, beta_up=25.0, claimed_bound=10.0)

    assert not r.ok
    assert r.solver_status == "sat"
    assert r.counterexample is not None


def test_g4c_verify_activation_bound_rejects_bad_beta() -> None:
    """[G4c] β <= 0 は拒否する."""
    from llcore.verifier import verify_activation_bound

    with pytest.raises(ValueError, match="beta"):
        verify_activation_bound(beta_gate=0.0, beta_up=25.0)
