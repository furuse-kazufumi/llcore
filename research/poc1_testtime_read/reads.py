# SPDX-License-Identifier: Apache-2.0
"""PoC-1 Stage 2 read 変種 — 凍結状態 S [B,sd,sd] と query q̂ [B,sd] から値ベクトル o [B,sd] を読む。

規約: S[b,i,j] は key 空間 (index j) → value 空間 (index i)。R0: o_i = Σ_j S[b,i,j] q_j。
  - S を value ベクトル o に転置適用: (Sᵀo)_j = Σ_i S[b,i,j] o_i = einsum('bij,bi->bj', S, o)。
全変種は **状態 S を一切変えない**(read-only test-time)。softmax は非 softmax 変種の対照としてのみ使う。

pre-reg §3.2 / §3.3。honest: R-ISTA/R-Hopfield は sparse-coding/modern-Hopfield の spirit
(機構 novelty は主張しない)。貢献は「学習・非直交 key の凍結蒸留状態への post-hoc 反復 read」regime。
"""
from __future__ import annotations

import torch


def _St(S: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Sᵀ を value ベクトル o[B,sd] に適用し key 空間 [B,sd] を返す。"""
    return torch.einsum("bij,bi->bj", S, o)


def _S(S: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """S を key ベクトル a[B,sd] に適用し value 空間 [B,sd] を返す。"""
    return torch.einsum("bij,bj->bi", S, a)


def _soft_threshold(z: torch.Tensor, t: float) -> torch.Tensor:
    return torch.sign(z) * torch.clamp(z.abs() - t, min=0.0)


def _sparsemax(z: torch.Tensor) -> torch.Tensor:
    """sparsemax (Martins & Astudillo 2016) — 非 softmax の疎確率射影。dim=-1。"""
    zs, _ = torch.sort(z, dim=-1, descending=True)
    rng = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
    cssv = zs.cumsum(-1) - 1.0
    cond = 1.0 + rng * zs > cssv
    k = cond.sum(-1, keepdim=True).clamp(min=1)
    tau = cssv.gather(-1, k - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


# ─── 変種 (すべて S 不変) ─────────────────────────────────────────

def r0(S: torch.Tensor, qhat: torch.Tensor) -> torch.Tensor:
    """R0: 単発線形 read o = S q̂ (ベースライン = セル native read)。"""
    return _S(S, qhat)


def r_ccq(S: torch.Tensor, qhat: torch.Tensor, *, lam: float) -> torch.Tensor:
    """R-CCQ: 単発 curvature contraction (2606.01294 再現・kill-risk 対照)。

    key 相関 C=SᵀS で query を 1 回だけ decorrelate: q' = (I − λC) q̂, o = S q'。反復無し。
    """
    C_q = _St(S, _S(S, qhat))  # SᵀS q̂  (key 空間)
    q2 = qhat - lam * C_q
    return _S(S, q2)


def r_hopfield(S: torch.Tensor, qhat: torch.Tensor, *, K: int, tau: float) -> torch.Tensor:
    """R-Hopfield: K step 非 softmax 連想 cleanup (snap-to-stored-value)。

    o0=S q̂ から反復: a=softthreshold(Sᵀo, τ) (key 活性を疎化=支配 key を強調) → o=S a。
    """
    o = _S(S, qhat)
    for _ in range(K):
        a = _soft_threshold(_St(S, o), tau)
        norm = a.abs().sum(-1, keepdim=True).clamp(min=1e-6)
        o = _S(S, a / norm)
    return o


def r_ista(S: torch.Tensor, qhat: torch.Tensor, *, K: int, lam: float, eta: float) -> torch.Tensor:
    """R-ISTA: K step の unrolled soft-threshold 疎復元 (NOODL 流 linear+soft-threshold)。

    o0=S q̂ を測定と見なし、S x≈o0 を満たす疎 key 活性 x を ISTA で復元 → o=S x (crosstalk 除去)。
    """
    o0 = _S(S, qhat)
    x = torch.zeros_like(qhat)
    for _ in range(K):
        resid = _S(S, x) - o0                 # S x − o0  (value)
        grad = _St(S, resid)                  # Sᵀ(S x − o0)  (key)
        x = _soft_threshold(x - eta * grad, lam)
    return _S(S, x)


# ─── mandated baselines (深掘り verdict 指定) ─────────────────────

def r_softmax_hopfield(S: torch.Tensor, qhat: torch.Tensor, *, beta: float) -> torch.Tensor:
    """softmax modern-Hopfield read (2008.02217 系, softmax energy 1-step)。非softmax変種の対照。"""
    a = torch.softmax(beta * _St(S, _S(S, qhat)), dim=-1)
    return _S(S, a)


def r_fy_hopfield(S: torch.Tensor, qhat: torch.Tensor) -> torch.Tensor:
    """sparse Fenchel-Young Hopfield read (2411.08590, 非softmax sparse retrieval = sparsemax)。"""
    a = _sparsemax(_St(S, _S(S, qhat)))
    return _S(S, a)
