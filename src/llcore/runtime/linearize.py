# SPDX-License-Identifier: Apache-2.0
"""Internal attention surgery: swap Qwen2 softmax attention for constant-state linear attention.

This is the deep-internal research lever for bounded-memory × unbounded-context conversation.
Standard attention keeps a KV cache that grows O(T) with the conversation and costs O(T²); a
*linear* attention replaces ``softmax(QKᵀ/√d)V`` with a running outer-product state
``S_t = Σ_{s≤t} φ(k_s) ⊗ v_s`` (and normalizer ``z_t = Σ φ(k_s)``), so the per-head state is
O(d²) and **constant in sequence length** — the conversation can be arbitrarily long at fixed
memory. We reuse the *pretrained* q/k/v/o projections (so the swap operates on a capable model,
not a fresh net) and apply RoPE exactly as in the verified forward; only the attention CORE
changes. A hybrid swap (``linearize_qwen2(model, [layer indices])``) keeps softmax in some layers
and linear attention in others, which is what lets us measure each layer's linearization
tolerance — the honest research output. Prior art exists (linear attention: Katharopoulos et al.
2020; linearizing pretrained LLMs: SUPRA, LoLCATs, Mamba-in-Llama); the contribution here is the
on-prem internal surgery in llcore's own instrumentable code with rigorous per-layer measurement,
to be recovered by llcore's constant-state distillation.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from llcore.runtime.qwen2 import (
    Qwen2Attention,
    Qwen2LM,
    Qwen2Params,
    _apply_rope,
    _repeat_kv,
)


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Positive feature map for linear attention (elu+1; Katharopoulos et al. 2020)."""
    return F.elu(x) + 1.0


def _causal_linear_attn(
    qphi: torch.Tensor, kphi: torch.Tensor, v: torch.Tensor, chunk_size: int, eps: float
) -> torch.Tensor:
    """Exact causal linear attention via a running outer-product state, chunked for memory.

    ``qphi``/``kphi``: ``[B,H,T,Dk]`` (already feature-mapped), ``v``: ``[B,H,T,Dv]``.
    The running state ``S`` (``[B,H,Dk,Dv]``) / ``z`` (``[B,H,Dk]``) carries across chunks, so the
    result is independent of ``chunk_size`` (it only bounds the ``[B,H,chunk,Dk,Dv]`` temporary).
    """
    b, h, t, dk = qphi.shape
    dv = v.shape[-1]
    state_s = torch.zeros(b, h, dk, dv, dtype=qphi.dtype)
    state_z = torch.zeros(b, h, dk, dtype=qphi.dtype)
    outs: list[torch.Tensor] = []
    for start in range(0, t, chunk_size):
        stop = min(start + chunk_size, t)
        qc = qphi[:, :, start:stop]  # [B,H,C,Dk]
        kc = kphi[:, :, start:stop]
        vc = v[:, :, start:stop]  # [B,H,C,Dv]
        kv = kc.unsqueeze(-1) * vc.unsqueeze(-2)  # [B,H,C,Dk,Dv]
        a = kv.cumsum(dim=2)  # within-chunk prefix of k⊗v
        num = torch.einsum("bhcd,bhde->bhce", qc, state_s) + torch.einsum("bhcd,bhcde->bhce", qc, a)
        kc_cum = kc.cumsum(dim=2)  # [B,H,C,Dk]
        den = (
            torch.einsum("bhcd,bhd->bhc", qc, state_z).unsqueeze(-1)
            + torch.einsum("bhcd,bhcd->bhc", qc, kc_cum).unsqueeze(-1)
            + eps
        )
        outs.append(num / den)
        state_s = state_s + kv.sum(dim=2)
        state_z = state_z + kc.sum(dim=2)
    return torch.cat(outs, dim=2)


class LinearAttention(nn.Module):
    """Constant-state linear attention drop-in for :class:`Qwen2Attention` (reuses its weights)."""

    def __init__(
        self,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        o_proj: nn.Linear,
        params: Qwen2Params,
        chunk_size: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.p = params
        self.chunk_size = chunk_size
        self.eps = eps

    @classmethod
    def from_attention(cls, src: Qwen2Attention, params: Qwen2Params, **kw: object) -> LinearAttention:
        """Build a linear attention that REUSES a pretrained Qwen2Attention's projections."""
        return cls(src.q_proj, src.k_proj, src.v_proj, src.o_proj, params, **kw)  # type: ignore[arg-type]

    def state_bytes(self) -> int:
        """Bytes of the per-head running state (S + z), float32 — constant in sequence length."""
        p = self.p
        return p.n_head * p.head_dim * p.head_dim * 4 + p.n_head * p.head_dim * 4

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        past_len: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        p = self.p
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, p.n_head, p.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, p.n_kv_head, p.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, p.n_kv_head, p.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        n_rep = p.n_head // p.n_kv_head
        kf = _repeat_kv(k, n_rep)
        vf = _repeat_kv(v, n_rep)
        out = _causal_linear_attn(_phi(q), _phi(kf), vf, self.chunk_size, self.eps)
        out = out.transpose(1, 2).reshape(b, t, p.n_head * p.head_dim)
        return self.o_proj(out), (k, v)


def linearize_qwen2(
    model: Qwen2LM, layer_indices: Sequence[int], chunk_size: int = 64
) -> Qwen2LM:
    """In-place swap: replace ``self_attn`` with :class:`LinearAttention` for the given layers.

    Reuses each layer's pretrained projections. Returns the same (mutated) model for chaining.
    Pass a ``copy.deepcopy(model)`` first if you need to keep the softmax original intact.
    """
    layers = model.model.layers
    for i in layer_indices:
        attn = layers[i].self_attn
        if isinstance(attn, Qwen2Attention):
            layers[i].self_attn = LinearAttention.from_attention(
                attn, model.params, chunk_size=chunk_size
            )
    return model
