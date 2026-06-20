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
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.p = params
        self.chunk_size = chunk_size
        self.eps = eps
        self.learnable = learnable
        if learnable:
            # per-head affine on q/k BEFORE the feature map φ; identity at init so a freshly
            # learnable linear attention == the fixed one until distillation moves it (LoLCATs-style
            # learnable feature map, with the pretrained projections frozen).
            self.q_scale = nn.Parameter(torch.ones(params.n_head, params.head_dim))
            self.q_bias = nn.Parameter(torch.zeros(params.n_head, params.head_dim))
            self.k_scale = nn.Parameter(torch.ones(params.n_head, params.head_dim))
            self.k_bias = nn.Parameter(torch.zeros(params.n_head, params.head_dim))

    @classmethod
    def from_attention(cls, src: Qwen2Attention, params: Qwen2Params, **kw: object) -> LinearAttention:
        """Build a linear attention that REUSES a pretrained Qwen2Attention's projections."""
        return cls(src.q_proj, src.k_proj, src.v_proj, src.o_proj, params, **kw)  # type: ignore[arg-type]

    def feature_parameters(self) -> list[nn.Parameter]:
        """The trainable feature-map parameters (empty unless ``learnable``)."""
        if not self.learnable:
            return []
        return [self.q_scale, self.q_bias, self.k_scale, self.k_bias]

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
        if self.learnable:
            qf_in = q * self.q_scale[None, :, None, :] + self.q_bias[None, :, None, :]
            kf_in = kf * self.k_scale[None, :, None, :] + self.k_bias[None, :, None, :]
        else:
            qf_in, kf_in = q, kf
        out = _causal_linear_attn(_phi(qf_in), _phi(kf_in), vf, self.chunk_size, self.eps)
        out = out.transpose(1, 2).reshape(b, t, p.n_head * p.head_dim)
        return self.o_proj(out), (k, v)


class SlidingWindowAttention(nn.Module):
    """Softmax attention restricted to the last ``window`` keys — bounded O(window) KV cache.

    A NAS mixer between full softmax (O(T) KV, best quality) and linear attention (O(1) state): it
    keeps softmax (so local quality is preserved) but each query attends only the previous
    ``window`` keys, capping the KV cache. Reuses the pretrained q/k/v/o projections + RoPE, and is
    identical to full softmax when ``window`` covers the sequence.
    """

    def __init__(
        self,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        o_proj: nn.Linear,
        params: Qwen2Params,
        window: int,
    ) -> None:
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.p = params
        self.window = window

    @classmethod
    def from_attention(cls, src: Qwen2Attention, params: Qwen2Params, window: int) -> SlidingWindowAttention:
        return cls(src.q_proj, src.k_proj, src.v_proj, src.o_proj, params, window)

    def kv_bytes(self, context_len: int) -> int:
        """Resident KV bytes — capped at ``window`` keys regardless of context length."""
        eff = min(self.window, context_len)
        return 2 * self.p.n_kv_head * eff * self.p.head_dim * 4

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
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        new_cache = (k, v)
        n_rep = p.n_head // p.n_kv_head
        kf = _repeat_kv(k, n_rep)
        vf = _repeat_kv(v, n_rep)
        scores = torch.matmul(q, kf.transpose(-2, -1)) / (p.head_dim**0.5)
        tk = kf.size(2)
        qpos = torch.arange(t) + past_len
        kpos = torch.arange(tk)
        causal = kpos[None, :] <= qpos[:, None]
        within = (qpos[:, None] - kpos[None, :]) < self.window
        mask = torch.where(causal & within, 0.0, float("-inf"))
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, vf).transpose(1, 2).reshape(b, t, p.n_head * p.head_dim)
        return self.o_proj(out), new_cache


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
