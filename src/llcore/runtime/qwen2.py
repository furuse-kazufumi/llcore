# SPDX-License-Identifier: Apache-2.0
"""llcore-native Qwen2 forward (Qwen2.5 family) — instrumentable substrate, not the research.

This is a from-scratch PyTorch implementation of the Qwen2 decoder (RMSNorm, rotary position
embeddings, grouped-query attention, SwiGLU MLP, optionally tied embeddings) whose module names
mirror HuggingFace ``Qwen2ForCausalLM`` exactly, so a HF ``state_dict`` loads directly. It exists
so llcore can run a *capable pretrained* model in its OWN code — the prerequisite for the actual
research layer (bounded-memory conversation via a constant-state memory, evolved per-layer
quantization, etc.), which a black-box runtime (Ollama/llama.cpp) could not host. On its own this
forward is a re-derivation, not a contribution; the contribution is what gets built on top of it.

Correctness is pinned by golden tests against HF (``tests/unit/test_runtime_qwen2.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class Qwen2Params:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    tie_embeddings: bool
    max_position: int

    @classmethod
    def from_hf_config(cls, cfg: dict[str, object]) -> Qwen2Params:
        hidden = int(cfg["hidden_size"])  # type: ignore[arg-type]
        n_head = int(cfg["num_attention_heads"])  # type: ignore[arg-type]
        head_dim = int(cfg.get("head_dim", hidden // n_head))  # type: ignore[arg-type]
        return cls(
            vocab_size=int(cfg["vocab_size"]),  # type: ignore[arg-type]
            hidden_size=hidden,
            intermediate_size=int(cfg["intermediate_size"]),  # type: ignore[arg-type]
            n_layer=int(cfg["num_hidden_layers"]),  # type: ignore[arg-type]
            n_head=n_head,
            n_kv_head=int(cfg["num_key_value_heads"]),  # type: ignore[arg-type]
            head_dim=head_dim,
            rope_theta=float(cfg.get("rope_theta", 1000000.0)),  # type: ignore[arg-type]
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),  # type: ignore[arg-type]
            tie_embeddings=bool(cfg.get("tie_word_embeddings", True)),
            max_position=int(cfg.get("max_position_embeddings", 32768)),  # type: ignore[arg-type]
        )


class RMSNorm(nn.Module):
    """Qwen2 RMSNorm: x * rsqrt(mean(x^2) + eps) * weight, reduction done in float32."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed.to(dtype)).to(dtype)


def _rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)  # [T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin: [T, head_dim] -> [1, 1, T, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, kv, n_rep, t, hd).reshape(b, kv * n_rep, t, hd)


class Qwen2Attention(nn.Module):
    def __init__(self, p: Qwen2Params) -> None:
        super().__init__()
        self.p = p
        self.q_proj = nn.Linear(p.hidden_size, p.n_head * p.head_dim, bias=True)
        self.k_proj = nn.Linear(p.hidden_size, p.n_kv_head * p.head_dim, bias=True)
        self.v_proj = nn.Linear(p.hidden_size, p.n_kv_head * p.head_dim, bias=True)
        self.o_proj = nn.Linear(p.n_head * p.head_dim, p.hidden_size, bias=False)

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
        scores = torch.matmul(q, kf.transpose(-2, -1)) / (p.head_dim**0.5)  # [B,H,T,Tk]
        tk = kf.size(2)
        qpos = torch.arange(t) + past_len
        kpos = torch.arange(tk)
        mask = torch.where(kpos[None, :] <= qpos[:, None], 0.0, float("-inf"))
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, vf)  # [B,H,T,hd]
        out = out.transpose(1, 2).reshape(b, t, p.n_head * p.head_dim)
        return self.o_proj(out), new_cache


class Qwen2MLP(nn.Module):
    def __init__(self, p: Qwen2Params) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(p.hidden_size, p.intermediate_size, bias=False)
        self.up_proj = nn.Linear(p.hidden_size, p.intermediate_size, bias=False)
        self.down_proj = nn.Linear(p.intermediate_size, p.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, p: Qwen2Params) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(p.hidden_size, p.rms_norm_eps)
        self.self_attn = Qwen2Attention(p)
        self.post_attention_layernorm = RMSNorm(p.hidden_size, p.rms_norm_eps)
        self.mlp = Qwen2MLP(p)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        past_len: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, cache = self.self_attn(self.input_layernorm(x), cos, sin, past, past_len)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache


class _Qwen2Inner(nn.Module):
    def __init__(self, p: Qwen2Params) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(p.vocab_size, p.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(p) for _ in range(p.n_layer)])
        self.norm = RMSNorm(p.hidden_size, p.rms_norm_eps)


class Qwen2LM(nn.Module):
    """Qwen2 causal LM mirroring HF parameter names so a HF state_dict loads directly."""

    def __init__(self, params: Qwen2Params) -> None:
        super().__init__()
        self.params = params
        self.model = _Qwen2Inner(params)
        self.lm_head = nn.Linear(params.hidden_size, params.vocab_size, bias=False)

    def load_hf_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Load a HuggingFace Qwen2ForCausalLM state_dict (casting to float32)."""
        own = self.state_dict()
        filtered = {
            k: v.float() for k, v in sd.items() if k in own and tuple(v.shape) == tuple(own[k].shape)
        }
        self.load_state_dict(filtered, strict=False)
        if self.params.tie_embeddings and "lm_head.weight" not in filtered:
            self.lm_head.weight.data.copy_(self.model.embed_tokens.weight.data)

    def forward(
        self,
        input_ids: torch.Tensor,
        past: KVCache | None = None,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        p = self.params
        b, t = input_ids.shape
        past_len = int(past[0][0].size(2)) if past is not None else 0
        x = self.model.embed_tokens(input_ids)
        positions = torch.arange(past_len, past_len + t)
        cos, sin = _rope_cos_sin(positions, p.head_dim, p.rope_theta)
        new_cache: KVCache = []
        for i, layer in enumerate(self.model.layers):
            assert isinstance(layer, Qwen2DecoderLayer)
            x, kv = layer(x, cos, sin, past[i] if past is not None else None, past_len)
            new_cache.append(kv)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        if return_cache:
            return logits, new_cache
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        logits, cache = self.forward(input_ids, return_cache=True)  # type: ignore[misc]
        out = input_ids
        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :]
            if temperature <= 0:  # greedy
                next_id = step_logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = step_logits / temperature
                if top_k is not None:
                    k = min(top_k, scaled.size(-1))
                    kth = torch.topk(scaled, k).values[:, -1, None]
                    scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                probs = torch.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=gen)
            out = torch.cat((out, next_id), dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
            logits, cache = self.forward(next_id, past=cache, return_cache=True)  # type: ignore[misc]
        if was_training:
            self.train()
        return out
