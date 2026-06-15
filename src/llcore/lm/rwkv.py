# SPDX-License-Identifier: Apache-2.0
"""Minimal RWKV-4 style recurrent char LM with stable running-max WKV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import torch
from torch import nn
from torch.nn import functional as F


class RWKVLayerState(NamedTuple):
    prev_tm_x: torch.Tensor
    prev_cm_x: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    p: torch.Tensor


@dataclass
class RWKVConfig:
    """Configuration for :class:`RWKVLM`."""

    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    model_type: str = "rwkv-4"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be > 0, got {self.n_layer}")
        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be > 0, got {self.n_embd}")


def _mix(cur: torch.Tensor, prev: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    return cur * mix + prev * (1.0 - mix)


class RWKVTimeMix(nn.Module):
    """RWKV time-mix with stable running-max WKV state."""

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        d = config.n_embd
        self.mix_k = nn.Parameter(torch.rand(d))
        self.mix_v = nn.Parameter(torch.rand(d))
        self.mix_r = nn.Parameter(torch.rand(d))
        self.time_decay = nn.Parameter(torch.zeros(d))
        self.time_first = nn.Parameter(torch.zeros(d))
        self.key = nn.Linear(d, d, bias=config.bias)
        self.value = nn.Linear(d, d, bias=config.bias)
        self.receptance = nn.Linear(d, d, bias=config.bias)
        self.output = nn.Linear(d, d, bias=config.bias)

    def step(
        self, x: torch.Tensor, prev_x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xk = _mix(x, prev_x, self.mix_k)
        xv = _mix(x, prev_x, self.mix_v)
        xr = _mix(x, prev_x, self.mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        decay = -torch.exp(self.time_decay)

        q = torch.maximum(p, self.time_first + k)
        e1 = torch.exp(p - q)
        e2 = torch.exp(self.time_first + k - q)
        wkv = (e1 * a + e2 * v) / (e1 * b + e2)

        q2 = torch.maximum(p + decay, k)
        e1n = torch.exp(p + decay - q2)
        e2n = torch.exp(k - q2)
        next_a = e1n * a + e2n * v
        next_b = e1n * b + e2n
        next_p = q2
        out = self.output(r * wkv)
        return out, next_a, next_b, next_p


class RWKVChannelMix(nn.Module):
    """RWKV channel-mix FFN with squared-ReLU activation."""

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        d = config.n_embd
        hidden = 4 * d
        self.mix_k = nn.Parameter(torch.rand(d))
        self.mix_r = nn.Parameter(torch.rand(d))
        self.key = nn.Linear(d, hidden, bias=config.bias)
        self.value = nn.Linear(hidden, d, bias=config.bias)
        self.receptance = nn.Linear(d, d, bias=config.bias)

    def step(self, x: torch.Tensor, prev_x: torch.Tensor) -> torch.Tensor:
        xk = _mix(x, prev_x, self.mix_k)
        xr = _mix(x, prev_x, self.mix_r)
        k = F.relu(self.key(xk))
        kv = self.value(k * k)
        r = torch.sigmoid(self.receptance(xr))
        return cast(torch.Tensor, r * kv)


class RWKVBlock(nn.Module):
    """Pre-LN RWKV block with time-mix then channel-mix."""

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.time_mixer = RWKVTimeMix(config)
        self.channel_mixer = RWKVChannelMix(config)

    def step(self, x: torch.Tensor, state: RWKVLayerState) -> tuple[torch.Tensor, RWKVLayerState]:
        x_ln1 = self.ln1(x)
        att, next_a, next_b, next_p = self.time_mixer.step(
            x_ln1, state.prev_tm_x, state.a, state.b, state.p
        )
        x = x + att
        x_ln2 = self.ln2(x)
        ffn = self.channel_mixer.step(x_ln2, state.prev_cm_x)
        x = x + ffn
        next_state = RWKVLayerState(
            prev_tm_x=x_ln1,
            prev_cm_x=x_ln2,
            a=next_a,
            b=next_b,
            p=next_p,
        )
        return x, next_state


class RWKVLM(nn.Module):
    """RWKV-4 style char LM with O(1) generation state per layer."""

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.ln0 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([RWKVBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def init_state(
        self, batch_size: int, *, device: torch.device | None = None
    ) -> list[RWKVLayerState]:
        dev = device if device is not None else self.lm_head.weight.device
        zeros = torch.zeros(batch_size, self.config.n_embd, device=dev)
        neg_inf = torch.full((batch_size, self.config.n_embd), -1e30, device=dev)
        return [
            RWKVLayerState(
                prev_tm_x=zeros.clone(),
                prev_cm_x=zeros.clone(),
                a=zeros.clone(),
                b=zeros.clone(),
                p=neg_inf.clone(),
            )
            for _ in range(self.config.n_layer)
        ]

    def state_bytes(self, state: list[RWKVLayerState]) -> int:
        total = 0
        for layer_state in state:
            for tensor in layer_state:
                total += int(tensor.numel() * tensor.element_size())
        return total

    def step(
        self, idx_t: torch.Tensor, state: list[RWKVLayerState] | None = None
    ) -> tuple[torch.Tensor, list[RWKVLayerState]]:
        if idx_t.ndim != 1:
            raise ValueError(f"idx_t must be 1-D [B], got shape {tuple(idx_t.shape)}")
        cur_state = self.init_state(idx_t.size(0), device=idx_t.device) if state is None else state
        x = self.drop(self.ln0(self.emb(idx_t)))
        next_state: list[RWKVLayerState] = []
        for li, block in enumerate(self.blocks):
            assert isinstance(block, RWKVBlock)
            x, layer_state = block.step(x, cur_state[li])
            next_state.append(layer_state)
        x = self.ln_f(x)
        logits = cast(torch.Tensor, self.lm_head(x))
        return logits, next_state

    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self(idx)[0])

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError(f"idx must be 2-D [B,T], got shape {tuple(idx.shape)}")
        _, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(
                f"sequence length {t} exceeds block_size {self.config.block_size}"
            )
        state = self.init_state(idx.size(0), device=idx.device)
        logits_steps: list[torch.Tensor] = []
        for pos in range(t):
            logits_t, state = self.step(idx[:, pos], state)
            logits_steps.append(logits_t.unsqueeze(1))
        logits = torch.cat(logits_steps, dim=1)
        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if idx.ndim != 2:
            raise ValueError(f"idx must be 2-D [B,T], got shape {tuple(idx.shape)}")
        if idx.size(1) == 0:
            raise ValueError("idx must contain at least one prompt token")
        was_training = self.training
        self.eval()
        state = self.init_state(idx.size(0), device=idx.device)
        last_logits: torch.Tensor | None = None
        for pos in range(idx.size(1)):
            last_logits, state = self.step(idx[:, pos], state)
        assert last_logits is not None
        out = idx
        for _ in range(max_new_tokens):
            logits = last_logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            out = torch.cat((out, idx_next), dim=1)
            last_logits, state = self.step(idx_next[:, 0], state)
        if was_training:
            self.train()
        return out
