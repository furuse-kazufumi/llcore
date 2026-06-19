# SPDX-License-Identifier: Apache-2.0
"""Constant-state recurrent character LM for Phase 1.

The core recurrence is the verified bounded update already used elsewhere in llcore:

``s_t = decay * s_{t-1} + (1 - decay) * tanh(W s_{t-1} + x_t)``

with ``decay = sigmoid(raw_decay)`` and ``W = 2 * tanh(raw_W)``. The model keeps one
state vector per layer, so generation uses O(1) runtime memory with respect to the
generated sequence length.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RecurrentConfig:
    """Configuration for :class:`RecurrentLM`."""

    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_embd: int = 128
    state_size: int = 128
    dropout: float = 0.0
    bias: bool = True
    model_type: str = "gated-rnn"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be > 0, got {self.n_layer}")
        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be > 0, got {self.n_embd}")
        if self.state_size <= 0:
            raise ValueError(f"state_size must be > 0, got {self.state_size}")


class RecurrentCore(nn.Module):
    """One bounded recurrent layer with residual projection back to the token width."""

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        self.in_proj = nn.Linear(config.n_embd, config.state_size, bias=config.bias)
        self.out_proj = nn.Linear(config.state_size, config.n_embd, bias=config.bias)
        self.norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.raw_decay = nn.Parameter(torch.randn(config.state_size) * 0.5 + 1.0)
        self.raw_W = nn.Parameter(
            torch.randn(config.state_size, config.state_size)
            * (0.3 / math.sqrt(config.state_size))
        )

    def core_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.raw_decay)
        W = 2.0 * torch.tanh(self.raw_W)
        return decay, W

    def step(self, h: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decay, W = self.core_params()
        xc = torch.tanh(self.in_proj(h))
        next_state = decay * state + (1.0 - decay) * torch.tanh(state @ W.T + xc)
        next_h = self.norm(h + self.out_proj(next_state))
        return next_h, next_state


class RecurrentLM(nn.Module):
    """Char-level recurrent LM with constant-size per-layer generation state."""

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([RecurrentCore(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
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

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_state(
        self, batch_size: int, *, device: torch.device | None = None
    ) -> list[torch.Tensor]:
        dev = device if device is not None else self.lm_head.weight.device
        return [
            torch.zeros(batch_size, self.config.state_size, device=dev)
            for _ in range(self.config.n_layer)
        ]

    def state_bytes(self, state: list[torch.Tensor]) -> int:
        return sum(int(s.numel() * s.element_size()) for s in state)

    def step(
        self, idx_t: torch.Tensor, state: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if idx_t.ndim != 1:
            raise ValueError(f"idx_t must be 1-D [B], got shape {tuple(idx_t.shape)}")
        batch = idx_t.size(0)
        cur_state = self.init_state(batch, device=idx_t.device) if state is None else state
        h = self.transformer["drop"](self.transformer["wte"](idx_t))
        next_state: list[torch.Tensor] = []
        for li, layer in enumerate(cast(nn.ModuleList, self.transformer["h"])):
            assert isinstance(layer, RecurrentCore)
            h, s = layer.step(h, cur_state[li])
            next_state.append(s)
        h = self.transformer["ln_f"](h)
        logits = cast(torch.Tensor, self.lm_head(h))
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
            step_logits, state = self.step(idx[:, pos], state)
            logits_steps.append(step_logits.unsqueeze(1))
        logits = torch.cat(logits_steps, dim=1)
        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def streaming_nll(self, ids: torch.Tensor, chunk_size: int = 256) -> tuple[float, int]:
        """Mean next-token cross-entropy (nats) over a 1-D sequence of *any* length.

        The constant per-layer state has no architectural context limit, so this scores
        sequences longer than ``block_size`` (which only caps the batched :meth:`forward`)
        and never materializes O(T) logits: it steps through the sequence accumulating the
        loss, so peak activation memory is O(``chunk_size``) in T while the state stays
        O(1). Predicts ``ids[1:]`` from ``ids[:-1]``; returns ``(mean_nll, n_predicted)``.
        A GPT cannot do this — its attention is O(T²) and ``block_size``-bounded.
        """
        if ids.ndim != 1:
            raise ValueError(f"ids must be 1-D [T], got shape {tuple(ids.shape)}")
        n = int(ids.size(0))
        if n < 2:
            raise ValueError(f"streaming_nll needs >= 2 tokens, got {n}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        was_training = self.training
        self.eval()
        state: list[torch.Tensor] | None = None
        inputs, targets = ids[:-1], ids[1:]
        total = 0.0
        for start in range(0, n - 1, chunk_size):
            stop = min(start + chunk_size, n - 1)
            logits_chunk: list[torch.Tensor] = []
            for i in range(start, stop):
                logits, state = self.step(inputs[i : i + 1], state)
                logits_chunk.append(logits)
            chunk = torch.cat(logits_chunk, dim=0)
            total += float(F.cross_entropy(chunk, targets[start:stop], reduction="sum").item())
        if was_training:
            self.train()
        return total / (n - 1), n - 1

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
        batch, prompt_len = idx.shape
        state = self.init_state(batch, device=idx.device)
        last_logits: torch.Tensor | None = None
        for pos in range(prompt_len):
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
