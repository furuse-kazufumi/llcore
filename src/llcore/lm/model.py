# SPDX-License-Identifier: Apache-2.0
"""GPT-2 "nano" character-level language model (minGPT / nanoGPT topology, CPU-only).

The module tree is named to match Karpathy's minGPT *exactly* so that
``model.state_dict()`` yields the same keys the bbycroft/llm-viz visualizer loads.
This was verified against ``llcore-viz/public/gpt-nano-sort-model.json`` (the sample
that ships with the visualizer is a direct dump of ``GPT.state_dict()`` —
see ``llcore-viz/gen_test_data.py``). Consequently :mod:`llcore.lm.export` can emit a
viz-compatible JSON with a trivial ``state_dict`` walk — no key remapping needed.

Design constraints (capability-first replan P0,
``docs/LLM_CAPABILITY_FIRST_REPLAN_2026_06_13.md``):

- **CPU / ``torch.float32`` only** — this machine has no GPU (torch 2.12.0+cpu).
- **GPT-2 nano topology**: learned token + positional embeddings, pre-LayerNorm
  blocks, causal multi-head self-attention, 4x GELU MLP, final LayerNorm, weight-tied
  ``lm_head``. ``bias=True`` everywhere by default (the visualizer reads LayerNorm
  biases, so ``bias=False`` is *not* viz-exportable).

The submodules are split into small typed ``nn.Module`` classes (``MLP``,
``Transformer``) rather than ``nn.ModuleDict`` so the code is statically typed
(mypy strict) while producing identical ``state_dict`` keys.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Configuration for :class:`CharGPT`.

    Attributes
    ----------
    vocab_size : int
        Number of distinct tokens (characters). Must be > 0.
    block_size : int
        Maximum context length (positions). Must be > 0.
    n_layer, n_head, n_embd : int
        Transformer depth, attention heads, embedding width. ``n_embd`` must be
        divisible by ``n_head``.
    dropout : float
        Dropout probability (embeddings, attention, residual). 0.0 for the fast
        smoke config; ~0.2 for tiny-corpus P1 to curb overfitting.
    bias : bool
        Whether Linear / LayerNorm layers carry a bias. Keep ``True`` for
        llm-viz export compatibility (the viz expects LayerNorm biases).
    model_type : str
        Label echoed into the exported config (e.g. ``"gpt-nano"``).
    tie_weights : bool
        Tie ``lm_head.weight`` to ``transformer.wte.weight`` (nanoGPT convention).
    """

    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    model_type: str = "gpt-nano"
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )


class NewGELU(nn.Module):
    """GELU (tanh approximation), identical to minGPT's ``NewGELU``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention (minGPT-exact naming: ``c_attn`` / ``c_proj``).

    The lower-triangular causal mask is held in a persistent buffer named ``bias`` so
    it (a) is actually used in the masked softmax and (b) appears in ``state_dict()``
    as ``...attn.bias`` exactly like the llm-viz sample.
    """

    bias: torch.Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_dim = c // self.n_head
        k = k.view(b, t, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        q = q.view(b, t, self.n_head, head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return cast(torch.Tensor, self.resid_dropout(self.c_proj(y)))


class MLP(nn.Module):
    """Position-wise feed-forward: ``c_fc`` -> GELU -> ``c_proj`` (4x expansion)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act = NewGELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.dropout(self.c_proj(self.act(self.c_fc(x)))))


class Block(nn.Module):
    """A pre-LayerNorm transformer block (minGPT naming: ``ln_1/attn/ln_2/mlp``)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Container module so ``state_dict`` keys are prefixed ``transformer.*``."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)


class CharGPT(nn.Module):
    """A small GPT-2 char-level language model (CPU, float32).

    The submodule layout (``transformer.{wte,wpe,drop,h,ln_f}`` + ``lm_head``) is
    chosen so ``state_dict()`` keys match the llm-viz schema. See module docstring.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # GPT-2 residual-path scaled init for projections writing into the residual.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

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

    def num_params(self, non_embedding: bool = True) -> int:
        """Total parameter count. ``non_embedding`` subtracts positional embeddings.

        With tied weights, ``wte`` and ``lm_head`` share storage and are counted once
        (``parameters()`` de-duplicates shared tensors).
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(
                f"sequence length {t} exceeds block_size {self.config.block_size}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = cast(torch.Tensor, self.lm_head(x))
        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self(idx)[0])

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` ids, appended to ``idx``."""
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        if was_training:
            self.train()
        return idx
