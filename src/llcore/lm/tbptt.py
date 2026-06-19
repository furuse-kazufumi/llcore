# SPDX-License-Identifier: Apache-2.0
"""Truncated-BPTT long-context training for constant-state recurrent LMs (pure-add).

The baseline :class:`~llcore.lm.trainer.Trainer` trains on random ``block_size`` windows and
``forward`` re-inits the state to zeros every call, so its effective BPTT span is ``block_size``
— the model gets ZERO gradient for dependencies longer than ``block_size``. To teach a model to
*use* context beyond ``block_size`` (the prerequisite for any honest "uses long context" claim),
this trainer streams long contiguous segments through the per-token ``step`` API, carrying a
**detached** state across ``chunk_size`` sub-windows: gradients still truncate at ``chunk_size``
(bounded O(chunk_size) memory), but the forward state reflects the whole segment so far.

It is a strictly additive module: it never touches ``Trainer.train`` or the models' ``forward``.
For a fair TBPTT-vs-baseline comparison, use the SAME ``vocab``/params/optimizer/seed and match
total tokens & updates (``chunk_size`` == baseline ``block_size``, same ``batch_size`` and
``max_updates``); the ONLY difference is the carried state.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from llcore.lm.eval import estimate_loss
from llcore.lm.recurrent import RecurrentLM
from llcore.lm.rwkv import RWKVLM

ConstantStateLM = RecurrentLM | RWKVLM
TBPTTHistoryItem = dict[str, float | int]
EvalHook = Callable[[int, float, float], None]


def detach_state(state: list[object]) -> list[object]:
    """Detach a carried state from the autograd graph, preserving values.

    Handles both state shapes: ``RecurrentLM`` uses ``list[Tensor]``; ``RWKVLM`` uses
    ``list[RWKVLayerState]`` (a NamedTuple), whose fields are detached individually.
    """
    out: list[object] = []
    for s in state:
        if isinstance(s, torch.Tensor):
            out.append(s.detach())
        else:
            out.append(type(s)(*[t.detach() for t in s]))  # RWKVLayerState NamedTuple
    return out


def reset_state_slots(
    model: ConstantStateLM, state: list[object], mask: torch.Tensor
) -> list[object]:
    """Reset the batch rows selected by ``mask`` (bool ``[B]``) to a fresh zero/init state."""
    fresh = model.init_state(int(mask.size(0)))
    m = mask.view(-1, 1)
    out: list[object] = []
    for s, f in zip(state, fresh, strict=True):
        if isinstance(s, torch.Tensor):
            out.append(torch.where(m, f, s))
        else:
            out.append(type(s)(*[torch.where(m, ff, ss) for ss, ff in zip(s, f, strict=True)]))
    return out


@dataclass
class TBPTTConfig:
    """Hyperparameters for :class:`TBPTTTrainer`."""

    seg_len: int = 1024
    chunk_size: int = 128
    batch_size: int = 32
    max_updates: int = 4000
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_updates: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 40
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or self.seg_len < self.chunk_size + 1:
            raise ValueError(
                f"need seg_len ({self.seg_len}) >= chunk_size+1 ({self.chunk_size + 1})"
            )


class TBPTTTrainer:
    """Streams long segments through ``step`` carrying a detached state across chunks."""

    def __init__(self, model: ConstantStateLM, config: TBPTTConfig | None = None) -> None:
        self.model = model
        self.cfg = config or TBPTTConfig()
        self.optimizer = self._configure_optimizer()
        self.gen = torch.Generator().manual_seed(self.cfg.seed)
        self.eval_gen = torch.Generator().manual_seed(self.cfg.seed + 1)
        self.update_num = 0
        self.best_val = float("inf")
        self.history: list[TBPTTHistoryItem] = []

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        decay: list[torch.Tensor] = []
        no_decay: list[torch.Tensor] = []
        seen: set[int] = set()
        for p in self.model.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups, lr=self.cfg.learning_rate, betas=(self.cfg.beta1, self.cfg.beta2)
        )

    def get_lr(self, it: int) -> float:
        cfg = self.cfg
        if it < cfg.warmup_updates:
            return cfg.learning_rate * (it + 1) / (cfg.warmup_updates + 1)
        if it >= cfg.max_updates:
            return cfg.min_lr
        ratio = (it - cfg.warmup_updates) / max(1, cfg.max_updates - cfg.warmup_updates)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def _sample_starts(self, train_ids: torch.Tensor, k: int) -> torch.Tensor:
        hi = int(train_ids.size(0)) - self.cfg.seg_len - 1
        if hi < 1:
            raise ValueError(
                f"train length {train_ids.size(0)} too short for seg_len {self.cfg.seg_len}"
            )
        return torch.randint(0, hi + 1, (k,), generator=self.gen)

    def train(
        self,
        train_ids: torch.Tensor,
        val_ids: torch.Tensor,
        on_eval: EvalHook | None = None,
    ) -> dict[str, object]:
        cfg = self.cfg
        cs = cfg.chunk_size
        bsz = cfg.batch_size
        vocab = self.model.config.vocab_size
        block = self.model.config.block_size
        ar = torch.arange(cs)

        seg_starts = self._sample_starts(train_ids, bsz)
        seg_pos = torch.zeros(bsz, dtype=torch.long)
        state: list[object] = self.model.init_state(bsz)
        self.model.train()
        for it in range(self.update_num, cfg.max_updates):
            lr = self.get_lr(it)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            base = (seg_starts + seg_pos).unsqueeze(1)  # [B,1]
            x = train_ids[base + ar]  # [B, cs]
            y = train_ids[base + ar + 1]  # [B, cs]
            logits_steps: list[torch.Tensor] = []
            for t in range(cs):
                logits, state = self.model.step(x[:, t], state)  # type: ignore[arg-type]
                logits_steps.append(logits.unsqueeze(1))
            chunk_logits = torch.cat(logits_steps, dim=1)  # [B, cs, V]
            loss = F.cross_entropy(chunk_logits.reshape(-1, vocab), y.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            state = detach_state(state)
            seg_pos = seg_pos + cs
            exhausted = (seg_pos + cs + 1) > cfg.seg_len
            if bool(exhausted.any()):
                idx = exhausted.nonzero().flatten()
                seg_starts = seg_starts.clone()
                seg_starts[idx] = self._sample_starts(train_ids, int(idx.numel()))
                seg_pos = seg_pos.clone()
                seg_pos[idx] = 0
                state = reset_state_slots(self.model, state, exhausted)
            if it % cfg.eval_interval == 0 or it == cfg.max_updates - 1:
                tr = estimate_loss(self.model, train_ids, block, bsz, cfg.eval_iters, self.eval_gen)
                va = estimate_loss(self.model, val_ids, block, bsz, cfg.eval_iters, self.eval_gen)
                self.model.train()
                self.best_val = min(self.best_val, va)
                self.history.append({"update": it, "lr": lr, "train_loss": tr, "val_loss": va})
                if on_eval is not None:
                    on_eval(it, tr, va)
            self.update_num = it + 1
        return {"history": list(self.history), "best_val_loss": self.best_val}
