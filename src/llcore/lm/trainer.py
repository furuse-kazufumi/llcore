# SPDX-License-Identifier: Apache-2.0
"""Training loop for the char-LM: AdamW + cosine LR (warmup) + grad clipping.

CPU-only. The LR schedule and optimizer parameter grouping follow nanoGPT: matmul
(>=2-D) weights are weight-decayed, while biases / LayerNorm / 1-D params are not.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from llcore.lm.data import get_batch
from llcore.lm.eval import estimate_loss
from llcore.lm.model import CharGPT


@dataclass
class TrainConfig:
    """Hyperparameters for :class:`Trainer`. Defaults = nanoGPT CPU "nano" recipe."""

    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    lr_decay_iters: int = 2000
    max_iters: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    batch_size: int = 12
    eval_interval: int = 250
    eval_iters: int = 20
    seed: int = 1337
    log_interval: int = 100


EvalHook = Callable[[int, float, float], None]


class Trainer:
    """Trains a :class:`CharGPT` on a 1-D id tensor, tracking held-out loss."""

    def __init__(self, model: CharGPT, config: TrainConfig | None = None) -> None:
        self.model = model
        self.cfg = config or TrainConfig()
        self.optimizer = self._configure_optimizer()
        self.batch_gen = torch.Generator().manual_seed(self.cfg.seed)
        self.eval_gen = torch.Generator().manual_seed(self.cfg.seed + 1)

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
        """Linear warmup then cosine decay to ``min_lr`` (nanoGPT schedule)."""
        cfg = self.cfg
        if it < cfg.warmup_iters:
            return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def train(
        self,
        train_ids: torch.Tensor,
        val_ids: torch.Tensor,
        on_eval: EvalHook | None = None,
    ) -> dict[str, object]:
        """Run the training loop; return history + best held-out loss."""
        cfg = self.cfg
        block = self.model.config.block_size
        history: list[dict[str, float]] = []
        best_val = float("inf")
        self.model.train()
        for it in range(cfg.max_iters):
            lr = self.get_lr(it)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            x, y = get_batch(train_ids, block, cfg.batch_size, self.batch_gen)
            _, loss = self.model(x, y)
            assert loss is not None
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
                tr = estimate_loss(
                    self.model, train_ids, block, cfg.batch_size, cfg.eval_iters, self.eval_gen
                )
                va = estimate_loss(
                    self.model, val_ids, block, cfg.batch_size, cfg.eval_iters, self.eval_gen
                )
                best_val = min(best_val, va)
                history.append({"iter": it, "lr": lr, "train_loss": tr, "val_loss": va})
                if on_eval is not None:
                    on_eval(it, tr, va)
        return {"history": history, "best_val_loss": best_val}
