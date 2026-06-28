# SPDX-License-Identifier: Apache-2.0
"""Training loop for the char-LM: AdamW + cosine LR (warmup) + grad clipping.

CPU-only. The LR schedule and optimizer parameter grouping follow nanoGPT: matmul
(>=2-D) weights are weight-decayed, while biases / LayerNorm / 1-D params are not.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from llcore.lm.data import get_batch
from llcore.lm.device import model_device
from llcore.lm.eval import TrainableLM as EvalTrainableLM
from llcore.lm.eval import estimate_loss


class TrainableLM(EvalTrainableLM, Protocol):
    config: Any

    def parameters(self) -> Iterator[torch.Tensor]: ...


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


TrainHistoryItem = dict[str, float | int]
EvalHook = Callable[[int, float, float], None]


class Trainer:
    """Trains a :class:`CharGPT` on a 1-D id tensor, tracking held-out loss."""

    def __init__(self, model: TrainableLM, config: TrainConfig | None = None) -> None:
        self.model = model
        self.cfg = config or TrainConfig()
        self.optimizer = self._configure_optimizer()
        self.batch_gen = torch.Generator().manual_seed(self.cfg.seed)
        self.eval_gen = torch.Generator().manual_seed(self.cfg.seed + 1)
        self.iter_num = 0
        self.best_val = float("inf")
        self.history: list[TrainHistoryItem] = []

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

    def state_dict(self) -> dict[str, object]:
        """Serialize optimizer + RNG + progress so training can resume exactly."""
        return {
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val,
            "history": list(self.history),
            "optimizer": self.optimizer.state_dict(),
            "batch_gen_state": self.batch_gen.get_state(),
            "eval_gen_state": self.eval_gen.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore optimizer + RNG + progress from :meth:`state_dict`."""
        self.iter_num = int(cast(int, state["iter_num"]))
        self.best_val = float(cast(float, state["best_val_loss"]))
        history = cast(list[TrainHistoryItem], state["history"])
        self.history = [dict(item) for item in history]
        self.optimizer.load_state_dict(cast(dict[str, object], state["optimizer"]))
        self.batch_gen.set_state(cast(torch.Tensor, state["batch_gen_state"]))
        self.eval_gen.set_state(cast(torch.Tensor, state["eval_gen_state"]))
        torch.set_rng_state(cast(torch.Tensor, state["torch_rng_state"]))

    def train(
        self,
        train_ids: torch.Tensor,
        val_ids: torch.Tensor,
        on_eval: EvalHook | None = None,
    ) -> dict[str, object]:
        """Run the training loop; return history + best held-out loss."""
        cfg = self.cfg
        block = self.model.config.block_size
        dev = model_device(self.model)
        self.model.train()
        for it in range(self.iter_num, cfg.max_iters):
            lr = self.get_lr(it)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            x, y = get_batch(train_ids, block, cfg.batch_size, self.batch_gen)
            x, y = x.to(dev), y.to(dev)
            _, loss = self.model(x, y)
            assert isinstance(loss, torch.Tensor)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
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
                self.iter_num = it + 1
                self.best_val = min(self.best_val, va)
                self.history.append({"iter": it, "lr": lr, "train_loss": tr, "val_loss": va})
                if on_eval is not None:
                    on_eval(it, tr, va)
            else:
                self.iter_num = it + 1
        return {"history": list(self.history), "best_val_loss": self.best_val}
