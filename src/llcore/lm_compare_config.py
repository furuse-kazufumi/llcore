# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompareConfig:
    """Small comparison recipe intended for CPU smoke runs."""

    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    state_size: int = 64
    max_iters: int = 120
    batch_size: int = 12
    eval_iters: int = 4
    throughput_prompt_lens: tuple[int, ...] | None = None
    throughput_new_tokens: int = 16
    throughput_repeats: int = 3
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.n_head <= 0:
            raise ValueError(f"n_head must be > 0, got {self.n_head}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.throughput_new_tokens < 1:
            raise ValueError(
                f"throughput_new_tokens must be >= 1, got {self.throughput_new_tokens}"
            )
        if self.throughput_repeats < 1:
            raise ValueError(
                f"throughput_repeats must be >= 1, got {self.throughput_repeats}"
            )
        if self.throughput_prompt_lens is not None and any(
            prompt_len < 1 for prompt_len in self.throughput_prompt_lens
        ):
            raise ValueError("throughput_prompt_lens must contain only positive lengths")
