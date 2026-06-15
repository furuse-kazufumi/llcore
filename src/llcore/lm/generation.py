# SPDX-License-Identifier: Apache-2.0
"""Text generation + a degeneracy gate for the char-LM.

A model can game a low validation loss while still collapsing into repetition when
sampled, so the P0 acceptance check pairs the perplexity gate with a hard
non-degeneracy check on a generated sample (:func:`is_degenerate`).
"""
from __future__ import annotations

from typing import Protocol

import torch

from llcore.lm.tokenizer import CharTokenizer


class SupportsGenerate(Protocol):
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor: ...


def generate_text(
    model: SupportsGenerate,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    """Sample text continuing ``prompt``. Out-of-vocab prompt chars map to id 0.

    The returned string includes the prompt followed by the sampled continuation.
    """
    if seed is not None:
        torch.manual_seed(seed)
    ids = tokenizer.encode_safe(prompt) if prompt else [0]
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def is_degenerate(text: str, min_distinct: int = 15, max_repeat_run: int = 10) -> bool:
    """Heuristic degeneracy gate. Returns ``True`` if the sample looks collapsed.

    Flags either (a) too few distinct characters (``< min_distinct``) or (b) any
    back-to-back repetition of a substring of length ``>= max_repeat_run`` (a loop).
    """
    if len(set(text)) < min_distinct:
        return True
    n = len(text)
    for length in range(max_repeat_run, n // 2 + 1):
        for i in range(0, n - 2 * length + 1):
            if text[i : i + length] == text[i + length : i + 2 * length]:
                return True
    return False
