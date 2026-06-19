# SPDX-License-Identifier: Apache-2.0
"""Arch-tagged checkpoint persistence for the char-LM families.

The original CLI (``llcore.lm.__main__``) only knew how to save/load :class:`CharGPT`
checkpoints, with the flat schema ``{config, model_state, itos}``. The constant-state
recurrent models (:class:`RecurrentLM`, :class:`RWKVLM`) had no persistence path at all,
so a *trained* recurrent model could never be reloaded — only ``compare.py`` trained them,
in memory, and threw them away. That blocked the whole "score a trained recurrent model on
long context" workflow.

This module adds one arch-tagged save/load that round-trips all three families:

``{"kind": "llcore.lm.lm_ckpt.v1", "arch": "gpt"|"recurrent"|"rwkv", "config": {...},
   "model_state": {...}, "itos": [...]}``

and stays backward-compatible with the pre-existing un-tagged GPT checkpoints already on
disk (a missing ``arch`` is read as ``"gpt"``).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM
from llcore.lm.tokenizer import CharTokenizer

if TYPE_CHECKING:
    from torch import nn

CKPT_KIND = "llcore.lm.lm_ckpt.v1"

ARCH_GPT = "gpt"
ARCH_RECURRENT = "recurrent"
ARCH_RWKV = "rwkv"

LM = CharGPT | RecurrentLM | RWKVLM


def arch_of(model: nn.Module) -> str:
    """Return the arch tag for a char-LM instance."""
    if isinstance(model, CharGPT):
        return ARCH_GPT
    if isinstance(model, RecurrentLM):
        return ARCH_RECURRENT
    if isinstance(model, RWKVLM):
        return ARCH_RWKV
    raise TypeError(f"unsupported model type for checkpointing: {type(model).__name__}")


def save_lm_checkpoint(path: str | Path, model: LM, itos: list[str]) -> None:
    """Persist a char-LM (gpt / recurrent / rwkv) with its arch tag, config and tokenizer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": CKPT_KIND,
            "arch": arch_of(model),
            "config": asdict(model.config),
            "model_state": model.state_dict(),
            "itos": list(itos),
        },
        path,
    )


def read_lm_arch(path: str | Path) -> str:
    """Cheaply read the arch tag of a checkpoint (legacy un-tagged GPT reads as ``"gpt"``)."""
    head = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    return str(head.get("arch", ARCH_GPT))


def _build(arch: str, config: dict[str, object]) -> LM:
    if arch == ARCH_GPT:
        return CharGPT(GPTConfig(**config))  # type: ignore[arg-type]
    if arch == ARCH_RECURRENT:
        return RecurrentLM(RecurrentConfig(**config))  # type: ignore[arg-type]
    if arch == ARCH_RWKV:
        return RWKVLM(RWKVConfig(**config))  # type: ignore[arg-type]
    raise ValueError(f"unknown checkpoint arch {arch!r}")


def load_lm_checkpoint(path: str | Path) -> tuple[LM, CharTokenizer]:
    """Load a gpt / recurrent / rwkv checkpoint plus its tokenizer (tensor-only, safe load)."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    arch = str(ckpt.get("arch", ARCH_GPT))
    model = _build(arch, ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharTokenizer(ckpt["itos"])
