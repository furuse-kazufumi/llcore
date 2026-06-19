# SPDX-License-Identifier: Apache-2.0
"""Tests for arch-tagged checkpoint save/load across gpt / recurrent / rwkv.

The existing CLI only saves/loads CharGPT checkpoints (``{config, model_state, itos}``).
The constant-state recurrent models (RecurrentLM, RWKVLM) had no persistence path, so a
trained recurrent model could never be reloaded for long-context evaluation. This module
adds a single arch-tagged save/load that round-trips all three model families and stays
backward-compatible with the pre-existing un-tagged GPT checkpoints on disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM


def _itos() -> list[str]:
    return [chr(ord("a") + i) for i in range(16)]


def _gpt() -> CharGPT:
    torch.manual_seed(0)
    return CharGPT(GPTConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=16))


def _recurrent() -> RecurrentLM:
    torch.manual_seed(0)
    return RecurrentLM(
        RecurrentConfig(vocab_size=16, block_size=8, n_layer=2, n_embd=16, state_size=16)
    )


def _rwkv() -> RWKVLM:
    torch.manual_seed(0)
    return RWKVLM(RWKVConfig(vocab_size=16, block_size=8, n_layer=2, n_embd=16))


@pytest.mark.parametrize("factory", [_gpt, _recurrent, _rwkv])
def test_roundtrip_preserves_logits(factory, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from llcore.lm.checkpoint import load_lm_checkpoint, save_lm_checkpoint

    model = factory()
    model.eval()
    path = tmp_path / "model.pt"
    save_lm_checkpoint(path, model, _itos())

    loaded, tok = load_lm_checkpoint(path)
    loaded.eval()
    assert tok.itos == _itos()
    assert type(loaded) is type(model)

    x = torch.randint(0, 16, (1, 6))
    before = model.forward_logits(x)
    after = loaded.forward_logits(x)
    assert torch.equal(before, after)


@pytest.mark.parametrize(
    "factory,arch",
    [(_gpt, "gpt"), (_recurrent, "recurrent"), (_rwkv, "rwkv")],
)
def test_arch_tag_recorded(factory, arch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from llcore.lm.checkpoint import read_lm_arch, save_lm_checkpoint

    path = tmp_path / "model.pt"
    save_lm_checkpoint(path, factory(), _itos())
    assert read_lm_arch(path) == arch


def test_loads_legacy_untagged_gpt_checkpoint(tmp_path: Path) -> None:
    """Existing on-disk GPT checkpoints predate arch tagging; they must still load."""
    from llcore.lm.checkpoint import load_lm_checkpoint

    model = _gpt()
    model.eval()
    path = tmp_path / "legacy.pt"
    # exactly the legacy schema written by llcore.lm.__main__._save_checkpoint
    torch.save(
        {"config": vars(model.config), "model_state": model.state_dict(), "itos": _itos()},
        path,
    )
    loaded, tok = load_lm_checkpoint(path)
    assert isinstance(loaded, CharGPT)
    x = torch.randint(0, 16, (1, 6))
    assert torch.equal(model.forward_logits(x), loaded.forward_logits(x))


def test_load_rejects_unknown_arch(tmp_path: Path) -> None:
    from llcore.lm.checkpoint import load_lm_checkpoint

    path = tmp_path / "bogus.pt"
    torch.save({"kind": "llcore.lm.lm_ckpt.v1", "arch": "mamba", "config": {}, "model_state": {}, "itos": _itos()}, path)
    with pytest.raises(ValueError, match="mamba"):
        load_lm_checkpoint(path)
