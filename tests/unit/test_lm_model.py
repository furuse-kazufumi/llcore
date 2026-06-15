# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.model` — shapes, loss, viz-schema state_dict keys."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llcore.lm.model import CharGPT, GPTConfig

# The bbycroft/llm-viz sample model (gpt-nano: n_layer=3, n_head=3, n_embd=48,
# vocab_size=3, block_size=11) — used to pin the exact state_dict key contract.
SAMPLE_CFG = GPTConfig(vocab_size=3, block_size=11, n_layer=3, n_head=3, n_embd=48)
VIZ_SAMPLE = Path("D:/projects/llcore-viz/public/gpt-nano-sort-model.json")


def expected_viz_keys(n_layer: int) -> set[str]:
    """The exact tensor-key set the llm-viz schema expects for ``n_layer`` blocks."""
    keys = {"transformer.wte.weight", "transformer.wpe.weight"}
    for i in range(n_layer):
        p = f"transformer.h.{i}"
        keys |= {
            f"{p}.ln_1.weight",
            f"{p}.ln_1.bias",
            f"{p}.attn.bias",
            f"{p}.attn.c_attn.weight",
            f"{p}.attn.c_attn.bias",
            f"{p}.attn.c_proj.weight",
            f"{p}.attn.c_proj.bias",
            f"{p}.ln_2.weight",
            f"{p}.ln_2.bias",
            f"{p}.mlp.c_fc.weight",
            f"{p}.mlp.c_fc.bias",
            f"{p}.mlp.c_proj.weight",
            f"{p}.mlp.c_proj.bias",
        }
    keys |= {"transformer.ln_f.weight", "transformer.ln_f.bias", "lm_head.weight"}
    return keys


def test_forward_shape() -> None:
    cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    idx = torch.randint(0, 16, (3, 8))
    logits, loss = model(idx)
    assert logits.shape == (3, 8, 16)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss() -> None:
    cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    idx = torch.randint(0, 16, (3, 8))
    targets = torch.randint(0, 16, (3, 8))
    logits, loss = model(idx, targets)
    assert loss is not None
    assert loss.ndim == 0
    # A random-init model over 16 classes should be near ln(16) nats.
    assert 1.5 < loss.item() < 4.0


def test_block_size_enforced() -> None:
    cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, 16, (1, 9)))


def test_state_dict_keys_match_viz_schema() -> None:
    """The critical viz-export contract: state_dict keys == minGPT/llm-viz keys."""
    model = CharGPT(SAMPLE_CFG)
    assert set(model.state_dict().keys()) == expected_viz_keys(SAMPLE_CFG.n_layer)


def test_state_dict_shapes() -> None:
    model = CharGPT(SAMPLE_CFG)
    sd = model.state_dict()
    c = SAMPLE_CFG
    assert tuple(sd["transformer.wte.weight"].shape) == (c.vocab_size, c.n_embd)
    assert tuple(sd["transformer.wpe.weight"].shape) == (c.block_size, c.n_embd)
    assert tuple(sd["transformer.h.0.attn.c_attn.weight"].shape) == (3 * c.n_embd, c.n_embd)
    assert tuple(sd["transformer.h.0.attn.c_proj.weight"].shape) == (c.n_embd, c.n_embd)
    assert tuple(sd["transformer.h.0.attn.bias"].shape) == (1, 1, c.block_size, c.block_size)
    assert tuple(sd["transformer.h.0.mlp.c_fc.weight"].shape) == (4 * c.n_embd, c.n_embd)
    assert tuple(sd["transformer.h.0.mlp.c_proj.weight"].shape) == (c.n_embd, 4 * c.n_embd)
    assert tuple(sd["lm_head.weight"].shape) == (c.vocab_size, c.n_embd)
    # dtype is float32 (viz only accepts torch.float32)
    assert sd["transformer.wte.weight"].dtype == torch.float32


def test_weight_tying() -> None:
    model = CharGPT(SAMPLE_CFG)
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_causal_mask_is_lower_triangular() -> None:
    model = CharGPT(SAMPLE_CFG)
    mask = model.state_dict()["transformer.h.0.attn.bias"][0, 0]
    assert torch.equal(mask, torch.tril(torch.ones(SAMPLE_CFG.block_size, SAMPLE_CFG.block_size)))


def test_attention_is_causal() -> None:
    """Changing a later token must not affect an earlier position's logits."""
    cfg = GPTConfig(vocab_size=8, block_size=6, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = CharGPT(cfg).eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 0]])
    b = a.clone()
    b[0, -1] = 7  # change only the LAST token
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    # positions 0..4 must be identical (cannot attend to position 5)
    assert torch.allclose(la[0, :5], lb[0, :5], atol=1e-6)
    # the last position is allowed to differ
    assert not torch.allclose(la[0, 5], lb[0, 5], atol=1e-6)


def test_generate_appends_valid_tokens() -> None:
    cfg = GPTConfig(vocab_size=8, block_size=6, n_layer=1, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20, temperature=0.8, top_k=4)
    assert out.shape == (1, 21)
    assert out.min().item() >= 0 and out.max().item() < cfg.vocab_size


def test_generate_handles_context_overflow() -> None:
    """Generation past block_size must crop context, not raise."""
    cfg = GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=10)
    assert out.shape == (1, 11)


def test_generate_rejects_empty_prompt() -> None:
    cfg = GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    with pytest.raises(ValueError, match="at least one prompt token"):
        model.generate(torch.zeros((1, 0), dtype=torch.long), max_new_tokens=1)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(vocab_size=8, block_size=4, n_head=3, n_embd=16)
    with pytest.raises(ValueError, match="vocab_size"):
        GPTConfig(vocab_size=0, block_size=4)
    with pytest.raises(ValueError, match="block_size"):
        GPTConfig(vocab_size=8, block_size=0)


def test_num_params_sane() -> None:
    model = CharGPT(SAMPLE_CFG)
    # tied wte/lm_head counted once; positional embeddings subtracted by default.
    assert model.num_params(non_embedding=True) > 0
    assert model.num_params(non_embedding=False) > model.num_params(non_embedding=True)


@pytest.mark.skipif(not VIZ_SAMPLE.exists(), reason="llm-viz sample not present")
def test_against_real_viz_sample() -> None:
    """Cross-check the key contract against the actual llm-viz sample JSON."""
    data = json.loads(VIZ_SAMPLE.read_text())
    sample_keys = {k for k in data if k != "config"}
    model = CharGPT(SAMPLE_CFG)
    assert set(model.state_dict().keys()) == sample_keys
    # shapes agree key-by-key
    sd = model.state_dict()
    for k, v in data.items():
        if k == "config":
            continue
        assert tuple(sd[k].shape) == tuple(v["shape"]), k
