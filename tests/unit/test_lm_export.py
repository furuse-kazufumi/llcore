# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.export` — viz JSON structure + byte round-trip."""
from __future__ import annotations

import base64
import json

import numpy as np
import torch

from llcore.lm.export import save_viz_json, tensor_to_json, to_viz_dict
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.tokenizer import CharTokenizer
from tests.unit.test_lm_model import SAMPLE_CFG, expected_viz_keys


def test_tensor_to_json_roundtrip() -> None:
    t = torch.randn(4, 3)
    j = tensor_to_json(t)
    assert j["shape"] == [4, 3]
    assert j["dtype"] == "torch.float32"
    buf = base64.b64decode(j["data"])  # type: ignore[arg-type]
    arr = np.frombuffer(buf, dtype="<f4").reshape(4, 3)
    assert np.allclose(arr, t.numpy(), atol=1e-6)


def test_to_viz_dict_keys_and_config() -> None:
    model = CharGPT(SAMPLE_CFG)
    d = to_viz_dict(model)
    assert set(d.keys()) == {"config"} | expected_viz_keys(SAMPLE_CFG.n_layer)
    cfg = d["config"]
    assert isinstance(cfg, dict)
    assert cfg["model_type"] == "gpt-nano"
    assert cfg["n_layer"] == SAMPLE_CFG.n_layer
    assert cfg["vocab_size"] == SAMPLE_CFG.vocab_size
    assert cfg["block_size"] == SAMPLE_CFG.block_size


def test_to_viz_dict_tensor_values_match_state_dict() -> None:
    model = CharGPT(SAMPLE_CFG)
    d = to_viz_dict(model)
    sd = model.state_dict()
    for key in ("transformer.wte.weight", "transformer.h.0.attn.c_attn.weight", "lm_head.weight"):
        entry = d[key]
        assert isinstance(entry, dict)
        buf = base64.b64decode(entry["data"])
        arr = np.frombuffer(buf, dtype="<f4").reshape(entry["shape"])
        assert np.allclose(arr, sd[key].numpy(), atol=1e-6)


def test_include_vocab() -> None:
    text = "abc\n"
    tok = CharTokenizer.from_text(text)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = CharGPT(cfg)
    d = to_viz_dict(model, tok, include_vocab=True)
    assert d["vocab"] == tok.itos


def test_save_viz_json_is_loadable(tmp_path) -> None:
    model = CharGPT(SAMPLE_CFG)
    path = tmp_path / "model.json"
    save_viz_json(model, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"config"} | expected_viz_keys(SAMPLE_CFG.n_layer)
    # every tensor entry is well-formed
    for key, value in data.items():
        if key == "config":
            continue
        assert set(value.keys()) == {"shape", "dtype", "data"}
        assert value["dtype"] == "torch.float32"
