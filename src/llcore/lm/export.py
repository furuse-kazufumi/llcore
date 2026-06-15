# SPDX-License-Identifier: Apache-2.0
"""Export a trained :class:`CharGPT` to bbycroft/llm-viz model JSON.

Because the model's submodule tree matches minGPT exactly, the export is a direct
``state_dict()`` walk — no key remapping. Each tensor is encoded as
``{"shape": [...], "dtype": "torch.float32", "data": <base64 little-endian f32>}``,
the format the visualizer's ``TensorF32.fromJson`` decodes (verified against
``llcore-viz/public/gpt-nano-sort-model.json``).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

from llcore.lm.model import CharGPT
from llcore.lm.tokenizer import CharTokenizer


def tensor_to_json(t: torch.Tensor) -> dict[str, object]:
    """Encode a tensor as the viz ITensorJson dict (little-endian float32)."""
    arr = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
    data = base64.b64encode(arr.tobytes()).decode("ascii")
    return {"shape": list(t.shape), "dtype": "torch.float32", "data": data}


def to_viz_dict(
    model: CharGPT,
    tokenizer: CharTokenizer | None = None,
    include_vocab: bool = False,
) -> dict[str, object]:
    """Build the full viz model dict: ``{"config": {...}, "<tensor>": {...}, ...}``."""
    cfg = model.config
    out: dict[str, object] = {
        "config": {
            "model_type": cfg.model_type,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
            "embd_pdrop": cfg.dropout,
            "resid_pdrop": cfg.dropout,
            "attn_pdrop": cfg.dropout,
        }
    }
    for key, value in model.state_dict().items():
        out[key] = tensor_to_json(value)
    if include_vocab and tokenizer is not None:
        out["vocab"] = tokenizer.itos
    return out


def save_viz_json(
    model: CharGPT,
    path: str | Path,
    tokenizer: CharTokenizer | None = None,
    include_vocab: bool = False,
) -> None:
    """Write :func:`to_viz_dict` to ``path`` as compact JSON."""
    payload = to_viz_dict(model, tokenizer, include_vocab)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
