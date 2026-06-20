# SPDX-License-Identifier: Apache-2.0
"""Memory-frugal loader for a real pretrained Qwen2 model into the llcore-native forward.

``Qwen2LM.load_hf_state_dict`` builds a full float32 copy of the HF state dict before loading,
which momentarily holds (bf16 weights + float32 weights) at once — fine for 0.5B, but a needless
~1.5× spike that risks OOM for 1.5B+. This loader instead streams the weights **one tensor at a
time** straight from the (memory-mapped) safetensors file into the model's parameters, so peak RAM
is ~the model itself plus a single tensor. That is the llcore memory-efficiency stance applied to
*loading*: never hold the whole model twice.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import torch

from llcore.runtime.qwen2 import Qwen2LM, Qwen2Params


def load_qwen2(model_dir: str | Path, dtype: torch.dtype = torch.float32) -> tuple[Qwen2LM, Any, Qwen2Params]:
    """Load a Qwen2 model directory (config + safetensors + tokenizer) with per-tensor streaming.

    Returns ``(model, tokenizer, params)``. ``dtype`` is the resident weight dtype (float32 default;
    pass ``torch.bfloat16`` to halve resident RAM at some CPU-speed cost).
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    params = Qwen2Params.from_hf_config(cfg)
    model = Qwen2LM(params)
    if dtype != torch.float32:
        model = model.to(dtype)
    model.eval()

    own = dict(model.named_parameters())  # tied lm_head is deduplicated to embed_tokens
    loaded: set[str] = set()
    files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {model_dir}")
    with torch.no_grad():
        for fp in files:
            with safe_open(fp, framework="pt") as f:  # lazy / mmap — one tensor read at a time
                for name in f.keys():  # noqa: SIM118
                    if name in own and tuple(f.get_slice(name).get_shape()) == tuple(own[name].shape):
                        own[name].copy_(f.get_tensor(name).to(dtype))
                        loaded.add(name)
    missing = [n for n in own if n not in loaded]
    if missing and not (params.tie_embeddings and missing == []):
        # tied embeddings legitimately leave nothing missing (lm_head shares embed_tokens);
        # anything else missing is a real load gap worth surfacing.
        unexpected = [m for m in missing if "lm_head" not in m]
        if unexpected:
            raise RuntimeError(f"{len(unexpected)} parameters not found in safetensors: {unexpected[:5]}")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    return model, tok, params
