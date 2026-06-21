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
from typing import Any, cast

import torch
from torch import nn

from llcore.lm.quant import Int8Linear, quantize_per_channel_int8
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
            # cast→Any: safetensors の版で safe_open が typed/untyped 両方あり得るため、
            # no-untyped-call(untyped 版)も unused-ignore(typed 版)も出さない頑健形にする。
            with cast("Any", safe_open)(fp, framework="pt") as f:  # lazy/mmap
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


def _module_by_path(root: nn.Module, path: str) -> nn.Module:
    obj: Any = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj  # type: ignore[no-any-return]


def load_qwen2_int8(model_dir: str | Path) -> tuple[Qwen2LM, Any, Qwen2Params]:
    """Streaming-int8 load: quantize the decoder Linears per tensor straight from safetensors.

    The fp32 model is NEVER materialized. The model is built on the ``meta`` device (zero memory),
    its decoder-layer Linears (q/k/v/o, gate/up/down) are swapped for zero-init :class:`Int8Linear`,
    the remaining tensors (token embedding, RMSNorm weights, attention biases, tied lm_head) are
    materialized as fp32, and weights are filled one tensor at a time from the (mmap'd) safetensors —
    each Linear weight quantized to int8 on read and the transient fp32 freed immediately. Resident
    RAM ≈ int8 decoder weights + the fp32 embedding (kept full-precision for quality), instead of the
    whole fp32 model. Embedding and lm_head stay fp32 (and tied); only the matmul-heavy Linears go int8.
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    params = Qwen2Params.from_hf_config(cfg)

    with torch.device("meta"):
        model = Qwen2LM(params)
    for blk in model.model.layers:
        for sub in (blk.self_attn, blk.mlp):
            for name, child in list(sub.named_children()):  # type: ignore[union-attr]
                if isinstance(child, nn.Linear):
                    setattr(sub, name, Int8Linear(child.in_features, child.out_features, bias=child.bias is not None))
    model.to_empty(device="cpu")
    if params.tie_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()

    own_params = dict(model.named_parameters())
    files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {model_dir}")
    with torch.no_grad():
        for fp in files:
            with cast("Any", safe_open)(fp, framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    mod_path, _, leaf = key.rpartition(".")
                    mod = _module_by_path(model, mod_path) if mod_path else model
                    if isinstance(mod, Int8Linear) and leaf == "weight":
                        q, scale = quantize_per_channel_int8(f.get_tensor(key).float())
                        mod.qweight.copy_(q)
                        mod.scale.copy_(scale)
                    elif isinstance(mod, Int8Linear) and leaf == "bias" and mod.bias is not None:
                        mod.bias.copy_(f.get_tensor(key).float())
                    elif key in own_params:
                        own_params[key].copy_(f.get_tensor(key).float())

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    return model, tok, params
