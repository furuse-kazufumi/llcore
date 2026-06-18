# SPDX-License-Identifier: Apache-2.0
"""int8 weight-only quantization + streaming-dequant / mmap inference for the char-LM.

Promotes the memory-efficiency PoCs (``scripts/int8_streaming_infer.py``,
``scripts/mmap_ram_exceed_poc.py``, ``scripts/int8_quant_footprint.py``) into a reusable
inference path on :class:`~llcore.lm.model.CharGPT`:

- :func:`quantize_per_channel_int8` — per-output-row symmetric int8 of a 2-D weight.
- :class:`Int8Linear` — a drop-in ``nn.Linear`` that keeps the weight as int8 and
  dequantizes it *inside* ``forward`` (a transient fp32 weight, freed after the matmul),
  so the resident weight footprint is ~1/4 of fp32 (measured: 539 MB → 149 MB for a
  130M-param model; ``docs/MEMORY_EFFICIENCY_FINDINGS.md`` (c)).
- :func:`save_int8_checkpoint` / :func:`load_int8_model` — persist and ``mmap``-load an
  int8 checkpoint so cold weight pages stay on disk until touched (llama.cpp style;
  ``docs/MEMORY_EFFICIENCY_FINDINGS.md`` (a')).

honest scope: weights-only (activations stay fp32); the dequantized matmul runs in fp32
(this is a *footprint*/working-set optimization, not an int8-GEMM speedup). Only
``nn.Linear`` weights are quantized — ``nn.Embedding`` and ``LayerNorm`` stay fp32.
PPL cost is < 0.1% per-channel on the trained char-LMs (``docs/...`` (b)).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llcore.lm.model import CharGPT, GPTConfig

INT8_QMAX = 127
INT8_CKPT_KIND = "llcore.lm.int8.v1"


def quantize_per_channel_int8(w: Tensor) -> tuple[Tensor, Tensor]:
    """Per-output-row symmetric int8 quantization of a 2-D weight.

    Returns ``(qweight[int8], scale[float32])`` where ``scale`` has one entry per output
    row (``w.size(0)``). ``amax`` is floored to avoid division by zero on all-zero rows.
    """
    if w.dim() != 2:
        raise ValueError(f"quantize_per_channel_int8 expects a 2-D weight, got {w.dim()}-D")
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / INT8_QMAX
    q = torch.clamp(torch.round(w / scale), -INT8_QMAX, INT8_QMAX).to(torch.int8)
    return q, scale.to(torch.float32)


def dequantize(q: Tensor, scale: Tensor) -> Tensor:
    """Reconstruct an fp32 weight from ``(qweight, scale)``."""
    return q.to(torch.float32) * scale


class Int8Linear(nn.Module):
    """A drop-in replacement for ``nn.Linear`` that stores int8 weights.

    The int8 ``qweight`` (buffer) and per-row ``scale`` (buffer) stay resident; the fp32
    weight is materialized only transiently inside :meth:`forward` and freed afterwards,
    bounding the simultaneous fp32 footprint to a single layer.
    """

    qweight: Tensor
    scale: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float32))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Int8Linear:
        """Quantize an existing ``nn.Linear`` into an :class:`Int8Linear`."""
        module = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        q, scale = quantize_per_channel_int8(linear.weight.data)
        module.qweight.copy_(q)
        module.scale.copy_(scale)
        if linear.bias is not None and module.bias is not None:
            module.bias.data.copy_(linear.bias.data)
        return module

    def forward(self, x: Tensor) -> Tensor:
        # Transient fp32 weight: dropped when this scope exits, so the simultaneous
        # fp32 materialization is bounded by this single layer's weight.
        weight = dequantize(self.qweight, self.scale)
        return F.linear(x, weight, self.bias)


def convert_linears_to_int8(model: nn.Module) -> nn.Module:
    """Recursively replace every ``nn.Linear`` in ``model`` with an :class:`Int8Linear`."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, Int8Linear):
            setattr(model, name, Int8Linear.from_linear(child))
        else:
            convert_linears_to_int8(child)
    return model


def _build_int8_skeleton(config: GPTConfig) -> CharGPT:
    """A CharGPT whose Linears are zero-initialised Int8Linears (load target)."""
    model = CharGPT(config)
    convert_linears_to_int8(model)
    return model


def save_int8_checkpoint(model: CharGPT, path: str | Path, itos: list[str]) -> None:
    """Quantize ``model``'s Linears to int8 and write a mmap-friendly checkpoint.

    The checkpoint stores the model config, the int8 state dict (int8 ``qweight`` +
    fp32 ``scale`` for each Linear, fp32 for embeddings / LayerNorm), and the tokenizer
    ``itos`` so :func:`load_int8_model` can round-trip without external metadata.
    """
    torch.save(
        {
            "kind": INT8_CKPT_KIND,
            "config": vars(model.config),
            "model_state": _dense_to_int8_state(model),
            "itos": list(itos),
        },
        path,
    )


def _dense_to_int8_state(model: CharGPT) -> dict[str, Tensor]:
    """Build an int8-skeleton state dict from a trained fp32 CharGPT.

    Iterates the int8 skeleton's keys and fills them from the dense model's
    ``state_dict()`` (which, unlike ``named_parameters()``, keeps tied keys such as
    ``lm_head.weight`` under both names — needed because ``lm_head`` becomes an
    :class:`Int8Linear` while ``wte`` stays a fp32 embedding).
    """
    skeleton = convert_linears_to_int8(CharGPT(model.config))
    src = model.state_dict()
    state: dict[str, Tensor] = {}
    for name in skeleton.state_dict():
        if name.endswith(".qweight"):
            q, _ = quantize_per_channel_int8(src[name[: -len(".qweight")] + ".weight"])
            state[name] = q
        elif name.endswith(".scale"):
            _, scale = quantize_per_channel_int8(src[name[: -len(".scale")] + ".weight"])
            state[name] = scale
        else:
            state[name] = src[name]  # embeddings / LayerNorm / biases / attn mask buffer
    return state


def load_int8_model(path: str | Path, *, mmap: bool = True) -> tuple[CharGPT, list[str]]:
    """Load an int8 checkpoint as a streaming-dequant CharGPT (optionally ``mmap``).

    With ``mmap=True`` the int8 weights are file-backed (cold pages stay on disk until
    a forward touches them). Returns ``(model, itos)``; the model's ``forward`` /
    ``generate`` run with per-layer int8→fp32 dequantization.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if ckpt.get("kind") != INT8_CKPT_KIND:
        raise ValueError(f"{path} is not a {INT8_CKPT_KIND} checkpoint")
    model = _build_int8_skeleton(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"], assign=True)
    model.eval()
    return model, list(ckpt["itos"])


def int8_footprint_bytes(model: CharGPT) -> dict[str, int]:
    """Resident weight bytes of ``model`` as fp32 vs int8 (Linear-only quantization).

    Honest accounting: int8 body (1 B/elem for Linear weights) + fp32 scales + fp32 for
    everything not quantized (embeddings, LayerNorm, biases).
    """
    fp32 = 0
    int8 = 0
    seen: set[int] = set()
    linear_weight_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)
    }
    for tensor in list(model.parameters()) + list(model.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        numel = tensor.numel()
        fp32 += numel * 4
        if id(tensor) in linear_weight_ids:
            int8 += numel * 1 + tensor.size(0) * 4  # int8 body + per-row fp32 scale
        else:
            int8 += numel * 4
    return {"fp32_bytes": fp32, "int8_bytes": int8}
