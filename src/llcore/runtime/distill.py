# SPDX-License-Identifier: Apache-2.0
"""Distillation recovery for linearized attention layers (LoLCATs-style, on-prem CPU).

The per-layer tolerance profile (``scripts/linearize_tolerance.py``) measures the *zero-shot*
quality cost of replacing softmax attention with constant-state linear attention. This module is
the recovery step — the genuine contribution beyond measurement: give the linear attention a small
learnable feature map (the projections stay frozen) and train it so its attention OUTPUT matches the
softmax teacher's output on a calibration corpus. After distillation a layer can be linearized at a
smaller quality cost than zero-shot, so more layers fit the constant-memory budget.

Output distillation, per layer, decoupled from the full model: a single softmax forward captures the
layer's attention input (its ``input_layernorm`` output) and the softmax attention output; the
student (linear attention reusing the same projections) is then trained offline to match — cheap on
CPU because only the small attention runs in the loop, not the whole model.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as F

from llcore.runtime.linearize import LinearAttention
from llcore.runtime.qwen2 import Qwen2Attention, Qwen2LM, _rope_cos_sin


def _capture_layer_io(
    model: Qwen2LM, layer: int, calib_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one softmax forward; return (attention input x_norm, softmax attention output)."""
    block = model.model.layers[layer]
    captured: dict[str, torch.Tensor] = {}

    def grab_x(_m: object, _i: object, o: torch.Tensor) -> None:
        captured["x"] = o.detach()

    def grab_teacher(_m: object, _i: object, o: tuple[torch.Tensor, object]) -> None:
        captured["teacher"] = o[0].detach()

    h1 = block.input_layernorm.register_forward_hook(grab_x)  # type: ignore[union-attr]
    h2 = block.self_attn.register_forward_hook(grab_teacher)  # type: ignore[union-attr]
    try:
        with torch.no_grad():
            model(calib_ids)
    finally:
        h1.remove()
        h2.remove()
    return captured["x"], captured["teacher"]


def distill_layer(
    model: Qwen2LM,
    layer: int,
    calib_ids: torch.Tensor,
    steps: int = 200,
    lr: float = 5e-2,
    seed: int = 0,
    chunk_size: int = 64,
    feature_map: str = "diag",
) -> dict[str, float]:
    """Distill layer ``layer``'s linear attention to match its softmax teacher; install it on ``model``.

    Returns ``{mse_before, mse_after, steps}``. ``mse_before`` is the zero-shot linear-vs-softmax
    output gap (identity-init feature map); ``mse_after`` is after training the feature map.
    ``feature_map`` selects the learnable map: ``"diag"`` (per-head affine, original) or ``"full"``
    (per-head full-linear matrix — higher capacity, the L7/Hedgehog-motivated upgrade).
    """
    torch.manual_seed(seed)
    src = model.model.layers[layer].self_attn
    if not isinstance(src, Qwen2Attention):
        raise ValueError(f"layer {layer} is not a softmax Qwen2Attention to distill from")
    x_norm, teacher = _capture_layer_io(model, layer, calib_ids)
    t = x_norm.size(1)
    cos, sin = _rope_cos_sin(torch.arange(t), model.params.head_dim, model.params.rope_theta)

    student = LinearAttention.from_attention(
        src, model.params, learnable=True, chunk_size=chunk_size, feature_map=feature_map
    )
    opt = torch.optim.AdamW(student.feature_parameters(), lr=lr)

    with torch.no_grad():
        mse_before = float(F.mse_loss(student(x_norm, cos, sin, None, 0)[0], teacher).item())
    mse_after = mse_before
    for _ in range(steps):
        out, _ = student(x_norm, cos, sin, None, 0)
        loss = F.mse_loss(out, teacher)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        mse_after = float(loss.item())

    model.model.layers[layer].self_attn = student
    return {"mse_before": mse_before, "mse_after": mse_after, "steps": float(steps)}


def distill_all_layers(
    model: Qwen2LM,
    calib_ids: torch.Tensor,
    *,
    steps: int = 200,
    lr: float = 5e-2,
    seed: int = 0,
    chunk_size: int = 64,
    on_layer: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[int, LinearAttention]:
    """Distill every layer independently from the pristine all-softmax teacher.

    For each layer ``i``, :func:`distill_layer` installs a learnable :class:`LinearAttention`
    student on the model; we capture it, then restore layer ``i`` to its original softmax attention
    before moving on, so each layer is distilled from the *unmodified* model (per-layer output
    distillation — joint multi-layer distillation, where errors may compound, is a separate
    experiment). The model is left in its original all-softmax state, so the caller may install any
    subset of the returned students at will. Returns ``{layer_index: distilled_student}``.

    Raises ``ValueError`` unless every layer is a softmax :class:`Qwen2Attention` (a valid teacher).
    ``on_layer`` (if given) is called with ``(layer_index, distill_layer_result)`` after each layer.
    """
    n_layer = model.params.n_layer
    originals: dict[int, Qwen2Attention] = {}
    for i in range(n_layer):
        attn = model.model.layers[i].self_attn
        if not isinstance(attn, Qwen2Attention):
            raise ValueError(
                f"distill_all_layers needs an all-softmax model as teacher; "
                f"layer {i} is {type(attn).__name__}"
            )
        originals[i] = attn
    students: dict[int, LinearAttention] = {}
    for i in range(n_layer):
        info = distill_layer(model, i, calib_ids, steps=steps, lr=lr, seed=seed, chunk_size=chunk_size)
        student = model.model.layers[i].self_attn
        assert isinstance(student, LinearAttention)
        students[i] = student
        model.model.layers[i].self_attn = originals[i]  # restore pristine softmax for the next teacher
        if on_layer is not None:
            on_layer(i, info)
    return students
