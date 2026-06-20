# SPDX-License-Identifier: Apache-2.0
"""Tests for distillation recovery of linearized attention layers.

The per-layer tolerance profile shows the *zero-shot* ceiling. The genuine contribution is to
*recover* quality: give the linear attention a small learnable feature map (frozen projections) and
distill its output to match the softmax teacher on a calibration corpus (LoLCATs-style). Then more
layers can be linearized at low cost. These tests pin: (a) the learnable feature is identity at init
(so a freshly-learnable linear attention == the fixed one — no silent behavior change), and (b)
distilling one layer reduces the teacher/student output gap.
"""
from __future__ import annotations

import torch

from llcore.runtime.linearize import LinearAttention
from llcore.runtime.qwen2 import Qwen2LM, Qwen2Params, _rope_cos_sin


def _tiny() -> Qwen2LM:
    torch.manual_seed(0)
    p = Qwen2Params(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        head_dim=8,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_embeddings=True,
        max_position=256,
    )
    return Qwen2LM(p).eval()


def test_learnable_feature_is_identity_at_init() -> None:
    model = _tiny()
    src = model.model.layers[0].self_attn
    fixed = LinearAttention.from_attention(src, model.params)
    learn = LinearAttention.from_attention(src, model.params, learnable=True)
    x = torch.randn(1, 18, model.params.hidden_size)
    cos, sin = _rope_cos_sin(torch.arange(18), model.params.head_dim, model.params.rope_theta)
    with torch.no_grad():
        a, _ = fixed(x, cos, sin, None, 0)
        b, _ = learn(x, cos, sin, None, 0)
    assert torch.allclose(a, b, atol=1e-6)
    # the learnable variant exposes trainable feature parameters
    trainable = [n for n, p in learn.named_parameters() if p.requires_grad and "proj" not in n]
    assert trainable, "learnable feature must expose its own parameters"


def test_distill_layer_reduces_teacher_student_gap() -> None:
    from llcore.runtime.distill import distill_layer

    model = _tiny()
    torch.manual_seed(1)
    ids = torch.randint(0, 48, (1, 48))
    result = distill_layer(model, layer=1, calib_ids=ids, steps=60, lr=5e-2, seed=0)
    assert result["mse_after"] < result["mse_before"]
    # the distilled layer is installed on the model and is a learnable LinearAttention
    assert isinstance(model.model.layers[1].self_attn, LinearAttention)


def test_distill_all_layers_returns_student_per_layer_and_restores() -> None:
    from llcore.runtime.distill import distill_all_layers
    from llcore.runtime.qwen2 import Qwen2Attention

    model = _tiny()  # n_layer = 2, all softmax
    torch.manual_seed(2)
    ids = torch.randint(0, 48, (1, 40))
    students = distill_all_layers(model, ids, steps=30, lr=5e-2)
    assert set(students) == {0, 1}
    assert all(isinstance(s, LinearAttention) for s in students.values())
    # each layer was distilled from a pristine softmax teacher → model is left all-softmax
    assert all(isinstance(model.model.layers[i].self_attn, Qwen2Attention) for i in range(2))


def test_distill_all_layers_rejects_non_softmax_model() -> None:
    import pytest

    from llcore.runtime.distill import distill_all_layers
    from llcore.runtime.linearize import linearize_qwen2

    model = _tiny()
    linearize_qwen2(model, [0])  # layer 0 is now LinearAttention, not a softmax teacher
    ids = torch.randint(0, 48, (1, 24))
    with pytest.raises(ValueError, match="all-softmax"):
        distill_all_layers(model, ids, steps=5)
