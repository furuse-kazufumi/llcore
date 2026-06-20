# SPDX-License-Identifier: Apache-2.0
"""Tests for internal attention surgery: replace Qwen2 softmax attention with a constant-state
linear (recurrent) attention, reusing the pretrained q/k/v/o projections.

This is the deep-internal research lever (not a wrapper): the attention CORE changes from
``softmax(QKᵀ/√d)V`` (KV cache O(T), compute O(T²)) to a linear attention whose running state
``S = Σ φ(k)⊗v`` is O(d²) and constant in sequence length — the architectural basis for
bounded-memory × unbounded-context conversation. A hybrid swap (linearize a subset of layers)
lets us measure each layer's linearization tolerance. Reusing the pretrained projection weights
is what makes the swap a surgery on a *capable* model rather than a fresh untrained net.
"""
from __future__ import annotations

import torch

from llcore.runtime.qwen2 import Qwen2Attention, Qwen2LM, Qwen2Params


def _tiny() -> Qwen2LM:
    torch.manual_seed(0)
    p = Qwen2Params(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        n_layer=3,
        n_head=4,
        n_kv_head=2,
        head_dim=8,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_embeddings=True,
        max_position=512,
    )
    return Qwen2LM(p).eval()


def test_linearize_no_layers_is_identity() -> None:
    from llcore.runtime.linearize import linearize_qwen2

    model = _tiny()
    ids = torch.randint(0, 48, (1, 16))
    with torch.no_grad():
        before = model(ids)
        linearize_qwen2(model, [])  # swap nothing
        after = model(ids)
    assert torch.equal(before, after)


def test_linearize_swaps_selected_layers_and_shares_weights() -> None:
    from llcore.runtime.linearize import LinearAttention, linearize_qwen2

    model = _tiny()
    original_q = model.model.layers[1].self_attn.q_proj
    linearize_qwen2(model, [1])
    assert isinstance(model.model.layers[1].self_attn, LinearAttention)
    assert isinstance(model.model.layers[0].self_attn, Qwen2Attention)
    assert isinstance(model.model.layers[2].self_attn, Qwen2Attention)
    # the linearized layer REUSES the pretrained projection (same object, not a copy)
    assert model.model.layers[1].self_attn.q_proj is original_q


def test_linear_attention_chunk_size_invariant() -> None:
    """The running-state recurrence must be exact: chunk_size only bounds memory, not the result."""
    from llcore.runtime.linearize import LinearAttention

    model = _tiny()
    attn = LinearAttention.from_attention(model.model.layers[0].self_attn, model.params)
    x = torch.randn(1, 40, model.params.hidden_size)
    pos = torch.arange(40)
    from llcore.runtime.qwen2 import _rope_cos_sin

    cos, sin = _rope_cos_sin(pos, model.params.head_dim, model.params.rope_theta)
    with torch.no_grad():
        attn.chunk_size = 4
        a, _ = attn(x, cos, sin, None, 0)
        attn.chunk_size = 64
        b, _ = attn(x, cos, sin, None, 0)
    assert torch.allclose(a, b, atol=1e-5)


def test_linearized_model_forward_finite_and_differs() -> None:
    from llcore.runtime.linearize import linearize_qwen2

    model = _tiny()
    ids = torch.randint(0, 48, (1, 24))
    with torch.no_grad():
        softmax_logits = model(ids)
        linearize_qwen2(model, list(range(model.params.n_layer)))  # fully linearize
        linear_logits = model(ids)
    assert linear_logits.shape == softmax_logits.shape
    assert torch.isfinite(linear_logits).all()
    # linear attention with reused weights is a real architectural change -> different distribution
    assert not torch.allclose(linear_logits, softmax_logits, atol=1e-2)


def test_constant_state_bytes_independent_of_length() -> None:
    """The linear-attention running state is O(d^2) per head — constant in sequence length."""
    from llcore.runtime.linearize import LinearAttention

    model = _tiny()
    attn = LinearAttention.from_attention(model.model.layers[0].self_attn, model.params)
    assert attn.state_bytes() == attn.state_bytes()  # deterministic
    p = model.params
    expected = p.n_head * p.head_dim * p.head_dim * 4 + p.n_head * p.head_dim * 4  # S + z, float32
    assert attn.state_bytes() == expected
