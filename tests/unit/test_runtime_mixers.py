# SPDX-License-Identifier: Apache-2.0
"""Tests for the NAS mixer zoo: a sliding-window softmax attention that bounds the KV cache.

For Level-2 architecture search each layer can pick a token mixer with a different memory/quality
profile. Sliding-window softmax sits between full softmax (O(T) KV, best quality) and linear
attention (O(1) state, lower zero-shot quality): it keeps softmax but attends only the last
``window`` keys, so its KV cache is bounded at O(window). It reuses the pretrained q/k/v/o
projections + RoPE, and reduces to full softmax when ``window`` >= sequence length.
"""
from __future__ import annotations

import torch

from llcore.runtime.qwen2 import Qwen2LM, Qwen2Params, _rope_cos_sin


def _tiny() -> Qwen2LM:
    torch.manual_seed(0)
    p = Qwen2Params(
        vocab_size=48, hidden_size=32, intermediate_size=64, n_layer=2, n_head=4,
        n_kv_head=2, head_dim=8, rope_theta=1000000.0, rms_norm_eps=1e-6,
        tie_embeddings=True, max_position=256,
    )
    return Qwen2LM(p).eval()


def test_sliding_window_equals_softmax_when_window_covers_sequence() -> None:
    from llcore.runtime.linearize import SlidingWindowAttention

    model = _tiny()
    src = model.model.layers[0].self_attn  # Qwen2Attention (full softmax)
    sw = SlidingWindowAttention.from_attention(src, model.params, window=10_000)
    x = torch.randn(1, 20, model.params.hidden_size)
    cos, sin = _rope_cos_sin(torch.arange(20), model.params.head_dim, model.params.rope_theta)
    with torch.no_grad():
        ref, _ = src(x, cos, sin, None, 0)
        got, _ = sw(x, cos, sin, None, 0)
    assert torch.allclose(ref, got, atol=2e-5)


def test_sliding_window_differs_with_small_window() -> None:
    from llcore.runtime.linearize import SlidingWindowAttention

    model = _tiny()
    src = model.model.layers[0].self_attn
    sw = SlidingWindowAttention.from_attention(src, model.params, window=4)
    x = torch.randn(1, 30, model.params.hidden_size)
    cos, sin = _rope_cos_sin(torch.arange(30), model.params.head_dim, model.params.rope_theta)
    with torch.no_grad():
        ref, _ = src(x, cos, sin, None, 0)
        got, _ = sw(x, cos, sin, None, 0)
    assert not torch.allclose(ref, got, atol=1e-3)
    # finite + right shape
    assert got.shape == ref.shape and torch.isfinite(got).all()


def test_sliding_window_kv_bytes_bounded() -> None:
    from llcore.runtime.linearize import SlidingWindowAttention

    model = _tiny()
    sw = SlidingWindowAttention.from_attention(model.model.layers[0].self_attn, model.params, window=16)
    p = model.params
    # at long context the bounded KV is window-capped, not T-grown
    assert sw.kv_bytes(context_len=4096) == 2 * p.n_kv_head * 16 * p.head_dim * 4
    assert sw.kv_bytes(context_len=8) == 2 * p.n_kv_head * 8 * p.head_dim * 4  # below window: actual len
