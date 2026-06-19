# SPDX-License-Identifier: Apache-2.0
"""Golden correctness tests for the llcore-native Qwen2 forward.

The honest-disclosure requirement for path A ("llcore runs the conversation, not an opaque
external runtime") is that llcore's own forward must reproduce the reference implementation
exactly. We verify the native Qwen2 forward against HuggingFace ``Qwen2ForCausalLM`` on a
tiny random config (cheap RAM, full architecture coverage: GQA, RoPE, RMSNorm, SwiGLU, tied
embeddings) for both full-sequence logits and greedy generation (which also exercises the KV
cache + position offset). If these pass, the same code on the real 0.5B weights is trustworthy.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")


def _tiny_hf():  # type: ignore[no-untyped-def]
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    return cfg, model


def test_native_qwen2_matches_hf_logits() -> None:
    from llcore.runtime.qwen2 import Qwen2LM, Qwen2Params

    cfg, hf = _tiny_hf()
    native = Qwen2LM(Qwen2Params.from_hf_config(cfg.to_dict())).eval()
    native.load_hf_state_dict(hf.state_dict())
    x = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        ref = hf(x).logits
        got = native(x)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=2e-4, rtol=1e-4)


def test_native_qwen2_greedy_generation_matches_hf() -> None:
    from llcore.runtime.qwen2 import Qwen2LM, Qwen2Params

    cfg, hf = _tiny_hf()
    native = Qwen2LM(Qwen2Params.from_hf_config(cfg.to_dict())).eval()
    native.load_hf_state_dict(hf.state_dict())
    prompt = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        ref = hf.generate(prompt, max_new_tokens=8, do_sample=False, use_cache=True)
        got = native.generate(prompt, max_new_tokens=8)
    assert torch.equal(got, ref)


def test_native_qwen2_kv_cache_matches_full_forward() -> None:
    """Incremental decode (KV cache) must equal a fresh full-sequence forward."""
    from llcore.runtime.qwen2 import Qwen2LM, Qwen2Params

    cfg, hf = _tiny_hf()
    native = Qwen2LM(Qwen2Params.from_hf_config(cfg.to_dict())).eval()
    native.load_hf_state_dict(hf.state_dict())
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        full = native(ids)
        # step token-by-token through the cache
        logits, cache = native.forward(ids[:, :1], return_cache=True)
        last = logits[:, -1:]
        for t in range(1, ids.size(1)):
            logits, cache = native.forward(ids[:, t : t + 1], past=cache, return_cache=True)
            last = logits[:, -1:]
    assert torch.allclose(last[:, -1], full[:, -1], atol=2e-4)
