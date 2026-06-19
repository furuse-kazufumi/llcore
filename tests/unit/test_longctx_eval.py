# SPDX-License-Identifier: Apache-2.0
"""Tests for confound-controlled long-context evaluation of constant-state LMs.

The headline metric is a *context-length scaling curve*: for a fixed set of target
positions, score each target conditioning on exactly ``c`` preceding tokens (fresh
zero-state window), sweeping ``c``. Because the target positions are held fixed across
all ``c``, text difficulty is perfectly controlled and the only thing varying is the
amount of available context — so a drop-then-plateau in NLL is a clean measurement of
the model's *effective* context length, with no cold-start/warmup asymmetry (every
measurement starts cold by construction). Crucially these helpers drive the model via
``step`` so they are NOT capped by ``block_size`` (which only guards the batched
``forward``), letting us probe ``c`` far longer than the training window.
"""
from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM


def _recurrent(block: int = 16) -> RecurrentLM:
    torch.manual_seed(0)
    m = RecurrentLM(RecurrentConfig(vocab_size=32, block_size=block, n_layer=2, n_embd=24, state_size=24))
    m.eval()
    return m


def _rwkv(block: int = 16) -> RWKVLM:
    torch.manual_seed(0)
    m = RWKVLM(RWKVConfig(vocab_size=32, block_size=block, n_layer=2, n_embd=24))
    m.eval()
    return m


# --- the step-loop logits must exactly equal the batched forward within block_size ---


@pytest.mark.parametrize("factory", [_recurrent, _rwkv])
def test_run_steps_matches_forward(factory) -> None:  # type: ignore[no-untyped-def]
    from llcore.lm.longctx_eval import run_steps

    model = factory()
    torch.manual_seed(1)
    x = torch.randint(0, 32, (3, 12))  # within block_size
    forward_logits, _ = model(x)
    step_logits = run_steps(model, x)
    assert step_logits.shape == forward_logits.shape
    assert torch.allclose(step_logits, forward_logits, atol=1e-5)


# --- conditioning on exactly c tokens equals the last-token CE of a forward over the window ---


def test_nll_at_positions_matches_last_token_forward() -> None:
    from llcore.lm.longctx_eval import nll_at_positions_with_context

    model = _recurrent()
    torch.manual_seed(2)
    ids = torch.randint(0, 32, (40,))
    c = 10
    positions = [12, 20, 33]
    nll, count = nll_at_positions_with_context(model, ids, positions, c)
    assert count == len(positions)

    # reference: for each p, predict ids[p] from a fresh forward over ids[p-c:p]
    ref_total = 0.0
    with torch.no_grad():
        for p in positions:
            window = ids[p - c : p].unsqueeze(0)
            logits, _ = model(window)
            last = logits[0, -1]
            ref_total += float(F.cross_entropy(last.unsqueeze(0), ids[p : p + 1], reduction="sum"))
    assert nll == pytest.approx(ref_total / len(positions), abs=1e-5)


# --- the context-length curve: same positions across c, handles c > block_size, finite ---


def test_context_length_curve_uses_same_positions_and_exceeds_block_size() -> None:
    from llcore.lm.longctx_eval import context_length_curve

    model = _recurrent(block=16)
    ids = torch.randint(0, 32, (400,))
    context_lens = [4, 16, 64]  # 64 >> block_size=16: must not raise
    result = context_length_curve(model, ids, context_lens, n_positions=20, seed=7)
    assert set(result["nll_by_context"].keys()) == {4, 16, 64}
    assert result["n_positions"] == 20
    for c, nll in result["nll_by_context"].items():
        assert torch.isfinite(torch.tensor(nll)) and nll > 0.0
    # determinism: same seed -> same positions -> same curve
    again = context_length_curve(model, ids, context_lens, n_positions=20, seed=7)
    assert again["nll_by_context"] == result["nll_by_context"]


def test_context_length_curve_rejects_too_short() -> None:
    from llcore.lm.longctx_eval import context_length_curve

    model = _recurrent()
    ids = torch.randint(0, 32, (30,))
    with pytest.raises(ValueError):
        context_length_curve(model, ids, [64], n_positions=20, seed=0)  # not enough room


# --- block-reset streaming NLL: resetting beyond the sequence length == plain streaming_nll ---


def test_block_reset_equals_streaming_when_no_reset() -> None:
    from llcore.lm.longctx_eval import block_reset_nll

    model = _recurrent()
    ids = torch.randint(0, 32, (50,))
    nll_reset, count = block_reset_nll(model, ids, reset_every=1000)  # never resets
    nll_stream, count2 = model.streaming_nll(ids)
    assert count == count2
    assert nll_reset == pytest.approx(nll_stream, abs=1e-5)


def test_block_reset_shorter_window_differs() -> None:
    from llcore.lm.longctx_eval import block_reset_nll

    model = _recurrent()
    torch.manual_seed(5)
    ids = torch.randint(0, 32, (200,))
    nll_short, _ = block_reset_nll(model, ids, reset_every=8)
    nll_full, _ = model.streaming_nll(ids)
    # both finite & positive; resetting changes the number (sanity, not a direction claim)
    assert nll_short > 0.0 and nll_full > 0.0
    assert nll_short != pytest.approx(nll_full, abs=1e-6)


# --- correctness gate: chunk_size must NOT change the streaming loss (state carries) ---


@pytest.mark.parametrize("factory", [_recurrent, _rwkv])
def test_streaming_nll_chunk_size_invariant(factory) -> None:  # type: ignore[no-untyped-def]
    model = factory()
    torch.manual_seed(9)
    ids = torch.randint(0, 32, (300,))
    a, _ = model.streaming_nll(ids, chunk_size=16)
    b, _ = model.streaming_nll(ids, chunk_size=64)
    c, _ = model.streaming_nll(ids, chunk_size=257)  # not a divisor of len
    assert a == pytest.approx(b, abs=1e-5)
    assert a == pytest.approx(c, abs=1e-5)


# --- banded streaming metrics: reconcile with streaming_nll, partition, top-k, unigram ---


def _band_edges() -> list[int]:
    return [0, 16, 32, 64, 128]


def test_streaming_metrics_by_band_reconciles_and_partitions() -> None:
    from llcore.lm.longctx_eval import streaming_metrics_by_band

    model = _recurrent()
    torch.manual_seed(4)
    ids = torch.randint(0, 32, (300,))
    out = streaming_metrics_by_band(model, ids, _band_edges())
    # token counts across bands partition the n-1 predicted positions exactly
    assert sum(b["n_tok"] for b in out["bands"]) == ids.size(0) - 1
    # token-weighted mean over bands == the scalar streaming_nll (same carried pass)
    stream_nll, _ = model.streaming_nll(ids)
    weighted = sum(b["mean_nll"] * b["n_tok"] for b in out["bands"]) / (ids.size(0) - 1)
    assert weighted == pytest.approx(stream_nll, abs=1e-4)
    assert out["full_mean_nll"] == pytest.approx(stream_nll, abs=1e-4)
    # every band reports the standard char-LM units
    for b in out["bands"]:
        assert b["ppl"] == pytest.approx(float(torch.tensor(b["mean_nll"]).exp()), rel=1e-5)
        assert b["bpc"] == pytest.approx(b["mean_nll"] / 0.6931471805599453, rel=1e-5)
        assert 0.0 <= b["top1"] <= 1.0 and b["top1"] <= b["top5"]


def test_streaming_metrics_by_band_tail_excludes_warmup() -> None:
    from llcore.lm.longctx_eval import streaming_metrics_by_band

    model = _recurrent()
    torch.manual_seed(4)
    ids = torch.randint(0, 32, (300,))
    out = streaming_metrics_by_band(model, ids, _band_edges(), tail_start=64)
    # tail mean is the token-weighted mean of bands whose target positions are >= 64
    tail_bands = [b for b in out["bands"] if b["lo"] >= 64]
    tok = sum(b["n_tok"] for b in tail_bands)
    expect = sum(b["mean_nll"] * b["n_tok"] for b in tail_bands) / tok
    assert out["tail_mean_nll"] == pytest.approx(expect, abs=1e-6)
    assert out["tail_start"] == 64


def test_streaming_metrics_by_band_unigram_floor() -> None:
    from llcore.lm.eval import unigram_nll  # noqa: F401  (ensures helper exists)
    from llcore.lm.longctx_eval import streaming_metrics_by_band

    model = _recurrent()
    torch.manual_seed(4)
    ids = torch.randint(0, 32, (300,))
    # add-1 unigram log-probs over the same vocab, computed on this sequence
    counts = torch.bincount(ids, minlength=32).double()
    probs = (counts + 1.0) / (ids.numel() + 32.0)
    logp = torch.log(probs)
    out = streaming_metrics_by_band(model, ids, _band_edges(), unigram_logp=logp)
    for b in out["bands"]:
        assert "unigram_nll" in b and "beats_unigram" in b
        assert isinstance(b["beats_unigram"], bool)


# --- GPT sliding-window baseline: scores every target exactly once ---


def test_gpt_sliding_window_scores_all_targets() -> None:
    from llcore.lm.model import CharGPT, GPTConfig
    from llcore.lm.longctx_eval import gpt_sliding_window_nll

    torch.manual_seed(0)
    gpt = CharGPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=24))
    gpt.eval()
    ids = torch.randint(0, 32, (120,))
    nll_nonoverlap, n1 = gpt_sliding_window_nll(gpt, ids, stride=16)
    nll_gold, n2 = gpt_sliding_window_nll(gpt, ids, stride=1)
    # every target 1..n-1 scored exactly once under either stride (only the context differs)
    assert n1 == n2 == ids.size(0) - 1
    assert nll_nonoverlap > 0.0 and nll_gold > 0.0
    assert torch.isfinite(torch.tensor([nll_nonoverlap, nll_gold])).all()


def test_gpt_sliding_window_rejects_bad_stride() -> None:
    from llcore.lm.model import CharGPT, GPTConfig
    from llcore.lm.longctx_eval import gpt_sliding_window_nll

    gpt = CharGPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=24))
    ids = torch.randint(0, 32, (50,))
    with pytest.raises(ValueError):
        gpt_sliding_window_nll(gpt, ids, stride=0)
    with pytest.raises(ValueError):
        gpt_sliding_window_nll(gpt, ids, stride=17)  # > block_size
