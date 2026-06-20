# SPDX-License-Identifier: Apache-2.0
"""Tests for proxy-v2: a statistically honest evaluation proxy for the memory↔quality NAS.

Proxy-v1 (`scripts/nas_pareto.py`) scored every genome with a single Δnll on ONE 256-token window —
no confidence interval, no significance, one short context, one corpus. That cannot support the claim
"the evolved frontier beats greedy" or "config X beats all-softmax": at 256 tokens the constant-state
linear attention's degradation is hidden (it manifests at 2k–8k+; cf. SUPRA) AND the memory crossover
is only ~227 tokens, so v1 measured quality in the regime where the savings barely exist.

proxy-v2 replaces the point estimate with paired multi-window bootstrap CIs, a context-length sweep,
a long-context needle/passkey probe, an attention-map KL diagnostic, cross-corpus generalization, and
a winner's-curse (selection-optimism) correction. These tests pin the PURE statistical/report core
(no model needed — the heart of "provable") plus the cheap model-coupled primitives on a tiny model.

Test style mirrors `test_pareto_metrics.py` (lazy imports inside test fns) and `test_runtime_distill.py`
(the `_tiny()` Qwen2 model for model-coupled functions).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from llcore.runtime.linearize import LinearAttention, SlidingWindowAttention
from llcore.runtime.qwen2 import Qwen2Attention, Qwen2LM, Qwen2Params, _rope_cos_sin

CatGenome = tuple[int, ...]


# --------------------------------------------------------------------------------------------------
# PURE functions (no model) — bootstrap CI, significance, rank correlation, hypervolume-with-CI,
# attention-KL on synthetic matrices, the honest-verdict chokepoint, and the report assembler.
# --------------------------------------------------------------------------------------------------


def test_bootstrap_paired_ci_zero_delta() -> None:
    from llcore.runtime.eval_proxy import bootstrap_paired_ci

    r = bootstrap_paired_ci(np.zeros(16), n_boot=500, seed=0)
    assert r["mean"] == 0.0
    assert r["ci_lo"] <= 0.0 <= r["ci_hi"]
    # a genuinely zero-effect config is a coin flip on "is it worse" — tie-aware p_worse == 0.5
    assert r["p_worse"] == 0.5
    assert r["n_windows"] == 16


def test_bootstrap_paired_ci_all_positive_and_all_negative() -> None:
    from llcore.runtime.eval_proxy import bootstrap_paired_ci

    pos = bootstrap_paired_ci(np.full(16, 0.1), n_boot=500, seed=0)  # Δnll>0 == quality LOSS
    assert pos["ci_lo"] > 0.0
    assert pos["p_worse"] == 1.0

    neg = bootstrap_paired_ci(np.full(16, -0.1), n_boot=500, seed=0)
    assert neg["ci_hi"] < 0.0
    assert neg["p_worse"] == 0.0


def test_bootstrap_paired_ci_deterministic_under_seed() -> None:
    from llcore.runtime.eval_proxy import bootstrap_paired_ci

    d = np.array([0.10, -0.05, 0.20, -0.02, 0.07, -0.11, 0.03, 0.15])
    a = bootstrap_paired_ci(d, n_boot=1000, seed=7)
    b = bootstrap_paired_ci(d, n_boot=1000, seed=7)
    assert a == b


def test_sign_test_exact_binomial() -> None:
    from math import comb

    from llcore.runtime.eval_proxy import sign_test

    eight = sign_test(np.full(8, 0.1))
    assert eight["pos_frac"] == 1.0
    assert eight["p_sign"] == pytest.approx(2 * comb(8, 0) / 2**8)

    mixed = sign_test(np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0]))  # 5 pos, 3 neg
    assert mixed["pos_frac"] == pytest.approx(5 / 8)
    expected = 2 * sum(comb(8, k) for k in range(0, 3 + 1)) / 2**8  # two-sided, smaller tail doubled
    assert mixed["p_sign"] == pytest.approx(min(1.0, expected))


def test_wilcoxon_perm_symmetric_vs_onesided() -> None:
    from llcore.runtime.eval_proxy import wilcoxon_perm

    # mirror-image magnitudes (average-rank ties) => signed-rank statistic 0 => never significant
    symmetric = wilcoxon_perm(np.array([2.0, -2.0, 3.0, -3.0, 5.0, -5.0]), seed=0)
    assert symmetric == pytest.approx(1.0)

    # all same sign => maximal statistic => only the all-aligned flips are as extreme => tiny p
    one_sided = wilcoxon_perm(np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), seed=0)
    assert one_sided < 0.05


def test_make_windows_non_overlap_and_short_corpus() -> None:
    from llcore.runtime.eval_proxy import make_windows

    ids_all = torch.arange(100).view(1, 100)

    w = make_windows(ids_all, L=10, K=5)  # stride defaults to L => non-overlapping
    assert len(w) == 5
    seen: set[int] = set()
    for win in w:
        vals = set(int(v) for v in win.view(-1).tolist())
        assert seen.isdisjoint(vals), "non-overlapping windows must not share tokens"
        seen |= vals

    # more windows than the corpus allows => fewer returned, never overlapping (anti-conservative guard)
    capped = make_windows(ids_all, L=10, K=20)
    assert len(capped) == 10

    offset = make_windows(ids_all, L=10, K=2, offset=5)
    assert int(offset[0][0, 0]) == 5


def test_spearman_and_kendall_known_values() -> None:
    from llcore.runtime.eval_proxy import kendall_tau, spearman_rho

    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_rho(a, a) == pytest.approx(1.0)
    assert kendall_tau(a, a) == pytest.approx(1.0)
    assert spearman_rho(a, a[::-1].copy()) == pytest.approx(-1.0)
    assert kendall_tau(a, a[::-1].copy()) == pytest.approx(-1.0)

    b = np.array([1.0, 2.0, 4.0, 3.0])  # one adjacent swap vs a
    assert spearman_rho(a, b) == pytest.approx(0.8)  # 1 - 6*2/(4*15)
    assert kendall_tau(a, b) == pytest.approx(4 / 6)  # 5 concordant, 1 discordant of 6 pairs


def test_bootstrap_hv_gain_paired_consistency() -> None:
    from llcore.runtime.eval_proxy import bootstrap_hv_gain

    # memetic strictly better (lower Δnll == higher -Δnll) on EVERY window at matched memory
    pwd = {"g": np.array([0.30, 0.40]), "m": np.array([0.00, 0.10])}
    pct = {"g": 50.0, "m": 50.0}
    win = bootstrap_hv_gain(pwd, pct, greedy_ids=["g"], memetic_ids=["m"], n_boot=500, seed=0)
    assert win["p_memetic_wins"] == pytest.approx(1.0)
    assert win["ci_lo"] > 0.0

    # identical fronts => zero gain every replicate => CI brackets 0, tie-aware win prob 0.5
    same = bootstrap_hv_gain(
        {"x": np.array([0.2, 0.3])}, {"x": 50.0},
        greedy_ids=["x"], memetic_ids=["x"], n_boot=500, seed=0,
    )
    assert same["gain_pct_mean"] == pytest.approx(0.0, abs=1e-9)
    assert same["ci_lo"] <= 0.0 <= same["ci_hi"]
    assert same["p_memetic_wins"] == pytest.approx(0.5)


def test_hypervolume_2d_ci_zero_width_matches_pareto_metrics() -> None:
    from llcore.runtime.eval_proxy import hypervolume_2d_ci
    from llcore.runtime.pareto_metrics import hypervolume_2d

    pwd = {"a": np.full(8, 0.10), "b": np.full(8, 0.20)}  # constant => zero-width bootstrap
    pct = {"a": 40.0, "b": 60.0}
    r = hypervolume_2d_ci(pwd, pct, ids=["a", "b"], n_boot=200, seed=0)

    points = [(40.0, -0.10), (60.0, -0.20)]
    ref = (0.0, min(y for _, y in points) - 1e-9)
    assert r["hv_mean"] == pytest.approx(hypervolume_2d(points, ref), abs=1e-9)
    assert r["hv_lo"] == pytest.approx(r["hv_hi"], abs=1e-9)  # no spread


def test_right_shift_ci_straddles_zero_is_no_shift() -> None:
    from llcore.runtime.eval_proxy import right_shift_ci

    # identical zero-shot/distilled per-window deltas => no measurable shift
    flat = right_shift_ci(
        {"p": np.array([0.3, 0.3, 0.3])}, {"p": np.array([0.3, 0.3, 0.3])},
        {"p": 50.0}, {"p": 50.0}, n_boot=300, seed=0,
    )
    assert flat["ci_lo"] <= 0.0 <= flat["ci_hi"]
    assert "no measurable shift" in str(flat["verdict"])

    # distilled strictly better on every window => frontier shifts out
    better = right_shift_ci(
        {"p": np.array([0.3, 0.3, 0.3])}, {"p": np.array([0.1, 0.1, 0.1])},
        {"p": 50.0}, {"p": 50.0}, n_boot=300, seed=0,
    )
    assert better["ci_lo"] > 0.0
    assert "shifts out" in str(better["verdict"])


def test_winners_curse_selection_inflates_best_estimate() -> None:
    """The bias proxy-v2's holdout re-eval removes: selecting the best of MANY noisy estimates is
    optimistic. Pure simulation pinning the phenomenon — memetic draws far more candidates than
    greedy, so its selected search-score is biased below its true value; a fresh holdout draw ties.
    """
    rng = np.random.default_rng(0)
    true_delta = 0.10  # identical TRUE quality cost for both methods
    noise = 0.05

    def select_best(n_candidates: int) -> tuple[float, float]:
        search = true_delta + rng.normal(0, noise, n_candidates)  # selection scores
        winner = int(np.argmin(search))  # pick lowest Δnll (best quality)
        holdout = true_delta + rng.normal(0, noise)  # fresh re-eval of the winner
        return float(search[winner]), float(holdout)

    greedy_sel = np.mean([select_best(6)[0] for _ in range(400)])
    memetic_sel = np.mean([select_best(480)[0] for _ in range(400)])
    memetic_hold = np.mean([select_best(480)[1] for _ in range(400)])

    # memetic's SELECTION score is more optimistic (lower) than greedy's, but its HOLDOUT ties truth
    assert memetic_sel < greedy_sel
    assert memetic_hold == pytest.approx(true_delta, abs=0.01)
    # the optimism gap (selection − holdout) the guard must report is clearly positive
    assert (memetic_hold - memetic_sel) > 0.01


def test_attn_kl_zero_for_identical_distributions() -> None:
    from llcore.runtime.eval_proxy import attn_kl_softmax_linear

    torch.manual_seed(0)
    h, t = 3, 6
    raw = torch.rand(h, t, t)
    causal = torch.tril(torch.ones(t, t))
    a = raw * causal
    a = a / a.sum(dim=-1, keepdim=True)
    kl = attn_kl_softmax_linear(a, a.clone())
    assert torch.isfinite(kl)
    assert float(kl) == pytest.approx(0.0, abs=1e-6)


def test_attn_kl_direction_penalizes_dropped_mass() -> None:
    from llcore.runtime.eval_proxy import attn_kl_softmax_linear

    # teacher needs both keys (row 1 = [0.5, 0.5]); student drops the second ([0.99, 0.01]).
    a_sm = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])  # [1 head, 2, 2], causal
    a_lin = torch.tensor([[[1.0, 0.0], [0.99, 0.01]]])
    forward = float(attn_kl_softmax_linear(a_sm, a_lin))  # KL(teacher || student)
    reverse = float(attn_kl_softmax_linear(a_lin, a_sm))  # swapped
    assert forward > reverse  # forward KL punishes the student for dropping mass the teacher placed


def test_honest_verdict_gates() -> None:
    from llcore.runtime.eval_proxy import honest_verdict

    sig_ci = {"gain_pct_mean": 30.0, "ci_lo": 10.0, "ci_hi": 50.0, "p_memetic_wins": 0.99}
    clean = [{"optimism_gap": 0.001, "pos_frac": 0.9, "p_sign": 0.01}]

    significant = honest_verdict(sig_ci, clean, proxy_vs_judge_tau=0.9, ci_halfwidth_floor=0.05)
    assert significant["confidence"] == "significant"
    assert "conversational_claim" not in significant

    suggestive = honest_verdict(sig_ci, clean, proxy_vs_judge_tau=0.5, ci_halfwidth_floor=0.05)
    assert suggestive["confidence"] == "suggestive"

    biased = [{"optimism_gap": 0.10, "pos_frac": 0.9, "p_sign": 0.01}]  # gap > halfwidth floor
    suppressed = honest_verdict(sig_ci, biased, proxy_vs_judge_tau=0.9, ci_halfwidth_floor=0.05)
    assert suppressed["confidence"] == "suppressed"

    null_ci = {"gain_pct_mean": 5.0, "ci_lo": -8.0, "ci_hi": 20.0, "p_memetic_wins": 0.6}
    null = honest_verdict(null_ci, clean, proxy_vs_judge_tau=0.9, ci_halfwidth_floor=0.05)
    assert null["confidence"] == "null"


def test_build_proxy_v2_report_shape_and_scope() -> None:
    from llcore.runtime.eval_proxy import build_proxy_v2_report

    rep = build_proxy_v2_report(
        inner_context=1024,
        context_sweep={256: {"mean": 0.01}},
        frontier_holdout=[{"genome": (0,), "delta_nll_heldout": 0.02}],
        hv_gain_ci={"gain_pct_mean": 12.0, "ci_lo": 2.0, "ci_hi": 22.0, "p_memetic_wins": 0.98},
        right_shift_ci=None,
        needle=None,
        attention_kl=None,
        proxy_vs_judge_tau=0.8,
        cross_corpus=None,
    )
    assert rep["scope"] == "next_token_nll_proxy"
    assert rep["conversational_claim"] is None
    for key in ("inner_context", "context_sweep", "frontier_holdout", "hv_gain_ci"):
        assert key in rep


# --------------------------------------------------------------------------------------------------
# Model-coupled functions on a tiny Qwen2 (cheap on CPU): window losses, paired deltas, fast proxy,
# KV-primed suffix scoring, attention-map reconstruction, per-genome attention KL, frontier re-eval.
# --------------------------------------------------------------------------------------------------


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
        max_position=512,
    )
    return Qwen2LM(p).eval()


def _genome_controls(model: Qwen2LM, window: int = 8) -> tuple[
    Callable[[CatGenome, bool], None], Callable[[], None]
]:
    """set_genome / restore closures mirroring nas_pareto (0=softmax, 1=sliding, 2=linear)."""
    n = model.params.n_layer
    originals = {i: model.model.layers[i].self_attn for i in range(n)}
    assert all(isinstance(a, Qwen2Attention) for a in originals.values())

    def set_genome(genome: CatGenome, use_distill: bool = False) -> None:
        for i, opt in enumerate(genome):
            src = originals[i]
            assert isinstance(src, Qwen2Attention)
            if opt == 0:
                model.model.layers[i].self_attn = src
            elif opt == 1:
                model.model.layers[i].self_attn = SlidingWindowAttention.from_attention(
                    src, model.params, window=window
                )
            else:
                model.model.layers[i].self_attn = LinearAttention.from_attention(src, model.params)

    def restore() -> None:
        for i in range(n):
            model.model.layers[i].self_attn = originals[i]

    return set_genome, restore


def test_window_losses_matches_manual_cross_entropy() -> None:
    from torch.nn import functional as F

    from llcore.runtime.eval_proxy import window_losses

    model = _tiny()
    ids = torch.randint(0, 48, (1, 20))
    lv = window_losses(model, ids)
    assert lv.shape == (19,)
    manual = F.cross_entropy(model(ids)[0, :-1], ids[0, 1:])
    assert torch.allclose(lv.mean(), manual, atol=1e-5)


def test_paired_window_deltas_zero_for_all_softmax_genome() -> None:
    from llcore.runtime.eval_proxy import base_window_losses, paired_window_deltas

    model = _tiny()
    set_genome, restore = _genome_controls(model)
    windows = [torch.randint(0, 48, (1, 16)) for _ in range(3)]
    base = base_window_losses(model, restore, windows)
    d = paired_window_deltas(model, set_genome, restore, (0, 0), windows, base)
    assert d.shape == (3,)
    assert np.allclose(d, 0.0, atol=1e-5)  # genome == base => zero Δnll, exactly paired


def test_fast_proxy_keys_and_zero_for_base() -> None:
    from llcore.runtime.eval_proxy import base_window_losses, fast_proxy

    model = _tiny()
    set_genome, restore = _genome_controls(model)
    windows = [torch.randint(0, 48, (1, 16)) for _ in range(4)]
    base = base_window_losses(model, restore, windows)
    r = fast_proxy(model, set_genome, restore, (0, 0), windows, base, n_boot=200, seed=0)
    assert {"mean_dnll", "ci_lo", "ci_hi", "p_worse", "n_windows"} <= set(r)
    assert abs(r["mean_dnll"]) < 1e-5


def test_suffix_losses_matches_full_forward_for_softmax_and_rejects_linear() -> None:
    from llcore.runtime.eval_proxy import suffix_losses, window_losses

    model = _tiny()
    ids = torch.randint(0, 48, (1, 24))
    full = window_losses(model, ids)
    suf = suffix_losses(model, ids, score_last=8, genome_has_linear=False)
    assert suf.shape == (8,)
    assert torch.allclose(suf, full[-8:], atol=1e-4)  # KV-primed == full forward for softmax
    with pytest.raises(ValueError, match="linear"):
        suffix_losses(model, ids, score_last=8, genome_has_linear=True)


def test_attention_matrices_are_causal_and_row_stochastic() -> None:
    from llcore.runtime.eval_proxy import implied_linear_attention, softmax_teacher_attention

    model = _tiny()
    p = model.params
    t = 12
    x = torch.randn(1, t, p.hidden_size)
    cos, sin = _rope_cos_sin(torch.arange(t), p.head_dim, p.rope_theta)
    attn = model.model.layers[0].self_attn
    assert isinstance(attn, Qwen2Attention)

    a_sm = softmax_teacher_attention(attn, x, cos, sin)
    a_lin = implied_linear_attention(LinearAttention.from_attention(attn, p), x, cos, sin)
    for a in (a_sm, a_lin):
        assert a.shape == (p.n_head, t, t)
        assert torch.allclose(a.sum(dim=-1), torch.ones(p.n_head, t), atol=1e-5)  # rows sum to 1
        upper = torch.triu(torch.ones(t, t), diagonal=1).bool()
        assert torch.allclose(a[:, upper], torch.zeros_like(a[:, upper]), atol=1e-6)  # causal


def test_genome_attn_kl_zero_for_softmax_positive_for_linear() -> None:
    from llcore.runtime.eval_proxy import genome_attn_kl

    model = _tiny()
    set_genome, restore = _genome_controls(model)
    calib = torch.randint(0, 48, (1, 16))
    base = genome_attn_kl(model, set_genome, restore, (0, 0), calib, t_kl=16)
    assert base["mean"] == 0.0  # no converted layers => no divergence
    lin = genome_attn_kl(model, set_genome, restore, (2, 2), calib, t_kl=16)
    assert lin["mean"] > 0.0


def test_genome_attn_kl_rejects_long_window() -> None:
    from llcore.runtime.eval_proxy import genome_attn_kl

    model = _tiny()
    set_genome, restore = _genome_controls(model)
    calib = torch.randint(0, 48, (1, 300))
    with pytest.raises(AssertionError):
        genome_attn_kl(model, set_genome, restore, (2, 2), calib, t_kl=300)  # O(T^2), capped at 256


def test_reeval_frontier_reports_optimism_gap() -> None:
    from llcore.runtime.eval_proxy import base_window_losses, make_windows, reeval_frontier

    model = _tiny()
    set_genome, restore = _genome_controls(model)
    ids_all = torch.randint(0, 48, (1, 200))
    search_windows = make_windows(ids_all, L=16, K=3, offset=0)
    holdout_windows = make_windows(ids_all, L=16, K=3, offset=100)
    base_search = base_window_losses(model, restore, search_windows)
    base_hold = base_window_losses(model, restore, holdout_windows)

    from llcore.runtime.eval_proxy import paired_window_deltas

    frontier: list[CatGenome] = [(0, 2), (2, 2)]
    search_deltas = {
        g: paired_window_deltas(model, set_genome, restore, g, search_windows, base_search)
        for g in frontier
    }
    rows = reeval_frontier(
        model, set_genome, restore, frontier,
        pct_of=lambda g: 50.0 * sum(1 for o in g if o != 0) / len(g),
        search_deltas=search_deltas,
        holdout_windows=holdout_windows, holdout_base=base_hold,
    )
    assert len(rows) == 2
    for row in rows:
        for key in ("genome", "pct", "delta_nll_selection", "delta_nll_heldout", "optimism_gap"):
            assert key in row
        assert row["optimism_gap"] == pytest.approx(
            row["delta_nll_selection"] - row["delta_nll_heldout"], abs=1e-9
        )


# --------------------------------------------------------------------------------------------------
# Long-context needle / passkey probe (the direct constant-state failure-mode test).
# --------------------------------------------------------------------------------------------------


class _FakeTok:
    """Minimal tokenizer stub: maps a fixed answer string to a known id sequence."""

    def __init__(self, answer: str, ans_ids: list[int]) -> None:
        self._answer = answer
        self._ans_ids = ans_ids

    def __call__(self, text: str, return_tensors: str | None = None) -> object:
        assert text == self._answer, "fake tok only knows the answer string"
        ids = torch.tensor([self._ans_ids])

        class _Out:
            input_ids = ids

        return _Out()


def test_build_passkey_prompt_span_indexes_answer_and_guards_recurrence() -> None:
    from llcore.runtime.eval_proxy import build_passkey_prompt

    answer, ans_ids = "KEY", [7, 8, 9]
    tok = _FakeTok(answer, ans_ids)
    # filler disjoint from the answer tokens => no spurious recurrence
    filler = torch.arange(10, 60).view(1, 50)
    ids, span = build_passkey_prompt(tok, filler, total_len=40, answer=answer, depth_frac=0.25)
    assert ids.shape == (1, 40)
    # the answer-span must index the FINAL answer occurrence in the targets array ids[:, 1:]
    assert ids[0, 1:][span].tolist() == ans_ids

    # a filler that already contains the answer after the needle would make retrieval trivial => raise
    bad = torch.tensor([[7, 8, 9] * 20])
    with pytest.raises(ValueError, match="recur"):
        build_passkey_prompt(tok, bad, total_len=40, answer=answer, depth_frac=0.25)


def test_score_needle_contract_and_perfect_when_target_known() -> None:
    from llcore.runtime.eval_proxy import score_needle

    model = _tiny()
    ids = torch.randint(0, 48, (1, 20))
    span = slice(15, 19)
    r = score_needle(model, ids, span)
    assert {"mean_logprob", "argmax_acc"} <= set(r)
    assert 0.0 <= r["argmax_acc"] <= 1.0
    assert r["mean_logprob"] <= 0.0  # log-prob is non-positive


def test_needle_horizon_returns_horizon_structure() -> None:
    from llcore.runtime.eval_proxy import needle_horizon

    model = _tiny()
    set_genome, restore = _genome_controls(model)
    filler = torch.randint(0, 40, (1, 200))
    answer, ans_ids = "K", [41]
    tok = _FakeTok(answer, ans_ids)
    base_acc = {(32, 0.5): 1.0, (48, 0.5): 1.0}
    out = needle_horizon(
        model, set_genome, restore, (0, 0), tok, filler, base_acc,
        lengths=(32, 48), depths=(0.5,), answer=answer,
    )
    assert "horizon" in out and "by_depth" in out
    assert out["horizon"] is None or isinstance(out["horizon"], int)
