# SPDX-License-Identifier: Apache-2.0
"""proxy-v2: a statistically honest evaluation proxy for the memory↔quality NAS.

`scripts/nas_pareto.py`'s v1 proxy scored each genome with a single Δnll on ONE 256-token window —
no error bar, one short context, one corpus. That cannot support "the evolved frontier beats greedy"
or "config X beats all-softmax": constant-state linear attention's quality cost manifests at long
context (2k–8k+; cf. SUPRA), so a 256-token window *systematically under-detects* it, and the memory
crossover is only ~227 tokens, so v1 measured quality where the savings barely exist.

This module supplies the pieces of a two-tier proxy:

* **fast inner-loop** (every NAS genome, cheap): paired multi-window Δnll at a single moderate context
  (default L=1024, on the right side of both the 227-tok memory crossover and the degradation onset),
  reduced to the scalar mean the GA selects on, with a bootstrap CI riding along for disclosure
  (:func:`window_losses`, :func:`make_windows`, :func:`base_window_losses`,
  :func:`paired_window_deltas`, :func:`bootstrap_paired_ci`, :func:`fast_proxy`).

* **rigorous frontier-only** (the handful of non-dominated genomes, once): re-evaluate on a FRESH
  disjoint holdout pool to remove the winner's-curse selection optimism, sweep the context length to
  expose regime dependence, and run a long-context needle/passkey retrieval probe
  (:func:`reeval_frontier`, :func:`context_sweep`, :func:`suffix_losses`, :func:`build_passkey_prompt`,
  :func:`score_needle`, :func:`needle_horizon`, :func:`sign_test`, :func:`wilcoxon_perm`).

* **diagnostics** (per study, never feeding selection): bootstrap CIs on the hypervolume gain and the
  distillation right-shift, an attention-map KL fidelity check, and a proxy-vs-judge rank-correlation
  gate (:func:`bootstrap_hv_gain`, :func:`hypervolume_2d_ci`, :func:`right_shift_ci`,
  :func:`implied_linear_attention`, :func:`softmax_teacher_attention`, :func:`attn_kl_softmax_linear`,
  :func:`genome_attn_kl`, :func:`spearman_rho`, :func:`kendall_tau`).

* **honest disclosure**: a single chokepoint (:func:`honest_verdict`) turns every guard into one
  disclosed verdict, and :func:`build_proxy_v2_report` assembles the additive report block that pins
  ``scope='next_token_nll_proxy'`` and refuses any conversational claim.

All scores are paired Δnll *within* a corpus, never raw nll and never averaged across corpora. The
attention-KL diagnostic is hard-capped at 256 tokens (it is O(T²) per head and shares v1's short-context
blind spot) and is NEVER wired into the NAS fitness.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from math import comb
from typing import Protocol

import numpy as np
import torch
from torch.nn import functional as F

from llcore.runtime.linearize import LinearAttention, SlidingWindowAttention, _phi
from llcore.runtime.pareto_metrics import hypervolume_2d
from llcore.runtime.qwen2 import (
    Qwen2LM,
    Qwen2Params,
    _apply_rope,
    _repeat_kv,
    _rope_cos_sin,
)

CatGenome = tuple[int, ...]
SetGenome = Callable[[CatGenome, bool], None]
Restore = Callable[[], None]


# ==================================================================================================
# Pure statistics: paired bootstrap CI, distribution-free significance, rank correlation.
# ==================================================================================================


def bootstrap_paired_ci(
    deltas: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap CI on the mean of per-window Δnll (Δnll>0 == quality LOSS).

    Resamples the K **already-paired** per-window deltas with replacement (not tokens), so it costs
    zero model forwards. ``p_worse`` is the bootstrap P(mean>0) with exact ties counted as half, so a
    genuinely zero-effect config reads as a 0.5 coin flip rather than a spurious 0 or 1.
    """
    d = np.asarray(deltas, dtype=np.float64)
    k = int(d.shape[0])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, k, size=(n_boot, k))].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_worse = float((boot > 0).mean() + 0.5 * (boot == 0).mean())
    return {
        "mean": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_worse": p_worse,
        "n_windows": float(k),
    }


def sign_test(deltas: np.ndarray) -> dict[str, float]:
    """Two-sided exact binomial sign test on per-window Δnll (zeros dropped from the test count).

    ``pos_frac`` is the fraction of all windows that are worse (Δnll>0); ``p_sign`` doubles the
    smaller binomial tail under p=0.5 (capped at 1.0). Answers 'does it hurt on EVERY window or only
    on average?' — a guard against one pathological window dominating the bootstrap mean.
    """
    d = np.asarray(deltas, dtype=np.float64)
    n_all = int(d.shape[0])
    pos = int((d > 0).sum())
    neg = int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return {"pos_frac": 0.0, "p_sign": 1.0}
    tail = sum(comb(n, k) for k in range(0, min(pos, neg) + 1)) / 2.0**n
    return {"pos_frac": pos / n_all if n_all else 0.0, "p_sign": min(1.0, 2.0 * tail)}


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties resolved to their average (the rankdata 'average' method)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(x.shape[0], dtype=np.float64)
    n = x.shape[0]
    i = 0
    while i < n:
        j = i
        while j < n and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def wilcoxon_perm(deltas: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    """Wilcoxon signed-rank permutation p-value under the sign-flip null (no scipy).

    The exact null is enumerated for ``n<=12`` non-zero deltas (so small-sample tests are exact and
    deterministic); otherwise ``n_perm`` random sign-flips are sampled. Robust in the few-windows
    regime and to the paired-symmetric null.
    """
    d = np.asarray(deltas, dtype=np.float64)
    nz = d[d != 0.0]
    n = int(nz.shape[0])
    if n == 0:
        return 1.0
    ranks = _rankdata_average(np.abs(nz))
    w = float(np.sum(np.sign(nz) * ranks))
    if n <= 12:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.float64) * 2.0 - 1.0
        null = signs @ ranks
    else:
        rng = np.random.default_rng(seed)
        null = rng.choice([-1.0, 1.0], size=(n_perm, n)) @ ranks
    return float((np.abs(null) >= abs(w) - 1e-12).mean())


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    return float((xc * yc).sum() / denom) if denom > 0 else 0.0


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation: Pearson on the average ranks of ``a`` and ``b``."""
    return _pearson(
        _rankdata_average(np.asarray(a, dtype=np.float64)),
        _rankdata_average(np.asarray(b, dtype=np.float64)),
    )


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b rank correlation (ties-corrected), O(n²) concordant/discordant count."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    n = int(av.shape[0])
    conc = disc = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = av[i] - av[j]
            db = bv[i] - bv[j]
            s = da * db
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
            else:
                if da == 0:
                    tie_a += 1
                if db == 0:
                    tie_b += 1
    n0 = n * (n - 1) / 2.0
    denom = float(np.sqrt((n0 - tie_a) * (n0 - tie_b)))
    return float((conc - disc) / denom) if denom > 0 else 0.0


# ==================================================================================================
# Hypervolume with uncertainty (diagnostics) — paired window resampling so a real win is not hidden.
# ==================================================================================================


def _ref_from_means(means: Sequence[float]) -> tuple[float, float]:
    """Shared lower-left reference for hypervolume: x=0, y just below the deepest -Δnll."""
    return (0.0, min(-m for m in means) - 1e-9)


def hypervolume_2d_ci(
    per_point_delta: dict[str, np.ndarray],
    pct: dict[str, float],
    ids: list[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI on one front's 2-D hypervolume by resampling the per-point Δnll windows.

    The memory axis (``pct``) is deterministic; only the ``-Δnll`` axis carries window noise. With
    zero-width inputs (constant per-point deltas) this reproduces :func:`pareto_metrics.hypervolume_2d`
    exactly (backward compatible).
    """
    arrs = {i: np.asarray(per_point_delta[i], dtype=np.float64) for i in ids}
    k = arrs[ids[0]].shape[0]
    ref = _ref_from_means([float(arrs[i].mean()) for i in ids])
    rng = np.random.default_rng(seed)
    hvs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        bidx = rng.integers(0, k, size=k)
        pts = [(pct[i], -float(arrs[i][bidx].mean())) for i in ids]
        hvs[b] = hypervolume_2d(pts, ref)
    lo, hi = np.quantile(hvs, [0.025, 0.975])
    return {"hv_mean": float(hvs.mean()), "hv_lo": float(lo), "hv_hi": float(hi)}


def bootstrap_hv_gain(
    per_window_delta: dict[str, np.ndarray],
    pct: dict[str, float],
    greedy_ids: list[str],
    memetic_ids: list[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI on the memetic-vs-greedy hypervolume gain (%), the error bar on the '+X%' claim.

    PAIRED resample: one window-index vector per replicate is applied to ALL genomes (independent
    per-genome resampling would inflate the CI and hide a real win). ``p_memetic_wins`` counts exact
    ties as half. The verdict 'memetic beats greedy' should fire only if ``ci_lo>0``.
    """
    all_ids = list(dict.fromkeys([*greedy_ids, *memetic_ids]))
    arrs = {i: np.asarray(per_window_delta[i], dtype=np.float64) for i in all_ids}
    k = arrs[all_ids[0]].shape[0]
    ref = _ref_from_means([float(arrs[i].mean()) for i in all_ids])
    rng = np.random.default_rng(seed)
    gains = np.empty(n_boot, dtype=np.float64)
    wins = 0.0
    for b in range(n_boot):
        bidx = rng.integers(0, k, size=k)
        ymean = {i: -float(arrs[i][bidx].mean()) for i in all_ids}
        hv_g = hypervolume_2d([(pct[i], ymean[i]) for i in greedy_ids], ref)
        hv_m = hypervolume_2d([(pct[i], ymean[i]) for i in memetic_ids], ref)
        gains[b] = 100.0 * (hv_m - hv_g) / max(hv_g, 1e-9)
        wins += 1.0 if hv_m > hv_g else (0.5 if hv_m == hv_g else 0.0)
    lo, hi = np.quantile(gains, [0.025, 0.975])
    return {
        "gain_pct_mean": float(gains.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_memetic_wins": float(wins / n_boot),
    }


def right_shift_ci(
    zs_delta: dict[str, np.ndarray],
    ds_delta: dict[str, np.ndarray],
    zs_pct: dict[str, float],
    ds_pct: dict[str, float],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap CI on the distillation right-shift (zero-shot → distilled hypervolume gain, %).

    Paired window resample across both fronts against a shared reference. Verdict reports 'shifts out'
    only if ``ci_lo>0``, 'regresses' if ``ci_hi<0``, else 'no measurable shift' with the CI quoted.
    """
    zs_ids = list(zs_delta)
    ds_ids = list(ds_delta)
    zsa = {i: np.asarray(zs_delta[i], dtype=np.float64) for i in zs_ids}
    dsa = {i: np.asarray(ds_delta[i], dtype=np.float64) for i in ds_ids}
    k = zsa[zs_ids[0]].shape[0]
    ref = _ref_from_means(
        [float(zsa[i].mean()) for i in zs_ids] + [float(dsa[i].mean()) for i in ds_ids]
    )
    rng = np.random.default_rng(seed)
    shifts = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        bidx = rng.integers(0, k, size=k)
        hz = hypervolume_2d([(zs_pct[i], -float(zsa[i][bidx].mean())) for i in zs_ids], ref)
        hd = hypervolume_2d([(ds_pct[i], -float(dsa[i][bidx].mean())) for i in ds_ids], ref)
        shifts[b] = 100.0 * (hd - hz) / max(hz, 1e-9)
    lo, hi = (float(v) for v in np.quantile(shifts, [0.025, 0.975]))
    mean = float(shifts.mean())
    if lo > 0:
        verdict = f"distillation shifts the frontier out: +{mean:.1f}% HV (95% CI {lo:.1f}..{hi:.1f}%)"
    elif hi < 0:
        verdict = f"distillation regresses the frontier: {mean:.1f}% HV (95% CI {lo:.1f}..{hi:.1f}%)"
    else:
        verdict = f"no measurable shift (95% CI {lo:.1f}..{hi:.1f}%)"
    return {"shift_pct_mean": mean, "ci_lo": lo, "ci_hi": hi, "verdict": verdict}


# ==================================================================================================
# Per-window loss vectors and the paired-delta primitive (fast inner-loop signal).
# ==================================================================================================


@torch.no_grad()
def window_losses(model: Qwen2LM, ids: torch.Tensor) -> torch.Tensor:
    """Per-token nll vector ``[T-1]`` for one window — the paired-bootstrap primitive (no ``.mean()``).

    A single full forward (``past=None``), correct for any genome including linear (which ignores the
    KV cache anyway). ``window_losses(model, ids).mean()`` equals the old ``nas_pareto.nll(ids)``.
    """
    out = model(ids)
    assert isinstance(out, torch.Tensor)
    return F.cross_entropy(out[0, :-1], ids[0, 1:], reduction="none")


@torch.no_grad()
def base_window_losses(
    model: Qwen2LM, restore: Restore, windows: list[torch.Tensor]
) -> list[torch.Tensor]:
    """All-softmax baseline per-window loss vectors, computed once per pool (paired comparison base)."""
    restore()
    return [window_losses(model, w) for w in windows]


@torch.no_grad()
def paired_window_deltas(
    model: Qwen2LM,
    set_genome: SetGenome,
    restore: Restore,
    genome: CatGenome,
    windows: list[torch.Tensor],
    base_losses: list[torch.Tensor],
    use_distill: bool = False,
) -> np.ndarray:
    """Per-window mean Δnll ``[K]`` for a genome vs the cached all-softmax base (same windows = paired)."""
    set_genome(genome, use_distill)
    deltas = np.empty(len(windows), dtype=np.float64)
    for i, w in enumerate(windows):
        deltas[i] = float(window_losses(model, w).mean().item()) - float(base_losses[i].mean().item())
    restore()
    return deltas


def fast_proxy(
    model: Qwen2LM,
    set_genome: SetGenome,
    restore: Restore,
    genome: CatGenome,
    windows: list[torch.Tensor],
    base_losses: list[torch.Tensor],
    use_distill: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """The NAS inner-loop signal: paired multi-window mean Δnll + a bootstrap CI for disclosure.

    The GA selects on ``mean_dnll`` (so :func:`evolve_multiobjective` is unchanged); the CI rides
    along for free (pure numpy on a length-K vector).
    """
    deltas = paired_window_deltas(model, set_genome, restore, genome, windows, base_losses, use_distill)
    ci = bootstrap_paired_ci(deltas, n_boot=n_boot, seed=seed)
    return {
        "mean_dnll": ci["mean"],
        "ci_lo": ci["ci_lo"],
        "ci_hi": ci["ci_hi"],
        "p_worse": ci["p_worse"],
        "n_windows": ci["n_windows"],
    }


@torch.no_grad()
def suffix_losses(
    model: Qwen2LM, ids: torch.Tensor, score_last: int, genome_has_linear: bool
) -> torch.Tensor:
    """KV-primed per-token nll over the last ``score_last`` targets (softmax/sliding genomes only).

    Primes the prefix once with ``return_cache=True`` so the long-context forward is ~O(L) not O(L²).
    Raises ``ValueError`` for a linear-containing genome: :class:`LinearAttention` ignores ``past``,
    so a primed suffix would see only the suffix state and give WRONG losses (use :func:`window_losses`).
    For all-softmax this equals ``window_losses(model, ids)[-score_last:]`` (pinned by a golden test).
    """
    if genome_has_linear:
        raise ValueError(
            "suffix KV-priming is invalid for a linear-containing genome: LinearAttention.forward "
            "ignores `past`; use window_losses (full forward) for linear genomes"
        )
    t = int(ids.size(1))
    if score_last >= t:
        return window_losses(model, ids)[-score_last:]
    p = t - score_last  # prefix length; targets are ids[0, p:t]
    primed = model(ids[:, :p], past=None, return_cache=True)
    assert isinstance(primed, tuple)
    logits_p, cache = primed
    first = logits_p[0, p - 1 : p]  # predicts ids[0, p]
    suffix_in = ids[:, p : t - 1]  # predicts ids[0, p+1:t]
    if int(suffix_in.size(1)) > 0:
        out = model(suffix_in, past=cache, return_cache=False)
        assert isinstance(out, torch.Tensor)
        logits = torch.cat([first, out[0]], dim=0)
    else:
        logits = first
    return F.cross_entropy(logits, ids[0, p:t], reduction="none")


# ==================================================================================================
# Rigorous frontier-only: winner's-curse re-eval + context-length sweep.
# ==================================================================================================


@torch.no_grad()
def reeval_frontier(
    model: Qwen2LM,
    set_genome: SetGenome,
    restore: Restore,
    frontier: list[CatGenome],
    pct_of: Callable[[CatGenome], float],
    search_deltas: dict[CatGenome, np.ndarray],
    holdout_windows: list[torch.Tensor],
    holdout_base: list[torch.Tensor],
    use_distill: bool = False,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Re-evaluate each frontier genome on a FRESH disjoint holdout pool (removes winner's curse).

    The GA selects its frontier by search-window Δnll over hundreds of genomes, so those numbers are
    optimistically biased (max-of-N on noise). The headline verdict must use ``delta_nll_heldout``;
    ``optimism_gap = selection - heldout`` quantifies the bias. ``per_window_holdout`` is returned so
    :func:`bootstrap_hv_gain` needs zero extra forwards.
    """
    rows: list[dict[str, object]] = []
    for g in frontier:
        held = paired_window_deltas(
            model, set_genome, restore, g, holdout_windows, holdout_base, use_distill
        )
        sel_mean = float(np.mean(search_deltas[g]))
        held_mean = float(np.mean(held))
        ci = bootstrap_paired_ci(held, seed=seed)
        st = sign_test(held)
        rows.append(
            {
                "genome": g,
                "pct": float(pct_of(g)),
                "delta_nll_selection": sel_mean,
                "delta_nll_heldout": held_mean,
                "optimism_gap": sel_mean - held_mean,
                "ci_lo": ci["ci_lo"],
                "ci_hi": ci["ci_hi"],
                "p_worse": ci["p_worse"],
                "pos_frac": st["pos_frac"],
                "p_sign": st["p_sign"],
                "per_window_holdout": held,
            }
        )
    return rows


@torch.no_grad()
def context_sweep(
    model: Qwen2LM,
    set_genome: SetGenome,
    restore: Restore,
    genome: CatGenome,
    ids_all: torch.Tensor,
    lengths: tuple[int, ...] = (256, 512, 1024, 2048),
    K: int = 12,
    offset: int = 0,
    seed: int = 0,
    use_distill: bool = False,
) -> dict[int, dict[str, float]]:
    """Paired Δnll + bootstrap CI per swept context length, on FRESH holdout windows.

    Resolves v1's core error: 256 tokens hides linear degradation AND sits below the memory crossover.
    The per-L curve makes regime dependence explicit; the deployment-quality verdict cites the longest
    affordable L, not the inner-loop length. (Uses full forwards — correct for any mixer; for long
    softmax/sliding genomes :func:`suffix_losses` is the available speed-up.)
    """
    result: dict[int, dict[str, float]] = {}
    for length in lengths:
        windows = make_windows(ids_all, length, K, offset=offset)
        if not windows:
            continue
        base = base_window_losses(model, restore, windows)
        deltas = paired_window_deltas(model, set_genome, restore, genome, windows, base, use_distill)
        ci = bootstrap_paired_ci(deltas, seed=seed)
        st = sign_test(deltas)
        result[length] = {
            "mean": ci["mean"],
            "ci_lo": ci["ci_lo"],
            "ci_hi": ci["ci_hi"],
            "p_worse": ci["p_worse"],
            "pos_frac": st["pos_frac"],
            "p_sign": st["p_sign"],
            "n_windows": float(len(windows)),
        }
    return result


def make_windows(
    ids_all: torch.Tensor, L: int, K: int, stride: int | None = None, offset: int = 0
) -> list[torch.Tensor]:
    """Cut up to ``K`` ``[1,L]`` windows from a tokenized corpus; ``stride>=L`` keeps them non-overlapping.

    Non-overlap (the default ``stride=L``) is the cheap independence guard so the window bootstrap is
    not anti-conservative. If the corpus is too short for ``K`` windows, fewer are returned — never
    overlapping. ``offset`` (in tokens) skips a consumed prefix; the holdout pool uses a disjoint one.
    """
    s = stride if stride is not None else L
    n = int(ids_all.size(1))
    out: list[torch.Tensor] = []
    for i in range(K):
        start = offset + i * s
        if start + L > n:
            break
        out.append(ids_all[:, start : start + L])
    return out


# ==================================================================================================
# Long-context needle / passkey probe (direct constant-state failure-mode test).
# ==================================================================================================


def build_passkey_prompt(
    tok: object,
    filler_ids: torch.Tensor,
    total_len: int,
    answer: str = "49271",
    depth_frac: float = 0.5,
) -> tuple[torch.Tensor, slice]:
    """Build a passkey/needle prompt: plant the answer at ``depth_frac``, repeat it at the very end.

    The model must carry the early needle across the context to predict the final answer (an induction
    / long-range copy probe). Returns ``(input_ids [1,total_len], answer_span)`` where ``answer_span``
    indexes the FINAL answer occurrence in the targets array ``ids[:, 1:]``. Non-triviality guard: the
    answer must NOT recur in the gap between the needle and the final answer (else it is locally
    copyable) — raises ``ValueError`` if it does.
    """
    enc = tok(answer, return_tensors="pt")  # type: ignore[operator]
    ans = enc.input_ids[0]
    a = int(ans.shape[0])
    if total_len < 3 * a + 1:
        raise ValueError(f"total_len {total_len} too short for answer of {a} tokens")
    flat = filler_ids.view(-1)
    reps = total_len // int(flat.shape[0]) + 1
    seq = flat.repeat(reps)[:total_len].clone()
    needle_start = max(a, min(int(depth_frac * (total_len - a)), total_len - 2 * a))
    seq[needle_start : needle_start + a] = ans
    seq[total_len - a : total_len] = ans
    gap = seq[needle_start + a : total_len - a]
    g = int(gap.shape[0])
    for i in range(g - a + 1):
        if torch.equal(gap[i : i + a], ans):
            raise ValueError("answer recurs in the gap (needle..final): retrieval would be trivial")
    ids = seq.view(1, total_len)
    span = slice(total_len - a - 1, total_len - 1)  # targets ids[:,1:] whose next-token is the answer
    return ids, span


@torch.no_grad()
def score_needle(model: Qwen2LM, input_ids: torch.Tensor, answer_span: slice) -> dict[str, float]:
    """Teacher-forced retrieval score over the answer span: mean answer-token logprob + argmax accuracy.

    A single full forward (no generation/KV-cache, so position handling is identical across mixers).
    ``argmax_acc`` is exact-match retrieval; the caller gates it with an in-window control accuracy.
    """
    out = model(input_ids)
    assert isinstance(out, torch.Tensor)
    logprobs = F.log_softmax(out[0, :-1], dim=-1)
    targets = input_ids[0, 1:]
    span_lp = logprobs[answer_span]
    span_tgt = targets[answer_span]
    chosen = span_lp.gather(1, span_tgt[:, None]).squeeze(1)
    acc = float((span_lp.argmax(dim=-1) == span_tgt).float().mean().item())
    return {"mean_logprob": float(chosen.mean().item()), "argmax_acc": acc}


@torch.no_grad()
def needle_horizon(
    model: Qwen2LM,
    set_genome: SetGenome,
    restore: Restore,
    genome: CatGenome,
    tok: object,
    filler_ids: torch.Tensor,
    base_acc: dict[tuple[int, float], float],
    lengths: tuple[int, ...] = (2048, 4096),
    depths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9),
    answer: str = "49271",
    use_distill: bool = False,
) -> dict[str, object]:
    """Per-genome long-context retrieval horizon: the shortest length where the genome fails (acc<1)
    while the all-softmax base succeeds (``base_acc==1``). ``None`` if it never fails. ``base_acc`` is
    the all-softmax reference accuracy per ``(length, depth)``, computed once by the caller.
    """
    set_genome(genome, use_distill)
    by_depth: dict[str, dict[str, float]] = {}
    horizon: int | None = None
    for length in lengths:
        failed_here = False
        for depth in depths:
            try:
                ids, span = build_passkey_prompt(tok, filler_ids, length, answer=answer, depth_frac=depth)
            except ValueError:
                continue
            sc = score_needle(model, ids, span)
            base = base_acc.get((length, depth), 1.0)
            by_depth[f"{length}:{depth}"] = {
                "argmax_acc": sc["argmax_acc"],
                "mean_logprob": sc["mean_logprob"],
                "control_acc": base,
            }
            if base >= 1.0 and sc["argmax_acc"] < 1.0:
                failed_here = True
        if failed_here and horizon is None:
            horizon = int(length)
    restore()
    return {"horizon": horizon, "by_depth": by_depth}


# ==================================================================================================
# Attention-map KL fidelity (diagnostic only; short-window, never wired into fitness).
# ==================================================================================================


class _AttnProj(Protocol):
    q_proj: torch.nn.Linear
    k_proj: torch.nn.Linear
    p: Qwen2Params


def _masked_softmax_attention(
    attn: _AttnProj, x_norm: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None
) -> torch.Tensor:
    """Causal softmax attention ``[H,T,T]`` from a module's q/k projections; optional sliding window."""
    p = attn.p
    b, t, _ = x_norm.shape
    q = attn.q_proj(x_norm).view(b, t, p.n_head, p.head_dim).transpose(1, 2)
    k = attn.k_proj(x_norm).view(b, t, p.n_kv_head, p.head_dim).transpose(1, 2)
    q, k = _apply_rope(q, k, cos, sin)
    kf = _repeat_kv(k, p.n_head // p.n_kv_head)
    scores = torch.matmul(q, kf.transpose(-2, -1)) / (p.head_dim**0.5)
    pos = torch.arange(t)
    keep = pos[None, :] <= pos[:, None]
    if window is not None:
        keep = keep & ((pos[:, None] - pos[None, :]) < window)
    scores = scores + torch.where(keep, 0.0, float("-inf"))
    return torch.softmax(scores, dim=-1)[0]


@torch.no_grad()
def softmax_teacher_attention(
    attn: _AttnProj, x_norm: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Full causal softmax attention ``[H,T,T]`` (the teacher) from the module's pretrained q/k."""
    return _masked_softmax_attention(attn, x_norm, cos, sin, None)


@torch.no_grad()
def implied_linear_attention(
    attn: LinearAttention, x_norm: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Reconstruct the causal row-stochastic attention implied by linear attention ``[H,T,T]``.

    ``A[i,j] = φ(q_i)·φ(k_j) / Σ_{j'≤i} φ(q_i)·φ(k_{j'})`` using the exact ``linearize.py`` path
    (q/k projections → RoPE → repeat_kv → learnable affine if present → φ=elu+1).
    """
    p = attn.p
    b, t, _ = x_norm.shape
    q = attn.q_proj(x_norm).view(b, t, p.n_head, p.head_dim).transpose(1, 2)
    k = attn.k_proj(x_norm).view(b, t, p.n_kv_head, p.head_dim).transpose(1, 2)
    q, k = _apply_rope(q, k, cos, sin)
    kf = _repeat_kv(k, p.n_head // p.n_kv_head)
    if attn.learnable:
        q = q * attn.q_scale[None, :, None, :] + attn.q_bias[None, :, None, :]
        kf = kf * attn.k_scale[None, :, None, :] + attn.k_bias[None, :, None, :]
    qphi = _phi(q)[0]  # [H,T,Dk]
    kphi = _phi(kf)[0]
    scores = torch.einsum("hid,hjd->hij", qphi, kphi)
    scores = scores * torch.tril(torch.ones(t, t))
    return scores / (scores.sum(dim=-1, keepdim=True) + eps)


def attn_kl_softmax_linear(a_sm: torch.Tensor, a_lin: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Forward KL(softmax || linear) in nats, averaged over heads and query rows ``i≥1``.

    Forward KL (teacher as reference) penalizes the student DROPPING attention the teacher placed —
    the failure mode that hurts long-context retrieval. Row 0 is trivially ``[1]`` and skipped.
    """
    p = a_sm.clamp_min(0.0) + eps
    qd = a_lin.clamp_min(0.0) + eps
    kl = (a_sm * (torch.log(p) - torch.log(qd))).sum(dim=-1)  # [H,T]; a_sm==0 terms vanish
    if kl.size(1) > 1:
        kl = kl[:, 1:]
    return kl.mean()


@torch.no_grad()
def genome_attn_kl(
    model: Qwen2LM,
    set_genome: SetGenome,
    restore: Restore,
    genome: CatGenome,
    calib_ids: torch.Tensor,
    t_kl: int = 256,
    head_stride: int = 1,
    use_distill: bool = False,
) -> dict[str, object]:
    """Per-genome attention-KL over converted (opt≠0) layers at a SHORT window (``t_kl<=256`` asserted).

    Diagnostic only — measures how far each converted layer's attention diverges from the softmax it
    replaced, at the realistic runtime input (captured under the installed genome). All-softmax → 0.0.
    O(T²) per head, so the short cap; this NEVER feeds the NAS fitness.
    """
    assert t_kl <= 256, "attention-KL is O(T^2) per head; capped at 256 tokens"
    ids = calib_ids[:, :t_kl]
    set_genome(genome, use_distill)
    p = model.params
    t = int(ids.size(1))
    cos, sin = _rope_cos_sin(torch.arange(t), p.head_dim, p.rope_theta)
    per_layer: dict[int, float] = {}
    for i, opt in enumerate(genome):
        if opt == 0:
            continue
        block = model.model.layers[i]
        captured: dict[str, torch.Tensor] = {}

        def grab(_m: object, _inp: object, o: torch.Tensor, store: dict[str, torch.Tensor] = captured) -> None:
            store["x"] = o.detach()

        handle = block.input_layernorm.register_forward_hook(grab)  # type: ignore[union-attr]
        try:
            model(ids)
        finally:
            handle.remove()
        x_norm = captured["x"]
        student = block.self_attn
        a_sm = softmax_teacher_attention(student, x_norm, cos, sin)  # type: ignore[arg-type]
        if isinstance(student, LinearAttention):
            a_st = implied_linear_attention(student, x_norm, cos, sin)
        elif isinstance(student, SlidingWindowAttention):
            a_st = _masked_softmax_attention(student, x_norm, cos, sin, student.window)  # type: ignore[arg-type]
        else:
            continue
        per_layer[i] = float(attn_kl_softmax_linear(a_sm, a_st).item())
    restore()
    if not per_layer:
        return {"mean": 0.0, "max": 0.0, "sum": 0.0, "per_layer": []}
    vals = list(per_layer.values())
    return {
        "mean": float(np.mean(vals)),
        "max": float(np.max(vals)),
        "sum": float(np.sum(vals)),
        "per_layer": [{"layer": float(i), "kl": per_layer[i]} for i in sorted(per_layer)],
    }


# ==================================================================================================
# Honest-disclosure chokepoint + report assembler (pure).
# ==================================================================================================


def honest_verdict(
    hv_gain_ci: dict[str, float],
    frontier_holdout: list[dict[str, object]],
    proxy_vs_judge_tau: float | None,
    ci_halfwidth_floor: float,
) -> dict[str, str]:
    """Turn every guard into one disclosed verdict on memetic-vs-greedy.

    Precedence: (1) suppress if any frontier ``optimism_gap`` exceeds the noise floor (selection
    optimism dominates); (2) null if the hypervolume-gain CI brackets 0; (3) null if the CI is entirely
    below 0 (memetic worse); (4) otherwise memetic beats greedy — 'significant', downgraded to
    'suggestive' if the proxy-vs-judge rank correlation ``τ<0.7``. Never emits a conversational claim.
    """
    ci_lo = float(hv_gain_ci["ci_lo"])
    ci_hi = float(hv_gain_ci["ci_hi"])
    max_gap = max((float(p.get("optimism_gap", 0.0)) for p in frontier_holdout), default=0.0)
    if max_gap > ci_halfwidth_floor:
        return {
            "memetic_vs_greedy": "verdict suppressed: selection optimism exceeds noise floor",
            "confidence": "suppressed",
            "notes": f"max optimism_gap {max_gap:.4f} > CI half-width floor {ci_halfwidth_floor:.4f}",
        }
    if ci_lo <= 0.0 <= ci_hi:
        return {
            "memetic_vs_greedy": f"no significant difference (95% CI {ci_lo:.1f}..{ci_hi:.1f}% HV)",
            "confidence": "null",
            "notes": "hypervolume-gain CI includes 0",
        }
    if ci_hi < 0.0:
        return {
            "memetic_vs_greedy": f"greedy not beaten (memetic worse, 95% CI {ci_lo:.1f}..{ci_hi:.1f}%)",
            "confidence": "null",
            "notes": "memetic frontier is dominated on holdout",
        }
    if proxy_vs_judge_tau is not None and proxy_vs_judge_tau < 0.7:
        return {
            "memetic_vs_greedy": f"memetic beats greedy: +{float(hv_gain_ci['gain_pct_mean']):.1f}% HV "
            f"(95% CI {ci_lo:.1f}..{ci_hi:.1f}%)",
            "confidence": "suggestive",
            "notes": f"proxy-vs-judge tau {proxy_vs_judge_tau:.2f} < 0.7: NAS objective may mis-track judge",
        }
    return {
        "memetic_vs_greedy": f"memetic beats greedy: +{float(hv_gain_ci['gain_pct_mean']):.1f}% HV "
        f"(95% CI {ci_lo:.1f}..{ci_hi:.1f}%)",
        "confidence": "significant",
        "notes": "holdout hypervolume-gain CI entirely above 0",
    }


def build_proxy_v2_report(
    *,
    inner_context: int,
    context_sweep: dict[int, dict[str, float]],
    frontier_holdout: list[dict[str, object]],
    hv_gain_ci: dict[str, float],
    right_shift_ci: dict[str, object] | None,
    needle: dict[str, object] | None,
    attention_kl: dict[str, object] | None,
    proxy_vs_judge_tau: float | None,
    cross_corpus: dict[str, object] | None,
) -> dict[str, object]:
    """Assemble the additive ``proxy_v2`` report block (pure dict; testable without a model).

    Pins ``scope='next_token_nll_proxy'`` and ``conversational_claim=None`` — a conversational quality
    claim must live in a separate disclosed generation eval, never inferred from these perplexity proxies.
    """
    return {
        "scope": "next_token_nll_proxy",
        "conversational_claim": None,
        "inner_context": inner_context,
        "context_sweep": {str(k): v for k, v in context_sweep.items()},
        "frontier_holdout": [
            {k: v for k, v in row.items() if k != "per_window_holdout"} for row in frontier_holdout
        ],
        "hv_gain_ci": hv_gain_ci,
        "right_shift_ci": right_shift_ci,
        "needle": needle,
        "attention_kl": attention_kl,
        "proxy_vs_judge_tau": proxy_vs_judge_tau,
        "cross_corpus": cross_corpus,
        "note": (
            "next-token-nll proxy with paired multi-window bootstrap CIs; the headline memetic-vs-greedy "
            "verdict uses fresh holdout windows (winner's-curse removed). Conversational quality is NOT "
            "assessed here."
        ),
    }
