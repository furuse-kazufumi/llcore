# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/feature_map_spikiness.py`` metric helpers (L7 Hedgehog ablation).

Pins the spikiness/monotonicity metrics so the ablation's numbers mean what the report claims:
entropy is 0 for a one-hot row and maximal for a uniform row; top-1 mass tracks spikiness; and the
Spearman rank correlation is +1 for identical key orderings, -1 for reversed.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "feature_map_spikiness.py"
_spec = importlib.util.spec_from_file_location("feature_map_spikiness", _SCRIPT)
assert _spec is not None and _spec.loader is not None
fms = importlib.util.module_from_spec(_spec)
sys.modules["feature_map_spikiness"] = fms
_spec.loader.exec_module(fms)


def _causal_uniform(h: int, t: int) -> torch.Tensor:
    """[H,T,T] causal weights, each row uniform over keys j<=i."""
    w = torch.zeros(h, t, t)
    for i in range(t):
        w[:, i, : i + 1] = 1.0 / (i + 1)
    return w


def _causal_onehot_diag(h: int, t: int) -> torch.Tensor:
    """[H,T,T] causal weights putting all mass on the diagonal key (perfectly spiky)."""
    w = torch.zeros(h, t, t)
    for i in range(t):
        w[:, i, i] = 1.0
    return w


def test_entropy_uniform_is_log_n() -> None:
    t = 16
    w = _causal_uniform(2, t)
    # last row has 16 keys -> entropy ln(16); helper averages rows with >= min_keys keys
    ent = fms.mean_row_entropy(w, min_keys=t)  # only the full row qualifies
    assert math.isclose(ent, math.log(t), rel_tol=1e-5)


def test_entropy_onehot_is_zero_and_top1_is_one() -> None:
    w = _causal_onehot_diag(3, 12)
    assert fms.mean_row_entropy(w, min_keys=4) < 1e-6
    assert math.isclose(fms.mean_top1_mass(w, min_keys=4), 1.0, rel_tol=1e-6)


def test_top1_uniform_is_one_over_n() -> None:
    t = 10
    w = _causal_uniform(1, t)
    # only the last (full) row qualifies -> top1 = 1/t
    assert math.isclose(fms.mean_top1_mass(w, min_keys=t), 1.0 / t, rel_tol=1e-5)


def test_spearman_identical_and_reversed() -> None:
    a = torch.tensor([0.1, 0.2, 0.3, 0.9])
    assert math.isclose(fms._spearman(a, a), 1.0, rel_tol=1e-6)
    assert math.isclose(fms._spearman(a, a.flip(0)), -1.0, rel_tol=1e-6)


def test_avg_ranks_handles_ties() -> None:
    # tied values must share their mean rank (1-based): [10,10,20,30] -> ranks [1.5,1.5,3,4]
    r = fms._avg_ranks(torch.tensor([10.0, 10.0, 20.0, 30.0]))
    assert torch.allclose(r, torch.tensor([1.5, 1.5, 3.0, 4.0]))


def test_spearman_tie_aware_constant_vector() -> None:
    # an all-equal vector (e.g. a near-uniform row collapsed to ties) has zero rank variance ->
    # correlation falls back to 0 (no spurious +/-1 from arbitrary argsort tie-breaking)
    a = torch.tensor([0.25, 0.25, 0.25, 0.25])
    b = torch.tensor([0.1, 0.2, 0.3, 0.9])
    assert abs(fms._spearman(a, b)) < 1e-6


def test_rank_corr_perfect_when_weights_share_order() -> None:
    # softmax and linear rows that rank keys identically -> rank corr +1
    h, t = 1, 8
    a_sm = _causal_uniform(h, t).clone()
    a_lin = _causal_uniform(h, t).clone()
    for i in range(t):
        order = torch.linspace(0.1, 1.0, i + 1)
        a_sm[0, i, : i + 1] = order / order.sum()
        a_lin[0, i, : i + 1] = (order**2) / (order**2).sum()  # monotone transform -> same ranking
    assert math.isclose(fms.mean_rank_corr(a_sm, a_lin, min_keys=4), 1.0, rel_tol=1e-6)


def test_linear_weights_are_causal_and_normalized() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 6, 4)
    kf = torch.randn(2, 6, 4)
    w = fms.linear_weights(q, kf)
    # rows sum to 1 over valid keys
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 6), atol=1e-5)
    # strictly causal: no mass above the diagonal
    upper = torch.triu(torch.ones(6, 6), diagonal=1).bool()
    assert float(w[:, upper].abs().max()) < 1e-7


def test_softmax_weights_match_manual() -> None:
    torch.manual_seed(1)
    q = torch.randn(1, 4, 8)
    kf = torch.randn(1, 4, 8)
    w = fms.softmax_weights(q, kf, head_dim=8)
    assert torch.allclose(w.sum(dim=-1), torch.ones(1, 4), atol=1e-5)
    # row 0 attends only key 0
    assert math.isclose(float(w[0, 0, 0]), 1.0, rel_tol=1e-6)
