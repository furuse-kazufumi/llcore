# SPDX-License-Identifier: Apache-2.0
"""Tests for 2-D Pareto-frontier comparison metrics (memory↔quality NAS).

Hypervolume is the scalar that lets us say one frontier dominates another; ``frontier_right_shift``
uses it to quantify how far per-layer distillation pushes the frontier out vs zero-shot. Pure
functions — no model in the loop — so the comparison logic is pinned independently of the heavy run.
"""
from __future__ import annotations


def test_hypervolume_single_point_is_rectangle() -> None:
    from llcore.runtime.pareto_metrics import hypervolume_2d

    assert hypervolume_2d([(2.0, 3.0)], (0.0, 0.0)) == 6.0


def test_hypervolume_staircase_of_two_points() -> None:
    from llcore.runtime.pareto_metrics import hypervolume_2d

    # (3,1) contributes 3×1; (1,3) raises the ceiling to 3 over width 1 → +2; total 5
    assert hypervolume_2d([(3.0, 1.0), (1.0, 3.0)], (0.0, 0.0)) == 5.0


def test_hypervolume_ignores_dominated_points() -> None:
    from llcore.runtime.pareto_metrics import hypervolume_2d

    base = hypervolume_2d([(3.0, 1.0), (1.0, 3.0)], (0.0, 0.0))
    with_dom = hypervolume_2d([(3.0, 1.0), (1.0, 3.0), (2.0, 0.5)], (0.0, 0.0))
    assert with_dom == base  # (2, 0.5) is dominated → adds no area


def test_hypervolume_respects_reference() -> None:
    from llcore.runtime.pareto_metrics import hypervolume_2d

    # raising the reference floor shrinks the dominated area: (2-0)*(3-1) == 4
    assert hypervolume_2d([(2.0, 3.0)], (0.0, 1.0)) == 4.0


def test_right_shift_positive_when_distilled_dominates() -> None:
    from llcore.runtime.pareto_metrics import frontier_right_shift

    # distilled reaches the same memory savings at strictly better quality (higher -Δnll)
    zs = [(40.0, -0.10), (60.0, -0.30)]
    ds = [(40.0, -0.02), (60.0, -0.12)]
    res = frontier_right_shift(zs, ds)
    assert isinstance(res["shift_pct"], float) and res["shift_pct"] > 0.0
    assert "shifts the frontier out" in res["verdict"]


def test_right_shift_zero_when_identical() -> None:
    from llcore.runtime.pareto_metrics import frontier_right_shift

    pts = [(40.0, -0.10), (60.0, -0.30)]
    res = frontier_right_shift(pts, list(pts))
    assert abs(res["shift_pct"]) < 1e-6  # type: ignore[arg-type]
    assert "does not measurably move" in res["verdict"]


def test_right_shift_negative_when_distilled_worse() -> None:
    from llcore.runtime.pareto_metrics import frontier_right_shift

    zs = [(40.0, -0.02), (60.0, -0.12)]
    ds = [(40.0, -0.10), (60.0, -0.30)]
    res = frontier_right_shift(zs, ds)
    assert isinstance(res["shift_pct"], float) and res["shift_pct"] < 0.0
    assert "regresses the frontier" in res["verdict"]
