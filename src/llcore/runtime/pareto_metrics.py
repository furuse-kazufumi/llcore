# SPDX-License-Identifier: Apache-2.0
"""2-D Pareto-frontier comparison metrics for the memory<->quality NAS.

A NAS run produces a frontier of ``(% attention memory saved, Δnll)`` points; comparing two
frontiers (greedy vs memetic, or zero-shot vs distillation-aware) needs a single scalar that
respects domination. 2-D hypervolume — the area a maximization front dominates above a shared
lower-left reference — is that scalar: front A beats front B iff it dominates strictly more area.
``frontier_right_shift`` wraps it to quantify how far per-layer distillation pushes the frontier
out relative to the zero-shot frontier (step ② of the conversational-llcore line).
"""
from __future__ import annotations


def hypervolume_2d(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    """2-D hypervolume (dominated area) of a **maximization** front.

    ``points`` are ``(x, y)`` maximization objectives; ``ref`` is the shared lower-left reference
    so two fronts are directly comparable. Points are swept in descending ``x``; each one that
    raises the running ``y`` ceiling contributes the rectangle it newly dominates, so dominated
    points add nothing and the result is the staircase area above ``ref``.
    """
    hv = 0.0
    y_prev = ref[1]
    for x, y in sorted(points, key=lambda p: -p[0]):
        if y > y_prev:
            hv += (x - ref[0]) * (y - y_prev)
            y_prev = y
    return hv


def frontier_right_shift(
    zero_shot: list[tuple[float, float]],
    distilled: list[tuple[float, float]],
    *,
    tol: float = 0.005,
) -> dict[str, float | str]:
    """Quantify how far distillation moved the memory/quality frontier, by 2-D hypervolume.

    Each list holds ``(pct_mem_saved, -delta_nll)`` maximization points. A converted layer keeps the
    same constant-state memory whether or not it was distilled, so distillation can only move the
    *quality* axis — the frontier shifts up/out, never changing the memory a genome costs. A shared
    lower-left reference derived from both fronts makes the two hypervolumes directly comparable.
    Returns ``{zero_shot_hv, distilled_hv, shift_pct, verdict}``; ``shift_pct`` is the relative
    hypervolume gain (positive = distillation enlarged the dominated region).
    """
    min_y = min((y for _, y in zero_shot + distilled), default=0.0)
    ref = (0.0, min_y - 1e-6)
    zs = hypervolume_2d(zero_shot, ref)
    ds = hypervolume_2d(distilled, ref)
    shift = 100.0 * (ds - zs) / max(zs, 1e-9)
    if ds > zs * (1.0 + tol):
        verdict = f"distillation shifts the frontier out: hypervolume +{shift:.1f}%"
    elif ds < zs * (1.0 - tol):
        verdict = f"distillation regresses the frontier: hypervolume {shift:.1f}%"
    else:
        verdict = "distillation does not measurably move the frontier"
    return {"zero_shot_hv": zs, "distilled_hv": ds, "shift_pct": shift, "verdict": verdict}
