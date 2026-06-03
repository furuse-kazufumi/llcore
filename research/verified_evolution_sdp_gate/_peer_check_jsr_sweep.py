# SPDX-License-Identifier: Apache-2.0
"""Peer-review verification: re-run the JSR_lb sweep ml=1..8 on the deg6 residual.

Independent of jsr_bracket.py / verify_certifier_jsr_soundness.py; only reuses the
same vertex extraction (_vertices_n2) and the same residual gene list. Reports for
each max_len: n_expansive, frac_gap, the expansive index set, and per-gene jsr_lb so
we can check (a) convergence length, (b) 5->6 stability, (c) the specific genes the
reviewer cited (gene[20], gene[49]).
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coupled_map import CoupledGene  # noqa: E402
from verifier_deg4 import _vertices_n2  # noqa: E402


def jsr_lb(vertices, max_len):
    V = [np.asarray(M, dtype=np.float64) for M in vertices]
    best = 0.0
    best_len = 1
    for k in range(1, max_len + 1):
        for combo in itertools.product(range(len(V)), repeat=k):
            P = np.eye(V[0].shape[0])
            for i in combo:
                P = V[i] @ P
            val = float(np.max(np.abs(np.linalg.eigvals(P)))) ** (1.0 / k)
            if val > best:
                best, best_len = val, k
    return best, best_len


def main():
    with open(os.path.join(_HERE, "exp_deg6_residual_genes.json"), encoding="utf-8") as f:
        residual = json.load(f)["residual_uncert"]
    n = len(residual)
    print(f"n_residual = {n}")

    # precompute vertex sets per gene once
    verts = []
    for rec in residual:
        g = CoupledGene.make(decay=np.asarray(rec["decay"]),
                             W=np.asarray(rec["W"]).reshape(2, 2))
        verts.append(_vertices_n2(g))

    THR = 1.0 - 1e-9
    last_jsr = {}
    for ml in range(1, 9):
        jsrs, lens = [], []
        exp_idx = []
        for i, vv in enumerate(verts):
            j, bl = jsr_lb(vv, ml)
            jsrs.append(j)
            lens.append(bl)
            if j >= THR:
                exp_idx.append(i)
        frac_gap = (n - len(exp_idx)) / n
        print(f"ml={ml}: n_expansive={len(exp_idx):2d} frac_gap={frac_gap:.4f} "
              f"exp_idx={exp_idx}")
        last_jsr = {"ml": ml, "jsrs": jsrs, "lens": lens, "exp_idx": exp_idx}

    # detail on the cited genes at max_len=6 (and 5,8 for transition)
    print("\n--- cited-gene tracking (jsr_lb at ml=5,6,8) ---")
    for idx in (20, 49):
        row = []
        for ml in (5, 6, 8):
            j, bl = jsr_lb(verts[idx], ml)
            row.append((ml, round(j, 5), bl))
        print(f"gene[{idx}]: {row}")

    # argmax_len of the 6 expansive genes at ml=6 (reviewer: 5 of 6 are argmax_len=1)
    print("\n--- expansive genes argmax_len at ml=6 ---")
    for idx in last_jsr["exp_idx"]:
        j, bl = jsr_lb(verts[idx], 6)
        print(f"gene[{idx}]: jsr_lb={round(j,4)} argmax_len={bl}")


if __name__ == "__main__":
    main()
