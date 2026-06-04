# SPDX-License-Identifier: Apache-2.0
"""L0 realization smoke: does the tiny verified-core reservoir LM actually function as an LM?

Run: py -3.11 demo_lm.py
PASS (L0) iff the best contracting gene's held-out per-byte CE beats the unigram baseline.
Also checks the Lemma-1 soundness contract (|s|<1) and the empirical-contraction oracle.
"""
from __future__ import annotations

import numpy as np

from lm_substrate import (
    ByteEmbedding,
    CoupledNDGene,
    LMTask,
    cert_inf,
    cert_two,
    classify_region,
    empirical_contraction,
    hidden_stability,
    load_corpus,
    to_ids,
)


def main() -> None:
    n = 32
    data = load_corpus(max_bytes=16384)
    ids = to_ids(data)
    emb = ByteEmbedding.make(n=n, seed=0)
    task = LMTask(emb=emb, ids=ids, alpha=1e-2)

    print(f"corpus bytes={len(data)}  distinct bytes={len(set(data))}  n={n}")
    print(f"unigram: CE={task.unigram_ce:.4f} nats  fitness(exp-CE)={task.unigram_fitness:.4f}")

    rng = np.random.default_rng(7)
    # candidate genes: a diagonal-decay (no coupling) gene + random contracting genes
    cands: list[tuple[str, CoupledNDGene]] = []
    cands.append(("diag_decay0.6", CoupledNDGene.make(decay=np.full(n, 0.6), W=np.zeros((n, n)))))
    n_contracting = 0
    tries = 0
    while n_contracting < 8 and tries < 4000:
        tries += 1
        decay = rng.uniform(0.3, 0.95, size=n)
        # small random coupling, scaled to give a decent chance of contraction
        W = rng.standard_normal((n, n)) * (0.6 / np.sqrt(n))
        g = CoupledNDGene.make(decay=decay, W=W)
        if cert_two(g):  # sound contraction (>= inf-norm region)
            cands.append((f"rand_two#{n_contracting}", g))
            n_contracting += 1

    print(f"\n{'gene':22s} {'region':14s} {'emp_rho':>8s} {'max|s|':>7s} {'CE':>8s} {'fit':>7s} {'<unigram?':>9s}")
    best_ce = float("inf")
    best_name = ""
    for name, g in cands:
        ce = task.held_out_ce(g)
        fit = float(np.exp(-ce)) if np.isfinite(ce) else 0.0
        rho = empirical_contraction(g, task._emb_seq)
        mx, nan = hidden_stability(g, task._emb_seq)
        reg = classify_region(g)
        beats = "YES" if ce < task.unigram_ce else "no"
        print(f"{name:22s} {reg:14s} {rho:8.4f} {mx:7.4f} {ce:8.4f} {fit:7.4f} {beats:>9s}")
        if ce < best_ce:
            best_ce, best_name = ce, name

    print("\n--- L0 verdict ---")
    print(f"best gene = {best_name}  CE={best_ce:.4f}  unigram CE={task.unigram_ce:.4f}")
    improvement = task.unigram_ce - best_ce
    print(f"improvement over unigram = {improvement:.4f} nats "
          f"({100*improvement/task.unigram_ce:.1f}%)  -> L0 {'PASS' if improvement > 0.02 else 'FAIL'}")

    # soundness contract sanity (Lemma 1): every gene keeps |s|<1
    all_bounded = all(hidden_stability(g, task._emb_seq)[0] < 1.0 + 1e-9 for _, g in cands)
    print(f"Lemma-1 (|s|<1) holds for all candidates: {all_bounded}")


if __name__ == "__main__":
    main()
