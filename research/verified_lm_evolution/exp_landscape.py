# SPDX-License-Identifier: Apache-2.0
"""L3 (mechanism, non-circular) — verifier-region attribution of held-out LM perplexity.

The decisive, gate-independent evidence for the arc's signature claim, ported to a REAL byte-LM:
classify every gene by its tightest sound contraction certifier (inf / two_norm_only / sdp_only /
non_certified) and measure the best achievable held-out cross-entropy per region. If a
less-conservative (but still sound) verifier region contains genes with strictly lower perplexity,
then a better verifier unlocks more reachable LM fitness — on real language-model loss, not a
synthetic task. If the regions tie, that is an honest NULL: on a real byte-LM the conservative gate
suffices (the optimal recurrence is near-diagonal), and the verifier-fitness frontier is
task-structural (synthetic-rotation-specific), which is itself a publishable honest result.

Also yields L1 (admitted => empirically contracting) and L2 (non_certified contains expansive genes).

Run: py -3.11 exp_landscape.py [n_genes] [max_bytes]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from lm_substrate import (
    ByteEmbedding,
    CoupledNDGene,
    LMTask,
    classify_region,
    empirical_contraction,
    load_corpus,
    to_ids,
)

N = 8


def sample_gene(rng) -> CoupledNDGene:
    """Sample across regions: vary decay floor + coupling scale so all 4 regions are populated."""
    decay = rng.uniform(rng.uniform(0.0, 0.5), 1.0, size=N)
    w_scale = rng.choice([0.15, 0.3, 0.6, 1.0, 1.5]) / np.sqrt(N)
    W = np.clip(rng.standard_normal((N, N)) * w_scale, -2.0, 2.0)
    return CoupledNDGene.make(decay=decay, W=W)


def main() -> None:
    n_genes = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    max_bytes = int(sys.argv[2]) if len(sys.argv) > 2 else 16384

    data = load_corpus(max_bytes=max_bytes)
    ids = to_ids(data)
    emb = ByteEmbedding.make(n=N, seed=0)
    task = LMTask(emb=emb, ids=ids, readout_steps=100, lr=0.5)
    uni = task.unigram_ce
    print(f"corpus={len(data)}B  n={N}  n_genes={n_genes}  unigram_CE={uni:.4f}", flush=True)

    rng = np.random.default_rng(20260604)
    regions: dict[str, list] = {"inf": [], "two_norm_only": [], "sdp_only": [], "non_certified": []}
    t0 = time.time()
    rows = []
    for i in range(n_genes):
        g = sample_gene(rng)
        reg = classify_region(g)
        ce = task.held_out_ce(g)
        rho = empirical_contraction(g, task._emb_seq, stride=11)
        contracting = bool(rho < 1.0) and np.isfinite(ce)
        regions[reg].append((ce, contracting, rho))
        rows.append({"region": reg, "ce": None if not np.isfinite(ce) else round(ce, 5),
                     "emp_rho": round(rho, 5), "contracting": contracting})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_genes}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'region':16s} {'n':>5s} {'best_CE':>8s} {'best_CE(contr)':>14s} "
          f"{'median_CE':>9s} {'<unigram(best)':>14s} {'%expansive':>10s}")
    summary = {}
    for reg, items in regions.items():
        if not items:
            print(f"{reg:16s} {0:5d}")
            summary[reg] = {"n": 0}
            continue
        ces = np.array([c for c, _, _ in items if np.isfinite(c)])
        ces_contr = np.array([c for c, ok, _ in items if ok])
        n_exp = sum(1 for _, ok, _ in items if not ok)
        best = float(ces.min()) if ces.size else float("nan")
        best_contr = float(ces_contr.min()) if ces_contr.size else float("nan")
        med = float(np.median(ces)) if ces.size else float("nan")
        pct_exp = 100.0 * n_exp / len(items)
        print(f"{reg:16s} {len(items):5d} {best:8.4f} {best_contr:14.4f} {med:9.4f} "
              f"{uni-best:14.4f} {pct_exp:10.1f}")
        summary[reg] = {"n": len(items), "best_ce": best, "best_ce_contracting": best_contr,
                        "median_ce": med, "pct_expansive": pct_exp}

    # L3 frontier read: best-contracting CE by region (the honest, contraction-filtered ceiling)
    print("\n--- L3 frontier (best CONTRACTING held-out CE per region; lower=better) ---")
    order = ["inf", "two_norm_only", "sdp_only", "non_certified"]
    prev = None
    for reg in order:
        s = summary.get(reg, {})
        if s.get("n", 0) == 0:
            continue
        bc = s.get("best_ce_contracting", float("nan"))
        delta = "" if prev is None or not np.isfinite(bc) or not np.isfinite(prev) else f"  (Δ vs prev={prev-bc:+.4f})"
        print(f"  {reg:16s} best_contracting_CE={bc:.4f}{delta}")
        if np.isfinite(bc):
            prev = bc

    out = {"corpus_bytes": len(data), "n": N, "n_genes": n_genes, "unigram_ce": uni,
           "summary": summary, "rows": rows, "elapsed_s": round(time.time() - t0, 1)}
    with open("exp_landscape_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved exp_landscape_results.json  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
