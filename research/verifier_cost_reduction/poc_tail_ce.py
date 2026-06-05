# SPDX-License-Identifier: Apache-2.0
"""PoC-2.5: does the cheap-bound's MISSED tail carry the navigable low-perplexity dynamics?

Decision-relevant question for whether the genuine robust-LMI / SDP (R-LLM-1, PoC-3) is worth building.

PoC-2: B2 = σ(|M|+R) (1 SVD, vertex-free, sound) recovers 77.6% of the exact 2^n cert_two reach. The
remaining ~22% = genes cert_two ADMITS but B2 REJECTS — only the full vertex enumeration (or an SDP)
certifies them. If those B2-missed genes carry STRICTLY LOWER held-out LM perplexity than the
B2-admitted genes, then the cheap bound forfeits real modeling capacity and the SDP is worth building.
If they do NOT (their CE distribution is no better), then B2 already captures the useful dynamics and
the SDP is unnecessary for the LM objective — a clean, honest go/no-go for PoC-3.

Method (real byte-LM, same substrate as L3): sample genes; for those cert_two admits, compute held-out
CE; split into {B2-admitted} vs {B2-rejected}; compare best/median CE. Reuses lm_substrate (real CE)
and poc_l2lite_v2.bound_b2. src/ untouched.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.normpath(os.path.join(_HERE, "..", "verified_lm_evolution"))):
    if p not in sys.path:
        sys.path.insert(0, p)

from lm_substrate import ByteEmbedding, LMTask, cert_two, load_corpus, to_ids  # noqa: E402
from poc_l2lite import MAX_INPUT_ABS, sample_gene  # noqa: E402
from poc_l2lite_v2 import bound_b2  # noqa: E402

N = 8


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    target_admitted = int(sys.argv[1]) if len(sys.argv) > 1 else 400  # # of cert_two genes to score
    max_bytes = int(sys.argv[2]) if len(sys.argv) > 2 else 8192       # match the gated L3 corpus

    data = load_corpus(max_bytes=max_bytes)
    ids = to_ids(data)
    emb = ByteEmbedding.make(n=N, seed=0)
    task = LMTask(emb=emb, ids=ids, readout_steps=100, lr=0.5)
    uni = task.unigram_ce
    print(f"corpus={len(data)}B n={N} target_two_admitted={target_admitted} unigram_CE={uni:.4f}", flush=True)

    rng = np.random.default_rng(20260606)
    ce_b2_admit, ce_b2_reject = [], []
    n_seen = 0
    t0 = time.time()
    while len(ce_b2_admit) + len(ce_b2_reject) < target_admitted:
        g = sample_gene(rng, N)
        n_seen += 1
        if not cert_two(g, MAX_INPUT_ABS):      # only score genes the exact 2-norm certifier admits
            continue
        ce = task.held_out_ce(g)
        if not np.isfinite(ce):
            continue
        if bound_b2(g) < 1.0:
            ce_b2_admit.append(ce)
        else:
            ce_b2_reject.append(ce)
        scored = len(ce_b2_admit) + len(ce_b2_reject)
        if scored % 50 == 0:
            print(f"  scored {scored}/{target_admitted}  (seen {n_seen}, {time.time()-t0:.0f}s)", flush=True)

    a = np.array(ce_b2_admit)
    r = np.array(ce_b2_reject)

    def stats(x):
        return None if x.size == 0 else {
            "n": int(x.size), "best_ce": round(float(x.min()), 4),
            "median_ce": round(float(np.median(x)), 4), "mean_ce": round(float(x.mean()), 4),
            "frac_below_unigram": round(float(np.mean(x < uni)), 3),
        }

    res = {
        "corpus_bytes": len(data), "n": N, "unigram_ce": round(uni, 4),
        "genes_seen": n_seen, "cert_two_admitted_scored": int(a.size + r.size),
        "b2_admitted": stats(a), "b2_rejected_tail": stats(r),
        "decision": None, "elapsed_s": round(time.time() - t0, 1),
    }
    # Honest go/no-go: does the B2-missed tail reach strictly lower CE than B2-admitted?
    if a.size and r.size:
        tail_better_best = float(r.min()) < float(a.min()) - 1e-4
        tail_better_median = float(np.median(r)) < float(np.median(a)) - 1e-4
        res["decision"] = {
            "tail_best_lower": tail_better_best,
            "tail_median_lower": tail_better_median,
            "sdp_motivated": bool(tail_better_best or tail_better_median),
            "note": ("SDP (PoC-3) motivated: the B2-missed tail reaches lower LM perplexity"
                     if (tail_better_best or tail_better_median) else
                     "SDP NOT motivated by LM perplexity: B2-missed tail is no better than B2-admitted"),
        }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(os.path.join(_HERE, "poc_tail_ce_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("wrote poc_tail_ce_results.json")


if __name__ == "__main__":
    main()
