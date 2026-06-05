# SPDX-License-Identifier: Apache-2.0
"""L3 (payoff) — gated evolution of the recurrent LM core: does a better verifier lower perplexity?

Runs the UNCHANGED evolvable_core.evolve() with each contraction-verifier backend (none / inf_norm /
two_norm / sdp) as the fail-closed admission gate, optimizing real held-out byte-LM likelihood.
Paired seeds (common random numbers) across gates. The arc's signature claim, on real LM loss:
if a less-conservative sound gate (two/sdp) reaches lower held-out CE than the conservative inf-norm
gate, a better verifier unlocks more reachable LM fitness. --null shuffles the corpus (destroys
sequential structure) => memory useless => all gates should tie (honest null control).

Run: py -3.11 exp_gated.py [seeds] [--null]
"""
from __future__ import annotations

import io
import json
import sys
import time

import numpy as np


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 for safe printing on cp932 consoles."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8")
            except (ValueError, OSError):
                pass
        elif hasattr(stream, "buffer"):
            setattr(sys, stream_name, io.TextIOWrapper(stream.buffer, encoding="utf-8"))


_ensure_utf8_stdout()

# evolvable_core lives in the sibling sdp_gate dir (added to sys.path by lm_substrate import)
import lm_substrate as L
from coupled_nd import CoupledNDGeneCodec, make_nd_verifier
from evolvable_core import EvolveConfig, evolve

N = 8
GATES = ("none", "inf_norm", "two_norm", "sdp")


def build_task(seeds_shuffle: bool, max_bytes: int) -> L.LMTask:
    data = L.load_corpus(max_bytes=max_bytes)
    ids = L.to_ids(data)
    if seeds_shuffle:
        rng = np.random.default_rng(999)
        ids = ids.copy()
        rng.shuffle(ids)
    emb = L.ByteEmbedding.make(n=N, seed=0)
    return L.LMTask(emb=emb, ids=ids, readout_steps=60, lr=0.5)


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
    null = "--null" in sys.argv
    max_bytes = 8192

    task = build_task(null, max_bytes)
    codec = CoupledNDGeneCodec(N)
    cfg = EvolveConfig(pop_size=12, n_generations=10, mutation_sigma=0.15, elitism=1, resample_cap=40)
    uni = task.unigram_ce
    print(f"{'NULL ' if null else ''}corpus={len(task.ids)}B n={N} seeds={n_seeds} "
          f"pop={cfg.pop_size} gens={cfg.n_generations} unigram_CE={uni:.4f}", flush=True)

    results: dict[str, list[float]] = {g: [] for g in GATES}
    region_of_winner: dict[str, list[str]] = {g: [] for g in GATES}
    t0 = time.time()
    for seed in range(n_seeds):
        for gate in GATES:
            rng = np.random.default_rng(1000 + seed)  # CRN: same seed across gates
            res = evolve(codec, task, make_nd_verifier(gate), cfg, rng=rng, gate_initial=(gate != "none"))
            results[gate].append(res.best_fitness)
            region_of_winner[gate].append(L.classify_region(res.best_gene))
        print(f"  seed {seed}: " + " ".join(f"{g}={results[g][-1]:.4f}" for g in GATES)
              + f"  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'gate':10s} {'mean_fit':>9s} {'mean_CE':>8s} {'best_fit':>9s}")
    summ = {}
    for g in GATES:
        arr = np.array(results[g])
        mean_fit = float(arr.mean())
        mean_ce = float(-np.log(np.clip(arr, 1e-12, None)).mean())
        print(f"{g:10s} {mean_fit:9.4f} {mean_ce:8.4f} {float(arr.max()):9.4f}")
        summ[g] = {"fits": results[g], "mean_fit": mean_fit,
                   "winner_regions": region_of_winner[g]}

    # paired sdp/two vs inf (one-sided: better gate >= inf)
    def paired(a, b):
        da = np.array(results[a]) - np.array(results[b])
        psd = float(np.mean(da > 0))  # fraction a>b
        return {"mean_delta": float(da.mean()), "frac_a_gt_b": psd, "deltas": da.round(5).tolist()}

    print("\n--- L3 payoff (paired, vs inf_norm; positive delta = better gate lowers perplexity) ---")
    for g in ("two_norm", "sdp"):
        p = paired(g, "inf_norm")
        print(f"  {g} vs inf_norm: mean_delta_fit={p['mean_delta']:+.4f}  frac({g}>inf)={p['frac_a_gt_b']:.2f}")
        summ[f"{g}_vs_inf"] = p

    out = {"null": null, "corpus_bytes": int(len(task.ids)), "n": N, "n_seeds": n_seeds,
           "unigram_ce": uni, "config": cfg.__dict__, "summary": summ,
           "elapsed_s": round(time.time() - t0, 1)}
    fname = "exp_gated_null_results.json" if null else "exp_gated_results.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {fname}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
