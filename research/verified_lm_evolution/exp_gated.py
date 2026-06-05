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
    fname = "exp_gated_null_results.json" if null else "exp_gated_results.json"
    t0 = time.time()

    def _paired(a, b, n_done):
        # one-sided: better gate >= inf. CRN-paired across gates (same seed).
        da = np.array(results[a][:n_done]) - np.array(results[b][:n_done])
        return {"mean_delta": float(da.mean()), "frac_a_gt_b": float(np.mean(da > 0)),
                "deltas": da.round(5).tolist()}

    def _build_out(n_done):
        summ = {}
        for g in GATES:
            arr = np.array(results[g][:n_done])
            summ[g] = {"fits": list(results[g][:n_done]), "mean_fit": float(arr.mean()),
                       "mean_ce": float(-np.log(np.clip(arr, 1e-12, None)).mean()),
                       "best_fit": float(arr.max()), "winner_regions": list(region_of_winner[g][:n_done])}
        for g in ("two_norm", "sdp"):
            summ[f"{g}_vs_inf"] = _paired(g, "inf_norm", n_done)
        return {"null": null, "corpus_bytes": int(len(task.ids)), "n": N,
                "n_seeds_requested": n_seeds, "n_seeds_done": n_done,
                "unigram_ce": uni, "config": cfg.__dict__, "summary": summ,
                "elapsed_s": round(time.time() - t0, 1)}

    def _checkpoint(n_done):
        # incremental write: a partial-but-valid JSON survives an early stop (kill-safe).
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(_build_out(n_done), f, indent=2)

    for seed in range(n_seeds):
        for gate in GATES:
            rng = np.random.default_rng(1000 + seed)  # CRN: same seed across gates
            res = evolve(codec, task, make_nd_verifier(gate), cfg, rng=rng, gate_initial=(gate != "none"))
            results[gate].append(res.best_fitness)
            region_of_winner[gate].append(L.classify_region(res.best_gene))
        print(f"  seed {seed}: " + " ".join(f"{g}={results[g][-1]:.4f}" for g in GATES)
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        _checkpoint(seed + 1)  # partial JSON saved after every seed

    final = _build_out(n_seeds)
    print(f"\n{'gate':10s} {'mean_fit':>9s} {'mean_CE':>8s} {'best_fit':>9s}")
    for g in GATES:
        s = final["summary"][g]
        print(f"{g:10s} {s['mean_fit']:9.4f} {s['mean_ce']:8.4f} {s['best_fit']:9.4f}")
    print("\n--- L3 payoff (paired, vs inf_norm; positive delta = better gate lowers perplexity) ---")
    for g in ("two_norm", "sdp"):
        p = final["summary"][f"{g}_vs_inf"]
        print(f"  {g} vs inf_norm: mean_delta_fit={p['mean_delta']:+.4f}  frac({g}>inf)={p['frac_a_gt_b']:.2f}")
    print(f"\nsaved {fname}  ({final['elapsed_s']}s)")


if __name__ == "__main__":
    main()
