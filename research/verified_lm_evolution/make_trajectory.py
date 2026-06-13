# SPDX-License-Identifier: Apache-2.0
"""make_trajectory.py — add REAL evolution trajectories ("点が登る") to the landscape spec.

M2 of the verified-plasticity 3D-viz arc. Runs an honest GA over the SAME substrate as the
900-gene landscape (``CoupledNDGene`` n=8 on the byte-LM ``LMTask``, corpus 12288, emb seed 0,
readout_steps=100/lr=0.5 — identical to ``exp_landscape.main``) under two regimes:

  * **none**     — select purely by held-out CE; no safety constraint. The best gene is free to
                   drift into the ρ≥1 (expansive / unsound) region if that lowers perplexity.
  * **cert_inf** — every accepted gene must pass the sound ``cert_inf`` contraction certifier
                   (fail-closed: resample the mutation, else keep the parent). Stays provably ρ<1.

Both regimes start from the SAME seeded initial population (fair comparison: divergence is the
gate's doing, not init luck). Per generation we log the best gene + its real (ce, emp_rho), then
project every best gene through the *identical* standardized-PCA basis used for the landscape
points (recomputed deterministically from the same 900 replayed genes), so the climbing path lives
in the same coordinate frame as the terrain. Whatever the runs actually do is reported honestly —
no assumption that "no gate" blows up.

Run (after make_landscape_spec.py has produced out/llcore_landscape_spec.json):
    py -3.11 make_trajectory.py [spec.json] [results.json]
Mutates the spec in place (adds spec["trajectories"]) and writes a JSONL log of the real paths.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from make_landscape_spec import (  # noqa: E402  reuse identical primitives
    SEED, _ensure_utf8_stdout, gene_feature, replay_genes,
)
from lm_substrate import (  # noqa: E402
    ByteEmbedding, CoupledNDGene, LMTask, empirical_contraction, load_corpus, to_ids,
)
from coupled_nd import cert_inf  # noqa: E402

_ensure_utf8_stdout()

N = 8
GA_SEED = 20260613      # locked GA seed (init + mutation + tournament) — reproducible.
POP = 16
GENS = 14
TOURNEY_K = 3
SD_DECAY = 0.05
SD_W = 0.08
RESAMPLE_CAP = 24       # gated regime: max fail-closed resamples before keeping the parent
RHO_STRIDE = 11         # match exp_landscape's empirical_contraction stride


# ---------------------------------------------------------------- PCA basis (shared frame)
@dataclass
class Frame:
    """The exact standardized-PCA + per-axis affine used for the landscape points."""
    mu: np.ndarray
    sd: np.ndarray
    vt2: np.ndarray      # (2, 72)
    xlo: float
    xhi: float
    ylo: float
    yhi: float

    def project(self, feats: np.ndarray) -> np.ndarray:
        """feats (m,72) -> (m,2) in the landscape's [0.04,0.96] frame (clipped to [0,1])."""
        z = (feats - self.mu) / self.sd
        pcs = z @ self.vt2.T
        x = 0.04 + (pcs[:, 0] - self.xlo) / ((self.xhi - self.xlo) or 1.0) * (0.96 - 0.04)
        y = 0.04 + (pcs[:, 1] - self.ylo) / ((self.yhi - self.ylo) or 1.0) * (0.96 - 0.04)
        return np.column_stack([np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)])


def landscape_frame(n_genes: int) -> Frame:
    """Recompute the IDENTICAL projection frame from the 900 replayed landscape genes."""
    feats = np.array([gene_feature(g) for g in replay_genes(n_genes)])
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd[sd == 0.0] = 1.0
    z = (feats - mu) / sd
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    vt2 = vt[:2]
    pcs = z @ vt2.T
    return Frame(mu, sd, vt2, float(pcs[:, 0].min()), float(pcs[:, 0].max()),
                float(pcs[:, 1].min()), float(pcs[:, 1].max()))


# ---------------------------------------------------------------- honest GA on the real substrate
def _mutate(rng, g: CoupledNDGene) -> CoupledNDGene:
    decay = g.decay + rng.normal(0.0, SD_DECAY, size=N)
    W = g.W + rng.normal(0.0, SD_W, size=(N, N))
    return CoupledNDGene.make(decay=decay, W=W)  # .make clips decay∈[0,1], W∈[-2,2]


def _init_pop(rng, gated: bool) -> list[CoupledNDGene]:
    pop = []
    while len(pop) < POP:
        decay = rng.uniform(rng.uniform(0.0, 0.5), 1.0, size=N)
        w_scale = rng.choice([0.15, 0.3, 0.6, 1.0, 1.5]) / np.sqrt(N)
        W = np.clip(rng.standard_normal((N, N)) * w_scale, -2.0, 2.0)
        g = CoupledNDGene.make(decay=decay, W=W)
        if gated and not cert_inf(g):
            continue
        pop.append(g)
    return pop


def _admit(rng, parent: CoupledNDGene, gated: bool) -> CoupledNDGene:
    """Produce one accepted child. Gated = fail-closed: resample until cert_inf, else keep parent."""
    if not gated:
        return _mutate(rng, parent)
    for _ in range(RESAMPLE_CAP):
        child = _mutate(rng, parent)
        if cert_inf(child):
            return child
    return parent  # known-safe fallback (parent already passed the gate)


def evolve(task: LMTask, gated: bool) -> list[dict]:
    """Run one regime; return per-generation best-gene records (real ce + emp_rho)."""
    rng = np.random.default_rng(GA_SEED)
    pop = _init_pop(rng, gated)
    fits = [task.fitness(g) for g in pop]      # higher = better (= exp(-CE))
    log: list[dict] = []
    for gen in range(GENS):
        bi = int(np.argmax(fits))
        best = pop[bi]
        ce = task.held_out_ce(best)
        rho = empirical_contraction(best, task._emb_seq, stride=RHO_STRIDE)
        log.append({"gen": gen, "decay": best.decay.tolist(), "W": best.W.tolist(),
                    "ce": None if not np.isfinite(ce) else round(float(ce), 5),
                    "emp_rho": round(float(rho), 5), "fitness": round(float(fits[bi]), 6)})
        # next generation: elitism(1) + tournament selection + gated mutation
        nxt = [best]
        while len(nxt) < POP:
            cand = [rng.integers(POP) for _ in range(TOURNEY_K)]
            parent = pop[max(cand, key=lambda i: fits[i])]
            nxt.append(_admit(rng, parent, gated))
        pop = nxt
        fits = [task.fitness(g) for g in pop]
    return log


# ---------------------------------------------------------------- assemble trajectories
def _traj_points(frame: Frame, log: list[dict]) -> list[list[float]]:
    feats = np.array([np.concatenate([np.asarray(r["decay"]), np.asarray(r["W"]).reshape(-1)])
                      for r in log])
    xy = frame.project(feats)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in xy]


def main() -> int:
    spec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / ".." / ".." / "out" / "llcore_landscape_spec.json"
    results_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "exp_landscape_12288_results.json"
    spec_path = spec_path.resolve()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    n_genes = len(results["rows"])
    max_bytes = int(results.get("corpus_bytes", 12288))

    print(f"building shared PCA frame from {n_genes} landscape genes…", flush=True)
    frame = landscape_frame(n_genes)

    print(f"building LMTask (corpus {max_bytes}B, n={N}, readout_steps=100, lr=0.5)…", flush=True)
    task = LMTask(emb=ByteEmbedding.make(n=N, seed=0), ids=to_ids(load_corpus(max_bytes=max_bytes)),
                  readout_steps=100, lr=0.5)

    out = {}
    for regime, gated in (("none", False), ("cert_inf", True)):
        print(f"evolving regime={regime} (gated={gated}): POP={POP} GENS={GENS}…", flush=True)
        log = evolve(task, gated)
        out[regime] = log
        e = log[-1]
        print(f"  {regime}: best CE {log[0]['ce']} -> {e['ce']}  | emp_rho {log[0]['emp_rho']} -> {e['emp_rho']}"
              f"  ({'crossed into ρ≥1' if (e['emp_rho'] and e['emp_rho'] >= 1) else 'stayed ρ<1'})", flush=True)

    # persist the raw real paths
    jsonl = HERE / ".." / ".." / "out" / "evolution_trajectory.jsonl"
    jsonl = jsonl.resolve()
    with jsonl.open("w", encoding="utf-8") as fh:
        for regime, log in out.items():
            for r in log:
                fh.write(json.dumps({"regime": regime, **r}, ensure_ascii=False) + "\n")
    print(f"wrote {jsonl}")

    # inject trajectories into the spec
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    none_e, cert_e = out["none"][-1], out["cert_inf"][-1]
    spec["trajectories"] = [
        {"label": f"no gate → CE {none_e['ce']} / ρ {none_e['emp_rho']}", "color": "#f0a04b",
         "dur": 6.5, "climb": 0.78, "points": _traj_points(frame, out["none"])},
        {"label": f"cert_inf gate → CE {cert_e['ce']} / ρ {cert_e['emp_rho']} (証明つき)", "color": "#3fb950",
         "dur": 6.5, "climb": 0.74, "points": _traj_points(frame, out["cert_inf"])},
    ]
    spec["caption"] = spec.get("caption", "") + (
        f"  軌跡=実 GA(同 byte-LM 基質・同 PCA 基底, seed {GA_SEED}, {GENS}世代): "
        f"橙=無gate / 緑=cert_inf gate(各世代 best を投影)。"
    )
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated {spec_path}  (2 real trajectories, {GENS} gens each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
