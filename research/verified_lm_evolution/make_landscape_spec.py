# SPDX-License-Identifier: Apache-2.0
"""make_landscape_spec.py — build an HONEST fitness/stability-landscape spec from the
real ``exp_landscape`` results, for ``raptor-render-landscape``.

This is the publishable (clean-room, Apache-2.0, zero third-party-viz-code) data adapter
for the "verified plasticity の地形" explainer. It consumes the *real* L3 landscape run
(``exp_landscape_<bytes>_results.json``: 900 sampled CoupledND genes, each with held-out
cross-entropy ``ce`` and empirical contraction ``emp_rho``) and emits the JSON spec that the
existing renderer turns into a viridis terrain + per-gene points coloured by the ρ=1 gate.

Honesty contract (see [[feedback_benchmark_honest_disclosure]] / [[feedback_article_visualization_real_data]]):
  * The 2D placement is NOT invented. The results file never serialised the genes, but they
    were sampled from a *fixed seed* (``exp_landscape.sample_gene`` with rng seed 20260604),
    so we **deterministically replay** the same 900 genes and project the real 72-dim gene
    vector (decay[8] ⊕ W[8×8]) to 2D via standardized PCA. We then **prove** the replay is
    faithful by re-deriving ``classify_region`` for every gene and asserting it matches the
    stored ``region`` for all 900 rows. If the match is not 900/900 we abort rather than emit
    a placeholder dressed up as real data.
  * Terrain height = "bits gained" = ``unigram_ce - ce`` (how much better than the unigram
    baseline; higher = better LM fitness), interpolated (Gaussian RBF) over the gene plane and
    labelled as interpolated. Genes whose CE diverged (``ce is None``) contribute a point but
    no height sample.
  * Point colour = empirical ρ with the gate: green ρ<1 (sound / contracting) vs red ρ≥1
    (unsound / expansive). This is the real per-gene scalar, passed straight through.

Run:
    py -3.11 make_landscape_spec.py [results.json] [out_spec.json]
Defaults: exp_landscape_12288_results.json -> ../../out/llcore_landscape_spec.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Make lm_substrate (which itself wires up coupled_nd) importable.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lm_substrate import CoupledNDGene, classify_region  # noqa: E402

N = 8
SEED = 20260604  # locked seed in exp_landscape.main; do NOT change (replay correctness).


def sample_gene(rng) -> CoupledNDGene:
    """Verbatim copy of exp_landscape.sample_gene so the rng draw sequence is identical."""
    decay = rng.uniform(rng.uniform(0.0, 0.5), 1.0, size=N)
    w_scale = rng.choice([0.15, 0.3, 0.6, 1.0, 1.5]) / np.sqrt(N)
    W = np.clip(rng.standard_normal((N, N)) * w_scale, -2.0, 2.0)
    return CoupledNDGene.make(decay=decay, W=W)


def replay_genes(n_genes: int) -> list[CoupledNDGene]:
    """Reproduce the exact genes used by the landscape run (fixed seed)."""
    rng = np.random.default_rng(SEED)
    return [sample_gene(rng) for _ in range(n_genes)]


def gene_feature(g: CoupledNDGene) -> np.ndarray:
    """72-dim gene vector: decay (8) followed by flattened W (8×8)."""
    return np.concatenate([np.asarray(g.decay, float).reshape(-1),
                           np.asarray(g.W, float).reshape(-1)])


def standardized_pca_2d(feats: np.ndarray) -> np.ndarray:
    """Per-feature z-score then project onto the top-2 principal components (numpy SVD)."""
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd[sd == 0.0] = 1.0
    z = (feats - mu) / sd
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    return z @ vt[:2].T  # (n, 2)


def _unit(col: np.ndarray, lo: float = 0.04, hi: float = 0.96) -> np.ndarray:
    """Normalise a 1D array into [lo, hi]."""
    cmin, cmax = float(col.min()), float(col.max())
    span = (cmax - cmin) or 1.0
    return lo + (col - cmin) / span * (hi - lo)


def rbf_terrain(xs: np.ndarray, ys: np.ndarray, vals: np.ndarray,
                nx: int = 64, ny: int = 36, bw: float = 0.085) -> list[list[float]]:
    """Gaussian-RBF interpolation of scattered (xs, ys, vals) onto an ny×nx grid.

    values[j][i] matches the renderer's indexing (j = y row, i = x col).
    """
    gx = np.linspace(0.0, 1.0, nx)
    gy = np.linspace(0.0, 1.0, ny)
    gxx, gyy = np.meshgrid(gx, gy)              # (ny, nx)
    grid = np.column_stack([gxx.ravel(), gyy.ravel()])   # (ny*nx, 2)
    pts = np.column_stack([xs, ys])             # (m, 2)
    d2 = ((grid[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)  # (G, m)
    w = np.exp(-d2 / (2.0 * bw * bw))
    wsum = w.sum(axis=1)
    wsum[wsum == 0.0] = 1.0
    interp = (w @ vals) / wsum
    return interp.reshape(ny, nx).tolist()


def main() -> int:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "exp_landscape_12288_results.json"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / ".." / ".." / "out" / "llcore_landscape_spec.json"
    out_path = out_path.resolve()

    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = data["rows"]
    n = len(rows)
    unigram_ce = float(data["unigram_ce"])
    corpus_bytes = int(data.get("corpus_bytes", 0))
    print(f"results: {results_path.name}  n_genes={n}  unigram_ce={unigram_ce:.4f}  corpus={corpus_bytes}B")

    # 1) Deterministically replay the genes and PROVE the replay matches the stored regions.
    genes = replay_genes(n)
    mismatches = [i for i, (g, r) in enumerate(zip(genes, rows)) if classify_region(g) != r["region"]]
    if mismatches:
        print(f"FATAL: gene replay region-mismatch on {len(mismatches)}/{n} rows "
              f"(first few: {mismatches[:5]}). Aborting — refusing to emit non-faithful coordinates.",
              file=sys.stderr)
        return 2
    print(f"replay validation: region match {n}/{n}  (genes are faithful — coordinates are real)")

    # 2) Standardized-PCA project the real 72-dim genes to 2D.
    feats = np.array([gene_feature(g) for g in genes])
    pcs = standardized_pca_2d(feats)
    xs = _unit(pcs[:, 0])
    ys = _unit(pcs[:, 1])

    # 3) Real scalars per gene.
    rho = np.array([float(r["emp_rho"]) for r in rows])
    ce = np.array([float(r["ce"]) if r["ce"] is not None else np.nan for r in rows])
    bits = unigram_ce - ce  # height; NaN where CE diverged
    finite = np.isfinite(bits)
    n_safe = int((rho < 1.0).sum())
    n_div = int((rho >= 1.0).sum())
    print(f"rho: min={rho.min():.3f} max={rho.max():.3f}  ρ<1 safe={n_safe}  ρ≥1 divergent={n_div}")
    print(f"bits-gained (finite): n={int(finite.sum())}  "
          f"min={np.nanmin(bits):.3f} max={np.nanmax(bits):.3f}")

    # 4) Terrain from finite-CE genes only (honest: divergent genes have no fitness height).
    terrain = rbf_terrain(xs[finite], ys[finite], bits[finite])

    # 5) Individuals = all 900 real genes, coloured by the real ρ gate.
    individuals = [{"x": round(float(xs[i]), 4), "y": round(float(ys[i]), 4),
                    "rho": round(float(rho[i]), 4), "r": 2.6} for i in range(n)]

    spec = {
        "title": "検証付き可塑性の地形 — 実 byte-LM 遺伝子 900 個 (CoupledND, n=8)",
        "size": [960, 540],
        "grid": {"values": terrain},
        "individuals": individuals,
        "trajectories": [],
        "caption": (
            f"x,y = 実 72 次元遺伝子(decay⊕W)の標準化 PCA 投影・seed {SEED} で再現(region 一致 {n}/{n})。"
            f"地形=獲得 bits(unigram_ce−CE, 高い=低 perplexity, RBF 補間)。"
            f"点色=実測 ρ: 緑 ρ<1 健全 {n_safe} / 赤 ρ≥1 発散 {n_div}。corpus {corpus_bytes}B。"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}  ({n} individuals, {len(terrain)}×{len(terrain[0])} terrain grid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
