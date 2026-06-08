# SPDX-License-Identifier: Apache-2.0
"""feasibility 追補: drift-init (実膨張域) での勾配被覆率・飽和 (設計 §6f 忠実版).

本体 (e)(f) は scaled-random violating init で測った (飽和=0)。設計 §6f が要求するのは
「膨張 step での raw_W 有効勾配被覆率 + 飽和 fraction」= CE drift が自然に作る膨張状態
(レビュー「94.5% 勾配≈0」)。本スクリプトは NONE CE で実モデルを infnorm_sup≥1.5 まで drift
させ、その drift-core で max/logsumexp/topk/gated の行被覆率・飽和を測る。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

import feasibility_internalization as M
H = M.H


def drift_to_inflation(n, seed, target=1.5, max_steps=400):
    data = H.make_data(30000, False)
    tr, va, vocab = data
    cfg = dict(layers=1, d=64, T=48, B=12, lr=3e-3)
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    m = H.GatedRecurrentLM(vocab, n, 1, cfg["d"])
    for _ in range(50):
        d_, W_ = m.core_np(0)
        if H.cert_inf(d_, W_):
            break
        with torch.no_grad():
            m.raw_W[0].mul_(0.5)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    reached = 0.0
    for it in range(max_steps):
        x, y = H.batches(tr, cfg["T"], cfg["B"], rng)
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward(); opt.step()
        d_, W_ = m.core_np(0)
        reached = H.infnorm_sup(d_, W_, H.t_min_per_coord(d_, W_))
        if reached >= target:
            break
    return m, reached


def coverage_on_core(m, variant, tau, k):
    thr = 1.0
    m.zero_grad()
    decay, W = m.core(0)                                  # float32, differentiable via raw_W
    R = M.rows_torch(decay.double(), W.double())
    true_max = R.max().detach()
    if variant == "max":
        agg = R.max()
    elif variant == "logsumexp":
        agg = torch.logsumexp(tau * R.reshape(-1), 0) / tau
    elif variant == "topk":
        kk = min(max(1, k), R.numel())
        agg = torch.topk(R.reshape(-1), kk).values.mean()
    elif variant == "gated":
        gate = (true_max >= thr).to(R.dtype)
        agg = gate * (torch.logsumexp(tau * R.reshape(-1), 0) / tau)
    loss = torch.relu(agg - thr)
    loss.backward()
    g = m.raw_W[0].grad.detach()
    row_norms = g.abs().sum(dim=1)
    eps = 1e-12
    cov = float((row_norms > eps).sum()) / g.shape[0]
    with torch.no_grad():
        deriv = 2.0 * (1.0 - torch.tanh(m.raw_W[0]) ** 2)
        touched = row_norms > eps
        sat = (deriv[touched] < 0.05).float().mean().item() if touched.any() else float("nan")
    m.zero_grad()
    return cov, sat


def main():
    tau = M.BINDING["logsumexp_tau"]
    out = {}
    for n in (8, 32):
        k = int(np.ceil(0.25 * 2 * n))
        rows = {}
        for seed in range(3):
            m, inf = drift_to_inflation(n, seed)
            for variant in ("max", "logsumexp", "topk", "gated"):
                cov, sat = coverage_on_core(m, variant, tau, k)
                rows.setdefault(variant, []).append((inf, cov, sat))
        out[n] = {v: {"mean_inf0": float(np.mean([r[0] for r in rows[v]])),
                      "mean_cov": float(np.mean([r[1] for r in rows[v]])),
                      "mean_sat": float(np.nanmean([r[2] for r in rows[v]]))}
                  for v in rows}
        print(f"n={n}: drift inf0≈{out[n]['max']['mean_inf0']:.2f}")
        for v in ("max", "logsumexp", "topk", "gated"):
            print(f"   {v:10s} row_coverage={out[n][v]['mean_cov']:.3f} "
                  f"saturation_touched={out[n][v]['mean_sat']:.3f}")
    json.dump(out, open(Path(__file__).resolve().parent / "results_drift_coverage.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
