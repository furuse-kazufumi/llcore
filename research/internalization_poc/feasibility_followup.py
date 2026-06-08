# SPDX-License-Identifier: Apache-2.0
"""feasibility 追補 II — 敵対レビュー (16 confirmed) の data 反映.

レビューが正しく突いた over-claim を「推測でなく追加 probe」で決着させる:
  (A) gated margin probe: plain logsumexp は relu 勾配が threshold 非依存で全 margin 同一だった
      (退化)。gated は true infnorm を gate するので margin が equilibrium を動かす仮説を**実証/反証**。
      residence (固定帯 ≥0.98) が margin 依存になるか + admit-core-active が 0 か。
  (B) gated silence を配備 θ=0.85 (margin=0.15) で再検証 + pull-back latency を per-seed で記録。
  (C) 死削減の帰属 ablation: {NONE / max-ENDO / logsumexp-ENDO / gated-ENDO} @ n=32 同条件で
      死率比較 → 削減は logsumexp の coverage 由来か gate/margin 由来かを切り分け。
CPU, n=32, 3 seeds。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import torch

import feasibility_internalization as M
H = M.H
DT = M.DT


def surrogate_scalar(kind, decay, W, threshold, tau, k):
    """kind ∈ {max, logsumexp, gated, topk}. 返り値 = relu hinge スカラー。"""
    R = M.rows_torch(decay, W)
    true_max = R.max().detach()
    if kind == "max":
        agg = R.max()
    elif kind == "topk":
        kk = min(max(1, k), R.numel()); agg = torch.topk(R.reshape(-1), kk).values.mean()
    elif kind == "logsumexp":
        agg = torch.logsumexp(tau * R.reshape(-1), 0) / tau
    elif kind == "gated":
        gate = (true_max >= threshold).to(R.dtype)
        return gate * torch.relu(torch.logsumexp(tau * R.reshape(-1), 0) / tau - threshold)
    else:
        raise ValueError(kind)
    return torch.relu(agg - threshold)


def model_train(n, seed, cfg, data, lam, margin, kind, tau, k, measure_m=5):
    tr, va, vocab = data
    threshold = 1.0 - margin
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    m = H.GatedRecurrentLM(vocab, n, cfg["layers"], cfg["d"])
    for _ in range(50):
        d_, W_ = m.core_np(0)
        if H.cert_inf(d_, W_):
            break
        with torch.no_grad():
            m.raw_W[0].mul_(0.5)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    deaths = res_fixed = res_mdep = core_active = core_steps = n_meas = 0
    infs = []
    for it in range(cfg["grad_steps"]):
        x, y = H.batches(tr, cfg["T"], cfg["B"], rng)
        opt.zero_grad()
        ce = torch.nn.functional.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
        if kind == "none" or lam == 0.0:
            loss = ce
        else:
            decay, W = m.core(0)
            loss = ce + lam * surrogate_scalar(kind, decay.double(), W.double(), threshold, tau, k)
        loss.backward(); opt.step()
        if (it + 1) % measure_m == 0 or it == cfg["grad_steps"] - 1:
            n_meas += 1
            d_, W_ = m.core_np(0)
            inf = H.infnorm_sup(d_, W_, H.t_min_per_coord(d_, W_))
            rho = H.empirical_rho(d_, W_, n_samples=200, seed=seed)
            infs.append(inf)
            if rho >= 1.0:
                deaths += 1
            if inf >= 0.98:                    # 固定近境界帯 (true boundary 近傍)
                res_fixed += 1
            if inf >= threshold:               # margin 依存帯 [1-margin, 1+)
                res_mdep += 1
            if kind == "gated" and inf < threshold:    # 真 admit 中核
                core_steps += 1
                dd, WW = m.core(0)
                cc = surrogate_scalar("gated", dd.double(), WW.double(), threshold, tau, k)
                if float(cc) > 0:
                    core_active += 1
    ce_final = H.eval_ce(m, va, cfg["T"], cfg["B"], cfg["eval_batches"], np.random.default_rng(seed + 7))
    return {"n": n, "seed": seed, "kind": kind, "margin": margin, "lam": lam,
            "final_ce": float(ce_final), "death_rate": deaths / n_meas,
            "residence_fixed": res_fixed / n_meas, "residence_mdep": res_mdep / n_meas,
            "admit_core_active": core_active, "admit_core_steps": core_steps,
            "mean_inf": float(np.mean(infs))}


def pull_back_per_seed(kind, n, seeds, tau, k):
    """gated/plain logsumexp の pull-back latency を per-seed で記録。"""
    push_to = 1.0 - 0.05
    out = []
    for seed in seeds:
        raw_W, raw_decay = M.make_scaled_violating(n, seed, 1.95)
        opt = torch.optim.Adam([raw_W, raw_decay], lr=3e-3)
        steps = 2000
        for it in range(2000):
            decay, W = M.cores_from_raw(raw_W, raw_decay)
            with torch.no_grad():
                if float(M.rows_torch(decay, W).max()) < 1.0:
                    steps = it; break
            loss = surrogate_scalar("gated" if kind == "gated" else "logsumexp",
                                    decay, W, push_to, tau, k)
            opt.zero_grad(); loss.backward(); opt.step()
        out.append(steps)
    return out


def main():
    tau = M.BINDING["logsumexp_tau"]
    data = H.make_data(30000, False)
    cfg = dict(layers=1, d=64, T=48, B=12, lr=3e-3, grad_steps=120, eval_batches=3)
    n = 32; k = math.ceil(0.25 * 2 * n); seeds = [0, 1, 2]
    out = {}

    # (A) gated margin probe
    print("[A] gated margin probe (residence が margin 依存になるか)")
    A = []
    for margin in (0.0, 0.05, 0.10, 0.15):
        for seed in seeds:
            A.append(model_train(n, seed, cfg, data, 0.1, margin, "gated", tau, k))
        rf = np.mean([x["residence_fixed"] for x in A if x["margin"] == margin])
        rm = np.mean([x["residence_mdep"] for x in A if x["margin"] == margin])
        ca = sum(x["admit_core_active"] for x in A if x["margin"] == margin)
        cs = sum(x["admit_core_steps"] for x in A if x["margin"] == margin)
        dr = np.mean([x["death_rate"] for x in A if x["margin"] == margin])
        mi = np.mean([x["mean_inf"] for x in A if x["margin"] == margin])
        print(f"   margin={margin:.2f} res_fixed(>=.98)={rf:.3f} res_mdep(>=1-m)={rm:.3f} "
              f"admit-core-active={ca}/{cs} death={dr:.3f} mean_inf={mi:.3f}")
    out["gated_margin_probe"] = A

    # (B) gated silence at deployed theta=0.85 + per-seed latency
    print("\n[B] gated silence @ deployed θ=0.85 (margin=0.15) + per-seed latency")
    b_runs = [x for x in A if x["margin"] == 0.15]
    ca = sum(x["admit_core_active"] for x in b_runs); cs = sum(x["admit_core_steps"] for x in b_runs)
    print(f"   θ=0.85 admit-core-active (should be 0) = {ca}/{cs}")
    lat_g = pull_back_per_seed("gated", n, [0, 1, 2, 3], tau, k)
    lat_p = pull_back_per_seed("plain", n, [0, 1, 2, 3], tau, k)
    print(f"   pull-back latency per-seed gated={lat_g} plain={lat_p}")
    out["gated_theta085_active"] = {"active": ca, "steps": cs}
    out["latency_per_seed"] = {"gated": lat_g, "plain": lat_p}

    # (C) 死削減の帰属 ablation
    print("\n[C] death-reduction attribution: NONE vs max-ENDO vs logsumexp-ENDO vs gated-ENDO (margin=0.05)")
    C = []
    for kind in ("none", "max", "logsumexp", "gated"):
        for seed in seeds:
            C.append(model_train(n, seed, cfg, data, 0.0 if kind == "none" else 0.1, 0.05, kind, tau, k))
        dr = np.mean([x["death_rate"] for x in C if x["kind"] == kind])
        mi = np.mean([x["mean_inf"] for x in C if x["kind"] == kind])
        ce = np.mean([x["final_ce"] for x in C if x["kind"] == kind])
        print(f"   {kind:10s} death={dr:.3f} mean_inf={mi:.3f} ce={ce:.3f}")
    out["death_attribution"] = C

    json.dump(out, open(Path(__file__).resolve().parent / "results_followup.json", "w"), indent=1)
    print("\n[saved] results_followup.json")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
