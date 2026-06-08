# SPDX-License-Identifier: Apache-2.0
"""feasibility 追補: gated-logsumexp surrogate の検証 (C14 admit-core grad-zero 復元).

feasibility 本体で判明: 素の logsumexp は max より offset(~0.1-0.28) 高いため、true infnorm が
admit 中核 (< 1-margin) にあっても surrogate がアクティブ (admit_core_dot ≠ 0) = C14 違反 = 安全域でも
CE に課税。修正案 = **detached true-infnorm gate**:
    loss = 1[ infnorm_sup(detach) >= threshold ] · relu( logsumexp(rows)/... - threshold )
ゲートが true max を見るので admit 中核で厳密に 0 (grad 0)。アクティブ時は全行へ勾配 (coverage 維持)。

本スクリプトは (1) admit 中核 silence (2) pull-back latency が素 logsumexp と同等 (3) 短 ENDO_GRAD で
admit 中核アクティブ step がゼロ化 を検証し、verdict の修正案を「実コードで検証済み」にする。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import torch

import feasibility_internalization as M   # rows_torch / make_scaled_violating / cores_from_raw / H / DT

DT = M.DT
H = M.H


def gated_logsumexp_loss(decay, W, threshold, tau):
    R = M.rows_torch(decay, W)
    true_max = R.max().detach()
    gate = (true_max >= threshold).to(R.dtype)
    soft = torch.logsumexp(tau * R.reshape(-1), dim=0) / tau
    return gate * torch.relu(soft - threshold)


def check_silence(n_list, tau):
    """admit 中核 (true infnorm < threshold) で gated loss=0, grad=0 を厳密検証。"""
    print("\n[1] admit-core silence (gated logsumexp)")
    ok_all = True
    for n in n_list:
        # 確実に admit 中核 (true infnorm < 0.85) を作る: scale を bisection で詰める
        g = torch.Generator().manual_seed(n)
        base_W = torch.randn(n, n, generator=g, dtype=DT)
        raw_decay_t = torch.randn(n, generator=g, dtype=DT) * 0.5 + 1.0
        scale = 0.5
        for _ in range(60):
            decay = torch.sigmoid(raw_decay_t); W = 2.0 * torch.tanh(base_W * scale)
            if float(M.rows_torch(decay, W).max()) < 0.85:
                break
            scale *= 0.6
        raw_W = (base_W * scale).clone().requires_grad_(True)
        raw_decay = raw_decay_t.clone().requires_grad_(True)
        decay, W = M.cores_from_raw(raw_W, raw_decay)
        true_inf = float(M.rows_torch(decay, W).max())
        thr = 0.95
        loss = gated_logsumexp_loss(decay, W, thr, tau)
        loss.backward()
        gnorm = float(raw_W.grad.abs().sum())
        silent = (true_inf < thr and float(loss) == 0.0 and gnorm == 0.0)
        ok_all = ok_all and silent
        print(f"   n={n:4d} true_inf={true_inf:.3f} (<{thr}) loss={float(loss):.2e} |grad|={gnorm:.2e} "
              f"({'SILENT OK' if silent else 'NOT SILENT'})")
    return ok_all


def gated_latency(variant, n, seed, target, admit_thr, lr, cap, tau, push_margin):
    push_to = admit_thr - push_margin
    raw_W, raw_decay = M.make_scaled_violating(n, seed, target)
    opt = torch.optim.Adam([raw_W, raw_decay], lr=lr)
    for it in range(cap):
        decay, W = M.cores_from_raw(raw_W, raw_decay)
        with torch.no_grad():
            cur = float(M.rows_torch(decay, W).max())
        if cur < admit_thr:
            return it
        if variant == "gated":
            loss = gated_logsumexp_loss(decay, W, push_to, tau)
        else:
            R = M.rows_torch(decay, W)
            loss = torch.relu(torch.logsumexp(tau * R.reshape(-1), 0) / tau - push_to)
        opt.zero_grad(); loss.backward(); opt.step()
    return cap


def check_latency(n_list, tau):
    print("\n[2] pull-back latency: gated vs plain logsumexp (median over 4 seeds)")
    res = {}
    for n in n_list:
        g_lat = [gated_latency("gated", n, s, 1.95, 1.0, 3e-3, 2000, tau, 0.05) for s in range(4)]
        p_lat = [gated_latency("plain", n, s, 1.95, 1.0, 3e-3, 2000, tau, 0.05) for s in range(4)]
        res[n] = (float(np.median(g_lat)), float(np.median(p_lat)))
        print(f"   n={n:4d} gated median={int(np.median(g_lat)):4d}  plain median={int(np.median(p_lat)):4d}")
    return res


def check_model_silence(n, seeds, tau, lam):
    """短 ENDO_GRAD (gated) で admit 中核アクティブ step がゼロ化するか + 死回避維持。"""
    print("\n[3] gated ENDO_GRAD: admit-core active steps -> 0 + death reduction maintained")
    data = H.make_data(30000, False)
    tr, va, vocab = data
    cfg = dict(layers=1, d=64, T=48, B=12, lr=3e-3, grad_steps=120, eval_batches=3)
    out = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
        m = H.GatedRecurrentLM(vocab, n, cfg["layers"], cfg["d"])
        for li in range(cfg["layers"]):
            for _ in range(50):
                d_, W_ = m.core_np(li)
                if H.cert_inf(d_, W_):
                    break
                with torch.no_grad():
                    m.raw_W[li].mul_(0.5)
        opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
        thr = 1.0
        deaths = 0; n_meas = 0; core_active = 0; core_steps = 0
        for it in range(cfg["grad_steps"]):
            x, y = H.batches(tr, cfg["T"], cfg["B"], rng)
            opt.zero_grad()
            ce = torch.nn.functional.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
            decay, W = m.core(0)
            cert = gated_logsumexp_loss(decay.double(), W.double(), thr, tau)
            (ce + lam * cert).backward(); opt.step()
            if (it + 1) % 5 == 0 or it == cfg["grad_steps"] - 1:
                n_meas += 1
                d_, W_ = m.core_np(0)
                inf = H.infnorm_sup(d_, W_, H.t_min_per_coord(d_, W_))
                rho = H.empirical_rho(d_, W_, n_samples=200, seed=seed)
                if rho >= 1.0:
                    deaths += 1
                if inf < thr:                          # true admit core
                    core_steps += 1
                    dd, WW = m.core(0)
                    cc = gated_logsumexp_loss(dd.double(), WW.double(), thr, tau)
                    if float(cc) > 0:                  # surrogate active in true admit core?
                        core_active += 1
        out.append((deaths / n_meas, core_active, core_steps))
        print(f"   seed={seed} death_rate={deaths/n_meas:.3f} admit-core-active={core_active}/{core_steps}")
    return out


def main():
    tau = M.BINDING["logsumexp_tau"]
    sil = check_silence([8, 32, 128], tau)
    lat = check_latency([32, 128, 256], tau)
    mod = check_model_silence(32, [0, 1, 2, 3], tau, 0.1)
    summary = {
        "silence_ok": bool(sil),
        "latency_gated_vs_plain": {str(k): v for k, v in lat.items()},
        "model_gated": [{"death_rate": d, "core_active": a, "core_steps": s} for (d, a, s) in mod],
    }
    json.dump(summary, open(Path(__file__).resolve().parent / "results_gated_check.json", "w"), indent=1)
    print("\n=== gated-logsumexp check ===")
    print(f"  admit-core silence: {'OK (C14 restored)' if sil else 'FAILED'}")
    core_active_total = sum(a for (_, a, _) in mod)
    print(f"  model admit-core active steps (should be 0): {core_active_total}")
    print(f"  death_rate (gated, n=32): {np.mean([d for (d,_,_) in mod]):.3f} (NONE feasibility=0.188)")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
