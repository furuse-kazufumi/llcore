# SPDX-License-Identifier: Apache-2.0
"""CPU feasibility for "検証シグナル勾配内蔵" (gradient-embedded supervision) — go/no-go 前置.

設計 = HD1_INTERNALIZATION_DESIGN.md v2 §6 (敵対レビュー 18 agents 反映後)。
目的: 本走 (GPU) に進む前に **本走形 (surrogate 変種 / λ / margin / threshold) を
結果取得前に機械的に確定**し、go/no-go を判定する。

probes (設計 §6):
  (e) pull-back latency  [GO/NO-GO]  — violating init (infnorm_sup≈1.95) → admissible (infnorm_sup<1)
        復帰 step 数を 3 surrogate 変種 (max / logsumexp / topk) × n で表化。
  (f) gradient coverage  — 膨張 step での raw_W 有効勾配「行被覆率」(非ゼロ行/n) と飽和 fraction。
  (g) margin-matched 真ρ — HARNESS の admit 境界 (infnorm_sup≈1) の真 ρ 分布 → MATCHED threshold 決定材料。
  (h) tautology probe    — ENDO_GRAD が「窓内で contract_death を 1 度も踏まない」seed 比率。
  (c') admit-core grad-zero — margin>0 で admit 中核 (infnorm_sup<1-margin) の surrogate grad≈0 数値検証。
  (a) F 条項 active      — NONE 窓契約死率 ≥ 5% (死 regime が立っているか)。
  (b) λ/margin 選定規則  — §4 の結果非依存の閉じた式を計測データに適用。

実行系統:
  - analytic (e/f/g): CE 不要 → n∈{8,32,64,128,256} 走破 (大 n スケール劣化を見る)。
  - model-based (h/a/c'-train/margin残留): CE 訓練が要る → CPU 予算で n∈{8,32}。
  - 妥当性: n∈{8,32} で「drift-init (CE で自然膨張)」 vs 「scaled-random violating init」の latency/coverage
    を突合し、cheap proxy が drift と桁で乖離しないことを確認 → 大 n は scaled-random トレンドで go/no-go
    (この外挿は honest に limitation として記録)。

CPU only. resumable (checkpoint after every probe block)。

実行:  py -3.11 research/internalization_poc/feasibility_internalization.py
"""
from __future__ import annotations

import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")   # cp932 console 対策 (既知パターン)
except Exception:
    pass

# ----- substrate (numpy 正本: infnorm_sup / empirical_rho / cert_inf / model) ----- #
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "highdim_evolution"))
import hd1_highdim_evo as H          # noqa: E402
from cert_surrogate import infnorm_sup_torch as _ref_inf_torch   # noqa: E402 (cross-check)

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
DEVICE = "cpu"
DT = torch.float64                   # surrogate 最適化は float64 (Stage-B の float32 飽和罠回避)

_T0 = time.time()
def _log(tag, msg): print(f"[{time.time()-_T0:7.1f}s][{tag}] {msg}", flush=True)
def _now(): return datetime.datetime.now().isoformat(timespec="seconds")


# ============================================================================ #
#  BINDING (結果取得前に固定): go/no-go 閾値・選定規則の数値                      #
# ============================================================================ #
BINDING = {
    "violating_target_infnorm": 1.95,   # NONE drift 終端 ρ_hp≈1.969 帯に合わせた違反 init
    "admit_threshold": 1.0,             # cert_inf 境界 (infnorm_sup<1)
    "pullback_margin": 0.05,            # pull-back の押し先 = 1 - margin (境界ちょうどは hinge grad=0 で停滞)
    "main_grad_steps": 400,             # 本走 (GPU) の grad_steps 予算
    "latency_cap": 2000,                # pull-back を打ち切る step 上限
    "lr": 3e-3,                         # 本走と同じ Adam lr
    "logsumexp_tau": 10.0,              # logsumexp の温度 (soft-max → max as tau→∞)
    "topk_frac": 0.25,                  # top-k の k = ceil(topk_frac * 2n)
    # --- go/no-go 規則 (§6e/§6f) ---
    "latency_frac_of_budget": 0.5,      # GO: latency ≤ FRAC·main_grad_steps (=200 step) を全 main n で
    "main_ns": [64, 128, 256],          # go/no-go を課す本走 n
    "coverage_min_at_maxn": 0.25,       # GO: 行被覆率 ≥ 0.25 を最大 n(256) で
    "variant_preference": ["max", "logsumexp", "topk"],  # GO なら最も単純な変種を採る
    # --- margin 選定規則 (§4): 近境界帯滞在率 < X% になる最小 margin ---
    "margin_band": 0.02,                # 近境界帯 = infnorm_sup ≥ 1 - band (=0.98)
    "margin_residence_X": 0.20,         # 滞在率上限 X
    "margin_grid": [0.0, 0.05, 0.10, 0.15],
    # --- λ 選定規則 (§4): |λ·∂surrogate| と |∂CE| が同じ decade ---
    "lambda_decades": True,
}

# probe scope
N_ANALYTIC = [8, 32, 64, 128, 256]
N_MODEL = [8, 32]
SEEDS_ANALYTIC = [0, 1, 2, 3]
SEEDS_MODEL = [0, 1, 2, 3]
RPATH = str(_HERE / "results_internalization_feasibility.json")


# ============================================================================ #
#  surrogate: 行テンソル (2, n) と 3 変種の集約                                  #
# ============================================================================ #
def rows_torch(decay: torch.Tensor, W: torch.Tensor, max_input_abs: float = 1.0) -> torch.Tensor:
    """infnorm_sup の (2, n) 行寄与テンソル。 .max() == infnorm_sup (numpy 正本)。

    diag_i = |decay_i + (1-decay_i)·t_i·W_ii|,  row_i = diag_i + (1-decay_i)·t_i·off_i
    box 端点 t ∈ {t_lo, 1.0} の 2 通り → (2, n)。
    """
    decay = decay.clamp(0.0, 1.0)
    W = W.clamp(-2.0, 2.0)
    absW = W.abs()
    diag_W = torch.diagonal(W)
    M = absW.sum(dim=1) + max_input_abs
    t_lo = 1.0 - torch.tanh(M) ** 2
    off = absW.sum(dim=1) - diag_W.abs()
    out = []
    for ti in (t_lo, torch.ones_like(t_lo)):
        diag = (decay + (1.0 - decay) * ti * diag_W).abs()
        out.append(diag + (1.0 - decay) * ti * off)
    return torch.stack(out)            # (2, n)


def agg(variant: str, R: torch.Tensor, tau: float, k: int) -> torch.Tensor:
    """(2, n) 行寄与を 1 スカラーへ集約 (= surrogate が押す量)。"""
    flat = R.reshape(-1)
    if variant == "max":
        return flat.max()
    if variant == "logsumexp":
        # (1/tau)·logsumexp(tau·flat) ≥ max → max より保守 (sound: <1 ⟹ infnorm_sup<1)
        return torch.logsumexp(tau * flat, dim=0) / tau
    if variant == "topk":
        kk = min(max(1, k), flat.numel())
        return torch.topk(flat, kk).values.mean()    # ≤ max (sound 保証なし; 死は emp_rho で判定)
    raise ValueError(variant)


def surrogate_loss(variant, decay, W, threshold, tau, k):
    """片側 hinge: 集約量 > threshold で押し返し、内側で 0。"""
    return torch.relu(agg(variant, rows_torch(decay, W), tau, k) - threshold)


def true_infnorm(decay: torch.Tensor, W: torch.Tensor) -> float:
    """真の infnorm_sup (= rows.max(), numpy cert と一致)。判定・進捗計測用 (検出は detach)。"""
    with torch.no_grad():
        return float(rows_torch(decay, W).max())


# ---- self-test: rows.max() == numpy infnorm_sup == cert_surrogate.infnorm_sup_torch ---- #
def _self_test():
    rng = np.random.default_rng(0)
    me = mr = 0.0
    for n in (8, 16, 32):
        for _ in range(5):
            decay = rng.uniform(0.1, 0.95, n)
            W = rng.normal(0, 0.4, (n, n))
            np_val = H.infnorm_sup(decay, W, H.t_min_per_coord(decay, W))
            dt = torch.tensor(decay, dtype=DT); Wt = torch.tensor(W, dtype=DT)
            mine = float(rows_torch(dt, Wt).max())
            ref = float(_ref_inf_torch(dt, Wt))
            me = max(me, abs(np_val - mine)); mr = max(mr, abs(ref - mine))
    ok = me < 1e-9 and mr < 1e-12
    _log("selftest", f"rows.max vs numpy max abs err={me:.2e}; vs cert_surrogate={mr:.2e} "
         f"({'OK' if ok else 'MISMATCH'})")
    if not ok:
        raise RuntimeError("surrogate self-test failed — fail-closed")
    return ok


# ============================================================================ #
#  violating init: scaled-random (cheap, n 全域)                                #
# ============================================================================ #
def make_scaled_violating(n, seed, target):
    """model init スケールの raw_W を、infnorm_sup≈target に届くまで scalar 倍 (bisection)。

    返り値 = leaf tensor (raw_W, raw_decay) requires_grad。core = (sigmoid(raw_decay), 2tanh(raw_W))。
    """
    g = torch.Generator().manual_seed(1000 + seed)
    raw_W0 = torch.randn(n, n, generator=g, dtype=DT) * (0.3 / n ** 0.5)
    raw_decay0 = torch.randn(n, generator=g, dtype=DT) * 0.5 + 1.0

    def inf_at(scale):
        decay = torch.sigmoid(raw_decay0)
        W = 2.0 * torch.tanh(raw_W0 * scale)
        return float(rows_torch(decay, W).max())

    lo, hi = 1.0, 1.0
    # expand hi until violating reachable
    for _ in range(60):
        if inf_at(hi) >= target:
            break
        hi *= 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if inf_at(mid) < target:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    raw_W = (raw_W0 * scale).clone().requires_grad_(True)
    raw_decay = raw_decay0.clone().requires_grad_(True)
    return raw_W, raw_decay


def cores_from_raw(raw_W, raw_decay):
    return torch.sigmoid(raw_decay), 2.0 * torch.tanh(raw_W)


# ============================================================================ #
#  (e) pull-back latency                                                        #
# ============================================================================ #
def pull_back_latency(variant, n, seed, target, threshold, lr, cap, tau, k):
    raw_W, raw_decay = make_scaled_violating(n, seed, target)
    decay, W = cores_from_raw(raw_W, raw_decay)
    inf0 = float(rows_torch(decay, W).max())
    opt = torch.optim.Adam([raw_W, raw_decay], lr=lr)
    steps = cap
    reached = False
    for it in range(cap):
        decay, W = cores_from_raw(raw_W, raw_decay)
        if float(rows_torch(decay, W).max()) < threshold:
            steps = it; reached = True; break
        loss = torch.relu(agg(variant, rows_torch(decay, W), tau, k) - threshold)
        opt.zero_grad(); loss.backward(); opt.step()
    decay, W = cores_from_raw(raw_W, raw_decay)
    inf_final = float(rows_torch(decay, W).max())
    return {"variant": variant, "n": n, "seed": seed, "inf0": inf0,
            "steps": steps, "reached": bool(reached), "inf_final": inf_final}


# ============================================================================ #
#  (f) gradient coverage (行被覆率 + 飽和 fraction + grad-norm 比材料)            #
# ============================================================================ #
def coverage_probe(variant, n, seed, target, threshold, tau, k):
    raw_W, raw_decay = make_scaled_violating(n, seed, target)
    decay, W = cores_from_raw(raw_W, raw_decay)
    inf0 = float(rows_torch(decay, W).max())
    loss = torch.relu(agg(variant, rows_torch(decay, W), tau, k) - threshold)
    if raw_W.grad is not None:
        raw_W.grad = None
    loss.backward()
    g = raw_W.grad.detach()
    row_norms = g.abs().sum(dim=1)
    eps = 1e-12
    rows_nonzero = int((row_norms > eps).sum())
    coverage = rows_nonzero / n
    # 飽和: W = 2 tanh(raw_W) の Jacobian factor 2·(1 - tanh^2(raw_W)); 触れた行内で勾配を殺す割合
    with torch.no_grad():
        deriv = 2.0 * (1.0 - torch.tanh(raw_W) ** 2)         # ∂W/∂raw_W (要素別)
        touched_rows = row_norms > eps
        if touched_rows.any():
            sat = (deriv[touched_rows] < 0.05).float().mean().item()
        else:
            sat = float("nan")
    return {"variant": variant, "n": n, "seed": seed, "inf0": inf0,
            "row_coverage": coverage, "rows_nonzero": rows_nonzero,
            "saturation_frac_touched": float(sat),
            "surrogate_gradnorm": float(g.abs().sum())}


# ============================================================================ #
#  (g) margin-matched 真ρ: admit 境界 (infnorm_sup≈t) の empirical_rho 分布       #
# ============================================================================ #
def slack_probe(n, seeds, infnorm_targets, samples):
    """各 target infnorm_sup へ random W を scale し、empirical_rho を測る → mean slack。"""
    out = {}
    rng = np.random.default_rng(7000 + n)
    for tgt in infnorm_targets:
        rhos = []
        infs = []
        for s in seeds:
            g = torch.Generator().manual_seed(2000 + s + int(tgt * 1000))
            raw_W0 = torch.randn(n, n, generator=g, dtype=DT) * (0.3 / n ** 0.5)
            raw_decay0 = torch.randn(n, generator=g, dtype=DT) * 0.5 + 1.0

            def inf_at(scale):
                decay = torch.sigmoid(raw_decay0); W = 2.0 * torch.tanh(raw_W0 * scale)
                return float(rows_torch(decay, W).max())
            lo, hi = 1e-3, 1.0
            for _ in range(60):
                if inf_at(hi) >= tgt:
                    break
                hi *= 1.5
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if inf_at(mid) < tgt:
                    lo = mid
                else:
                    hi = mid
            scale = 0.5 * (lo + hi)
            decay = torch.sigmoid(raw_decay0).numpy()
            W = (2.0 * torch.tanh(raw_W0 * scale)).numpy()
            infs.append(H.infnorm_sup(decay, W, H.t_min_per_coord(decay, W)))
            rhos.append(H.empirical_rho(decay, W, n_samples=samples, seed=900 + s))
        out[f"{tgt:.3f}"] = {"mean_infnorm": float(np.mean(infs)),
                             "mean_emp_rho": float(np.mean(rhos)),
                             "std_emp_rho": float(np.std(rhos)),
                             "mean_slack": float(np.mean(np.array(infs) - np.array(rhos)))}
    return out


# ============================================================================ #
#  model-based: GRAD 訓練 (CE + λ·surrogate) — (h)/(a)/(c')-train/margin 残留     #
# ============================================================================ #
def make_model(vocab, n, cfg):
    return H.GatedRecurrentLM(vocab, n, cfg["layers"], cfg["d"]).to(DEVICE)


def _admit_init(m, cfg):
    for li in range(cfg["layers"]):
        for _ in range(50):
            d_, W_ = m.core_np(li)
            if H.cert_inf(d_, W_):
                break
            with torch.no_grad():
                m.raw_W[li].mul_(0.5)


def model_run(arm, n, seed, cfg, data, lam, margin, variant, tau, k, measure_m=5):
    """arm ∈ {NONE, ENDO_GRAD}. NONE は lam=0。死は empirical_rho≥1 で判定 (設計 §3)。"""
    tr, va, vocab = data
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    m = make_model(vocab, n, cfg)
    _admit_init(m, cfg)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    threshold = BINDING["admit_threshold"] - margin
    li_list = list(range(cfg["layers"]))
    deaths_contract = 0
    residence_near = 0
    n_meas = 0
    ce_grad_dots = []          # admit-core grad-zero operationalize (c')
    rho_at_death = []
    inf_at_death = []
    infs_seen = []
    for it in range(cfg["grad_steps"]):
        x, y = H.batches(tr, cfg["T"], cfg["B"], rng)
        opt.zero_grad()
        loss_ce = torch.nn.functional.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
        if lam > 0.0:
            loss_cert = 0.0
            for li in li_list:
                decay, W = m.core(li)
                loss_cert = loss_cert + surrogate_loss(
                    variant, decay.double(), W.double(), threshold, tau, k)
            loss = loss_ce + lam * loss_cert
        else:
            loss = loss_ce
        loss.backward()
        opt.step()

        if (it + 1) % measure_m == 0 or it == cfg["grad_steps"] - 1:
            n_meas += 1
            li_inf = []
            li_rho = []
            for li in li_list:
                d_, W_ = m.core_np(li)
                inf = H.infnorm_sup(d_, W_, H.t_min_per_coord(d_, W_))
                rho = H.empirical_rho(d_, W_, n_samples=(200 if n <= 64 else 96), seed=seed + li)
                li_inf.append(inf); li_rho.append(rho)
            inf = max(li_inf); rho = max(li_rho)
            infs_seen.append(inf)
            if rho >= 1.0:
                deaths_contract += 1
                rho_at_death.append(rho); inf_at_death.append(inf)
            if inf >= (1.0 - BINDING["margin_band"]):
                residence_near += 1
            # (c') admit 中核 step での λ·∂surrogate と ∂CE の内積 ≈ 0 を確認 (lam>0 のみ)
            if lam > 0.0 and inf < (1.0 - margin) and inf < 1.0:
                ce_grad_dots.append(_admit_core_dot(m, x, y, vocab, li_list, lam,
                                                    margin, variant, tau, k))
    ce = H.eval_ce(m, va, cfg["T"], cfg["B"], cfg["eval_batches"], np.random.default_rng(seed + 7))
    return {"arm": arm, "n": n, "seed": seed, "lam": lam, "margin": margin, "variant": variant,
            "final_ce": float(ce), "deaths_contract": deaths_contract, "n_meas": n_meas,
            "death_rate": deaths_contract / max(1, n_meas),
            "residence_near_rate": residence_near / max(1, n_meas),
            "never_touched_death": bool(deaths_contract == 0),
            "mean_inf": float(np.mean(infs_seen)) if infs_seen else float("nan"),
            "admit_core_dot_mean": float(np.mean(ce_grad_dots)) if ce_grad_dots else float("nan"),
            "admit_core_dot_n": len(ce_grad_dots)}


def _admit_core_dot(m, x, y, vocab, li_list, lam, margin, variant, tau, k):
    """admit 中核 step: g_ce·(λ g_surr) の cosine ≈ 0 を測る (surrogate が CE を歪めない確認)。"""
    threshold = BINDING["admit_threshold"] - margin
    # ∂CE
    m.zero_grad()
    ce = torch.nn.functional.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
    ce.backward()
    g_ce = torch.cat([m.raw_W[li].grad.reshape(-1) for li in li_list])
    # ∂surrogate
    m.zero_grad()
    cert = 0.0
    for li in li_list:
        decay, W = m.core(li)
        cert = cert + surrogate_loss(variant, decay.double(), W.double(), threshold, tau, k)
    if float(cert) == 0.0:
        m.zero_grad()
        return 0.0                       # admit 中核 = surrogate loss 0 → grad 0 (期待)
    cert.backward()
    g_su = torch.cat([(lam * m.raw_W[li].grad).reshape(-1) for li in li_list])
    m.zero_grad()
    denom = (g_ce.norm() * g_su.norm()).item()
    return float((g_ce @ g_su).item() / denom) if denom > 0 else 0.0


def lambda_grad_norm_probe(n, seed, cfg, data, margin, variant, tau, k):
    """λ 選定材料: 違反域近傍で |∂CE/∂raw_W| と |∂surrogate/∂raw_W| の比 → decade。"""
    tr, va, vocab = data
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = make_model(vocab, n, cfg)
    _admit_init(m, cfg)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    # 少し走らせて自然に膨張側へ寄せる (NONE drift 風)
    for _ in range(40):
        x, y = H.batches(tr, cfg["T"], cfg["B"], rng)
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward(); opt.step()
    li_list = list(range(cfg["layers"]))
    threshold = BINDING["admit_threshold"] - margin
    x, y = H.batches(tr, cfg["T"], cfg["B"], rng)
    m.zero_grad()
    ce = torch.nn.functional.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1)); ce.backward()
    gce = float(torch.cat([m.raw_W[li].grad.reshape(-1) for li in li_list]).abs().mean())
    m.zero_grad()
    cert = 0.0
    for li in li_list:
        decay, W = m.core(li)
        cert = cert + surrogate_loss(variant, decay.double(), W.double(), threshold, tau, k)
    gsu = 0.0
    if float(cert) > 0:
        cert.backward()
        gsu = float(torch.cat([m.raw_W[li].grad.reshape(-1) for li in li_list]).abs().mean())
    m.zero_grad()
    return {"n": n, "seed": seed, "mean_grad_ce": gce, "mean_grad_surrogate": gsu,
            "surrogate_active": bool(gsu > 0)}


# ============================================================================ #
#  orchestration                                                                #
# ============================================================================ #
def load_prior():
    if os.path.exists(RPATH):
        try:
            return json.load(open(RPATH, encoding="utf-8"))
        except Exception:
            pass
    return None


def main():
    _self_test()
    prior = load_prior()
    out = prior or {"meta": {"experiment": "internalization feasibility (gradient-embedded supervision)",
                             "design": "HD1_INTERNALIZATION_DESIGN.md v2 §6", "device": DEVICE,
                             "binding": BINDING, "start_time": _now(), "status": "running"},
                    "probes": {}}
    out["meta"]["binding"] = BINDING

    def save():
        out["meta"]["update_time"] = _now()
        json.dump(out, open(RPATH, "w", encoding="utf-8"), indent=1)

    thr = BINDING["admit_threshold"]
    tau = BINDING["logsumexp_tau"]
    cap = BINDING["latency_cap"]
    lr = BINDING["lr"]
    target = BINDING["violating_target_infnorm"]
    variants = BINDING["variant_preference"]

    # ---------- (e) pull-back latency ----------
    if "latency" not in out["probes"]:
        _log("e", "pull-back latency (3 variants × n_analytic × seeds)")
        recs = []
        for n in N_ANALYTIC:
            k = math.ceil(BINDING["topk_frac"] * 2 * n)
            for variant in variants:
                for seed in SEEDS_ANALYTIC:
                    r = pull_back_latency(variant, n, seed, target, thr, lr, cap, tau, k)
                    recs.append(r)
                _log("e", f"n={n:4d} {variant:10s} steps(median)="
                     f"{int(np.median([x['steps'] for x in recs if x['n']==n and x['variant']==variant]))}"
                     f" reached={sum(x['reached'] for x in recs if x['n']==n and x['variant']==variant)}/{len(SEEDS_ANALYTIC)}")
            save()
        out["probes"]["latency"] = recs
        save()

    # ---------- (f) gradient coverage ----------
    if "coverage" not in out["probes"]:
        _log("f", "gradient coverage (行被覆率 + 飽和)")
        recs = []
        for n in N_ANALYTIC:
            k = math.ceil(BINDING["topk_frac"] * 2 * n)
            for variant in variants:
                for seed in SEEDS_ANALYTIC:
                    recs.append(coverage_probe(variant, n, seed, target, thr, tau, k))
            for variant in variants:
                cov = np.mean([x["row_coverage"] for x in recs if x["n"] == n and x["variant"] == variant])
                _log("f", f"n={n:4d} {variant:10s} row_coverage={cov:.3f}")
        out["probes"]["coverage"] = recs
        save()

    # ---------- (g) margin-matched 真ρ slack ----------
    if "slack" not in out["probes"]:
        _log("g", "admit-boundary true-rho slack (HARNESS の真ρ分布)")
        recs = {}
        for n in N_ANALYTIC:
            samples = 400 if n <= 64 else 150
            recs[str(n)] = slack_probe(n, SEEDS_ANALYTIC, [1.0, 0.98, 0.95, 0.90], samples)
            b = recs[str(n)]["1.000"]
            _log("g", f"n={n:4d} at infnorm≈1.0: mean_emp_rho={b['mean_emp_rho']:.3f} "
                 f"slack={b['mean_slack']:.3f}")
        out["probes"]["slack"] = recs
        save()

    # ---------- model cfg (CPU 予算) ----------
    model_cfg = dict(layers=1, d=64, T=48, B=12, lr=3e-3, grad_steps=120, eval_batches=3,
                     max_chars=30000, cert_every=4)
    data = None

    def get_data():
        nonlocal data
        if data is None:
            data = H.make_data(model_cfg["max_chars"], False)
        return data

    # ---------- λ grad-norm probe (b) ----------
    if "lambda_probe" not in out["probes"]:
        _log("b", "lambda grad-norm probe (|∂CE| vs |∂surrogate|)")
        d = get_data()
        recs = []
        for n in N_MODEL:
            k = math.ceil(BINDING["topk_frac"] * 2 * n)
            for seed in SEEDS_MODEL[:2]:
                recs.append(lambda_grad_norm_probe(n, seed, model_cfg, d, 0.0, "logsumexp", tau, k))
        out["probes"]["lambda_probe"] = recs
        save()
        for r in recs:
            _log("b", f"n={r['n']:4d} seed={r['seed']} |gCE|={r['mean_grad_ce']:.2e} "
                 f"|gSurr|={r['mean_grad_surrogate']:.2e}")

    # selected λ (decade match): λ ≈ 10^round(log10(|gCE|/|gSurr|))
    def select_lambda():
        recs = out["probes"]["lambda_probe"]
        ratios = [r["mean_grad_ce"] / r["mean_grad_surrogate"]
                  for r in recs if r["mean_grad_surrogate"] > 0]
        if not ratios:
            return 1.0
        return float(10 ** round(math.log10(float(np.median(ratios)))))

    lam_sel = select_lambda()
    out["meta"]["selected_lambda"] = lam_sel
    _log("b", f"selected λ (decade match) = {lam_sel:g}")

    # provisional variant for model probes (will reconcile w/ go-no-go below):
    # use logsumexp if max NO-GO (decided after latency analysis), else preference.
    sel_variant = _select_variant(out, BINDING)
    out["meta"]["selected_variant_provisional"] = sel_variant
    _log("decide", f"provisional selected variant = {sel_variant}")

    # ---------- (b) margin selection: 近境界滞在率 < X% の最小 margin ----------
    if "margin_probe" not in out["probes"]:
        _log("b", "margin selection (near-boundary residence)")
        d = get_data()
        recs = []
        n_m = 32 if 32 in N_MODEL else N_MODEL[-1]
        k = math.ceil(BINDING["topk_frac"] * 2 * n_m)
        for margin in BINDING["margin_grid"]:
            for seed in SEEDS_MODEL[:3]:
                r = model_run("ENDO_GRAD", n_m, seed, model_cfg, d, lam_sel, margin,
                              sel_variant, tau, k)
                recs.append(r)
            res = np.mean([x["residence_near_rate"] for x in recs if x["margin"] == margin])
            _log("b", f"margin={margin:.2f} near-boundary residence={res:.3f}")
        out["probes"]["margin_probe"] = recs
        save()

    def select_margin():
        recs = out["probes"]["margin_probe"]
        X = BINDING["margin_residence_X"]
        for margin in BINDING["margin_grid"]:
            res = np.mean([x["residence_near_rate"] for x in recs if x["margin"] == margin])
            if res < X:
                return margin
        return BINDING["margin_grid"][-1]   # none satisfied → largest

    margin_sel = select_margin()
    out["meta"]["selected_margin"] = margin_sel
    _log("b", f"selected margin = {margin_sel}")

    # ---------- (a) F 条項 + (h) tautology: NONE vs ENDO_GRAD ----------
    if "model_arms" not in out["probes"]:
        _log("ah", "NONE (F条項) + ENDO_GRAD (tautology) model runs")
        d = get_data()
        recs = []
        for n in N_MODEL:
            k = math.ceil(BINDING["topk_frac"] * 2 * n)
            for seed in SEEDS_MODEL:
                recs.append(model_run("NONE", n, seed, model_cfg, d, 0.0, margin_sel,
                                      sel_variant, tau, k))
            none_death = np.mean([x["death_rate"] for x in recs if x["arm"] == "NONE" and x["n"] == n])
            _log("a", f"n={n:4d} NONE death_rate={none_death:.3f} "
                 f"({'ACTIVE' if none_death >= 0.05 else 'inactive→excluded'})")
            for seed in SEEDS_MODEL:
                recs.append(model_run("ENDO_GRAD", n, seed, model_cfg, d, lam_sel, margin_sel,
                                      sel_variant, tau, k))
            eg = [x for x in recs if x["arm"] == "ENDO_GRAD" and x["n"] == n]
            never = np.mean([x["never_touched_death"] for x in eg])
            egd = np.mean([x["death_rate"] for x in eg])
            _log("h", f"n={n:4d} ENDO_GRAD death_rate={egd:.3f} never_touched={never:.2f} "
                 f"dot={np.nanmean([x['admit_core_dot_mean'] for x in eg]):.3f}")
            save()
        out["probes"]["model_arms"] = recs
        save()

    # ---------- VERDICT ----------
    verdict = compute_verdict(out, BINDING, lam_sel, margin_sel, sel_variant)
    out["verdict"] = verdict
    out["meta"]["status"] = "done"
    out["meta"]["end_time"] = _now()
    save()
    _print_verdict(verdict)
    return 0


def _variant_latency_table(out):
    recs = out["probes"]["latency"]
    tbl = {}
    for r in recs:
        tbl.setdefault((r["variant"], r["n"]), []).append(r["steps"])
    return {k: float(np.median(v)) for k, v in tbl.items()}


def _variant_coverage_table(out):
    recs = out["probes"]["coverage"]
    tbl = {}
    for r in recs:
        tbl.setdefault((r["variant"], r["n"]), []).append(r["row_coverage"])
    return {k: float(np.mean(v)) for k, v in tbl.items()}


def _select_variant(out, B):
    """GO 規則を満たす最も単純な変種 (preference 順)。無ければ logsumexp を暫定。"""
    if "latency" not in out["probes"] or "coverage" not in out["probes"]:
        return "logsumexp"
    lat = _variant_latency_table(out)
    cov = _variant_coverage_table(out)
    budget = B["latency_frac_of_budget"] * B["main_grad_steps"]
    maxn = max(B["main_ns"])
    for variant in B["variant_preference"]:
        lat_ok = all(lat.get((variant, n), 1e9) <= budget for n in B["main_ns"])
        cov_ok = cov.get((variant, maxn), 0.0) >= B["coverage_min_at_maxn"]
        if lat_ok and cov_ok:
            return variant
    # none GO with preference; pick the one with best (lowest) latency at maxn that meets coverage
    cand = [v for v in B["variant_preference"] if cov.get((v, maxn), 0.0) >= B["coverage_min_at_maxn"]]
    if cand:
        return min(cand, key=lambda v: lat.get((v, maxn), 1e9))
    return "logsumexp"


def compute_verdict(out, B, lam_sel, margin_sel, sel_variant):
    lat = _variant_latency_table(out)
    cov = _variant_coverage_table(out)
    budget = B["latency_frac_of_budget"] * B["main_grad_steps"]
    maxn = max(B["main_ns"])
    per_variant = {}
    for variant in B["variant_preference"]:
        lat_ok = all(lat.get((variant, n), 1e9) <= budget for n in B["main_ns"])
        cov_ok = cov.get((variant, maxn), 0.0) >= B["coverage_min_at_maxn"]
        per_variant[variant] = {
            "latency_by_n": {n: lat.get((variant, n)) for n in N_ANALYTIC},
            "coverage_by_n": {n: cov.get((variant, n)) for n in N_ANALYTIC},
            "latency_ok": bool(lat_ok), "coverage_ok": bool(cov_ok),
            "GO": bool(lat_ok and cov_ok)}
    any_go = any(v["GO"] for v in per_variant.values())
    # F 条項
    f_active = {}
    if "model_arms" in out["probes"]:
        for n in N_MODEL:
            dr = [x["death_rate"] for x in out["probes"]["model_arms"]
                  if x["arm"] == "NONE" and x["n"] == n]
            f_active[n] = bool(np.mean(dr) >= 0.05) if dr else None
    # tautology
    taut = {}
    if "model_arms" in out["probes"]:
        for n in N_MODEL:
            eg = [x for x in out["probes"]["model_arms"]
                  if x["arm"] == "ENDO_GRAD" and x["n"] == n]
            if eg:
                taut[n] = {"never_touched_frac": float(np.mean([x["never_touched_death"] for x in eg])),
                           "death_rate": float(np.mean([x["death_rate"] for x in eg]))}
    # slack → MATCHED threshold material
    slack = out["probes"].get("slack", {})
    decision = "GO" if (any_go and any(f_active.values())) else "NO-GO"
    if any_go and not any(f_active.values()):
        decision = "NO-GO (F条項: 死 regime 不在 — 全 n で NONE 死率<5%, 死地形再設計へ)"
    if not any_go:
        decision = ("NO-GO (latency/coverage: 全変種が go/no-go を満たさず — "
                    "λ 増強 or drift 検知後トリガ or 構造的不適へ方向転換)")
    return {"decision": decision, "selected_variant": sel_variant, "any_variant_GO": bool(any_go),
            "per_variant": per_variant, "selected_lambda": lam_sel,
            "selected_margin": margin_sel, "F_active_by_n": f_active,
            "tautology_by_n": taut, "slack_table": slack,
            "budget_steps": budget, "main_ns": B["main_ns"]}


def _print_verdict(v):
    print("\n" + "=" * 78)
    print("  FEASIBILITY VERDICT — 検証シグナル勾配内蔵 (gradient-embedded supervision)")
    print("=" * 78)
    print(f"  DECISION: {v['decision']}")
    print(f"  selected variant (provisional) = {v['selected_variant']}  "
          f"λ={v['selected_lambda']:g}  margin={v['selected_margin']}")
    print(f"  budget (FRAC·main_grad_steps) = {v['budget_steps']:.0f} steps; main n = {v['main_ns']}")
    print("  --- per-variant go/no-go ---")
    for variant, d in v["per_variant"].items():
        lats = "  ".join(f"n{n}:{(d['latency_by_n'][n] if d['latency_by_n'][n] is not None else -1):.0f}"
                         for n in d["latency_by_n"])
        print(f"   {variant:10s} GO={d['GO']!s:5s} lat_ok={d['latency_ok']!s:5s} "
              f"cov_ok={d['coverage_ok']!s:5s} | latency[{lats}]")
        covs = "  ".join(f"n{n}:{(d['coverage_by_n'][n] or 0):.2f}" for n in d["coverage_by_n"])
        print(f"   {'':10s} coverage[{covs}]")
    print(f"  F条項 active by n: {v['F_active_by_n']}")
    print(f"  tautology by n   : {v['tautology_by_n']}")
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
