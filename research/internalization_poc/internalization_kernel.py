# SPDX-License-Identifier: Apache-2.0
# 検証シグナル勾配内蔵 — MAIN RUN kernel (self-contained for Kaggle T4).
#
# Preregistration (binding) = HD1_INTERNALIZATION_PREREG.md (committed BEFORE results).
# Feasibility (確定本走形) = HD1_INTERNALIZATION_FEASIBILITY.md (敵対レビュー 41 agents 反映).
# Design = HD1_INTERNALIZATION_DESIGN.md v2. Substrate copied verbatim from hd1_grounding_kernel.py.
#
# 測るもの: 検証シグナルの作用経路 (勾配経由 = gradient-embedded vs 事後 rollback) の差が
#   死回避 (empirical_rho≥1) と CE に与える影響。死判定は empirical_rho 単独 (surrogate は内部量, A4)。
#
# arms:
#   NONE              : 補助損失なし (drift baseline; F 条項)
#   ENDO_HARNESS      : cert_inf gate k=4 -> fail で core+Adam 同期 rollback (HD-1 ENDO 流用)。surrogate なし
#   ENDO_GRAD         : 毎 step L = CE + λ·gated-logsumexp(θ=0.95)。rollback なし (主 arm)
#   ENDO_GRAD_MATCHED : ENDO_GRAD で θ=1.0 (margin 0; HARNESS 境界真ρ整合で margin 交絡分離)
#   ENDO_BOTH         : ENDO_GRAD + ENDO_HARNESS rollback (主張限定: 共存して壊れない動作確認)
#   ENDO_GRAD_L{...}  : H2 用 λ-sweep (λ∈{0.03,0.1,0.3,1.0}); RUN_STAGE=2 のみ
#
# surrogate = gated-logsumexp: 1[infnorm_sup.detach() >= θ] · relu(logsumexp(rows, τ=10) − θ)
#   .max()==numpy infnorm_sup; logsumexp≥max なので sound (loss=0 ⟹ admit)。gate で admit 中核 silence (C14)。
#
# RUN_STAGE=1 (first-check): NONE+ENDO_GRAD を 4 seeds → S1-S5 判定 (PREREG §4)。
# RUN_STAGE=2 (confirmatory): 全 arm を 16 seeds (+ H2 λ-sweep 32 seeds)。
# Output: result_internalization_s{STAGE}_n{N}.json (resumable; checkpoint after every record).
from __future__ import annotations

import copy
import datetime
import json
import os
import sys
import time
import traceback
import urllib.request

# ===================== USER TOGGLES ===================== #
RUN_N = int(os.environ.get("RINT_N", "64"))             # 64 | 128 | 256
RUN_STAGE = int(os.environ.get("RINT_STAGE", "1"))      # 1 (first-check) | 2 (confirmatory)
RUN_MODE = os.environ.get("RINT_MODE", "main")          # "main" | "smoke"
# ======================================================== #

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_T0 = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _log(tag, msg):
    print(f"[{time.time()-_T0:7.1f}s][{tag}] {msg}", flush=True)


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


_log("setup", f"internalization | RUN_N={RUN_N} RUN_STAGE={RUN_STAGE} RUN_MODE={RUN_MODE} device={DEVICE}"
     + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))


# ===================================================================== #
#  cert_inf / empirical_rho machinery (copied verbatim from hd1_highdim_evo.py)
# ===================================================================== #
def _clip(decay, W):
    return np.clip(np.asarray(decay, float).reshape(-1), 0, 1), np.clip(np.asarray(W, float), -2, 2)


def t_min_per_coord(decay, W, max_input_abs=1.0):
    decay, W = _clip(decay, W)
    M = np.abs(W).sum(axis=1) + max_input_abs * 1.0
    return 1.0 - np.tanh(M) ** 2


def infnorm_sup(decay, W, t_lo):
    decay, W = _clip(decay, W); n = decay.shape[0]
    absW = np.abs(W)
    diag_idx = np.arange(n)
    off = absW.sum(axis=1) - absW[diag_idx, diag_idx]
    best = 0.0
    for i in range(n):
        row = 0.0
        for ti in (t_lo[i], 1.0):
            diag = abs(decay[i] + (1.0 - decay[i]) * ti * W[i, i])
            row = max(row, diag + (1.0 - decay[i]) * ti * off[i])
        best = max(best, row)
    return float(best)


def cert_inf(decay, W, max_input_abs=1.0):
    return bool(infnorm_sup(decay, W, t_min_per_coord(decay, W, max_input_abs)) < 1.0)


def empirical_rho(decay, W, n_samples, seed=0):
    decay, W = _clip(decay, W); n = decay.shape[0]; rng = np.random.default_rng(seed); mx = 0.0
    for _ in range(n_samples):
        s = rng.uniform(-1, 1, n); x = rng.uniform(-1, 1, n)
        t = 1.0 - np.tanh(W @ s + x) ** 2
        J = np.diag(decay) + np.diag((1.0 - decay) * t) @ W
        mx = max(mx, float(np.max(np.abs(np.linalg.eigvals(J)))))
    return mx


# ===================================================================== #
#  gated-logsumexp surrogate (torch, differentiable; feasibility 確定形 A1)
# ===================================================================== #
def rows_torch(decay, W, max_input_abs=1.0):
    """(2, n) 行寄与テンソル。 .max() == numpy infnorm_sup (numpy 一致 3.55e-15)。"""
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
    return torch.stack(out)


def gated_logsumexp_loss(decay, W, theta, tau=10.0):
    """1[infnorm_sup.detach() >= θ] · relu(logsumexp(rows, τ)/τ − θ)。admit 中核で厳密 0 (C14)。"""
    R = rows_torch(decay, W)
    gate = (R.max().detach() >= theta).to(R.dtype)
    return gate * torch.relu(torch.logsumexp(tau * R.reshape(-1), dim=0) / tau - theta)


# ===================================================================== #
#  corpus / data / model (copied verbatim from hd1_grounding_kernel.py)
# ===================================================================== #
_TS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_corpus(max_chars):
    try:
        txt = urllib.request.urlopen(_TS_URL, timeout=30).read().decode("utf-8", "ignore")
        _log("corpus", f"downloaded tiny-shakespeare ({len(txt)} chars), using {min(len(txt), max_chars)}")
    except Exception as e:
        if RUN_MODE == "main":
            raise RuntimeError(f"corpus download failed — fail-fast in main mode ({e})")
        txt = ("To be, or not to be, that is the question. " * 4000)
        _log("corpus", f"download FAILED ({e}); offline fallback (smoke only)")
    return txt[:max_chars]


def make_data(max_chars):
    txt = load_corpus(max_chars); chars = sorted(set(txt)); vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in txt], dtype=np.int64)
    cut = int(len(ids) * 0.9); return ids[:cut], ids[cut:], vocab


def batches(ids, T, B, rng):
    ix = rng.integers(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[i:i + T] for i in ix]); y = np.stack([ids[i + 1:i + 1 + T] for i in ix])
    return torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE)


def unigram_ce(tr, va, vocab):
    c = np.bincount(tr, minlength=vocab).astype(float) + 1.0; p = c / c.sum()
    return float(-np.mean(np.log(p[va[:-1]])))


class GatedRecurrentLM(nn.Module):
    def __init__(self, vocab, n, layers, d):
        super().__init__(); self.n, self.layers = n, layers
        self.emb = nn.Embedding(vocab, d)
        self.U = nn.ModuleList([nn.Linear(d, n) for _ in range(layers)])
        self.P = nn.ModuleList([nn.Linear(n, d) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.raw_decay = nn.ParameterList([nn.Parameter(torch.randn(n) * 0.5 + 1.0) for _ in range(layers)])
        self.raw_W = nn.ParameterList([nn.Parameter(torch.randn(n, n) * (0.3 / n ** 0.5)) for _ in range(layers)])
        self.readout = nn.Linear(d, vocab)

    def core(self, li):
        return torch.sigmoid(self.raw_decay[li]), 2.0 * torch.tanh(self.raw_W[li])

    def core_np(self, li):
        d, W = self.core(li); return d.detach().cpu().numpy(), W.detach().cpu().numpy()

    def forward(self, idx):
        h = self.emb(idx)
        for li in range(self.layers):
            decay, W = self.core(li); xc = torch.tanh(self.U[li](h))
            h = self.norm[li](h + self.P[li](_recur(decay, W, xc)))
        return self.readout(h)


def _recur(decay, W, xc):
    B, T, n = xc.shape; s = torch.zeros(B, n, device=xc.device); outs = []
    for t in range(T):
        s = decay * s + (1 - decay) * torch.tanh(s @ W.T + xc[:, t]); outs.append(s)
    return torch.stack(outs, 1)


@torch.no_grad()
def eval_ce(model, ids, T, B, n_batches, rng):
    model.eval(); tot = 0.0
    for _ in range(n_batches):
        x, y = batches(ids, T, B, rng); lg = model(x)
        tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1)).item()
    model.train(); return tot / n_batches


# ===================================================================== #
#  preregistered configuration (HD1_INTERNALIZATION_PREREG.md §2)
# ===================================================================== #
if RUN_MODE == "smoke":
    CFG = dict(layers=1, d=32, T=16, B=4, lr=3e-3, grad_steps=20, max_chars=4000, eval_batches=2)
    N = 16
    RHO_SAMPLES_PT = 16
    RHO_SAMPLES_HP = 32
else:
    CFG = dict(layers=1, d=96, T=64, B=24, lr=3e-3, grad_steps=400, max_chars=80000, eval_batches=6)
    N = RUN_N
    RHO_SAMPLES_PT = {64: 200, 128: 96, 256: 48}[RUN_N]
    RHO_SAMPLES_HP = {64: 600, 128: 250, 256: 250}[RUN_N]

GATE_K = 4                       # ENDO_HARNESS cert gate cadence
MEASURE_M = 5                    # measurement cadence
SEP_T = 60                       # state-separation probe horizon
TAU = 10.0                       # logsumexp 温度 (feasibility A1)
LAM = 0.1                        # λ_cert (decade match, A2)
MARGIN = 0.05                    # ENDO_GRAD margin (θ=0.95; A3)
THETA_GRAD = 1.0 - MARGIN        # 0.95
THETA_MATCHED = 1.0              # MATCHED (margin 0)

# arm spec: (use_surrogate, theta, lam, use_harness_rollback)
ARM_SPEC = {
    "NONE":               (False, None, 0.0, False),
    "ENDO_HARNESS":       (False, None, 0.0, True),
    "ENDO_GRAD":          (True, THETA_GRAD, LAM, False),
    "ENDO_GRAD_MATCHED":  (True, THETA_MATCHED, LAM, False),
    "ENDO_BOTH":          (True, THETA_GRAD, LAM, True),
}
# H2 λ-sweep arms (RUN_STAGE=2 のみ)
LAM_SWEEP = [0.03, 0.1, 0.3, 1.0]
for _lam in LAM_SWEEP:
    ARM_SPEC[f"ENDO_GRAD_L{str(_lam).replace('.', '')}"] = (True, THETA_GRAD, _lam, False)

if RUN_STAGE == 1:
    ARMS = ["NONE", "ENDO_GRAD"]
    SEEDS = [2026 + i for i in range(4)]
else:
    ARMS = ["NONE", "ENDO_HARNESS", "ENDO_GRAD", "ENDO_GRAD_MATCHED", "ENDO_BOTH"]
    SEEDS = [2026 + i for i in range(16)]
    H2_ARMS = [f"ENDO_GRAD_L{str(l).replace('.', '')}" for l in LAM_SWEEP]
    H2_SEEDS = [2026 + i for i in range(32)]              # H2 は 32 seeds (PREREG §3)


# ===================================================================== #
#  measurement (death = empirical_rho>=1; surrogate = 内部量, A4)
# ===================================================================== #
def _np_rollout_pair(decay, W, T, rng):
    n = decay.shape[0]
    s_a = rng.choice([-1.0, 1.0], size=n); s_b = -s_a
    rates = []
    for _ in range(T):
        x = rng.uniform(-1, 1, n)
        d_prev = float(np.linalg.norm(s_a - s_b)) or 1e-12
        s_a = decay * s_a + (1 - decay) * np.tanh(W @ s_a + x)
        s_b = decay * s_b + (1 - decay) * np.tanh(W @ s_b + x)
        d_now = float(np.linalg.norm(s_a - s_b))
        if d_now > 0 and d_prev > 0:
            rates.append(np.log(d_now / d_prev))
    return float(np.mean(rates)) if rates else 0.0


def measure_point(model, li_list, seed, step, n_samples):
    rng = np.random.default_rng(seed * 100003 + step)
    rho = inf_sup = -1.0
    sep_rates = []
    for li in li_list:
        d_, W_ = model.core_np(li)
        rho = max(rho, empirical_rho(d_, W_, n_samples=n_samples, seed=seed + li))
        inf_sup = max(inf_sup, infnorm_sup(d_, W_, t_min_per_coord(d_, W_)))
        sep_rates.append(_np_rollout_pair(d_, W_, SEP_T, rng))
    return {"step": step, "rho_hat": float(rho), "sep_rate": float(max(sep_rates)),
            "infnorm_sup": float(inf_sup), "contract_death": bool(rho >= 1.0),
            "harm_death": bool(max(sep_rates) >= 0.0)}


# ===================================================================== #
#  one run
# ===================================================================== #
def run_arm(arm, seed, data):
    use_surr, theta, lam, use_harness = ARM_SPEC[arm]
    tr, va, vocab = data
    cfg = dict(CFG); cfg["n"] = N
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    m = GatedRecurrentLM(vocab, N, cfg["layers"], cfg["d"]).to(DEVICE)
    li_list = list(range(cfg["layers"]))

    # shared admissible init for ALL arms (PREREG §2)
    for li in li_list:
        for _ in range(50):
            d_, W_ = m.core_np(li)
            if cert_inf(d_, W_):
                break
            with torch.no_grad():
                m.raw_W[li].mul_(0.5)

    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    prev_core = prev_opt = None
    if use_harness:
        prev_core = [(m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone()) for li in li_list]
        prev_opt = copy.deepcopy(opt.state_dict())

    traj = []
    deaths_contract = deaths_harm = rollbacks = 0
    admit_core_active = admit_core_steps = 0          # S3 (gate silence) 材料

    for it in range(cfg["grad_steps"]):
        x, y = batches(tr, cfg["T"], cfg["B"], rng)
        opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
        if use_surr:
            cert = 0.0
            for li in li_list:
                decay, W = m.core(li)
                cert = cert + gated_logsumexp_loss(decay.double(), W.double(), theta, TAU)
            loss = loss + lam * cert
        loss.backward()
        opt.step()

        # ENDO_HARNESS / ENDO_BOTH: cert gate -> fail で core+Adam 同期 rollback
        if use_harness and ((it + 1) % GATE_K == 0 or it == cfg["grad_steps"] - 1):
            failed = any(not cert_inf(*m.core_np(li)) for li in li_list)
            if failed:
                with torch.no_grad():
                    for li in li_list:
                        m.raw_decay[li].copy_(prev_core[li][0]); m.raw_W[li].copy_(prev_core[li][1])
                opt.load_state_dict(prev_opt)
                rollbacks += 1
            else:
                prev_core = [(m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone()) for li in li_list]
                prev_opt = copy.deepcopy(opt.state_dict())

        if (it + 1) % MEASURE_M == 0 or it == cfg["grad_steps"] - 1:
            pt = measure_point(m, li_list, seed, it + 1, RHO_SAMPLES_PT)
            traj.append(pt)
            if pt["contract_death"]:
                deaths_contract += 1
            if pt["harm_death"]:
                deaths_harm += 1
            # S3: gated surrogate が真 admit 中核 (infnorm<θ) で silent か (use_surr のみ)
            if use_surr and pt["infnorm_sup"] < theta:
                admit_core_steps += 1
                with torch.no_grad():
                    act = 0.0
                    for li in li_list:
                        decay, W = m.core(li)
                        act += float(gated_logsumexp_loss(decay.double(), W.double(), theta, TAU))
                if act > 0:
                    admit_core_active += 1

    rho_hp = max(empirical_rho(*m.core_np(li), n_samples=RHO_SAMPLES_HP, seed=seed * 7919 + 999 + li)
                 for li in li_list)
    ce = eval_ce(m, va, cfg["T"], cfg["B"], cfg["eval_batches"], np.random.default_rng(seed + 7))
    n_meas = len(traj)
    return {
        "arm": arm, "n": N, "seed": seed, "lam": lam, "theta": theta, "final_ce": float(ce),
        "deaths_contract": deaths_contract, "deaths_harm": deaths_harm, "rollbacks": rollbacks,
        "n_meas": n_meas, "death_rate": deaths_contract / max(1, n_meas),
        "never_touched_death": bool(deaths_contract == 0),
        "window_max_rho": float(max((p["rho_hat"] for p in traj), default=0.0)),
        "rho_excess_integral": float(sum(max(p["rho_hat"] - 1.0, 0.0) for p in traj)),
        "admit_core_active": admit_core_active, "admit_core_steps": admit_core_steps,
        "final_rho_hp": float(rho_hp), "trajectory": traj,
    }


# ===================================================================== #
#  stage-1 first-check 判定 (PREREG §4)
# ===================================================================== #
def stage1_verdict(records, n_meas_per_run):
    none = [r for r in records if r["arm"] == "NONE" and r.get("status", "ok") == "ok"]
    egrad = [r for r in records if r["arm"] == "ENDO_GRAD" and r.get("status", "ok") == "ok"]
    none_death = float(np.mean([r["death_rate"] for r in none])) if none else None
    s1 = (none_death is not None and none_death >= 0.05)
    never = float(np.mean([r["never_touched_death"] for r in egrad])) if egrad else None
    s2 = (never is not None and never < 1.0)          # 少なくとも 1 seed が死に触れる
    core_active = sum(r["admit_core_active"] for r in egrad)
    core_steps = sum(r["admit_core_steps"] for r in egrad)
    s3 = (core_active == 0)
    egrad_inf_drift = float(np.mean([r["window_max_rho"] for r in egrad])) if egrad else None
    return {
        "n": N,
        "S1_NONE_death_active": {"none_death_rate": none_death, "pass": bool(s1),
                                 "note": "death regime active (>=5%)" if s1 else "INACTIVE -> exclude this n"},
        "S2_tautology_nonapplicable": {"never_touched_frac": never, "pass": bool(s2),
                                       "note": "non-tautological (some seed touches death)" if s2
                                       else "tautology suspect (gate-like)"},
        "S3_gate_silence": {"admit_core_active": core_active, "admit_core_steps": core_steps,
                            "pass": bool(s3), "note": "C14 gated silence OK" if s3 else "gate leaked -> recalibrate θ/τ"},
        "S5_egrad_window_max_rho": egrad_inf_drift,   # margin residence 再確認材料 (記述)
        "go_stage2": bool(s1),                        # S1 が confirmatory の前提 (S2/S3 は解釈規律)
    }


def main():
    rpath = (f"result_internalization_s{RUN_STAGE}_n{N}.json" if RUN_MODE == "main"
             else f"result_internalization_smoke_s{RUN_STAGE}_n{N}.json")
    data = make_data(CFG["max_chars"])
    tr, va, vocab = data
    if RUN_MODE == "main":
        assert vocab >= 40, f"corpus sanity failed (vocab={vocab}; offline fallback?)"
    uni = unigram_ce(tr, va, vocab)

    # build the job list (arm, seed)
    jobs = [(arm, seed) for arm in ARMS for seed in SEEDS]
    if RUN_STAGE == 2:
        jobs += [(arm, seed) for arm in H2_ARMS for seed in H2_SEEDS]
    _log("run", f"vocab={vocab} unigram_CE={uni:.4f} n={N} stage={RUN_STAGE} arms={ARMS} "
         f"jobs={len(jobs)}")

    out = {"meta": {"experiment": "gradient-embedded supervision vs post-hoc rollback",
                    "prereg": "HD1_INTERNALIZATION_PREREG.md", "stage": RUN_STAGE, "mode": RUN_MODE,
                    "n": N, "device": DEVICE,
                    "gpu": (torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"),
                    "cfg": {**CFG, "tau": TAU, "lam": LAM, "margin": MARGIN, "gate_k": GATE_K,
                            "measure_m": MEASURE_M, "sep_T": SEP_T, "rho_samples_pt": RHO_SAMPLES_PT,
                            "rho_samples_hp": RHO_SAMPLES_HP, "unigram_ce": uni, "arm_spec":
                            {k: list(v) for k, v in ARM_SPEC.items()}},
                    "status": "running", "error": None, "start_time": _now()},
           "records": []}
    done = set()
    if os.path.exists(rpath):
        try:
            prior = json.load(open(rpath, encoding="utf-8"))
            out["records"] = prior.get("records", [])
            done = {(r["arm"], r["seed"]) for r in out["records"] if r.get("status", "ok") == "ok"}
            if done:
                _log("run", f"resume: {len(done)} records done -> skip")
        except Exception:
            pass

    def save():
        json.dump(out, open(rpath, "w", encoding="utf-8"), indent=1)

    try:
        for arm, seed in jobs:
            if (arm, seed) in done:
                continue
            t1 = time.time()
            try:
                r = run_arm(arm, seed, data); r["status"] = "ok"
            except Exception as e:
                r = {"arm": arm, "n": N, "seed": seed, "status": "error",
                     "error": f"{e}\n{traceback.format_exc()}"}
            out["records"].append(r); save()
            if r["status"] == "ok":
                _log("run", f"n={N} {arm:18s} seed={seed} ce={r['final_ce']:.4f} "
                     f"d_con={r['deaths_contract']:2d} dr={r['death_rate']:.3f} "
                     f"rb={r['rollbacks']} wmaxrho={r['window_max_rho']:.3f} "
                     f"core_act={r['admit_core_active']}/{r['admit_core_steps']} "
                     f"rho_hp={r['final_rho_hp']:.3f} ({time.time()-t1:.0f}s)")
            else:
                _log("run", f"n={N} {arm:18s} seed={seed} ERROR")
        out["meta"]["status"] = "done"
    except Exception as e:
        out["meta"]["status"] = "error"; out["meta"]["error"] = f"{e}\n{traceback.format_exc()}"
        _log("run", f"FATAL: {e}")
    out["meta"]["end_time"] = _now()

    if RUN_STAGE == 1:
        ok = [r for r in out["records"] if r.get("status") == "ok"]
        n_meas = ok[0]["n_meas"] if ok else 0
        out["stage1_verdict"] = stage1_verdict(ok, n_meas)
        _log("S", f"stage-1 verdict n={N}: {json.dumps(out['stage1_verdict'], ensure_ascii=False)}")
    save()

    ok = [r for r in out["records"] if r.get("status") == "ok"]
    print(f"\n=== internalization stage-{RUN_STAGE} n={N} (unigram={uni:.4f}; {len(ok)} ok; "
          f"status={out['meta']['status']}; wall={time.time()-_T0:.0f}s) ===")
    for arm in (ARMS + (H2_ARMS if RUN_STAGE == 2 else [])):
        rs = [r for r in ok if r["arm"] == arm]
        if not rs:
            continue
        mean = lambda k_: float(np.mean([r[k_] for r in rs]))
        print(f"  {arm:18s} ce={mean('final_ce'):.4f} dr={mean('death_rate'):.3f} "
              f"wmaxrho={mean('window_max_rho'):.3f} rho_excess={mean('rho_excess_integral'):.2f} "
              f"rho_hp={mean('final_rho_hp'):.3f}")
    print(f"[saved] {rpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
