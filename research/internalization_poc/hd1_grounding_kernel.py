# SPDX-License-Identifier: Apache-2.0
# HD-1 grounding MAIN RUN kernel (self-contained for Kaggle T4).
#
# Preregistration (binding) = HD1_GROUNDING_PREREG.md (committed BEFORE results).
# Design = HD1_GROUNDING_DESIGN.md v3. One kernel instance per n (RUN_N toggle);
# kernel ids: furusekazufumi/hd1-grounding-n{64,128,256}.
#
# arms: NONE / ENDO / REVIVE / OBSERVE_P1 / OBSERVE_P2 (+ ENDO_K8 exploratory @ n=128)
# - shared admissible init (cert_inf admit via raw_W halving) for ALL arms
# - ENDO: cert_inf gate cadence k=4 -> fail => core+Adam synchronized rollback
# - REVIVE: no gate; independent monitor (m=5) detects contract death -> record death ->
#   raw_W <- c*raw_W (bisection 24 iters to admit) + Adam state reset for raw_W -> continue
# - OBSERVE: proxy = moving average (window=4 measurements) of g_t = state-norm growth;
#   threshold = 10th pctl of death-event proxy log; exceed => shrink last m steps of
#   updates by beta=0.5. P1 = self history only; P2 = pooled P1 death log (within n).
# - measurement (ALL arms, cadence m=5): empirical_rho (per-point samples by n),
#   state-separation probe (box-edge pair, SEP_T=60), proxy g_t, infnorm_sup.
#   contract death = rho_hat >= 1; harm death = sep_rate >= 0.
# Output: result_hd1g_n{N}.json (resumable; checkpoint after every record).
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
RUN_N = int(os.environ.get("HD1G_N", "64"))            # 64 | 128 | 256 (one kernel per n)
RUN_MODE = os.environ.get("HD1G_MODE", "main")         # "main" | "smoke" (wiring check)
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


_log("setup", f"HD-1 grounding | RUN_N={RUN_N} RUN_MODE={RUN_MODE} device={DEVICE}"
     + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))


# ----- cert_inf machinery (copied verbatim from hd1_highdim_evo.py) ----- #
def _clip(decay, W):
    return np.clip(np.asarray(decay, float).reshape(-1), 0, 1), np.clip(np.asarray(W, float), -2, 2)


def t_min_per_coord(decay, W, max_input_abs=1.0):
    decay, W = _clip(decay, W)
    M = np.abs(W).sum(axis=1) + max_input_abs * 1.0           # V=I
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
    """From-below sup of rho(J) over the (s,x) box (same estimator as HD-1)."""
    decay, W = _clip(decay, W); n = decay.shape[0]; rng = np.random.default_rng(seed); mx = 0.0
    for _ in range(n_samples):
        s = rng.uniform(-1, 1, n); x = rng.uniform(-1, 1, n)
        t = 1.0 - np.tanh(W @ s + x) ** 2
        J = np.diag(decay) + np.diag((1.0 - decay) * t) @ W
        mx = max(mx, float(np.max(np.abs(np.linalg.eigvals(J)))))
    return mx


# ----- corpus / data / model (copied verbatim from hd1_highdim_evo.py) ----- #
_TS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_corpus(max_chars):
    try:
        txt = urllib.request.urlopen(_TS_URL, timeout=30).read().decode("utf-8", "ignore")
        _log("corpus", f"downloaded tiny-shakespeare ({len(txt)} chars), using {min(len(txt), max_chars)}")
    except Exception as e:
        if RUN_MODE == "main":
            # binding run must not silently degrade to a fallback corpus (prereg sec.2)
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


# ----- preregistered configuration (HD1_GROUNDING_PREREG.md sec.2) ----- #
if RUN_MODE == "smoke":
    CFG = dict(layers=1, d=32, T=16, B=4, lr=3e-3, grad_steps=20, max_chars=4000,
               eval_batches=2)
    N = 16
    SEEDS = [2026, 2027]
    RHO_SAMPLES_PT = 16
    RHO_SAMPLES_HP = 32
else:
    CFG = dict(layers=1, d=96, T=64, B=24, lr=3e-3, grad_steps=400, max_chars=80000,
               eval_batches=6)                              # HD-1 full equivalent (E3)
    N = RUN_N
    SEEDS = [2026 + i for i in range(16)]
    RHO_SAMPLES_PT = {64: 200, 128: 96, 256: 48}[RUN_N]      # per-point (prereg A5)
    RHO_SAMPLES_HP = {64: 600, 128: 250, 256: 250}[RUN_N]    # final point, E3

GATE_K = 4                       # ENDO gate cadence (ENDO_K8 uses 8)
MEASURE_M = 5                    # monitor / trajectory cadence
SEP_T = 60                       # state-separation probe horizon
OBSERVE_PCTL = 10.0              # death-event proxy percentile threshold
OBSERVE_BETA = 0.5               # avoidance shrink factor
OBSERVE_MA_W = 4                 # proxy moving-average window (measurements)
ARMS = ["NONE", "ENDO", "REVIVE", "OBSERVE_P1", "OBSERVE_P2"]
if RUN_MODE == "main" and RUN_N == 128:
    ARMS.append("ENDO_K8")       # exploratory k-sensitivity (prereg A7 / E9)


# ----- measurement: two-layer death + proxy + sound margin ----- #
def _np_rollout_pair(decay, W, T, rng):
    """Box-edge pair under identical input: mean log decay rate + state-norm growth g."""
    n = decay.shape[0]
    s_a = rng.choice([-1.0, 1.0], size=n)
    s_b = -s_a
    s = np.zeros(n)
    rates, norms = [], []
    for _ in range(T):
        x = rng.uniform(-1, 1, n)
        d_prev = float(np.linalg.norm(s_a - s_b)) or 1e-12
        s_a = decay * s_a + (1 - decay) * np.tanh(W @ s_a + x)
        s_b = decay * s_b + (1 - decay) * np.tanh(W @ s_b + x)
        s = decay * s + (1 - decay) * np.tanh(W @ s + x)
        d_now = float(np.linalg.norm(s_a - s_b))
        if d_now > 0 and d_prev > 0:
            rates.append(np.log(d_now / d_prev))
        norms.append(float(np.linalg.norm(s)))
    half = len(norms) // 2
    g = float(np.log((np.mean(norms[half:]) + 1e-12) / (np.mean(norms[:half]) + 1e-12)))
    return float(np.mean(rates)), g


def measure_point(model, li_list, seed, step, n_samples):
    rng = np.random.default_rng(seed * 100003 + step)
    rho = inf_sup = -1.0
    sep_rates, gs = [], []
    for li in li_list:
        d_, W_ = model.core_np(li)
        rho = max(rho, empirical_rho(d_, W_, n_samples=n_samples, seed=seed + li))
        inf_sup = max(inf_sup, infnorm_sup(d_, W_, t_min_per_coord(d_, W_)))
        r, g = _np_rollout_pair(d_, W_, SEP_T, rng)
        sep_rates.append(r)
        gs.append(g)
    return {"step": step, "rho_hat": float(rho), "sep_rate": float(max(sep_rates)),
            "proxy_g": float(max(gs)), "infnorm_sup": float(inf_sup),
            "contract_death": bool(rho >= 1.0), "harm_death": bool(max(sep_rates) >= 0.0)}


# ----- REVIVE repair (raw_W bisection to admit, then mandatory cert check) ----- #
def repair_raw(model, li):
    raw = model.raw_W[li].detach().clone()
    lo, hi = 0.0, 1.0
    ok_c = 0.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        with torch.no_grad():
            model.raw_W[li].copy_(raw * mid)
        d_, W_ = model.core_np(li)
        if cert_inf(d_, W_):
            ok_c = mid
            lo = mid
        else:
            hi = mid
    with torch.no_grad():
        model.raw_W[li].copy_(raw * ok_c)
    d_, W_ = model.core_np(li)
    assert cert_inf(d_, W_), "repair failed to land in admit set"
    return ok_c


def _reset_adam_state(opt, params):
    for p in params:
        if p in opt.state:
            del opt.state[p]


# ----- one run ----- #
def run_arm(arm, seed, data, shared_death_log=None):
    """shared_death_log: OBSERVE_P2 only — pooled P1 death-event proxies (within n)."""
    tr, va, vocab = data
    cfg = dict(CFG)
    cfg["n"] = N
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    m = GatedRecurrentLM(vocab, N, cfg["layers"], cfg["d"]).to(DEVICE)
    li_list = list(range(cfg["layers"]))

    # shared admissible init for ALL arms (fair start; prereg sec.2)
    for li in li_list:
        for _ in range(50):
            d_, W_ = m.core_np(li)
            if cert_inf(d_, W_):
                break
            with torch.no_grad():
                m.raw_W[li].mul_(0.5)

    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    gate_k = 8 if arm == "ENDO_K8" else GATE_K
    is_endo = arm in ("ENDO", "ENDO_K8")
    is_observe = arm in ("OBSERVE_P1", "OBSERVE_P2")
    prev_core = None
    prev_opt_state = None
    snap_theta = None
    g_window = []                                       # raw g_t history for MA proxy
    death_proxy_log = list(shared_death_log or [])      # P2 starts from pooled P1 log
    obs_threshold = (float(np.percentile(death_proxy_log, OBSERVE_PCTL))
                     if death_proxy_log else None)
    own_death_proxies = []                              # this run's death-event proxies
    traj = []
    deaths_contract = deaths_harm = repairs = rollbacks = avoids = 0

    def core_snapshot():
        return [(m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone()) for li in li_list]

    def core_restore(snap):
        with torch.no_grad():
            for li in li_list:
                m.raw_decay[li].copy_(snap[li][0])
                m.raw_W[li].copy_(snap[li][1])

    if is_endo:
        prev_core = core_snapshot()
        prev_opt_state = copy.deepcopy(opt.state_dict())
    if is_observe:
        snap_theta = copy.deepcopy(m.state_dict())

    for it in range(cfg["grad_steps"]):
        x, y = batches(tr, cfg["T"], cfg["B"], rng)
        opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        opt.step()

        # ENDO / ENDO_K8: cert gate -> fail => core+Adam synchronized rollback
        if is_endo and ((it + 1) % gate_k == 0 or it == cfg["grad_steps"] - 1):
            failed = any(not cert_inf(*m.core_np(li)) for li in li_list)
            if failed:
                core_restore(prev_core)
                opt.load_state_dict(prev_opt_state)
                rollbacks += 1
            else:
                prev_core = core_snapshot()
                prev_opt_state = copy.deepcopy(opt.state_dict())

        # independent monitor (cadence m): trajectory for ALL arms + REVIVE/OBSERVE reaction
        if (it + 1) % MEASURE_M == 0 or it == cfg["grad_steps"] - 1:
            pt = measure_point(m, li_list, seed, it + 1, RHO_SAMPLES_PT)
            g_window.append(pt["proxy_g"])
            if len(g_window) > OBSERVE_MA_W:
                g_window.pop(0)
            pt["proxy_ma"] = float(np.mean(g_window))
            traj.append(pt)
            if pt["contract_death"]:
                deaths_contract += 1
            if pt["harm_death"]:
                deaths_harm += 1

            if arm == "REVIVE" and pt["contract_death"]:
                for li in li_list:
                    repair_raw(m, li)
                    _reset_adam_state(opt, [m.raw_W[li]])
                repairs += 1

            if is_observe:
                if pt["contract_death"]:
                    death_proxy_log.append(pt["proxy_ma"])
                    own_death_proxies.append(pt["proxy_ma"])
                    obs_threshold = float(np.percentile(death_proxy_log, OBSERVE_PCTL))
                if obs_threshold is not None and pt["proxy_ma"] >= obs_threshold:
                    cur = m.state_dict()
                    with torch.no_grad():
                        for k_ in cur:
                            cur[k_].copy_(snap_theta[k_] + OBSERVE_BETA * (cur[k_] - snap_theta[k_]))
                    m.load_state_dict(cur)
                    avoids += 1
                snap_theta = copy.deepcopy(m.state_dict())

    # final high-precision rho (E3 descriptive)
    rho_hp = max(empirical_rho(*m.core_np(li), n_samples=RHO_SAMPLES_HP, seed=seed * 7919 + 999 + li)
                 for li in li_list)
    ce = eval_ce(m, va, cfg["T"], cfg["B"], cfg["eval_batches"], np.random.default_rng(seed + 7))
    return {
        "arm": arm, "n": N, "seed": seed, "final_ce": float(ce),
        "deaths_contract": deaths_contract, "deaths_harm": deaths_harm,
        "repairs": repairs, "rollbacks": rollbacks, "avoids": avoids,
        "death_proxy_log": own_death_proxies, "final_rho_hp": float(rho_hp),
        "trajectory": traj,
    }


def main():
    rpath = f"result_hd1g_n{N}.json" if RUN_MODE == "main" else f"result_hd1g_smoke_n{N}.json"
    data = make_data(CFG["max_chars"])
    tr, va, vocab = data
    if RUN_MODE == "main":
        assert vocab >= 40, f"corpus sanity failed (vocab={vocab}; offline fallback?)"
    uni = unigram_ce(tr, va, vocab)
    _log("run", f"vocab={vocab} unigram_CE={uni:.4f} n={N} arms={ARMS} seeds={len(SEEDS)}")

    out = {"meta": {"experiment": "HD-1 grounding main run (preregistered)",
                    "prereg": "HD1_GROUNDING_PREREG.md", "mode": RUN_MODE, "n": N,
                    "device": DEVICE,
                    "gpu": (torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"),
                    "cfg": {**CFG, "seeds": SEEDS, "gate_k": GATE_K, "measure_m": MEASURE_M,
                            "sep_T": SEP_T, "observe_pctl": OBSERVE_PCTL, "beta": OBSERVE_BETA,
                            "ma_window": OBSERVE_MA_W, "rho_samples_pt": RHO_SAMPLES_PT,
                            "rho_samples_hp": RHO_SAMPLES_HP, "unigram_ce": uni},
                    "status": "running", "error": None, "start_time": _now()},
           "records": []}
    done = set()
    if os.path.exists(rpath):
        try:
            prior = json.load(open(rpath))
            out["records"] = prior.get("records", [])
            done = {(r["arm"], r["seed"]) for r in out["records"] if r.get("status", "ok") == "ok"}
            if done:
                _log("run", f"resume: {len(done)} records done -> skip")
        except Exception:
            pass

    def save():
        json.dump(out, open(rpath, "w"), indent=1)

    try:
        for arm in ARMS:
            # OBSERVE_P2 needs the pooled P1 death log (within this n)
            shared = None
            if arm == "OBSERVE_P2":
                p1 = [r for r in out["records"] if r["arm"] == "OBSERVE_P1"
                      and r.get("status", "ok") == "ok"]
                if len(p1) < len(SEEDS):
                    raise RuntimeError(f"OBSERVE_P2 requires all P1 done ({len(p1)}/{len(SEEDS)})")
                shared = [v for r in p1 for v in r["death_proxy_log"]]
                _log("run", f"OBSERVE_P2 pooled death log: {len(shared)} events")
            for seed in SEEDS:
                if (arm, seed) in done:
                    continue
                t1 = time.time()
                try:
                    r = run_arm(arm, seed, data, shared_death_log=shared)
                    r["status"] = "ok"
                except Exception as e:
                    r = {"arm": arm, "n": N, "seed": seed, "status": "error",
                         "error": f"{e}\n{traceback.format_exc()}"}
                out["records"].append(r)
                save()
                if r["status"] == "ok":
                    _log("run", f"n={N} {arm:11s} seed={seed} ce={r['final_ce']:.4f} "
                         f"d_con={r['deaths_contract']:2d} d_harm={r['deaths_harm']:2d} "
                         f"rep={r['repairs']} rb={r['rollbacks']} av={r['avoids']} "
                         f"rho_hp={r['final_rho_hp']:.3f} ({time.time()-t1:.0f}s)")
                else:
                    _log("run", f"n={N} {arm:11s} seed={seed} ERROR")
        out["meta"]["status"] = "done"
    except Exception as e:
        out["meta"]["status"] = "error"
        out["meta"]["error"] = f"{e}\n{traceback.format_exc()}"
        _log("run", f"FATAL: {e}")
    out["meta"]["end_time"] = _now()
    save()

    # descriptive summary (no tests here; confirmatory analysis is local, prereg sec.3)
    ok = [r for r in out["records"] if r.get("status") == "ok"]
    print(f"\n=== HD-1 grounding n={N} (unigram={uni:.4f}; {len(ok)} ok; "
          f"status={out['meta']['status']}; wall={time.time()-_T0:.0f}s) ===")
    n_meas = len(ok[0]["trajectory"]) if ok else 0
    for arm in ARMS:
        rs = [r for r in ok if r["arm"] == arm]
        if not rs:
            continue
        mean = lambda k_: float(np.mean([r[k_] for r in rs]))
        print(f"  {arm:11s} ce={mean('final_ce'):.4f} d_con={mean('deaths_contract'):5.1f} "
              f"d_harm={mean('deaths_harm'):5.1f} rep={mean('repairs'):4.1f} "
              f"rb={mean('rollbacks'):5.1f} av={mean('avoids'):5.1f} rho_hp={mean('final_rho_hp'):.3f}")
    rs = [r for r in ok if r["arm"] == "NONE"]
    if rs and n_meas:
        frac = float(np.mean([r["deaths_contract"] / n_meas for r in rs]))
        print(f"  F-check (contract): NONE window death rate = {frac:.2%} "
              f"({'ACTIVE' if frac >= 0.05 else 'inactive -> n excluded'})")
        frach = float(np.mean([r["deaths_harm"] / n_meas for r in rs]))
        print(f"  harm activity     : NONE window harm rate  = {frach:.2%} "
              f"({'active' if frach >= 0.05 else 'inactive -> harm indicator excluded'})")
    print(f"[saved] {rpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
