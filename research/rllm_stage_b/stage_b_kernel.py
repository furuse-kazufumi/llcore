# SPDX-License-Identifier: Apache-2.0
# R-LLM Stage-B - GPU: the verified core inside a REAL gradient-trained softmax-attention Transformer.
#
# Derived from ../highdim_evolution/hd1_highdim_evo.py (same self-contained, checkpointed, resumable
# Kaggle/Colab GPU scaffold). Pre-registration: PREREGISTRATION_STAGE_B.md (gates B-G1..B-G4 fixed
# before any GPU run).
#
# DESIGN: char-LM, 2 pre-LN Transformer blocks with CAUSAL SLIDING-WINDOW softmax attention (w_att=8,
# stacked receptive field ~15) over context T=160  =>  information beyond ~15 chars can ONLY flow
# through the verified recurrent core s_t = decay*s + (1-decay)*tanh(W s + xc_t), xc = tanh(U h)
# (|xc|<=1 => cert_inf with max_input_abs=1.0 stays sound). The certified object is ONLY (decay, W).
#
# CONDITIONS (core training regime; all else identical, CRN-paired):
#   pure    - no core channel (is the memory channel load-bearing at all? B-G1)
#   none    - core unconstrained (+ derived post-hoc projection metric, B-G4)
#   project - after EVERY step, if cert_inf fails scale W by largest gamma in [0,1] (bisection) that
#             certifies; deterministic, never reverts (constraint without rejection-friction)
#   reject  - HD-1-style: every cert_every steps, revert core (only) to last passing snapshot
# B-Q2: friction vs expressivity = where does CE(project) land between CE(none) and CE(reject)?
from __future__ import annotations

# ===================== USER TOGGLES ===================== #
RUN_MODE = "feasibility"   # "smoke" (local CPU) | "feasibility" (n=64, cheap) | "full" (n=64+256)
RUN_NULL = False           # True = shuffled-corpus null control
# ======================================================== #

import json, os, sys, time, datetime, traceback, urllib.request
import numpy as np

SEED0 = 1234
_T0 = time.time()
def _log(p, m): print(f"[{time.time()-_T0:6.1f}s][{p}] {m}", flush=True)
def _now(): return datetime.datetime.now().isoformat(timespec="seconds")
_log("setup", f"Stage-B starting | RUN_MODE={RUN_MODE} RUN_NULL={RUN_NULL}")

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _log("deps", f"torch {torch.__version__} ok")
except Exception as e:
    print("[deps][FATAL] torch missing. Pick a GPU runtime (Colab/Kaggle ships torch).", e); raise

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_log("setup", f"device={DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else " (CPU -> slow; enable GPU accelerator)"))


# ----- arc cert_inf ONLY (O(n^2) closed form; identical to HD-1) ----- #
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

def project_gamma(decay, W):
    """Largest gamma in [0,1] s.t. cert_inf(decay, gamma*W) passes (bisection).
    gamma=0 passes iff max_i decay_i < 1 — GUARANTEED by the strict affine decay reparam in
    StageBLM.core() (decay <= 1-1e-6; red-team fix: float32 sigmoid saturates to exactly 1.0,
    which would make every gamma infeasible). Caller still re-verifies defensively."""
    if cert_inf(decay, W): return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if cert_inf(decay, mid * W): lo = mid
        else: hi = mid
    return lo * 0.999  # safety margin inside the certified region

def empirical_rho(decay, W, n_samples, seed=0):
    """From-below sup of rho(J) over the (s,x) box (sampled estimator, disclosed in pre-reg)."""
    decay, W = _clip(decay, W); n = decay.shape[0]; rng = np.random.default_rng(seed); mx = 0.0
    for k in range(n_samples):
        s = rng.uniform(-1, 1, n); x = rng.uniform(-1, 1, n)
        t = 1.0 - np.tanh(W @ s + x) ** 2
        J = np.diag(decay) + np.diag((1.0 - decay) * t) @ W
        mx = max(mx, float(np.max(np.abs(np.linalg.eigvals(J)))))
    return mx

def rho_samples_for(n):
    return 1500 if n <= 16 else (600 if n <= 64 else 250)


# ----- corpus (tiny-shakespeare char-level; self-contained; same as HD-1) ----- #
_TS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
def load_corpus(max_chars):
    try:
        txt = urllib.request.urlopen(_TS_URL, timeout=30).read().decode("utf-8", "ignore")
        _log("corpus", f"downloaded tiny-shakespeare ({len(txt)} chars), using {min(len(txt), max_chars)}")
    except Exception as e:
        txt = ("To be, or not to be, that is the question. " * 8000)
        _log("corpus", f"download FAILED ({e}); offline fallback")
    return txt[:max_chars]

def make_data(max_chars, null):
    txt = load_corpus(max_chars); chars = sorted(set(txt)); vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in txt], dtype=np.int64)
    if null:
        np.random.default_rng(999).shuffle(ids); _log("corpus", "NULL: corpus shuffled")
    cut = int(len(ids) * 0.9); return ids[:cut], ids[cut:], vocab

def batches(ids, T, B, rng):
    ix = rng.integers(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[i:i + T] for i in ix]); y = np.stack([ids[i + 1:i + 1 + T] for i in ix])
    return torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE)

def unigram_ce(tr, va, vocab):
    c = np.bincount(tr, minlength=vocab).astype(float) + 1.0; p = c / c.sum()
    return float(-np.mean(np.log(p[va[:-1]])))


# ----- model: windowed-attention Transformer (+ optional verified recurrent channel) ----- #
class WindowedBlock(nn.Module):
    """Pre-LN Transformer block: causal sliding-window softmax MHA + FFN."""
    def __init__(self, d, heads, w_att, T):
        super().__init__(); self.h = heads; self.dk = d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        i = torch.arange(T).view(-1, 1); j = torch.arange(T).view(1, -1)
        allowed = (j <= i) & (i - j < w_att)                      # causal sliding window
        mask = torch.zeros(T, T).masked_fill(~allowed, float("-inf"))
        self.register_buffer("mask", mask, persistent=False)      # additive mask (version-robust)
    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / self.dk ** 0.5 + self.mask[:T, :T]
        y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(y)
        return x + self.ff(self.ln2(x))

def _recur(decay, W, xc):
    B, T, n = xc.shape; s = torch.zeros(B, n, device=xc.device); outs = []
    for t in range(T):
        s = decay * s + (1 - decay) * torch.tanh(s @ W.T + xc[:, t]); outs.append(s)
    return torch.stack(outs, 1)

class StageBLM(nn.Module):
    def __init__(self, vocab, cfg, hybrid):
        super().__init__(); self.hybrid = hybrid; n, d, T = cfg["n"], cfg["d"], cfg["T"]
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(T, d))
        self.blocks = nn.ModuleList([WindowedBlock(d, cfg["heads"], cfg["w_att"], T)
                                     for _ in range(cfg["blocks"])])
        self.ln_f = nn.LayerNorm(d)
        self.readout = nn.Linear(d, vocab)
        if hybrid:  # built LAST so core params never perturb the shared trunk's RNG stream
                    # (red-team B-G1 fix: pure & hybrid now share bit-identical emb/blocks/readout init)
            self.U = nn.Linear(d, n); self.P = nn.Linear(n, d)
            self.raw_decay = nn.Parameter(torch.randn(n) * 0.5 + 1.0)
            self.raw_W = nn.Parameter(torch.randn(n, n) * (0.3 / n ** 0.5))
    def core(self):
        # strict decay in (1e-6, 1-1e-6): float32 sigmoid saturates to EXACTLY 1.0 (red-team
        # soundness fix) which would empty the certified region (gamma=0 infeasible at decay=1).
        d = torch.sigmoid(self.raw_decay) * (1.0 - 2e-6) + 1e-6
        return d, 2.0 * torch.tanh(self.raw_W)
    def core_np(self):
        d, W = self.core(); return d.detach().cpu().numpy(), W.detach().cpu().numpy()
    def set_core_np(self, decay, W):
        with torch.no_grad():
            decay = np.clip(decay, 1e-6, 1 - 1e-6); W = np.clip(W, -2 + 1e-6, 2 - 1e-6)
            p = np.clip((decay - 1e-6) / (1.0 - 2e-6), 1e-9, 1 - 1e-9)   # invert the affine (float64)
            self.raw_decay.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32, device=DEVICE))
            self.raw_W.copy_(torch.atanh(torch.tensor(W / 2.0, dtype=torch.float32, device=DEVICE)))
    def forward(self, idx):
        B, T = idx.shape
        h = self.emb(idx) + self.pos[:T]
        for blk in self.blocks: h = blk(h)
        if self.hybrid:
            decay, W = self.core()
            xc = torch.tanh(self.U(h))                            # |xc|<=1 => cert max_input_abs=1.0 sound
            h = h + self.P(_recur(decay, W, xc))
        return self.readout(self.ln_f(h))

@torch.no_grad()
def eval_ce(model, ids, T, B, n_batches, rng):
    model.eval(); tot = 0.0
    for _ in range(n_batches):
        x, y = batches(ids, T, B, rng); lg = model(x)
        tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1)).item()
    model.train(); return tot / n_batches


# ----- one training run under a condition ----- #
def train_run(cond, seed, cfg, data):
    tr, va, vocab = data; T = cfg["T"]
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    m = StageBLM(vocab, cfg, hybrid=(cond != "pure")).to(DEVICE)
    n_params = sum(p.numel() for p in m.parameters())

    if cond != "pure":   # certified init for ALL hybrid conds -> none/project/reject share the
                         # exact same starting core (red-team fix: isolates regime, not init scale)
        for _ in range(50):
            d_, W_ = m.core_np()
            if cert_inf(d_, W_): break
            with torch.no_grad(): m.raw_W.mul_(0.5)

    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    steps = cfg["grad_steps"]; ce_every = cfg.get("cert_every", 4)
    rejects = checks = projections = fallbacks = 0; gammas = []
    if cond == "reject":
        prev = (m.raw_decay.detach().clone(), m.raw_W.detach().clone())

    def _reset_core_opt_state():
        # symmetric Adam-moment reset after any out-of-band core mutation (red-team fix:
        # stale momentum after revert/projection would otherwise bias the B-Q2 comparison)
        for p in (m.raw_decay, m.raw_W):
            st = opt.state.get(p)
            if st:
                if "exp_avg" in st: st["exp_avg"].zero_()
                if "exp_avg_sq" in st: st["exp_avg_sq"].zero_()

    for it in range(steps):
        x, y = batches(tr, T, cfg["B"], rng); opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1)); loss.backward(); opt.step()
        # project & reject share the SAME cadence (red-team fix: cadence asymmetry would confound
        # the B-Q2 mechanism comparison) — certified at every check incl. the final step.
        if cond == "project" and ((it + 1) % ce_every == 0 or it == steps - 1):
            checks += 1; d_, W_ = m.core_np()
            g = project_gamma(d_, W_)
            if g < 1.0:
                projections += 1; gammas.append(g); m.set_core_np(d_, g * W_)
                d2, W2 = m.core_np()
                if not cert_inf(d2, W2):   # defensive; unreachable after the strict decay reparam
                    fallbacks += 1
                    m.set_core_np(np.full(cfg["n"], 0.7), np.zeros((cfg["n"], cfg["n"])))
                _reset_core_opt_state()
        elif cond == "reject" and ((it + 1) % ce_every == 0 or it == steps - 1):
            checks += 1; d_, W_ = m.core_np()
            if not cert_inf(d_, W_):
                with torch.no_grad(): m.raw_decay.copy_(prev[0]); m.raw_W.copy_(prev[1])
                rejects += 1; _reset_core_opt_state()
            else:
                prev = (m.raw_decay.detach().clone(), m.raw_W.detach().clone())

    ce = eval_ce(m, va, T, cfg["B"], cfg["eval_batches"], np.random.default_rng(seed + 7))
    met = {"ce": ce, "n_params": int(n_params)}
    if cond != "pure":
        d_, W_ = m.core_np(); ns = rho_samples_for(cfg["n"])
        met["max_emp_rho"] = empirical_rho(d_, W_, n_samples=ns, seed=seed + 50)
        met["cert_pass_final"] = cert_inf(d_, W_)
        met["sound"] = bool(met["max_emp_rho"] < 1.0)
    if cond == "project":
        met["project_rate"] = projections / max(1, checks)
        met["mean_gamma"] = float(np.mean(gammas)) if gammas else 1.0
        met["fallbacks"] = fallbacks
    if cond == "reject":
        met["reject_rate"] = rejects / max(1, checks)
    if cond == "none":                                            # B-Q4: post-hoc projection price
        d_, W_ = m.core_np(); g = project_gamma(d_, W_)
        m.set_core_np(d_, g * W_)
        met["postproject_gamma"] = g
        met["ce_postproject"] = eval_ce(m, va, T, cfg["B"], cfg["eval_batches"], np.random.default_rng(seed + 7))
        d2, W2 = m.core_np()
        met["rho_postproject"] = empirical_rho(d2, W2, n_samples=rho_samples_for(cfg["n"]), seed=seed + 51)
    return met


def main():
    if RUN_MODE == "smoke":
        NS = [16]; seeds = [SEED0]
        base_cfg = dict(d=32, T=32, B=4, heads=2, w_att=4, blocks=2, lr=3e-3,
                        grad_steps=8, eval_batches=2, max_chars=4000, cert_every=4)
    elif RUN_MODE == "feasibility":
        NS = [64]; seeds = [SEED0 + i for i in range(2)]
        base_cfg = dict(d=128, T=160, B=24, heads=4, w_att=8, blocks=2, lr=3e-3,
                        grad_steps=300, eval_batches=8, max_chars=100000, cert_every=4)
    else:  # full
        NS = [64, 256]; seeds = [SEED0 + i for i in range(4)]
        base_cfg = dict(d=128, T=160, B=24, heads=4, w_att=8, blocks=2, lr=3e-3,
                        grad_steps=1200, eval_batches=8, max_chars=300000, cert_every=4)
    conds = ["pure", "none", "project", "reject"]
    rpath = f"result_stageb{'_null' if RUN_NULL else ''}.json"

    data = make_data(base_cfg["max_chars"], RUN_NULL); tr, va, vocab = data
    uni = unigram_ce(tr, va, vocab)
    _log("run", f"vocab={vocab} unigram_CE={uni:.4f} NS={NS} conds={conds} seeds={len(seeds)}")

    out = {"meta": {"experiment": "R-LLM Stage-B verified core in gradient-trained Transformer",
                    "mode": RUN_MODE, "null": RUN_NULL, "device": DEVICE,
                    "gpu": (torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"),
                    "NS": NS, "conds": conds, "n_seeds": len(seeds), "base_cfg": base_cfg,
                    "vocab": vocab, "unigram_ce": uni, "start_time": _now(), "end_time": None,
                    "status": "running", "error": None}, "records": []}
    done = set()
    if os.path.exists(rpath):
        try:
            prior = json.load(open(rpath)); out["records"] = prior.get("records", [])
            done = {(r["n"], r["cond"], r["seed"]) for r in out["records"] if r.get("status") == "ok"}
            if done: _log("run", f"resume: {len(done)} records done -> skip")
        except Exception: pass
    def save(): json.dump(out, open(rpath, "w"), indent=2)

    try:
        for n in NS:
            cfg = dict(base_cfg); cfg["n"] = n
            for cond in conds:
                for seed in seeds:
                    if (n, cond, seed) in done: continue
                    rec = {"n": n, "cond": cond, "seed": seed, "start_time": _now(),
                           "status": "running", "metrics": {}, "error": None}
                    try:
                        met = train_run(cond, seed, cfg, data)
                        rec["metrics"] = met; rec["status"] = "ok"
                        extra = ""
                        if "max_emp_rho" in met: extra += f" rho={met['max_emp_rho']:.3f}"
                        if "reject_rate" in met: extra += f" rej={met['reject_rate']:.2f}"
                        if "project_rate" in met: extra += f" proj={met['project_rate']:.2f} mg={met['mean_gamma']:.3f}"
                        if "ce_postproject" in met: extra += f" ce_pp={met['ce_postproject']:.4f} (g={met['postproject_gamma']:.3f})"
                        _log("run", f"n={n:4d} cond={cond:8s} seed={seed} ce={met['ce']:.4f}{extra}")
                    except Exception as ex:
                        rec["status"] = "error"; rec["error"] = f"{type(ex).__name__}: {ex}"
                        _log("run", f"n={n} cond={cond} seed={seed} ERROR: {rec['error']}")
                    rec["end_time"] = _now(); out["records"].append(rec); save()
        out["meta"]["status"] = "done"
    except Exception as ex:
        out["meta"]["status"] = "fatal"
        out["meta"]["error"] = "".join(traceback.format_exception_only(type(ex), ex)).strip()
    finally:
        out["meta"]["end_time"] = _now(); save()

    print("\n[saved] " + rpath)
    print(f"=== Stage-B summary (held-out CE, lower=better; unigram={uni:.4f}; status={out['meta']['status']}) ===")
    ok = [r for r in out["records"] if r["status"] == "ok"]
    print(f"{'n':>5s} {'cond':8s} {'CE':>8s} {'rho':>7s} {'ce_pp':>8s} {'rate':>6s}")
    for n in NS:
        for cond in conds:
            rs = [r["metrics"] for r in ok if r["n"] == n and r["cond"] == cond]
            if not rs: continue
            mean = lambda k: float(np.mean([x[k] for x in rs if k in x])) if any(k in x for x in rs) else float("nan")
            rate = mean("reject_rate") if cond == "reject" else (mean("project_rate") if cond == "project" else float("nan"))
            print(f"{n:5d} {cond:8s} {mean('ce'):8.4f} {mean('max_emp_rho'):7.3f} {mean('ce_postproject'):8.4f} {rate:6.2f}")
    print(f"[done] {len(ok)}/{len(out['records'])} ok in {time.time()-_T0:.0f}s -> {rpath}")
    if RUN_MODE == "smoke": print("SMOKE_OK" if len(ok) == len(out["records"]) and ok else "SMOKE_FAIL")


if __name__ == "__main__":
    main()
