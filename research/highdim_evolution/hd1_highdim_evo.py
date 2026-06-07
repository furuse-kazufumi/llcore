# SPDX-License-Identifier: Apache-2.0
# HD-1 - GPU: what does evolution of the verified-core do at HIGH dimension, UNRESTRICTED?
#
# Derived from research/verifier_navigability_gpu/bg10_gpu_lm.py (same self-contained, checkpointed,
# Kaggle/Colab-ready GPU scaffold). Question the user posed (2026-06-06): "what happens to evolution at
# higher dimension without restriction?" — a regime CPU could not reach (the sound 2-norm/SDP certifiers
# enumerate 2^n t-box vertices, infeasible past n~16; speed also walls it).
#
# KEY DESIGN NOTE: the substrate s_t = decay*s + (1-decay)*tanh(W s + x) is tanh-bounded, so |s|<1 ALWAYS
# — "unrestricted" evolution never NaNs. What is at stake is the CONTRACTION / echo-state property
# (rho(J)<1 = fading memory / homeostasis), not boundedness. So HD-1 asks, as n grows:
#   (1) does UNRESTRICTED (gate="none") evolution drift to rho>=1 (lose the echo-state property)?
#   (2) does it find LOWER held-out CE by going expansive — i.e. is the contraction gate a help or a
#       handicap at scale? (= the L2 "gate load-bearing" question, at high n)
#   (3) the only SOUND gate that SCALES is cert_inf (O(n^2)); 2-norm/SDP are 2^n => NEVER called here.
#       How restrictive (admit-rate) and how costly (CE gap) is the cheap inf gate as n grows?
#
# gates = ["none", "inf"] ONLY. n in {8,32,64,128(,256)}. Output: result_hd1[_null].json (resumable).
from __future__ import annotations

# ===================== USER TOGGLES ===================== #
RUN_MODE = "feasibility"   # "feasibility" (cheap, n<=64) | "full" (n<=128/256, more seeds/steps)
RUN_NULL = False           # True = shuffled-corpus null control
# ======================================================== #

import json, os, sys, time, datetime, traceback, urllib.request
import numpy as np

SEED0 = 1234
_T0 = time.time()
def _log(p, m): print(f"[{time.time()-_T0:6.1f}s][{p}] {m}", flush=True)
def _now(): return datetime.datetime.now().isoformat(timespec="seconds")
_log("setup", f"HD-1 starting | RUN_MODE={RUN_MODE} RUN_NULL={RUN_NULL}")

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _log("deps", f"torch {torch.__version__} ok")
except Exception as e:
    print("[deps][FATAL] torch missing. Pick a GPU runtime (Colab/Kaggle ships torch).", e); raise

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_log("setup", f"device={DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else " (CPU -> slow; enable GPU accelerator)"))


# ----- arc cert_inf ONLY (O(n^2) closed form; SCALES). cert_two/sdp (2^n) intentionally absent. ----- #
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
    off = absW.sum(axis=1) - absW[diag_idx, diag_idx]         # row off-diag abs-sum, vectorized
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

def gate_pass(name, decay, W):
    if name == "none": return True
    if name == "inf":  return cert_inf(decay, W)
    raise ValueError(name)

def empirical_rho(decay, W, n_samples, seed=0):
    """From-below sup of rho(J) over the (s,x) box. n_samples capped by caller for high n."""
    decay, W = _clip(decay, W); n = decay.shape[0]; rng = np.random.default_rng(seed); mx = 0.0
    for k in range(n_samples):
        s = rng.uniform(-1, 1, n); x = rng.uniform(-1, 1, n)
        t = 1.0 - np.tanh(W @ s + x) ** 2
        J = np.diag(decay) + np.diag((1.0 - decay) * t) @ W
        mx = max(mx, float(np.max(np.abs(np.linalg.eigvals(J)))))
    return mx

def rho_samples_for(n):
    return 1500 if n <= 16 else (600 if n <= 64 else 250)


# ----- corpus (tiny-shakespeare char-level; self-contained) ----- #
_TS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
def load_corpus(max_chars):
    try:
        txt = urllib.request.urlopen(_TS_URL, timeout=30).read().decode("utf-8", "ignore")
        _log("corpus", f"downloaded tiny-shakespeare ({len(txt)} chars), using {min(len(txt), max_chars)}")
    except Exception as e:
        txt = ("To be, or not to be, that is the question. " * 4000)
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


# ----- model: same gated-recurrent LM; n parameterized (high-dim) ----- #
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
    def core(self, li): return torch.sigmoid(self.raw_decay[li]), 2.0 * torch.tanh(self.raw_W[li])
    def core_np(self, li):
        d, W = self.core(li); return d.detach().cpu().numpy(), W.detach().cpu().numpy()
    def set_core_np(self, li, decay, W):
        with torch.no_grad():
            decay = np.clip(decay, 1e-6, 1 - 1e-6); W = np.clip(W, -2 + 1e-6, 2 - 1e-6)
            self.raw_decay[li].copy_(torch.tensor(np.log(decay / (1 - decay)), dtype=torch.float32, device=DEVICE))
            self.raw_W[li].copy_(torch.atanh(torch.tensor(W / 2.0, dtype=torch.float32, device=DEVICE)))
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


# ----- regimes ----- #
def train_grad(gate, seed, cfg, data):
    tr, va, vocab = data; T = cfg["T"]
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    m = GatedRecurrentLM(vocab, cfg["n"], cfg["layers"], cfg["d"]).to(DEVICE)
    for li in range(cfg["layers"]):
        for _ in range(50):
            d_, W_ = m.core_np(li)
            if gate_pass(gate, d_, W_): break
            with torch.no_grad(): m.raw_W[li].mul_(0.5)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"]); rejects = 0; steps = cfg["grad_steps"]
    prev = [(m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone()) for li in range(cfg["layers"])]
    ce_every = cfg.get("cert_every", 4)
    for it in range(steps):
        x, y = batches(tr, T, cfg["B"], rng); opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1)); loss.backward(); opt.step()
        if gate != "none" and ((it + 1) % ce_every == 0 or it == steps - 1):
            for li in range(cfg["layers"]):
                d_, W_ = m.core_np(li)
                if not gate_pass(gate, d_, W_):
                    with torch.no_grad(): m.raw_decay[li].copy_(prev[li][0]); m.raw_W[li].copy_(prev[li][1])
                    rejects += 1
                else:
                    prev[li] = (m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone())
    ce = eval_ce(m, va, T, cfg["B"], cfg["eval_batches"], np.random.default_rng(seed + 7))
    ns = rho_samples_for(cfg["n"])
    rho = max(empirical_rho(*m.core_np(li), n_samples=ns, seed=seed + li) for li in range(cfg["layers"]))
    return {"ce": ce, "reject_rate": rejects / max(1, steps * cfg["layers"]), "max_emp_rho": rho}

def _warm_base(gate, seed, cfg, data, vocab):
    tr, va, _ = data; T = cfg["T"]
    torch.manual_seed(seed + 5); rng = np.random.default_rng(seed + 5)
    m = GatedRecurrentLM(vocab, cfg["n"], cfg["layers"], cfg["d"]).to(DEVICE)
    for li in range(cfg["layers"]):
        for _ in range(50):
            d_, W_ = m.core_np(li)
            if gate_pass(gate, d_, W_): break
            with torch.no_grad(): m.raw_W[li].mul_(0.5)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"]); ws = cfg["grad_steps"] // 2
    prev = [(m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone()) for li in range(cfg["layers"])]
    ce_every = cfg.get("cert_every", 4)
    for it in range(ws):
        x, y = batches(tr, T, cfg["B"], rng); opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1)); loss.backward(); opt.step()
        if gate != "none" and ((it + 1) % ce_every == 0 or it == ws - 1):
            for li in range(cfg["layers"]):
                d_, W_ = m.core_np(li)
                if not gate_pass(gate, d_, W_):
                    with torch.no_grad(): m.raw_decay[li].copy_(prev[li][0]); m.raw_W[li].copy_(prev[li][1])
                else:
                    prev[li] = (m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone())
    for p in [m.emb.weight, m.readout.weight, m.readout.bias]: p.requires_grad_(False)
    return m

def evolve_core(gate, seed, cfg, data, base):
    tr, va, vocab = data; T = cfg["T"]; rng = np.random.default_rng(seed + 100); m = base
    def fit(cores):
        for li, (d_, W_) in enumerate(cores): m.set_core_np(li, d_, W_)
        return -eval_ce(m, va, T, cfg["B"], cfg["eval_batches"], np.random.default_rng(seed + 7))
    cur = []
    for li in range(cfg["layers"]):
        d0, W0 = m.core_np(li)
        if not gate_pass(gate, d0, W0): d0, W0 = np.full(cfg["n"], 0.7), np.zeros((cfg["n"], cfg["n"]))
        cur.append((d0, W0))
    best = fit(cur); admit = tries = 0
    for g in range(cfg["evo_gens"]):
        cand = [(np.clip(d_ + rng.normal(0, cfg["sigma"], d_.shape), 0, 1),
                 np.clip(W_ + rng.normal(0, cfg["sigma"], W_.shape), -2, 2)) for (d_, W_) in cur]
        tries += 1
        if all(gate_pass(gate, d_, W_) for (d_, W_) in cand):
            admit += 1; f = fit(cand)
            if f > best: best, cur = f, cand
    ns = rho_samples_for(cfg["n"])
    rho = max(empirical_rho(d_, W_, n_samples=ns, seed=seed + 50 + li) for li, (d_, W_) in enumerate(cur))
    return {"ce": -best, "admit_rate": admit / max(1, tries), "max_emp_rho": rho}


def main():
    feasibility = (RUN_MODE == "feasibility")
    if feasibility:
        NS = [8, 32, 64]; seeds = [SEED0 + i for i in range(2)]
        base_cfg = dict(layers=1, d=64, T=64, B=16, lr=3e-3, grad_steps=150, evo_gens=80,
                        sigma=0.12, eval_batches=4, max_chars=40000, cert_every=4)
    else:  # full
        NS = [8, 32, 64, 128, 256]; seeds = [SEED0 + i for i in range(4)]
        base_cfg = dict(layers=1, d=96, T=64, B=24, lr=3e-3, grad_steps=400, evo_gens=150,
                        sigma=0.12, eval_batches=6, max_chars=80000, cert_every=4)
    gates = ["none", "inf"]
    rpath = f"result_hd1{'_null' if RUN_NULL else ''}.json"

    data = make_data(base_cfg["max_chars"], RUN_NULL); tr, va, vocab = data
    uni = unigram_ce(tr, va, vocab)
    _log("run", f"vocab={vocab} unigram_CE={uni:.4f} NS={NS} gates={gates} seeds={len(seeds)}")

    out = {"meta": {"experiment": "HD-1 high-dim unrestricted evolution", "mode": RUN_MODE,
                    "null": RUN_NULL, "device": DEVICE,
                    "gpu": (torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"),
                    "NS": NS, "gates": gates, "n_seeds": len(seeds), "base_cfg": base_cfg,
                    "vocab": vocab, "unigram_ce": uni, "start_time": _now(), "end_time": None,
                    "status": "running", "error": None}, "records": []}
    done = set()
    if os.path.exists(rpath):
        try:
            prior = json.load(open(rpath)); out["records"] = prior.get("records", [])
            done = {(r["n"], r["gate"], r["seed"]) for r in out["records"] if r.get("status") == "ok"}
            if done: _log("run", f"resume: {len(done)} records done -> skip")
        except Exception: pass
    def save(): json.dump(out, open(rpath, "w"), indent=2)

    try:
        for n in NS:
            cfg = dict(base_cfg); cfg["n"] = n
            for gate in gates:
                for seed in seeds:
                    if (n, gate, seed) in done: continue
                    rec = {"n": n, "gate": gate, "seed": seed, "start_time": _now(),
                           "status": "running", "metrics": {}, "error": None}
                    try:
                        g = train_grad(gate, seed, cfg, data)
                        e = evolve_core(gate, seed, cfg, data, _warm_base(gate, seed, cfg, data, vocab))
                        rec["metrics"] = {"grad_ce": g["ce"], "grad_reject_rate": g["reject_rate"],
                                          "grad_max_emp_rho": g["max_emp_rho"], "evo_ce": e["ce"],
                                          "evo_admit_rate": e["admit_rate"], "evo_max_emp_rho": e["max_emp_rho"],
                                          "grad_sound": bool(g["max_emp_rho"] < 1.0),
                                          "evo_sound": bool(e["max_emp_rho"] < 1.0)}
                        rec["status"] = "ok"
                        _log("run", f"n={n:4d} gate={gate:4s} seed={seed} GRAD ce={g['ce']:.4f} "
                                    f"rho={g['max_emp_rho']:.3f} rej={g['reject_rate']:.2f} | "
                                    f"EVO ce={e['ce']:.4f} rho={e['max_emp_rho']:.3f} admit={e['admit_rate']:.2f}")
                    except Exception as ex:
                        rec["status"] = "error"; rec["error"] = f"{type(ex).__name__}: {ex}"
                        _log("run", f"n={n} gate={gate} seed={seed} ERROR: {rec['error']}")
                    rec["end_time"] = _now(); out["records"].append(rec); save()
        out["meta"]["status"] = "done"
    except Exception as ex:
        out["meta"]["status"] = "fatal"
        out["meta"]["error"] = "".join(traceback.format_exception_only(type(ex), ex)).strip()
    finally:
        out["meta"]["end_time"] = _now(); save()

    print("\n[saved] " + rpath)
    print(f"=== HD-1 summary (held-out CE, lower=better; unigram={uni:.4f}; status={out['meta']['status']}) ===")
    ok = [r for r in out["records"] if r["status"] == "ok"]
    print(f"{'n':>5s} {'gate':5s} {'GRAD_ce':>8s} {'GRAD_rho':>8s} {'EVO_ce':>8s} {'EVO_rho':>8s} {'EVO_admit':>9s}")
    for n in NS:
        for gate in gates:
            rs = [r["metrics"] for r in ok if r["n"] == n and r["gate"] == gate]
            if not rs: continue
            mean = lambda k: float(np.mean([x[k] for x in rs]))
            print(f"{n:5d} {gate:5s} {mean('grad_ce'):8.4f} {mean('grad_max_emp_rho'):8.3f} "
                  f"{mean('evo_ce'):8.4f} {mean('evo_max_emp_rho'):8.3f} {mean('evo_admit_rate'):9.3f}")
    print(f"[done] {len(ok)}/{len(out['records'])} ok in {time.time()-_T0:.0f}s -> {rpath}")


if __name__ == "__main__":
    main()
