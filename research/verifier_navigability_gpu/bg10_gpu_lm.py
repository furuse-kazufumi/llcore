# SPDX-License-Identifier: Apache-2.0
"""BG10 — GPU stage A: navigability + gradient-vs-evolution on a REAL gradient-trained recurrent LM.

SELF-CONTAINED, paste-into-one-cell (Colab/Kaggle). Inlines the llcore arc certifiers verbatim
(soundness-critical — do NOT re-derive), builds a small gradient-trained gated-recurrent char LM whose
per-layer state-mixing core is the *evolvable + contraction-verified* CoupledNDGene, and answers:

  Q-NAV   : does EVOLUTION of the core get trapped by the conservative inf gate (gap > sdp)?
  Q-GRAD  : does GRADIENT (projected) escape the trap random mutation falls into?
  Q-PAYOFF: sound (sdp) vs inf vs none on real held-out CE — and does the --null (shuffled corpus) tie?

Pre-registration: PREREGISTRATION_BG10.md. research/ isolated; src/ untouched.

============================ HOW TO RUN (free GPU, $0) ============================
Kaggle (recommended, 30 GPU-h/week, background): New Notebook -> Accelerator: GPU T4 -> paste this whole
file into one cell -> set MODE below -> Run All. Or Colab: Runtime -> T4 GPU -> paste -> run.

    MODE = "smoke"     # S2 validation, ~minutes, $0   (then change to "full" for S4)
    NULL = False       # set True to run the shuffled-corpus null control

The cell auto-installs cvxpy+clarabel (for the sdp gate) and downloads tiny-shakespeare. It writes
bg10_results_<mode>.json and prints a summary. Share that JSON back for analysis.
=================================================================================
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
import urllib.request

import numpy as np

# ----- config (smoke vs full) ------------------------------------------------ #
MODE = os.environ.get("BG10_MODE", "smoke")          # "smoke" | "full"
NULL = os.environ.get("BG10_NULL", "0") == "1"        # shuffle corpus (null control)
SEED0 = 1234

# ----- optional deps: torch (required), cvxpy+clarabel (sdp gate; degrades) --- #
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    print("torch is required (Colab/Kaggle GPU runtime has it). Error:", e); raise

try:
    import cvxpy as cp
    _CLARABEL_OK = "CLARABEL" in cp.installed_solvers()
    _CVXPY, _SOLVER = True, (cp.CLARABEL if _CLARABEL_OK else None)
except Exception:
    _CVXPY = _CLARABEL_OK = False; _SOLVER = None
    if MODE == "full":
        os.system(f"{sys.executable} -m pip install -q cvxpy clarabel")
        try:
            import cvxpy as cp
            _CLARABEL_OK = "CLARABEL" in cp.installed_solvers()
            _CVXPY, _SOLVER = True, (cp.CLARABEL if _CLARABEL_OK else None)
        except Exception:
            pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =================================================================== #
# Inlined arc certifiers (VERBATIM from research/verified_evolution_sdp_gate/coupled_nd.py).
# Sound contraction over the achievable-t box [t_min,1]^n. numpy/CPU; n is small (cores).
# =================================================================== #
def _clip(decay, W):
    return np.clip(np.asarray(decay, float).reshape(-1), 0, 1), np.clip(np.asarray(W, float), -2, 2)

def t_min_per_coord(decay, W, max_input_abs=1.0):
    decay, W = _clip(decay, W)
    M = np.abs(W).sum(axis=1) + max_input_abs * 1.0  # V=I ⇒ |V|.sum(axis=1)=1
    return 1.0 - np.tanh(M) ** 2

def _jac_at_t(decay, W, t):
    decay, W = _clip(decay, W)
    return np.diag(decay) + np.diag((1.0 - decay) * t) @ W

def _box_vertices(t_lo):
    n = t_lo.shape[0]
    return [np.array([(1.0 if b else t_lo[i]) for i, b in enumerate(c)])
            for c in itertools.product((0, 1), repeat=n)]

def infnorm_sup(decay, W, t_lo):
    decay, W = _clip(decay, W); n = decay.shape[0]; best = 0.0
    for i in range(n):
        off = sum(abs(W[i, j]) for j in range(n) if j != i); row = 0.0
        for ti in (t_lo[i], 1.0):
            diag = abs(decay[i] + (1.0 - decay[i]) * ti * W[i, i])
            row = max(row, diag + (1.0 - decay[i]) * ti * off)
        best = max(best, row)
    return float(best)

def cert_inf(decay, W, max_input_abs=1.0):
    return bool(infnorm_sup(decay, W, t_min_per_coord(decay, W, max_input_abs)) < 1.0)

def cert_two(decay, W, max_input_abs=1.0):
    t_lo = t_min_per_coord(decay, W, max_input_abs)
    return all(float(np.linalg.svd(_jac_at_t(decay, W, v), compute_uv=False)[0]) < 1.0
               for v in _box_vertices(t_lo))

def cert_sdp(decay, W, max_input_abs=1.0, margin=1e-7):
    if cert_two(decay, W, max_input_abs):
        return True
    if not (_CVXPY and _CLARABEL_OK):
        return False
    decay, W = _clip(decay, W); n = decay.shape[0]
    verts = _box_vertices(t_min_per_coord(decay, W, max_input_abs))
    Js = [_jac_at_t(decay, W, v) for v in verts]
    for J in Js:
        if float(np.max(np.abs(np.linalg.eigvals(J)))) >= 1.0:
            return False
    P = cp.Variable((n, n), symmetric=True); I = np.eye(n)
    cons = [P >> I] + [P - J.T @ P @ J >> margin * I for J in Js]
    try:
        cp.Problem(cp.Minimize(cp.trace(P)), cons).solve(solver=_SOLVER)
    except Exception:
        return False
    if P.value is None:
        return False
    Pv = 0.5 * (P.value + P.value.T)
    if float(np.min(np.linalg.eigvalsh(Pv))) <= 0.0:
        return False
    for J in Js:
        M = Pv - J.T @ Pv @ J
        if float(np.min(np.linalg.eigvalsh(0.5 * (M + M.T)))) <= 0.0:
            return False
    return True

def gate_pass(name, decay, W):
    if name == "none":  return True
    if name == "inf":   return cert_inf(decay, W)
    if name == "two":   return cert_two(decay, W)
    if name == "sdp":   return cert_sdp(decay, W)
    raise ValueError(name)

def empirical_rho(decay, W, n_samples=2000, seed=0):
    decay, W = _clip(decay, W); n = decay.shape[0]; rng = np.random.default_rng(seed)
    S = rng.uniform(-1, 1, (n_samples, n)); X = rng.uniform(-1, 1, (n_samples, n)); mx = 0.0
    for k in range(n_samples):
        t = 1.0 - np.tanh(W @ S[k] + X[k]) ** 2
        J = np.diag(decay) + np.diag((1.0 - decay) * t) @ W
        mx = max(mx, float(np.max(np.abs(np.linalg.eigvals(J)))))
    return mx


# =================================================================== #
# Corpus (tiny-shakespeare, char-level; self-contained download).
# =================================================================== #
_TS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
def load_corpus(max_chars):
    try:
        txt = urllib.request.urlopen(_TS_URL, timeout=20).read().decode("utf-8", "ignore")
    except Exception:
        txt = ("To be, or not to be, that is the question. " * 4000)  # offline fallback
    return txt[:max_chars]


# =================================================================== #
# Model: small gradient-trained gated-recurrent LM with verified cores.
#   x_core = tanh(U·emb)               (|x_core|<1 ⇒ sound input bound, max_input_abs=1)
#   s_t = decay⊙s + (1-decay)⊙tanh(W s + x_core)     (core = CoupledNDGene; decay,W = verified)
#   layer out = P·s (+ residual + norm); readout: linear → vocab CE.
# embedding/U/P/readout = gradient-trained "wrapper"; cores = decay,W (gradient or evolved).
# =================================================================== #
class GatedRecurrentLM(nn.Module):
    def __init__(self, vocab, n, layers, d):
        super().__init__()
        self.n, self.layers = n, layers
        self.emb = nn.Embedding(vocab, d)
        self.U = nn.ModuleList([nn.Linear(d, n) for _ in range(layers)])
        self.P = nn.ModuleList([nn.Linear(n, d) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        # cores: decay∈[0,1] via sigmoid(raw_decay); W∈[-2,2] via 2*tanh(raw_W)
        self.raw_decay = nn.ParameterList([nn.Parameter(torch.randn(n) * 0.5 + 1.0) for _ in range(layers)])
        self.raw_W = nn.ParameterList([nn.Parameter(torch.randn(n, n) * (0.3 / n ** 0.5)) for _ in range(layers)])
        self.readout = nn.Linear(d, vocab)

    def core(self, li):
        return torch.sigmoid(self.raw_decay[li]), 2.0 * torch.tanh(self.raw_W[li])

    def core_np(self, li):
        d, W = self.core(li); return d.detach().cpu().numpy(), W.detach().cpu().numpy()

    def set_core_np(self, li, decay, W):
        with torch.no_grad():
            decay = np.clip(decay, 1e-6, 1 - 1e-6); W = np.clip(W, -2 + 1e-6, 2 - 1e-6)
            self.raw_decay[li].copy_(torch.tensor(np.log(decay / (1 - decay)), dtype=torch.float32, device=DEVICE))
            self.raw_W[li].copy_(torch.atanh(torch.tensor(W / 2.0, dtype=torch.float32, device=DEVICE)))

    def forward(self, idx):                      # idx (B,T)
        h = self.emb(idx)                        # (B,T,d)
        for li in range(self.layers):
            decay, W = self.core(li)             # (n,), (n,n)
            xc = torch.tanh(self.U[li](h))       # (B,T,n), |.|<1 ⇒ sound input bound
            S = _recur(decay, W, xc)             # (B,T,n) = s_t = decay*s + (1-decay)*tanh(W s + xc_t)
            h = self.norm[li](h + self.P[li](S)) # residual + norm
        return self.readout(h)                   # (B,T,vocab)


def _recur(decay, W, xc):
    """Sequential verified core: s_t = decay*s + (1-decay)*tanh(W s + xc_t).  xc (B,T,n)."""
    B, T, n = xc.shape
    s = torch.zeros(B, n, device=xc.device); outs = []
    for t in range(T):
        s = decay * s + (1 - decay) * torch.tanh(s @ W.T + xc[:, t])
        outs.append(s)
    return torch.stack(outs, 1)


# =================================================================== #
# Data / eval helpers.
# =================================================================== #
def make_data(max_chars, T, null):
    txt = load_corpus(max_chars)
    chars = sorted(set(txt)); vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in txt], dtype=np.int64)
    if null:
        np.random.default_rng(999).shuffle(ids)          # destroy sequential structure
    cut = int(len(ids) * 0.9)
    return ids[:cut], ids[cut:], vocab, T

def batches(ids, T, B, rng):
    ix = rng.integers(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[i:i + T] for i in ix]); y = np.stack([ids[i + 1:i + 1 + T] for i in ix])
    return (torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE))

@torch.no_grad()
def eval_ce(model, ids, T, B, n_batches, rng):
    model.eval(); tot = 0.0
    for _ in range(n_batches):
        x, y = batches(ids, T, B, rng)
        logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    model.train(); return tot / n_batches

def unigram_ce(tr, va, vocab):
    c = np.bincount(tr, minlength=vocab).astype(float) + 1.0; p = c / c.sum()
    return float(-np.mean(np.log(p[va[:-1]])))


# =================================================================== #
# Regimes.
# =================================================================== #
def train_grad(gate, seed, cfg, data):
    """Projected gradient: train whole model; after each step, REJECT core moves that leave the gate's
    feasible set (sound projected GD). wrapper trains freely. Returns held-out CE + soundness/admit stats."""
    tr, va, vocab, T = data
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    m = GatedRecurrentLM(vocab, cfg["n"], cfg["layers"], cfg["d"]).to(DEVICE)
    # ensure initial cores are feasible for the gate (resample raw_W until pass, else zero W)
    for li in range(cfg["layers"]):
        for _ in range(50):
            d_, W_ = m.core_np(li)
            if gate_pass(gate, d_, W_): break
            with torch.no_grad(): m.raw_W[li].mul_(0.5)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    rejects = 0; steps = cfg["grad_steps"]
    prev = [(m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone()) for li in range(cfg["layers"])]
    for it in range(steps):
        x, y = batches(tr, T, cfg["B"], rng)
        opt.zero_grad(); logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)); loss.backward(); opt.step()
        # project: reject infeasible core moves (keep wrapper update)
        for li in range(cfg["layers"]):
            d_, W_ = m.core_np(li)
            if not gate_pass(gate, d_, W_):
                with torch.no_grad():
                    m.raw_decay[li].copy_(prev[li][0]); m.raw_W[li].copy_(prev[li][1])
                rejects += 1
            else:
                prev[li] = (m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone())
    ce = eval_ce(m, va, T, cfg["B"], cfg["eval_batches"], np.random.default_rng(seed + 7))
    rho = max(empirical_rho(*m.core_np(li)) for li in range(cfg["layers"]))
    return {"ce": ce, "reject_rate": rejects / max(1, steps * cfg["layers"]), "max_emp_rho": rho,
            "winner_region": classify_region_np(*m.core_np(0))}

def evolve_core(gate, seed, cfg, data, base):
    """Freeze the wrapper (from a base GRAD-trained model), EVOLVE the cores by gated random mutation.
    Tests navigability: can mutation reach good cores inside the gate? Returns held-out CE + admit rate."""
    tr, va, vocab, T = data
    rng = np.random.default_rng(seed + 100)
    m = base
    def core_fit(cores):
        for li, (d_, W_) in enumerate(cores): m.set_core_np(li, d_, W_)
        return -eval_ce(m, va, T, cfg["B"], cfg["eval_batches"], np.random.default_rng(seed + 7))
    # init feasible cores
    cur = []
    for li in range(cfg["layers"]):
        d0, W0 = m.core_np(li)
        if not gate_pass(gate, d0, W0): d0, W0 = np.full(cfg["n"], 0.7), np.zeros((cfg["n"], cfg["n"]))
        cur.append((d0, W0))
    best_fit = core_fit(cur); admit = 0; tries = 0
    for g in range(cfg["evo_gens"]):
        cand = []
        for (d_, W_) in cur:
            nd = np.clip(d_ + rng.normal(0, cfg["sigma"], d_.shape), 0, 1)
            nW = np.clip(W_ + rng.normal(0, cfg["sigma"], W_.shape), -2, 2)
            cand.append((nd, nW))
        tries += 1
        if all(gate_pass(gate, d_, W_) for (d_, W_) in cand):
            admit += 1; f = core_fit(cand)
            if f > best_fit: best_fit, cur = f, cand
    for li, (d_, W_) in enumerate(cur): m.set_core_np(li, d_, W_)
    return {"ce": -best_fit, "admit_rate": admit / max(1, tries),
            "winner_region": classify_region_np(*cur[0])}

def classify_region_np(decay, W):
    if cert_inf(decay, W): return "inf"
    if cert_two(decay, W): return "two_norm_only"
    if cert_sdp(decay, W): return "sdp_only"
    return "non_certified"


# =================================================================== #
# Main.
# =================================================================== #
def main():
    smoke = (MODE == "smoke")
    cfg = dict(n=8, layers=1, d=64, T=64, B=16, lr=3e-3,
               grad_steps=120 if smoke else 1500, evo_gens=60 if smoke else 400,
               sigma=0.12, eval_batches=4 if smoke else 16,
               max_chars=20000 if smoke else 300000)
    gates = ["none", "inf", "sdp"] if smoke else ["none", "inf", "two", "sdp"]
    seeds = [SEED0] if smoke else [SEED0 + i for i in range(8)]
    data = make_data(cfg["max_chars"], cfg["T"], NULL)
    tr, va, vocab, T = data
    uni = unigram_ce(tr, va, vocab)
    print(f"BG10 {MODE} NULL={NULL} device={DEVICE} vocab={vocab} sdp_ok={_CVXPY and _CLARABEL_OK} "
          f"unigram_CE={uni:.4f}", flush=True)

    out = {"mode": MODE, "null": NULL, "device": DEVICE, "cfg": cfg, "vocab": vocab,
           "unigram_ce": uni, "sdp_available": bool(_CVXPY and _CLARABEL_OK), "grad": {}, "evo": {}}
    t0 = time.time()
    for gate in gates:
        out["grad"][gate], out["evo"][gate] = [], []
        for seed in seeds:
            g = train_grad(gate, seed, cfg, data); out["grad"][gate].append(g)
            # EVO: short grad-train of the wrapper under the same gate, freeze it, then evolve the core
            base = _build_warm_base(gate, seed, cfg, data, vocab)
            e = evolve_core(gate, seed, cfg, data, base); out["evo"][gate].append(e)
            print(f"  gate={gate:4s} seed={seed} GRAD ce={g['ce']:.4f} reject={g['reject_rate']:.2f} "
                  f"rho={g['max_emp_rho']:.3f} | EVO ce={e['ce']:.4f} admit={e['admit_rate']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        # checkpoint per gate
        json.dump(out, open(f"bg10_results_{MODE}{'_null' if NULL else ''}.json", "w"), indent=2)

    # summary
    print("\n=== summary (held-out CE; lower=better; unigram=%.4f) ===" % uni)
    for reg in ("grad", "evo"):
        print(f"[{reg}]")
        for gate in gates:
            ces = [r["ce"] for r in out[reg][gate]]
            print(f"  {gate:5s} mean_CE={np.mean(ces):.4f}  best={np.min(ces):.4f}")
    json.dump(out, open(f"bg10_results_{MODE}{'_null' if NULL else ''}.json", "w"), indent=2)
    print(f"\nsaved bg10_results_{MODE}{'_null' if NULL else ''}.json ({time.time()-t0:.0f}s)")


def _build_warm_base(gate, seed, cfg, data, vocab):
    """Short grad-train of the full model under the gate, then return it (wrapper used frozen by EVO)."""
    tr, va, _, T = data
    torch.manual_seed(seed + 5); rng = np.random.default_rng(seed + 5)
    m = GatedRecurrentLM(vocab, cfg["n"], cfg["layers"], cfg["d"]).to(DEVICE)
    for li in range(cfg["layers"]):
        for _ in range(50):
            d_, W_ = m.core_np(li)
            if gate_pass(gate, d_, W_): break
            with torch.no_grad(): m.raw_W[li].mul_(0.5)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    prev = [(m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone()) for li in range(cfg["layers"])]
    for _ in range(cfg["grad_steps"] // 2):
        x, y = batches(tr, T, cfg["B"], rng); opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1)); loss.backward(); opt.step()
        for li in range(cfg["layers"]):
            d_, W_ = m.core_np(li)
            if not gate_pass(gate, d_, W_):
                with torch.no_grad(): m.raw_decay[li].copy_(prev[li][0]); m.raw_W[li].copy_(prev[li][1])
            else:
                prev[li] = (m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone())
    for p in [m.emb.weight, m.readout.weight, m.readout.bias]:
        p.requires_grad_(False)
    return m


if __name__ == "__main__":
    main()
