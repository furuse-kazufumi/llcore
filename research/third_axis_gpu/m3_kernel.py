# SPDX-License-Identifier: Apache-2.0
# M3 (③本丸) - GPU: is QD/behavioral niching (MAP-Elites) load-bearing on a REAL LLM loss landscape?
#
# Derived from ../rllm_stage_b/stage_b_kernel.py (same self-contained, checkpointed, resumable
# Kaggle scaffold). Pre-registration: PREREGISTRATION_M3.md (verdict rules + controls fixed before
# any GPU run). BG9 structural hypothesis: 3 wins only when the difficulty lives in HIGH-DIM
# behavior space whose good region is measure-zero under direct sampling (BG9_VERDICT.md S2/S4).
#
# Substrate: Stage-B hybrid char-LM (windowed attention trunk, gradient-warmed then FROZEN);
# genome = core (decay, W) = n^2+n dims (n=64 -> 4,160 continuous dims). Fitness = -held-out CE on
# FIXED eval batches (CRN; trunk outputs h and core input xc are PRECOMPUTED once per seed, so one
# fitness eval = recurrence + readout only).
#
# Methods (equal eval budget E): M1 random search / M2 RR-hillclimb (restart=E/10) / M3 panmictic GA
# (pop16, tournament-2, mutation-only) / M4 MAP-Elites 8x8 (descriptors B1 dynamics, B2 functional)
# / M5 Adam-on-core (E/3 steps, FLOP-matched approx; reported, not a 3 judge).
# Controls: P+ = Step4-transplant deceptive corridor on mean(W) (validates the harness; MAP-E must
# beat M1/M2/M3 or verdict = N/A) / N0 = smooth concave (all tie; MAP-E "win" = false positive).
from __future__ import annotations

# ===================== USER TOGGLES ===================== #
RUN_MODE = "feasibility"   # "smoke" (local CPU) | "feasibility" (preview) | "full" (verdict)
# ======================================================== #

import json, os, sys, time, datetime, traceback, urllib.request, zlib
import numpy as np

SEED0 = 1234
_T0 = time.time()
def _log(p, m): print(f"[{time.time()-_T0:6.1f}s][{p}] {m}", flush=True)
def _now(): return datetime.datetime.now().isoformat(timespec="seconds")
_log("setup", f"M3 starting | RUN_MODE={RUN_MODE}")

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _log("deps", f"torch {torch.__version__} ok")
except Exception as e:
    print("[deps][FATAL] torch missing.", e); raise

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_log("setup", f"device={DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else " (CPU)"))


# ----- cert/rho metrics (identical to HD-1/Stage-B; metrics only, NOT gating) ----- #
def _clip(decay, W):
    return np.clip(np.asarray(decay, float).reshape(-1), 0, 1), np.clip(np.asarray(W, float), -2, 2)

def t_min_per_coord(decay, W, max_input_abs=1.0):
    decay, W = _clip(decay, W)
    M = np.abs(W).sum(axis=1) + max_input_abs * 1.0
    return 1.0 - np.tanh(M) ** 2

def infnorm_sup(decay, W, t_lo):
    decay, W = _clip(decay, W); n = decay.shape[0]
    absW = np.abs(W); diag_idx = np.arange(n)
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
    for k in range(n_samples):
        s = rng.uniform(-1, 1, n); x = rng.uniform(-1, 1, n)
        t = 1.0 - np.tanh(W @ s + x) ** 2
        J = np.diag(decay) + np.diag((1.0 - decay) * t) @ W
        mx = max(mx, float(np.max(np.abs(np.linalg.eigvals(J)))))
    return mx


# ----- corpus (tiny-shakespeare; identical to HD-1/Stage-B) ----- #
_TS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
def load_corpus(max_chars):
    try:
        txt = urllib.request.urlopen(_TS_URL, timeout=30).read().decode("utf-8", "ignore")
        _log("corpus", f"downloaded tiny-shakespeare ({len(txt)} chars), using {min(len(txt), max_chars)}")
    except Exception as e:
        txt = ("To be, or not to be, that is the question. " * 8000)
        _log("corpus", f"download FAILED ({e}); offline fallback")
    return txt[:max_chars]

def make_data(max_chars):
    txt = load_corpus(max_chars); chars = sorted(set(txt)); vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in txt], dtype=np.int64)
    cut = int(len(ids) * 0.9); return ids[:cut], ids[cut:], vocab

def batches_with_pos(ids, T, B, rng):
    ix = rng.integers(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[i:i + T] for i in ix]); y = np.stack([ids[i + 1:i + 1 + T] for i in ix])
    octile = np.minimum((ix * 8) // max(1, (len(ids) - T - 1)), 7)   # source-position octile (B2)
    return (torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE),
            torch.tensor(octile, device=DEVICE))

def unigram_ce(tr, va, vocab):
    c = np.bincount(tr, minlength=vocab).astype(float) + 1.0; p = c / c.sum()
    return float(-np.mean(np.log(p[va[:-1]])))


# ----- model (identical structure to Stage-B; red-team fixes inherited) ----- #
class WindowedBlock(nn.Module):
    def __init__(self, d, heads, w_att, T):
        super().__init__(); self.h = heads; self.dk = d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        i = torch.arange(T).view(-1, 1); j = torch.arange(T).view(1, -1)
        allowed = (j <= i) & (i - j < w_att)
        mask = torch.zeros(T, T).masked_fill(~allowed, float("-inf"))
        self.register_buffer("mask", mask, persistent=False)
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

def _recur_t(decay_t, W_t, xc):
    B, T, n = xc.shape; s = torch.zeros(B, n, device=xc.device); outs = []
    for t in range(T):
        s = decay_t * s + (1 - decay_t) * torch.tanh(s @ W_t.T + xc[:, t]); outs.append(s)
    return torch.stack(outs, 1)

class HybridLM(nn.Module):
    def __init__(self, vocab, cfg):
        super().__init__(); n, d, T = cfg["n"], cfg["d"], cfg["T"]
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(T, d))
        self.blocks = nn.ModuleList([WindowedBlock(d, cfg["heads"], cfg["w_att"], T)
                                     for _ in range(cfg["blocks"])])
        self.ln_f = nn.LayerNorm(d)
        self.readout = nn.Linear(d, vocab)
        self.U = nn.Linear(d, n); self.P = nn.Linear(n, d)     # core params built LAST (RNG order)
        self.raw_decay = nn.Parameter(torch.randn(n) * 0.5 + 1.0)
        self.raw_W = nn.Parameter(torch.randn(n, n) * (0.3 / n ** 0.5))
    def core_np(self):
        d = (torch.sigmoid(self.raw_decay) * (1.0 - 2e-6) + 1e-6).detach().cpu().numpy()
        W = (2.0 * torch.tanh(self.raw_W)).detach().cpu().numpy()
        return d, W
    def trunk(self, idx):
        B, T = idx.shape
        h = self.emb(idx) + self.pos[:T]
        for blk in self.blocks: h = blk(h)
        return h
    def forward(self, idx):
        h = self.trunk(idx)
        decay = torch.sigmoid(self.raw_decay) * (1.0 - 2e-6) + 1e-6
        W = 2.0 * torch.tanh(self.raw_W)
        xc = torch.tanh(self.U(h))
        h = h + self.P(_recur_t(decay, W, xc))
        return self.readout(self.ln_f(h))


# ----- genome ops (numpy; boxes match HD-1: decay [0,1], W [-2,2]) ----- #
def sample_genome(n, rng):
    decay = 1.0 / (1.0 + np.exp(-(rng.normal(0, 0.5, n) + 1.0)))   # sigmoid(randn*0.5+1)
    W = 2.0 * np.tanh(rng.normal(0, 0.3 / n ** 0.5, (n, n)))
    return decay, W

def mutate(g, sigma, rng):
    d, W = g
    return (np.clip(d + rng.normal(0, sigma, d.shape), 0, 1),
            np.clip(W + rng.normal(0, sigma, W.shape), -2, 2))


# ----- fitness: REAL (frozen-trunk precompute) ----- #
class RealFitness:
    """One eval = recurrence + readout over precomputed (h, xc, y). Deterministic (fixed batches)."""
    def __init__(self, model, va, cfg, seed):
        self.m = model; self.cfg = cfg
        rng = np.random.default_rng(seed + 7)
        hs, xcs, ys, octs = [], [], [], []
        with torch.no_grad():
            for _ in range(cfg["eval_batches"]):
                x, y, oc = batches_with_pos(va, cfg["T"], cfg["B"], rng)
                h = model.trunk(x)
                hs.append(h); xcs.append(torch.tanh(model.U(h))); ys.append(y); octs.append(oc)
        self.h, self.xc, self.y, self.oct = hs, xcs, ys, octs
        self.calls = 0
    def __call__(self, genome, need_b2=False, grad_tensors=None):
        """Returns (ce, shard8) with shard8 = per-source-octile CE (B2 raw vector)."""
        self.calls += 1
        if grad_tensors is None:
            d_np, W_np = genome
            decay = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)
            W = torch.tensor(W_np, dtype=torch.float32, device=DEVICE)
            ctx = torch.no_grad()
        else:
            decay, W = grad_tensors
            import contextlib; ctx = contextlib.nullcontext()
        tot, shard_sum, shard_cnt = 0.0, np.zeros(8), np.zeros(8)
        loss_acc = None
        with ctx:
            for h, xc, y, oc in zip(self.h, self.xc, self.y, self.oct):
                s = _recur_t(decay, W, xc)
                lg = self.m.readout(self.m.ln_f(h + self.m.P(s)))
                ce_tok = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1),
                                         reduction="none").view(y.shape)
                ce_b = ce_tok.mean(dim=1)                         # per-sample CE
                tot += float(ce_b.mean().item())
                if grad_tensors is not None:
                    loss_acc = ce_b.mean() if loss_acc is None else loss_acc + ce_b.mean()
                if need_b2:
                    for k in range(8):
                        mk = (oc == k)
                        if mk.any():
                            shard_sum[k] += float(ce_b[mk].mean().item()); shard_cnt[k] += 1
        ce = tot / len(self.h)
        if grad_tensors is not None:
            return ce, loss_acc / len(self.h)
        if not need_b2:
            return ce, None
        present = shard_cnt > 0                                   # red-team fix: empty octile would
        shard8 = np.zeros(8)                                      # read as CE=0 (dominating outlier)
        shard8[present] = shard_sum[present] / shard_cnt[present]
        if not present.all():
            shard8[~present] = shard8[present].mean()             # impute cross-octile mean
        return ce, shard8


# ----- fixed low-dim probe slices + unit-window mapping (red-team blocker fix, take 2) ----- #
# Two requirements verified by CPU simulation before any GPU run:
#   (i) behavior axes must be MUTATION-MOBILE: per-step drift >= ~1/4 bin width. Full-4096-mean
#       coordinates are CLT-frozen (drift sigma/64); even a /4-window 24-slice was too slow AND a
#       sloped deceptive valley breaks MAP-E's within-bin ratchet (validated FAIL 0/4).
#  (ii) the corridor must be Step4's PROVEN shape (exp2_highdim_deceptive.deceptive_eval): fitness =
#       max(broad local Gaussian, narrow global corner Gaussian) — outside the local basin the
#       GLOBAL tail supplies an outward within-bin gradient, which is what makes the archive ratchet
#       work. A V-shaped valley (take 1) pulls elites back inward and kills the ratchet.
# Unit window u = clip(w + 0.5, 0, 1): per-entry mutation noise 0.12 maps 1:1 (Step4's scale);
# behavior = means over 10-entry halves (drift 0.038/step ~= bin/3.3); init behavior std ~0.047 so
# the (1,1) corner is ~10.6 sigma of SAMPLING away (teleport-proof) yet stepping-stone-reachable.
def _u20(W):
    u = np.clip(np.asarray(W).reshape(-1)[:20] + 0.5, 0, 1)
    return u

def behavior20(genome):
    u = _u20(genome[1])
    return (float(np.mean(u[:10])), float(np.mean(u[10:])))


# ----- synthetic control fitnesses (no model; same genome space) ----- #
def fitness_pplus(genome):
    """Step4 exp2 deceptive_eval base (noiseless), faithfully transplanted:
    broad local optimum at behavior (0.3,0.3), narrow global at the (1,1) corner.
    NOTE: the run-time fitness ADDS Step4's observation noise N(0, 0.01) (suite closure) — the
    noise is load-bearing: on the flat stretch between basins the Gaussian tails are << 0.01, so
    noise lets place-if-better churn neutrally and the archive random-walks outward (the ratchet).
    Recorded bests are honest-rescored with THIS noiseless base (Step4's honest_n_trials analogue)."""
    b = np.array(behavior20(genome))
    local = 0.60 * np.exp(-np.sum((b - np.array([0.3, 0.3])) ** 2) / (2 * 0.18 ** 2))
    glob = 1.00 * np.exp(-np.sum((b - np.array([1.0, 1.0])) ** 2) / (2 * 0.07 ** 2))
    return float(max(local, glob))

PPLUS_NOISE = 0.01   # Step4 exp2 _NOISE

def fitness_n0(genome, n):
    """Smooth strictly-concave; unimodal at (0.7, 0)."""
    d, W = genome
    return -float(np.mean((d - 0.7) ** 2) + np.mean(W ** 2))


# ----- behavior descriptors (axes mutation-mobile by construction; see pre-reg table) ----- #
def desc_b1(genome, _shard=None):
    d, W = genome
    return (float(np.clip(np.mean(d[:16]), 0, 1)),               # decay probe: drift 0.03/step
            float(np.clip(np.mean(np.asarray(W).reshape(-1)[:10]) + 0.5, 0, 1)))  # drift 0.038/step

def desc_b1p(genome, _shard=None):
    return behavior20(genome)                                    # corridor-aligned (P+ only)

_B2_PROJ = None
def desc_b2(genome, shard8):
    global _B2_PROJ
    if _B2_PROJ is None:
        _B2_PROJ = np.random.default_rng(777).normal(0, 1, (8, 2)) / np.sqrt(8)
    c = shard8 - shard8.mean()
    z = c / max(float(c.std()), 1e-3)                            # std-adaptive (red-team fix)
    v = np.tanh(z @ _B2_PROJ)
    return (float((v[0] + 1) / 2), float((v[1] + 1) / 2))

def bin_of(desc, bins=8):
    return (min(bins - 1, max(0, int(desc[0] * bins))), min(bins - 1, max(0, int(desc[1] * bins))))


# ----- methods (equal eval budget E; fit(genome)->(fitness, shard8)) ----- #
def run_m1_random(fit, n, E, warm, rng):
    best, bg = -1e18, None
    for i in range(E):
        g = warm if i == 0 else sample_genome(n, rng)
        f, _ = fit(g)
        if f > best: best, bg = f, g
    return best, bg

def run_m2_rr(fit, n, E, warm, rng, sigma):
    """Warm-restart RR (red-team fix): restart = re-perturb the WARM core (3*sigma) instead of a
    cold random genome — on the real landscape a cold restart is a liability, not the BG9
    teleport; warm-restart keeps M2 the fair strong direct-sampling baseline."""
    R = max(1, E // 10)
    best, bg = -1e18, None
    cur = warm; fcur, _ = fit(cur); used = 1
    if fcur > best: best, bg = fcur, cur
    while used < E:
        if used % R == 0:
            cur = mutate(warm, 3.0 * sigma, rng); fcur, _ = fit(cur); used += 1
            if fcur > best: best, bg = fcur, cur
            continue
        cand = mutate(cur, sigma, rng); f, _ = fit(cand); used += 1
        if f >= fcur: cur, fcur = cand, f
        if f > best: best, bg = f, cand
    return best, bg

def run_m3_ga(fit, n, E, warm, rng, sigma, pop=16):
    P = [warm] + [mutate(warm, sigma, rng) for _ in range(pop - 1)]
    fits = []
    for g in P:
        f, _ = fit(g); fits.append(f)
    used = pop; best = max(fits); bg = P[int(np.argmax(fits))]
    while used + pop <= E:
        newP = []
        for _ in range(pop):
            i, j = rng.integers(0, pop, 2)
            parent = P[i] if fits[i] >= fits[j] else P[j]
            newP.append(mutate(parent, sigma, rng))
        newF = []
        for g in newP:
            f, _ = fit(g); newF.append(f)
        used += pop; P, fits = newP, newF
        if max(fits) > best: best, bg = max(fits), P[int(np.argmax(fits))]
    while used < E:   # partial final generation -> exact budget parity (red-team fix)
        i, j = rng.integers(0, pop, 2)
        parent = P[i] if fits[i] >= fits[j] else P[j]
        g = mutate(parent, sigma, rng); f, _ = fit(g); used += 1
        if f > best: best, bg = f, g
    return best, bg

def run_m4_mapelites(fit, n, E, warm, rng, sigma, desc, need_b2):
    archive = {}   # bin -> (fitness, genome)
    def place(g):
        f, shard = fit(g, need_b2=need_b2) if need_b2 else fit(g)
        b = bin_of(desc(g, shard))
        if b not in archive or f > archive[b][0]: archive[b] = (f, g)
        return f, g
    best, bg = -1e18, None
    f, g = place(warm)
    if f > best: best, bg = f, g
    for _ in range(15):
        f, g = place(sample_genome(n, rng))
        if f > best: best, bg = f, g
    used = 16
    while used < E:
        ks = list(archive.keys())
        elite = archive[ks[rng.integers(0, len(ks))]][1]
        f, g = place(mutate(elite, sigma, rng)); used += 1
        if f > best: best, bg = f, g
    return best, bg, len(archive)

def run_m5_grad(rf, model, E, seed, cfg):
    steps = max(1, E // 3)
    torch.manual_seed(seed + 11)
    snap = (model.raw_decay.detach().clone(), model.raw_W.detach().clone())   # restore after (hardening)
    for p in model.parameters(): p.requires_grad_(False)
    model.raw_decay.requires_grad_(True); model.raw_W.requires_grad_(True)
    opt = torch.optim.Adam([model.raw_decay, model.raw_W], lr=cfg["lr"])
    for it in range(steps):
        decay = torch.sigmoid(model.raw_decay) * (1.0 - 2e-6) + 1e-6
        W = 2.0 * torch.tanh(model.raw_W)
        ce, loss = rf(None, grad_tensors=(decay, W))
        opt.zero_grad(); loss.backward()
        if it == 0:
            assert model.raw_W.grad is not None, "M5: autograd graph broken (raw_W.grad is None)"
        opt.step()
    d_np, W_np = model.core_np()
    f, _ = rf((d_np, W_np))
    with torch.no_grad():   # leave shared model at the warm baseline (red-team hardening)
        model.raw_decay.copy_(snap[0]); model.raw_W.copy_(snap[1])
        model.raw_decay.requires_grad_(False); model.raw_W.requires_grad_(False)
    return f, (d_np, W_np), steps


def main():
    if RUN_MODE == "smoke":
        cfg = dict(n=16, d=32, T=32, B=4, heads=2, w_att=4, blocks=2, lr=3e-3,
                   warm_steps=8, eval_batches=2, max_chars=4000, E=64, sigma=0.12, landscape=24)
        seeds = [SEED0]; descriptors = ["B1"]
    elif RUN_MODE == "feasibility":
        cfg = dict(n=64, d=128, T=160, B=24, heads=4, w_att=8, blocks=2, lr=3e-3,
                   warm_steps=300, eval_batches=3, max_chars=100000, E=1504, sigma=0.12, landscape=300)
        seeds = [SEED0 + i for i in range(2)]; descriptors = ["B1"]
    else:  # full
        cfg = dict(n=64, d=128, T=160, B=24, heads=4, w_att=8, blocks=2, lr=3e-3,
                   warm_steps=300, eval_batches=3, max_chars=300000, E=6000, sigma=0.12, landscape=600)
        seeds = [SEED0 + i for i in range(4)]; descriptors = ["B1", "B2"]
    rpath = "result_m3.json"

    tr, va, vocab = make_data(cfg["max_chars"])
    uni = unigram_ce(tr, va, vocab)
    _log("run", f"vocab={vocab} unigram_CE={uni:.4f} n={cfg['n']} E={cfg['E']} seeds={len(seeds)}")

    out = {"meta": {"experiment": "M3 third-axis QD on real LLM core landscape", "mode": RUN_MODE,
                    "device": DEVICE, "gpu": (torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"),
                    "cfg": cfg, "vocab": vocab, "unigram_ce": uni, "descriptors": descriptors,
                    "start_time": _now(), "end_time": None, "status": "running", "error": None},
           "records": []}
    done = set()
    if os.path.exists(rpath):
        try:
            prior = json.load(open(rpath)); out["records"] = prior.get("records", [])
            done = {(r["control"], r["method"], r.get("descriptor", ""), r["seed"])
                    for r in out["records"] if r.get("status") == "ok"}
            if done: _log("run", f"resume: {len(done)} records done -> skip")
        except Exception: pass
    def save(): json.dump(out, open(rpath, "w"), indent=2)

    def record(control, method, descriptor, seed, payload, status="ok", error=None):
        out["records"] = [r for r in out["records"]
                          if (r["control"], r["method"], r.get("descriptor", ""), r["seed"])
                          != (control, method, descriptor, seed)]
        out["records"].append({"control": control, "method": method, "descriptor": descriptor,
                               "seed": seed, "status": status, "error": error,
                               "metrics": payload, "time": _now()})
        save()

    try:
        for seed in seeds:
            # --- per-seed warm model (CRN: same trunk+core init/training for all methods) --- #
            torch.manual_seed(seed); np.random.seed(seed)
            m = HybridLM(vocab, cfg).to(DEVICE)
            opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
            wrng = np.random.default_rng(seed)
            for it in range(cfg["warm_steps"]):
                x, y, _ = batches_with_pos(tr, cfg["T"], cfg["B"], wrng)
                loss = F.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            warm_core = m.core_np()
            rf = RealFitness(m, va, cfg, seed)
            f_warm, _ = rf(warm_core)
            _log("run", f"seed={seed} warm done: warm-core CE={f_warm:.4f} (unigram {uni:.4f})")

            E, n, sigma = cfg["E"], cfg["n"], cfg["sigma"]

            # --- L0 landscape map (user-requested GPU extension; descriptive, no gate) --- #
            # 600 random + 100x4 warm-perturbed (sigma sweep) genomes -> real CE + B1 coords.
            # Mechanism evidence for the verdict: direct-sampling difficulty (frac of random
            # genomes beating warm), local smoothness (perturbation CE curve vs sigma).
            if ("real", "L0", "", seed) not in done and cfg.get("landscape", 0) > 0:
                t0 = time.time(); rngL = np.random.default_rng(seed * 1000 + 424242)
                pts = []
                plan = ([("random", None)] * cfg["landscape"]
                        + [("perturb", s) for s in (0.03, 0.12, 0.5, 1.0)
                           for _ in range(cfg["landscape"] // 6)])
                for kind, s in plan:
                    g = sample_genome(n, rngL) if kind == "random" else mutate(warm_core, s, rngL)
                    ce, _ = rf(g)
                    pts.append({"k": kind, "s": s, "ce": round(ce, 4),
                                "b1": [round(x, 3) for x in desc_b1(g)]})
                rnd = [p["ce"] for p in pts if p["k"] == "random"]
                pay = {"warm_ce": f_warm, "n_points": len(pts),
                       "frac_random_better_than_warm": float(np.mean([c < f_warm for c in rnd])),
                       "random_ce_min": float(np.min(rnd)), "random_ce_median": float(np.median(rnd)),
                       "perturb_ce_median": {str(s): float(np.median(
                           [p["ce"] for p in pts if p["k"] == "perturb" and p["s"] == s]))
                           for s in (0.03, 0.12, 0.5, 1.0)},
                       "points": pts}
                record("real", "L0", "", seed, pay)
                _log("run", f"seed={seed} real  L0 landscape: rand_min={pay['random_ce_min']:.4f} "
                            f"rand_med={pay['random_ce_median']:.4f} warm={f_warm:.4f} "
                            f"frac_better={pay['frac_random_better_than_warm']:.3f} ({time.time()-t0:.0f}s)")

            def fit_real(g, need_b2=False):
                ce, shard = rf(g, need_b2=need_b2)
                return -ce, shard
            def fit_pp(g, need_b2=False): return fitness_pplus(g), None
            def fit_n0(g, need_b2=False): return fitness_n0(g, n), None

            suites = [("real", fit_real), ("pplus", fit_pp), ("n0", fit_n0)]
            for control, fit in suites:
                t0 = time.time()
                for method in ("M1", "M2", "M3", "M4"):
                    descs = (descriptors if (method == "M4" and control == "real")
                             else (["B1p"] if (method == "M4" and control == "pplus")
                                   else ["B1"] if method == "M4" else [""]))
                    for dname in descs:
                        if (control, method, dname, seed) in done: continue
                        rng = np.random.default_rng(   # deterministic across processes (NOT python hash)
                            seed * 1000 + zlib.crc32(f"{control}|{method}|{dname}".encode()) % 9973)
                        try:
                            if method == "M1": best, bg = run_m1_random(fit, n, E, warm_core, rng)
                            elif method == "M2": best, bg = run_m2_rr(fit, n, E, warm_core, rng, sigma)
                            elif method == "M3": best, bg = run_m3_ga(fit, n, E, warm_core, rng, sigma)
                            else:
                                dfun = {"B1": desc_b1, "B1p": desc_b1p, "B2": desc_b2}[dname]
                                best, bg, ncell = run_m4_mapelites(fit, n, E, warm_core, rng, sigma,
                                                                   dfun, need_b2=(dname == "B2"))
                            pay = {"best_fitness": best,
                                   "best_ce": (-best if control == "real" else None),
                                   "cert_inf": cert_inf(*bg),
                                   "emp_rho": empirical_rho(*bg, n_samples=200, seed=seed + 99)}
                            if method == "M4": pay["archive_cells"] = ncell
                            record(control, method, dname, seed, pay)
                            _log("run", f"seed={seed} {control:5s} {method}{('/' + dname) if dname else '':4s}"
                                        f" best={best:.4f}" + (f" ce={-best:.4f}" if control == "real" else ""))
                        except Exception as ex:
                            record(control, method, dname, seed, {}, "error", f"{type(ex).__name__}: {ex}")
                            _log("run", f"seed={seed} {control} {method} ERROR: {ex}")
                # M5 GRAD only on real
                if control == "real" and ("real", "M5", "", seed) not in done:
                    try:
                        ce5, bg, steps = run_m5_grad(rf, m, E, seed, cfg)
                        record(control, "M5", "", seed,
                               {"best_fitness": -ce5,           # fitness = -CE (same sign as M1-M4 real)
                                "best_ce": ce5, "grad_steps": steps,
                                "cert_inf": cert_inf(*bg),
                                "emp_rho": empirical_rho(*bg, n_samples=200, seed=seed + 98)})
                        _log("run", f"seed={seed} real  M5      ce={ce5:.4f} (steps={steps})")
                    except Exception as ex:
                        record(control, "M5", "", seed, {}, "error", f"{type(ex).__name__}: {ex}")
                        _log("run", f"seed={seed} real M5 ERROR: {ex}")
                _log("run", f"seed={seed} suite={control} done in {time.time()-t0:.0f}s (rf calls={rf.calls})")
        out["meta"]["status"] = "done"
    except Exception as ex:
        out["meta"]["status"] = "fatal"
        out["meta"]["error"] = "".join(traceback.format_exception_only(type(ex), ex)).strip()
    finally:
        out["meta"]["end_time"] = _now(); save()

    print("\n[saved] " + rpath)
    ok = [r for r in out["records"] if r["status"] == "ok"]
    print(f"=== M3 summary (status={out['meta']['status']}; real fitness = -held-out CE, unigram={uni:.4f}) ===")
    for control in ("real", "pplus", "n0"):
        for method in ("M1", "M2", "M3", "M4", "M5"):
            rs = [r for r in ok if r["control"] == control and r["method"] == method]
            if not rs: continue
            for dname in sorted({r.get("descriptor", "") for r in rs}):
                rr = [r["metrics"]["best_fitness"] for r in rs if r.get("descriptor", "") == dname
                      and "best_fitness" in r["metrics"]]
                ce = [r["metrics"].get("best_ce") for r in rs if r.get("descriptor", "") == dname]
                ce = [c for c in ce if c is not None]
                if rr:
                    print(f"{control:6s} {method}{('/' + dname) if dname else '':5s} "
                          f"best_fit={np.mean(rr):8.4f}" + (f"  ce={np.mean(ce):7.4f}" if ce else ""))
    print(f"[done] {len(ok)}/{len(out['records'])} ok in {time.time()-_T0:.0f}s -> {rpath}")
    if RUN_MODE == "smoke": print("SMOKE_OK" if len(ok) == len(out["records"]) and ok else "SMOKE_FAIL")


if __name__ == "__main__":
    main()
