# BG10 — GPU stage A PRE-REGISTRATION (navigability + gradient-vs-evolution on a REAL recurrent LM)

> **Billing gate (CLAUDE.md).** This is the ONLY billable thread. Charge target = **cloud GPU compute
> time only** (Claude API + local CPU = $0). Plan = **free-first (Colab/Kaggle T4) → paid-fallback
> (RunPod/Vast RTX4090) only if free-tier limits bite**. Same code both. No spend or launch without the
> user's explicit GO at each gate (S2 smoke, S4 full). Expected: $0 on free tier; ≤$40 cap (exp $3–8)
> if paid.
>
> **Why GPU (what CPU cannot do).** CPU established (`../verifier_navigability/`) that *random-mutation
> evolution* is trapped by an over-conservative-but-sound gate (navigability, not ceiling). The two things
> only a GPU can add: (1) a **real gradient-trained** sequence model (backprop needs GPU at useful scale),
> to ask whether **gradient escapes the navigability trap** that evolution falls into; (2) a model large
> enough (multi-layer, learned projections, longer context, bigger corpus) that the answer is not a
> tiny-n artifact. This is the full-scale version of the navigability thread's N5 (optimizer-independence).

## 0. One-line questions (falsifiable)

**Q-NAV (scale):** Does the navigability trap survive on a real, multi-layer, gradient-trained recurrent
LM — i.e. under *evolution of the verified core*, does the inf gate still trap far below a ceiling it
contains, with gap(inf) > gap(sdp)?

**Q-GRAD (the killer):** Does **gradient** (backprop, which follows the loss surface) **escape** the trap
that random mutation falls into — i.e. does inf-gated *gradient* training reach the inf-region ceiling
where inf-gated *evolution* could not? YES ⇒ navigability is an **EA/random-search property, not a
gradient property** (huge, honest, reframes the verifier-payoff story for real LMs). NO ⇒ the trap is
**fundamental to the feasible-set geometry**, independent of optimizer.

**Q-PAYOFF (real LM):** For a real **gradient-trained** recurrent LM, does a sound verifier
(sdp) help / hurt / not-matter for held-out perplexity vs inf vs none — and does the **null control**
(shuffled corpus) tie at scale (closing the structure-independence question the CPU null left open)?

## 1. Model (real, GPU-trained, small enough that n-state certs stay tractable)

A **gated linear-recurrent byte/char LM** (tiny Mamba/RWKV/linear-attention family):
```
tokens → learned embedding (GPU, gradient) → [ L stacked layers:
            verified recurrent state-mix: s_t = decay⊙s_{t-1} + (1-decay)⊙tanh(W s_{t-1} + U x_t)
            (W,decay = CoupledNDGene per layer; STATE dim n kept small=8–16 so cert_two/sdp tractable)
            + learned in/out projections U, P (GPU, gradient) + MLP + residual + norm ] × L
         → learned readout (GPU, gradient) → softmax CE (real held-out perplexity)
```
- The **evolvable + contraction-verified** part = the per-layer `(decay, W)` core (small n). Everything
  else (embedding, U, P, MLP, readout, ≥2 layers, real corpus, longer context) is **real gradient-trained**.
- Certifiers reused **unchanged** from `../verified_evolution_sdp_gate/coupled_nd.py` (cert_inf/two/sdp,
  CLARABEL-pinned). n≤16 keeps 2ⁿ vertex enumeration tractable; n>16 is the deferred vertex-free
  certifier R-LLM-1 (out of scope, soundness-first). The soundness contract (|input|<1 via tanh/norm so
  `max_input_abs=1`) is enforced and re-derived for the projected input U·x.

## 2. Two optimizer regimes under each gate (the core comparison)

For gate ∈ {none, inf, two, sdp}:
- **EVO:** train embedding/U/P/MLP/readout by gradient (fixed core), but the **core (decay,W) is EVOLVED**
  (random mutation, gate-filtered admission) — the CPU regime, now with a real gradient-trained wrapper.
- **GRAD:** the **core is also gradient-trained**, kept in the gate's feasible set by either (a) projection
  after each step (project (decay,W) to the nearest cert-passing point — or reject-and-halve the step), or
  (b) a differentiable soundness penalty (e.g. inf-norm/SDP-surrogate contraction penalty) — both reported.

Q-GRAD = does GRAD@inf reach the inf-region ceiling that EVO@inf cannot?

## 3. Pre-registered FALSIFIABLE gates

| gate | claim | PASS condition |
|---|---|---|
| **G0 realize** | the verified-core recurrent LM trains on GPU and beats unigram; a no-verifier real baseline (small Transformer) bounds the gap | held-out CE < unigram by a clear margin; baseline reported |
| **G1 soundness@scale** | gate-admitted/projected cores are empirically contracting on the real corpus (0 divergence), null-vacuity avoided (ungated cores diverge) | per-layer empirical ρ<1 for all admitted; >0 ungated divergent |
| **G2 Q-NAV** | EVO navigability trap survives at scale: gap(inf) > gap(two) > gap(sdp), ≥8 seeds | monotone, bootstrap-separated; gap = ceiling(random-sample of core) − reached(EVO) |
| **G3 Q-GRAD** | gradient escapes (or not) | GRAD@inf reached CE vs EVO@inf reached CE: ESCAPE if GRAD@inf ≈ inf-ceiling ≫ EVO@inf (paired, ≥8 seeds); FUNDAMENTAL if GRAD@inf still ≈ EVO@inf |
| **G4 Q-PAYOFF + null** | real-LM verifier payoff + structure-dependence | (a) sdp vs inf vs none on held-out CE for GRAD-trained model; (b) **null (shuffled corpus): gates must tie** (p>0.1) — the clean closure of the CPU §3c open question |
| **G5 honest-negative is OK** | any of "trap is fundamental", "gradient escapes", "verifier doesn't matter for gradient LMs", "null ties / doesn't" are all valid recorded outcomes | reported without distortion |

## 4. Cost / staged gates (matches the S0–S5 procedure)

- **S2 smoke ($0 free / ~$2 paid):** 1 layer, n=8, 1 seed, tiny corpus, few min — validate pipeline +
  positive control (ungated diverges; sdp admits) + that the question is non-vacuous. GO/NO-GO.
- **S4 full ($0 free / ≤$40 cap, exp $3–8):** L=2–4, n=8–16, {none,inf,two,sdp} × {EVO,GRAD} × ≥8 seeds,
  real corpus + shuffled null. Fits a free Colab/Kaggle T4 in ~1–3 h (checkpoint per seed so a session
  timeout loses nothing).
- All seeds fixed; CLARABEL pinned; 0-unsound re-confirmed; per-gate Codex pair-review before verdict.

## 5. Adversarial red-team (post-measurement)
- projection vs penalty give the same G3 verdict? (else it's a method artifact)
- GRAD "escape" not just from the gradient-trained *wrapper* compensating (freeze wrapper, vary only core)
- null tie checked under BOTH optimizers
- ceiling estimated by a fair random sample of cert-passing cores at the SAME n/corpus (no corpus mismatch — the §3c lesson)
- small-Transformer baseline is fairly tuned (not a strawman) so "real LM" is credible

## 6. Deliverable for S0 (now, $0)
`bg10_gpu_lm.py` — one self-contained, Colab/Kaggle-pasteable script: inlines the cert functions (no repo
needed), builds the model, runs `--smoke` (S2) or `--full` (S4), checkpoints per seed, exports results
JSON. Plus a 6-line run recipe for Colab free / Kaggle free / RunPod.
