# PRE-REGISTRATION — R-LLM Stage-B: the verified core inside a REAL gradient-trained Transformer

> Written 2026-06-06 BEFORE any GPU run (discipline: gates fixed before measurement,
> `feedback_benchmark_honest_disclosure`). Derived scaffold: `../highdim_evolution/hd1_highdim_evo.py`.
> Substrate lineage: `../verified_lm_evolution/PREREGISTRATION.md §1` explicitly deferred
> "softmax-attention の純 Transformer・end-to-end 学習" to GPU stage B — this is that stage.

## 1. Question

L0–L3 established the verified core as a *reservoir* byte-LM (no gradient through the core's
surroundings); BG10 established that *gradient* avoids the EA navigability trap; HD-1 established that
at full budget *ungated gradient itself* leaves the contractive region (entropic, null-confirmed) and
that the sound cheap gate (`cert_inf`) costs real CE — leaving **open whether that cost is lost
expressivity or optimization friction**.

Stage-B asks, on a **real softmax-attention Transformer trained end-to-end** with the verified
recurrent core as its **only long-range memory path**:

- **B-Q1 (load-bearing memory):** does the hybrid (Transformer + verified core) beat the attention-only
  Transformer when attention is windowed (receptive field ≪ context)? I.e. does the core actually carry
  long-range information under end-to-end gradient training?
- **B-Q2 (cost decomposition — HD-1's open question):** with three matched training regimes for the core
  — `none` (unconstrained), `project` (always-certified via deterministic projection, no rejection),
  `reject` (HD-1-style revert-on-cert-failure) — does the gate's CE cost come from the **constraint**
  (expressivity) or the **rejection mechanism** (friction)?
- **B-Q3 (drift with attention):** does the unconstrained core still drift to ρ≥1 when attention shares
  the modeling load (HD-1 said yes when the core is alone)?
- **B-Q4 (price of post-hoc verification):** project the *final* unconstrained core into the certified
  region once after training — how much CE does "train free, certify later" lose vs training-time gating?

## 2. Design

**Model (StageBLM):** char-level LM. emb(d) + learned pos-emb → 2 × pre-LN Transformer blocks
(4-head softmax attention with **causal sliding window w_att=8**, FFN 4d) → recurrent verified channel
`xc=tanh(U h)`, `s_t = decay⊙s_{t-1} + (1−decay)⊙tanh(W s_{t-1} + xc_t)`, `h ← LN(h + P s)` → readout.
Context T=160 ≫ stacked receptive field (≈ 2·(8−1)+1 = 15), so **information beyond ~15 chars can flow
ONLY through the core's state s** — the memory channel is load-bearing by construction.
`pure` = identical minus the U/P/core channel (param counts recorded; not param-matched — disclosed).

**Certified object:** ONLY the core `(decay, W)` (contraction / echo-state of the s-channel, arc
`cert_inf`, O(n²), `max_input_abs=1.0` — sound because `|xc| ≤ 1` by tanh). Attention/embeddings/readout
are never certified. The verified property is the core's homeostasis, nothing more.

**Conditions (core training regime; everything else identical, CRN-paired seeds):**
1. `pure` — no core channel (B-Q1 baseline).
2. `none` — core unconstrained. (+ derived metric: post-hoc projection of the final core, B-Q4.)
3. `project` — every `cert_every=4` steps (**same cadence as `reject`** — red-team fix: cadence
   asymmetry would confound the B-Q2 mechanism comparison), if `cert_inf` fails, scale W by the
   largest γ∈[0,1] (bisection) s.t. `infnorm_sup(decay, γW) < 1` — deterministic, never reverts.
   Certified at every check incl. the final step; between checks the core may transiently leave the
   region (same exposure as `reject`). γ=0 feasibility is guaranteed by a strict decay
   reparametrisation `decay = (1−2e-6)·σ(·)+1e-6` (red-team fix: float32 sigmoid saturates to exactly
   1.0, which would empty the certified region); a defensive post-projection re-check falls back to a
   trivially-certified core (decay=0.7, W=0) and is counted (`fallbacks`, expected 0).
4. `reject` — HD-1 GRAD-gate: every `cert_every=4` steps, if `cert_inf` fails, revert core (only) to
   last passing snapshot.

**Symmetry measures (red-team fixes, all disclosed):** shared trunk modules (emb/pos/blocks/ln_f/
readout) are constructed BEFORE the core params so `pure` and hybrids share bit-identical trunk inits
(pure-vs-hybrid differs ONLY in the extra channel + param count); ALL hybrid conditions get the same
certified-init loop (W halved until cert passes) so `none`/`project`/`reject` start from the exact
same core (caveat: stage-B `none` therefore starts certified, unlike HD-1's raw-init `none` — B-G3
cross-study comparison carries this asymmetry); Adam moments of the core params are reset after any
out-of-band mutation (projection or revert) in BOTH arms — stale momentum would otherwise re-push the
core out of region and bias Δf upward (this also means stage-B `reject` is a *cleaner* revert than
HD-1's, disclosed for cross-comparability).

**Sweep:** n_core ∈ {64, 256} × 4 conditions × 4 seeds (full) = 32 runs; feasibility = n=64 × 2 seeds
= 8 runs. d=128, T=160, B=24, Adam lr 3e-3, steps: 1200 (full) / 300 (feasibility), eval_batches=8,
corpus tiny-shakespeare (full 300 kB / feas 100 kB), held-out = last 10 % (temporal split).
**Null control (shuffled corpus) planned for the full design** — same protocol, answers whether any
gate-cost found is structure-independent (HD-1 precedent).

**Metrics per run:** held-out CE; core `empirical_rho` (sampled-from-below estimator, disclosed);
cert pass at end; reject/project intervention rates; param count; for `none` additionally
`ce_postproject`, `rho_postproject`.

## 3. Pre-registered gates (fixed BEFORE running)

- **B-G1 (PASS if):** mean CE(`none`) < mean CE(`pure`) at every n_core and in ≥ 3/4 seeds (full).
  FAIL ⇒ the hybrid premise is broken at this scale; report honestly, downstream gates still reported
  but flagged as resting on a non-load-bearing channel.
- **B-G2 (decomposition; let Δf = CE(`reject`) − CE(`none`) ≥ 0 be the full gate cost):**
  - *friction-dominated* if CE(`project`) − CE(`none`) < 0.25·Δf (project recovers ≥ 75 % of the cost)
  - *expressivity-dominated* if CE(`project`) − CE(`none`) > 0.75·Δf
  - *mixed* otherwise. (If Δf ≤ 0 at some n: gate-cost absent there; report sign per n.)
- **B-G3 (drift):** fraction of `none` seeds with final core ρ ≥ 1, per n_core, vs HD-1's full-run
  fractions (4/4 at n=256). Prediction registered: attention absorbs modeling pressure ⇒ drift weaker
  than HD-1. (Either outcome informative.)
- **B-G4:** report `ce_postproject` − CE(`none`) vs CE(`project`) − CE(`none`): post-hoc ≤ training-time
  cost ⇒ "certify after" viable at this scale.

Primary comparisons are paired per-seed; with 4 seeds the resolvable effect is coarse — sign consistency
(≥ 3/4) + magnitude reported, no p-value theater at n=4. Secondary: Wilcoxon if ≥ 8 paired points
pooled across n_core.

## 4. Honest limits (declared up front)

- Tiny model (~0.5 M params), char-level, one corpus, T4. We measure **relative** effects between
  matched conditions, not absolute LM quality.
- `pure` has fewer params than hybrids (no U/P/W/decay) — B-G1 is therefore *generous* to the hybrid;
  a hybrid loss to a smaller pure model is a strong negative signal, a narrow win is weak evidence.
- `empirical_rho` is a sampled lower estimator of the sup; cert is the sound side.
- BPTT through T=160 sequential core steps is the wall-clock bottleneck (Python loop over T) — sizes
  chosen for a single-session Kaggle run.
- The feasibility→full budget sensitivity seen in HD-1 means feasibility numbers here are previews only;
  conclusions are drawn from the full run.
