# Verified-Plasticity Evaluation: Measuring Guarantee, not Capability, in Online Structural Adaptation of Small LLM Adapters

**Draft / extended abstract — 2026-06-10.** Not submitted. 規律: honest-disclosure(capability ≠ guarantee を混同しない)。
正本データ = `research/rllm_pivot/PHASE_{M1,1,2}_VERDICT.md` + 全 `phase*.py`/`*_results.json`。capstone = `docs/VERIFIED_PLASTICITY_FRAMEWORK.md`。

---

## Abstract

We ask whether a small recurrent adapter bolted onto a frozen small LLM can adapt its *structure* online without (i) diverging or (ii) catastrophically forgetting — and whether such adaptation can be driven by *evolution* competitively against *gradient descent*. We contribute a **Verified-Plasticity Evaluation Framework**: a methodology that treats "does the online structural adaptation stay provably contractive (ρ<1)?" as a first-class, falsifiable measure, and compares candidate adaptation methods under a six-device statistical-rigor harness (pre-registration, Holm conjunction, artifact discipline, falsification clauses, self-power audit, anti-over-claim critic). Across four phases of $0/CPU experiments we find: (1) verified structural evolution is feasible **only at small per-component width (n≤6)** — no presently-known sound certifier is simultaneously *navigable* and *scalable* (the 2^n vertex wall); (2) under real Net2Net width-growth surgery, sound certifiers (cert_two/cert_sdp) achieve **0 observed false-admits**, while an experience-based "STABLE-style" gate **false-admits 84%** of genuinely divergent genes; (3) **capability is NEGATIVE**: on a real SmolLM2-135M-derived cross-entropy terrain, MAP-Elites beats a *finite-difference* gradient 20/20, but a **strong analytic (autograd) gradient reverses this 19/20** — the evolutionary "win" is an artifact of a weak baseline; this holds (in mean) on a synthetic terrain too. We conclude the framework's value lies in **guarantee** (provably-stable online structural adaptation), not capability, and we report the honest-disclosure mechanism that caught the capability false-positive before publication.

## 1. Introduction

Online adaptation of an LLM's *structure* (not just its weights) sits on a classic stability–plasticity contradiction: more structural freedom risks divergence and forgetting. Most neural-architecture-search work competes on accuracy/latency/FLOPs. We instead ask a guarantee question — *can online structural adaptation be kept provably contractive?* — and build a falsifiable framework to measure it across methods. We deliberately separate two claims that are routinely conflated: **capability** (does the adapter predict better, lower cross-entropy?) and **guarantee** (is the adapted structure provably stable, ρ<1?). Our headline finding is that capability does *not* favor evolution once a strong gradient baseline is used, so the framework's deliverable is the guarantee measurement itself.

## 2. The Framework

**Substrate.** A coupled n-dim recurrent adapter `s' = decay⊙s + (1−decay)⊙tanh(Ws + x)`, with `decay∈[0,1]^n, W∈[−2,2]^{n×n}`, applied to a low-dim projection of a frozen LLM's hidden states. The tanh keeps the state bounded; instability therefore manifests as failure of the echo-state property (perturbation persistence/amplification in the linearized sensitivity), *not* norm divergence — a subtlety that defeats norm-based and finite-horizon "forgetting" tests.

**Certifiers (sound contraction over the achievable Jacobian box, 2^n vertices).** `cert_inf` (closed-form ‖J‖_∞<1), `cert_two` (σ_max<1 at every box vertex), `cert_sdp` (common quadratic Lyapunov LMI, cvxpy/CLARABEL, fail-closed). All imply ρ(J)<1. An independent eigenvalue oracle `empirical_rho` (from-below) checks soundness.

**Three plug-points** (framework-ness, single-object swap): GeneCodec (substrate), Objective (direction), VerifierBackend (gate). The gate admits only by the *stability* certificate, never by fitness — a homeostatic constraint that permits adaptation but forbids divergence/forgetting.

**Six-device harness.** Pre-registration → result order; Holm conjunction; artifact discipline; falsification clauses; self-power audit; anti-over-claim critic. This methodology layer is what licenses any "evolution is real" claim — and is itself the framework's backbone.

## 3. Experiments and Results

All $0/CPU; each phase adversarially re-verified (MAJOR 0).

**Phase −1 (numerical co-viability scan).** width_grow soundness-preserving bands exist for cert_two at small n (n=4: 58–67%), but are empty for the only scalable certifier (cert_inf). Vertex-free B2 is navigable only at n=4, collapsing to cert_inf by n=8. ⇒ **verified structural evolution is small-n per-component only**; a navigable-and-scalable sound certifier does not presently exist.

**Phase 0/1 (real harness + soundness under growth, Decision gate 1 = PASS).** On real SmolLM2-135M hidden states, a cert_two gate is load-bearing (certified-stable 1.000 vs no-gate 0.680). Under real Net2Net width-growth surgery, sound certifiers give **0 observed false-admits**; cert_sdp is the most navigable sound certifier (admits ~0.9–0.99 of the truly-contracting set) but the 2^n vertex cost is unchanged. per-block AND is unsound under coupling (forbidden); full-system cert required. Small-n feasibility 0.013h ≪ 30h budget.

**Phase 2 (framework validity + capability, Decision gate 2).**

*Discrimination (H-discriminative, PASS).* Over a spanning population (95 divergent / 305 contracting genes), false-admit of divergent genes: no-gate 100%, **STABLE-style experience gate 84%**, sound certs **0%**. cert_sdp: 0% false-admit and only 4.6% over-rejection (sound + most navigable). Mamba-synthetic positive control: 0 false-reject.

*Base-level discrimination (positive control, PASS).* Every Mamba-130M layer's SSM has continuous diagonal `A = −exp(A_log) < 0` (100% of 589,824 channel,state entries), so λ_max = max(Δ·A) ≤ 0 for all Δ>0 — trivially stable-by-construction. SmolLM2-135M (Llama) has no SSM state recurrence; stability must be imposed by the bolted-on gate. The framework cleanly separates the two bases.

*Capability (ARTIFACT+NEGATIVE).* On a real SmolLM2-derived next-hidden-cluster cross-entropy terrain (n=6, K=6, held-out sentences, 20 seeds), held-out fitness (=−CE): **analytic gradient −1.446 (best)** > MAP-Elites −1.454 > random −1.473 > finite-diff gradient −1.483. MAP-Elites beats the finite-difference gradient **20/20** (mean +0.029, p<1e-6) — but a strong **analytic (autograd Adam) gradient reverses it 19/20** (p=3.5e-4). The evolutionary advantage is an **artifact of the weak finite-difference baseline** (cold-start, dim+1 evals/step, ~95 updates), not capability. A synthetic-terrain cross-check agrees in mean (analytic gradient best, 0.575 > 0.535) though variance keeps the paired test a tie there.

*Framework-ness (F8).* The three plug-points swap with a single object (17 unit tests pass). The hypothesis "structural diversity is load-bearing for generalization" is **NULL**.

## 4. Honest Limitations

(i) Verified structural evolution is **small-n per-component only**; high-dim scaling needs an undiscovered navigable-scalable certifier. (ii) **Capability is not sellable**; a strong gradient beats evolution on both terrains. (iii) Soundness is **0 *observed* false-admits** (from-below consistency), not a machine proof. (iv) The real-terrain CE is a hidden-cluster proxy (full-vocab degenerates at small n); gating costs −0.028 held-out (it trims plasticity). (v) "Strong gradient is best" assumes free exact gradients (backprop) — realistic for LLM training. (vi) The dissemination/market value of a guarantee-only framework is unestablished (consumer story pending).

## 5. Conclusion

The Verified-Plasticity Evaluation Framework establishes a falsifiable, method-agnostic way to measure whether online structural adaptation of a small LLM adapter stays provably contractive, and to discriminate sound certifiers from experience-based gates that miss 84% of divergence. Its honest verdict is that **the value is guarantee, not capability**: evolution does not beat a strong gradient on real or synthetic terrains. The most consequential methodological result is negative-by-design and self-inflicted — the framework's strong-gradient meta-gate caught a tempting capability false-positive (evolution 20/20 vs a weak baseline) before it could be claimed. We argue this self-skeptical discipline, not any single mechanism, is what a "provably evolvable" system most needs.

---

**Reproducibility.** All code/data: [github.com/furuse-kazufumi/llcore](https://github.com/furuse-kazufumi/llcore). Models: SmolLM2-135M, Mamba-130M (both Apache-2.0). No GPU required.
**Status/next.** Framework established. Open: scale beyond n≤6 (navigable-scalable certifier); consumer story + demand evidence (user judgment); venue selection.
