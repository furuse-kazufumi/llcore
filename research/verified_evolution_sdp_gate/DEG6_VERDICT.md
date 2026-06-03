# VERDICT — verifier-ladder coverage, the SCS-artifact correction, and degree-8/JSR (2026-06-03)

> Roadmap **#5 + #6**. Goal (user): *"CPU で llcore の進化に効く要素を追加検証"* + *"良すぎる数値は
> 疑い、手戻りしないよう細かく検証"*. Discipline: `research/`, **src untouched**, pre-registration first,
> honest disclosure, adversarial red-team + a parallel multi-agent skeptic review.
>
> **HEADLINE (after a major self-correction the user's caution forced):** An earlier draft reported a
> *non-nested "complementary" degree-4/6 ladder* and an *89% finite-degree-gap residual*. **Both were
> cvxpy SCS-solver artifacts.** Under the accurate CLARABEL solver the truth is cleaner and humbler:
> **a common quadratic Lyapunov (SDP) certifies ~95% (286/300) of contracting evolved dynamics; the
> ladder is NESTED; degree-4/6/8 add only a tiny near-boundary residual.** The arc thesis "the right
> verifier is SDP/Lyapunov" is **STRENGTHENED**; the dramatic "degree unlocks much more" story is
> retracted. Soundness was never at risk (the artifact was false *negatives*).

## 0. The solver-artifact correction (the most important finding)

cvxpy's `.solve()` with no solver argument defaults to **SCS** (first-order ADMM — the "Solution may
be inaccurate" warnings). For these feasibility-**boundary** SDPs, SCS returns **FALSE NEGATIVES**:
it fails to find a Lyapunov certificate that exists. The independent eigen re-check guards *soundness*
(it rejects bad certificates) but **cannot recover a certificate SCS never found** — so SCS
fabricated apparent structure. Verified first-party (`verify_solver_artifact.py`, cvxpy 1.9.1):

| on the 54 SCS-"deg-certified" residual genes | deg4∖deg6 | deg6∖deg4 | both | "complementary"? |
|---|---|---|---|---|
| **SCS** (default) | 23 | 13 | 18 | YES (the artifact) |
| **CLARABEL** (accurate) | **0** | **0** | **54** | **NO → nested** |

And CLARABEL certifies **42–43 of the 53 SCS-"uncertified" residual** genes (SCS false negatives).
**Fix:** pin `solver=CLARABEL` in `verifier_deg4`, `verifier_deg6`, and `coupled_components._sdp_certifies`.
**Lesson (memory `feedback_cvxpy_pin_accurate_solver`):** never trust cvxpy's SCS default for a
feasibility-boundary SDP; the margin sweep (red-team lens4) is *blind* to this — only a solver swap
is decisive. The adversarial multi-agent review caught it before the verdict was published.

## 1. What was built

`verifier_deg6.py` — degree-6 Lyapunov via the symmetric 3rd Kronecker power (general
`sym_power(A,d)`, brute-force-verified n=2,3,4, d=2,3,4; reproduces `sym2_power` exactly). Drop-in
`VerifierBackend`, `evolve()`/codecs untouched. `verifier_jsr.py` — JSR bracket `[jsr_lb (Gripenberg),
γ*_d (SOS upper bound via bisection)]`. Plus the verification battery: `verify_solver_artifact.py`,
`verify_certifier_jsr_soundness.py`, `verify_region_ceiling.py`, `verify_deg4_payoff.py`,
`exp_deg6_ladder/jsr_bracket/deg8_ladder/dimension.py`, `redteam_deg6.py`. `test_deg6.py` 13 tests
(rewritten for the nested ladder + a deterministic residual fixture).

## 2. The COVERAGE frontier (CLARABEL — honest)

300 empirically-contracting genes (seed 2024), cumulative certified:

| | inf | +2-norm | +sdp | +deg4 | +deg6 | +deg8 | exact-JSR tail |
|---|---|---|---|---|---|---|---|
| certified | 88 | 137 | **286** | 289 | 290 | ~292–294 | 2 near-boundary remain |
| % of 300 | 29 | 46 | **95** | 96.3 | 96.7 | ~98 | — |

- **The common quadratic SDP is the workhorse: 286/300 (95%).** (SCS had said 193/64%.)
- deg4 adds +3, deg6 adds +1, **degree-8 adds +2–4** (closes finite-gap residual genes deg4/6 miss).
- Of the 14 the quadratic SDP can't certify: **4** have a higher-degree Lyapunov (deg4/6/8), **6** are
  **switched-expansive** (JSR≥1 — pointwise-ρ<1 but a vertex product expands; correctly refused), and
  ~**2–4** are near-boundary finite-gap (jsr_lb 0.98–0.99) needing exact-JSR.
- **G-A1 (advance ≥+5 over sdp): FAIL** (+4) — honest: degree-4/6 barely advance under an accurate
  solver. **G-A2 (non-nested): RETRACTED** — deg4 ⊆ deg6 (the 23/13 split was SCS). **G-A3 (sound):
  PASS** (0 unsound).

## 3. Certifier soundness — confirmed by the JSR oracle (unaffected by the solver bug)

The artifact was false *negatives*; soundness (false-positive freedom) was never at risk and is
independently confirmed. Every deg4/deg6-certified gene must have JSR{vertices}<1 (common vertex
Lyapunov ⇒ JSR<1 ⇒ convex-hull theorem ⇒ the nonlinear map contracts). `verify_certifier_jsr_soundness.py`:
**4 certified genes, max JSR_lb 0.966, 0 with JSR_lb≥1** → **CERTIFIER SOUND**. (The skeptic confirmed
a constructed switched-expansive pair — each vertex ρ=0.95<1 but JSR=1.57 — is correctly REJECTED by
the LMI itself, not just the ρ pre-screen.) Soundness conclusion survives the whole correction.

## 4. degree-8 / JSR-exact (Roadmap #6 — user-chosen coverage thrust)

On the 10-gene true residual: `exp_deg8_ladder.py` → degree-8 certifies **4** (deg8_only=4, neither=6,
0 unsound); `exp_jsr_bracket.py` → γ*_8<1 closes **2 of the 4 finite-gap**, mean bracket width 0.009,
near-boundary tail = **2** genes (jsr_lb 0.9915, 0.9787). **Honest findings:**
- **degree-8 is the genuine top rung** — it certifies finite-gap genes the quadratic SDP, deg4, and
  deg6 cannot, soundly. (G-8A PASS, G-8C sound.)
- **The lifted SOS family is NON-MONOTONE** (`verifier_jsr` smoke: γ* *rose* deg2→deg8 0.767→0.780 on
  a near-normal gene; `higher_degree_worse_count`>0). The full-space LMI's conservativeness grows with
  the lift, so the tightest bound is `γ*_min = min_d γ*_d` (the lifted union). This — not a monotone
  hierarchy — is why deg4/deg8 each catch a different slice. **"Climbing the lifted SOS ladder
  monotonically reaches exact-JSR" is FALSE.**
- **exact-JSR closes the tail only via proper SOS-on-variety or branch-and-bound** (NP-hard); the 2
  near-boundary genes (jsr_lb→0.99) stay open at finite CPU lift degree. The coverage frontier
  **asymptotes to the JSR=1 boundary** rather than closing — an honest, clean limit.

## 5. The CAPABILITY frontier — NULL at n=2 (robust), with a pre-registration deviation disclosed

`verify_deg4_payoff.py` (region-constrained optimisation + winner attribution + 50k soundness, 12
seeds): L2_sdp max 0.9795 (winner **sdp_only**, sound), L3_deg4 0.9954 (winner **sdp_only**, sound),
L4_deg6 0.9674 (winner deg4_only). The rotation optimum is **SDP-certifiable**, so degree-4/6 give **no
capability payoff**; the +0.0159 is GA variance (winner is sdp_only, not residual), and L4<L3 with a
superset gate confirms GA-reach noise. NULL — robust (two opposite-biased measurements both land NULL;
winner-attribution, not the margin, drives it).

**DISCLOSED DEVIATION (adversarial review):** the *pre-registered* G-B2 = strict-gate(**residual_reach**,
Wilcoxon p<0.05 ∧ |psd|≥0.147 ∧ **n=15**). `verify_deg4_payoff.py` is the **rotation** positive-control
task, a 0.02 margin (not Wilcoxon), **n=12** — a conservative proxy, NOT the pre-registered test. The
actual `exp_deg6_capability.py` (residual_reach, n=15, CLARABEL) is now **RUN (2026-06-03, elapsed
1308 s; `exp_deg6_capability_results.json`)**: the **positive control G-B1 PASSES** (rotation L1_two vs
L0_inf: Δ+0.336, paired_sign_delta=1.0, Wilcoxon p=3.05e-5 — the harness detects a real payoff, so the
test is powered), while **G-B2 is NULL** — neither higher-degree rung beats the SDP gate on the
residual_reach target: **L4_deg6 vs L2_sdp Δ+0.020, psd=0.33, p=0.138 (fail); L3_deg4 vs L2_sdp Δ+0.008,
psd=0.13, p=0.265 (fail)** (strict gate = p<0.05 ∧ |psd|≥0.147 ∧ n≥15). On residual_reach the inf/2-norm
gates cannot reach the target at all (L0_inf=0.066, L1_two=0.059) while the **SDP gate already reaches
R²≈0.92** of it (L2_sdp=0.916), and deg4/deg6 add no significant capability (0.924 / 0.936). The
capability NULL above SDP is therefore confirmed by the **pre-registered G-B2 with a validated positive
control**, not merely the conservative rotation proxy. *Honest nuance:* residual_reach is a reachability
proxy — the exact optimum is quad-rejected by construction, yet SDP-gated evolution approximates its
free response to ≈0.92 R² because near-optimal quad-admissible genes track the target; the +0.02 deg6
edge is within GA variance (psd 0.33, not significant).

## 6. The DIMENSION-threshold claim — RETRACTED

The earlier "+0.4→+2.0 gap jumps at n≥3 ⇒ higher-degree verifiers become load-bearing as the core
scales" is **retracted**, for two reasons the adversarial review established:
1. **Premise moot:** under CLARABEL the higher-degree rungs add almost nothing even at n=2 (§2), so
   there is barely any deg-rung capability for dimension to "gate."
2. **Proxy unsound:** the `decay_ratio<0.6` gate never re-checked ρ; ~49–58% of the n=4 "residual
   contracting" genes were empirically EXPANSIVE (empirical_ρ≥1, a tanh-saturation artifact at fixed
   ‖s0‖), and the `T_residual_max` gene itself had ρ≈1.8 — exactly the "switched-expansive, correctly
   rejected" class. Plus 5–27× residual/quad pool-size asymmetry inflated the max-vs-max gap.

`exp_deg6_dimension.py` was fixed (empirical_ρ<1 soundness gate) and **re-run (2026-06-03, ρ-gated, CLARABEL — `exp_deg6_dimension_results.json`): noisy / not robust across runs** — the canonical 1600-scan run gives T_residual 2.755 / 4.145 / 3.569 and T_quad 2.365 / 1.813 / 1.535 (n=2/3/4) ⇒ gap +0.39 / +2.33 / +2.03, **non-monotone** (`G_C_gap_monotone_increasing`=false, `T_residual_monotone_increasing`=false); a CPU-time-capped re-run flipped to an *accidentally*-monotone profile (gap +0.12 / +0.30 / +1.13) purely from truncated scan coverage — the cross-run flip is itself evidence the residual-vs-quad gap is a **sampling artifact, not a dimension law**;
the only defensible cross-n signal is T_quad shrinkage (the quadratic class covers less of higher-dim
space), itself partly a max-over-shrinking-sample effect. No dimension-gated capability claim stands.

## 7. Adversarial verification (what caught what)

- **`redteam_deg6.py` ALL_PASS (4 lenses)** — but lens4's SCS-only margin sweep was **structurally
  blind** to the artifact (it reported "complement robust" — confirming, not catching, the bug). Margin
  sweeps cannot detect a solver false-negative; only a solver swap can.
- **Parallel multi-agent skeptic review (`_review_workflow.js`, 11 agents)** — caught all three of:
  the SCS complementarity artifact (HIGH), the unsound dimension proxy (HIGH), the G-B2 pre-reg
  deviation (HIGH); confirmed certifier soundness and the capability NULL survive. This is the review
  that forced the correction — the discipline working.

## 8. Honest bottom line

After correcting the solver artifact: **with an accurate solver, a common quadratic Lyapunov (SDP)
certifies ~95% of the contracting evolved-dynamics on the n=2 substrate; degree-4/6/8 SOS close nearly
all the rest, leaving a ~2-gene near-boundary tail for exact-JSR; 6 pointwise-contracting genes are
correctly refused as switched-expansive.** This *strengthens* the arc's core claim — the right verifier
backend is SDP/Lyapunov, and it is nearly *complete* on CPU — while *retracting* the (artifactual)
complementarity, the inflated residual, and the dimension-gated-capability narrative. The verified CPU
evolution skeleton, the certifier's soundness, and the capability NULL all survive. **The single most
valuable deliverable may be the cautionary methodology: a reproducible demonstration that the default
SOS/SDP solver fabricates false structure near the feasibility boundary, and that adversarial
multi-perspective review (not margin sweeps) is what catches it.**

**Next:** (1) finish the two running confirmations (real G-B2 at n=15; ρ-gated EXP-C). (2) proper
SOS-on-variety or branch-and-bound JSR for the 2 near-boundary genes (the only way to truly close the
frontier). (3) audit the *earlier* arc verdicts (#2 deg4 "33% recovery", Track C/D coverage counts) for
the same SCS contamination — pin CLARABEL and re-measure. (4) the src SDP backend (`src/llcore/verifier/
backends.py`) should also pin an accurate solver before any production use.

Artifacts: `verifier_deg6.py` · `verifier_jsr.py` · `verify_solver_artifact.py` · `test_deg6.py` (13) ·
`exp_deg6_ladder/jsr_bracket/deg8_ladder/dimension.py` · `verify_*` · `redteam_deg6.py` · result JSONs ·
`DEG6_PREREGISTRATION.md` · `DEG8_JSR_PREREGISTRATION.md`. src/ untouched; push deferred (exposure avoidance).
