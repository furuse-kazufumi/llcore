# VERDICT — degree-6 Lyapunov & the two frontiers of certificate strength (2026-06-03)

> Roadmap **#5** (skeleton extension). Goal (user): *"CPU で llcore の進化に効果がありそうな要素を
> 追加検証する"* + *"良すぎる数値は疑い、手戻りしないよう細かく検証する"*. Discipline:
> `research/verified_evolution_sdp_gate/`, **src untouched**, pre-registration first
> (`DEG6_PREREGISTRATION.md`), seeds fixed, honest disclosure, adversarial red-team.
> **Headline: the COVERAGE frontier of the verifier ladder keeps advancing with degree (and the
> rungs are complementary, non-nested, all sound), but the CAPABILITY frontier — the fitness
> evolution can reach — SATURATES at SDP on the n=2 substrate. The two frontiers decouple above
> SDP.** A surprising "deg4 unlocks more fitness" signal turned out to be GA noise — caught by
> careful verification (the user's caution was correct).

## 1. What was built (new VerifierBackend rung + verification battery)

`verifier_deg6.py` — a **degree-6 homogeneous Lyapunov** certificate V(z)=m₃(z)ᵀ P m₃(z) over the
degree-3 monomial (Veronese order-3) vector, via a common-P LMI on the **symmetric 3rd Kronecker
power** J^[3] of each t-box vertex Jacobian (Parrilo–Jadbabaie SOS hierarchy, k=3). One generic
`sym_power(A, degree)` (brute-force-verified vs the monomial transform for n=2,3,4, degree=2,3, max
error ~1e-16; reproduces `verifier_deg4.sym2_power` exactly). The certifier never trusts the
solver: P≻0 and every decrease-LMI ≻0 are re-checked by independent eigendecomposition, with a
ρ(J_v)<1 vertex pre-screen. **Only the backend is new** — `evolvable_core.evolve()` and the codecs
are untouched; `make_deg6_verifier_n2()` is a drop-in `VerifierBackend` (sdp ∪ deg4 ∪ deg6).
`test_deg6.py`: **13 tests pass** (sym-power correctness, soundness, residual recovery, both
complementarity directions, union⊇sdp). Whole repo regression: **255 → 268 tests, 0 regressions.**

## 2. The COVERAGE frontier advances + the ladder is NON-NESTED (EXP-A, 300 contracting genes)

| cumulative gate | inf | +2-norm | +sdp | +deg4 | +deg6 |
|---|---|---|---|---|---|
| certified (of 300 empirically-contracting) | 88 | 137 | 193 | 234 | **247** |
| % | 29 % | 46 % | 64 % | 78 % | **82 %** |

- **G-A1 (coverage advances) — PASS.** The deg-union certifies **+54 genes over the quadratic
  class** (sdp 193 → deg6 247); degree-6 adds **+13 beyond sdp∪deg4**.
- **G-A2 (ladder non-nested) — PASS, and NON-OBVIOUS.** In the *lifted full-space* form the
  degree-4 and degree-6 certificates are **complementary**: **deg4∖deg6 = 23** and **deg6∖deg4 =
  13**, both non-empty. So "higher degree = strict superset" is FALSE here — the sound admission
  set is the **union**. (Mechanism: the full-space decrease is imposed on all of the lifted space,
  not only the Veronese variety, so a higher degree does not dominate a lower one.) This is a
  genuine methodological finding, reproduced by two seeded test searches.
- **G-A3 (soundness) — PASS.** 0 unsound among deg-certified genes at the EXP-A check (20k) — and
  see §3 for the far stronger JSR test.

## 3. Certifier soundness — verified by the STRONGEST practical oracle (JSR), not just pointwise ρ

A skeptic's foundational concern: the lifted LMI imposes Lyapunov decrease only at the t-box
**vertices** {J_v}; the nonlinear map's mean Jacobian J̄(s)=∫₀¹J(τs)dτ is a **convex-hull** point,
and for degree≥2 the lift A^[k] is **non-convex** in A, so vertex feasibility does NOT directly
imply hull-decrease of *this* V (unlike the quadratic k=1 case, which is saved by matrix
convexity). Soundness instead follows via the **JSR convex-hull theorem**: a common vertex
Lyapunov ⇒ JSR{J_v}<1 ⇒ JSR{conv J_v}<1 ⇒ the nonlinear map contracts. A *necessary consequence*
is therefore that **every certified gene has JSR{J_v}<1**.

`verify_certifier_jsr_soundness.py` checks exactly this (JSR lower bound = max over length-≤6
vertex products of ρ(∏)^{1/k}):

| | value |
|---|---|
| deg4/deg6-certified genes tested | 54 |
| max JSR_lb among certified | **0.9974** (< 1) |
| mean JSR_lb among certified | 0.869 |
| **certified genes with JSR_lb ≥ 1 (would be a soundness bug)** | **0** |

**CERTIFIER SOUND.** No certified gene has an expanding product — strictly stronger than the
pointwise-ρ oracle (which can miss switched expansion). The max 0.9974 confirms the deg4/deg6
rungs reach genuinely *near-boundary* contractions the quadratic class cannot.

## 4. The CAPABILITY frontier SATURATES at SDP (n=2) — honest-null CONFIRMED, after catching a trap

This is where "good numbers" had to be distrusted (the user's directive). Three measurements:

1. **2-seed gated-GA smoke (MISLEADING):** L3=sdp∪deg4 reached rotation R²=**0.9765** vs L2=sdp
   **0.8881** → looked like a deg4 capability payoff.
2. **Random-sample region ceilings (ALSO MISLEADING, opposite bias):** deg4-region max 0.605 < sdp
   0.72 → looked like a clear null — but random sampling is **density-biased** (it under-samples
   small regions; even the sdp ceiling read 0.72 vs the true ~0.98).
3. **Region-constrained OPTIMISATION + winner attribution + 50k soundness (DECISIVE,
   `verify_deg4_payoff.py`, 12 seeds):**

| gate | max reach | mean | best-gene region | ρ@50k | sound |
|---|---|---|---|---|---|
| L2 = sdp | 0.9795 | 0.874 | **sdp_only** | 0.932 | ✓ |
| L3 = sdp∪deg4 | 0.9954 | 0.865 | **sdp_only** | 0.935 | ✓ |
| L4 = sdp∪deg4∪deg6 | 0.9674 | 0.907 | deg4_only | 0.921 | ✓ |

**Verdict: NO sound deg4/deg6 capability payoff on rotation.** The rotation optimum is **SDP-
certifiable** (L2 and L3 winners both live in `sdp_only`, ρ<1, at ~0.98–0.99). The +0.0159 of L3
over L2 is **GA variance, not a deg4 effect** — L3's winner is an sdp-region gene, so the gain came
from finding a better *sdp* gene, not a residual one. **L4 (0.9674) < L3 (0.9954) despite admitting
a superset** is a clean demonstration that single-run GA-reach is noisy — which is exactly why the
2-seed smoke's 0.9765 was a seed-luck artifact, and why region attribution (not raw reach) is the
honest measure. The deg4/deg6 regions top out *below* the sdp ceiling on this task.

→ **The capability frontier saturates at SDP for n=2: degree-4/6 add provable COVERAGE but no new
reachable CAPABILITY.** This is the pre-registered honest-null (G-B2 NULL), now properly verified.
(Positive control G-B1: the inf→2-norm payoff reproduces — L0 caps far below L1+, the known arc
result; full ladder reach confirms inf is the only real bottleneck.)

## 5. JSR attribution of the irreducible residual (honest accounting)

Of the 53 genes that are pointwise-contracting but **even degree-6 cannot certify**:

| | count | % |
|---|---|---|
| **finite-degree-gap candidate** (JSR_lb<1, deg6 fails ⇒ a degree-8+/exact-JSR rung would reach) | **47** | 89 % |
| **switched-expansive, correctly rejected** (JSR_lb≥1 ⇒ a vertex product expands; pointwise-ρ<1 was misleading) | 6 | 11 % |

Robust from max_len 5 → 6 (unchanged). So the coverage frontier has **genuine remaining headroom**
(89 % of the residual is a finite-degree gap, not a fundamental wall), and the 11 % the ladder
refuses are genes it *should* refuse (the cheap pointwise oracle would have wrongly passed them).

## 6. The capability gap is DIMENSION-GATED (EXP-C, coupled_nd n=2,3,4) — forward-looking

Why does capability saturate at SDP at n=2? Because the residual's achievable **transient
amplification** is weak at n=2, so SDP-certifiable genes approximate residual targets. Mechanism
hypothesis: non-normal transient amplification grows with dimension. Measured (max behavioural
transient of residual vs quad-certified contracting genes):

| n | T_residual | T_quad | **gap** | quad genes found |
|---|---|---|---|---|
| 2 | 2.76 | 2.37 | **+0.39** | 387 |
| 3 | 4.15 | 1.81 | **+2.33** | 64 |
| 4 | 3.57 | 1.54 | **+2.03** | **6 (under-powered)** |

The gap **jumps ~5× from n=2 to n≥3** — a clear dimension threshold. **Honest caveats:** the strict
monotone gate (G-C) **FAILS** (n=4 +2.03 < n=3 +2.33), but n=4 found only **6 quad-certified
contracting genes** in the scan budget (severe under-power), so the n=4 dip is most plausibly
sampling, not a real reversal; T_quad shrinking with n (2.37→1.81→1.54) is the robust signal (the
quadratic class covers ever less as n grows). **Reading:** the deg-rung *capability* payoff is
**dimension-gated** — negligible at n=2, substantial at n≥3 — so it should **return in high-dim /
full-LLM regimes**, structurally motivating the GPU bet (echoing the ③-arc BG9 insight that
selection needs high dimension). *Not yet a proven monotone law — n=4 needs a larger-budget rerun.*

## 7. Pre-registered gates — verdicts

- **G-A1 coverage advances — PASS** (L4−L2 = +54 ≥ 5).
- **G-A2 ladder non-nested — PASS** (deg4∖deg6 = 23, deg6∖deg4 = 13, both > 0).
- **G-A3 soundness — PASS** (0 unsound; JSR test 0 certified genes with JSR_lb≥1).
- **G-B1 harness valid (positive control) — PASS** (inf is the bottleneck; L1+ ≫ L0, the arc result).
- **G-B2 capability payoff — NULL (pre-registered honest-null CONFIRMED).** No sound deg4/deg6
  payoff over sdp on rotation; the rotation optimum is sdp-certifiable.
- **G-C dimension monotone — FAIL (strict), THRESHOLD supported.** Gap jumps at n≥3 but is not
  strictly monotone (n=4 under-powered); reported honestly, not as a proven law.

## 8. Adversarial red-team — (pending `redteam_deg6.py`: 50k soundness on all deg-certified,
admission-size-artifact under random fitness, circularity of the residual reference, numerical
margin 1e-6..1e-8 robustness of the complementarity) + a parallel multi-lens skeptic review
(`_review_workflow.js`). [To be filled on completion.]

## 9. Honest disclosure / the two-frontier thesis

- **Two distinct frontiers of certificate strength, which DECOUPLE above SDP.** *Coverage* (what a
  verifier can prove) keeps climbing with degree — and the degree-4/6 rungs are **complementary**,
  so the ladder is a *union*, not a chain. *Capability* (what evolution can reach) saturates at SDP
  on n=2 because the rotation optimum is already SDP-certifiable. Conflating the two is the trap
  that produced both the "0.98 deg4 payoff" (GA luck) and the "0.605 null" (sampling bias). The
  honest, verified statement: **above SDP, a stronger verifier buys provable coverage and soundness
  headroom, not new reachable fitness — at n=2. The capability payoff is dimension-gated.**
- **Two measurement traps caught (kept as lessons, [[feedback_benchmark_honest_disclosure]]):**
  (a) single/2-seed GA-reach is high-variance (L4<L3 with a superset gate proves it) → use region
  attribution + many seeds; (b) random-sample region ceilings are density-biased → use region-
  constrained optimisation. Both biased the answer in *opposite* directions; only the decisive
  method (optimise-within-region + attribute winner + 50k soundness) is trustworthy.
- **Solver "Solution may be inaccurate" warnings are benign:** the certifier's independent eigen
  re-check (P≻0 and every decrease-LMI ≻0), not the solver status, is the sound authority — and the
  JSR test (§3) corroborates 0 unsound certified genes.
- **EXP-C n=4 under-power** (6 quad genes) and **JSR_lb finite truncation** (max_len 6) are
  disclosed; the dimension claim is a *threshold*, not a proven monotone law.
- **Push:** none (llcore remote not created — exposure avoidance). Local commits only.

## 10. Bottom line

A concrete **degree-6 Lyapunov VerifierBackend** plugs into the unchanged skeleton and is **sound
by the JSR oracle**. It sharpens the arc's "stronger verifier ⇒ more reachable safe fitness" into a
**two-frontier** picture: the **coverage** frontier keeps advancing (inf→2norm→sdp→deg4→deg6, the
top rungs complementary/non-nested, 82 % of contracting genes certified, 89 % of the rest a finite-
degree gap), while the **capability** frontier **saturates at SDP** on the n=2 substrate — degree-4/6
add provable coverage and soundness headroom but no new evolved capability, because the task optimum
is already SDP-certifiable. Crucially, the capability gap is **dimension-gated** (jumps 5× at n≥3),
so the higher-degree rungs are predicted to become capability-load-bearing as the core scales toward
full-LLM dimensionality — a falsifiable, GPU-motivating hypothesis. The headline "deg4 unlocks more
fitness" was a GA-noise artifact, caught by region-attributed optimisation — the discipline working.

**Next (each a plug-in):** (1) degree-8 / exact-JSR `VerifierBackend` to reach the 89 % finite-gap
residual; (2) a larger-budget EXP-C (n=4,5,6) to test whether the capability gap is monotone in
dimension (the GPU go/no-go signal); (3) promote the deg4/deg6 union into the `src/` verifier
backend plugin (Stage 3b is already there for sdp); (4) a high-dim Objective whose optimum is
*provably* in the deg-residual, to test capability payoff where dimension makes it non-trivial.

Artifacts: `verifier_deg6.py` · `test_deg6.py` (13) · `exp_deg6_ladder.py` · `exp_deg6_capability.py`
· `exp_deg6_dimension.py` · `jsr_bracket.py` · `verify_region_ceiling.py` · `verify_deg4_payoff.py`
· `verify_certifier_jsr_soundness.py` · `redteam_deg6.py` · `DEG6_PREREGISTRATION.md` · result JSONs.
src/ untouched; push deferred (exposure avoidance).
