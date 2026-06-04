# The Right Verifier for Evolving Neural Dynamics is SDP/Lyapunov, not SMT — A CPU Study and an Adversarially-Audited ~95% Completeness Result

*Draft — 2026-06-04. This is an honest, reproducibility-first paper. Its central engineering claim is positive (a common quadratic Lyapunov certificate, solved as a semidefinite program, is the right contraction gate for an evolutionary search over small neural dynamics, and it is near-complete on CPU); its most transferable contribution is methodological (we caught and corrected a solver artifact in our own pipeline that had fabricated a richer-than-real "degree ladder"). All results are an n = 2, CPU-only study on a single pool and seed. Negative and retracted sub-results are labelled as corrected, not presented as findings.*

---

## Abstract

An evolutionary search over the *dynamics* of small neural systems — coupled, RWKV-style state-update rules — needs a soundness gate that admits only genes whose dynamics contract (converge) and rejects those that diverge. The natural design question is not *whether* to gate but *which verifier* to use. We study this question on a CPU-only, on-premise substrate and reach a clear answer: the right contraction verifier is a **common quadratic Lyapunov certificate solved as a semidefinite program (SDP/LMI)**, not a satisfiability-modulo-theories (SMT/Z3) closed-form check. On our pool of 300 empirically-contracting n = 2 genes, the SDP certificate certifies 286 (95.3%), the single largest jump on a verifier-fitness "ladder" that begins with conservative induced norms; higher-degree sum-of-squares (SOS) rungs add only a handful more (+4 over the SDP, to cumulative 290), and of the 10-gene degree-≤6 residual that remains, 6 are *correctly-rejected* switched-expansive genes rather than misses — leaving a ~2-gene near-boundary tail (jsr_lb ≈ 0.99) that the degree-8 SOS certifier admits but that exact joint-spectral-radius (JSR) methods would be needed to *close exactly*. Placed inside a minimal genetic algorithm, this gate admits zero divergent children (an ungated baseline lets 17–20% of admitted children drift into non-contraction) and a stronger sound verifier unlocks additional reachable safe fitness up to the SDP rung (inf-norm-gated rotation fitness tops ~0.41; SDP-gated reaches ~0.86, p = 3.1 × 10⁻⁵), beyond which the higher-degree capability gain is a measured null.

The result that we believe travels furthest is not the number but how we earned it. An earlier draft, run on cvxpy's default first-order solver (SCS), reported a *richer* picture — a non-nested, "complementary" degree-4/degree-6 ladder and an ~89% finite-degree-gap residual. That was a **solver artifact**: SCS false-negatives near the SDP feasibility boundary, and the resulting missing certificates fabricated apparent structure. Pinning the accurate interior-point solver (CLARABEL) collapsed the ladder to a nested one and lifted quadratic-SDP coverage from 64% to 95%. Crucially, the red-team device that *should* have caught this — a margin sweep — was structurally blind to it, because it wobbles inside the same buggy solver; the decisive checks were a **solver swap** and **multi-perspective adversarial review** (Codex plus a six-agent skeptic workflow). We present the corrected, humbler, and stronger result, and we treat the self-correction as the paper's core methodological contribution. We emphasize the scope: this is an n = 2, CPU, single-pool study about the right *verifier*, not a claim that evolving neural dynamics is broadly useful.

---

## 1. Introduction

A recurring ambition in machine learning research is to *search over* learning systems rather than only to train a fixed one — to let an outer evolutionary loop discover the dynamics, update rules, or architectural pieces of a model. When the object being evolved is a *dynamical system* — a state-update rule that maps a hidden state forward in time — a basic safety property is at stake: the discovered dynamics must **contract** (their state must not blow up under iteration), or the evolved system is useless and unstable. An evolutionary loop that admits divergent dynamics will happily breed them, because a transiently lucky divergent gene can score well before it explodes.

The clean way to prevent this is to put a **soundness gate** inside the loop: every candidate child is checked by a verifier, and only certified-contracting genes are admitted. This reframes a learning problem as a *verified* search — "Verified × Evolvable" — and immediately raises the question this paper answers: *what is the right verifier?* The candidates are not interchangeable. They differ in soundness (do they ever admit a divergent gene?), in completeness (what fraction of genuinely contracting genes do they certify?), and in the *kind* of certificate they produce (a closed-form inequality, a fixed induced norm, a state-dependent quadratic Lyapunov function, a higher-degree SOS polynomial). Each choice has consequences not only for safety but, as we show, for how much *reachable* fitness the evolutionary search can attain without ever admitting an unsafe gene.

The substrate is `llcore`, a CPU-only, on-premise framework that evolves the dynamics of small coupled neural systems (n = 2 coupled RWKV-style state-update genes). Two verifier families are natural candidates and were both implemented and tested. The first is **SMT/Z3**: encode the contraction condition as a logical formula over reals and discharge it with a satisfiability-modulo-theories solver. The second is **SDP/Lyapunov**: search for a common positive-definite matrix P such that the quadratic form V(s) = sᵀPs decreases along the dynamics, expressed as a linear matrix inequality (LMI) and solved as a semidefinite program. We find, across a sequence of pre-registered experiments, that for this substrate SMT is *decorative* (it adds no discriminating power over a closed-form algebraic test) while the SDP/Lyapunov certificate is *load-bearing* (it captures a class of P-weighted contractions that no fixed induced norm can express) and, on CPU, *nearly complete* (it certifies ~95% of contracting genes).

Our motivation is partly methodological, and the paper is organized around a single discipline of honest disclosure: when a result looks unusually rich or favourable, decompose it before believing it. That discipline paid off concretely. An intermediate version of this very paper reported a more impressive "degree ladder beyond SDP." We caught it as a numerical-solver artifact, corrected it, and the corrected story is both humbler and stronger. We present that correction in full (§5), because the most reusable thing we learned was not about Lyapunov functions but about how a default solver can fabricate structure and how to catch it.

**Contributions.**

- **A verifier thesis with numbers.** For evolving contracting neural dynamics on this CPU substrate, the right verifier is a common quadratic Lyapunov certificate solved as an SDP/LMI — not SMT/Z3. We support this with a verifier-fitness ladder in which the quadratic SDP certifies 286/300 (95.3%) of contracting genes, the single largest rung, while SMT adds zero discriminating power on two independent corpora.
- **A near-completeness result on CPU.** With an accurate solver, the quadratic SDP is nearly complete: degree-4/6 SOS add +4 (to cumulative 290), and of the 10-gene degree-≤6 residual, six are *correctly-rejected* switched-expansive genes (not misses) and four are finite-gap genes the degree-8 SOS certifier closes — leaving ~2 near-boundary genes (jsr_lb ≈ 0.99) that degree-8 admits but whose *exact* JSR the bracket does not pin at finite CPU lift degree, an honest asymptote toward the JSR = 1 boundary rather than a closed frontier.
- **The methodology core: catching our own solver artifact.** We document, reproducibly, how cvxpy's default first-order solver (SCS) false-negatives near the SDP feasibility boundary and fabricated a non-nested "complementary" degree ladder and an inflated residual. Pinning CLARABEL collapsed the artifact (64% → 95% quadratic-SDP coverage; "+54" extra coverage → "+4"; a 23/13/18 split → a nested 0/1/54). The transferable lesson is that margin-sweep red-teams are *structurally blind* to this failure, and the decisive checks are solver-swap and multi-perspective adversarial review.
- **A hardened, soundness-precise gate.** We add an opt-in machine-checked positive-definiteness test (Rump verified-PD) inside an OR-of-solvers gate, reproduce the float gate's admit set exactly (286 == 286), and state the soundness guarantee precisely: it rests on the certifier *theorems* plus an independent eigenvalue recheck, with a one-sided JSR falsifier observing zero false admits — strong evidence, not an end-to-end machine-checked proof except where the additive Rump component is used.

We are explicit about what this paper does *not* claim. It does not claim that evolving neural dynamics is broadly useful; it characterizes the right verifier for the narrow, well-posed contraction-gate problem. It does not claim that SMT is useless in general; it shows SMT is decorative *for closed-form-reducible contraction invariants on this substrate*. And it does not claim a clean monotone "climb the SOS ladder to exact-JSR," because that is false (§7).

---

## 2. Background

### 2.1 Discrete-time contraction and the gate's job

The substrate evolves coupled state-update rules of the form s′ = decay ⊙ s + (1 − decay) ⊙ tanh(Ws + Vx), an n = 2 coupled RWKV-style gene. A gene is *contracting* if iterating its update drives any two trajectories together — equivalently, on this substrate, if the spectral radius of the relevant Jacobian (or its worst case over an achievable parameter box) is strictly below one. The gate's job is to **admit only contracting genes and reject divergent ones**, soundly: a single false admit (an expansive gene certified as contracting) would let the evolutionary loop breed instability. A verifier may be *conservative* — rejecting some genuinely contracting genes — without being *unsound*; soundness is the non-negotiable property, completeness is the figure of merit.

### 2.2 Induced norms: the cheap, conservative rungs

The simplest sound certificates bound an induced operator norm of the Jacobian over the achievable parameter box. If ‖J‖ < 1 for an induced p-norm uniformly over the box, the map contracts. The **∞-norm** certificate (a row-absolute-sum bound) and the **2-norm** certificate (the largest singular value, computed by an SVD at the box vertices, since ‖J‖₂ is convex so its box supremum is attained at a vertex) are both solver-free and closed-form. They are sound but *conservative*: a contraction whose contracting direction is not aligned with any fixed coordinate or singular basis can have every fixed induced norm exceed one while it still contracts under a *weighted* metric. The conservatism is structural, not a tuning issue, and it is exactly what the SDP certificate repairs.

### 2.3 A common quadratic Lyapunov function as an SDP/LMI

The richer certificate searches for a *metric* under which the map contracts. For a quadratic Lyapunov function V(s) = sᵀPs with P ≻ 0, the contraction condition that V decreases along the dynamics is, at each vertex Jᵥ of the achievable Jacobian box, the linear matrix inequality P − JᵥᵀPJᵥ ≻ 0. Feasibility of this system of LMIs in the matrix variable P is a semidefinite program. A feasible P is a certificate that the convex hull of the vertex maps shares a common contraction metric, which (by convexity of the quadratic form) implies the nonlinear map contracts. The key expressive gain over §2.2 is the *non-identity P*: setting P = I recovers the 2-norm test (so the SDP is a true superset of the 2-norm certificate on this substrate), but a genuinely non-identity P expresses contractions that no fixed induced norm can. This is the geometry the SDP buys, and the reason it is load-bearing rather than decorative.

### 2.4 The SOS/Veronese hierarchy and its link to JSR

When even a quadratic Lyapunov function does not exist but the map still contracts, one can lift to higher-degree polynomial Lyapunov functions. The standard device (Parrilo & Jadbabaie) lifts the dynamics through the Veronese / symmetric-power map of degree d and searches for a sum-of-squares certificate in the lifted space, which is again an SDP. As d grows, this family relates to the **joint spectral radius (JSR)** of the vertex set — the asymptotic worst-case growth rate over arbitrary products of the vertex maps — and provides increasingly tight upper bounds γ\*_d on it. We implement degree-4, degree-6, and degree-8 SOS certifiers via the symmetric Kronecker power (brute-force-verified to reproduce the closed-form symmetric square exactly for n = 2, 3, 4 and d = 2, 3, 4) and a JSR bracket combining a Gripenberg lower bound with the SOS upper bound. An important caveat, established empirically below, is that this lifted family is **not monotone**: a higher lift degree can yield a *looser* bound, so the tightest certificate is the minimum over degrees, not the highest degree (§7).

### 2.5 SMT/Z3 for closed-form invariants

The alternative verifier family encodes the contraction condition as a first-order formula over the reals — universally quantified over the achievable parameter box — and discharges it with an SMT solver (Z3). This is attractive when the invariant is genuinely a logical/algebraic condition with no convex-optimization structure. We test it head-to-head against the closed-form algebraic contraction test on both the scalar and coupled substrates (§4) and find it adds no discriminating power: every contraction condition we needed reduces to a closed-form inequality (a row-absolute-sum that is convex in the box parameter, hence maximized at a vertex), so the SMT solver merely re-derives what the algebra already gives. SMT would only become load-bearing for a *non-closed-form* invariant; the spectral-radius / Lyapunov-stability condition is exactly such an invariant, but it lives in SDP territory, not SMT territory.

---

## 3. The Verifier-Fitness Frontier

We organize the central evidence as a **ladder**: a sequence of verifiers of increasing expressive power, each a sound contraction certificate, applied to a fixed pool of empirically-contracting genes. The pool is 300 genes (scanned from 786 candidates, seed = 2024), each verified empirically contracting (ρ < 1 sampled at 4000 points from the same seeded stream) so that *every miss is a verifier's conservatism, not an actually-divergent gene*. All SDP/SOS rungs use the accurate interior-point solver CLARABEL (the reason for that pin, and what it corrected, is §5).

### 3.1 The ladder, with the exact numbers

Counting cumulatively how many of the 300 contracting genes each rung certifies:

| rung | certificate | new admits | cumulative | % of 300 |
|---|---|---|---|---|
| inf-norm | closed-form ‖J‖_∞ < 1 over the box | 88 | 88 | 29.3 |
| + 2-norm | vertex σ_max(J) < 1 | +49 | 137 | 45.7 |
| **+ quadratic SDP** | **common-P Lyapunov LMI** | **+149** | **286** | **95.3** |
| + degree-4 SOS | lifted SOS (degree 4) | +3 | 289 | 96.3 |
| + degree-6 SOS | lifted SOS (degree 6) | +1 | 290 | 96.7 |

The headline is the **quadratic SDP rung: +149 genes, lifting cumulative coverage from 45.7% to 95.3%** — by far the single largest jump on the ladder. The SOS rungs above it are read top-to-bottom as a *cumulative* ladder: degree-4 certifies the 3 genes that lie in both the degree-4 and degree-6 SOS cones (cumulative 289), and degree-6 adds the 1 gene that only its cone certifies (cumulative 290), for a degree-≤6 residual of 10 (300 − 290). One footnote keeps the bookkeeping honest: degree-4 is *nested* in degree-6 here (degree-4 minus degree-6 = 0), so degree-4 has no admit unique to it; the +3 attributed to the degree-4 rung is the degree-4 SOS cone's increment over the SDP, which degree-6 also achieves. The ladder's own per-set counts are inf = 88, two-only = 49, sdp-only = 149, deg4-and-deg6 = 3, deg6-only = 1, residual = 10, so the SOS family adds 3 + 1 = **+4 over the SDP**, reaching cumulative 290 (see §3.4 and §5).

### 3.2 The non-identity-P geometry is what does the work

The 149 SDP-only admits are precisely the genes that contract under a non-trivial weighted metric but exceed every fixed induced norm. Setting P = I in the LMI recovers the 2-norm test, so the SDP certificate is a *true superset* of the 2-norm certificate on this pool (we observe zero genes that the 2-norm certifies but the SDP does not). The +149 is therefore not a relabelling of the same contractions under a different name; it is a strictly larger, sound admit set unlocked by allowing P ≠ I — the geometric content of "the SDP is load-bearing."

### 3.3 The scale check (Track D, 3270 genes)

The ladder above is a curated 300-gene pool; we confirm the same shape at scale on an independent 3270-gene corpus (Track D, CLARABEL). The SDP certificate certifies 1291 of 1363 contracting genes (95%), and it beats the 2-norm certificate by **+692 genes** (1291 vs 599), with **zero** genes that the 2-norm certifies but the SDP does not (the SDP is again a true superset). The same superset-and-~95% pattern holds on a third corpus (Track B exp1, 10175 genes): the SDP-only admits rose from a previously-reported 817 to 2168 under the accurate solver, and the SDP certifies 94% of the 5021 contracting genes there. The three independent corpora agree on *certified coverage* — ~94–95% — but, honestly, not on a single tight residual: the uncertified tail is ~3.3% on the curated 300-gene pool (10/300), ~5–6% on Track D (≈72–80/1363), and ~6% on Track B exp1 (≈301/5021). The spread is pool-dependent: exp1's larger residual reflects its 10175-gene composition (a parameter grid plus heavier non-normal sampling), which over-represents near-boundary and switched-expansive dynamics. The robust, seed-independent statement is the one the corpora share: a common quadratic Lyapunov certificate certifies ~95% of contracting evolved dynamics on this n = 2 substrate, with a near-boundary residual in the 3–6% range depending on how the pool is sampled.

### 3.4 Per-rung coverage above the SDP

Of the 14 contracting genes the quadratic SDP cannot certify on the 300-gene pool (300 − 286), the higher rungs and an exact-JSR analysis partition them honestly:

- **degree-4 and degree-6 SOS together add 4 over the SDP**: degree-4 is *nested* in degree-6 (degree-4-minus-degree-6 = 0, so no admit is unique to degree-4), but the degree-4 SOS cone certifies 3 genes the quadratic SDP cannot (cumulative 289), and degree-6 certifies 1 further gene that only its cone reaches (degree-6-only = 1, cumulative 290). The combined gain beyond the SDP is therefore **+4** (the 3 in-both plus the 1 degree-6-only), leaving a degree-≤6 residual of **10** (300 − 290);
- of the **10** the degree-≤6 ladder leaves, two different certifiers attack the four that are *not* switched-expansive, and it is important not to conflate them. The **full-space degree-8 SOS LMI certifies all 4** of these finite-gap genes soundly (it is the genuine top rung). The **exact-JSR bracket** — a Gripenberg lower bound paired with the SOS upper bound γ\*_d — is the stricter, *exactness*-oriented device, and its degree-8 upper bound drops below 1 (closes the bracket) on only **2 of those 4**; on the other **2 near-boundary genes** (jsr_lb 0.9915 and 0.9787) the bracket's tightest upper bound stays just above 1 (γ\*_min ≈ 1.0003 and 1.0004). So those two genes are admitted by the degree-8 SOS certifier but are **not bracketed to exact-JSR at finite CPU lift degree**: they sit in the asymptotic tail that a proper exact-JSR method (SOS-on-variety or branch-and-bound, both NP-hard) would be needed to *close exactly*, not merely to certify (§7). The remaining **6 of the 10 are correctly-rejected switched-expansive genes** — pointwise-ρ < 1 at each vertex but a vertex *product* expands (JSR ≥ 1), so the gate rightly refuses them. They are not misses.

At every rung, on every corpus, we observe **0 unsound admits**.

### 3.5 The verified × evolvable payoff (the practical motivation)

The ladder is not an end in itself; it matters because a *more complete sound verifier unlocks more reachable safe fitness*. We wire each verifier as a fail-closed gate inside a minimal genetic algorithm and measure both safety and fitness.

*Safety.* An ungated GA lets **17–20%** of admitted children drift into non-contraction (per-task: rotation 19.7%, benign 16.7%); the SDP gate admits **0** divergent children. The gate is load-bearing — it measurably changes what evolution breeds — and sound.

*Reachable fitness.* On a rotation task whose optimum lives in the region only the richer verifiers can certify, the conservative inf-norm gate caps evolved fitness at ~0.41, while the SDP gate reaches ~0.86 (paired one-sided Wilcoxon p = 3.1 × 10⁻⁵, complete separation, robust across three independent seed families and surviving Bonferroni correction). On a benign task whose optimum is inf-norm-certifiable, the gates are statistically indistinguishable (an honest null) — confirming that the rotation advantage is task-structural, not a generic "admit more, score more" artifact. The big jump is inf → 2-norm (capturing rotational contractions the ∞-norm over-rejects); the SDP's unique gain over the 2-norm is real but thin on the rotation optimum, and lives in the near-boundary SDP-only shell.

This is the practical reason the verifier choice matters: a *stronger* sound verifier unlocks more reachable fitness at no soundness cost, up to the SDP rung — beyond which (the higher-degree SOS rungs) the capability gain is a measured null on this substrate, not a continued climb. The SDP/Lyapunov gate is implemented as a production, pluggable, **fail-closed** backend in the framework's source tree (Stage 3b): it refuses to certify if CLARABEL is absent rather than silently falling back to the artifact-prone solver. The full test suite passes (255 source tests plus 313 research tests), CPU-only, at zero marginal cost, fully on-premise.

---

## 4. SMT is Decorative; SDP is Load-Bearing

The verifier thesis has two halves, and the negative half is as important as the positive one: for this substrate, SMT/Z3 contributes nothing that closed-form algebra does not already supply, while the SDP/Lyapunov certificate contributes a genuinely richer class of contractions.

### 4.1 SMT adds zero discriminating power (two head-to-heads)

We tested Z3 against the closed-form contraction test on two corpora. On the coupled substrate (Track C, 3270 genes), a Z3 contraction check and the closed-form ∞-norm endpoint-enumeration test **disagreed on 0 of 3270** genes. On the scalar substrate (Track B, 20000 evaluations), Z3 and the closed-form scalar contraction inequality **agreed on 20000 of 20000**. The reason is structural: the ∞-norm contraction condition is a row-absolute-sum that is convex in the box parameter, so its worst case is attained at a box vertex and the universally-quantified formula reduces to a finite closed-form check; the SMT solver merely re-derives the algebra. For closed-form-reducible contraction invariants, **SMT adds zero discriminating power**, and the soundness comes from the convexity/LMI theorem and the algebra, not from invoking a solver. "Z3-gated verification" is, on this substrate, an honest but *decorative* framing.

### 4.2 SDP captures what no fixed norm can

By contrast, the SDP/Lyapunov certificate is the one place a solver genuinely earns its keep. The genuinely richer certificate — a non-identity Lyapunov P that the induced norms cannot express — is exactly the SDP one (§3.2): on Track D the SDP certifies +692 genes the 2-norm rejects, P-weighted contractions that no fixed induced norm can capture. SMT cannot express this either; the spectral-radius / Lyapunov-stability condition is not a closed-form inequality, and it is precisely the *non-closed-form* invariant that motivates a solver. The clean conclusion of the verifier study is therefore directional: where the invariant is closed-form, SMT is decorative and algebra suffices; where it is not closed-form, the right tool is SDP/LMI, not SMT.

---

## 5. A Solver Artifact and How We Caught It

This section is the paper's methodological core. We report, in full and as a *correction*, that an intermediate version of these results was contaminated by a numerical-solver artifact that fabricated a richer-than-real picture, and we describe the discipline that caught it.

### 5.1 The artifact: SCS false-negatives near the feasibility boundary

cvxpy's `.solve()` with no explicit solver defaults to **SCS**, a first-order ADMM solver. For the feasibility-*boundary* SDPs that arise on this substrate — genes that contract only barely, whose Lyapunov certificate sits very close to the edge of feasibility — SCS returns **false negatives**: it fails to find a Lyapunov certificate that demonstrably exists (the tell-tale sign is its "Solution may be inaccurate" warning). The pipeline's independent eigenvalue recheck guards *soundness* — it rejects bad certificates — but it cannot recover a certificate the solver never found. So SCS's misses did not produce unsound admits; they produced *missing* admits, and the missing admits fabricated apparent structure further up the ladder.

### 5.2 The fabricated finding

Run under SCS, the degree ladder looked rich and *complementary*. The reported picture was: degree-4-only = 23, degree-6-only = 13, both = 18 (a non-nested ladder in which degree-4 and degree-6 each catch genes the other misses); a quadratic-SDP coverage of only **193/300 (64%)**; a degree-4/6 contribution of **+54** genes over the SDP; and a residual of **53** genes for the higher degrees to recover. A separate degree-4 experiment reported a degree-4 residual of 172 with degree-4 "recovering 33%" of it. These numbers told a compelling story — "the degree ladder unlocks much more than the SDP" — and that story was wrong.

### 5.3 Pinning CLARABEL collapsed it

Pinning the accurate interior-point solver **CLARABEL** across every SDP certifier (and the production source backend) collapsed the fabricated structure:

| quantity | SCS (reported) | CLARABEL (honest) | reading |
|---|---|---|---|
| quadratic-SDP coverage (of 300) | 193 (64%) | **286 (95%)** | SDP undercounted |
| degree-4/6 add over SDP | +54 | **+4** | "degree unlocks much" was the artifact |
| degree-4∖degree-6 / degree-6∖degree-4 | 23 / 13 | **0 / 1** | "complementary" was the artifact (nested) |
| residual (degree-≤6 misses) | 53 | **10** | residual inflated ~5× by SCS |
| degree-4 #2 residual (1200 genes) | 172 | **31** | residual inflated ~5.5× |
| degree-4 #2 "recovery" | 57 (33%) | **4 (13%)** | "33% recovery" was mostly SCS-missed SDP |

CLARABEL **recovered 42–43 of the 53** SCS false-negative residual genes. The nested, humbler truth — a quadratic SDP at 95%, a tiny near-boundary residual, a *nested* (not complementary) higher-degree ladder — replaced the fabricated rich ladder.

### 5.4 The bias ran both ways; the thesis only got stronger

The artifact did not bias every number in the same direction, and disclosing this is part of the honesty. *Where the SDP itself was the headline* (Track D, "SDP beats 2-norm"), SCS made the SDP look **weaker**: it reported the SDP beating the 2-norm by only +254 genes, whereas the accurate solver shows **+692** — the SDP advantage was *understated* by 2.7×. *Where "beyond SDP" was the headline* (the degree-6 complementarity, the degree-4 "33% residual recovery"), SCS's missing SDP certificates inflated the residual and made the higher degrees look like they recovered a large class — those "beyond-SDP" findings were largely *fabricated* and are retracted. So SCS biased magnitudes in *both* directions, but the unifying truth is single and clean: with an accurate solver, **a common quadratic Lyapunov (SDP) certifies ~95% of contracting evolved dynamics**, and the verifier thesis is confirmed and strengthened. The retracted sub-narratives — a "complementary / non-nested" degree-4/6 ladder, an "89% finite-gap residual," "degree unlocks +54 coverage" — are corrections, never findings.

### 5.5 The lesson: margin sweeps are blind; solver-swap and multi-perspective are decisive

The most transferable outcome is *how* the artifact was caught, and how it was *not*. Our adversarial red-team had a margin-sweep lens — perturb the certificate margin and check robustness — and it reported the complementary ladder as *robust*, i.e. it **confirmed the bug rather than catching it**. This is not a bug in the red-team; it is structural. A margin sweep wobbles the constraint *inside the same first-order solver*, so it cannot detect a systematic solver false-negative; every perturbation lands in the same biased regime. The two checks that *were* decisive were (1) a **solver swap** — re-running the identical SDPs under CLARABEL and observing the admit sets diverge — and (2) **multi-perspective adversarial review**, in which independent reviewers recomputed the headline against real code. We have since frozen this lesson into a standing, deterministic regression test (a sub-minute pytest guard on an immutable SCS-era fixture) that asserts the CLARABEL ladder is nested, that the SCS-vs-CLARABEL admit sets diverge (so the solver-swap detector still fires), and that every certified gene passes the JSR soundness oracle — fail-closing permanently if the certifiers ever silently revert to the SCS default.

### 5.6 Provenance

The correction and its narrowing were carried out under an explicit cross-AI pair-review discipline. A Codex (gpt-5.4) read-only adversarial review plus a six-agent skeptic workflow (2026-06-04, ~564k tokens, ~45 minutes) re-verified every headline against the actual code. The review **narrowed three over-claims** — an over-stated "every solve is pinned / fail-closed completeness" claim (a latent footgun where a falsy-but-non-None solver string could fall back to SCS, now fixed and regression-tested); the soundness-proof strength (§6); and a "preserve-or-grow / never shrinks" rationale for the hardened gate that was *direction-inverted* (§6) — and **closed one latent footgun**. No headline number changed. We treat this self-correction as a strength of the result, not something to hide: the value of the discipline is precisely that it narrows over-claims before publication.

---

## 6. Hardening and Soundness

Soundness — never admitting a divergent gene — is the gate's non-negotiable property, and we state our guarantee precisely rather than overclaiming it.

### 6.1 The Rump verified-PD + OR-of-solvers hardening (additive, opt-in)

Beside the float-recheck SDP certifier we add an opt-in **machine-checked** positive-definiteness test. "Rump verified PD" is a rigorous test that returns *True* only when a Cholesky backward-error bound proves the matrix is positive definite; a *True* verdict is a proof, and the test was validated to produce **0 false positives over ~7 million adversarial matrices**. The hardened gate accepts a gene iff **any** of {CLARABEL, SCS} returns a P that passes the Rump recheck on the same no-margin matrices the float path tests — an OR, not a vote, so each solver is an independent chance to find a verifiable certificate. On the 300-gene pool the hardened gate reproduced the float gate's admit set **exactly (286 == 286, 0 lost, 0 extra)**, with 149 of the 286 admitted through the genuine SDP-solve-plus-Rump branch (the SDP-only count from the ladder's bookkeeping; the other 137 = inf 88 + two-only 49 take the solver-free fast path), so the equality reflects agreement on a real SDP workload, not an all-fast-path artifact.

We are careful about *why* 286 == 286 holds, because an earlier draft got the reasoning backwards (a correction the pair-review forced). The Rump test returns a sound *lower bound* on the minimum eigenvalue, so on the same matrix it is the **stricter** test — the set of Rump-accepted certificates is a subset of the float-accepted ones, and in principle it could *reject* a barely-PD P the float test accepts. The exact reproduction is therefore **empirical**, explained by an ~8-order-of-magnitude headroom (the SDP enforces a 1e-7 decrease margin while the Rump backward-error bound is ~1e-15), confirmed numerically (4000 barely-PD matrices, 0 Rump rejections, 0 unsound lower bounds). The honest statement is "286 == 286 observed on this pool at this margin," not "the admit set can never shrink." The hardening's contribution is to upgrade the *basis of trust* from a single solver's float eigenvalue claim to a machine-checked PD proof across two solvers; it is additive and not the default.

### 6.2 The soundness framing, stated precisely

Soundness rests on three components, in order of strength:

1. **The certifier theorems.** For the quadratic certificate, a feasible common vertex-LMI (P − JᵥᵀPJᵥ ≻ 0 at every vertex Jᵥ) implies, by convexity of the quadratic form, that P − JᵀPJ ≻ 0 holds for every J in the convex hull of the vertices. The contraction conclusion for the *nonlinear* map then needs two further steps, which we make explicit rather than fold into "by convexity": (i) a *containment* argument that the pointwise Jacobian J(s) lies in that hull over the achievable state/parameter box — here the box is built from the tanh-derivative bound (1 − tanh²) ∈ [0, 1], so each Jacobian entry is a convex combination of its vertex extremes and J(s) ∈ conv{Jᵥ}; and (ii) an incremental / mean-value argument that a common contraction metric on every J(s) along a segment makes the map itself contract between any two trajectories. For degree ≥ 2, the lifted full-space SOS certificate is a *sufficient* condition for contraction. These are the guarantee; they are mathematical, not solver-dependent.
2. **An independent float eigenvalue recheck.** The pipeline independently rechecks the eigenvalues of the solver's returned P (P ≻ 0 and the decrease LMI ≻ 0), so it rejects a numerically bad certificate the solver might have returned. This is a numerical guard, not a float-level proof.
3. **A one-sided JSR falsifier.** The JSR product oracle (a Gripenberg lower bound over products up to length 6) is a *falsification* check: a certified gene with jsr_lb ≥ 1 would *prove* an unsound admit. We observe **0** such genes across every corpus: pool-wide, the worst certified jsr_lb over the 286 admits is **0.9999** (< 1, stable at product length 8), and over the small four-gene higher-degree-certifier battery the worst is 0.966. This is *necessary but not sufficient* evidence — "**0 observed false admits**," not an end-to-end machine-checked proof at float precision.

The only component that supplies a machine-checked float-level PD proof is the additive Rump gate (§6.1), and it is not the default. Two facts make the soundness conclusion robust despite this precision. First, the artifact we corrected was false *negatives* — missing certificates, not spurious ones — so pinning CLARABEL cannot, in direction, introduce false positives; it can only recover admits the solver should have found. Second, the independent recheck plus the JSR falsifier jointly observe **0** unsound admits on every pool we tested. We state this as strong, multi-corpus evidence of soundness, and we decline to call it a proof except where the Rump component is engaged.

---

## 7. Limitations

We list the honest limits explicitly, because several of them bound the results materially.

- **The lifted SOS family is non-monotone.** A higher lift degree can yield a *looser* bound: in a smoke test the SOS upper bound γ\* *rose* from degree-2 to degree-8 (0.767 → 0.780) on a near-normal gene, and the count of "higher-degree-worse" cases is positive. The tightest certificate is therefore the *minimum over degrees* (the union of the lifted certificates), not the highest degree, which is why degree-4 and degree-8 each catch a different slice. The intuitive claim "climbing the SOS ladder monotonically reaches exact-JSR" is **false**, and we do not make it.
- **Exact JSR is NP-hard; the frontier asymptotes rather than closes.** Closing the last near-boundary genes requires proper SOS-on-variety or branch-and-bound JSR, which is NP-hard; the 2 near-boundary genes (jsr_lb → 0.99) stay open at finite CPU lift degree. The coverage frontier *asymptotes* to the JSR = 1 boundary rather than closing — an honest, clean limit, not a closed result.
- **n = 2, CPU, this pool and seed.** Every number is on the n = 2 coupled substrate, on CPU, on a specific pool and seed (300-gene seed = 2024, plus the 3270- and 10175-gene corpora). The diagonal-scalar special case used in some earlier-track refinements is exact only on a [−1, 1] box; non-diagonal / vector maps need the full Jacobian operator norm, and a different pool, seed, dimension, or solver set could in principle surface a non-zero delta in either direction. The soundness, load-bearing, and "SMT decorative" results are structural and seed-independent; the exact coverage percentages and margins are pool-dependent.
- **This is about the right *verifier*, not about the usefulness of evolving neural dynamics.** We make no claim that evolving the dynamics of small neural systems is broadly useful. We characterize, narrowly and rigorously, which verifier is the right contraction gate for that search, and how complete it is on CPU.

---

## 8. Related Work

**SOS and Lyapunov certificates.** The sum-of-squares relaxation of Lyapunov stability, and its formulation as a semidefinite program, originates with Parrilo's thesis and subsequent work [verify]. The specific lifting of switched/uncertain linear dynamics through a polynomial (Veronese) map to obtain SOS Lyapunov certificates, and its connection to the joint spectral radius, is due to Parrilo & Jadbabaie [verify]. Our degree-4/6/8 certifiers sit squarely in this literature. Importantly, the non-monotonicity we observe empirically (§7) is *not* a novel surprise but a known feature of degree-parameterized polynomial Lyapunov families: the tradeoff between certificate degree/structure and bound tightness is treated directly by Ahmadi, Jungers, Parrilo & Roozbehani [verify], whose path-complete graph Lyapunov framework shows that richer certificate structure does not monotonically improve the bound — exactly the phenomenon our γ\*_d(degree) data exhibits. We position our finding as an empirical confirmation of that literature on this substrate, not as a new theoretical result.

**Joint spectral radius.** The JSR and its approximation are studied by Parrilo & Jadbabaie [verify], by Gripenberg's branch-and-bound lower/upper bracketing [verify], and comprehensively by Jungers [verify]. The NP-hardness/inapproximability of the JSR (and Lyapunov exponent) is due to Tsitsiklis & Blondel [verify], and the undecidability of boundedness of all products of a matrix pair is Blondel & Tsitsiklis [verify]; these are the first sources that motivate our "asymptotes rather than closes" framing — closing the last near-boundary genes exactly is not merely expensive but provably hard. We use a Gripenberg-style lower bound as our one-sided soundness falsifier.

**Contraction analysis and metrics.** The gate's mathematical object is a contraction property of a discrete-time nonlinear map, and the common-quadratic-Lyapunov certificate we use is precisely a *constant* contraction (Riemannian) metric. Contraction analysis for nonlinear systems is due to Lohmiller & Slotine [verify], and the convex search for state-dependent contraction metrics — control contraction metrics — to Manchester & Slotine [verify]. Our claim in §2.3 that "a non-identity P expresses contractions no fixed induced norm can" is the constant-metric special case of this body of work: the SDP/LMI search is a search for a contraction metric, and that framing, rather than substrate-specific geometry, is what grounds the soundness theorem of §6.2.1.

**Linear matrix inequalities and convex optimization in control.** The LMI formulation of stability and contraction, and the broader programme of casting control problems as convex feasibility, is canonically Boyd, El Ghaoui, Feron & Balakrishnan [verify]; Boyd & Vandenberghe's convex optimization text is the standard reference for the SDP machinery [verify].

**SDP solvers.** Our central methodological finding turns on the difference between an interior-point and a first-order solver. We pin **CLARABEL**, an interior-point conic solver (Goulart & Chen) [verify], and trace the artifact to **SCS**, the first-order operator-splitting conic solver (O'Donoghue, Chu, Parikh & Boyd) [verify]. The general phenomenon — first-order solvers struggling near the feasibility boundary where interior-point methods remain accurate — is folklore in the optimization community but, to our knowledge, is rarely documented as a *fabricated scientific finding* in a downstream pipeline.

**Verified positive-definiteness.** The machine-checked PD test we use as the additive hardening is based on Rump's rigorous error bounds for verified numerical linear algebra (Rump 2006, and the broader verified-computing literature) [verify], with Higham's accuracy-and-stability treatment as background [verify].

**Neural-network verification.** The broader enterprise of verifying neural systems — Marabou, the α,β-CROWN family, and the VNN-COMP competition series [verify] — addresses input-output robustness of trained networks rather than contraction of evolved dynamics, but shares our central concern: a *sound* verifier whose completeness is the figure of merit, and the discipline of not over-claiming a verifier's guarantees. The contrast also sharpens our thesis rather than contradicting it. Those tools reach for SMT/MILP and branch-and-bound precisely because their invariant — the input-output relation of a ReLU network — is *not* closed-form-reducible: the piecewise-linear activation pattern induces a combinatorial set of linear regions that a convex relaxation alone cannot decide exactly, so a case-splitting (SMT/MILP) search is genuinely load-bearing there. Our "SMT is decorative" result is therefore explicitly scoped to *closed-form-reducible contraction invariants on this n = 2 substrate* (§4, §7); it is not a claim that SMT is useless for neural verification in general, where the non-closed-form structure is exactly what makes SMT/MILP load-bearing.

**Neuroevolution and Quality-Diversity.** The outer-loop motivation — evolving the dynamics of neural systems — connects to neuroevolution and Quality-Diversity search, in particular MAP-Elites (Mouret & Clune) [verify]; a companion `llcore` paper characterizes when the selection/diversity factor is load-bearing in this same substrate.

**Reproducibility and honest disclosure.** Methodologically, this paper is in the tradition of pre-registration and adversarial self-checking in empirical ML. Our specific contribution to that tradition is concrete: a documented case where the standard red-team device (a margin sweep) was *structurally blind* to a solver artifact, and where a solver-swap plus multi-perspective review were the decisive checks.

---

## 9. Reproducibility and Honest Disclosure

The work is laid out under a `research/` tree isolated from the framework's `src/` (which was not modified during the verifier study, beyond pinning the production SDP backend to CLARABEL). The keystone artifacts are the ladder results (`exp_deg6_ladder_results.json`, recording inf = 88, two-only = 49, sdp-only = 149, deg6-only = 1, deg4-and-deg6 = 3, residual = 10), the scale-check results (Track D and Track B JSONs), and the SCS-vs-CLARABEL audit.

The following reproducibility measures are in place:

- **Fixed seeds.** The 300-gene pool is regenerated deterministically (seed = 2024, ρ < 1 empirical-contraction filter at 4000 samples from the same seeded stream), and the scale-check corpora are seeded similarly.
- **CLARABEL pin.** Every SDP certifier — `verifier_deg4`, `verifier_deg6`, the coupled `_sdp_certifies`, the Track-D runner, and the production `src` Stage-3b backend — explicitly pins `solver=CLARABEL`, with the production backend fail-closing (refusing to certify) when CLARABEL is absent rather than silently using SCS. A latent footgun (a falsy-but-non-None solver string falling through to the bare SCS default) was found by the pair-review and fixed with a regression test.
- **The solver-swap regression test.** An immutable SCS-era fixture (the 54-deg-certified / 53-residual battery, restored verbatim from the originating git commit) backs a standing, deterministic, sub-minute pytest guard asserting (a) the CLARABEL ladder is nested (degree-4-only == 0), (b) the SCS-vs-CLARABEL admit sets diverge (the solver-swap detector still fires), and (c) every CLARABEL-certified gene passes the JSR soundness oracle (jsr_lb < 1). This permanently fail-closes if the certifiers ever revert to the SCS default.
- **The JSR soundness oracle.** The one-sided Gripenberg falsifier (products up to length 6) is wired as a standing check; it observes 0 genes with jsr_lb ≥ 1 across every corpus and is the soundness *falsification* mechanism (§6.2).

**What is needed before submission.** This is a draft. Before external submission it requires (1) cleaned, single-command reproduction scripts that regenerate every table from the seeds; (2) figures (the ladder coverage curve, the SCS-vs-CLARABEL admit-set divergence, the verifier-fitness frontier); and (3) bibliography verification — every reference marked [verify] in §8 must be checked against its first source (authors, year, venue, and that the cited content actually supports the claim) before any citation is finalized.

---

## 10. Conclusion

We asked which verifier is the right contraction gate for an evolutionary search over small neural dynamics, and the honest answer is directional and clear. For closed-form-reducible contraction invariants on this CPU substrate, **SMT/Z3 is decorative** — it agreed with the closed-form algebraic test on 0/3270 and 20000/20000 head-to-heads, adding no discriminating power. The genuinely richer certificate — a non-identity quadratic Lyapunov P that no fixed induced norm can express — is the **SDP/LMI** one, and on CPU it is *nearly complete*: it certifies ~95% of contracting evolved dynamics (286/300, 1291/1363, 94% of 5021 on three independent corpora), the single largest jump on the verifier ladder, while higher-degree SOS rungs add only a handful of near-boundary genes and six of the residual are correctly-rejected switched-expansive genes rather than misses. Inside a minimal GA the SDP gate admits zero divergent children and unlocks more reachable safe fitness up to the SDP rung (~0.41 → ~0.86, p = 3.1 × 10⁻⁵; the higher-degree rungs add no significant capability), the operational meaning of "a better verifier is load-bearing."

The result we most want to travel is the methodology. An earlier version of this paper reported a richer "degree ladder beyond SDP" that was a numerical-solver artifact: cvxpy's default first-order solver false-negated near the feasibility boundary and fabricated a non-nested complementary ladder and an inflated residual. The standard red-team device, a margin sweep, was structurally blind to it; the decisive checks were a solver swap and multi-perspective adversarial review, which we have since frozen into a standing regression test. We caught our own artifact, corrected it, and the corrected story is both humbler and stronger. The soundness guarantee we state precisely — certifier theorems plus an independent eigenvalue recheck plus a one-sided JSR falsifier observing zero false admits, with a machine-checked PD proof available as an additive option — and the limits we state plainly: a non-monotone SOS family, an NP-hard exact-JSR tail that asymptotes rather than closes, and an n = 2, CPU, single-pool scope. This is a paper about the right *verifier*, supported by audited numbers and an audited correction, not a claim that evolving neural dynamics is broadly useful.

---

## References

*All citations are provisional. Every entry marked [verify] requires first-source verification (authors, year, venue, and that the content supports the in-text claim) before submission.*

1. P. A. Parrilo. *Structured Semidefinite Programs and Semialgebraic Geometry Methods in Robustness and Optimization.* PhD thesis, California Institute of Technology, 2000. [verify]
2. P. A. Parrilo and A. Jadbabaie. "Approximation of the joint spectral radius using sum of squares." *Linear Algebra and its Applications*, 428(10):2385–2402, 2008. [verify]
3. R. M. Jungers. *The Joint Spectral Radius: Theory and Applications.* Lecture Notes in Control and Information Sciences, Springer, 2009. [verify]
4. G. Gripenberg. "Computing the joint spectral radius." *Linear Algebra and its Applications*, 234:43–60, 1996. [verify]
5. J. N. Tsitsiklis and V. D. Blondel. "The Lyapunov exponent and joint spectral radius of pairs of matrices are hard — when not impossible — to compute and to approximate." *Mathematics of Control, Signals, and Systems*, 10(1):31–40, 1997. [verify]
5a. V. D. Blondel and J. N. Tsitsiklis. "The boundedness of all products of a pair of matrices is undecidable." *Systems & Control Letters*, 41(2):135–140, 2000. [verify]
6. S. Boyd, L. El Ghaoui, E. Feron, and V. Balakrishnan. *Linear Matrix Inequalities in System and Control Theory.* SIAM Studies in Applied Mathematics, 1994. [verify]
7. S. Boyd and L. Vandenberghe. *Convex Optimization.* Cambridge University Press, 2004. [verify]
8. P. J. Goulart and Y. Chen. "Clarabel: An interior-point solver for conic programs with quadratic objectives." 2024. [verify]
9. B. O'Donoghue, E. Chu, N. Parikh, and S. Boyd. "Conic optimization via operator splitting and homogeneous self-dual embedding." *Journal of Optimization Theory and Applications*, 169(3):1042–1068, 2016. [verify]
10. S. M. Rump. "Verification of positive definiteness." *BIT Numerical Mathematics*, 46(2):433–452, 2006. [verify]
11. N. J. Higham. *Accuracy and Stability of Numerical Algorithms.* 2nd edition, SIAM, 2002. [verify]
12. S. Diamond and S. Boyd. "CVXPY: A Python-embedded modeling language for convex optimization." *Journal of Machine Learning Research*, 17(83):1–5, 2016. [verify]
13. G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer. "Reluplex: An efficient SMT solver for verifying deep neural networks" (and the Marabou successor, Katz et al., CAV 2019). [verify]
14. H. Zhang, T.-W. Weng, P.-Y. Chen, C.-J. Hsieh, and L. Daniel, and the α,β-CROWN line (Wang et al., "Beta-CROWN," NeurIPS 2021). [verify]
15. C. Brix, M. N. Müller, S. Bak, T. T. Johnson, and others. "The VNN-COMP neural network verification competition" (report series). [verify]
16. L. de Moura and N. Bjørner. "Z3: An efficient SMT solver." *TACAS*, 2008. [verify]
17. J.-B. Mouret and J. Clune. "Illuminating search spaces by mapping elites." arXiv:1504.04909, 2015. [verify]
18. A. M. Lyapunov. "The general problem of the stability of motion" (1892 doctoral thesis; English translation, *International Journal of Control*, 55(3):531–534, 1992). [verify]
19. W. Lohmiller and J.-J. E. Slotine. "On contraction analysis for non-linear systems." *Automatica*, 34(6):683–696, 1998. [verify]
20. A. Megretski and A. Rantzer. "System analysis via integral quadratic constraints." *IEEE Transactions on Automatic Control*, 42(6):819–830, 1997. [verify]
21. A. A. Ahmadi, R. M. Jungers, P. A. Parrilo, and M. Roozbehani. "Joint spectral radius and path-complete graph Lyapunov functions." *SIAM Journal on Control and Optimization*, 52(1):687–717, 2014. [verify]
22. I. R. Manchester and J.-J. E. Slotine. "Control contraction metrics: Convex and intrinsic criteria for nonlinear feedback design." *IEEE Transactions on Automatic Control*, 62(6):3046–3053, 2017. [verify]
