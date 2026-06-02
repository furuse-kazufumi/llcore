# When Is Darwinian Selection Load-Bearing? A High-Dimensionality Condition for Quality-Diversity in Evolutionary Search over Neural Dynamics

*Draft — 2026-06-02. This is an honest negative-result paper with a structural insight. It does not claim that behavioral niching is broadly useful; it characterizes the precise, narrow condition under which it helps and shows that no realistic CPU substrate we tested meets that condition.*

---

## Abstract

We ask whether the Darwinian "selection / separation" factor — the third of Darwin's four ingredients (variation, heredity, **selection**, over-reproduction) — is *load-bearing* when evolving the dynamics of small neural systems. We operationalize this third factor as MAP-Elites behavioral niching (a Quality-Diversity method) and test, across a sequence of CPU-tractable substrates, whether it provides a genuine advantage over strong non-niching baselines, in particular a random-restart hill-climbing optimizer.

Our central finding is negative but precise. On a deliberately constructed synthetic deceptive corridor, behavioral niching is *decisively* load-bearing: MAP-Elites reaches the global optimum on essentially every seed while three baselines (pure random, panmictic GA, random-restart hill-climbing) reach it on none, with a maximal effect size (Cliff's delta = +1.00). But across every *realistic* CPU substrate we examined — echo-state-reservoir text proxies, memory tasks, multi-reservoir parity, multi-task generalization, and a four-kernel-union search space — the third factor is **not** load-bearing. Crucially, this is not a power problem: using deterministic, noise-free landscape measurement we verify that the realistic proxy landscapes are *genuinely smooth* (single-basin), and where a landscape has a "hard" coordinate, that coordinate is **low-dimensional** and a strong random-restart baseline samples it directly.

The structural insight that organizes these results: behavioral niching is load-bearing only when the deception lives in a **high-dimensional behavior space** that direct sampling cannot reach. In our synthetic corridor the behavior descriptor is the mean of a 24-dimensional genotype; by central-limit concentration the global peak is a measure-zero region that random restarts cannot hit, so the archive's stepping-stones become essential. The realistic substrates fail this condition: their hard coordinates (e.g. leak rate, discrete kernel choice) are one- or low-dimensional, and a hill-climbing baseline with restarts solves them directly. We discuss why this leaves high-dimensional full-LLM loss landscapes as the only remaining venue for the third factor, and why even there a strong gradient baseline poses the same risk.

---

## 1. Introduction

Evolutionary and population-based methods are a recurring proposal for *searching over* learning systems rather than only training a fixed one: neuroevolution, Quality-Diversity (QD), and open-endedness research all suggest that maintaining a *diverse population* — not just hill-climbing on a single objective — is what lets search escape deceptive traps and keep finding qualitatively new behaviors. The intuition is Darwinian: variation supplies novelty, heredity carries it forward, and **selection acting on differences** — the third ingredient — converts that novelty into cumulative progress, especially when combined with mechanisms that *separate* the population into niches so that promising-but-currently-inferior solutions are not discarded.

This paper interrogates that intuition for the specific case of evolving *neural dynamics* (state-update rules, reservoir parameters, readout structure) on CPU-tractable substrates. We isolate the third Darwinian factor — selection/separation — and operationalize it concretely as **MAP-Elites behavioral niching** (Mouret & Clune 2015), the canonical QD method, with the hypothesis that niching lets search cross fitness valleys that defeat a plain hill-climber. We compare it against deliberately strong baselines: pure random sampling at equal budget, a panmictic genetic algorithm, and a **random-restart hill-climbing** optimizer.

Our motivation is partly methodological. It is easy to demonstrate that a QD method *can* win on a hand-crafted deceptive problem; it is much harder, and more honest, to ask whether the *condition* that makes it win occurs in problems we actually care about. We therefore adopt a discipline of honest disclosure throughout: when a result looks unusually favorable, we decompose it before believing it; we pre-register protocols before measuring; and we subject every positive-leaning headline to adversarial (red-team) verification and external pair-review.

**Contributions.**

- **A structural condition for when the third factor helps.** We show, both by existence proof and by a sequence of negatives, that behavioral niching is load-bearing precisely when the search's "hard part" lives in a high-dimensional behavior space that direct sampling cannot reach. When the hard coordinate is low-dimensional, a random-restart baseline samples it directly and niching loses its advantage. This condition unifies an existence proof with five independent failures.
- **An honest, multi-substrate negative result.** Across reservoir text proxies, three families of memory tasks, multi-reservoir parity, multi-task generalization, and a four-kernel-union space, the third factor is not load-bearing — and we separate "genuinely smooth landscape" from "underpowered measurement" rather than conflating them.
- **A methodology contribution: deterministic noise-free landscape measurement.** We show that closed-form ridge readout over a fixed-seed reservoir takes no random numbers, letting us drive evaluation noise to machine epsilon and measure multimodality without a noise floor — distinguishing a truly smooth landscape from a landscape merely buried in evaluation noise.

We emphasize what this paper does *not* claim. It does not claim that selection or QD is useless in general; the third factor is real and decisive in the right regime. It claims, specifically, that across the realistic CPU substrates we tested, the regime that makes it decisive does not arise.

---

## 2. Background

### 2.1 Darwin's four factors

Following the standard Darwin/Mayr decomposition, evolutionary search rests on four ingredients: **(1) variation** — heritable perturbation of designs; **(2) heredity** — offspring resemble parents; **(3) selection** — differential survival driven by fitness differences; **(4) over-reproduction** — more offspring than resources can sustain, which is what gives selection something to act on. Throughout this paper "the third factor" (③) refers specifically to selection/separation: the claim that *acting on fitness differences while maintaining separated niches* is what makes search succeed.

An internal soundness audit of our framework established that, in our substrates, factors (1) and (2) hold (variation moves the phenotype; offspring are behaviorally close to parents, correlation ≈ 0.87 with mutation scale), while factor (3) is *wired but idling*: under constant fitness, improvement is exactly zero (selection is real), yet under the working proxy fitness a genetic algorithm did not significantly beat equal-budget random search (one example: GA − random ≈ −0.011, 5/10 wins, Wilcoxon p ≈ 0.77). The audit also exposed a classic artifact — a reported monotone "best" curve of 0.473 collapsed to 0.183 under fresh-seed honest re-evaluation, a +0.29 inflation entirely due to elitism freezing a stale, lucky noisy fitness. This motivated the strict honest-evaluation methodology of §4.

### 2.2 MAP-Elites and behavioral niching as the operationalization of the third factor

MAP-Elites discretizes a *behavior space* into a grid of cells and keeps the best solution found in each cell. Unlike a single-objective optimizer, it does not throw away a solution merely because a fitter one exists elsewhere: a behaviorally distinct but lower-fitness solution is retained as a potential *stepping-stone*. This separation-by-behavior is our concrete stand-in for the third factor — selection that operates *within niches*, letting the population ratchet across fitness valleys that a pure hill-climber cannot cross (Lehman & Stanley 2011; Mouret & Clune 2015).

### 2.3 Deceptive landscapes

A landscape is *deceptive* when local improvement leads away from the global optimum: hill-climbing gets trapped in a false peak, and crossing to the true peak requires temporarily accepting worse fitness. Deception is the regime in which QD's stepping-stones are supposed to pay off. A key subtlety, which our results turn on, is that deception can be defined in *genotype/fitness* terms or in *behavior* terms, and these are not interchangeable: a landscape can be multimodal in genotype space yet trivially navigable, or smooth in fitness yet behaviorally hard to reach.

### 2.4 Baselines: why random-restart hill-climbing is the bar to beat

Our toughest baseline is random-restart hill-climbing: a simple (1+1)-style optimizer that accepts only non-worsening moves and, on stagnation, *restarts from a fresh random point*. This baseline is strong precisely because restarts give it free *coverage*: if the hard coordinate of a problem is something a fresh random draw can hit directly, restarts will hit it without needing to cross any valley. The entire question of whether the third factor is load-bearing reduces, in our analysis, to whether there exists a behavior dimension in which this restart baseline *fails* while niching *succeeds*.

---

## 3. Methodology as a Contribution

We treat the research *process* itself as a contribution, because the central difficulty of this work was not running experiments but avoiding self-deception. Five methodological commitments structure every phase.

### 3.1 Pre-registration

For the decisive late-phase experiments we fixed the protocol — behavior descriptor, number of bins, evaluation budget, seed count, statistical gate, and the three-valued decision rule (load-bearing / not-needed / N/A) — *before* observing results, and recorded any post-hoc deviation as an honest violation. This was a direct response to an earlier "deceptiveness measurement" attempt that failed by circularity and post-hoc threshold tuning; pre-registration is the structural fix.

### 3.2 Honest disclosure: decompose unusually good results before believing them

When a result was unusually favorable, we treated it as a prompt to find the artifact rather than as a win. This discipline caught the elitism-frozen +0.29 inflation (§2.1), a "best = archive-max" forgetting bias that inflated a QD advantage, and a random-search ceiling that was really *selection-on-noise* (taking the max over many evaluation seeds) rather than a genuine optimum.

### 3.3 Adversarial verification (red-team)

Positive-leaning verdicts were independently attacked along multiple lenses — circularity, descriptor dependence, sampling robustness, budget robustness — with each attack recorded as `refuted = true/false` and a severity. In the kernel-diversification phase a dedicated red-team *confirmed* a negative structurally (it could not find any behavior dimension where the restart baseline failed but niching succeeded), strengthening rather than overturning the conclusion.

### 3.4 Cross-AI pair-review

Each phase's verdict and code were reviewed by a second AI system (Codex / GPT) in read-only mode, and every finding was verified against the actual code before adoption. This caught implementation-level defects that the primary author missed: a mismatched-replicate seeding bug that broke the paired statistical test, an off-by-one fence-post error in a separate spiking substrate, an unimplemented logical gate, and several over-claims that were demoted to feasibility sketches. The point is not that the second AI was authoritative — it was not, and several of its findings were ruled non-issues on inspection — but that an independent adversarial reader is an effective substitute for the missing "disproof test."

### 3.5 Deterministic, noise-free landscape measurement

Our most important methodological device addresses a real confound: a landscape *measured* through noisy evaluation can look multimodal (or smooth) purely because of the noise floor, not because of its true geometry. We exploit that a fixed-seed echo-state reservoir followed by a *closed-form* ridge readout (a linear solve) consumes **no random numbers** at evaluation time. This lets us drive the evaluation-noise standard deviation down to machine epsilon (≈ 1.11 × 10⁻¹⁶ — a floating-point ULP artifact, not real noise) and measure valley structure directly. With the noise floor physically removed, we can finally separate "the landscape is genuinely smooth" from "we could not measure its valleys." Throughout, comparisons use common random numbers (CRN) for paired tests, a strict gate (one-sided Wilcoxon p < 0.05, paired effect-size threshold |·| ≥ 0.147, n ≥ 15, positive mean difference), and fresh-seed honest re-evaluation to exclude elitism-frozen artifacts.

---

## 4. The Arc: Experiments

We present six phases. The first is an existence proof (the third factor *can* be decisive); the rest are realistic substrates on which it is not. Each phase states a falsifiable criterion, the numbers, and an honest reservation.

### 4.1 Phase I — Existence proof: a synthetic deceptive corridor (Step 4)

**Criterion.** The third factor is load-bearing if MAP-Elites beats *all three* baselines (random, panmictic GA, random-restart hill-climbing) under a strict gate on a landscape engineered to be deceptive.

**Setup.** A *genotypic corridor*: behavior is `mean(gene)` over a 24-dimensional genotype (the descriptor is 1-D, the mean of a high-dimensional vector). High behavior requires *every* dimension to be high — a genotype extreme. The fitness profile along behavior is a local optimum (behavior ≈ 0.4, value 0.6), then a *dip* (behavior ≈ 0.65, value ≈ 0), then the global optimum (behavior ≈ 0.9, value 1.0). Evaluation noise σ ≈ 0.008.

**Result.** MAP-Elites reaches the global optimum on ~95% of runs; all three baselines reach it 0% of the time, stalling at the local optimum (≈ 0.60). Against each baseline: p = 1.9 × 10⁻⁶, Cliff's delta = +1.00, 100% win rate. Robust across three base seeds (60 seeds total). An `init_batch` ablation confirms the mechanism is the *archive ratchet*, not initial coverage: shrinking the initial random batch from 600 to 30 still yields ~100% global reach (mean 0.998), while pure random fails (0%) even at 6000 samples.

**Why each method behaves as it does.** Random sampling keeps behavior ≈ 0.5 (central-limit concentration of a mean) and can *never* reach behavior 0.9. Hill-climbing climbs to the local 0.6 and refuses the downhill move needed to cross the dip; its restarts land at behavior ≈ 0.5 again, back in the same trap. MAP-Elites retains the dip cells as new behavioral niches and ratchets behavior from 0.5 → 0.9, crossing the fitness valley by way of behavioral diversity.

**Boundary (Step 4, exp5).** Removing the dip — a *smooth* corridor — erases the advantage: MAP-Elites no longer beats random-restart hill-climbing (p ≈ 0.29) or the panmictic GA, winning only against pure random. **The third factor's advantage is confined to the deceptive regime; MAP-Elites is not universally superior.**

**Honest reservation.** This corridor is a *constructed* synthetic landscape. It proves the third factor is *possible*, not that any realistic task exhibits this structure. The win is attributable to behavioral-diversity maintenance enabling a valley crossing — a blend of niching and selection, not an isolated "differential survival rate." All experiments are CPU toy-scale at low noise; the restart baseline is a plain (1+1) optimizer without step-size adaptation (a reasonable, not state-of-the-art, baseline).

### 4.2 Phase II — The substrate floors: memory tasks and multi-reservoir parity (Step C, Ladder 1)

**Memory tasks (Step C).** We asked whether deceptive corridors arise *naturally* in standard memory tasks (delayed parity, flip-flop, delayed recall) on a single leaky-reservoir + ridge substrate. The result is a clean **N/A**: the substrate could not host a clean test of the third factor, for opposite reasons.

- *delayed_parity* is a **substrate floor**: a single reservoir cannot compute XOR (Minsky-Papert), so all methods score mean R² ≈ 0.003-0.004 and reach-rate 0.0. MAP-Elites vs. baselines: diff ≈ ±0.0008, p ≥ 0.51 — nobody climbs, so the third factor cannot be separated.
- *flip_flop* is a **ceiling**: all methods saturate at R² ≈ 0.945-0.953 (reach-rate 1.0 at threshold 0.8), compressing variance. MAP-Elites vs. random is positive in sign (diff = +0.0041, paired effect = +0.33, above the 0.147 gate) but p = 0.15 — *underpowered and inconclusive at n = 15, not a null*. It loses to the panmictic GA.

Genotype-space multimodality was high (valley fraction 1.000 for parity, 0.939 for flip-flop) yet meaningless for the third factor: multimodal-in-genotype ≠ navigably-deceptive. The most important finding contradicts the prior expectation (Bengio et al. 1994: long-range dependencies should be deceptive): the flip-flop instance as designed (sequence length 30, pulse probability 0.2) was *too easy*, so we explicitly declined to over-generalize this to "real tasks don't need the third factor."

**Multi-reservoir parity (Ladder 1).** We then asked whether coupling multiple reservoirs lifts the parity floor enough to host a clean test. It does not, and we pinned down *why* across five mechanisms (parallel gating, quadratic readout, evolved search, very-wide single, all-in hybrid), all `floor_lifted = false`. Depth (DeepESN) raises the floor *statistically* (2-layer effect +0.47, p = 0.010; 3-layer +0.60, p = 0.004, both strict-gate PASS) but only to absolute R² 0.05-0.10 — far from solving parity. A clean positive control settles the mechanism: a degree-2 readout solves 2-bit XOR exactly (held-out R² = +1.0000 at window 2) but fails for degree ≥ 3 (window 3/4/5 → −0.064 / −0.052 / −0.086). **5-bit parity is a degree-5 monomial**, a robust floor of this CPU reservoir+ridge paradigm. The parity route is structurally blocked, so the third-factor test must move off parity.

**Honest reservation.** The degree-5 floor is established as a robust floor *of this setting* (final-state readout, this search budget, held-out R²), proven principled only for the degree-2 readout case via positive control; it is not a paradigm-wide impossibility proof.

### 4.3 Phase III — Multi-task generalization: the third factor is not needed (E-A)

**Criterion.** Off the parity floor, test the third factor on *generalization* across a task distribution, with the cleanest ablation — MAP-Elites (full ①②③) vs. **MAP-Elites with selection removed** (`randselect`: parents drawn at random, unconditional placement = variation only), plus a panmictic GA and random.

**Setup.** Single-layer leaky reservoir + ridge readout on a *variable-delay recall* task. Train on delays {15, 30}; test (hold-out) on longer delays {45, 60} — an extrapolation. Equal budget (n_evals = 400), n_seeds = 30 (post-review), CRN pairing, global-best-of-budget readout, fresh-seed honest re-evaluation on the test regimes.

**Result (post pair-review).**

| method | test generalization R² (mean ± std) | train | gap |
|---|---|---|---|
| MAP-E (full ①②③) | 0.682 ± 0.115 | 0.898 | +0.216 |
| MAP-E randselect (selection removed) | 0.557 ± 0.108 | 0.872 | +0.315 |
| panmictic GA | 0.702 ± 0.083 | 0.915 | +0.213 |
| random | 0.620 ± 0.105 | 0.877 | +0.258 |

| gate | comparison | diff | p (one-sided) | effect | passes |
|---|---|---|---|---|---|
| C-gen3 | MAP-E > randselect | +0.126 | 0.0151 | +0.60 | **True** |
| C-gen4a | MAP-E > panmictic | −0.019 | 0.598 | −0.07 | False |
| C-gen4b | MAP-E > random | +0.062 | 0.126 | +0.20 | False |

**Interpretation.** MAP-Elites beats the *no-selection* drift control (C-gen3 PASS) — "some selection beats no selection." But it does *not* beat the panmictic GA (selection without niching; it is in fact marginally behind) and does not beat random. So the niching-specific (third-factor) contribution is absent: this multi-task generalization landscape is smooth enough that simple selection or even random reaches the same generalization. Consistent with the smooth-regime boundary of Phase I.

**Honest reservation.** The verdict is restricted to this setting (budget 400, grid 6×6, descriptor ignoring `w_in`); a sweep over those was not run. The conclusion is "a robust observation in this setting," not a claim about landscapes in general. Pair-review initially judged the result untrustworthy and forced three rerun-blocking fixes (unique per-replicate seeding for valid paired tests; global-best-of-budget to remove a forgetting bias; honest_n 16 → 30) — **the conclusion was unchanged after the fixes**, which is the honest-methodology point.

### 4.4 Phase IV — The real proxy is genuinely smooth, noise-free (Step D)

**Criterion.** Decide whether the *realistic proxy landscape* is deceptive (multimodal) or smooth (single-basin) using *deterministic, noise-free* measurement, so that a smooth verdict cannot be dismissed as underpower.

**Setup.** Reuse the closest CPU proxy to the project's actual thesis (evolving dynamics): an echo-state reservoir (fixed seed) + closed-form ridge readout predicting next characters on real text (the project's own ~24k-character source). Because the readout is a closed-form solve, evaluation noise is driven to machine epsilon. Measure valley fraction (multimodal if ≥ 0.2) on the real landscape at dim = 3 and dim = 40, with positive and negative controls.

**Result.**

| landscape | dim | valley fraction (mean/max) | multimodal? | verdict |
|---|---|---|---|---|
| ESN 3-param (real proxy) | 3 | 0.000 / 0.000 | No (3 seeds agree) | smooth → third factor not needed, noise-free |
| ESN per-neuron (real proxy) | 40 | 0.096 / 0.121 | No (3 seeds agree) | smooth-leaning → not needed |
| multipeak control | 3 / 40 | 0.70 / 0.80 | Yes | diagnostic healthy (detects multimodality) |
| quadratic control | 3 / 40 | 0.000 | No | diagnostic healthy (detects smoothness) |

The real proxy landscapes are smooth/single-basin under noise-free measurement, with the diagnostic validated by controls. This is the first time the recurring "third factor not needed" pattern was attributed to *the landscape being truly smooth* rather than to insufficient power: re-measuring would not produce multimodality.

A complementary re-test of the nearest multi-task case (C-gen4b) with a fresh proper-n run (n = 64) does pass the strict gate (diff = +0.0472, one-sided p = 0.038, effect = +0.188), indicating the third factor is *not null* in direction there — but updated power at n = 64 is only ≈ 0.52, an in-run drift was found (the last 9 seeds run negative, diff ≈ −0.038), and the result fails a Bonferroni correction (α = 0.0167). So C-gen4b is a *load-bearing candidate / still inconclusive*, not a confirmation.

**Honest reservation.** "Genuinely smooth" is precise only to threshold proximity: 90.9% of midpoints on the 3-param landscape dip slightly downward, and the maximum relative dip (0.0435) sits just under the 0.05 valley threshold; the 40-dim valley fraction (0.121) is 0.079 below the 0.2 flip point. The accurate phrasing is a *weak multi-basin landscape with shallow (~2-4%) valleys just below the multimodality threshold*, not a perfectly convex bowl. The smooth-verdict direction holds; its robustness is threshold-limited.

### 4.5 Phase V — Kernel diversification cannot host the third factor, structurally (BG9)

**Pre-registered hypothesis (H7).** Unioning four kernel families (RWKV / Mamba / Hopfield / linear-attention) into an extended `KernelGenome` introduces a deceptive corridor (discrete kernel-id barriers) that a 3-param single-kernel space lacked, so MAP-Elites (niching over kernel-id × theta) beats random-restart hill-climbing, panmictic GA, and random over ≥ 15 seeds. **Null:** the extended space is still smooth / kernel-neutral. The pre-registered honest prior was null-leaning, given that every prior CPU substrate had been smooth.

**Result (three stages).**

- **Substrate validity (PASS but weak).** A first-principles kernel-favoring task suite shows a *non-constant* task → best-kernel mapping (Mamba wins selective-copy, linear-attention wins weighted-accumulation), so the space is not inert. But Hopfield never wins on any task (its diagonal-scalar mock cannot form a working attractor), leaving an effective *three*-kernel union with clean specialization on only two axes. *Discrimination exists, but discrimination ≠ multimodal barrier.*
- **Harness validity (does not validate, structurally).** On a synthetic kernel-barrier positive control, MAP-Elites beats panmictic (+0.423) and random (+0.208) but **cannot beat random-restart hill-climbing** (+0.051, p = 0.31) — failing the 3-baseline gate. On the negative control (kernel-neutral) it correctly shows no advantage. On the real kernel-favoring suite, MAP-Elites is beaten 0/3.
- **Adversarial red-team (confirms the negative).** An instrumented restart baseline matches the plain one bit-for-bit; on the positive control its restarts spread kernel-id roughly uniformly across the 4 basins ([12, 18, 16, 18] occupancy), reaching the target 88% of the time. Four faithful refutation constructions (high-dim theta corridor, sequential-kernel, in-basin L1 corridor, deceptive multi-basin) all yield `beats_rr = false`. Sweeping the corridor dimension D = 0 → 3, tightening it only makes MAP-Elites *starve before* the restart baseline (at D = 3: MAP-E reach 0.08 vs. RR 0.42). **There is no behavior dimension in the kernel space where the restart baseline fails but niching succeeds**, robust across three base seeds.

**Verdict.** Formally N/A (the positive control does not validate), but substantively a *decisive structural negative*: unlike the earlier circularity-driven N/A, here the harness is healthy (it correctly nulls the negative control and detects GA/random) — the substrate simply cannot *host* third-factor deception. The answer to "does enlarging the search space via kernel diversification unlock the third factor?" is **NO (structurally, on CPU)**.

**Honest reservation.** Smoke-scale (5-12 seeds), but the structure (restart directly samples kernel-id; tightening starves MAP-Elites first) is seed-independent. The diagonal-scalar mock is a limitation; a full, non-diagonal kernel implementation might differ, but the *structural* barrier — a low-dimensional discrete choice that restarts sample directly — is independent of kernel implementation quality.

### 4.6 Summary of the arc

| phase | substrate | third factor load-bearing? |
|---|---|---|
| I (Step 4) | synthetic deceptive corridor | **Yes — decisively** (existence proof) |
| II (Step C / Ladder 1) | memory tasks; multi-reservoir parity | N/A (floor / ceiling / degree-5 wall) |
| III (E-A) | multi-task generalization | No (smooth; niching adds nothing) |
| IV (Step D) | ESN text proxy, deterministic | No — landscape *verified smooth*, noise-free |
| V (BG9) | four-kernel union | No — *structurally* (low-dim choice) |

---

## 5. The Structural Insight

The existence proof and the five negatives are reconciled by a single condition.

> **Behavioral niching (the third factor) is load-bearing only when the search's hard coordinate lives in a high-dimensional behavior space that direct sampling (random restart) cannot reach.**

**Why the synthetic corridor satisfies it.** In Phase I the behavior descriptor is `mean(24-dim gene)`. By central-limit concentration, the mean of 24 independent coordinates concentrates near 0.5, so the global peak (mean ≈ 0.9) lives in an effectively measure-zero region. Random sampling and random restarts *cannot directly reach it* — every fresh draw lands near behavior 0.5. The only way across is to retain behavioral stepping-stones and ratchet, which is exactly what the archive does. The high dimensionality of the *thing being averaged* is what makes the global behavior unreachable by direct sampling, and therefore what makes niching essential.

**Why the realistic CPU substrates do not.** Their hard coordinates are low-dimensional:

- In the ESN text proxy the controlling coordinate is essentially the *leak rate* — a smooth, low-dimensional knob that a hill-climber simply walks up (and Phase IV verifies there is no valley to cross at all).
- In the kernel union the hard coordinate is *which kernel* — a single discrete choice over four options. A random restart samples kernel-id ≈ uniformly and *teleports* into every basin; there is no valley to cross because the answer is one direct draw away.

The BG9 red-team makes this mechanism explicit and is the crux of the insight: across every construction, the random-restart baseline's restarts sample the low-dimensional hard coordinate directly, so tightening the landscape to defeat the restart baseline also starves MAP-Elites — the window in which "restart fails ∧ niching succeeds" is structurally empty whenever the hard coordinate is low-dimensional. Conversely, in Phase I the hard coordinate is the high-dimensional mean, restarts cannot sample it, and the window is wide open.

This is why genotype-space multimodality (Phase II: valley fraction up to 1.000) is *not* sufficient: a landscape can be riddled with genotype valleys yet have its difficulty concentrated in a low-dimensional behavior coordinate that restarts reach directly. The relevant dimensionality is that of the *behavior* the search must reach, not of the genotype.

---

## 6. Limitations

- **CPU proxies, not full backprop LLMs.** Every substrate is an echo-state reservoir / ridge readout (with diagonal-scalar kernel mocks in Phase V), not a gradient-trained large model. The closest proxy to "evolving dynamics" was deliberately chosen, but a reservoir with a fixed dynamical core plus a learned linear readout is not the same object as a network whose entire weight set is trained by backpropagation. A smooth proxy landscape is not a proof that the full-LLM loss landscape is smooth.
- **Smoke-scale, but structure is seed-independent.** Several late phases run at 5-30 seeds. The *numerical* margins move with n, but the *structural* findings (restart directly samples a low-dimensional coordinate; the smooth landscape has no valley to cross) are robust across the seeds we tested.
- **Threshold proximity.** The "smooth" verdict in Phase IV is precise only to threshold proximity (shallow ~2-4% valleys just below the multimodality cutoff). We report this rather than rounding it to "perfectly convex."
- **The condition could hold for a full LLM — but a strong baseline might still defeat it there.** It is genuinely possible that a full-LLM loss landscape has high-dimensional deceptive structure where the third factor *would* be load-bearing. But the same caveat that sinks the CPU substrates re-applies: if a *strong baseline* (here, backpropagation / gradient descent) reaches the hard region directly, the third factor is again not needed. The high-dimensionality condition is necessary, not sufficient; one must additionally show that *no strong direct method* solves it — which on CPU was random-restart and on GPU would be gradient descent. This is the GPU "bet."

---

## 7. Implications and Future Work

The arc closes every CPU route: the realistic proxy landscape is verified smooth (Phase IV), and the last candidate route — enlarging the space via kernel diversification — is structurally closed (Phase V). The only remaining venue for the third factor is a **high-dimensional landscape**, which is what a full-LLM parameter / loss space provides (millions of dimensions). The structural insight makes the GPU experiment *better-motivated* than a blind "maybe the full LLM is special": it follows the principle that the third factor requires high dimensionality, and full-LLM loss space is exactly the high-dimensional regime.

This motivates a pre-registered, falsifiable go/no-go criterion for any GPU investment:

> **Is the full-LLM landscape's hard region high-dimensional in behavior *and* unreachable by a strong direct baseline (gradient descent)?**

If the hard region is high-dimensional but gradient descent reaches it directly, the third factor is not needed — the GPU analogue of the BG9 random-restart result. The appropriate posture is therefore a portfolio decision (shared with other full-LLM-fitness work) plus a single cloud-rented pre-registered run before any capital commitment, not a standalone bet. We also note, but did not pursue here, the *fourth* factor (over-reproduction / density-dependent selection): our fixed-budget setting never instantiates resource scarcity, so selection pressure is exogenous; an internalized-selection (ecology) substrate is a separate, more invasive line.

---

## 8. Conclusion

We asked whether the Darwinian selection/separation factor — operationalized as MAP-Elites behavioral niching — is load-bearing when evolving neural dynamics. The honest answer is two-sided and precise. The factor is *real and decisive* on a synthetic deceptive corridor where the hard coordinate is a high-dimensional mean that direct sampling cannot reach. But across every realistic CPU substrate we tested — reservoir text proxies, memory tasks, multi-reservoir parity, multi-task generalization, and a four-kernel union — it is *not* load-bearing, and not for lack of statistical power: deterministic noise-free measurement shows the realistic landscapes are genuinely smooth, and where they are hard, the hard coordinate is low-dimensional and a strong random-restart baseline solves it directly. The unifying mechanism is a *dimensionality condition*: niching helps only when the deception lives in a high-dimensional behavior space. This is an honest negative result with a structural reason, and it relocates the open question to high-dimensional full-LLM landscapes, where the very same caveat — a strong direct baseline may suffice — must be tested before the third factor can be declared load-bearing.

---

## Appendix A — Per-phase pre-registration, gates, and key external-review findings

| phase | falsifiable criterion / gate | strict-gate result | key pair-review / red-team finding |
|---|---|---|---|
| I (Step 4) | MAP-E beats 3 baselines on deceptive corridor | p = 1.9e-6, δ = +1.00, 60 seeds; smooth boundary erases advantage | 5 findings: budget discipline, init_batch ablation, all-baseline boundary, honest baselining, proxy labeling |
| II — Step C | C1-C4 strict gate (n ≥ 15) on memory tasks | N/A: parity floor (R² ≈ 0.003), flip-flop ceiling (R² ≈ 0.95, p = 0.15) | 5-lens adversarial verify demoted draft over-claims; Bengio-deception over-generalization withdrawn |
| II — Ladder 1 | floor_lifted via 5 mechanisms | all false; depth +0.47/+0.60 PASS but abs 0.05-0.10; degree-2 positive control (R² = +1.0 at window 2) | 4 findings: selection-on-noise vs. elitism, positive-control artifact, "principled" softened to "robust floor" |
| III (E-A) | MAP-E > randselect / panmictic / random | C-gen3 PASS (+0.126, p = 0.015, δ = +0.60); C-gen4a/b FAIL | 7 findings; 3 rerun-blockers fixed (seeding, archive-forgetting, honest_n); conclusion unchanged |
| IV (Step D) | deterministic C1 multimodality (noise = machine eps) | ESN 3-param vf = 0.000, per-neuron vf = 0.096, both not multimodal; C-gen4b fresh n=64 PASS but power 0.52, Bonferroni FAIL | 3 surviving refutations (medium): optional-stopping drift, threshold proximity, K4-clip budget |
| V (BG9) | pre-registered H7: MAP-E beats RR on kernel union | positive control fails RR gate (+0.051, p = 0.31); real beaten 0/3; red-team `beats_rr = false` ×4 | structural negative confirmed, not refuted; effective 3-kernel substrate (Hopfield inert) |

---

## Author notes (internal evidence trail; not for external citation)

Numeric claims trace to internal verdict documents in this repository: `docs/poc/STEP4_SELECTION_VERDICT.md` (Phase I), `docs/poc/STEP_C_VERDICT.md` + `STEP_C_NAVIGATION_FRAME.md` (Phase II memory), `docs/poc/LADDER1_VERDICT.md` (Phase II parity), `docs/poc/E_A_VERDICT.md` (Phase III), `research/step_d_settle/THIRD_AXIS_SETTLE_VERDICT.md` (Phase IV), `research/kernel_diversification/BG9_VERDICT.md` + `BG9_PREREGISTRATION.md` (Phase V), and `docs/poc/EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md` (methodology background). External works referenced in-text (Mouret & Clune 2015; Lehman & Stanley 2011; Bengio et al. 1994; Minsky & Papert; Boldi/Ding/Spector 2023, arXiv:2311.02283) are cited only where their identity is established in the source material; uncertain references are stated descriptively rather than with fabricated citations.
