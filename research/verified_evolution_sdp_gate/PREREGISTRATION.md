# PRE-REGISTRATION — SDP-Lyapunov-gated coupled evolution (Track B + D integration)

> Pre-committed **before** running any experiment cell. Falsifiable gates fixed here.
> Discipline: `research/verified_evolution_sdp_gate/` isolated, **src/ untouched**, seed fixed,
> honest disclosure ([[feedback_benchmark_honest_disclosure]]), Codex pair-review after
> ([[feedback_codex_pair_review_for_llcore]]). Date: 2026-06-03.

## 0. Motivation (what this settles)

The CPU Verification-pillar arc (Track A→B→C→D, `CPU_VERIFICATION_RESEARCH_2026-06-02_VERDICT.md`)
established: **the right contraction verifier for llcore is SDP/LMI (Lyapunov), not Z3/SMT**.
- Track B proved a contraction **gate** is *load-bearing* in the (scalar) GA: it excludes 15–30 %
  non-contraction drift while being sound.
- Track D proved the **SDP** certifier genuinely beats every fixed induced norm on the *coupled*
  map: it certifies **+254 genes (+43 %)** that the ∞-norm / 2-norm reject — P-weighted
  contractions a fixed norm structurally cannot see.

Track B (gate is load-bearing) was run on the **scalar** gene, where all certifiers collapse to
the same closed form (so "which verifier" did not matter). Track D (SDP beats norms) was a static
*audit* of a fixed gene pool, not an evolutionary loop. **This experiment is their integration**:
put the SDP gate **inside** the coupled evolutionary loop and ask whether a *better verifier*
changes **what evolution finds** — the practical payoff of SDP over the conservative norm gate.

This realizes llcore's founding thesis — *evolve the dynamics without letting the verifier break
them* — with the verifier backend the arc concluded is correct (SDP), on CPU.

## 1. Substrate (fixed)

Coupled n=2 RWKV-style map (Track C `coupled_map.py`, reused verbatim, no src touch):

    s' = decay ⊙ s + (1 − decay) ⊙ tanh(W s + V x),   s,x ∈ R², decay∈[0,1]², W∈[−2,2]^{2×2}, V=I

Genotype = 6 reals (decay[2] + W[2×2]). Off-diagonal W = coupling (this is why SDP can beat
the ∞-norm; in the diagonal/scalar regime they coincide).

State Jacobian J(t) = diag(decay) + diag((1−decay)·t)·W, t_i = sech²(pre_i) ∈ (0,1].

## 2. Gates (verifier backends) — all SOUND for *admitting only contractions*

A gate admits a child gene iff its certifier certifies contraction over the achievable-t box
(`t_domain="tmin1"`, the tighter sound floor). Fail-closed: reject → resample (cap), else fallback.

| mode        | certifier | source | admits |
|-------------|-----------|--------|--------|
| `none`      | (accept all) | — | every child |
| `inf_norm`  | closed-form ‖J‖_∞ < 1 over box (= Z3 ∞-norm, identical) | Track C | smallest set (conservative) |
| `two_norm`  | vertex σ_max(J) < 1 | Track D `two_norm_vertex_certifier` | ⊇ inf on near-symmetric W |
| `sdp`       | common-P quadratic Lyapunov LMI (cvxpy) | Track D `lyapunov_sdp_certifier` | richest (P-weighted) |

Established (Track D): SDP ⊇ 2-norm (P=I always feasible); SDP vs ∞-norm are **non-nested**
(∞-norm catches a few SDP misses), but SDP admits far more on net (855 vs 513 over 3270 genes).
A maximally-permissive sound gate would be the **union**; we report `sdp` and also `union`
(inf ∨ 2norm ∨ sdp) as the "best available sound verifier".

## 3. Tasks (deterministic fitness — NO eval RNG, per Step D noise-free lesson)

Fitness is a deterministic function of the gene (no per-eval random draws), so a flat landscape
is genuinely flat, not noise (the Step D EXP2 correction). Both tasks are autonomous (x=0) or
fixed-input free responses scored against a fixed target trajectory.

- **Task R (rotation / oscillatory memory)** — rewards *coupled* contracting dynamics near the
  stability boundary. Target = a damped 2-D rotation trajectory the map must reproduce from a
  fixed initial state. Producing rotation needs complex Jacobian eigenvalues (antisymmetric W
  component) ⇒ large row-abs-sums ⇒ ‖J‖_∞ ≥ 1 even when ρ(J) < 1. So the *good* genes tend to
  be **∞-norm-rejected but 2-norm/SDP-certified**. We do NOT assume the optimum is SDP-exclusive;
  we measure which region the high-fitness genes occupy (§5 exp1).
- **Task B (benign decay)** — rewards simple diagonal-ish contraction (target = damped
  exponential decay, no rotation). Optimum lives **inside** the ∞-norm region. Used as the
  honest-null control: a generic SDP advantage here would expose an artifact.

Fitness = exp(−MSE(trajectory, target)) ∈ (0,1] (bounded, deterministic). Higher = better.

## 4. GA config (fixed)

pop=30, generations=40, tournament_k=3, uniform crossover rate=0.5, Gaussian mutation σ=0.15
(clipped to box), elitism=1, resample_cap=50, fallback = known-safe gene
(decay=[0.5,0.5], W=0 ⇒ ‖J‖_∞=0.5). n_seeds = 20 (paired: same seed → same init pop & RNG
stream across gate modes, common-random-numbers for paired tests). Certifier results memoized
per rounded genotype to bound cvxpy cost.

## 5. Experiments

- **exp1 (landscape attribution, NON-circular mechanism evidence)**: independent of the gated
  runs. Sample a large gene pool (random + a no-gate GA's visited genes) and classify each by
  (a) deterministic fitness on Task R / B, (b) tightest certifier class ∈ {non_contracting,
  inf, two_norm_only, sdp_only}. **Question**: do the *high-fitness* Task-R genes concentrate in
  the ∞-norm-rejected-but-(2norm/SDP)-certified region? If yes → the better verifier's extra
  admissions are fitness-relevant (mechanism). If high-fitness genes are all ∞-norm-admissible →
  no payoff possible (honest negative, reported as such).
- **exp2 (gated evolution)**: 4 gates × 2 tasks × 20 seeds. Record best-fitness curve, final
  winner gene + its region class, and per-gate admit/reject/fallback counts.

## 6. Falsifiable gates (PASS/FAIL fixed now)

- **G0 — control / src-untouched**: `gate='none'` coupled GA is deterministic (same seed →
  identical best/diversity curve) and admits 100 % of children (0 rejections). `git status`
  shows **no src/ change**. *FAIL if any nondeterminism, or src/ modified.*
- **G1 — gate soundness (the safety guarantee)**: every gene **admitted** by `inf_norm`,
  `two_norm`, `sdp` over the whole run is genuinely contracting: empirical spectral radius
  ρ(J) < 1 from a dense (s,x) box oracle (≥20 k samples + sign-corners). **false_admit = 0**
  for all three gates. *FAIL if any admitted gene has empirical ρ ≥ 1 (sound gate let a
  divergent gene through).*
- **G2 — gate is load-bearing**: the `none` (ungated) final populations contain a non-trivial
  fraction (pre-registered threshold **≥ 5 %** averaged over seeds) of empirically non-contracting
  genes (ρ ≥ 1 or ‖J‖_∞-divergent under the trajectory), which all three sound gates drive to
  **0 admitted**. *FAIL if ungated drift < 5 % (gate has nothing to exclude = no-op, like the
  scalar state_norm gate) — reported honestly as no-op if so.*
- **G3 — permissiveness ordering**: over the union of genes encountered, admit-set sizes satisfy
  |sdp| ≥ |two_norm| and |sdp| > |inf_norm| (SDP strictly more permissive on net). Report the
  non-nesting count (genes inf-admits that sdp rejects). *FAIL if sdp does not admit strictly
  more than inf_norm (then SDP adds no reach in this regime).*
- **G4 — PRACTICAL PAYOFF (headline)**: on **Task R**, best-fitness(`sdp`) > best-fitness
  (`inf_norm`), **paired over the 20 seeds**, one-sided Wilcoxon signed-rank p < 0.05 AND
  paired-sign effect |δ| ≥ 0.147 (the project's standard gate from `strict_compare`). AND the
  fitness advantage is **attributable** to winners in the ∞-norm-rejected region (exp1 + winner
  region class). *FAIL → SDP gate gives no measurable evolutionary payoff over the conservative
  gate (honest negative: conservative verifier costs nothing here).*
- **G5 — HONEST NULL (anti-overclaim)**: on **Task B** (benign), best-fitness(`sdp`) ≈
  best-fitness(`inf_norm`): NO significant difference (two-sided Wilcoxon p > 0.05 OR |δ| < 0.147).
  *FAIL if SDP beats inf on the benign task too → the G4 effect is a generic SDP/admission-size
  artifact, not task-structural; headline retracted.*

## 7. Adversarial red-team (run AFTER, independent lenses)

- **Lens A — circular / tautology**: random-fitness control. Replace fitness with a gene-
  independent random scalar; if `sdp` still beats `inf_norm`, the G4 effect is an artifact of
  admission-set SIZE (more genes to sample), not fitness structure. Must show sdp ≈ inf on
  random fitness (alongside G5 benign null).
- **Lens B — soundness independence**: re-verify G1 false_admit=0 on every admitted winner with
  an INDEPENDENT brute-force oracle (different seed, 100 k samples + dense corner/edge search +
  long-horizon trajectory divergence). Any divergent admitted gene falsifies soundness.
- **Lens C — power / seed robustness**: re-run G4 across ≥3 base-seed families; report achieved
  power at n=20; flag optional-stopping / multiple-comparison fragility (Bonferroni over the
  gates compared).
- **Lens D — mechanism attribution**: confirm the G4 winners genuinely live in the region
  ∞-norm rejects (2norm-only ∪ sdp-only), not a coincidental ∞-norm-admissible optimum. If the
  payoff winners are ∞-norm-admissible, the stated mechanism is wrong (retract mechanism claim
  even if the fitness gap is real).

## 8. What each outcome means (pre-committed interpretation)

- **G1 PASS + G2 PASS + G4 PASS + G5 PASS + all lenses survive** → **SDP-gated verified evolution
  is a real CPU mechanism**: a better (sound) verifier lets evolution reach higher-fitness
  contracting dynamics the conservative gate forbids, never admitting a divergent gene. This is
  the positive integration result (Verified × Evolvable, correct backend).
- **G1 PASS + G2 PASS + G4 FAIL (G5 PASS)** → verified evolution is sound & load-bearing, but the
  *choice* of sound verifier (SDP vs ∞-norm) has **no evolutionary payoff for these tasks**
  (the conservative gate costs nothing). Honest negative; still a clean Verified×Evolvable result.
- **G4 PASS but Lens A or D fails** → the apparent payoff is an artifact (admission-size or
  mis-attributed); headline retracted, reported as honest correction.

No push (local-only phase at the time). git via orchestrator, one batched commit.
