> ⚠️ **CORRECTION (2026-06-03 audit, `research/AUDIT_SCS_CLARABEL_2026-06-03.md`):** the SDP coverage
> numbers below (e.g. "sdp admits 3369", Track D "+254 / +43%", §3 "39/50 top contracting genes in the
> D4 residual / even SDP over-conservative") used cvxpy's **SCS** default, which false-negatives near
> the feasibility boundary. Under **CLARABEL**: SDP certifies **~95%** of contracting genes (Track D
> 1291/1363; it beats 2-norm by **+692**, not +254 — the SDP advantage was *understated*), and the D4
> residual is only **~3–5%** (the "even SDP over-conservative / 39-of-50-in-residual" framing is largely
> an SCS artifact). **The core conclusion (the right verifier is SDP/Lyapunov) is strengthened**; only
> the "rich beyond-SDP residual" magnitudes are corrected. Soundness unaffected (G1/D1 still PASS).

# VERDICT — SDP-Lyapunov-gated coupled evolution (Track B + D integration)

> Goal (user, 2026-06-03): 「llcore 研究の続き。CPU 環境下で AI 進化の仕組みを実現する。」
> 「進化の実現が最優先。一度実現すれば骨組みとして拡張するだけ。」
> Result: **a working, verified, extensible CPU mechanism for evolving an AI dynamics
> core** — evolution climbs, the SDP-Lyapunov verifier provably keeps every admitted
> gene contracting, and a *better* verifier measurably unlocks more reachable fitness.
> Discipline: `research/verified_evolution_sdp_gate/`, **src untouched**, seeds fixed,
> pre-registration first, honest disclosure, adversarial red-team, Codex pair-review.

## 1. What was built (the 骨組み / skeleton)

`evolvable_core.py` — a minimal verified-evolution loop with **three pluggable
extension points**: `GeneCodec` (substrate), `Objective` (evolution direction),
`VerifierBackend` (fail-closed safety gate). Adding a direction = a new Objective;
extending the substrate or verifier = a new codec/backend. `coupled_components.py`
supplies the n=2 coupled-RWKV plugins + 4 verifier backends (none / inf_norm /
two_norm / sdp) wrapping the Track C/D certifiers. 12 TDD tests pass; `demo_evolve.py`
shows the realization end-to-end; README.md documents extension.

## 2. Gates (verifier backends), all SOUND for admitting only contractions

| backend | certificate | admits (of 10175-gene pool, exp1) |
|---|---|---|
| `none` | accept all | all |
| `inf_norm` | closed-form ‖J‖_∞<1 over achievable-t box | 1860 |
| `two_norm` | vertex σ_max(J)<1 | 1860 + 692 = 2552 |
| `sdp` | common-P quadratic Lyapunov LMI (cvxpy) | 1860 + 692 + 817 = **3369** |

SDP admits **1.8× more** genes than inf_norm (G3 ✓): the extra 692 (2-norm) + 817
(SDP-only) are genuine contractions the conservative ∞-norm structurally over-rejects.

## 3. exp1 — landscape attribution (mechanism, NON-circular, 10175 genes)

The decisive, gate-independent evidence: classify every gene by tightest certifier
and measure the best achievable **rotation-task** fitness per region.

| region (tightest sound certificate) | genes | max rotation fitness |
|---|---|---|
| `inf` (∞-norm certifies) | 1860 | **0.38** |
| `two_norm_only` | 692 | 0.89 |
| `sdp_only` | 817 | 0.90 |
| `non_certified` (ρ<1 but no quadratic/norm cert = Track-D D4 residual) | 6806 | 0.99 |

**The mechanism is the region ceiling, not GA luck**: the ∞-norm-admissible region
*cannot* exceed 0.38 on the rotation task; the 2-norm/SDP regions reach 0.90. So a
verifier that admits more (while staying sound) lets evolution reach strictly higher
fitness. Benign-task best gene sits in `inf` (fitness 1.0) — no region advantage there
(the basis of the G5 null).

**Honest limitation (must disclose):** among empirically-CONTRACTING high-fitness
rotation genes, **39 of the top 50 are `non_certified`** (ρ<1 but provable by neither a
fixed induced norm nor a common quadratic Lyapunov P — the Track-D D4 residual). All
sound gates, SDP included, reject them. So **even the SDP gate is over-conservative**
relative to the true contraction set; the highest-fitness contracting dynamics (0.99)
need a *stronger* verifier still (JSR / non-quadratic Lyapunov). The verifier-fitness
frontier is therefore monotone but unfinished:

    inf (0.38)  →  two_norm (0.89)  →  sdp (0.90)  →  [JSR / non-quadratic?]  →  empirical contraction (0.99)

SDP's unique gain over the 2-norm is small on rotation (0.90 vs 0.89; the thin SDP-only
shell, consistent with Track D). The large win is **inf → two_norm**.

*Correction (Codex #8):* the per-region max in the original `exp1_results.json` was NOT
filtered to empirically-contracting genes (a code bug, now fixed), so the `non_certified`
0.99 entry can include genuinely **expansive** genes. The certified regions (inf 0.38 /
two 0.89 / sdp 0.90) are by definition contracting, so those numbers stand. For
`non_certified`, the *contracting-filtered* ceiling is the honest one: the top-50-
*contracting* analysis (39/50 in `non_certified`) confirms high-fitness contracting genes
do populate the D4 residual, and a 3000-gene re-check shows its contracting max sits just
below the raw max (0.571 vs 0.595 on that pool) — i.e., a modest fraction of high-fitness
`non_certified` genes are divergent, but the residual genuinely holds high-fitness
*contracting* dynamics that all current sound gates miss. The frontier conclusion (a
stronger verifier would unlock them) stands; the exact 0.99 figure should be read as the
raw region max, not a certified-contracting ceiling.

## 4. exp2 — gated evolution (G0–G5)   [n=15 seeds, pop=24, gens=25]

Mean best-fitness per gate (4 gates × 3 tasks × 15 paired seeds):

| task | none | inf_norm | two_norm | sdp | sdp vs inf (paired) |
|---|---|---|---|---|---|
| **rotation** | 0.928 | **0.411** | 0.767 | **0.859** | Δ=+0.448, psd=1.00, **p=3.1e-5** |
| benign | 0.999 | 0.999 | 0.999 | 0.999 | Δ=−0.000, psd=0.20, p=0.72 (null) |
| nonnormal | 0.992 | 0.963 | 0.986 | 0.986 | Δ=+0.023, psd=1.00, p=3.1e-5 |

Winner region (rotation, 15 runs): **sdp → 12 sdp_only + 3 two_norm_only** (every
winner in an inf-REJECTED region); **inf_norm → 13 inf + 2 non_certified**. (nonnormal
sdp → 8 sdp_only + 6 two_norm_only + 1 inf.) The GA realises the exp1 region ceilings:
inf-gated rotation tops out at 0.41 (≈ region max 0.38); sdp at 0.86 (≈ 0.90).

**Fallback transparency (Codex #4):** inf-gated rotation *rejects* heavily (≈373
rejections/run) but falls back only **≈0.5×/run** — resampling almost always finds an
inf-admissible child. So inf is **not fallback-flooded**; it genuinely searches its (inf)
region and hits the 0.38 ceiling — the payoff is the region ceiling, not a fallback
artifact. The inf mean 0.41 (slightly above 0.38) is lifted by the 2 ungated init-elites.

## 5. Pre-registered gates — verdicts

- **G0 control / src-untouched — PASS.** `none` gate is deterministic (same seed →
  identical curves, test_skeleton) and admits 100 % of children. `git status` shows
  **no src/ change** throughout.
- **G1 gate soundness — PASS (by certificate; consistency-checked).** Soundness is
  *guaranteed by the certifier theorems*, not by sampling: ‖J‖_∞<1, σ_max<1, and the
  common-P LMI each mathematically imply ρ<1 (Track C/D + their red-teams). The empirical
  oracle is a **from-below falsification/consistency check** (Codex #1/#2): on the
  *admitted children* at **20 000** samples it found **0 divergent** across inf/two/sdp
  (150 unique children/gate, rotation). Honest caveat: the inf-gated rotation *final populations* held
  **1 divergent gene** total — an **ungated initial-population elite** the over-restrictive
  inf gate could not replace (it rejects almost all rotation children), kept alive by
  elitism; two/sdp showed **0** (they admit rotation children and self-clean). End-to-end
  soundness ⇒ gate the initial population too: the `gate_initial=True` option yields **0**
  divergent in inf-rotation final pops (verified), confirming the gate itself never
  false-admits.
- **G2 load-bearing — PASS (rotation/benign), marginal (nonnormal).** Ungated final pops
  drift to non-contraction: **rotation 19.7 %, benign 16.7 %, nonnormal 1.9 %**; all sound
  gates drive admitted children to 0. (nonnormal 1.9 % < 5 % → the gate has little to
  exclude there; reported honestly, not a no-op elsewhere.)
- **G3 permissiveness ordering — PASS.** |sdp|=3369 > |two|=2552 > |inf|=1860 over the
  exp1 pool (SDP admits 1.8× more, all sound).
- **G4 PRACTICAL PAYOFF — PASS (strong, on the non-circular rotation task).** sdp ≫ inf
  (Δ=+0.448, one-sided Wilcoxon **p=3.1e-5**, psd=1.00) and two ≫ inf (p=3.1e-5);
  **attributed** to winners in the inf-rejected region (15/15 sdp winners
  two_norm_only/sdp_only). The headline p=3.1e-5 **survives Bonferroni** over all ~9
  gate/task comparisons (α=0.05/9=5.6e-3) and psd=1.00 is complete separation (post-hoc
  power ≈ 1). The `nonnormal` task also shows sdp>inf (p=3.1e-5, Δ=0.023) but is
  **circular by construction** (optimum SDP-only by design) → counted as a *reachability
  demonstration*, NOT independent payoff evidence (Codex #7).
- **G5 HONEST NULL — PASS.** On benign, sdp vs inf is a null (Δ=−0.000, p=0.72): no
  generic SDP/admission-size advantage. The G4 effect is task-structural (Lens A confirms).

## 6. Adversarial red-team (Lenses A–D) — all PASS / survive

- **Lens A — admission-size artifact: REFUTED (good).** With a fitness UNCORRELATED with
  dynamics (a per-gene pseudo-random score), sdp vs inf is a null (psd=−0.40, Δ=−0.0005,
  **p=0.91**) — SDP gets *no* edge from merely admitting more genes. So the G4 payoff is
  **fitness-structural**, not an artifact of the larger admission set. *Caveat (Codex #5):*
  the control is gene-KEYED random (a fixed random value per gene), not the prereg's pure
  per-evaluation noise; the numeric-tuple key hash is deterministic across processes
  (verified), so Lens A is reproducible. A pure-noise variant is left as a stronger future
  control.
- **Lens B — independent soundness: PASS (reduced scope).** sdp+two_norm winners on
  rotation+nonnormal (60 winners) re-checked with an independent 50k-sample oracle
  (different seed) + long-horizon separation: **0 divergent**. *Deviation (Codex #3):* the
  prereg asked for every admitted winner at 100k + dense edge search; this run covers the
  two richest sound gates (the main soundness risk) at 50k. The soundness *guarantee* does
  not rest on this sampling anyway — see G1 / §7.
- **Lens C — power / seed robustness: PASS.** G4 (sdp ≫ inf on rotation) holds across **3
  independent base-seed families** (Δ=0.46/0.46/0.44, each **p=3.1e-5**), all surviving
  **Bonferroni** (α=0.0167). Not seed-cherry-picked.
- **Lens D — mechanism attribution: CONFIRMED.** sdp winners live in the inf-REJECTED
  regions — rotation: **12 sdp_only + 3 two_norm_only / 0 inf**; nonnormal: 8 sdp_only +
  6 two_norm_only + 1 inf. inf winners stay in `inf` (+2 ungated init-elites). The payoff
  comes precisely from the extra reach the better verifier unlocks.

The supplementary admitted-child check confirmed the §5 G1 caveat: with `gate_initial=True`
the inf-gated rotation final populations carry **0** divergent genes (end-to-end sound),
proving the single default-run divergent was an ungated initial elite, not a gate false-admit.

## 7. Honest disclosure / deviations

- **Config deviation from PREREGISTRATION §4** (pop=30, gens=40, n=20): run at
  pop=24, gens=25, **n=15** for CPU tractability (cvxpy per-call canonicalisation is the
  bottleneck). n=15 still satisfies the G4 strict-gate threshold (n≥15). The gates are
  structural (soundness / region-ceiling / null) and seed-robust, not config-sensitive.
- **Soundness rests on the certifier THEOREMS, not the empirical oracle (Codex #1/#2).**
  ‖J‖_∞<1, σ_max<1, and the common-P LMI each *mathematically imply* ρ<1 (Track C/D, with
  their own red-teams). `empirical_spectral_radius` is a **from-below** sampler: ρ<1
  observed only means *no violation found* — it can falsify but not certify soundness. So
  the oracle is used as a **consistency / bug check** (it found 0 divergent admitted
  children at 20k — see §6), and the guarantee itself is the certificate. The default-run
  G1 measurement using the *final population* @2500 was weaker than the prereg's
  *admitted-children* @≥20k; the proper admitted-child check is reported in §6.
- **Multiplicity / power (Codex #6):** exp2 produces several gate/task comparisons; the
  headline rotation p=3.1e-5 ≪ Bonferroni α (0.05/9=5.6e-3), and Lens C confirms across 3
  seed families. psd=1.00 (complete separation) ⇒ post-hoc power ≈ 1 for the observed
  effect. No achieved-power simulation was run (deviation from prereg Lens C wording).
- **Thin SDP-only shell (Codex #9):** the sdp_only genes sit very near the quadratic-
  stability boundary (common-P min-eigenvalue margins ~1e-7, not a solver artifact — same
  with CLARABEL). The Track-D certifier independently re-checks the LMI eigenvalues (not
  solver-blind), but near-boundary numerical robustness is a real caveat; it is *why* SDP's
  unique gain over the 2-norm is small/fragile, and it does not affect the inf→2norm bulk.
- **Fitness ceiling caught & fixed**: the first fitness (exp(−MSE)) saturated at ~0.997
  for most genes (the ③/llive ceiling trap); replaced with **R²** (project-standard,
  headroom) before any verdict — disclosed per [[feedback_benchmark_honest_disclosure]].
- **NonNormalObjective is a *reachability test*, not a naturalness claim**: its target is
  the free response of a hand-picked SDP-only reference gene, so its optimum lives in the
  SDP-only region by construction. It tests whether the SDP gate can *reach* dynamics the
  conservative gates forbid; it does NOT claim natural tasks are non-normal. Stated openly.
- **A behaviour-preserving SDP fast-path** (reject when ρ(J_v)≥1 at any t-box vertex,
  a necessary condition for the LMI) was added to make the sweep tractable. Gate
  decisions are provably identical (verified: 12 TDD tests unchanged).
- **Push**: none (llcore remote not created — exposure avoidance). Local commit only.

## 8. Bottom line

**A working, verified, extensible CPU mechanism for evolving an AI dynamics core is
realized** (the user's "進化の実現"). Evolution climbs; the SDP-Lyapunov verifier keeps
every admitted gene provably contracting (and, with `gate_initial=True`, the whole
population end-to-end). The skeleton's three plug-points (GeneCodec / Objective /
VerifierBackend) make "adding an evolution direction" and "feature extension" one-object
changes — the骨組み the user asked for.

**Scientific result (all gates + 4 red-team lenses survive):** *a better contraction
verifier monotonically unlocks more evolutionary fitness at no soundness cost.* The
conservative ∞-norm gate severely over-restricts evolution on coupled/rotational dynamics
(rotation fitness capped at ~0.41), and the SDP/2-norm gate recovers it to ~0.86–0.90
(G4, p=3.1e-5, robust across seed families, **not** an admission-size artifact, attributed
to the inf-rejected region), while a benign task shows no difference (G5 null). The big
win is **inf → 2-norm** (capturing rotational contractions the ∞-norm over-rejects); SDP's
unique gain over 2-norm is real but thin (the SDP-only shell). And even SDP is
over-conservative: the highest-fitness *contracting* dynamics sit in the Track-D **D4
residual** (ρ<1 but no quadratic/norm certificate), so the verifier-fitness frontier
continues to **JSR / non-quadratic Lyapunov** as the next backend.

**This validates the llcore arc thesis operationally**: the right verifier backend is
SDP/Lyapunov (not Z3/SMT), and putting it *inside* the evolutionary loop is the correct,
load-bearing implementation of "Verified × Evolvable" — on CPU.

**Next (skeleton extensions, each a plug-in):** (1) richer `GeneCodec`s — n≥3 coupled,
multi-kernel union, learning-rule genes; (2) a JSR / non-quadratic-Lyapunov
`VerifierBackend` to reach the D4 residual; (3) promote the SDP gate into a `src/`
`verifier backend plugin` (Stage 3b) once a codec stabilises; (4) more `Objective`s
(memory/oscillation/control tasks) as evolution directions.

Artifacts: `evolvable_core.py` (skeleton) · `coupled_components.py` · `test_skeleton.py`
(12 PASS) · `demo_evolve.py` · `exp_runner.py` (exp1/exp2) · `redteam.py` (A–D) ·
`PREREGISTRATION.md` · `README.md` · results JSONs. src/ untouched; push deferred
(llcore remote not created — exposure avoidance).
