# AUDIT — re-measuring the verified-evolution arc with an accurate SDP solver (CLARABEL) (2026-06-03)

> User directive: *"過去成果も CLARABEL で再監査して"*. Trigger: the degree-6 work discovered that
> cvxpy's **SCS** default (first-order) returns **false negatives** near the SDP feasibility boundary
> (memory `feedback_cvxpy_pin_accurate_solver`), fabricating false structure. This audit pins
> **CLARABEL** across every SDP certifier and re-runs the arc's coverage experiments. Soundness was
> never at risk *in direction* (the artifact is false *negatives*), and we observe **0 false admits**;
> but read the soundness claim precisely (pair-review F2, 2026-06-04): admission rests on the
> certifier **theorems** (quadratic: vertex-LMI ⇒ hull stability by convexity; deg≥2: the lifted SOS
> sufficient condition) plus an **independent float eigen re-check** of the solver's `P`. The JSR
> product oracle is a **one-sided, finite-length (≤6) falsification check** — `jsr_lb≥1` would prove
> an unsound admit, and we find **0** such genes; that is strong *evidence* of soundness, NOT an
> end-to-end machine-checked proof at float precision. The additive Rump verified-PD gate is the only
> component that supplies such a proof, and it is not the default. See `PAIRREVIEW_audit_rump_2026-06-04.md`.

## Solver pinned (whole codebase, additive)

`verifier_deg4.py`, `verifier_deg6.py`, `coupled_components._sdp_certifies`,
`spectral_lyapunov_contraction/exp_d_runner.py` (Track D), `coupled_nd.cert_sdp`, **and `src/llcore/
verifier/backends.py`** (the production Stage-3b SdpLyapunov backend). `prob.solve(solver=CLARABEL)`
with fail-closed fallback. Tests after: src 255 + research deg 19 + skeleton/nd/obj 27 + backends 7
= all PASS, 0 regressions.

## Results — SCS (original) vs CLARABEL (honest)

| experiment | quantity | SCS (reported) | CLARABEL (honest) | reading |
|---|---|---|---|---|
| **EXP-A deg6 ladder** (300 contr.) | quadratic SDP certifies | 193 (64%) | **286 (95%)** | SDP undercounted |
| | deg4/6 add over sdp | +54 | **+4** | "degree unlocks much" = artifact |
| | deg4∖deg6 / deg6∖deg4 | 23 / 13 | **0 / 1** | complementarity = artifact (nested) |
| | residual (deg6 misses) | 53 | **10** | residual inflated 5× by SCS |
| **deg4 #2** (1200 genes) | D4 residual | 172 | **31** | residual inflated 5.5× |
| | deg4 recovers | 57 (33.1%) | **4 (12.9%)** | "33% recovery" = mostly SCS-missed sdp |
| **Track D** (3270 genes) | SDP certifies (tmin1) | ~855 | **1291 (95% of 1363)** | SDP undercounted |
| | SDP beats 2-norm (ρ<1) | 254 (+43%) | **692** | SDP advantage UNDERSTATED 2.7× |
| | two-beats-sdp | (some) | **0** | SDP is a true superset of 2-norm |
| | D4 residual | (inflated) | **72 (5.3%)** | residual deflated under accurate solver |
| **Track B exp1** (10175 genes) | sdp_only / sdp total | 817 / 3369 | **2168 / 4720 (94% of 5021 contr.)** | SDP undercounted (same pattern) |
| | non_certified (contracting) | 6806 | **5455 (only 301 contr.)** | residual deflated |
| | top-50 rotation genes in residual | 39 | **23** | "even SDP over-conservative" reduced |

Soundness in every audit: **0 *observed* false admits** (Track D D1 PASS worst pnorm-gain ≤1.0;
deg4/deg6 unsound 0; JSR oracle 0 genes with jsr_lb≥1). NB (pair-review F2): the JSR oracle is a
one-sided lower-bound check over products to length 6 — `0 genes with jsr_lb≥1` is *necessary but not
sufficient* for soundness; the guarantee itself is the certifier theorem + independent eigen re-check,
not the oracle.

## The two-directional contamination pattern

SCS false negatives biased magnitudes **both ways**, but the core thesis the same direction:
- **Where SDP was the headline (Track D "SDP beats 2-norm"):** SCS made SDP look *weaker* (254 vs the
  true 692) → the SDP advantage was **understated**; the conclusion is **strengthened**.
- **Where "beyond SDP" was the headline (deg6 complementarity, deg4 "33% residual recovery"):** SCS's
  missed-sdp-certificates inflated the *residual* and made degree-4/6 look like they recovered a big
  class → those "beyond-SDP" findings were largely **fabricated** and are **retracted**.

**Unifying truth:** with an accurate solver, a **common quadratic Lyapunov (SDP) certifies ~95% of
contracting evolved dynamics** (EXP-A 286/300, Track D 1291/1363). The degree-4/6/8 SOS rungs add only
a **tiny near-boundary residual** (a handful of genes; degree-8 closes most, ~2 need exact-JSR). The
arc thesis — **the right verifier backend is SDP/Lyapunov** — is **confirmed and strengthened, and
shown to be nearly *complete* on CPU**. The "rich degree ladder beyond SDP" and "complementarity"
sub-narratives were SCS artifacts.

## What this corrects in the prior verdicts

- **DEG6_VERDICT.md** — already corrected (complementarity retracted, residual 53→10, dimension
  retracted, degree-8 closes 2-4 of the tiny residual). Stands.
- **DEG4_VERDICT.md (#2)** — "D4 residual 172, recovered 57 (33%)" → CLARABEL: **residual 31, recovered
  4 (13%)**. The 33% was an SCS artifact; true degree-4 contribution is marginal. *Add correction note.*
- **VERDICT.md (Track B+D integration) / D_VERDICT.md (Track D)** — "SDP +254 over 2-norm (+43%, 855 vs
  599)" → CLARABEL: **SDP +692 (1291 vs 599)**, two-beats-sdp 0. SDP's advantage was understated; the
  conclusion strengthens. The "39/50 top contracting genes in the D4 residual / even SDP
  over-conservative" framing (VERDICT §3) is largely an SCS artifact — under CLARABEL the D4 residual is
  ~5% (Track D) / ~3% (EXP-A), and SDP captures the high-fitness contracting dynamics. *Add correction.*
- **STAGE3B_VERDICT.md / src backends.py** — the production SDP backend used SCS; now pinned to
  CLARABEL (255 + 7 tests still pass). Production verifier is now accurate. *Add note.*

## Reproducibility — the keystone is now permanently runnable (2026-06-03 fix)

The SCS→CLARABEL solver-swap keystone (DEG6_VERDICT.md §0) is once again reproducible from a fixed
fixture, after an audit found the demonstration had silently broken: the later CLARABEL ladder run
**overwrote** `verified_evolution_sdp_gate/exp_deg6_residual_genes.json` (SCS-era 54 deg_certified /
53 residual_uncert → post-correction 4 / 10), destroying the battery `verify_solver_artifact.py`
depended on. Fix:

- **Immutable snapshot** `exp_deg6_residual_genes_scs_era.json` — the SCS-era battery (deg_certified
  = 54, residual_uncert = 53) restored verbatim from git commit `cd400ef` and treated as a fixed
  regression fixture.
- **`verify_solver_artifact.py`** now reads that snapshot (not the overwritten live file) and
  reproduces §0 first-party: **SCS** deg4_only=23 / deg6_only=13 / both=18 (complementarity = the
  artifact), **CLARABEL** 0 / 0 / 54 (nested), and CLARABEL recovers **42–43 of the 53** SCS false
  negatives.
- **`verified_evolution_sdp_gate/test_solver_swap_regression.py`** — a standing, deterministic,
  sub-minute pytest guard on the same snapshot asserting (a) the CLARABEL ladder is nested
  (deg4_only == 0), (b) the SCS vs CLARABEL admit sets diverge (the solver-swap detector still
  fires), and (c) every CLARABEL-certified gene passes the JSR soundness oracle (jsr_lb < 1). This
  permanently fail-closes if the certifiers ever silently revert to the SCS default.
- The pre-correction `redteam_deg6_results.json` (still reporting the retracted
  `complement_both_pos = true` / `n_deg_certified = 54`) is annotated as superseded by
  `redteam_deg6_results.STALE.md`; the canonical nested result is `exp_deg6_ladder_results.json`
  (deg4∖deg6 = 0).

## Honest bottom line of the audit

The verified-evolution arc's **central claim is intact and stronger** ("the right contraction verifier
is SDP/Lyapunov, and on CPU it is ~95% complete"). Its **quantitative "beyond-SDP degree ladder"
results were inflated by an SCS solver artifact** and are corrected: the true beyond-SDP residual is a
~3–5% near-boundary tail, degree-8 closes most of it, and exact-JSR is needed only for a couple of
genes. **The most valuable transferable outcome is the methodology lesson** (`feedback_cvxpy_pin_
accurate_solver`): pin an accurate SDP solver, never trust SCS near the feasibility boundary, and use
adversarial multi-perspective review + solver-swap (not margin sweeps) to catch it.

**Track B exp1 — DONE (confirms the pattern at scale):** sdp_only 817→2168, sdp certifies **94%** of
the 5021 contracting genes (4720), non_certified deflated to 5455 (only **301 contracting**), top-50
rotation-in-residual 39→23. *Honest nuance:* exp1's residual is ~6% (301/5021) vs EXP-A's ~3% (10/300)
— pool-dependent (exp1's 10175 includes a grid + heavier non-normal sampling). The 23 top-50 rotation
genes in the residual do NOT resurrect a capability payoff: `verify_deg4_payoff.py` decisively showed
the sdp gate's *optimisation* ceiling reaches the rotation optimum (0.98, sdp_only winner) while the
deg regions top out below it (random-sample region max 0.90 under-estimates; gated optimisation is the
right measure). So the capability NULL above SDP stands; the residual is a near-boundary/expansive tail.

**Track B exp2 (G4 evolution payoff) — not re-run:** the inf→sdp payoff only strengthens under CLARABEL
(sdp certifies more), and the capability NULL above SDP is settled by DEG6_VERDICT §5. Low value to re-run.

Artifacts: `exp_deg6_ladder_results.json`, `exp_deg4_results.json`, `exp_d_results.json`,
`verify_solver_artifact.py`, this audit. src backends.py + 5 research certifiers pinned. push deferred.
