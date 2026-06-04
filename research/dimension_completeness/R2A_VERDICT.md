# R2a VERDICT — higher-degree SOS RECOVERS only a MINORITY of the dimension-grown residual; the bulk is FUNDAMENTAL (switched-expansive)

**Verdict: PARTIAL, leaning FUNDAMENTAL.** The lifted higher-degree SOS hierarchy (degree-4 =
symmetric-2nd power, degree-6 = sym-3, degree-8 = sym-4) recovers **only ~22% (n=3) and ~29% (n=4)**
of the R1 dimension-grown residual. It does NOT close a majority. Decisively, the bulk of the
still-residual is **switched-expansive** (a product of the box-vertex Jacobians genuinely expands,
`jsr_lb ≥ 1`): those genes are FUNDAMENTALLY beyond ANY common-Lyapunov / box-switched SOS approach
at any degree — not merely beyond a finite degree. **0 unsound recoveries.** Lift math re-verified
(max error 7.1e-14). CLARABEL pinned and confirmed.

## Headline numbers (exact, per-n / per-degree, cumulative)

R1 residual = empirically-contracting genes (ρ<1 at high samples) that the **quadratic** common-P
SDP could NOT certify (R1: n=3 → 41 genes, n=4 → 104 genes; the grown residual).

| n | R1 residual | deg4 (sym-2) | +deg6 (sym-3) | +deg8 (sym-4) | cumulative recovered | still residual | unsound |
|---|-------------|--------------|----------------|----------------|----------------------|----------------|---------|
| 3 | 41          | 7  (17.1%)   | 9  (22.0%)     | 9  (22.0%)     | **9 / 41 = 21.9%**   | 32             | **0**   |
| 4 | 104         | 17 (16.3%)   | 28 (26.9%)     | 30 (28.8%)     | **30 / 104 = 28.8%** | 74             | **0**   |

NEW recoveries first achieved at each rung (cumulative ⇒ deg4 ⊆ +deg6 ⊆ +deg8):
- n=3: deg4 +7, deg6 +2, deg8 +0  → degree-8 adds NOTHING beyond degree-6 at n=3.
- n=4: deg4 +17, deg6 +11, deg8 +2 → degree-8 adds only 2 of 104 (diminishing returns).

## Why the verdict is FUNDAMENTAL, not "just needs more degree" — the JSR attribution

The still-residual was attributed via the n-dim JSR product lower bound over the 2^n box-vertex
Jacobians (`jsr_lb`, capped at the same per-n product length R1 used: n=3 len 4, n=4 len 3):

| n | still residual | switched-expansive (`jsr_lb ≥ 1`) | finite-gap (`jsr_lb < 1`) | max gap `jsr_lb` |
|---|----------------|-----------------------------------|---------------------------|------------------|
| 3 | 32             | **20** (62.5%)                    | 12 (37.5%)                | 0.9981           |
| 4 | 74             | **47** (63.5%)                    | 27 (36.5%)                | 0.9996           |

- **Switched-expansive (jsr_lb ≥ 1, the MAJORITY of the still-residual at both n):** a product of
  the t-box vertex Jacobians has spectral radius ≥ 1, i.e. the box-switched linear system is genuinely
  expansive. NO common Lyapunov function of ANY degree can certify contraction over that box —
  this is a hard, degree-independent obstruction. These genes are nonetheless empirically
  contracting because the actual nonlinear map never visits the conservative box's expansive corner
  (the `t_min`-based box over-approximates the reachable Jacobian set). Recovering them requires a
  **tighter, non-box reachability bound** (or exact JSR of the actual reachable set — NP-hard), NOT
  a higher SOS degree. This is the FUNDAMENTAL part of the dimension degradation.
- **Finite-gap (jsr_lb < 1, the minority):** JSR{box vertices} < 1, so a sufficiently high SOS
  degree could in principle certify them. But the degree ladder is already nearly flat here (deg8
  added 0 at n=3, 2 at n=4, with max gap jsr_lb crowding 0.998–1.000), so even these sit in the
  near-boundary tail where CPU-feasible degrees stall.

## Soundness (RG-sound) — PASS, 0 unsound

Every gene admitted by a lifted-SOS certifier was re-verified from below, INDEPENDENT of the solver:
- (a) empirical spectral radius at 20,000 samples: max over ALL recovered genes was **0.9951 (n=3)**
  and **0.9974 (n=4)** — strictly < 1.
- (b) n-dim JSR product lower bound over the 2^n box-vertex Jacobians: max over ALL recovered genes
  was **0.9967 (n=3)** and **0.9988 (n=4)** — strictly < 1.
- **0 recoveries with ρ ≥ 1 or jsr_lb ≥ 1 at either n.** The lifted full-space LMI on the symmetric
  k-th power is a genuine (sound) sufficient certificate here. The solver status was NEVER trusted:
  the certifier independently re-checks P ≻ 0 and every decrease-LMI ≻ 0 by eigenvalue
  (`verifier_deg6.certify_degN`). A CLARABEL "solution may be inaccurate" warning fired on some
  near-boundary solves; it is benign because the admit is gated by the independent eigen re-check,
  not the solver status — an inaccurate solution that still yields a positive-definite, decrease-LMI-
  satisfying P is a valid certificate, and any that does not is rejected.

## CLARABEL confirmation (HARD guardrail) — PASS

`_CLARABEL_OK=True`, `_SOLVER is cp.CLARABEL` → confirmed at run start; the experiment ABORTS
otherwise. Installed solvers: `['CLARABEL','SCS','SCIPY','HIGHS','OSQP']`. No SCS path is ever taken
(the certifiers fail-closed to `False` if CLARABEL is absent). This matches the post-SCS-artifact
discipline of the surrounding arc.

## Lift-math sanity (RG, pre-run) — PASS

`sym_power(A, k)` was checked against a brute-force monomial transform `m(Az) = sym_power(A,k) m(z)`
on random n=3 and n=4 matrices for k ∈ {2,3,4} BEFORE trusting any recovery. Max error
**7.1e-14** (< 1e-8 gate). Per-case (worst over 3 random draws):

| n | degree | sym order | lift dim | max err |
|---|--------|-----------|----------|---------|
| 3 | 4 | 2 | 6  | 4.4e-16 |
| 3 | 6 | 3 | 10 | 3.6e-15 |
| 3 | 8 | 4 | 15 | 1.1e-15 |
| 4 | 4 | 2 | 10 | 2.7e-15 |
| 4 | 6 | 3 | 20 | 8.7e-15 |
| 4 | 8 | 4 | 35 | 7.1e-14 |

deg8 at n=4 (35×35 P over 16 vertices) was tractable: the n=4 sweep over all 104 residual genes
through deg4+deg6+deg8 ran in ~2233 s; n=3 (41 genes) in ~25 s. No degree was skipped.

## Gate ledger (registered in PREREGISTRATION_R2a.md before measuring)

- **RG-recover** (cumulative through deg8): n=3 → 9/41 = 21.9%; n=4 → 30/104 = 28.8%. Recorded.
- **RG-sound** (0 unsound across all n, all degrees): **PASS** (0 unsound).
- **RG-verdict** (thresholds fixed in prereg): recoverable iff > 50% at BOTH n; fundamental iff
  < 25% at n=4; partial otherwise. Observed: 21.9% (n=3) and 28.8% (n=4) → **"partial"** by the
  letter of the rule (n=4 is in 25%–50%). The JSR attribution sharpens "partial" toward
  FUNDAMENTAL: ~63% of the still-residual at both n is switched-expansive (degree-independent),
  and degree-8 contributes essentially nothing (0 at n=3, 2 at n=4).

## Honest answer to the question

Does the lifted higher-degree SOS hierarchy RECOVER the dimension-grown residual, or is the
degradation FUNDAMENTAL?

**Mostly FUNDAMENTAL.** Higher-degree SOS recovers a non-trivial but MINORITY slice (≈22–29%), and
that slice is essentially exhausted by degree-6 — degree-8 buys almost nothing. The dominant
remainder (~63% of what's left at each n) is **switched-expansive over the conservative t-box**:
no common Lyapunov of any degree can certify it, because the box itself contains an expanding
switched product. The n-dimension degradation R1 found is therefore NOT an artifact of using too low
an SOS degree — it is a real limit of the box-switched common-Lyapunov / CPU-SOS approach. Closing
it would require a fundamentally different tool (tighter non-box reachability of the Jacobian set, or
exact JSR of the actual reachable set), not a higher rung on the SOS ladder.

## Limits / honest caveats

- `jsr_lb` is a capped one-sided lower bound (product length 4 at n=3, 3 at n=4, each 2^n^len ≈ 4096
  products). It can only PROVE expansion (jsr_lb ≥ 1 ⇒ switched-expansive); a gene labelled
  "finite-gap" might still turn expansive at a longer product length — so the switched-expansive
  count is a LOWER bound and the true fundamental fraction is ≥ what's reported. This makes the
  "fundamental" reading conservative, not inflated.
- The lifted full-space LMI on the symmetric k-th power is a sufficient (slightly conservative) form
  of the exact SOS-on-the-variety condition; a small additional residual could in principle be
  shaved by the exact (variety-restricted) SOS, but that does not change the switched-expansive
  obstruction, which is degree-independent.
- Recovery is over the SAME conservative achievable-t box R1 used; a tighter reachability box (not
  attempted here) is the natural next lever and would target precisely the switched-expansive set.
- Fixed seed; research/ only; src/ untouched; existing verified lift math reused (not reinvented).
- NOTE: the working-tree copy of `dim_completeness_residual_genes.json` was found transiently
  truncated to 79 n=4 entries (by an unrelated concurrent process) AFTER this experiment had already
  loaded the canonical 104-gene set; it was restored from HEAD (911c410). The experiment itself ran
  against the full, committed 104-gene R1 residual (run log: `residual_in=104`).
