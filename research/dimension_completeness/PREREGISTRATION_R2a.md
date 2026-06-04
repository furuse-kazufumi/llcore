# PREREGISTRATION — R2a: does the LIFTED higher-degree SOS hierarchy RECOVER the dimension-grown residual?

Registered BEFORE measuring. Fixed seed. research/ only. src/ untouched.

## Background (R1 finding — the thing R2a interrogates)

R1 (`exp_dim_completeness.py`, `dim_completeness_results.json`) found that the **quadratic**
common-P SDP contraction-completeness DEGRADES with coupled dimension:

| n | pool | quadratic SDP certifies | completeness | contracting-but-uncertified residual |
|---|------|-------------------------|--------------|---------------------------------------|
| 2 | 200  | 184                     | 92.0%        | 16  (8.0%)   |
| 3 | 200  | 159                     | 79.5%        | 41  (20.5%)  |
| 4 | 200  | 96                      | 48.0%        | 104 (52.0%)  |

The residual GROWS with dimension (16 → 41 → 104). R1's 41 (n=3) and 104 (n=4)
**residual genes** — empirically contracting (ρ<1 at high samples) but NOT certifiable by the
quadratic SDP — are saved in `dim_completeness_residual_genes.json` and are the **test set** here.

## Question

Does the LIFTED higher-degree SOS Lyapunov hierarchy
(degree-4 = symmetric-2nd power, degree-6 = symmetric-3rd power, degree-8 = symmetric-4th power)
RECOVER (certify) that grown residual at n=3 and n=4 — or is the n-dimension degradation
**FUNDAMENTAL** (the residual is genuinely beyond the CPU SOS hierarchy: switched-expansive,
or needing exact-JSR which is NP-hard)?

## Method (locked)

- Reuse the ALREADY-VERIFIED lift math (do NOT reinvent): `verifier_deg6.sym_power(A, degree)`
  (degree=2 ≡ deg4, 3 ≡ deg6, 4 ≡ deg8) and `verifier_deg6.certify_degN(vertices, degree)`.
- For each R1-residual gene at n∈{3,4}: build its 2^n t-box vertex Jacobians exactly as
  `coupled_nd` does (`_jac_at_t` over `_box_vertices(t_min_per_coord(g))`), then attempt
  `certify_degN` at degree=2 (deg4), degree=3 (deg6), degree=4 (deg8).
- CLARABEL is PINNED for every cvxpy SDP solve (inherited from `verifier_deg6._SOLVER`); the run
  ABORTS if CLARABEL is not the confirmed solver. The certifier does an INDEPENDENT eigenvalue
  re-check of P and of every decrease-LMI (never trusts the solver status), and pre-screens
  ρ(J_v)<1 at every vertex.
- Recovery is CUMULATIVE: deg4_rec ⊆ (deg4∪deg6)_rec ⊆ (deg4∪deg6∪deg8)_rec. A gene counts as
  recovered at the lowest degree that certifies it.
- SOUNDNESS oracle for EVERY recovered gene, INDEPENDENT of the solver:
  (a) empirical spectral radius at high samples (ρ<1, `empirical_rho_fast` @ ≥20k samples), AND
  (b) an n-dim JSR product lower bound over the 2^n vertex Jacobians (`jsr_lb` with the same
  per-n product-length cap R1 used: n=3 len 4 → 8^4=4096, n=4 len 3 → 16^3=4096).
  Any recovered gene with ρ≥1 OR jsr_lb≥1 is UNSOUND.
- Lift-math sanity: BEFORE trusting any recovery, verify `sym_power(A, k)` against a brute-force
  monomial transform `m(Az) = sym_power(A,k) m(z)` on several random n=3 and n=4 matrices for
  k∈{2,3,4}; require max error < 1e-8.

## Falsifiable gates (registered before measuring)

- **RG-recover**: at each n∈{3,4}, record the CUMULATIVE recovered count after deg4, then +deg6,
  then +deg8, as a fraction of that n's R1 residual (41 at n=3, 104 at n=4).
- **RG-sound**: ZERO unsound recoveries across all n and all degrees (every admit has ρ<1 AND
  jsr_lb<1). This is a HARD gate — any unsound admit invalidates the run (must be 0).
- **RG-verdict** (decision rule, thresholds fixed now):
  - **"recoverable"** iff the cumulative lifted SOS (through deg8) closes a **MAJORITY (> 50%)**
    of the grown residual at BOTH n=3 and n=4, with 0 unsound.
  - **"fundamental"** iff the cumulative lifted SOS closes only a **SMALL fraction (< 25%)** of the
    grown residual at n=4 (the dimension where degradation is worst), with 0 unsound — i.e. the
    higher-degree hierarchy does NOT rescue the dimension-grown residual.
  - **"partial"** for anything between (25%–50% at n=4, or recoverable at n=3 but not n=4) —
    reported honestly with the exact per-n/per-degree numbers.
- Honest-disclosure commitment: if higher-degree SOS recovers LITTLE, report
  "the n-dimension degradation is fundamental / beyond the CPU SOS hierarchy" plainly. Do NOT
  inflate recovery. A small recovery fraction is a VALID, valuable finding.

## Tractability note

Lift dimensions m = C(n+degree-1, degree):
- n=3: deg4 → 6, deg6 → 10, deg8 → 15.
- n=4: deg4 → 10, deg6 → 20, deg8 → 35.
All are tractable LMIs for CLARABEL at the residual sizes (41 / 104 genes). deg8 at n=4 (35×35 P,
16 vertices) is the heaviest; if it is intractable in the time budget it is reported as such and
excluded honestly (the verdict then rests on deg4+deg6 plus an explicit deg8-skip note).
