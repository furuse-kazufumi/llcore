# Track D — Tighter matrix contraction certificates (2-norm vertex / SDP-Lyapunov): PRE-REGISTRATION

**Written BEFORE `exp_d_runner.py` was run** (the convexity/vertex soundness fact was numerically
spot-checked first — `sigma_max(J(t))` attained at a t-box vertex over 2000 random genes × 400
interior points, max interior-minus-vertex excess = 0.0 — but the pre-registered gates D1/D2/D3/D4
below were frozen before the full 3270-gene sweep produced `exp_d_results.json`). Results live in
`D_VERDICT.md`.

This project's core rule (**HONEST DISCLOSURE**): if a tighter certificate or a solver looks like it
"earns its keep", decompose it before believing it. A **negative outcome — including "even the SDP
solver adds little over closed-form-ish 2-norm vertex enumeration" — is a fully valid and acceptable
honest finding.** Negative / null / N-A results are valid. We report ONLY numbers observed by running
the scripts.

All new code is ADDITIVE under `research/spectral_lyapunov_contraction/`. `src/` is NOT touched. No git.
Determinism: every rng is `numpy.random.default_rng(seed)`; seeds reported below and in the verdict.

---

## What this builds on (Track C)

Track C (`research/coupled_z3_contraction/`) studied the n=2 COUPLED RWKV-style state map

    s' = decay (.) s + (1 - decay) (.) tanh(W @ s + V @ x),   s, x in R^2, (.) = elementwise
    gene = (decay in [0,1]^2, W in [-2,2]^{2x2} off-diagonal allowed),  V = identity (fixed).

Exact state Jacobian:

    J(t) = diag(decay) + diag((1 - decay) (.) t) @ W,   t_i = sech^2(pre_i) in (t_min_i, 1],  pre = W@s + V@x
    Row i: J_ii = decay_i + (1-decay_i)*t_i*W_ii,   J_ij = (1-decay_i)*t_i*W_ij  (j != i)

J is **AFFINE in t** over the box `[t_min, 1]^2` (each entry is constant + linear-in-t_i).

Track C's CAPSTONE finding (from the independent red-team oracle `redteam_results.json`, seed 777,
n_emp=6000 + structured corners; 60k heavy-confirm seed 999):

- **C1 (soundness):** the induced ∞-norm certifier `||J||_inf < 1` admitted **513** genes, **0 false
  admits**, worst admitted empirical ∞-norm = 0.99986.
- **C2 (coupling value):** the diagonal scalar heuristic admitted **1267 (red-team count, also reported
  as 1796 scalar-admit total)** genes that are empirically EXPANSIVE; the coupling-aware ∞-norm rejects
  all of them. Coupling-awareness is load-bearing.
- **C3 (conservativeness):** `||J||_inf < 1` is SUFFICIENT, not NECESSARY. It **over-rejects 850 genes**
  that are TRULY contractive in the spectral sense (empirical spectral-radius sup `rho(J) < 1` over the
  box), with **median(∞-norm − rho) gap = 0.477**, max 1.998. Of those 850, **0 have empirical
  ∞-norm < 1** — i.e. the ∞-norm certifier is internally correct (it really does see ∞-norm ≥ 1 there);
  it is merely loose *relative to the spectral radius*.
- Population: 3270 genes, **rho_lt1 = 1363** (genes with empirical spectral-radius sup < 1).
- **R3:** Z3 vs closed-form endpoint ∞-norm disagreement = 0/3270 ⇒ Z3 is DECORATIVE for ∞-norm; the
  ∞-norm box-sup is a closed-form per-row 1-D convex max, no solver needed.

**Open question (Track D):** Does a TIGHTER matrix contraction certificate certify those **850**
`rho<1` genes the ∞-norm rejected — and does a SOLVER (SDP/Lyapunov) beat a closed-form-ish certificate
(2-norm vertex enumeration)?

## Population (IDENTICAL to Track C — reused, not regenerated)

- **Grid (270 genes):** `decay in {0.2,0.5,0.8}^2` (9) × W from `W_diag in {-0.5,0.5,0.9}` (3) and
  `W_off in {-1.5,-0.9,0.0,0.9,1.5}` (5), each applied symmetrically `[[wd,wo],[wo,wd]]` and
  asymmetrically `[[wd,wo],[-wo,wd]]` (×2). 9×3×5×2 = 270.
- **Random (3000 genes):** `numpy.default_rng(0)`: `decay ~ U[0,1]^2`, `W ~ U[-2,2]^{2x2}`.
- Total **3270** genes, all `.clipped()` to the legal box. `max_input_abs = 1.0`.
- Built by `research/coupled_z3_contraction/redteam_fast.py::build_population(3000, 0)` — Track D
  re-imports that exact builder so the population (and thus the index of every gene) matches bit-for-bit.

## The three certifiers compared (Track D)

1. **∞-norm certifier (baseline, from Track C):** closed-form per-row endpoint enumeration
   `coupled_z3_contraction/coupled_map.infnorm_over_box_freeT`. Admits iff `||J||_inf < 1` over the
   t-box. SOUND, conservative. Two domains: `free01` (t in [0,1]^2) and `tmin1` (t in [t_min,1]^2).
2. **2-norm vertex certifier (NEW, `two_norm_vertex_certifier.py`):** `sigma_max(J)` (largest singular
   value, = induced 2-norm) is a CONVEX function of J; J is AFFINE in t over the box; therefore
   `sup_{t in box} sigma_max(J(t))` is attained at a BOX VERTEX. The certifier evaluates `sigma_max`
   via numpy SVD at the **4 vertices** of `[t_lo,1]^2` and admits iff the max `< 1`. **NO solver needed
   — closed-form-ish (SVD + vertex enumeration).** SOUND: `||J||_2 < 1` over the box ⟹ contraction in
   2-norm ⟹ `rho(J) < 1` and non-expansive. Two domains: `free01` (t_lo=0) and `tmin1` (t_lo=t_min).
3. **SDP-Lyapunov certifier (NEW, `lyapunov_sdp_certifier.py`):** the tightest quadratic certificate.
   Seek a common Lyapunov `P >> 0` with `J_v' P J_v - P << 0` at all 4 t-box VERTICES `J_v` (quadratic
   stability LMIs; since J is affine in t, vertex-LMIs ⟹ the LMI holds for the whole box by convexity).
   This is an SDP/LMI = genuine SOLVER territory, needs **cvxpy**. If cvxpy import fails, the module
   MUST degrade gracefully and report `available=False` (NO crash, NO fabricated result).

The **empirical oracle** is `coupled_z3_contraction/redteam_fast.emp_infnorm_fast` (∞-norm sup),
`emp_rho_min_fast` (spectral-radius sup), plus a fresh DENSE `||J||_2` sup computed in Track D over the
same structured + random (s,x) box sample (seed reported). These are from-BELOW sups; ground truth for
soundness.

## PRE-REGISTERED GATES (verbatim)

### D1 — Soundness (0 false admits required)
Every gene the **2-norm-vertex** certifier (and the **SDP-Lyapunov** certifier, if cvxpy runs) ADMITS
must be empirically NON-EXPANSIVE on a dense (s,x) box sample: empirical `||J||_2 <= 1 + tol` AND
empirical spectral-radius sup `rho(J) < 1 + tol` (tol = 1e-6 to absorb from-below SVD float noise; the
certifier itself uses strict `< 1`). **PASS iff 0 false admits** (no admitted gene has empirical
`||J||_2 > 1 + tol` or `rho > 1 + tol`). Run for both `free01` and `tmin1` domains.
3-valued: PASS (0 false admits) / FAIL (>=1 false admit — would indicate an UNSOUND certificate).

### D2 — Tightness gain of 2-norm-vertex over ∞-norm
Count genes the **2-norm-vertex** certifier ADMITS that the **∞-norm** certifier REJECTS, restricted to
genes with empirical `rho < 1` (i.e. a subset of Track C's 850 rho<1 over-rejects). Report the count
and the fraction of the 850. (We also report the raw count of all 2-norm-admit-but-∞-norm-reject genes
for completeness, but the headline D2 number is the rho<1 subset, matching the Track C framing.)
Report for both domains. No hard threshold — this characterizes the gain. PASS = "ran and reported".

### D3 — Does the SOLVER (SDP) earn its keep?
If cvxpy installs and runs: count genes the **SDP-Lyapunov** certifier ADMITS that the
**2-norm-vertex** certifier REJECTS.
- If this count **> 0** ⇒ the solver (SDP) genuinely beats the closed-form-ish 2-norm vertex method
  here. Report the count, the fraction of the 850, and confirm those genes pass D1 soundness.
- If this count **~0** ⇒ honest negative: **"even the solver adds little over 2-norm vertex
  enumeration"** (a deeper negative, consistent with Track A/B/C's "solver is decorative for this
  invariant class"). Report it as such.
- If cvxpy **cannot be installed / imported**: report that honestly, mark SDP as `not_run`, and present
  the 2-norm-vertex result as the partial answer with SDP flagged as an optional follow-up. The
  **2-norm-vertex result must stand regardless of cvxpy availability.**
3-valued: PASS (ran and reported, either sign) / N-A (cvxpy unavailable, SDP not run).

### D4 — Honest residual
Among the 1363 genes with empirical `rho < 1`, count those that NO certifier (∞-norm OR 2-norm-vertex
OR SDP, whichever ran) admits. These need a strictly stronger tool (joint-spectral-radius, non-quadratic
/ parameter-dependent Lyapunov functions, or a time-varying / common-but-non-quadratic certificate),
because a single common quadratic Lyapunov function / a single induced norm cannot certify them.
Report the count, the fraction of the rho<1 set, and characterize WHY (e.g. the achievable-t variation
forces switching-system behavior with no common quadratic P).
No hard threshold — characterization gate. PASS = "ran and reported".

## Determinism & honesty commitments

- Population builder reused verbatim from `redteam_fast.build_population(3000, 0)` ⇒ identical genes /
  indices to Track C.
- All empirical sampling rng = `numpy.random.default_rng(SEED)` with SEED reported in the verdict and
  JSON (empirical ∞-norm / rho reuse Track C's seed 777, n_emp=6000 + structured corners; the new
  empirical `||J||_2` sup uses the same sample for apples-to-apples comparison).
- numpy SVD / eigvals are deterministic; cvxpy SDP uses its default solver (reported), tolerance noted.
- 2-norm-vertex soundness rests on: `sigma_max` convex in J + J affine in t ⇒ box-sup at a vertex.
  This was numerically spot-checked (max interior-minus-vertex excess = 0.0 over 2000×400 samples)
  BEFORE freezing this pre-registration, and is RE-verified inside `exp_d_runner.py` (a soundness
  self-check on a sample of genes) as part of the run.
- Empirical sups are from-BELOW; a small positive `(certified_bound − empirical)` gap is EXPECTED and is
  NOT unsoundness. Only `empirical > certified bound` (D1) would be an alarm.
- If the SDP turns out co-equal with the 2-norm vertex method (a plausible outcome — quadratic stability
  of a 2x2 affine-in-t family may be governed by the same singular-value envelope), we will NOT spin it
  as "the solver is load-bearing"; we will report it as "even the SDP solver adds little over closed-form
  2-norm vertex enumeration for this n=2 family", continuing the Track A/B/C honest-disclosure arc.
