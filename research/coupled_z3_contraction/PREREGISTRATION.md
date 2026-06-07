# Track C — Coupled-map Z3 inf-norm contraction certification: PRE-REGISTRATION

**Written BEFORE the full grid/random experiment was run** (smoke tests of individual
modules were run first to confirm wiring; the pre-registered gates C1/C2/C3 below were frozen
before `exp_c_runner.py` produced `exp_c_results.json`). Results live in `C_VERDICT.md`.

This project's core rule (HONEST DISCLOSURE): if a number is unusually good, or Z3 looks like
it "earns its keep", decompose it before believing it. A conservative outcome -- including
"Z3 still does not clearly beat a cheaper closed-form method" -- is a fully acceptable honest
finding. Negative / null / N-A results are valid.

All new code is ADDITIVE under `research/coupled_z3_contraction/`. `src/` is NOT touched. No git.
Determinism: every rng is `numpy.random.default_rng(seed)`; seeds reported in the verdict.

---

## Background this builds on (Tracks A & B)

On the DIAGONAL scalar RWKV map `s' = decay*s + (1-decay)*tanh(mix*x + gate_str*s)`, two prior
tracks established that the Z3 contraction certifier is mathematically IDENTICAL to a closed-form
scalar inequality `max(|decay|, |decay+(1-decay)*gate_str|) < 1` -- Z3 added ZERO discrimination
("decorative"; A_VERDICT.md, B_VERDICT.md). The diagonal Jacobian has a single entry per
coordinate, so the inf-norm collapses to a per-coordinate scalar test.

**Open question (Track C):** does Z3 (or more precisely, a *coupling-aware* contraction check)
earn its keep on a COUPLED (non-diagonal) map, where the simple per-coordinate scalar inequality
is no longer SOUND?

## The map (n=2 coupled)

    s' = decay (.) s + (1-decay) (.) tanh(W @ s + V @ x),   s,x in R^2, (.) = elementwise
    gene = (decay in [0,1]^2, W in [-2,2]^{2x2} off-diagonal allowed),  V = identity (fixed).

State Jacobian: `J = diag(decay) + diag((1-decay) (.) t) @ W`, `t_i = sech^2(pre_i) in (0,1]`,
`pre = W@s + V@x`. Row i: `J_ii = decay_i + (1-decay_i)*t_i*W_ii`, `J_ij = (1-decay_i)*t_i*W_ij`.

## The three checks compared

1. **Z3 inf-norm certifier** (`z3_infnorm_certifier.py`): SUFFICIENT, SMT-encodable. Searches
   `t` in `[0,1]^2` (`free01`) or `[t_min,1]^2` (`tmin1`) for `OR_i (row abs-sum_i >= 1)`.
   `unsat` => `||J||_inf < 1` certified over the box => inf-norm contraction => unique fixed
   point + bounded state (Banach). `sat`/`unknown` => conservative / fail-closed reject.
   SOUND because free `t in [0,1]^2` strictly over-approximates the achievable `t` set.
2. **Scalar diagonal heuristic** (`scalar_heuristic.py`): the PRIOR diagonal-only check.
   Admits iff `max_i sup_{t_i} |J_ii| < 1`, IGNORING off-diagonal `W_ij`. UNSOUND for coupled
   maps by construction.
3. **Empirical oracle** (`coupled_map.empirical_box_norms`, `iterate_state_growth`): dense
   (s,x) box sample (20000 random + 16 sign-corners + zero point) of the EXACT Jacobian; reports
   from-below sups of `||J||_inf`, `||J||_2`, and spectral radius `rho(J)`, plus trajectory
   separation growth. Ground truth for soundness and expansiveness.

## Gene sampling (fixed, pre-registered)

- **Grid:** `decay in {0.2, 0.5, 0.8}^2` (9) x `W` built from `{W_diag in {-0.5,0.5,0.9}}` and
  `{W_off in {-1.5,-0.9,0.0,0.9,1.5}}` applied symmetrically and asymmetrically. Exact grid
  construction is in `exp_c_runner.py::build_grid` and is FROZEN here by reference.
- **Random:** 3000 genes, `numpy.default_rng(0)`: `decay ~ U[0,1]^2`, `W ~ U[-2,2]^{2x2}`.
- All genes `.clipped()` to the legal box before any check.
- `max_input_abs = 1.0`. Empirical sample seed = 0, n_samples = 20000 (>= dense), plus corners.

## PRE-REGISTERED GATES (verbatim)

### C1 — Z3 soundness (0 false admits required)
Every gene the Z3 inf-norm certifier ADMITS (`certified=True`) must be empirically
non-expansive: its empirical `||J||_inf` over the dense box sample must be `<= certified bound`
(`<= 1.0`, allowing float tol `1e-9`). Cross-check `||J||_2` too (`||J||_2 <= ||J||_inf` for
2x2? -- NO in general; we report `||J||_2` for context but the gate is on the inf-norm, which is
what Z3 certifies). **PASS iff 0 false admits** (no admitted gene has `emp_infnorm > 1 + 1e-9`).
Run BOTH `free01` and `tmin1` domains.

3-valued: PASS (0 false admits) / FAIL (>=1 false admit -- would indicate an UNSOUND encoding).

### C2 — Z3 adds value over scalar (is it decorative?)
Count genes where the SCALAR diagonal heuristic ADMITS (`scalar admit=True`, says contraction)
but the coupled map is actually EXPANSIVE (empirical `||J||_inf > 1 + 1e-9`). Call this count
`N_scalar_false_admit_expansive`.
- If `N_scalar_false_admit_expansive > 0`: the scalar heuristic is UNSOUND for coupled maps, and
  the Z3 (coupling-aware) certifier correctly REJECTS those same genes => **Z3 is NOT decorative
  here** (it adds discrimination the scalar test cannot). Report the count + fraction, and CONFIRM
  each such gene is truly expansive empirically (`emp_infnorm > 1`), and CONFIRM Z3 rejects them.
- If `N_scalar_false_admit_expansive == 0`: scalar and Z3 agree on this gene family => Z3 stays
  decorative even for the coupled map (honest null -- acceptable outcome).

HONEST sub-question pre-committed: distinguish "Z3 adds value over the SCALAR heuristic" from
"Z3 adds value over a full CLOSED-FORM inf-norm computation". Because each row abs-sum is
piecewise-linear in `t_i`, a closed-form endpoint enumeration (`coupled_map.infnorm_over_box_freeT`)
ALSO computes `||J||_inf` exactly. We will report the agreement rate between Z3's verdict and the
closed-form `||J||_inf < 1` predicate. If they agree exactly, the honest framing is: **the
COUPLING (not Z3 specifically) is what beats the scalar heuristic; Z3 and closed-form are
co-equal for this sufficient condition.** This is the same "decorative-but-now-over-the-right-
baseline" nuance Track A surfaced, stated up front.

### C3 — Honest limits (inf-norm is sufficient, not necessary)
`||J||_inf < 1` is SUFFICIENT for contraction, not NECESSARY. Report genes that are truly locally
contractive (empirical spectral radius `rho(J) < 1` over the whole box, AND not state-growing /
non-finite) but the inf-norm certifier REJECTS (`certified=False`). Call this
`N_conservative_false_reject`. Report count, fraction, and the distribution of
`(emp_infnorm, emp_2norm, emp_spectral_radius)` for those genes to characterize HOW conservative
the inf-norm gate is (typical gap `emp_infnorm - emp_spectral_radius`).
Also state honestly: the EXACT contraction condition (spectral radius / 2-norm < 1 over the box)
is NOT naturally SMT-encodable -- it is an eigenvalue / SDP / Lyapunov-LMI problem, outside
linear-real-arithmetic Z3 can decide efficiently. So Z3's inf-norm is a SOUND-but-CONSERVATIVE
sufficient gate, and the conservativeness is exactly the `rho < 1 <= inf-norm` band measured here.

Report (N-A allowed): C3 has no pass/fail threshold; it is a characterization gate.

## Determinism & honesty commitments

- All sampling rng = `numpy.random.default_rng(0)`; Z3 is deterministic (exact rational
  arithmetic over `RealVal` constants). Re-running `exp_c_runner.py` reproduces identical JSON.
- Only numbers OBSERVED by RUNNING the scripts are reported. `t_min` uses float64 `sech^2(M)`
  injected as `z3.RealVal` (exact rational of that float) -- the only non-exact step, ~1e-16,
  noted as in Track A.
- Empirical sups are from-BELOW; a small positive `(certified_bound - empirical)` gap is expected
  and is not unsoundness. Only `empirical > certified bound` (C1) or `rho >= 1` claimed-contractive
  (C3) would be alarms.
- If Z3 turns out co-equal with the closed-form inf-norm (likely), we will NOT spin it as
  "Z3 load-bearing"; we will report it as "coupling-aware contraction beats the diagonal scalar
  heuristic; Z3 and closed-form endpoint enumeration are equivalent realizations of that check".
