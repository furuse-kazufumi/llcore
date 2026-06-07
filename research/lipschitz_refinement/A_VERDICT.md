# A_VERDICT — Track A: Achievable-t Lipschitz Refinement

**Date:** 2026-06-02 | **Repo:** <llcore-root> | **Python:** py -3.11 | **z3-solver:** 4.16.0
**Seed:** 0 (all sampling) | **Genes:** 6096 total (4096-pt grid 16^3 + 2000 random, full box)
**All scripts ran to completion.** Results JSON: `exp_a1_a2_results.json`, `exp_a3_results.json`, `exp_a4_results.json`.

---

## Headline

The achievable-t refinement is **sound and produces a strictly tighter bound VALUE**
(L_achievable <= L_free, strictly smaller for 31.8% of genes, mean delta 0.052, max 0.244),
but it does **NOT enlarge the certified set** — the binary "L<1 certified?" decision is
**byte-identical** to the existing free-t certifier (gain = 0). **Gate A2 (strict
certification gain > 0) FAILS.** A1, A3a, A3b pass. A4 reported. This is reported honestly
rather than spun; the structural reason is given below.

---

## Measured gate results

| Gate | Metric | Threshold | Observed | Verdict |
|------|--------|-----------|----------|---------|
| **A1** | `#(free-cert AND NOT ach-cert)` | `== 0` | **0** | **PASS** |
| **A2** | `#(ach-cert AND NOT free-cert)` | `> 0` | **0** (fraction 0.0) | **FAIL** |
| **A3a** | `#(empirical_L > L_achievable + 1e-6)` over 2000 admitted | `== 0` | **0** (worst excess 5.79e-11) | **PASS** |
| **A3b** | `#(L_achievable > L_free + 1e-9)` over all 6096 | `== 0` | **0** | **PASS** |
| **A4** | residual gap + over-rejection characterized | report | reported (below) | **N/A (reported)** |

Free-certified = 4022, Achievable-certified = 4022. **certified_sets_identical = true.**

---

## Why A2 is structurally 0 (honest root cause — not a bug)

Both certifiers search `t` for `|J| >= 1`, where `J(t) = decay + (1-decay)*gate_str*t` is
affine in t, so `sup|J|` is attained at an **endpoint**.

- Free-t endpoints: `t in {0, 1}` -> `L_free = max(|J(0)|, |J(1)|) = max(|decay|, |decay+(1-decay)*gate_str|)`.
- Achievable-t endpoints: `t in {t_min, 1}` -> `L_achievable = max(|J(t_min)|, |J(1)|)`.

**Both share the `J(1)` endpoint.** The ONLY endpoint the refinement removes is `t=0`, whose
value is `|J(0)| = |decay|`. But `decay` is clipped to `[0,1]`, so `|J(0)| <= 1` ALWAYS. The
`t=0` endpoint can therefore **never** be the term that pushes the free bound to `>= 1` — a
free-t rejection is always caused by `|J(1)| >= 1`, which the achievable bound shares verbatim.
Hence: free rejects `<=>` `|J(1)| >= 1` `<=>` achievable rejects. The certified sets coincide
exactly, and no gene can flip from reject to certify. This was confirmed two ways:
(1) closed-form scan over 96000 grid points -> gain 0, containment-violation 0;
(2) Z3-level scan -> gain 0, and Z3 verdict matched the closed-form `L<1` predicate with
**0 mismatches** on the boundary-focused grid.

The task's A2 hypothesis ("achievable-t certifies strictly more genes") is **falsified for
this specific diagonal RWKV map with decay in [0,1]**. The premise that the free bound is
"conservative" is true for the bound VALUE but does not translate into extra certifications,
because the conservatism lives entirely at the t=0 / |decay|<=1 endpoint that is already
dominated.

---

## Where the refinement DOES help (the genuine, measurable win)

The bound **value** is strictly tighter for **1939 / 6096 = 31.8%** of genes
(A3b guarantees it never increases). Among those:
- mean `L_free - L_achievable` = **0.0518**
- max `L_free - L_achievable` = **0.2437**

Against empirical reality (A4): the gap to the from-below empirical sup shrinks from
`L_free - emp` (mean 0.0170, max 0.248) to `L_achievable - emp` (mean 5.04e-4, median 3.05e-8,
max 9.66e-3) — a **mean gap reduction of 0.0165**. So `L_achievable` is a near-exact box
constant (median gap ~3e-8, i.e. matched to float noise), whereas `L_free` is meaningfully
loose. **This matters for any downstream use that consumes the bound value** (e.g. ranking
genes by contraction margin, fitness shaping, or proving a tighter convergence rate), even
though it does not change the pass/fail gate.

---

## A3 soundness (vs reality)

For all 2000 achievable-admitted genes, `empirical_L <= L_achievable + 1e-6` held with
**0 violations**; the worst single excess was **5.79e-11** (central-difference + float64
noise, far below tol). `L_achievable <= L_free` held for all 6096 genes (0 violations).
The empirical estimator used 5000 (s,x) samples (>= 4000 required) via both llcore's
`empirical_lipschitz` and an independent central-difference stream (max of the two taken).

## A4 honest tightness

- Residual gap `L_achievable - empirical_sup`: mean 5.04e-4, median 3.05e-8, max 9.66e-3,
  min -7.78e-11. **negative_gap_count (soundness alarm) = 0** — empirical never exceeds the
  analytic box constant beyond float noise.
- **Over-rejection** (empirical_L < 1 but L_achievable >= 1): **16 genes**, ALL at the
  `gate_str = -2.0` boundary with `decay = 1/3`, where `empirical_L ≈ 0.99999...` (within
  ~1e-7 to 1e-12 of 1) and `L_achievable = 1.0000000000000002`. These are **true box-sup-≈-1**
  cases: the exact box sup is genuinely 1 (attained as `s,x -> ±1` corners), and the finite
  empirical sample landed just under it. This is NOT a refinement defect — and the FREE
  certifier over-rejects the **same 16** genes (over_reject_reduction = 0), reconfirming the
  binary decisions coincide.

---

## Honest reservations / caveats

1. **A2 falsified.** The headline pre-registered "non-triviality" gain is 0. The refinement's
   value is a tighter bound *value*, not a larger certified set, for this map. I did not
   round this away.
2. **Exactness scope.** `L_achievable` is the EXACT box Lipschitz constant only w.r.t.
   (a) the `[-1,1]` input box `|s|<=1, |x|<=1`, and (b) the per-coordinate **diagonal** map
   (each state coordinate updates independently; no cross-coordinate coupling). For a
   non-diagonal / vector map the analysis would need the full Jacobian operator norm.
2b. It is the *local* state-direction Lipschitz constant `sup|ds'/ds|`; it does not by itself
    bound the input-direction sensitivity.
3. **float -> rational step.** `t_min = sech^2(|mix|+|gate_str|)` is evaluated in float64,
   then injected into Z3 as `z3.RealVal(t_min_float)` (an exact rational of that float). Z3
   then reasons exactly over that rational. So the *only* non-exact step in the achievable Z3
   certifier is rounding the true `sech^2(M)` to the nearest float64 before rationalizing.
   For the clipped box `M = |mix|+|gate_str| <= 3`, this rounding is ~1e-16 and the closed-form
   and Z3 verdicts agreed with 0 mismatches in testing. (Direction of this rounding could in
   principle relax `t_min` by ~1 ulp; it had no observed effect on any verdict here.)
4. **empirical_sup is from-below.** It under-estimates the true box sup, so a small positive
   `L_achievable - empirical_sup` is expected and is not unsoundness; only a *negative* gap
   would alarm, and there were none.
5. **Determinism.** Every rng is `numpy.random.default_rng(0)`. Z3 is deterministic (rational
   arithmetic). Re-running reproduces identical JSON.

---

## Bottom line

The math is correct and implemented soundly; the certifier is a faithful achievable-t
refinement that yields a provably tighter (and near-empirically-exact) bound value. But for
the RWKV diagonal map with `decay in [0,1]`, **tightening the bound does not certify any new
gene**, because the loose endpoint the refinement removes (`t=0`, value `|decay| <= 1`) is
already dominated. A2 is a negative result and is reported as such per the project's HONEST
DISCLOSURE rule.
