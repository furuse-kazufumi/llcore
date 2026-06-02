# PREREGISTRATION — Track A: Achievable-t Lipschitz Refinement

**Date:** 2026-06-02
**Author:** Track A research agent (llcore Z3 verification pillar)
**Status:** Pre-registered BEFORE running any experiment. Gates and thresholds below are fixed.

---

## Context

`src/llcore/verifier/invariants.py` certifies state-direction contraction (Lipschitz `L < 1`)
for an `StateUpdateGene(decay, mix, gate_str)` whose update is

    s' = decay*s + (1-decay)*tanh(mix*x + gate_str*s)

The local state Jacobian is

    J(s,x) = ds'/ds = decay + (1-decay)*gate_str*t,   t = sech^2(pre),  pre = mix*x + gate_str*s

The **existing (free-t)** certifier over-approximates the achievable set of `t` by the full
interval `[0, 1]` and certifies when `sup_{t in [0,1]} |J| < 1`. Its closed-form bound is
`L_free = max(|decay|, |decay + (1-decay)*gate_str|)` (endpoints t=0 and t=1).

This refinement uses the **EXACT achievable** set of `t` over the box `|s|<=1, |x|<=1`.
Since `pre` ranges exactly over `[-M, M]` with `M = |mix| + |gate_str|` (linear in s,x;
corners attain extremes) and `sech^2` is even and decreasing in `|pre|`, the achievable set
of `t` is exactly `[t_min, 1]` with `t_min = sech^2(M) = 1 - tanh^2(M)` (t=1 attained at
pre=0, e.g. s=x=0). Because J is affine (monotone) in t, the EXACT box Lipschitz constant is

    L_achievable = max(|J(t_min)|, |J(1)|)
                 = max(|decay + (1-decay)*gate_str*t_min|, |decay + (1-decay)*gate_str|)

`L_achievable <= L_free` always (achievable interval is a sub-interval of [0,1], so its
endpoint maximum cannot exceed the wider interval's endpoint maximum).

---

## PRE-REGISTERED GATES (verbatim)

- **A1 (containment, soundness of refinement):** free-t certified set is a SUBSET of
  achievable-t certified set. Violations must be **0**.

- **A2 (strict gain, non-triviality):** achievable-t certifies strictly MORE genes than
  free-t (gain count **> 0**). Report the fraction.

- **A3 (soundness vs reality):** for all achievable-admitted genes,
  `empirical_L <= L_achievable` (**0 violations**); and `L_achievable <= L_free`
  for all genes.

- **A4 (honest tightness):** characterize residual over-approximation; report it rather
  than rounding to "exact".

---

## Thresholds (fixed before results)

| Gate | Metric | Threshold |
|------|--------|-----------|
| A1 | `#(free-certified AND NOT achievable-certified)` | `== 0` (MUST) |
| A2 | `#(achievable-certified AND NOT free-certified)` | `> 0` (MUST); report fraction of total |
| A3a | `#(genes with empirical_L > L_achievable + 1e-6)` over admitted genes | `== 0` (MUST) |
| A3b | `#(genes with L_achievable > L_free + 1e-9)` over all genes | `== 0` (MUST) |
| A4 | residual gap `L_achievable - empirical_sup`; count of `empirical_L < 1 but L_achievable >= 1` | report, not pass/fail |

**Tolerance:** `tol = 1e-6` for empirical-vs-analytic comparison (central-difference noise +
float64). `1e-9` for analytic-vs-analytic (`L_achievable <= L_free`).

**Determinism:** all random sampling uses `numpy.random.default_rng(seed)` with `seed=0`.
Z3 certifiers are deterministic (rational arithmetic). `t_min` is computed in float64 then
fed to Z3 as a rational via `z3.RealVal(float)` — this float->rational step is the only
non-exact step in the Z3 certifier and is noted in A_VERDICT.md.

**Sample sizes:** >= 4000 genes total for A1/A2 (deterministic grid + random seed=0).
A3 samples up to ~2000 admitted genes; empirical_lipschitz uses >= 4000 (s,x) samples.
