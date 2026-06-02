# Track B — Verifier-gated Evolution: PRE-REGISTRATION

**Written BEFORE any results were generated.** Frozen spec of hypotheses, gates,
metrics, seeds, and 3-valued honest outcomes. Results live in `B_VERDICT.md`.

This project's core rule (HONEST DISCLOSURE): if a number is unusually good or
the gate looks "free", decompose it before believing it. Negative / null / "the
gate is a no-op here" outcomes are valid and explicitly valued.

---

## Research question

llcore couples two pillars: **Verified** (Z3 invariant gates on the state-update
gene) and **Evolvable** (a minimal tournament GA). The baseline GA
(`src/llcore/evolution/minimal_ga.py::evolve`) accepts every child
**unconditionally** — there is NO verifier gate in the loop.

> **Does inserting the Z3 soundness gate into the GA's child-admission change
> WHAT evolution finds, and at WHAT fitness cost?**

We answer this WITHOUT touching `src/` (semver discipline). All new code is
additive under `research/verified_evolution/`. `gated_evolve.py` re-implements
the GA loop reusing llcore operators (`tournament_select`, `uniform_mutate`,
`crossover_uniform`, `evaluate_population`, `initialize_random_population`) and
is verified to reproduce `evolve()` byte-identically when the gate mode is
`"none"` (so any difference is attributable solely to the gate).

## Gate modes (the independent variable)

- **`none`**   — baseline. Children accepted unconditionally. MUST match
  `src` `evolve()` curves exactly (control for the re-implementation).
- **`state_norm`** — admit a child only if `verify_gene_safe(gene).ok` (Z3 online
  gate certifying `|s|<=1` is preserved for `|x|<=1`). Fail-closed.
- **`contraction`** — admit a child only if
  `verify_lipschitz_contraction(gene).contraction is True` (Z3 certifies
  state-direction Lipschitz `L<1`). Fail-closed.

Rejected children are **resampled** (regenerate from tournament parents) up to a
cap of **50 resamples**; if the cap is hit, fall back to a **known-safe gene**
(decay=0.5, mix=0.0, gate_str=0.0 — trivially contractive: closed-form
`L_upper_bound = max(0.5, 0.5) = 0.5 < 1` and state_norm-safe) and increment a
`fallback_count` event counter. RNG is threaded through every resample so a fixed
seed is fully deterministic.

## Tasks (the regime axis)

Two CONTRASTING regimes built from the official `CopyTask` (so wiring mirrors the
canonical `scripts/poc_0c_minimal_ga.py` pattern exactly: calibrate baseline →
`dataclasses.replace` → `evaluate_gene` closure):

- **`copy_d0`** (easy / short-horizon): `CopyTask(state_dim=8, out_dim=8, delay=0)`
  — recall the most recent input. Solvable by fast-forgetting (low-decay,
  low-recurrence) genes, which sit comfortably inside the contraction region.
- **`copy_d8`** (hard / long-horizon): `CopyTask(state_dim=8, out_dim=8, delay=8)`
  — recall the input from 8 steps back through a 32-step sequence. Rewards
  long-memory genes (high decay, strong recurrence), which push toward and across
  the `L<1` contraction boundary.

Hypothesis on regime (B4): the contraction gate should be a NO-OP where the
fitness optimum is already contractive and LOAD-BEARING / COSTLY where the
optimum lives at or beyond the `L=1` boundary.

## Fixed experimental parameters

- GA: `pop_size=10, n_generations=10, tournament_k=3, mutation_sigma=0.15,`
  `crossover_rate=0.5, elitism=1` (identical to PoC 0c so the `none` mode is a
  faithful baseline). 110 fitness evaluations per run.
- `evaluate_gene` uses `n_trials=5` (noise reduction).
- Readout: `make_fixed_readout(8, 8, seed=1001)`, shared across all cells.
- Baseline MSE calibrated per task with `calibrate_baseline` (default seed).
- **Seeds: N = 20 per (task, gate) cell** (>=15 required). GA seeds are
  `1000..1019`. Test-fitness of the evolved best is re-measured on a held-out
  **test RNG** `default_rng(900000 + ga_seed)` with `n_trials=20` (independent of
  the GA's training RNG stream) to avoid train-on-test optimism.
- resample cap = 50, fallback gene = (0.5, 0.0, 0.0).
- z3 timeouts: state_norm 500 ms, contraction 1000 ms (library defaults).

## Pre-registered gates / outcomes

### B1 — Fitness cost of safety
For each (task, gate in {state_norm, contraction}), over the 20 paired seeds
compute `delta = best_test_fitness(gated) - best_test_fitness(none)` (paired by
seed). Report median delta, paired one-sided **Wilcoxon signed-rank** test, and
effect size (matched-pairs rank-biserial correlation `r`).

3-valued honest outcome per cell:
- **FREE**       — `|median delta|` small AND no significant loss (p>=0.05 on the
  "gated < none" one-sided test).
- **COSTLY**     — significant loss (one-sided p<0.05 that gated < none) with a
  non-trivial negative median delta.
- **BENEFICIAL** — significant gain (one-sided p<0.05 that gated > none).

Threshold for "small": `|median delta| < 0.02` fitness (≈ noise floor at
n_trials=20). Wilcoxon needs >=1 non-zero pair difference; if all deltas are
exactly 0 we report **FREE (degenerate: gate never altered the winner)**.

### B2 — Pathology prevention (is the gate load-bearing?)
In the UNGATED (`none`) final populations across all 20 seeds, measure the rate of
genes that VIOLATE the gated invariant:
- for `contraction`: `empirical_lipschitz(gene) >= 1.0` (sequence-free Jacobian
  estimate over 2000 sampled (s,x) points).
- for `state_norm`: state blow-up or non-finite over a LONG sequence (`L=512`,
  `|x|<=1`): `max|s| > 1 + 1e-6` or any non-finite.

Then confirm the corresponding GATED final populations have **0** such genes.

3-valued: if ungated violation rate ≈ 0 → **gate is a NO-OP on this task** (the
high-fitness region is already safe; HONEST: report it). If ungated rate is high
and gated rate is 0 → **gate is LOAD-BEARING** (it removes a real pathology
class). Mixed → partial.

### B3 — Gate soundness (0 false admits)
Every gene ADMITTED through a gate during evolution must INDEPENDENTLY satisfy the
invariant under empirical check: for `contraction`, `empirical_lipschitz < 1.0`
(must be `<= L_upper_bound`, the closed-form Z3 bound); for `state_norm`, no
blow-up / non-finite over `L=512`. We log every admitted gene and check 0 false
admits. We do NOT require 0 false REJECTS — the gates are deliberately
conservative (sound over-approximation), so false rejects are expected and fine.

### B4 — Regime characterization (honest)
Connect to the project theme ("like selection, the gate may only matter in
specific regimes"): state, per task, whether each gate is load-bearing (B2) vs a
no-op, and whether it is FREE / COSTLY / BENEFICIAL (B1). Produce the regime map:
(task × gate) → {no-op, load-bearing} × {free, costly, beneficial}.

## Pre-committed honest caveats

- Fitness is a **fixed-readout probe** (per `tasks.py` docstring), not a pure gene
  fitness — a known caveat inherited from the upstream task design.
- `verify_gene_safe` is documented to hold for the WHOLE clip box, so we EXPECT
  the state_norm gate to admit everything (no-op). This is pre-registered as the
  likely null result, not a surprise to be explained away.
- The contraction gate uses a conservative free-`t` over-approximation; a gene
  rejected for `L>=1` may still be empirically fine on a given task. Reported
  costs are costs of *conservative* safety, stated as such.
- N=20 paired seeds is modest; Wilcoxon p-values are reported as-is without
  multiple-comparison correction across the 4 (task,gate) B1 cells. We note this.
