# Track B — Verifier-gated Evolution: VERDICT

All numbers below were OBSERVED from `exp_b_results.json` (produced by
`py -3.11 research/verified_evolution/exp_b_runner.py`). Nothing is hand-typed
from intuition. Pre-registration: `PREREGISTRATION.md` (written first).

**Run:** 2 tasks × 3 gates × 20 seeds (GA seeds 1000..1019), 110 fitness
evals/run, `n_trials`=5 train / 20 held-out test. Wall time **134.9 s**, CPU.

**Control (re-implementation is faithful):** `gated_evolve(gate_mode="none")`
produces curves **byte-identical** to `src/llcore/evolution/evolve()`
(`control_none_matches_src = True`, verified across 3 seeds in
`assert_none_matches_src`). So every difference below is attributable solely to
the gate, not to a re-implemented loop.

---

## One-paragraph answer to the research question

Inserting the Z3 soundness gate into the GA's child-admission **does** change what
evolution finds — **but only the contraction (`L<1`) gate, and only in the
short-horizon regime**. The state_norm (`|s|<=1`) gate is a **complete no-op** on
both tasks: it admits every clipped gene (0 rejections, 0 pathologies in the
ungated runs), exactly as its source docstring predicts. The contraction gate is
**load-bearing** on both tasks (the ungated GA drifts 15.5%–23.0% of its final
population into empirically non-contractive `L>=1` genes) and **fully prevents**
that pathology in admitted children (0/1800 false admits). Its fitness cost is
**regime-dependent**: **COSTLY** on the easy `copy_d0` task (median test-fitness
delta −0.0056, one-sided Wilcoxon p=0.032), but **FREE** on the harder `copy_d8`
task (median delta −0.0004, p=0.25, not significant). The fitness optimum on the
easy task sits *just past* the `L=1` boundary, so forbidding it costs a little;
on the hard task the best contractive genes are competitive with the best
non-contractive ones, so safety is free.

---

## B1 — Fitness cost of safety (paired, N=20 seeds)

`delta = test_fitness(gated) − test_fitness(none)`, paired by seed. One-sided
Wilcoxon signed-rank H1: gated < none. Effect size = matched-pairs rank-biserial.

| cell | mean none | mean gated | median Δ | Wilcoxon p (gated<none) | rank-biserial | n≠0 pairs | **verdict** |
|---|---|---|---|---|---|---|---|
| copy_d0 / state_norm | 0.2114 | 0.2114 | +0.0000 | 1.000 | 0.000 | 0 | **FREE** (degenerate — gate never changed the winner) |
| copy_d0 / contraction | 0.2114 | 0.2001 | **−0.0056** | **0.032** | −0.497 | 18 | **COSTLY** |
| copy_d8 / state_norm | 0.1519 | 0.1519 | +0.0000 | 1.000 | 0.000 | 0 | **FREE** (degenerate) |
| copy_d8 / contraction | 0.1519 | 0.1472 | −0.0004 | 0.251 | −0.191 | 16 | **FREE** (no significant loss) |

- **state_norm = FREE everywhere, degenerate.** 0 rejections in 20×110 children →
  identical RNG stream → identical result to baseline. The gate literally never
  fired. This is the pre-registered null result, not a surprise.
- **contraction = COSTLY on copy_d0** (significant, moderate effect rb≈−0.5). The
  cost is *small in absolute terms* (≈0.6 fitness pp on a 0.21 mean) but it is a
  genuine, statistically significant loss. Honest read: forbidding the
  best-but-non-contractive gene costs a little on the easy task.
- **contraction = FREE on copy_d8** (not significant, p=0.25). Honest: the
  best-contractive gene is competitive with the best-unrestricted gene here.

**Honest caveat (pre-registered):** N=20 is modest; the 4 B1 cells are reported
without multiple-comparison correction. The COSTLY verdict (p=0.032) would not
survive a strict Bonferroni across 4 tests (α/4 = 0.0125) — so it is "significant
at α=0.05, marginal under correction." We report it as a small genuine cost, not
a dramatic one. Fitness is a fixed-readout probe (upstream task-design caveat).

## B2 — Pathology prevention (is the gate load-bearing?)

Violation = independent empirical check on each final-population gene:
contraction → `empirical_lipschitz(gene) >= 1.0` (2000 sampled points); state_norm
→ `|s|>1+1e-6` or non-finite over an `L=512`, `|x|<=1` sequence.

| cell | ungated violation rate | gated violation rate |
|---|---|---|
| copy_d0 / state_norm | **0.000** (0/2000) | 0.000 |
| copy_d0 / contraction | **0.155** (310/2000) | **0.000** (0/2000) |
| copy_d8 / state_norm | **0.000** (0/2000) | 0.000 |
| copy_d8 / contraction | **0.230** (460/2000) | **0.005** (1/2000) |

- **state_norm: ungated rate is 0** → the high-fitness region is already
  `|s|<=1`-safe (convex-combination structure guarantees it). The gate is a
  **NO-OP** — it removes nothing because there is nothing to remove. HONEST: this
  invariant is not load-bearing for these tasks.
- **contraction: ungated rate is high (15.5% / 23.0%)** → the ungated GA actively
  drifts into non-contractive territory (more so on the hard task that rewards
  long memory). The gate drives admitted children to **0** violations. The
  contraction gate is **LOAD-BEARING**.
- **The single gated violation (copy_d8/contraction, 1/2000, rate 0.005)** was
  investigated, not waved away: it is gene `(0.828, 0.973, 1.004)`, empL=1.001,
  which is an **un-gated initial-population survivor** of seed 1013 (verified by
  reconstructing that seed's initial pop; the contraction gate *would* reject it
  as a child — verdict False — but the initial population is un-gated by design,
  matching `src` where the gate acts on the child-admission step). It is a
  borderline `L=1.001` gene, not a soundness leak. See B3.

## B3 — Gate soundness (0 false admits)

Every gene ADMITTED through a gate as a child must independently satisfy the
invariant. We logged every admitted child (1800 per gated cell = 20 seeds × 90
non-elite children) and re-checked it empirically.

| cell | false admits / total admitted children | fallbacks (cap hit) | total rejections | total resamples |
|---|---|---|---|---|
| copy_d0 / state_norm | **0 / 1800** | 0 | 0 | 0 |
| copy_d0 / contraction | **0 / 1800** | 0 | 270 | 270 |
| copy_d8 / state_norm | **0 / 1800** | 0 | 0 | 0 |
| copy_d8 / contraction | **0 / 1800** | 0 | 236 | 236 |

**0 false admits across all 7200 admitted children.** The 50-resample cap was
never hit (0 fallbacks) — one resample always sufficed (rejections == resamples,
each rejection resolved on the next draw). Z3's gate is sound on this gene family:
no gene it admitted ever empirically violated `L<1` or `|s|<=1`. (The one B2
gated violation is an un-gated *initial* survivor, not an admitted child — B2
note.)

## B4 — Regime characterization

Regime map (load from MEASURED ungated violation rate; cost from B1):

| task | gate | load-bearing? | fitness cost | ungated pathology rate |
|---|---|---|---|---|
| copy_d0 (easy, delay=0) | state_norm | **no-op** | FREE | 0.000 |
| copy_d0 (easy, delay=0) | **contraction** | **load-bearing** | **COSTLY** | 0.155 |
| copy_d8 (hard, delay=8) | state_norm | **no-op** | FREE | 0.000 |
| copy_d8 (hard, delay=8) | **contraction** | **load-bearing** | FREE | 0.230 |

**Connection to the project theme ("like selection, the verifier gate may only
matter in specific regimes"):** This is exactly what we observe — the gate's
relevance is **not** a fixed property of the gate, it is a property of the
*(invariant × task)* pair:

1. **Which invariant matters depends on the gene family's structure.** The
   state_norm invariant is structurally implied by the clip box (convex
   combination), so it is a no-op *for every task* — a verified-but-vacuous gate.
   The contraction invariant cuts across the high-fitness region, so it is
   load-bearing.

2. **Whether the load-bearing gate COSTS fitness depends on the task regime.** On
   the easy short-horizon task, the fitness optimum sits just past the `L=1`
   boundary (best ungated genes are non-contractive), so the gate is load-bearing
   AND costly. On the hard long-horizon task, the gate is *even more*
   load-bearing (higher ungated pathology rate, 23% vs 15.5%) yet the cost
   vanishes (FREE) — the best contractive genes match the best unrestricted ones.

**Summary verdict:** Inserting the Z3 contraction gate changes WHAT evolution
finds (it removes a 15–23% non-contractive fraction and shifts the winner) at a
fitness cost that is small-but-significant on the easy task and free on the hard
task. The state_norm gate changes nothing (vacuously safe here). The
Verified×Evolvable coupling is real but **regime-gated**: safety is free, costly,
or vacuous depending on where the task's fitness optimum sits relative to the
certified region.
