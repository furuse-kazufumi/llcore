# Verifier cost-reduction thread — design sketch (R-LLM-1 adjacent)

> Status: **design-first, not yet measured** (user endorsement 2026-06-06). This is a
> *soundness-claiming reduction* — the exact class where the R-reach lesson applies: prove
> soundness at theorem level BEFORE measuring, because an empirical oracle cannot catch a
> plausible-but-unsound bound. Do NOT auto-run heavy experiments. CPU/$0 minimal rung first.

## Where the cost actually is (honest profile)

The evolvable genome is `(decay, W)` = `n + n²` reals (dense, `CoupledNDGeneCodec`). The binding
runtime cost is **not** the genome size — it is the **sound certifier's `2^n` t-box vertex
enumeration**:

- `cert_inf` — O(n²) closed form, no enumeration. Cheap, but over-conservative (traps the EA at
  unigram; see `../verified_lm_evolution/VERDICT.md` §3b/§3c).
- `cert_two` — enumerates the `2^n` vertices of the achievable-`t` box, one SVD (O(n³)) each
  → O(2^n · n³).  (`src/llcore/verifier/backends.py:175`, `coupled_nd.py:134`)
- `cert_sdp` — same `2^n` vertices folded into an LMI / Lyapunov-`P` SDP → the dominant cost
  (the null run took ~1000–2000 s/seed, SDP-bound). (`backends.py:212,228,232`)

`2^n` is on the **state dimension n**, NOT on the `n²` weights. This is the wall that puts n=32
out of reach.

## Three levers (which cost each cuts — honest)

| lever | what it is | cuts | soundness | leverage |
|---|---|---|---|---|
| **L1 low-rank / pruned W** | truncated-SVD / `W≈AB` (rank r) / magnitude prune | genome `n²→2nr` → cheaper EA search, less overfit | trivially sound (constrained W; certifier unchanged) | **modest** — cuts EA cost, NOT the `2^n` verifier cost |
| **L2 vertex-free sound certifier (R-LLM-1)** | one structured robust-LMI / interval-matrix 2-norm / μ-analysis / SOS over the t-box instead of `2^n` enum | `2^n → poly(n)` | **TRAP**: a slightly-loose bound = unsound admit. Theorem-level proof first. | **highest** — kills the dominant cost, rank-independent |
| **L3 model-order reduction (balanced truncation)** | discard small Hankel-σ modes → certify r-dim reduced system | `2^n → 2^r` (attacks the exponent) | **hardest**: reduced model must soundly over-approximate the full Jacobian set; nonlinear (tanh) ⇒ LTI error bounds don't transfer | **highest payoff, hardest** |

**Synergy (the integrated scaling path):** L1 makes the active recurrent subspace explicit (rank r)
→ L3 reduces the certified state dimension to r → L2 certifies the reduced system vertex-free at
`2^r` or poly cost. Low-rank + MOR + vertex-free compose.

**Already present:** the spectral primitives L2/L3 need (Jacobian SVD, the Lyapunov/Gramian-like SPD
`P` from `cert_sdp`) are ALREADY in `backends.py` — wired to *prove*, not to *reduce*. The work is
re-pointing them + the soundness argument, not building from scratch.

## Minimal first rung (sound, CPU, $0, falsifiable)

**"L2-lite" interval-matrix 2-norm enclosure.** Replace the `2^n` per-vertex SVD in `cert_two` with a
single sound upper bound on `‖J(t)‖₂` over the t-box (e.g. via `|J|` / interval-matrix spectral-norm
enclosures, or a Perron bound on `diag(decay) + diag((1−decay)·t_max)·|W|`). Pre-registered checks:

1. **Soundness (load-bearing):** on a large random gene pool, the enclosure must NEVER admit a gene
   the exact `2^n` enumeration rejects (conservative-or-equal). One counterexample ⇒ NO-GO.
2. **Cost:** wall-clock vs the `2^n` method at n=8,12,16 (expect crossover where enumeration explodes).
3. **Tightness:** admit-rate vs exact two_norm region (how much conservatism the speed buys; if it
   collapses toward `cert_inf`'s over-conservatism it inherits the navigability trap — measure it).

If L2-lite is sound AND not as conservative as inf, it is the cheapest real progress toward R-LLM-1
and unlocks n>8 without GPU. If it collapses to inf-like conservatism, the lesson is that vertex-free
2-norm is insufficient and the SDP/SOS route (true R-LLM-1) is required — also a clean result.

## L4 — cost as an internal SELECTION PRESSURE (not just external engineering)

(User insight, 2026-06-06: "reducing cost is itself a valid direction of evolution — in nature
simpler structures sometimes survive.") Instead of engineering cost down *around* the EA, fold a
**structural-cost term into fitness** and let evolution prefer cheap-to-verify genes by itself:
multi-objective `(maximize held-out likelihood, minimize structural cost)` where structural cost =
rank(W) / sparsity / active state-dimension.

**Biological precedent (real):** reductive evolution / genome streamlining (Prochlorococcus,
Pelagibacter/SAR11 = smallest free-living genomes, selected for economy & replication speed),
minimal symbiont genomes (Buchnera, Mycoplasma), loss of unused costly traits (cave-fish eyes,
flightless birds) under "use it or lose it" / relaxed selection. **TRIZ:** this is the law of
increasing Ideality (= function / cost) + Trimming — same function at less cost is the central
direction of system evolution. **ML:** hardware-aware NAS (latency/FLOPs in the objective),
L1/weight-decay, MDL / Bayesian Occam, NSGA-II accuracy-vs-complexity Pareto fronts.

**CRITICAL caveat — two kinds of "cheap", only one is good (llcore's own L3 result bites here):**
- ✅ **good cheap = structural simplicity** (low rank / sparse W / small active dim): genuinely
  cheaper to certify AND more navigable. Reward this.
- ❌ **bad cheap #1 = certifier conservatism**: `cert_inf` is the cheapest certifier (O(n²)) but
  §3b/§3c show it traps evolution at unigram. A naive "cheap-to-verify = good" reward would push the
  EA straight INTO the inf trap. The cost term must target *structural* cost, never "use the cheap
  conservative certifier".
- ❌ **bad cheap #2 = degenerate behavior**: the unigram collapse itself is "a too-simple structure
  that survives because it is safe/cheap but useless" — the obligate-parasite failure mode. So
  scalarizing "cheap = good" is dangerous; use a **Pareto (CE vs structural cost)** front, not a
  weighted sum.

**Why llcore is uniquely equipped:** most evolutionary systems cannot tell "good simple" (low-rank,
still beats unigram) from "degenerate simple" (unigram, learns nothing). llcore has a **soundness
oracle** that distinguishes them — so it can study cost-pressured evolution honestly. This is the
FullSense thesis made literal: *evolution toward verifiability* — organisms selected not just to be
fit but to be cheap-to-prove-safe. Design-first + Pareto framing; honest disclosure that adding a
cost objective changes what "best" means. User-gated like L1–L3.

## PoC-1 result (L2-lite, n=8/12/16, 2026-06-06) — `poc_l2lite.py` / `poc_l2lite_results.json`

Minimal first rung run. **Cost = decisive win; tightness = the cheap bound is too loose.** Honest
verdict: the naive interval midpoint+radius 2-norm bound is **sound but more conservative than even
`cert_inf`**, so it is NOT a drop-in — the cost payoff needs a *structure-aware* tighter bound or the
real robust-LMI.

| metric | result |
|---|---|
| **soundness** (3000 genes, region-populating sampler) | **0 violations** — L2-lite never admitted a gene `cert_two` rejects (387 actual admits, non-vacuous). Conservative-by-construction confirmed. |
| **cost speedup** vs exact `2^n` `cert_two` | **60× (n=8) → 980× (n=12) → 12,520× (n=16)**; L2-lite ≈ constant ~0.0002 s/gene (2 SVDs), `cert_two` = 0.006→0.082→2.76 s/gene. The `2^n` wall is broken. |
| **tightness** | L2-lite admits only **29.5%** of the exact `two_norm` region; rejects **700 of `cert_inf`'s 1072** admits, gaining only **15** inf misses → **strictly more conservative than inf overall**. |

**Why the cheap bound is loose (and the fix direction):** `σ(J) ≤ σ(M) + σ(R)` treats the t-box as
`n²` *independent* entry-intervals, but the real perturbation is `Δ = diag((1−decay)(t−t_mid))·W`,
parameterized by only the **n** values `t_i` (row `i` scales as one shared `t_i`). The naive bound
gives each entry its own worst case → over-inflated radius. A tighter vertex-free certificate must
exploit this **n-parameterized (low-rank-ish) structure** of the perturbation — which is exactly the
bridge to L1/L3 (low rank / reduced dimension) and to the genuine **robust-LMI / SDP = R-LLM-1**.

**Conclusion (PoC-first, spec kept small):** vertex-free 2-norm *is* enormously cheaper and provably
sound, but the *naive* interval split is too conservative to use as the gate (it would worsen the
navigability trap). Next rung = a structure/correlation-aware sound bound (or the SDP robust-LMI),
measured the same way (soundness-consistency + tightness vs exact + cost). Not auto-run; user-gated.

## Relation to existing plan
R-LLM-1 = "vertex-free sound certifier for n=32+" is already the named next step
(`../verified_lm_evolution/VERDICT.md §5`, `CPU_MEMORY_EFFICIENCY_PLAN.md §3`). This sketch refines it
into the L1/L2/L3 taxonomy + a sound minimal first experiment. `CPU_MEMORY_EFFICIENCY_PLAN.md` covers
*constant-factor* CPU/memory wins; this thread is the *algorithmic-complexity* (`2^n`) attack.
