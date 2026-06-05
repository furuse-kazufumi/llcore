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

## Relation to existing plan
R-LLM-1 = "vertex-free sound certifier for n=32+" is already the named next step
(`../verified_lm_evolution/VERDICT.md §5`, `CPU_MEMORY_EFFICIENCY_PLAN.md §3`). This sketch refines it
into the L1/L2/L3 taxonomy + a sound minimal first experiment. `CPU_MEMORY_EFFICIENCY_PLAN.md` covers
*constant-factor* CPU/memory wins; this thread is the *algorithmic-complexity* (`2^n`) attack.
