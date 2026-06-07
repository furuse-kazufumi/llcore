## 7. Breaking the 2^n verifier wall: vertex-free sound certification

The sound certifier that gates evolution is not bottlenecked by the genome size but by
its inner verification step. This section reports a design taxonomy for attacking that
bottleneck and a CPU-only, $0 proof-of-concept that breaks the dominant cost while
keeping soundness intact. We state up front that this is a *soundness-claiming reduction*:
the failure mode is not a slow experiment but a plausible-but-unsound bound that admits a
gene it should reject, so soundness is argued at the theorem level and then checked
empirically as a falsifier, not as the proof (SKETCH.md).

### 7.1 Where the cost actually is

The evolvable genome is `(decay, W)` = `n + n²` reals (dense `CoupledNDGeneCodec`), but the
binding runtime cost is the sound certifier's `2^n` t-box vertex enumeration, not the genome
size (SKETCH.md). Three certifier backends sit on a cost/conservatism trade-off:

- `cert_inf` is an O(n²) closed form with no enumeration. It is cheap but over-conservative,
  and prior work in this codebase reports it traps the evolutionary search at a unigram model
  (SKETCH.md, citing `../verified_lm_evolution/VERDICT.md §3b/§3c`).
- `cert_two` enumerates the `2^n` vertices of the achievable-`t` box and computes one SVD,
  O(n³), at each, giving O(2^n · n³) (SKETCH.md). This is the exact 2-norm reach we treat as
  the yardstick.
- `cert_sdp` folds the same `2^n` vertices into an LMI/Lyapunov-`P` SDP and is the dominant
  cost; SKETCH.md reports the null run took ~1000–2000 s/seed, SDP-bound (SKETCH.md).

Critically, the `2^n` blow-up is on the **state dimension n**, not on the `n²` weights, which
is what puts `n=32` out of reach (SKETCH.md).

### 7.2 Four levers, classified by what they cut

SKETCH.md organizes the attack into four levers, honest about which cost each one actually
reduces and what soundness risk it carries:

- **L1 — low-rank / pruned W.** Truncated-SVD, `W≈AB` (rank r), or magnitude pruning shrinks
  the genome `n²→2nr`, making the EA search cheaper and less prone to overfit. It is trivially
  sound (the certifier is unchanged on a constrained W), but its leverage is *modest*: it cuts
  EA cost, **not** the `2^n` verifier cost (SKETCH.md).
- **L2 — vertex-free sound certifier (R-LLM-1).** One structured robust-LMI / interval-matrix
  2-norm / μ-analysis / SOS bound over the t-box replaces the `2^n` enumeration, taking the cost
  from `2^n` to poly(n). This has the highest leverage (it kills the dominant cost and is
  rank-independent) but is also the trap: a slightly loose bound becomes an unsound admit, so a
  theorem-level proof must come first (SKETCH.md).
- **L3 — model-order reduction (balanced truncation).** Discarding small Hankel-σ modes and
  certifying an r-dimensional reduced system attacks the exponent itself (`2^n → 2^r`). SKETCH.md
  flags this as the hardest: the reduced model must soundly over-approximate the full Jacobian
  set, and because the dynamics are nonlinear (tanh), LTI error bounds do not transfer directly
  (SKETCH.md).
- **L4 — cost as an internal selection pressure.** Rather than engineering cost down around the
  EA, fold a structural-cost term into fitness and let evolution prefer cheap-to-verify genes,
  framed as a multi-objective `(maximize held-out likelihood, minimize structural cost)` with
  structural cost = rank(W) / sparsity / active state-dimension (SKETCH.md).

SKETCH.md describes a synergy path in which L1 makes the active recurrent subspace explicit
(rank r), L3 reduces the certified state dimension to r, and L2 certifies the reduced system
vertex-free at `2^r` or poly cost. It also notes the spectral primitives L2/L3 need (Jacobian
SVD, the Lyapunov/Gramian-like SPD `P` from `cert_sdp`) are already present in the codebase,
wired to *prove* rather than to *reduce*, so the work is re-pointing them plus the soundness
argument (SKETCH.md).

A caveat on L4 is load-bearing and worth stating in full, because it is where llcore's own L3
result bites: there are two kinds of "cheap" and only one is good (SKETCH.md). Good cheap is
**structural simplicity** (low rank / sparse W / small active dimension), which is genuinely
cheaper to certify and more navigable. Bad cheap comes in two forms — **certifier conservatism**
(rewarding "use the cheap conservative `cert_inf`" pushes the EA straight into the unigram trap)
and **degenerate behavior** (the unigram collapse is itself a too-simple structure that survives
because it is safe and cheap but useless). SKETCH.md concludes the cost term must target
structural cost only, and must be expressed as a Pareto front (CE vs structural cost), not a
weighted scalar sum. The reason llcore can study this honestly is that it has a soundness oracle
that distinguishes "good simple" from "degenerate simple" — most evolutionary systems cannot
(SKETCH.md).

> Figure: Four-lever taxonomy (L1 low-rank, L2 vertex-free, L3 MOR, L4 cost-as-selection) versus
> what each cuts and its soundness risk (data: SKETCH.md, "Three levers" table + L4 section)

### 7.3 The bound and its soundness argument

The Jacobian over the achievable-`t` box is affine in `t`:
`J(t) = diag(decay) + diag((1−decay)⊙t)·W` for `t ∈ [t_lo, 1]^n` (poc_l2lite.py). Writing
`M = J(t_mid)` for the box midpoint and `R` for the entrywise half-width matrix
`R_ij = (1−decay_i)·((1−t_lo_i)/2)·|W_ij|` (the constant decay diagonal contributes zero),
every `J` in the box satisfies `|J − M| ≤ R` entrywise (poc_l2lite.py). Because the spectral
norm is monotone under nonnegative entrywise domination, this yields the two candidate
single-SVD-class upper bounds evaluated in the PoCs (poc_l2lite_v2.py):

- **B1** `= σ(M) + σ(R)` — the triangle split, 2 SVDs (poc_l2lite_v2.py).
- **B2** `= σ(|M| + R)` — abs-domination, since `|J| ≤ |M| + R` entrywise and `σ_max` is monotone
  under nonnegative domination, 1 SVD (poc_l2lite_v2.py).

Both are upper bounds on `sup_{t} σ_max(J(t))`, so by construction every admit set is a subset
of the genes that contract over the box, i.e. of the `cert_two` admit set; therefore a soundness
violation (admitting a gene `cert_two` rejects) is impossible unless there is a bug, and the
PoC checks it empirically only as a falsifier (poc_l2lite.py, poc_l2lite_v2.py).

### 7.4 PoC results: B1 too loose, B2 recovers the bulk

PoC-1 (n=8, 3000 genes, seed 20260606, region-populating sampler) found the cost win is decisive
and B1 is sound but too loose (poc_l2lite_results.json). Across 3000 genes B1 produced **0**
soundness violations against `cert_two` (poc_l2lite_results.json). On cost, the vertex-free bound
stays roughly constant per gene while `cert_two` explodes with `2^n`: speedup was
**60× at n=8, 980× at n=12, and 12,520× at n=16** (poc_l2lite_results.json), where `cert_two` cost
0.006402 → 0.082118 → 2.761525 s/gene and the bound ≈ 0.0002 s/gene over n=8/12/16
(poc_l2lite_results.json). At n=16 this is the `2^n = 65536`-vertex enumeration being replaced by a
single SVD class (poc_l2lite_results.json). On tightness, however, B1 admitted only **387** genes —
**29.5%** of `cert_two`'s reach — rejecting 700 of `cert_inf`'s admits while gaining only 15 of inf's
misses, leaving it strictly more conservative than inf overall (poc_l2lite_results.json).

PoC-2 (same n=8, 3000 genes, seed 20260606) tested B2 and reversed that pessimism
(poc_l2lite_v2_results.json). The full admit comparison against the `cert_two` yardstick of 1310:

> Figure: Admit-set coverage by certifier, n=8 / 3000 genes (data: poc_l2lite_v2_results.json)

| certifier | admits | % of exact `cert_two` (1310) |
|---|---|---|
| `cert_inf` | 1072 | 81.8% |
| `cert_two` (exact, 2^n) | 1310 | 100% (yardstick) |
| B1 = σ(M)+σ(R) | 387 | 29.5% |
| **B2 = σ(\|M\|+R)** | **1017** | **77.6%** |
| inf ∪ B2 | 1142 | 87.2% |

(All counts from poc_l2lite_v2_results.json.) B2 recovers **77.6%** of the exact `2^n` `cert_two`
reach with a **single SVD**, at **0** soundness violations (poc_l2lite_v2_results.json), and the
PoC-1 cost table places that single-SVD class at ~12,520× cheaper than the `2^n` method at n=16
(poc_l2lite_results.json). The cheap union gate **inf ∪ B2 admits 1142, which exceeds inf's own 1072**
(poc_l2lite_v2_results.json) — the only configuration in PoC-2 that beats inf's coverage
(`beats_inf_coverage` is true for `inf_or_b2` only; poc_l2lite_v2_results.json). The PoC also records
that `cert_inf ⊄ cert_two` on this pool: 75 of inf's admits are not 2-norm-contracting because inf
is a different norm (poc_l2lite_v2_results.json). The honest reading is that PoC-1's headline ("naive
vertex-free 2-norm is worse than inf") was an artifact of the bad bound B1, not a property of
vertex-free 2-norm certification per se (SKETCH.md). The structural reason B1 is loose, per SKETCH.md,
is that `σ(M)+σ(R)` treats the t-box as `n²` independent entry-intervals, whereas the real
perturbation `Δ = diag((1−decay)(t−t_mid))·W` is parameterized by only the **n** values `t_i` (each
row shares one `t_i`), so the naive split over-inflates the radius (SKETCH.md).

### 7.5 Scope, residual, and open questions

The scaling win is in hand: `inf ∪ B2` is a poly-cost, vertex-free, sound gate covering ~87% of the
`2^n` certifier, so `n>8` is reachable on CPU without the `2^n` wall (SKETCH.md). But ~22% of
`cert_two`'s 2-norm reach is still missed by B2 (the 1310 − 1017 genes), and SKETCH.md states that
remaining tail — together with the genes only an LMI can certify — is the territory of the genuine
robust-LMI / SDP route (R-LLM-1, designated PoC-3), which remains user-gated and is not auto-run
(SKETCH.md).

We deliberately do not over-claim here. First, the headline numbers come from a single
configuration: n=8, 3000 genes, one seed (20260606), `max_input_abs = 1.0`, and a specific
region-populating sampler (decay biased high, small Gaussian `W` scaled by `1/√n`); the cost
speedups were measured at n=8/12/16 with only 300/60/8 genes respectively
(poc_l2lite_results.json, poc_l2lite_v2_results.json). Second, and most important for honesty:
whether the ~22% tail that B2 misses carries the navigable, low-perplexity dynamics — i.e. whether
losing those genes actually costs the evolutionary search anything useful — is **not yet measured**.
SKETCH.md is explicit that the cross-entropy of the B2-missed-but-`cert_two`-admitted genes must be
measured before building the SDP, because that is what decides whether the SDP tail is worth its cost
(SKETCH.md). Until that measurement exists, the claim is bounded to: a single-SVD vertex-free sound
certifier recovers most of the exact `2^n` certifier's admit set at orders-of-magnitude lower cost,
with the navigability of the missed tail an open question. Consistent with the L4 framing, this is a
statement about *verifiability cost*, not about language-learning capability; the soundness oracle
gates which dynamics are admissible, it does not by itself demonstrate that the admitted dynamics
learn language.
