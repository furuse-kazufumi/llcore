> ⚠️ **SOLVER NOTE (2026-06-03 audit, `research/AUDIT_SCS_CLARABEL_2026-06-03.md`):** this verdict's
> **headline is UNAFFECTED by the SCS→CLARABEL correction.** The §2 scaling table (none / inf_norm /
> two_norm at n=2,3,4; the ∞-norm ceilings 0.491 / 0.459 / 0.343 and the two−inf payoff) is produced by
> `exp_nd.py`, whose sweep uses only the `none` / `inf_norm` / `two_norm` gates — **the SDP/cvxpy gate is
> deliberately excluded** (§2 already states this). Those two certifiers are **closed-form and
> solver-independent**: `cert_inf` is the numpy closed-form `infnorm_sup`, and `cert_two` is
> `np.linalg.svd` over the 2ⁿ box vertices. Neither touches cvxpy or SCS, so SCS false-negatives cannot
> have biased the headline; the inf-vs-two-norm dimensional payoff stands exactly as measured. For
> completeness the `coupled_nd.cert_sdp` path (used only by the `classify_region` winner labels and the
> n=2 SDP shell, never the headline table) is now **CLARABEL-pinned** (`coupled_nd.py`
> `_SOLVER = cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None`), matching the rest of the
> audited codebase. Soundness unaffected (G1 still 0 divergent admitted). Read with the audit.

# VERDICT — richer GeneCodec (n-dim) + dimension scaling (2026-06-03)

> Roadmap item #1 (skeleton extension): add an n-dimensional `GeneCodec` and matching
> n-dim `VerifierBackend`s, plug into the **unchanged** `evolvable_core.evolve()`, and ask:
> does the conservative ∞-norm verifier's evolutionary cost — and the better-verifier
> payoff — scale with dimension? `research/`, src untouched, seeds fixed, honest disclosure.

## 1. Skeleton extensibility — DEMONSTRATED

`coupled_nd.py` adds a brand-new substrate (n-dim coupled-RWKV: decay∈Rⁿ, W∈Rⁿˣⁿ) as a
`CoupledNDGeneCodec` + n-dim certifiers (inf/2-norm closed-form & 2ⁿ-vertex; SDP via cvxpy)
+ a block-rotation `Objective`. **Only the codec and backend changed — `evolve()` is byte-for-
byte the same.** This is the骨組み payoff: a new evolution substrate is a drop-in plug-in.
Regression: the n-dim inf/2-norm certifiers agree with the validated n=2 Track-C/D
certifiers on **0 / 600** genes. `test_nd.py` (codec, soundness, climb, gate) passes.

## 2. Result (block-rotation, 10 seeds, pop=40, gens=50; gates none/inf_norm/two_norm)

| n | none | inf_norm | two_norm | **two − inf** (paired) | G1 divergent admitted |
|---|---|---|---|---|---|
| 2 | 0.981 | 0.491 | 0.880 | **+0.388** (p=9.8e-4, psd=1.00) | 0 |
| 3 | 0.767 | 0.459 | 0.582 | **+0.123** (p=9.8e-4, psd=1.00) | 0 |
| 4 | 0.840 | **0.343** | 0.603 | **+0.260** (p=9.8e-4, psd=1.00) | 0 |

(SDP gate not in this sweep — its 2ⁿ-vertex cvxpy LMI is impractically slow at n=4, and its
unique gain over the 2-norm is the *thin shell* already established at n=2; the cheap 2-norm
is the "better verifier" for the bulk effect here.)

## 3. What scales, what doesn't (honest)

- **The better-verifier payoff GENERALISES to higher dimensions.** At every n, the
  conservative ∞-norm gate significantly under-performs the 2-norm gate (two ≫ inf,
  p=9.8e-4, psd=1.00 at n=2,3,4), with **0 divergent admitted** (soundness holds in n-dim).
  So "a better sound verifier lets verified evolution reach more fitness" is **not an n=2
  artifact** — it holds as the AI core scales.
- **The ∞-norm ceiling drops monotonically with n** (0.491 → 0.459 → 0.343): as predicted,
  more coordinates ⇒ more rotation blocks ⇒ the ∞-norm (row abs-sums) over-rejects *more* of
  the rotational optimum. This is the dimensional mechanism, and it is monotone.
- **But the payoff MAGNITUDE does NOT cleanly grow with n** (+0.388, +0.123, +0.260 — non-
  monotonic). The hypothesis "payoff grows with dimension" is **not cleanly supported.** The
  confound is the **GA's convergence difficulty at higher n**: with fixed pop/gens the
  achievable fitness (even ungated: none = 0.98 / 0.77 / 0.84, itself non-monotonic) is
  limited by search, not the verifier, so the two-vs-inf *gap* mixes "verifier reach" with
  "how far the GA got." At n=3 both gates are search-limited (none only 0.77), compressing
  the gap; n=4 with a better ungated reach shows a larger gap again. A clean
  dimension-scaling law would need search-effort normalised to convergence per n (future).

## 4. Bottom line

- ✅ **Skeleton extension works**: a new n-dim substrate is a pure plug-in; `evolve()` unchanged.
- ✅ **Verified-evolution + better-verifier payoff generalise to higher dimensions** (n=2,3,4,
  p=9.8e-4, sound).
- ✅ **The ∞-norm's dimensional blind spot is real and monotone** (its rotational ceiling
  falls with n).
- ⚠️ **"Payoff grows with dimension" is NOT cleanly shown** — confounded by GA convergence;
  honest negative on the *magnitude-scaling* claim, while the *existence-at-all-n* claim is
  strong.

Artifacts: `coupled_nd.py`, `test_nd.py`, `exp_nd.py`, `exp_nd_results.json`. No push.
Next roadmap item: #2 — a degree-4 / non-quadratic Lyapunov `VerifierBackend`
(`verifier_deg4.py`) to reach the Track-D D4 residual.
