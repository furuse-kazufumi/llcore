## 2. Background: an evolvable, contraction-verified recurrent core

This section describes the substrate and verification machinery that the rest of the
paper builds on. The core idea is to *evolve* the dynamics of a small recurrent system
while a **fail-closed contraction verifier** admits only children whose dynamics are
provably non-divergent. We describe (2.1) the `CoupledNDGene` substrate, (2.2) three
*sound* contraction certifiers and the soundness arguments behind them, and (2.3) the
framing in which a certifier acts as a *homeostatic constraint on evolution*. We are
careful throughout to keep the certifiers' guarantees at the strength their source
documents actually justify — no stronger.

### 2.1 Substrate: the `CoupledNDGene` recurrent map

The evolvable substrate is an `n`-dimensional coupled recurrent map (`coupled_nd.py`):

```
s' = decay ⊙ s + (1 - decay) ⊙ tanh(W s + V x)
```

with state and input `s, x ∈ R^n`, element-wise decay `decay ∈ [0,1]^n`, coupling matrix
`W ∈ [-2,2]^{n×n}`, and input projection `V = I` fixed to the identity (`coupled_nd.py`,
class `CoupledNDGene`; the map generalises an earlier `n=2` coupled map to arbitrary `n`).
A single update is the convex blend of the previous state and a `tanh` nonlinearity of the
pre-activation `W s + V x` — i.e. a leaky, saturating, RWKV-style recurrent step.

The **gene** subject to evolution is the pair `(decay, W)`; `V` is held at the identity.
The genotype is therefore a flat vector of `n + n²` real numbers (decay coordinates plus
the full coupling matrix), encoded by `CoupledNDGeneCodec` with the box bounds
`decay ∈ [0,1]^n`, `W ∈ [-2,2]^{n×n}` (`coupled_nd.py`, `CoupledNDGeneCodec.__init__`,
`self.dim = n + n*n`). This codec plugs into an *unchanged* evolutionary loop — adding a
substrate is one new codec object and nothing else changes (`README.md`, "How to extend").

The quantity that the verifiers reason about is the Jacobian of the update with respect to
the state:

```
J(t) = diag(decay) + diag((1 - decay) ⊙ t) · W,   where t_i = sech²(pre_i) ∈ (0,1]
```

(`coupled_nd.py`, `jacobian` / `_jac_at_t`). The factor `t_i = sech²(pre_i) = 1 - tanh²(pre_i)`
is the local slope of the `tanh`, which depends on the pre-activation `pre_i = (Ws)_i + (Vx)_i`.
Because the state and the (bounded) input are themselves bounded, each `t_i` ranges over a
known interval; reasoning over that interval — rather than over any single operating point —
is what lets the verifiers make a guarantee that holds for *all* reachable inputs.

### 2.2 Three sound contraction certifiers over the achievable-`t` box

The verifiers certify **contraction** — the spectral radius `ρ(J) < 1` over the set of all
reachable Jacobians — which (for this autonomous recurrence) gives the *echo-state property*:
the hidden state stays bounded, does not blow up, and forgets its initial condition
(`PREREGISTRATION.md`, §2 "帰結"). Rather than sampling operating points, all three certifiers
work over the **achievable-`t` box** `[t_min, 1]^n`, where `t_min` is computed per coordinate
from the row sums of `|W|` and `|V|` and the input bound `max_input_abs`
(`coupled_nd.py`, `t_min_per_coord`). The box has `2^n` vertices (`coupled_nd.py`,
`_box_vertices`).

The three certifiers (`coupled_nd.py`):

- **`cert_inf` — closed-form ∞-norm, O(n²).** Computes the supremum of `‖J(t)‖_∞` over the
  box in closed form. Each row's absolute-sum is V-shaped in `t_i`, so its maximum is attained
  at an endpoint `t_i ∈ {t_min_i, 1}` (`coupled_nd.py`, docstring of `infnorm_sup`). The gene
  is admitted iff `sup ‖J‖_∞ < 1`. This is the only certifier that scales: it is `O(n²)` and
  enumerates no vertices (`PREREGISTRATION.md`, §1 honest 制約).

- **`cert_two` — vertex `σ_max`, `2^n` vertices.** Takes the maximum over the `2^n` box
  vertices of the largest singular value `σ_max(J)`; admits iff every vertex has `σ_max(J) < 1`.
  This is sound because `σ_max` is convex and `J` is affine in `t`, so the maximum over the box
  is attained at a vertex (`coupled_nd.py`, comment at `two_norm`). It is solver-independent
  (vertex SVD only).

- **`cert_sdp` — vertex common-`P` Lyapunov LMI, `2^n` vertices.** Seeks a single quadratic
  Lyapunov matrix `P ≻ 0` satisfying `P − J_v^T P J_v ≻ 0` at every box vertex `J_v`, solved
  via `cvxpy` (`coupled_nd.py`, `cert_sdp`). The `2-norm` certificate is the special case
  `P = I`, so `cert_sdp` short-circuits to `True` whenever `cert_two` already passes.

**Soundness — what guarantees it.** The guarantee rests on the *certifier theorems*, not on
sampling: `‖J‖_∞ < 1`, `σ_max(J) < 1`, and a feasible common-`P` LMI each mathematically imply
`ρ(J) < 1` over the box (`VERDICT.md`, G1; `coupled_nd.py`, module docstring "All three imply
ρ(J)<1 over the box ⇒ contraction"). The pre-registration makes the box-covers-reachability
step explicit through three lemmas (`PREREGISTRATION.md`, §2):

- **Lemma 1 (bounded state).** If `s_0 ∈ [-1,1]^n` then every `s_t ∈ (-1,1)^n`, because each
  coordinate is a convex blend of values in `[-1,1]`.
- **Lemma 2 (bounded input).** With a `tanh`-bounded embedding `e(x) = tanh(E[x])`, every input
  coordinate satisfies `|e(x)_i| < 1`, so `max_input_abs = 1.0` is a sound (in fact strict)
  upper bound.
- **Lemma 3 (the `t`-box covers all reachable Jacobians).** From Lemmas 1–2,
  `|pre_i| < Σ_j |W_ij| + 1 = M_i`, hence `t_i = sech²(pre_i) > sech²(M_i) = t_min_i`, which is
  exactly the value `coupled_nd.t_min_per_coord` computes. Therefore every Jacobian reached
  during operation lies inside `[t_min, 1]^n`.

Because the box covers all reachable Jacobians, a certificate over the box transfers to a
guarantee over the actual dynamics. The empirical spectral-radius sampler (`coupled_nd.py`,
`empirical_rho`) is explicitly a **from-below consistency / falsification check**, not a
proof: observing `ρ < 1` only means *no violation was found*; the guarantee itself is the
certificate (`VERDICT.md`, §7; `PREREGISTRATION.md`, §2).

**Fail-closed engineering.** The SDP path is fail-closed in two ways. The genuine LMI solve
runs only under the CLARABEL solver; if CLARABEL is absent, `cert_sdp` *refuses* (returns
`False`) rather than silently falling back to `cvxpy`'s SCS default, which false-negatives near
the feasibility boundary (`coupled_nd.py`, `_CLARABEL_OK` / `_SOLVER`; comments at lines 35–46,
146–150). The solver's `P` is then independently re-checked by eigenvalue tests — the certifier
is never solver-blind (`coupled_nd.py`, end of `cert_sdp`). A behaviour-preserving fast path
rejects any vertex with `ρ(J_v) ≥ 1` (a necessary condition for LMI feasibility) before invoking
the solver (`coupled_nd.py`, `cert_sdp` pre-screen; `VERDICT.md`, §7).

**Scalability ceiling (honest, load-bearing).** The two stronger certifiers enumerate the `2^n`
box vertices, which is tractable only for small `n` (demonstrated for `n = 2, 3, 4` in the
module's `__main__`, and used at `n = 8` so that `2^8 = 256` vertices remain enumerable —
`PREREGISTRATION.md`, §1 honest 制約 and §6). At a real LM dimension such as `n = 32`, `2^32`
vertices is infeasible, so *only* the closed-form `cert_inf` (`O(n²)`) scales
(`PREREGISTRATION.md`, §1). A vertex-free sound 2-norm/SDP certifier for `n = 32+` is left as
explicit future work (the pre-registration's R-LLM-1 stage), precisely because rushing a
robust-LMI formulation risks unsoundness (`PREREGISTRATION.md`, §1). This is a genuine
frontier limit of the present work — both the SDP completeness degradation with dimension *and*
the exponential cost of vertex enumeration.

*Figure: the `tanh` recurrent step, its state-Jacobian `J(t)`, and the achievable-`t` box*
*`[t_min,1]^n` whose `2^n` vertices the certifiers reason over (data: coupled_nd.py).*

### 2.3 The certifier as a homeostatic constraint on evolution

The three certifiers are wired into the evolutionary loop as a `VerifierBackend` admission
gate: each mutated child must `certifies(gene) -> True` before it is admitted, so evolution can
search aggressively while the dynamical system is provably never driven into divergence
(`README.md`, "What it is"; `coupled_nd.py`, `make_nd_verifier`). The loop itself
(`evolvable_core.evolve()`) is unchanged across substrates and verifiers; the codec
(`GeneCodec`), the fitness (`Objective`, the *direction* of evolution), and the verifier
(`VerifierBackend`, the *safety*) are the three pluggable extension points (`README.md`,
"The skeleton"; `VERDICT.md`, §1). In this sense the verifier functions as a **homeostatic
constraint**: it does not steer *toward* any behaviour, but it bounds the search to the region
of provably stable dynamics, the way a homeostat keeps an evolving organism inside its viable
envelope.

This framing is supported, not merely asserted, by the source documents:

- **The gate is load-bearing.** Ungated evolution drifts into non-contraction (final
  populations measured at **rotation 19.7 %, benign 16.7 %, nonnormal 1.9 %** non-contracting),
  while all sound gates drive admitted children to 0 % (`VERDICT.md`, §5 G2). The verifier is
  removing genes that genuinely diverge, not policing a no-op.

- **A stronger verifier unlocks more reachable fitness, at no soundness cost.** On a coupled
  *rotation* task, the conservative ∞-norm gate over-rejects the rotation region and caps mean
  best fitness at **0.411**, while the 2-norm and SDP gates recover it to **0.767 / 0.859**
  (`VERDICT.md`, §4 exp2; paired one-sided Wilcoxon `p = 3.1e-5`, complete separation
  `psd = 1.00`, surviving Bonferroni over ~9 comparisons — `VERDICT.md`, §5 G4). A gate-free,
  certifier-attributed landscape analysis shows this is a *region ceiling*, not GA luck: the
  ∞-norm-admissible region cannot exceed **0.38** on rotation, whereas the 2-norm and SDP
  regions reach **0.89 / 0.90** (`VERDICT.md`, §3 exp1, 10175-gene pool). The bulk of the gain
  is **∞-norm → 2-norm**; SDP's unique gain over the 2-norm is real but thin, sitting in a
  near-boundary "SDP-only shell" with common-`P` margins of order `1e-7` (`VERDICT.md`, §7
  Codex #9, §8).

- **The effect is task-structural, not an admission-size artifact.** On a *benign* task all
  four gates tie at **0.999** (`VERDICT.md`, §4), an honest null (G5). A red-team control with a
  fitness uncorrelated with dynamics also shows no SDP edge (`p = 0.91`), so the rotation payoff
  is fitness-structural rather than a consequence of merely admitting more genes
  (`VERDICT.md`, §6 Lens A).

**Scope discipline.** Two limits are stated openly and we do not strengthen them. First, the
permissiveness ordering `|sdp| (3369) > |two| (2552) > |inf| (1860)` over the exp1 pool, and the
0.99 ceiling reported for the `non_certified` region, were corrected during audit: under
CLARABEL the SDP advantage is in fact larger than first reported (the original SCS-based numbers
*understated* it), while the raw `non_certified` 0.99 figure is a *raw region max* that can
include divergent genes and should not be read as a certified-contracting ceiling
(`VERDICT.md`, top correction banner; §3 Codex #8 correction). The honest residual finding still
stands: even the SDP gate is over-conservative relative to the true contraction set — among the
top-50 empirically-contracting rotation genes, **39 of 50 are `non_certified`** (`ρ < 1` but
provable by neither a fixed induced norm nor a common quadratic Lyapunov `P`), so the
verifier–fitness frontier continues toward JSR / non-quadratic-Lyapunov backends
(`VERDICT.md`, §3, §8). Second — and this is the scope boundary the present paper keeps — the
demonstrated result is **evolvability under a verified stability constraint**, established on
synthetic dynamical tasks (rotation / benign / nonnormal). Whether the same "stronger sound
verifier monotonically unlocks reachable fitness" claim holds when the fitness is a *real*
held-out language-model perplexity is, at this stage, an *open, pre-registered question*, not a
result: the pre-registration explicitly notes that the substrate "is not yet operating as an
actual LLM/VLM" and that a null outcome (contraction is free for real LMs; the verifier payoff
is synthetic-task-specific) would be a valid honest negative narrowing the paper's scope
(`PREREGISTRATION.md`, §0). We therefore treat the contraction-verified evolvable core as a
*substrate property* — bounded, non-divergent, evolvable dynamics — and reserve any
language-learning claim for the experiments that test it directly.
