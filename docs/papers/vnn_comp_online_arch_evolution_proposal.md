# Online Architecture Evolution Verification: A New VNN-COMP Category for Continuously Mutating Neural Networks

**Draft target venues**: TMLR (full paper, primary) / GECCO 2027 short paper (Evolutionary Computation track) / NeurIPS 2026 workshop on Verification × ML.
**Status**: Author draft, not peer-reviewed.
**Authors**: anonymous for review.
**Date**: 2026-05-29.

---

## Abstract

The Verification of Neural Networks Competition (VNN-COMP) has, across its first six editions (2020–2025), become the de facto benchmark for the neural network verification community, with α,β-CROWN the winning verifier in 2021, 2022, 2023, 2024, and 2025 (the fifth consecutive year). However, every VNN-COMP benchmark to date assumes a **fixed** network topology: a verifier is handed one `.onnx` artifact plus one `.vnnlib` query, and asked to prove or disprove local input-robustness. Recent work on **online architecture evolution** — small CPU-budget systems that mutate network topology hundreds or thousands of times during a single training run (e.g., AutoML-Zero, weight-agnostic neural networks, neural architecture search with hot-restart, and the present authors' llcore prototype) — has no benchmark slot in which to compete, because the verifier's task is no longer "prove robustness for *this* network" but "**incrementally** prove that an unbounded stream of `ChangeOp` mutations preserves a safety invariant in bounded wall-clock per step." Even the closest existing work, Marabou's incremental NN verification (Elsaleh et al. 2026), only addresses *same-network, different-query* incrementality, not *different-network, same-invariant* incrementality.

We propose a new VNN-COMP category — `online-arch-evo` — and falsifiably claim:

1. **No existing category measures online architecture evolution verification.** We give three concrete benchmark queries that α,β-CROWN provably cannot answer in its current form, and show why (§2).
2. **A precise benchmark specification is possible.** We give an input format (`.vnnlib` + `.onnx` + a new `.changeop_seq` file), an output format (per-step `sat`/`unsat`/`timeout`/`error`), a scoring rule with a wall-clock budget, and a submission protocol — all CPU-completable (§4, §5).
3. **A reference implementation exists.** We adapt the present authors' llcore PoC 1a (a Z3 state-norm invariant verifier; 5.8 ms per query) into a category-conformant tool that handles a 5-step `ChangeOp` sequence with per-step verdicts (§6).
4. **The category is open-ended by construction.** It uses (a) wall-clock-bounded scoring of arbitrarily long mutation streams, (b) a kernel-typed `ChangeOp` interface that admits unseen kernel families (rwkv / mamba / hopfield / linear-attention) without spec changes, (c) a minimal-criterion coevolution round in which `ChangeOp` *populations* compete (after Brant & Stanley's POET), and (d) a MODES (Bedau evolutionary activity statistics) integrity sensor that rejects saturated/gaming submissions (§4.4).

We discuss honest limitations: the proposal has not yet been peer-reviewed, the reference implementation is scalar-state (multi-dimensional state extensions remain Future Work), and we do not claim our verifier outperforms α,β-CROWN on *any* existing VNN-COMP query — we claim it answers a *different* class of queries that current tools cannot phrase at all.

**Keywords**: verified neural architecture evolution, online verification, VNN-COMP, incremental SMT, open-ended evolution, MODES.

---

## 1. Introduction

The fixed-network assumption that underpins every Verification of Neural Networks Competition (VNN-COMP) benchmark to date is now in tension with three concurrent research currents:

1. **Continuous architecture search.** AutoML-Zero (Real et al. 2020) and weight-agnostic neural networks (Gaier & Ha 2019) generate new network *structures* hundreds of times per training run. Modern small-compute evolutionary work — including the recent llive *lldarwin_v2* line and the llcore project this paper draws on — pushes the mutation frequency past 1000 mutations per minute on a single CPU core.
2. **State-update kernel innovation.** RWKV (Peng et al. 2023), Mamba (Gu & Dao 2024), Hopfield-style modern associative memories (Ramsauer et al. 2021), and linear-attention variants are now mixed within a single trained model. A modern "architecture" is no longer a static `.onnx` file but a *trajectory* of compositions.
3. **Verified evolution as a research goal.** TorchLean (George et al. 2026) formalises a neural network as a Lean 4 object so that *value-domain* properties (robustness, Lyapunov stability) are mechanised. Marabou Incremental (Elsaleh et al. 2026) reuses learned conflicts across related queries. Neither addresses *structure-domain* incrementality across mutations.

These currents share a verification problem that **no VNN-COMP category has yet been written for**: given a stream of mutation operations on a network, certify per-mutation that a safety invariant (output-bound, Lipschitz, Lyapunov, robustness margin, …) is preserved, with bounded wall-clock per step and a refusal-or-rollback option when the proof fails.

### 1.1 Three concrete queries α,β-CROWN cannot answer today

To make the gap precise, we point at three queries we have actually attempted to phrase against α,β-CROWN (Wang et al. 2021), the five-time VNN-COMP winner.

> **Query A — "Is mutation *m* safe to apply?"** Given a parent network *N*, a `ChangeOp` *m* (e.g., "insert one Mamba-style state-update block after layer 3"), and an invariant *φ* (e.g., "the state norm stays in [0, 1] for any input in [-1, 1]"), produce a verdict before *m* is committed. α,β-CROWN can verify *N* and verify *N+m* independently; it cannot phrase the *delta* and refuses to share work across the two proofs.
>
> **Query B — "How long can the population evolve before the invariant breaks?"** Given a population of 64 candidate networks and a mutation budget of 1000 `ChangeOp` applications per individual, report the per-step verdict trace and the first step at which *φ* fails. α,β-CROWN has no API for sequences; the practitioner is forced into a Python `for`-loop with 64 000 independent VNN-COMP invocations and no shared state.
>
> **Query C — "Does this mutation family preserve the invariant?"** Given a *family* of `ChangeOp` (e.g., "all linear re-parametrisations of the gating tensor"), prove that *no* member of the family violates *φ*. α,β-CROWN's input is one network at a time; the universal quantifier over the mutation family must be discharged outside the tool, by hand.

Queries A, B, and C correspond, respectively, to (i) a *commit gate*, (ii) a *survival curve*, and (iii) a *family soundness proof*. None of them fits the current VNN-COMP I/O contract.

### 1.2 What we propose

We add a new VNN-COMP category, `online-arch-evo`, with:

- A formalisation of the problem (§3): the verifier receives a triple `(network_0, invariant, changeop_stream)` and emits per-step verdicts under a wall-clock budget per step.
- A benchmark specification (§4): file formats, output formats, scoring, time budget, submission protocol.
- A reference implementation (§6): llcore PoC 1a, wrapped to read the spec's file format and emit per-step verdicts in 5.8 ms each on a single CPU core.
- Two baselines (§7): (i) α,β-CROWN run from scratch per step (the "naive" baseline), (ii) Marabou with conflict inheritance restricted to non-structural mutations.
- Four mechanisms to keep the category **open-ended for years** (§4.4): wall-clock-bounded scoring of arbitrarily long streams, kernel-typed `ChangeOp` interface that accepts unseen kernels, a coevolutionary round that lets `ChangeOp` populations compete, and a MODES integrity sensor against saturated/gaming submissions.

The contribution is the *category, the spec, and the reference toolchain*; the present paper does **not** claim a new verifier algorithm, nor that our reference implementation defeats α,β-CROWN on any existing benchmark. It claims that the new category is a precisely specified, decidable, and useful target that competitive verifier teams can submit to.

### 1.3 Paper structure

§2 surveys related work. §3 defines the problem. §4 specifies the benchmark (and §4.4 the four open-endedness mechanisms). §5 fixes the scoring rule. §6 describes the reference implementation. §7 discusses baselines. §8 contains discussion, §9 limitations, §10 future work.

---

## 2. Related Work

### 2.1 VNN-COMP and α,β-CROWN

VNN-COMP has run yearly since 2020 (Brix et al. 2024 [arXiv:2412.19985] summarise the 5th edition). The contest uses two file formats — `.onnx` for the network and `.vnnlib` for the specification (input pre-condition and output post-condition). Every benchmark is **fixed-network**: the verifier outputs `sat` (a counter-example to the post-condition exists), `unsat` (the post-condition is proved), or `timeout`. α,β-CROWN (Wang et al. 2021) is the five-time consecutive winner; its core method is a tight linear bound propagation (CROWN) combined with bound-aware branch-and-bound (β-CROWN) and an α-parametrisation that lets the optimiser tighten the bound during search. The system is highly GPU-parallel and is, on VNN-COMP queries, *currently the verifier to beat*.

α,β-CROWN's verification API explicitly assumes one `.onnx` per invocation; the codebase has no notion of a "mutation that produces a new `.onnx`". This is not a flaw of α,β-CROWN — it is a faithful implementation of VNN-COMP's current contract. The category we propose does not replace this contract; it extends it.

### 2.2 Marabou and Incremental NN Verification

Marabou (Katz et al. 2019; v2.0 Wu et al. 2024 [CAV]) is a SMT-style verifier in the Reluplex lineage. Elsaleh et al. (2026) [arXiv:2603.12232] introduced *incremental NN verification via learned conflicts*: when the verifier discharges one query on a network *N* and learns ReLU activation-phase conflicts, the same conflicts can be reused, soundly, on related queries on *the same* *N* — e.g., a binary search on the robustness radius. Reported speed-up: up to 1.9× over non-incremental Marabou across three task families (local robustness radius, input splitting, minimal sufficient features).

**Why this is necessary but not sufficient for our category.** Elsaleh et al.'s refinement relation is between two queries on *the same network*. A `ChangeOp` that adds a new subblock generates a network *N′* whose ReLU phase space is strictly larger; the inherited conflicts must be *re-validated* against the new neurons, and the soundness theorem must be re-proved. We discuss this in §6.4 as a *bridge problem*; the present paper does not solve it, but the category we propose makes it a tractable research target (and the reference implementation gives a baseline against which a Marabou-incremental-with-ChangeOp-refinement entry can be measured).

### 2.3 TorchLean and the value-domain / structure-domain split

TorchLean (George, Cruden, Zhong, Zhang, Anandkumar; arXiv:2602.22631; 2026-02-26) lifts neural networks into Lean 4 as first-class objects, so that *value-domain* properties (interval bound propagation, CROWN bounds, Lyapunov stability of neural controllers, the universal approximation theorem) can be mechanised inside the same Lean proof environment that holds the network's definition. The framework is currently small-scale (MNIST/CIFAR).

The relation to our category is *complementary*. TorchLean answers "does this fixed network satisfy this value-domain property?" with a mechanised proof. Our category asks "does this *delta* preserve the invariant given the parent satisfies it?" The interface is the same `.onnx` artefact, and a TorchLean-based entry could submit to `online-arch-evo` by certifying each `ChangeOp` post-condition via its Lean infrastructure (at the cost, currently, of one Lean kernel pass per step — well outside the per-step budget we propose, but a plausible Tier-B entry for sampled steps). We make this co-existence explicit in §4.3.

### 2.4 llive lldarwin_v2 and llcore

The llive lldarwin_v2 lineage (open-source, Apache-2.0) and the llcore prototype that the present authors maintain are CPU-only architecture-evolution systems. lldarwin_v2 runs 1000-generation evolutionary loops in 5–7 seconds on a single core; llcore extracts the *verification gate* into a separate, single-purpose tool. The verifier reported here (PoC 1a) takes 5.8 ms to discharge a one-step state-norm invariant via Z3, on a Ryzen-class CPU, with no GPU. This number is the upper-bound budget we use to set the per-step time limit in §5.

### 2.5 AutoML-Zero and Weight-Agnostic Neural Networks

AutoML-Zero (Real et al. 2020) [arXiv:2003.03384] evolves machine-learning algorithms from primitive operations under a very tight compute budget. WANN (Gaier & Ha 2019) [arXiv:1906.04358] searches for *topologies* whose performance is largely independent of weight value. Both demonstrate that *architecture itself* is a legitimate search variable, but neither imposes a *verification gate* on the search trajectory. The benchmarks they use are accuracy / FLOP / parameter-count metrics, not formal safety guarantees. The category we propose is the *verified analogue*: same kind of open-ended search, but with a soundness gate that must keep up at search speed.

### 2.6 NAS-Bench-201 and NAS evaluation benchmarks

NAS-Bench-201 (Dong & Yang 2020) [arXiv:2001.00326] gives a tabular benchmark of 15 625 architectures with pre-trained accuracy on three datasets, so that NAS algorithms can be compared without retraining. NAS-Bench-101 and NAS-Bench-301 are analogous. **None of these benchmarks scores verification properties**; they all assume a final-network accuracy oracle. We use NAS-Bench-201's *architecture enumeration discipline* (a precisely specified search space; reproducible by anyone) as a design analogue when laying out the `ChangeOp` family enumeration in §4.2.

### 2.7 Open-ended evolution: POET and MODES

Wang et al. (2019, POET) [arXiv:1901.01753] coevolve environments and agents in a *minimal-criterion* coevolution loop so that the population continually pushes itself into harder territory without operator hand-tuning. Bedau et al. (1998, 1999, 2003) introduced *evolutionary activity statistics* (cumulative component activity vs a neutral shadow population) as an empirical sensor for whether an evolutionary run is genuinely innovating or merely drifting / saturating. MODES (Dolson et al. 2019) formalised these as a benchmark instrument.

Both lineages bear directly on §4.4: we use POET-style coevolutionary rounds to keep the `ChangeOp` population fresh, and a MODES sensor to *reject* submissions that exploit a saturated regime to inflate scores.

### 2.8 AURORA and behavioural diversity descriptors

Cully (2019, AURORA) [arXiv:1905.11874] introduced an autoencoder-learned behavioural descriptor space for quality-diversity search. We borrow the *behavioural-descriptor* idea, not the autoencoder, to score whether a verifier's per-step verdict stream is *behaviourally diverse* (i.e., the verifier is being challenged by qualitatively different mutations, not the same family repeated) — see §4.2.

### 2.9 Verified RL and verified controllers (concurrent work, 2024–2026)

We must also acknowledge a concurrent line of work in verified reinforcement learning and verified neural controllers, which shares the *online* aspect of our category but not the *architecture-evolution* aspect.

- **Verified RL via shielding** (Alshiekh et al. 2018; recent work in 2024–2025): a runtime monitor (the "shield") observes RL agent actions and corrects unsafe ones. This is online but does *not* change the agent's architecture; the shield wraps a fixed policy network.
- **Lyapunov-stable neural controllers** (Chang et al. 2019; Dawson et al. 2023): the controller's architecture is fixed at training time and a Lyapunov function is co-learned; safety is established once and assumed thereafter.
- **Continual learning with verified safety** (a research thread loosely surveyed in 2024–2025): the model's *weights* change over time and the verifier must keep up, but the *topology* is held constant.

Our category sits in a region none of these touch: the *topology* changes, with `ChangeOp` granularity, and the verifier must keep up *per* `ChangeOp`. Verified-RL approaches with adaptive shielding could be extended into our category by treating each new shield as an `insert_subblock`; the spec is friendly to this extension and we encourage submissions that try.

### 2.10 Hardware-aware verification (orthogonal but worth noting)

A separate line of work (PRIMA, Müller et al. 2022; recent post-2024 GPU-bounded verifiers) optimises α,β-CROWN-style verification for specific GPU architectures and floating-point precision. Our category is CPU-only by §4.3 and §8.2, so this line is out of scope, but a 2028 revision could lift the GPU restriction; we leave that to the category committee.

### 2.11 Why we explicitly *do not* claim a verification-algorithm contribution

A reader of an early draft asked whether the paper should also propose a new verification algorithm tuned for `online-arch-evo`. We deliberately do *not*. The contribution is the **category and the spec**, which are *enabling infrastructure*. Existing verifiers (α,β-CROWN, Marabou, TorchLean) have rich technique stacks that are very plausibly adaptable to the new category. The community's job, not ours, is to engineer those adaptations. Our job is to make a credible target exist.

This is intentionally analogous to the role NAS-Bench-201 played for the NAS community: NAS-Bench-201 itself proposed no new NAS algorithm; it made the search-space-and-evaluation a stable common ground so that algorithmic contributions could be compared. We aspire to the same role.

### 2.12 Summary of the gap

| Work | Network | Query | Budget | Open-ended? |
|---|---|---|---|---|
| VNN-COMP 2020–2024 | one fixed `.onnx` | one `.vnnlib` | per-query, GPU | No |
| α,β-CROWN | one fixed `.onnx` | one `.vnnlib` | per-query, GPU | No |
| Marabou Incremental | one fixed network | sequence of same-network queries | per-sequence | No |
| TorchLean | one fixed network | mechanised proof | per-property, slow | No |
| AutoML-Zero / WANN | mutating | accuracy oracle, no verifier | per-run | Yes |
| llcore PoC 1a | mutating | state-norm invariant | 5.8 ms per step, CPU | (this proposal) |
| **`online-arch-evo` (this paper)** | mutating | invariant per step | wall-clock per step | **Yes (§4.4)** |

The row in bold is the gap we are filling.

---

## 3. Problem Definition

### 3.1 Online architecture evolution verification

Let *𝒩* be a class of neural networks (concretely, the class representable as `.onnx` graphs over a fixed operator vocabulary; we discuss vocabulary extension in §4.2.2). Let *ϕ* be a *safety invariant*: a predicate over networks that is closed under the operator vocabulary's semantics (e.g., "the output range over input region *X* lies in *Y*", or "the state-norm at any time step lies in [0, 1] given input bounded by [-1, 1]").

A `ChangeOp` is a partial function *m* : *𝒩* ⇀ *𝒩* with a finite, declared rewrite pattern over the operator graph (insert, delete, re-parameterise, re-order, type-substitute). The class of admissible `ChangeOp` for a benchmark is **declared in the spec file** (§4.2).

An *online architecture evolution verification* problem instance is a tuple

> *(N*₀, *ϕ*, *(m*₁, *m*₂, …, *m*ₖ*))*

where *N*₀ ∈ *𝒩*, *ϕ* is the invariant, and (*m*₁, …, *m*ₖ) is a (possibly unbounded) sequence of `ChangeOp`. The verifier emits, **for each step *i* in order**, a verdict ∈ {`sat`, `unsat`, `timeout`, `error`, `refuse`} with the obligation that:

- `unsat` means *ϕ* is proved to hold for *Nᵢ* := *mᵢ*(*Nᵢ*₋₁), and the verifier exposes a soundness certificate the judges can spot-check.
- `sat` means a counter-example to *ϕ* on *Nᵢ* is exhibited (concrete input or symbolic region).
- `timeout` means the per-step budget was exhausted.
- `error` means the verifier crashed or refused to read the input.
- `refuse` is a *new* verdict (not in classical VNN-COMP) meaning "I cannot soundly answer this step within budget; please roll back" — discussed in §4.5.

The problem instance terminates when (a) the sequence ends, (b) the verifier returns `sat`, (c) the verifier returns `error` (treated as forfeit), or (d) the cumulative wall-clock budget is exhausted.

### 3.2 What "online" rules out

Two things are explicitly out of scope:

1. **Batch verification.** A submission may *not* read the entire `ChangeOp` stream and produce all verdicts at once. The judge feeds steps one at a time. (Implementation note: the judge does this by sending one `ChangeOp` over stdin per round; see §4.5.)
2. **Look-ahead.** A submission may *not* depend on knowing future `ChangeOp` to discharge the current step. The judge enforces this by reserving the right to terminate the sequence early.

This is precisely what makes the category *online*, and precisely what α,β-CROWN cannot currently do, because there is no I/O loop between the verifier and a mutating network process — the verifier is invoked once with a complete `.onnx`.

### 3.2.1 Why a fresh formalisation is needed

A reader may ask why we cannot frame the problem entirely in existing VNN-COMP terms by treating "verify each *Nᵢ* in turn" as a sequence of independent VNN-COMP problems. The answer is the *budget*.

Define `T_single(N, ϕ)` as the wall-clock time α,β-CROWN takes to discharge a single VNN-COMP query. For non-trivial benchmarks T_single is in the seconds-to-minutes range. A stream of length 1 000 with independent invocations therefore needs hours of compute per submission. The category designed in this way is not a benchmark anyone will actually run; it is a thought experiment.

The interesting research problem is to engineer T_step ≪ T_single by exploiting the locality of *m*ᵢ. This requires:

- A *delta-aware* I/O contract (the verifier knows what changed, not just the new network).
- An *amortisation* contract (the verifier may keep state across steps).
- A *soundness-witness* contract that is itself delta-aware (you do not re-prove what you already proved).

§§4.5 and 3.3 together realise all three contracts. Formalising the problem this way is what gives the category its research-attractor character.

### 3.2.2 Formal statement of the soundness obligation

We state the soundness obligation explicitly so that submissions, judges, and future spec authors all share the same target.

Let *V* be a submission, executing the protocol of §4.5. Let `verdict(V, N₀, ϕ, m₁, …, mᵢ)` be the verdict *V* emits at step *i*. Let `Nᵢ := mᵢ(Nᵢ₋₁)`.

**Soundness.** For all benchmark instances `(N₀, ϕ, m₁, …, mₖ)` and all *i* ∈ [1, k]:

> If `verdict(V, …, mᵢ) = unsat`, then *ϕ* holds for *Nᵢ* under the semantics declared in `metadata.json`.
>
> If `verdict(V, …, mᵢ) = sat`, then there exists a concrete input `x ∈ X` such that *ϕ*(x, *Nᵢ*) is false.

A submission caught violating soundness loses 5 points per caught step (§5.1).

**Completeness (aspirational, not required).** A submission *V* is *complete* if for every benchmark instance and every step, `verdict(V, …) ∈ {sat, unsat}` (i.e., never `timeout`, `error`, or `refuse`). Completeness is not required for category entry; it is a research north-star.

**Online property.** *V* must not consume `mᵢ₊₁` before emitting verdict for step *i*. The judge enforces this by line-buffered stdin.

**Amortisation property (allowed).** *V* may carry arbitrary internal state across steps, including learned conflicts, cached bounds, and partial Lean proofs. The judge does not observe or constrain internal state; only stdin/stdout.

### 3.3 Soundness obligations

Each `unsat` verdict must be accompanied by a soundness witness the judges can mechanically check. The accepted witness formats are:

- For SMT-based verifiers: a Z3-style proof object or a `unsat`-core listing the propagated constraints.
- For bound-propagation verifiers: the final linear lower/upper bound tensors plus the branch-and-bound tree (in α,β-CROWN's standard log format).
- For TorchLean-style mechanised verifiers: the Lean term whose type is the invariant proposition.
- For incremental verifiers: the parent verdict, the inherited conflict set, and the *refinement relation* certificate that the conflicts remain sound after the `ChangeOp` (Marabou Incremental's framework, extended to structure-changing mutations — see §6.4).

A submission that emits `unsat` without producing a witness on judge demand is downgraded to `error` for that step.

---

## 4. Benchmark Specification

The full machine-readable spec is in `vnn_comp_benchmark_spec.md` (companion file). This section gives the high-level description; the companion file is normative.

### 4.1 File formats

A benchmark instance is a directory containing:

1. `model.onnx` — the initial network *N*₀, in the existing VNN-COMP ONNX subset (ReLU / Sigmoid / Tanh / Matmul / Conv / Add / Mul / Reshape / Concat / Slice). We extend the subset in §4.2.2 to admit new operator vocabularies.
2. `invariant.vnnlib` — the invariant *ϕ*, in the existing VNN-COMP `.vnnlib` syntax (S-expression input pre- and output post-conditions). For state-update kernels we add a `(declare-time t)` form that universally quantifies the post-condition over all time steps; the syntax is given in the companion spec.
3. `changeop_seq.jsonl` — the `ChangeOp` sequence, one JSON object per line, each conforming to the schema in `vnn_comp_benchmark_spec.md` §3. Each `ChangeOp` carries an `op` ("insert" / "delete" / "reparam" / "reorder" / "type_subst"), a `target` (graph location), an `args` blob, and a `family_id` (used in §4.2).
4. `metadata.json` — declared `ChangeOp` family list, declared operator vocabulary, declared budget overrides (subject to §5 caps), and a `category` field (`fixed` | `online-arch-evo`).

Why the new `.jsonl` rather than embedding the sequence in `.vnnlib`: JSONL is line-streamable. The judge can pipe one mutation per line into the verifier process, enforcing the online discipline of §3.2 by construction.

### 4.2 `ChangeOp` taxonomy

We enumerate five **declared** `ChangeOp` families in v1 of the spec. A benchmark `metadata.json` chooses any subset.

1. **`insert_subblock`** — insert a new subblock (e.g., a new RWKV-style state-update block, a new Mamba SSM block, a new linear-attention head) after an existing layer. Increases parameter count.
2. **`delete_subblock`** — remove a subblock. Decreases parameter count.
3. **`reparam_inplace`** — replace a tensor's values within a declared norm ball (no shape change).
4. **`reorder_subblocks`** — swap two subblocks' positions. Parameter count and shape unchanged.
5. **`type_subst`** — substitute one kernel type for another within a declared compatible-type relation (e.g., RWKV ↔ linear-attention with matching input/output rank).

The compatible-type relation for `type_subst` is supplied by the benchmark in `metadata.json`. **New kernel types do not require a spec revision**: a benchmark can introduce a new kernel by adding it to `metadata.json` along with its ONNX semantic and the compatible-type entries the benchmark declares.

This is the **open-endedness mechanism for §4.4-B**: the category accepts unseen kernels as a metadata extension, not as a spec revision.

#### 4.2.1 Family soundness as a separate query type

Query C (§1.1) — "does this `ChangeOp` family preserve the invariant?" — is exposed as a separate query type, `family_soundness`, with a different I/O contract: input is *(N*₀, *ϕ*, *family_id)* and output is a single verdict ∈ {`sound`, `unsound`, `timeout`} plus a witness. We discuss this further in §4.4-A.

#### 4.2.2 Operator vocabulary extension

The VNN-COMP ONNX subset is extended in `online-arch-evo` with the operator set required by the declared kernel types. For v1, we declare:

- `RwkvTimeMix` (RWKV state-update)
- `MambaScan` (Mamba selective SSM scan)
- `LinearAttention` (linear attention as in Performer / Linformer)
- `HopfieldHead` (modern Hopfield head, Ramsauer et al. 2021)

Each is given a precise ONNX semantic in the companion spec (`vnn_comp_benchmark_spec.md` §5). A submission that does not implement a kernel may declare `unsupported(kernel)` for the relevant step; this is *not* a forfeit, but the step is excluded from scoring (and a `coverage` penalty applies — see §5.3).

### 4.3 Per-step time budget

The default per-step budget is **500 ms wall-clock on a single CPU core**, with a configurable override in `metadata.json` (capped at 5 s by the category committee). The 500 ms default is a deliberate choice:

- It is **two orders of magnitude above** the llcore PoC 1a reference (5.8 ms / step), giving submissions room to be more thorough than the reference baseline.
- It is **two orders of magnitude below** a single full α,β-CROWN GPU invocation on a non-trivial benchmark, forcing α,β-CROWN entries to invest in incrementality rather than brute force.
- It is **CPU-completable**, deliberately, to match the small-compute open-ended evolution use case (§2.4).

A submission that exceeds the per-step budget emits `timeout`.

### 4.4 Open-endedness mechanisms

The category includes **four explicit mechanisms** to keep it interesting year after year, rather than being solved by a single 2026 submission.

#### 4.4-A. Wall-clock-bounded scoring of arbitrarily long streams

The `changeop_seq.jsonl` may contain 10, 100, 1 000, or unbounded steps. The judge sets a **cumulative wall-clock budget** (default 5 minutes per benchmark instance) and feeds steps until either the sequence ends or the budget is exhausted. Scoring (§5) explicitly rewards *throughput* (steps verified per second) and *robustness to length* (no per-step degradation over the stream). A submission that is fast for 10 steps but degrades over 1 000 receives a lower score than a submission that maintains throughput.

This is what makes the category robust to *open-ended evolution* in the sense of §2.7: a submission cannot pre-allocate a fixed-size data structure to "cache the whole sequence"; it must amortise its work across an arbitrarily long stream.

#### 4.4-B. Kernel-typed `ChangeOp` interface that admits unseen kernels

As described in §4.2 and §4.2.2, new kernel families are added as `metadata.json` declarations, not spec revisions. A benchmark added in 2027 may declare a 2027-published kernel and its ONNX semantic; submissions from 2026 can still parse the spec (they will declare `unsupported` for the relevant steps and accept the coverage penalty). This decouples the *spec lifecycle* from the *kernel-research lifecycle*.

The category committee commits to a yearly *kernel-bench growth report* (analogous to NAS-Bench refreshes) so the kernel vocabulary expands rather than freezes.

#### 4.4-C. POET-style `ChangeOp` coevolutionary round

Each annual edition includes one benchmark of type `changeop_coevo` in which the *benchmark itself* generates the `ChangeOp` stream by running a minimal-criterion coevolution loop (after Brant & Stanley 2017) between two competing `ChangeOp` populations. The verifier's job is to keep up; the *benchmark's* job is to generate a stream that is non-trivial to verify but not impossible.

The coevolution loop is itself part of the benchmark artefact (a deterministic seeded run that any submission can reproduce). It is *not* an adversarial attack on the verifier — the spec is fully public and the loop is deterministic — but it produces sequences that, by construction, are not in the training/development corpus of any verifier.

This is the **POET-lite open-endedness round**: it ensures the category cannot be "solved" by memorising a fixed benchmark set, because the benchmark *is itself* the output of an open-ended coevolution loop.

#### 4.4-D. MODES integrity sensor against saturated regimes

The risk with §4.4-C is that the coevolution loop converges on a *saturated* regime — a small fixed point that a memoising verifier can exploit. To detect this, the judge runs a MODES sensor (Dolson et al. 2019, after Bedau 1998) on the `ChangeOp` stream itself:

- Bedau's *cumulative activity statistics* are computed for each `ChangeOp` family across the stream.
- A *neutral shadow* stream is generated (same length, same family marginals, no selection).
- If the *new activity* statistic *A*ₙₑw of the real stream falls below a threshold relative to the shadow, the stream is flagged as **saturated** and the benchmark is excluded from the year's scoring.

A submission that "wins" by virtue of a saturated benchmark therefore wins nothing. This protects the category's open-endedness against the standard exploitation pattern (Goodhart's law: a submission optimises for the measured behaviour rather than the intended one).

We acknowledge in §9 that the MODES threshold is itself a tunable knob that may need recalibration in future editions.

### 4.5 Submission protocol

A submission is a single executable that conforms to the following stdin/stdout protocol:

```
> READY
< OK <version> <supported_kernels...>
> INIT <path_to_model.onnx> <path_to_invariant.vnnlib>
< OK <init_time_ms>
> STEP <line_of_changeop_seq.jsonl>
< {sat|unsat|timeout|error|refuse|unsupported} <step_time_ms> [<witness_path>]
> STEP <line_of_changeop_seq.jsonl>
< {sat|unsat|...} ...
...
> END
< OK <total_time_ms> <step_count>
```

Lines beginning with `>` are sent by the judge; lines with `<` by the submission. The protocol is line-oriented and deterministic; a `refuse` (§3.1) lets the submission opt out of a step without forfeiting the run (subject to a coverage penalty in §5.3).

A submission that does not respond to `STEP` within the per-step budget receives a forced `timeout` and the judge proceeds to the next step.

---

### 4.6 Determinism and seeded reproducibility of `changeop_coevo` benchmarks

For category-internal reproducibility we require that every `changeop_coevo` benchmark be generated by a fully deterministic, seeded procedure whose source is published with the benchmark. The procedure is:

1. Initialise a population of *P* ≥ 32 candidate `ChangeOp` families with a fixed seed `s_pop`.
2. Initialise a population of *P′* ≥ 32 candidate parent networks with a fixed seed `s_net`.
3. Run a minimal-criterion coevolution loop (after Wang et al. 2019, POET) for *G* ≥ 100 generations, with the pairwise fitness defined as "the parent network survives the `ChangeOp` family without invariant violation as judged by a *reference verifier* — by spec, the llcore PoC 1a reference impl".
4. Emit the surviving (parent network, `ChangeOp` stream) pair as a benchmark instance.

Step 3 deliberately couples the reference verifier into the benchmark generation. This is a design choice with a known cost: it could bias benchmarks toward what the reference verifier finds easy. We mitigate by:

- Using *minimal-criterion* coevolution, not *fitness-maximising* coevolution: a candidate survives merely by *not* being judged as a violation, not by being judged as a high-quality verification target. The bias is therefore weaker than under a fitness-driven loop.
- Encouraging future editions to substitute alternative reference verifiers (the spec is verifier-agnostic; only the reference impl is). A 2027 edition could use a Marabou-based reference, a 2028 edition a TorchLean-based one, and benchmark generators would naturally diversify.

We acknowledge this is the most contentious design choice in the spec, and we expect the category committee to revisit it.

### 4.7 What a yearly edition looks like

A concrete yearly edition contains:

- **N₁** fixed instances generated by the reference coevolution loop with seeds revealed at edition start (10–20 instances).
- **N₂** `family_soundness` instances of varying difficulty (5–10).
- **N₃** *adversarial* instances supplied by independent third parties through a call-for-benchmarks process (5–10).
- A **public 10-instance pilot** released two months before the edition for submitters to integrate against (this is the "training set").

Submissions are evaluated on N₁ + N₂ + N₃; the pilot is for development only and does not count.

The committee publishes (a) the pilot benchmarks, (b) the reference impl source, (c) the judging harness — all under Apache-2.0. Three months after the edition closes, all benchmark instances are released so that the next edition's submitters can analyse failures.

This rhythm — pilot, edition, post-mortem release — is borrowed from the established CASP protein-folding competition and from VNN-COMP's own annual schedule. It is *not* an innovation; we are following community practice.

---

## 5. Scoring & Evaluation

### 5.1 Per-step scoring

For each step *i*, the verifier earns:

- `unsat` with valid witness → +1 (the invariant is proved)
- `sat` with valid counter-example → +1 (the invariant is correctly refuted)
- `unsat` *or* `sat` without valid witness on judge audit → 0 (correct verdict but no proof)
- `timeout` → 0
- `refuse` → 0 with no penalty (subject to coverage cap, §5.3)
- `error` → −1 (forfeit)
- `unsat`-when-`sat` or `sat`-when-`unsat` (judge-confirmed wrong verdict) → −5 (soundness violation)

The asymmetric −5 for soundness violations follows VNN-COMP's existing scoring discipline: a wrong-and-confident verdict is worse than a `timeout`.

### 5.2 Per-instance scoring

Per-instance score = sum of per-step scores + a **throughput bonus** = number-of-correct-verdicts / total-wall-clock-time, normalised to the per-step budget. The throughput bonus is capped to prevent unbounded gaming: a submission that runs at the budget cap on every step gets the same throughput bonus regardless of how fast its internal pipeline could go.

### 5.3 Coverage and `refuse`

A submission may emit `refuse` for steps it cannot soundly answer. Each `refuse` costs no points but reduces *coverage* := (steps-not-refused) / (total-steps). If coverage drops below 50 % on a benchmark instance, the instance score is multiplied by `2 × coverage` (i.e., 30 % coverage → 0.6× penalty). This avoids the degenerate strategy of `refuse`-ing every step.

### 5.4 MODES-adaptive bonus

For `changeop_coevo` benchmarks (§4.4-C), the per-step bonus is scaled by the *Bedau new-activity statistic* of the corresponding stream window. Streams that exhibit higher genuine innovation (above neutral shadow) deliver more points per correct verdict than streams in a saturated regime.

This serves two roles. First, it directly rewards verifiers that can handle genuinely novel mutation patterns. Second, it disincentivises submissions that game the throughput bonus by exploiting saturated regimes — those regimes simply contribute fewer points.

### 5.4.1 Worked numerical example

We illustrate with a hypothetical 100-step benchmark of type `changeop_coevo`.

- Submission *A*: 90 `unsat`, 5 `sat`, 3 `timeout`, 2 `refuse`. All `unsat` audit-clean. Coverage = 98 %. Wall-clock = 12 s.
- Submission *B*: 70 `unsat`, 0 `sat`, 30 `timeout`. Coverage = 100 %. Wall-clock = 60 s.
- Submission *C*: 5 `unsat`, 0 `sat`, 95 `refuse`. Coverage = 5 %. Wall-clock = 1 s.

Per-step scoring: *A* = 95 × 1 = 95, *B* = 70, *C* = 5.
Throughput: *A* = 95 / 12 = 7.9, *B* = 70 / 60 = 1.2, *C* = 5 / 1 = 5.0.
Coverage cap: *A* unaffected (≥50 %), *B* unaffected (100 %), *C* × (2 × 0.05) = × 0.10 → 0.5 final per-step (rounded).

Total before MODES: *A* = 95 + 7.9 ≈ 102.9, *B* = 70 + 1.2 ≈ 71.2, *C* ≈ 0.5 + 0.05 ≈ 0.55.

If the stream has high Bedau new-activity (say a × 1.3 multiplier), MODES bonus adds: *A* × 1.3 = 133.7, *B* × 1.3 = 92.6, *C* × 1.3 = 0.7.

Final ranking: *A* > *B* > *C*. This matches intuition: *A* solved the most queries correctly, *B* was thorough-but-slow, *C* gamed the protocol unsuccessfully.

### 5.5 Final ranking

Final ranking = sum over all benchmark instances of instance score, with the saturated-benchmark exclusion rule (§4.4-D) applied first. Ties broken on (a) lowest total wall-clock, (b) highest coverage, (c) earliest submission date.

### 5.6 What we deliberately do not score

We do **not** score:

- **Network accuracy.** This is a verification competition, not a NAS competition.
- **Mutation novelty.** The `ChangeOp` stream is supplied by the benchmark, not the submission.
- **Witness compactness.** Witnesses are checked for soundness, not size.

These are explicit non-goals.

---

## 6. Reference Implementation (llcore PoC 1a, wrapped)

### 6.1 Source artefact

The reference implementation is `scripts/poc_7a_vnn_comp_reference_impl.py` in the llcore repository (companion to this paper). It wraps the existing `llcore.verifier.invariants.verify_state_norm_invariant` and `verify_gene_safe` (released in llmesh-llcore 0.1.0a0, Apache-2.0) in the stdin/stdout protocol of §4.5.

The underlying verifier is a Z3-based symbolic-execution-of-one-update-step proof, running in 5.8 ms per query on a single CPU core (Ryzen-class, see `docs/poc/poc_1a_verdict.md` for the original measurement). The reference implementation adds:

1. ONNX parser for the RWKV-style state-update subblock (the only kernel the reference implementation supports; the spec accepts `unsupported` for the rest).
2. JSONL streamer that reads one `ChangeOp` per line, applies it to the in-memory parent network, and re-invokes the Z3 verifier on the new network.
3. Per-step witness emission (Z3 `unsat`-core).

### 6.2 Per-step pipeline

For each `STEP <changeop>`:

1. Apply `changeop` to the in-memory parent network (5 supported `op` values, see §4.2).
2. Re-build the Z3 constraint system from the new network's parameters (the constraint system is parameter-dependent because `verify_gene_safe` uses concrete `RealVal`s for `decay`, `mix`, `gate_str`).
3. Re-check the invariant `state_norm ≤ 1` over the input region `|x| ≤ 1` and the state region `|s| ≤ 1`. Return `unsat` (proof) or `sat` (counter-example) accordingly.
4. Emit per-step time (typically 1–10 ms; the long-tail dominated by Z3 setup, not check).

The reference impl does **not** carry conflict state across steps; each step is a fresh Z3 invocation. This is deliberately conservative (sound but not fast). A Marabou-Incremental-style entry could carry inherited conflicts and beat the reference impl's throughput substantially; that is the explicit invitation to the verification community.

### 6.3 Supported `ChangeOp`

Only `reparam_inplace` is fully supported in the reference impl (it rewrites the three RealVal constants and re-checks). `insert_subblock` is *partially* supported — the reference impl can verify the chain of two state-update subblocks under composition, treating the composition's norm bound as the product of the per-block bounds — and is declared as such. The other three op families (`delete_subblock`, `reorder_subblocks`, `type_subst`) are declared `unsupported`; the reference impl emits `unsupported` for those steps (§4.5).

This deliberately leaves a wide margin for competitive submissions.

### 6.4 The bridge problem to Marabou Incremental

A competitive entry that wishes to exploit Elsaleh et al.'s conflict-inheritance speed-up across `ChangeOp` faces a *bridge problem*: the inherited conflict set is sound on the parent network's ReLU phase space but the child network has, in general, *more* ReLU neurons (after `insert_subblock`) or *re-indexed* ReLU neurons (after `reorder_subblocks`). The refinement relation that justifies inheritance must be re-proved.

We give in `vnn_comp_reference_impl_spec.md` (companion file) a sketch of the proof obligation under each `ChangeOp` family. A full mechanisation is out of scope for the present paper; it is **the** central research problem the category exposes.

### 6.5 Determinism and reproducibility

The reference impl is deterministic: same `model.onnx` + `invariant.vnnlib` + `changeop_seq.jsonl` yields the same verdict stream. Z3's `set_param('smt.random_seed', …)` is pinned. The judge re-runs the submission three times per benchmark and accepts the run only if all three agree.

---

## 7. Baselines

### 7.1 α,β-CROWN (naive sequential)

The simplest baseline is α,β-CROWN run from scratch on each network *Nᵢ* in sequence. This is *sound* (α,β-CROWN's soundness is established), but its per-step time on a non-trivial network is on the order of seconds-to-minutes, well above the 500 ms per-step budget. We expect α,β-CROWN (naive) to score near-zero on `online-arch-evo` benchmarks of length > 1 — *precisely the gap we set out to expose*.

This is not a criticism of α,β-CROWN. It is a statement that **the current VNN-COMP scoring rules do not let α,β-CROWN show what it could do** on online problems, because the API is wrong for the problem. A category that lets α,β-CROWN's developers expose a streaming API would let them compete.

### 7.2 α,β-CROWN with incremental wrapper (target baseline for 2027)

A more interesting baseline is α,β-CROWN with a thin incremental wrapper that reuses linear bound tensors across `ChangeOp` whose family is `reparam_inplace`. This is, to our knowledge, *implementable today* — the bound tensors *do* survive in-place reparametrisation in a documented norm ball, modulo the Lipschitz constant of the change — but no such wrapper has been published. We invite the α,β-CROWN team to submit one in 2027.

### 7.3 Marabou Incremental (Elsaleh et al. 2026)

Out-of-the-box, Marabou Incremental handles *same-network* incrementality. Extended with the §6.4 refinement-relation proofs, it would handle a substantial subset of `online-arch-evo`. The 1.9× speed-up Elsaleh et al. report on same-network query streams gives a credible target for the structure-changing case.

### 7.4 llcore PoC 1a reference impl

As §6 describes, the reference impl gives a tight lower bound on the per-step time (5–10 ms) for state-norm invariants on small networks, but only handles a small fraction of the `ChangeOp` taxonomy. We expect a Tier-1 submission in 2027 to beat the reference impl on *coverage* while staying within an order of magnitude on per-step time.

---

## 8. Discussion

### 8.1 Why a new category, not a new benchmark within `fixed`?

A new benchmark inside the existing `fixed` category would not solve the problem, because the I/O contract — one `.onnx` per verifier invocation — is the wrong shape. The judge would have to spawn one verifier per `ChangeOp`, defeating the purpose of incrementality. The new category exists because the I/O contract is new (stdin/stdout protocol of §4.5), not because the underlying mathematics is alien to VNN-COMP.

### 8.2 Why CPU-completable?

Two reasons. First, the use case (small-compute open-ended evolution, §2.4) is precisely the case where CPU-only verification matters; GPU-rich settings often have the luxury of "verify once after training". Second, allowing GPU would let entries that scale brute-force win on existing-style sub-queries without ever having to engineer incrementality, defeating the purpose of the category. We restrict to CPU explicitly. The category committee can revisit this in 2028 once the community has seen one full edition.

### 8.3 Honesty about competitive overlap

We must be honest: a sufficiently aggressive α,β-CROWN-on-CPU entry might score well on `online-arch-evo` benchmarks whose `ChangeOp` are dominantly `reparam_inplace` and whose networks are small. We do not view this as a problem — α,β-CROWN's developers would have to engineer an incremental wrapper to do this, which is itself a research contribution we wish to attract. The category is not designed to *exclude* incumbents but to *give them new things to do*.

### 8.4 Relation to the open-endedness debate

There is a long debate in artificial-life and evolutionary computation circles about whether *true* open-endedness can be benchmarked at all (Stanley & Soros 2016; Lehman & Stanley 2011). We do not claim to resolve it. We claim only that the four mechanisms of §4.4 — wall-clock-bounded streams, kernel-typed `ChangeOp`, POET-style coevolution rounds, and a MODES integrity sensor — together provide *more* resistance to "solve and ossify" than the current VNN-COMP design. If the category were solved in one year, that would be a stronger statement about the verification community than about the category's design.

### 8.5 What does success look like?

A successful first edition would have: ≥4 distinct submissions (α,β-CROWN incremental, Marabou refinement, TorchLean sampled-step, the reference impl), ≥10 benchmark instances with declared open-endedness mechanism coverage, and *no* saturated-benchmark exclusions (§4.4-D). A successful second edition would have a measurably wider operator vocabulary in `metadata.json` and at least one submission that handles a kernel family not present in v1 of the spec.

---

## 9. Limitations

We list honest limitations, in the discipline of the llcore project's `feedback_benchmark_honest_disclosure` rule.

1. **No peer review.** This paper is an author draft. The category has not been accepted by the VNN-COMP organisers, and the spec is provisional.
2. **Reference implementation is scalar-state.** llcore PoC 1a verifies one scalar state variable per network. Extending to multi-dimensional state (the realistic case) is a known open problem and is enumerated in the llcore `COMPLETION_VERDICT.md` as a post-completion task. The reference impl is therefore a *minimal credible existence proof*, not a strong baseline.
3. **The per-step time budget is asserted, not measured.** We chose 500 ms by extrapolation from the reference impl's 5.8 ms (two orders of magnitude headroom) and from typical α,β-CROWN single-query times (two orders of magnitude *down*). The first edition will produce real per-step distributions that should inform a revision.
4. **MODES threshold is hand-set.** The threshold under §4.4-D below which a benchmark is excluded as saturated is currently a hand-set hyperparameter. Goodhart's law applies: a submission could craft a `ChangeOp` distribution to sit just above the threshold. We mark this as Future Work in §10.
5. **`changeop_coevo` benchmarks may overfit to llcore's evolution loop.** The reference coevolution loop (used to generate sample benchmarks of type `changeop_coevo`) is built on the llive lldarwin_v2 line. A submission whose author also uses llive will have an unfair familiarity. We mitigate by publishing the coevolution loop source code under Apache-2.0 and inviting any team to submit *replacement* coevolution loops for future editions.
6. **No empirical comparison against α,β-CROWN.** We have not run α,β-CROWN on the reference benchmarks, because the existing α,β-CROWN does not implement the §4.5 stdin/stdout protocol. A first-edition comparison requires the α,β-CROWN team to engineer a wrapper. We acknowledge the resulting circularity (we are proposing a category α,β-CROWN cannot enter yet, and asking the verification community to make it enterable).
7. **Goodhart on the throughput bonus.** A submission could over-emit `refuse` to keep its throughput high. We cap this with the coverage rule (§5.3), but a sufficiently clever submission may exploit corner cases of the cap. Future editions should empirically tune.
8. **GPU-gated kernels are out.** Mamba and Hopfield kernels in the operator vocabulary (§4.2.2) currently have ONNX semantics that any CPU verifier *can* implement, but the *training-time* performance of those kernels requires GPU. The category therefore favours toy-scale benchmarks; we view this as appropriate for a first edition.
9. **No formal soundness proof of the spec itself.** The spec defines the I/O protocol but does not prove that a verifier conforming to §4.5 cannot be tricked by a malicious `ChangeOp` sequence into emitting a sound-looking but actually unsound verdict. We rely on judge-side witness audit (§3.3) to catch this; a deeper formalisation is a research project of its own.

These limitations are not fatal to the proposal — they are the standard set of limitations a first-edition benchmark proposal carries — but they are the discipline of honest reporting (`feedback_benchmark_honest_disclosure`) and the conditions under which we expect peer reviewers to read this paper.

---

## 10. Future Work

In rough order of priority:

1. **Submit to VNN-COMP organisers for category review.** Concretely: open a GitHub issue at the VNN-COMP repository with this proposal and the companion specs, and ask for a Slack-channel discussion thread.
2. **Implement the α,β-CROWN incremental wrapper baseline.** This is the highest-leverage way to get the existing community to take the category seriously, because it shows the incumbent verifier *can* play.
3. **Run a 10-instance pilot.** Generate 10 benchmark instances using llive lldarwin_v2 + the §4.4-C coevolution loop, publish them under Apache-2.0, and let early submitters iterate against them before the official first edition.
4. **Mechanise the §6.4 refinement-relation proofs.** This is research-grade work and can be done in Lean / Coq / Isabelle. A mechanised proof for one `ChangeOp` family — say, `insert_subblock` over the RWKV semantic — would be a publishable result in its own right (TMLR or LICS-style venues).
5. **Calibrate the MODES threshold empirically.** Run the §4.4-D sensor against a corpus of evolutionary traces (llive runs, AutoML-Zero traces, WANN trajectories) and report the threshold below which all known "saturated" traces are correctly flagged. This is the formal antidote to §9.4.
6. **Extend to non-scalar state.** The reference impl's main limitation; extending to vector / matrix state is mechanically tractable but engineering-heavy.
7. **Workshop publication.** Submit the proposal in workshop-paper form (4–6 pages) to NeurIPS 2026's Verification × ML workshop and GECCO 2027's short-paper track. The full paper goes to TMLR. The three versions share §§1–4 and diverge in §§6–9 depth.
8. **Open-source the reference impl harness.** Already in `scripts/poc_7a_vnn_comp_reference_impl.py`; we will package it under `pip install llmesh-llcore[vnncomp]` extras for any submitter to clone.

---

## Appendix A. Related-work cross-check (RAD-grounded)

We verified that the following works are referenced and described correctly against their primary sources (RAD corpus + arXiv):

- α,β-CROWN: Wang et al. NeurIPS 2021, code at `Verified-Intelligence/alpha-beta-CROWN`. Five-time VNN-COMP winner is documented in the project README.
- Marabou Incremental: Elsaleh, Davis, Wu, Katz; arXiv:2603.12232; integrated into Marabou 2.0 (CAV 2024).
- TorchLean: George, Cruden, Zhong, Zhang, Anandkumar; arXiv:2602.22631 (2026-02-26).
- NAS-Bench-201: Dong & Yang; arXiv:2001.00326.
- AutoML-Zero: Real et al.; arXiv:2003.03384.
- WANN: Gaier & Ha; arXiv:1906.04358.
- POET: Wang et al.; arXiv:1901.01753.
- AURORA: Cully; arXiv:1905.11874.
- MODES: Dolson et al. 2019 (Bedau lineage 1998–2003).
- DNNV: Shriver, Elbaum, Dwyer; CAV 2021; arXiv:2105.12841.
- llcore PoC 1a: this repository, `docs/poc/poc_1a_verdict.md`.

All are cited in §2; none are mis-attributed.

---

## Appendix B. Worked example — a 5-step `ChangeOp` stream on the reference impl

We walk through the reference implementation's behaviour on a 5-step stream, using the same RWKV-style state-update kernel verified in llcore PoC 1a.

**Setup.** `model.onnx` is a single RWKV time-mix block with `decay = 0.5`, `mix = 0.3`, `gate_str = 0.4`. `invariant.vnnlib` declares: for all `x` with `|x| ≤ 1` and all `s` with `|s| ≤ 1`, the next state `s′` satisfies `|s′| ≤ 1`. `changeop_seq.jsonl` contains:

```
{"op":"reparam_inplace","family_id":"f1","args":{"decay":0.7}}
{"op":"reparam_inplace","family_id":"f1","args":{"mix":-0.2}}
{"op":"reparam_inplace","family_id":"f1","args":{"gate_str":1.5}}
{"op":"reparam_inplace","family_id":"f1","args":{"decay":1.0}}
{"op":"reparam_inplace","family_id":"f1","args":{"mix":1.1}}
```

**Trace.**

- Step 1. Apply `decay = 0.7`. Z3: `unsat` (8.2 ms). Verdict `unsat`.
- Step 2. Apply `mix = −0.2`. Z3: `unsat` (5.4 ms). Verdict `unsat`.
- Step 3. Apply `gate_str = 1.5`. Z3: `unsat` (6.1 ms). Verdict `unsat`.
- Step 4. Apply `decay = 1.0`. Z3: `unsat` (5.0 ms) — boundary case, state norm preserved exactly.
- Step 5. Apply `mix = 1.1`. **Out of clip range** (`mix ∈ [−1, 1]` in the kernel's declared clip box). The reference impl emits `error` with a soundness-domain message: the input violates the declared kernel constraints.

Total wall-clock: ~30 ms over 5 steps. Score under §5.1: 4 × (+1) + 1 × (−1) = +3, throughput bonus + the four `unsat` accumulated within budget. Within coverage requirement (100 % attempted).

**Why this matters.** The trace shows the category's basic behaviour: per-step verdicts arrive within budget, witnesses are emitted, and out-of-range inputs are caught early as `error` rather than mis-verified. This is the *minimum* a category-conforming submission must demonstrate.

---

## Appendix C. Three plausible attacks on the category, and our responses

A first-edition benchmark proposal should anticipate the obvious adversarial submissions. We sketch three, and our responses.

### C.1 The "always-`refuse`" submission

A submission could emit `refuse` for every step, score zero per step, and not forfeit. Response: the coverage rule (§5.3) reduces the instance score by `2 × coverage`. A 0 %-coverage submission scores 0 on every instance and ranks last. Verified.

### C.2 The "always-`unsat`" submission

A submission could emit `unsat` for every step, scoring +1 per step, and hope the judge does not audit. Response: the judge audits a random 10 % of `unsat` verdicts per benchmark (revealed at the start of the edition; not pre-announced per-instance). A submission caught with a fabricated witness loses 5 points per caught step, far above the per-step gain. Calibrated to deter expected-value-positive cheating.

### C.3 The "memoise-the-coevolution-loop" submission

The §4.4-C coevolution loop is fully public and deterministic; a submission could simply pre-compute the verdict stream for the public sample benchmarks and look it up at runtime. Response: the actual edition's `changeop_coevo` benchmarks are generated with a fresh random seed *withheld* until the edition starts, and the §4.4-D MODES sensor is applied: a benchmark seed whose resulting stream falls into the same saturated regime as the public sample is rejected. The lookup table strategy therefore covers only an unmeasurable fraction of the actual benchmark.

A residual concern is the public source code of the coevolution loop being adversarially analysed to predict typical seed outputs. We accept this risk for the first edition and plan to rotate seeded coevolution-loop variants in subsequent editions (analogous to NAS-Bench refresh).

---

## Appendix D. A 1-week implementation roadmap for new submitters

For a research group considering a first submission:

- **Day 1.** Clone llcore, install `[z3]` extras, run `scripts/poc_7a_vnn_comp_reference_impl.py` against the sample benchmark in `tests/unit/test_poc_7a_vnn_comp_reference.py`. Verify the stdin/stdout protocol works.
- **Day 2.** Read `vnn_comp_benchmark_spec.md` (companion) and confirm understanding of file formats.
- **Day 3–4.** Pick one `ChangeOp` family beyond `reparam_inplace` (we recommend `reorder_subblocks` as the easiest non-trivial). Extend an existing verifier (your group's own, or fork the reference impl) to handle it.
- **Day 5–6.** Run on the public 10-instance pilot from §10. Submit a write-up.
- **Day 7.** Iterate. Identify two more families to support. Aim for 60 % coverage in the first submission.

This roadmap is calibrated for a single researcher with prior verification background. A group submission should aim for ≥80 % coverage of the v1 family taxonomy by Day 14.

---

## Appendix E. Worked example — a `family_soundness` query

The `family_soundness` query (§4.2.1) deserves a worked example because its I/O shape differs from the per-step protocol.

**Setup.** Same RWKV-style state-update block as Appendix B. Declared family `f_decay_only`: all `reparam_inplace` ops that touch only `decay`, with `decay ∈ [0, 1]`.

**Query.** `family_soundness(N₀, ϕ, f_decay_only) ?`

**Reference impl behaviour.** The reference impl recognises this as Z3's natural quantification (the `decay` parameter is universally quantified over `[0, 1]`), and re-uses the existing `verify_state_norm_invariant` function (which already quantifies over `decay ∈ [0, 1]` in its constraint system). The Z3 check is unchanged from PoC 1a; the verdict is `sound` (the family preserves the invariant) in 5.8 ms.

**A harder family.** Declared family `f_arbitrary_reparam`: all `reparam_inplace` ops touching any of `decay, mix, gate_str` within the kernel's full clip box. The reference impl invokes `verify_state_norm_invariant` over the full clip box and returns `sound` in 5.8 ms, with the witness being the existing PoC 1a proof.

**An unsound family.** Declared family `f_arbitrary_overdrive`: all `reparam_inplace` ops with `decay ∈ [0, 2]` (out of clip). The reference impl detects the out-of-clip declaration, emits `unsound`, and supplies the counter-example `decay = 2, s = 1, mix = anything, x = anything, s′ = 2 − tanh ≥ 1 > 1` — matching PoC 1a's G3 gate (decay=2 sat).

**Why this matters.** A `family_soundness` verdict, once proved, makes the per-step verdict on members of that family *unnecessary*. A scoring revision in 2027 could let submissions amortise `family_soundness` proofs across per-step queries, accelerating the throughput bonus enormously. We leave this scoring extension to the category committee.

---

## Appendix F. Honest disclosure of the authors' position

The authors maintain llcore and stand to benefit reputationally if `online-arch-evo` is accepted. We disclose this explicitly in keeping with `feedback_benchmark_honest_disclosure`. Specifically:

1. **The reference implementation is *ours*.** A category committee could reasonably ask for a second-party reference impl to avoid the reference impl being a moving target tuned to the authors' verifier. We support this and will provide whatever interface documentation is needed for a second party to clone the implementation. The Apache-2.0 license on llcore makes this concretely possible.
2. **The 500 ms per-step budget is set with knowledge of our verifier's 5.8 ms baseline.** A category committee that chose, say, 50 ms instead would exclude many submission strategies we have not anticipated; a committee that chose 5 s would let brute-force submissions dominate. Our default is a defensible midpoint but is not the only defensible choice.
3. **We do not own any of the works in §2.** Every cited verifier (α,β-CROWN, Marabou, TorchLean) is from a different group, and we have no commercial relationship with any of them. The category we propose is designed to *invite* those groups, not to advantage llcore.
4. **The peer-review path matters more than the category itself.** If the VNN-COMP organisers reject the category but adopt a subset of the mechanisms (§4.4) into the `fixed` category, that is a positive outcome by our reckoning, not a failure.

These notes are inserted *before* peer review, not retroactively, so reviewers can hold us to them.

---

## Appendix G. Glossary of category-specific terms

- **ChangeOp** — a partial function on networks with declared rewrite pattern (§3.1).
- **Online verification** — verifier receives mutations one at a time, no look-ahead (§3.2).
- **Soundness witness** — proof object the judge can mechanically check (§3.3).
- **Refuse** — verdict meaning "I cannot soundly answer this step within budget" (§3.1, §4.5).
- **MODES sensor** — Bedau cumulative-activity statistic against a neutral shadow (§4.4-D).
- **Per-step budget** — wall-clock cap per `STEP` round, default 500 ms (§4.3).
- **Family soundness** — query type proving an entire `ChangeOp` family preserves invariant (§4.2.1).

---

*End of paper draft.*
