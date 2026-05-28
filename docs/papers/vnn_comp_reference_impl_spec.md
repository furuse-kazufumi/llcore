# Reference Implementation Spec — llcore PoC 1a wrapper

**Status**: provisional, accompanying `vnn_comp_online_arch_evolution_proposal.md` and `vnn_comp_benchmark_spec.md`.
**Audience**: implementers wishing to clone or extend the reference impl.
**License**: Apache-2.0.
**Last edit**: 2026-05-29.

This document describes the *reference* `online-arch-evo` submission, built by wrapping llcore PoC 1a (`scripts/poc_1a_z3_invariant.py` + `src/llcore/verifier/invariants.py`) in the stdin/stdout protocol of the v0.1 benchmark spec.

The reference impl is **deliberately minimal**: it supports one kernel (`RwkvTimeMix`), one `ChangeOp` family fully (`reparam_inplace`), and one query type (per-step `unsat`/`sat`). The other op families and kernels return `unsupported`.

The point of the reference impl is to give every submitter a working baseline they can clone, extend, and out-compete.

---

## 1. File locations

- Source: `scripts/poc_7a_vnn_comp_reference_impl.py`
- Tests: `tests/unit/test_poc_7a_vnn_comp_reference.py`
- Underlying verifier: `src/llcore/verifier/invariants.py` (`verify_state_norm_invariant`, `verify_gene_safe`)
- Kernel definition: `src/llcore/state_update/genes.py` (`StateUpdateGene`)

---

## 2. Capabilities

| Capability | Status | Notes |
|---|---|---|
| `READY` / `INIT` / `END` protocol | full | line-buffered stdin |
| `RwkvTimeMix` kernel | full | uses existing PoC 1a Z3 encoding |
| `MambaScan` kernel | `unsupported` | scalar SSM, future |
| `LinearAttention` kernel | `unsupported` | future |
| `HopfieldHead` kernel | `unsupported` | future |
| `reparam_inplace` op | full | rewrites RealVal, re-checks |
| `insert_subblock` op | partial | only chain-of-RwkvTimeMix supported, composition bound |
| `delete_subblock` op | `unsupported` | shape re-validation not implemented |
| `reorder_subblocks` op | `unsupported` | identity for ≤2 blocks; otherwise unsupported |
| `type_subst` op | `unsupported` | future |
| `family_soundness` query | full for `RwkvTimeMix` + `reparam_inplace` families | re-uses PoC 1a |
| Witness emission | `unsat` only (`.smt2`) | `sat` witness not implemented (PoC 1a returns counter-example for state_bound=0.3 case; not yet plumbed to `.json`) |
| MODES sensor integration | n/a | sensor lives on judge side |
| Determinism | yes | Z3 random_seed pinned to 20260529 |

This table is the **honest disclosure** of what the reference impl can and cannot do. A submission that supports more capabilities will beat the reference impl on coverage and score.

---

## 3. Per-step pipeline

For each `STEP <changeop>`:

1. **Parse** the JSON line. Validate against §3 of the spec. On parse failure, emit `error 0`.
2. **Apply** the `op` to the in-memory parent network:
   - `reparam_inplace`: update the `StateUpdateGene` record. If `args` contains a key outside `{"decay","mix","gate_str"}` or a value outside the kernel's clip box, emit `error <t>` (graceful refusal for out-of-clip).
   - `insert_subblock` with `RwkvTimeMix`: append a new `StateUpdateGene` record to the chain. Composition bound treats the chain as the product of per-block bounds (sound, conservative).
   - Other ops: emit `unsupported <t>`.
3. **Re-check** the invariant. For a single block, call `verify_gene_safe(gene, ...)`. For a chain of blocks, call `verify_gene_safe` on each and conjoin (all-safe → admit).
4. **Emit verdict**:
   - `unsat` if all `verify_gene_safe` calls return `ok=True` with `used_z3=True`.
   - `sat` if any `verify_gene_safe` returns `ok=False` with a counter-example.
   - `timeout` if Z3 returned `unknown` within the per-step budget.
5. **Write witness**:
   - For `unsat`: serialise the Z3 query to `.smt2` file in the judge's witness directory (passed via env at start). Append `(check-sat)` for re-verifiability.
   - For `sat`: serialise the counter-example to `.json`.

---

## 4. Stdin / stdout reference dialogue

```
> READY
< OK 0.1 RwkvTimeMix
> INIT /abs/path/model.onnx /abs/path/invariant.vnnlib
< OK 12
> STEP {"step":1,"op":"reparam_inplace","family_id":"f_decay_only","target":{"layer":"L0"},"args":{"decay":0.7}}
< unsat 8 /tmp/witness_step1.smt2
> STEP {"step":2,"op":"reparam_inplace","family_id":"f_mix_only","target":{"layer":"L0"},"args":{"mix":-0.2}}
< unsat 5 /tmp/witness_step2.smt2
> STEP {"step":3,"op":"reparam_inplace","family_id":"f_gate_only","target":{"layer":"L0"},"args":{"gate_str":1.5}}
< unsat 6 /tmp/witness_step3.smt2
> STEP {"step":4,"op":"reparam_inplace","family_id":"f_decay_only","target":{"layer":"L0"},"args":{"decay":1.0}}
< unsat 5 /tmp/witness_step4.smt2
> STEP {"step":5,"op":"reparam_inplace","family_id":"f_mix_only","target":{"layer":"L0"},"args":{"mix":1.1}}
< error 1
> END
< OK 37 5
```

This dialogue is the canonical example. Any implementation reproducing it exactly under the same `model.onnx` + `invariant.vnnlib` + `changeop_seq.jsonl` is a conforming reference impl.

---

## 5. Witness format (reference impl)

### 5.1 `unsat` `.smt2` example

For step 1 (`decay=0.7`, kernel `RwkvTimeMix`), the witness file is:

```smtlib
; llcore PoC 7a unsat witness, step 1
; kernel = RwkvTimeMix, params: decay=0.7, mix=0.3, gate_str=0.4
(set-logic QF_NRA)
(declare-const s Real)
(declare-const x Real)
(declare-const tanh_val Real)
(assert (and (>= s (- 1)) (<= s 1)))
(assert (and (>= x (- 1)) (<= x 1)))
(assert (and (>= tanh_val (- 1)) (<= tanh_val 1)))
(assert (<= (* tanh_val tanh_val) (* (+ (* 0.3 x) (* 0.4 s)) (+ (* 0.3 x) (* 0.4 s)))))
(assert (>= (* tanh_val (+ (* 0.3 x) (* 0.4 s))) 0))
(assert (or (> (+ (* 0.7 s) (* 0.3 tanh_val)) 1) (< (+ (* 0.7 s) (* 0.3 tanh_val)) -1)))
(check-sat)
; expected: unsat
```

A judge can re-run this with stand-alone Z3 (`z3 witness_step1.smt2`) and confirm `unsat` independently. This is the soundness audit path.

### 5.2 `sat` `.json` example (hypothetical)

```json
{
  "step": 7,
  "kernel": "RwkvTimeMix",
  "params": {"decay": 0.3, "mix": 0.9, "gate_str": 1.8},
  "input": {"x": 1.0},
  "state": {"prev": 1.0, "next_estimate": 1.05},
  "violation": "|next| = 1.05 > 1.0 = state_bound"
}
```

(The reference impl does not yet emit `sat` witnesses; this is documented as a Limitation in §2.)

---

## 6. Composition under `insert_subblock` (partial support)

If `model.onnx` already contains *k* `RwkvTimeMix` blocks in sequence and `insert_subblock` appends one more, the reference impl treats the chain as:

> `s_{out} = block_k(block_{k-1}(... block_1(x)))`

The state-norm invariant `|s| ≤ 1` is preserved iff every individual block preserves it (each is a convex combination by §5.1 of the spec). The reference impl calls `verify_gene_safe` on each new block and conjoins; this is sound but conservative (it does not exploit per-block tightness across the chain).

If a block fails `verify_gene_safe`, the verdict is `sat` and the witness is the failing block's per-block counter-example. The judge audits as usual.

---

## 7. Determinism

- Z3 random seed is set via `solver.set('random_seed', 20260529)` at every `Solver()` instantiation.
- The reference impl does no I/O other than reading the stdin / `.onnx` / `.vnnlib` and writing `.smt2` witnesses. No filesystem temp file is read across steps.
- Three judge re-runs produce identical verdict streams; the reference impl is verified for determinism in `tests/unit/test_poc_7a_vnn_comp_reference.py`.

---

## 8. Per-step timing budget

PoC 1a measured 5.8 ms per `verify_state_norm_invariant` call on a Ryzen-class CPU. The reference impl adds ~2–5 ms of JSON parsing and Z3 query setup per `STEP`. Typical step time is 7–15 ms; budget headroom is ~485 ms under the spec default.

For `insert_subblock` chains of length *k*, step time is ~7 ms × *k*. A 64-block chain therefore costs ~450 ms — at the edge of the budget. The reference impl emits `timeout` for chains > 64; a real submission should improve on this via incremental conflict reuse (the explicit research invitation of §6.4 in the proposal paper).

---

## 9. Bridge to Marabou Incremental (the §6.4 research problem)

For each `ChangeOp` family, we sketch the proof obligation a Marabou-Incremental-style entry must discharge.

### 9.1 `reparam_inplace`

If the parent verdict was `unsat` via a conflict set *C* in the parent's ReLU phase space, the child has the *same* ReLU graph. The conflict set *C* is sound on the child iff the change to network parameters does not invalidate any constraint in *C*. For a change inside a declared norm ball, the Lipschitz constant of the network gives a bound on the worst-case constraint violation; if this is below the constraint slack, *C* remains sound.

**Proof obligation**: given Lipschitz constant *L* and change magnitude δ, show that for every constraint in *C* the slack exceeds *L* δ. This is tractable; we are not aware of a published mechanisation, but the obligation is clear.

### 9.2 `reorder_subblocks`

If the two reordered blocks have shape-compatible inputs and outputs, and the reordering does not change the function computed (i.e., the blocks commute), then *C* is sound on the child *as-is*. In general the blocks do not commute, and the conflict set must be re-indexed.

**Proof obligation**: define a re-indexing map *r* such that for every constraint `(neuron_i ≥ 0)` in *C*, the re-indexed constraint `(neuron_{r(i)} ≥ 0)` is sound on the child. For commuting blocks *r* is identity; for general blocks *r* is undefined and the conflict set is discarded.

### 9.3 `insert_subblock`

The child has strictly more ReLU neurons than the parent. The parent's conflict set is sound on the child *for the subset of neurons that survive insertion*. The new neurons' phase space is unconstrained by *C*; *C* therefore does not need re-validation, only extension.

**Proof obligation**: show that no constraint in *C* references a neuron whose index has changed due to insertion. With ONNX node names as stable identifiers, this is trivial.

### 9.4 `delete_subblock`

The child has fewer ReLU neurons. Constraints in *C* referencing deleted neurons are vacuously satisfied; constraints referencing surviving neurons are sound iff the surviving neurons' inputs are unaffected by the deletion.

**Proof obligation**: identity-bypass case (the deletion replaces a block with the identity) preserves *C* on the surviving subgraph. Non-identity deletions invalidate *C*.

### 9.5 `type_subst`

The child has a different operator at the substituted node. Constraints in *C* referencing the substituted operator's ReLU neurons must be re-validated against the new operator's semantics.

**Proof obligation**: for each constraint in *C* referencing the substituted node, prove the new operator's semantics implies the constraint. Often false; type_subst is the hardest case for conflict inheritance.

A submission that mechanises any one of these (e.g., 9.1 in Lean) would be a publishable research contribution and would dominate the reference impl on `reparam_inplace`-heavy benchmarks.

---

## 10. Where to clone

```
git clone https://github.com/<TBD>/llcore.git
cd llcore
py -3.11 -m pip install -e .[z3]
py -3.11 scripts/poc_7a_vnn_comp_reference_impl.py --self-test
```

(Repository URL pending — fill in at category acceptance.)

---

## 11. Honest disclosure (reference impl)

In keeping with `feedback_benchmark_honest_disclosure`:

1. The reference impl was implemented in the **same session** as the spec was written. Both reflect a single research perspective; reviewer feedback on either is welcome and likely to improve both.
2. The reference impl supports only one kernel and one fully-supported op family. It is **not** representative of what a 2027 competitive submission should look like.
3. The witness audit checker is `z3 witness.smt2`, an off-the-shelf tool. We have not implemented a `.smt2` → `.lean` bridge for cross-checker auditing.
4. Per-step times reported in §8 are extrapolations from PoC 1a's 5.8 ms baseline plus measured overhead in a single environment; numbers in other environments may differ.
5. The `family_soundness` query under §9.5 of the spec is supported only for families using `reparam_inplace`. Other families return `unsupported`.

---

*End of reference impl spec.*
