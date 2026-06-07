# `online-arch-evo` — Benchmark Specification, v0.1

**Status**: provisional, accompanying the proposal paper `vnn_comp_online_arch_evolution_proposal.md`.
**Audience**: benchmark authors, submission authors, judges.
**License**: Apache-2.0.
**Last edit**: 2026-05-29.

This document is **normative**. The proposal paper is descriptive; if the two disagree, this file wins.

---

## 1. Scope

This spec defines the file formats, I/O protocol, scoring rule, and submission rules for the proposed VNN-COMP category `online-arch-evo` (online architecture evolution verification). The spec is versioned (this is v0.1); subsequent revisions appear as v0.2, v0.3, … under change control of the VNN-COMP category committee (if accepted).

A benchmark instance conforming to v0.1 must declare `"spec_version": "0.1"` in its `metadata.json`. A submission conforming to v0.1 must declare its supported spec versions in its `OK <version>` response (§6).

---

## 2. Benchmark instance layout

A benchmark instance is a directory `<bench_id>/` containing exactly:

```
<bench_id>/
├── model.onnx                # initial network N0
├── invariant.vnnlib          # invariant φ, possibly using (declare-time t)
├── changeop_seq.jsonl        # per-step ChangeOp stream (one JSON object per line)
├── metadata.json             # category, declared families, declared kernel vocabulary, budget overrides
└── reference_verdicts.jsonl  # (optional) reference impl per-step verdicts, for offline checking
```

`reference_verdicts.jsonl` is non-normative; it exists for submitter convenience.

---

## 3. `changeop_seq.jsonl` schema

One JSON object per line. Each line conforms to:

```json
{
  "step": <int, >=1, monotone increasing>,
  "op": "insert_subblock" | "delete_subblock" | "reparam_inplace" | "reorder_subblocks" | "type_subst",
  "family_id": <string, declared in metadata.json>,
  "target": <op-specific dict>,
  "args": <op-specific dict>
}
```

### 3.1 `insert_subblock`

```json
{
  "step": 1,
  "op": "insert_subblock",
  "family_id": "f_rwkv_insert",
  "target": {"after_layer": "<onnx_node_name>"},
  "args": {
    "kernel": "RwkvTimeMix" | "MambaScan" | "LinearAttention" | "HopfieldHead" | "<declared>",
    "init_params": {"decay": 0.5, "mix": 0.3, "gate_str": 0.4}
  }
}
```

The newly inserted block's outputs feed the next consumer of the targeted layer's outputs.

### 3.2 `delete_subblock`

```json
{
  "step": 2,
  "op": "delete_subblock",
  "family_id": "f_delete",
  "target": {"layer": "<onnx_node_name>"},
  "args": {}
}
```

The deleted block's inputs are short-circuited to its consumers (identity bypass). A delete that breaks shape compatibility is rejected as a malformed benchmark.

### 3.3 `reparam_inplace`

```json
{
  "step": 3,
  "op": "reparam_inplace",
  "family_id": "f_decay_only",
  "target": {"layer": "<onnx_node_name>"},
  "args": {"decay": 0.7}
}
```

`args` is a partial parameter map; unspecified parameters retain their current value.

### 3.4 `reorder_subblocks`

```json
{
  "step": 4,
  "op": "reorder_subblocks",
  "family_id": "f_swap",
  "target": {"layer_a": "<name>", "layer_b": "<name>"},
  "args": {}
}
```

Both layers must have shape-compatible inputs and outputs; verifier must re-validate.

### 3.5 `type_subst`

```json
{
  "step": 5,
  "op": "type_subst",
  "family_id": "f_rwkv_to_linatt",
  "target": {"layer": "<name>"},
  "args": {
    "new_kernel": "LinearAttention",
    "params_mapping": {"decay": "alpha", "mix": "beta"}
  }
}
```

The substitution is admissible iff the (`old_kernel`, `new_kernel`) pair is in `metadata.json.type_subst_compat`.

### 3.6 Line ordering and validation

- Lines must be in monotone step order; gaps are forbidden.
- A line whose `op` is not in the declared family vocabulary is a malformed benchmark.
- Lines may include arbitrary additional fields beginning with `_`; submissions must ignore them.

---

## 4. `metadata.json` schema

```json
{
  "spec_version": "0.1",
  "category": "online-arch-evo",
  "bench_id": "<string>",
  "declared_kernels": ["RwkvTimeMix", "MambaScan", ...],
  "declared_families": [
    {"family_id": "f_decay_only", "ops": ["reparam_inplace"], "description": "..."},
    ...
  ],
  "type_subst_compat": [
    ["RwkvTimeMix", "LinearAttention"],
    ...
  ],
  "per_step_budget_ms": 500,
  "total_wallclock_budget_s": 300,
  "audit_rate": 0.10,
  "judge_seed": "<int, withheld until edition start>",
  "modes_sensor": {"threshold_ratio": 0.20, "window": 50}
}
```

Fields:

- `per_step_budget_ms`: per-step wall-clock cap (default 500, max 5000).
- `total_wallclock_budget_s`: cumulative cap (default 300).
- `audit_rate`: fraction of `unsat` verdicts the judge audits (default 0.10).
- `judge_seed`: random seed for audit-step selection; published at edition start.
- `modes_sensor.threshold_ratio`: A_new / A_shadow below which stream is flagged saturated.
- `modes_sensor.window`: window length (steps) over which MODES statistics are accumulated.

---

## 5. Operator vocabulary

The VNN-COMP ONNX subset is extended with the following ops. Each is defined by a precise ONNX semantic. v0.1 specifies four kernels; future versions may add more by appending to this section (and listing the addition in `declared_kernels` of any benchmark using them).

### 5.1 `RwkvTimeMix`

**Inputs**: `x[t]`, `state[t]`, parameters `decay`, `mix`, `gate_str`.
**Output**: `state[t+1] = decay * state[t] + (1 - decay) * tanh(mix * x[t] + gate_str * state[t])`.
**Clip box**: `decay ∈ [0, 1]`, `mix ∈ [-1, 1]`, `gate_str ∈ [-2, 2]`.
**Invariant**: `|state[t]| ≤ 1` for all `t`, given `|x[t]| ≤ 1`.

(This is the kernel verified by llcore PoC 1a; it serves as the lingua franca for the reference impl.)

### 5.2 `MambaScan` (provisional)

**Inputs**: `x[t]`, `h[t]`, parameters `A`, `B`, `C`, `Δt`.
**Output**: selective state-space scan as in Gu & Dao 2024.
**Clip box**: spectral radius of `A` ≤ 1; entries of `B`, `C` ≤ 1; `Δt ∈ (0, 1]`.
**Invariant**: `|h[t]|₂ ≤ K` for some K depending on `B`, `C`, declared per-instance.

(Provisional; the precise ONNX semantic for `MambaScan` is left to the kernel's authors to publish. Until then, benchmarks using `MambaScan` must include a `kernel_semantics.py` reference implementation alongside `model.onnx`.)

### 5.3 `LinearAttention`

**Inputs**: query `Q`, key `K`, value `V`.
**Output**: `softmax(QK^T / √d) V` linearised as `Q (K^T V) / (Q · K_sum)` per Performer.
**Clip box**: ‖Q‖, ‖K‖, ‖V‖ ≤ 1 entrywise.
**Invariant**: output norm ≤ √d.

### 5.4 `HopfieldHead`

**Inputs**: query `Q`, memory `M`, temperature `β`.
**Output**: `softmax(β · M Q) M` per Ramsauer et al. 2021.
**Clip box**: ‖Q‖, ‖M‖ ≤ 1, β ∈ (0, 100].
**Invariant**: output is a convex combination of memory rows; output norm ≤ max-row-norm of `M`.

---

## 6. Submission protocol (stdin/stdout)

A submission is a single executable. The judge starts it, communicates via stdin/stdout, and tears it down at end-of-instance.

**6.1 Handshake**

```
> READY
< OK <version> <supported_kernels_csv>
```

`version` is the spec version the submission supports (e.g., `0.1`). `supported_kernels_csv` is a comma-separated list of kernels for which the submission can emit `unsat`/`sat`; for unsupported kernels the submission emits `unsupported`.

**6.2 Init**

```
> INIT <abs_path_to_model.onnx> <abs_path_to_invariant.vnnlib>
< OK <init_time_ms>
```

`init_time_ms` is the submission's self-reported init wall-clock. The judge also records its own measurement and uses the maximum.

**6.3 Step**

```
> STEP <one_line_of_changeop_seq.jsonl>
< <verdict> <step_time_ms> [<witness_path>]
```

`<verdict>` ∈ {`sat`, `unsat`, `timeout`, `error`, `refuse`, `unsupported`}.

`<witness_path>` is required for `sat` and `unsat`; the witness file must exist when the judge reads it. Witness format depends on verdict (§6.5).

**6.4 End**

```
> END
< OK <total_time_ms> <step_count>
```

After `END` the judge gives the submission 1 s to exit cleanly; SIGKILL after.

**6.5 Witness formats**

For `unsat`:

- SMT-style: a `.smt2` file with the `unsat`-core constraints.
- Bound-propagation style: a `.npy` file with the final lower/upper bound tensors and a `.json` file with the branch-and-bound tree.
- Mechanised style: a `.lean` file with the theorem statement and a proof term that type-checks under a published Lean library version.

For `sat`:

- A `.json` file with the concrete counter-example: `{"input": [...], "state": [...], "expected_invariant": "...", "violation": "..."}`.

The judge verifies witnesses for the audited subset (`audit_rate` × `unsat`-count) by re-running an independent witness-checker.

---

## 7. Judge harness

The judge (the executable that runs benchmarks) implements:

1. **Process launch**: starts the submission with declared CPU pinning (1 core), declared memory cap (default 4 GB), declared timeout per step.
2. **stdin pipe**: line-buffered, never sends step *i+1* before reading verdict for step *i*.
3. **Stop conditions**: any of (a) `END` line, (b) `error` verdict, (c) `total_wallclock_budget_s` exceeded, (d) submission exits.
4. **Audit**: after the run, samples `audit_rate × (unsat-count + sat-count)` verdicts uniformly at random using `judge_seed`, runs an independent witness-checker on each, and records soundness violations.
5. **MODES sensor**: computes Bedau new-activity over the realised stream window (length `modes_sensor.window`), compares to a seeded neutral shadow, and emits a `saturated: bool` flag for the instance.

The judge is open-source under Apache-2.0; the reference implementation will be released at category acceptance.

---

## 8. Submission rules

1. **No network access** during a run. The judge runs submissions in a network-isolated sandbox.
2. **CPU only.** The submission's process is started under cgroup CPU pinning; GPU access is blocked at OS level.
3. **Deterministic** within a run. The judge re-runs the submission three times per instance and rejects the run if verdicts differ across the three.
4. **No look-ahead.** Enforced by the stdin pipeline (§7.2).
5. **No side channels.** The submission may not read environment variables, network state, or filesystem outside the explicitly passed paths. The judge sandbox prevents this; we list it as a rule for clarity.
6. **No pre-computed verdict tables.** Forbidden in principle; in practice we do not police this because the §4.4-C `changeop_coevo` benchmarks use withheld seeds that make pre-computation infeasible. Submissions caught with hard-coded verdicts on the public pilot benchmarks are not disqualified; submissions caught on the real edition benchmarks are.

---

## 9. Reporting and per-category leaderboard

After the edition, the judge publishes per-instance verdict tables, witness audit results, MODES flags, and the final ranking. The format mirrors VNN-COMP's existing per-edition publication (one CSV per category in the `vnncomp<year>_results` GitHub repo).

Per-submission per-instance row:

```
submission_id, bench_id, n_steps, n_unsat, n_sat, n_timeout, n_refuse, n_error, n_unsupported, coverage, wall_clock_s, score_raw, audit_violations, modes_saturated, score_final
```

---

## 10. Versioning and change control

Spec revisions are at semver `MAJOR.MINOR`:

- `MAJOR` bumps are breaking (e.g., changed `changeop_seq.jsonl` schema): benchmarks and submissions must update in lockstep.
- `MINOR` bumps are additive (e.g., new declared kernels): older submissions still parse benchmarks but emit `unsupported` for new kernels.

The category committee is expected to bump `MAJOR` no more than once per three editions, to give submitters stability.

---

## Appendix A. End-to-end example (toy)

A minimal benchmark instance demonstrating the v0.1 spec:

```
bench_toy_rwkv_5step/
├── model.onnx                # single RwkvTimeMix block, decay=0.5, mix=0.3, gate_str=0.4
├── invariant.vnnlib          # ∀x∈[-1,1] ∀s∈[-1,1]. |s'| ≤ 1
├── changeop_seq.jsonl        # see below
├── metadata.json             # see below
└── reference_verdicts.jsonl  # ["unsat","unsat","unsat","unsat","error"]
```

**`changeop_seq.jsonl`**:

```
{"step":1,"op":"reparam_inplace","family_id":"f_decay_only","target":{"layer":"L0"},"args":{"decay":0.7}}
{"step":2,"op":"reparam_inplace","family_id":"f_mix_only","target":{"layer":"L0"},"args":{"mix":-0.2}}
{"step":3,"op":"reparam_inplace","family_id":"f_gate_only","target":{"layer":"L0"},"args":{"gate_str":1.5}}
{"step":4,"op":"reparam_inplace","family_id":"f_decay_only","target":{"layer":"L0"},"args":{"decay":1.0}}
{"step":5,"op":"reparam_inplace","family_id":"f_mix_only","target":{"layer":"L0"},"args":{"mix":1.1}}
```

**`metadata.json`** (excerpt):

```json
{
  "spec_version": "0.1",
  "category": "online-arch-evo",
  "bench_id": "bench_toy_rwkv_5step",
  "declared_kernels": ["RwkvTimeMix"],
  "declared_families": [
    {"family_id": "f_decay_only", "ops": ["reparam_inplace"], "description": "vary decay only, in [0,1]"},
    {"family_id": "f_mix_only", "ops": ["reparam_inplace"], "description": "vary mix only, in [-1,1]"},
    {"family_id": "f_gate_only", "ops": ["reparam_inplace"], "description": "vary gate_str only, in [-2,2]"}
  ],
  "type_subst_compat": [],
  "per_step_budget_ms": 500,
  "total_wallclock_budget_s": 60,
  "audit_rate": 0.20,
  "judge_seed": 20260529,
  "modes_sensor": {"threshold_ratio": 0.20, "window": 5}
}
```

**Expected reference impl trace** (under the llcore PoC 7a wrapper, deterministic):

```
> READY
< OK 0.1 RwkvTimeMix
> INIT /path/model.onnx /path/invariant.vnnlib
< OK 12
> STEP {"step":1,"op":"reparam_inplace","family_id":"f_decay_only","target":{"layer":"L0"},"args":{"decay":0.7}}
< unsat 8 /tmp/w1.smt2
> STEP {"step":2,...}
< unsat 5 /tmp/w2.smt2
> STEP {"step":3,...}
< unsat 6 /tmp/w3.smt2
> STEP {"step":4,...}
< unsat 5 /tmp/w4.smt2
> STEP {"step":5,...}
< error 1
> END
< OK 37 5
```

Step 5 returns `error` because `mix = 1.1` is outside the declared clip box.

---

## Appendix B. Witness format examples

### B.1 SMT-style `unsat` witness for the toy `RwkvTimeMix` invariant

```smtlib
; witness for step 1: decay=0.7, mix=0.3, gate_str=0.4
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

### B.2 JSON `sat` witness (hypothetical, for a different benchmark)

```json
{
  "input": {"x": 1.0, "s": 1.0},
  "state": {"prev": 1.0, "next": 1.3},
  "expected_invariant": "|next| <= 1.0",
  "violation": "1.3 > 1.0",
  "kernel": "RwkvTimeMix",
  "params": {"decay": 0.5, "mix": 0.8, "gate_str": 1.6}
}
```

---

## Appendix C. Reserved field names

The following keys are reserved for future spec extensions and should not be used in third-party benchmark `metadata.json` files: `_internal`, `gpu_hint`, `parallel_steps`, `quantum_hint`. Future spec revisions may give these meaning; current submissions must ignore them.

---

*End of spec v0.1.*
