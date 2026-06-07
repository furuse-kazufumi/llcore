# CPU Memory & Virtual-Memory Efficiency — Incorporation Plan

> **Provenance / honest disclosure.** Produced by a 6-agent investigation Workflow (`llcore-cpu-memory-efficiency`,
> 2026-06-05). Numeric speedups below (6.85× BLAS, 4.2× batched-SVD, 3.0× process-parallel) were measured by the
> agents on **micro-benchmarks while a live L3 batch was running on the same box** (CPU-contended). They are
> directional, **not yet confirmed on the real workload** — §6 is the clean re-measurement plan to run after the
> batch frees the CPU. Do not cite a speedup as final until §6 re-confirms it. The certifier path stays float64,
> full stop (see §7). `src/` is untouched; all targets are `research/` files + the runner.

---

## 1. TL;DR — is memory the bottleneck?

**No. At the current `n=8` scale this workload is compute-bound, not memory-bound, and the single largest realized win is a runner env var, not a memory change.** The largest live array anywhere in the pipeline is the `(T,256)` float64 softmax pair (`logits`+`P`) inside `fit_logistic_readout` — ~52 MiB at `T≈16k`, trivially RAM-resident on this 15.7 GB box, and the certifier path's vertex/Jacobian payload at `n=8` is only ~0.13 MiB. The investigations measured a **6.85× wall-clock speedup purely from capping BLAS threads to 1** (`OPENBLAS_NUM_THREADS=1`), dwarfing every allocation tweak. Memory only becomes a *binding* constraint when you **scale**: float32-fitness halves the `(T,256)` buffers, the `2^n` vertex enumeration in `cert_two`/`cert_sdp` hits a memory wall around `n≈16` and is dead (1099 GB, ~9 days/gene) at `n=32`, and longer corpora grow the readout temporaries linearly. So treat memory work as **scaling insurance + allocator-churn hygiene**, not as today's hot-path fix — and never claim a speedup without re-running `run_l3.ps1`'s `Stopwatch`/`elapsed_s` to confirm it.

---

## 2. Now-safe wins (adopt / adopt-scoped, trivial–small, soundness none or fitness-only)

Apply these to `research/` files + the runner only (`src/` untouched). Each is bit-checkable against the current deterministic fitness vector (`held_out_ce` verified `ce1==ce2` to ~1e-12).

- [ ] **Cap BLAS threads in the runner** — `run_l3.ps1`. Add `$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'` next to the existing `$env:PYTHONUTF8='1'`. *Measured 6.85× on the fitness path; bit-identical output (SHA-256 stable across NT=1/2/4/8).* **soundness_risk: none** — thread count does not perturb float64 results or certifier margins. **Highest-value change in the plan.** **[APPLIED 2026-06-05 for future runs; the in-flight batch launched before this edit.]**

- [ ] **Reuse one preallocated `(T,256)` logits buffer across all GD steps** — `lm_substrate.py:fit_logistic_readout` (the `n_steps` loop, lines 230–238). Preallocate `logits` once, then per step: `np.matmul(F, R, out=logits)`, in-place max-shift, `np.exp(logits, out=logits)`, in-place `/= sum`, in-place `logits[idx, y] -= 1.0`, `grad = FT @ logits`. *Verified bit-equivalent (grad max diff 8.7e-19); per-step transient peak 26.96 MB → 0.12 MB at T=6552.* **soundness_risk: none** — fitness-only arrays; the certifier never sees them. **Scoping note:** keep the held-out logits (`_augment(S_ho) @ R`, line 309) and final `cross_entropy` float64.

- [ ] **Eliminate the redundant 2nd reservoir rollout per gene** — `exp_landscape.py:main` lines 84–85. `held_out_ce(g)` rolls the reservoir, then `empirical_contraction(g, ...)` rolls the **identical** `(T-1,n)` reservoir again. Thread the already-computed `S` out of `held_out_ce` (helper returning `(ce, S)`) and pass `S` into `empirical_contraction`. *Halves landscape reservoir rollouts (1800 → ~900).* **soundness_risk: low-fitness-only.** (Effort: medium — touches two functions.)

- [ ] **In-place reservoir step scratch** — `lm_substrate.py:reservoir_states` loop (lines 148–150). Use `out=` scratch. **soundness_risk: low-fitness-only.** **Honest:** at `n=8` this is allocator-pressure relief, *not* a FLOP win (the python loop dispatch dominates the tiny `8×8` GEMV); modest, must be profiled, do not over-claim. Grows in value with the doubled landscape rollout count.

- [ ] **`assert F.flags['C_CONTIGUOUS']` guard** — `lm_substrate.py:fit_logistic_readout` before the loop. Regression-prevention (stops a future float32/buffer refactor from silently forcing a hidden BLAS copy of the 33–134 MB `F`). **soundness_risk: none.** Fold in *with* the buffer-reuse edit.

**Do NOT in this pass:** swap the hand-rolled max-shift/log-sum-exp for `scipy.special.logsumexp` — matches to 3.5e-15 (no bug) but allocates internal temporaries with **no `out=`**, regressing the buffer-reuse goal. Keep the in-place hand-rolled path.

---

## 3. Scaling levers (for `n>8`, longer corpus, larger pools)

Inactive today (sub-MB / RAM-resident); earn their place only as the problem grows. Distinguish **delays the wall** vs **removes the wall**.

| Lever | Target | Delays or removes? | soundness_risk |
|---|---|---|---|
| **float32-scoped fitness path** | `reservoir_states`, `_augment`, `fit_logistic_readout` F@R/softmax, `cross_entropy`, `held_out_ce` (NOT `coupled_nd.py` cert_*) | Halves every fitness buffer ((T,256) 33.55→16.78 MB; scaled 134→67 MB). **Delays** ~1 doubling. | low-fitness-only — with the §3 caveat below |
| **Process parallelism** (`ProcessPoolExecutor`, NT=1/worker) | `exp_landscape.py` 900-gene loop; `exp_gated.py` 60 independent `evolve()` runs | Measured 3.0× over serial-NT=1 with 4 workers. Use `max_workers = cpu_count()//2` (4 physical, avoid HT oversubscription). Bind read-only `_emb_seq`/`ids` once per worker via pool `initializer`, not per-task args. | none |
| **Stream/chunk + batched-SVD `cert_two`** | `coupled_nd.py:_box_vertices` → generator; chunked batched `np.linalg.svd(..., compute_uv=False)` | Measured 4.2× at `n=8`, bit-identical verdicts. Restores `O(chunk·n²)` live memory + early-exit. **Delays** the `2^n` wall to ~`n=16`; does NOT remove it. | low-fitness-only (verdict-parity verified) |
| **Symmetry pruning of t-box vertices** | `coupled_nd.py:_box_vertices` via `t_min_per_coord` | When `t_lo[i]==1.0` (weakly-coupled row) that coord collapses → effective `2^k`. Free + exact, data-dependent (helps near-diagonal genes). **Delays.** | none |
| **`numpy.memmap` for corpus / gene archive** | `load_corpus`, `ByteEmbedding.table` | **Defer.** Corpus ≤16 KB, embedding 16 KB, 900 genes ≈ 0.5 MB — all fit in L2. Revisit only if a future read-only array exceeds ~1 GB AND is sequential-access. | none |
| **Vertex-free sound certifier (R-LLM-1)** — *the REAL fix* | new `cert_two_vertexfree`/`cert_sdp_vertexfree`; robust-LMI / S-procedure / interval-arithmetic over the box `[t_min,1]^n` (`J(t)` is **affine** in `t`) | **REMOVES** the `2^n` wall — cost polynomial in `n`. The only lever that makes `n=32+` feasible. Documented next stage (`PREREGISTRATION.md:48–54`), which warns a rushed robust-LMI is itself an unsoundness risk (the "R-reach trap"). | HIGH-certifier-path |

### The float32-fitness caveat (read before adopting)
The f32 reservoir deviates from f64 by ~9.5e-8 — negligible *on fitness*. But that ~1e-7 deviation is the **same order as `cert_sdp`'s `margin=1e-7`** and the strict-`<1.0` boundary in `cert_inf`/`cert_two`. It is *only* safe because `classify_region`/`cert_*` derive **purely from `(decay,W)` via `t_min_per_coord`** and never read any reservoir/fitness array (verified). The scoping boundary is **structural, not a convention**. Hard rule: float32 stops at `held_out_ce`/`fitness`; never a value that flows into a certificate.

---

## 4. Windows virtual-memory config

**Honest framing first:** for this CPU-bound numpy loop, **virtual memory cannot speed anything up.** Python working set stays at tens of MB. The only thing a VM change protects is **run completion** — the determinism / Objective-contract integrity a mid-run OOM-kill would destroy during a long marathon with `ccr` + editor open alongside `run_l3.ps1`.

**The one concrete action — fix the pagefile (VERIFIED 2026-06-05):** 15.7 GB RAM, `AutomaticManagedPagefile=False`, ~3.7 GB physical free, **23.9 GB of 37.7 GB commit limit in use** (live). The `D:` pagefile is set to `Initial=Max=1907000 MB` (~1.86 TB ≈ whole drive) while the actually-allocated `pagefile.sys` is the 22.5 GB one on `C:` — almost certainly unintended. Recommend a **fixed 24–48 GB pagefile on the fast `D:`** (1140 GB free), `C:` small. **This is a Windows system change → user decision; it does NOT speed up any gene eval — it just stops a commit-limit OOM from killing a marathon run.**

```powershell
Get-CimInstance Win32_PageFileSetting | Select-Object Name,InitialSize,MaximumSize
Get-CimInstance Win32_PageFileUsage   | Select-Object Name,CurrentUsage,PeakUsage,AllocatedBaseSize
(Get-Counter '\Memory\% Committed Bytes In Use').CounterSamples.CookedValue
Get-Process python | Select-Object Name,Id,@{n='WS_MB';e={[math]::Round($_.WorkingSet64/1MB,1)}},@{n='Commit_MB';e={[math]::Round($_.PrivateMemorySize64/1MB,1)}}
```
**Reject / no-op:** memmap'ing the KB-scale corpus; OS buffer-cache "tuning" (already automatic); paging out the hot `(T,256)` softmax arrays (rewritten every GD step → thrash); `SetProcessWorkingSetSize` caps (induce paging). Keep `run_l3.ps1` **sequential** so each phase gets full RAM + CPU.

---

## 5. Runner env settings (`run_l3.ps1`)

```powershell
$env:PYTHONUTF8 = '1'
$env:OPENBLAS_NUM_THREADS = '1'   # measured 6.85x on the fit_logistic_readout F@R path
$env:OMP_NUM_THREADS      = '1'   # numpy ships scipy-openblas64; both honored
```
**Why:** on this 4-physical / 8-logical box, default threads oversubscribe — `fit_logistic_readout`'s `F@R` (`T×9 @ 9×256`) explodes to 244 ms/step at NT=8 vs 43 ms at NT=1, matching OpenBLAS #1614 (sched_yield busy-wait + alloc-lock churn on tiny GEMMs). **Determinism preserved** — outputs SHA-256 bit-identical across NT=1/2/4/8 (small dims → single block, stable reduction order). float64 results, fitness, certifier margins untouched.

**Honest caveat:** the `cert_sdp` eigvals-256 pre-screen is *faster* with threads (178 ms NT=1 → 50 ms NT=8), but it is memoised per `(decay,W)` and only reached for genes `cert_two` doesn't certify — rarely hot. **For the cert-heavy gated run, NT=1's net effect must be measured** (readout-win vs cert-loss); for landscape (readout-dominated) NT=1 is clearly right. Process-wide NT=1 is the right default; revisit only if a profile shows `cert_sdp` dominating.

**If you parallelize:** keep `sum(jobs × NT) ≤ physical cores`. Two NT=8 jobs = 16 threads on 4 cores thrashed (had to be killed). With NT=1 per job, up to 4 concurrent jobs — or one `ProcessPoolExecutor` over genes/seeds (§3).

---

## 6. What to profile after the batch frees the CPU (measure-baseline-first)

1. **Wall-clock baseline** — run `run_l3.ps1` as-is; `Stopwatch`/`elapsed_s` already log. Record landscape (`900 12288`) + gated (`15`) totals. *(2026-06-05 actuals on the in-flight, contended run: landscape 6562.5s; gated ~1900s/seed.)*
2. **Bit-exact fitness regression vector** — dump per-gene `held_out_ce` (landscape) + per-(seed,gate) `best_fitness` (gated); re-dump after each edit; require ≤1e-12 max abs diff.
3. **Compute/memory split** — `cProfile`/`py-spy --native` on `exp_landscape.py 900 12288` + `exp_gated.py 15`, split between (a) `reservoir_states`, (b) `fit_logistic_readout`, (c) `classify_region`/`cert_*`. **Predict:** reservoir+logistic ≈ 95%+; `cert_*` cheap at `n=8`.
4. **Peak memory** — `tracemalloc` around one `fit_logistic_readout`/`held_out_ce`. **Confirm** ~52 MiB live peak; re-measure after buffer-reuse (→ ~0.12 MB transient) and float32 (→ ~½).
5. **Call-count instrumentation** — count reservoir rollouts/readout fits (landscape ≈1800→~900 after the redundant-rollout fix; gated ≥7320 pairs); count **unique** genes hitting `cert_two`/`cert_sdp` after memoise.
6. **Thread-cap re-confirm on the REAL batch** — re-run with/without NT=1; confirm the ~6.85× (was a 20-gene micro-bench) and SHA-256 fitness identity hold at full scale, **separately for landscape (predict win) and gated (cert-heavy — measure)**.
7. **BLAS-vs-interpreter check** — time `reservoir_states` vs a hoisted-`out=` variant; confirm the `n=8` loop is interpreter-bound (in-place = allocator relief, not FLOP relief); report honestly as such.

---

## 7. Explicitly reject / defer

**Reject (soundness — never touch the certifier path):**
- **float32 / memmap / page-eviction on `cert_inf`, `cert_two`, `cert_sdp`, CLARABEL, `_jac_at_t`, `_box_vertices`, `t_min_per_coord`, the `eigvals`/`eigvalsh` re-checks** (`coupled_nd.py`). The `<1.0` margin, `margin=1e-7`, and eigvalsh re-checks are load-bearing for the fail-closed certificate. float32 `σ_max` near 1.0 can flip a verdict to a false "certified" — the same error class the design refuses SCS-fallback to avoid. **soundness_risk: HIGH-certifier-path. float64 locked.**
- **`scipy.special.logsumexp`/`softmax` in the GD loop** — no `out=`, regresses buffer-reuse.
- **Paging out the hot `(T,256)` softmax arrays** — thrash; the real lever is float32/buffer-reuse (algorithmic), not VM.
- **memmap for the KB-scale corpus / embedding** — zero benefit, page-fault overhead, breaks `np.frombuffer`.

**Defer (premature until the problem scales / a profile justifies it):**
- float32-scoped fitness, process-parallelism, batched/streamed/symmetry-pruned `cert_two` — real + verified, but earn their cost only at `n>8` / larger pools.
- **Vertex-free sound certifier (R-LLM-1)** — the only thing that *removes* the `2^n` wall; soundness-path redesign that must be proven sound before measurement. Out of scope for a perf pass.
- Extending cert-memoisation to landscape `classify_region` (distinct random genes → little reuse).
- `fit_ridge_readout` one-hot — reference/pre-screen only, NOT on the fitness path (`held_out_ce` uses `fit_logistic_readout`); `exp_*` never call it.

**Bottom line:** ship §5 (BLAS NT=1) + the §2 checklist after a clean baseline; fix the §4 pagefile (user decision) to protect marathon runs. Hold §3's float32/parallelism/batched-SVD for the scaling push and the vertex-free certifier for a dedicated, soundness-first effort. Profile per §6 before claiming any speedup; the certifier stays float64, full stop.
