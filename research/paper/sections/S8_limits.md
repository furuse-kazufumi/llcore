## 8. Limitations, reproducibility, and the verified-evolution roadmap

This section states what our results do *not* show, makes the runs reproducible, and lays out the
roadmap that follows from the limitations. We hold the red-teamed verdict's scope unchanged: the L3
"payoff" is **evolvability, not language learning** (VERDICT.md §0).

### 8.1 Limitations of scope

The verified-evolution core was demonstrated on a deliberately small substrate, and the boundaries of
that substrate bound every claim in this paper.

- **Substrate is a reservoir LM, not a gradient-trained Transformer.** The model is a reservoir/ESN
  byte-LM with a fixed embedding and a per-gene logistic readout (VERDICT.md §1, §5); the recurrent
  dynamics `(decay, W)` are evolved, not trained by backpropagation. Whether the verified-evolution
  effect transfers to a gradient-trained Transformer is an open question deferred to GPU stage B
  (VERDICT.md §5).
- **State dimension n = 8.** The sound certifiers enumerate the `2^n` vertices of the achievable-`t`
  box (SKETCH.md, "Where the cost actually is"); `cert_two` is O(2^n · n³) and `cert_sdp` folds the
  same `2^n` vertices into an LMI (SKETCH.md). The `2^n` cost is on the **state dimension n**, not on
  the `n²` weights (SKETCH.md). This is the wall that puts larger n out of reach: the enumeration hits
  a memory wall around n ≈ 16 and is effectively dead at n = 32 (1099 GB, ~9 days/gene per
  CPU_MEMORY_EFFICIENCY_PLAN.md §1). So n = 8 is not a tuning choice but a hard limit of the current
  vertex-enumerating certifier.
- **2^n proof method.** Soundness on this substrate is established by the certifier proof
  (Lemmas 1–3 + the cert_inf/two/sdp theorems, VERDICT.md §2), with the empirical 0%-expansive figure
  a from-below *consistency check*, not the proof itself (VERDICT.md §2). The proof method is tied to
  the `2^n` enumeration; a vertex-free method would need its own soundness argument (see §8.3).
- **CPU-only.** All reported runs are CPU-only. At n = 8 the workload is compute-bound, not
  memory-bound (CPU_MEMORY_EFFICIENCY_PLAN.md §1); the largest live array is the `(T,256)` float64
  softmax pair inside the readout fit, ~52 MiB at T ≈ 16k, while the certifier payload at n = 8 is only
  ~0.13 MiB (CPU_MEMORY_EFFICIENCY_PLAN.md §1). The single largest realized speedup was a runner
  environment variable (capping BLAS threads to 1, measured 6.85× on the fitness path —
  CPU_MEMORY_EFFICIENCY_PLAN.md §1, §5), and that 6.85× was measured on a 20-gene micro-bench while a
  live batch contended for the same box, so it is directional and not yet re-confirmed at full scale
  (CPU_MEMORY_EFFICIENCY_PLAN.md §5, §6). No GPU result is claimed here.

Figure: certifier cost vs state dimension n (closed-form cert_inf vs 2^n cert_two/cert_sdp, with the
n≈16 memory wall and n=32 infeasibility) (data: research/verifier_cost_reduction/SKETCH.md,
research/verified_lm_evolution/CPU_MEMORY_EFFICIENCY_PLAN.md §1).

### 8.2 Honest-disclosure box

> **What survives, stated at full strength and no stronger (VERDICT.md §0, §3c, §4).**
>
> - **L3 = evolvability, not language learning.** Under evolution, relaxing the over-conservative
>   inf-norm gate to a sound relaxation (two_norm / sdp) lets evolution reach strictly lower held-out
>   CE than inf — robustly, 10/10 gated seeds, p = 0.000977 (VERDICT.md §0, §3b). But the mechanism is
>   **navigability**: the inf region *contains* genes 0.118 nats better than unigram on the gated
>   8192 B corpus (inf best 3.4395 vs unigram 3.5571), yet inf-gated evolution collapses to unigram
>   (3.557) — so inf's pinning is an evolvability/navigability failure, not a low region ceiling
>   (VERDICT.md §3c-i). We therefore call the L3 payoff **evolvability, not learning** (VERDICT.md §0).
> - **The null does NOT tie.** The honest-null control (shuffled corpus, 10 paired seeds) was predicted
>   to make all gates tie once sequential structure is destroyed; instead the gate ordering persists
>   and both sound relaxations still beat inf in 10/10 null seeds (sign p = 0.000977 — the same
>   significance as the real run) (VERDICT.md §3c-ii). On the held-out CE (nats) scale the null gap is
>   ~107% of the real gap (sdp−inf dCE: real 0.0429 vs null 0.0459) (VERDICT.md §3c-ii). The gate-gap
>   is therefore **essentially structure-independent** — an optimization/regularization artifact of how
>   the contraction constraint interacts with the fixed readout, **not** evidence the verifier helps
>   learn real language (VERDICT.md §3c-ii). We do **not** claim a structure-dependent gate-gap
>   residual: the paired real−null difference is not significant (CE-scale mean −0.0031, 5/10 positive;
>   fitness-scale sign p ≈ 0.17) (VERDICT.md §3c-ii). The single genuinely structure-dependent signal
>   is the **unigram-crossing**: sound gates beat the no-context unigram in the real run (10/10) but no
>   gate does on the null (VERDICT.md §3c-ii).
> - **The cheap-certifier boundary has a tail.** The viable cheap vertex-free gate `inf ∪ B2` covers
>   ~87.2% of the exact `2^n` cert_two reach (1142 of 1310 admits) at poly cost (SKETCH.md PoC-2); the
>   abs-domination bound B2 = σ(|M|+R) alone recovers 77.6% with a single SVD (SKETCH.md PoC-2). The
>   remaining ~22% of cert_two's 2-norm reach (and the genes only an LMI can certify) is **missed** by
>   the cheap bound (SKETCH.md PoC-2). Whether that tail matters depends on whether those
>   hard-to-certify genes carry the navigable low-perplexity dynamics — which is unmeasured (SKETCH.md
>   PoC-2). The earlier PoC-1 pessimism ("naive vertex-free 2-norm is worse than inf") was an artifact
>   of the bad triangle-split bound B1, not of vertex-free certification per se (SKETCH.md PoC-1 / PoC-2).
>
> **Not claimed (VERDICT.md §0, §4):** a strict monotone multi-rung ladder; the two-vs-sdp ordering
> (sampling noise: median reversal, MW ≈ 0.5 — VERDICT.md §3a); a corpus-robust region-ceiling
> (the 12288 B inf-worst-ceiling ladder does not replicate on 8192 B — VERDICT.md §3c-i); a
> structure-dependent gate-gap residual; or "a better verifier unlocks real LM learning."

### 8.3 Reproducibility

All runs are deterministic and CPU-only; the certifier path is float64, full stop
(CPU_MEMORY_EFFICIENCY_PLAN.md §7).

- **Determinism.** Runs are CRN-paired (common random numbers) across gates and use fixed seeds:
  the gated experiment used 10 paired seeds (`exp_gated.py`, pop12 / gens10, 8192 B corpus,
  unigram_CE = 3.5571 — VERDICT.md §3b). Fitness is bit-reproducible: `held_out_ce` was verified
  `ce1 == ce2` to ~1e-12, and runner outputs are SHA-256 bit-identical across BLAS thread counts
  NT = 1/2/4/8 (CPU_MEMORY_EFFICIENCY_PLAN.md §2, §5). The null control's stop is **external and
  outcome-blind** — killed at a fixed seed-boundary checkpoint after 10 of 15 requested seeds, with a
  kill-safe partial JSON written per seed, so the 10/10 sign test is not subject to optional-stopping
  bias (VERDICT.md §3c-ii).
- **Runner environment.** The runner pins `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
  `PYTHONUTF8=1` (`run_l3.ps1`); determinism is preserved across thread counts because the small
  matrix dimensions produce a single block with stable reduction order
  (CPU_MEMORY_EFFICIENCY_PLAN.md §5).
- **Isolation.** All targets live under `research/`; `src/` is untouched and the work was not pushed
  at the time of the verdict (VERDICT.md header; CPU_MEMORY_EFFICIENCY_PLAN.md §1).
- **Reproduction scripts and primary result JSON.** The following table is the authoritative list of
  scripts and result artifacts referenced by the primary sources. (Several of these are referenced by
  filename only in the source documents; we list them as the named artifacts to reproduce, with the
  source that names each.)

  | What to run / read | Artifact | Reported figures | Named in |
  |---|---|---|---|
  | Substrate (reservoir byte-LM) | `lm_substrate.py` | reservoir + logistic readout; baseline = byte-unigram | VERDICT.md header, §1 |
  | Certifiers (reused, unchanged) | `../verified_evolution_sdp_gate/coupled_nd.py`; `src/llcore/verifier/backends.py` | cert_inf O(n²), cert_two O(2^n·n³), cert_sdp LMI | VERDICT.md header; SKETCH.md |
  | Pre-registration | `PREREGISTRATION.md` | gates, soundness theorems (Lemmas 1–3), L3 / L3-null gate wording | VERDICT.md §0, §2, §5 |
  | Landscape (12288 B, region ceilings) | `exp_landscape.py` | unigram_CE 5.2399; region best contracting CE inf 4.8377 / two 4.7954 / sdp 4.7525 / non_certified 4.7052; expansive % | VERDICT.md §1, §3a |
  | Gated evolution (real, 10 seeds) | `exp_gated.py` → `exp_gated_real10_results.json` | inf CE 3.5568 / two 3.5310 / sdp 3.5140 / none 3.4926; frac_a_gt_b = 1.0; p = 0.000977 | VERDICT.md §3b |
  | Same-corpus 8192 B landscape (500 genes) + null control | `run_l3b.ps1` (driver) | inf best 3.4395 (0.118 below unigram); ceilings ~equal (inf 3.4395 ≈ two 3.4294 ≈ sdp 3.4413); null gate ordering persists, 10/10 | VERDICT.md §3c-i, §3c-ii |
  | Runner / env + Stopwatch timing | `run_l3.ps1` | NT=1 6.85× (micro-bench); landscape 6562.5 s, gated ~1900 s/seed (contended) | CPU_MEMORY_EFFICIENCY_PLAN.md §5, §6, §6.1 |
  | Navigability cross-check (other side) | `../verifier_navigability_gpu/NAVIGABILITY_GPU_VERDICT.md` | inf trap is EA-specific; gradient training avoids it | VERDICT.md §3c-ii |
  | Codex pair-review (read-only) | `CODEX_PAIRREVIEW_L3.md` | 16 findings, none overturn the result | VERDICT.md §4.5 |
  | Cost-reduction PoCs (CPU, $0) | `poc_l2lite.py` → `poc_l2lite_results.json`; `poc_l2lite_v2.py` → `poc_l2lite_v2_results.json` | PoC-1: 0 violations, 60×/980×/12,520× at n=8/12/16, B1 admits 29.5%; PoC-2: B2 77.6%, inf∪B2 87.2%, 0 violations | SKETCH.md PoC-1 / PoC-2 |
  | Kaggle GPU cross-check (BG10) | `bg10_kaggle.py` | pending / external | VERDICT.md §5 |

Figure: reproduction-script dependency graph (substrate → certifiers → landscape/gated/null drivers →
result JSON → verdict) (data: research/verified_lm_evolution/VERDICT.md, research/verifier_cost_reduction/SKETCH.md).

### 8.4 The verified-evolution roadmap

The limitations in §8.1 define the roadmap. Each item is named in the primary sources as the next step,
not as a completed result.

1. **R-LLM-1: vertex-free sound certifier (the real fix for the `2^n` wall).** Replace the `2^n`
   vertex enumeration with one structured robust-LMI / interval-matrix 2-norm / μ-analysis / SOS bound
   over the `t`-box, taking cost `2^n → poly(n)` and making n = 32+ feasible (SKETCH.md "Three levers"
   L2; CPU_MEMORY_EFFICIENCY_PLAN.md §3). This is a soundness-first, design-first effort: a
   slightly-loose bound is an unsound admit, so soundness must be proven at theorem level *before*
   measuring — the "R-reach trap" (SKETCH.md status header; CPU_MEMORY_EFFICIENCY_PLAN.md §3, §7). The
   cheap-PoC progress so far is `inf ∪ B2` covering ~87.2% of cert_two at poly cost with 0 soundness
   violations (SKETCH.md PoC-2); the genuine robust-LMI/SDP rung (PoC-3) is user-gated and pending
   measurement of whether the missed ~22% tail carries navigable low-perplexity dynamics (SKETCH.md
   PoC-2).
2. **GPU stage B: a true gradient-trained Transformer.** Move from the reservoir LM to a
   gradient-trained Transformer on GPU (VERDICT.md §5, PREREGISTRATION.md §1). The navigability
   cross-check indicates the inf trap is an EA(random-mutation)-specific artifact that gradient
   training avoids, so for gradient-trained substrates the verifier should be chosen on
   soundness/coverage alone (VERDICT.md §3c-ii). Whether the verified-evolution effect even appears
   under gradient training is therefore an explicit open question, not an assumed transfer.
3. **Multimodal = sensory organs.** Multimodal extension is on the roadmap as a future direction; we
   note it here as a named next direction rather than a result (no multimodal figure is claimed in the
   primary sources).
4. **L4: cost as an internal selection pressure.** Fold a structural-cost term into fitness and let
   evolution prefer cheap-to-verify genes via a multi-objective `(maximize held-out likelihood,
   minimize structural cost)` Pareto front, where structural cost = rank(W) / sparsity / active
   state-dimension (SKETCH.md L4). The critical caveat — and exactly where llcore's own L3 result bites
   — is that there are two kinds of "cheap": good cheap is **structural simplicity** (low-rank / sparse,
   genuinely more navigable), while bad cheap is **certifier conservatism** (cert_inf is the cheapest
   certifier yet traps evolution at unigram) and **degenerate behavior** (the unigram collapse itself)
   (SKETCH.md L4). A naive "cheap = good" scalar reward would push the EA straight into the inf trap, so
   the objective must target *structural* cost via a Pareto front, never a weighted sum, and never "use
   the cheap conservative certifier" (SKETCH.md L4). llcore is equipped to study this honestly precisely
   because its soundness oracle distinguishes "good simple" (still beats unigram) from "degenerate
   simple" (learns nothing) (SKETCH.md L4). This direction is design-first and user-gated like L1–L3
   (SKETCH.md L4).

Figure: roadmap dependency — L1 low-rank W → L3 model-order reduction → L2 vertex-free certifier
(R-LLM-1), composing toward n=32+, with L4 cost-pressure as an orthogonal multi-objective layer
(data: research/verifier_cost_reduction/SKETCH.md "Three levers" + L4).
