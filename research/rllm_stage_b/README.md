# R-LLM Stage-B — the verified core inside a REAL gradient-trained Transformer (GPU)

> Status (2026-06-06): **kernel written + pre-registered + CPU-smoke-validated (`SMOKE_OK`, 4/4);
> awaiting user GO for Kaggle push.** Pre-reg: `PREREGISTRATION_STAGE_B.md` (gates B-G1..B-G4 fixed
> before any GPU run). Kernel: `stage_b_kernel.py` (same self-contained, checkpointed, resumable
> scaffold as `../highdim_evolution/`).

## What this is

The stage the L0–L3 thread explicitly deferred (`../verified_lm_evolution/PREREGISTRATION.md §1`):
the arc verified core `s_t = decay⊙s + (1−decay)⊙tanh(W s + x)` wired into a **softmax-attention
Transformer trained end-to-end**, where **causal sliding-window attention (w_att=8, stacked receptive
field ≈ 15 ≪ T=160) makes the core the ONLY long-range information path**. Certified object = (decay, W)
only (`cert_inf`, O(n²); input bounded by `xc = tanh(U h)` ⇒ `max_input_abs=1.0` stays sound).

## Questions → conditions

| Q | question | conditions |
|---|---|---|
| B-Q1 | is verified memory load-bearing under end-to-end training? | `pure` vs `none` |
| B-Q2 | HD-1's open Q: gate cost = expressivity or friction? | `none` vs `project` vs `reject` |
| B-Q3 | does the core still drift ρ≥1 when attention shares the load? | `none` (vs HD-1 fractions) |
| B-Q4 | price of post-hoc verification ("train free, certify later") | `none` + final-core projection |

`project` = after every step, scale W by the largest γ (bisection) that certifies — constraint without
rejection; `reject` = HD-1-style revert-on-failure (cert_every=4). Cadence asymmetry disclosed.

## Smoke (local CPU, torch 2.12.0+cpu, 6 s — NOT conclusive)

n=16, T=32, 8 steps: pure 3.4430 / none 3.3276 (ρ 0.919; postproject γ=0.380 → 3.3302) /
project 3.3249 / reject 3.3249. All 4 paths run; hybrid < pure already at smoke scale.

## How to run (Kaggle T4)

Same procedure as HD-1 (`../highdim_evolution/README.md` — incl. the 4 gotchas: access_token auth,
`PYTHONUTF8=1`, **`"machine_shape": "NvidiaTeslaT4"` mandatory** (P100 breaks Kaggle torch), status-500
→ use `kernels output` as completion probe). Feasibility (n=64 × 2 seeds × 4 conds) first, then full
(n ∈ {64,256} × 4 seeds, ~30–90 min) + null (shuffled corpus) per pre-reg.

src/ untouched; research/ isolated.
