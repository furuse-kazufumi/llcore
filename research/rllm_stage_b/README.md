# R-LLM Stage-B — the verified core inside a REAL gradient-trained Transformer (GPU)

> Status (2026-06-06): **COMPLETE — feasibility 8/8 + full 32/32 + null 32/32 on Kaggle T4
> (~42 min/kernel), all four pre-registered gates resolved.** Pre-reg: `PREREGISTRATION_STAGE_B.md`
> (gates fixed before any GPU run; 3-lens adversarial review, 5 majors fixed pre-push).
> Kernels: <https://www.kaggle.com/code/furusekazufumi/rllm-stage-b> (+`-full`, `-full-null`).

## VERDICTS (full run, n ∈ {64,256} × 4 conds × 4 seeds; unigram 3.3987 real / 3.3257 null)

**REAL** (mean over 4 seeds; ρ = core empirical rho; pp = post-hoc projection of `none`'s final core):

|  n | cond | CE | ρ | cert 4/4 | interv. rate | mean γ | ce_pp |
|---:|------|------:|------:|:--:|---:|---:|------:|
| 64 | pure | 1.7919 | — | — | — | — | — |
| 64 | none | **1.7581** | 1.114 (4/4≥1) | 0/4 | — | — | 2.1366 (γ_pp≈0.06) |
| 64 | project | 1.7797 | 0.944 | 4/4 | 1.00 | 0.388 | — |
| 64 | reject | 1.7863 | 0.900 | 4/4 | 1.00 | — | — |
| 256 | pure | 1.7919 | — | — | — | — | — |
| 256 | none | **1.7196** | 1.277 (4/4≥1) | 0/4 | — | — | 2.8369 (γ_pp≈0.02) |
| 256 | project | 1.7796 | 0.962 | 4/4 | 1.00 | 0.115 | — |
| 256 | reject | 1.7777 | 0.926 | 4/4 | 1.00 | — | — |

- **B-G1 — PASS at both n, 4/4 seeds.** The verified-core channel is genuinely load-bearing inside a
  real gradient-trained Transformer: none−pure = **−0.034 (n=64), −0.072 (n=256)** — the benefit GROWS
  with core dimension. NULL sanity: the channel's edge vanishes on shuffled data (−0.001, 2/4 at n=64)
  ⇒ the real-data win is structure (learning), not extra parameters. First R-LLM-line demonstration
  that certified memory carries long-range language information end-to-end.
- **B-G2 — EXPRESSIVITY-dominated at both n (HD-1's open question CLOSED).** Δf (reject−none) =
  +0.0282 / +0.0581, both above the resolvability floor; project−none = +0.0215 / +0.0600 ⇒ dp/Δf =
  0.76 (n=64, just past the 0.75 line — borderline, disclosed) and ≥1.0 (n=256, unambiguous). The
  constraint itself is the cost; the rejection mechanism adds ~nothing. The HD-1 n=8 "friction hint"
  was anecdotal noise.
- **B-G2-null — the gate cost is STRUCTURE-DEPENDENT: Δf ≈ 0 on shuffled data** (−0.0025 / −0.0038,
  "gate cost ABSENT"). Unlike L3's EA gate-gap (which persisted on null = optimization artifact), the
  gradient-trained expressivity cost exists ONLY where real structure is being modeled. This is the
  first genuinely structure-dependent gate effect in the whole arc.
- **B-G3 — drift survives attention: 4/4 seeds ρ≥1 at both n (real).** Registered prediction
  ("attention absorbs the pressure") was half right: the drift *fraction* is unchanged (4/4, same as
  HD-1) but the *magnitude* is tamer (ρ 1.11/1.28 vs HD-1's 1.22/1.95 at the same n). Null drifts
  harder (ρ→2.19 at n=256) with zero CE payoff — entropic-drift reading re-confirmed in the
  Transformer setting.
- **B-G4 — "train free, certify later" FAILS at scale.** Post-hoc projection costs **+0.378 (n=64) /
  +1.117 (n=256)** vs training-time gating's +0.022/+0.060 — a 17–19× penalty. Mechanism: the
  unconstrained core lands so deep outside the certifiable region that projection must shrink W to
  γ≈0.06/0.02 of its trained value, destroying what was learned. **Verification must live inside the
  training loop.**
- Honest bookkeeping: project/reject final cores certified 4/4 everywhere; defensive fallbacks fired
  0 times (the strict-decay fix is doing the work); null CE never dips below unigram (no overfit
  artifact); pure CE identical across n (consistency: pure has no n-dependence) — 72/72 runs ok.

### Net framing (for the paper)

A certified memory channel still beats no memory at all — gated CE 1.778–1.786 vs pure 1.792 — but
the gate claws back **64–83 % of the channel's unconstrained benefit** (n=256: keeps −0.014 of
−0.072). The verifier tax: real, modest, expressivity-shaped, structure-dependent, and ~19× cheaper
paid during training than after. Together with HD-1 (drift is entropic; real data anchors) this gives
the homeostasis-vs-capability trade a complete first regime map on a real Transformer.

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
