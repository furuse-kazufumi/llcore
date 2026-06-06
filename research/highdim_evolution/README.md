# HD-1 — High-dimension, unrestricted evolution of the verified core (GPU)

> Status (2026-06-06): **PUSHED to Kaggle (user GO) — v2 running on T4.**
> <https://www.kaggle.com/code/furusekazufumi/hd1-highdim-evo>
> - v1 (08:52 JST): FAILED 0/12 — `enable_gpu` only ⇒ Kaggle assigned **P100 (sm_60)**, but Kaggle's
>   torch 2.10.0+cu128 supports sm_70+ ⇒ `cudaErrorNoKernelImageForDevice` on every run. ~5 s, no quota lost.
> - v2 (09:0x JST): re-pushed with **`"machine_shape": "NvidiaTeslaT4"`** in kernel-metadata.json
>   (`enable_gpu` is DEPRECATED in CLI 2.2.1; machine_shape ∈ {NvidiaTeslaT4, NvidiaTeslaP100, Tpu1VmV38}).
> - Auth gotcha: classic `kaggle.json` (username+key) is REJECTED for write ops by CLI 2.2.1 — the stored
>   key is a new-style token; it works via `KAGGLE_API_TOKEN` env or `~/.kaggle/access_token` (now written).
>   Also `PYTHONUTF8=1` required on cp932 consoles (em-dashes in the script break metadata read otherwise).
> - `kernels status` returns 500 for this script kernel; use `kernels output` as the completion probe
>   (it returns the latest *finished* version's output + log).

## Question (user, 2026-06-06)
"What happens to evolution at **higher dimension, without restriction**?" — a regime CPU could not reach
(the sound 2-norm/SDP certifiers enumerate `2^n` t-box vertices; infeasible past n≈16, plus speed).

## Design
`hd1_highdim_evo.py` — derived from `../verifier_navigability_gpu/bg10_gpu_lm.py` (same self-contained,
checkpointed, resumable Kaggle/Colab GPU scaffold). Sweeps **n ∈ {8,32,64(,128,256)}** × gates
**{none, inf}** × seeds, both GRAD (gradient-trained, projected) and EVO (gated random mutation on a
gradient-warm frozen base). Measures held-out CE, `empirical_rho` (contraction/echo-state proxy),
GRAD reject-rate, EVO admit-rate, soundness flag.

**Key design note:** the substrate `s=decay·s+(1-decay)·tanh(Ws+x)` is tanh-bounded ⇒ `|s|<1` ALWAYS,
so "unrestricted" never NaNs. What is at stake is the **contraction / echo-state property (`ρ(J)<1` =
fading memory / homeostasis)**, not boundedness. HD-1 asks, as n grows: (1) does unrestricted (`none`)
evolution drift to `ρ≥1` (lose the echo-state property)? (2) does it reach **lower CE by going
expansive** — i.e. is the contraction gate a help or a handicap at scale (= L2 "gate load-bearing" at
high n)? (3) the only SOUND gate that SCALES is `cert_inf` (O(n²)); 2-norm/SDP are `2^n` and are
**never called** here — how restrictive (admit-rate) / costly (CE gap) is the cheap inf gate as n grows?
This connects the cost-reduction thread (PoC-2.6: cheap coverage degrades with n) and the navigability
thread (BG10) at scale.

## RESULTS — feasibility run (v2, Kaggle T4, 113 s, 12/12 ok, 2026-06-06)

`result_hd1_feasibility.json` (n ∈ {8,32,64} × {none,inf} × 2 seeds; unigram CE 3.2512; d=64, T=64,
grad_steps=150, evo_gens=80):

|  n | gate | GRAD ce | GRAD ρ | EVO ce | EVO ρ | EVO admit | EVO sound (per seed) |
|---:|------|--------:|-------:|-------:|------:|----------:|----------------------|
|  8 | none | 2.4518 | 0.896 | 2.5886 | 0.959 | 1.000 | 1/2 unsound (ρ: 0.92, 1.00) |
|  8 | inf  | 2.4575 | 0.845 | 2.5800 | 0.821 | 0.006 | 2/2 sound |
| 32 | none | 2.4146 | 0.920 | 2.5607 | 1.013 | 1.000 | 1/2 unsound (0.95, 1.08) |
| 32 | inf  | 2.4257 | 0.861 | 2.5401 | 0.908 | 0.000 | 2/2 sound |
| 64 | none | 2.3973 | 0.943 | 2.5801 | 1.218 | 1.000 | **2/2 unsound (1.04, 1.39)** |
| 64 | inf  | 2.4046 | 0.886 | 2.5127 | 0.873 | 0.000 | 2/2 sound |

**Answers to the three questions (feasibility-level, 2 seeds — full run still pending):**
1. **YES — unrestricted evolution drifts expansive, monotonically with n.** Mean EVO ρ: 0.959 → 1.013 →
   1.218; unsound seeds 1/2 → 1/2 → **2/2 at n=64**. The echo-state property is lost by default at scale
   (consistent with a measure/volume argument: the contracting region's volume fraction shrinks with n —
   same geometry as PoC-2.6's coverage degradation).
2. **NO — going expansive does not pay; it costs, and the cost grows with n.** EVO ce (none − inf):
   +0.009 → +0.021 → **+0.067**. The gate is a HELP at scale, not a handicap (L2 "gate load-bearing"
   confirmed at high n, in the strong form: even the freeze the gate imposes beats unrestricted drift).
3. **The cheap sound gate (`cert_inf`) becomes totally restrictive at scale** — admit 0.006 → 0.000 →
   0.000 (EVO is trapped at the gradient-warm base) — **yet that freeze is CE-protective**: gated EVO has
   the best EVO ce at every n. "Infinitely restrictive but free protection."
   Honest caveat: with admit ≈ 0, "gated EVO" ≈ frozen warm base; the comparison is really
   *unrestricted-EVO endpoint vs. base*, i.e. unrestricted evolution at high n is worse than not evolving.
4. GRAD stays sound at all n (ρ 0.845–0.943, reject_rate 0.0) and beats EVO everywhere — reconfirms the
   navigability verdict (gradient avoids the trap unaided) at 8× the dimension.

## RESULTS — full run + null control (Kaggle T4, 2026-06-06) — **supersedes feasibility conclusions**

`result_hd1_full.json` (real, 40/40 ok, 1076 s) + `result_hd1_full_null.json` (shuffled-corpus null,
40/40 ok, 1144 s). Pushed as separate kernels `hd1-highdim-evo-full` / `-full-null` (parallel, repo
toggles untouched; variants patched in temp dirs). n ∈ {8,32,64,128,256} × {none,inf} × 4 seeds;
d=96, grad_steps=400, evo_gens=150; unigram CE 3.3542 (real) / 3.3021 (null).

**REAL** (mean over 4 seeds; unsound = #seeds with ρ≥1):

|  n | gate | GRAD ce | GRAD ρ | GRAD unsound | EVO ce | EVO ρ | EVO unsound |
|---:|------|--------:|-------:|---:|-------:|------:|---:|
|  8 | none | 2.18 | 1.07 | 3/4 | 2.26 | 0.90 | 0/4 |
|  8 | inf  | 2.21 | 0.80 | 0/4 | 2.27 | 0.85 | 0/4 |
| 32 | none | 2.07 | 1.20 | 4/4 | 2.18 | 0.97 | 0/4 |
| 32 | inf  | 2.16 | 0.87 | 0/4 | 2.22 | 0.86 | 0/4 |
| 64 | none | 2.01 | 1.22 | 4/4 | 2.15 | 0.98 | 2/4 |
| 64 | inf  | 2.13 | 0.89 | 0/4 | 2.19 | 0.87 | 0/4 |
|128 | none | 2.01 | 1.42 | 4/4 | 2.15 | 1.25 | 3/4 |
|128 | inf  | 2.11 | 0.91 | 0/4 | 2.17 | 0.90 | 0/4 |
|256 | none | 2.06 | 1.95 | 4/4 | 2.19 | 1.89 | 4/4 |
|256 | inf  | 2.10 | 0.92 | 0/4 | 2.15 | 0.92 | 0/4 |

**NULL** (shuffled corpus — no learnable sequential structure):

|  n | gate | GRAD ce | GRAD ρ | GRAD unsound | EVO ce | EVO ρ | EVO unsound |
|---:|------|--------:|-------:|---:|-------:|------:|---:|
|  8 | none | 3.33 | 1.06 | 3/4 | 3.33 | 1.53 | 4/4 |
| 32 | none | 3.32 | 1.58 | 4/4 | 3.33 | 1.67 | 4/4 |
| 64 | none | 3.32 | 1.89 | 4/4 | 3.33 | 1.84 | 4/4 |
|128 | none | 3.32 | 2.21 | 4/4 | 3.32 | 1.82 | 4/4 |
|256 | none | 3.32 | 2.61 | 4/4 | 3.32 | 1.78 | 4/4 |
| all| inf  | 3.32–3.33 | 0.81–0.92 | 0 | 3.32–3.33 | 0.87–0.92 | 0 |

### Conclusions (full + null; supersede the feasibility section where they conflict)

1. **Ungated GRADIENT training itself leaves the contractive region (19/20 real seeds unsound)** — ρ
   grows with n (1.07 → 1.95). The feasibility "GRAD stays sound" was a small-budget artifact
   (150 steps); at 400 steps gradient crosses ρ=1 at every n. The navigability verdict ("gradient
   avoids the trap") holds for *EA-style traps*, not for staying contractive — gradient has no reason
   to stay and doesn't.
2. **The drift is ENTROPIC, not structure-seeking — the null kills the edge-of-chaos reading.** On
   shuffled data the same (stronger: GRAD ρ → 2.61) drift occurs with ZERO CE payoff (all cells pinned
   at unigram). Crossing ρ=1 is the default geometry of unconstrained walks at high n (the contractive
   region's volume fraction vanishes — same geometry as PoC-2.6 coverage decay), not evidence that
   prediction *needs* criticality. Real data actually *anchors* the dynamics (real drift < null drift
   at every n).
3. **The sound cheap gate costs real CE at full budget: GRAD none beats inf by 0.03–0.12 at every n**
   (peak at n=64–128, shrinking at 256). "Gate ≈ free" from feasibility was budget-limited. Open
   question (stage-B): is the cost lost expressivity or optimization *friction* (gate rejects ~25 % of
   steps)? Hint for friction: the single sound none-seed (n=8, ρ 0.95) got the *best* CE in its group.
4. **EVO's expansive payoff is non-monotone and collapses at n=256**: EVO ce (none − inf) =
   −0.013 / −0.035 / −0.040 / −0.019 / **+0.042** (deepest at n=64; values recomputed exactly from
   the JSONs in the paper-integration verification pass). Mild expansiveness helps mutation up to n=128;
   at n=256 (ρ ≈ 1.6–2.1) unrestricted EVO loses to the frozen gated base. Gradient exploits the
   expansive region in a controlled way; random mutation drowns in it at scale.
5. **`cert_inf` stays totally restrictive for EVO (admit 0.000 at every n)** — gated EVO ≡ warm base.
   And EVO unsoundness on real data is cleanly monotone in n: 0/4 → 0/4 → 2/4 → 3/4 → 4/4.
6. **Harness integrity (honest-disclosure check): PASS** — null CE never dips below unigram anywhere
   (no selection-noise overfit), 80/80 runs ok, both kernels same scaffold.

Caveats: 4 seeds; CE gaps are 0.02–0.12 nats on one corpus (tinyshakespeare) and one substrate
(tanh reservoir + linear head); `empirical_rho` is a sampled estimator; feasibility↔full sign flip for
EVO-gate shows conclusions here are budget-sensitive — treat as regime map, not universals.

## Local CPU smoke (tiny, NOT conclusive)
n=8, d=32, grad_steps=8, evo_gens=8, 3000-char corpus, unigram_CE 3.132:
- `none`: GRAD ce 3.818 ρ 0.923 | **EVO ce 4.051 ρ 1.025** (drifted *expansive*, admit 1.0)
- `inf` : GRAD ce 3.821 ρ 0.890 | **EVO ce 4.064 ρ 0.810** (contracting but **admit 0.0 = trapped**)
Already previews the phenomenon (unrestricted EVO goes ρ>1; inf EVO is sound-but-trapped) — but it is a
seconds-long smoke; real signal needs the GPU feasibility/full run.

## How to run (Kaggle GPU T4, $0)
1. Push as a kernel (kaggle CLI, token set up — see memory `reference_api_keys`):
   ```
   kaggle kernels init -p D:/projects/llcore/research/highdim_evolution
   # edit kernel-metadata.json: id "furusekazufumi/hd1-highdim-evo", code_file "hd1_highdim_evo.py",
   #   language python, kernel_type script, enable_gpu true, enable_internet true (corpus download)
   kaggle kernels push -p D:/projects/llcore/research/highdim_evolution
   ```
   Toggle `RUN_MODE="feasibility"` (n≤64, cheap) first, then `"full"` (n≤256). `RUN_NULL=True` for the
   shuffled-corpus control.
2. Monitor: `kaggle kernels status furusekazufumi/hd1-highdim-evo`
3. Pull (kernel must be Public for apikey API, or use OAuth): `kaggle kernels output ... -p <dir>`

## The broader GPU program (user picked all 3 + this, 2026-06-06)
1. **HD-1** (this) — high-dim unrestricted evolution (scale-up + cost-reduction n>8 + curiosity).
2. **R-LLM stage-B** — verified core in a real gradient-trained Transformer (paper roadmap → result).
3. **③ 本丸** — is Darwinian selection / QD load-bearing on a real (small) LLM loss landscape (the
   long-gated GPU question; needs careful pre-registration; BG9 showed ③ needs high-dim behavior).

src/ untouched; research/ isolated; push deferred (llcore has no git remote).
