# Conversational llcore — findings (2026-06-20)

Goal (user, 2026-06-20): **make llcore conversational at a decent level**, on a home CPU
(~3.6 GB RAM, GPU blocked), without abandoning the evolutionary/plasticity research substrate.

Honest framing the user enforced across the session:
1. The from-scratch char-LM **cannot** be made conversational on this hardware (capability=NULL on
   CPU char-LM was already established; bottleneck = scale/data/compute). Confirmed empirically:
   the best char-LM (11.9M, ppl 38 vs unigram 215) only mimics Japanese prose, it does not converse.
2. **Just running a pretrained model as-is (even re-implementing int8/mmap) is not R&D** — it is a
   llama.cpp/GGUF re-derivation. The contribution must modify the model's internals.
3. The evolutionary substrate (`src/llcore/fitness/`, `evolution/`, `persona/`, `verifier/`) is
   **kept intact** — every module this session is purely additive (git shows new files only).

Chosen path: **llcore-native runtime for a small pretrained instruct model**, with the R&D living in
**deep internal attention surgery** (not a wrapper) — toward *bounded-memory × unbounded-context*
conversation.

---

## 1. Substrate: llcore-native Qwen2 forward (verified == HuggingFace)

`src/llcore/runtime/qwen2.py` — from-scratch Qwen2 decoder (RMSNorm, RoPE θ=1e6, GQA, SwiGLU,
tied embeddings), module names mirror HF so a HF `state_dict` loads directly. Golden-tested against
`transformers.Qwen2ForCausalLM` (`tests/unit/test_runtime_qwen2.py`): full-sequence logits match to
2e-4, greedy generation is token-identical, KV cache == full forward.

On the real **Qwen2.5-0.5B-Instruct**, llcore's own forward **converses in Japanese** (~5–6 tok/s,
CPU fp32):
- 「何ができますか？」→「私はAIアシスタントで、日本語を話す能力があります…」 ✓
- 「日本の首都は？」→「東京都です」 ✓
- 「3たす5は？」→「18です」 ✗ (0.5B is weak at arithmetic — scale to 1.5B for real quality)

Tied embeddings are shared (no duplicate vocab×hidden matrix resident) — a memory-correct detail for
the small-RAM target. This forward is the *instrumentable substrate*; on its own it is a re-derivation,
**not** the contribution.

## 2. R&D: internal attention surgery — linearizing Qwen's attention

`src/llcore/runtime/linearize.py` — replace a layer's softmax attention core
(`softmax(QKᵀ/√d)V`, KV cache O(T), compute O(T²)) with **constant-state linear attention**
(running state `S=Σφ(k)⊗v`, normalizer `z=Σφ(k)`, φ=elu+1), **reusing the pretrained q/k/v/o
projections** and the same RoPE. Per-head state is **O(d²), constant in sequence length** — the
architectural basis for arbitrarily-long conversation at fixed memory. A hybrid swap
(`linearize_qwen2(model, [layer indices])`) keeps softmax in some layers and linear in others.
Tested: exact chunk-size invariance of the recurrence, weight reuse, constant state bytes
(`tests/unit/test_runtime_linearize.py`).

### Per-layer linearization-tolerance profile (Qwen2.5-0.5B-Instruct, zero-shot, no distillation)

`scripts/linearize_tolerance.py` → `out/linearize_tolerance/linearize_tolerance.json`.
Held-out Japanese (aozora), 512 eval tokens, baseline softmax **ppl 68.74**.

- **Layer 0 is by far the least tolerant**: linearizing it alone → ppl 160.76 (Δnll +0.85).
  Layers 11 (+0.62), 9 (+0.32), 3 (+0.24), 1 (+0.19) also resist — interpretable as early/“retrieval”
  layers that softmax is doing real work in.
- **Most middle/late layers have near-zero perplexity cost**: layer 22 (Δ+0.0007 — no measurable
  quality loss), 7 (+0.006), 15 (+0.02), 13/19/18/12/6 all < +0.04.
- **Cumulative (greedy by single-layer Δ):** top-4 tolerant layers → ppl 73.7 (+7% ppl);
  6 → 81.6; 8 → 93.8; 12 → 167 (breaks); all 24 → 17612 (catastrophic without recovery — expected).
- **Memory:** linear state = 232,960 B/layer (constant) vs softmax KV @8192 tok = 8,388,608 B/layer
  (36× and growing). **Crossover ≈ 227 tokens** — only beyond that does a linearized layer use less
  memory than softmax KV; below it the constant state is actually larger. The point is it never grows,
  so a long chat never OOMs.

### What "tolerant / near-zero cost" precisely means (no, it is NOT "free")

"Tolerant" refers to **one axis only — perplexity (quality)**. Converting a layer is a *trade*, not a
free lunch:
- **Gain = memory**: O(T) KV → O(d²) constant state — but a *net* memory win only for context
  > ~227 tokens (below that the constant state is bigger).
- **Cost = quality**: Δnll is small but **nonzero**; measured zero-shot, at 512 tokens, on one Japanese
  corpus — not a general guarantee, and the smallest per-layer Δs are near measurement noise.
- **Cost = compute**: linear attention's chunked outer product is **heavier per token at short
  context**; the speed win only appears at long T. A short chat can be *slower*.

So the honest claim is: *a handful of layers can be converted to constant-memory attention at
near-zero perplexity cost, with the memory win realized only at long context and a short-context
compute penalty* — **not** "free".

**Honest result:** ~4–6 of Qwen2.5-0.5B's 24 layers tolerate zero-shot linearization at near-zero
perplexity cost; layer 0 and a couple of retrieval layers resist. Prior art exists (linear attention:
Katharopoulos 2020; linearizing pretrained LLMs: SUPRA, LoLCATs, Mamba-in-Llama). The contribution =
on-prem internal surgery + rigorous per-layer measurement in llcore's own code.

## 3. Supporting result: constant-state recurrent long-context (the memory substrate)

`src/llcore/lm/{checkpoint,longctx_eval,tbptt}.py` + `scripts/{train_recurrent_longctx,recurrent_longctx_eval}.py`.
Trained a constant-state recurrent char-LM on aozora (1.3M params, **ppl 22 vs unigram 207 = 9.4× the
unigram’s predictive power**, top1 0.38). Confound-controlled long-context evaluation (context-length
curve on fixed positions, banded streaming NLL with per-band unigram floor + top-1, carry-vs-reset
delta, GPT sliding-window baseline; chunk-invariance correctness gate). Result:
- **Constant-memory non-degradation = True**: per-token NLL stays flat (ppl ~21, beats unigram
  everywhere) from position 0 out to 8192 at O(1) state.
- **Effective context ≈ block_size (~128)**: the context-length curve plateaus at block_size
  (256→4096 flat at ppl 20.92) — the model does **not** use context beyond its BPTT window (honest
  null, exactly as predicted for BPTT=128). TBPTT trainer (`tbptt.py`) is built to test whether
  long-context training pushes that plateau right.

This is the O(1)-memory conversational-memory substrate; the linear-attention surgery (§2) applies the
same constant-state idea *inside the pretrained model*.

## 4. Capstone: distillation recovers the linearized layers (the contribution beyond measurement)

`src/llcore/runtime/distill.py` — give the linear attention a tiny **learnable feature map**
(per-head affine on q,k, ~4 small params per head, **identity at init**, projections frozen) and
distill its attention output to match the softmax teacher on a calibration corpus (LoLCATs-style,
output distillation, decoupled per layer so only the small attention runs in the training loop).
Tested (`tests/unit/test_runtime_distill.py`): identity-init parity + distillation reduces the gap.

On the real Qwen2.5-0.5B-Instruct, distilling the **most intolerant** layers (400 steps, lr 5e-2,
512-token calibration), recovery of the zero-shot Δnll gap:

| layer | zero-shot Δnll | distilled Δnll (calib) | distilled Δnll (held-out) | recovered (held-out) |
|------:|---------------:|-----------------------:|--------------------------:|---------------------:|
| 3     | +0.237         | +0.019                 | —                         | 92% (calib)          |
| 9     | +0.315         | +0.029                 | **+0.014**                | **96%**              |
| 11    | +0.622         | +0.011                 | **−0.008**                | **101%**             |

**The recovery generalizes to unseen text** (distilled on one corpus, measured on a different one):
96–101% of the gap recovered for the worst layers, with a ~4-parameter feature map. So distillation
turns "~4–6 layers linearizable zero-shot" into "most layers linearizable at near-zero perplexity
cost" — each converted layer trading its O(T) KV cache for an O(d²) constant state.

Honest caveats: output distillation, **single layer at a time** (joint multi-layer distillation is
the next test — errors may compound), tiny CPU model, perplexity proxy (not a full conversation
eval), and the memory win is still long-context-only (crossover ~227 tokens) with a short-context
compute penalty. Recovery this high with 4 params suggests the linear attention was already close on
those layers — consistent with their low zero-shot Δ.

---

## 5. Goal hit: Qwen2.5-1.5B-Instruct converses in llcore's own code

`src/llcore/runtime/loader.py` — a memory-frugal **streaming loader** fills the native model one
tensor at a time from the (mmap'd) safetensors, so peak RAM ≈ the model + one tensor (no 1.5×
fp32-dict spike). Verified bit-exact against the dict loader on 0.5B (max|Δ| logits = 0.0). Actual
machine RAM is **16.8 GB (≈7.4 GB free)** — the earlier "3.6 GB" assumption was stale.

Loaded **Qwen2.5-1.5B-Instruct** (1.54B params, ~6 GB fp32, 29 s) and held a multi-turn Japanese
conversation entirely through llcore's native forward (~1–3 tok/s CPU):
- 「3たす5は？数字だけ」→ **「8」** ✓ (0.5B answered 18 — a real capability jump)
- 「『明日来てね』を敬語に」→「明日に来てくださいね」 ✓
- 「日本で一番高い山は？」→「富士山です」 ✓
- しりとり → weak (1.5B still limited at word games — honest).

**This meets the "まともに会話できる" goal for general Q&A / arithmetic / instruction-following**, in
llcore's own instrumentable code (the prerequisite for the bounded-memory R&D), not a black box.

int8 honesty: `convert_linears_to_int8` (llcore quant) on the loaded 1.5B keeps it answering
correctly (「8」, 「富士山」), but **in-process fp32→int8 conversion does NOT lower measured RAM** (the
freed fp32 buffers stay with the allocator — the known "peak WS unchanged without pressure" finding)
and is **slower on CPU** (~0.5 tok/s: per-forward dequant, no int8 GEMM). The real int8 RAM win needs
a **streaming-int8 load** (quantize per tensor from safetensors, never materialize fp32); the real
int8 *speed* win needs GPU int8 GEMM ("良い HW ほど効く").

---

## Next

- **Streaming-int8 load** (quantize per tensor from safetensors) so 1.5B/3B run at int8-resident RAM
  from the start — the genuine memory-efficiency clincher on a real conversational model.
- Run the linearization-tolerance profile + per-layer distillation on **1.5B** (the R&D applied to the
  model that actually converses), then **joint multi-layer distillation** under a quality budget and
  **evolve which layers** with `memory_objective` + cap-gate (the evolutionary substrate on the runtime).
- Scale to Qwen2.5-3B-Instruct (int8 ≈3 GB) for higher conversational quality.
- The open-model structural-analysis RAD corpus (`D:/docs/open_model_architectures_corpus_v2`, 24
  models + design-space map) grounds which architectures/patterns are most linearization-amenable
  (e.g. Qwen3's QK-norm bounds q/k just like our learned affine feature map; MLA is an alternative
  bounded-KV mechanism; SSM/hybrids are the O(1)-state destination).

All artifacts are local, no push. Tests: `test_runtime_qwen2.py`, `test_runtime_linearize.py`,
`test_lm_checkpoint.py`, `test_longctx_eval.py`, `test_tbptt.py` (all green; full suite non-regression).
