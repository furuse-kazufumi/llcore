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
- **Most middle/late layers are nearly free**: layer 22 (Δ+0.0007 ≈ free), 7 (+0.006), 15 (+0.02),
  13/19/18/12/6 all < +0.04.
- **Cumulative (greedy by single-layer Δ):** top-4 tolerant layers → ppl 73.7 (+7%, ~free);
  6 → 81.6; 8 → 93.8; 12 → 167 (breaks); all 24 → 17612 (catastrophic without recovery — expected).
- **Memory:** linear state = 232,960 B/layer (constant) vs softmax KV @8192 tok = 8,388,608 B/layer
  (36× and growing). **Crossover ≈ 227 tokens** — beyond a short context a linearized layer is
  strictly cheaper and never OOMs as the chat grows.

**Honest result:** ~4–6 of Qwen2.5-0.5B's 24 layers can be converted to constant-memory attention
**nearly for free, zero-shot**; layer 0 and a couple of retrieval layers resist. Prior art exists
(linear attention: Katharopoulos 2020; linearizing pretrained LLMs: SUPRA, LoLCATs, Mamba-in-Llama).
The contribution = on-prem internal surgery + rigorous per-layer measurement in llcore's own code.

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

---

## Next (the genuine capstone): recover linearized quality by distillation

The zero-shot profile shows the *ceiling without training*. The novel llcore contribution is to
**recover** capability so that many more layers can be linearized: add a learnable feature map to the
linear attention (frozen projections), distill its output to match the softmax teacher on a small
calibration corpus (LoLCATs-style), and re-measure the tolerance profile. Then evolve which layers to
linearize (memory_objective + cap-gate). Scale the substrate to Qwen2.5-1.5B-Instruct for genuinely
"まとも" conversation.

All artifacts are local, no push. Tests: `test_runtime_qwen2.py`, `test_runtime_linearize.py`,
`test_lm_checkpoint.py`, `test_longctx_eval.py`, `test_tbptt.py` (all green; full suite non-regression).
