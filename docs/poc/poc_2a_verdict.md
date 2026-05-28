# PoC 2a Verdict — factor_hook × state update kernel (mock)

調査日: 2026-05-29  
ファイル: `scripts/poc_2a_factor_hook_mock.py` + `src/llcore/factor_hook/hooks.py`  
Test: `tests/unit/test_poc_2a_factor_hook.py`

## falsifiable 命題

llcore 自前 factor_hook protocol が (a) 10 因子保持 + clamp、(b) Noop で Δ=1.0、
(c) Heuristic で directionality (uncertainty 高 → Δ<1)、(d) apply_hook で decay 動的調整、
(e) snapshot 差異で effective gene が変わる、(f) 決定論性、(g) 進化ループに統合可能。

## 破綻ゲート (G1-G7)

| Gate | 結果 |
|---|---|
| G1 10 factors + clamp | ✓ |
| G2 Noop Δ=1.0 | ✓ |
| G3 directionality (uncertainty=0.535 < baseline=1.284 < integrate=3.080) | ✓ |
| G4 apply_hook modifies decay | ✓ heuristic 0.5→0.536 |
| G5 snapshot distinguishability | ✓ dist=0.536 |
| G6 determinism | ✓ |
| G7 evolution smoke with hook | ✓ 10×10 monotonic best=0.496 |

pytest 10/10 PASS.

## Codex pair-review verdict

**Green-light** (3 wording fix 推奨):

1. **Q4 wording**: "factor_hook × RWKV mock" → "factor_hook × state update kernel (mock)"
   (実体は RWKV weight 未接続、RWKV-inspired state update kernel)
2. **Q3 wording**: "neutral" → "baseline 0.5" (中立固定点でなく all-0.5)
3. **Q1 follow-up debt**: `decay * (2 - Δ)` は Δ>2 で完全忘却に潰れる。
   v0.2 で `decay/Δ` or `decay + k*(1-Δ)` 改修候補。

→ 3 fix とも本 commit で反映済。

## honest 留保

- **mock 環境**: 実 RWKV-7 weight 接続は別 PoC (post-llcore-完成 phase)
- **G7 smoke の主張範囲**: "hookable evolution pipeline" の smoke、
  "dynamic hook" (snapshot 進化中変化) の本質実証は未 (固定 snap)
- **Heuristic 式 baseline bias**: all-0.5 で Δ≈1.28 (signal=(0.5+0.5+0.25-0.75)/2=0.25)
  → v0.2 で normalization 改修候補
- **factor_hook の API stability**: Protocol を安定化してから将来 GeneModHook
  (mix/gate_str も触る別 Protocol) を additive 追加する設計方針

## 次段

llcore Stage 0-2 全 PoC 完走 = **CPU PoC battery 完成**。
post-llcore-完成 phase:
- Stage 3 (kernel 多様化 gene)
- Stage 4 (learning_rule 進化)
- Stage 5 (Marabou Incremental bridge)
- 実 RWKV-7 weight 接続 (GPU/新 PC 後)

## 関連 memory

- [[project_llcore_init_2026_05_29]]
- [[feedback_codex_pair_review_for_llcore]]
