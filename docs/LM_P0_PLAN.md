# llcore P0 実装計画 — CPU char-Transformer トレーナ (`src/llcore/lm/`)

> 親計画: `docs/LLM_CAPABILITY_FIRST_REPLAN_2026_06_13.md`（capability-first 再計画）。
> 本書は P0（de-risk）の実装計画。リコン（2026-06-15, 4 観点並列 + 一次ソース検証）で確定した設計。

## 0. ゴール（P0 合格線）

日本語 char-LM が **held-out perplexity を char unigram baseline より明確に下回る**
（`model_ppl ≤ 0.85 × unigram_ppl`、実目標は 2×+）かつ **生成が崩れない**こと。
まず英 tiny-shakespeare で smoke（既知ベースライン）→ 日本語へ。CPU 完結（torch 2.12.0+cpu）。

honest 留保: P0 は「最低限 LLM = それっぽい生成」まで。なぞなぞ/QA 級は P3（クラウド GPU）。

## 1. アーキテクチャ決定（一次ソース検証済み）

- **topology = GPT-2 nano（minGPT/nanoGPT 準拠）**。学習済みを bbycroft/llm-viz の 3D で
  そのまま歩けるよう、**モジュール木を minGPT 命名に厳密一致**させる。
- **決定的発見**: llm-viz が読むモデル JSON は `model.state_dict()` の直接ダンプ
  （`llcore-viz/gen_test_data.py:38`、サンプル `llcore-viz/public/gpt-nano-sort-model.json` =
  45 キー / config + 44 tensor で実機検証済）。
  → exporter は `{config, **{k: tensor_to_json(v) for k,v in state_dict().items()}}` だけ。

### viz export schema（権威）
- top-level: `{"config": {...}, "<tensor key>": {shape,dtype,data}, ...}`
- config: `model_type, n_layer, n_head, n_embd, vocab_size, block_size`（+ optional dropouts）
- tensor: `{"shape":[...], "dtype":"torch.float32", "data": base64(little-endian float32, row-major)}`
- 重み命名（per block i）: `transformer.wte.weight`[V,C] / `transformer.wpe.weight`[T,C] /
  `transformer.h.{i}.ln_1.{weight,bias}` / `.attn.bias`[1,1,T,T] causal mask（登録バッファ） /
  `.attn.c_attn.{weight,bias}`[3C,C] / `.attn.c_proj.{weight,bias}`[C,C] / `.ln_2.{weight,bias}` /
  `.mlp.c_fc.{weight,bias}`[4C,C] / `.mlp.c_proj.{weight,bias}`[C,4C] /
  `transformer.ln_f.{weight,bias}` / `lm_head.weight`[V,C]
- LayerNorm は bias 必須（viz が ln bias を読む）→ P0 は `bias=True` 固定。
- 語彙 map は JSON に不要（viz は id→'A'+id でハードコード）。任意で `vocab` フィールド付加可。

### nano 構成（nanoGPT 検証値）
- **Config A（smoke, ~3-6分 CPU）**: n_layer=4, n_head=4, n_embd=128, block=64, dropout=0.0,
  batch=12, max_iters=2000, lr=1e-3→min 1e-4, warmup=100, wd=0.1, betas=(0.9,0.99), grad_clip=1.0, cosine。
- **Config B（P1 目標）**: 6/6/384, block=256, dropout=0.2, batch=64（CPU は 16-32 に縮小）。
- init: std=0.02、residual proj（c_proj）は std=0.02/√(2·n_layer)。GELU=tanh近似（minGPT NewGELU）。
  lm_head は wte と weight-tied（nanoGPT）。compile=False（CPU）。

## 2. コーパス決定

- **日本語**: 青空文庫（**public domain**）夏目漱石「吾輩は猫である」card 000148。
  zip `https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip`（**Shift-JIS**）。
  decode shift_jis → ruby `《》`・`｜`・注記 `［＃…］`・dashed header・`底本：` footer 除去 →
  ~400-500KB UTF-8、vocab ~2300-2600。train/val = 末尾10% 連続 held-out。
- **smoke**: tiny-shakespeare
  `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
  （~1.1MB, vocab 65。model PPL≈4.3 vs unigram≈16-17 = 約4×）。
- **unigram baseline**: train 文字頻度 + add-1 smoothing → held-out 平均 per-char NLL（自然対数）→
  `unigram_ppl = exp(mean_nll)`。

## 3. 再利用 vs 新規（リコン inventory より）

- **新規実装**（src/llcore/lm/）: model / trainer / generation / eval / export（minGPT topology は
  research/third_axis_gpu の HybridLM が参考だが research コード=参照のみ。viz 互換のため自前で厳密一致）。
- **参照**: `research/verified_lm_evolution/lm_substrate.py` の cross_entropy / unigram_ce 式（torch で再実装）。
- 慣習: SPDX Apache-2.0 header / `from __future__ import annotations` / NumPy-style docstring /
  tests `tests/unit/test_lm_*.py` / ruff line-100 py311 / mypy strict。

## 4. モジュール構成

```
src/llcore/lm/
  __init__.py     公開 API
  model.py        GPTConfig, NewGELU, CausalSelfAttention, Block, CharGPT(+generate)
  tokenizer.py    CharTokenizer（vocab=sorted(set(text)), stoi/itos, encode/decode, save/load）
  data.py         clean_aozora, fetch_*, train_val_split, get_batch
  eval.py         unigram_nll/ppl, held_out_perplexity, estimate_loss
  generation.py   generate_text, is_degenerate（崩壊ゲート）
  export.py       to_viz_dict / save_viz_json（state_dict → viz JSON）
  trainer.py      TrainConfig, Trainer（AdamW + cosine LR + warmup + grad_clip + eval）
  __main__.py     CLI: prepare / train / eval / generate / export
tests/unit/test_lm_*.py
```

## 5. 検証（honest disclosure）

1. 単体テスト全 green（model 形状/state_dict キー = viz サンプルと一致 / tokenizer round-trip /
   unigram / overfit sanity / export キー一致 / 生成崩壊検出）。
2. smoke: tiny-shakespeare で held-out PPL < unigram を実証（trainer が正しい証拠）。
3. 本番: 日本語コーパスで `model_ppl ≤ 0.85 × unigram_ppl` + 非崩壊生成 → 4 数値 + サンプルを verdict に記録。
4. export した JSON を再ロードしキー集合/shape がサンプルと構造一致することを確認。

## 6. 状態 / 結果

- 2026-06-15 着手・**P0 完遂**（branch `feat/lm-p0-char-transformer`、commit fdb66fd）。
- 実装: `src/llcore/lm/` 9 モジュール + `tests/unit/test_lm_*.py` 7 ファイル（48 tests green）/ ruff /
  mypy strict すべてクリーン。
- **検証結果（held-out PPL < unigram、両コーパスで PASS）**:

  | コーパス | vocab | params | unigram PPL | model PPL | 比 | gate | 崩壊 |
  |---|---|---|---|---|---|---|---|
  | tiny-shakespeare (smoke) | 65 | 0.81M | 28.43 | 6.78 | 0.238× | PASS | False |
  | 青空文庫「吾輩は猫である」 | 3044 | 1.19M | 215.38 | 36.43 | 0.169× | PASS | False |

  - smoke は nanoGPT 既知挙動（val_loss ~1.9）と一致＝トレーナの正しさを実証。
  - 日本語生成は「」会話・漱石語彙（主人/鼻/不平）・と云う/でしょう調を再現＝「最低限 LLM=それっぽい生成」達成。
  - viz export: 両モデルとも tensor 57 キー・per-block スキーマが実 viz サンプルと一致、weight tying
    （wte==lm_head バイト一致）込みで正しく出力（shakespeare 4.3MB / aozora 8.16MB の `model_viz.json`）。
  - honest 留保: 意味的整合（QA/なぞなぞ）は未到達＝P3（クラウド GPU + BPE + 大コーパス）。CPU-nano の範囲内。
  - held-out 評価は非重複窓（境界で文脈リセット）= モデルに**保守的**（実力より低めに出る）方向のバイアス。
  - 敵対レビュー（gem-critic）で eval/比較の健全性を確認: critical バグ無し・全非対称性がモデルに不利方向。
    推奨に従い `held_out_report` で **unigram をモデルと同一トークンで採点**（airtight 化）→ verdict 不変。

## 7. 次（P1）

- P1: 日本語で PPL をさらに下げる（dropout 付き・iters/モデル拡大、`--config p1`）+ **学習済みモデルを
  clean-room 3D で歩く**（`model_viz.json` を自前 Apache-2.0 ビューアでロード。bbycroft コードは非依存）。
- P2: 進化 = NAS（アーキ/ハイパーを proxy 短学習で探索）。P3: クラウド GPU + BPE。
