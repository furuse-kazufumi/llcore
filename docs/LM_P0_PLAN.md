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

### P1 ablation（2026-06-15, 日本語コーパス・honest disclosure）

| 設定 | train_loss | val_loss | gap | model PPL (=exp val) |
|---|---|---|---|---|
| smoke dropout=0, 2000 iters | 3.24 | 3.61 | 0.37 | 36.43 |
| smoke dropout=0, 3500 iters | 2.55 | 3.54 | 0.99（過学習） | 34.85 |
| **smoke dropout=0.1, 3500 iters** | 2.89 | 3.46 | 0.57 | **32.40 (0.150×)** |

- **教訓**: 小コーパス（289K字）では **iters を増やすだけだと train のみ下がり held-out 頭打ち＝過学習**。
  **dropout=0.1 で過学習を抑制（gap 0.99→0.57）→ held-out 最良**。ボトルネックは反復数でなく
  正則化/データ量（リコンの nano 予測どおり）。CLI に `--dropout` 追加。生成も改善（迷亭先生/寒月君/令嬢など漱石登場人物再現）。

### P1 残り / P2 / P3

- 2026-06-16 追記: `out/lm_aozora_realp1` の実 `--config p1` run は **完走済み**。`max_iters=1000`, `ctx=256`, `dropout=0.2` で `model_ppl=38.3152`, `unigram_ppl=215.0577`, `ratio_model_over_unigram=0.1782`, gate PASS。比較スクリプト `py -3.11 scripts/p1_compare.py` では、smoke-best `lm_aozora_drop` (`ratio=0.1504`) に対して **held-out は未改善** と再確認された。したがって現時点の honest 結論は「`p1` 実 config は動くが、小コーパス・1000 iter の条件では smoke-best を上回れていない」。
- 2026-06-16 追記: `--extra-corpus-file` 導入後の `_load_corpus()` について、`out/corpus_aozora.txt` 単独入力では **旧経路とバイト同一**であることを実測確認した（len `320730` 一致、SHA256 `58ed1634a9880b2659c212ca162f2d18ab126bf99e16dfccfcb075431e2f7a93` 一致）。よって上の `lm_aozora_drop` vs `lm_aozora_realp1` 比較に、少なくともこの単一路径の改行正規化差は混入していない。
- 2026-06-16 追記: 次の「データ追加」rerun を安く選別するため、`py -3.11 scripts/p1_corpus_probe.py out/corpus_aozora.txt <extra...>` を追加した。`train` と同じ改行正規化で extra corpus 候補の chars / vocab / SHA256 / **new chars vs base / OOV rate vs base tokenizer** を先に観測できる。`out/corpus_aozora.txt` 単体 probe では chars `320730`、vocab `3044`、SHA256 `58ed1634a988...` を再確認し、既存記録と整合した。
- 2026-06-16 追記: 上の probe はその後、正規化/連結を `src/llcore/lm/corpus.py` の pure helper へ切り出して `train` と**単一実装**に揃えた。表記も `new chars vs base (uniq)` / `OOV vs base (occurs)` に補強し、preview の 12 文字 cap は `...(+N more)` で明示、extra `sha256` は**単体候補ファイルの指紋**であって combined 中の寄与バイト列ではない旨を note に追記した。これで cheap triage の honest disclosure が一段締まった。
- 2026-06-16 追記: extra corpus 束の運用も軽くし、`train` / `eval` / `scripts/p1_corpus_probe.py` は `--extra-corpus-manifest` を受けられるようにした。manifest は UTF-8 の 1 path/line 形式で、blank 行と `#` comment を許す。これで candidate 束を 1 ファイルへ固定し、probe で tokenizer drift を見てから同じ束を train/eval へそのまま流せる。
- 2026-06-16 追記: `scripts/p1_corpus_probe.py` はその後 `--write-manifest` と `--max-oov-rate` / `--max-new-chars` も受けるようにした。したがって次の cheap triage は「候補群を probe → 条件で filtered manifest を自動生成 → その同じ manifest で train/eval」まで一筆でつながる。
- 2026-06-16 追記: さらに、filter 判定は丸め表示値ではなく生の OOV rate で行い、選別が走った場合は **selected subset の combined** も再計算して表示/JSON に残すようにした。manifest は可能な限り出力先基準の相対 path で書くため、probe→train/eval の round-trip をそのまま持ち回しやすい。
- 2026-06-16 追記: extra corpus の実データ準備も 1 手で回せるよう、[scripts/p1_prepare_aozora.py](D:/projects/llcore/scripts/p1_prepare_aozora.py) を追加した。これは Aozora Bunko の zip URL 群（直指定または `--url-manifest`）を取得して `clean_aozora()` と同じ cleaning を通し、UTF-8 corpus 群 + `--write-manifest` の train-ready manifest + JSON report をまとめて生成する。manifest は可能なら出力先基準の相対 path で書くため、`p1_prepare_aozora.py` → `p1_corpus_probe.py` → `llcore.lm train/eval --extra-corpus-manifest` を同一 bundle のままつなげられる。
- 2026-06-16 追記: prepare script の prepared-output commit もその後 harden し、corpus/sidecar は tmp file へ stage してから `os.replace()` で反映、途中失敗時はこの batch が作った final outputs を cleanup する。加えて `--write-manifest` / `--json` の親ディレクトリ自動作成、URL 0 件の `argparse` error 化も入ったため、bundle 準備の CLI 摩擦はかなり減った。
- 2026-06-16 追記: manifest 自体の provenance も追加した。`--write-manifest` を使う prepare / probe の両方で `<manifest>.bundle.json` を自動生成し、ordered corpus file 群の `path / chars / vocab / sha256` と normalized combined corpus の `sha256` を固定する。これで cheap triage から train/eval へ流した束を、manifest の path 列だけでなく **内容ハッシュ付きの bundle 記録**として残せる。
- 2026-06-16 追記: その後、probe の empty selection と bundle 契約も整理した。全 extras が filter 落ちした場合でも header-only manifest は正常に書き、bundle JSON は意図的に未生成へ寄せる。probe manifest の bundle `combined.sha256` は `base_file=` を通して **base+selected extras** の train 順ハッシュになり、prepare manifest 側は extras-only のまま `includes_base=false` を明示する。`bundle_sha256` も path 依存をやめ、ordered file `sha256` 列だけの fingerprint に変えた。
- 2026-06-16 追記: さらに manifest **消費側**も sibling bundle を検証するようにした。`resolve_extra_corpus_files()` は `<manifest>.bundle.json` があれば manifest 本体の `sha256` と bundle payload を照合するため、probe/prepare 後に manifest や base corpus が drift した場合は train/eval/probe の再実行時点で fail-closed する。これで cheap triage → manifest 化 → train/eval の導線は、書込み側だけでなく消費側でも provenance を見て止まれる。
- 2026-06-16 追記: その後、prepare bundle と probe bundle の契約差も吸収した。verify は bundle の `combined.includes_base` に従って base を混ぜる/混ぜないを切り替え、比較から `files[].path` を外すため、prepare が書く extras-only bundle と probe が書く base+extras bundle が同じ消費層を通る。これで「prepare → probe → train/eval を同一 bundle のまま流せる」という導線は、実 round-trip test でも通る状態になった。
- 2026-06-16 追記: その運用確認をさらに軽くするため、[scripts/p1_manifest_inspect.py](D:/projects/llcore/scripts/p1_manifest_inspect.py) も追加した。manifest と sibling bundle を読み、ordered file SHA / `combined.sha256` / `includes_base` を一覧しつつ、`--base-corpus-file` があればその場で `verify_corpus_manifest_bundle()` を叩く。次の重い rerun 前は、prepare/probe が書いた束をまずここへ通して drift が無いことを cheap に確認してから train/eval へ渡す。manifest 不在や壊れた bundle payload は traceback ではなく `[verify] ...` + rc=1 へ統一し、base 未指定で verify を skip した表示には `(unverified)` を明示する。subprocess smoke は `py -3.11` launcher 固有ではなく **`sys.executable` で現在の pytest 解釈系から script entrypoint を起動する** 契約で固定している。
- 2026-06-16 追記: `train` / `eval` の consume 成功時にも 1 行の verified summary を出すようにした。sibling bundle 付き manifest を読んだ場合、CLI は `entries` / `includes_base` / `combined_sha256` / `bundle_sha256` をログへ出すため、prepare/probe で作った束が実際にどの内容ハッシュで rerun へ流れたかを heavy run の標準出力から直接監査できる。
- 2026-06-16 追記: その provenance は stdout だけでなく `verdict.json` / `train_state.pt` にも保存するようにした。`manifest_verification` には verified/unverified 各 summary が入り、**verified は raw manifest entry が collapse せず、そのまま実消費集合へ入る場合に限る**。manifest に base 混入や duplicate extra がある場合は、dedup 後 hash を verified として付け替えるのではなく `collapse after base/duplicate filtering` で fail-closed reject する。したがって rerun 後にログが欠けても、artifact に残る verified hash は「実際に consume した corpus を bundle で attest できたケース」だけを指す。
- 2026-06-16 追記: cheap 監査用の `scripts/p1_manifest_inspect.py` も同じ contract B に揃えた。manifest の raw entry 数に加えて `effective_entries` を表示し、verify 成功時は effective corpus に対する `combined_sha256` / `bundle_sha256` を `[effective]` 行へ出す。ただしこれは **raw entry が collapse しない manifest に限る**。base 混入や duplicate 行で `entries != effective_entries` になった manifest は reject するため、inspect 表示と rerun 本体の verified summary が一致するのは single-manifest / no-collapse ケースに限られる。
- 2026-06-16 追記: その後の edge も詰め、`effective_entries=[]` + `includes_base=false` でも inspect は `_verified_manifest_summary()` と同じ empty-summary 契約で通すようにした。壊れた `combined` payload は表示前に `ValueError` へ寄せ、verify=failed の表示にも unverified 印を付ける。なお inspect の `effective_entries` は単一 manifest の隔離ビューであり、`--extra-corpus-file` や複数 manifest をまたぐ cross-source dedup までは反映しない。
- 2026-06-16 追記: `p1_manifest_inspect --json` も runtime 側の `manifest_verification` schema へ寄せた。ただし **verified エントリのみ**が runtime `verdict.json` と同型で、inspect には `skipped` / `failed` という専用 status が残る。また inspect の `effective_entries` は単一 manifest の隔離ビューなので、`--extra-corpus-file` や複数 manifest を併用した実 run と完全一致するのは cross-source dedup が絡まないケースに限られる。
- 2026-06-16 追記: その温度差を CI 側で埋めるため、`p1_manifest_inspect.py` には `--require-verified` を追加した。既定では `skipped` を rc=0 の cheap pre-check として許容し、strict な自動化だけがこの flag で `verification.status != "passed"` を rc=1 に切り替える。
- 2026-06-16 追記: さらに provenance contract を contract B へ固定した。manifest に base path や duplicate extra が混入して `entries != effective_entries` になる場合は、runtime / inspect とも **`collapse after base/duplicate filtering` で reject** し、effective hash を `verified` として流用しない。これにより `verified` は「raw manifest entry がそのまま実消費集合へ入る」ケースだけを指す。shape 検証も `bool ⊂ int` を弾くよう補正したため、`chars=true` などの malformed bundle は skip/display-only 経路でも fail-closed に落ちる。
- 2026-06-16 追記: その最後の監査導線として、`scripts/p1_manifest_reconcile.py` を追加した。これは **1 本以上の** `p1_manifest_inspect.py --json` report を argv 順に連結し、`--runtime` で明示指定した runtime 側の `verdict.json` / `train_state.pt` に残る `manifest_verification` を文字列整形なしで照合する。差分時は per-entry summary を出して rc=1 で止まり、比較は seed #32 に合わせて **content/contract フィールドだけ**で行い、`manifest_path` の絶対パス差は表示専用へ落とす。`.pt` は `weights_only=True` で読み、壊れた checkpoint は `[reconcile] ...` の整形エラーへ寄せる。さらに `--json` で matched / mismatch を structured report 化でき、report payload 自体も `comparison_mode="positional"` を持つため、single-manifest / no-collapse ケースでは cheap inspect → rerun → reconcile をそのまま CI 比較へ流せる。multi-manifest は位置対応・順序依存比較であることも docstring / help に明記済みで、order-mismatch は **内容差のある entry** を入れ替えた場合に限る一方、同一内容で `manifest_path` だけが異なる entry の swap は content-only 契約により意図的に検出しない。最新 report 用 gate は `py -3.11 -m pytest tests/unit -k "lm or corpus or probe or p1_compare or p1_prepare or p1_manifest_reconcile" -q` = `204 passed, 401 deselected`。
- 2026-06-16 追記: `reconcile` の entry schema も fail-closed に締めた。`manifest_verification` の `verified` entry は `entry_count / generated_by / includes_base / combined_sha256 / bundle_sha256` を必須、degraded entry は `reason` を必須にし、両側同時欠落でも false-positive `"matched"` にならないよう load 時に reject する。
- 2026-06-16 追記: その後 `p1_corpus_probe.py` と `p1_prepare_aozora.py` の 2 manifest を順序どおり `llcore.lm train` に渡す **actual multi-manifest happy-path** 回帰も追加した。inspect 側も shipped CLI 契約へ寄せ、`p1_manifest_inspect.py --json` を **manifest ごとに 1 回ずつ**実行し、`p1_manifest_reconcile.py` が複数 inspect JSON を argv 順に positional 連結して runtime `manifest_verification` と照合する。この回帰が実際に pin しているのは、probe+prepare の **manifest ごとの on-disk inspect handoff** と runtime が **manifest 群の順序** および **各 producer の `includes_base` provenance 契約** を保ったまま `manifest_verification` を emit すること。順序入れ替え reject は synthetic reconcile fixture に加えて、**actual producer/runtime artifact + shipped inspect JSON 群**を使った order-mismatch 回帰でも固定済みだが、この「固定済み」は `reconcile` の **現 `COMPARABLE_FIELDS` 契約範囲**に限る。また `includes_base` 自体は per-entry provenance 表示であり、base の二重計上防止や corpus 合成保証そのものは runtime の dedup/resolve ロジックが担う。
- 2026-06-16 追記: resume の provenance 欠落も補修した。snapshot `train_meta` には **`requested_extra_corpus_files` / `extra_corpus_manifests`** を追加し、manifest-backed run を `--resume-checkpoint` だけで再開しても saved manifest 群から bundle 再検証をやり直して `manifest_verification` を維持する。manifest path 群を持たない pre-fix snapshot では saved `manifest_verification` をそのまま保持する fallback を残し、`_restore_training_snapshot()` / `_load_checkpoint()` は **`weights_only=True`** に切り替えた。これで resume/eval の CLI 入力から full pickle を実行せずに済み、report 用 gate も `py -3.11 -m pytest tests/unit -k "lm or corpus or probe or p1_compare or p1_prepare or p1_manifest_reconcile" -q` = `206 passed, 401 deselected` まで更新済み。
- 2026-06-16 追記: その軽微 follow-up として、`verdict.json` には **full training/eval input の `corpus_sha256`** も載せるようにした。prepare 由来 manifest の `combined_sha256` は extras-only provenance のまま維持しつつ、学習コーパス全体の指紋を別キーで見られる形にした。加えて、完了済み snapshot (`iter_num>=max_iters`) を `--resume-checkpoint` で読む場合も artifact を **再 emit** するので、`verdict.json` や `model.pt` を手で消した後の completed resume でも成果物ゼロにならない。manifest collapse の文言も、manifest 側だけでなく overlapping CLI extras も原因になり得ることが分かるよう補正済み。最新 report 用 gate は `py -3.11 -m pytest tests/unit -k "lm or corpus or probe or p1_compare or p1_prepare or p1_manifest_reconcile" -q` = `207 passed, 401 deselected`。
- P1 残: さらなる PPL 改善（block_size 拡大・`--config p1`・データ追加）+ **学習済みモデルを clean-room
  3D で歩く**（`model_viz.json` を自前 Apache-2.0 ビューアでロード。bbycroft コードは無ライセンス＝非依存で再実装）。
- P2: 進化 = NAS（アーキ/ハイパーを proxy 短学習で探索）。P3: クラウド GPU + BPE + 大コーパスで質問応答級。
