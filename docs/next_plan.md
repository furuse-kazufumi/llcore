# next_plan (正本) — llcore

> 最終更新: 2026-06-23 ~06:40 JST（ccr セッションでユーザー指示により再構築・肥大化解消）
> SESSION_SUMMARY.md は raptor Stop hook で毎ターン自動上書きされるため、**このファイルが再開の正本**。
> 旧 EXIT(43)–(87)（Kaggle ポーリングログ約 2100 行・ほぼ全て「v3 RUNNING 待機継続」の重複）は
> `docs/archive/next_plan_kaggle_polling_through_EXIT87_20260623.md` に全文退避済（可逆）。

## 現在地（2026-06-23）

研究（科学）は健全。詳細の正本は `docs/CONVERSATIONAL_LLCORE_FINDINGS.md` §1–11：
- **会話基盤**: llcore 自前 Qwen2 forward（`runtime/qwen2.py`、HF golden 一致 2e-4）が 0.5B/1.5B で日本語会話成立。
  streaming-int8 ローダで **1.5B を 2.44GB resident**（fp32 5.7GB から）。
- **メモリ効率 R&D**: 層別 linearization-tolerance profile → 蒸留で gap **96–101% 回復**（~4 params/head, held-out 汎化）。
- **進化×構造**: memetic NSGA-II が greedy を **+34.3% HV** 支配、蒸留 frontier **+16.4%**、**proxy-v2**（統計的に正直な eval）で
  「単一 256tok 窓は constant-state 劣化を検出不能」を実証。両 Goal（会話可能＋進化が証明可能）実証済。
- **記事 32 本（技術 16 + 一般 16, JA）**。**2026-06-23 ccr が C フェーズ（挿絵）+ 内容監査を完遂**:
  全 16 トピックに house-style 静的 SVG 概念図を embed（旧 3 → 全 16、33 枚／全 32 ドラフト網羅）+ 監査 punch-list の
  content 修正を適用・検証（b1 数値 84.0% 整合 / a8 callout 331.8・205.4 整合 / a7・a8・b2 の制作メモ全削除 /
  用語注・敬称・免責バナー・噛み砕き肉付け）。leaked note 0・TODO 0・全 SVG Qiita-safe 静止。
- **B フェーズ（アーク 4 本圧縮 → 多言語）完遂（2026-06-23 ccr）**: 16 トピックを背骨に沿って **4 アーク記事（S/A/B/C, JA）**へ
  非破壊統合（`docs/articles/arcs/`, 元 16 トピックは source 温存）→ **en/zh/ko へ翻訳 = 計 16 多言語アーク記事**（ja/en/zh/ko ×4,
  829k 字）。検証: 全言語版で見出し数・SVG 数・代表数値（28.66% / 331.8 / 84.0% / 1,572,864 / +16.8% / 0.619 等）が JA と一致・
  honest disclosure/dead link/leaked note クリア。
- **残フォローアップ**: ①公開時の sibling 相互参照 URL マッピング（記事 slug→Qiita URL）②SVG 図中ラベルの言語別ローカライズ
  （現状 en/zh/ko 記事内も図は日本語ラベル・alt は訳済）③実 Qiita 公開（human gate A）。
- Kaggle needle offload は **任意 polish**（b2 の 2048 実測値差替のみ）。**2026-06-23 ccr が sweep-only(needle 除去) v5 に切替済**。

---

## 次の一手（優先順・2026-06-23 再編）

### P0 — human gate（ユーザーのみ実行可・これが真の critical path）
- **Qiita 公開**: 32 本 publish-ready。公開順 s1→s2→a 系→b 系→c 系、各 技術版→一般版ペア。
  raw 絶対 URL / imgix 静止化 / 姉妹リンク確認。**Kaggle 結果を待たない**（b2 は L137-138 が honest 留保を
  narrative 化済で実測値なしでも publish-ready）。

### P1 — 本研究（AI が進められる本命・これまで Kaggle 沼で starve していた）
`CONVERSATIONAL_LLCORE_FINDINGS.md` §Next。
1. **memetic NAS を本走スケール**: full proxy-v2 = K≥12 holdout・context sweep **2048–4096**（SUPRA 級長文脈崩壊帯）・
   cross-corpus・pop/gen 拡大・`--distill right_shift_ci`。← b2 の実測値はここから「正規に」得られる（Kaggle 任意 polish より本筋）。
2. **1.5B へ適用**: linearization-tolerance + per-layer 蒸留を、実際に会話する 1.5B で。
3. **joint 多層蒸留**（誤差合成の検証）→ `memory_objective` + cap-gate でどの層を進化させるか。
4. **wider mixer**（SSM/RWKV ブロック、RAD コーパス `open_model_architectures_corpus_v2` 接地）。
5. 3B int8（≈3GB）へスケールで会話品質向上。

### P2 — Kaggle（任意・time-box・無限ポーリング禁止）
- v5（sweep-only, needle 除去・`--context-sweep 256,512,1024,2048`）を **1 回の 12h window のみ**監視。
- COMPLETE → `kaggle kernels output furusekazufumi/llcore-needle-offload -p out/needle_kaggle`
  → `py -3.11 scripts/extract_needle_results.py out/needle_kaggle/nas_pareto.json`（test 正パス=`tests/unit/test_extract_needle_results.py` 4 passed）
  → b2（`docs/articles/drafts/b2-suppress-your-win.md`）L115/L138 + SVG L113 を実測値に差替（任意）。
- **未完 / ERROR / 12h 超 → 諦めて b2 を UNTESTED narrative のまま publish。needle 版は二度と再 push しない。**
  P1.1 の full proxy-v2 本走で 2048 実測が取れればそちらを使う。

---

## 運用規律（EXIT43–87 の沼の教訓・2026-06-23 必読）

1. **完了確認できない外部ジョブは time-box（1 window）で打ち切る**。状態文字列だけの無限ポーリング禁止。
2. **状態無変化の周回で next_plan にフル EXIT ブロックを append しない**（1 行 or 無記録）。← 2277 行肥大化の主因。
3. **任意 polish を blocker 扱いしない**。critical path = publish（human gate）であって Kaggle ではない。
4. **ツール不安定時（偽出力）は成功表示を信用せず別手段で着地検証**（`git status`/`wc -l`/`ls`、`date && echo OK` で健全性確認）。EXIT-87-TRUE 事故の教訓。
5. これら ①–④ を llcore memory（`feedback-*.md`）に着地させる（過去 2 回 phantom で未着地）。

## 環境メモ
- HEAD=`ba60620`、ブランチ `feat/lm-recurrent`。push なし（human gate B）。
- Kaggle: 無料 CPU 30GB/12h。同一 slug は最新 version のみ実行（push で旧 run 置換）。enable_internet=true。
- 機械 RAM は 16.8GB（≈7.4GB free）。旧「3.6GB」想定は stale。

---

## P1.1 proxy-v2 本走 — CPU 完走を断念・GPU 待ちに整理 (2026-06-25)

- CPU で full proxy-v2 を 2 回試走: 4096 sweep=14.5h で kill / 2048 sweep=**15.4h でも未完**(CPU 59.8 時間)。
  律速は sweep 長でなく **rigorous frontier tier の base コスト**(全 frontier 点 × K=12 × 2 front + distilled frontier + cross-corpus 再評価=毎走 recompute)。
  → **CPU では非現実的**と実証。memory `project_gpu_pc_consideration`(forward 律速→GPU 必須) を裏付け。
- **eval_cache 温存**: `out/nas_pareto_v2full_local/eval_cache.json` (236KB) + `.bak`。GPU 確保後に **同コマンドで resume** すれば GA 探索は再利用。
  resume 手順: `out/nas_pareto_v2full_local/RESUME_INSTRUCTIONS.md`(無ければ本節のコマンドで再起動、`--out` 同一で自動 resume)。
- **記事は非ブロック**: b2/B アークは needle UNTESTED + smoke decay で publish-ready。full verdict は GPU 走後に b2 の 2048 narrative を実測差替で polish(任意)。
- 次の P1 候補(CPU でも可): 1.5B linearization-tolerance + per-layer 蒸留 / joint 多層蒸留(frontier 再評価を伴わない単発検証は CPU でも現実的)。
