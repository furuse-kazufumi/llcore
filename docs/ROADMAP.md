# llcore ROADMAP (自走運用の単一の真実, 2026-06-11〜)

> **運用方式 (ユーザー確定 2026-06-11): ロードマップ自走。**
> Claude が本 ROADMAP を保持し、優先順に**自律実行** → 各マイルストンで commit + 結果報告。
> 確認を取るのは **決定点のみ** = 不可逆操作 (force push/削除/DB drop) / 仕様分岐 / 課金 / 明示制約との矛盾。
> ユーザーは方向修正だけ行う。準拠 memory: feedback_max_plan_autonomy / feedback_session_marathon /
> feedback_parallel_first_execution。進捗は本ファイルの ✅/🔄/⬜ で管理し、毎マイルストンで更新する。

## 北極星
llcore = **進化可能な LLM の検証フレームワーク** (verified plasticity) +
**会話→アノテーション→連結性→世界モデル**の獲得経路。既存モデルは「進化型ベース」であり製品ではない。

## ★戦略課題: 世界知識の不在 (ユーザー指摘 2026-06-11)
llcore の進化コア/検証器は**世界知識を内包しない** (tiny adapter + verifier)。これは make-or-break な弱点。
世界知識の供給源は 3 つ、いずれも AnnotationStore/連結性グラフが統合層になる:
1. **frozen base LLM** (SmolLM2/sarashina) — llcore.chat が borrow (知識はあるが進化しない・小型は不正確)。
2. **会話アノテーション** — 連結性グラフに蓄積 (slow・会話で言及された範囲のみ)。
3. **★RAD コーパス (21 分野 49k docs) = 既製の世界知識** — AnnotationStore に取り込めば連結性グラフが
   **世界知識グラフ**になる。これが最有力 (loop-engineering corpus も新規ノードとして合流)。
→ ROADMAP M3 で RAD→AnnotationStore 取込を検証 (世界知識の注入が retrieval/grounding を改善するか)。

## マイルストン

### M0 完了済 (2026-06-10〜11)
- ✅ llcore.chat (基本会話, SmolLM2-360M default, JA=sarashina)
- ✅ llcore.clip (SigLIP) + AnnotationStore (uint64 ID/連続float32/int8/id_cosine 厳密/二層リンク)
- ✅ 検証付き会話デモ (sound cert のみ真ρ<1, 25-agent レビュー honest 修正済)
- ✅ 1D-SegFormer PoC (日本語境界 F1 0.774>規則)
- ✅ 優先1 事実抽出 / 優先2 head-to-head (CLIP 反証・差別化は連結性グラフと判明) /
     優先3 連結性グラフ (MRR 0.056→0.389, R@1 0→1/3, IDF hub 抑制)

### M1 ✅ 連結性グラフ — 降格のままクローズ (2026-06-12)
- ✅ M1.2 評価ベンチ拡張 (3→22 probe, `scripts/connectivity_bench.py`)
- ⚠️ **★honest 訂正**: 22 probe で **IDF 連結は cosine と完全一致 (0/22 差, MRR 0.727)**、IDF なし連結は
  大半を害す (0.321)。「連結性が差別化」は 3 probe の過剰適合 + 弱い cosine baseline のアーティファクトだった。
  **実の勝因は事実抽出 (質問/依頼除外)** = cosine を 0.727 に。詳細=CONNECTIVITY_BENCH_CORRECTION_2026_06_11.md。
- ✅ entity coref エッジ (2026-06-12) — 加算マージン方式 (`query_connected(entity_hop=True)`, 既定 off)。
  CLIP で 5 probe 改善/0 悪化 (MRR 0.727→0.797)、MiniLM では効果ゼロ・非破壊 (0.947 不変) =
  **弱 encoder の補償としてのみ価値**。正本 = textseg1d/M1_ENTITY_ENCODER_RESULTS_2026_06_12.md
- ✅ encoder 差し替えオプション (2026-06-12) — `SentenceEncoderBackend` (all-MiniLM-L6-v2, optional extra
  `text`)。**MiniLM cosine 単独 MRR 0.947 (R@1 0.909)** = この規模の会話 retrieval はほぼ解決。
  追加知見: cooccur hop は強 encoder で微害 (0.947→0.902) — 単調改善主張を down-claim。
- → **差別化の主軸を M3 (世界知識) + M2 (cert×教師) に移す**。retrieval は事実抽出+MiniLM cosine で大半解決。

### M2 ⬜ cert gate × 連結性教師の配線 (差別化の本命)
- ⬜ 連結グラフ (ターン境界/照応/話題) を verified adapter の進化・学習信号に
- ⬜ sound gate vs 経験 gate vs 無 gate で「会話教師下の連結性保持」を実 LLM hidden で比較

### M3 🔄 世界知識の注入 (ユーザー指摘への回答 = 差別化の主軸)
- ✅ **取込 PoC 成立 (2026-06-12)**: loop-engineering corpus 39 docs → MiniLM AnnotationStore。
  世界知識 18 probe (事前登録) MRR **0 → 0.639** (11/18 が rank 1)、**会話 22 probe への干渉ゼロ**
  (store 11 倍化 97→1,071 でも per-probe 完全一致 0.947)。取込 17.0s。失敗 5 問の主犯 =
  gold の日英混在 + 断片化 (真の retrieval 失敗は 1 問のみ) → 0.639 は下限値。
  正本 = textseg1d/M3_RAD_INGEST_POC_2026_06_12.md + out/rad_ingest_poc.json
- ✅ **検証 (i) 大規模化 (2026-06-12)**: 3 corpus 1,989 docs → 59,971 annotations (56 倍, cap 60k,
  silent cap なし)。会話 22 probe MRR 0.947→**0.890** (軽微, rank 変化 2 probe のみ)、loop 18 probe
  0.639→**0.306** (大幅埋もれ)。劣化はトピック重複 probe に集中 → **「干渉ゼロ」(M3.0) は
  トピック非重複の条件付きと down-claim**。埋もれの主因 = 規模でなくトピック重複 ((iii) の部分的
  先行回答)。正本 = textseg1d/M3_SCALE_MULTILINGUAL_2026_06_12.md + out/rad_scale_poc_scale.json
- ✅ **検証 (ii) 多言語 encoder head-to-head (2026-06-12)**: multilingual-e5-small prefix なしは
  world +0.034 (M3.0 失敗 5 問中 3 問を rank 1 救出 = 日英混在 undercount 説を実証) だが
  conv −0.015 + レイテンシ 2 倍。prefix あり (公式推奨) は逆効果 (world 0.597)。
  **結論 = MiniLM 続投**、undercount は gold 判定の多言語化で解消する方が筋。
  正本 = 同上 + out/rad_scale_poc_ml.json
- ✅ **検証 (iii) トピック重複干渉 (2026-06-12)**: astrophysics 100/400/800 docs 段階注入。
  会話 MRR 0.947→0.849 だが **R@3 1.000 維持・天文 probe は rank 2 止まり = fail モードは
  壊滅でなく漸進的押し下げ**。劣化 9 probe 中 7 はトピック/語彙重複で説明 (実ヒット目視)、
  直交 probe は不変。重複 23k ann の干渉 > 非重複 60k ann ((i) 比) = 規模より中身。
  corpus が rank 1 を取った事例は全て role="corpus" → **role フィルタで会話 R@1 全防衛可**。
  正本 = textseg1d/M3_TOPIC_OVERLAP_2026_06_12.md + out/rad_topic_overlap_poc.json
- ✅ **role スコープ絞り込み実装 + 実証 (2026-06-12)**: `query(exclude_roles={"corpus"})` 追加
  (negative・複数可・矛盾指定 fail-closed・既定 None 後方互換、unit +3)。(iii) +800 store で
  会話 22 probe が **0.849 → 0.947 (R@1 0.909) = 注入前に完全復元**。限界 = corpus 間
  (loop vs astro) の食い合いは role では防げない (loop 0.490 のまま)。
  正本 = M3_TOPIC_OVERLAP 追記 + out/rad_role_filter_check.json
- ✅ **分野単位スコープ実装 + 実証 (2026-06-12)**: per-row domain タグ (設計 (b)) —
  `add_text(domain="loop")` + `query(domain=..., exclude_domains=...)` (role と直交・
  fail-closed・後方互換、unit +6 = 399 PASS)。(iii) +800 store で loop 18 probe が
  `domain="loop"` で **0.4895 → 0.6389 = astro 混入前に全 metric 一致で復元**
  (role="corpus" は nofilter と同値 = role の限界を in-run 再現)。会話側回帰なし (0.947)。
  honest: 復元は構造的必然 (スコープ = M3.0 と同一集合)、価値は実装確認 + 回帰なし証明。
  限界 = クエリ→分野ルータ未設計 / per-row 単一値は初出優先。
  正本 = textseg1d/M3_DOMAIN_SCOPE_2026_06_12.md + out/rad_domain_filter_check.json
- ✅ **ANN 化実装 + 実測 (2026-06-12)**: faiss HNSW (`query(ann=True)`, optional extra
  `llcore[ann]`、不在時 ImportError = fail-closed・黙って exact 劣化しない)。フィルタとは
  over-fetch で両立、quantized 併用は fail-closed。23k store 40 probe で **recall@10 0.9825
  (min 0.8)・domain="loop" 併用 MRR 0.6389 = exact 完全一致**。honest: 23k では速度メリット
  なし (19.6→16.9ms、支配項は query encode ~15ms) — ANN の本領は ~10 万行超。HNSW 構築
  1.94s。unit +4 = 403 PASS。正本 = textseg1d/M3_ANN_HNSW_2026_06_12.md + out/rad_ann_check.json
- ⬜ RAD コーパス全量取込 (実測 49 分野 ~17.8k docs → ~50 万 ann 見込み) → 連結性グラフ =
  世界知識グラフ。encode ~87 分 (96 ann/s 実測) が支配項 → save/load 永続化で一度きり運用。
  取込後に ANN 速度メリット + recall を全量規模で再測
- ✅ **★言語 RAD コーパス = M4 で新設済み (重複記載を整理 2026-06-12)**: corpus/language
  (D:/docs/language_corpus_src, 64 docs) が要求 5 領域と完全対応 — syntax-grammar /
  morphology-lexicon / semantics-pragmatics / phonetics-phonology / multilingual-japanese
  (各 12-13 docs)。全量取込 (51 corpora) に domain="language" で含まれる。
- ⬜ 世界知識注入が retrieval/grounding を改善するか実測 (会話のみ vs 会話+RAD)

### M4 ✅ RAD コーパス生成 (世界知識ギャップ + 環境設計の根拠) — 完了
- ✅ ループ手法調査 (50 手法, 4 スコープ) → corpus/loop_engineering (39 doc / 12 cluster) 登録済
- ✅ 言語学調査 (67 トピック, 5 領域) → corpus/language (64 doc / 15 cluster) 登録済
- ✅ /corpus2skill で skill 階層化し RAD 全体 INDEX に登録 (raptor/.claude/skills/corpus/)

### M5 ✅ ループエンジニアリング環境 llloop v0.1.0a0 構築済 (ユーザー指示 2026-06-11)
- ✅ ユーザー決定: 配置=**D:/projects/llloop** (新規proj) / 自律=**push まで** / ccr=**当面独立** /
  初回タスク=**llcore テスト緑維持**。ccr は安全網として温存・無改変。
- ✅ 構築: MAPE-K runner + ★fail-closed 安全層 (SafetyPolicy/CircuitBreaker/Budget/認証検知) +
  差し替え戦略 (plan_execute_verify/reflexion) + green-keeper タスク。pytest 26 green/ruff/mypy clean。
- ✅ 実証: green-keeper が llcore で goal_reached / ドリフト→ruff自己修復→検証緑 の閉ループ実機確認。
- ⬜ 次: LLM actor 統合 (Plan を LLM 駆動に) / ccr claude-loop キュー連携 / 戦略比較実験。

## 直近の自走順 (上から実行)
1. ✅ M4 ループ調査 workflow 完了 → RAD corpus2skill 化
2. ✅ M1.2 評価ベンチ拡張 (3→22 probe) → hub 抑制を確定調整 (honest 訂正で決着)
3. ✅ M1 entity coref エッジ + encoder 差し替え (2026-06-12 クローズ)
4. M3 RAD→AnnotationStore 取込 (世界知識注入) ← **進行中**。PoC + (i)(ii)(iii) + role 絞り込み
   + 分野スコープ + ANN 化 ✅ (2026-06-12)。残 = 全量取込 (~50 万 ann) + 言語コーパス新設
5. M2 cert × 連結性教師 配線

各完了で本 ROADMAP の ✅ 更新 + commit + 1 段報告。
