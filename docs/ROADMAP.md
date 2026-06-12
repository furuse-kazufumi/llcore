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
- ⬜ RAD コーパス (~48 分野) を AnnotationStore に取込 → 連結性グラフ = 世界知識グラフ。
  次の検証 3 点: (i) 10 万 annotations 級での会話干渉再測 (ii) 多言語 encoder head-to-head
  (日英混在 gold の undercount 解消) (iii) 会話トピックと重なる corpus での干渉測定
- ⬜ **★言語 RAD コーパス新設** (ユーザー指摘 2026-06-11: 既存 ~48 分野に言語/語彙/文法/発音が無い):
  linguistics (syntax/morphology/semantics/pragmatics) / lexicon (WordNet/語彙意味論) /
  grammar (CFG/dependency/construction) / phonetics・phonology (IPA/調音/韻律) / 多言語 (特に日本語:
  形態素解析/活用/仮名漢字)。1D-SegFormer・事実抽出・アノテーション連結 (類義/形態エッジ) の基盤。
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
4. M3 RAD→AnnotationStore 取込 (世界知識注入) ← **次**。最初の取込 = loop_engineering corpus
   (dogfooding)。encoder は MiniLM (M1 確定)。大規模 store での MRR 保持も再測する
5. M2 cert × 連結性教師 配線

各完了で本 ROADMAP の ✅ 更新 + commit + 1 段報告。
