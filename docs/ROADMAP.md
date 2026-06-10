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

### M1 ⚠️ 連結性グラフ — 22-probe で over-claim 判明・降格 (2026-06-11)
- ✅ M1.2 評価ベンチ拡張 (3→22 probe, `scripts/connectivity_bench.py`)
- ⚠️ **★honest 訂正**: 22 probe で **IDF 連結は cosine と完全一致 (0/22 差, MRR 0.727)**、IDF なし連結は
  大半を害す (0.321)。「連結性が差別化」は 3 probe の過剰適合 + 弱い cosine baseline のアーティファクトだった。
  **実の勝因は事実抽出 (質問/依頼除外)** = cosine を 0.727 に。詳細=CONNECTIVITY_BENCH_CORRECTION_2026_06_11.md。
- ⬜ entity coref エッジ — 語彙不一致の難ケース (name 等) 用の**小改善に降格** (差別化の主軸にしない)
- ⬜ encoder 差し替えオプション (MiniLM backend; CLIP は cross-modal 専用) — head-to-head で MiniLM 優位
- → **差別化の主軸を M3 (世界知識) + M2 (cert×教師) に移す**。retrieval は事実抽出+cosine/MiniLM で大半解決。

### M2 ⬜ cert gate × 連結性教師の配線 (差別化の本命)
- ⬜ 連結グラフ (ターン境界/照応/話題) を verified adapter の進化・学習信号に
- ⬜ sound gate vs 経験 gate vs 無 gate で「会話教師下の連結性保持」を実 LLM hidden で比較

### M3 ⬜ 世界知識の注入 (ユーザー指摘への回答)
- ⬜ RAD コーパス (21 分野) を AnnotationStore に取込 → 連結性グラフ = 世界知識グラフ
- ⬜ loop-engineering corpus (本セッション調査) を最初の取込対象に (dogfooding)
- ⬜ 世界知識注入が retrieval/grounding を改善するか実測 (会話のみ vs 会話+RAD)

### M4 ⬜ 自律ループの工学化 (ループエンジニアリング調査の実装反映)
- 🔄 ループ手法調査 → RAD コーパス化 (workflow 進行中, 自律/制御/学習/運用 4 スコープ)
- ⬜ 調査結果から FullSense/Claude Code 自律マラソンに採用する手法を選別・実装

## 直近の自走順 (上から実行)
1. M4 ループ調査 workflow 完了 → RAD corpus2skill 化
2. M1.2 評価ベンチ拡張 (3→20 probe) → hub 抑制を確定調整
3. M1 entity coref エッジ + encoder 差し替え
4. M3 RAD→AnnotationStore 取込 (世界知識注入)
5. M2 cert × 連結性教師 配線

各完了で本 ROADMAP の ✅ 更新 + commit + 1 段報告。
