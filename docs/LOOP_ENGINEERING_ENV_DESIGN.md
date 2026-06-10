# ループエンジニアリング環境 設計 (M5, 2026-06-11 起草)

> **ユーザー指示 (2026-06-11)**: 一通りコーパス生成が終わったら、ループエンジニアリングをできる
> 環境を**検討構築**。**安全のため ccr は残し、新しい形で構築**。
> = ccr (claude-auto.mjs / claude-loop / raptor-loop-queue) は安全網として温存・無改変。別系統を新設。

本 doc は M4 調査コーパス (50 手法) を根拠にした設計。**構築前にユーザー確認を取る決定点**を末尾に明記。

## 何を作るか (定義)
「ループエンジニアリングをできる環境」= **自律ループを設計・実行・実験できるハーネス**。
単一の固定ループでなく、**ループ戦略を差し替えて比較・改良できる**枠組み (= loop engineering)。

## 設計の骨格 (調査コーパスからの統合)
### 制御バックボーン = MAPE-K (autonomic computing)
4 段閉ループ + 共有 Knowledge:
- **Monitor**: 現状態収集 (git status / test 結果 / タスク進捗 / リソース)。
- **Analyze**: 目標状態との差 (symptom) 検出。SPC 的な異常検知 (進捗が止まる/逸脱) を併設。
- **Plan**: 次アクション生成。**plan-execute-verify** (ReWOO/Plan-and-Solve) を採用。
- **Execute**: アクション実行 → 結果を Monitor へ (閉ループ)。
- **Knowledge (K)**: 自己/環境/目標/適応知識を周回横断で蓄積 (= AnnotationStore/連結グラフと接続可能)。

### エージェント内ループ = plan → execute → verify + Reflexion
- 各反復で **Chain-of-Verification** (commit/適用前に検証) → 失敗時 **Reflexion** (自己批判して次案)。
- 同一アプローチ N 回失敗で戦略転換 (CLAUDE.md /goal ルールと整合)。

### ★安全層 (ユーザー最優先) — fail-closed
- **Circuit breaker**: 連続失敗/異常で**ループ即停止** (暴走防止)。watchdog-supervisor パターン。
- **Lyapunov/SPC 風の発散検知**: 進捗指標が改善せず発散方向なら停止 (llcore の収縮ゲート思想と同形)。
- **危険操作ゲート**: 削除/push/submodule 改変/DB drop は **constraints に明示が無い限り実行不可**、
  人間 checkpoint 必須 (CLAUDE.md MCP fail-closed 規約)。
- **予算上限**: トークン/反復回数/時間の hard cap。超過で停止。
- **再ログイン/認証要求でループ継続しない** (ccr と同じ規律)。

### 観測可能性 = GitOps reconciliation 風
- desired (goal) vs actual (現状) を毎周回ログ化。各反復の Monitor/Analyze/Plan/Execute と K の差分を記録。
- 監査可能 (重要状態変化はログ)。

### ループ戦略の差し替え (= engineering)
- strategy plugin: `react` / `reflexion` / `plan_execute_verify` / `rewoo` を差し替え可能に。
- 同一タスクで戦略を比較し、どのループが速く/安全に収束するか実験できる (learning-loops の知見を反映)。

## ccr との関係 (並存・非干渉)
- ccr = **長期セッションの起動・自動継続・claude-loop キュー** (人間介在点つき) = 安全網として温存。
- 新環境 = **1 タスク/ゴールに対する MAPE-K 自律ループの実行系**。ccr の claude-loop キューから
  タスクを受けて新環境で回す、という連携も将来可能 (ccr 無改変で上に乗る)。

## 実装方針 (案)
- 言語: Python 3.11 (llcore/FullSense と整合)。重い依存なし、stdlib + 既存資産。
- 配置候補 (★要ユーザー判断): (a) 新規プロジェクト `D:/projects/llloop` (b) llcore 配下
  `src/llcore/loopeng/` (c) fullsense/tools 配下。
- LLM 呼び出し: 既存の Claude/Codex 資産を再利用 (オーケストレータ=Claude, 配下=Codex 可)。
- skeleton: MAPE-K runner + 安全層 + 1 戦略 (plan-execute-verify) + ログ から最小着手、戦略を順次追加。

## ★構築前の決定点 (ユーザー確認)
1. **配置**: llloop 新規 proj / llcore 配下 / fullsense/tools のどれか。
2. **自律範囲**: 新環境が無確認で実行してよい操作の境界 (read/scan/test/commit まで? push は常に人間?)。
3. **ccr 連携**: 当面は完全独立か、claude-loop キューから受ける連携を最初から入れるか。
4. **最初の実証タスク**: 新環境で最初に回す安全なゴール (例: llcore のテスト緑維持 + 小改善ループ)。
