# Stage 3 評価 Harness — 残 3 独自軸 PoC の横断評価設計

**作成**: 2026-05-29 (Stage 3 着手)
**対象**: PoC 2b (persona-indexed × verifier) / PoC 3a (Marabou bridge) / PoC 7a (VNN-COMP 提案)
**目的**: 3 PoC を横断する統一評価軸で「確定独自軸が機構として成立しているか」を falsifiable に判定。llive lleval/lldarwin_v2 の知見を流用しつつ llcore 自前で完結。
**前提**: 進化に上限を設けない (open-ended) 機構 4 種 (適応難易度ゲート / 中立貯蔵庫 / MODES 計器 / MCC カリキュラム) が全 PoC に埋込済。

---

## 評価軸 (5 軸、全 PoC 横断)

### 軸 A: 機構実証 (Mechanism Feasibility)
**問い**: 独自軸の主張する機構が小 PoC で動作するか?
**測り方**: 各 PoC の falsifiable G ゲートの PASS/FAIL 集計 (3 PoC 合計 24 ゲート目安)
**llive 知見流用**: [[feedback_poc_feasibility_first]] (要件→PoC→フィジビリティ→詳細設計の梯子)
**判定**: 全 PoC で G1-Gn の PASS 率 >= 80% かつ critical gate (PoC 2b G3/G7 + PoC 3a G1/G2/G3 + PoC 7a G3) 100%

### 軸 B: 開放端性 (Open-Endedness)
**問い**: 進化に上限を設けない工夫が saturated/neutral regime に陥らず adaptive regime を維持するか?
**測り方**:
- MODES A_new (新規行動採用件数 / 世代) が 90% 世代で >0 (PoC 2b G6)
- 適応難易度 floor が単調非減少 (PoC 2b G5)
- 中立貯蔵庫で persona 絶滅回避 (PoC 2b G3)
- ChangeOp カリキュラムの frontier slope > 0 (PoC 3a G8)
- benchmark spec が無限 ChangeOp 列を許容 (PoC 7a G4)

**llive 知見流用**:
- `poc_evolutionary_activity_modes.py` の honest 発見 「A_new 単独では adaptive↔saturated を分離不能 → A_new + 多様性崩壊 AND gate」
- `poc_minimal_criterion_coevolution.py` の MCC frontier 2.39x 改善
- `lldarwin_v2_poc_marathon_2026_05_26.md` の 12h ラン失敗 (gen5 で 1.0 飽和) の真因 = 固定ものさし

**判定**: 全 PoC で saturated 兆候なし (honest disclosure に記録された逸脱含む) かつ adaptive 維持

### 軸 C: 健全性 (Soundness)
**問い**: 機構の正当性 (verifier の sound / 数学的 invariant / refinement relation) が形式的に保証されるか?
**測り方**:
- PoC 2b: Z3 state_norm gate latency < 10ms (G8)
- PoC 3a: sound 拡張 refinement relation の Z3 unsat 証明 (G1, G2, G3) + 病的 ChangeOp を反例検出 (G4)
- PoC 7a: reference impl が α,β-CROWN baseline と TPR/FPR 公正比較可能 (G3)

**llive 知見流用**:
- `2026-05-28_presurvey_verifier_stack.md` (Agent D) の Marabou Incremental + TorchLean + VNN-COMP 議論
- llive `verifier.py` (EVO-04) の構造 invariants + SMT 層既存実装

**判定**: 全 Z3/refinement gate で sound 証明 (反例なし or 反例が意味のある病的ケース)

### 軸 D: 独自性 (Originality)
**問い**: 各 PoC が先行研究との非自明差別化を保つか?
**測り方**:
- PoC 2b: persona-indexed specialist 集団 × verifier は NAS (単一最良) と差別化される証拠 (集団内 verifier rejection 差別化 = persona が探索を分割)
- PoC 3a: "異なる構造" refinement relation 拡張は Marabou Incremental (同構造内) と差別化される証拠 (ChangeOp 粒度)
- PoC 7a: 新カテゴリ提案が α,β-CROWN 連続 5 年優勝の "固定 network 入力 robustness" と差別化される証拠 (online architecture evolution)

**llive 知見流用**:
- `2026-05-29_core_evolution_master_survey.md` の 7 独自軸 negation work なし verdict
- `feedback_originality_over_imitation` 「採用は網羅でなく選別」

**判定**: 各 PoC で関連研究 3 件以上引用 + 差別化 sharp に表現 (論文 § Related Work 完成度)

### 軸 E: honest disclosure
**問い**: 異常に良い結果は内訳が疑われたか? 留保事項が明示されたか?
**測り方**:
- 各 verdict doc の "honest 留保" section が空でない
- Codex review prompt で confound / Goodhart / artifact のリスクを明示的に問う Q を含む
- proxy vs 実 LLM の境界が明示される (mock 中心、実 LLM は post-llcore phase)

**llive 知見流用**:
- [[feedback_benchmark_honest_disclosure]] (異常な結果は内訳疑う)
- [[feedback_llive_measurement_purity]] (測定純度)
- llive PoC battery 6 要素の横断教訓「単一スカラー指標は誤判定しやすい → AND gate」

**判定**: 全 PoC で honest 留保が記録され、Codex review が confound を問う Q を含む

---

## 評価 harness 実装 (post-Agent 完了時)

3 Agent 完了後、main で以下を実行:

### Step 1: 機構実証集計 (軸 A)
- 各 PoC の `pytest -v` 結果から G1-Gn の PASS/FAIL を抽出
- 統合 verdict (`docs/poc/STAGE_3_VERDICT.md`) に 24 gate 集計表を着地

### Step 2: 開放端性指標集計 (軸 B)
- 各 PoC の verdict doc から MODES A_new / 適応難易度 floor / 中立貯蔵庫 / ChangeOp curriculum の数値を抽出
- saturated 兆候の有無を honest 検査

### Step 3: 健全性検証 (軸 C)
- Z3 timeout / refinement relation sound 証明 / α,β-CROWN baseline 比較可能性を検査
- PoC 1a の 5.8ms 実績を baseline に PoC 2b/3a の verifier latency を比較

### Step 4: 独自性確認 (軸 D)
- PoC 7a 論文 draft の Related Work § で関連研究包含確認
- PoC 2b/3a verdict doc の "独自性 sketch" を確認

### Step 5: honest disclosure 監査 (軸 E)
- 各 verdict doc の honest 留保が空でないか
- Codex review prompt の confound / Goodhart リスク Q が含まれるか

### Step 6: Codex × Claude pair-review 結果統合
- 各 PoC verdict doc 末尾の Codex review prompt template に従い codex exec
- review 結果を verdict doc に追記
- blocker は修正 dispatch、green-light で commit

---

## llive 知見流用一覧 (本 Stage 3 で参照)

| llive 資産 | llcore 流用先 | 性質 |
|---|---|---|
| `pressures.py:AdaptivePercentileGate` | PoC 2b adaptive_floor.py | Read 参照 → llcore 自前 minimal 実装 |
| `lineage_reservoir.py` | PoC 2b lineage_reservoir.py | Read 参照 → llcore 自前 minimal 実装 |
| `quality_diversity.py:FactorSubspaceNovelty` | PoC 2b descriptor 設計 | アイデア参照 |
| `poc_evolutionary_activity_modes.py` | PoC 2b modes_meter.py | Read 参照 → llcore 自前 |
| `poc_minimal_criterion_coevolution.py` | PoC 3a curriculum.py | Read 参照 → llcore 自前 |
| `poc_aurora_descriptors.py` | PoC 7a benchmark spec の行動 descriptor | アイデア参照 |
| `poc_cvt_map_elites.py` | PoC 7a benchmark spec の archive scaling | アイデア参照 |
| `verifier.py` (EVO-04) | PoC 3a refinement relation 設計 | アイデア参照 |
| `2026-05-28_presurvey_verifier_stack.md` (Agent D) | PoC 3a + 7a 関連研究 | doc 流用 |
| `2026-05-29_research_plan_core_evolution.md` | PoC 7a 論文 motivation | doc 流用 |
| `2026-05-28_presurvey_verified_evolution_existing.md` (Agent B) | PoC 7a 関連研究 negation 確認 | doc 流用 |
| `lldarwin_v2_poc_marathon_2026_05_26.md` | 全 PoC 反面教師 (固定ものさし飽和) | doc 流用 |

**規律**: llive コードは Read のみ、import 禁止 (llcore 非依存維持)。アイデア・数式・実装パターンを抽出して llcore 自前で minimal 実装する。

---

## 評価結果着地予定 (Stage 3 完了時)

- `docs/poc/STAGE_3_VERDICT.md` — 24 gate 集計表 + 5 軸評価 + Codex review 結果統合
- `docs/poc/llive_assets_referenced.md` — llive 流用一覧 (本 doc から独立化)
- `docs/papers/vnn_comp_online_arch_evolution_proposal.md` — 論文 draft (TMLR full + GECCO short + NeurIPS workshop 分岐)
- memory 更新: `project_llcore_init_2026_05_29.md` に Stage 3 結果追記
- 関連 memory: 必要なら `feedback_open_ended_mechanism_must` 新規作成 (open-ended 4 機構の必須性が確立した場合)
