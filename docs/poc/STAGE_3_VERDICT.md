# Stage 3 統合 Verdict — 残 3 独自軸 PoC + Codex pair-review 完走

**Stage 3 範囲**: 確定独自軸 7 つ中 残 3 つ (#4 persona-indexed × verifier / #5 Marabou bridge / #7 VNN-COMP 新カテゴリ)
**作成**: 2026-05-29 (Agent dispatch + Codex pair-review + claim 修正後 完成版)
**前提**: Codex × Claude pair-review 規律 + 進化に上限を設けない工夫 (open-ended 4 機構: 適応難易度 / 中立貯蔵庫 / MODES 計器 / MCC カリキュラム) を全 PoC に埋込

---

## 1. PoC 着地 summary

| PoC | Stage | 実装ファイル | tests | gates | Codex verdict |
|---|---|---|---|---|---|
| 2b | persona-indexed × verifier (独自軸 #4) | 7 file (priors/adaptive_floor/reservoir/modes_meter/poc/test/verdict) | 26 PASS | 8/8 | wording (Q1/Q5/Q6) + blocker 1 件 (Q4 AND gate 未実装 → 修正済) |
| 3a | Marabou bridge skeleton (独自軸 #5) | 7 file (changeop/refinement/curriculum/poc/test/verdict/paper) | 26 PASS | 9/9 | 全 7 Q で gap 指摘 → **claim 範囲を mechanism feasibility / sketch に降格** |
| 7a | VNN-COMP 新カテゴリ提案 (独自軸 #7) | 6 file (paper/spec/ref-spec/poc/test/verdict) | 17 PASS | 7/7 | **positive 評価** ("proposal の核は強い") + Findings 4 件 (parser/sat-witness 未実装 等) |
| **計** | **3 PoC** | **20+ files** | **69 PASS** | **24/24** | 5/5 Green-light path (修正 / 降格 / 正式承認 で完了) |

**+既存**: Stage 0-2 既存 76 tests + Stage 3 新規 69 tests = **計 145 tests / 全 PASS**。Stage 3 着地後の回帰検出も baseline (76 → 145) で OK。

---

## 2. 5 軸評価結果 (`docs/eval/STAGE_3_EVAL_HARNESS.md` 基準)

### 軸 A 機構実証 — **PASS (24/24 gate)**
- PoC 2b: G1-G8 全 PASS (kernel coverage / verifier rejection diff / persona 生存 / fitness 単調 / floor ratchet / A_new active / 全滅回避 / verifier latency)
- PoC 3a: G1-G9 全 PASS (Z3 両方向判定 / 合成性 / 100-step bound / 病的反例検出 / Marabou 包含 sketch / curriculum / Z3<100ms / frontier slope / mock 完走)
- PoC 7a: G1-G7 全 PASS (論文 8198 word / spec 曖昧性なし / ref impl 動作 / 上限なし scoring / α,β-CROWN 差別化 sharp / 関連研究包含 / honest disclosure)
- critical gate (PoC 2b G3/G7 + PoC 3a G1/G2/G3 + PoC 7a G3) すべて PASS、PASS 率 100%

### 軸 B 開放端性 — **PASS (機構実証) / 上限なし claim は限定**
- 機構実証 OK: 3 PoC 全てで MODES / 適応難易度 / reservoir / ChangeOp curriculum 各機構が動作
- **claim 限定** (Codex finding 受容):
  - PoC 2b ratchet は **bounded fitness 上の monotone threshold** であり真の "上限なし" でない (Codex Q2)。50 世代スケールでは fitness hard cap 未到達のため別固定点未出現。**"上限なし" は ratchet + reservoir + MODES の組合せ機構**で初めて成立
  - PoC 3a curriculum は `magnitude_cap` で明示上限あり (Codex Q4)。frontier 上昇 (0.032→0.800) は機構実証、1000 世代 long-run falsify は別 PoC
  - PoC 7a benchmark spec は MODES 閾値が hand-set (Codex Q8)、yearly continuity は rotating families / withheld seeds 追加で強化

### 軸 C 健全性 — **PASS**
- PoC 2b: Z3 state_norm gate mean=6.07ms (PoC 1a baseline 5.8ms と整合)、smoke は warm-up 不足で 10-15ms (honest 留保)
- PoC 3a: Z3 per-step max=2.2ms / mean=1.4ms (100-step total 146.9ms = budget 内)
- PoC 7a: ref impl per-step max=5.8ms = baseline 整合、500ms budget の 1.2%
- ただし: PoC 3a の **"sound 拡張 refinement relation R" formal proof は未** (Codex Q1) → sketch のみ、post phase で formal proof

### 軸 D 独自性 — **PASS / 一部 claim 修正**
- PoC 7a 関連研究 13 件本文 + Appendix A 11 件 cross-check + Codex Q5 推奨 4 件 (verified RL / adaptive shielding) 追加要
- 既存 VNN-COMP との差別化 sharp: paper §1.1 Query A/B/C + §2.12 gap 表
- **claim 修正**: PoC 3a "Marabou ⊂ llcore 包含" は **完全撤回** (Codex Q5: 型違いで包含と言えない、Marabou=query subset / llcore=behavioral inequality)。代替 = 「両者は異なる型の refinement、両方を有する verifier stack が将来研究」

### 軸 E honest disclosure — **PASS / 規律の実演**
- 3 PoC 全 verdict doc に honest 留保章 (PoC 2b 10 項目 / PoC 3a 6 項目 + 7 Q 降格 / PoC 7a 12 項目 + Codex 4 Findings)
- Codex review prompt 全 PoC に埋込済、確実に confound / Goodhart / artifact を問う Q を含む
- pair-review 規律 [[feedback_codex_pair_review_for_llcore]] が機能した実証:
  - **Stage 0-2 (5 PoC)**: 5 件中 4 件で Claude 単独見落とし → Codex 検出
  - **Stage 3 (3 PoC)**: 3 件中 3 件で Codex が claim と実装のズレを指摘
    - PoC 2b Q4: AND gate 未実装 → 実装修正で対応
    - PoC 3a Q1-Q7: 全 Q で gap → claim を mechanism feasibility / sketch に **降格** ([[feedback_benchmark_honest_disclosure]] 規律で対応)
    - PoC 7a Q1-Q8: positive 評価 + 4 Findings → honest 留保拡張 + venue 戦略確定

---

## 3. Codex × Claude pair-review 結果統合

### 修正したもの (実装変更)
- PoC 2b Q4: `modes_meter.is_adaptive_active(require_no_diversity_collapse=True)` 追加、`gate_g6_a_new_active` を AND gate に切替、test 2 件追加 (26/26 PASS)、verdict G6 数値 AND gate ベースで再走 PASS

### claim 降格したもの (verdict 修正)
- PoC 3a Q1-Q7: "sound 拡張 refinement relation R" formal claim → **数式 sketch のみ** に降格。"Marabou ⊂ llcore" → **完全撤回**。論文素材 `marabou_sound_extension_sketch.md` を **workshop position paper / idea sketch** に honest 降格 (TMLR full submission は post-llcore phase)

### honest 留保拡張したもの (verdict 追記)
- PoC 2b Q1/Q2/Q5/Q6: persona prior strict identifiability なし / ratchet は bounded 上の monotone threshold / convex hull claim は AABB proxy / verifier rejection は confounded を留保 §8-10 に追記
- PoC 7a Findings 1, 2: parser missing / sat witness missing を honest 留保 §11, 12 に追記

### Codex から受け取った前向き示唆 (post-llcore phase)
- PoC 7a venue 優先順位確定: **TMLR > NeurIPS workshop > GECCO short**、現原稿で **NeurIPS workshop submission は可能水準**、TMLR は parser + sat witness 実装後
- PoC 7a 追加すべき adjacent work 4 本 (Runtime Safety through Adaptive Shielding 2025-05-20, ProSh 2025-10-17, Adaptive GR(1) shielding repair 2025-11-04, The Effect of Architecture During Continual Learning 2026-01-27)

---

## 4. 確定独自軸 7 つ 最終 status

| # | 独自軸 | Stage 2 まで | Stage 3 | 最終 |
|---|---|---|---|---|
| 1 | ChangeOp → Z3 online gate | mechanism 実証 (1a) | refinement の **数式 sketch** 拡張 (3a) | ✓ mechanism 実証 + refinement sketch |
| 2 | state update gene 化 (RWKV-style) | mechanism 実証 (0a v2) | — | ✓ mechanism 実証 |
| 3 | factor_hook 認知駆動 Δ | mechanism 実証 (2a mock) | — | ✓ mechanism 実証 (mock) |
| 4 | **persona-indexed × verifier** | 基盤 (0c + 1a) | mechanism 実証 (2b, AND gate 修正後 8/8 PASS) | ✓ **新規 mechanism 実証** |
| 5 | **Marabou Incremental "異なる構造" 拡張** | 素地 | **sketch + skeleton** (3a, Codex で claim 降格) | △ **mechanism feasibility / sketch のみ** (formal proof は post phase) |
| 6 | Lipschitz/Hurwitz invariants | state_norm 着地 (1a) | — | △ state_norm のみ、Lipschitz は post phase |
| 7 | **VNN-COMP 新カテゴリ** | 提案論文素材 | **paper draft 8198 word + spec + ref impl** (7a, Codex で venue 確定 + workshop 可能水準) | ✓ **新規 mechanism 実証 + workshop submission 可能** |

**結論**: 7 つ中 **5 つで mechanism 実証完了** (Stage 0-3)。**残 2 つ (#5 Marabou / #6 Lipschitz) は post-llcore phase 候補**。Stage 0-2 完成宣言時 "4/7" → Stage 3 完了で "5/7 + 1 sketch"。

---

## 5. push 状態 / commit 計画

- 現在 (Stage 3 完了直前): ローカル 6 commits (Stage 0-2 完成宣言まで)、GitHub push 未
- Stage 3 commit 計画 (main が atomic で実施、`docs/poc/llive_assets_referenced.md` 統合分含む):
  1. `feat(poc-2b): persona-indexed × verifier with open-ended guards (Stage 2b) + Codex Q4 AND gate fix`
  2. `feat(poc-3a): Marabou bridge skeleton + Z3 mock refinement (Stage 3a, claim-降格 per Codex)`
  3. `feat(poc-7a): VNN-COMP new category proposal + benchmark spec + ref impl (workshop-ready)`
  4. `docs(stage-3): 統合 verdict + 評価 harness 5 軸 + Codex pair-review record`

push 解禁はユーザー指示後 (現状ローカル保持方針)。

---

## 6. 次のアクション候補 (Stage 3 完了後)

### 短期 (即着手可)
- **PoC 7a NeurIPS workshop submission**: abstract framing 修正 + adjacent work 2 本追加で原稿提出可
- **PoC 2b Stage 2c**: 公平 verifier-rate 比較 (fresh sample 固定 + 進化外で verifier 当てる) で Codex Q6 confound を分離
- **PoC 3a Stage 3a-v2**: ε(c) の数式を formal sound proof で再導出、合成性 G2 を **implication encoding** で証明、Marabou 包含 sketch を撤回した代替 "両者の型の違い" doc に書き直し

### 中期 (GPU / parser / Marabou native 必要)
- PoC 7a TMLR full submission: `.onnx` / `.vnnlib` parser 実装 + `sat` witness 実装 + 10-instance pilot
- PoC 3a Marabou native install (Docker) + α,β-CROWN incremental wrapper
- Stage 3b kernel 多様化 gene (rwkv/mamba/hopfield/linear-attn)
- Stage 4 学習則 (FF/EP/PCN/Hebb) を impl_chromosome gene 化

### 長期 (paper phase)
- GECCO 2027 short paper (10-instance pilot 結果蓄積後)
- TMLR full submission (parser + sat witness 実装後)
- Lipschitz/Hurwitz invariants の Z3 拡張 (確定独自軸 #6)

---

## 関連 memory (本 verdict 関連)

- [[project_llcore_init_2026_05_29]] — llcore project 発足 + Stage 0-2 完成宣言
- [[project_core_evolution_survey_2026_05_28]] — Agent A-D + RAD 14 分野事前調査
- [[feedback_codex_pair_review_for_llcore]] — review 規律 (Stage 3 で更に立証)
- [[feedback_benchmark_honest_disclosure]] — claim 降格規律の根拠 (PoC 3a Stage 3)
- [[feedback_staged_poc_individual_structure]] — PoC battery 文化
- [[feedback_external_ai_verify]] — Codex finding を実コード検証して採用 (PoC 2b Q4 修正で実践)
