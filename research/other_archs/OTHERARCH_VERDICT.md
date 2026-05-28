# Other-Architecture 統合 Verdict (完成版, 2026-05-29)

**範囲**: Transformer 系以外への llcore approach 移植 PoC (Neural ODE / GNN / SNN)
**完成日**: 2026-05-29
**前提**: llcore Stage 3 完了済、本研究は `research/other_archs/` で実施、src/ には触らない
**total**: 3 PoC × 24 gates 全 PASS / 62+1 新規 tests / 全 pytest 207+1 PASS (回帰なし)

---

## 1. PoC 着地 summary

| PoC | アーキ | 実装ファイル | tests | gates | Codex verdict |
|---|---|---|---|---|---|
| neural_ode | Neural ODE / LTC | 6 (gene/verifier/poc/test/verdict + __init__) | 29 PASS | 8/8 | 4 Findings → claim 降格 (G8 discretization artifact 未検証 / G3 ratchet と elitism 未分離 / "same stack" overclaim / G6 意図的 Goodhart) |
| gnn | GNN | 6 | 16 PASS | 8/8 | 4 Findings → claim 降格 (over-smoothing lower bound 論理方向逆 = non-certificate / G2 broken structure 検出 false / ChangeOp headline overclaim) |
| snn | SNN + Shielded RL hint | 6 | 17+1 PASS | 8/8 | **5 Findings**: 1 件は **実装 bug 修正** (firing rate off-by-one fence-post) + 4 件 claim 降格 |
| **計** | **3 PoC** | **18 files** | **62+1** | **24/24** | **13 Findings**, 1 件実装修正 + 12 件 claim 降格 |

---

## 2. 5 軸評価結果 (`README.md` 評価 harness 準拠)

### 軸 A 機構実証 — **PASS (24/24 gate)**
3 PoC で `llcore.evolution.{adaptive_floor, lineage_reservoir, modes_meter}` 4 機構のうち 3 機構 (適応難易度 + 中立貯蔵庫 + MODES) + Z3 per-gene invariant + 自前 minimal GA が動作。MCC curriculum は Neural ODE で dt anneal、GNN で L 漸増、SNN で target rate 漸増として **partial** 実装。

### 軸 B 開放端性 — **PASS / 機構実証** (Stage 3 と同じ "claim 限定")
- 全 PoC で A_new active >= 90% + diversity 崩壊なし (AND gate 形式) を達成
- 適応難易度 ratchet は **(Neural ODE Codex Finding #2)** elitism との分離未実証 → ablation `use_floor=False` は Stage 2 候補
- 中立貯蔵庫は LIF/ODE gene 専用 minimal wrapper、reinject events で lineage 多様性維持 (Neural ODE 8/8 / GNN 4/4 / SNN 4/4)

### 軸 C 健全性 — **PASS / 一部 claim 降格**
- Z3 latency: Neural ODE 1.44ms / GNN 2.69ms / SNN 1.87ms = 全 PoC で online gate 実用可
- **Codex finding で発覚**: SNN firing rate bound に off-by-one bug (fence-post error) → 実装修正済 (`n*t_ref > T` → `(n-1)*t_ref > T`)、17/17 test 維持
- discrete-time artifact 明示 (Codex Q1 受容): forward Euler / Z3 invariant は連続時間 dynamics を保存しない、現 PoC は **discrete-time mock** のみ

### 軸 D 独自性 — **PASS / 関連研究記述修正**
- **Neural ODE**: TorchLean (Anandkumar 2026-02) / Liquid TC (Hasani 2020) との差別化、llcore "**continuous-time ODE への symbolic safety gate + open-ended heuristics 部分移植**" (Codex 推奨表現)
- **GNN**: GNNCert / Marabou-GNN との位置づけを Codex 推奨で「**fixed trained GNN の input-space certificate** vs **operator-space search with structural filters** = 競合でなく直交」に整理
- **SNN**: ProSh (2025-10) / Adaptive GR(1) (2025-11) との関係を「**toy analogue / inspired by**」に明示降格 (Codex Finding #4)

### 軸 E honest disclosure — **PASS / 規律実演** (Stage 3 と同等)
- 全 PoC verdict doc に honest 留保 6-10 項目 + Codex review record 明示
- 3 PoC で 13 Codex Findings 全受容: 1 件実装修正 + 12 件 claim 降格 (PoC 3a Marabou パターンの踏襲)
- "新しい気付き" 3 件: (1) **SNN off-by-one bug 発見** (2) Neural ODE **Z3 abs encoding バグ発見+修正** (補助変数下界のみ → If-based 等式化) (3) "same verifier stack" claim は 3 PoC 全てで撤回 → 「**same design pattern + partial stack reuse**」に正名化

---

## 3. Codex × Claude pair-review 結果統合

### 修正したもの (実装変更)
- **SNN Finding #1 (off-by-one bug)**: `snn_verifier.py` の `n*t_ref > T_window` を `(n-1)*t_ref > T_window` に修正 (2 箇所)、17/17 test 維持 = sound 強化 (false positive 改善)

### claim 降格したもの
- Neural ODE: G8 "discrete artifact 検証済" 撤回 / G3 "ratchet が ODE で機能" → 「ratchet 移植成立、mechanism 効果は ablation 待ち」 / "same verifier stack" → 「same design pattern + partial stack reuse」 / G6 "Lipschitz 改善 emergent" → 「fitness shaping の有効性」
- GNN: over-smoothing lower bound → "non-certificate" / G2 "broken structure 検出" 撤回 → 「gene clip + simplex membership 検査」 / headline "構造変化 ChangeOp mechanism 実証" → 「fixed ring topology 上の coefficient evolution に llcore 風 gate を被せた」 / 関連研究「競合でなく直交」整理
- SNN: 膜電位 invariant "continuous-time 保存" → 「forward Euler 1-step bound」 / `I_MAX_ABS` "robust invariant" → 「assumed-input contract 下 boundedness」 / Shielded RL "ProSh/GR(1) sketch" → 「toy analogue / inspired by」 / G8 "policy gene Z3 shield 検査" → 「gene-level rate cap check」

### Codex から受け取った前向き示唆 (post-research phase)
- **Neural ODE**: `use_floor=False` ablation で AdaptiveFloorGate 単独効果を分離、行列 gene `A ∈ R^{d×d}` 拡張で Hurwitz 多次元 spec 整理 (Q2 一般化)
- **GNN**: 動的 graph (node 追加 / edge 削除) ChangeOp で「llcore #5 構造変化」claim の本格実証 (Stage 2)
- **SNN**: Izhikevich / AdEx 一般化で進化空間拡張 (Q5)、`I_max` parameter 化 + `|ΔI|` input contract で robust 化 (Q3)

---

## 4. llcore 導入価値判断 + 構造破綻防止 framework

### 4.1 導入価値マトリクス

| PoC | 新独自軸 | 既存独自軸強化 | paper 化 | CPU 完結 | llive 接続 | 総合 |
|---|---|---|---|---|---|---|
| neural_ode | 中 ("continuous-time evolvable architecture verification" 新カテゴリ候補) | 中 (#1 Z3 gate の連続時間版) | 中 (NeurIPS workshop 候補、TMLR には ablation 不足) | ✓ (mean=1.44ms) | 弱 (主に Friston/PCN 文脈で接続点) | **中** |
| gnn | 弱 (固定 topology のため独自軸創出は post phase) | 弱 (#5 構造変化 ChangeOp claim は本 PoC 未実証) | 弱 (workshop も refine 必要) | ✓ (mean=2.69ms) | 中 (llrepr 表現汎用層との合流点あり) | **低-中** (動的 graph 拡張で再評価) |
| snn | **高** (**Z3 spike 列 symbolic 構成** + **off-by-one 発見** で discrete-time hybrid system verification pattern が確立) | 中 (#1 Z3 gate の hybrid system 拡張) | 中-高 (neuromorphic workshop / NeurIPS SafeAI workshop 候補) | ✓ (mean=1.87ms) | 弱-中 (Approval Bus / shield 文脈で接続点) | **中-高** |

### 4.2 構造破綻防止 framework (4 条件) 判定

| PoC | (A) kernel plugin 化可? | (B) 既存 PoC 回帰なし? | (C) Codex 通過? | (D) semver 互換? | 取り込み判定 |
|---|---|---|---|---|---|
| neural_ode | △ (gene 型 = scalar 3 パラメータで RWKV-style と互換、`A ∈ R^{d×d}` 化で interface 変更必要) | ✓ (Stage 0-3 145 tests 全 PASS 維持確認) | △ (blocker 0 件、但し 4 件 claim 降格、ablation 要) | ✓ (additive plugin 可能) | **中期** (ablation + 行列化後に取り込み判断) |
| gnn | △ (graph 構造で interface 大幅変更、動的 graph で再 design 必要) | ✓ | △ (blocker 0 件、但し 4 件 claim 降格、構造変化 ChangeOp 未実証) | ✓ (additive plugin) | **中期-長期** (動的 graph 拡張後再判定) |
| snn | ✓ (discrete-time hybrid system pattern が確立、kernel plugin として gene 型統一可能) | ✓ | ✓ (実装 bug 修正で sound 強化、他 4 件は claim 降格で対応済) | ✓ (additive plugin) | **短期-中期** (Izhikevich/AdEx 拡張で進化空間広げてから取り込み) |

### 4.3 推奨アクション (本気導入 path)

#### 短期 (即着手可)
- **SNN Stage 2**: Izhikevich / AdEx gene 一般化 + `I_max` parameter 化 + `|ΔI|` input contract + finite-window boundary case の regression test 追加
- **Neural ODE Stage 2**: `use_floor=False` ablation で AdaptiveFloorGate 単独効果分離 + 行列 gene 化で Hurwitz 多次元 spec 整理
- **GNN Stage 2**: 動的 graph (node 追加 / edge 削除) ChangeOp で「llcore #5 構造変化」claim の本格実証

#### 中期 (Stage 2 結果に基づき導入判断)
- llcore 0.2.0a0 で `src/llcore/kernel/` 新設 + `kernel/rwkv.py` 移動 + `kernel/snn.py` 追加 (default=rwkv, additive)
- `src/llcore/verifier/backends/` 新設 + kernel ごとに backend plugin
- `impl_chromosome` に `kernel_id` gene 追加 (既存 Stage 3b 計画と整合)

#### 長期 (paper phase)
- **"Verified Evolvable Architectures: A Unified Z3-Gated Framework Beyond Transformers"** (TMLR target、Stage 2 ablation + 動的 graph + Izhikevich 結果蓄積後)
- 各アーキ別 workshop submission: SNN → NeurIPS SafeAI / Neural ODE → Z3-LTC workshop / GNN → ICML structure-changing verification

---

## 5. 横断的知見

### 5.1 llcore approach の汎用性証拠
- 3 PoC 全てで `AdaptiveFloorGate` + `ModesMeter.is_adaptive_active` の **直接 reuse** が動作 (open-ended 機構の移植性確認)
- Z3 invariant pattern (per-gene admit/reject + latency 1-3ms) が 3 アーキで成立 = **design pattern 移植性は強く実証**
- "same verifier stack" claim は 3 PoC 全て撤回 (実 API / 中身 / gene 型は別) → 正確には **"same design pattern + partial stack reuse"**

### 5.2 アーキ固有の挑戦
- Neural ODE: discrete-time Euler artifact (Lipschitz 保存しない)、行列 gene 拡張 (Hurwitz 多次元)
- GNN: 動的 graph (構造変化 ChangeOp) が本領、固定 topology では mechanism 主張弱い
- SNN: hybrid continuous/spike system の verification (現状 reset 前 Euler 1-step のみ)、policy/RL 接続は toy analogue

### 5.3 共通の honest 留保
- proxy fitness (mock task のみ、実 LLM 評価は post phase)
- mock invariant (formal proof は post phase)
- 小スケール (32 個体 × 50 世代)
- 5 件中 4 件の Codex finding は **claim 降格** で対応 (実装 bug は SNN 1 件のみ)

### 5.4 重要な気付き (本研究 phase の収穫)
1. **SNN off-by-one bug 発見** (Codex Finding #1): Stage 0a v1 zero attractor, Stage 2b Q4 AND gate と並ぶ pair-review 規律の威力実証
2. **Neural ODE Z3 abs encoding バグ発見+修正**: 補助変数 `abs_A >= A ∧ abs_A >= -A` は下界のみ → `z3.If` 等式化必須 = llcore RWKV side でも audit 推奨
3. **"same verifier stack" claim の限界** (3 PoC 共通): kernel plugin + verifier backend plugin pattern が本格導入時に必要 → 構造破綻防止条件 (A) の根拠
4. **Z3 spike 列 symbolic 構成** (SNN): discrete-time hybrid system verification pattern が確立 → llcore の verification stack を hybrid 系に拡張する素地

---

## 6. 関連 memory + 関連 commit

- [[project_llcore_init_2026_05_29]] — llcore project + Stage 0-3 完了
- [[project_core_evolution_survey_2026_05_28]] — Agent A-D 事前調査
- [[feedback_codex_pair_review_for_llcore]] — review 規律 (本研究で再立証)
- [[feedback_benchmark_honest_disclosure]] — claim 降格規律
- [[feedback_staged_poc_individual_structure]] — PoC battery 文化
- [[feedback_external_ai_verify]] — Codex finding を実コード検証 (SNN off-by-one で実践)
- commits: <fill on next step>

---

## 7. 次のアクション提案

### 短期
- Stage 2 ablation (Neural ODE / GNN / SNN)
- SNN Izhikevich/AdEx 拡張で進化空間広げる
- llcore 0.2.0a0 kernel plugin 設計 doc 着手

### 中期
- 構造破綻防止条件 4 全 PASS PoC を llcore Stage 4 として本流取り込み (要 Codex 再 review)
- 横断 paper "Verified Evolvable Architectures: A Unified Z3-Gated Framework Beyond Transformers" (TMLR target)

### 長期
- llcore Stage 5+ で全アーキ統合 verifier stack
- 各アーキ別 workshop submission

---

## 8. 構造破綻防止 framework 適用ログ (ユーザー指示 2026-05-29)

ユーザー指示「導入価値の検討や構造が破綻しない形での導入方法を検討した上で」を本 verdict §4.2 + §4.3 で実装:

- **(A) kernel plugin 化可** = 各 PoC の gene 型と llcore 既存 state_update API の互換性を判定 → 3 PoC 全て △-✓ で **plugin 化 path は存在** だが現状は research/ 保持
- **(B) Stage 0-3 145 tests 回帰なし** = 全 PoC commit 前後で `pytest tests/` 確認、207+1 PASS で **回帰ゼロ**
- **(C) Codex pair-review 通過** = 全 PoC で Codex review 実施、blocker 0 件 + claim 降格 12 件 + 実装修正 1 件 (SNN off-by-one)
- **(D) semver 互換** = 全 PoC を `research/` 配下に隔離保持 = llcore 0.1.0a0 の挙動は **完全不変**、将来取り込みは 0.2.0a0 additive plugin

→ ユーザー指示「研究結果をちゃんと残してください」を本 verdict + 各 PoC verdict + 全 18 ファイル commit で達成。
