# llcore research — Other Architectures (Transformer 以外への移植 PoC)

**作成**: 2026-05-29 (Stage 3 完了後、ユーザー Goal「他のアーキテクチャでも試してみましょう」を受けた研究 phase 着手)
**位置付け**: **llcore 研究記録**。**src/ には触らない**。導入価値が証明されたら将来 llcore 本流に取り込む判断材料。
**動機**: llcore の核心 = "core algorithm を低次元 gene 化 + Z3 invariant gate + 進化 + open-ended 機構" は Transformer 系 (RWKV-style) で Stage 0-3 完了 (5/7 確定独自軸 mechanism 実証, 145 tests / 24 gates)。これが Transformer 固有ではなく **(a) invariant が形式記述可能 + (b) parametric ChangeOp 可能 + (c) CPU mock 可能** の 3 条件を満たすアーキならどれにも展開できる、という仮説の検証。

---

## 適用条件 (3 ヶ条)

llcore approach が他アーキで成立するための必要条件:

1. **invariant の形式記述可能性**: Lipschitz / stability / capacity / equivariance / firing rate bound などが数式で書け、Z3 SMT で symbolic 検査可能
2. **parametric ChangeOp の定義可能性**: アーキの構造変化 (parameter shift / op 置換 / 層追加削除) が低次元 gene 上の操作として表せる
3. **CPU mock の成立**: 小スケール instance (小 graph / 少 neuron / 短 trajectory) で feasibility 検証が回せる

---

## 3 並列 PoC (本 phase)

| Arch | 場所 | gene 化対象 | Z3 invariant | 相性 | 既存 llcore との接続 |
|---|---|---|---|---|---|
| **Neural ODE / LTC** | `neural_ode/` | vector field 係数 (A, W, b) | Lipschitz 上界 + Hurwitz stability | ★★★ | Stage 0a (RWKV-style 離散時間) の **連続時間版**, Stage 1a の Z3 state_norm の **連続時間 Lipschitz 版** |
| **GNN** | `gnn/` | message passing op (aggregation + update) | over-smoothing lower bound + permutation equivariance | ★★★ | 独自軸 #5 (Marabou "異なる構造" 拡張) の **structure-changing ChangeOp 自然 fit** |
| **SNN + Shielded RL** | `snn/` | LIF neuron model (τ_m, V_th, V_reset, t_ref) | firing rate ≤ 1/t_ref + 膜電位 bounded + Shielded RL hint | ★★ | Codex Q5 推奨 ProSh / Adaptive GR(1) shielding と verifier 統合 sketch |

それぞれ:
- `gene.py` (アーキ固有 gene 構造)
- `verifier.py` (Z3 invariant)
- `poc.py` (G1-Gn gate runner, 単独実行可)
- `test_*.py` (pytest 10+ tests, gate ごと)
- `verdict.md` (命題 / 結果 / Codex review prompt / honest 留保)

---

## llcore 既存資産の流用方針

**llcore.evolution.* は import OK** (本研究は llcore 内 directory):

| llcore 資産 | 用途 |
|---|---|
| `llcore.evolution.adaptive_floor.AdaptiveFloorGate` | 適応難易度 ratchet |
| `llcore.evolution.lineage_reservoir.LineageReservoir` | 中立貯蔵庫 |
| `llcore.evolution.modes_meter.ModesMeter` | A_new + diversity AND gate |
| `llcore.verifier.invariants.is_z3_available` | Z3 環境確認 |
| `llcore.evolution.minimal_ga` | 自前 minimal GA パターン |

**自前 minimal 実装** (各アーキ専用):
- アーキ固有 gene 構造 (vector field / message passing / LIF model)
- アーキ固有 invariant (Lipschitz / over-smoothing / firing rate)
- ChangeOp + selection + fitness proxy
- Codex review prompt template

**llive Read のみ**: import 禁止 (llcore 一貫性維持)、パターン参照のみ。

---

## 評価 harness (5 軸, Stage 3 と同じ)

### 軸 A 機構実証 (PASS 条件: 全 gate PASS + critical gate 100%)
- 各 PoC G1-Gn の PASS/FAIL 集計
- critical gate: 各 PoC の Z3 invariant 検査 + 進化 fitness 改善

### 軸 B 開放端性 (PASS 条件: saturated/neutral に陥らず adaptive 維持)
- MODES A_new active >= 90% + diversity 崩壊なし AND gate
- 適応難易度 ratchet 単調非減少
- 中立貯蔵庫 lineage 多様性

### 軸 C 健全性 (PASS 条件: Z3 sound + invariant 反例検出)
- 各アーキの Z3 invariant が病的 gene を反例検出
- Z3 latency < 10ms / call

### 軸 D 独自性 (PASS 条件: 関連研究との sharp 差別化)
- Neural ODE: TorchLean (Anandkumar 2026-02) / Liquid TC (Hasani 2020) との差別化
- GNN: GNNCert (Wang 2021) / Marabou-GNN (Sälzer 2023) / Modular Robustness 研究との差別化
- SNN: ProSh (2025-10) / Adaptive GR(1) (2025-11) / Loihi/TrueNorth context との差別化

### 軸 E honest disclosure (PASS 条件: 留保明示 + Codex confound Q 包含)
- 各 verdict の honest 留保が空でない
- proxy vs 実 task の境界明示
- Codex pair-review 結果を verdict に追記

---

## llcore 導入価値の判断軸 (本研究の goal)

各 PoC が完了したら以下で **llcore 本流への導入価値** を判断する:

| 判断項目 | 高価値 | 中価値 | 低価値 (研究記録のみ保持) |
|---|---|---|---|
| **新独自軸の創出** | 新カテゴリ提案論文化可 | 既存独自軸の拡張 | 既存と重複 |
| **既存独自軸の強化** | #5 ChangeOp 構造変化の真の実装 | #1 Z3 gate の応用幅拡張 | 主張範囲狭い |
| **paper 化価値** | TMLR / NeurIPS workshop 直接候補 | GECCO short / 補助 | workshop 弱い |
| **CPU 完結維持** | CPU で完全動作 | optional GPU で 5x | GPU 必須 |
| **llive 知見との接続** | llive 表現汎用層 (llrepr) / VLM と直結 | llive 補助 | 接続なし |

---

## 出力

各 PoC 完了後、本 dir 直下に統合 verdict `OTHERARCH_VERDICT.md` を着地。導入価値判断 + Codex pair-review 結果 + 次フェーズ提案を含む。

---

## 関連 memory

- [[project_llcore_init_2026_05_29]] — llcore project 発足 + Stage 0-3 完了
- [[project_core_evolution_survey_2026_05_28]] — Agent A-D 事前調査 (本研究の素地)
- [[feedback_codex_pair_review_for_llcore]] — review 規律 (本研究でも継続)
- [[feedback_benchmark_honest_disclosure]] — claim 降格規律
- [[feedback_staged_poc_individual_structure]] — PoC battery 文化
