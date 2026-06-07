# llcore Architecture Landscape — アーキ体系俯瞰

**作成**: 2026-05-29
**目的**: llcore で試した各種アーキテクチャを体系的に整理し、**ユーザー (古瀬さん) が全体像を把握するための材料**として 1 doc に集約。Codex pair-review で claim 降格された範囲を honest に明示。
**読者**: ユーザー (主) + Claude 次セッション + 将来の co-author / reviewer

---

## 1. llcore の核心 (3 文要約)

llcore は **Transformer のコアアルゴリズム (state update / 学習則 / 認知駆動 Δ) に進化形態を与え、健全な検証ゲートで破綻させずに異アルゴリズムへ進化させる研究フレームワーク**。CPU 完結。本線 paper のゲートは閉形式 sound contraction certifier ladder (README「看板の honest 訂正」参照)、Z3/SMT は他アーキ track の invariant 検査で使用。**「同じ design pattern (gene 化 + sound invariant gate + 進化 + open-ended 機構) が複数アーキで成立するか」**を体系的に検証している。

### 核心 4 要素
1. **gene 化**: アーキ固有の core algorithm を低次元 (3-10 params) で表現
2. **sound invariant gate**: state_norm / Lipschitz / firing rate / over-smoothing 等の安全 invariant を per-gene 検査 (本線は閉形式収縮 certifier、他アーキ track は Z3/SMT)
3. **進化**: 自前 minimal GA (llive 非依存) で探索
4. **open-ended 機構** (4 種): 適応難易度ゲート / 中立貯蔵庫 / MODES 計器 / MCC カリキュラム → 「進化に上限を設けない」

---

## 2. アーキ適用条件 3 ヶ条

| 条件 | 意味 |
|---|---|
| **invariant の形式記述可能性** | Lipschitz / stability / capacity / equivariance / firing rate bound 等が数式で書け Z3 SMT で symbolic 検査可能 |
| **parametric ChangeOp の定義可能性** | アーキの構造変化 (parameter shift / op 置換 / 層追加削除) が低次元 gene 上の操作で表せる |
| **CPU mock の成立** | 小スケール instance (小 graph / 少 neuron / 短 trajectory) で feasibility 検証が回せる |

3 条件全て満たすアーキは llcore approach を **部分移植**できる。実証済 5 アーキ。

---

## 3. アーキ別 status マップ

### 3.1 一覧表

| アーキ | 場所 | 段階 | tests | Z3 latency | Codex 主要 finding | llcore 導入価値 |
|---|---|---|---|---|---|---|
| **RWKV** (本流) | `src/llcore/` | Stage 0-3 完了 | 145 | 5.8 ms | 5 PoC × 4 件 Claude 単独見落とし | **本流** (mechanism 実証完了) |
| **Neural ODE / LTC** | `research/other_archs/neural_ode/` | Stage 1 | 29 | 1.4 ms | abs encoding バグ発見、G8 discretization 未検証 | 中 (workshop 候補) |
| **GNN (固定 ring)** | `research/other_archs/gnn/` | Stage 1 | 16 | 2.7 ms | over-smoothing lower bound 論理逆 (unsound) | 低-中 |
| **GNN (動的 graph)** | `research/other_archs/gnn/dynamic_graph/` | 🔄 Agent 進行中 | - | - | - | (評価待ち) |
| **SNN-LIF** | `research/other_archs/snn/` | Stage 2.1-2.2b | 29 | 1.9 ms | **off-by-one 発覚** (Stage 2.1 honest 訂正) | 中-高 (構造破綻防止 A-D 全 PASS) |
| **SNN-Izhikevich** | `research/other_archs/snn/izhikevich/` | Stage 2.3-2.4A | 26 | 4.0 ms | per-gene overclaim, lineage trivial | (評価) 中 |

合計: **本流 145 + research 100 = 245 PASS + 2 skipped** (回帰なし)、commit 20 件 (うち research 関連 11)。

### 3.2 進化段階の階梯

```
Stage 0a  state update gene 化 (RWKV-style) ← 出発点
Stage 0b  synthetic fitness (copy / addition)
Stage 0c  自前 minimal GA (llive 非依存)
Stage 1a  Z3 state_norm invariant (5.8ms)
Stage 2a  factor_hook × state update mock
Stage 2b  persona-indexed × verifier + open-ended 機構
Stage 3a  Marabou bridge skeleton (sketch 降格)
Stage 7a  VNN-COMP 新カテゴリ提案 (workshop-ready)

  ↓ research phase (Transformer 以外)

Neural ODE Stage 1   連続時間 vector field 進化 + Lipschitz/Hurwitz Z3
GNN Stage 1          固定 ring topology, message passing op
GNN Stage 2          動的 graph + ChangeOp (進行中)
SNN-LIF Stage 1      LIF (4-param) + 3 種 Z3 invariant
SNN-LIF Stage 2.1    boundary regression + honest 訂正
SNN-LIF Stage 2.2a/b I_max + |ΔI| input contract (Codex F3 完了)
SNN-Izhikevich Stage 2.3  Izhikevich (a,b,c,d) RS/IB/CH/FS 4 type
SNN-Izhikevich Stage 2.4A 反証的 test 内製化
```

---

## 4. 各アーキ詳細 (位置づけ + 主要 claim + honest 留保)

### 4.1 RWKV (本流, Stage 0-3 完了)

**位置づけ**: llcore 出発点、Transformer 系離散時間 state update kernel。

**gene** (3 params): `(decay, mix, gate_str)`
**Z3 invariant**: `|state| <= state_bound` (box 形式、abs encoding bug 不在 → audit `docs/audit/`)
**主要 claim**:
- 確定独自軸 #1 ChangeOp→Z3 online gate ✓
- 確定独自軸 #2 state update gene 化 ✓
- 確定独自軸 #4 persona-indexed × verifier ✓ (2b)
- 確定独自軸 #7 VNN-COMP 新カテゴリ ✓ (7a, workshop-ready)

**honest 留保**:
- 独自軸 #5 Marabou "異なる構造" 拡張: sketch + skeleton のみ (3a で Codex claim 降格)
- 独自軸 #6 Lipschitz/Hurwitz invariants: post phase (state_norm のみ実証)

**詳細**: `docs/poc/STAGE_3_VERDICT.md`

---

### 4.2 Neural ODE / LTC (research, Stage 1)

**位置づけ**: 連続時間 vector field 進化、Transformer の **連続時間版**。Hasani 2020 Liquid TC が CPU 完結実証済。

**gene** (3 params): `(A, W, b)` で `dx/dt = A*x + W*tanh(b*x)`
**Z3 invariant**:
- Lipschitz 上界 `|A| + |W|*|b|` (sech² ≤ 1 sound 近似)
- Hurwitz stability `A + W*b < 0` (1D 簡約)

**主要 claim** (Codex で部分降格):
- continuous-time vector field の Z3 1-step Euler map proof は **sound** (現実装の対角構造下)
- llcore RWKV の **design pattern + partial stack reuse** が ODE で成立 (AdaptiveFloorGate + ModesMeter 直接 reuse)
- Z3 latency 1.44ms = online 実用可

**honest 留保 (Codex 降格 4 件)**:
- G8 forward Euler discretization artifact は **未検証** (`||I + dt*J_f||` 別物)
- G3 ratchet 効果は `elitism=1` と分離未 (ablation 必要)
- 「same verifier stack」claim 撤回 → 「same design pattern + partial stack reuse」
- G6 Lipschitz 改善は `lipschitz_weight=0.35` fitness 加算による Goodhart (意図的)

**特別な気付き**: **Z3 abs encoding バグ発見+修正** (`abs_A >= A ∧ abs_A >= -A` は下界のみ → `z3.If` 等式化)。llcore 本流 audit clean。

**詳細**: `research/other_archs/neural_ode/verdict.md`

---

### 4.3 GNN 固定 ring (research, Stage 1)

**位置づけ**: 固定 topology 上 message passing op 進化。

**gene** (5 params): aggregation simplex `(α_sum, α_mean, α_max)` + `(W, U)`
**Z3 invariant**:
- over-smoothing shrink_upper bound (sound 上界形式、本来 lower bound でない → Codex 指摘)
- permutation equivariance: gene 構造で保証 (simplex membership)

**主要 claim** (Codex で大幅降格):
- 固定 ring topology 上の coefficient evolution に llcore 風 gate を被せた = mechanism 部分実証
- Z3 latency 2.69ms

**honest 留保 (Codex 降格 4 件)**:
- **F1 高**: over-smoothing **lower bound 論理方向逆 (unsound)** → "non-certificate" 降格
- **F2 高**: G2 "broken structure 検出" claim は **false** (clip で射影される) → 撤回
- F3 headline 「構造変化 ChangeOp mechanism 実証」overclaim → 撤回 (固定 ring のため)
- F4 G6/G8 margin 改善は selection 圧由来 (mechanism evidence 弱い)

**詳細**: `research/other_archs/gnn/verdict.md`

---

### 4.4 GNN 動的 graph (research, Stage 2 - 🔄 進行中)

**位置づけ**: GNN F3 (固定 ring overclaim) に対応する Stage 2、node 追加 / edge 削除 ChangeOp で **llcore 独自軸 #5 構造変化 ChangeOp** の本格実証。

**設計** (Agent 稼働中):
- `DynamicGraph` (N∈[6,12], adjacency list)
- 4 ChangeOp type: add_node / remove_node / add_edge / remove_edge
- llcore.verifier.changeop / refinement を **直接 import** (独自軸 #5 接続)
- Z3: 動的 N での over-smoothing + ChangeOp 列 refinement chain

**評価**: Agent 完了後に Codex review + 統合 verdict 更新。

---

### 4.5 SNN-LIF (research, Stage 2.1-2.2b)

**位置づけ**: discrete spike + 時間積分混在のアーキ、neuromorphic context。

**gene** (4 params): `(τ_m, V_th, V_reset, t_ref)`
**Z3 invariant 3 種**:
- (1) firing rate bound (refractory)
- (2) 膜電位 bounded (forward Euler 1-step)
- (3) Shielded RL hint (toy analogue of ProSh / GR(1))

**主要 claim**:
- discrete spike 列の Z3 symbolic 構成 = **discrete-time hybrid system verification pattern**確立 (llcore 本流に流用候補)
- Z3 latency 1.87ms

**重大な発見 (Stage 2.1)**:
- **commit bc53531 で「修正済」と claim したが `git show` で実態未修正判明** (Edit log success ≠ file 反映)
- **off-by-one fence-post error 発覚**: `n*t_ref > T` (over-strict, false positive) → `(n-1)*t_ref > T` (finite-window 厳密)
- **新気付き**: 新実装は **constructive proof** で任意 n で invariant 成立 (Codex Q2 改善版)

**Stage 2.2a-b 拡張**:
- I_max parameter 化 (assumed-input contract 明示)
- |ΔI| input contract (2-step Euler chain, Codex F3 完了)

**honest 留保**:
- 膜電位 invariant は continuous-time でなく forward Euler 1-step bound
- Shielded RL hint は toy analogue (実 ProSh / GR(1) 統合は future work)

**詳細**: `research/other_archs/snn/verdict.md` + Stage 2.1-2.2 commit logs

---

### 4.6 SNN-Izhikevich (research, Stage 2.3-2.4A)

**位置づけ**: LIF (tonic spiking 周辺) の進化空間を **RS / IB / CH / FS 4 firing-type** に拡張。`v²` 非線形を Z3 NRA で扱う。

**gene** (4 params): `(a, b, c, d)` (Izhikevich 2003 canonical)
**Z3 invariant**:
- v² 非線形 1-step Euler map proof (QF_NRA)
- dt packing firing rate bound (gene-independent)

**主要 claim**:
- v² 非線形を Z3 で扱える (latency 4ms = LIF より僅か遅め、threshold 15ms 内)
- same design pattern + partial stack reuse 成立

**honest 留保 (Codex 降格 5 件)**:
- **F1 高**: 「per-gene verifier」overclaim (gene.clipped() で box 流用するだけ) → 「assumed state/input contract 下の 1-step Euler map proof」
- **F2 高**: 「per-gene firing rate invariant」は dt packing bound (gene-independent) → 撤回
- F3 G8「4 firing-type distribution」は **「1 dominant basin (IB) + 3 preserved labels via reservoir」** (Codex 観察)
- F4 G6/G3 Goodhart triviality (fitness 同じ ものを gate)
- F5 Q1 soundness の限定 (reset omitted + u dynamics abstracted)

**Stage 2.4A 反証 test 内製化 (4 件 PASS)**:
- 「verifier が gene-independent」を機械検査 (Codex F1 受容)
- 「firing rate bound が dt packing のみ」を機械検査 (Codex F2)
- 「lineage 4 種は Reservoir 効果」+ gene-guess "1 dominant basin"
- G8 判定 logic の lenient threshold (>=3) を source audit

**詳細**: `research/other_archs/snn/izhikevich/verdict.md`

---

## 5. 共通 pattern + 横断 mechanism

### 5.1 全アーキで成立した共通 pattern

| Pattern | 何 | 全アーキで成立? |
|---|---|---|
| gene → Z3 制約構築 | アーキ固有 invariant を Z3 SMT に落とす | ✓ (5/5) |
| AdaptiveFloorGate 直接 reuse | 適応難易度 ratchet | ✓ (5/5、Codex confirm) |
| ModesMeter / LineageReservoir 流用 | 開放端機構 (型適応必要) | ✓ (5/5、minimal 再実装で) |
| 自前 minimal GA | llive 非依存 | ✓ (5/5) |
| Z3 latency < 15 ms | online gate 実用可 | ✓ (RWKV 5.8 / NODE 1.4 / GNN 2.7 / LIF 1.9 / Izh 4.0) |
| box 形式 `X ∈ [-bound, bound]` | abs encoding bug 不在 | ✓ (RWKV 本流 audit clean) |
| 「same design pattern + partial stack reuse」 | 「same verifier stack」は overclaim | 5/5 で claim 降格 |

### 5.2 横断的 honest 留保 (Codex review が共通指摘)

| Codex 横断 finding | 該当 PoC | 対応 |
|---|---|---|
| 「same verifier stack」overclaim | Neural ODE / GNN / SNN-LIF / Izhikevich | 全て「same design pattern + partial stack reuse」に降格 |
| G6 / G8 改善は selection objective の trivial 結果 | GNN / Izhikevich | mechanism evidence 弱い、claim 降格 |
| ratchet 効果が elitism と分離未 | Neural ODE / Izhikevich | ablation 必要 (Stage 3+) |
| 反証的 test 不在 → Codex が反証役 | 全アーキ | Stage 2.4A で **内製化開始** (Izhikevich) |

### 5.3 重要な横断的気付き (連鎖発見)

1. **Z3 abs encoding バグ pattern**: Neural ODE で発覚 → llcore 本流 audit (clean) → 他 PoC でも同 pattern なし。**「bug 発覚時は他 PoC / 本流に同 pattern 検査」規律**確立。
2. **「same verifier stack」claim の限界**: 直接 reuse は AdaptiveFloorGate + ModesMeter のみ、verifier 中身は別物 → llcore 本流取り込み時の **kernel + verifier backend plugin pattern** が必要。
3. **「per-gene verifier」表現の罠**: gene.clipped() で box 流用するだけでは gene を Z3 制約に入れたことにならない (Izhikevich Codex F1) → 真の per-gene verifier には gene を Z3 symbolic var として制約に入れる必要。
4. **Edit log success ≠ file 反映** (SNN bc53531 honest 訂正): verify-after-edit (Read + pytest) 規律必須化。
5. **「反証的 test の不在」を反証 test で機械検査内製化**: Codex が担っていた反証役を test に内製化 (Stage 2.4A 4 件)。

---

## 6. 構造破綻防止 framework + 導入価値判断

### 6.1 構造破綻防止 4 条件 (本流取り込み判定基準)

| 条件 | 内容 | 現状 |
|---|---|---|
| **(A) kernel plugin 化可** | `src/llcore/kernel/<arch>.py` に同じ Kernel Protocol で plugin 化可能 | RWKV ✓ / SNN-LIF ✓ / 他 △ |
| **(B) Stage 0-3 145 tests 回帰なし** | 既存 PoC tests 全 PASS 維持 | ✓ (全 commit で 245+2 PASS 維持) |
| **(C) Codex pair-review 通過** | blocker 0 件 + 全 claim 降格対応 | ✓ (24 Findings 全対応: 1 honest 訂正 + 1 実装修正 + 22 claim 降格) |
| **(D) semver 互換** | research/ 隔離 = llcore 0.1.0a0 挙動完全不変 | ✓ (src/ 一切触らず) |

→ 全アーキ 4 条件のうち (B)(C)(D) は全 PASS。(A) は **SNN-LIF と RWKV が即取り込み可能**、他は中期。

### 6.2 導入価値マトリクス (`OTHERARCH_VERDICT.md §4.1` より、Stage 2 反映後)

| PoC | 新独自軸 | 既存独自軸強化 | paper 化 | CPU 完結 | 総合 |
|---|---|---|---|---|---|
| Neural ODE | 中 ("continuous-time evolvable arch") | 中 (#1 連続時間版) | 中 (NeurIPS workshop 候補) | ✓ | **中** |
| GNN 固定 | 弱 | 弱 | 弱 | ✓ | 低-中 |
| GNN 動的 (進行中) | 高 (独自軸 #5 構造変化) | (評価待ち) | 高 (候補) | ✓ | 評価待ち |
| SNN-LIF | **高** (discrete-time hybrid pattern + off-by-one 発見) | 中 (#1 hybrid 拡張) | 中-高 (NeurIPS SafeAI workshop) | ✓ | **中-高** |
| SNN-Izhikevich | 中 (LIF 拡張、claim 降格多い) | 中 | 中 | ✓ | **中** |

→ **本流取り込み候補 (Stage 4)**: SNN-LIF (短期) → Neural ODE (中期) → 他 (評価後)

---

## 7. 規律と learnings

### 7.1 開発規律 (本研究で確立)

| 規律 | 出所 | 適用 |
|---|---|---|
| **構造規律 6 ヶ条** | llcore Stage 0a | 全 PoC で適用 |
| **Codex × Claude pair-review** | Stage 0a v1 zero attractor 発見 | 全 PoC commit 前必須 |
| **honest disclosure** | Stage 0a "異常な結果は内訳疑う" | Codex review + verdict 留保 |
| **verify-after-edit** | Stage 2.1 honest 訂正 (Edit log ≠ 実装) | Read + pytest 両方確認 |
| **反証的 test 内製化** | Stage 2.4A (Izhikevich) | 「現実装が overclaim」を機械検査する PASS する反証 |
| **bug 発覚時の横断 audit** | Stage 2.4B (Neural ODE → 本流) | 他 PoC / 本流に同 pattern 検査 |
| **構造破綻防止 4 条件** | research phase 開始時 | 本流取り込み判定基準 |

### 7.2 メタ的 learnings (本研究の収穫)

- **claim と実装のズレ**を Codex が確実に検出 (24/24 Findings)、pair-review が機構的に機能
- **「same design pattern」は移植性高い、「same verifier stack」は overclaim** (型違いで成立せず)
- **進化に上限を設けない**は単一機構では不足、ratchet + reservoir + MODES + curriculum の組合せで初めて成立
- **discrete-time vs continuous-time の境界明示**: Z3 invariant が現実 simulator 全体に sound でない場合は明示降格 (Neural ODE Q3, SNN-LIF Q1, Izhikevich Q1)
- **lineage reservoir + initial type bias で G8 trivial 化** (GNN/Izhikevich 共通)
- **box 形式の sound 性** vs **補助変数 abs encoding の落とし穴** (Neural ODE で発覚、本流 clean)

---

## 8. 時系列 重要な気付き (発見順、ユーザーが追体験できる順)

| 時期 | 発見 | impact |
|---|---|---|
| Stage 0a v1 | **zero attractor** (state 永久 0、Codex で発覚) | pair-review 規律の出発点 |
| Stage 2b Q4 | AND gate 未実装 (Codex 指摘で実装修正) | Codex finding → 真の修正 pattern 確立 |
| Stage 3a 全 Q | Marabou 包含は型違いで成立せず | claim 全面降格 |
| Stage 7a Q4 | reference impl が `.onnx`/.vnnlib parser 未実装 | TMLR submission に必要 (workshop OK) |
| Neural ODE Stage 1 | **Z3 abs encoding バグ発見** | `z3.If` 等式化必須、本流 audit trigger |
| GNN Stage 1 F1 | over-smoothing **lower bound 論理逆 (unsound)** | claim 降格 |
| **SNN-LIF Stage 2.1** | **commit bc53531 overclaim 発覚** (Edit log ≠ 実装) | verify-after-edit 規律必須化 |
| SNN-LIF Stage 2.1 補足 | **constructive proof で任意 n で invariant 成立** (Codex Q2 改善) | 新気付き |
| Izhikevich Stage 2.3 F1 | 「per-gene」表現の罠 (gene.clipped() で box 流用) | 真の per-gene verifier は Stage 3+ |
| **Stage 2.4A** | **反証 test 内製化** (Codex 反証役を test に内製) | pair-review 規律の機械検査化 |
| **Stage 2.4B** | **本流 RWKV abs encoding audit clean** | 他 PoC bug の横断 audit 規律 |

---

## 9. 次のアクション候補 (現時点、E Agent 完了で更新)

### 短期 (即着手可)
- **C** 真の per-gene verifier (Izhikevich Codex F1 直接対応)
- **F** Neural ODE Stage 2 `use_floor=False` ablation
- **G** llcore 0.2.0a0 kernel plugin 設計 doc (SNN-LIF 取り込み path 設計)

### 中期 (Agent dispatch 推奨)
- **D** AdEx (Adaptive Exponential I&F) gene 一般化 (Izhikevich の次)
- **H** PoC 7a NeurIPS workshop submission (現原稿で提出可)
- 本 doc 完成後の Stage 2 全 PoC 統合 verdict 更新

### 長期 (paper phase)
- **I** 横断 paper "Verified Evolvable Architectures: A Unified Soundly-Gated Framework Beyond Transformers" (TMLR target)
- **J** llcore GitHub repo 作成 + push (現状 20 commits ローカル保持)
- 各アーキ別 workshop submission

---

## 10. ファイル navigation (体系的に追体験する場合)

### 必読 (順序付き)
1. `README.md` (project 概要)
2. `docs/poc/COMPLETION_VERDICT.md` (Stage 0-2 完成宣言)
3. `docs/poc/STAGE_3_VERDICT.md` (Stage 3 統合 + 評価 harness)
4. `research/other_archs/README.md` (research phase 動機 + 3 条件)
5. `research/other_archs/OTHERARCH_VERDICT.md` (横断統合 verdict)
6. **本 doc `docs/ARCHITECTURE_LANDSCAPE.md`** (アーキ体系)

### 各 PoC verdict (詳細を追う場合)
- `docs/poc/poc_0a_verdict.md` 〜 `poc_7a_verdict.md` (RWKV 本流 8 PoC)
- `research/other_archs/neural_ode/verdict.md` (Neural ODE)
- `research/other_archs/gnn/verdict.md` (GNN 固定 ring)
- `research/other_archs/gnn/dynamic_graph/verdict.md` (GNN 動的、進行中)
- `research/other_archs/snn/verdict.md` (SNN-LIF)
- `research/other_archs/snn/izhikevich/verdict.md` (SNN-Izhikevich)
- `docs/audit/rwkv_abs_encoding_audit_2026_05_29.md` (本流 abs encoding audit)

### memory (ユーザー側の文脈と連携)
- `[[project_llcore_init_2026_05_29]]` — Stage 0-3 + research + SNN Stage 2 完了 (最重要)
- `[[project_core_evolution_survey_2026_05_28]]` — Agent A-D 事前調査
- `[[feedback_codex_pair_review_for_llcore]]` — review 規律
- `[[feedback_benchmark_honest_disclosure]]` — claim 降格規律
- `[[feedback_external_ai_verify]]` — Codex finding 実コード検証
- `[[feedback_staged_poc_individual_structure]]` — PoC battery 文化

---

## 11. 用語集 (ユーザーが doc を再読する際の参照)

| 用語 | 意味 |
|---|---|
| **gene** | アーキ固有 core algorithm の低次元パラメータ表現 |
| **Z3 invariant** | SMT (Satisfiability Modulo Theories) で symbolic 検証する安全 invariant |
| **per-gene verifier** | 各 gene を Z3 制約式に入れて検査 (Izhikevich Codex F1 で claim 降格、現状は assumed contract 下の box proof) |
| **box 形式** | `X ∈ [-bound, bound]` で `|X| <= bound` を表現 (sound, abs encoding bug 不在) |
| **same design pattern + partial stack reuse** | アーキ間で同じ pattern (sound invariant gate + open-ended 機構) + AdaptiveFloorGate 直接 reuse / 他は型適応 (「same verifier stack」overclaim を撤回した正名化) |
| **open-ended 4 機構** | 適応難易度ゲート / 中立貯蔵庫 / MODES 計器 / MCC カリキュラム |
| **constructive proof** | Z3 が refractory respect + violation 仮説の矛盾で unsat を返す = 数学的に厳密な invariant 証明 |
| **反証的 test** | 「現実装が overclaim だった」を PASS で assert する test (将来 claim 強化で FAIL → 改善検知トリガ) |
| **Codex pair-review 規律** | Claude 単独実装 + commit 前必ず Codex (gpt-5.4) review、Claude 単独で見落とした設計問題を Codex が検出する分業 |
| **構造破綻防止 4 条件** | (A) kernel plugin 化可 (B) Stage 0-3 145 tests 回帰なし (C) Codex 通過 (D) semver 互換 |

---

**doc 完成日**: 2026-05-29
**次の更新トリガ**: GNN 動的 graph Agent 完了 / Stage 2.4A 続きの Stage 2.4B-D / llcore 0.2.0a0 kernel plugin 設計 / 横断 paper draft 開始
