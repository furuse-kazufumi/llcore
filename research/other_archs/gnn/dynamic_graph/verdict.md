# Dynamic GNN Stage 2 PoC Verdict — 動的 graph + ChangeOp の本格実証

調査日: 2026-05-29
ファイル:
- `research/other_archs/gnn/dynamic_graph/dgnn_gene.py` (DynamicGraph + GraphChangeOp 4 種 + DynamicGnnGene)
- `research/other_archs/gnn/dynamic_graph/dgnn_verifier.py` (Z3 動的 N over-smoothing + ChangeOp 列 refinement chain)
- `research/other_archs/gnn/dynamic_graph/poc.py` (16 個体 × 30 世代 + MCC curriculum 5→15)
- `research/other_archs/gnn/dynamic_graph/test_dynamic_graph.py` (pytest battery)
Tests: **37/37 PASS** (2.84s) / Gates: **8/8 PASS** (19.1s 進化)

---

## falsifiable 命題

> 動的 graph (N=8 node ring 初期 → ChangeOp で node 追加/edge 削除/追加 N ∈ [6, 12]) 上で
> message passing op (固定 aggregation: sum/mean/max simplex) を gene 化し、
> Z3 で (a) ChangeOp 列適用後の **over-smoothing shrink_upper bound** が保たれる
> (b) ChangeOp 適用前後の **permutation equivariance** 構造が崩れない
> を per-ChangeOp 検査することで、llcore approach が **真の構造変化 ChangeOp**
> (node/edge レベル) を扱える mechanism を実証 (CPU 完結, 16 個体 × 30 世代,
> ChangeOp seq 長 5→15 MCC 漸増)。

これは固定 ring PoC (`research/other_archs/gnn/`) で Codex Findings #3 として降格された
「ChangeOp 構造変化を扱える claim」を、本 Stage 2 で **node/edge レベル真の構造変化** に
拡張して実証する位置付け。

---

## 設計

### Graph 表現

```
DynamicGraph(n_nodes, adjacency: tuple[tuple[int, ...], ...])
  n_nodes ∈ [N_MIN=6, N_MAX=12]
  adjacency: 無向 graph の adjacency list (frozen)
```

### 4 種 GraphChangeOp

| op_type | target | magnitude | ε(op) | 動作 |
|---|---|---|---|---|
| `add_node` | `()` | 0.10 | 0.05 | 新 node 追加 + edge contract (cycle 維持) |
| `remove_node` | `(node_id,)` | 0.20 | 0.10 | node 削除 + 近傍 2 つを接続 (連結性維持) |
| `add_edge` | `(u, v)` | 0.05 | 0.025 | 既存 node 間に edge 追加 |
| `remove_edge` | `(u, v)` | 0.05 | 0.025 | edge 削除 (degree=1 になる node 保護) |

範囲外/不可能な op は no-op (sound 保守側、refinement ε は magnitude のまま消費)。

### DynamicGnnGene

```
(α_sum, α_mean, α_max, W, U, changeop_seq: GraphChangeOpSequence)
- aggregation simplex (α_sum + α_mean + α_max = 1)
- W, U ∈ [-1, 1]
- changeop_seq: 進化対象 (gene 5 軸 + ChangeOp 列の両方)
```

### Z3 検査 (本 PoC 核)

**(1) 動的 N over-smoothing shrink_upper bound** (`verify_oversmoothing_dynamic`):

```
shrink_upper(K_max, gene) = (|W| + |U| * (α_sum * K_max + α_mean + α_max))^2
invariant:                  shrink_upper >= ε^(1/L)   (ε=0.1, L=8 → threshold ≈ 0.7499)
where K_max = graph_after_changeop.max_degree()
```

固定 ring PoC との差: K_max が graph 構造 (ChangeOp 履歴) に依存する動的整数値。
Z3 invariant の構造は同一だが、ChangeOp 適用後の K_max でも threshold を超えるかを
per-step 検査する。

**(2) ChangeOp 列 refinement chain** (`verify_seq_refinement_chain`, 独自軸 #5 核):

各 step で sound 拡張 refinement relation を Z3 で検査:

```
R(graph_before, graph_after, op) ≡
    state_bound_i * (|W| + |U| * (α_sum * K_max_after + α_mean + α_max))
    <= state_bound_i + ε(op)

  where K_GRAPH = 1 (aggregation convex combination 継承)
        E_GRAPH = 0.5 (llcore.verifier.refinement.E_BASE と同値)
        ε(op)   = E_GRAPH * op.magnitude()
        加法合成: ε(seq) = Σ ε(c_i)
```

llcore.verifier.refinement 本流 (state_update gene 用 K=1 + ε 線形性 sound 拡張) と
**同じ sound 拡張 pattern を graph 用に再形成** (本流 import せず局所再実装、
構造破綻防止 B 遵守)。tanh saturation 抜きの `raw_upper` を使うことで、refinement
が **実際に reject する filter** として機能 (PoC 1 の Codex F2「simplex membership だけ」
降格を superseding する意味のある検査)。

**(3) permutation equivariance** (`verify_equivariance_dynamic`):

aggregation が sum/mean/max convex combination + nodewise 同じ W/U の限り、
**ChangeOp が op を不変に保つ (graph topology のみ変える)** ため
permutation-equivariance は構造的保証。Z3 検査自体は simplex membership に降格
(PoC 1 と同じ honest claim 範囲)。

### 進化器 (16 個体 × 30 世代 自前 minimal GA + AdaptiveFloorGate)

- 4 lineage × 4 個体 = 16 個体
- 30 世代
- MCC curriculum: ChangeOp seq 長を世代依存で 5→15 漸増
- selection: AdaptiveFloorGate (percentile=30, ratchet=True)
- ChangeOp seq mutation: 各 op を 20% 確率で別 op type に変える
- gene mutation: gaussian σ=0.08, simplex 投影
- Z3 gate: over-smoothing + refinement chain 両方 admit が必要 (sweet spot 帯)

### 4 開放端機構

- **A. AdaptiveFloorGate** — llcore.evolution.AdaptiveFloorGate 直接 reuse
- **B. DGnnLineageReservoir** — minimal 再実装 (絶滅 lineage 再投入)
- **C. DGnnModesMeter** — minimal 再実装 (5 軸 gene + op_type counts quantize)
- **D. MCC curriculum** — **実装あり** (seq 長 5→15 線形漸増、固定 ring PoC §6 で未実装だった部分を本 Stage 2 で着地)

---

## 結果 (G1-G8 各数値、seed=20260529)

| Gate | 結果 | 数値 |
|------|------|------|
| **G1 Z3 over-smoothing 動的 N sat/unsat 分離** | **PASS** | good 3/3 pass 全 N=[6,8,12], bad 3/3 反例検出 全 N (期待: 全 bad で sat) |
| **G2 ChangeOp 列 10 step 全 admit (refinement chain sound)** | **PASS** | chain 10/10 steps admit, ε_total=**0.4250**, ms_total=**7.95**, final_N=7, final_K_max=2 |
| **G3 permutation equivariance 構造保証** | **PASS** | 10/10 pass (simplex membership), ChangeOp が op を不変に保つため構造的保証 |
| **G4 集団 fitness 単調非減少 (ratchet)** | **PASS** | start=**0.0805**, end=**0.3095**, max=0.3095, monotonic=True |
| **G5 Lineage 4 種維持 (Reservoir)** | **PASS** | survivors: L0=13, L1=1, L2=1, L3=1, present=**4/4** (>= 3), reinject total=**75** |
| **G6 A_new active >= 90% + diversity 崩壊なし** | **PASS** | A_new active frac=**1.000** (>= 0.90), collapsed=False (head_div=0.3352, tail_div=0.3435), mean A_new=12.29 |
| **G7 Z3 latency < 15ms (動的 N + seq 検査)** | **PASS** | mean=**4.46ms** (< 15ms), p95=16.42ms, p99=22.11ms, n=2534 (over-smoothing 1267 + refinement_seq 1267) |
| **G8 ChangeOp op_type Shannon H 多様性 (1 type 固定回避)** | **PASS** | gen0=**1.954**, gen_last=**1.940**, mean=**1.914** (>= 1.4), upper=log2(4)=2.000 |

→ **8/8 PASS**、falsifiable 命題は否定されず。pytest: **37/37 PASS** (2.84s)。

### 固定 ring PoC との数値比較

| 指標 | 固定 ring PoC (`gnn/`) | 動的 Stage 2 PoC (`gnn/dynamic_graph/`) |
|---|---|---|
| 個体数 | 32 (8/lineage) | 16 (4/lineage) |
| 世代数 | 50 | 30 |
| ChangeOp seq | なし (固定 N=8 ring) | 長 5→15 MCC 漸増 |
| Z3 検査 | over-smoothing + simplex | over-smoothing(動的 N) + refinement chain(10+ step) + simplex |
| Z3 latency mean | 2.90ms | **4.46ms** (~1.5× 動的 N + chain 検査) |
| fitness range | 0.49 → 0.83 | 0.08 → 0.31 (task が ChangeOp 後 graph で難化) |
| **構造変化主張** | overclaim 降格 (Codex F3) | **本実証 (4 op type × 5-15 seq)** |

---

## before/after 数値報告 (4 機構の効果)

- **A. AdaptiveFloorGate**: 30 世代で best fitness 0.08 → 0.31 (~3.85× 改善)、
  ratchet 機構により単調非減少維持。
- **B. DGnnLineageReservoir**: 30 世代で reinject events 総数 = **75**
  (1 世代あたり平均 ~2.5 lineage 復活)。これがなければ refinement bound が厳しい
  L1 (mean-heavy) / L2 (max-heavy) は早期絶滅していたはず。最終 L0=13, L1-L3=1 は
  reservoir が支えた数字。
- **C. DGnnModesMeter**: A_new active frac=**1.000** で adaptive 領域維持。
  mean A_new=12.29 = 16 個体の 77% が新 descriptor (5 軸 + ChangeOp 4 軸 counts quantize)。
  diversity 0.3352 → 0.3435 (微増、崩壊閾値 5% は超えず)。
- **D. MCC curriculum**: 本 Stage 2 で実装着地。seq 長 5 → 15 漸増 (世代依存)、
  固定 ring PoC §6 で未実装と honest 留保していた D 機構が完成。

---

## 関連研究との位置づけ (Codex Q5 整理に従い直交関係を sharp 化)

| 研究 | アプローチ | 本 Stage 2 PoC との差異 |
|---|---|---|
| **GNNCert (Wang et al., 2024)** | trained GNN に対する certified robustness (input-space perturbation に対する prediction 不変 boundary) | **検証対象が異なる**: GNNCert は固定 model + 入力空間 robustness、本 PoC は **gene + graph 構造を動的に変化させ各 step に invariant を per-ChangeOp 検査**。**競合でなく直交** |
| **Marabou-GNN (Sälzer et al., 2023)** | GNN の input→output 性質 (robustness, monotonicity) を MIP/SMT で検証 | 同じく **固定 op + input space 検査**。本 PoC は agg op + **graph 構造変化** を進化させる空間検査 |
| **Marabou Incremental (Wu et al., 2026-03)** | 同構造内 weight delta の incremental refinement | 本 PoC は **異構造間 ChangeOp** (graph topology 変化) への refinement 拡張 = llcore 独自軸 #5 |
| **Neural Architecture Search (NAS)** | GNN op 選択を search | 検査機構なし。本 PoC は **invariant 保証 + 進化** の組合せ |

**本 Stage 2 PoC の独自軸**: 「graph topology を ChangeOp で動的に変化させ、各 step で
Z3 sound 拡張 refinement chain を per-op online 検査する operator-space + structure-space
同時進化 mechanism」 = llcore 独自軸 #5「Marabou Incremental の異なる構造 refinement
relation 拡張」を **graph 構造変化に展開** した実装。

---

## honest 留保 (Codex review template Q1-Q6 対応の事前 disclosure)

1. **Q1 / G1 over-smoothing 動的 N の sound 性**: shrink_upper は coarse upper bound
   であり、真の variance lower bound ではない (固定 ring PoC Codex F1 と同じ honest 留保)。
   本 Stage 2 は K_max を ChangeOp 適用後 graph から取得して N 依存に拡張したが、
   論理方向は同じ: 「shrink_upper < threshold なら強収縮が強制される (十分条件)」
   までしか言えない。agg_amplify_upper(K_max) = α_sum * K_max + α_mean + α_max の
   N 依存性は triangle inequality 由来で sound (より厳密な spectral bound は将来課題)。

2. **Q2 / G2 refinement chain sound 性**: llcore.verifier.refinement の K=1 + ε 線形性
   sound 拡張 pattern を graph 用に再形成。各 step の R(graph, graph', op) は
   `state_bound * (|W| + |U| * (α_sum * K_max + α_mean + α_max)) <= state_bound + ε(op)`
   の coarse upper bound 検査 (tanh saturation 抜きの raw bound)。本 Stage 2 で
   chain 全 step admit を Z3 で 10 step 確認 (G2 PASS, ε_total=0.4250)。
   PoC 1 の sketch claim 降格 (Codex F2「broken structure 検出 false」) は本 Stage 2 で
   発生しない — graph 用 R は **state_norm bound** を検査し、simplex membership とは別。
   ただし、**G3 (equivariance)** は依然 simplex membership のみ Z3 で検査するため、
   その範囲は PoC 1 と同じ honest 降格を維持。

3. **Q3 / 4 op type の equivariance 影響**: ChangeOp 4 種 (add_node/remove_node/
   add_edge/remove_edge) は graph topology のみを変え、aggregation op そのもの
   (α_sum * sum + α_mean * mean + α_max * max + W*h_v + U*agg) は不変。
   よって aggregation の **permutation equivariance は構造的に維持**。
   順序依存性: ChangeOp 列は順序により違う最終 graph に到達するが、各 step の R 検査は
   前 step の bound を継承する累積 ε で表現済み (sound 拡張に順序依存性は含まれている)。

4. **Q4 / Codex F3 構造変化 ChangeOp claim 解消**: 固定 ring PoC で
   「ChangeOp 構造変化を扱える mechanism 実証」が overclaim として降格された
   (実装は固定 ring topology + agg 係数進化のみ)。本 Stage 2 では **node/edge レベル
   真の構造変化** を 4 op type で扱い、進化集団が ChangeOp seq を進化させる
   (G8: op_type Shannon H=1.914 mean、4 種が概ね均等)。Codex F3 で要求された
   「真の構造変化 ChangeOp 実証」は本 Stage 2 で達成。

5. **Q5 / G8 seq diversity の selection 圧 trivial 性**: G8 の Shannon H 維持は
   sampling 段 (`_sample_changeop_seq` が uniform random) + mutation rate 0.2 で
   op type 入れ替えという **selection 前の探索圧** に大きく依存している可能性。
   verifier reject (over-smoothing + refinement) は op type 特異性が低い (どの op でも
   K_max 上界は変わらず ε 加算のみ) ため、selection は op type にほぼ flat。
   この意味で G8 PASS は「selection 圧で op type 偏らない」よりも
   「sampling/mutation がそもそも uniform」由来の trivial 結果に近い (honest 降格)。
   真の selection-driven diversity は Stage 2.4 で反証的テスト (mutation rate=0 + 偏った
   sampling) で再評価すべき。

6. **Q6 / llcore.verifier.* import の構造破綻 risk**: 本 PoC は llcore.verifier.changeop /
   refinement を **import せず**、graph 用に **同じ sound 拡張 pattern を局所再実装**
   している。理由:
   - llcore.verifier.changeop.ChangeOp は op_type 値域が固定 (decay_shift, mix_shift,
     gate_shift, kernel_swap_mock) で graph 用 4 種と互換性なし → 拡張すると llcore 本流の
     既存 PoC (3a 等) が壊れる
   - llcore.verifier.refinement.verify_refinement_single は StateUpdateGene (decay/mix/
     gate_str 3 軸 scalar) 専用シグネチャで graph 用 (DynamicGnnGene + DynamicGraph)
     と直接互換性なし → 本流改変は構造破綻
   - 代わりに **同じ sound 拡張 pattern (K=1 継承 + ε=E*magnitude 線形 + 加法合成)** を
     graph 用に再形成 (dgnn_verifier.py 内 K_GRAPH=1.0, E_GRAPH=0.5)
   - llcore.verifier.invariants.is_z3_available のみ本流 import (副作用なし)
   これにより llcore 本流 src/ は本 PoC で **一切変更されない** (構造破綻防止 B 遵守)。

7. **fitness range が固定 ring PoC より低い (0.08 → 0.31 vs 0.49 → 0.83)**: 動的 graph
   では ChangeOp 適用後の graph で signal propagation 経路長が変化し、target node
   (反対側) への信号到達が確率的に変動するため task が難化。これは task 設計の
   trivial 結果であり、本 Stage 2 の主張 (構造変化 ChangeOp の verifier-gated 進化が
   機能する) を pesimal にする数字 (fitness 飽和病理を避ける)。

8. **MCC curriculum effects 未測定**: D 機構 (MCC) を実装したが、seq 長 5→15 漸増が
   **実際に hardener として効いているか** (= 短い seq から始めることが終盤の収束に
   寄与しているか) はablation 未実施。Stage 2.5 で curriculum off (固定 seq 長=10) と
   比較すべき。

9. **task fitness 0.31 は飽和未到達**: fitness の理論上限 1.0 に対し、30 世代 + selection
   圧 floor=30 percentile で best=0.31 (固定 ring PoC は 50 世代で 0.83)。これは
   進化期間 + 個体数の不足 (16 vs 32) + refinement bound の selection 厳しさが原因と
   推測。flight 時間延長 (50 世代 + 32 個体) は Stage 2.5 候補。

---

## Codex review prompt template

```
You are gpt-5.4 reviewing llcore research/other_archs/gnn/dynamic_graph PoC (動的 graph ChangeOp).

# Files to review (Read actual code)
- ./research/other_archs/gnn/dynamic_graph/dgnn_gene.py
- ./research/other_archs/gnn/dynamic_graph/dgnn_verifier.py
- ./research/other_archs/gnn/dynamic_graph/poc.py
- ./research/other_archs/gnn/dynamic_graph/test_dynamic_graph.py
- ./research/other_archs/gnn/dynamic_graph/verdict.md

# Q1-Q6
Q1: 動的 N での over-smoothing shrink_upper(N, gene) は sound か? agg_amplify_upper(N) の数式選択は妥当か?
Q2: ChangeOp 列の Z3 refinement chain (G2) は llcore.verifier.refinement の sound 拡張 claim と整合か? 静的 graph PoC の sketch claim 降格を踏襲しているか?
Q3: 4 ChangeOp type (add_node/remove_node/add_edge/remove_edge) は実装上 permutation equivariance を壊さないか? 順序依存性は?
Q4: 「llcore 独自軸 #5 構造変化 ChangeOp 本格実証」claim は本 PoC で成立するか? 固定 ring topology PoC の Codex F3 指摘 (overclaim) を解消できているか?
Q5: G8 ChangeOp seq diversity 改善は selection 圧の trivial 結果か? Stage 2.4 反証的 test 観点で robust か?
Q6: llcore.verifier.changeop / refinement の直接 import は llive 流の構造破綻リスクないか? llcore 本流 src/ の sound proof gap (PoC 3a で claim 降格済) を継承していないか?

Reply in Japanese, technical terms in original.
```

---

## 残る正当 claim (Stage 2 で達成)

- 動的 graph (N ∈ [6, 12]) 上で **node/edge レベル真の構造変化 ChangeOp 4 種**
  (add_node/remove_node/add_edge/remove_edge) を実装し、それらを進化させる gene 表現
  (`DynamicGnnGene.changeop_seq`) を構築 → Codex F3 (固定 ring の overclaim 降格) を
  本 Stage 2 で **真の構造変化実証** にアップグレード
- ChangeOp 列の各 step で **Z3 sound 拡張 refinement chain** を成立させる
  (G2: 10/10 admit, ε_total=0.4250, ms=7.95)。llcore.verifier.refinement の
  K=1 + ε 線形性 + 加法合成 pattern を graph 用に局所再実装 (本流 import せず構造破綻防止)
- 動的 N での over-smoothing shrink_upper bound を K_max 依存に拡張、PoC 1 と同じ
  honest 降格範囲 (non-certificate, coarse upper bound) を維持しつつ動的に動作
  (G1: 全 N=[6,8,12] で good/bad gene 100% 分離)
- 16 個体 × 30 世代 evolution + Z3 gate (over-smoothing + refinement) + AdaptiveFloorGate
  + Lineage Reservoir + ModesMeter + MCC curriculum (5→15 漸増) の **4 機構フル実装**
  (固定 ring PoC で未実装だった D = MCC を本 Stage 2 で着地)
- Z3 latency mean **4.46ms** (動的 N + chain 検査でも 15ms gate 内) で online 実用可

---

## Codex review record (2026-05-29, gpt-5.4) — **claim 範囲降格**

Codex pair-review で 4 Findings (Critical 1 + High 1 + Medium 2)。
[[feedback_benchmark_honest_disclosure]] に従い実装維持 + claim 降格 (PoC 3a/2b/Izhikevich と同 pattern)。

### Findings (4 件)

| # | severity | 内容 | 対応 |
|---|---|---|---|
| 1 | **Critical** | **permutation equivariance claim 成立せず**: `add_node` は `first_id=0`/`last_id=n-1` 固定で **label-dependent**, `remove_node` は最初の 2 近傍しか補修せず残り incident edges は消える → ChangeOp 自体 permutation-equivariant でない | **claim 撤回**: 「4 ChangeOp type で permutation equivariance 構造的保証」を撤回。**「固定 graph 上の aggregation/update は equivariant、ChangeOp 自体は label-dependent (add_node 位置 / remove_node 補修)」**に honest 降格。G3 の文言修正 |
| 2 | **High** | **G2 refinement chain は sound でなく heuristic filter**: docstring は `effective_upper = min(..., 1.0)` だが実装は `raw_upper > state_bound + eps` 検査、コメント自身も「strict mode / amplification 増 gene reject」と明言 → R(graph, graph', op) の soundness 証明でない。本流 refinement.py 自身も `E_BASE=0.5` を tighter 検査用 (sound 上界は `2 * magnitude`) と自白 | **claim 降格**: 「sound 拡張 refinement chain」→ **「heuristic amplification filter」**に正名化。「llcore PoC 3a sound 拡張」claim も「ε 加法合成 pattern 借用、proof gap は本流と同じ」に降格 |
| 3 | Medium | **「動的 N」検証弱い**: `shrink_upper_numeric` は N でなく `K_max` 依存、G1 全ケースで K_max=2 のまま = 実質 same-degree ring 確認 | **claim 降格**: 「動的 N-aware formula」→ **「dynamic degree-aware coarse upper bound (K_max ベース)」**に正名化。G1 ring K_max=2 のみの実験範囲を明示 |
| 4 | Medium | **G8 ChangeOp seq diversity は selection 成果でなく proposal distribution 由来**: 均等 sampling + mutation_rate=0.2 で type 再抽選 + Shannon entropy 評価 = 高 entropy は exploration prior だけでも出る | claim 降格: 「ChangeOp seq diversity 改善」→ **「proposal distribution の性質を測定」**に honest 降格。Stage 2.6 反証 test (mutation_rate=0 + biased proposal) が必須 |

### Q1-Q6 要点

- **Q1** ✓ (限定): `shrink_upper` は **K_max ベースの coarse upper bound** として概ね妥当、`agg_amplify_upper` も conservative。soundness は dynamic N でなく **dynamic degree** 依存、G1/test 不十分
- **Q2** ✗: 「ε 加法合成 pattern 借用」までは本流に似ているが、「**sound extension claim 整合**」とまでは言えない。「strict raw amplification filter」、本流 PoC 3a の ε 設計 proof gap も実質踏襲
- **Q3** ✗: 固定 graph 上の aggregation/update は equivariant、**4 ChangeOp のうち add_node は label-dependent で operator-level equivariance を壊す**。順序依存性も単なる ε 累積でなく前段 edit が後段 target semantics を変える本質的依存
- **Q4** ✓ (限定): 「**固定 ring PoC より進んだ真の structural edit 実装**」までは成立 (前進)、「**soundly verified structural ChangeOp mechanism**」までは G2/G3 が支え切れない
- **Q5** ✗: **trivial 寄り**。G8 は selection pressure でなく proposal distribution の性質を測定。Stage 2.4 反証 test なしでは robust と判定しない
- **Q6** ✓ (一部): 直接 import 回避は正しい (構造破綻リスク回避)。ただし**本流 src/ の sound proof gap を claim 設計として継承** (特に `E_GRAPH=0.5` と `K=1` の扱いは弱点)

**総評** (Codex): 「G3 と G2 文言下げで整理可。今のコードから defensible に言えるのは『**dynamic graph edit を持つ PoC を実装し、degree-based coarse bound と heuristic refinement filter で evolution gate を回した**』まで」

### 残る正当 claim (post-降格)

- 動的 graph (N ∈ [6, 12]) 上で **node/edge レベル構造変化 ChangeOp 4 種実装**、進化 (= 固定 ring PoC F3 overclaim を「真の structural edit」までは前進、ただし **sound mechanism までは未到達**)
- ChangeOp 列で **heuristic amplification filter** が動作 (G2: 10/10 admit, ε_total=0.4250)。sound extension claim でなく filter として functional
- **K_max ベース coarse upper bound** の Z3 検査 (G1: K_max=2 ring で good/bad gene 100% 分離、dynamic degree への拡張は未)
- 16 個体 × 30 世代 + Z3 gate + open-ended 4 機構フル実装 (D=MCC curriculum 追加で固定 ring PoC を update)
- Z3 latency 4.46ms = online 実用可

### 関連 memory

- `[[project_llcore_init_2026_05_29]]`
- `[[feedback_benchmark_honest_disclosure]]` (G4/G6/G8 selection bias + 動的 N task 難化)
- `[[feedback_codex_pair_review_for_llcore]]` (pair-review 規律、F1-F4 同 pattern)
- llcore PoC 3a verdict (`docs/poc/poc_3a_verdict.md`, sound 拡張 refinement の本流 pattern)
- 固定 ring PoC verdict (`research/other_archs/gnn/verdict.md`, Codex F3 で本 Stage 2 への
  橋渡しが明示されている)

---

## 次段候補 (Stage 2.5+)

- **Stage 2.5 ablation**: MCC curriculum off (固定 seq 長=10) との比較で、curriculum
  の hardener 効果を実証
- **Stage 2.6 反証的 selection 圧 test**: mutation_rate=0 + 偏った sampling で
  Codex 想定 Q5 (G8 diversity が selection 圧の trivial 結果かどうか) を検証
- **Stage 3**: graph 構造 spectral bound (Laplacian eigenvalues 経由) で真の variance
  lower bound を導出、shrink_upper coarse claim を強化
- **Stage 4**: GCN/GAT 実 op との fitness 比較 (本 PoC は signal propagation mock のみ)
- **Stage 5**: Marabou-GNN 実 bridge (本流 llcore.verifier.refinement.is_marabou_available
  に倣う) で Z3 mock を超える検査
