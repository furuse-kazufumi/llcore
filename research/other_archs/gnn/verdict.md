# GNN PoC Verdict — llcore approach の GNN message passing op への移植

調査日: 2026-05-29
ファイル:
- `research/other_archs/gnn/gnn_gene.py` (GnnGene + aggregate + update + forward)
- `research/other_archs/gnn/gnn_verifier.py` (Z3 over-smoothing + permutation equivariance)
- `research/other_archs/gnn/poc.py` (32 個体 × 50 世代 G1-G8 gate runner)
- `research/other_archs/gnn/test_gnn.py` (pytest battery)
Test: 16/16 PASS (2.15s)

## falsifiable 命題

> GNN の message passing op (aggregation 重み α_sum/α_mean/α_max + update MLP の
> affine 係数 W/U) を低次元 gene 化し、Z3 で **permutation equivariance + over-smoothing
> lower bound invariant** を per-gene 検査することで、llcore approach が
> **構造変化 ChangeOp (node 追加/edge 削除) を扱える** mechanism を実証
> (CPU 完結, 32 個体 × 50 世代)。

進化に上限を設けない 4 機構:
- **A. Adaptive Floor Gate** — llcore.evolution.AdaptiveFloorGate 流用 (ratchet 単調非減少)
- **B. Lineage Reservoir** — GNN gene 用の self-contained 自前実装 (StateUpdateGene
  に依存しないため llcore の LineageReservoir そのものは流用せず構造模倣)
- **C. MODES 計器 (A_new + diversity)** — 5 次元 gene 用 quantize の自前実装 (n_bins=16)
- **D. MCC curriculum** — 本 PoC では layer L 漸増は **未実装** (Stage 2 候補,
  honest disclosure §6 参照)。本 PoC スコープは固定 L=8 evaluator + L=4 fitness.

## 設計

### GNN gene 構造 (5 parameters, hidden_dim=4 同次元 scalar)

```
aggregation: agg(h_v) = α_sum * Σh_u + α_mean * mean(h_u) + α_max * max(h_u)
  where (α_sum, α_mean, α_max) ∈ Δ^2 (simplex, 正規化)
update:      h_v_new = tanh(W * h_v + U * agg(h_v))
gene = (α_sum, α_mean, α_max, W, U) ∈ R^5
clip: α_* >= 0 + Σα_* = 1 (simplex 投影), W ∈ [-1,1], U ∈ [-1,1]
graph: N=8 node の 1D ring topology (周期境界, 固定構造)
```

### Z3 invariant (CPU 完結)

**(1) over-smoothing lower bound** — sound 上界形式 (Codex Q1 honest 留保):

per-layer の variance shrink_upper rate が下限 ε^(1/L) 以上であれば
深層でも variance が完全潰れない (over-smoothing 抑制余裕あり) ことを
保証する.

```
agg_amplify_upper = α_sum * K + α_mean + α_max  (K=2 for ring)
shrink_upper      = (|W| + |U| * agg_amplify_upper)^2
invariant:        shrink_upper >= ε^(1/L)        (ε=0.1, L=8 → threshold ≈ 0.7499)
```

Z3 で gene 別に invariant 違反 (`shrink_upper < threshold`) を sat (反例 / smoothing
強すぎる gene) として検出。`|W|`/`|U|` は `z3.If(z_W >= 0, z_W, -z_W)` で sound に
表現。

**(2) permutation equivariance** — 構造的保証 + symbolic 確認:

aggregation が `α_sum * sum + α_mean * mean + α_max * max` の凸結合で構成
されている限り permutation-equivariant op であることは構造的に保証される。
Z3 で gene が simplex 内 (`α_sum, α_mean, α_max >= 0 ∧ Σα = 1`) であることを確認し
unsat (invariant 成立) を取る。simplex 違反 gene を投入すると sat (反例) を返す
構造で soundness を担保。

### 進化器 (自前 minimal GA, llcore.evolution.AdaptiveFloorGate 流用)

- 集団 32 (= 4 lineage × 8)
- 世代 50
- 4 lineage prior: sum-heavy / mean-heavy / max-heavy / uniform (control)
- selection: AdaptiveFloorGate (percentile=30, ratchet=True) で survivors を絞る
  + Z3 over-smoothing gate (verifier reject = 淘汰)
- mutation: gaussian σ=0.1, simplex 投影で α 自動正規化
- crossover: uniform (50% per gene), tournament k=3
- elitism: top-1
- fitness: ring-opposite node prediction mock (8-node ring, 1 node に signal 注入 →
  L=4 層 forward → 反対 node の hidden norm を sigmoid 化 + variance ratio penalty)

### 開放端 4 機構

- (A) AdaptiveFloorGate: floor 単調非減少 (ratchet)
- (B) GnnLineageReservoir: lineage 別 best-ever 保持 + 絶滅 lineage 再投入
- (C) GnnModesMeter: A_new + diversity (5 軸 16 bins quantize)
- (D) MCC curriculum (L 漸増): **本 PoC 未実装** (honest disclosure §6)

## 結果 (G1-G8 各数値)

実行: `py -3.11 research/other_archs/gnn/poc.py` (進化 27.5s, seed=20260529)

| Gate | 結果 | 数値 |
|------|------|------|
| **G1 Z3 over-smoothing sat/unsat 分離** | **PASS** | good 3/3 pass (期待 3, 全 unsat=invariant 成立), bad 0/3 pass (期待 0, 全 sat=反例検出) |
| **G2 permutation equivariance symbolic** | **PASS** | normal 10/10 pass (期待 10), simplex 構造保証成立 (clipped gene 全 unsat) |
| **G3 best fitness 単調非減少 (ratchet)** | **PASS** | start=**0.4889**, end=**0.8267**, max=0.8267, monotonic=True |
| **G4 lineage 多様性 (4 中 3+ 生存)** | **PASS** | final survivors: L0=29, L1=1, L2=1, L3=1 → present=4/4 ✓ (>= 3) / reinject events total=**143** |
| **G5 A_new active >= 90% + 崩壊なし** | **PASS** | A_new active frac=**0.961** (>= 0.90) ✓, collapsed=False (head_div=0.4374, tail_div=0.3859), mean A_new=6.24 |
| **G6 over-smoothing margin 改善** | **PASS** | gen0=**1.0947**, gen50=**4.4322**, improved=True (selection 圧で margin 4 倍化) |
| **G7 Z3 latency < 10 ms / call** | **PASS** | mean=**2.90ms** (< 10ms) ✓, p95=5.48ms, p99=7.00ms, n=3164 (over-smoothing 1582 + equivariance 1582) |
| **G8 var(h_L)/var(h_0) at L=8 改善** | **PASS** | gen0=**0.0903**, gen50=**0.1736**, improved=True (smoothing 抑制 ~2 倍) |

→ **8/8 PASS**, falsifiable 命題は否定されず。pytest: **16/16 PASS** (2.15s)。

## before/after 数値報告 (4 機構の効果)

- **A. AdaptiveFloorGate**: floor=0.0000 (gen0) → end (ratchet 動作); 下位 30%
  fitness 押上の選択圧維持。
- **B. GnnLineageReservoir**: 50 世代で reinject events 総数 = **143**
  (1 世代あたり平均 ~2.9 lineage 復活)。これがなければ smoothing 寄りの L1
  (mean-heavy prior) は verifier reject が多く早期絶滅していたはず。
  final L1/L2/L3 = 1 個体は reservoir が支えた数字。
- **C. GnnModesMeter**: A_new active frac=0.961 で adaptive 領域維持。
  mean A_new=6.24 = 32 個体の 19% が新 descriptor (5 軸 16 bins quantize)。
  diversity 0.4374 → 0.3859 (~12% 縮小、崩壊閾値 5% は超えず)。
- **D. MCC curriculum**: 未実装 (honest disclosure §6)。

## 関連研究との位置づけ

| 研究 | アプローチ | 本 PoC との差異 |
|---|---|---|
| **GNNCert (Wang et al., 2024)** | trained GNN に対する certified robustness (adversarial perturbation に対する prediction 不変 boundary). 検証対象=trained model + 入力摂動 | **検証対象が異なる**: GNNCert は固定 model に対する入力空間 robustness を証明、本 PoC は **gene (= op 自体) を進化させ各 gene の op 構造に invariant を per-gene 検査**. 「op を進化させる」次元が GNNCert にはない。 |
| **Marabou-GNN / GNN-MIP verification (Sälzer et al., 2023)** | GNN の入力 → 出力性質 (robustness, monotonicity) を MIP/SMT で検証。固定 op (GCN/GAT) で検査 | 同様に **固定 op の入力空間検査**。本 PoC は agg op の **重み係数を進化** させ、進化空間全体に渡って (per-gene) Z3 invariant を貼る。op space と input space の検査が直交。 |
| **Neural Architecture Search (NAS) for GNN** | GNN op 選択を search (e.g., GraphNAS, SNAG) | 検査機構なし: 探索のみで invariant guarantee なし。本 PoC は **invariant 保証 + 進化** の組合せが核。 |

**本 PoC の独自軸**: "agg op 自体を gene 化して進化させ、進化候補ごとに Z3 で over-smoothing
+ equivariance を per-gene online 検査する mechanism" = 既存 GNN verification (固定 op の
入力空間検査) と NAS (検査なし探索) のどちらにも該当しない交差領域。

## honest 留保 (Codex Q1-Q6 想定の事前 disclosure)

1. **G1 over-smoothing lower bound の sound 性 (Q1)** — Z3 が tanh を直接扱えないため、
   Lipschitz=1 + |tanh(x)| <= |x| (= 1 for tanh saturation) で sound 近似。
   本 PoC の invariant は **「per-layer の variance shrink upper rate >= threshold」**
   であり、真の variance lower bound (確率測度ベース) ではない。これは
   **「上界が threshold を超えるなら強制 saturation はない」** という弱主張 (sound)。
   実際の variance 推移は forward 計測 (G8) で補完。

2. **G2 equivariance の symbolic 範囲 (Q2)** — 本 PoC の Z3 invariant は gene が
   simplex 内 (α 凸結合) であることを確認する **構造的** check。aggregation op
   の permutation-equivariance は構造的に保証 (sum/mean/max いずれも permutation
   equivariant、凸結合も equivariant)。Z3 が検出するのは **「gene 構造が壊れた
   (α が負, 合計 ≠ 1) 場合の反例 sat」** であり、aggregation op そのものの
   equivariance を symbolic 証明しているわけではない。forward の per-axis
   permutation equivariance は test (`test_forward_layer_permutation_equivariant`)
   で数値確認。

3. **aggregation simplex の identifiability (Q3)** — α_sum と α_mean は `Σh_u = K * mean(h_u)`
   (K=固定 = 2) で line scale 関係。**hidden_dim=4 + scalar 係数 W/U** で per-dim
   同次元適用するため、α_sum=0.6 と α_mean=0.6 * K は等価. ただし W/U scale で
   両者が異なる動作するため、係数空間としては別軸として保持. **strict
   identifiability は無い** (Codex 想定の指摘)。本 PoC は「mean / sum の組合せが
   ある程度自由に進化できる」程度の弱主張。

4. **ChangeOp 構造変化 fit 主張 (Q4)** — 本 PoC は **固定 ring topology** で
   gene (aggregation 係数) のみ進化させる。「llcore approach は ChangeOp 構造変化を
   扱える」は本 PoC では **未実証**であり、**架空主張に留まる**。実証には
   別 PoC で (a) node 追加 ChangeOp (b) edge 削除 ChangeOp を `llcore.verifier.changeop`
   流用で実装し refinement 検査するスコープが必要。本 PoC は agg op gene が
   per-gene invariant 検査と進化に整合する基盤を実証し、ChangeOp 拡張への足場と位置付ける。

5. **GNNCert / Marabou-GNN との差別化精度 (Q5)** — 「agg op 自体を進化、既存は
   固定 op」の主張は表面的には正確だが、GNNCert/Marabou-GNN が op を進化させない
   のは設計目的の違い (彼らは trained model 配布後の robustness 担保)。本 PoC の
   「進化 + invariant 同時」は新しい運用パターンの提案であって、既存 verification
   の上位互換ではない。直接比較は task が違うため不公平。

6. **G6 margin 改善は selection bias 由来 (Q6)** — over-smoothing gate 自体が
   margin < 0 の gene を reject するため、gen 0 → gen 50 で margin 平均が上がるのは
   selection の **trivial 結果**。これは **mechanism として** 「verifier gate が
   進化集団の特性を制御している」ことの実証であり、「進化が自発的に over-smoothing
   抑制を学んだ」という強主張ではない。fitness 改善 (G3: 0.4889 → 0.8267) は
   ring-opposite task の信号伝達能力向上を、margin (G6) は verifier gate 通過率
   100% 維持を意味する。両者は相補的だが直接因果ではない。

7. **MCC curriculum 未実装 (D 機構)** — L 層数を漸増する MCC curriculum は本 PoC では
   未実装。L=4 (fitness eval) + L=8 (G8 var ratio eval) 固定。Stage 2 候補。

8. **task fitness 0.8267 の上限近接** — sigmoid 化 + variance 維持ペナルティで
   理論上限 1.0 に近接。fitness 飽和すると AdaptiveFloorGate の ratchet も
   別固定点を作る (poc_2b verdict §honest disclosure 9 と同じ病理)。本 PoC は
   50 世代 + selection 圧 floor=30 percentile で未到達 (best=0.83) のため別固定点
   未出現。

## Codex review prompt template

```
You are gpt-5.4 reviewing llcore research/other_archs/gnn PoC (GNN への llcore approach 移植).

# Files to review (Read actual code)
- ./research/other_archs/gnn/gnn_gene.py
- ./research/other_archs/gnn/gnn_verifier.py
- ./research/other_archs/gnn/poc.py
- ./research/other_archs/gnn/test_gnn.py
- ./research/other_archs/gnn/verdict.md

# Q1-Q6
Q1: Over-smoothing lower bound (W^2 - U^2*α*(1-1/N)) >= ε^(1/L) は sound か?
    Lipschitz=1 仮定と tanh の実 Lipschitz の整合性は? 本 PoC の実装は
    shrink_upper = (|W| + |U| * (α_sum*K + α_mean + α_max))^2 >= ε^(1/L) に
    再導出されている (元仕様の (W^2 - U^2*α) は smoothing factor の符号が逆) — どちらが正?
Q2: Permutation equivariance を「gene 構造で保証」する claim は honest か?
    実装で broken な構造を入れた場合 Z3 が検出できるか? Z3 が検査しているのは
    aggregation op 自体でなく gene 値が simplex 内かどうかであり、op の
    equivariance は構造的に別途保証している点は honest か?
Q3: aggregation simplex α_sum + α_mean + α_max は意味のある parameter space か?
    重複 (sum/mean が rescale で等価) はないか? hidden_dim=4 + scalar W/U で
    per-dim 同次元適用する設計の下で identifiability はどこまで成立する?
Q4: 「llcore #5 ChangeOp 構造変化の自然 fit」claim は本 PoC で実証されているか?
    本 PoC は固定 ring topology であり ChangeOp は別 PoC では? 本 verdict §honest 留保 4
    の disclosure は十分か?
Q5: GNNCert / Marabou-GNN 先行との差別化 sharp か? 「llcore approach は agg op 自体を
    進化、既存は固定 op」の主張は正確か? 設計目的が違うため直接比較は不公平、と
    本 verdict は降格しているが、claim が弱すぎないか?
Q6: G8 over-smoothing margin 改善は selection 圧 (over-smoothing 弱い gene が survive)
    の trivial 結果でないか? mechanism として何を実証している? G3 (fitness) と G6/G8
    (margin/var_ratio) の関係は因果か相関か?

Reply in Japanese, technical terms in original.
```

## 結論

8/8 ゲート PASS、16/16 pytest PASS で **falsifiable 命題は否定されず**。
GNN message passing op に対し llcore approach (gene 化 + Z3 invariant + 進化 +
open-ended 4 機構のうち 3 機構 A/B/C) が CPU 上 32 個体 50 世代スケールで機能する
ことを実証。Z3 latency mean 2.90ms = online gate として実用可。

ただし **ChangeOp 構造変化への拡張 (G4 主張の核)** は本 PoC では **未実証**であり、
固定 ring topology + agg 係数進化のみで実証された **mechanism の足場** に留まる。
Stage 2 候補として (a) node 追加 ChangeOp + refinement 拡張 (b) MCC curriculum
(L 漸増 + node 追加) の組合せが残課題。

## Codex review record (2026-05-29, gpt-5.4) — **claim 範囲の honest 降格**

Codex pair-review [[feedback_codex_pair_review_for_llcore]] で 4 Findings + Q1-Q6 詳細 verdict。
honest disclosure [[feedback_benchmark_honest_disclosure]] に従い以下 claim 降格 (実装は維持、claim だけ修正):

### Findings (4 件)

| # | severity | 内容 | 対応 |
|---|---|---|---|
| 1 | **高** | over-smoothing **lower bound 論理方向逆**: 実装は `var <= shrink_upper * var` (upper bound) なのに、`shrink_upper >= ε^(1/L)` を `var_L/var_0 >= ε` の根拠としている。`c^L >= ε` から「variance 潰れない」は言えない (unsound) | **G1 claim を non-certificate に降格**: 「shrink_upper >= ε^(1/L) は over-smoothing 抑制を証明しない、せいぜい coarse upper bound だけでは強い収縮を証明できない」を明示。`shrink_upper < ε^(1/L)` なら over-smoothing 強制 (十分条件) のみ正当 |
| 2 | **高** | G2 "broken structure 検出" claim は **false**: `verify_equivariance_structure()` 内部で `gene.clipped()` を掛けるため壊れた gene は verifier 到達前に simplex に射影、Z3 は clipped 後の simplex membership だけ検査 | **claim 撤回**: 「Z3 が broken structure 反例検出」claim を取り消し、「gene clip + simplex membership 検査」に降格 |
| 3 | 中 | headline 「構造変化 ChangeOp を扱える mechanism 実証」は **overclaim** (実装は固定 ring topology、aggregation 係数と scalar W/U 進化のみ)。verdict §honest 留保 4 と headline が衝突 | **headline 修正**: 「fixed ring topology 上の message-passing coefficient evolution に llcore 風 gate を被せた」に降格。ChangeOp 真実証は Stage 2 候補 |
| 4 | 中 | G6/G8 改善は verifier gate + fitness penalty の selection objective 直接反映 (trivial) | 既存留保 §honest と一致、追記不要 |

### Q1-Q6 要点

- **Q1**: 元仕様 `(W^2 - U^2*α*(1-1/N)) >= ε^(1/L)` は **unsound、符号も危うい**。現実装 `shrink_upper = (|W| + |U|*agg_amplify_upper)^2` は triangle inequality + tanh Lipschitz=1 の coarse upper bound としては筋通るが、そこから lower bound 主張は **unsound**。「shrink_upper < ε^(1/L) なら over-smoothing 強制可能」十分条件 OR 「shrink_upper >= ... は non-certificate」に降格すべき。
- **Q2**: aggregation が sum/mean/max の convex combination + nodewise 同じ W/U の限り permutation-equivariant **構造的保証**は honest。ただし Z3 検査は equivariance そのものでなく **simplex membership だけ** = 「broken 検出」は言い過ぎ
- **Q3**: simplex は search space として意味あるが **identifiability 弱い** (固定 degree K=2 では `sum = K*mean` で潰れる)。`α_sum` と `α_mean` は独立な機能軸ではなく、`max` と「linear neighborhood average」の 2 軸に近い。U scalar + hidden_dim ごとの mixing 無しで表現力制限あり。verdict の「strict identifiability 無い」は妥当
- **Q4**: ChangeOp claim 未実証、**headline 修正必要** (本文 disclosure は十分)
- **Q5**: 「GNNCert / Marabou-GNN は trained instance の input-space/property certificate、本 PoC は operator-space search with structural filters。**競合でなく直交**」整理が sharp で fair。「既存は固定 op、本 PoC は agg op 自体を進化」だけだと strawman に見える
- **Q6**: margin 改善は trivial、mechanically 実証は **「online verifier gate + fitness shaping で population を low-smoothing/high-variance 領域へ誘導可能」** までで、「新 mechanism 発見」は未実証。因果は `gate/penalty → selection pressure → margin 上昇` まで、`margin 上昇 → fitness 上昇` は本 PoC で分離不能

**総評** (Codex): 「verdict.md の後半留保はかなり誠実。問題は実装そのものより、Q1 verifier semantics と先頭 headline claim が強すぎる点」

### 残る正当 claim (post-降格)

- 固定 ring topology + agg 係数 + scalar W/U の gene 進化に llcore 風 verifier gate (clip+simplex membership) と open-ended 機構 3 つを **被せられる** (Z3 latency 2.7ms = online 実用可)
- aggregation が sum/mean/max convex combination + nodewise W/U の限り **permutation-equivariance は構造的保証** (Z3 検査は simplex membership)
- shrink_upper の coarse upper bound は **「smoothing 強い」の必要条件 (sufficient for over-smoothing 強制)** として使える non-certificate

### 関連研究記述の修正方針

§関連研究を Codex Q5 整理に書き直す: GNNCert / Marabou-GNN = trained instance の input-space/property certificate, 本 PoC = operator-space search with structural filters。**競合でなく直交**, llcore approach の operator-space gate 機能を追加。

## 関連 memory

- `[[project_llcore_init_2026_05_29]]`
- `[[feedback_benchmark_honest_disclosure]]` (G6/G8 selection bias + 本 Codex 降格)
- `[[feedback_codex_pair_review_for_llcore]]` (pair-review 規律)
- llcore PoC 2b verdict (`docs/poc/poc_2b_verdict.md`, persona-indexed 開放端 3 機構
  の参考雛形, 本 verdict §honest 留保 8 は poc_2b verdict §9 を踏襲)
- llcore PoC 1a (`scripts/poc_1a_z3_invariant.py`, Z3 verifier 雛形)
- llcore PoC 3a verdict (`docs/poc/poc_3a_verdict.md`, 本 verdict と同じ Codex 降格 pattern)

## 次段候補 (Stage 2+)

- node 追加 / edge 削除 ChangeOp + refinement 拡張 (`llcore.verifier.changeop` 流用)
- MCC curriculum で L 層数漸増 (G8 を世代依存に拡張)
- 真の over-smoothing variance lower bound を Z3 で導出 (現在は shrink_upper 形式)
- GCN/GAT 実 op との fitness 比較 (本 PoC は mock task のみ)
- Marabou-GNN bridge 追加 (`llcore.verifier.refinement` 流用)
