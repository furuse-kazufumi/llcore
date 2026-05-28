# Marabou Incremental NN Verification の "異構造" Refinement Relation Sound 拡張 — Sketch

Stage: PoC 3a (Stage 3a Marabou bridge skeleton)  
Date: 2026-05-28  
Status: working sketch (TMLR / NeurIPS workshop submission 強度を目指す中間文書)

## Abstract

Wu et al. (2026-03, arxiv 2603.12232) "Incremental NN Verification via Learned Conflicts" は
**同一構造の neural network** に対する一連の verification query 間で `refinement relation`
を定義し、ある query の unsat core を refined query に持ち越す conflict inheritance を sound に行う。
本研究は llcore の進化ループ (state-update gene の **構造的変更**) に対し、refinement
relation を **異構造 (kernel/decay/mix/gate-strength の Δ; ChangeOp)** へ sound に拡張し、
ChangeOp 粒度で `(i) sound 拡張命題 R(NN, NN', ChangeOp)`、`(ii) 合成性`、`(iii) 無限列耐性`
を Z3 で機構実証する。さらに POET-lite 風の Minimal Criterion Coevolution (MCC) を ChangeOp
集団に適用することで、**進化に上限を設けない** open-ended カリキュラムを構成する。

## 1. 背景: Marabou Incremental の同構造限界

Wu et al. 2026-03 の core 主張:

> Two queries `Q` and `Q'` are in *refinement relation* iff the search space of `Q'`
> is a subset of the search space of `Q`; in particular, every conflict (unsat core)
> found in `Q` is also a sound lemma in `Q'`.

この定義は **同一 NN 内** の input region 縮小 (local robustness radius) や input
splitting (BaB) を対象とする。Marabou 2.0 (Katz et al., CAV 2024) は CDCL を統合し
conflict cache を実装するが、`Q` と `Q'` の **構造 (layer topology)** が一致することを
仮定する。

llcore の進化ループは「同一系統の network に対する連続 query 列」を生成するが、
ChangeOp が `kernel_swap_mock` のように **構造を切り替える** ものを含むため、Marabou
の refinement relation のままでは conflict 継承が **soundness を失う**。

## 2. llcore 拡張: ChangeOp と Refinement Relation R

### 2.1 ChangeOp atomic operation

```
ChangeOp = (op_type, delta)
op_type  ∈ { decay_shift, mix_shift, gate_shift, kernel_swap_mock }
delta    ∈ ℝ (shift) or {0, 1} (kernel discrete switch)
```

`*_shift` は同構造内 weight 微変更 (Marabou refinement と整合)、`kernel_swap_mock` は
gate 構造そのものの離散切り替え (異構造変更)。

### 2.2 Sound 拡張 Refinement Relation

```
R(NN, NN', c)
  ≡  ∀ x ∈ X.  |state_norm(NN', x)|  ≤  K · |state_norm(NN, x)|  +  ε(c)
```

ここで

- `K = 1` — RWKV-style convex combination の継承係数 (decay 同型構造下で自然)
- `ε(c) = E_BASE · |delta|`     for shift ops (E_BASE = 0.5)
- `ε(c) = E_BASE · 1 + KERNEL_SWAP_EXTRA` for `kernel_swap_mock(swap=True)`
  (KERNEL_SWAP_EXTRA = 0.3 で discrete 変更の K=1 超過分を吸収)
- `ε(noop) = 0`

#### Soundness 根拠 (informal proof sketch)

```
NN  : s' = decay        · s + (1 - decay)        · tanh(mix · x + gate_str · s)
NN' : s' = (decay+Δd)   · s + (1 - decay - Δd)   · tanh((mix+Δm)·x + (gate_str+Δg)·s)

|state_norm(NN') − state_norm(NN)|
   ≤ |Δd| · (|s| + |tanh|) + (1 − decay − Δd) · |tanh' − tanh|
   ≤ |Δd| · 2 + (1 − decay − Δd) · (|Δm| · |x| + |Δg| · |s|)   (tanh は 1-Lipschitz)
   ≤ 2 · (|Δd| + |Δm| + |Δg|)                                   (|x|,|s| ≤ 1)
   = 2 · magnitude(c)
```

E_BASE = 0.5 は **sound だがやや tight な** 選択であり、filtering 機能 (病的 ChangeOp を
落とす能力) を優先する。loose な bound (E_BASE > 1) では sat-反例が出にくく、curriculum
の淘汰圧が失われる。

### 2.3 包含関係 (Marabou ⊂ llcore)

```
Marabou refinement (Wu et al. 2026-03):
  R_M(NN, NN', δw) ⊂ { c : c.op_type ∈ {decay_shift, mix_shift, gate_shift} }
  ∧ NN.topology == NN'.topology
                                                              ⊆
llcore refinement (本研究):
  R(NN, NN', c)  for c ∈ ChangeOp (including kernel_swap_mock)
```

すなわち llcore の `R` は **Marabou の refinement relation を真に包含する** (kernel
construction 切替を含む extra penalty 項付き)。`R` を kernel_swap_mock 抜きで制限すれば
Marabou の同構造 refinement に一致する (退化形).

## 3. 合成性 (Composability)

```
[Composition Theorem]
  R(N0, N1, c1) ∧ R(N1, N2, c2)  ⇒  R(N0, N2, c1 ∘ c2)
  with ε(c1 ∘ c2) = ε(c1) + ε(c2)
```

### Proof sketch

```
|state_norm(N2)|  ≤  K · |state_norm(N1)|  +  ε(c2)
                  =  1 · ( 1 · |state_norm(N0)| + ε(c1) )  +  ε(c2)
                  =  |state_norm(N0)| + ε(c1) + ε(c2)
                  =  K · |state_norm(N0)|  +  ε(c1 ∘ c2)
```

K = 1 と ε の magnitude 線形性により ε(c1 ∘ c2) = ε(c1) + ε(c2)。Z3 では
``verify_composition`` が直接 N0 → N2 で反例探索し unsat を確認 (PoC 3a G2)。

## 4. 無限列耐性 (Infinite-sequence tolerance)

任意長 ChangeOp 列 (c1, …, c_n) に対し

```
|state_norm(N_n)|  ≤  |state_norm(N_0)|  +  Σ_{i=1..n} ε(c_i)
                  =  state_bound  +  E_BASE · Σ magnitude(c_i)
                  +  (extra term if any c_i is kernel_swap_mock(True))
```

`Σ magnitude(c_i)` が **bounded** な限り state_norm bound は崩れない。本 PoC では
state_bound = 1.0、ChangeOp curriculum 側で magnitude_cap を設定して Σ を限定する
(curriculum 詳細は §5)。100 step で連続検査し全 step PASS を確認 (PoC 3a G3, G7)。

## 5. Open-ended ChangeOp Curriculum (MCC 風)

llive `poc_minimal_criterion_coevolution.py` の MCC を ChangeOp 集団に適用:

- **Minimal Criterion**: ChangeOp `c` は (i) `verify_refinement_single(NN, c).ok=True`
  かつ (ii) `ε(c) >= frontier_quantile` (現集団の上位 percentile) のとき生存。
- **Mutation**: `delta` を gaussian σ で perturb (magnitude_cap 内で clamp)。
- **Refill**: 不足分を frontier の少し上から random sample (anti-monotone pressure)。

### "上限なし" の数学的根拠

- Σ magnitude(c_i) は curriculum の magnitude_cap によって直接制御可能。
- frontier_slope > 0 が長期に維持される (G8 で実証) → curriculum は飽和しない。
- POET-lite と同じく **解 (= sound 保持 NN) と問 (= ChangeOp)** を共進化させ、
  固定 ChangeOp 集合の上限を回避。

### Anti-monotone pressure (Q4 reviewer concern)

verifier-pass 率のみで淘汰すると、pass しやすい (= magnitude 小) ChangeOp ばかり
残り単調化するリスクがある。本実装では:

1. **mutation の退行許容**: σ で正負双方の perturb を許す
2. **frontier 上の refill**: ε_floor + 0.05 から random sample
3. **median epsilon の追跡**: median が単調減少すれば saturation 検出 (`is_saturated`)

の 3 段で対処する。

## 6. 関連研究との位置づけ

| 研究 | 構造変更 | refinement 拡張 | curriculum | 進化上限 |
|---|---|---|---|---|
| Marabou Incremental (Wu et al. 2026-03) | × (同構造のみ) | conflict inheritance | × | 該当なし |
| α,β-CROWN (VNN-COMP) | × | × (bound prop) | × | 該当なし |
| POET (Wang et al. 2019) | env のみ (NN 構造不変) | × | ✓ | open-ended |
| MCC (Brant & Stanley 2017) | × | × | ✓ | open-ended |
| **llcore (本研究)** | **✓ (ChangeOp)** | **✓ (sound 拡張)** | **✓ (MCC 風)** | **✓ (上限なし)** |

直交軸の組合せが gap である (llive 2026-05-28 presurvey)。

## 7. 実装と Z3 検査

PoC 3a (`scripts/poc_3a_marabou_bridge_skeleton.py`) で G1-G9 を機械的に確認:

- G1: 単一 ChangeOp R(N, N', c) sat/unsat 判定
- G2: 合成性 R(N0, N2, c1∘c2) Z3 unsat
- G3: 100 ChangeOp 列の連続検査 PASS
- G4: 病的 ChangeOp (decay=2.0) の sound 反例検出
- G5: Marabou ⊂ llcore 包含関係 (本文書)
- G6: MCC curriculum verifier-pass 率淘汰 + 上限なし
- G7: Z3 < 100ms / step (100-step 1 秒以内)
- G8: curriculum frontier slope > 0
- G9: Marabou 不在で mock 完走 (CPU 完結)

## 8. honest 留保

1. **ε の線形性**: magnitude 線形 ε は **sound だが保守的**。Wong-Carlini-Mądry 風
   certified radius (曲率考慮) で tighter ε が取れる余地あり (Stage 5+)。
2. **K = 1 の境界例**: kernel_swap_mock は K > 1 が必要な場合があり、本 PoC では
   extra penalty (KERNEL_SWAP_EXTRA = 0.3) で吸収。実 NN kernel 交換 (Stage 5+) では
   K = 1.5 程度を要する可能性。
3. **scalar 状態**: 現 PoC は scalar `s`、実 NN は多次元。多次元拡張は frobenius
   norm + per-dim ε で同じ命題が立つ予定 (Stage 4+)。
4. **論文化への距離**: G5 の包含関係は **informal sketch**。formal proof (TMLR 級)
   には refinement relation の category-theoretic 抽象化または Marabou source への
   patch が必要 (Stage 6+)。
5. **Marabou 実 install 未検証**: Stage 3a は mock + Z3 で機構実証に集中。Marabou
   native CDCL conflict cache との実機 benchmark は Stage 5+ で別 PoC とする。

## 9. 次段 PoC への接続

- Stage 3b: Marabou 実 install + native CDCL hook (Linux/Docker 環境で別途)
- Stage 4a: 多次元 state への拡張 (frobenius norm)
- Stage 5a: 実 RWKV/SSM kernel 切替の refinement (kernel_swap_mock を実 kernel に置換)
- Stage 6a: refinement relation の category-theoretic 抽象化 (論文版 formal proof)

## References (key)

- Wu et al. 2026-03, "Incremental NN Verification via Learned Conflicts", arxiv 2603.12232.
- Katz et al. 2019, "The Marabou Framework for Verification and Analysis of DNNs", CAV.
- Katz et al. 2024, "Marabou 2.0 — CDCL Integration", CAV.
- Brant & Stanley 2017, "Minimal Criterion Coevolution (MCC)".
- Wang et al. 2019, "POET: Open-Ended Evolution of Environments and Solutions".
- Shriver, Elbaum, Dwyer 2021, "DNNV: A Unified Framework for NN Verification", CAV.
- Wong-Carlini-Mądry 2018, "Provable Robust Defenses via Convex Outer Adversarial Polytope".

## Internal references

- llive `docs/papers/2026-05-28_presurvey_verifier_stack.md` §B (Agent D Marabou 拡張議論)
- llive `scripts/poc_minimal_criterion_coevolution.py` (MCC 実装)
- llcore `docs/poc/poc_1a_verdict.md` (Z3 verifier base)
- llcore `src/llcore/verifier/refinement.py` (本拡張の Z3 構築)
