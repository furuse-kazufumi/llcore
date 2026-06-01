# Stage 3b Kernel 多様化 — 破綻ゲート (Break Gates) 登録簿

falsifiable な破綻条件を ID 付きで列挙。各 gate は **「これが起きたら設計が破綻」**を明文化し、
計測法と判定閾値を固定する。実装時に smoke / 本実験で各 gate を測り pass/fail を記録する。
honest 規律: 各 gate は独立に pass/fail/N-A を持つ。mechanism feasibility (BG1-5) と
specialist 実証 (BG6-8) と欺瞞地形 (BG9) を **混同しない**。

凡例: gate が **fail** = その行の「破綻条件」が真 = 設計の当該主張が崩れる。

---

## Stage 3a — mechanism feasibility (各 kernel が Stage1 gate を通せるか)

### BG1 — state_norm admit 集合が非空
- **破綻条件**: いずれかの kernel (rwkv/mamba/hopfield/linear_attn) で、clip 範囲から N=2000 個
  サンプルした gene の **state_norm Z3 gate admit 率が 0** (admit 集合が空)。
- **計測**: 各 kernel の bounds から一様サンプル → `verify_state_norm_kernel(kernel_id, theta)`
  (research 側、既存 `verify_gene_safe` の対角 over-approx パターンを kernel 別に一般化) →
  admit 率 = `#(solver_status=="unsat") / N`。
- **pass 閾値**: admit 率 > 0 (全 kernel)。linear_attn は §2.4 gene 制約 (`lam+softplus(2)·|v_gain|≤1`)
  満たす部分集合で > 0 を確認。

### BG2 — Lipschitz Z3 verdict と閉形式上界が一致
- **破綻条件**: あるサンプル gene で `(L_upper_bound < 1)` と `(solver_status == "unsat")` が
  **食い違う** (Z3 が certified と言うのに閉形式上界 ≥1、または逆)。
- **計測**: 各 kernel で N サンプル → Z3 `verify_kernel_lipschitz` と endpoint 閉形式
  `kernel_L_upper_bound` を両方計算 → 一致率。
- **pass 閾値**: 一致率 = 1.0 (timeout/unknown は除外集計し別途記録)。これは既存 rwkv の
  `_lipschitz_upper_bound` ⟺ Z3 unsat 一致 (BG3/(iv)) の kernel 拡張版。

### BG3 — empirical Lipschitz が Z3 上界以下 (over-approx soundness)
- **破綻条件**: ある kernel・gene で `empirical_L > L_upper_bound + tol` (経験微分が解析上界を超過
  = over-approx が unsound)。
- **計測**: 各 kernel の対角 `eval_step` を中央差分 (既存 `empirical_lipschitz` と同手法) で
  `max|∂s'/∂s|` 推定 → Z3 `L_upper_bound` と比較。
- **pass 閾値**: 全 gene で `emp_L ≤ L_upper_bound + 1e-3`。

### BG4 — kernel_swap 変異後の genome が decode 可能 & trajectory finite
- **破綻条件**: random walk で `kernel_id_shift` を 1000 回適用した genome 系列で、いずれかが
  decode 不能 (codec 例外) または `simulate` が NaN/Inf trajectory を返す。
- **計測**: ランダム `KernelGenome` から開始 → mutation (theta gaussian + kernel_id_shift) を
  1000 step → 各 step で decode + simulate (L=128, dim=8 有界入力) → finite チェック。
- **pass 閾値**: 全 step finite & decode OK。

### BG5 — 既存 rwkv 結果が KernelGenome 埋め込みで数値不変 (後方互換)
- **破綻条件**: `kernel_id=0` 経路の `simulate` 出力が、既存 `state_update.run_sequence` の出力と
  **bit 一致しない** (後方互換破壊)。
- **計測**: ランダム StateUpdateGene を `KernelGenome(kernel_id=0, theta=[d,m,g,junk])` に埋め込み →
  research decode → run_sequence と `np.array_equal` 比較 (同一入力列)。
- **pass 閾値**: 全サンプルで完全一致。**これが src 不変・後方互換の機械的担保**。

---

## Stage 3b — specialist 出現 / 単一 kernel に固定しない

### BG6 — specialist 出現 (task→best-kernel 写像が非定数)
- **破綻条件**: multi-task の全 task で **同一 kernel_id が best** (specialist 不在 = 1 kernel 万能)。
- **計測**: 各 task を単独 fitness にして進化 → best gene の kernel_id を集計 → task→kernel 写像表。
- **pass 閾値**: 写像が非定数 (≥2 種の kernel_id が少なくとも 1 task で best)。
- **honest**: fail (全 task 同一 kernel) は negative として正当 = 「memory_tasks は kernel 中立」。

### BG7 — archive kernel 多様性 (collapse しない)
- **破綻条件**: final MAP-Elites archive 占有 cell の kernel_id 分布が **単一値に collapse**
  (Shannon entropy ≈ 0)。
- **計測**: final archive の kernel_id ヒストグラム → 正規化 Shannon entropy H (bits)。
  gate-on / gate-off 両条件で測る (§7 リスク6: gate が多様性を殺す交絡)。
- **pass 閾値**: H > 0.1 bits (≥2 kernel が live occupancy)。

### BG8 — kernel 選択が load-bearing (固定 ablation に勝つ)
- **破綻条件**: kernel_id を進化させた版が、**各 kernel 固定 4 本のうち best**と比べ multi-task
  test 汎化で **有意に上回らない** (kernel 選択の付加価値ゼロ)。
- **計測**: kernel_id 進化版 vs 固定 4 本 (rwkv-only/mamba-only/hopfield-only/linear-only) を
  ≥15 seed、`selection_lab.compare` で paired Wilcoxon p + Cliff δ。test (hold-out) 主指標。
- **pass 閾値**: 進化版 - max(固定) で diff>0 & p<0.05 & |δ| 非無視。
- **honest**: fail (固定の方が良い) は negative として正当 = 「多様化は不要、最良 1 kernel で十分」。

---

## ③ 欺瞞地形 (Step4「探索空間拡張で unlock するか」の検定)

### BG9 — 拡張空間が ③ load-bearing な欺瞞地形を持つか
- **破綻条件 (= ③不要側)**: positive control では ③ が立つのに、実 multi-task では MAP-Elites が
  RR-hillclimb / panmictic-GA / random に対し優位消失 (Step4 §7 の「proxy 滑らか」が拡張空間でも再現)。
- **N/A 条件**: positive control すら ③ が立たない、または記述子 (n_bins) / 予算で判定反転
  (DECEPTIVENESS_MEASURE_VERDICT の循環論法/記述子依存が未解消)。
- **③成立条件 (= 欺瞞地形あり)**: positive control で ③ 成立 **かつ** 実 multi-task でも MAP-E が
  3 baseline 全勝 (p<0.05, δ 非無視) + 交絡 ablation (kernel_id を behavior から抜く) で優位が
  diversity 維持に帰属。
- **計測**: §6.3 の positive / negative control + 実 memory_tasks multi-task を Step4 exp4-5 harness
  (`selection_lab.run_methods_over_seeds`) を `KernelGenome` 空間で再走。事前登録: 固定 n_bins /
  固定予算 / 記述子不変 behavior。
- **判定**: ③成立 / ③不要 / N/A の 3 値。honest: 「整いすぎた ③成立」は内訳を疑う
  (`feedback_benchmark_honest_disclosure`)。

---

## gate 依存関係 (実行順序)

```
BG5 (後方互換) ──┐
BG1 (state_norm) ─┼─→ BG4 (swap 健全) ─→ Stage 3a feasibility 確定
BG2 (Lip 一致) ──┤
BG3 (over-approx)─┘
                            │
                            ▼
BG6 (specialist) ─→ BG7 (collapse なし) ─→ BG8 (load-bearing) ─→ Stage 3b 確定
                            │
                            ▼
BG9 (③ 欺瞞地形) ← positive/negative control 先行
```

BG1-5 が全 pass しないと BG6-9 は意味を持たない (壊れた kernel で specialist を測っても無意味)。
