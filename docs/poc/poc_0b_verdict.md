# PoC 0b Verdict — Synthetic sequence fitness (copy / addition)

調査日: 2026-05-29  
ファイル: `scripts/poc_0b_synthetic_fitness.py` + `src/llcore/fitness/tasks.py`  
Test: `tests/unit/test_poc_0b_fitness.py`

## falsifiable 命題

> decay/mix/gate_str の 3 パラメータ gene + 固定線形 readout で、
> copy task / addition task それぞれに対し
> (a) fitness は決定論的に計算可能で範囲 [0, 1] に収まり、
> (b) random gene 集団で fitness の分散が非自明 (各 task で gene が差別化される),
> (c) copy と addition で「最適 gene」が異なる (task 依存性),
> (d) baseline-fit gene の fitness は random gene 中央値より高い。

## 破綻ゲート (G1-G7)

- [x] G1: fitness ∈ [0, 1] + finite
- [x] G2: 決定論性 (同 gene/seed で完全一致)
- [x] G3: non-degenerate (random pop N=20 で variance > 1e-4)
- [x] G4: task dependency (copy vs add で rank correlation < 0.7 or mean diff > 0.01)
- [x] G5: gene sensitivity (各 parameter 摂動で fitness 変化 > 1e-3)
- [x] G6: baseline calibration finite (mse ∈ [1e-4, 1e4])
- [x] G7: reasonable best (200 random search で fitness > 0.3 達成)

## 実行結果 (2026-05-29)

```
PoC 0b verdict: PASS — fitness が機能、task 依存性 + gene sensitivity を満たす.
                 次段 PoC 0c (自前 minimal GA) に進めます.
```

| ゲート | 結果 |
|---|---|
| G1 fitness range | f_copy=0.313, f_add=0.271 |
| G2 determinism | diff=0 |
| G3 variance | var(copy)=1.09e-02, var(add)=1.26e-02 |
| G4 task dep | mean_diff=0.13, rank_corr=**-0.190** (低相関 = 強い task 依存性) |
| G5 sensitivity | decay+0.2: 0.058 / mix+0.3: 0.224 / gate+0.5: 0.136 |
| G6 baseline | copy=4.05e-01 / add=86.3 |
| G7 best 200 | best_copy=0.555, best_add=0.628 |

### 重要発見
- **rank_corr=-0.190** = copy と addition で gene 順位が**逆相関気味** → 強い task 依存性 (PoC 0c 進化で specialist 出現の前提が立つ)
- **mix perturbation が最強** (0.224) → 進化で mix gene が支配的に動く可能性
- **best fitness 0.555/0.628 > 0.3** → 進化探索の余地が十分残っている

## 設計判断

### Fixed linear readout (gene 進化対象外)
本 PoC は **state update gene の表現力**にフォーカス。readout を進化対象に
すると confound (gene と readout のどちらが効いているか分離不能)。Stage 後期で
readout も進化対象に拡張する場合は別 PoC として切り分け。

### Baseline-MSE 正規化
random gene 集団 (N=20, N_trials=5, seed 固定 calibration) の MSE 中央値を
baseline として固定。fitness = 1 - clip(MSE / baseline, 0, 1) で random gene
中央値が ≈ 0.5 になるように設計。

### Copy task の delay=0
最も単純な「直近入力を覚えてられるか」テスト。memory horizon は Stage 後期で
delay > 0 に拡張。

## 実行方法

```powershell
cd <llcore-root>
py -3.11 scripts/poc_0b_synthetic_fitness.py
py -3.11 -m pytest tests/unit/test_poc_0b_fitness.py -v
```

## 次段 (PoC 0c)

→ 自前 minimal GA で進化 10×10
   - `src/llcore/evolution/` に tournament + uniform mutation の minimal GA
   - PoC 0b fitness を使い 10 個体 × 10 世代の進化が完走
   - 全滅しない / best が単調非減少 / 多様性測定可
   - llive lldarwin_v2 への依存なし (独立路線)

## honest 留保

- mock task のみ (実 LLM scale ではない、proxy mechanism feasibility)
- copy delay=0 は memory horizon ゼロ、後段で 1, 4, 16 等の長期 delay test
- readout 固定で confound 排除しているが、readout 質に fitness が依存する
  inherent な制約は残る (Stage 後期 readout 進化で対応)
- baseline_mse 2 digit 差 (copy 0.4 vs add 86.3) は task 性質依存 (正常)
- G4 OR logic (mean_diff OR rank_corr) はやや lax の懸念あり (codex 指摘予定の場合 AND 強化検討)

## Codex pair review (v1→v2 fix)

### v1 で指摘された blocker (修正済)
- **AdditionTask.score() (abs比較) と calibrate_baseline (通常 MSE) の MSE 定義不整合**
  → SyntheticTask Protocol に raw_error() を追加、score と calibrate_baseline 両方が
  task.raw_error を呼ぶ形に統一。AdditionTask は raw_error 内で `(|pred| - target)^2`、
  CopyTask は `(pred - target)^2`。fitness は `1 - clip(raw_error / baseline, 0, 1)`。

### v1 で指摘された非 blocker (対応済)
- **G4 OR ロジック → rank_corr 必須 (主判定) + mean_diff 補助 (report only)**
- **G6 baseline_mse range が広すぎ → seed sweep ratio (max/min < 3) で頑健性 gate に**
- **copy delay=0 のみ → copy delay=4 (memory horizon) を G7 に追加** で memory-capable 主張強化
- **claim wording: "fixed-readout probe-based fitness" 表記に統一**
- **AdditionTask 命名注意: 実体は ||sum_t x_t||_2 regression と honest 明記**

### v2 実測 (修正反映後)
| Gate | v1 | v2 |
|---|---|---|
| G4 rank_corr | -0.190 (OR-pass) | **-0.201 (rank_corr 必須 pass)** |
| G6 baseline | range [1e-4, 1e4] 通過のみ | **copy ratio=1.11, add ratio=1.19 (頑健)** |
| G7 best | copy=0.555, add=0.628 | **copy0=0.518, copy4=0.525, add=0.703 (memory-capable)** |

### v2 Codex verdict
- **Green-light** (修正反映後 commit 可)

## 関連 memory

- [[project_llcore_init_2026_05_29]]
- [[feedback_codex_pair_review_for_llcore]]
- [[feedback_benchmark_honest_disclosure]]
