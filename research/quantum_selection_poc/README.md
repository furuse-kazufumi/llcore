# QUBO 多様性選択 — 最小 PoC

進化ループの「次世代に残す K 個体を選ぶ」段を **QUBO** に定式化し、量子アニーラー (QPU) へ
drop-in できる構造で古典サンプラーが機能するかを確認する最小 PoC。**深追いしない方針** で、
量子コーパス novelty 判定 (memory:project_creative_corpora_2026_06_07) を受けた sanity 実験。

## スコープ (厳格)

- **測ること**: (1) QD selection の QUBO 定式化が正しく解けるか (2) QPU に drop-in できる構造か。
- **測らないこと**: QPU 優位。本 PoC は QA vs 古典のベンチではない。
  honest disclosure 規律 ([[feedback_benchmark_honest_disclosure]]): QPU vs tuned SA vs random の
  3 比較は将来課題。本 PoC 自体は novelty を主張しない (QUBO 選択は先行多数)。

## 定式化

集団 N 個体から K 個選ぶ二値 `x_i ∈ {0,1}`:

```
minimize  E(x) = − Σ f_i x_i + λ Σ_{i<j} s_ij x_i x_j + A (Σ x_i − K)²  [+ μ Σ (1−cert_i) x_i]
```

`f` = fitness、`s` = cosine 類似度 (多様性ペナルティ)、`A` = 基数制約、`μ` = **cert gate**。
対称行列 `Q` は `dimod.BinaryQuadraticModel` へ 1:1 変換でき、`neal` / D-Wave QPU に無改造で乗る。

## 結果 (`results_qubo_selection.json`)

| 設定 | random | sa_classical | greedy 群 |
|---|---|---|---|
| small (N=16, 全 1820) | **opt** (空間が小さく解けてしまう) | opt | gap 1.05 |
| medium (N=28, 全 118 万, random は 0.2%) | gap 1.37 (未到達) | **opt** | gap 1.92 / 6.93 |

- **結論**: 定式化は古典 SA で解ける。medium で SA だけが exact に一致し、random・fitness-greedy・
  diverse-greedy を上回る。fitness-only は多様性が死ぬ (div 0.333 vs SA 0.458)。
- **honest 所見**: small (N=16) では random も最適に届く — 問題が小さすぎる証拠であって SA の弱さでも
  random の強さでもない。規模を上げて初めて定式化の価値が見える。
- **cert gate 接続点**: unsound 高 fitness 個体を `μ` penalty で選択から排除できる (両設定で全排除)。
  = llcore の Z3 健全性 gate を「圧縮/選択の受理関数」へ写像する象限 (novelty 判定で唯一未踏寄り)。

## 実行

```
py -3.11 research/quantum_selection_poc/qubo_diversity_selection_poc.py
```

依存 = numpy のみ (Optional extras 規律: `neal`/`dimod` は不在で自前 SA が代替)。

## 深追いする場合の次手 (未着手)

1. **QPU vs tuned SA vs random の 3 比較** (D-Wave Leap 無料枠, 分単位) — QA 優位は falsifiable に。
2. **certificate-gated evolutionary compression** (novelty の芯): TN bond/rank/mask を genome 化し、
   llcore Z3 gate を受理関数にした verified compression × 進化 (verified_compression_gap.md)。
3. llive selection への実配線 (本 PoC は self-contained で本体非接続)。
