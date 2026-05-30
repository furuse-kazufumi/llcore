# 梯子段1: 複数 reservoir 結合 (多細胞基質) — 設計

> Step C verdict の「次 (CPU 安価 2 本柱)」(1) の実装設計。
> research/ 隔離 (`research/ladder1_multi_reservoir/`)、src 非変更、push 未。

## 背景 (どこから来たか)

Step C verdict (2026-05-30, `STEP_C_VERDICT.md`) の結論:

- 単一 leaky-delay-line reservoir + ridge readout = **単細胞レベル基質**。
- `delayed_parity` (5-bit XOR) は **基質の床** — 単一 reservoir は XOR を線形分離可能な
  特徴を作れず (Minsky-Papert)、全 method で R²≈0。③ (選択圧) 以前の表現力ボトルネック。
- `flip_flop` は天井効果で underpowered。
- → ③ をクリーンに検定できる「欺瞞的だが航行可能な回廊」が本基質に自然発生しなかった。

「③ は無力」ではなく「**この基質・タスク帯で③の有無を測れなかった**」が honest な現状。
複雑さの梯子を一段上げる = **単細胞 → 多細胞 (複数 reservoir 結合)**。

## 仮説 (梯子段1)

**線形記憶 × 非線形分業**: 浅い層が異なる時定数 (leak) で過去を線形に保持し、深い層が
前層の状態を tanh で非線形合成すれば、ridge (線形) readout からでも XOR を分離可能な
特徴空間が立ち上がる (DeepESN: Gallicchio & Micheli 2017)。

→ もし床が外れれば、その新しい landscape が多峰/欺瞞的かを測り、③ (分離が選択を可能に
するか) をクリーンに検定できる土俵が初めて生まれる。

## 基質: DeepReservoir (`multi_reservoir.py`)

K 層スタック leaky-integrator:

```
layer 0:   u_0[t] = x[t]                          (in_dim_0 = task.in_dim)
layer k>0: u_k[t] = h_{k-1}[t]                     (in_dim_k = n_taps_{k-1})
h_k[t] = (1 - a_k) ⊙ h_k[t-1] + a_k ⊙ tanh(W_in_k @ u_k[t] + h_k[t-1])
readout 特徴 = concat([h_0[T-1], ..., h_{K-1}[T-1]])
```

- gene = 層ごとの (leak_raw, W_in)。値域は単一版と同一 (leak_raw∈[-4,4], W_in∈[-2,2])。
- **単一層 `DeepReservoir((N,))` は step_c の `LeakyDelayLineReservoir` と数値一致**
  (test で atol=1e-12 検証済) → 多層化は単一層の真の一般化、回帰なし。
- `make_eval_once` / `make_behavior` は単一版と同契約 (selection_lab / strict_compare 流用可)。

## 実験計画 (3 段、安価→決定的の順)

### ① 表現力 sanity — 床が外れるか (`exp_l1_expressivity.py`)

random search で各構成の **到達可能天井 (max held-out R²)** を測る。

| 構成 | total_taps | 深さ | 役割 |
|---|---|---|---|
| 1L-8 | 8 | 1 | Step C の床の再現確認 |
| 1L-16 | 16 | 1 | taps 増のみ |
| 2L-8×8 | 16 | 2 | **1L-16 と同規模・深さ2 = 深さの効果を分離** |
| 3L-8×8×8 | 24 | 3 | 深さ3 |

**誤帰属の回避**: 「床が外れた」が深さ (非線形合成) のおかげか単なる taps 増かを、
total_taps を揃えた `2L-8×8 vs 1L-16` の strict gate で分離する。各 seed で全 gene を
同一 train/eval データで評価 → 構成間 paired 公平。

判定:
- 床が外れた = 多層 max R² が 1L-8 を強化 honest 基準 (片側 Wilcoxon p<0.05・
  |paired_sign_delta|≥0.147・n_seeds≥15相当) で有意に上回る。
- 深さの寄与 = 2L-8×8 が 1L-16 を有意に上回る (上回らなければ規模で説明可)。

### ② landscape 多峰性 (床が外れた場合のみ)

外れた最小構成で C1 多峰性診断 (収束点間の谷検出, `landscape_map.py` 流用)。
単峰 broad-basin なら ③≡hill-climbing で ③ は不要 (exp7 と同型)。

### ③ niching vs baseline + lexicase (多峰なら)

MAP-E vs {random, RR-hillclimb, panmictic-GA, **lexicase**} を strict gate。
- **lexicase 追加理由**: Boldi/Ding/Spector 2023 (arXiv:2311.02283) — 欺瞞では
  lexicase > QD。Step C で「QD≈baseline は③無力でなく QD 次善かも (lexicase 未検定)」と
  保留した wrong-tool 仮説を排除する。各 test case (sequence) ごとの段階フィルタで親選択。

## §6(g) 反証打ち切り (無限後退の停止)

Step C verdict §6(g): 結合 reservoir + lexicase baseline + 軌跡 fitness を全部入れても
C3 (③ の strict gate) が全条件で不通過なら、**③ は本 landscape パラダイムで非
load-bearing 確定**。梯子段1 はこの打ち切り条件の主要部分 (結合 reservoir + lexicase) を担う。

## 規律

- 各 exp 後に Codex (gpt-5.4, read-only) pair-review、findings を実コード検証して採否
  ([[feedback_codex_pair_review_for_llcore]] / [[feedback_external_ai_verify]])。
- honest disclosure: 床が外れない / 深さが効かない / ③ が立たない、いずれも negative-
  but-informative として記録 ([[feedback_benchmark_honest_disclosure]])。
- 結果は `LADDER1_VERDICT.md` に集約。push 未維持。

## 結果

### ① 表現力 sanity — deep 機構 (DeepESN, N_SEEDS=15, random search)

`exp_l1_results.json` (delayed_parity, seq_len=20/window=5, N_RANDOM=400):

| 構成 | total_taps | max R² (mean±std) | vs 1L-8 (strict) |
|---|---|---|---|
| 1L-8 (床) | 8 | 0.021 ± 0.036 | — (Step C 床を再現) |
| 1L-16 (taps増のみ) | 16 | 0.019 ± 0.035 | δ=−0.33 FAIL (無効) |
| 2L-8×8 (深さ2・同規模) | 16 | 0.051 ± 0.068 | δ=+0.47 p=0.010 **PASS** |
| 3L-8×8×8 (深さ3) | 24 | 0.096 ± 0.096 | δ=+0.60 p=0.004 **PASS** |
| 深さの寄与: 2L vs 1L-16 (同規模16) | — | +0.032 | δ=+0.47 p=0.010 **PASS** |

**確定した2点 (honest に分離)**:
1. ✅ **深さ (層間非線形合成) が統計的に有意に効く**。同規模の taps 増 (1L-16) は無効 (δ=−0.33) なのに深さ2は床を有意に持ち上げる → 「効くのは規模でなく深さ＝多細胞の非線形分業」。Step C の単細胞床を**部分的に緩和**。
2. ⚠️ **絶対値は低く parity 未解決** (2L=0.05/3L=0.10、完全解 R²=1 に程遠い)。**「床が統計的に持ち上がった」≠「外れた」**。random search の探索不足が残り、真の表現力天井は進化探索で測る必要。

**最初の run の教訓 (自己訂正)**: 初回 N_SEEDS=10 < strict_compare の min_seeds=15 で全 `passes=False` の機械的偽陰性を出した。N_SEEDS=15 + `assert N_SEEDS>=15` で修正済 ([[feedback_benchmark_honest_disclosure]])。

### 床外し機構の診断 (ワークフロー ladder1-floor-break, 5機構)

深さ以外の床外し原理を並列に切り分け中: `parallel_gated` (乗法結合=多細胞分業) / `quadratic_readout` (明示2次=readout 対照) / `evolved_search` (進化で真天井) / `wide_single` (幅の効果) / `hybrid_max` (上限 anchor)。各機構を held-out R² で測定し、床外しを reservoir 表現力/readout/幅/探索/artifact のどれに帰属するか adversarial verify。

(ワークフロー完了後に追記 → LADDER1_VERDICT.md に統合)
