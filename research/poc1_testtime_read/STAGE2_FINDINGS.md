# PoC-1 Stage 2 — 凍結 Gated DeltaNet 状態への post-hoc read 変種比較 (2026-07-11, CPU)

> pre-reg: `docs/research/preregistration/prereg_poc1_testtime_read_2026-06-29.md` §3.2/§4/§5。
> honest 規律: `feedback_benchmark_honest_disclosure`。★異常/都合の良い結果は内訳を疑う。
> 図・データ: `figures/` (recall_vs_K.png / variants_bar.png / cleanup_prob.png / figures_data.json)、
> 生結果: `stage2_results.json`。

## 設計 (honest)

- **1 層モデル** = 「単一の凍結状態 S」を明確化。各 query 位置で **native forward の next_state** を凍結し
  R0 = セル native read `S q̂` と**厳密一致**(assert 済)。downstream(gate/norm/proj/lm_head)は学習済みを固定
  → read 変種は `o` だけを差し替える純粋比較。
- ハイパラ (λ/τ/K/η/β) は **val split で選択 → test で評価**(test 上フィッティング禁止)。
- 統計: **paired bootstrap CI**(2000 resample)。3 seed。SWA 窓不使用。負荷(pair 数)=系列長の CPU 代理。

## 結果 (num_pairs=6, chance=0.062, 3 seed 平均)

| read 変種 | test recall | Δ vs R0 | 有意 seed |
|---|---|---|---|
| R0 (single) | 0.268 | — | — |
| R-CCQ (単発曲率収縮=kill-risk 対照) | 0.268 | +0.000 | 0/3 |
| sparse-FY (1-step sparsemax) | 0.273 | +0.005 | 0/3 |
| softmax-Hopfield | 0.277 | +0.009 | 2/3 |
| **R-Hopfield (soft-threshold cleanup)** | **0.286** | **+0.018** | **3/3** |
| R-ISTA | 0.286 | +0.018 | 3/3 (但し下記) |

## ★honest 内訳(図 recall_vs_K が露呈した重要な訂正)

1. **効果は小さいが頑健**: 単発 read (R0 0.268) に対し cleanup read で +0.018 (0.286)、3/3 seed で CI 超。
   R-CCQ(賢い単発)は +0.000、sparse-FY(sparsemax 1-step)も非有意 → gain は「単発の工夫」でも
   「単なる疎性」でも説明できない(H2 kill-risk を通過)。
2. **★"反復"ではなく"1 回の cleanup"が本体**: recall_vs_K 図で、**R-Hopfield は K=1 で既に +0.018 に到達し
   K=5 まで平坦**。すなわち反復は不要で、**単発の非 softmax soft-threshold cleanup 1 ステップ**が gain の源。
3. **★R-ISTA は K に対し不安定に振動**(K=2/4/6 で recall 0.07–0.14 に崩壊、K=1/3/5/7 で 0.29)。
   val 選択の K=5 がたまたま良い位相を引いただけで、R-ISTA の「反復 gain」は **step-size 由来の
   artifact**(NOODL 流 unrolled ISTA の条件数依存)。→ R-ISTA を「反復 read の勝ち」と読むのは誤り。
4. 従って pre-reg H1 の「**反復** read (K=3–5)」framing は**本 setting では支持されない**。支持されるのは
   「学習・非直交 key の凍結 gated-delta 状態に対し、**単発の非 softmax cleanup read** が単発線形 read を
   小幅だが頑健に上回る」。

## 判定 (pre-reg §5、H1/H2 のみ。H3 未実施)

- **GO-with-caveats**(H1∧H2 は R-Hopfield が満たす: R0 も R-CCQ も CI 超で上回る、3/3 seed)。
  ただし **"iteration" は不成立**(K フラット/ISTA 不安定)→ 主張は「反復」でなく「cheap cleanup read」。
- **read 側 test-time は dead ではない**(NULL 棄却)= 凍結線形状態の read 改善に**実シグナルあり**。
  → GPU スケール追確認の価値を確定(pre-reg の GO 条件を満たす方向)。
- ただし効果小・CPU tiny・ceiling-relaxation(天井は不変=突破ではない)。

## 未実施 → 次(Stage 2b / GPU)

- **H3(状態種 ablation)必須**: gated-delta / vanilla-additive / delta-rule で cleanup gain が
  「gated/delta のみで出る」ことを確認(pre-reg GO の第3条件)。vanilla-additive で同じ gain が出るなら
  「どの状態でも効く一般的 read トリック」= novelty 縮小、を切り分ける。
- mandated baseline は softmax-Hopfield(2/3 有意で cleanup に近い)まで確認済。sparse-FY は非有意。
- **GPU 本走**: Qwen3-1.7B 蒸留 linear-attn 状態 + 長系列(2k/4k/8k)。CPU の +0.018 が長系列で拡大/消失かを見る。
- honest 注記: cleanup gain の source は「crosstalk 部分除去」(cleanup_prob 図: 正解値の確率が上がる)。
  これは fixed-resolution 状態からの読み出し改善であって **plateau 突破ではない**(P7=2504.14366)。

---

## H3 結果 (状態種 ablation, 2026-07-11, 3 状態種 × 3 seed) — SUPPORTED

faithful セルを変えず instance レベルで step を差し替え、3 状態種で cleanup gain (R-Hopfield vs R0) を測定。

| 状態種 | mean R0 | mean cleanup gain | 有意 seed |
|---|---|---|---|
| gated_delta (本命) | 0.268 | +0.019 | 3/3 |
| delta_rule (忘却なし) | 0.264 | +0.021 | 3/3 |
| vanilla_additive (純 Hebbian・最弱) | 0.286 | +0.001 | 0/3 |

**H3 SUPPORTED**: cleanup gain は gated-delta/delta-rule で出て **vanilla-additive で消える**。
しかも vanilla-additive は R0 自体が高い (0.286)=単発読みが既にクリーンで headroom が無い。
→ gain は「どの線形状態でも効く一般トリック」ではなく **gated/delta 状態の crosstalk 特有**。
pre-reg の novelty regime (学習・非直交 key の凍結 gated-delta 状態への post-hoc cleanup) を裏付け。

## 総合判定 (H1/H2/H3) — GO(反復 framing のみ訂正)

- H1 (cleanup > R0): 支持 / H2 (> R-CCQ kill-risk): 支持 / H3 (vanilla で消失): 支持。
- **pre-reg §5 GO 条件を実質すべて満たす**。唯一の訂正 = 「反復 read」ではなく「単発の非 softmax
  cleanup read」が本体 (R-Hopfield は K でフラット・R-ISTA は不安定)。
- → **read 側 test-time は状態依存の実シグナルあり**。GPU 本走 (Qwen3-1.7B 蒸留状態・長系列 2k/4k/8k)
  の価値を確定。honest: 効果小 (+0.02)・CPU tiny・ceiling-relaxation (突破ではない)。
