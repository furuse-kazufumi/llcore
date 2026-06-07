# HD-1 接地 本登録 (preregistration) — 記憶形成機構の gradient 基質比較

> **状態: 本登録 (binding)。本 doc の commit は GPU 本走の結果取得前に行う** (2 段階登録の
> step 3 = 設計 v3 §5; feasibility 結果で空欄を埋めて確定したもの)。
> 正本設計 = [[HD1_GROUNDING_DESIGN.md]] v3 / レビュー = [[HD1_DESIGN_REVIEW_2026_06_07.md]]。
> 本 doc 確定後の変更は「逸脱」として結果 doc に明記しない限り行わない。

## 0. feasibility 要約 (本登録の入力; n∈{8,32}, 4 seeds, budget 120 step)

`results_hd1_grounding_feas.json` (72 records) + `analyze_hd1_feas.py`:

1. **ρ(step) は非単調**: n=32 で step≈25 に ρ→1.26 のピーク → step≈65 以降 ρ≈0.87 の
   contractive 域へ自然回帰。**契約死は transient excursion (step 10–50) に集中**し、
   後半 plateau での死は 0% (tail-30/50% 窓の NONE 死亡率 = 0.00%, tail-70% で 9.38%)。
   全窓では n=8: 1.04% / n=32: **34.38% (ACTIVE)**。
   → 設計 v3 §1 の「plateau を measure 窓に」は **premise が小 n で偽** (plateau は安全域)。
2. **実害 probe は全 72 records で 0 発火** (n=32 NONE の sep_rate max −0.001, p95 −0.002 =
   閾値 0 に接近。n↑ で発火の可能性はあるが feasibility では縮退)。
3. **EXO_init ≡ NONE が bit-identical** (全 arm 共通 admissible init の帰結; CE 完全一致で確認)。
4. proxy↔rho 相関 (記述): Spearman(proxy_g, rho_hat) = n=8: +0.05 / n=32: **+0.57**。
5. E4 (REVIVE vs REVIVE_ABLATE): CE Δ = −0.0094 (n=32; 修復は純正則化に非ず — 記述)。
6. Adam-sync ablation: ENDO 2.5968 vs ENDO_NOSYNC 2.6106 (n=32 CE)。死回避は両方 0 —
   **Adam 同期は死回避軸に影響なし、CE 軸に小寄与** (交絡懸念は解消、記述)。
7. OBSERVE β sweep (n=32): CE 2.7159 / 2.6362 / 2.6235 (β=0.25/0.5/0.75)、
   d_con 11.5 / 8.8 / 8.8。**β=0.25 は CE・死とも悪化** (縮小しすぎは境界滞留を延ばす)。
8. **honest prior**: feasibility の OBSERVE (pass1 相当・瞬時 proxy) は NONE より死を減らせて
   いない (d_con 8.8 vs 8.2 @ n=32)。H1 は本当に開いた問いである。
9. init 検査: n∈{64,128,256} で random init は cert_inf を**通らない** (0/6) →
   admit-init loop は本走スケールで実際に縮小する (§5 E3 の注意)。

## 1. 設計 v3 からの変更 (amendments; 各根拠つき, 結果取得前に確定)

- **A1 (measure 窓)**: plateau 窓は撤回し **全窓 = 全測定点 (step m, 2m, …, 400 の 80 点)** を
  measure 窓とする。根拠: 死現象は transient excursion であり (feas §0-1)、plateau 窓では
  全 arm 死 0 の縮退で confirmatory が成立しない。全窓は小 n (transient) と大 n (持続 —
  HD-1 §7 の ρ→1.95 prior) の両 regime を n 別チューニングなしに覆う。
  窓感度 (tail 30/50/70%) は探索的に 1 回だけ報告 (E5)。
- **A2 (EXO_init 廃止)**: 共有 admissible init の下で EXO_init ≡ NONE (bit-identical,
  feas §0-3)。本走の NONE が「初期のみ監督」の意味論を既に持つ。arm 数 5→4+1
  (OBSERVE 2-pass を 2 slot と数えて計 5 slot)。
- **A3 (実害 indicator の縮退条項; F 条項の対称拡張)**: ある n で NONE の窓実害死率
  (harm-death step 比率の seed 平均) **< 5% ならその n の実害 indicator を判定から除外**し、
  H1 は契約死 indicator のみで判定する (除外の事実は結果に明記)。feasibility で全 0 発火の
  ため縮退時の扱いを事前固定する。
- **A4 (OBSERVE proxy = 移動平均)**: 設計 §3-1 どおり **proxy = g_t の直近 4 測定点移動平均**。
  feasibility runner は瞬時値を使っており設計から逸脱していた (honest 記載)。本走は設計準拠。
- **A5 (測定予算)**: 軌跡測定の CPU コスト実測 (n=256 で 7.9 s/点 @250 samples) に基づき:
  測定 cadence m=5 (80 点; feasibility と同一)、per-point empirical_rho サンプル数 =
  **n=64: 200 / n=128: 96 / n=256: 48** (n 内比較のみ行うため n 間の分解能差は比較を汚さない)。
  最終測定点のみ高精度 (HD-1 同等: n=64: 600 / 128: 250 / 256: 250) で E3 用に別途記録。
  from-below 推定で少サンプルは死検出が保守側 (境界すれすれを見逃す) — 全 arm 同一条件。
- **A6 (proxy-sound 検査の操作化)**: H1 反証条項 (a) の判定統計量 =
  **NONE arm の全測定点 (n 内 pool) での Spearman(g_t, infnorm_sup_t) ≥ 0.8** で
  「proxy が sound 主成分化」と判定。0.8 未満は (b) 独立 proxy とみなす。
  根拠: 死回避が可能な proxy は不安定域近傍で ρ と相関すること自体は不可避
  (feas で rho_hat と +0.57)。問うべきは **sound certificate (infnorm_sup) の代理になって
  いるか**なので、相関先を infnorm_sup に固定し、強相関 (≥0.8) のみ無効化条件とする。
  本走 runner は infnorm_sup を毎測定点記録する。
- **A7 (k 感度の探索 arm)**: 設計 §4 H2 自白の cadence k 感度を **ENDO_K8 (k=8),
  n=128 のみ, 16 seeds** の探索 arm として 1 回だけ走らせる (E9; confirmatory 外)。

## 2. 本走の固定パラメータ (binding)

| 項目 | 値 | 備考 |
|---|---|---|
| 基質 cfg | layers=1, d=96, T=64, B=24, lr=3e-3, **grad_steps=400**, max_chars=80000, eval_batches=6 | HD-1 full 同等 (E3 比較) |
| n | {64, 128, 256} | |
| seeds | 2026..2041 (16) | 全 arm 共通 (paired) |
| arms | NONE / ENDO / REVIVE / OBSERVE_P1 / OBSERVE_P2 (+ ENDO_K8 @n=128 探索) | A2 |
| 共有 init | 全 arm: cert_inf admit まで raw_W←0.5·raw_W (≤50 回) | 公平な開始; HD-1 と異なる (§5 E3) |
| ENDO | cert_inf を k=4 cadence 検査 → fail で core+Adam 同期 rollback | |
| REVIVE | gate なし。独立判定 (m=5) の契約死検出時: raw_W←c·raw_W (c admit まで二分探索 24 iter) + 当該 layer Adam リセット | 死は踏む |
| OBSERVE | proxy = g_t の直近 4 測定点 MA。閾値 = 死イベント時 proxy log の 10 pctl。超過で直近 m step の更新を β=0.5 縮小 | cert_inf 不使用 |
| OBSERVE 2-pass | P1 = 自己履歴のみ。P2 = P1 の死イベント proxy を n 内 16 seeds pool して初期 log とし、以後自己の死も追記 (閾値は都度再計算)。**判定は P2** | P1 は E8 用 |
| 測定 | cadence m=5 (80 点): empirical_rho (A5 サンプル数) + state-separation probe (SEP_T=60, 箱端 ±1 初期差) + proxy g_t + **infnorm_sup** | 全 arm 同一 |
| 契約死 | empirical_rho ≥ 1 (測定点単位) | |
| 実害死 | sep_rate ≥ 0 (測定点単位) | |
| 窓死亡率 | (死フラグの立った測定点数) / 80 | run 単位の指標 |

## 3. confirmatory 仮説 (binding; family = {H1, H2}, Holm)

- **H1 (主) — sound vs empirical 死回避**: 各 active n・各 active indicator で
  **OBSERVE_P2 の窓死亡率 < NONE** (paired Wilcoxon signed-rank, 両側, seed 対応 16 対)。
  - **成立条件**: 全 active n で方向一致 (median 差 < 0) **かつ** 各 active n の
    Holm 調整後 p < 0.05。indicator は契約死 + 実害死の co-primary 連言
    (実害は A3 の縮退条項に従う)。
  - **ENDO 反証条項**: OBSERVE_P2 vs ENDO の窓死亡率が全 active n で Wilcoxon p > 0.05
    かつ median 差 = 0 のとき (= empirical が sound に追いついた): A6 検査を実施し、
    (a) Spearman ≥ 0.8 → 「proxy 設計の問題」として H1 判定を無効化 /
    (b) < 0.8 → **「sound ≫ empirical は EA 固有」へ正式格下げ** (設計 §6 前例に従う)。
- **H2 — repair の学習保存**: 各 active n で **REVIVE の最終 CE < ENDO の最終 CE**
  (paired Wilcoxon, 両側)。成立条件は H1 と同形 (全 active n 方向一致 + 各 n Holm p<0.05)。
- **多重性**: 仮説単位の代表 p = その仮説の (active n × active indicator) 検定群の **max p**
  (連言主張のため binding は最悪値)。Holm を {p_H1, p_H2} に適用
  (min 側 < 0.025, 残り < 0.05)。
- **F 条項 (binding)**: ある n で NONE の窓契約死率 (seed 平均) **< 5%** → その n を
  H1/H2 とも判定から除外。**全 n 除外なら本実験は INVALID** (confirmatory 主張なし、
  記述結果のみ報告し、死現象が立つ regime の再設計に戻る)。
- 検出力の自白: 16 seeds で CE の小効果は拾えない可能性 — その場合「検出せず」を報告 (盛らない)。

## 4. 探索的解析 (補正外, 記述; 各 1 回)

- **E1**: NONE arm 内の窓契約死率 × 最終 CE の関係 (n 別, seed 横断 Spearman)。
- **E2**: 契約死フラグ × sep_rate の step 単位 Spearman (n 内 pool)。|ρ|≥0.5 で契約死を
  実害 proxy として採用、未満なら「死回避軸は実害 probe で再定義」分岐 (設計どおり)。
  実害が全 0 縮退の場合は「縮退 (判定不能)」と報告。
- **E3 (接地サニティ; 非 binding)**: (i) NONE が admit set を離れ ρ≥1 域に到達するか
  (方向のみ) (ii) ENDO−NONE の CE コストが HD-1 §7 帯 0.03–0.12 と整合するか。
  **注意 (honest)**: 本走は admissible init を全 arm に課す (HD-1 の gate="none" は生 init で
  n≥64 では init 時点で非 admit — feas §0-9)。よって ρ→1.95 帯の定量再現は期待せず、
  方向一致のみ確認する。最終点の高精度 rho (A5) で記述。
- **E5**: 窓感度 (tail 30/50/70%) — H1 の主要対比を各窓で 1 回だけ再計算 (記述)。
- **E8**: OBSERVE_P1 vs P2 (2-pass 死履歴共有の寄与; 記述)。
- **E9**: ENDO_K8 vs ENDO (k 感度 @ n=128; H2 の従属所見確認; 記述)。
- (feasibility 済として報告のみ: E4 = REVIVE vs ABLATE / E6 = Adam-sync / E7 = β sweep)

## 5. 結果の解釈マップ (取得前に固定)

| 結果 | 解釈 (これ以上は主張しない) |
|---|---|
| H1 成立 | empirical 観察 (proxy 閾値回避) は gradient 基質で部分的な死回避を提供するが sound に届かない — toy の sound≫empirical の gradient 版を支持 |
| H1 不成立 (OBSERVE≈NONE 以下) | **この proxy 設計の** empirical 回避は gradient 基質で死回避を提供しない。設計 §6 のとおり「OBSERVE の上限」ではなく 1 実装の実力 — empirical 一般の限界とは区別不能と明記 |
| H1 で OBSERVE≈ENDO | 反証条項 (a)/(b) に従う (A6) |
| H2 成立 | 死後修復は時間的退行 (rollback) より学習を保存する — REVIVE 価値の gradient 接地 |
| H2 不成立 | repair の CE 優位は toy 固有 or 16 seeds で検出不能 — どちらか区別せず報告 |
| F 全除外 (INVALID) | admissible init + d=96 規模では死 regime が立たない — 死現象の regime 設計に戻る (excursion を引き起こす条件の同定が先) |

## 6. 実行計画

1. kernel = `hd1_grounding_kernel.py` (self-contained, RUN_N toggle)。Kaggle T4 で n ごとに
   別 kernel 3 本 (furusekazufumi/hd1-grounding-n{64,128,256})、resumable JSON checkpoint。
   概算: n=64 ≈ 1.2h / n=128 ≈ 3.7h / n=256 ≈ 6h (いずれも 9h 制限内; 量子余裕で順次 or 2 並列)。
2. pull 後、分析スクリプト (検定は本 doc §3 を実装) で confirmatory 判定 → VERDICT 追補。
3. 逸脱が必要になった場合は結果 doc に「逸脱」節を設け、本 doc は書き換えない。
