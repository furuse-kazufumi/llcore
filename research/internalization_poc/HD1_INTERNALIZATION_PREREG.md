# 検証シグナル勾配内蔵 本登録 (preregistration) — gradient-embedded supervision vs 事後 rollback

> **状態: 本登録ドラフト (binding 候補)**。本 doc の commit は GPU 本走の結果取得前に行う
> (= 2 段階登録 step 3; feasibility 結果で空欄を埋めて確定したもの)。
> **GPU 予算 (Kaggle T4 時間 × λ-sweep × arms) はユーザー承認後に最終化**してから結果取得前 commit。
> 正本設計 = [[HD1_INTERNALIZATION_DESIGN.md]] v2 / feasibility = [[HD1_INTERNALIZATION_FEASIBILITY.md]]
> (敵対レビュー 41 agents confirmed 16 反映済)。本 doc 確定後の変更は「逸脱」として結果 doc に明記。
>
> **用語規律 (HD-1 由来, 維持)**: 「内的化」は使わない。測るのは「検証シグナルが loss に入る
> (gradient-embedded) か rollback で外から効くか」の作用経路差のみ。真の内的化は §未到達上位目標。

## 0. feasibility 要約 (本登録の入力; CPU, n∈{8,32} model / {8..256} analytic, $0)

`results_internalization_feasibility.json` + `results_gated_check.json` + `results_drift_coverage.json` +
`results_followup.json`:

1. **(e) pull-back latency [go/no-go の核]**: violating(infnorm≈1.95)→admissible 復帰 step (4 seeds 中央値) —
   `.max()` = 165/104/318/**1736**/**2000(cap, 0/4 復帰)** (n=8/32/64/128/256) = 予算 200 を n≥64 で超過 = **NO-GO**。
   logsumexp = 80/21/21/29/33 / top-k = 86/24/22/34/28 = flat ≪200 = **GO**。
2. **(f) coverage**: `.max()` 行被覆 = 厳密 1/n (0.125→0.004), logsumexp = 1.0 (n=256 で 0.984), top-k = 0.5。
   scaled-random + drift-init 両方同一傾向。**飽和 fraction = 0** (feasibility 規模では tanh 飽和未到達 →
   `.max()` 劣化は行数不足由来で飽和非依存)。
3. **C14 (admit 中核 grad-zero)**: 素 logsumexp は max+offset(~0.1-0.28) を押すため admit 中核でも活動
   (admit_core_dot −0.09〜−0.24)。**detached true-infnorm gate** で復元: 配備 θ=0.85 (margin=0.15) で
   admit-core-active = 0/16 (全 margin で 0)、pull-back latency は gate あり/なし per-seed 一致 [20,21,30,21]。
4. **λ = 0.1** (decade match): |∂CE| ≈ 1-2.6e-3, |∂surrogate| ≈ 2.2-8e-2、比中央値 0.042 → 10^round(log10)=10⁻¹。
5. **margin の退化と gate 解決 (実証)**: plain logsumexp は relu 勾配が threshold 非依存 (常時アクティブ) で
   near-boundary residence が全 margin 0.222 一定 = 退化。gated は true infnorm を gate するので margin が
   equilibrium を制御 (residence(≥0.98) = 0.694/0.250/0.222/0.222 @ margin 0/0.05/0.10/0.15)。
6. **死削減は変種非依存 (帰属 ablation, n=32, margin=0.05)**: NONE 0.194 → max/logsumexp/gated 全て 0.069。
   → 死削減は「surrogate 補助損失が在る事」由来で coverage/gate 由来でない。**変種選定 (logsumexp/gated ≫ max)
   は死削減でなく latency/coverage の大 n スケール論拠のみに立脚** (n=32 では死では分離不能)。
7. **slack (HARNESS admit 境界 infnorm≈1.0 の真 ρ)**: 0.846/0.874/0.895/0.891/0.909 (n=8/32/64/128/256)、
   slack 0.15→0.09 → MATCHED arm の θ=1.0 (true-infnorm) は HARNESS 境界真 ρ に整合。
8. **F 条項 (NONE 死 regime)**: n=8 inactive (0.000) / n=32 active (0.188-0.194)。main n は §4 stage-1 で確認。
9. **honest prior**: feasibility の死削減 (0.194→0.069) は変種非依存・小 n。H1 (gradient 内蔵が死を減らすか) と
   H2 (勾配経由が rollback より効率か) は本当に開いた問い (feasibility は検出力不足で confirmatory 不可)。

## 1. 設計 v2 からの確定 (amendments; 各根拠つき, 結果取得前に固定)

- **A1 (surrogate 形)**: 本走 surrogate = **gated-logsumexp** = `1[infnorm_sup.detach() ≥ θ] · relu(
  logsumexp(rows(decay,W), τ=10) − θ)`。根拠: feasibility (e)(f) で `.max()` NO-GO (単一行 drift 律速)、
  logsumexp は GO だが offset で C14 違反 → gate で復元 (§0-3)。`rows(decay,W)` = (2,n) 行寄与テンソル
  (`.max()` == numpy infnorm_sup; numpy 一致 3.55e-15)。logsumexp ≥ max なので sound (gated loss=0 ⟹ admit)。
- **A2 (λ 固定)**: 主 arms の λ_cert = **0.1** (全 n 固定; decade match §0-4)。H2 の Pareto 用 λ-sweep のみ別 (§3-H2)。
- **A3 (margin / threshold)**: ENDO_GRAD の margin = **0.05** (θ=0.95; feasibility の residence knee §0-5 +
  admit-core silence 維持 + 死削減維持で選定)。ENDO_GRAD_MATCHED の θ = **1.0** (margin 0; slack §0-7 で
  HARNESS 境界真 ρ に整合)。**最終 margin は stage-1 で長 drift residence を再確認** (transient floor 0.222 が
  薄まるか; §4)。
- **A4 (死判定の独立性; A6 circularity ガード移植)**: 契約死の confirmatory 判定は **empirical_rho ≥ 1 単独**
  (from-below 固有値サンプリング)。surrogate (infnorm_sup) は **補助損失の内部量**で判定・主張に一切用いない
  (surrogate が infnorm_sup を押す→infnorm_sup 下がるは恒等式)。死イベント時の infnorm_sup(surrogate) と
  empirical_rho(death) の相関を記述報告し、高相関 (Spearman ≥ 0.8) なら「死回避は surrogate の sound 近似性の
  産物」留保を発動 (HD-1 A6 と同形)。
- **A5 (測定予算)**: HD-1 grounding と同一 — cadence m=5 (80 点)、per-point empirical_rho サンプル =
  n=64:200 / 128:96 / 256:48、最終点高精度 = 64:600 / 128:250 / 256:250。infnorm_sup を毎測定点記録。
- **A6 (F 条項 + 実害縮退; HD-1 から移植)**: ある n で NONE 窓契約死率 (seed 平均) < 5% → その n を H1/H2 とも
  除外。**全 n 除外なら INVALID** (死 regime 再設計へ)。実害死率 (sep_rate ≥ 0) < 5% → その n の実害 indicator 除外。
- **A7 (トートロジー非該当の実証)**: ENDO_GRAD 専用に「窓内で contract_death を 1 度も踏まない seed 比率」を記録
  (gate 的先回りで死回避が自明化していないことの裏取り; feasibility n=32 で 0/4 = 非該当を確認済、main n で再確認)。

## 2. 本走の固定パラメータ (binding)

| 項目 | 値 | 備考 |
|---|---|---|
| 基質 cfg | layers=1, d=96, T=64, B=24, lr=3e-3, **grad_steps=400**, max_chars=80000, eval_batches=6 | HD-1 grounding 同等 |
| n | {64, 128, 256} | |
| seeds | 2026..2041 (16; H2 は 32 検討 §3) | 全 arm 共通 (paired) |
| 共有 init | 全 arm: cert_inf admit まで raw_W←0.5·raw_W (≤50 回) | 公平な開始 |
| surrogate | **gated-logsumexp (τ=10)**, λ=0.1 | A1/A2 |
| **NONE** | 補助損失なし (drift baseline) | F 条項 |
| **ENDO_HARNESS** | cert_inf を k=4 cadence 検査 → fail で core+Adam 同期 rollback (HD-1 ENDO 流用) | 事後 rollback 経路 |
| **ENDO_GRAD** | 毎 step L = CE + λ·gated-logsumexp (θ=0.95=1−margin)。rollback なし | 勾配経路 (主) |
| **ENDO_GRAD_MATCHED** | ENDO_GRAD で θ=1.0 (margin 0); HARNESS 境界真 ρ 整合で margin 交絡分離 | C2 対照 |
| **ENDO_BOTH** | ENDO_GRAD + ENDO_HARNESS rollback。**主張限定**: 「rollback が GRAD と共存して壊れない」動作確認のみ | 寄与分離せず |
| 測定 | cadence m=5 (80 点): empirical_rho (A5) + infnorm_sup + state-sep probe (SEP_T=60) | 全 arm 同一 |
| 契約死 | empirical_rho ≥ 1 (測定点単位) | A4 |
| 窓死亡率 | (契約死フラグの立った測定点数) / 80 | run 単位 |
| corpus | tiny-shakespeare 先頭 80000 chars。download 失敗で fail-fast + vocab ≥ 40 assert | silent 汚染防止 |

## 3. confirmatory 仮説 (binding; family = {H1, H2}, Holm)

- **H1 (主) — 勾配内蔵は死を減らすか (量の階層で非自明)**: 各 active n で
  **ENDO_GRAD の窓契約死率 < NONE** (paired Wilcoxon signed-rank, 両側, seed 対応 16 対)。
  - **非自明性の核**: 補助損失は infnorm_sup (**上界**) を θ へ押すが、死は empirical_rho (**下界**) で測る。
    量の階層 `infnorm_sup ≥ true ρ ≥ empirical_rho`。上界を押す勾配が drift に勝って下界 death を減らせるかは
    自明でない (上界→下界伝播効率は無保証)。
  - **成立条件**: 全 active n で方向一致 (median 差 < 0) かつ各 active n の Holm 調整後 p < 0.05。
  - **アーティファクト規律 (binding 解釈; α 不変)**: soft surrogate が「死亡測定点を消す」計数効果でないことを、
    連続量 2 指標 (窓内 max empirical_rho / ρ 超過積分 Σ max(rho−1,0)) でも ENDO_GRAD < NONE 方向一致で確認。
    不一致なら主張を「窓計数上の低下」へ弱める。
  - **A4 留保**: 死イベント時 infnorm_sup↔empirical_rho 相関 ≥ 0.8 なら「surrogate sound 近似性の産物」留保。
- **H2 — 勾配内蔵は事後 rollback より効率的か (点比較禁止 → Pareto 曲線 vs 点)**:
  単一 λ の点比較は λ チューニング産物 (片側だけ自由 hyperparameter) のため禁止。
  - **手続き**: ENDO_GRAD を **λ-sweep {0.03, 0.1, 0.3, 1.0}** (4 点) で走らせ各 λ の (窓契約死回避率,
    最終 CE) を得て **Pareto 曲線**を引く。ENDO_HARNESS の (死回避率, CE) 点を求める。
  - **成立条件**: 各 active n で、HARNESS の死回避率に対応する ENDO_GRAD 曲線の **内挿 CE < HARNESS の CE**
    (= 同じ死回避率で勾配経路の方が CE が低い)。seed 上の bootstrap (2000 resample) で
    P(内挿 CE_GRAD < CE_HARNESS) を求め、全 active n で > 0.975 (Holm 調整後 0.95) を成立とする。
    HARNESS の死回避率が GRAD 曲線の範囲外なら「比較不能 (検出せず)」を報告。
  - **検出力 (binding 自白)**: HD-1 H2 は 16 seeds で n=128 を落とした前例。**H2 は 32 seeds で走らせる**
    (MC 見積を stage-1 で seed 分散から確定。32 seeds でも小差は拾えない場合「検出せず」を報告、盛らない)。
- **多重性**: 仮説単位代表 p = (active n × 検定) の max p。Holm を {p_H1, p_H2} に適用。
- **F 条項 (binding)**: ある n で NONE 窓契約死率 < 5% → その n を H1/H2 とも除外。全 n 除外なら INVALID。

## 4. GPU stage-1 first-check (binding; stage-2 confirmatory の前提条件)

feasibility は n≤32 のみ測定。本走 GPU は **stage-1 で main n の前提を確認してから stage-2 本測定**へ進む。
stage-1 が失敗したら stage-2 に進まず結果 doc に記録 (逸脱でなく設計):

| # | first-check | 失敗時 |
|---|---|---|
| S1 | **NONE 窓契約死率 ≥ 5%** @ 各 main n (死 regime active) | その n を H1/H2 除外 (A6); 全 n 失敗で INVALID |
| S2 | **ENDO_GRAD never_touched_death 比率 > 0** @ active n (gate 的先回りでない) | tautology 懸念 → H1 解釈を「gate 的回避」へ明記し弱める |
| S3 | **gated-logsumexp admit 中核 silence (admit-core-active = 0)** @ 配備 θ, main n | gate 失効 → θ/τ 再較正 (stage-1 内, 結果非参照) |
| S4 | **大 n drift 状態の飽和 fraction を記録** (記述) | 飽和あり → `.max()` NO-GO 強化 (変種結論不変) |
| S5 | **margin residence を長 drift で再確認** → ENDO_GRAD margin を最終確定 (transient floor 0.222 が薄まるか) | floor 残存なら margin=0.05 維持 (knee) |

## 5. 結果の解釈マップ (取得前に固定)

| 結果 | 解釈 (これ以上は主張しない) |
|---|---|
| H1 成立 (アーティファクト規律通過) | 検証シグナルの**勾配内蔵**は gradient 基質で死を減らす — 上界 surrogate を押す勾配が下界 death に伝播。連続量不一致なら「窓計数上の低下」へ弱める |
| H1 不成立 | この surrogate 設計の勾配内蔵は死を減らさない (1 実装の実力; gradient 内蔵一般の限界とは区別不能)。検出力併記 |
| H2 成立 | 同じ死回避率で勾配経路は事後 rollback より低 CE — 監督を勾配へ移す効率優位の接地 |
| H2 不成立 | 勾配経路の CE 優位は検出されず (32 seeds で検出不能 or 実在せず、区別せず報告) |
| S1 全除外 (INVALID) | 死 regime が main n で立たない — 死現象 regime 設計に戻る |
| 死削減が変種非依存 (feasibility 既知) | 「変種選定は死削減でなく latency/coverage 由来」を維持 (over-claim しない) |

## 6. 実行計画 (GPU 予算はユーザー承認後に最終化)

1. kernel = `internalization_kernel.py` (未作成; hd1_grounding_kernel.py を踏襲し arm/surrogate を gated-logsumexp に
   置換, RUN_N + RUN_STAGE toggle, resumable)。Kaggle T4。**stage-1 → first-check 判定 → stage-2**。
2. arm × n × seeds の概算: NONE/HARNESS/GRAD/MATCHED/BOTH (5) + H2 用 λ-sweep ENDO_GRAD 4 点 = 計 ~8-9 arm-config
   × {64,128,256} × 16 (H2 は 32) seeds。**GPU 時間見積を stage-1 で実測 → ユーザーに budget 提示してから本走**。
3. pull 後、分析スクリプト (検定は §3 実装) で confirmatory 判定 → VERDICT。逸脱は結果 doc に明記。
4. VERDICT → 論文 §9.8「勾配内蔵監督 vs 事後 rollback」。

## 7. 実装 (本登録 commit で結果取得前に固定; 作成・検証済)

- `internalization_kernel.py` (GPU kernel; gated-logsumexp arm + stage-1/stage-2 + H2 λ-sweep, resumable)。
  **CPU smoke 検証済**: stage-1 (8 jobs) / stage-2 (208 jobs = 5 arm×16 + 4 λ×32) 完走、stage-1 verdict 自動算出。
- `analyze_internalization.py` (H1 Wilcoxon + アーティファクト連続量 + A4 相関 / H2 Pareto bootstrap / F 条項 / Holm)。
  **合成データ検証済**: H1 dir_ok/p、H2 Pareto 内挿 P、Holm 正常動作。degenerate (死 0) で F 条項 INVALID graceful。
- 両 doc とも本登録と同 commit で**結果取得前に固定** (HD-1 grounding と同規律)。GPU 実行は無料 T4、stage-1 で時間
  実測 → 無料週次 quota 内なら $0 で stage-2、超過/有料が要る場合はコスト見積をユーザー承認後 (「安ければ可」)。
