# 検証シグナル勾配内蔵 — CPU feasibility VERDICT (go/no-go 前置, 2026-06-08)

> 設計 = [[HD1_INTERNALIZATION_DESIGN.md]] v2 §6 (敵対レビュー 18 agents 反映後) /
> ランナー = `feasibility_internalization.py` (go/no-go 閾値を冒頭 `BINDING` に結果取得前固定) /
> 追補 = `feasibility_gated_check.py` (C14) + `feasibility_drift_coverage.py` (§6f 飽和) +
> `feasibility_followup.py` (敵対レビュー 16 confirmed の data 反映) /
> データ = `results_internalization_feasibility.json` / `results_gated_check.json` /
> `results_drift_coverage.json` / `results_followup.json` (全 CPU, $0) /
> **本 verdict は敵対レビュー (Workflow 41 agents, 4 lens → verify → synthesize, confirmed 16 /
> refuted 20) を反映済**。over-claim は追加 probe で訂正 (推測でなく実測)。逸脱なし。

## DECISION: **GO (conditional)** — 構造的 go/no-go は決着、パラメータは stage-1 で再確認

設計 v2 が前置した **2 つの構造リスクは feasibility で実測決着**:

| リスク (設計 §1/§6) | feasibility 実測 | 帰結 |
|---|---|---|
| **C13 単一行 subgradient (drift 律速)** | `.max()` の pull-back latency が n と共に超線形爆発 (n=128 で 1736 step ≫ 予算 400, n=256 は cap 2000 未復帰)、行被覆率 = 厳密に 1/n | `.max()` 単独は **NO-GO**。logsumexp/top-k は latency フラット (~20-35) + 全行被覆 = **GO** |
| **C14 admit 中核 grad-zero** | 素の logsumexp は max+offset(~0.1-0.28) を押すため true infnorm が admit 中核でも surrogate が活動 (dot −0.08〜−0.24) = 安全域で CE 課税 | **detached true-infnorm gate** で復元 (配備 θ=0.85 でも admit-core-active=0、実装・検証済) |

→ **本走 surrogate = gated-logsumexp** (`1[infnorm_sup.detach() ≥ θ] · relu(logsumexp(rows, τ=10) − θ)`)、λ=0.1。
GO は **直接測定された go/no-go (latency/coverage)** に立脚。F条項/tautology/MATCHED の **main n での再確認は
GPU stage-1 の first-check** (feasibility は n≤32 のみ測定; 下記「stage-1 で確認する仮定」)。

## (e) pull-back latency [GO/NO-GO の核] — `.max()` は構造的不適 (直接測定)

violating init (infnorm_sup≈1.95) → admissible (true infnorm_sup<1.0) 復帰 step (4 seeds 中央値)。
判定 = FRAC·main_grad_steps = 0.5·400 = **200 step 以下を全 main n (64/128/256) で**。

| variant | n=8 | n=32 | n=64 | n=128 | n=256 | 判定 |
|---|---|---|---|---|---|---|
| **max** | 165 | 104 | 318 | **1736** | **2000 (cap, 0/4 復帰)** | **NO-GO** (n≥64 で >200; 超線形) |
| **logsumexp** | 80 | 21 | 21 | 29 | 33 | **GO** (フラット, ≪200) |
| **top-k (k=0.25·2n)** | 86 | 24 | 22 | 34 | 28 | GO |

`.max()` の n=64 実測 318 step はレビュー agent の独立見積「n=64 で 210–533」と整合。n=256 は cap 2000 でも
未復帰 (inf_final 1.55–2.24)。**この latency は n∈{64,128,256} で直接測定**(外挿でない; scaled-random init)。

## (f) gradient coverage — `.max()` の劣化機構は「1/n 行被覆」(飽和でなく, 直接測定)

膨張 step の raw_W 行被覆率 (非ゼロ行/n)。**scaled-random init は n∈{8..256} 全域で直接測定**、
drift-init (CE 自然膨張, §6f 忠実) は n∈{8,32} で突合 (両者同一傾向):

| variant | n=8 | n=32 | n=64 | n=128 | n=256 |
|---|---|---|---|---|---|
| max | 0.125 | 0.031 | 0.016 | 0.008 | 0.004 | **= 厳密に 1/n (毎 step 1 行のみ)** |
| logsumexp | 1.000 | 1.000 | 1.000 | 1.000 | 0.984 | 全行 |
| top-k | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | k/n=0.5 |

- **飽和 fraction = 0.000** (scaled-random と drift-init 両方, infnorm≈1.5–1.95)。feasibility 規模では
  tanh 飽和は未到達 → `.max()` の劣化は**純粋に 1 行/step の被覆不足**で飽和非依存に確定 (drift-init で再現)。
- **limitation (honest)**: レビューの「膨張域 94.5% 勾配≈0」(飽和) は更に高い膨張・大 n の現象で CPU
  feasibility (n≤32, ≤400 drift step) では未再現。飽和は `.max()` を**更に**悪化 (NO-GO は保守側)、
  logsumexp の被覆は行数ベースで飽和に頑健。**事前登録 sanity**: GPU 本走の大 n drift 状態で飽和を計測し、
  もし飽和すれば `.max()` NO-GO は強化、無ければ被覆論拠のみで成立 — どちらでも変種結論は不変。

## C14 修正 = gated-logsumexp (配備 θ=0.85 で検証済)

素 logsumexp は max+offset を押すため admit 中核でも活動 (model 実測 admit_core_dot −0.09〜−0.24)。
修正 = ハーネス真量を見る **detached gate** `1[infnorm_sup.detach() ≥ θ]`。検証 (`feasibility_followup.py`):

| 検証項目 | 結果 |
|---|---|
| admit 中核 silence (unit, n=8/32/128, true infnorm<0.95) | loss=0,|grad|=0 全 OK |
| admit 中核 silence (実 ENDO_GRAD 訓練, **配備 θ=0.85=margin0.15**, n=32) | **admit-core-active = 0/16** (全 margin 0/0.05/0.10/0.15 でも 0) |
| pull-back latency (gated vs plain logsumexp, **per-seed**) | gated=[20,21,30,21] plain=[20,21,30,21] = **per-seed 完全一致** (gate は復帰を遅らせない) |

(訂正: 旧 verdict は θ=1.0 のみ検証だったが、配備 θ=0.85 で再検証し silence 維持を確認。
latency も中央値でなく per-seed 一致を確認 — 敵対レビュー C14/latency major 反映。)

## (g) margin-matched 真ρ — MATCHED arm の threshold = 1.0 (slack probe = 設計根拠)

HARNESS の admit 境界 (infnorm_sup≈1.0) の真 ρ 分布 (random W ensemble):

| n | 真 ρ @ infnorm≈1.0 | slack |
|---|---|---|
| 8 | 0.846 | 0.154 |
| 32 | 0.874 | 0.126 |
| 64 | 0.895 | 0.105 |
| 128 | 0.891 | 0.109 |
| 256 | 0.909 | 0.091 |

∞-norm slack は n と共に縮小 (0.15→0.09)。MATCHED arm の threshold=1.0 (true-infnorm) は HARNESS 境界の
真 ρ (0.85–0.91) に整合する**設計根拠**。**status: 設計妥当だが feasibility で MATCHED arm 自体は未訓練**
(slack probe は analytic 材料のみ) → MATCHED の交絡分離効果は GPU stage-1 で ENDO_GRAD と対比し確認
(敵対レビュー MATCHED_THETA major 反映)。

## (b) λ / margin 選定 (結果非依存規則 + margin 退化の gate 解決を実証)

- **λ = 0.1** (decade match): |∂CE/∂raw_W| ≈ 1.0–2.6e-3, |∂surrogate| ≈ 2.2–8.0e-2、比中央値 ≈ 0.042
  → λ = 10^round(log10 0.042) = 10⁻¹。全 n 固定。
- **margin 退化 → gate で解決 (実証, `results_followup.json`)**: **plain logsumexp** は relu 勾配が threshold
  非依存 (常時アクティブ) のため near-boundary 滞在率が全 margin で 0.222 一定 = 退化。**gated logsumexp** は
  true infnorm を gate するので margin が equilibrium を動かす:

  | margin | res(≥0.98) plain | res(≥0.98) **gated** | mean_inf gated | admit-core-active |
  |---|---|---|---|---|
  | 0.00 | 0.222 | 0.694 | 1.029 | 0/40 |
  | 0.05 | 0.222 | **0.250** | 0.985 | 0/37 |
  | 0.10 | 0.222 | 0.222 | 0.954 | 0/26 |
  | 0.15 | 0.222 | 0.222 | 0.934 | 0/16 |

  gated では margin が residence/equilibrium を実際に制御 (0→0.05 で 0.694→0.250 の大幅低下) = **退化解消を実証**。
  **honest**: 厳密規則「residence<0.20」は floor 0.222 (warm-up transient 由来) のため未達 → **margin=0.05 を
  「residence knee (0.694→0.250 の屈曲点) + admit-core silence 維持 (0/37) + 死削減維持」で選定**、MATCHED は
  θ=1.0 (margin 0)。最終 margin は GPU stage-1 の長 drift で再確認 (transient floor が薄まるか)。

## (a) F 条項 + (h) トートロジー (n=32 で確認、main n は stage-1 で再確認)

- **F 条項 (死 regime active?)**: n=8 NONE 0.000 (**inactive → 除外**; ∞-norm bound が loose で mean_inf
  1.03–1.14 でも empirical_rho<1)。n=32 NONE 0.188–0.194 (**ACTIVE**, mean_inf 1.79–1.88)。死 regime は
  n と共に立つ。**main n (64/128/256) は同一基質 (GatedRecurrentLM d=96, tiny-shakespeare, 400 step) の
  NONE arm を HD-1 grounding が測定済 = 0.476/0.778/0.961** ([[HD1_GROUNDING_VERDICT.md]] 行14, NOT 捏造)。
  本走 NONE は同一基質 → 死 regime 継承は妥当だが、**GPU stage-1 で NONE 死率 ≥5% を first-check**
  (満たさねば GO 再考)。
- **(h) トートロジー非該当 (C11)**: n=32 ENDO_GRAD「死を一度も踏まない seed 比率 = 0/4」=
  全 seed が一度は死に触れ → soft surrogate は gate 的先回りでない。死率 0.188→0.069 は真の soft 圧。
  **n=8 の never_touched=1.0 は「gate 的」ではなく「F-inactive (死 regime 不在)」の帰結** — NONE も死なない
  n=8 では当指標は無情報 (敵対レビュー TAUTOLOGY_N8 反映; cherry-pick でなく F 条項で除外)。
  main n の tautology は **GPU stage-1 で再確認** (n に依存しうる)。

## 死削減の帰属 (敵対レビューで訂正 — over-claim だった)

旧 verdict は死削減を「gated-logsumexp の C14 復元」に帰属したが **誤り**。ablation (n=32, margin=0.05,
`results_followup.json`):

| arm | death | mean_inf | ce |
|---|---|---|---|
| NONE | 0.194 | 1.844 | 2.366 |
| max-ENDO | 0.069 | 1.023 | 2.370 |
| logsumexp-ENDO | 0.069 | 0.925 | 2.365 |
| gated-ENDO | 0.069 | 0.985 | 2.362 |

**死削減 (0.194→0.069) は変種非依存** — max でも logsumexp でも gated でも同等。→ n=32 の死削減は
**「surrogate 補助損失が在る事」由来**で、coverage でも gate でもない。**変種選定 (logsumexp/gated ≫ max) は
死削減でなく latency/coverage の大 n スケール論拠のみに立脚** (n=32 では死では分離不能)。gate は死削減でなく
**CE-tax (C14) の解消**が役割。CE は 4 arm でほぼ同等 (2.362–2.370, feasibility は検出力不足で confirmatory 不可)。

## 確定した本走形 (事前登録 ready, stage-1 再確認項目つき)

| 項目 | 値 | status |
|---|---|---|
| surrogate | **gated-logsumexp** (τ=10) | **確定** ((e)(f) で max NO-GO / C14 配備 θ 検証済) |
| λ_cert | **0.1** (全 n 固定) | **確定** (decade match) |
| margin (ENDO_GRAD) | **0.05** (knee 選定) / MATCHED θ=1.0 | gate で退化解消を実証; floor は stage-1 で再確認 |
| 死判定 | **empirical_rho ≥ 1 単独** | 確定 (設計 §3, circularity 回避) |
| arms | NONE / ENDO_HARNESS / ENDO_GRAD / ENDO_GRAD_MATCHED(θ=1.0, 設計提案) / ENDO_BOTH | MATCHED 効果は stage-1 で確認 |
| GPU n | 64 / 128 / 256 × 16 seeds (H2 は 32 検討) | — |
| **stage-1 first-check (満たさねば GO 再考)** | (1) NONE 死率≥5% @ main n (2) tautology 非該当 @ main n (3) MATCHED の交絡分離 (4) 大 n 飽和 | feasibility は n≤32 のみ |

## 結論 (1 行)

**設計 v2 の 2 構造リスク (C13 単一行 drift 律速 / C14 admit 中核課税) は CPU feasibility で両方決着 —
本走 surrogate = gated-logsumexp (λ=0.1, margin 0.05/MATCHED θ=1.0)。go/no-go = GO (conditional)、
死 regime・tautology・MATCHED・飽和の main n 確認を GPU stage-1 first-check に置く。**

## 次手順

1. **事前登録**: 確定本走形 (上表) + 仮説 H1/H2 (Pareto 曲線 vs 曲線) + F/A6/A3 条項 + stage-1 first-check を
   結果取得前 commit。
2. **GPU 本走 (stage-1 先頭で死 regime/tautology/飽和を確認 → 満たせば stage-2 本測定)**: Kaggle T4,
   n∈{64,128,256}, NONE/HARNESS/GRAD/MATCHED/BOTH。
3. VERDICT → 論文 §9.8「勾配内蔵監督 vs 事後 rollback」。
