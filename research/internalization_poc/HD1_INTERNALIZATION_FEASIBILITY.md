# 検証シグナル勾配内蔵 — CPU feasibility VERDICT (go/no-go 前置, 2026-06-08)

> 設計 = [[HD1_INTERNALIZATION_DESIGN.md]] v2 §6 (敵対レビュー 18 agents 反映後) /
> ランナー = `feasibility_internalization.py` (go/no-go 閾値を冒頭 `BINDING` に結果取得前固定) /
> データ = `results_internalization_feasibility.json` (CPU, $0) +
> `results_gated_check.json` (C14 修正検証) + `results_drift_coverage.json` (§6f 飽和) /
> 生出力 = `feasibility_run.log`。**逸脱なし** — 全 go/no-go は BINDING 規則の機械適用。

## DECISION: **GO** (事前登録へ) — ただし本走形に 1 つの修正を反映 (実コード検証済)

設計 v2 が前置した 2 つの構造リスクは feasibility で **両方とも実測決着**:

| リスク (設計 §1/§6) | feasibility 実測 | 帰結 |
|---|---|---|
| **C13 単一行 subgradient (drift 律速)** | `.max()` の pull-back latency が n と共に超線形爆発 (n=128 で 1736 step ≫ 予算 400)、行被覆率 = 厳密に 1/n | `.max()` 単独は **NO-GO**。logsumexp/top-k は latency フラット (~20-35) + 全行被覆 = **GO** |
| **C14 admit 中核 grad-zero** | 素の logsumexp は max+offset(~0.1-0.28) を押すため true infnorm が admit 中核でも surrogate が活動 (dot −0.08〜−0.24) = 安全域で CE 課税 | **detached true-infnorm gate** で復元 (実装・検証済、下記) |

→ **本走形を確定**: **gated-logsumexp** surrogate (`1[infnorm_sup.detach() ≥ θ] · relu(logsumexp(rows, τ=10) − θ)`)、λ=0.1、MATCHED θ=1.0。

## (e) pull-back latency [GO/NO-GO の核] — `.max()` は構造的不適

violating init (infnorm_sup≈1.95) → admissible (true infnorm_sup<1.0) 復帰 step (4 seeds 中央値)。
予算判定 = FRAC·main_grad_steps = 0.5·400 = **200 step 以下を全 main n (64/128/256) で**。

| variant | n=8 | n=32 | n=64 | n=128 | n=256 | 判定 |
|---|---|---|---|---|---|---|
| **max** | 165 | 104 | 318 | **1736** | **2000 (cap, 0/4 復帰)** | **NO-GO** (n≥64 で >200; 超線形) |
| **logsumexp** | 80 | 21 | 21 | 29 | 33 | **GO** (フラット, ≪200) |
| **top-k (k=0.25·2n)** | 86 | 24 | 22 | 34 | 28 | GO |

`.max()` の n=64 実測 318 step はレビュー agent の独立見積「n=64 で 210–533」と整合 (相互確認)。
n=256 では 2000 step (cap) でも復帰せず (inf_final 1.55–2.24) = 単独 surrogate は完全に drift 律速。

## (f) gradient coverage — `.max()` の劣化機構は「1/n 行被覆」(飽和でなく)

膨張 step での raw_W 行被覆率 (非ゼロ行/n)。scaled-random init と **drift-init (CE 自然膨張, §6f 忠実)** の両方で同一傾向:

| variant | n=8 | n=32 | n=64 | n=128 | n=256 |
|---|---|---|---|---|---|
| max | 0.125 | 0.031 | 0.016 | 0.008 | 0.004 | **= 厳密に 1/n (毎 step 1 行のみ)** |
| logsumexp | 1.000 | 1.000 | 1.000 | 1.000 | 0.984 | 全行 |
| top-k | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | k/n=0.5 |

- **飽和 fraction = 0.000** (両 init, infnorm≈1.5–1.95)。feasibility 規模では tanh 飽和は未到達 →
  `.max()` の劣化は **純粋に 1 行/step の被覆不足**で、飽和非依存に確定 (drift-init で再現)。
- **limitation (honest)**: レビューの「膨張域 94.5% 勾配≈0」(飽和) は更に高い膨張・大 n の現象で
  CPU feasibility (n≤32, ≤400 drift step) では再現せず。飽和は `.max()` を**更に**悪化させるだけ
  (NO-GO は保守側)、logsumexp の被覆は行数ベースで飽和に頑健。GPU 本走前に大 n drift-init での
  飽和確認を pre-flight sanity として残す。

## C14 修正 = gated-logsumexp (実装・検証済, `feasibility_gated_check.py`)

素の logsumexp は max より高く出る (sound: logsumexp≥max ⟹ admit 内なら必ず true 安全) が、その offset の
ため admit 中核でも push を続け CE に課税 (model 実測 admit_core_dot −0.09〜−0.24, 活動 step 8〜19/24)。

修正 = ハーネスの真量 (infnorm_sup) を見る **detached gate**: `gate = 1[infnorm_sup.detach() ≥ θ]`。
admit 中核では gate=0 で厳密に loss=0 (grad=0)、外では logsumexp の全行勾配。検証:

| 検証項目 | 結果 |
|---|---|
| admit 中核 silence (n=8/32/128, true infnorm<0.95) | loss=0, |grad|=0 **全 OK (C14 復元)** |
| pull-back latency (gated vs plain logsumexp) | n=32/128/256 で **21/29/33 = 完全一致** (gate は復帰を遅らせない) |
| 実 ENDO_GRAD 訓練の admit 中核活動 step (n=32, 4 seeds) | **0 / 13〜14 = 完全 silence** (素 logsumexp は 8〜19 活動) |
| 死回避維持 | death_rate 0.073 (NONE 0.188) = 素 logsumexp と同等の削減を維持 |

## (g) margin-matched 真ρ — MATCHED arm の threshold = 1.0

HARNESS の admit 境界 (infnorm_sup≈1.0) の真 ρ 分布 (random W ensemble, 各 n で 4 seeds × 150–400 samples):

| n | 真 ρ @ infnorm≈1.0 | slack (infnorm−ρ) |
|---|---|---|
| 8 | 0.846 | 0.154 |
| 32 | 0.874 | 0.126 |
| 64 | 0.895 | 0.105 |
| 128 | 0.891 | 0.109 |
| 256 | 0.909 | 0.091 |

∞-norm bound の slack は n と共に縮小 (0.15→0.09)。設計 §5 の懸念 (margin が bound slack を二重計上) は
**ENDO_GRAD_MATCHED arm の threshold を 1.0 (true-infnorm 空間, gate 基準)** に設定すれば解消 —
HARNESS 境界の真 ρ (0.85–0.91) に一致し、margin 由来の追加縮小を排除して交絡分離。

## (b) λ / margin 選定規則 (結果非依存の閉じた式)

- **λ = 0.1** (decade match): 膨張近傍で |∂CE/∂raw_W| ≈ 1.0–2.6e-3, |∂surrogate/∂raw_W| ≈ 2.2–8.0e-2、
  比中央値 ≈ 0.042 → λ = 10^round(log10 0.042) = 10⁻¹。|λ·∂surrogate| と |∂CE| が同 decade。全 n 固定。
- **margin = 規則 inconclusive → gate 化で解消**: 素 logsumexp では near-boundary 滞在率が margin 全格子
  (0.00/0.05/0.10/0.15) で **0.208–0.25 と一定** = offset が threshold を支配し margin が λ から分離不能。
  **gated-logsumexp では gate が true infnorm を見るので margin が真量に作用** → MATCHED は margin=0 (θ=1.0)、
  ENDO_GRAD は小 margin (≤0.05) を gated 損失で再 probe して事前登録で 1 値確定 (本走前の唯一の未決)。

## (a) F 条項 + (h) トートロジー非該当

- **F 条項 (死 regime active?)**: n=8 NONE 窓契約死率 0.000 (**inactive → 除外**, mean_inf 1.03–1.14 だが
  empirical_rho<1 = ∞-norm bound が loose)。n=32 NONE 0.188 (**ACTIVE**, mean_inf 1.79–1.88)。死 regime は
  HD-1 同様 n と共に立つ → 本走 n (64/128/256) は HD-1 grounding で 0.476/0.778/0.961 既確立 (active 継承)。
- **(h) トートロジー非該当 (C11)**: n=32 ENDO_GRAD の「死を一度も踏まない seed 比率 = 0/4」=
  **全 seed が少なくとも一度は死に触れる** → soft surrogate の死回避は gate 的先回りでない。死率 0.188→0.073
  (約 61% 減) は真の soft 圧の効果。設計 §0「死は依然踏みうる」が空証文でないことを実測裏取り。
- **副次 (探索的, 補正外)**: ENDO_GRAD は infnorm を制御 (mean 0.90 vs NONE 1.85) しつつ CE 同等
  (2.34–2.40 vs NONE 2.34–2.41) = 本走の H1 (死減) / H2 (CE トレードオフ) に前向きな兆候 (feasibility は
  検出力不足で confirmatory 主張せず)。

## 確定した本走形 (事前登録 ready)

| 項目 | 確定値 | 根拠 |
|---|---|---|
| surrogate | **gated-logsumexp** (τ=10) | (e)(f) で `.max()` NO-GO / C14 修正検証済 |
| λ_cert | **0.1** (全 n 固定) | decade match (§4 規則) |
| 死判定 | **empirical_rho ≥ 1 単独** | 設計 §3 (surrogate は補助損失の内部量, circularity 回避) |
| arms | NONE / ENDO_HARNESS / ENDO_GRAD / ENDO_GRAD_MATCHED (θ=1.0) / ENDO_BOTH | 設計 §2 |
| GPU n | 64 / 128 / 256 × 16 seeds (H2 は 32 検討) | HD-1 と同 (死 regime active) |
| 唯一の未決 | ENDO_GRAD の margin (≤0.05) を gated 損失で 1 回再 probe → 事前登録で確定 | (b) margin 規則の gate 化後の再適用 |

## 結論 (1 行)

**設計 v2 が前置した 2 構造リスク (C13 単一行 drift 律速 / C14 admit 中核課税) は CPU feasibility で
両方決着 — 本走 surrogate は gated-logsumexp (λ=0.1) に確定。go/no-go = GO、残作業は margin 1 点の
gated 再 probe → 事前登録 → GPU 本走。**

## 次手順

1. **margin 再 probe (gated)**: gated-logsumexp で near-boundary 滞在率 vs margin を 1 回測り margin 確定。
2. **事前登録**: 確定本走形 (上表) + 仮説 H1/H2 (Pareto 曲線 vs 曲線) + F/A6/A3 条項を結果取得前 commit。
3. **GPU 本走**: Kaggle T4, n∈{64,128,256}, NONE/HARNESS/GRAD/MATCHED/BOTH。
4. VERDICT → 論文 §9.8「勾配内蔵監督 vs 事後 rollback」。
