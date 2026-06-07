# 検証シグナルを勾配に内蔵する — 設計 v2 (敵対レビュー反映)

> 状態: **設計 v2**。骨子 v1 への敵対レビュー (Workflow 18 agents, 4 lenses → **confirmed 14 /
> refuted 0**; 記録 = [[HD1_INTERNALIZATION_REVIEW_2026_06_08.md]]) を全反映。骨子 v1 が粗かった
> ことを規律が捕捉した = 恥でなく fail-closed が機能した証拠。次 = feasibility → 事前登録 → GPU。
>
> **用語是正 (C8/C9, 最重要)**: 「内的化」を本文・見出し・仮説名・論文節名から除去し
> **「検証シグナルの勾配内蔵 (gradient-embedded supervision)」** に統一。HD-1 設計 §2 が
> 敵対レビューの帰結として禁じた「内的/外的」二項対立を、対象を変えただけで再導入していた。
> 「真の内的化 (R-endo 本来の問い)」は **§8 の未到達上位目標**として残し、本実験が測るのは
> その 1 段手前 = 「検証シグナルが loss に入るか rollback で外から効くか」の差のみ。

## 0. 位置づけ — 何を測り、何を測らないか

HD-1 接地 (H1 PASS: 監督階層 sound ≫ shared-empirical ≫ self-empirical ≈ none) で測ったのは
**ハーネスが事後 rollback する** 形 (ENDO_HARNESS)。本実験の問い: cert_inf = `infnorm_sup < 1` の
sound 上界 `infnorm_sup` を微分可能 surrogate にし、補助損失でモデルの勾配に流す
(ENDO_GRAD) と、HARNESS と比べて何が変わるか。

**測る**: 検証シグナルの作用経路 (勾配 vs 事後 rollback) の差が、死回避と CE に与える影響。
**測らない (C9)**: 「目的の内発性」「自律的検証」。検証すべき性質 (ρ<1)・surrogate 形・閾値は
すべてハーネスが指定する。真に違うのは「そのハーネス由来シグナルが loss に入るか」の 1 点。

## 1. 微分可能 surrogate (実コード検証済み + 構造リスク明記)

`cert_surrogate.py` に実装。`infnorm_sup_torch(decay, W)` は numpy 正本と一致 (15 ケース max
abs err 3.55e-15)、`W=2·tanh(raw_W)` 越しに勾配が流れる。補助損失:
`L_total = CE + λ_cert · relu(infnorm_sup_torch(decay, W) − threshold)`。

- **margin/grad-zero の整合是正 (C14)**: threshold = 1 − margin (admit 内部の knee)。
  「admit 内で grad ゼロ」は **margin=0 専用**の性質 — margin>0 では admit 帯 (1−margin, 1) で
  loss>0。正しくは「**admit 中核 (infnorm_sup < 1−margin) で grad ゼロ**」。self-test に margin>0
  ケースを追加し、本文の実構成と検証根拠を一致させる。
- **単一行 subgradient リスク (C13, feasibility の核心)**: `torch.stack(rows).max()` は単一
  argmax 行のみに subgradient を流す = 毎 step W の 1/n 行しか押せない。さらに膨張 regime
  (surrogate が必要な領域) で `W=2·tanh(raw_W)` が飽和し勾配が消える。**drift は CE 勾配が全 n 行
  に働くのに押し返しは 1 行/step** = n とともに不利化 (HD-1 E10 の empirical 回避 76%→3% 崩壊と
  同型の大 n 劣化構造)。feasibility で勾配被覆率を測り、soft-max 上界 (logsumexp) / top-k 行平均の
  surrogate 変種を比較して本走形を結果取得前に確定する (§6)。

## 2. arms (5; ラベルは「監督がモデルに作用する経路」— C8)

| arm | 作用経路 | 実装 |
|---|---|---|
| **NONE** | なし | baseline (drift; ρ→1.95 帯へ) |
| **ENDO_HARNESS** | 事後 rollback (ハーネス) | HD-1 の ENDO 完全流用 (cadence k=4 検査 → fail で core+Adam rollback) |
| **ENDO_GRAD** | 勾配経由 (補助損失) | 毎 step `L_total` に surrogate 補助損失。rollback・gate なし |
| **ENDO_GRAD_MATCHED** | 勾配経由 (真ρ整合) | threshold を「HARNESS の admit 境界の **平均真 ρ** に一致」させた GRAD (margin 交絡分離; C2) |
| **ENDO_BOTH** | 両方 (安全網) | GRAD 補助損失 + HARNESS rollback。**主張限定 (C6)**: 「rollback が GRAD と共存して壊れない」動作確認のみ。効率/機構主張はしない (寄与分離不能 — ABLATE 対照が無いため) |

- 全 arm 共有 admissible init (公平な開始)。
- 「内的/外的」ラベルは表・本文から除去 (C8)。脚注で §0 の留保に直結。

## 3. confirmatory 仮説 (family Holm; 文言は事前登録で最終化)

### 死回避の判定量を先に固定 (C3, blocker)

**契約死の confirmatory 判定は `empirical_rho ≥ 1` (from-below 固有値サンプリング) 単独**。
surrogate (`infnorm_sup`) は **補助損失の内部量** であり、判定・主張に一切用いない。§2 測定リスト
から判定用途の infnorm_sup を外す。理由: surrogate が infnorm_sup を押す→infnorm_sup が下がるは
恒等式 (HD-1 A6 が OBSERVE に課した proxy-sound 独立性条項を ENDO_GRAD はそっくり継ぐ)。

### H1 — 勾配内蔵は死を減らすか (量の階層で非自明性を明文化; C5)

**ENDO_GRAD の窓契約死率 (empirical_rho≥1) < NONE** (paired Wilcoxon, active n)。
**非自明性の核 (C5)**: 補助損失は infnorm_sup (**上界**) を 1−margin へ押すが、死は empirical_rho
(**下界**) で測る。量の階層は `infnorm_sup ≥ true ρ ≥ empirical_rho`。上界を押す勾配が drift
速度に勝って **下界側の death を減らせるか** は自明でない (上界縮小が下界に伝播する効率は無保証)。
- **A6 移植 (C3)**: 死イベント時の surrogate(infnorm_sup) と death(empirical_rho) の相関を記述
  報告。高相関なら「ENDO_GRAD≈死回避は surrogate の sound 近似性の産物」と留保発動。
- **exploratory**: push 前後で上界−下界 gap がどう動くか報告 (H1 PASS が gap 縮小か単なる上界従属か
  切り分け)。

### H2 — 勾配内蔵は事後 rollback より効率的か (点比較禁止 → 曲線 vs 曲線; C4, blocker)

骨子 v1 の「ENDO_GRAD の CE < ENDO_HARNESS の CE かつ死回避同等」は **λ チューニングの産物**に
なる (片側だけ自由 hyperparameter λ を持つ非対称比較; HARNESS の動作点に λ を合わせれば CE は
当然下がる)。**confirmatory は曲線 vs 曲線にする**:
- ENDO_GRAD を **λ_cert sweep (3–5 点)** で走らせ「窓死回避率 vs 最終 CE」の Pareto 曲線を引く。
  ENDO_HARNESS の (死回避率, CE) 点がその曲線の **劣 (上) 側** にあるかで判定 = 同じ死回避率での
  CE を内挿比較。単一 λ の点比較は禁止。
- 検出力 (C12): HD-1 の H2 は 16 seeds で n=128 を落とした。feasibility seed 分散ベースで H2 用
  MC 検出力見積を binding 化 + **seeds 16→32 を事前判断** (または H2 を exploratory 降格)。
  「非検出 ≠ 無効」を解釈マップに事前固定。

### F 条項・実害縮退条項 (HD-1 から移植; C11, blocker)

- **F 条項**: ある n で NONE の窓契約死率 (seed 平均) < 5% → その n を H1/H2 とも除外。
  **全 n 除外なら INVALID** (「死 regime 再設計に戻る」分岐を明記)。
- **A3 実害縮退**: NONE の窓実害死率 < 5% → その n の実害 indicator を除外。
- **トートロジー非該当の実証 (C11)**: ENDO_GRAD 専用に「窓内で contract_death を 1 度も踏まない
  seed 比率」を記録。soft surrogate でも gate 的先回りで死回避が自明化していないことを測定で裏取り
  (設計 §0 の「死は依然踏みうる」を空証文にしない)。

## 4. λ_cert / margin の選択規則 (結果非依存の閉じた式; C10/C7, blocker)

feasibility 結果を見てから H1/H2 に有利な λ を選ぶ garden of forking を塞ぐ。**選定目的に CE 良さ・
死回避同等を使わない**:
- **margin**: feasibility で境界滞在率 (infnorm_sup の boundary 帯滞在率) が X% 未満になる最小値を
  機械的に選ぶ (X は feasibility 前に固定)。D1 を結果取得前に 1 値へ確定。
- **λ_cert**: 「admit 中核 grad-zero (§6) を満たしつつ、勾配ノルム比 |λ·∂surrogate/∂θ| と |∂CE/∂θ|
  の桁が揃う」1 桁に固定 (outcome 非参照)。本走 n ごとに再選定せず **feasibility 固定値を全 n に
  適用** (HD-1 と同じ保守; λ-by-n 交互作用の回避)。この制約が H2 を不利にしうることを §5 で自白。
- 選定規則は事前登録 §2 相当の binding table に数値で書く (HD-1 PREREG の OBSERVE 全固定に倣う)。

## 5. 弱点の自白 (v2 で増補)

- **margin が bound slack を二重計上 (C2)**: admit 境界 (infnorm_sup≈1) で真の ρ は既に 1 未満
  (実測 slack n=8 ≈0.10 / n=32 ≈0.05)。HARNESS は cert_inf<1 でこの slack ぶん縮める。GRAD は
  threshold=1−margin を押すので **HARNESS + margin ぶん深く** 縮小 = H2 の CE 優位方向を自分で課税。
  → ENDO_GRAD_MATCHED arm で真ρ整合し交絡分離。HD-1 §7 gate cost 帯 0.03–0.12 に対し GRAD は
  上端〜超過側に振れる蓋然性が高い (楽観しない)。
- **単一行 subgradient + tanh 飽和 (C13)**: 上記 §1。drift に負ける失敗モードを H1 不成立時の
  D3 分岐 (λ 不足 vs 原理的限界) の切り分け材料に。
- **検出力 (C12)**: 16 seeds で CE 小差は拾えない (HD-1 H2 前例)。MC 見積 + 32 seeds 検討。
- surrogate は ∞-norm 保守性を継ぐ (1 設計の留保; HD-1 OBSERVE と同じ「1 つの誠実な実装の実力」)。

## 6. feasibility ゲート (go/no-go 条項を前置; C1/C13/C2/C11/C14)

CPU feasibility (n∈{8,32}, 4 seeds)。**最初に go/no-go を判定**してから本走へ:
- **(e) pull-back latency (C1, go/no-go)**: violating 初期化 (infnorm_sup=1.95) から純 surrogate で
  admissible 復帰までの step 数を n×lr×λ で表化。実測 (レビュー agent) で n=64 は復帰に 210–533
  step 要 = 本走予算 400 を超過。**latency > k·grad_steps なら ENDO_GRAD 単独は drift 律速で構造的
  不適** → λ 増強 (drift 検知後) or surrogate 変種 (logsumexp) へ方向転換。
- **(f) 勾配被覆率 probe (C13)**: 膨張 step での raw_W 有効勾配被覆率 (非ゼロ行数/n, 飽和 fraction)
  を n∈{8,32} で記録 → n スケール劣化を確認。`.max()` vs logsumexp vs top-k の surrogate 変種を
  1 回比較し本走形を確定。
- **(g) margin-matched 真ρ実測 (C2)**: HARNESS の admit 境界の真 ρ 分布を実測 → ENDO_GRAD_MATCHED
  の threshold を決める。
- **(h) トートロジー非該当 (C11)**: ENDO_GRAD の「死を 1 度も踏まない seed 比率」を測定。
- **(c′) admit 中核 grad-zero の operationalize (C14)**: 「admit 中核 step で λ·∂surrogate と ∂CE の
  内積 ≈ 0」「margin off vs on で CE 軌跡差が閾未満」等、数値基準を結果取得前に固定。
- (a) F 条項 active 確認 (b) λ/margin 選定規則 (§4) の適用。

**段階**: feasibility → 事前登録最終化 (結果取得前 commit) → GPU 本走 (n∈{64,128,256} × 16〜32
seeds) → VERDICT → 論文 §9.8「勾配内蔵監督 vs 事後 rollback」。

## 7. 未決 (feasibility で詰める)

- **D1**: margin (§4 の境界滞在率規則で 1 値確定)。
- **D2**: λ 固定 vs drift 検知後増強 (C1 の latency 観点で再評価; 骨子 v1 の「固定」は撤回検討)。
- **D3**: ENDO_GRAD が NONE 同等だった場合の分岐 — λ 不足 / 単一行 subgradient の原理的限界 /
  上界→下界伝播の非効率 (C5) の 3 つを (e)(f) の probe で切り分け。
- **D4**: ENDO_BOTH の rollback 時 Adam 履歴交絡 (主張限定なので影響小)。

## 8. 真の内的化 = 未到達の上位目標 (本実験のスコープ外; C9)

本実験 (ENDO_GRAD) は「検証シグナルの勾配内蔵」を測る = 真の内的化の **1 段手前**。真の内的化は
「モデルが **検証すべき性質を自分で発見** し、自分の certificate を自分で設計する」自律で、surrogate
形・閾値をハーネスが与える本実験はそこに達しない。集団化 (PBT 風の選択・復活) も大規模で別。

## 9. 正本リンク

- 前段: [[HD1_GROUNDING_VERDICT.md]] / レビュー記録: [[HD1_INTERNALIZATION_REVIEW_2026_06_08.md]] /
  surrogate: `cert_surrogate.py` / 基質: `research/highdim_evolution/` / 論文 §7・§9.6・§9.7 /
  方針: [[project_llcore_one_year_policy]]。
