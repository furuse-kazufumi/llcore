# BG9 事前登録 (Pre-Registration) — ③ 欺瞞地形 in KernelGenome 空間

> 2026-06-02。**results を見る前に protocol を固定する**(谷深さ DECEPTIVENESS_MEASURE の
> 循環論法/post-hoc 失敗を構造的に回避する唯一の方法)。隔離 `research/kernel_diversification/`、
> src/llcore 無改変、git は orchestrator 一括。本書を確定 (commit) してから BG9 driver を実装・実行する。
> 確定後に閾値・記述子・予算・task 群を結果に合わせて動かしたら honest 違反として記録する。

---

## 0. 命題 (falsifiable)

> **仮説 H7**: 4 kernel (rwkv/mamba/hopfield/linear_attn) を union した `KernelGenome` 拡張空間は、
> 3-param 単一 kernel 空間が持たなかった **欺瞞的 corridor 構造**(離散 kernel_id 障壁 + 非対称 gate 透過性)を
> multi-task fitness 上に持ち、**MAP-Elites (kernel_id × theta niching) が同予算 RR-hillclimb / panmictic-GA / random を
> ≥15 seed で有意に上回る (③ load-bearing)**。

**帰無 H0**: 拡張空間でも地形は滑らか/kernel 中立で、MAP-E は強 baseline に優位を持たない (③不要)。

**honest prior (明記)**: kernel は対角スカラ mock (kernels.py スコープ宣言)。③研究の全 CPU 基質
(ESN/ridge proxy・多タスク・記憶タスク・BG6 の memory_tasks) はこれまで滑らか/中立だった。
よって **事前確率は ③不要/N/A 寄り**。clean な③成立が出たら `feedback_benchmark_honest_disclosure` に従い
**内訳を疑い**、§5 の敵対検証 4 レンズ全通過を要求する。

---

## 1. 固定する設計パラメータ (確定後に動かさない)

### 1.1 behavior descriptor (固定)
- `behavior(gene_vec5) = (kernel_id 連続値, theta L1 norm)` の 2D。実装 = `kernel_fitness.kernel_behavior` / `kernel_behavior_bounds`。
- **kernel_id は fitness を直接定義しない選択変数**ゆえ、谷深さの「behavior=mean(g) を彫った循環」は構造的に起きにくい(§5.1 で検定)。

### 1.2 n_bins (固定)
- kernel_id 軸 = **4 bin** (= N_KERNELS、各 kernel basin に 1 bin)。
- theta L1 norm 軸 = **8 bin** (Step4 / BG6 と同桁)。
- 合成 control・negative control・real すべてで **同一 n_bins**。**robustness 検査として ±1 段 (kernel 4 固定 / theta {6,8,10}) を §5 で別途掃引**するが、主判定は (4, 8) 固定。

### 1.3 予算 n_evals (固定)
- 全 method **equal budget** `n_evals = 4000` (Step4 exp4 / BG6 full の桁。実 wall-clock を G1<900s に収める chunk 分割は許容、予算自体は固定)。
- MAP-E `init_batch` = 64 (BG6 と同)。

### 1.4 seeds (固定)
- **n_seeds = 15** (strict_compare の下限)。CRN (共通乱数) で method 間 paired。各 seed で SeedSequence 一意化
  (E-A の F3 教訓: 進化 seed エイリアス + paired 検定の matched replicate 崩れを回避)。
- best は archive-max でなく **global best-of-budget** で取る (E-A の F2 教訓: randselect の忘却バイアス除去)。

### 1.5 honest 再評価 (固定)
- 進化は train 系列、**判定は fresh held-out 再評価** (elitism 凍結持越し artifact 排除)。`HONEST_N` = 30 (E-A の F7 教訓)。

### 1.6 統計ゲート (固定)
- method M が baseline B に勝つ = **片側 Wilcoxon `alternative="greater"` p<0.05 ∧ Cliff δ |δ|≥0.2 ∧ paired mean diff>0** (strict_compare 準拠)。
- MAP-E が「3 baseline 全勝」= RR-hillclimb ∧ panmictic-GA ∧ random の **3 つ全てに上記ゲート成立**。

---

## 2. 3 基質 (substrate) — 事前定義

### 2.1 positive control (harness validity 確立、§5 の前提)
- **synthetic kernel-barrier multi-task**: 一部 task は **hopfield basin の gene でしか高 fitness** に到達できないよう構成
  (= 離散 kernel_id 障壁を意図的に inject)。rwkv basin から連続 hill-climb で到達不能 = Step4 exp4 deceptive corridor の kernel 版。
- **要件**: MAP-E が ③ load-bearing (3 baseline 全勝) を **先に示すこと**。示せなければ harness が「③が在るとき検出できる」ことを保証できず → **BG9 = N/A** (測定器不成立)。

### 2.2 negative control (smooth 対照、§5)
- **kernel 中立 task**: BG6 で全 kernel 飽和と判明した `delayed_recall` (全 kernel R²≈1.0) を用いる。
- **要件**: MAP-E が ③ 優位を **持たない** (smooth ゆえ niching 無益) ことを確認。優位が出たら harness の false-positive → N/A 警告。

### 2.3 real (公正な実テスト = BG9 の本番)
- **kernel-favoring task suite** (BG9-3 で第一原理設計 + 強 BG6 で validity 確認したもの)。各 kernel が構造的に得意な task を含む:
  mamba=input-selective copy / hopfield=associative recall / linear_attn=long-range accumulation / rwkv=gated leaky-integration。
- **validity gate (事前登録)**: real suite は **強 BG6 (per-kernel probe + GA で task→best-kernel 写像が真に非定数、≥2 kernel が別 task で best、margin 非僅差)** を通過したものに限る。
  通過しない suite で BG9 を回しても inert(BG6 で実証済の轍) → その場合 real は「kernel 中立」と honest 開示し ③不要 方向。
- **post-hoc 禁止**: real suite は **BG9 を回す前に**強 BG6 で確定させ、BG9 結果を見て task を入替えない。

---

## 3. 交絡 ablation (事前登録) — ③優位の帰属を切り分ける

谷深さの循環論法 (severity HIGH) を避けるため、③優位が出た場合に **何に帰属するか**を以下で分離する:

- **(A1) kernel_id-ablation**: behavior から kernel_id を抜いた **theta-only MAP-E** と比較。
  - ③優位が theta-only で **消える** → 優位は **kernel diversity 維持由来** (= ③ が kernel 障壁を跨ぐ stepping-stone、H7 支持)。
  - ③優位が theta-only でも **残る** → 優位は単なる探索量/theta niching 由来で **kernel-union 特有でない** (H7 弱い)。
- **(A2) gate-on / gate-off**: state_norm gate を on/off 両条件で測り、gate が欺瞞構造を inject する効果 (非対称透過性) を honest 開示。
- **(A3) init_batch ablation**: MAP-E init_batch を {32, 64} で振り、勝因が coverage でなく archive ratchet であることを確認 (Step4 C4 と同思想)。

---

## 4. 判定 (3 値) — 確定

| 判定 | 条件 (全て満たす) |
|---|---|
| **③ load-bearing (欺瞞地形あり)** | positive control PASS (§2.1) **かつ** real で MAP-E が 3 baseline 全勝 (§1.6) **かつ** (A1) で優位が kernel diversity 維持に帰属 (theta-only で優位減衰) **かつ** §5 敵対検証 4 レンズ全通過 |
| **③不要 (欺瞞地形なし)** | positive control PASS だが real で MAP-E が 3 baseline 全勝に届かない (優位消失) → Step4 §7「proxy 滑らか」が拡張空間でも再現 = honest negative |
| **N/A (測定不能)** | positive control すら ③ 不成立 (harness が③を検出できない) **または** real suite が強 BG6 不通過 (kernel 中立=inert) **または** §1.2 robustness 掃引や §5 で判定が反転 |

**「整いすぎた③成立は内訳を疑う」**: ③ load-bearing を出すには上 4 条件 (positive validity + real 全勝 + 帰属 + 敵対全通過) を **全て**満たす必要があり、1 つでも欠ければ ③不要 か N/A に倒す。

---

## 5. 敵対検証 4 レンズ (事前登録、谷深さで効いた lens を継承)

BG9 verdict draft に対し独立に以下を当て、各 `refuted=true/false + severity` で記録。positive 寄り headline を弱める方向に限定する:

1. **循環論法**: ③優位は「behavior=fitness を定義する量」のなぞりでないか。kernel_id は選択変数で fitness を直接定義しないが、
   real suite の fitness が kernel_id と高 corr を持つなら準循環 → `corr(fitness, kernel_id)` を測り開示。
2. **記述子依存**: §1.2 の n_bins ±1 段掃引 (theta {6,8,10}) で verdict が反転しないか。反転したら N/A。
3. **サンプリング頑健性**: 別 base_seed 群 (3 系) で再走し verdict が一致するか。CV が大きく再測定で反転したら N/A 寄り。
4. **予算頑健性 / honest 再評価**: n_evals {2000,4000} と fresh held-out (elitism 持越し排除) で優位が survive するか。

---

## 6. 実行順 (確定)

1. BG9-3: kernel-favoring suite を第一原理設計 + 実装 + **強 BG6 validity gate** (§2.3) を通す。通らなければその事実を honest 記録し real=「中立」で進む。
2. BG9-4: driver 実装 (positive/negative/real × 4 method × ablation A1/A2/A3、chunked-resumable、本 pre-reg の固定 param を参照)。
3. BG9-5: positive control を**先に**回し PASS (harness validity) を確認 → negative control → real。≥15 seed。
4. 3 値 verdict (§4) → 敵対検証 4 レンズ (§5) → honest 反映。
5. BG9-6: Codex pair-review (findings は実コード検証してから採用) → commit → memory/index 反映 + GPU 判断含意。

---

## 7. GPU 判断への含意 (事前に固定)

- **③ load-bearing (CPU で立つ)** → ③の存在は CPU kernel-union で証明。GPU は「実 LLM でも一致するか」の scale 確認に格下げ = 投資の緊急度低下。
- **③不要 (CPU でも滑らか)** → 全 CPU 基質が滑らかで一貫。GPU は唯一の残り路だが事前確率は低い = クラウド GPU で事前登録1本のみ、ポートフォリオ判断。
- **N/A** → 測定器/substrate の限界。GPU 前に CPU 測定器を直すべき (谷深さと同型の教訓)。

*UTF-8 / py -3.11 / src 無改変 / git は orchestrator 一括。本 pre-reg は確定後 immutable (変更は honest 記録付きで別節追記)。*
