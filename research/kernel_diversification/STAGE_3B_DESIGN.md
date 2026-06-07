# Stage 3b — Kernel 多様化 gene の設計 + scout + smoke 統合

**作成**: 2026-06-01
**位置づけ**: llcore 研究記録 (`research/kernel_diversification/` 隔離)。**src/ には一切触らない**。
既存 `src/llcore/kernel/{protocol.py, rwkv.py}` / `state_update/genes.py` /
`verifier/invariants.py` は **読むだけ**で再利用設計する。git は orchestrator が一括 commit。

**ステータス (honest)**:
- §1-§4 (gene 表現 / kernel 写像 / Stage1・Lipschitz gate 適用 / smoke 数値接地) =
  **mechanism feasibility が smoke レベルで PASS** (BG1/2/3/5, z3 available, N=400 gene/kernel)。
- §5 (③ 欺瞞地形仮説) / §6 (heavy 実験手順) / §7 (リスク) = **設計と仮説**。BG4/BG6-9 は **未実証**。
- 「できる (mechanism feasible)」と「やった (specialist/欺瞞地形を実証した)」を **厳密に区別**する。
  本書は前者を smoke で示し、後者は falsifiable な実験計画として登録する段階。

**本書の役割**: `DESIGN_kernel_diversification_3b.md` (設計本体, 29KB) /
`SCOUT_kernel_lit.md` (文献 scout) / `SCOUT_NOTES_smoke.md` (smoke 一次解釈) /
`BREAK_GATES.md` (BG1-9 登録簿) を **タスク 7 章構成で統合した上位サマリ**。各章の詳細は
当該ファイルへ参照を貼る (重複転記は避ける)。

参照ファイル一覧:
- 設計本体: `DESIGN_kernel_diversification_3b.md`
- 文献 scout (kernel 安定性 / specialist 先行研究): `SCOUT_kernel_lit.md`
- gate 登録簿 (BG1-9 falsifiable): `BREAK_GATES.md`
- smoke 一次解釈 (負の結果含む): `SCOUT_NOTES_smoke.md`
- skeleton 実装: `kernels.py` (KernelGenome + 4 kernel forward dynamics)
- gate smoke: `smoke_kernel_gates.py` + `smoke_kernel_gates_results.json` (BG1/2/3/5)
- 動的 smoke: `smoke_kernels.py` + `smoke_results.json` (finite + state_norm 有界)

---

## 0. 戦略的接続 — なぜ kernel 多様化か (③ 研究の論理的次段)

Step4 (`docs/poc/STEP4_SELECTION_VERDICT.md`) の確定結論:

- **③ (選択圧 / 分離 = MAP-Elites niching) は欺瞞的 corridor landscape でのみ load-bearing。**
- 3-param RWKV gene 空間 (decay/mix/gate_str) + 実 substrate proxy (ESN × 実テキスト, step6) は
  **滑らか / 単峰 broad-ridge** で、欺瞞 corridor が自然出現しない → そこでは③不要。
- Step4 §6 + DECEPTIVENESS_MEASURE_VERDICT の含意: **真の unlock は「探索空間を 3-param から拡張」**
  すること。richer な architecture 空間なら欺瞞地形を持ちうるかを問い直す。

本 Stage 3b は **まさにその探索空間拡張**である。3 連続 param 空間 → **離散 kernel 種別 ×
kernel 固有連続 param** の混合空間に拡張し、

- **Stage 3a (§1-§4)** = kernel 種別を遺伝子化し、各 kernel が Stage1 gate (state_norm /
  Lipschitz contraction) を pass する破綻ゲートを定義 (mechanism feasibility)。**smoke PASS 済**。
- **Stage 3b (§5-§6 + 命題3b)** = multi-task で specialist が出現するか (集団が単一 kernel に
  固定しない) + **混合空間が欺瞞地形 (③ load-bearing) を持ちうるか** を検定。**未着手**。

OTHERARCH_VERDICT.md の教訓「same design pattern, NOT same verifier stack」(Neural ODE / GNN / SNN
3 PoC で overclaim 撤回) を全前提に組む: 各 kernel の `Trajectory.kind` と verifier 系統を型で分け、
意味論差を消さない。modern Hopfield は energy descent (Lyapunov) 系統で既存 Jacobian gate と
**質的に別系統**である点 (SCOUT_kernel_lit.md key_finding #4) は §3 で honest に扱う。

---

## 1. 命題 (3a / 3b) — falsifiable

### 命題3a (mechanism feasibility) — smoke PASS

> 4 kernel (rwkv / mamba_selective / hopfield_dense / linear_attn) それぞれについて、clip 範囲内に
> **state_norm gate (`|s|≤state_bound` 不変) を Z3 unsat で pass する gene が非空に存在**し、かつ
> rwkv / mamba / linear_attn は **Lipschitz contraction gate (L<1) を pass する gene も非空**に存在する。
> hopfield は **L<1 pass 集合が空でないが clip 範囲の一部に限る** (multi-basin 由来)。

**falsifiable / 破綻条件**:
- (反証A) ある kernel で state_norm admit 集合が空 → 対角写像 / bounds 設計が破綻。
- (反証B) Z3 verdict と閉形式 `L_upper_bound` が矛盾 → encoding 健全性破綻。
- (反証C) `empirical_L > L_upper_bound` → over-approx が unsound。

**判定 (smoke, N=400 gene/kernel, z3 available, state_bound=1, |x|≤1)**: → **§4 表参照。
BG1/BG2/BG3/BG5 全 PASS = 命題3a は smoke レベルで支持**。
ただし mamba Lipschitz encoding の保守性 (free `a∈[0,1]` 端点 `a=1` で contract率 0.000) が
DESIGN「mamba 無条件 contraction」claim と乖離 → §3 / §6 で精緻化を要追記事項として残す
(honest: 整いすぎた claim を smoke が反証した。消さずに記録)。

### 命題3b (specialist 出現 / 単一 kernel に固定しない) — 未実証

> 既存 `TaskMixture` (memory_tasks の FlipFlop / DelayedRecall / DelayedParity を regime とする
> 多タスク分布) 上で `KernelGenome` 集団を進化させたとき、
> **(i) 異なる task に異なる kernel_id が best に選ばれ (task→best-kernel 写像が非定数)**、かつ
> **(ii) MAP-Elites archive の kernel_id 分布が単一値に collapse しない**
> (final archive で ≥2 kernel が live occupancy, Shannon entropy>0)。

**falsifiable / 破綻条件**:
- (反証D) 全 task で同一 kernel best → specialist 不在 (1 kernel 万能 = 多様化に価値なし)。
- (反証E) archive kernel entropy ≈ 0 → 集団が単一 kernel に固定 (多様化 gene 不機能)。
- (反証F) kernel_id 固定 ablation が進化版と **同等以上**の test 汎化 → kernel 選択が
  load-bearing でない (Step4 ③ ablation 規律と同型 = niching が effort でなく diversity に
  帰属するかの切り分けと同じ思想)。

**計測法 (未実施, §6 で手順化)**: `ea_lab.run_ea_methods_over_seeds` を `KernelGenome` (dim=5) に
拡張流用。behavior descriptor = `(kernel_id 連続値 1D, theta L1 norm 1D)` の 2D を MAP-Elites grid。
`split_regimes` で train/test 分離 hold-out 汎化を主指標、per-task best-kernel 写像表、final archive
kernel entropy、ablation は固定 4 本 vs 進化版を ≥15 seed paired Wilcoxon p<0.05 + Cliff δ
(`selection_lab.compare` 流用)。

**honest 留保**: specialist 出現には task 間で best kernel が実際に違う構造が要る。出なければ
honest negative (「memory_tasks は kernel 中立 = 多様化の土俵でない」)。捏造して positive に倒さない
(DECEPTIVENESS_MEASURE_VERDICT の N/A 受容規律)。緩和 = 各 kernel が構造的に得意な task を含める
(hopfield → 連想 / DelayedRecall, mamba → selective / FlipFlop, linear_attn → 長距離累積)。

---

## 2. Kernel gene 表現 + 各 kernel の写像表 (タスク 1+2)

### 2.1 タグ付き union genome `KernelGenome`

```
KernelGenome = (kernel_id: float,  theta: np.ndarray[MAX_DIM=4])
GA_DIM = 1 + MAX_DIM = 5
```

- **`kernel_id`** — 離散遺伝子を **連続実数 `k∈[0, N_KERNELS)` で保持**し
  `int(floor(clip(k, 0, N-1)))` で離散化 (`kernels.py::KernelGenome.kernel_index`)。
  連続値で持つことで既存連続ベクトル GA operator (`rng.normal`, `_clip`) を **無改変流用**。
- **`theta`** — 固定長 4 次元。各 kernel は先頭 `dim(kernel_id)=3` 次元のみ自分の codec で解釈し、
  残余 1 次元は **junk DNA** として無視 (decode 時 `decode_theta` が `theta[:dim]` のみ抽出)。
  junk 次元は探索暴走防止に `[0,1]` clip (`clipped()`、意味論非干渉)。

**固定長 union を選ぶ TRIZ 根拠 (#1 分割 / #3 局所性質)**: 可変長 genome は固定 dim GA operator
(`_clip`, `rng.normal(size=dim)`) を壊す。固定長 + junk なら `map_elites_*` / `panmictic_ga` を
**1 行も変えず**流用できる (dim=5)。後方互換を operator レベルで担保。

### 2.2 既存資産との整合 (実コードで確認済)

- `kernel/protocol.py::Kernel[GeneT]` は gene 型を **パラメトリック化済** (RWKV=`StateUpdateGene` 3-dim,
  SNN=`LIFGene` 4-dim) → 可変 dim 連続ベクトル符号化の契約を既に持つ (docstring「gene 型は kernel
  ごとに異なる」)。
- **後方互換 (BG5)**: `kernel_id=0` のとき `theta[:3]` を既存 `StateUpdateGene(decay,mix,gate_str)` に
  decode し、**既存 `run_sequence` をそのまま呼ぶ** (`kernels.py::run_sequence_kernel`)。3-param 既存実験
  (step4/step6/ea_multitask) は kernel_id=0 部分空間に **完全埋め込み** → Step4「3-param は滑らか」
  結論を退化させず包含する。**smoke で bit 一致 (all_bit_match=True, N=50) を機械担保**。
- **ChangeOp 整合**: 新 op_type `kernel_id_shift` は **research 側**に追加 (`apply_kernel_changeop`)。
  src の `OP_TYPES` (decay/mix/gate/kernel_swap_mock) は不変。これは protocol.py の S3 延期事項
  「op_type 検証を kernel 別 (`change_op_types` 参照) に拡張」の research 先行 PoC に相当
  (S3 を src で実装する誘惑は §7 リスク7 で抑制)。

### 2.3 各 kernel の state-update 写像表

全 kernel を **対角 (各座標独立) スカラ写像** `s' = f(s, x; theta)` に定式化。理由: Stage1b の Lipschitz
証明 (`verify_lipschitz_contraction`) が **対角写像前提** (座標ヤコビ `J=∂s'/∂s` が 1 変数 = Z3 高速可解)。
多様化しても対角構造を保てば既存 Z3 stack がそのまま効く (= same design pattern / partial stack reuse)。

**honest 留保**: full vector/matrix 版ではなく **core dynamics を対角スカラ recurrence に簡約した CPU
mock** (rwkv 以外)。"toy analogue / inspired by" 降格 (OTHERARCH 規律)。full 版は Stage 5+。

| kernel | state_update (対角, `kernels.py` 実装) | theta (dim=3) | full との差 (honest) |
|---|---|---|---|
| rwkv | `s' = decay·s + (1-decay)·tanh(mix·x + gate·s)` | decay∈[0,1], mix∈[-1,1], gate∈[-2,2] | なし (本流 `run_sequence` 再利用) |
| mamba_selective | `a=σ(α·x+β); s' = a·s + (1-a)·(gain·x)` | α∈[-2,2], β∈[-2,2], gain∈[-1,1] | 対角 1-state selective SSM mock (full Mamba の multi-head/selective-scan を落とす) |
| hopfield_dense | `z=β·(ξ·tanh(s)+x); s' = (1-η)·s + η·tanh(z)` | η∈[0,1], β∈[0,3], ξ∈[-1,1] | 1-pattern 連想想起 対角 mock (dense 多パターン retrieval を落とす) |
| linear_attn | `φ=softplus(w·x); s' = lam·s + φ·(v_gain·x)` | w∈[-2,2], lam∈[0,1], v_gain∈[-1,1] | bounded 変種 mock (full は累積で非有界 → lam<1 減衰で有界化) |

文献根拠 (SCOUT_kernel_lit.md): mamba ZOH 離散化 `Ā=exp(Δ·A)` の対角 A 負実部 → `Δ·a_i<0` 線形制約
(Gu&Dao 2312.00752, Sparse Mamba 2409.00563)。modern Hopfield (Ramsauer 2008.02217) の更新は CCCP
energy descent で β が specialist 度を支配し臨界 β_c で相転移 (2311.18434)。素の linear attention
(Katharopoulos 2006.16236) は decay なし積分器で発散しうる honest negative 候補、GLA/RetNet の
`S_t=γ·S_{t-1}+kv^T` (γ<1) のみ Banach 縮約 `‖S‖≤‖kv‖/(1-γ)` 成立。

---

## 3. 各 kernel への Stage1 / Lipschitz gate 適用 (タスク 3+6)

### 3.1 state_norm gate (既存 `verify_gene_safe` の対角 over-approx 一般化)

全 kernel が convex combination 風の有界性論証を持つ (詳細 DESIGN §2):
- **rwkv / mamba / hopfield**: `|s'| ≤ a·|s| + (1-a)·1` 型 → `|s|≤1` で `≤1` (RWKV と同型)。
- **linear_attn**: softplus 非有界のため **追加 gene 制約** `lam + softplus(|w|)·|v_gain| ≤ 1` を
  満たす gene のみ admit。→ **最も通りにくい** (smoke admit率=0.420)。

### 3.2 Lipschitz contraction gate (既存 `verify_lipschitz_contraction` の kernel 別拡張)

既存 (rwkv) の **sound free-variable abstraction** を kernel 別の座標ヤコビ `J(s,x;theta)` に一般化。
鍵 = どの非線形項を free 変数で over-approx するか。実コード (`invariants.py`) で確認した既存設計:
`J = decay + (1-decay)·gate_str·t` (`t=sech²(pre)∈(0,1]` を free `t∈[0,1]` に over-approx →
t について 1 次線形 → Z3 が `Or(J≥1, J≤-1)` を unsat なら L<1 certified)。閉形式
`_lipschitz_upper_bound = max(|decay|, |decay+(1-decay)·gate_str|)` が Z3 unsat と一致 (BG2)。

| kernel | 座標ヤコビ `J=∂s'/∂s` | free over-approx | Z3 適合度 | smoke 結果 |
|---|---|---|---|---|
| rwkv | `decay+(1-decay)·gate·t` (`t=sech²`) | free `t∈[0,1]` (既存) | ◎ 1 変数線形 | contract率 0.718, L_ub 0.030..1.925 |
| mamba | `J=a` (a=σ(.) は x のみ依存, `∂a/∂s=0`) | free `a∈[0,1]` | ○ trivial だが端点保守 | contract率 **0.000** (artifact, 下記) |
| hopfield | `(1-η)+η·u·β·ξ·v` (`u,v=sech²`) | free `u,v∈[0,1]` 双線形 (4隅 max) | △ 双線形, 一部 reject | contract率 0.797, L_ub 0.052..2.470 |
| linear_attn | `J=lam` (φ は x のみ依存) | free 不要 `L_ub=lam` | ○ mamba と同型 | contract率 **1.000**, L_ub 0.003..0.9997 |

**統一インターフェース設計 (research 側 `kernel_lipschitz.py`、未実装)**:
`verify_kernel_lipschitz(kernel_id, theta) -> LipschitzResult`。各 kernel が
`jacobian_freevars(theta) -> (free_var_bounds, J_expr_builder)` を提供、共通ドライバが z3 free 変数宣言
→ `J_expr` 構築 → `Or(J≥1, J≤-1)` 反例探索 → 既存 `LipschitzResult` 型をそのまま返す (partial stack
reuse)。閉形式 `L_upper_bound` は free 変数 hypercube 頂点での `|J|` max (rwkv の閉形式の一般化)。
state 方向 contraction のみ主張 (∂s'/∂x は別)、fail-closed (timeout/unknown は reject) を継承。

**honest — smoke が露呈させた問題 (mamba contraction率 0.000)**: 閉形式 `mamba_L_ub` が free `a∈[0,1]`
の端点 `a=1` で常に 1.0 を返すため `L_ub<1` が一度も真にならない。だが真の `J=a=σ(.)∈(0,1)` は常に
strict contraction。**開区間 (0,1) を端点込み [0,1] に over-approx した代償**。fail-closed 規律上は
健全だが DESIGN の「mamba 無条件 contraction」claim と乖離。→ **精緻化 (achievable 上界
`σ(|α|+|β|)<1` を定数注入) が要**。整いすぎた claim を smoke が反証した = honest disclosure の威力。
§6 の heavy 実験前に encoding 精緻化を行う (要追記事項)。

### 3.3 honest — Hopfield は別系統 (将来課題)

SCOUT_kernel_lit.md key_finding #4: modern Hopfield の収束は **energy descent (ΔE≤0, Lyapunov)** で
Lipschitz sup-norm とは質的に別系統。本 mock (対角 1-pattern) は外側 tanh で擬似的に Jacobian gate に
乗せているが、full dense Hopfield では **ΔE≤0 か softmax 作用素ノルム上界 (≤β·λ_max(cov)) の新 gate**
が要る。本 Stage はあくまで対角 mock の範囲で「Jacobian gate に乗る hopfield-inspired dynamics」を
扱う。full Hopfield の verifier 系統分離は Stage 5+ (型 `Trajectory.kind` で区別する設計済)。

---

## 4. smoke 結果 — どの kernel が skeleton 化 + 有界確認できたか (タスク 4+5)

### 4.1 gate smoke (`smoke_kernel_gates_results.json`, N=400 gene/kernel, z3 available)

| kernel | BG1 state_norm admit率 | BG2 Z3⟺閉形式 一致 | BG3 over-approx 違反 | L_ub 範囲 | 閉形式 contract率 |
|---|---|---|---|---|---|
| rwkv | 1.000 (PASS) | 1.0 (PASS) | 0 (PASS) | 0.030..1.925 | 0.718 |
| mamba_selective | 1.000 (PASS) | 1.0 (PASS) | 0 (PASS) | 1.000..1.000 | **0.000** (artifact) |
| hopfield_dense | 1.000 (PASS) | 1.0 (PASS) | 0 (PASS) | 0.052..2.470 | 0.797 |
| linear_attn | **0.420** (PASS, 232/400 reject) | 1.0 (PASS) | 0 (PASS) | 0.003..0.9997 | 1.000 |
| **BG5 後方互換** | — | — | — | — | bit 一致 PASS (N=50) |

### 4.2 動的 smoke (`smoke_results.json`, N_GENE=32/kernel, L=64, dim=8, |x|≤1, seed=20260601)

| kernel | finite_state | state_norm_ok | max_state_norm | 経路 |
|---|---|---|---|---|
| rwkv | True | True | 2.651 | 既存 `run_sequence` (backcompat) |
| mamba_selective | True | True | 1.781 | research mock |
| hopfield_dense | True | True | 2.264 | research mock |
| linear_attn | True | True | **21.817** | research mock |

(動的 smoke の state_norm_ok は緩い sanity bound `K=√8·20≈56.6`; 厳格 STATE_BOUND=1 gate は §4.1 が担当)

### 4.3 結論 (smoke スコープ)

- **skeleton 化 + 有界確認できた kernel = 4/4** (rwkv / mamba_selective / hopfield_dense / linear_attn)。
  全 kernel が finite + bounded に回り、Z3 state_norm admit 集合が非空。実装できなかった kernel は無し。
- **Stage 3a mechanism feasibility = smoke レベルで PASS** (BG1/2/3/5)。命題3a 支持。
- **支持された設計仮説**:
  - linear_attn の state_norm 過剰 reject (admit=0.420) と ungated max_norm=21.8 →
    **「kernel 間で gate 透過性が非対称」(§5 / 命題3b 土台) を数値・動的の両面で裏付け**。
    gate は bug でなく load-bearing 特徴。
  - hopfield の multi-basin (L_ub 最大 2.470, contract率 0.797, 約 20% reject) → Lipschitz gate で
    一部 reject される領域を持つ (DESIGN 予測通り)。
- **要修正 / honest 留保**:
  - mamba contraction率 0.000 = over-approx 保守性 artifact (§3.2)。encoding 精緻化要。**消さず記録**。
  - timeout 0 (hopfield 双線形 J でも N=400/1000ms で顕在化せず)。大 N / tighter bound で再評価要。
- **未実施 (本 smoke 対象外)**: BG4 (kernel_id_shift mutation 絡みの random walk decode 健全)、
  BG6-8 (specialist 出現)、BG9 (③ 欺瞞地形)。各命題は別 BG で分離済 (混同しない構造)。

---

## 5. ③ 欺瞞地形仮説 — 拡張空間が ③ load-bearing になりうるか (タスク 7)

### 5.1 背景接続 (Step4 + 正規化交絡 ablation)

Step4: 3-param + 実 proxy は滑らか → ③不要。残る問い = 「探索空間拡張で欺瞞地形が出るか」。
DECEPTIVENESS_MEASURE_VERDICT: 欺瞞性を **magnitude として測るのは困難** (循環論法 / CLT影 / 記述子依存 /
予算依存)。negative も N/A も正当。→ 本検定はその失敗教訓を **全前提に組む**。

### 5.2 欺瞞地形が出うる構造的根拠 (smoke で一部実測)

`KernelGenome` 空間は 3-param と質的に違う 2 性質を持つ:

1. **離散 kernel_id 障壁**: kernel 間遷移 (`kernel_id_shift`) は不連続な fitness 段差を作りうる。
   ある task の best が hopfield basin にあり rwkv basin から連続 hill-climb で到達不能なら、
   Step4 exp4 の "genotypic corridor + dip" の **kernel 版 corridor**。
2. **非対称 gate 透過性 (smoke 実測)**: linear_attn admit率 0.420 / hopfield Lipschitz 約 20% reject の
   ように、高性能 gene が gate で reject される領域に偏在 → 実効 feasible 集合が dip / 穴を持つ
   (gate が landscape に欺瞞構造を inject する経路)。**この 2 数値は smoke で接地済**。

> **仮説H7**: `KernelGenome` 拡張空間 (4 kernel × theta + gate 制約) は、3-param 単一 kernel 空間が
> 持たなかった欺瞞的 corridor 構造 (kernel 障壁 / gate 穴) を multi-task fitness 上に持ちうる。
> 帰結 = MAP-Elites (kernel_id × theta behavior niching) が RR-hillclimb / panmictic-GA / random を
> ≥15 seed Wilcoxon p<0.05 + Cliff δ で上回る (③ load-bearing)。**仮説であり未実証**。

### 5.3 検定法 (DECEPTIVENESS_MEASURE の失敗を回避する事前登録)

Step4 exp4-5 の MAP-Elites vs 3 baseline 比較を `KernelGenome` 空間で再走。negative を license する
5 条件を **全て事前登録** (DECEPTIVENESS_MEASURE_VERDICT §5):

- (a) **記述子不変 behavior**: `(kernel_id 連続値, theta L1 norm)` を固定。ビン数も合成・実で同一固定
  (VERIFY_descriptor_dependence のビン反転回避)。
- (b) **固定 n_bins** を事前登録 (判定がビンで反転するため必須)。
- (c) **固定予算** n_evals を事前登録 (magnitude は予算敏感)。
- (d) **positive control**: hopfield kernel 障壁を意図的に強調した synthetic multi-task (一部 task は
  hopfield でしか高 fitness) を作り、MAP-Elites が ③ load-bearing になることを **先に確認**
  (メトリック validity 確立 = Step4 exp4 deceptive corridor positive control と同役割)。
- (e) **negative control**: kernel 中立 task で ③ が立たない (優位消失) ことを確認 (exp5 smooth 対照)。

### 5.4 正規化交絡 ablation との接続 (結論と合わせる)

DECEPTIVENESS_MEASURE の循環論法 (severity HIGH) = 「behavior=mean を彫った合成で ρ=1 は定義のなぞり」。
本検定では behavior に **kernel_id (離散障壁) を入れる**ため、fitness landscape を behavior=mean で彫る
循環は構造的に起きにくい (kernel_id は fitness を直接定義しない選択変数)。新たな交絡対策:
**behavior から kernel_id を抜いた theta-only MAP-E と比較**し、優位が kernel diversity 維持由来か
単なる探索量由来かを切り分ける (Step4 C4 init_batch ablation と同思想)。さらに **gate-on / gate-off
両条件**で測り、gate が欺瞞構造を inject する効果を honest 開示。

### 5.5 判定 (3 値)

- **③ load-bearing (欺瞞地形あり)**: positive control で ③ 成立 **かつ** 実 multi-task でも
  MAP-E が 3 baseline 全勝 (p<0.05, δ 非無視) + 交絡 ablation で優位が diversity 維持に帰属 →
  Step4 の予想「拡張で unlock」が支持される。
- **③不要 (欺瞞地形なし)**: positive 成立も実 multi-task で優位消失 → Step4 §7 結論が拡張空間でも
  再現 (honest negative)。
- **N/A**: positive すら立たない or 記述子 / 予算で判定反転 → 測定不能で保留。

honest: full LLM でなく memory_tasks proxy (Step4 §7 留保継承)。**整いすぎた③成立は内訳を疑う**
(`feedback_benchmark_honest_disclosure`)。本検定 = BG9。

---

## 6. 次の heavy 実験手順 (Stage 3a gate sweep → 3b specialist 進化)

依存順 (`BREAK_GATES.md` の DAG): BG1-5 (feasibility, smoke PASS 済) → BG4 → BG6-8 (specialist) →
BG9 (③ 欺瞞地形, positive/negative control 先行)。壊れた kernel で specialist を測るのは無意味。

### Step A — encoding 精緻化 + Stage 3a 本 sweep (gate)
1. `kernel_lipschitz.py` を実装 (§3.2 統一インターフェース)。**mamba/linear_attn の achievable 上界
   定数注入** (`a ≤ σ(|α|+|β|) < 1`) で over-approx を緩和 (mamba contract率 0.000 artifact 解消)。
2. BG1-3 を本サイズ (N=2000 gene/kernel, Sobol サンプル) で再走。admit率 / Z3⟺閉形式一致 / over-approx
   soundness を kernel 別 timeout 調整 (hopfield 双線形 J のみ長め) で記録。
3. **BG4 (kernel_id_shift random walk)**: random `KernelGenome` から `kernel_id_shift` + theta gaussian を
   1000 step → 各 step decode + simulate (L=128, dim=8) finite チェック (smoke_kernels.py の finite 確認を
   mutation 絡みに拡張)。`apply_kernel_changeop` を research に新規実装 (src OP_TYPES 不変)。

### Step B — KernelGenome を ea_lab / selection_lab に接続 (dim=5)
4. `kernels.py::KernelGenome.as_array/from_array` (dim=5, 着地済) を `ea_lab.run_ea_methods_over_seeds` /
   `selection_lab` に流し込む adapter を research に実装。GA operator (`_clip`, `rng.normal(size=5)`) は
   無改変流用 (固定長 union の設計目的)。online gate は §3 の state_norm + Lipschitz を kernel 別適用。
5. behavior descriptor = `(kernel_id 連続値 1D, theta L1 norm 1D)` の 2D MAP-Elites grid を実装
   (n_bins 事前登録)。

### Step C — Stage 3b specialist 進化 (BG6-8)
6. **BG6 (specialist 写像)**: memory_tasks 各 task (FlipFlop / DelayedRecall / DelayedParity) を単独
   fitness にして進化 → best gene の kernel_id 集計 → task→kernel 写像表。非定数なら命題3b(i) 支持。
   緩和 = 各 kernel が得意な task を含める (hopfield→DelayedRecall, mamba→FlipFlop, linear_attn→長距離)。
7. **BG7 (archive collapse なし)**: TaskMixture 上で MAP-Elites を回し final archive 占有 cell の
   kernel_id 正規化 Shannon entropy を gate-on / gate-off 両条件で測る (H>0.1 bits を pass)。
8. **BG8 (load-bearing)**: kernel_id 進化版 vs 固定 4 本 (rwkv/mamba/hopfield/linear-only) を ≥15 seed
   `selection_lab.compare` で paired Wilcoxon p<0.05 + Cliff δ、`split_regimes` の test 汎化主指標。

### Step D — BG9 (③ 欺瞞地形)
9. §5.3 の positive control (hopfield 障壁強調 synthetic) → negative control (kernel 中立) → 実
   memory_tasks の順で Step4 exp4-5 harness を `KernelGenome` 空間で再走。事前登録 (固定 n_bins / 予算 /
   記述子不変)。theta-only behavior ablation + gate-on/off で交絡切り分け。判定 ③成立/③不要/N/A の 3 値。
   `step_c_deceptiveness_measure/` の FDC/elite-dip 計測を kernel 混合空間に適用するのが論理的接続。

honest: Step A の mamba encoding 精緻化を **必ず先**に済ませる (artifact のまま Step C/D に進むと
contract率の誤読が specialist / 欺瞞地形の判定を汚染する)。

---

## 7. honest 留保・リスク (タスク 8)

1. **specialist が出ない (BG6 反証)** — memory_tasks が kernel 中立なら命題3b は honest negative。
   緩和 = 各 kernel が構造的に得意な task を含める。それでも出なければ「土俵でない」と honest 報告。
2. **対角 mock が full kernel と乖離** — 1-state 対角簡約は multi-head 二次相互作用 / 高次元 retrieval を
   落とす。claim を toy analogue / inspired by に降格 (OTHERARCH 規律)、full 版 Stage 5+ 明示。
   特に hopfield は energy descent 別系統 (§3.3) で対角 mock は近似に過ぎない。
3. **Z3 nonlinear timeout (hopfield 双線形 J)** — smoke では N=400/1000ms で timeout 0 だが大 N で
   増えうる → fail-closed reject + endpoint 閉形式クロスチェック (BG2) + kernel 別 timeout 調整。
   timeout 多発は honest 記録。
4. **欺瞞地形検定の循環論法再発** — positive/negative control + 事前登録で予防、magnitude 転移が
   取れなければ N/A 保留。整いすぎた③成立は内訳を疑う。
5. **kernel_id 連続→離散 floor が GA 探索を歪める** — 境界 (k=0.99→1.0) で不連続。緩和 = behavior の
   kernel_id は連続値で記録 (探索は滑らか) + kernel_swap_mock 併用検討。
6. **gate が kernel 多様性を選択前に殺す** — smoke で linear_attn admit 0.420 のように gate が
   hopfield/linear_attn を過剰 reject すると集団が rwkv/mamba に collapse (BG7 反証)。これは設計交絡で
   あり同時に発見 (gate が欺瞞構造を inject) → gate-on/gate-off 両条件で差分 honest 開示。
7. **既存資産改変の誘惑** — protocol.py S3 延期事項 (op_type kernel 別検証) / ChangeOp OP_TYPES 拡張を
   src でやりたくなるが research 隔離厳守。BG5 (rwkv 埋め込み bit 一致, smoke pass) で src 不変担保。
8. **honest disclosure 違反リスク** — mechanism feasible (§4 smoke PASS) を specialist 実証 (§6 未着手) や
   欺瞞地形 (§5 未着手) と混同するリスク → 全命題を別 BG で分離し各々独立に pass/fail/N/A を出す構造。
   smoke が早速 mamba「無条件 contraction」claim を反証 (contract率 0.000) = honest disclosure の実演。

---

## 8. 参照 (すべて読むだけ、research 隔離 / src 非変更)

- `src/llcore/kernel/protocol.py` — Kernel/GeneCodec/Trajectory/VerifierBackend (S3 延期事項)
- `src/llcore/kernel/rwkv.py` — RWKV 準拠例 (本流委譲 wrapper, 後方互換の核)
- `src/llcore/state_update/genes.py` — StateUpdateGene (3-param, kernel_id=0 埋め込み先)
- `src/llcore/verifier/invariants.py` — state_norm (`verify_gene_safe`) + Lipschitz
  (`verify_lipschitz_contraction` / `_lipschitz_upper_bound` / `empirical_lipschitz`)
- `src/llcore/verifier/changeop.py` — ChangeOp/OP_TYPES (kernel_id_shift 拡張先, src 不変)
- `research/step4_selection/selection_lab.py` — MAP-Elites + 3 baseline + compare (③ 検定流用)
- `research/ea_multitask/{ea_lab.py, task_mixture.py}` — multi-task train/test 分離 + ③ ablation
- `research/step_c_memory_tasks/memory_tasks.py` — FlipFlop/DelayedRecall/DelayedParity (regime 源)
- `research/step_c_deceptiveness_measure/DECEPTIVENESS_MEASURE_VERDICT.md` — 欺瞞 magnitude 測定失敗教訓
- `research/other_archs/OTHERARCH_VERDICT.md` — kernel plugin 化 4 条件 + "partial stack reuse" 正名化
- `docs/poc/STEP4_SELECTION_VERDICT.md` — ③ 欺瞞 corridor 限定 + 実 proxy 滑らか (本設計の出発点)
- 本 directory: `DESIGN_kernel_diversification_3b.md` / `SCOUT_kernel_lit.md` / `BREAK_GATES.md` /
  `SCOUT_NOTES_smoke.md` / `kernels.py` / `smoke_kernel_gates.py` / `smoke_kernels.py`
