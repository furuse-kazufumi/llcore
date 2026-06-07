# Stage 3b — Kernel 多様化 Gene の falsifiable 設計

**作成**: 2026-06-01
**位置づけ**: llcore 研究記録 (`research/kernel_diversification/` 隔離)。**src/ には触らない**。
既存 `src/llcore/kernel/{protocol.py, rwkv.py}` / `state_update/` / `verifier/invariants.py` は
**読むだけ** で再利用設計する。git は orchestrator が一括 commit。
**ステータス**: DESIGN (まだ実装も実証もしていない。本書は「やる」設計であり「やった」結果ではない)。

---

## 0. 戦略的接続 (なぜ kernel 多様化なのか)

Step4 (`docs/poc/STEP4_SELECTION_VERDICT.md`) の確定結論:

- **③ (選択圧/分離 = MAP-Elites niching) は欺瞞的 corridor landscape でのみ load-bearing。**
- 3-param RWKV gene 空間 (decay/mix/gate_str) + 実 substrate proxy (ESN×実テキスト, step6) は
  **滑らか/単峰 broad-ridge** で、欺瞞 corridor が自然出現しない → そこでは③不要。
- Step4 §6 + DECEPTIVENESS_MEASURE_VERDICT の含意: **真の unlock は「探索空間を 3-param から拡張」**
  すること。richer な architecture 空間なら欺瞞地形を持ちうるか、を問い直す。

本 Stage 3b は **まさにその探索空間拡張**である。3 連続 param 空間 → **離散 kernel 種別 ×
kernel 固有連続 param** の混合空間に拡張する。

- **Stage 3a (本書 §1-§6)** = kernel 種別を遺伝子化し、各 kernel が Stage1 gate (state_norm /
  Lipschitz contraction) を pass する破綻ゲートを定義 (mechanism feasibility)。
- **Stage 3b (本書 §1-§5 + §7)** = multi-task で specialist が出現するか (集団が単一 kernel に
  固定しない) + **混合空間が欺瞞地形 (③ load-bearing) を持ちうるか** を検定。

honest 留保 (最重要): 本書は **設計**。「できる (mechanism feasible)」の論証と「やった (実証)」は
厳密に区別する。§3a/§3b の命題は falsifiable に書くが、結果はまだ無い。

---

## 1. Gene 表現 — kernel 種別 + param を 1 genome に符号化 (§タスク 1)

### 1.1 設計制約 (既存 protocol との整合 / 後方互換)

既存資産から読み取れる固定点:

- `kernel/protocol.py` の `Kernel[GeneT]` Protocol は **gene 型をパラメトリック** (`GeneT`)。
  RWKV=`StateUpdateGene` (3-dim)、SNN=`LIFGene` (4-dim) と **kernel ごと gene 型が違う**設計が
  すでに織り込まれている (docstring「gene 型は kernel ごとに異なる」)。
- `GeneCodec[GeneT]` が `dim` / `lower` / `upper` / `to_array` / `from_array` / `clip` を宣言。
  各 kernel は **可変 dim の連続ベクトル** に符号化できる契約をすでに持つ。
- `verifier/changeop.py` の `ChangeOp.__post_init__` は現状 RWKV 4 op_type のみ許可
  (`OP_TYPES = (decay_shift, mix_shift, gate_shift, kernel_swap_mock)`)。
  protocol.py docstring の **S3 延期事項** が「op_type 検証を kernel 別 (`change_op_types` 参照) に
  拡張」と明記 → 本設計はこの延期事項を research で先行検証する。
- semver: src 既存シンボルは一切変更しない。本設計の符号化は **research 側に新規 codec を置き**、
  既存 `StateUpdateGene` を 1 kernel (`rwkv`) として包含する後方互換構造にする。

### 1.2 提案: タグ付き union genome (`KernelGenome`)

genome を 2 区画に分ける:

```
KernelGenome = (kernel_id: int,  theta: np.ndarray[max_dim])
```

- **`kernel_id`** — 離散遺伝子。`0=rwkv / 1=mamba / 2=hopfield / 3=linear_attn`。
  GA 上は実数 1 次元 `k ∈ [0, n_kernels)` で持ち、`int(floor(clip(k, 0, n_kernels-eps)))` で離散化
  (selection_lab/ea_lab が連続ベクトル前提なので、連続→離散 floor で既存 GA operator を無改変流用)。
- **`theta`** — 連続遺伝子ベクトル。`max_dim = max_k dim(k)` の固定長 (本設計 4 kernel で max_dim=4)。
  各 kernel は先頭 `dim(kernel_id)` 次元のみを **自分の codec で解釈**し、残余次元は無視 (junk DNA)。
  これにより GA の crossover/mutation を **固定長ベクトル上で kernel 横断に**実行でき、
  `kernel_swap` 変異後も theta 区画が連続的に流用される (=漸進的 kernel 遷移が表現可能)。

固定長 union を選ぶ理由 (TRIZ #1 分割 / #3 局所性質):
- 可変長 genome は既存の固定 dim GA operator (`_clip`, `rng.normal(0, sigma, size=dim)`) を壊す。
- 固定長 + junk 区画なら **既存 `map_elites_*` / `panmictic_ga` を 1 行も変えず**に流用できる
  (dim = 1 + max_dim = 5)。後方互換性を operator レベルで担保。

### 1.3 既存 `StateUpdateGene` との後方互換

- `kernel_id=0 (rwkv)` のとき `theta[:3]` を `StateUpdateGene(decay, mix, gate_str)` に decode。
  → 既存 `RWKVCodec.from_array` / `run_sequence` / `verify_gene_safe` を **そのまま** 呼べる。
- 3-param 既存実験 (step4/step6/ea_multitask) は `KernelGenome` の `kernel_id=0` 部分空間に
  **完全埋め込み**される (=既存 landscape は本拡張空間の slice)。
  → Step4 の「3-param は滑らか」結論を **退化させずに包含** し、拡張空間が新たに欺瞞地形を
  持つかを差分として測れる (§7)。

### 1.4 ChangeOp 整合 (op_type の kernel 別拡張)

- 既存 `change_op_types` が kernel ごとに宣言されている (RWKV: decay/mix/gate_shift)。
- 本設計は **新 op_type `kernel_id_shift`** を research 側 ChangeOp 拡張に追加 (src の `OP_TYPES` は
  変えない。research に `apply_kernel_changeop` を新規実装)。`kernel_id_shift` は kernel_id を
  ±1 する離散遷移 op。既存 `*_shift` は対応 kernel の theta 区画に作用する形に一般化。
- これは protocol.py の S3 延期事項「op_type 検証を kernel 別に拡張」の research 先行 PoC に相当。

---

## 2. 各 kernel の state-update 写像 + protocol 適合性 + Stage1 gate 適用可否 (§タスク 2)

全 kernel を **対角 (各座標独立) スカラ写像** `s' = f(s, x; theta)` として定式化する。
理由: Stage1b の Lipschitz 証明 (`verify_lipschitz_contraction`) が **対角写像前提** (座標ヤコビ
`J = ∂s'/∂s` が 1 変数線形 = Z3 高速可解)。多様化しても **この対角構造を保てば既存 Z3 stack が
そのまま効く** (= same design pattern, partial stack reuse、OTHERARCH_VERDICT の正名化と整合)。

honest 留保: これは「教科書 mamba/hopfield/linear-attn の full 実装」ではなく、**各 kernel の
core dynamics を対角スカラ recurrence に簡約した CPU mock**。full vector/matrix 版は Stage 5+。
このスコープ宣言は OTHERARCH_VERDICT の "toy analogue / inspired by" 降格規律に従う。

### 2.1 rwkv (既存、baseline)

```
s' = decay·s + (1-decay)·tanh(mix·x + gate_str·s)
```
- theta = (decay∈[0,1], mix∈[-1,1], gate_str∈[-2,2]), dim=3。
- protocol 適合: **完全適合** (既存 `StateUpdateGene` / `RWKVCodec` / `run_sequence` そのもの)。
- Stage1 gate: state_norm `verify_gene_safe` ✓ / Lipschitz `verify_lipschitz_contraction` ✓
  (既存実装が J=decay+(1-decay)·gate_str·t を扱う)。

### 2.2 mamba_selective (input-selective Δ, 対角 SSM mock)

Mamba の選択的 state space の核 = **入力依存の離散化 Δ(x)** で decay を gating。対角簡約:
```
a = sigmoid(alpha·x + beta)            # 入力依存 forget gate ∈ (0,1)
s' = a·s + (1-a)·(gain·x)              # selective leak + 線形入力注入
```
- theta = (alpha∈[-2,2], beta∈[-2,2], gain∈[-1,1]), dim=3。
- protocol 適合: `MambaCodec` を新規 (dim=3, bounds 上記)。`simulate` は対角 recurrence を回し
  `Trajectory(kind="state")` を返す → **既存 Trajectory 型に完全適合**。
- **state_norm gate 適用可否**: ✓ 可能。`|s'| ≤ a·|s| + (1-a)·|gain|·|x|`。`|x|≤1, |gain|≤1` で
  `≤ a·|s| + (1-a)` → `|s|≤1` なら `≤1` (convex combination, RWKV と同型の有界性論証)。
  Z3 化: a を free t∈(0,1) に over-approx、`s_next = t·s + (1-t)·gain·x`、反例 `|s_next|>1` 探索。
- **Lipschitz gate 適用可否**: △ 注意。`∂s'/∂s = a + s·∂a/∂s`。ここで a は x のみの関数なので
  `∂a/∂s=0` → **`∂s'/∂s = a ∈ (0,1)` で常に L<1** (無条件 contraction)。
  ただし `∂s'/∂x` 経由の入力方向ヤコビは別 (state 方向のみ L<1 を主張)。既存 stack の
  state-direction 定義と整合。Z3 化は RWKV より易しい (J が theta 非依存で a∈(0,1) 一定)。

### 2.3 hopfield_dense (associative retrieval, 対角 mock)

Modern Hopfield = energy 最小化への retrieval dynamics。対角簡約 (1 パターン保持の連想想起):
```
s' = (1-eta)·s + eta·tanh(beta·(xi·tanh(s) + x))    # xi=蓄積パターン方向, beta=逆温度
```
- theta = (eta∈[0,1], beta∈[0,3], xi∈[-1,1]), dim=3。
- protocol 適合: `HopfieldCodec` 新規。`simulate` は同 recurrence → `Trajectory(kind="state")` 適合。
- **state_norm gate 適用可否**: ✓ 可能。外側 tanh で更新項 `|tanh(.)|≤1`、convex 風
  `|s'| ≤ (1-eta)|s| + eta·1` → `|s|≤1` で `≤1`。eta∈[0,1] が convex 係数。RWKV と同じ論証骨格。
- **Lipschitz gate 適用可否**: △ 要注意 (retrieval は **複数固定点**を持つのが本質 → state 方向
  contraction L<1 を **全域では満たさない設計**)。ここが重要: hopfield は構造的に **non-contractive な
  領域 (basin 境界, ‖J‖≥1)** を持ちうる。
  `∂s'/∂s = (1-eta) + eta·sech²(z)·beta·xi·sech²(s)` (z=外 tanh 引数)。
  beta·xi が大きいと J>1 領域あり → **Lipschitz gate は一部 hopfield gene を reject する**。
  これは bug ではなく **kernel 間で gate pass 率が違う** ことの設計上の核心 (§3a/§7 で利用)。

### 2.4 linear_attn (kernel feature map, 対角 mock)

Linear attention = `s' = s + φ(k)·v` の累積 KV state。対角スカラ簡約 (φ=elu+1 風を 1 次元化):
```
phi = softplus(w·x)                    # 非負 feature map (φ(k)≥0)
s' = lam·s + phi·(v_gain·x)            # 減衰付き KV 累積
```
- theta = (w∈[-2,2], lam∈[0,1], v_gain∈[-1,1]), dim=3。
- protocol 適合: `LinearAttnCodec` 新規。`simulate` → `Trajectory(kind="state")` 適合。
- **state_norm gate 適用可否**: ✓ **ただし要追加制約**。`|s'| ≤ lam·|s| + phi·|v_gain|·|x|`。
  softplus は非有界 (`phi→∞` as `w·x→∞`) → **無制約だと state_norm 違反しうる**。
  `|x|≤1, |w|≤2` で `phi ≤ softplus(2) ≈ 2.13` を上界に。`|s|≤1` で `|s'| ≤ lam + 2.13·|v_gain|`。
  → `|s|≤1` 不変量を満たすには **`lam + 2.13·|v_gain| ≤ 1`** という gene 制約が要る (= state_bound を
  緩めるか v_gain を絞る)。これも kernel 間 gate 率差の源 (linear_attn は naive だと reject 多)。
  honest 留保: linear attention の state は本来「累積」で有界でない設計 → 有界化のため
  `lam<1` 減衰を入れた **bounded 変種** に簡約している (full linear attn とは別物)。
- **Lipschitz gate 適用可否**: ✓ `∂s'/∂s = lam ∈ [0,1)` (phi は x のみ依存)。lam<1 なら無条件 L<1。
  state_norm より Lipschitz の方が通りやすい (mamba と同じく state 方向は単純減衰)。

### 2.5 適合性サマリ表

| kernel | dim | state_norm gate | Lipschitz gate | full との差 (honest) |
|---|---|---|---|---|
| rwkv | 3 | ✓ (既存) | ✓ (既存) | なし (本流) |
| mamba_selective | 3 | ✓ convex | ✓ 無条件 (a∈(0,1)) | 対角 1-state SSM mock |
| hopfield_dense | 3 | ✓ convex | **△ 一部 reject** (multi-basin) | 1-pattern 対角 mock |
| linear_attn | 3 | ✓ **要 gene 制約** | ✓ (lam<1) | bounded 変種 mock |

**設計の核心**: 4 kernel とも state_norm は通せるが、**Lipschitz contraction の通りやすさが kernel
ごとに構造的に違う** (mamba/linear=易, rwkv=中, hopfield=難)。この **非対称な gate 透過性** が
Stage 3b specialist 出現 (§3b) と欺瞞地形 (§7) の load-bearing 仮説の土台になる。

---

## 3. 命題 (falsifiable)

### 3a. 命題3a — 各 kernel が Stage1 gate を pass しうる (§タスク 3)

> **命題3a**: 4 kernel (rwkv/mamba/hopfield/linear_attn) それぞれについて、clip 範囲内に
> **state_norm gate (`|s|≤state_bound` 不変) を Z3 unsat で pass する gene が非空に存在し**、
> かつ rwkv/mamba/linear_attn は **Lipschitz contraction gate (L<1) を pass する gene も非空**に
> 存在する。hopfield は **L<1 pass 集合が空でないが clip 範囲の一部に限られる** (multi-basin 由来)。

**falsifiable / 破綻条件**:
- (反証 A) ある kernel で clip 範囲全域が state_norm gate を **reject** する (admit 集合が空) →
  その kernel の対角写像 + bounds 設計が破綻 (state_norm 上界論証が間違い)。
- (反証 B) Lipschitz の L_upper_bound 閉形式 (`_lipschitz_upper_bound` 相当) と
  Z3 verdict (unsat/sat) が **矛盾** する → Z3 encoding の健全性破綻 (BG 参照)。
- (反証 C) `empirical_lipschitz` 経験値 > Z3 の `L_upper_bound` → over-approx が unsound (BG3 と同型)。

**計測法**: 各 kernel で clip 範囲から Sobol/grid で N=2000 gene サンプル → 各 gene に
state_norm Z3 gate + Lipschitz Z3 gate を適用 → **admit 率** を記録。
admit 率 > 0 (非空) を pass 条件、Z3 verdict と閉形式上界の一致を健全性条件とする。

### 3b. 命題3b — specialist 出現 / 単一 kernel に固定しない (§タスク 4)

> **命題3b**: 既存 `TaskMixture` (memory_tasks の FlipFlop / DelayedRecall / DelayedParity 等を
> regime とする多タスク分布) 上で `KernelGenome` 集団を進化させたとき、
> **(i) 異なる task に異なる kernel_id が選ばれ (task→best-kernel 写像が非定数)**、かつ
> **(ii) 進化集団 (MAP-Elites archive) の kernel_id 分布が単一値に collapse しない**
> (final archive で ≥2 kernel が live occupancy を持ち、Shannon entropy > 0)。

**falsifiable / 破綻条件**:
- (反証 D) 全 task で同一 kernel_id が best → specialist 不在 (1 kernel が万能 = 多様化に価値なし)。
- (反証 E) archive の kernel_id 分布が単一値に collapse (entropy ≈ 0) → 集団が単一 kernel に固定
  (多様化 gene が機能していない = 命題3b 反証)。
- (反証 F) kernel_id を **固定**した ablation (各 kernel 単独で全 task を解く) が、
  kernel_id 進化版と **同等以上**の multi-task test 汎化を出す → kernel 選択が load-bearing でない
  (多様化の付加価値ゼロ。Step4 の③ ablation 規律と同型 = niching が effort でなく diversity に
  帰属するかを切り分けるのと同じ思想)。

**計測法**: `ea_lab.run_ea_methods_over_seeds` を `KernelGenome` (dim=5) に拡張流用。
- behavior descriptor = `(kernel_id_onehot 縮約 1D, theta の代表量 1D)` の 2D を MAP-Elites grid に。
- train/test regime 分離 (`split_regimes`) で hold-out 汎化を主指標 (E-A 方針流用、リーク防止)。
- per-task best-kernel: 各 task を単独 fitness にした eval で best gene の kernel_id を集計 (写像表)。
- archive kernel entropy: final archive 占有 cell の kernel_id ヒストグラム → Shannon entropy。
- ablation: kernel_id 固定 4 本 vs kernel_id 進化版を ≥15 seed paired Wilcoxon p<0.05 + Cliff δ
  (Step4/E-A と同じ honest 統計、`selection_lab.compare` 流用)。

honest 留保: specialist が出るには **task 間で best kernel が実際に違う構造**が要る。出なければ
それ自体が honest negative (「memory_tasks は kernel 中立 = 多様化の価値が出る土俵でない」)。
捏造して positive に倒さない (DECEPTIVENESS_MEASURE_VERDICT の N/A 受容規律)。

---

## 4. 破綻ゲート + 計測法 (§タスク 5)

falsifiable な break gate を ID 付きで定義 (実装時に各 gate を smoke で測る)。
詳細は `BREAK_GATES.md` に転記。要約:

| ID | 破綻条件 (これが起きたら設計が破綻) | 計測 |
|---|---|---|
| BG1 | いずれかの kernel で state_norm admit 集合が空 | N=2000 gene サンプルの Z3 admit 率 > 0 |
| BG2 | Lipschitz Z3 verdict と閉形式 L_upper_bound が矛盾 | 全サンプルで `(L_ub<1) == (status=="unsat")` |
| BG3 | empirical_L > Z3 L_upper_bound (over-approx unsound) | 各 kernel で `emp_L ≤ L_ub + tol` |
| BG4 | kernel_swap 変異後 genome が decode 不能 / NaN trajectory | random walk 1000 swap で finite & decode OK |
| BG5 | 既存 rwkv 結果が KernelGenome 埋め込みで数値変化 | kernel_id=0 経路 == 既存 `run_sequence` bit 一致 |
| BG6 | specialist 不在 (全 task 同一 best kernel) | per-task best-kernel 写像が非定数 |
| BG7 | archive kernel collapse (entropy≈0) | final archive kernel entropy > 0 閾値 |
| BG8 | kernel 選択が load-bearing でない (ablation 同等) | 進化版 > 固定版 paired Wilcoxon p<0.05 |
| BG9 (③) | 拡張空間でも欺瞞地形が無い → ③不要が再確認 | §7 の欺瞞メトリック (positive control 付き) |

BG1-BG5 = mechanism feasibility (Stage 3a)。BG6-BG8 = specialist (Stage 3b)。BG9 = 欺瞞地形 (§7)。
honest: BG9 が「欺瞞なし」を出しても negative として正当 (Step4 §7 と同じ結論になりうる)。

---

## 5. Stage 1b Lipschitz contraction invariant の各 kernel 拡張 (§タスク 6)

既存 `verify_lipschitz_contraction` の設計骨格 (sound free-variable abstraction) を **kernel 別の
座標ヤコビ `J(s,x;theta)` に一般化**する。鍵 = **どの非線形項を free 変数で over-approx するか**。

既存 (rwkv): `t = sech²(pre) ∈ (0,1]` を free `t∈[0,1]` に over-approx → `J = decay+(1-decay)·gate_str·t`
を t について 1 次線形 → Z3 が `|J|≥1` を unsat 判定なら L<1 certified。

各 kernel への拡張 (同 pattern):

- **mamba**: `J = a` (a=sigmoid(.) は x のみ依存、∂a/∂s=0)。free `a∈[0,1]` で `|J|=a<1` を Z3 判定。
  → **最も易しい** (J が theta 完全非依存、ほぼ trivial unsat)。閉形式 `L_ub = 1` 端点除外で `<1`。
- **hopfield**: `J = (1-eta) + eta·u·beta·xi·v`、`u=sech²(z)∈(0,1]`, `v=sech²(s)∈(0,1]`。
  **free 2 変数** `u,v∈[0,1]` で over-approx → `J` は u,v について双線形 (各 1 次)。Z3 は
  双線形でも端点列挙で可解 (4 隅 max)。閉形式 `L_ub = max over (u,v)∈{0,1}² of |J|`。
  beta·xi 大 → `L_ub≥1` で reject。**これが hopfield の一部 reject の Z3 的根拠**。
- **linear_attn**: `J = lam` (phi は x のみ依存)。free 不要、`L_ub = lam < 1`。mamba と同型に易しい。

**統一インターフェース設計** (research 側 `kernel_lipschitz.py`):
```
verify_kernel_lipschitz(kernel_id, theta) -> LipschitzResult
```
- 各 kernel が `jacobian_freevars(theta) -> (free_var_bounds, J_expr_builder)` を提供。
- 共通ドライバが z3 free 変数を bounds で宣言 → `J_expr` を構築 → `Or(J≥1, J≤-1)` 反例探索 →
  既存 `LipschitzResult` (contraction/L_upper_bound/used_z3/solver_status) を **そのまま返す**
  (= 既存 result 型再利用、OTHERARCH の "partial stack reuse" 正名化と整合)。
- 閉形式 `L_upper_bound` は kernel 別 endpoint enumeration (rwkv の `_lipschitz_upper_bound` を
  一般化: free 変数の hypercube 頂点で `|J|` の max)。Z3 unsat と必ず一致すべき (BG2)。

honest 留保:
- **state 方向 contraction のみ**を主張 (入力方向 `∂s'/∂x` は別)。既存 stack の定義と一致。
- free 変数 over-approx は **conservative reject** を許す (sat = achievable でない点かも)。
  fail-closed 規律 (既存 docstring) を継承: timeout/unknown は reject。
- hopfield の双線形 J は z3 で nonlinear real arithmetic になり **timeout 率が rwkv より上がりうる**
  → timeout は fail-closed reject (= hopfield admit 率がさらに下がる方向、健全)。

---

## 6. ③ 欺瞞地形を持ちうるかの仮説と検定法 (§タスク 7)

### 6.1 背景接続 (Step4 + 正規化交絡 ablation)

Step4: **3-param + 実 proxy は滑らか → ③不要**。残る問い = 「探索空間拡張で欺瞞地形が出るか」。
DECEPTIVENESS_MEASURE_VERDICT: 欺瞞性を **magnitude として測るのは困難** (循環論法/CLT影/記述子依存/
予算依存)。negative も N/A も正当。→ 本検定は **その失敗教訓を全て前提に組む** (positive control 必須、
事前登録、rank だけでなく magnitude 転移条件、記述子不変性)。

### 6.2 欺瞞地形が出うる **構造的根拠** (仮説)

`KernelGenome` 空間は 3-param と質的に違う 2 性質を持つ:

1. **離散 kernel_id 障壁**: kernel 間遷移 (`kernel_id_shift`) は **不連続な fitness 段差**を作りうる。
   ある task の best が hopfield basin にあり、rwkv basin から連続 hill-climb で到達できないなら、
   これは Step4 exp4 の "genotypic corridor + dip" の **kernel 版 corridor**。
2. **非対称 gate 透過性 (§2.5)**: hopfield は高性能 gene が **Lipschitz gate で reject される領域**に
   偏在しうる → fitness 高い領域が gate で削られ、**実効 feasible 集合が dip/穴**を持つ
   (= gate が landscape に欺瞞構造を inject する経路)。

> **仮説H7**: `KernelGenome` 拡張空間 (4 kernel × theta + gate 制約) は、3-param 単一 kernel 空間が
> 持たなかった **欺瞞的 corridor 構造 (kernel 障壁 / gate 穴)** を multi-task fitness 上に持ちうる。
> その帰結として MAP-Elites (kernel_id × theta behavior niching) が
> RR-hillclimb / panmictic-GA / random を ≥15 seed Wilcoxon p<0.05 + Cliff δ で上回る (③ load-bearing)。

### 6.3 検定法 (DECEPTIVENESS_MEASURE の失敗を回避する設計)

Step4 exp4-5 の MAP-Elites vs 3 baseline 比較を **`KernelGenome` 空間で再走**する。
ただし DECEPTIVENESS_MEASURE_VERDICT の §5「negative を license する 5 条件」を **全て事前登録**:

- (a) **記述子不変な behavior 定義**: behavior = `(kernel_id 連続値, theta L1 norm)` を **固定**。
  ビン数も合成・実タスクで同一に固定 (記述子依存 BG = VERIFY_descriptor_dependence の反転を回避)。
- (b) **固定 n_bins** を事前登録 (判定がビンで反転するため必須)。
- (c) **固定予算** n_evals を事前登録 (magnitude は予算敏感)。
- (d) **positive control**: §2.3 の hopfield kernel 障壁を **意図的に強調**した synthetic multi-task
  (一部 task は hopfield でしか高 fitness 出ない) を作り、MAP-Elites が ③ load-bearing になる
  ことを **先に確認** (メトリック validity 確立)。Step4 exp4 の deceptive corridor positive control
  と同じ役割。これが取れて初めて実 (memory_tasks) で測る意味が出る。
- (e) **negative control**: kernel 中立 task (どの kernel でも同等) で ③ が立たない (優位消失) ことを
  確認 (Step4 exp5 smooth 対照と同型)。

判定:
- **③ load-bearing (欺瞞地形あり)**: positive control で ③ 成立 **かつ** 実 multi-task でも
  MAP-E が 3 baseline 全勝 (p<0.05, δ 非無視) → 拡張空間は欺瞞地形を持つ (Step4 の予想「拡張で
  unlock」が支持される)。
- **③不要 (欺瞞地形なし)**: positive control で ③ 成立するが実 multi-task で優位消失 →
  「拡張しても memory_tasks 上は滑らか」= Step4 §7 の結論が拡張空間でも再現 (honest negative)。
- **N/A**: positive control すら ③ が立たない、または記述子/予算で判定反転 → 測定不能で保留
  (DECEPTIVENESS_MEASURE と同じ honest N/A)。

### 6.4 正規化交絡 ablation との接続

DECEPTIVENESS_MEASURE の循環論法 (severity HIGH) = 「behavior=mean を彫った合成で ρ=1 は定義のなぞり」。
本検定では behavior に **kernel_id (離散障壁) を入れる**ため、fitness landscape を behavior=mean で
彫る循環は構造的に起きにくい (kernel_id は fitness を直接定義しない選択変数)。
ただし **新たな交絡**: kernel_id behavior が fitness と相関しすぎると別の循環になる →
**交絡 ablation**: behavior から kernel_id を抜いた (theta-only behavior) MAP-E と比較し、kernel_id を
behavior に入れた効果が「kernel diversity 維持」由来か「単なる探索量」由来かを切り分ける
(Step4 C4 の init_batch ablation と同じ思想)。

honest 留保: full LLM でなく memory_tasks proxy。Step4 §7 留保「proxy が滑らかでも full LLM が
欺瞞的でない保証ではない」を継承。本検定の結論も proxy スコープに限定する。

---

## 7. リスク (§タスク 8)

1. **specialist が出ない (BG6 反証)** — memory_tasks が kernel 中立で、どの kernel でも同等性能なら
   命題3b は honest negative。緩和: 各 kernel が **構造的に得意な task** を設計に含める
   (hopfield→連想/DelayedRecall, mamba→selective/FlipFlop, linear_attn→長距離累積)。
   それでも出なければ「memory_tasks は土俵でない」と honest 報告。

2. **hopfield/linear_attn の対角 mock が full と乖離しすぎ** — 1-state 対角簡約は教科書 kernel の
   本質 (multi-head attention の二次相互作用 / 高次元 retrieval) を落とす。claim を
   "toy analogue / inspired by" に降格 (OTHERARCH 規律)。full 版は Stage 5+ と明示。

3. **Z3 nonlinear timeout (hopfield 双線形 J)** — 双線形/二次 real arithmetic で z3 が unknown を
   返す率が上がる。緩和: fail-closed reject (健全) + endpoint 閉形式上界をクロスチェック (BG2) +
   timeout_ms を kernel 別に調整 (hopfield のみ長め)。timeout 多発は honest disclosure に記録。

4. **欺瞞地形検定の循環論法再発 (DECEPTIVENESS_MEASURE と同じ罠)** — positive/negative control +
   事前登録 (n_bins/予算/記述子) で予防するが、それでも magnitude 転移が取れなければ N/A 保留。
   「整いすぎた ③ load-bearing 結果が出たら内訳を疑う」(honest_disclosure 規律) を適用。

5. **kernel_id の連続→離散 floor が GA 探索を歪める** — kernel_id を実数で持ち floor 離散化すると、
   境界 (k=0.99→1.0) で不連続。緩和: behavior 記述子の kernel_id は連続値のまま記録 (BG7 entropy は
   離散後で測るが、探索は連続で滑らか)。kernel_swap_mock (既存) のような明示離散 op も併用検討。

6. **state_norm/Lipschitz gate が「kernel 多様性」を選択前に殺す** — gate が hopfield/linear_attn を
   過剰 reject すると、進化集団が gate を通りやすい rwkv/mamba に collapse (BG7 反証)。これは
   **設計上の交絡であり同時に発見でもある** (§6.2 の "gate が欺瞞構造を inject"): gate を online で
   適用するか post-hoc で測るかで結論が変わる → 両条件 (gate-on / gate-off) で測り差分を honest 開示。

7. **既存資産改変の誘惑** — protocol.py S3 延期事項 (op_type kernel 別検証) や ChangeOp OP_TYPES 拡張を
   src でやりたくなるが、本 Stage は research 隔離厳守。BG5 (rwkv 埋め込み bit 一致) で src 不変を担保。

8. **honest disclosure 違反リスク** — 「mechanism feasible (§3a)」を「specialist 実証 (§3b)」や
   「欺瞞地形あり (§7)」と混同して報告するリスク。本書は全命題を別 BG で分離し、各々独立に
   pass/fail/N/A を出す構造にしてある。

---

## 8. 参照 (すべて読むだけ、本研究は research 隔離 / src 非変更)

- `src/llcore/kernel/protocol.py` — Kernel/GeneCodec/Trajectory/VerifierBackend Protocol (S3 延期事項)
- `src/llcore/kernel/rwkv.py` — RWKV 準拠例 (本流委譲 wrapper, 後方互換の核)
- `src/llcore/state_update/genes.py` — StateUpdateGene (3-param, kernel_id=0 埋め込み先)
- `src/llcore/verifier/invariants.py` — state_norm (`verify_gene_safe`) + Lipschitz
  (`verify_lipschitz_contraction` / `_lipschitz_upper_bound` / `empirical_lipschitz`)
- `src/llcore/verifier/changeop.py` — ChangeOp/OP_TYPES (kernel_id_shift 拡張先, src 不変)
- `research/step4_selection/selection_lab.py` — MAP-Elites + 3 baseline + compare (③ 検定流用)
- `research/ea_multitask/{ea_lab.py, task_mixture.py}` — multi-task train/test 分離 + ③ ablation
- `research/step_c_memory_tasks/memory_tasks.py` — FlipFlop/DelayedRecall/DelayedParity (regime 源)
- `docs/poc/STEP4_SELECTION_VERDICT.md` — ③ 欺瞞 corridor 限定 + 実 proxy 滑らか (本設計の出発点)
- `research/step_c_deceptiveness_measure/DECEPTIVENESS_MEASURE_VERDICT.md` — 欺瞞 magnitude 測定の
  失敗教訓 + negative license 5 条件 (§6 検定設計の制約源)
- `research/other_archs/OTHERARCH_VERDICT.md` — kernel plugin 化 4 条件 + "partial stack reuse" 正名化
- `research/other_archs/snn/snn_gene.py` — 非 RWKV kernel gene の符号化先例 (clipped/as_array pattern)
