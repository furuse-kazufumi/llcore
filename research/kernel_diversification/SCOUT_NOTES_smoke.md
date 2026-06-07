# Scout Notes — kernel 多様化 mechanism feasibility smoke (2026-06-01)

`smoke_kernel_gates.py` を実行した一次結果 (`smoke_kernel_gates_results.json`) の honest 解釈。
DESIGN_kernel_diversification_3b.md の §2-§5 + BREAK_GATES.md BG1/BG2/BG3/BG5 を小サンプル
(各 kernel 400 gene, z3 available=True) で検証した feasibility smoke。**完全実証ではない**。

## 数値サマリ (N=400 gene/kernel, state_bound=1, |x|<=1)

| kernel | BG1 admit率 | BG2 一致 | BG3 違反 | L_ub 範囲 | 閉形式 contract率 |
|---|---|---|---|---|---|
| rwkv | 1.000 | 1.0 | 0 | 0.030..1.925 | 0.718 |
| mamba_selective | 1.000 | 1.0 | 0 | 1.000..1.000 | **0.000** |
| hopfield_dense | 1.000 | 1.0 | 0 | 0.052..2.470 | 0.797 |
| linear_attn | **0.420** | 1.0 | 0 | 0.003..1.000 | 1.000 |
| BG5 後方互換 | — | — | — | — | bit 一致 pass=True |

## 設計仮説の検証 (何が支持され、何が要修正か)

### 支持された (DESIGN 通り)
- **BG1 非空 admit (命題3a 前半)**: 4 kernel すべて admit 集合が非空 → state_norm gate を
  通せる gene が存在。**mechanism feasible**。
- **linear_attn の state_norm 過剰 reject (DESIGN §2.4 予測)**: admit=0.420 (232/400 reject)。
  softplus 非有界 → `lam + softplus(|w|)·|v_gain| ≤ 1` 制約を満たさない gene が gate で落ちる。
  **「kernel 間で gate 透過性が非対称」という §2.5 / §6.2 の核心仮説を数値で裏付け** (linear_attn が
  最も通りにくい)。これは bug でなく設計の load-bearing 特徴。
- **BG2 verdict 一致 = 1.0**: Z3 unsat ⟺ 閉形式 L_ub<1 が全 kernel で完全一致 → Z3 encoding 健全。
- **BG3 over-approx sound = 違反 0**: empirical_L ≤ 閉形式 L_ub が全 kernel で成立。
- **BG5 後方互換 bit 一致**: kernel_id=0 経路 == 既存 run_sequence → src 不変・後方互換を機械担保。
- **hopfield の multi-basin (DESIGN §2.3 予測)**: L_ub が最大 2.470 まで伸び、contract率 0.797
  (約 20% が L≥1) → hopfield は Lipschitz gate で一部 reject される領域を持つ (予測通り)。

### 要修正 / honest 留保 (smoke が露呈させた問題)
- **mamba contraction率 0.000 は over-approx の保守性 artifact (真の negative ではない)**:
  閉形式 `mamba_L_ub` が free `a∈[0,1]` の端点 a=1 で常に 1.0 を返すため `L_ub<1` が一度も真に
  ならない。だが **真の `J=a=sigmoid(.)∈(0,1)` は常に strict contraction**。
  → 開区間 (0,1) を端点込み [0,1] に over-approx した代償。**Lipschitz gate としては「mamba は
  certified contraction を出せない (端点 1 で reject 寄り)」になり、これは fail-closed 規律上は
  健全だが、設計意図 (mamba=無条件 contraction) と乖離**。
  本実験での教訓 = DESIGN §2.2 の「mamba は無条件 L<1」claim は **strict 開区間を扱える
  encoding (a の真の上界 sigmoid(|alpha|+|beta|)<1 を定数注入) が必要**。本 smoke の素朴
  over-approx では出ない。→ DESIGN §5 の mamba 拡張に「a の achievable 上界を定数化」を追記すべき
  (現状 free [0,1] は緩すぎる)。**消さずに記録**: 整いすぎた「mamba 無条件 contraction」を
  smoke が反証した = honest disclosure の威力。
- **timeout 0**: hopfield 双線形 J でも z3 timeout 0 (N=400, 1000ms)。DESIGN §7 リスク3 の
  timeout 懸念は本スケールでは顕在化せず (より大 N / tighter bound で再評価要)。

## BG 判定 (smoke スコープ)
- BG1 pass (全 kernel admit>0) / BG2 pass (一致 1.0) / BG3 pass (違反 0) / BG5 pass (bit 一致)。
- **Stage 3a mechanism feasibility = smoke レベルで PASS。** ただし mamba Lipschitz encoding の
  保守性が DESIGN claim と乖離 → 要 encoding 精緻化 (上記)。
- BG4 (swap 健全) / BG6-9 (specialist / 欺瞞地形) は **本 smoke 対象外** (別実験、未着手)。

## 追記 (2026-06-01): 最小 skeleton + 動的 smoke 着地

`kernels.py` (KernelGenome union genome + 4 kernel forward dynamics, 対角 mock,
rwkv は既存 run_sequence 再利用) と `smoke_kernels.py` (有界入力動的 smoke) を追加。
`smoke_results.json` に数値接地。**src 非改変** (BG5 bit 一致で機械担保)。

### 動的 smoke 結果 (N_GENE=32/kernel, L=64, dim=8, |x|<=1, seed=20260601)

| kernel | finite_state | state_norm_ok | max_state_norm | 経路 |
|---|---|---|---|---|
| rwkv | True | True | 2.651 | 既存 run_sequence |
| mamba_selective | True | True | 1.781 | research mock |
| hopfield_dense | True | True | 2.264 | research mock |
| linear_attn | True | True | **21.817** | research mock |
| BG5 後方互換 | — | — | bit 一致 | — |

- **all_ok=True** (全 kernel finite + state_norm 緩い上界 K=√8·20≈56.6 以下 + BG5 bit 一致)。
- **honest 留保 (linear_attn)**: un-gated 経路で max_norm=21.8 と唯一 1 を大きく超える。
  これは `s'=lam*s + softplus(w*x)*v_gain*x` の softplus*x 寄与が 1 step で >1 を作るため
  (gate 検証 BG1 で admit=0.420 = 最も reject される、という gate smoke の結果と整合)。
  発散はしない (lam<1 減衰で有界化) が、「ungated だと state が 1 を大きく超え得る」=
  state_norm gate が load-bearing であることの動的裏付け。緩い K では pass だが、
  STATE_BOUND=1 の厳格 gate (smoke_kernel_gates.py BG1) では reject 寄りになる二面性を記録。
- BG4 (kernel 軌跡 finite) の最小版に相当 = 上記「次ステップ候補 2」を消化。

## 次の実装ステップ候補 (本 scout の含意)
1. mamba/linear_attn の Lipschitz encoding を achievable 上界の定数注入に精緻化 (over-approx 緩和)。
2. ~~BG4 (kernel_swap random walk finite) smoke を追加。~~ → smoke_kernels.py で finite 確認済 (2026-06-01)。
   残: kernel_id_shift mutation を絡めた random walk 版 (op 適用後 decode 健全)。
3. `KernelGenome` codec + decode を実装し ea_lab/selection_lab に dim=5 で接続 (Stage 3b 着手)。
   → kernels.py に KernelGenome.as_array/from_array (dim=5) 着地済。残: ea_lab 接続。
4. memory_tasks multi-task で BG6 (specialist 写像) を測る = 命題3b の最初の falsification。
