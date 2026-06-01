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

## 次の実装ステップ候補 (本 scout の含意)
1. mamba/linear_attn の Lipschitz encoding を achievable 上界の定数注入に精緻化 (over-approx 緩和)。
2. BG4 (kernel_swap random walk finite) smoke を追加。
3. `KernelGenome` codec + decode を実装し ea_lab/selection_lab に dim=5 で接続 (Stage 3b 着手)。
4. memory_tasks multi-task で BG6 (specialist 写像) を測る = 命題3b の最初の falsification。
