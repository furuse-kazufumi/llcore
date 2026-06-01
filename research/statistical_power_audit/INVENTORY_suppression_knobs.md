# 統計的検出力 自己監査 — 容疑 knob インベントリ (read-only)

> 2026-06-01。research/statistical_power_audit/ 隔離。src 非変更・git 非実行。
> 目的: 「llcore ③ 研究の厳格な統計ゲートが **偽陰性 (Type II error) に偏り、本当はある進化を検出できていない** 可能性」を自己監査するための、容疑 knob の正確な所在・偽陰性発生機序・toggle 方法・既知データの inventory。
> honest disclosure: これは「緩めれば進化が見える」を主張する文書ではない。緩和が **偽陽性 (Type I)** をどれだけ増やすかを併記する前提の materials 集めである。

---

## 0. falsifiable 命題 (本監査が検証する対象)

- **H_audit (本命題)**: 現行の strict gate (n≥15 ∧ 片側 Wilcoxon p<0.05 ∧ |paired_sign_delta|≥0.147 ∧ clip=True) は、real な中効果 (|δ|≈0.2–0.5) の③寄与を p<0.05 に解像できず、honest-negative の一部は「③不在の証拠」でなく「underpowered=inconclusive」である。
- **破綻ゲート (H_audit が偽になる条件)**: 既知真陽性 (合成欺瞞回廊 d≥0.16) で gate が一貫して PASS し、かつ既知 negative を緩めても **偽陽性が爆発しない** (= ground-truth smooth control d=0 で依然 FAIL を保つ) なら、gate は「厳しすぎ」でなく「適切に校正されている」= H_audit 棄却側。
- 計測すべき2量: (1) gate 緩和で既知 negative が PASS に転じる閾値 (Type II 改善)、(2) 同緩和を **smooth control (d=0)** に当てたときの偽陽性率 (Type I 悪化)。両者のトレードオフを必ず併記。

---

## 1. 容疑 knob K1–K4 (file:symbol・偽陰性機序・toggle)

### K1 — honest_eval の fresh-seed 再評価 + 完全 strict gate
- **file**: `src/llcore/evolution/honest_eval.py`
- **symbol**:
  - `honest_reevaluate(eval_once, gene, *, n_trials, rng)` (L87) — 進化と独立な fresh rng で n_trials 回平均。
  - `evolution_vs_random(...)` (L173) の `passes` 判定 (L271–276) と既定パラメータ `n_seeds=15, honest_n_trials=30, alpha=0.05, min_seeds=15, min_effect=0.147` (L184–188)。
  - `_paired_p` (L155, 片側 Wilcoxon、scipy 不在時 `_sign_test_p_greater` L139)。
  - `_paired_sign_delta` (L119, `(#正−#負)/n`、`_cliff_delta` は alias L136)。
- **偽陰性 (Type II) をどう生むか**:
  1. **fresh-seed 再評価**: elitism の凍結持越し (前世代 noisy fitness の保持) を排除し +0.29 の水増しを消す = 偽陽性防止が主目的。だが副作用として、進化中に当たった「運の良い高評価個体」を独立 seed で測り直すと regression-to-mean で平均が下がり、**本物の小さな改善も過小評価**しうる。n_trials=30 平均は分散を下げるが、real effect が再評価ノイズと同オーダだと検出力を食う。
  2. **min_effect=0.147 床**: paired_sign_delta (符号バランス効果量) に Cliff δ small-effect 境界を流用。docstring 自身が「教科書的 Cliff's delta ではない」と明記。`(#正−#負)/n` なので n=15 では刻みが粗く (1 seed=0.133)、real な medium effect でも符号がばらつくと床割れ → FAIL。
  3. **片側 Wilcoxon + n=15**: 片側で過剰な厳しさは避けているが n=15 は medium effect を p<0.05 に乗せにくい (flip_flop 実例参照)。
  4. **4 条件 AND**: `diff>0 ∧ p<alpha ∧ n≥min_seeds ∧ |δ|≥min_effect` の連言。どれか1つでも落ちれば FAIL。連言は偽陽性を抑えるが、各条件が独立に検出力を削るので合成検出力は最も弱い条件に律速。
- **toggle (src 非変更で)**: `evolution_vs_random` / 校正スクリプトを research 側から呼ぶ際に keyword 引数で `min_effect`, `min_seeds`, `n_seeds`, `alpha`, `honest_n_trials` を振る。`passes` を再計算したいだけなら返り値の `diff/wilcoxon_p/paired_sign_delta/n_seeds` を取り出して research 側で別閾値の AND を組む (関数本体は触らない)。

### K2 — strict_compare の閾値 (memory tasks 用 strict 版)
- **file**: `research/step_c_memory_tasks/strict_compare.py`
- **symbol**: `strict_compare(scores_a, scores_b, name_a, name_b, *, alpha=0.05, min_seeds=15, min_effect=0.147)` (L34)。`passes = diff>0 ∧ _paired_p<alpha ∧ len≥min_seeds ∧ |_paired_sign_delta|≥min_effect` (L50)。honest_eval の `_paired_p`/`_paired_sign_delta` を import 流用 (L17)。
- **偽陰性機序**: K1 と同一の 4 条件 AND を 2 スコア配列に適用する薄いラッパ。0.147 床が「中効果切り捨て」の主犯。E-A exp_ea3 の C-gen4b (MAP-E vs random: diff=+0.0625, p=0.126, δ=+0.20) はこのゲートで FAIL。δ=+0.20 は 0.147 を超えるが p=0.126 で落ちる = **効果量床でなく p で落ちる**ケース。逆に flip_flop (δ=+0.33, p=0.15) も p で落ちる。
- **toggle**: research 側呼出で `alpha`, `min_seeds`, `min_effect` を keyword で振る。ファイル無改変。

### K3 — ea_lab の equal-budget / global best-of-budget / ③ablation
- **file**: `research/ea_multitask/ea_lab.py`
- **symbol**:
  - `map_elites_full` (L42, selection_mode='elite' ①②③) / `map_elites_randselect` (L67, selection_mode='random' ②③殺し①のみ = ③ablation の中核)。
  - `_map_elites_core` (L94) の **global best-of-budget** ロジック (`_track` L136, gbest_gene/gbest_f L133)。Codex F2 で「archive-max 占有者」から「予算内全評価個体の max」へ変更 (L129–132 コメント)。
  - `run_ea_methods_over_seeds` (L185)。`honest_n_trials` を要求、CRN seed 設計 (`_evo_rng` L224, `_honest_both` L227)。
- **偽陰性機序**:
  1. **global best-of-budget 化 (Codex F2)**: randselect は無条件上書きで強個体を忘れるため、archive-max だと random (全 n_evals から best) に不当に不利 → C-gen3 の③有利 gap を水増ししていた。これを公平化した = **③検出力をわざと削る方向の修正** (偽陽性除去だが、もし archive-ratchet 自体が③の効果なら、その効果を「読み出し方」で消している可能性)。verdict 注記: 「randselect を強くした (0.536→0.557) のに C-gen3 はまだ PASS」= この修正は C-gen3 を殺してはいない。
  2. **equal-budget**: 全 method 同一 n_evals。MAP-E の archive 維持コスト (評価予算外の bookkeeping) は budget に入らないが、stepping-stone を「同じ評価回数」で比べるので、③が予算効率で勝つ余地を測れる設計。検出力には中立〜やや保守。
  3. **HONEST_N=30** (exp_ea3 の定数、ea_lab は引数受け): honest_eval §5 準拠。fresh-seed 再評価ノイズを平均化 = K1 同様、real な小改善を平均化で薄める副作用。
- **toggle**:
  - ③ablation 強度: `map_elites_full` vs `map_elites_randselect` を runner に渡し分け (research 側)。`_map_elites_core(..., selection_mode=...)` の 'elite'/'random' が ON/OFF。
  - global-best vs archive-max を比較したい場合は `_map_elites_core(..., archive_out=<dict>)` を渡すと最終 archive 占有者を取り出せる (L107 probe) → research 側で archive-max を再計算し global-best と対比 (production 不変)。
  - `n_evals`, `n_seeds`, `honest_n_trials`, `sigma`, `base_seed` は `run_ea_methods_over_seeds` の keyword。

### K4 — ridge_readout の clip=True (飽和で信号消失)
- **file**: `src/llcore/fitness/ridge_readout.py`
- **symbol**: `ridge_fitness(gene, task, *, n_train=64, n_eval=64, ridge_lambda=1e-2, rng, clip=True)` (L161)、clip 適用は L211 `np.clip(r2, 0.0, 1.0) if clip else r2`。`make_ridge_eval_once(..., )` (L214) は `clip` を露出せず常に既定 (=True) で `ridge_fitness` を呼ぶ (L235–239)。
- **偽陰性機序**: held-out R² が負 (mean 予測以下) を [0,1] に clip。**clip 後の 0.0 は raw R²<0 を潰した値で「raw=0 の信号皆無」と識別不能** (Codex pair-review High finding, docstring L31–33/L187–190 明記)。選択圧として使うと、平坦化した floor 0.0 域では gene 間の fitness 差 (spread) が消え、進化が登るべき勾配が無くなる → landscape が人工的に平坦化し「③不要/滑らか」に見える。raw R² の符号・spread を見れば構造があるのに、clip でそれを隠している可能性。
- **toggle**: `ridge_fitness(..., clip=False)` で raw R² を露出 (research 側で eval_once を自作する)。`make_ridge_eval_once` は clip を露出しないため、診断時は `ridge_fitness` を直接 wrap する eval_once を research 側で書く (src 無改変)。clip=False で spread/符号を測り、clip=True と同 gene 集団で fitness 分散を比較 → 「平坦化が clip 由来か landscape 由来か」を切り分け。

---

## 2. 既知 真陽性 (ground truth) — 統計が「進化あり」を出すべきケース

- **canonical 存在証明 (exp4)**:
  - **entrypoint**: `research/step4_selection/exp4_genotypic_corridor.py::main` (eval=`corridor_eval` L48, behavior=`behavior_mean` L43, D=24, n_evals=6000, n_seeds=20, honest_n_trials=30, sigma=0.10)。判定は `selection_lab.compare` (非 strict: passes=diff>0 ∧ p<alpha のみ)。
  - **効果量**: MAP-E が 3 baseline 全勝、p=1.9e-6, Cliff δ=+1.00, 勝率 100%。大域到達 95% vs baseline 0% (`STEP4_SELECTION_VERDICT.md` L16–17/28–29)。robustness: base_seed {20260530,777,31337} 計 60 seed で reach≥0.95 (L32–33)。
- **連続 knob 版 (推奨される primary ground-truth = 効果量を振れる)**:
  - **entrypoint**: `research/step_c_applicability/exp_knob_sweep.py::run_sweep` (+ `main`)。eval=`make_corridor_eval(d)` (L130) で **dip depth d ∈[0,1] を引数化** = 欺瞞性 (dip 深さ) を連続変化。strict gate = `strict_gate` (L176, K1/K2 と同一 4 条件)。
  - **効果サイズを振る引数**: `make_corridor_eval(d)` の **d** (dip depth)。d=0=smooth control (③不要), d=1=deep dip (deceptive)。d* (③が3 baseline 全勝になる最小 d) = **0.16** (base_seed {20260530,777,31337} 一致, `exp_knob_sweep_results.json` `threshold_d_star_strict`)。loose 閾値 (climbing baseline 初勝利) = 0.13。
  - **実測効果量 (results JSON)**: d=0.16 で MAP-E vs panmictic diff=+0.283, p=4.8e-6, δ=+0.8 / vs random diff=+0.221, p=9.5e-7, δ=+1.0 (PASS)。d=0.13 で vs RR diff=+0.109, p=0.027, δ=+0.2 (PASS) だが vs panmictic δ=−0.4 で全勝に届かず load_bearing=false。d=0.0 で MAP-E vs RR diff=−0.0023, p=0.97, δ=−0.5 (FAIL=正しく smooth で③不要)。
  - **校正に最適な理由**: 同一 corridor 構造のまま d だけ動かし、gate が真陽性 (d≥0.16) を捉え偽陽性 (d=0) を出さないかを **連続軸**で測れる。Type I/Type II トレードオフ計測の主装置。

---

## 3. ③ablation の entrypoint (進化の③要素を on/off する実験本体)

- **`research/ea_multitask/exp_ea3_ablation.py::main`** — C-gen3 (MAP-E full vs MAP-E_randselect=②③殺し①のみ) + C-gen4 (vs panmictic / vs random)。多タスク hold-out 汎化。strict_compare で判定。最も clean な③分離 (randselect が ablation 本体)。
- **`research/ea_multitask/ea_lab.py::map_elites_randselect`** — ③(fitness ゲート placement) と ②(niche elite 親) を殺し ①変異のみ残す ablation 関数本体 (selection_mode='random')。`map_elites_full` が対照 (①②③)。
- **`research/step_c_applicability/exp_knob_sweep.py::run_methods_crn`** — MAP-E vs {rr_hillclimb, panmictic_ga, random} を CRN で比較 (③の有無を baseline 群で挟む ablation)。
- **`research/step4_selection/exp4_genotypic_corridor.py::main` / `exp5_boundary_control.py`** — ③成立 (exp4) / smooth で消失 (exp5) の境界 ablation。
- **`research/step6_real_proxy/exp7_method_comparison.py::main`** — 実 ESN×テキスト proxy で③ablation (negative)。
- **`research/step_c_memory_tasks/exp_c4_ablation.py::main`** — MAP-E 勝因が coverage か ratchet かの init_batch ablation (MAP-E 非勝利のため moot)。

---

## 4. 既知 real negative の効果量・p・n 実値 (verdict / results JSON 由来)

| case | 比較 | diff (delta) | p (片側) | δ (paired_sign) | n | gate | 出典 |
|---|---|---|---|---|---|---|---|
| **flip_flop** (Step C) | MAP-E vs random | +0.0041 | 0.15 | **+0.33** | 15 | FAIL (p) | STEP_C_VERDICT §3.2 (`exp_c2c3_results.json`) |
| flip_flop | MAP-E vs RR-hillclimb | +0.0036 | 0.26 | +0.07 | 15 | FAIL | 同上 |
| flip_flop | MAP-E vs panmictic-GA | −0.0041 | 0.97 | −0.20 | 15 | FAIL (diff<0) | 同上 |
| delayed_parity | MAP-E vs random | −0.0008 | 0.85 | −0.20 | 15 | FAIL (床, R²≈0.003) | 同上 §3.2 |
| delayed_parity | MAP-E vs RR | −0.0007 | 0.83 | −0.20 | 15 | FAIL (床) | 同上 |
| delayed_parity | MAP-E vs panmictic | −0.0002 | 0.51 | +0.07 | 15 | FAIL (床) | 同上 |
| **E-A C-gen3** | MAP-E vs randselect | +0.126 | **0.0151** | +0.60 | 15 | **PASS** | E_A_VERDICT (`exp_ea3_results.json`) ※negativeではないが対照 |
| **E-A C-gen4a** | MAP-E vs panmictic | **−0.0193** | 0.598 | −0.067 | 15 | FAIL (diff<0) | 同上 |
| **E-A C-gen4b** | MAP-E vs random | +0.0625 | 0.126 | +0.20 | 15 | FAIL (p) | 同上 |
| **step6 exp7 (A)** 3-param ESN | MAP-E vs RR | −0.0016 | 0.078 | (記載なし) | 8 | FAIL (smooth) | STEP4_VERDICT §7 L114 |
| step6 exp7 (B) per-neuron 40dim | MAP-E vs RR/panmictic | 負け / random のみ勝 | — | — | 6 | FAIL (smooth) | 同上 L115–116 |

**underpowered の自認 (verdict 一次情報)**:
- flip_flop MAP-E vs random: δ=+0.33 (medium, 0.147 床超) なのに p=0.15。verdict 自身が「**non-significant かつ underpowered = inconclusive (保留)**、null の受容ではない」と明記 (STEP_C_VERDICT L6/L83/L153)。**n=15 で medium effect が p<0.05 に届かないのは典型** = K1/K2 の n=15 制約が偽陰性側に効いている直接証拠。
- E-A C-gen4b: δ=+0.20 (床超) / p=0.126 = やはり p で落ちる underpowered 候補。
- step6 (A): p=0.078 (n=8) = n が更に小さく検出力不足が顕著。

**注意 (Type I 側、honest)**:
- 上記の「p で惜しく落ちる」群は δ も小さい (0.20–0.33)。緩和で PASS させると **smooth control (exp_knob_sweep d=0)** での偽陽性 (d=0 で MAP-E が RR に δ=−0.5/p=0.97 と明確に負け) との分離が崩れないかを §0 破綻ゲートで必ず検証する。緩和が flip_flop を PASS にしつつ d=0 を FAIL に保てなければ「厳しすぎ」でなく「校正が正しい」= H_audit 棄却。

---

## 5. 次の計測 (results JSON で残すべき生数値)

1. **校正 (calibration)**: `exp_knob_sweep::run_sweep` を d を密に振り (特に d∈[0.10,0.20])、各 gate パラメータ (`min_effect`∈{0.0,0.05,0.147,0.3}, `alpha`∈{0.05,0.1}, `n_seeds`∈{15,30,60}) で d* がどう動くかを `calibration_dstar_vs_gate.json` に。
2. **逆算 (power back-calc)**: flip_flop / C-gen4b / step6(A) の観測 δ・分散から、p<0.05 到達に必要な n を逆算 (scipy あれば Wilcoxon、無ければ符号検定の二項) → `power_backcalc.json`。
3. **ablation (clip)**: K4 を `clip=True` vs `clip=False` で同 gene 集団の fitness spread/符号を測り、平坦化の clip 寄与を `clip_ablation_spread.json` に。
4. **Type I/II トレードオフ**: 各緩和で (既知 negative の PASS 化数) と (smooth control d=0 の偽陽性率) を対で `type1_type2_tradeoff.json` に。**負の結果 (= gate は健全=進化は本当に無い) も valid として残す**。

すべて `research/statistical_power_audit/` に JSON 永続化。git は orchestrator 一括 commit。
