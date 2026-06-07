<!-- SPDX-License-Identifier: Apache-2.0 -->
# 正規化交絡 検定 最終判定 (NORMALIZATION_CONFOUND_VERDICT)

- 対象: `variable_delay_recall` (VDR) の「fitness 地形が滑らか / 多様性維持 ③ 不要 / 進化しない」honest-negative
- 検定軸: 正規化 (fitness clip / 状態 scale / 記述子 binning) が地形の多峰構造を消した「正規化人工物」か否か
- 主診断: C1 midpoint-valley test (`research/step_c_memory_tasks/landscape_map.py:multimodality_report`, read-only 流用)
- 構造規律: `src/` 完全非改変 (`git status --porcelain src/` = 空、`git diff --stat src/` = 空)、git 操作なし、UTF-8、`py -3.11`、read-only import のみ、全成果 `research/normalization_confound/` 隔離
- 日付: 2026-06-01

---

## 1. 命題と非循環診断

### 1.1 falsifiable 命題 P

> **P**: 「VDR (E-A 勝者分布, D∈{15,30,45,60}, distractor_amp=0.2, leaky-delay-line reservoir 基質) の fitness 地形が滑らか (= C1 で連結 manifold = 単峰 = `is_multimodal=False` = ③不要)」という現行 honest-negative は、fitness/記述子/状態スケールの正規化が地形の多峰構造を消した『正規化人工物』である。

**検定形 (否定形)**: 全正規化条件 (clip on/off/soft × sigma{0.10,0.15,0.20} × bounds{現行,拡大}) で C1 `valley_fraction` が `is_multimodal` 閾値 0.2 と baseline (clip=hard) 比を跨いで**反転しなければ P を棄却** (= robust に滑らか)。**反転すれば P を採択** (= 人工物確定) し③ablation escalation へ。

### 1.2 非循環診断の論証 (G4 ゲート 充足)

谷深さ verdict (`DECEPTIVENESS_MEASURE_VERDICT.md` / `VERIFY_circular_reasoning.md`, severity HIGH) が摘発した循環 = 「合成 `corridor_eval = f(behavior=mean(g))` でメトリックが behavior=mean 軸の d を定義上なぞる」を C1 は構造的に回避する。実コードで確認した非循環性:

1. **主診断シグネチャに method 比較が物理的に入らない**:
   `multimodality_report(eval_once, *, dim, bounds, n_restarts, n_evals, sigma, base_seed)` — `inspect.signature` で確認。method / passes / diff / behavior / win の引数は **ゼロ**。`valley_fraction` は raw 24 次元 gene 空間の random-restart 山登り収束 optima とその中点 fitness だけから算出され、behavior 軸射影も grid binning も使わない。
2. **fitness は reservoir 物理 (held-out R²) であり mean(gene) の関数ではない**:
   `c1_clip_eval.py:make_eval_once_clipswitch` L75-87 は `res.run → fit_ridge_readout → MSE/var → R²`。これは合成 corridor (corr +0.96 = 定義上のなぞり) とは別物。
3. **全 4 harness が ③ablation を import すらしていない**:
   `ea_lab` / `run_ea_methods_over_seeds` / `map_elites` / `make_behavior` / `strict_compare` の import ゼロ、`exp_ea3_results.json` の read ゼロ。③/panmictic の出現は「参照しない」と明記した docstring 内のみ。
4. **clip ablation は出力変換であって behavior 軸ではない**:
   clip=False で谷が出るなら「出力飽和が谷を床に潰していた」という純機構的因果。③の定義 (選択圧/分離) に依存しない。
5. **判定式が exp_ea3 を参照しない**:
   verdict logic (`c1_normalization_sweep.py` L295-324) は `valley_fraction` / `noisy_flat_null_vf` / `noise_dominated` / `is_multimodal_unanimous` のみ参照。exp_ea3 の diff/p/passes を一切含まない。

**結論**: 主診断・副次診断・人工物判定の全てが③の成否から独立。VERIFY の `circularity` lens は **refuted=false, severity=none** (循環なし)。

---

## 2. 正規化 knob × 設定 sweep 結果表

### 2.1 主 sweep (12 設定, 3 seed, quick mode: n_restarts=6, n_evals=100, n_train/eval=40)

`c1_sweep_results_quick.json` (正本)。total wall-clock = 1851.96s (30.9 分)。**any_flip = False**。

| # | 設定 | knob | verdict | valley_mean | noisy-flat null | eval_noise_std | valley_thr | R² range / spread |
|---|------|------|---------|-------------|-----------------|----------------|-----------|-------------------|
| 1 | vdr_D60 clip=hard sigma=0.15 current | **BASELINE** | noise_confounded | 0.889 | 1.000 | 0.157 | 0.0142 | [0.408, 0.829] / 0.4213 |
| 2 | vdr_D60 clip=none sigma=0.15 current | A: raw R² | noise_confounded | 0.867 | 1.000 | 0.206 | 0.0130 | [0.408, 0.829] / 0.4213 |
| 3 | vdr_D60 clip=soft sigma=0.15 current | A: tanh | noise_confounded | 0.867 | 1.000 | 0.194 | 0.0123 | [0.387, 0.680] / 0.2933 |
| 4 | vdr_D60 clip=hard sigma=0.10 current | C: sigma | noise_confounded | 0.889 | 1.000 | 0.157 | 0.0142 | [0.404, 0.839] / 0.4349 |
| 5 | vdr_D60 clip=hard sigma=0.20 current | C: sigma | noise_confounded | 0.933 | 1.000 | 0.157 | 0.0142 | [0.493, 0.826] / 0.3331 |
| 6 | vdr_D60 clip=hard sigma=0.15 wide | C: leak_raw[-6,6] | noise_confounded | 0.911 | 1.000 | 0.154 | 0.0187 | [0.459, 0.656] / 0.1973 |
| 7 | vdr_D15 clip=hard sigma=0.15 current | **BASELINE (対照)** | noise_confounded | 0.800 | 1.000 | 0.062 | 0.0388 | [0.916, 0.982] / 0.0661 |
| 8 | vdr_D15 clip=none sigma=0.15 current | A: raw R² | noise_confounded | 0.800 | 1.000 | 0.062 | 0.0388 | [0.916, 0.982] / 0.0661 |
| 9 | vdr_D15 clip=soft sigma=0.15 current | A: tanh | noise_confounded | 0.644 | 1.000 | 0.036 | 0.0324 | [0.724, 0.754] / 0.0300 |
| 10 | vdr_D15 clip=hard sigma=0.10 current | C: sigma | noise_confounded | 0.644 | 1.000 | 0.062 | 0.0388 | [0.873, 0.982] / 0.1087 |
| 11 | vdr_D15 clip=hard sigma=0.20 current | C: sigma | noise_confounded | 0.778 | 1.000 | 0.062 | 0.0388 | [0.936, 0.978] / 0.0417 |
| 12 | vdr_D15 clip=hard sigma=0.15 wide | C: leak_raw[-6,6] | noise_confounded | 0.622 | 1.000 | 0.065 | 0.0390 | [0.859, 0.978] / 0.1196 |

**全 12 設定が単一 verdict = `noise_confounded`**。baseline (clip=hard) からどの正規化を外しても `deceptive` へ反転しない。

### 2.2 診断器健全性 control (`budget_sensitivity_check.py`)

| control | verdict | valley_fraction | 解釈 |
|---------|---------|-----------------|------|
| noiseless 単峰 (二次関数, ノイズなし) | **smooth (true-negative)** | 0.000 | C1 はノイズなし単峰を正しく smooth と判定 → 診断器は健全 |
| noisy-flat (平坦 + 同 eval noise) | **deceptive (FALSE-positive)** | 1.000 | C1 は完全に平坦なノイズ地形を**最大 deceptive と誤判定** |

**これが本研究の決定的所見**: VDR で観測される valley≈0.8-0.9 は noisy-flat null (vf=1.0) と統計的に区別不能。eval ノイズ std (0.036-0.206) >> 谷閾 0.05·|fit| (0.012-0.039)。

### 2.3 noise-robust C1 escalation (頑健性補強, vdr_D15, 8/9 点完走)

`noise_robust_run.log` (正本; JSON は smoke 残骸)。設計: optima endpoint と midpoint を同一 n_avg seed で **CRN-paired 平均**評価し、谷閾を `max(0.05·|fit|, 2·SEM)` に SEM-scale。**反証予測**= 真の幾何多峰なら averaging 後も valley が noisy-flat null より高い plateau に残る。

| 設定 | n_avg | valley_mean | null_vf | margin | SEM | R² range | verdict |
|------|-------|-------------|---------|--------|-----|----------|---------|
| clip=hard current | 1 | 0.333 | 0.444 | **-0.111** | 0.061 | [0.881,0.971] | noise_confounded |
| clip=hard current | 4 | 0.611 | 0.389 | +0.222† | 0.030 | [0.879,0.969] | noise_confounded |
| clip=hard current | 16 | 0.667 | 0.889 | **-0.222** | 0.015 | [0.895,0.968] | noise_confounded |
| clip=none current | 1 | 0.333 | 0.444 | -0.111 | 0.061 | [0.881,0.971] | noise_confounded |
| clip=none current | 4 | 0.611 | 0.389 | +0.222† | 0.030 | [0.879,0.969] | noise_confounded |
| clip=none current | 16 | 0.667 | 0.889 | -0.222 | 0.015 | [0.895,0.968] | noise_confounded |
| clip=hard wide | 1 | 0.444 | 0.444 | +0.000 | 0.066 | [0.888,0.971] | noise_confounded |
| clip=hard wide | 4 | 0.667 | 0.389 | +0.278† | 0.033 | [0.867,0.961] | noise_confounded |

† n_avg=4 の正 margin は `mean_pairs < 8` (n_restarts=4 → ~6 pairs) の low-stats gate で `geometric_valley` 不採用 = 低ペア量子化のノイズ。`has_geometric_valley = False`。

**noise を std/4 (SEM 0.061→0.015) まで averaging しても valley は noisy-flat null を robust に超えず**、むしろ n_avg 増で null 自体が valley を追い越す (n_avg=16: valley 0.667 < null 0.889)。geometric valley plateau は不在。clip=none は clip=hard と数値完全一致 (交絡 A 非 load-bearing を再確認)。

**any_flip = False (全 sweep 通算)。flip_settings = []。**

---

## 3. 最終判定

### (c) なお不確定 — C1 は本 stochastic ridge fitness に対し「計測不能」(instrument 不能)

「滑らか / ③不要 / 進化しない」が **(a) 正規化人工物** か **(b) 本物の地形性質** かは、**C1 midpoint-valley test では支持も否定もできない**。理由は循環でも正規化でもなく、**診断器の谷閾 (0.05·|fit| ≈ 0.012-0.039) が fitness 評価ノイズ std (0.036-0.206) を大きく下回り、valley_fraction が noisy-flat null と区別不能になる noise floor 問題**である。

根拠 (3 層):
1. **直接対照証拠**: noiseless 単峰 → vf=0.0 (真陰性), noisy-flat → vf=1.0 (偽陽性)。VDR の valley≈0.8-0.9 は後者と同軌道 = ノイズ人工物の疑い。
2. **正規化不変**: 12 設定 (clip × sigma × bounds) すべて noise_confounded、any_flip=False。clip 解除でも収束 optima の R² 範囲・spread が clip=hard と完全一致 (spread_ratio=1.0) → 交絡 A (clip 飽和) は VDR では**非 load-bearing** (optima が全て R²>0.4 で [0,1] 床に届かず飽和が発火しない)。
3. **averaging 不変**: noise-robust C1 (CRN-paired n_avg 平均) で SEM を 1/4 に潰しても valley は null plateau を超えず、geometric valley 不在。

**重要な honest correction**: C1 raw は VDR を一度も `smooth` と返していない (常に高 valley_fraction)。「smooth」が出たのは noiseless 単峰 control のみ。したがって「prior の『VDR は滑らか』地形主張」は C1 幾何では **支持も否定もできない (instrument 不能)** であり、(a)(b) いずれにも確定的に倒れない。

### surviving refutation (VERIFY 3-lens の honest 開示)

| lens | refuted | severity | 残存リスク (本判定への影響) |
|------|---------|----------|------------------------------|
| flip_robustness | false | **medium** | C1 診断器自体が「浅勾配 stochastic fitness で多峰偽陽性に silently バイアス」する。raw C1 を信じた将来の解析者は vf=1.0 を見て誤って「deceptive/多峰」と高信頼で結論しうる。本研究は noisy-flat null + SEM-scale gate でこれを回避済だが、**instrument の構造バイアスは残る** → 判定を (c) 寄りに固定する根拠。flip 自体は存在せず隠蔽もなし。 |
| unclip_noise | false | low | clip=False が負 R² 散乱で偽多峰を作った疑いは **REFUTED** (全 fitness 値が strictly positive, clip 同一性 spread_ratio=1.0)。残存: opt-vs-midpoint dip は構造的 (z=5.57) であり、より高予算の noise-robust C1 なら一部解像できる可能性 = 「C1 完全 instrument 不能」は僅かに過小評価。next-step (1) で対応予定。 |
| circularity | false | none | 循環なし。判定が③勝敗を再エンコードする経路は実行されていない。 |

surviving medium refutation (flip_robustness) があるため、**判定は (c) 不確定に固定**する。これは「instrument が本 fitness の noise floor 以下」という診断器側の限界であり、研究結論として valid な負の結果。

---

## 4. ③ 研究 (diversity 維持 / MAP-Elites) への含意

- **③ablation escalation は未起動 (循環回避ルール G4 遵守)**。escalation は「C1 が baseline smooth → 正規化を外して deceptive へ反転」を gate 条件とするが、C1 は一度も deceptive を**確証**できていない (全 noise_confounded)。deceptive 未確証で③検定へ進むと循環を作るため意図的に停止。
- したがって本 ablation は exp_ea3 の honest-negative (③ NOT load-bearing: C-gen3 pass / C-gen4 fail, **panmictic 0.7015 > MAP-E 0.6822**) を **強化も反証もしない**。両者は独立だが、C1 が instrument 不能なため「正規化人工物が③結論を汚染していた」証拠も「robust に③不要」証拠も C1 からは得られない。
- **含意**: exp_ea3 の「③不要」は依然 honest-negative として保持されるが、その地形的裏付け (C1 幾何による独立追認) は**未達**。③再評価の要否は、地形が真に多峰か否かに依存し、それは現状 instrument 不能。

---

## 5. ユーザー懸念「正規化の影響で進化しない変な状態」への直接回答

**「正規化 (clip/scale) のせいで進化が止まる変な状態に陥っている」という懸念は、本検定では確認できなかった (= その証拠は出ていない)。ただし『陥っていない』とも断定できない。**

具体的に切り分けると:

1. **clip 飽和による「谷が床に埋まる平坦化」は VDR では起きていない**。収束 optima が全て R²>0.4 (D60) / R²>0.7 (D15) で、clip の [0,1] 床に届かない。clip=hard と clip=none で optima も spread も bit 単位一致 (spread_ratio=1.0)。→ **交絡 A (clip による地形潰し) は VDR では非 load-bearing で発火しない**。これは「正規化で進化が止まる」の最有力容疑を**棄却**できた部分。
2. **sigma / bounds (状態スケール) を振っても地形判定は反転しない** (any_flip=False)。→ scale 正規化が「見かけ単峰」を作っている証拠も出ていない。
3. **ただし「進化する/しない」を C1 で測ること自体ができない**。本 fitness は評価ごとに ±0.06-0.21 の R² ノイズを持つ stochastic ridge であり、C1 の谷検出はこのノイズに埋もれて計測不能。「進化しない変な状態」が地形の真の性質なのか、ノイズで見えていないだけなのかは C1 では分離不能。

**端的に**: 「正規化が原因で進化が止まっている」という機構 (clip 飽和) は VDR では否定できた。しかし「進化するか否か」自体の確定診断は、C1 ではなく (a) より高予算の noise-robust C1、または (b) deterministic fitness (step6 text proxy, rng 不使用) が必要。現状は「正規化のせいで変な状態、ではなさそうだが、地形が本当に滑らかかは未確定」が honest な答え。

---

## 6. honest 留保 (caveats)

1. **主結論は「smooth でも deceptive でもなく、C1 が本 fitness で計測不能」**。任務 branch (any_flip=false) の素案「smooth の追認」は前提誤り (C1 は一度も VDR を smooth と返していない) のため、honest disclosure 規律に従い「noise-confound の頑健性確認」に再定義した。
2. **G1 予算**: detect mode (n_evals=400) は 1 設定 535s で 30 分 gate 超過と判明 → quick mode に縮小。budget_sensitivity §2 で valley_fraction が n_evals 120/150/300 でほぼ不変 (1.0/0.893/0.893, Δ<0.05) を確認済 (G1 縮小条件充足)。それでも quick 全 12 設定の total=30.9 分で 30 分 soft gate を僅かに超過 (D60 が 1 設定 ~210s と重い)。confirm mode (n_restarts≥15/16) は未実行。
3. **G2 seed 再現**: smoke 2 回連続実行で valley_fraction 完全一致。3 seed 系列 (20260530/531/601) で is_multimodal が全設定一致 (flags=[T,T,T])。
4. **G3 degenerate なし**: 全設定で n_optima≥2 (6)、finite_rate=1.0、clip=none でも R² が NaN/-inf にならず。pairs==0 collapse なし。
5. **noise-robust 9 点目** (vdr_D15|clip=hard|bounds=wide|n_avg=16) は ~150s 計算が本ターン内未完 = 未取得。残り 8 点が全 noise_confounded・margin null 中心振動・geometric 不在で一致しており、1 点追加で集約 (has_geometric_valley=False) は不変。`c1_noise_robust_results.json` は smoke 残骸 (2 点) が残存、確定数値は `noise_robust_run.log` (8 点) が正本。
6. **vdr_D60 への noise-robust C1 は未実行** (D60 n_avg=16 は 1 点 ~3-4 分で本ターン予算超過)。12 設定 sweep で D60/D15 とも同一 noise_confounded、D15 で頑健性確認できたため D60 も同結論と推定 (未測定を明記)。
7. **low-stats 注意**: 補強 quick は予算制約で n_restarts=4 (mean_pairs≈6)。valley_fraction を 0/0.17/0.33/0.5/0.67 に粗く量子化し個別 margin を不安定化する。設計の low-stats gate (geometric は mean_pairs≥8 要求) で偽陽性抑止済だが、確定的 geometric/smooth 判定には n_restarts≥8 + n_avg≥16 + 多 null seed の confirm run が必要 (本ターン予算外)。
8. **knob B (behavior 記述子 / MAP-Elites binning) は未測定** (設計通り)。C1 は raw gene 空間で動くため構造的に効かず、escalation 限定。deceptive 未確証のため起動条件に達せず。deceptiveness verdict の「n_bins 8→16 で 17x 反転」を VDR 非依存軸で再現するかは未検証。
9. **briefing vs committed JSON 不一致 (prior 指摘の追認)**: briefing は「合成 corridor d*=0.16, 実タスクは全 below=滑らか」とするが、committed では VDR は strictly above (all_below_threshold=false)。本 C1 でも VDR baseline は smooth ではなく noise_confounded であり、prior の「VDR 滑らか・③不要」地形主張は C1 幾何では instrument 不能 = 支持も否定もできない。
10. **構造規律**: src/ 完全非改変 (`git status --porcelain src/` 空, `git diff --stat src/` 空)、git 操作なし、UTF-8 (日本語 docstring)、`py -3.11`、read-only import のみ、全成果 `research/normalization_confound/` 隔離内のみ。
11. **負の結果も valid**: 「C1 は本 stochastic ridge fitness の地形多峰性を計測できない」という診断器側の限界は、消さず研究結論として残す。これは deceptiveness verdict の教訓 (「診断が③勝敗の再エンコードであってはならない」「seed ノイズに閾値が埋もれた」) を VDR で独立に再確認したもの。

---

### 参照ファイル (全 `research/normalization_confound/` 内)
- `c1_clip_eval.py` — clip 切替 eval_once (src 非改変・read-only 流用)
- `c1_normalization_sweep.py` — 12 設定 正規化 sweep harness
- `budget_sensitivity_check.py` — 予算感度 + 診断器健全性 control (noiseless/noisy-flat)
- `c1_noise_robust_confirm.py` — noise-robust C1 (CRN-paired + SEM-scale) escalation harness
- `c1_sweep_results_quick.json` — 主 sweep 確定数値 (正本)
- `noise_robust_run.log` — noise-robust C1 確定数値 8 点 (正本; JSON は smoke 残骸)
- `quick_run.log` — 主 sweep 実行ログ
