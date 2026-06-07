# R-endo viability — 発散基質 × 記憶形成3機構 VERDICT (2026-06-07, 助走版確定)

なぜなぜ分析 (「無条件有界では収縮と生存がデカップル」) → 環境=recurrence ゲイン κ の発散基質を構成し、
ユーザー洞察 (「復活がないと経験が記憶に残らない」「他個体の死から学ぶ」) を実装した記憶形成3機構を
事前登録 A/B で検証した決着。正本コード = `viability_substrates.py` / `run_viability_ab.py` /
`results_viability_ab.json` (生結果)。

**本版 = ユーザー批判 2 件反映後の確定版** (旧版の数字は git 履歴 = commit 0f1da57 以前):
1. **REVIVE 修正** (commit e9a7b91): 旧「死を予見して先回り修復 (死=0)」は死回避がトートロジー →
   **死を経験してから蘇生** (deaths は被るが記憶を保って復活) に修正。死回避軸の対象から外し、
   真価は記憶保存軸 (pop_mean_fitness) で測る。
2. **助走 (warm-up) 期追加** (commit d284d10): κ_high 直後の初期 shock 死を G2_WARMUP=8 世代に吸わせ
   (OBSERVE の death_memory 学習・REVIVE の傷蓄積もここで進む)、**測定は定常状態 (G2_MEASURE=10) のみ**。
   OBSERVE への公平化 (旧 43→22 相当) + ENDO の定常性能を正確に測る。

## 結論一行

**2 軸とも事前登録方向で確定 (n=20, 死境界 active な linear/highgain)。
①死回避: ENDO (sound 予見) は定常死 0 (linear 0 vs NONE 27.2, p=0.000) — empirical な OBSERVE は
warm-up で学習させても 17.3 残る = 「証明は即時・捏造不能 / 学習は死んで学ぶ・不完全」を定常状態で定量化。
②記憶保存: REVIVE は死を経験しながら (deaths≈NONE) 集団記憶を有意に保つ (linear +0.060 p=0.001 /
highgain +0.017 p=0.034) = 「復活がないと経験が記憶に残らない」を実証。soundness viol=0 (3660 checked)。**

## 設計 (環境 = recurrence ゲイン κ, 助走版)

memory タスク (delay=8) を 3 発散基質で評価。環境 κ を mid-run でステップ:
phase1 (G1=10 世代) κ=1.0 → 助走 (G2_WARMUP=8, deaths 数えない) κ=2.0 → **測定 (G2_MEASURE=10)** κ=2.0。
κ↑ で実効収縮 κ·a が 1 を超え、以前安定だった gene (a<1) が**発散** = viability 脅威・入力ゲイン g で
回避不能・real divergence。死記憶/集団/rng は warm-up→measure 継続。
- 3 基質: **linear** (飽和除去, 幾何発散) / **softsat** (高天井 K=50) / **highgain** (高ゲイン M=20 観測)。
- 死 = 発散 (|state|>1e6/非有限) or 誤差包絡 > 生存閾 V。死=fitness 0。
- 指標 2 軸: **measure 期致命評価数** (死回避) と **pop_mean_fitness** (記憶保存 = 死んでも集団の経験が残るか)。

## 記憶形成3機構 (ユーザー洞察) + baseline

| arm | 機構 | 死への対応 |
|---|---|---|
| **ENDO** | 自己予見 | 内的健全検証器 (κ 環境結合) で死を予見し **reject** (zero-shot, sound) |
| **REVIVE** | 復活/修復 | pre-screen しない。**死を経験したら蘇生** (記憶 mix 保持・dynamics 安全化) して集団へ戻す |
| **OBSERVE** | 社会的観察 | 他個体の観察された死 (death_memory) 近傍を **経験的に回避** (kNN, lossy, Goodhart 可能) |
| NONE | — | gate なし (致命 gene も評価し死を被る; 死んだ個体の経験は消える) |
| EXO_fixed | 設計時固定 | gate κ=κ_low 固定 (κ_high 発散を見逃す) |

## 結果 (n=20 seeds 3000-3019, measure 期)

**軸1 死回避 (measure 期 deaths, 低いほど良い):**

| 基質 | NONE | EXO_fixed | **ENDO** | REVIVE | OBSERVE | soundness |
|---|---|---|---|---|---|---|
| linear | 27.2 | 10.2 | **0.0** | 18.8 | 17.3 | viol 0/1412 |
| softsat | 0.7 | 0.5 | 0.0 | 1.0 | 0.5 | viol 0/1245 |
| highgain | 5.0 | 3.4 | **0.0** | 6.3 | 2.6 | viol 0/1003 |

**軸2 記憶保存 (pop_mean_fitness, 高いほど良い):**

| 基質 | NONE | EXO_fixed | ENDO | REVIVE | OBSERVE |
|---|---|---|---|---|---|
| linear | 0.649 | 0.690 | **0.750** | **0.708** | 0.674 |
| highgain | 0.734 | 0.738 | 0.748 | **0.751** | 0.738 |

主要検定 (paired sign-flip 両側, n=20):
- **死回避**: NONE−ENDO: linear Δ=+27.2 (+20/−0, p=0.000) / highgain Δ=+5.1 (+17/−0, p=0.000)。
  OBSERVE−ENDO: linear Δ=+17.3 (+20/−0, p=0.000) / highgain Δ=+2.7 (+15/−0, p=0.0001)。
  = **sound 予見は定常死 0、empirical は学習後も死が残る**。
- **記憶保存**: REVIVE−NONE pop_mean: linear Δ=+0.060 (+16/−4, p=0.001) / highgain Δ=+0.017 (+14/−6,
  p=0.034)。REVIVE の deaths は NONE 並み (linear 18.8 vs 27.2, p=0.099 n.s. / highgain 6.3 vs 5.0,
  p=0.668 n.s.) = **同程度死んでいるのに記憶が残る差**。
- **ENDO の記憶**: linear では ENDO が記憶でも最強 (ENDO−NONE +0.102 p=0.000, ENDO−REVIVE +0.042
  p=0.000)。highgain では ENDO≈REVIVE (Δ=−0.003 p=0.373 n.s.)、ENDO−NONE は n.s. (p=0.125)。
- softsat: **INVALID (死境界 inactive, F2)** — NONE deaths=0.7 ≤ 1。下記 honest 留保参照。

## honest な発見 4 つ

1. **発散基質で内的 gate に自己保存の仕事が生まれ、定常状態で死を完全排除する**。有界 CopyTask (run_d)
   では autonomy null だったが、κ 環境では ENDO 定常死 0 (両 active 基質, +20/−0 と +17/−0)。
   なぜなぜが導いた「生存を収縮に再結合」の確認が warm-up 統制後も成立。

2. **死が消すのは elite 記憶でなく探索 (個体の経験) — 復活がそれを取り戻す**。REVIVE は NONE と同程度
   死にながら pop_mean が有意に高い = 死んだ個体の記憶 (mix) を保って復活させる分だけ、集団に経験が蓄積
   する。「復活がないと経験が記憶に残らない」(ユーザー洞察) の集団レベル実証。

3. **sound ≫ empirical は warm-up で公平化しても揺らがない**。OBSERVE は warm-up で death_memory を学習
   する機会を与えられ NONE より改善する (linear 17.3 < 27.2) が、ENDO の 0 には遠い。「経験的学習は
   機能するが、死んで学ぶ構造コストは消えない」— DGM/SEAL 型 empirical gate との定量対比。

4. **softsat の死は定常でなく transient だった** (warm-up の副産物的発見)。旧版 (助走なし) では
   NONE=2.2 で boundary active に見えたが、初期 shock を warm-up に吸わせると measure 期 NONE=0.7 =
   高天井飽和の死は κ 段差直後の過渡現象。死境界の「定常 active」判定には助走が必須という方法論的教訓。

## honest 留保 (over-claim 排除)

- **softsat は F2 で INVALID** (3 基質中 1 つは死境界が定常 inactive)。「全基質で成立」とは主張しない。
- **highgain の記憶保存は効果量小** (REVIVE−NONE +0.017, ENDO−NONE n.s.)。死圧が弱い (NONE 5.0) ため
  記憶差も縮む — 効果は死圧に比例するという整合的な読みだが、外挿はしない。
- **REVIVE の deaths≈NONE は設計どおり** (死を経験する) であり欠点ではないが、「REVIVE が死を減らす」
  とは主張できない (linear p=0.099 / highgain は符号逆 n.s.)。
- **OBSERVE の弱さは一部実装依存** (kNN radius=0.15)。tuning 余地はあるが「死んで学ぶ」構造コストは残る。
- scalar gene / 単一 κ 段差 / probe-based fitness のスコープ。κ_high=2.0 は「環境が再帰ゲインを 2 倍に
  destabilize」の honest な設定。null/n.s. も削除せず残す (feedback_benchmark_honest_disclosure)。

## llcore への含意 / 次

- **R-endo の null (有界基質) → viability 基質で positive** の図式は助走統制後も維持。HD-1 GPU
  (ungated gradient が ρ→1.95 で収縮域逸脱) が実モデルの該当例。
- 3 機構の階層 (sound 予見 > 復活 > 社会観察 > なし) が 2 軸で定量化された。taxonomy
  ([[MEMORY_FORMATION_TAXONOMY.md]]) の「認識の信頼性が下がる順の階層」と整合。
- **次**: ② 反証#2 robustness (`run_viability_robustness.py`, 順序の seed/κ/dim/hard-mem 頑健性 +
  可塑性) → ③ factorial 2³ (`run_viability_factorial.py`, E×R「予見+復活」相乗 / E が O を冗長化するか)
  → ④ regime-aware adaptive meta-controller。

## 追補 (同日): ② 反証#2 robustness — 12/12 全成立 (反証を潰した)

正本 = `run_viability_robustness.py` (2 軸意味論へ事前登録更新, 結果取得前 commit) +
`results_viability_robustness.json`。6 config (baseline / seed_shift 4000- / κ=1.5 / κ=3.0 / dim=24 /
hard_mem_delay20) × 2 基質 (linear, highgain) × 5 arms × n=12。**全 12 で死境界 active**。

| 事前登録条件 | 結果 |
|---|---|
| 軸1 死回避 (ENDO < NONE かつ ENDO < OBSERVE) | **12/12** — 全 config で ENDO 定常死 0.0 |
| 軸2 記憶保存 (REVIVE pop_mean > NONE) | **12/12** |
| 能力 (sound best ≥ unsafe best − 0.05) | **12/12** — 最大逆 gap −0.007 (hard_mem/linear) |
| soundness violations | **0** (全 config) |

- **反証#2 (「linear toy アーティファクト」) を潰した**: 順序は seed 集合・κ 規模 (1.5/2.0/3.0)・
  次元 (8/24)・記憶難度 (delay 8/20)・非線形 (highgain) の全変化で保持。
- **安定↔可塑性 TRIZ 矛盾は壊れなかった**: 記憶最適が発散境界近傍を要求する hard_mem_delay20 でも
  capability gap は −0.007/−0.004 (≥ −0.05) = **ρ<1 ゲートは記憶獲得能力を殺さない**。
  taxonomy の「最大リスク」(corpus paper 16) への実験的応答。
- 副次所見: OBSERVE は全 config で NONE より改善 (empirical 学習は機能する) が ENDO の 0 には一度も
  届かない — 「死んで学ぶ」構造コストの一貫した定量化。

## 追補 (同日): ③ factorial 2³ — 組み合わせと交互作用 (n=20, linear/highgain)

正本 = `run_viability_factorial.py` (事前登録 commit, 結果取得前) + `results_viability_factorial.json`。
事前登録 2 仮説とも**両基質で成立**。

**measure 期 deaths / pop_mean (8 combos):**

| combo | linear deaths | linear pop_mean | highgain deaths | highgain pop_mean |
|---|---|---|---|---|
| NONE | 27.2 | 0.649 | 5.0 | 0.734 |
| E | **0.0** | 0.750 | **0.0** | 0.748 |
| R | 18.8 | 0.708 | 6.3 | 0.751 |
| O | 17.3 | 0.674 | 2.6 | 0.738 |
| ER | **0.0** | 0.749 | **0.0** | 0.749 |
| EO | **0.0** | 0.750 | **0.0** | 0.749 |
| RO | 6.2 | 0.723 | 1.9 | **0.753** |
| ERO | **0.0** | 0.750 | **0.0** | 0.751 |

- **H_ER_best_of_both: True (両基質)** — ER は deaths=0 (E 並み) + pop_mean ≈ E。ただし内実は
  「E 単独で両軸最良に到達し、R の追加価値がゼロ」(int pop_mean__ER = −0.027 p=0.000 = E on で
  R の記憶保存ボーナスが消える)。「良いとこ取り」というより **sound 予見の単独支配**。
- **H_EO_redundant: True (両基質)** — int deaths__EO = +5.61 (p=0.000) / +1.71 (p=0.004)。
  sound gate は empirical 観察を完全冗長化 (E on で O の死削減効果が消滅)。
- **探索的発見: RO 相乗 (sound なし世界の最適)** — E を使えない場合、R+O の合成は単独を大幅に
  上回る (linear: RO 6.2 ≪ R 18.8 / O 17.3)。highgain では RO の pop_mean=0.753 が全 combo 最高。
  「観察が大半の死を避け、漏れた死を蘇生が保全する」相補性 = **empirical 機構は重ねるほど良いが、
  重ねても証明 1 つに勝てない** (RO 6.2 vs E 0.0)。
- honest 留保: E の支配は検証器が κ を**完全観測できる前提** (oracle sensing)。検証器の前提が壊れる
  regime (κ 推定ノイズ/遅延) では E が unsound 化しうる → ④ meta-controller の検証動機。3-way (ERO)
  と R 系交互作用は事前登録どおり探索的扱い。
