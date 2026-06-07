# HD-1 接地 — 記憶形成機構を実モデル gradient 基質で比較する詳細設計 (v3, 2026-06-07)

> 状態: **詳細設計 v3 (敵対レビュー反映済)**。v2 への 4-lens レビュー (blocker 5 / major 9;
> 記録 = [[HD1_DESIGN_REVIEW_2026_06_07.md]]) を全反映。次 = 事前登録 doc + runner 実装 →
> CPU feasibility → GPU 本走。1 年スパン方針 — 急がず丁寧に。
>
> **v2 からの構造転換 (レビューの帰結)**:
> 1. 主仮説を H_repair から **H_sound_vs_empirical (OBSERVE の実力)** へ移動 — 真の実験的
>    不確実性はそこにある。
> 2. REVIVE は **gate で先回りしない** — 独立判定が死を検出した時に修復 (toy 意味論の忠実移植。
>    v2 は toy で潰したトートロジーを再導入していた)。
> 3. **実害 probe (state-separation) を co-primary に昇格** — 契約死と CE のデカップルへの応答。
> 4. ラベル是正: 本実験は「記憶形成機構の gradient 基質比較」であり「内的化の検証」ではない
>    (それは §8 の将来実験)。toy 軸 2 (集団記憶保存) の移植とも呼ばない (集団選択が無いため)。

## 0. 基質 (確定, v2 から不変)

`GatedRecurrentLM` (hd1_highdim_evo.py)。core: `s' = decay·s + (1−decay)·tanh(W s + x_c)`,
effective W = 2·tanh(raw_W)。tanh 有界で実発散なし — 脅かされるのは echo-state 性 (ρ<1)。
既知 (§7): ungated gradient は ρ→1.95 へ逸脱 (19/20)、gate cost CE 0.03–0.12、post-hoc 17–19×。

## 1. 死の定義 — 二層・co-primary (D1 確定)

- **契約死**: `empirical_rho ≥ 1` (from-below 実測, cadence m で測定)。gate (cert_inf = sound 上界)
  と判定 (実測下界) の分離は維持。ただし **cert_inf を強制する arm (ENDO) の契約死 ≈0 は定理の
  帰結** であり実験結果ではない — 死回避軸の検定は NONE / EXO_init / OBSERVE / REVIVE vs ENDO の
  「非 sound arm がどこまで迫れるか」のみ。
- **実害死 (co-primary)**: state-separation probe — 初期状態差を**箱端付近 (大擾乱 regime)** に
  取り長 horizon で `‖s_a − s_b‖` の減衰率を実測。線形化定理 (ρ<1⟹局所幾何減衰) の自明な帰結に
  ならない regime で echo-state 喪失の実害を operational に測る。cadence m で契約死と同時測定。
- **measure 窓**: 「後半 50%」仮置きは撤回 — **feasibility で各 arm の ρ(step) 軌跡から plateau を
  実測同定し、その値を事前登録に固定**。窓感度 (30/50/70%) は探索的に 1 回だけ報告。

## 2. arms (5 + feasibility 限定 1)

| arm | 機構 | 実装 |
|---|---|---|
| **NONE** | 無拘束 | `gate="none"` (既存流用)。接地サニティ: ρ→1.95 (§7) を再現すること |
| **EXO_init** | 初期のみ監督 | init で cert_inf 充足 (既存ループ)、以後放置 |
| **ENDO** | 継続監督 (sound) | cert_inf を cadence k で検査 → fail で rollback (既存 `"inf"`)。**rollback 時に Adam state も同期復元** (交絡対策; スナップショット) |
| **REVIVE** | 死後修復 | **gate では検査しない (NONE と同じく走る)**。独立判定 (cadence m) が契約死を検出した時に: 死を記録 → `raw_W ← c·raw_W` (raw 空間で縮小; c を admit まで二分探索し修復後 cert_inf 検査で確認) → 当該 layer の Adam state リセット → 訓練継続。**死は踏む** (deaths は NONE 同様カウント) — 価値は CE/実害軸のみ |
| **OBSERVE** | 経験的回避 | 観測された死 (自 run 履歴 + 2-pass の他 run 履歴) の **操作的 proxy = 訓練中 forward で観測する state-norm 増大率** (cert_inf 構成要素と独立) の閾値を学習し、超えたら直近更新を縮小 `Δθ ← β·Δθ`。cert_inf は呼ばない |
| (feasibility のみ) **REVIVE_ABLATE** | 反事実 | REVIVE と同じ縮小を death memory と無関係に周期発動 (純正則化) — REVIVE の CE がただの正則化効果でないことの確認 1 回 |

- 用語規律: ENDO/EXO は「継続監督/初期のみ監督」(supervision schedule) — 「内的/外的」とは
  呼ばない (レビュー M: ハーネスが gate を持つ以上、内的化ではない)。
- rollback と repair の差の正確な記述: **rollback = 時間的退行** (k step 前の方向+大きさへ復元) /
  **repair = 現方向の大域縮小** (raw_W 空間)。「差は k step 分の情報だけ」は誤り (種類が違う)。

## 3. OBSERVE の自由度 — 全固定リスト (事前登録に転記)

1. proxy = measure cadence m ごとの `g_t = log(‖s‖_post / ‖s‖_pre)` 移動平均 (window=4 測定)
   — forward で無料観測できる操作量、cert_inf と構造独立。
2. 閾値則 = 観測された死イベント時 proxy の **10 percentile** (固定)。
3. 回避行動 = 直近 m step 分の更新を `β=0.5` で縮小 (固定; feasibility で {0.25, 0.5, 0.75} を
   1 回だけ報告し本走は 0.5)。
4. 死記憶共有 = **2-pass 固定**: pass1 で全 seed が自己履歴のみで走り死を記録、pass2 で全 seed が
   pass1 の完全な死履歴を共有して走る (seed 順序効果と time-leakage を排除)。本走の比較は pass2。
5. proxy が sound 界の主成分にならないことの確認: 死イベントでの proxy と infnorm_sup の相関を
   記述的に報告 (高相関なら「OBSERVE≈ENDO は proxy の sound 近似性による」留保を発動)。

## 4. 事前登録仮説 (階層; 文言は事前登録 doc で最終化)

**Confirmatory family (Holm 補正, family = 以下 2 本):**

- **H1 (主) sound vs empirical 死回避**: OBSERVE の measure 窓死亡率 (契約死 step 比率 + 実害
  probe 非減衰率の 2 指標 co-primary) は NONE より低いが ENDO より高い。
  **反証条項**: OBSERVE が ENDO と統計的同等以下まで迫った場合 — まず §3-5 の proxy-sound 相関を
  検査し、(a) proxy が sound 主成分化していれば「proxy 設計の問題」として無効、(b) 独立 proxy の
  まま同等なら **「sound≫empirical は EA 固有」へ正式格下げ** (§6 navigability 前例に従う)。
- **H2 repair の学習保存 (CE)**: REVIVE の最終 CE < ENDO の最終 CE (paired, Wilcoxon signed-rank)。
  ※ これは「同一 run 内で、死を経験して修復する方が、死を防いで巻き戻すより学習が残るか」という
  **本実験固有の主張** — toy 軸 2 (集団記憶保存) の移植とは呼ばない。
  従属所見の自白: 効果は cadence k に依存しうる — k は固定 (k=4)、k 感度は探索的 1 回。

**Exploratory (補正外, 記述的):**

- E1 死×性能のデカップル度: 契約死 step 比率 vs CE の関係 (NONE arm 内)。
- E2 H_harm: 契約死フラグ × 実害 probe 減衰率の **step 単位 Spearman** (D1 の proxy 妥当性)。
  |ρ|≥0.5 で契約死を proxy 採用、未満なら「死回避軸は実害 probe で再定義」を発動 (分岐を事前固定)。
- E3 接地サニティ: NONE の ρ 軌跡が §7 の ρ→1.95 帯を再現 / ENDO の CE コストが 0.03–0.12 帯を
  再現 (同 cadence・同 n 限定のパイプライン同一性チェック — 「整合検定」ではない)。
- E4 REVIVE_ABLATE (feasibility のみ): REVIVE − ABLATE の CE 差 (記憶保存が正則化効果でない確認)。

**統計設計 (レビュー反映):**

- seeds = **16** (toy n=20 に近づける; 5 arms × 16 seeds × 3 n = 240 runs, T4 2-3 セッション分割)。
- 検定 = **Wilcoxon signed-rank** (paired, 両側) — sign-flip の n=8 退化問題を回避。
- 多重性 = confirmatory family 2 本に Holm。n 水準は「全 active n で方向一致 + 各 n p<0.05」を
  結論条件とする保守則 (n またぎの集約はしない)。
- **F 条項 (数値固定)**: NONE の measure 窓契約死 step 比率 < 5% の n は判定除外 (toy F2 同型)。
  feasibility ゲートの合否も同条件。
- 非劣性 ε は廃止 (H_repair の死回避非劣性は定理帰結のため検定しない — sanity 確認のみ)。

## 5. 実行計画

1. **事前登録 doc + runner** (`run_hd1_grounding.py`; hd1 コード import 流用; 結果取得前 commit)
2. **CPU feasibility** (n ∈ {8, 32}, 4 seeds, 縮小 budget): (a) F 条項 active 確認 (b) ρ(step)
   plateau → measure 窓確定 (c) OBSERVE β 3 点 / REVIVE_ABLATE / Adam-sync ablation 各 1 回
3. **feasibility 結果で事前登録を最終化** (窓・β 固定値を埋める) → 本登録 commit
4. **GPU 本走** (Kaggle T4: n ∈ {64, 128, 256}, 16 seeds × 5 arms, resumable)
5. VERDICT → 論文追補 (ラベルは「gradient 基質での機構比較」) → スライド拡充素材

## 6. 弱点の自白 (v3 更新)

- OBSERVE の proxy/閾値/β は固定したが依然 1 設計 — 「OBSERVE の上限」ではなく「1 つの誠実な
  実装の実力」しか測れない (toy と同じ留保)。
- REVIVE の修復は raw_W 縮小 + Adam リセット — 「学習を保つ」は W の現方向の保存に限る比喩で、
  network 関数の保存ではない (非線形再パラメータ化のため)。
- 契約死≈0 (ENDO) は定理帰結 — ENDO の行は「定理どおり動くことの確認」であり発見ではない。
- 16 seeds でも CE の小効果は拾えない可能性 — その場合は「検出せず」を報告 (盛らない)。

## 7. 将来実験 (本設計のスコープ外と明記)

- **真の内的化 (R-endo 本来の問い)**: cert_inf の微分可能 surrogate (∞-norm 上界は subgradient 可)
  を補助損失としてモデル自身の勾配に組み込み、「モデルが自分を ρ<1 域へ押す」arm。ハーネス監督
  (本設計の ENDO) との比較で初めて「検証の内的化」が gradient 文脈で測れる。
- 集団化 (PBT 風の選択・復活で toy 軸 2 の真の移植) — 大規模。

## 8. 正本リンク

- レビュー記録: [[HD1_DESIGN_REVIEW_2026_06_07.md]] / 基質: `research/highdim_evolution/`
- toy 決着: `VIABILITY_VERDICT.md` / 論文 §7, §9.6 / 方針: [[project_llcore_one_year_policy]]
