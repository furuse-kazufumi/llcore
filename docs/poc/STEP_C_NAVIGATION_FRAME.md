# Step C — 探索の航行枠 (Navigation Frame) — 生成中の解釈枠

> **凡例 — 進化4要素**: ①変異 ②遺伝 ③適者生存・選択 ④過剰繁殖。用語集 → [`YOUGO_平易版.md`](./YOUGO_平易版.md)。
> **位置づけ**: ユーザーとの対話 (2026-05-30) で co-develop した「landscape 航行」の比喩群を、実験 machinery と
> 検証可能予言へ接地した**生きたメモ**。verdict の「解釈枠」セクション + future-work の素材。**比喩は発見的であり
> 物理的事実でない** — 各予言は反証条件つきで検証してから採用 (鵜呑み禁止)。

## 用語集 (本書で使う英語術語)

> 平易な「山登り」たとえ版は [`YOUGO_平易版.md`](./YOUGO_平易版.md) を参照。ここは本書頻出の英語術語を日本語(English)形で 1 行定義する補助表。固有名詞は訳さず注釈のみ添える。

### 探索の地形と航行

- **地形 (landscape)** — 設計の良し悪しを「高さ」で表した図。高い場所ほど良い設計 (適応度地形)。
- **適応度 (fitness)** — その設計の良さ・性能スコア (=地形の高さ)。
- **行動記述子 (behavior descriptor, BD)** — 設計を「どんなタイプか」で位置づける座標。fitness とは別軸で多様性を測る方位磁石。
- **山登り (hill-climbing)** — 今より少し良い方へ動くだけの素朴な探索。だまし地形ではニセ頂上で停止する。
- **欺瞞的 (deceptive)** — ニセ頂上 (局所最適) に引っかかり本物の頂上へ行けない地形。
- **欺瞞的回廊 (deceptive corridor)** — 本物の頂上へ至る、わざと作った細い騙し経路 (手で作らずとも自然に航行可能かが C3 の論点)。
- **谷率 (valley_fraction)** — 「2 解の中点が谷になる」割合。多峰性 (谷の多さ) の指標。
- **盆地 (basin)** — 一つの頂上へ転がり込む引力圏の幅。
- **障壁 (barrier)** — 頂上と頂上を隔てる谷の深さ。深いほど飛び越え (ratchet) が難しい。

### 品質多様性 (Quality-Diversity, QD) の機構

- **品質多様性 (Quality-Diversity, QD)** — 「良さ」だけでなく「多様なタイプ」も同時に保つ探索族。だまし地形で谷を飛び石で渡る。
- **MAP-Elites** — 行動空間を網格に区切り各マスの最良解を保管する代表的 QD 手法 (固有名詞、訳さない)。
- **アーカイブ (archive)** — MAP-Elites が各ニッチの最良解をしまっておく保管庫。飛び石の貯蔵庫。
- **ニッチ / ニッチ化 (niche / niching)** — 行動空間の区画。違うタイプの解を捨てずに残す多様性維持の工夫。
- **エリート主義 (elitism)** — 各ニッチで最良個体だけを残す方針 (MAP-Elites の "Elites" の由来)。
- **飛び石 (stepping-stone)** — 単独では最良でないが、後の改良への足場になる中間解。archive で保持する。
- **ラチェット (ratchet)** — 後戻りせず一方向に進む爪車。飛び石で谷を跨いで前進を固定する働き。
- **レキシケース選択 (lexicase)** — 多目的を順に課して選ぶ選択法。欺瞞領域では MAP-Elites を上回りうる対抗手法 (Boldi 2023)。

### リザバー基質 (本書の実験道具)

- **リザバー (reservoir)** — 中身を固定した超軽量 AI (ESN) の「固定した脳みそ」部分。CPU で速い。
- **リッジ読み出し (ridge readout)** — リザバーの状態から答えを一発で解く線形読み出し計算。
- **リーク (leak)** — リザバーが過去の状態をどれだけ持ち越すかの漏れ率。記憶の保持長を決める。
- **線アトラクタ (line attractor)** — 状態が一本の線上に安定して乗り、記憶を運ぶ力学 (川のたとえ)。Sussillo&Barak 2013 接地。

### 統計・検証のことば

- **基準 (baseline)** — 「これに勝てるか」の比較相手 (例: random / lexicase)。
- **厳格ゲート (strict gate)** — 採否の厳しい合格条件。本書では n≥15・片側 Wilcoxon p<0.05・|δ|≥0.147。
- **交絡 (confound)** — 真の原因と見分けがつかず結論を曇らせる要因 (例: 基質ボトルネックが③の効果を隠す)。
- **アブレーション (ablation)** — 部品を 1 つ外して効果を切り分ける実験 (基質を振って③の利得が残るか見る)。
- **p 値 (p-value)** — 「偶然でこうなる確率」。小さい (<0.05) ほど本物の差。
- **Cliff's delta** — 差の大きさの指標。+1 で圧勝、0 で互角、− で負け (固有名詞、訳さない)。

### 生態系 (④ 過剰繁殖) のことば

- **密度依存選択 (density-dependent selection)** — 個体が混み合うほど生存競争が厳しくなる選択。有限資源が選択圧を内生化する。
- **環境収容力 (carrying capacity)** — その環境が養える個体数の上限。過剰繁殖を希少へ突き当てる。
- **赤の女王 (Red Queen)** — 捕食者と獲物が走り続ける共進化の動的選択圧 (Van Valen 1973)。
- **ニッチ構築 (niche construction)** — 生物が自ら環境 (新ニッチ) を作り変え、選択圧を生み続ける営み。

### 文献・固有名詞 (訳さず、1 行注のみ)

- **Sussillo&Barak 2013** — flip-flop RNN が近似線アトラクタを作るとした研究。
- **Howard&Kahana 2002 (TCM)** — TCM (Temporal Context Model、ゆっくり漂う時間文脈で記憶を説明)。
- **Lehman&Stanley 2011** — novelty search (新規性だけを報酬にする探索) の提唱。
- **Mouret&Clune 2015** — MAP-Elites の原典。
- **Boldi/Ding/Spector 2023** — 「Objectives Are All You Need」(arXiv:2311.02283)。欺瞞領域で lexicase が QD を上回ると示した。
- **POET** — 環境と解を共進化させるオープンエンド探索の枠組み。
- **Hopfield 1982** — 連想記憶ネットワーク (内容アドレス記憶) の原典。modern-Hopfield (Ramsauer 2020) はその拡張。
- **Tierra (Ray 1991) / Avida (Adami&Ofria)** — 自己複製プログラムが計算資源を奪い合う ALife (人工生命) システム。
- **Lenia / Flow-Lenia (Chan 2019 / Plantec 2022)** — 連続セルオートマトンの媒質に生物的パターンが創発する系。
- **Wilcoxon** — 順位ベースのノンパラメトリック有意差検定 (固有名詞)。

## 1. 比喩 ↔ machinery ↔ 検証可能予言

| 比喩 (ユーザー) | 実験 machinery | 文献接地 | 検証可能予言 / 含意 | CPU |
|---|---|---|---|---|
| 記憶の宮殿 / loci (区別可能な目印) | MAP-Elites (行動空間を網羅する品質多様性探索) アーカイブ (archive) / behavior descriptor (行動記述子, BD) | method of loci | 区別可能な loci ほど物語が綴れる → richer BD で C3 が出やすい | low |
| 飛び石の物語 (順に辿る) | 飛び石 (stepping-stone) を archive で保持し dip を跨ぐ ラチェット (ratchet) | Lehman&Stanley novelty search | 手順4 で③の勝因と実証済 | — |
| **河川 / 有向フロー** (谷沿いに上流→下流) | 状態軌跡 h[0]→…→h[T] = **線アトラクタ (line attractor)** に乗ると記憶が運ばれる | **Sussillo&Barak 2013** (flip-flop RNN は近似線アトラクタ。spec §3 既引用) / **Howard&Kahana 2002 TCM** (記憶=ゆっくり漂う時間文脈=川) | (a) 流れの持続を BD 軸に / (b) attractor 基質では川が干上がらず 谷率 (valley_fraction) 低下 | low/med |
| **湖の小島の山** (平原に孤立した針、流れ無し) | 平坦低 適応度 (fitness) 平原の孤立最適解 (干し草の山の針 (needle in haystack、平原に孤立した最適解)) | 欺瞞的 (deceptive)/plateau 地形 (landscape) | **C3 = 川/湖 の判別器**。MAP-E>random=川(③実在) / MAP-E≈random=湖の小島(運のみ、③無力) | (測定中) |
| **川・湖の幅と深さ** (スケール) | 盆地 (basin) 幅 / 障壁 (barrier) 深さ / **mutation sigma=0.15・grid=12×12 = 測る定規** | scale/resolution | river が sigma より細い・grid より細かいと**不可視** → 結果は scale 依存 → **sigma・grid sweep 頑健性 (robustness)**。barrier 深さ = ratchet 難度 | low |
| **移動する太陽** (外部・時変の方位参照で平地を航行) | **BD = 方位磁石** (fitness 平坦時に behavior で航行 = 品質多様性 (Quality-Diversity, QD) が湖を渡れる理由) / 補助自己教師信号 / **軌跡** (終端でなく時間発展) | 補助目的 / 内発報酬 / TCM 文脈ドリフト | **軌跡ベース fitness/BD** (終端のみ評価の honest 限界を直撃) + **next-input 予測の補助"太陽"** が平坦な湖に勾配を与える | low/med |

## 2. C1-C4 への含意 (verdict 解釈枠)

- **C3 (MAP-E vs random) = 「川か湖の小島か」の決定的判別器**。
  - C3 ✓ → 実記憶タスクは**自然な川**を持つ (手で 回廊 (corridor) を作らずとも navigable) = ③実在。
  - C3 ✗ (MAP-E ≈ random) → **湖の小島** (多峰だが流れ無し、運のみ) = 精密化された撤退理由 (「多峰 ≠ 航行可能」)。
- **C1 (遺伝子・多峰, 谷率 (valley_fraction)) は川/湖を区別しない**。parity=1.000 (全中点が谷) は遺伝子空間では
  むしろ湖の小島寄りの兆候だが、MAP-E は behavior 空間で飛ぶため C3 が disambiguator。

## 3. 統合テーゼ (生成中・要検証)

> 欺瞞的な記憶 地形 (landscape) は**単一信号 (適応度 (fitness) 山登り) では航行不能**。現実の山岳踏破と同じく、
> **多手がかり航行** (地形の形 + 流れ + スケール + 外部の移動参照) が要る。それが計算的には
> **QD (品質多様性) + 補助/軌跡信号** に対応する。③ が load-bearing になるのは「川がある」regime に限る。

## 4. 検証可能実験キュー (CPU 安価順、実験完了後に着手候補)

1. **流れ診断 (アーカイブ (archive) 連結性)**: MAP-E の archive で最良セルが改善飛び石の連結鎖で到達可能か (川) / 孤立セルか (湖)。1 run から計算。CPU low。
2. **スケール 頑健性 (robustness)**: sigma ∈ {0.05,0.15,0.30} × grid ∈ {8,12,20} sweep で C1 多峰性・C3 が定規に依存するか。CPU low-med。
3. **★軌跡の太陽 (移動する太陽の軌跡ベース 適応度 (fitness)) — ユーザー確定 (2026-05-30, verdict 後に実施)**:
   現 `make_eval_once` は `res.run(gene, inputs)[-1]` で**終端状態しか読まない**。これを**軌跡ベース**へ:
   - **設計 (最小拡張)**: 終端1点でなく**後半の時間窓 (例 t ∈ [T-k, T]) の状態**で held-out R² を測り、
     窓平均を fitness にする。線アトラクタ (持続する川) に乗った解は答えを**広い窓で安定保持**して
     高得点、終端でたまたま符号化しただけの脆い解は窓平均で減点される = 「移動する太陽 (時間を通じた
     参照)」で航行する fitness。`reservoir.make_eval_once` に `eval_window:int` を足し、`_collect` が
     `run(...)[-1]` → `run(...)[-eval_window:]` を返す改修で実装可 (per-gene ridge は流用)。
   - **反証可能仮説**: 軌跡ベース fitness では C1 多峰性が下がり (中途半端な川も窓内で部分点 → 勾配が
     立つ) かつ C3 で③優位が変わる (流れ持続を直接報酬化 → ニッチ化 (niching) が効きやすい or 逆に 山登り (hill-climb) でも
     届く)。終端のみ fitness との A/B を 厳格ゲート (strict gate) で比較。
   - **honest 留保**: 窓を広げると「連続想起 (memory continuity B)」を測る別タスクに化けるので、
     窓幅は「答えが利用可能な時刻範囲」に限定し標準タスク定義を逸脱しない (地形捏造禁止 §7 と同精神)。
   - CPU med (eval_once コストは窓幅にほぼ不変、readout 評価が窓分増えるのみ)。
4. **補助の太陽**: next-input 予測の self-supervised 補助目的を足し、平坦な湖に勾配が出るか。CPU med。
5. **attractor 基質 vs リーキー (leaky、漏れ率を持つリザバー)**: 同タスクで Hopfield/gated 基質の 谷率 (valley_fraction) が落ちるか (川が干上がらない予言)。CPU med、要設計。

## 5. honest 留保

- 比喩接地は**発見的**。「川」「太陽」は物理事実でなく、machinery への写像が成り立つ範囲でのみ有効。
- 文献 (Sussillo&Barak / Howard&Kahana 等) は記憶ダイナミクスの定説だが、**我々の リザバー (reservoir)+ridge proxy は
  backprop full RNN/LLM とは別物**。予言の転用は proxy の限界内に留める。
- 本 doc は**生きたメモ** (対話で拡張中)。確定 verdict は C1-C4 実数 + adversarial verify を経てから。

## 6. story-method workflow 接地 (w2tb55l9v, 7 agents) — premise 監査 + 確定 future-work

記憶術 → 品質多様性 (Quality-Diversity, QD) の対応を文献接地 (Mouret&Clune 2015 / Lehman&Stanley 2011 / Maguire 2003 / Hopfield 1982 /
Bengio 1994,2009 / Sel4Sel=Frans,Soros,Witkowski 2021 arXiv:2106.09153 / CMA-ME=Fontaine 2020 GECCO /
AURORA-XCon=Coiffard 2025) し、gem-critic で premise まで監査した結果。

### 6a. ★premise レベルの honest 反証 (verdict 冒頭に必須)
- **Boldi/Ding/Spector 2023「Objectives Are All You Need」(arXiv:2311.02283, コーパス内確認済)**:
  **欺瞞的 (deceptive) 領域では many-objective レキシケース選択 (lexicase) 選択が MAP-Elites (行動空間を網羅する品質多様性探索) を上回り、QD は illumination(非欺瞞)領域でのみ優位**。
  Step C は「欺瞞 地形 (landscape) を QD で攻める」設定なので、**「QD/MAP-Elites が記憶 landscape の正しい道具」という枠組み前提自体が未検証 交絡 (confound)**。
  → C3 が MAP-E>random でも「QD 固有の③か、一般の多目的選択(lexicase)でも足りるか」を切り分けるまで「③(QD)が答え」と断定不可。
  逆に C3 が MAP-E≈random でも「③が無力」でなく「QD が wrong-tool で lexicase なら立つ」可能性が残る。**安価な lexicase 基準 (baseline) で前提を検定すべき**。

### 6b. C3 の confound 一覧 (これまでの対話 + workflow を統合; verdict の解釈に必須)
1. **descriptor 衝突** (前出): 2D 粗 BD が異質解を同セルに潰す → ③の多様性維持を削ぐ。
2. **lake-island**: 最適解が behavior でも孤立 (流れ無し) → C3≈random は「③無力」でなく「航行不能地形」。
3. **★基質ボトルネック** (新, P1): リーキー (leaky、漏れ率を持つリザバー) リザバー (reservoir) は連想束縛4要件 (選択的保持/固定点束縛/内容アドレス性/変数束縛) を欠き、③以前に 適応度 (fitness) 天井を作りうる → C3 失敗が「③」でなく「基質」のせいの可能性。
4. **★BD アライメント** (新, P3): 現 BD(平均実効記憶長/leak std)が基質パラメタ寄りで課題難度勾配に未アライン疑い → unaligned BD 上では③が踏み石を正しく選べない (Pugh/Soros/Stanley 2016 = 性能の支配因子)。
5. **★wrong-tool** (新, 6a): QD 自体が欺瞞領域で次善 (Boldi 2023)。

### 6c. 確定 future-work 実験 (verdict 後、critic 修正適用済の検証順)
> 全て CPU low・既存 step_c 部品の最小拡張・strict gate(n≥15, 片側Wilcoxon p<0.05, |δ|≥0.147)・two-way 反証。
> **順序が重要**: unaligned BD 上では③検証が無効化されるため P3 を先に。
1. **P3 [BD アライメント]** (③の前提チェック・最優先): 課題構造 BD(依存距離×系列位置) / AURORA風 PCA-BD vs 現 BD。弁別性指標(隣接セル間 behavior距離÷セル内分散>1)+collision率を測定。`make_behavior` 差替のみ。
2. **P1 [基質 アブレーション (ablation)]**: leaky(S0) → scalar forget gate(S1, 真に1パラメタ追加で最安, まず S0 vs S1) → 必要なら 1-step modern-Hopfield(S2, Ramsauer 2020)。③設定固定で基質だけ振り「③の利得が基質非依存に残るか/基質を強くすると逓減するか」を分離。
3. **P4 [選択スケジュール] + lexicase baseline**: λ(t)=1→0 (novelty→fitness, 手書きスケジュールであり Sel4Sel の学習選択の再現ではない=overclaim 除去) vs 純fitness/純novelty。**L4=lexicase 選択を baseline に追加** (6a の前提検定: parity を依存距離別 sub-objective に分解)。
4. **P2 [カリキュラム]**: 多様遅延ニッチ+③ vs 単調 continuation(Elman 1993/Bengio 2009)。固定遅延=最も rugged な一点での一発勝負(Bengio 1994 が指数的 rugged 化を理論保証)。注: Rohde&Plaut 1999 の阻害は言語課題知見で parity への転用は仮説止まり(過剰一般化しない)。
5. **★軌跡の太陽** (§4 #3, ユーザー確定): 終端のみ → 後半窓の R² 窓平均 fitness。

### 6d. 実装の根拠にしない表層比喩 (機序不一致、critic 指摘)
- 逐次軌跡=archive セル隣接 (幾何近接 ≠ 順序 retrieval cue) / narrative 駆動=stepping-stone(受動保存 ≠ 能動的意味づけ) /
  PAO 二重符号化=fitness×BD(双射符号化 ≠ 特徴づけ) / Dresler 2017 訓練=③(個体内 learning ≠ 集団 selection)。
  → 対応・予言の **発想源**には使うが、**実装の機序的根拠**には使わない。

### 6e. CPU→GPU transfer の honest 分解 (ユーザー方針 2026-05-30: 「CPU で原理検証できれば GPU でも進化成立」)
スコープ確定 = **CPU で出来る範囲の原理検証**を deliverable とする (GPU は条件付き次段)。transfer 論理を honest に2分解:
- **(a) 機構 (conditional)**: 「欺瞞構造+航行可能な流れを持つ landscape では ③(QD/選択)が 山登り (hill-climbing)/random を上回り load-bearing」。
  → **CPU で立証可能・GPU へ transfer する** (噛み合う歯車は大きくしても噛み合う)。ユーザーの transfer 論理はこの層では**正しい**。
- **(b) 前提 (antecedent)**: 「実 GPU/LLM 訓練 landscape が (a) の欺瞞構造を**実際に持つ**か」。
  → **CPU では立証できない**。手順6 で実テキスト ESN+ridge proxy は**滑らか**だった (③不要) = proxy 地形 ≠ full-LLM 地形の可能性。これは GPU でしか試せない経験的問。
- **帰結 (honest)**: 「CPU 原理検証 → GPU 成立」は**機構(a)については成立し、GPU 投資を合理化する** (賭けでなく前提検定にする)。
  ただし**前提(b)は保証されない** — 実 LLM 地形が滑らかなら CPU 原理が健全でも GPU で ③ は効かない。
  → verdict の framing = 「**CPU が機構(conditional)を立証 / GPU は前提(実 LLM 地形が欺瞞的か)を試す**」。
  P1-P4+軌跡 fitness は全て CPU で**機構(a)を厚くする** = 「出来る範囲」に合致。`feedback_benchmark_honest_disclosure` 準拠で
  transfer を unfalsifiable bridge にしない (機構と前提を必ず分離記載)。

## 7. ④過剰繁殖 と生態系 (ユーザー 2026-05-30: 「④もGPUでないと厳しい、捕食者・食料の概念まで要る」)

### 7a. ④空転の診断 (なぜ現 setup で ④ が働かないか)
- Darwin の④ (over-reproduction) = **有限資源下で生存可能数を超える個体が生まれる** → 競争 → ③(選択)が意味を持つ (Malthus→Darwin)。
- 現 Step C (MAP-Elites/GA, **固定 pop_size=20 / 固定 n_evals=2000**) は**固定予算の選択**であって**過剰繁殖×資源希少が無い**。
  → 選択はあるが「淘汰の圧」が外生的に与えた予算配分にすぎない = **④ 空転** (監査 EVOLUTION_SOUNDNESS_AUDIT の「③④空転」と一致する根本原因)。
- つまり ③ が空転して見えたのは、**④ (過剰繁殖→希少→密度依存選択) が構造的に不在**で選択圧が内生的に生まれていないから、とも読める。

### 7b. 捕食者・食料 の machinery 写像
- **食料/資源 = 密度依存選択 (density-dependent selection) / 環境収容力 (carrying capacity)**。有限食料が pop を律速 → 過剰繁殖が希少に出会う → 選択が内生化。logistic 成長 / Lotka-Volterra。
- **捕食者 = 共進化/赤の女王 (Van Valen 1973) の移動する選択標的**。動的選択圧 = 環境複雑性テーゼ「新ニッチを生み続ける環境」(wkackrdhl が裏取り中)。
- 合わせて = **内生的・動的選択圧を自己生成する生態系** → ④③ が空転をやめる。古典 ALife: Tierra (Ray 1991) / Avida (Adami&Ofria) / PolyWorld (Yaeger 1994)。

### 7c. CPU→GPU 分解 (§6e と同型、④ にも適用)
- **(a) 機構**: 「有限資源+過剰繁殖 → 密度依存選択が生まれ ③ が働く」「捕食者共進化 → 動的選択圧」。
  → **CPU の toy ecology (agent-based, 有限資源, 変動 pop) でプロトタイプ可能**。古典 ALife は modest hardware で動いた = 機構は GPU 非依存。
- **(b) スケール/現実性**: 大集団・実 LLM agent・豊かな per-agent 計算 → GPU。
- **honest 帰結**: ユーザー「④もGPUでないと厳しい」は**スケール/現実性については正しい**が、**機構(a) は CPU で立証可能**。
  → 「出来る範囲の CPU 原理検証」に **④ toy ecology**(「資源希少が選択圧を生み ③ を非空転化するか」の最小検証)を含められる。
  ただし現 Step C の固定予算 GA からは構造改修 (資源動態+変動 pop) が要り、③ の 地形 (landscape) 改修より大きい変更 = 段階を踏む。

### 7d. 統合 (③ landscape 航行 と ④ 生態系 の関係)
- §1-6 (③/navigation) は「**与えられた landscape をどう航行するか**」、§7 (④/ecology) は「**そもそも選択圧をどう内生させるか**」。
- 監査の「③④空転」= landscape が平坦(③側の問題) **かつ** 過剰繁殖×資源希少が不在(④側の問題) の二重欠如。
- environment-complexity workflow (wkackrdhl, 反証ガード付き) が Red Queen/niche construction/POET を裏取り中 → **完了時に「CPU toy ecology で ④ を非空転化する最小実験」を design 出力**へ統合する。
  反証可能形 (unfalsifiable 回避): 「有限資源+変動 pop を入れると、固定予算 baseline より ③ の load-bearing 度(MAP-E−random の δ)が有意に上がる」を strict gate で検定。上がらなければ「④ 生態系も Step C 課題では不要」と negative 決着。

## 8. 海 vs landscape パラダイム (ユーザー 2026-05-30: 「生命誕生に倣うなら landscape より海から始めた方が良かった」)

### 8a. 2つのパラダイム
- **landscape パラダイム** (現 Step C): Wright 1932 の適応度地形。**外部から固定された目的(地形)**を hill-climb/航行する。fitness は実験者が与える静的写像。
  → ③(選択圧)も④(過剰繁殖)も**外から課す**構造 = §7a の「④空転」と §6b の confound は**この枠組みに内在**する。
- **海パラダイム** (生命誕生): 海=**流体的・動的な媒質**。分子が拡散・混合・反応し、自己組織化する。地形を「登る」のでなく、媒質の力学から
  **個体性・選択・複製が内生的に創発**する。fitness/選択単位/集団動態を実験者が与えず、**媒質から立ち上げる**。

### 8b. これは ③④空転の最深層の根
- ③ が空転 = landscape が平坦 (§6b lake)。④ が空転 = 過剰繁殖×希少が不在 (§7a)。
- **その根** = そもそも「landscape (固定目的) を課した」こと自体が「③④を外部から押し付ける」問題を**焼き込んでいる**。
  海パラダイムなら ③④ は押し付けでなく**創発**する。ユーザーは「機構の調整」でなく「**パラダイムの選択**」のレイヤに到達した。

### 8c. ALife 接地
- **Tierra (Ray 1991) / Avida**: 自己複製プログラムが CPU/メモリ(=海)を奪い合う。選択は**差次的複製から創発**(imposed fitness でない) = 海パラダイムの計算的実装。
- **自己触媒集合 (Kauffman 1986) / ハイパーサイクル (Eigen)**: 自己複製が化学から創発。
- **散逸構造 (Prigogine)**: エネルギー流束下で秩序が創発 (非平衡)。
- **Lenia/Flow-Lenia (Chan 2019/Plantec 2022)**: 連続 CA の媒質に自己組織化する「生物」が創発。

### 8d. honest トレードオフ (なぜ我々は landscape を選んだか、そのコスト)
- **海の利点**: 開放端性・創発に忠実 (Stanley「目的は有害になりうる」と整合)。③④ を内生化する唯一の本筋。
- **海の難点 (致命的)**: **制御・測定・反証が極めて困難**。Tierra/Avida は「面白い結果」を出すのも解釈するのも難しく、**「lifelike だ!」式の手放し称賛に堕しやすい**
  (= `feedback_benchmark_honest_disclosure` が最も警戒する失敗様式)。landscape は限界だらけだが**清潔な反証可能問**(「③ は random に勝つか」)を与える。
- **帰結**: 現 Step C が landscape を選んだのは**反証可能性のため**であり、正しい第一歩。だが「③ がなぜ立ちにくいか」の**根本は海を選ばなかったこと**にある、というのが honest な自己診断。

### 8e. 統合と進め方 (パラダイムを混ぜない)
- §7 の CPU toy ecology (有限資源→密度依存選択) は**landscape→海への中間段**。海パラダイムの最小・反証可能な一歩。
- より大胆な海 (目的自体が創発する媒質) は次パラダイムの prototype。**ただし反証可能性を死守**(wkackrdhl の falsifiability guard が必須。「lifelike」では verdict にしない)。
- **現スコープ (CPU 原理検証) では landscape の清潔な問に決着をつける** (= STEP_C_VERDICT)。海は「landscape で③の限界を見切った後の、次の反証可能な一歩」として §7→海の順で段階化。
  パラダイムを混ぜると測定不能になるため、verdict は landscape 枠で閉じ、海は future-work に分離記載する。
