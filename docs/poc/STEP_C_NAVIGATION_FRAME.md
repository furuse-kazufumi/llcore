# Step C — 探索の航行枠 (Navigation Frame) — 生成中の解釈枠

> **凡例 — 進化4要素**: ①変異 ②遺伝 ③適者生存・選択 ④過剰繁殖。用語集 → [`YOUGO_平易版.md`](./YOUGO_平易版.md)。
> **位置づけ**: ユーザーとの対話 (2026-05-30) で co-develop した「landscape 航行」の比喩群を、実験 machinery と
> 検証可能予言へ接地した**生きたメモ**。verdict の「解釈枠」セクション + future-work の素材。**比喩は発見的であり
> 物理的事実でない** — 各予言は反証条件つきで検証してから採用 (鵜呑み禁止)。

## 1. 比喩 ↔ machinery ↔ 検証可能予言

| 比喩 (ユーザー) | 実験 machinery | 文献接地 | 検証可能予言 / 含意 | CPU |
|---|---|---|---|---|
| 記憶の宮殿 / loci (区別可能な目印) | MAP-Elites archive / behavior descriptor (行動記述子, BD) | method of loci | 区別可能な loci ほど物語が綴れる → richer BD で C3 が出やすい | low |
| 飛び石の物語 (順に辿る) | stepping-stone を archive で保持し dip を跨ぐ ratchet | Lehman&Stanley novelty search | 手順4 で③の勝因と実証済 | — |
| **河川 / 有向フロー** (谷沿いに上流→下流) | 状態軌跡 h[0]→…→h[T] = **線アトラクタ (line attractor)** に乗ると記憶が運ばれる | **Sussillo&Barak 2013** (flip-flop RNN は近似線アトラクタ。spec §3 既引用) / **Howard&Kahana 2002 TCM** (記憶=ゆっくり漂う時間文脈=川) | (a) 流れの持続を BD 軸に / (b) attractor 基質では川が干上がらず valley_fraction 低下 | low/med |
| **湖の小島の山** (平原に孤立した針、流れ無し) | 平坦低 fitness 平原の孤立最適解 (needle in haystack) | deceptive/plateau landscape | **C3 = 川/湖 の判別器**。MAP-E>random=川(③実在) / MAP-E≈random=湖の小島(運のみ、③無力) | (測定中) |
| **川・湖の幅と深さ** (スケール) | basin 幅 / barrier 深さ / **mutation sigma=0.15・grid=12×12 = 測る定規** | scale/resolution | river が sigma より細い・grid より細かいと**不可視** → 結果は scale 依存 → **sigma・grid sweep robustness**。barrier 深さ = ratchet 難度 | low |
| **移動する太陽** (外部・時変の方位参照で平地を航行) | **BD = 方位磁石** (fitness 平坦時に behavior で航行 = QD が湖を渡れる理由) / 補助自己教師信号 / **軌跡** (終端でなく時間発展) | 補助目的 / 内発報酬 / TCM 文脈ドリフト | **軌跡ベース fitness/BD** (終端のみ評価の honest 限界を直撃) + **next-input 予測の補助"太陽"** が平坦な湖に勾配を与える | low/med |

## 2. C1-C4 への含意 (verdict 解釈枠)

- **C3 (MAP-E vs random) = 「川か湖の小島か」の決定的判別器**。
  - C3 ✓ → 実記憶タスクは**自然な川**を持つ (手で corridor を作らずとも navigable) = ③実在。
  - C3 ✗ (MAP-E ≈ random) → **湖の小島** (多峰だが流れ無し、運のみ) = 精密化された撤退理由 (「多峰 ≠ 航行可能」)。
- **C1 (遺伝子・多峰, valley_fraction) は川/湖を区別しない**。parity=1.000 (全中点が谷) は遺伝子空間では
  むしろ湖の小島寄りの兆候だが、MAP-E は behavior 空間で飛ぶため C3 が disambiguator。

## 3. 統合テーゼ (生成中・要検証)

> 欺瞞的な記憶 landscape は**単一信号 (fitness 山登り) では航行不能**。現実の山岳踏破と同じく、
> **多手がかり航行** (地形の形 + 流れ + スケール + 外部の移動参照) が要る。それが計算的には
> **QD (品質多様性) + 補助/軌跡信号** に対応する。③ が load-bearing になるのは「川がある」regime に限る。

## 4. 検証可能実験キュー (CPU 安価順、実験完了後に着手候補)

1. **流れ診断 (archive 連結性)**: MAP-E の archive で最良セルが改善飛び石の連結鎖で到達可能か (川) / 孤立セルか (湖)。1 run から計算。CPU low。
2. **スケール robustness**: sigma ∈ {0.05,0.15,0.30} × grid ∈ {8,12,20} sweep で C1 多峰性・C3 が定規に依存するか。CPU low-med。
3. **★軌跡の太陽 (移動する太陽の軌跡ベース fitness) — ユーザー確定 (2026-05-30, verdict 後に実施)**:
   現 `make_eval_once` は `res.run(gene, inputs)[-1]` で**終端状態しか読まない**。これを**軌跡ベース**へ:
   - **設計 (最小拡張)**: 終端1点でなく**後半の時間窓 (例 t ∈ [T-k, T]) の状態**で held-out R² を測り、
     窓平均を fitness にする。線アトラクタ (持続する川) に乗った解は答えを**広い窓で安定保持**して
     高得点、終端でたまたま符号化しただけの脆い解は窓平均で減点される = 「移動する太陽 (時間を通じた
     参照)」で航行する fitness。`reservoir.make_eval_once` に `eval_window:int` を足し、`_collect` が
     `run(...)[-1]` → `run(...)[-eval_window:]` を返す改修で実装可 (per-gene ridge は流用)。
   - **反証可能仮説**: 軌跡ベース fitness では C1 多峰性が下がり (中途半端な川も窓内で部分点 → 勾配が
     立つ) かつ C3 で③優位が変わる (流れ持続を直接報酬化 → niching が効きやすい or 逆に hill-climb でも
     届く)。終端のみ fitness との A/B を strict gate で比較。
   - **honest 留保**: 窓を広げると「連続想起 (memory continuity B)」を測る別タスクに化けるので、
     窓幅は「答えが利用可能な時刻範囲」に限定し標準タスク定義を逸脱しない (地形捏造禁止 §7 と同精神)。
   - CPU med (eval_once コストは窓幅にほぼ不変、readout 評価が窓分増えるのみ)。
4. **補助の太陽**: next-input 予測の self-supervised 補助目的を足し、平坦な湖に勾配が出るか。CPU med。
5. **attractor 基質 vs leaky**: 同タスクで Hopfield/gated 基質の valley_fraction が落ちるか (川が干上がらない予言)。CPU med、要設計。

## 5. honest 留保

- 比喩接地は**発見的**。「川」「太陽」は物理事実でなく、machinery への写像が成り立つ範囲でのみ有効。
- 文献 (Sussillo&Barak / Howard&Kahana 等) は記憶ダイナミクスの定説だが、**我々の reservoir+ridge proxy は
  backprop full RNN/LLM とは別物**。予言の転用は proxy の限界内に留める。
- 本 doc は**生きたメモ** (対話で拡張中)。確定 verdict は C1-C4 実数 + adversarial verify を経てから。

## 6. story-method workflow 接地 (w2tb55l9v, 7 agents) — premise 監査 + 確定 future-work

記憶術 → QD の対応を文献接地 (Mouret&Clune 2015 / Lehman&Stanley 2011 / Maguire 2003 / Hopfield 1982 /
Bengio 1994,2009 / Sel4Sel=Frans,Soros,Witkowski 2021 arXiv:2106.09153 / CMA-ME=Fontaine 2020 GECCO /
AURORA-XCon=Coiffard 2025) し、gem-critic で premise まで監査した結果。

### 6a. ★premise レベルの honest 反証 (verdict 冒頭に必須)
- **Boldi/Ding/Spector 2023「Objectives Are All You Need」(arXiv:2311.02283, コーパス内確認済)**:
  **deceptive 領域では many-objective lexicase 選択が MAP-Elites を上回り、QD は illumination(非欺瞞)領域でのみ優位**。
  Step C は「欺瞞 landscape を QD で攻める」設定なので、**「QD/MAP-Elites が記憶 landscape の正しい道具」という枠組み前提自体が未検証 confound**。
  → C3 が MAP-E>random でも「QD 固有の③か、一般の多目的選択(lexicase)でも足りるか」を切り分けるまで「③(QD)が答え」と断定不可。
  逆に C3 が MAP-E≈random でも「③が無力」でなく「QD が wrong-tool で lexicase なら立つ」可能性が残る。**安価な lexicase baseline で前提を検定すべき**。

### 6b. C3 の confound 一覧 (これまでの対話 + workflow を統合; verdict の解釈に必須)
1. **descriptor 衝突** (前出): 2D 粗 BD が異質解を同セルに潰す → ③の多様性維持を削ぐ。
2. **lake-island**: 最適解が behavior でも孤立 (流れ無し) → C3≈random は「③無力」でなく「航行不能地形」。
3. **★基質ボトルネック** (新, P1): leaky reservoir は連想束縛4要件 (選択的保持/固定点束縛/内容アドレス性/変数束縛) を欠き、③以前に fitness 天井を作りうる → C3 失敗が「③」でなく「基質」のせいの可能性。
4. **★BD アライメント** (新, P3): 現 BD(平均実効記憶長/leak std)が基質パラメタ寄りで課題難度勾配に未アライン疑い → unaligned BD 上では③が踏み石を正しく選べない (Pugh/Soros/Stanley 2016 = 性能の支配因子)。
5. **★wrong-tool** (新, 6a): QD 自体が欺瞞領域で次善 (Boldi 2023)。

### 6c. 確定 future-work 実験 (verdict 後、critic 修正適用済の検証順)
> 全て CPU low・既存 step_c 部品の最小拡張・strict gate(n≥15, 片側Wilcoxon p<0.05, |δ|≥0.147)・two-way 反証。
> **順序が重要**: unaligned BD 上では③検証が無効化されるため P3 を先に。
1. **P3 [BD アライメント]** (③の前提チェック・最優先): 課題構造 BD(依存距離×系列位置) / AURORA風 PCA-BD vs 現 BD。弁別性指標(隣接セル間 behavior距離÷セル内分散>1)+collision率を測定。`make_behavior` 差替のみ。
2. **P1 [基質 ablation]**: leaky(S0) → scalar forget gate(S1, 真に1パラメタ追加で最安, まず S0 vs S1) → 必要なら 1-step modern-Hopfield(S2, Ramsauer 2020)。③設定固定で基質だけ振り「③の利得が基質非依存に残るか/基質を強くすると逓減するか」を分離。
3. **P4 [選択スケジュール] + lexicase baseline**: λ(t)=1→0 (novelty→fitness, 手書きスケジュールであり Sel4Sel の学習選択の再現ではない=overclaim 除去) vs 純fitness/純novelty。**L4=lexicase 選択を baseline に追加** (6a の前提検定: parity を依存距離別 sub-objective に分解)。
4. **P2 [カリキュラム]**: 多様遅延ニッチ+③ vs 単調 continuation(Elman 1993/Bengio 2009)。固定遅延=最も rugged な一点での一発勝負(Bengio 1994 が指数的 rugged 化を理論保証)。注: Rohde&Plaut 1999 の阻害は言語課題知見で parity への転用は仮説止まり(過剰一般化しない)。
5. **★軌跡の太陽** (§4 #3, ユーザー確定): 終端のみ → 後半窓の R² 窓平均 fitness。

### 6d. 実装の根拠にしない表層比喩 (機序不一致、critic 指摘)
- 逐次軌跡=archive セル隣接 (幾何近接 ≠ 順序 retrieval cue) / narrative 駆動=stepping-stone(受動保存 ≠ 能動的意味づけ) /
  PAO 二重符号化=fitness×BD(双射符号化 ≠ 特徴づけ) / Dresler 2017 訓練=③(個体内 learning ≠ 集団 selection)。
  → 対応・予言の **発想源**には使うが、**実装の機序的根拠**には使わない。

### 6e. CPU→GPU transfer の honest 分解 (ユーザー方針 2026-05-30: 「CPU で原理検証できれば GPU でも進化成立」)
スコープ確定 = **CPU で出来る範囲の原理検証**を deliverable とする (GPU は条件付き次段)。transfer 論理を honest に2分解:
- **(a) 機構 (conditional)**: 「欺瞞構造+航行可能な流れを持つ landscape では ③(QD/選択)が hill-climbing/random を上回り load-bearing」。
  → **CPU で立証可能・GPU へ transfer する** (噛み合う歯車は大きくしても噛み合う)。ユーザーの transfer 論理はこの層では**正しい**。
- **(b) 前提 (antecedent)**: 「実 GPU/LLM 訓練 landscape が (a) の欺瞞構造を**実際に持つ**か」。
  → **CPU では立証できない**。手順6 で実テキスト ESN+ridge proxy は**滑らか**だった (③不要) = proxy 地形 ≠ full-LLM 地形の可能性。これは GPU でしか試せない経験的問。
- **帰結 (honest)**: 「CPU 原理検証 → GPU 成立」は**機構(a)については成立し、GPU 投資を合理化する** (賭けでなく前提検定にする)。
  ただし**前提(b)は保証されない** — 実 LLM 地形が滑らかなら CPU 原理が健全でも GPU で ③ は効かない。
  → verdict の framing = 「**CPU が機構(conditional)を立証 / GPU は前提(実 LLM 地形が欺瞞的か)を試す**」。
  P1-P4+軌跡 fitness は全て CPU で**機構(a)を厚くする** = 「出来る範囲」に合致。`feedback_benchmark_honest_disclosure` 準拠で
  transfer を unfalsifiable bridge にしない (機構と前提を必ず分離記載)。
