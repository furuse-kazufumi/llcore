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
3. **軌跡の太陽**: 終端のみ → 状態軌跡の動的性質 (線アトラクタ度) を fitness/BD に。CPU med。
4. **補助の太陽**: next-input 予測の self-supervised 補助目的を足し、平坦な湖に勾配が出るか。CPU med。
5. **attractor 基質 vs leaky**: 同タスクで Hopfield/gated 基質の valley_fraction が落ちるか (川が干上がらない予言)。CPU med、要設計。

## 5. honest 留保

- 比喩接地は**発見的**。「川」「太陽」は物理事実でなく、machinery への写像が成り立つ範囲でのみ有効。
- 文献 (Sussillo&Barak / Howard&Kahana 等) は記憶ダイナミクスの定説だが、**我々の reservoir+ridge proxy は
  backprop full RNN/LLM とは別物**。予言の転用は proxy の限界内に留める。
- 本 doc は**生きたメモ** (対話で拡張中)。確定 verdict は C1-C4 実数 + adversarial verify を経てから。
