# M2 設計: cert gate × 会話連結性教師の配線 (2026-06-12)

> ROADMAP M2「cert gate × 連結性教師の配線 (差別化の本命)」の設計書。
> 接地 = RAD コーパス 3 テーマ調査 (本文 §1) + verified plasticity 枠組み
> (`VERIFIED_PLASTICITY_FRAMEWORK.md`) + phase2_capability_realce の実 LLM hidden 枠。

## 0. 一行要約

会話の連結構造 (ターン境界/照応/話題) を **学習信号 (Objective)** にして small-n
verified adapter を適応させたとき、**sound cert gate だけが「会話教師に騙されない」**
ことを、無 gate / 経験 gate / cert ladder の比較で guarantee 軸において実証する。

## 1. RAD 接地 (2026-06-12 調査、Explore agent による grep)

| テーマ | 直接重複 | 最接近の先行 (RAD corpus 内) |
|---|---|---|
| (a) 検証付き継続学習 | 中 | OGPSA (orthogonal gradient projection safety alignment, llm_corpus doc_0279) / Recovery Guarantees for CL (cognitive_ai doc_0598) / Trust Region CL (deep_learning doc_0033) |
| (b) グラフ構造を学習信号に | 高 | GraphWalk (llm doc_0516) / GAAMA 4 層グラフ連想記憶 (agents doc_0279) / CGFuse (neural_network doc_0748) |
| (c) 会話連結性を教師信号に | **低** | SIOP turn-level credit assignment (deep_learning doc_0725) / CHORUS (agents doc_0620) — いずれも coherence は評価指標止まりで学習信号でない |

- キーワード未検出 (honest): "coreference supervision" / "discourse structure learning"
  は corpus 内ヒットなし。
- **差別化 = 3 軸の交点**: (a) sound certificate (真 ρ<1) × (b) 会話グラフ構造 ×
  (c) 連結性を直接 objective 化 — 3 つを同時に扱う先行例は RAD 49 分野に見当たらない。
- 注意 (circularity 回避): 教師の gold は **会話 JSON の turn 構造という外部事実**に
  接地する。AnnotationStore の連結グラフ実装 (M1 で IDF 連結は cosine と同値と判明) を
  正解と見なす循環は避ける。

## 2. 実験設計

### 2.1 基質 (phase2_capability_realce を踏襲 — 検証済みの枠を再利用)

- SmolLM2-135M (CPU frozen) layer 15 hidden (576 次元)
- 入力系列 = **実会話 35 turns** (connectivity_bench.SOURCES の 3 JSON =
  chat_staged_smoke / chat_endurance / phase2_demo_verified_chat) の
  annotation 単位系列 (split_annotations で分割、出現順)
- train PCA top-32 → seed 別ランダム射影 P_s: 32→n (n=6) → 単位分散正規化
- adapter = CoupledNDGene (n=6): `s' = decay ⊙ s + (1−decay) ⊙ tanh(W s + x_t)`

### 2.2 教師 = 連結性 objective (新規 Objective plug-in、最小→拡張の 3 段)

- **T1 ターン境界 (M2.0 最小実装)**: 各位置 t で「annotation t+1 が新 turn の先頭か
  同一 turn 内継続か」の 2 値 CE。readout = 状態 s_t の線形 2 クラス
  centroid 分類 (realce と同じ centroid 方式、train のみで fit)。
  gold = 会話 JSON の turn 区切りそのもの (外部事実、リークなし)。
- **T2 照応 (拡張)**: 同一 entity に言及する annotation ペア (coref) は状態空間で
  近く、無関係ペアは遠く (contrastive margin)。gold = entity 文字列一致
  (M1 entity-coref エッジの抽出器を再利用、ただし教師としては表層一致のみ)。
- **T3 話題遷移 (拡張)**: 「次 annotation の話題クラスタ」予測 (realce の
  centroid readout 流用、クラスタ = train annotations の KMeans)。

fitness = −mean CE (T1)。T2/T3 は M2.1 以降の追加軸。

### 2.3 比較 method (gate 軸 — H-discriminative の会話教師版)

| method | 役割 |
|---|---|
| 無 gate | 負対照 (危険の床) |
| STABLE 風経験 gate (EPS_FORGET=1e-2/T=64/K_PROBE=8) | 既踏比較 (84% false-admit の再現対象) |
| cert_inf (sound, solver-free) | 安い sound |
| cert_sdp (sound, 最 navigable) | 本命 sound |

進化 = coupled_nd の MAP-Elites / evolve (Phase 2 と同予算規律: train-fitness
評価回数で揃える、gate の resample は予算非計上 — 既存 honest 留保を継承)。

### 2.4 測定の分離 (capability と guarantee を混同しない)

1. **guarantee (主役)**: 採用 gene の empirical_rho (from-below オラクル) 分布。
   発散 gene (ρ≥1) の false-admit 数を gate ごとに数える。
   - 事前予測 (反証可能): sound cert は false-admit 0 を維持。経験 gate は
     会話教師下でも発散 gene を有意に誤許可する (Phase 2 の 84% は設定依存値
     なので、率の再現でなく「>0 で sound と分離」を判定基準にする)。
2. **capability (脇役・NEGATIVE 前提)**: held-out 会話 (turn 単位分割) の T1 CE。
   Phase 2 で capability NEGATIVE 確定済み — M2 で「進化が強い」とは主張しない。
   測る理由 = 会話教師が gate 下でも学習可能な信号であること (non-degenerate) の確認。
3. **連結性保持 = 忘却 (M2.2)**: 会話前半で適応 → 後半で継続適応 → 前半 T1 CE の
   劣化を測る。gate が「忘却を防ぐ」とは主張しない (cert は発散を防ぐのであって
   忘却を防ぐ保証ではない — over-claim 禁止)。測るのは「sound gate が忘却を
   悪化させない」(中立性) の確認。

### 2.5 統計規律 (6 装置の継承)

- 事前登録: 本設計書の §2.4 判定基準を結果を見る前に固定 (このコミットが登録)。
- seed 族 15-20、paired 比較、Holm 連言 (capability 系のみ)。
- honest 留保 (事前): 35 turns は小規模。会話系列長 ~100-200 annotations は
  T=64 の経験 gate ホライズンと同オーダー = 経験 gate に有利寄りの設定であり、
  それでも騙されるなら主張は強くなる (不利設定での負対照)。

## 3. 自走順

1. **M2.0 PoC**: T1 objective + 無 gate vs cert_sdp の 2 条件 smoke
   (識別力 = CE が floor に張り付かないこと、を最初に確認 — realce の教訓)
2. **M2.1 本測定**: 4 gate × 15+ seed、§2.4 基準で判定
3. **M2.2 忘却測定**: 前半→後半→前半の連結性保持
4. verdict ドキュメント (research/rllm_pivot/M2_VERDICT.md) + ROADMAP 更新

## 4. しないこと (scope 制御)

- capability 優位の主張 (Phase 2 で NEGATIVE 確定済み)
- 高次元化 (n>6) — navigable-sound certifier 不在は既知の第一級 negative
- 連結性グラフ (IDF/cooccur hop) を教師に使うこと — M1 で cosine と同値/微害と
  判明済み。教師は **会話 JSON の構造的事実** (turn/entity 表層) のみ。
