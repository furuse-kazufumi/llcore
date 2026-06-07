# HD-1 接地 — 記憶形成 3 機構を実モデル gradient 基質で再評価する詳細設計 (v2, 2026-06-07)

> 状態: **詳細設計 (敵対レビュー前)**。v1 骨子 → 本 v2 で D1-D4 を決定し仮説を確定 →
> 敵対レビュー → 事前登録 (結果取得前 commit) → CPU feasibility → GPU 本走。
> 1 年スパン方針 ([[project_llcore_one_year_policy]]) — 急がず丁寧に。

## 0. 基質の一次接地 (hd1_highdim_evo.py 読解結果)

- 基質 = `GatedRecurrentLM` (emb → [U: d→n / recurrent core (decay, W) / P: n→d / LayerNorm residual]×L → readout)。
  core: `s' = decay·s + (1−decay)·tanh(W s + x_c)`。**tanh 有界なので実発散 (NaN) はしない** —
  脅かされるのは contraction / echo-state 性 (ρ(J)<1 = fading memory = homeostasis)。
- 既存 gate: `"none"` / `"inf"` (cert_inf O(n²), 4 step ごとに検査 → fail で **prev へ rollback**)。
- 既存測定: `empirical_rho` (from-below sampling), held-out CE, reject/admit rate。resumable JSON。
- 既知事実 (§7): ungated gradient は全 n で ρ→1.95 へ逸脱 (19/20 seeds)、drift は entropic、
  gate cost は CE 0.03–0.12、post-hoc 認証は 17–19 倍高い (B-G4)。

**含意**: HD-1 の `"inf"` rollback は ENDO の対応物が**既に動いている**。よって接地の新規性は
「ENDO が死を防ぐか」(ほぼ既知) ではなく、**(a) rollback と repair の対決 (b) empirical 機構
(OBSERVE) の gradient 文脈での実力** に置く。

## 1. arms (5) — toy → gradient の意味論移植

| arm | toy (GA) | HD-1 接地 (gradient) | 実装 |
|---|---|---|---|
| **NONE** | gate なし | `gate="none"` (既存) | 流用 |
| **EXO_init** | 設計時固定 gate | **初期化時のみ** cert_inf を満たす (既存 init ループ)、以後放置 | 流用+設定 |
| **ENDO** | 評価前 self-admit | cert_inf を cadence k で検査 → fail で **prev へ rollback** (既存 `"inf"`) — 学習も巻き戻す | 流用 |
| **REVIVE** | 死後修復 (mix 保持) | cert_inf fail 時、rollback でなく **直近重みを certificate 保持で最小修復**: W の方向を保ち大きさを縮める `W ← α·W` (α を二分探索で admit 最小縮小; decay は不変) — **gradient が学んだ方向を保ったまま安全化** | 新規 (~30 行) |
| **OBSERVE** | 他個体の死近傍を経験回避 | 集団 (同時並走する他 seed run) の**観測された逸脱** (ρ̂≥1 になった時点の cheap proxy 統計) から経験的閾値を学習し、自 run の proxy がそれに近づいたら回避行動 (直近 step を縮小 `Δθ ← β·Δθ`)。**cert_inf は呼ばない** (純 empirical) | 新規 (~60 行) |

- OBSERVE の proxy (D3 決定): `max_row_abs_sum(W)` と `mean(1−decay)` の 2 統計 (O(n²) 以下, sound 性なし,
  Goodhart 可能 — toy の kNN と同じ「経験的・不完全」役)。死記憶 = 全 run 共有リスト (run 間は
  ファイル/メモリ共有, 観測のみ)。
- REVIVE の率直な位置づけ (D4): spectral clipping 系は既存技法 — 差別軸は toy と同じく
  **certificate 保持** (admit 集合への最小修復を sound 検査で確認) + **学習方向の保存** (rollback 比較)。
  novelty は機構でなく**測定** (rollback vs repair の死/学習保存の同時比較) に置く。

## 2. 死の定義 (D1 決定) — 二層で循環を断つ

- **主 (契約死)**: `empirical_rho ≥ 1` (from-below 実測)。**gate (cert_inf = sound 上界) と判定
  (empirical_rho = 実測下界) を分離**することで「ゲートがゲートを測る」循環を断つ。
  測定 cadence: m step ごと (gate cadence k と独立に固定)。
- **副 (実害死)**: state-separation probe — 2 つの初期状態 `s_a ≠ s_b` に同一入力列を流し、
  `‖s_a − s_b‖` が幾何減衰しない (echo-state 喪失の実害 = 初期条件を忘れない = 記憶が文脈でなく
  初期化に支配される)。toy の「実測誤差包絡」に対応する operational な実害。
- **measure 窓**: 訓練後半 50% のみ (toy の助走教訓 — 初期 shock を統計に入れない)。

## 3. 事前登録仮説 (方向; 文言は事前登録 doc で確定)

- **H_repair (本丸・新規)**: REVIVE は ENDO (rollback) と**同等の死回避** (契約死 step 数 差 ≤ ε)
  を保ちながら、**最終 CE が有意に低い** (修復は学習を保つ; toy 記憶保存軸の gradient 版)。
  paired sign-flip, seed pair, 両側。
- **H_sound_vs_empirical**: OBSERVE は NONE より契約死を減らすが、ENDO/REVIVE には届かない
  (sound≫empirical の gradient 版)。**反証条項 (重要)**: OBSERVE ≥ ENDO 同等なら
  「sound≫empirical は EA 固有」へ格下げ — §6 (navigability は EA 固有) の前例があるため
  現実的リスクであり、だからこそ測る価値がある。
- **H_cost**: ENDO/REVIVE の CE コスト (vs NONE) が HD-1 実測帯 0.03–0.12 と整合 (接地の検証)。
- **H_harm**: 契約死と実害死 (state-separation) の相関 — 契約死が実害の proxy として妥当かを
  データで示す (D1 の正当化を結果で裏づける)。
- **F 条項**: NONE の measure 窓契約死がゼロに近い n は判定除外 (F2 同型; §7 より n≥32 で active 期待)。

## 4. 実行計画 (段階ゲート)

1. **敵対レビュー** (本 doc; Workflow 3-4 skeptics: 意味論移植の穴 / 測定循環 / 統計 / Goodhart)
2. **事前登録 doc + runner** (`run_hd1_grounding.py`, hd1 コードを import 流用; 結果取得前 commit)
3. **CPU feasibility** (n ∈ {8, 32}, 2 seeds, 縮小 budget — 死境界 active 確認が gate)
4. **GPU 本走** (Kaggle T4: n ∈ {64, 128, 256}, 8 seeds × 5 arms — HD-1 パイプライン流用)
5. VERDICT → 論文 §9.6 追補 or 新節 → スライド拡充素材

## 5. 規模見積もり

- 5 arms × 8 seeds × 3 n = 120 runs (GPU full)。HD-1 full (2 gates × 4 seeds × 5 n = 40 runs) の
  3 倍 — T4 で 2-3 セッションに分割可 (resumable 既存)。
- OBSERVE の「集団」は同 n 内の 8 seeds を集団と見做す (run 間共有は逐次実行でも履歴ファイルで可)。

## 6. 既知の弱点 (レビューに先回りして自白)

- OBSERVE の回避行動 (`Δθ ← β·Δθ`) は 1 設計 — 弱すぎれば NONE と同じ、強すぎれば EXO 的。
  β と proxy 閾値の選び方が結果を左右する (toy の kNN radius と同じ「1 実装」留保が必要)。
- REVIVE の `W ← α·W` は方向保存と言うが、rollback も「数 step 前の学習」は保持している —
  差は「直近 k step 分の gradient 情報」のみ。k (cadence) が小さいと差も小さい → cadence を
  事前登録で固定し、感度を探索的に 1 回だけ報告。
- 契約死 (ρ̂≥1) は HD-1 で「踏んでも CE は下がり続ける」ことが既知 — 「死」の重みづけは
  homeostasis 喪失 (実害 probe) との相関で裏づける必要 (H_harm がその役)。

## 7. 正本リンク

- 基質: `research/highdim_evolution/hd1_highdim_evo.py` / toy 決着: `VIABILITY_VERDICT.md`
- 論文: §7 (HD-1) + §9.6 (3 機構) / 方針: [[project_llcore_one_year_policy]]
