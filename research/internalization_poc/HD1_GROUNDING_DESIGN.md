# HD-1 接地 — 記憶形成 3 機構を実モデル gradient 基質で再評価する設計骨子 (2026-06-07 起草)

> 状態: **設計骨子 (draft)**。詳細設計 → 事前登録 → CPU feasibility → GPU 本走 の順で進める。
> 1 年スパン方針 ([[project_llcore_one_year_policy]]) の本流第一手。急がず丁寧に。

## 1. 目的 (なぜこれが次か)

§9.6 / VIABILITY_VERDICT の最重要 honest 留保 = 「スカラー gene の toy 基質であり、実 LLM スケールは
未証明」。一方 HD-1 (論文 §7) は既に **実在の発散性基質** を確立している: ungated gradient 訓練は
全次元で収縮域を出る (19/20 seeds, ρ→1.95 @ n=256, drift は entropic)。
つまり「κ を人工的にステップさせる」必要がない — **gradient 訓練そのものが viability 脅威**。
ここに 3 機構 (ENDO/REVIVE/OBSERVE) を移植して 2 軸 (死回避/記憶保存) を測れば、toy→実モデルの
接地が一段で進む。

## 2. 意味論の移植 (最大の設計リスク)

GA の「個体の死」を gradient 文脈にどう写すか。**population-based training (複数 run = 集団)** を橋に使う:

| viability PoC (GA) | HD-1 接地 (gradient) |
|---|---|
| 個体 = gene | 個体 = 訓練 run (seed/hyperparam で個性) |
| 評価 = rollout | 評価 = k step の訓練区間 |
| 死 = 発散/包絡超え | 死 = 収縮域逸脱 (certified ρ̂ ≥ 1) ※実発散ではなく契約上の死 (要設計判断 §5-D1) |
| ENDO = 評価前 self-admit | ENDO = step 後の重みを cert_inf (O(n²)) で self-certify → fail なら step reject/縮小 (= certified training の評価前版) |
| REVIVE = 死後修復 (mix 保持) | REVIVE = 逸脱した重みを certificate 保持で収縮域へ最小 projection (※post-hoc 認証が 17-19 倍高い事実 [§7.4 B-G4] との突合が見どころ) |
| OBSERVE = 他個体の死近傍を回避 | OBSERVE = 集団内の他 run の逸脱履歴 (step/領域) を経験的に回避 (PBT 風の社会学習) |
| NONE / EXO | ungated / 設計時固定 gate (κ_low 相当 = 訓練前のスペクトル仮定で固定) |

## 3. 事前登録仮説の方向 (確定は詳細設計で)

- **H_avoid**: ENDO は契約死 (逸脱 step) を NONE/OBSERVE より有意に減らす (toy の 2 軸第 1 軸の再現)。
- **H_memory**: REVIVE は逸脱を経験した run の事後性能 (CE 回復) を NONE より保つ (第 2 軸の再現)。
- **H_cost**: ENDO の CE コストが HD-1 実測の gate cost 帯 (0.03-0.12) と整合する (新規測定でなく既存
  事実との突合 = 接地の検証)。
- 反証条項: gradient では OBSERVE が ENDO に勝つ/同等なら「sound≫empirical は EA 固有」へ格下げ
  (§6 の navigability が EA 固有だった前例があるため、これは現実的なリスク = だからこそ測る価値)。

## 4. 実行計画 (段階ゲート)

1. **詳細設計 + 事前登録** (次セッション; Workflow ブレスト → 設計 → 結果取得前 commit)
2. **CPU feasibility** (n=64, 縮小 budget, smoke ~分オーダー) — 死境界が active か (F2 チェック) が gate
3. **GPU 本走** (Kaggle T4, $0, HD-1 と同じ end-to-end 自走パイプライン流用)
4. VERDICT → 論文 §9.6 への追補 or 新節 → スライド拡充の素材 (1 年方針と連動)

## 5. 未決の設計判断 (詳細設計で決める)

- **D1 死の定義**: 契約死 (certified ρ̂≥1) か、実害死 (摂動下の誤差爆発を実測) か、両方か。toy では
  実害死だった — 契約死だけだと「ゲートがゲートを測る」循環の懸念 (red-team 必須)。
- **D2 集団サイズ**: PBT 風 (8-20 runs) — Kaggle T4 で回る規模に。
- **D3 OBSERVE の情報共有粒度**: 重み空間 kNN は高次元で無意味 → スペクトル統計 (ρ̂, σ_max 軌跡) の
  近傍回避が現実的。toy との対応関係を honest に明記。
- **D4 REVIVE の projection**: spectral clipping は既存技法 — certificate 保持 (admit 集合への最小修復) と
  の差分を明確化しないと novelty が立たない (taxonomy の REVIVE 差別軸を踏襲)。

## 6. 正本リンク

- 基質: `research/highdim_evolution/` (HD-1) / 3 機構: `research/internalization_poc/` (toy 決着)
- 論文: `research/paper/PAPER_DRAFT.md` §7 (HD-1) + §9.6 (3 機構)
- 方針: [[project_llcore_one_year_policy]] (1 年スパン、submission 急がず)
