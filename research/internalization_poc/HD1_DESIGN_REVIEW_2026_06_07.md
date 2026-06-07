# HD-1 接地 詳細設計 v2 の敵対レビュー記録 (2026-06-07, Workflow 4 lenses)

> 対象 = HD1_GROUNDING_DESIGN.md v2 (commit 時点)。4 lenses (意味論/循環/統計/Goodhart) 全員
> **needs_fixes** (blocker 5 / major 9 / minor 6)。本記録は v3 への反映台帳 (全文は workflow
> `wtx7lxrw5` 出力)。

## blocker (全採用 → v3 へ)

1. **[意味論] 「個体=run」写像で toy の死の定義的性質 (集団からの除去) が消失** — REVIVE の
   記憶保存軸 (pop_mean) に HD-1 対応物が無い。→ v3: ラベル修正 (案 A)。「toy 軸 2 の移植」とは
   呼ばず「同一 run 内の rollback vs repair の CE 差」という別主張として登録。
2. **[循環/Goodhart ×2] REVIVE/ENDO の死回避が cert_inf 共有の定理帰結 (トートロジー)** —
   cert_inf admit ⟹ ρ<1 は定理。**toy は REVIVE を「先回りしない」設計で既に解いていた**のに
   v2 は gate fail 先回り修復で再導入していた。→ v3: REVIVE は gate では検査せず、**独立判定
   (empirical_rho≥1, cadence m) が死を検出した時に初めて修復** (toy 意味論の忠実移植)。
   死回避軸の検定は NONE/EXO/OBSERVE vs ENDO に限定 (ENDO vs REVIVE 死差の検定は無意味)。
3. **[統計] n=8 × two-sided sign-flip は構造的に検出不能** (最小 p=0.0078 は 8/0 全会一致のみ、
   多重補正で全滅; v2 の「最小 p≈0.004」は片側の誤り)。→ v3: 16 seeds + Wilcoxon signed-rank +
   仮説の階層化 (confirmatory 1 family + exploratory) + Holm。

## major (採用方針)

- **Adam state 交絡**: rollback が optimizer モーメントを巻き戻さない → ENDO vs REVIVE の CE 差に
  実装交絡。→ v3: rollback/repair とも opt.state スナップショット同期を基本 + ablation 1 回。
- **「内的化」ラベル over-claim**: HD-1 の ENDO はハーネス監督 (継続監督 vs 初期のみ監督)。→ v3:
  「内的化の検証」と呼ばない。真の内的化 (cert の微分可能 surrogate を loss に組む) は将来実験 §8。
- **OBSERVE proxy が cert_inf 主成分** (max_row_abs_sum = infnorm_sup の支配項) → 「純 empirical」
  崩壊 + 反証条項の誤発火リスク。→ v3: proxy を構造独立な操作量 (観測 state-norm 増大率) に変更。
- **実害 probe の主軸昇格**: 契約死と CE がデカップル (v2 自白) → state-separation probe を
  co-primary に昇格 + 大擾乱 regime で測定 (線形化定理の自明帰結を避ける)。
- **真の賭けは OBSERVE**: H_repair は k (cadence) 依存の従属所見 → 主仮説を H_sound_vs_empirical
  に置く。「rollback=時間的退行 / repair=現方向の大域縮小」と差の種類を正確に記述 (「k step 分
  だけ」は誤り)。raw_W=2·tanh 再パラメータ化で effective 空間の α 縮小は raw 空間で方向非保存 —
  修復は raw_W 空間で定義し直す。
- **H_cost 降格**: 「0.03-0.12 と整合」は判定不能 → sanity check に降格。接地サニティの主役は
  「NONE が ρ→1.95 (§7 既知) を再現するか」。
- **measure 窓「後半 50%」は無根拠** (toy の warm-up は環境ステップ紐づけ、HD-1 は連続ドリフト)
  → feasibility で ρ(step) 軌跡の plateau を実測同定して事前登録 + 窓感度 1 回報告。

## minor (採用)

- 非劣性 ε / F 条項閾値 / H_harm 検定法を数値固定 (F: NONE の measure 窓契約死 step 比率 <5% で
  除外。H_harm は step 単位 Spearman に再定義)。
- OBSERVE 自由度の固定列挙 (β / proxy 合成 / 閾値則 / 共有の因果順序 = 2-pass)。
- α 二分探索の単調性は raw_W 空間 c·raw_W なら成立しやすい — 修復後 admit 検査を必須に。
- REVIVE_ABLATE (death memory なし純正則化) は feasibility のみで 1 回確認。

## レビューが「正しい」と認めた点 (保持)

gate (sound 上界) と判定 (実測下界) の分離 / 新規性の (a)(b) 絞り込み / 反証条項
(OBSERVE≥ENDO → EA 固有へ格下げ) / F 条項 / 弱点自白の姿勢。
