# 検証シグナル勾配内蔵 設計骨子 v1 の敵対レビュー記録 (2026-06-08)

> 対象 = [[HD1_INTERNALIZATION_DESIGN.md]] 骨子 v1 + `cert_surrogate.py`。
> 体制 = Workflow 18 agents: 4 lenses (surrogate_math / confound / overclaim / feasibility_gaps)
> 並列レビュー → major/blocker 全件を独立 agent が実コード・一次データで敵対検証。
> 結果 = **confirmed 14 (blocker 4 / major 10) / refuted 0**。骨子 v1 の粗さを規律が捕捉
> (fail-closed が機能)。全件 設計 v2 に反映済み。

## Confirmed (設計 v2 反映済) — 3 系統

### A. 統計・事前登録の規律漏れ (HD-1 prereg が骨子 v1 に未移植)

- **C3 [blocker] circularity**: surrogate(infnorm_sup) で死回避を測ると同語反復。HD-1 A6 相当の
  独立性条項が欠落。→ v2 §3「死判定は empirical_rho 単独・surrogate は内部量」+ A6 移植 (相関報告)。
- **C4 [blocker] H2 が λ チューニングの産物**: 片側だけ自由 hyperparameter の非対称点比較。
  → v2 §3「Pareto 曲線 vs 曲線」(λ sweep で死回避率 vs CE、HARNESS 点が劣側か)。
- **C10 [blocker] λ/margin 固定基準が結果非依存で事前定義されていない** (garden of forking)。
  → v2 §4 結果非依存の閉じた式 (境界滞在率規則 / grad-norm 比) で binding。
- **C11 [blocker] F 条項・A3 縮退条項の欠落**。→ v2 §3 に HD-1 から移植 + トートロジー非該当の実証。
- **C7 [major] feasibility 最終化が H2 とリーク** (λ-by-n)。→ v2 §4 feasibility 固定値を全 n 適用。
- **C12 [major] 16 seeds 検出力不足** (HD-1 H2 が n=128 で連言破った前例)。→ v2 §3/§5 MC 見積 +
  32 seeds 検討 + 「非検出≠無効」事前固定。

### B. over-claim (HD-1 が潰した「内的化」語の再導入)

- **C8 [major] arms 表の「内的/外的」ラベル** = HD-1 設計が敵対レビューの帰結で禁じた語。
  → v2 §2 ラベルを「監督がモデルに作用する経路」(勾配経由/事後 rollback) に置換。
- **C9 [major] 「検証の内的化」は over-claim** (両 arm ともハーネスが cert 指定)。
  → v2 全体で「検証シグナルの勾配内蔵」に統一。真の内的化は §8 の未到達上位目標へ。

### C. surrogate の物理 (実コードで測った構造リスク)

- **C1 [major] pull-back latency が予算超過**: violating→admissible 復帰に n=64 で 210–533 step
  (本走予算 400 超)。→ v2 §6 (e) を go/no-go に前置 (latency > k·grad_steps なら ENDO_GRAD 不適)。
- **C2 [major] margin が bound slack 二重計上**: GRAD は HARNESS+margin ぶん深く縮小し H2 を自課税。
  → v2 §2 ENDO_GRAD_MATCHED arm 追加 + §6(g) 真ρ実測 + §5 自白訂正。
- **C5 [major] H1 非自明性が temporal soft 論だけ**: 上界を押して下界 death が減るかが核。
  → v2 §3 H1 を量の階層 (infnorm_sup ≥ true ρ ≥ empirical_rho) で書き直し + gap 報告。
- **C13 [major] 単一 argmax 行 subgradient + tanh 飽和**: 毎 step 1/n 行のみ・膨張域で 94.5%
  勾配≈0。drift (全 n 行) に n とともに負ける。→ v2 §1/§6(f) 勾配被覆率 probe + logsumexp 変種比較。
- **C14 [major] margin>0 と「admit 内 grad-zero」不整合**。→ v2 §1「admit 中核 grad-zero」訂正 +
  §6(c′) operationalize + self-test に margin>0 ケース追加。
- **C6 [major] ENDO_BOTH の交絡を探索的で逃げている** (ABLATE 対照なし)。→ v2 §2 主張を
  「rollback が GRAD と共存して壊れない動作確認」に限定。

## 教訓

- 骨子段階でも HD-1 prereg の binding 規律 (F 条項 / A6 独立性 / λ 事前固定 / 検出力自白) は
  最初から移植すべき — 「骨子だから後で」が confirmed 6 件を生んだ。
- 上界 surrogate で下界 death を測る設計は **circularity と「上界→下界伝播の非効率」の二重リスク**。
- 一度敵対レビューで潰したラベル (「内的化」) は対象が変わると無自覚に再導入される — 用語規律は
  プロジェクト横断で持ち越す。
