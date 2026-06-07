# HD-1 接地 事前登録の敵対レビュー記録 (2026-06-07, commit 前ゲート)

> 対象 = [[HD1_GROUNDING_PREREG.md]] 草稿 + `hd1_grounding_kernel.py`。
> 体制 = Workflow 13 agents: 4 lenses (stats / forking-paths / design-fidelity / impl) 並列レビュー
> → blocker/major 全件を独立 agent が一次データ・実コードで敵対検証 (反証を試みる)。
> 結果 = **confirmed 4 (blocker 1 / major 3) / refuted 5**。confirmed は全て prereg/kernel に反映済み。

## Confirmed (修正反映済)

1. **[blocker / stats] ENDO 反証条項が判定不能** — 全ゼロ対の Wilcoxon は p=NaN (scipy 実測)、
   かつスパース離散カウントでは median 差=0 が実差 (mean 差 > 0) を隠す。
   → **修正**: 発動規則を decidable な mean 差規則へ書換 — 全 active n で
   mean(OBSERVE_P2) − mean(ENDO) ≤ δ = 1/80 (全ゼロ退化を含む)。Wilcoxon/median は同等判定に不使用。
2. **[major / stats] H1 の三重保守則 (max-p × 方向一致 × Holm) の検出力自白が H2 (CE) のみで過少**
   → **修正**: MC 見積りを §3 に明記 — 均一効果なら base 6% × 相対 30% 削減でも 3n power ≈ 0.98、
   **効果異質 (benefit/harm 混合) なら 0.46→0.05 へ急落**。「H1 非検出 ≠ OBSERVE 無効」を §5 に併記。
3. **[major / forking] 全窓二値カウントは OBSERVE の「死亡測定点消し」計数アーティファクトに脆弱**
   (測定点直前の縮小でピーク測定点だけ潰せる; feasibility 実測でも max rho は OBSERVE < NONE なのに
   二値カウントは OBSERVE ≥ NONE — 二指標が別の物語を語る)。
   → **修正**: H1 アーティファクト規律 (binding 解釈規則) を §3 に追加 — H1 成立の解釈は連続量
   2 指標 (窓内 max rho / ρ 超過積分) の方向一致を要求。+ E10 (avoid↔死点消失タイミング) +
   E11 (NONE 定義の excursion 窓分割) を §4 に追加。
4. **[major / impl] corpus silent fallback が binding run を汚染しうる** (Kaggle 既定 internet OFF +
   metadata 不在 + bare except で退化コーパスのまま status=done 完走)。
   → **修正**: kernel main mode は download 失敗で fail-fast + vocab ≥ 40 assert。
   kernel-metadata.json を追加し enable_internet/gpu/machine_shape を固定。prereg §2/§6 に明記。

## Refuted (検証 agent が一次データ/実コードで棄却 — 修正不要)

- **実害 indicator が構造的に常時縮退 (= co-primary は名目のみ)**: 尺度差は設計意図 (持続的
  echo-state 喪失を測る高い棒) であり、sep_rate は n=8→32 で 0 へ単調接近 = 本走 n で発火の余地。
  A3 の条件付き事前固定が正しい preregistration 作法。
- **F 条項と A3 の非対称ケース規則欠落**: F 条項 = n レベル / A3 = indicator レベルの別 granularity
  で論理は閉じている。kernel に両ゲートが機械的に実装済み。
- **「全 active n 方向一致」が結果依存の脱出口**: F 条項は NONE 専属 (OBSERVE と自由度を共有しない)
  ため post-hoc 操作の経路が構造上ない。退化 regime を落とすのは正当な床。
- **A6 閾値 0.8 が確証バイアス**: infnorm_sup 相関は feasibility で未観測 (rho 相関 +0.57 とは別量)
  = 0.8 は真の a-priori 固定。条項の発火方向も「誠実な敗北側」でインセンティブが逆。
- **A5 per-n サンプル数差の arm 非対称**: サンプル数は n 内で全 arm 同一 + 全検定は n 内 paired
  のためキャンセル。repair は sound 内部 (infnorm_sup < 1) に着地し境界に張り付かない。

## 教訓

- 事前登録の同等判定はスパースカウントで必ず退化を踏む — 「全ゼロ対をどう裁くか」を先に書く。
- 二値カウント指標は介入側が測定点を狙い撃ちできる構造を持つ — 連続量の方向一致を解釈規律で縛る。
- インフラ既定値 (Kaggle internet OFF) × bare except = binding run の silent 汚染経路。fail-fast。
