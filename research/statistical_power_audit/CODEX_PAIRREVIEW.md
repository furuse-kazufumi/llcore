# Codex pair-review — Statistical Power Audit (2026-06-01)

Reviewer: Codex CLI gpt-5.4 (read-only), `feedback_codex_pair_review_for_llcore` 準拠。
全 finding を実コード (file:line) で検証済 ([[feedback_external_ai_verify]])。

## 総評

**(c)「部分的に underpowered」は概ね妥当 (Codex 同意)**。コード・JSON から支持される点:
- `n=15` で既知真陽性 (make_corridor_eval, d*=0.16, Cliff δ=+1.0) を正しく検出。
- **C-gen4b / flip_flop を inconclusive に格下げ**、**step6 exp7 (n=8/6) を保留** = 妥当。
- `calibration_circularity` の scope 限定 (family-internal, magnitude 非転送) は妥当。境界地図は無意味ではないが family-internal 以上には読めない。

**結論を覆すバグは無し**。ただし 2 本柱の claim 精度に訂正が要る (採用前に修正)。

## Findings (実コード検証済)

### F1 [high] Type I guard の FPR 定義が不一致 → sweet-spot 数値の根拠が不正確
`type1_guard_sweep.py` の `tpr_borderline` / `fpr_d0_corridor` / `fpr_pure_null` は「3 baseline 全勝の load_bearing」を測る一方、**`fpr_shuffle_null` だけは「固定 best baseline 1 本に対する gate pass」**を測っている (type1_guard_sweep.py:92/177/178)。この異なる FPR を `net_true_positive_gain` と sweet-spot 判定に直接混ぜ (type1_guard_sweep.py:192, consolidate_results.py:101)、本文で `0.047→0.172` を **gate 全体の null FPR** と読ませている (VERDICT §6 / :109)。
- **訂正**: 「α緩和 sweet spot なし」の **方向性は妥当**だが、提示 FPR 数値は full machinery と同一定義でない。**P1 修正 = shuffle-null も「3 baseline 全勝」で再計算**し TPR/FPR/net gain を同一基準に揃える。それまで `0.047→0.172` は「best-baseline-1本 基準の参考値」と明記。

### F2 [medium] K4 clip「真の suppression」は証拠より一段強い → 「有力候補」に降格
実装が直接測るのは `clip=True/False` での random gene 群の score spread と floor 率のみ (ablate_suppression_knobs.py:177/202)。**clip=False で MAP-E vs baseline の verdict が反転するかは未測**で、`verdict_flips` に K4 は入っていない (ablate_suppression_knobs_results.json)。それなのに本文は「K4 ... Yes (ridge系)」「唯一の能動的 suppression 機序」と断定 (VERDICT :116/:146)。
- **訂正**: 現エビデンスで言えるのは **「clip は raw 構造を強く潰す (spread 最大13x)」「suppression の有力候補」**まで。**確定は Task #12 で clip=False の MAP-E vs baseline paired verdict-flip + null-ridge FPR を測ってから**。

### F3 [medium] consolidate JSON が §9 訂正と未整合
`repower` 個票は C-gen4b bootstrap n80 近傍で psd 床が binding になり得ることを示すのに (repower_real_negatives_results.json)、`consolidate_results.py` は `limiting_condition_all = p_only ... min_effect 床ではない` をハードコード (consolidate_results.py:61/137)。markdown §9 は正しく留保済 (VERDICT :222) だが機械可読集約が不整合。
- **訂正**: P1 = consolidate の `limiting_condition_all` と verdict 文を **「@n=15 は p_only、bootstrap n80 域では psd floor が binding になり得る」**に直す。

### F4 [P3] sweet-spot ラベルの Monte Carlo 誤差
「sweet spot = K2_min_effect=0.05」の機械ラベルは MC 誤差込みで baseline と同等 → ラベルを落とすべき (over-precision 回避)。

## 影響と次手

- **③ への含意は不変**: E-A C-gen4b / flip_flop / step6 の「③不要」は **proper n で再検定するまで保留** (Task #12)。統計マシナリは一律 suppress ではないが、n≤10 盲点 + 中効果 (δ0.15-0.40) で Type II 偏向は実在。
- **Task #12 に追加**: clip=False での MAP-E vs baseline verdict-flip 測定 (F2 確定/反証)、Type I guard の event 定義統一 (F1)、consolidate JSON 訂正 (F3)。
- 本 review は外部 AI finding を実コード検証して採用した記録 ([[feedback_external_ai_verify]] / [[feedback_codex_pair_review_for_llcore]])。

## 関連
- [[project_llcore_init_2026_05_29]]
- STATISTICAL_POWER_VERDICT.md (本体), §9 (surviving refutation)
