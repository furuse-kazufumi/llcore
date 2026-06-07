# Codex pair-review — ③ 第三軸決着 capstone (2026-06-02)

Reviewer: Codex CLI gpt-5.4 (read-only)。`feedback_codex_pair_review_for_llcore` / `feedback_external_ai_verify` 準拠。

## 総評: **ブロッカーなし — ③ 結論を外部確認**

Codex は本文 4 主張を「全体として防御可能」と判定。特に:
- **C-gen4b を `load_bearing` でなく候補止まりにした判断は妥当** — 実 JSON で `updated_power_at_fresh_n=0.5174`、`p_psd_floor_ceiling_updated=0.7386` (< 0.80) で確証基準未達を確認 (exp1_repower_proper_n_results.json:297)。optional-stopping / 後半ドリフト / Bonferroni の caveat も verdict 本文に明記済 (THIRD_AXIS_SETTLE_VERDICT.md:162)。
- **EXP2 決定論・非循環は clean** — `eval_noise_std_max=1.11e-16`、実 landscape `vf_mean=0.0/0.096`、control 分離が JSON 一致 (exp2_results.json:668)。本文自認どおり「真に単峰」より「閾値下の弱 multi-basin」が精密。
- **EXP3 K4 降格は現 budget に限れば妥当** — `verdict_flip=false`、`null_fpr 0.0/0.0` (exp3_clip_flip_results.json:570)。ただし `0/0` + ~7x 縮小予算ゆえ「at this budget」限定。
- **総括「proxy 基質では③不要が (B) 確定、決定的検定は GPU full LLM (b) へ」は妥当な落とし所**。

## Findings (harness の将来 rerun 堅牢性 — 現結論は汚れていない)

### CF1 [medium] exp1 の null 枝が load_bearing 枝と非対称
`exp1_repower_proper_n.py:606` は `fresh_n>=15 && diff<=0` のみで `null_confirmed_at_power` を返す。現 C-gen4b はこの枝に入らないので**今回の主張は clean** だが、将来 rerun で単に負へぶれただけで「power で null 確定」と言えてしまう (load_bearing 側の `power>=0.80 && ceiling>=0.80` と非対称)。
- **fix (next-cycle)**: null 枝にも `power>=0.80` 相当を要求。

### CF2 [medium] exp3 総合判定が diagnostic_valid を見ない
`exp3_clip_verdict_flip.py:431` は `any_real_flip==false` なら `g3.diagnostic_valid` 不問で `null_confirmed_at_power` を返す。今回は `diagnostic_valid=true` で clean だが、再走時に sanity failure を無視して過大確定の余地。
- **fix (next-cycle)**: `diagnostic_valid` failure 時は `still_inconclusive` に落とす。

### CF3 [low] exp2 冒頭コメントが実装と不一致
`exp2_deterministic_c1.py:26` の説明は旧「corridor control 多峰 ∧ quadratic smooth」を G3 と記すが、実装は dim 別 multipeak control を使い corridor は note 扱い (同:416)。結論は正しいが監査トレースが紛らわしい。
- **fix (next-cycle)**: 冒頭コメントを実装に合わせる。

### CF4 [文言] K4 の精度
K4 は「null 確定」より **「not load-bearing at this budget」** が正確 (FPR 0/0 + 低予算)。verdict 本文 §6 は at-this-budget を開示済だが run_verdict ラベルは `null_confirmed_at_power`。
- **fix (next-cycle)**: ラベル/文言を「not load-bearing at this budget」に統一。

## 結論への影響

**なし — ③ 結論は外部検証で確認された**。CF1-CF4 は全て将来 rerun (= GPU 判断後に ③ を再検定する場合) の harness 堅牢性 + 文言精度であり、現 verdict (proxy で③不要が (B) 確定 / C-gen4b は候補 / K4 降格) を覆さない。GPU full LLM 損失地形での再検定 (選択肢 b) に進む際、これら 4 fixes を適用してから harness を再利用する。

## 関連
- [[project_llcore_init_2026_05_29]]
- THIRD_AXIS_SETTLE_VERDICT.md (本体, §6 surviving refutation)
- [[feedback_codex_pair_review_for_llcore]] / [[feedback_external_ai_verify]]
