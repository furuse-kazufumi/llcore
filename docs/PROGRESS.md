# PROGRESS — セッション再開ポインタ

このファイルは **Stop hook 非管理** の再開ポインタ。**再開フローの正本は `next_plan.md`** であり、本ファイルは再開地点の要約に限定する。

## 再開手順

1. まず [next_plan.md](next_plan.md) を読む
2. 特に「このセッションの最終到達点」と「次の具体的な一手」から再開する
3. `SESSION_SUMMARY.md` は hook に上書きされうるため、進捗の正本として扱わない

## 現在の要点

- **現在の主作業ブランチは `feat/lm-recurrent`**。LM recurrent 実験の正本は `docs/LM_RECURRENT_PLAN.md` と tracked artifact `docs/artifacts/lm_recurrent_*` で、再開フロー自体の正本は引き続き `next_plan.md`
- LM recurrent 比較は **verdict packet 完成** まで到達:
  - `docs/artifacts/lm_recurrent_verdict.md` に PPL 表 / memory@T 曲線 / caveat を集約
  - strongest claim は「**RWKV が最も再現性の高い候補**」。`64/160` の 3 seed と `64/240` の 3 seed で raw PPL 最良と unigram floor 通過を維持
  - ただし full winner は未宣言。GPT と Recurrent の相対順位は seed-sensitive のまま
- artifact/verdict 回帰保護は実装済み:
  - `tests/unit/test_lm_artifacts.py` が tracked JSON から verdict row / `6/6` claim / md / svg を再計算して照合
  - 正式 gate は `py -3.11 -m pytest tests/unit -k lm -q && py -3.11 -m mypy src/llcore/lm/ && py -3.11 -m ruff check src/llcore/lm/`
- LM recurrent の duplicate SVG 整理は **canonical 化まで完了済み**:
  - `13bcc26` で `lm_recurrent_pilot160.svg` を canonical として残し、byte 同一だった block_size=64 SVG 6 枚を削除
  - `f6c7bb2` で canonical 化後の doc 記述を現状へ整合
  - LM recurrent 側の承認待ちはこの件については解消済み
- 現在の人間ゲート待ちは別ストリームが中心:
  - `verified_safe_learning` staging の publish 判断
  - `self_evolving_agents` staging の publish / rerun 関連判断

- `verified_safe_learning` staging は完了、人間ゲート待ち
- `self_evolving_agents` staging も完了、人間ゲート待ち
- `self_evolving_agents` は `INDEX.md` 件数補正と `/` リンク修正まで反映済み
- `corpus2skill` 側では stopword / resume 問題の再発防止修正まで完了
- `--resume-summaries` の positive round-trip はテストで確認済み
- marker 無し legacy summary の互換再読込、empty vocabulary fallback、0-score 語借用防止まで反映済み
- 比較用 rerun `self_evolving_agents_corpus_v2_stopwordcheck` により top-level label は改善確認、ただし off-topic 混入は残存
- `queries_refined_candidate.txt` を staging に追加し、`fetch_arxiv_topical.py` は今後 `Source Query` を paper markdown へ保存する
- 2026-06-15 時点で `queries_refined_candidate.txt` はさらに精密化済み。ただし **効果は未検証** で、`ti:` 化により recall が落ちる可能性があるため、本実行前に「改善候補」として扱う
- `Source Query` は本文メタ行だと TF-IDF 汚染になるため、source-query comment + loader 側メタ除去へ修正済み
- temp fetch 検証では loader 後 top terms / `_make_label()` に `ti` `cat` `source` `query` は残らず、query 汚染遮断を確認
- `runner._strip_skill_header()` は H1 限定へ補正し、`## Overview` 始まりの legacy summary も resume 対象に戻した
- `D:\tools\raptor` 側の現 dirty は `_bazue_*` 3 件削除のみで、self_evolving_agents rerun 束には混ぜない
- 現在地点は「`self_evolving_agents` の (b) rerun 方針は承認済み、Anthropic 要約器の疎通確認も通過済み、残る主ブロッカーは rerun 本実行の人間判断待ち」という状態。最小 fetch 検証と query 汚染是正までは完了済み
- 現在の既存 dirty はなし（作業木 clean）
- 次は `verified_safe_learning` / `self_evolving_agents` の人間判断待ちを処理するか、人間判断なしで進めるなら `(b)` rerun 本実行の判断回収と準備再開へ進む
- 再開時は `next_plan.md` の「今回追加で進めた内容」と「次の具体的な一手」から続行
