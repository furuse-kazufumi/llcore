# PROGRESS — セッション再開ポインタ

このファイルは **Stop hook 非管理** の再開ポインタ。**詳細の正本は `next_plan.md` であり、本ファイルは再開地点の要約に限定する。**

## 再開手順

1. まず [next_plan.md](next_plan.md) を読む
2. 特に「このセッションの最終到達点」と「次の具体的な一手」から再開する
3. `SESSION_SUMMARY.md` は hook に上書きされうるため、進捗の正本として扱わない

## 現在の要点

- `verified_safe_learning` staging は完了、人間ゲート待ち
- `self_evolving_agents` staging も完了、人間ゲート待ち
- `self_evolving_agents` は `INDEX.md` 件数補正と `/` リンク修正まで反映済み
- `corpus2skill` 側では stopword / resume 問題の再発防止修正まで完了
- `--resume-summaries` の positive round-trip はテストで確認済み
- marker 無し legacy summary の互換再読込、empty vocabulary fallback、0-score 語借用防止まで反映済み
- 比較用 rerun `self_evolving_agents_corpus_v2_stopwordcheck` により top-level label は改善確認、ただし off-topic 混入は残存
- `queries_refined_candidate.txt` を staging に追加し、`fetch_arxiv_topical.py` は今後 `Source Query` を paper markdown へ保存する
- `Source Query` は本文メタ行だと TF-IDF 汚染になるため、source-query comment + loader 側メタ除去へ修正済み
- temp fetch 検証では loader 後 top terms / `_make_label()` に `ti` `cat` `source` `query` は残らず、query 汚染遮断を確認
- `runner._strip_skill_header()` は H1 限定へ補正し、`## Overview` 始まりの legacy summary も resume 対象に戻した
- `libexec/raptor-rad-ingest` の `_ensure_utf8_io()` 追加は別筋の未記録差分として残存。rptr commit 時は分離要
- 現在地点は「(b) precision 改善 rerun の準備 + 汚染是正完了」。最小 fetch 検証も完了済み
- 次は人間判断待ち項目を先に処理するか、人間判断なしで進めるなら `(b)` rerun の準備(材料確認まで)を再開する
- 再開時は `next_plan.md` の「今回追加で進めた内容」と「次の具体的な一手」から続行
