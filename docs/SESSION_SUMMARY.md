# Session Summary

> ⚠️ このファイルは Stop hook (`raptor-auto-summary`) に上書きされうる。
> 最終更新: 2026-06-16 (LM recurrent cleanup と rerun 準備は完了済み、残る人間ゲートは corpus staging 側。現状の作業木は主に記録更新で、`docs/ARTICLE_SEEDS.md` は append-only 追記が中心。`collect_research_seeds.py` 側では裸の `記事化` / `published` レグを除去済みだが、legacy shorthand `→ #...` / `→ 記事...` はまだ consumed 判定に残る。dirty の実体は `git status` を正とする)
> **実体の正本は `docs/next_plan.md`**。進捗と承認待ちの詳細はそちらを優先し、このファイルは再開ポインタに留める。
> 補助ポインタは `docs/PROGRESS.md`。

再開地点:
- 現在の主作業ブランチは `feat/lm-recurrent`
- LM recurrent 実装・verdict packet・tracked artifacts は完成済み
- strongest claim は **「RWKV が最も再現性の高い候補」**、full winner は未宣言
- LM recurrent の duplicate SVG canonical 化と `.llterm/loop_ledger.jsonl` 追跡解除はどちらも完了済み
- 現状の作業木は主に文書更新で、`docs/ARTICLE_SEEDS.md` は append-only 追記が中心。`collect_research_seeds.py` は 2026-06-16 に fullsense 側の local dirty 作業木で観測確認し、裸の `記事化` / `published` レグは除去済みだが、legacy shorthand `→ #...` / `→ 記事...` は残る状態だった。repo 外・dirty 作業木なので、この要約は**時点スナップショット**に留め、再利用前に現物を再取得する。**llcore repo 内**の dirty は `docs/*.md` の文書更新だけで、llcore 側のコード/アセットに未コミット変更は無い。fullsense repo 側は `collect_research_seeds.py` などが dirty。dirty の実体は各 repo の `git status` を正とし、ここでは固定列挙しない

未解決:
- `verified_safe_learning` staging の publish 判断
  - 既存 live v1 (`SKILL.md` 起点, 97 ノート + `SKILL.md`) と staging v2 (`INDEX.md` 起点, hierarchical) は入口互換が無く、publish 判断は「新規作成」ではなく migration 方式の選択になる
  - 詳細な 3 択比較、中間案 shim 草案、publish 前後チェックリスト、隔離チェック例、static gate の pass/fail 条件、`--reindex` の副作用メモ、人間ゲートでの選び分け基準、次に出す確認ダイアログ順は **`docs/next_plan.md` を正本**として参照する
- `self_evolving_agents` precision rerun 本実行の判断
- `self_evolving_agents` staging の publish 判断（rerun 後または現状維持）

次の具体的な一手:
1. `docs/next_plan.md` の `★ユーザー判断` と `次の具体的な一手` を開き、未回収の人間ゲートを確認する
2. 最優先で `verified_safe_learning` publish か `self_evolving_agents` precision rerun 本実行のどちらを先に処理するか、人間判断を回収する
3. rerun 実行が承認されたら、repo 外 `D:\docs\self_evolving_agents_corpus_v2.staging\_STAGING_META\queries_refined_candidate.txt` の SHA256 `2AB6A443E70D7A58DDDCFFE4213BF0156960C48E89109245CC9C34F74D6B7D73` を、**2026-06-16 10:19:33 +09:00 取得時点**の値として、保存先パス + 取得日時とセットで再確認してから fetch / corpus2skill 実行へ進む
   - この SHA は precision rerun の入力固定用に意図的に保持している gate 値だが、repo 外ファイル依存なので llcore 単体では再現不能。旧 `0E6C...` から `2AB6...` への更新は query tightening 反映

直近 gate:
- `py -3.11 -m pytest tests/unit -k lm -q && py -3.11 -m mypy src/llcore/lm/ && py -3.11 -m ruff check src/llcore/lm/`
- exit code `0`
- `91 passed, 401 deselected` / `mypy success` / `ruff success`
