# next_plan (正本) — EXIT 時点の再開地点

> 最終更新: 2026-06-15 ((b) rerun 準備の query 精密化まで完了)
> SESSION_SUMMARY.md は Stop hook で自動上書きされるため、**このファイルが再開の正本**。
> hook 非管理の再開ポインタ: `docs/PROGRESS.md`

## EXIT 時点の現在地

- **コア作業 (`corpus2skill` 修正 + 比較 rerun) は本セッション前半で完了済み**。この EXIT 整形ターンでは **新規作業を開始せず、記録更新のみ** 実施
- `Source Query` 汚染是正、legacy summary resume 修正、最小 runtime 検証までは完了
- 次セッションの最初の判断点は **人間判断待ち項目を先に処理するか**、または人間判断なしで進めるなら **(b) rerun の準備(材料確認まで)だけ進めるか** の 2 択
- 不可逆操作 (`publish` / rename / push / 削除) は引き続き人間承認なしに実行しない
- **(b) precision 改善 rerun は、準備(queries 確認・記録確認・既存成果の読取り)までは自律可だが、本フェッチ実行・`papers/` 作り直し・新規 output dir 生成は人間承認必須** とする

## 今セッションまでに完了したこと (重複作業禁止)

### 1. verified_safe_learning 分野 — 完了・人間ゲート待ち

- 生成物: `D:\docs\verified_safe_learning_corpus_v2.staging`
  - 818 docs / 64 clusters / 72 SKILL.md
  - 判断記録: `_STAGING_META/DECISIONS.md`
- 検証済み: fallback 残 0 / frontmatter 破損 0 / Navigation リンク全有効
- live (`D:\docs\verified_safe_learning_corpus_v2`) は未作成のまま

### 2. self_evolving_agents 分野 — 完了・人間ゲート待ち

- 生成物: `D:\docs\self_evolving_agents_corpus_v2.staging` (全体 1,821 files)
  - 階層スキル: 807 docs / 61 clusters / 69 SKILL.md (top 8 + subclusters, depth 2)
  - source: `papers/` = v1 seed 31 + arXiv 新規 776
  - クエリ: `_STAGING_META/queries.txt` (16 queries, since 2019)
  - 判断記録・publish 手順: `_STAGING_META/DECISIONS.md`
- 検証済み: fallback 残 0 / Navigation リンク切れ 0 / `INDEX.md` のトップリンクを `/` に補正 / Document Types をユニーク docs で 807 に補正 / live 書込みゼロ
- raptor 内部中間生成物: `D:\tools\raptor\.claude\skills\corpus\self_evolving_agents_corpus_v2.staging`

## このセッションの最終到達点

- 追加で `corpus2skill` の stopword / resume 問題の再発防止修正を実装し、比較用 rerun を 1 回実施
- EXIT 整形ステップでは新規コード作業は行わず、ここで打ち止め。以降は記録更新のみ
- `self_evolving_agents` staging について、統合修正指示のうち以下は反映済み:
  - `INDEX.md` の Document Types 件数を 2421 → 807 に補正
  - `INDEX.md` のトップレベルリンクを `\` → `/` に修正
  - `D:\tools\raptor\packages\corpus2skill\writer.py` も同修正を反映し再発防止
  - `D:\docs\self_evolving_agents_corpus_v2.staging\_STAGING_META\DECISIONS.md` に semantic quality 低下、`OPENAI_API_KEY` 未設定、外部パス配置理由、人間ゲート 3 択を追記
  - `docs/SESSION_SUMMARY.md` は正本ポインタのみに縮退
- 未着手:
  - `self_evolving_agents` の semantic quality 改善 rerun を staging 名で本実行
  - 2 つの staging の publish
  - API キー復旧

### 今回追加で進めた内容 (2026-06-13 継続)

- `D:\tools\raptor\packages\corpus2skill\embedder.py`
  - `TfidfVectorizer(stop_words="english")` を有効化し、`the / and / of` 系ラベル汚染を抑制
  - stopword 除去で `empty vocabulary` になる退化コーパスは stopword 無しで再 fit する fallback を追加
  - `top_terms_from_matrix()` で 0 スコア語を返さないよう修正
- `D:\tools\raptor\packages\corpus2skill\clusterer.py`
  - ラベル候補を 12 語から選び、generic term / 数字主体 token を落として 3 語へ圧縮
  - 0 スコア語借用を止め、generic term しか残らない場合は doc-type fallback へ戻す
  - `_GENERIC_LABEL_TERMS` から `learning` を除外
- `D:\tools\raptor\packages\corpus2skill\writer.py` / `runner.py`
  - LLM 要約に `<!-- summary-source: llm -->` マーカーを付与
  - `runner._load_existing_summaries()` は marker 判定を `body` ではなく生 `content` に対して行うよう修正
  - `SKILL.md` の header 剥がしを `# label` + 任意空行だけを除去する形へ修正し、marker / legacy summary を落とさないよう改善
  - marker 無しの旧正規要約は `## Key Knowledge` + `## When Useful` / `## Navigation` を持ち、既知 fallback 文言に一致しない場合のみ legacy summary として再利用
- `D:\tools\raptor\packages\corpus2skill\tests\test_corpus2skill.py`
  - stopword 除去、empty vocabulary fallback、generic label 除去、doc-type fallback、resume round-trip、legacy summary 再利用、fallback 誤検知防止の回帰テスト追加
- 検証
  - `pytest D:\tools\raptor\packages\corpus2skill\tests\test_corpus2skill.py -q` → `31 passed`
  - 上記の編集・テスト・rerun 出力は **`D:\tools\raptor` 側の別リポジトリ管理物**。llcore 側 diff には含まれず、未コミットの可能性がある
- 比較用 rerun
  - コマンド: `py -3.11 D:\tools\raptor\raptor_corpus2skill.py --source D:\docs\self_evolving_agents_corpus_v2.staging\papers --name self_evolving_agents_corpus_v2_stopwordcheck --max-depth 2 --min-cluster-size 5 --max-clusters 8`
  - 出力: `D:\tools\raptor\.claude\skills\corpus\self_evolving_agents_corpus_v2_stopwordcheck`
  - 結果: 807 docs / 60 clusters / summaries 0
  - 改善確認: 旧 top-level `the / and / of`, `and / the / to`, `the / and / models` が、新 top-level `self / arxiv / recursive`, `evolutionary / search / arxiv`, `prompt / optimization / prompts`, `memory / long / multi`, `test / time / training` へ置換
  - ただし off-topic 混入は残存:
    - `self / arxiv / recursive` に X-ray / perovskite / Yang-Baxter
    - `evolutionary / search / arxiv` に天文 disk model
    - `scientific / autonomous / arxiv` に AI safety consensus / reflexions 年次レビュー
    - `test / time / training` に GAN / speech recognition / molecule optimization
  - 結論: **(b) stopword 除去 + query を絞って staging 再生成** の優先度がさらに上がった。単なる publish より rerun 推奨。
  - 留保: 学術 stopword (`fig`, `et`, `al` など) の追加は未実施。コーパス横断で過剰除去の危険があるため、query 絞り込みと実コーパス観察の後に別途判断する

## 重要インシデント / 制約

1. **ANTHROPIC_API_KEY org disabled は未解決**
   - `corpus2skill` の要約器は verified_safe_learning に続いて self_evolving_agents でも全失敗
   - `--resume-summaries` の marker round-trip 自体は修正済み。ただし現 staging は決定論的補完主体で、API 復旧なしに高品質 summary へ置換はできない
2. **self_evolving_agents 実行時は `OPENAI_API_KEY` も環境未設定**
   - API 代替が使えず、今回は docs title / 子クラスタ / 頻出語ベースの決定論的補完で `SKILL.md` を完成
3. **self_evolving_agents は recall 優先で query を広く張っている**
   - 元 staging では少数 leaf どころかトップレベルから stopword 主導ラベル (`the / and / of` など) が残り、最大クラスタにも off-topic 混在がある
   - `self_evolving_agents_corpus_v2_stopwordcheck` では top-level label は改善したが、query 由来の off-topic 混入は依然残る
   - 決定論的補完は「辿れる導線」であって、意味的に信頼できる cluster summary ではない
   - publish 前に「この precision / semantic quality でよいか」を人間レビューする前提

## 次の具体的な一手 (優先順)

1. **【人間】verified_safe_learning staging の publish 判断**
   - `D:\docs\verified_safe_learning_corpus_v2.staging`
   - rename 手順は `..._STAGING_META/DECISIONS.md` の publish 節
2. **【人間】self_evolving_agents staging の publish 判断**
   - `D:\docs\self_evolving_agents_corpus_v2.staging`
   - 次の 3 択を判断:
     - (a) このまま publish
     - (b) stopword 除去 + query を絞って staging 再生成
     - (c) off-topic docs / clusters を手動で除去
   - 追加材料:
     - stopword 修正後の比較 rerun では top-level label quality は明確に改善
     - ただし source query 起因の off-topic 混入は残るため、現時点の推奨は **(b)** 
3. **【人間】API キー対処**
   - `ANTHROPIC_API_KEY` org disabled の復旧
   - 必要なら `OPENAI_API_KEY` も環境投入
4. **【Claude・次セッション】人間判断が (b) precision 改善 rerun の場合**
   - `D:\tools\raptor\packages\corpus2skill\embedder.py` / `clusterer.py` の stopword 修正は適用済みなので、そのまま使う
   - rerun 前提の最小 runtime 検証は完了済み。追加の確認を重複させず、まず `queries` / 記録 / 既存成果の読み合わせまで進めてよい
   - `_STAGING_META/queries.txt` を絞る
   - **ここから先の本実行 (`papers/` 作り直し、別 output dir への fetch、`fetch_arxiv_topical.py` → `raptor-corpus2skill` 実行) は人間承認後**
   - broad query 由来の off-topic cluster を主に削る
   - 必要なら学術 stopword (`fig`, `et`, `al`) は rerun 前に小さく追加検証する。ただし過剰除去リスクがあるので後回し
5. **【Claude・次セッション】人間判断が (c) 手動除去の場合**
   - staging 内で off-topic docs / clusters の候補を列挙
   - 影響範囲を確認してから staging 側だけ編集

## 次セッション開始時の最短手順

1. `docs/next_plan.md` を開く
2. `docs/PROGRESS.md` で再開地点を確認する
3. 人間判断が未投入なら、新規実装ではなく `next_plan` の判断待ち項目から処理する
4. 人間判断が `(a)` なら各 staging の `_STAGING_META/DECISIONS.md` にある publish / rename 手順から開始する
5. 人間判断が `(b)` なら query 絞り込み rerun、`(c)` なら staging 側の手動除去へ進む
6. 人間判断なしで自律継続する場合でも、この EXIT 時点では **最小検証済みなのは query 汚染遮断まで**。`queries_refined_candidate.txt` の precision 改善効果は未検証なので、同じ確認を繰り返す必要はないが、本 rerun 後に before/after の混入率と recall 低下を必ず再評価する
7. `D:\tools\raptor` 側の `git status` を確認し、`libexec/raptor-rad-ingest` の `_ensure_utf8_io()` 差分を fetch/corpus2skill 修正と分離コミットすべきか判断する

## 今回 repo 内で更新した記録

- `docs/SESSION_SUMMARY.md`
- `docs/next_plan.md`
- `docs/PROGRESS.md`
- `docs/ARTICLE_SEEDS.md` に記事ネタ 2 件 append

## 今回この再開セッションで追加したこと (2026-06-13 継続 2)

- `CLAUDE.md` は `D:\projects\llcore` 配下に見当たらず、SESSION START の参照元は不在
- RAD 研究接地:
  - `D:\docs\self_evolving_agents_corpus` を grep し、既存軸が provable self-mod / coding agent / skill library / memory / AI Scientist / model merging にあることを再確認
  - `D:\docs\hacker_corpus_v2` は本件の query 設計材料としては有効ヒット薄
- off-topic 混入の再確認:
  - stopword 修正後 rerun (`..._stopwordcheck`) でも `self / arxiv / recursive` に perovskite / Yang-Baxter、`evolutionary / search / arxiv` に disk model、`prompt / optimization / prompts` に地質・医療 prompt 最適化、`test / time / training` に chemistry 系が混在
  - 汚染源は clusterer というより `queries.txt` の broad query 側と判断
- 追加準備:
  - `D:\docs\self_evolving_agents_corpus_v2.staging\_STAGING_META\queries_refined_candidate.txt` を新規作成
  - 方針は `ti:` + `LLM/agent` 条件 + `cat:` 制限で precision を上げること
  - `D:\tools\raptor\fetch_arxiv_topical.py` に query 来歴保存の最小修正を追加したが、本文メタ行に入れる設計は TF-IDF 汚染になるため後続セッションで是正対象になった

## 次セッションで人間判断が (b) の場合の具体化メモ

- 推奨手順:
  1. 既存 `papers/` は残したまま別 staging 名で fetch rerun
  2. `queries_refined_candidate.txt` を初期値に使い、`--per-query` は 60 のまま維持
  3. 生成された paper markdown の source-query comment と `Categories` を見て、なお broad な query だけ個別に再修正
  4. その後 `raptor_corpus2skill.py` を rerun し、top-level cluster の semantic drift を再評価
- disclosure:
  - `queries_refined_candidate.txt` は **改善候補** であり、まだ fetch rerun で検証していない
  - 主な tightening は `all:` から `ti:` への変更と LLM/agent 条件追加なので、precision は上がる可能性がある一方、title に語を含めない正規論文を取りこぼす recall 低下リスクがある
  - 改善判断は rerun 後の before/after 比較でのみ確定する
- 想定効果:
  - `model merging` 由来の astro / medical 混入、
  - `prompt optimization` 由来の vision / geology 混入、
  - `test-time training` 由来の chemistry / molecule discovery 側の混入、
  - `recursive self-improvement` 周辺の数理系ノイズ
  を現在より追跡・除去しやすくする

## 今回この再開セッションで追加したこと (2026-06-15)

- `CLAUDE.md` は repo 内に存在せず、前回記録どおり `docs/next_plan.md` / `docs/PROGRESS.md` を再開起点として継続
- RAD 研究接地を再確認:
  - `D:\docs\self_evolving_agents_corpus_v2` の既存軸は引き続き prompt evolution / Reflexion / AI Scientist / model merging / memory evolution / recursive self-improvement に整理済み
  - `D:\docs\hacker_corpus_v2` は今回の query 精密化には有効ヒット薄
- `self_evolving_agents_corpus_v2_stopwordcheck` の off-topic 例を再点検し、query 汚染源を具体化:
  - `Reflexion` 系 broad hit が `Superstructure reflexions in tilted perovskites` を混入
  - `AI Scientist` 系 broad hit が `TianJi-Environ` のような domain-specific science agent を混入
  - `prompt optimization` 系 broad hit が `Task-driven Prompt Evolution for Foundation Models` のような医療画像 prompt 最適化を混入
  - `test-time training` 系 broad hit が `MiGrATe` / `FineMedLM-o1` / `CoTBox-TTT` / `HyperWalker` など広い domain adaptation を混入
- `D:\docs\self_evolving_agents_corpus_v2.staging\_STAGING_META\queries_refined_candidate.txt` を追加修正:
  - `verbal reinforcement learning` と `Reflexion` をそれぞれ `agent` / `language model` / `LLM` 条件付きで分離
  - `prompt evolution` query を追加し、GEPA / Promptbreeder 系を残しつつ vision prompt 系を落としやすくした
  - `test-time training` は `ti:` 化 + `agent` / `language model` / `LLM` 条件で絞っており、molecule / medical 側の混入抑制はこの tightening に依存する。効果は未検証
  - `AI Scientist` は `agent` / `language model` / `LLM` 条件に加え、`research` / `discovery` の broad 条件を外して `autonomous` に限定
  - `recursive self-improvement` から広すぎる `all:AI` を除去し、`open-ended` から広すぎる `all:self` を除去
- まだ未実行:
  - refined query での本 fetch rerun
  - 新規 output dir 作成
  - `papers/` 再生成

## 環境メモ

- llcore ブランチ: `phase2a-trajectory-tube-gate` (過去タスク時点の環境メモ。本ファイル内の現在地は後段「LM recurrent 現在地」を正本とする)
- リポジトリはもともと dirty。今回この repo で触ったのは `docs/next_plan.md` / `docs/PROGRESS.md` で、query 候補の編集は repo 外 `D:\docs\...`。`assets/articles/llcore_landscape_real.svg` / `research/verified_lm_evolution/make_trajectory.py` / `.llterm/loop_ledger.jsonl` は別件 dirty

## 統合修正指示の反映メモ (2026-06-13)

- 反映:
  - `fetch_arxiv_topical.py` の query 来歴は本文メタ行ではなく HTML comment 保存へ変更済み。loader 側でも `Authors/Date/arXiv/URL/Categories/Source Query` 行と source-query comment を TF-IDF 入力前に除去するよう反映済み
  - `runner._strip_skill_header()` は H1 (`# `) 限定に直し、`## Overview` 始まりの legacy/manual SKILL.md を落とさないよう補正
  - `DECISIONS.md` には source-query 追跡の制約 (`fp.exists()` skip で旧 papers に遡及しない / 複数 query 命中時は最初の保存分のみ残る) を追記
  - rerun 前の最小ランタイム確認として「1 query fetch → 1件目視 → 検索式語がラベルに出ないこと確認」を追加し、実施済み
- 来歴確認:
  - `D:\tools\raptor\libexec\raptor-rad-ingest` の `_ensure_utf8_io()` 追加は今回タスクとは別筋の未記録差分として現存。relevant ではないため巻き戻さず、raptor 側でコミット分離が必要
  - `D:\tools\raptor` は llcore 外の別リポジトリで、ここで独立コミットまでは実施していない。コミット時は fetch/corpus2skill 修正と `libexec/raptor-rad-ingest` を分離すべき
- 最小ランタイム確認:
  - temp dir `C:\Users\puruy\AppData\Local\Temp\rad_query_sanity` に 1 query だけ fetch して markdown 生成を確認
  - 生成物には `<!-- source-query: ... -->` comment が入る一方、loader 後の TF-IDF top terms から `ti` / `cat` / `source` / `query` / `authors` / `categories` は消えることを確認
  - 同じ 2 docs に対する `_make_label(...)` は `soundnessbench / soundness / atmospheric` となり、検索式語がラベルに残らないことを確認
- 見送り:
  - `_is_informative_label_term()` の `3d/2d/5g` まで落とし得る regex 指摘は妥当だが、この corpus の直近 blocker ではないため今回は未着手

## 承認待ちメモ (2026-06-16)

- LM recurrent 比較の統合修正指示 #1 は、tracked artifact の重複 SVG を `git rm` で整理する削除操作を含むため、人間承認が必要
- 実測確認:
  - `docs/artifacts/lm_recurrent_pilot160.svg`
  - `docs/artifacts/lm_recurrent_pilot240.svg`
  - `docs/artifacts/lm_recurrent_pilot240_seed2026.svg`
  - `docs/artifacts/lm_recurrent_pilot240_seed7.svg`
  は `git hash-object` が全て `2cb4574abc14ed8fcd3eeac471a3cb45bdee7af7` で byte 同一
- 承認された場合の実施内容:
  1. 重複 SVG の tracked copy を削除し、summary/doc の参照先を共有 SVG へ統一
  2. `strict gate` → `unigram floor` の文言整理で済まない実体の重複を是正
  3. 変更後に `py -3.11 -m pytest tests/unit -k lm -q && py -3.11 -m mypy src/llcore/lm/ && py -3.11 -m ruff check src/llcore/lm/` を再実行
- 不承認なら、削除を伴わない範囲（文言・注記・将来方針）だけで整合を維持する
- 2026-06-16 追記:
  - `docs/LM_RECURRENT_PLAN.md` の到達点整理を commit しようとした時点では `git commit` が `.git/index.lock` で停止したが、再確認時には lock は消滅しており、**lock 解除の手動削除**は不要になった（duplicate SVG の物理削除承認は引き続き保留）
  - 次の local commit は **pathspec 限定**で行い、対象は `docs/LM_RECURRENT_PLAN.md` と `docs/next_plan.md` のみとする。`.llterm/loop_ledger.jsonl` は自動ログなので巻き込まない

## LM recurrent 現在地 (2026-06-16)

- 主作業ブランチは `feat/lm-recurrent`
- 進捗の正本:
  - 再開フロー = `docs/next_plan.md`
  - LM recurrent 実験内容 = `docs/LM_RECURRENT_PLAN.md`
  - tracked artifact / verdict = `docs/artifacts/lm_recurrent_*`
- 到達点:
  - head-to-head verdict packet (`docs/artifacts/lm_recurrent_verdict.md`) は完成
  - strongest claim は **「RWKV が最も再現性の高い候補」**
  - 根拠は `64/160` の 3 seed と `64/240` の 3 seed で raw PPL best / unigram floor pass を維持したこと
  - ただし GPT と Recurrent の相対順位は seed-sensitive のままで、full winner は未宣言
- 未解決:
  - duplicate SVG の **物理削除 (`git rm`)** は承認待ち
  - `.llterm/loop_ledger.jsonl` と `docs/status/` は無関係 dirty / untracked として継続除外
- 自律継続の境界:
  - LM recurrent 本体では、verdict packet 完成以降に**承認なしで進めるべき必須タスクは残っていない**
  - 次に動くのは、(a) duplicate SVG の物理整理に承認が下りたとき、または (b) 追加 seed / 追加 budget / 比較基準変更の新要件が入ったとき
