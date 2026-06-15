# next_plan (正本) — EXIT 時点の再開地点

> 最終更新: 2026-06-16 (LM recurrent は verdict packet 完成・loop_ledger 追跡解除を上位承認として優先)
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

## 今回 repo 内で更新した記録 (2026-06-13 時点)

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
  - `docs/artifacts/lm_recurrent_pilot120.svg`
  - `docs/artifacts/lm_recurrent_pilot160.svg`
  - `docs/artifacts/lm_recurrent_pilot160_seed2026.svg`
  - `docs/artifacts/lm_recurrent_pilot160_seed7.svg`
  - `docs/artifacts/lm_recurrent_pilot240.svg`
  - `docs/artifacts/lm_recurrent_pilot240_seed2026.svg`
  - `docs/artifacts/lm_recurrent_pilot240_seed7.svg`
  は `git hash-object` が全て `2cb4574abc14ed8fcd3eeac471a3cb45bdee7af7` で byte 同一
- duplicate 判定はファイルサイズではなく **内容ハッシュ一致** を正本とする。物理削除時に対象一覧を出す場合も、この hash 一致を基準に列挙する
- 承認された場合の実施内容:
  1. 削除対象 SVG への参照を Markdown / tests / docs 全体で grep し、共有 SVG への切替済みを確認する
  2. 重複 SVG の tracked copy を削除し、summary/doc の参照先を共有 SVG へ統一する
  3. `strict gate` → `unigram floor` の文言整理で済まない実体の重複を是正
  4. 変更後に `py -3.11 -m pytest tests/unit -k lm -q && py -3.11 -m mypy src/llcore/lm/ && py -3.11 -m ruff check src/llcore/lm/` を再実行
- 削除前の参照確認（2026-06-16 実施）:
  - duplicate SVG stem への現参照は `docs/next_plan.md`, `docs/artifacts/lm_recurrent_interim_summary.md`, `docs/artifacts/lm_recurrent_verdict.md`, `tests/unit/test_lm_artifacts.py`, `docs/LM_RECURRENT_PLAN.md` に存在
  - したがって `git rm` 前に、少なくとも上記 docs/tests の参照先を共有 SVG へ張り替えた後で grep 再確認する
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
  - `.llterm/loop_ledger.jsonl` は無関係 dirty として継続除外
  - `docs/status/` は `llterm_status.svg` の生成物と判断し `.gitignore` へ退避
- 自律継続の境界:
  - LM recurrent 本体では、verdict packet 完成以降に**承認なしで進めるべき必須タスクは残っていない**
  - 次に動くのは、(a) duplicate SVG の物理整理に承認が下りたとき、または (b) 追加 seed / 追加 budget / 比較基準変更の新要件が入ったとき
  - `.llterm/loop_ledger.jsonl` の tracking 方針変更（`.gitignore` + `git rm --cached` 等）は LM recurrent 本体とは別件の repo 衛生タスクとして扱い、必要なら別途承認付きで処理する
  - 2026-06-16 追記: `.llterm/loop_ledger.jsonl` は tracked のまま append-only dirty を再発させるため、方針候補は「ignore + `git rm --cached` で追跡解除」に収束している。これは index からの削除操作を含むため、人間承認後に別コミットで実施する
  - 2026-06-16 追記: 承認要求は **別々に扱う**
    1. duplicate SVG の tracked copy 削除 + 共有 SVG 参照への統一
    2. `.llterm/loop_ledger.jsonl` の追跡解除（`.gitignore` + `git rm --cached`、runtime append を今後の commit から分離）
    どちらも削除系/追跡解除系のため fail-closed で人間承認後にだけ実施し、**承認もコミットも分離する**
  - 2026-06-16 追記: 上位承認を仰ぐ優先順位は `loop_ledger` 追跡解除を先、duplicate SVG 物理削除を後とする。理由は前者が毎ターン working tree を dirty にし続けるため
  - 2026-06-16 追記: 今後の状態報告では「コード変更ゼロ」と「planning/doc 更新は別途あり得る」を分けて書く。working tree に `.llterm/loop_ledger.jsonl` の自動追記 dirty がある場合は、それを明示して誤読を避ける

## EXIT 再開ポインタ (2026-06-16)

- 新セッションは `docs/SESSION_SUMMARY.md` と本節から再開する
- LM recurrent 本体では承認なしに進める必須タスクは残っていない
- 次の具体的な一手は **`.llterm/loop_ledger.jsonl` 追跡解除の承認可否を確認すること**
- 承認が下りたら:
  1. `.gitignore` に `.llterm/loop_ledger.jsonl`（または `.llterm/`）を追加
  2. `git rm --cached .llterm/loop_ledger.jsonl`
  3. 上記のみを単独コミット
- その次に duplicate SVG 物理削除の承認確認へ進む
- 直近 gate は `py -3.11 -m pytest tests/unit -k lm -q && py -3.11 -m mypy src/llcore/lm/ && py -3.11 -m ruff check src/llcore/lm/` で exit `0`（`90 passed, 401 deselected` / `mypy success` / `ruff success`）

## 再開メモ (2026-06-16, 現セッション)

- `docs/SESSION_SUMMARY.md` / 本ファイルを再読し、再開地点は前回記録どおり **LM recurrent 本体は完了、残る承認待ちは duplicate SVG 物理削除 1 件のみ** と確認
- repo 直下には `CLAUDE.md` / `AGENTS.md` は存在しないため、再開時は `docs/next_plan.md` / `docs/PROGRESS.md` を主に参照する。ただし上位指示は引き続き優先し、global `C:\Users\puruy\.claude\CLAUDE.md` の規約も有効
- 統合修正指示の反映:
  - `.git/hooks` に有効フックはなく、repo 内検索でも `.llterm/loop_ledger.jsonl` を `git add` する自動再 tracked 経路は未検出。`tools/llterm_status.py` は ledger を読むだけで stage しない
  - ignore 粒度は単一ファイルではなく **`.llterm/` 単位** を採用する。将来 tracked に戻す設定ファイルが要る場合のみ negate パターンで例外化する
  - `docs/status/` は **既に `.gitignore` 済み** のため、今回の解除コミットに追加反映は不要
  - `.llterm/loop_ledger.jsonl` の監査上の扱いは、将来の誤解を避けるため **ユーザー判断待ちの分類** とする。今回の実施理由は「tracked append ノイズの分離」であり、「監査証跡として不要」とまでは断定しない
  - 現在ブランチは `feat/lm-recurrent`。`.gitignore` 変更が他ブランチへ merge されるまでは、未反映ブランチで同種 dirty が再発しうる
  - 不要指摘の扱い:
    - 「`docs/status/` の ignore 取りこぼし」は **既に `.gitignore` 済み** のため不採用
    - 「JSONL 末尾の `(truncated)` 由来の整合性懸念」は **diff 表示上の切り詰めであり実ファイル破損ではない** ため不採用
  - 実施結果:
    - commit `3d1f6ab` (`Stop tracking llterm runtime artifacts`) で `.gitignore` に `.llterm/` を追加し、`git rm --cached -f .llterm/loop_ledger.jsonl` を単独コミットで実施
    - コミット範囲は `.gitignore` と ledger の index 解除だけに限定し、`docs/next_plan.md` は未ステージのまま維持
    - 実ファイルの ledger は作業木に残しつつ ignore 下へ移し、以後の append-only 追記を commit ノイズから分離した
  - 承認済み・実施結果:
    - 2026-06-16 承認受領: ユーザーは **選択肢 1** を選び、`lm_recurrent_pilot160.svg` を canonical として残し、byte 同一の block_size=64 duplicate SVG 6 枚を削除して共有参照へ統一する方針を承認した。実施は option A（JSON は保持、test / docs / `git rm` を同一コミットで原子的に更新）とし、option B（JSON 同時削除）は一次データ破棄のため採らない
    - 2026-06-16 実施完了: `lm_recurrent_pilot120.svg`, `lm_recurrent_pilot160_seed2026.svg`, `lm_recurrent_pilot160_seed7.svg`, `lm_recurrent_pilot240.svg`, `lm_recurrent_pilot240_seed2026.svg`, `lm_recurrent_pilot240_seed7.svg` を削除し、`interim_summary.md` / `tests/unit/test_lm_artifacts.py` / `docs/LM_RECURRENT_PLAN.md` を canonical shared SVG 前提へ更新した。検証は `py -3.11 -m pytest tests/unit/test_lm_artifacts.py -q` = `10 passed`, `py -3.11 -m pytest tests/unit -k lm -q` = `91 passed, 401 deselected`, `py -3.11 -m mypy src/llcore/lm/` = success, `py -3.11 -m ruff check src/llcore/lm/` = success
    - 残る不可逆操作は **現時点では無し**。以下は今回実施前の検討メモ / 監査ログとして保持する
    - ※以下の数項目は `13bcc26` 実施前の監査スナップショットとして保持する。現状の canonical 化済み状態とは区別して読むこと
    - `docs/artifacts/lm_recurrent_interim_summary.md` では、byte 同一 SVG の tracked copy は **現時点では tracked のまま残しつつ、将来は canonical 化候補**として扱っている。今回の diff で追加したのは主に renderer 由来の説明部分であり、保持 / canonical 化方針そのものはそれ以前から置かれていた。削除するなら、この現状維持方針を撤回するかどうかを先に人間判断で確定する必要がある
    - 実測では tracked SVG 8 枚のうち **7 枚が byte 同一**。同一集合は `lm_recurrent_pilot120.svg`, `lm_recurrent_pilot160.svg`, `lm_recurrent_pilot160_seed2026.svg`, `lm_recurrent_pilot160_seed7.svg`, `lm_recurrent_pilot240.svg`, `lm_recurrent_pilot240_seed2026.svg`, `lm_recurrent_pilot240_seed7.svg` で、`lm_recurrent_pilot256_40.svg` のみ別 hash
    - fail-closed 整合のため、承認前に一度入れていた「interim index の全面共有参照化」と「test の canonical 解決」は巻き戻した。現時点の tracked artifacts は **全 run が各自の `.svg` を参照**し、保持路線（選択肢 2 / 3）でも自然に読める状態へ戻してある
    - `pilotXXX.md` 本体に SVG 参照は存在しないため、削除時の**主たる整合対象**は **interim index + `lm_recurrent_verdict.md` + `docs/LM_RECURRENT_PLAN.md` + 物理 SVG 群**。加えて `tests/unit/test_lm_artifacts.py` には各 stem の `.svg` 実在前提があるため、test 側の SVG カップリング改修も別途要する。特に `docs/LM_RECURRENT_PLAN.md` の保持方針文言（tracked copy を当面保持し、将来 canonical 化するなら `lm_recurrent_pilot160.svg` に統一する旨）は option 1 実施時に更新/撤回が必要
    - 削除を実行するなら canonical shared SVG は **`lm_recurrent_pilot160.svg` に統一**する。理由は `lm_recurrent_verdict.md` が既にこれを共有参照先として使っており、`pilot240_seed7` index 行も同じ canonical へ張り替えるのが最小差分だから
    - `tests/unit/test_lm_artifacts.py` は現時点では各 stem の物理 SVG を個別検証する状態を維持する。削除を実行する場合は、**同一コミット内で** test の canonical 許容化、interim index の共有参照統一、duplicate 6 枚の `git rm`、`lm_recurrent_verdict.md` の共有参照確認、LM gate 再確認を順に行う
    - 不可逆削除は個別・明示の承認が必要であり、包括的な「確認不要」指示では自動承認しない
    - 2026-06-16 追記: 承認質問は `pilot120` を含む block_size=64 の同一 SVG 7 枚を対象とする形へ再発行済み。現在はその選択回答待ちであり、回答受領までは削除・参照更新・gate 再実行のいずれも開始しない
    - 2026-06-16 追記: 上の巻き戻しにより、選択肢 2 / 3 でも repo 状態は矛盾しない。option 1 が選ばれた場合のみ、削除専用の参照更新・note 更新・test 更新を単一コミットへ束ねる
    - 2026-06-16 追記: canonical shared SVG の候補は `lm_recurrent_pilot160.svg` に統一した。これは**削除または将来の共有参照化を行う場合の候補**であり、現 tracked state を即座に共有参照へ変えるものではない
    - 2026-06-16 追記: docs に書いた「全 block_size=64 SVG が byte 同一、`pilot256_40` のみ別」という**観測事実**は、drift 防止のため `tests/unit/test_lm_artifacts.py` の guard test で固定した。一方、seed / `max_iters` / `batch_size` / `eval_iters` が SVG に効かない理由は **renderer のコード読解にもとづく説明**であり、guard test が直接その因果まで証明しているわけではない
    - 2026-06-16 追記: `interim_summary.md` の保持理由は、旧版を「撤回」したというより、**より honest な文言へ整理した** と捉えるのが正確。現在は「tracked のまま残しつつ、将来 duplicate tracked SVG は canonical 化候補とする」という位置づけで、`verdict.md` が shared family reference、`interim_summary.md` が run ごとの artifact inventory という役割差を明記している
    - 2026-06-16 追記: 現 working tree の差分は **docs のみ**で、内容は **選択肢 3（保持方針維持 + 注記整理）に対応する記録更新** に留まる。現 tracked state では **全 run が各自の `.svg` を参照したまま**であり、canonical 共有参照への統一や test 改修はまだ実施していない。`git rm` を伴う選択肢 1 はこの差分に混ぜず、承認後に **別コミット** で `test canonical 許容化 → duplicate SVG 削除 → LM gate 再確認` の順で行う
    - 2026-06-16 追記: `py -3.11 -m pytest tests/unit/test_lm_artifacts.py -q` は `10 passed` を再確認済み。これは **現行の docs-only 状態**に対する green であり、canonical 名統一や test 改修の完了を意味しない
    - 2026-06-16 追記: 追加統合指示により、**test 改修を伴わない素朴版の選択肢 1 は非推奨** と整理した。主因は、JSON を残したまま SVG だけ削除する計画が `tests/unit/test_lm_artifacts.py` の JSON↔SVG カップリングと構造衝突するため
    - 選択肢 1 を再開するなら、削除前に plan とコミット説明で次の分岐を先に確定する必要がある:
      - `(A)` test を decouple し、JSON stem ごとの SVG 実在要求を外す
      - `(B)` JSON も同時削除し、summary row / reproduction block / verdict json-link 系 test まで含めて直す
      現時点の推奨は `(A)`。いずれにせよ **別コミット** での実施が必要
    - リスクの非対称性:
      - `(A)` は byte 同一の duplicate SVG を削るだけで、**情報損失は限定的だがゼロではない**。canonical SVG は JSON から `_render_memory_curve_svg` で再生成できる一方、物理削除で失われるのは **その時点の renderer 実装が出した歴史的 on-disk バイト列**であり、再生成一致は renderer 実装が不変な限りでのみ期待できる
      - `(B)` は一次データ JSON の破棄を含み、seed 比較証拠を失うため非推奨
    - `(A)` を採る場合の必須要件:
      - render 等価性ガードは失わない。`test_tracked_recurrent_svgs_are_well_formed_xml` は **生存 SVG（canonical `pilot160` + `pilot256_40`）に対象を絞る形で維持**し、`svg_text == _render_memory_curve_svg(result)` の検証を残す
      - byte 同一性ガードの**性質は変わる**。現 `test_block64_memory_svg_hashes_match_and_256_proxy_differs` は tracked on-disk SVG 同士の直接比較で drift を検知しているが、option A では **各 block_size=64 JSON を再描画し、canonical SVG と文字列等価で一致すること**を検証する形へ作り替える。そのため `_BLOCK64_IDENTICAL_SVG_STEMS` の 7 stem タプル定義も更新対象に含める
      - `interim_summary.md` の 6 枚分 markdown SVG リンクは、test 緩和とは別に **張替または削除を独立タスクとして実施**する。dangling link を test 緩和で隠さない
      - duplicate SVG の `git rm` と `interim_summary.md` の 6 リンク張替は **同一コミットで原子的に行う**。`test_interim_summary_links_target_existing_tracked_artifacts` は markdown target の `.resolve().exists()` まで検証するため、分離すると gate が赤になる
      - `test_interim_summary_links_target_existing_tracked_artifacts` は各 stem の `./{stem}.svg` **文字列の本文存在自体も assert** しているため、option A では summary 本文のリンク張替/削除だけでなく **`_summary_svg_target` ヘルパ（または同等の test ロジック）を canonical 解決へ改修**する必要がある
      - option A で `test_tracked_recurrent_svgs_are_well_formed_xml` の対象を生存 2 枚へ絞ると、削除 stem 分の per-seed render 等価検証は直接は消える。その分は `test_block64_memory_svg_hashes_match_and_256_proxy_differs` を **各 block64 JSON の再描画が canonical SVG と文字列等価で一致する** 形へ作り替えることで回収する
    - 選択肢 1 の削除コミットで最低限同時改修が必要な test は 3 本:
      - `test_block64_memory_svg_hashes_match_and_256_proxy_differs`
      - `test_tracked_recurrent_svgs_are_well_formed_xml`
      - `test_interim_summary_links_target_existing_tracked_artifacts`
      1 本でも漏れると suite が赤になる
      - 詳細:
        - `test_tracked_recurrent_svgs_are_well_formed_xml` は `_tracked_pilot_stems()` が JSON glob 基準のため、削除 stem の `.svg` を読みに行って `FileNotFoundError` になる。option A では SVG 実在を前提に回す反復ソース自体を **生存 stem (`pilot160` / `pilot256_40`) に絞る** 必要がある
        - `test_block64_memory_svg_hashes_match_and_256_proxy_differs` は削除 stem を `read_bytes()` するため、`_BLOCK64_IDENTICAL_SVG_STEMS` の更新も必要
        - `test_interim_summary_links_target_existing_tracked_artifacts` は `_summary_svg_target` が `./{stem}.svg` 実在を前提にしており、index 張替と canonical 解決を同時に行わないと落ちる
    - `(B)` を採る場合の追加影響 test:
      - `test_tracked_recurrent_markdown_matches_json_summary_values`
      - `test_verdict_doc_recomputes_rwkv_claims_from_tracked_json`
      - `test_verdict_doc_representative_rows_match_tracked_json`
      既記の summary row / reproduction block / verdict json-link 系に加えて上記 3 本も落ちるため、影響範囲へ含める
    - `(B)` は **SVG の重複削除ではなく一次データ(JSON)の破棄**に踏み込む。seed 比較証拠の喪失を伴うため、honest disclosure の観点で **非推奨** と扱う
    - 2026-06-16 追記: `test_block64_memory_svg_hashes_match_and_256_proxy_differs` は **block_size=64 の memory curve が seed/max_iters に依らず byte 同一** であることの回帰ガードでもある。選択肢 1 で 6 枚削除する場合は、tracked on-disk bytes の直接比較という監査価値は後退する。その代わり、望ましくは **JSON 再生成ベースの同一性検証へ作り替えて保全**し、何が残り何が失われるかを honest disclosure で明記する
    - 2026-06-16 追記: option A でも、`assert svg_text == _render_memory_curve_svg(result)` が担保していた **削除 stem ごとの render 等価性ガード**は直接には後退する。JSON は残るため必要時に再生成・再検証はできるが、保持方針を撤回する理由（重複 tracked SVG を減らす）と、この監査継続性トレードオフはセットで記録する
    - 2026-06-16 追記: option 3 では `interim_summary.md` の canonical 表記を実ファイル名 `lm_recurrent_pilot160.svg` に揃えることを **本作業に含める**。verdict の shared family 参照が run 固有図ではなく block_size=64 共通図である旨の補足も、必要なら同系統の doc 整理として後続で別コミット化できる
    - 2026-06-16 追記: option 1 実施時は `interim_summary.md` の **index だけでなく Note 段落本文**（"These tracked copies remain in place for now" を含む保持文言）も更新対象に含める。`docs/LM_RECURRENT_PLAN.md` の保持方針文言と同様、撤回/更新が必要
    - 2026-06-16 追記: 承認質問側の前提も現行文言に合わせる。つまり、撤回対象は旧「監査継続性のため保持」ではなく、**per-run inventory として当面保持する方針**である
    - 2026-06-16 追記: 承認質問の分岐 A に含める原子的コミット対象は `interim_summary.md` / `tests/unit/test_lm_artifacts.py` / duplicate SVG 6 枚の `git rm` だけでは不足で、**`docs/LM_RECURRENT_PLAN.md` の保持方針文言の更新/撤回も必須** とする
    - 現時点の推奨:
      - 即時リスク回避なら **選択肢 3**（保持方針維持 + 注記/参照整理のみ）
      - 選択肢 1 は、上記 `(A)/(B)` 分岐と回帰ガード保全策を plan に織り込んだうえで、承認後に別コミットで実施
    - 不採用指摘:
      - 「byte-identical 不変条件の test が diff に無い」は **既に `tests/unit/test_lm_artifacts.py` に guard test を追加済み** のため不採用
      - 既存 commit `153c0f7` のメッセージ scope を広げる案は妥当だが、**amend が必要**になるため今回は未実施


---

## ★ユーザー判断 (2026-06-16, ccr 経由) — 人間ゲート回答

- **self_evolving_agents staging**: 採用 = **(b) stopword 除去 + query 絞りで再生成**（ユーザー承認・選択肢②）。そのまま publish / 手動除去 ではなく **再生成**方針。
  - **★ブロッカー**: corpus2skill 要約器は直近の試行で失敗しており、現時点でも高品質 summary へ置換できる状態か未確認。現セッションの環境観測では `ANTHROPIC_API_KEY` は **存在**する一方、`OPENAI_API_KEY` は未設定のまま。要約器の実可用性確認が先決。
  - 手順: API 実可用性確認 → stopword/query 調整で corpus2skill 再生成 → off-topic 混入を再確認 → publish 判断。
- 他の人間ゲート（verified_safe_learning publish / precision rerun 本実行 / API キー復旧）は **ccr 側バッファで保留中**（ユーザー未回答）。
