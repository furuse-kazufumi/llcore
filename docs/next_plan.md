# next_plan (正本) — EXIT 時点の再開地点

> 最終更新: 2026-06-16 (LM recurrent の canonical 化、loop_ledger 追跡解除、HD-1 viz 追加、rerun 準備メモ更新まで完了。残る人間ゲートは corpus staging 側。現状の作業木は主に記録更新で、`docs/ARTICLE_SEEDS.md` は append-only 追記が中心。`collect_research_seeds.py` 側では裸の `記事化` / `published` レグを除去済みだが、legacy shorthand `→ #...` / `→ 記事...` はまだ consumed 判定に残る。dirty の実体は `git status` を正とする)
> SESSION_SUMMARY.md は Stop hook で自動上書きされるため、**このファイルが再開の正本**。
> hook 非管理の再開ポインタ: `docs/PROGRESS.md`

## ★★ 再起動時の最優先指示 (2026-06-16, 古瀬さん→ccr 経由) — このブロックを最初に実行

**方針: 人間ゲート待ちで「待機」して idle 停止せず、ゲート不要の可逆作業を自律で前進させる**（「動いてるのか分からん」を避ける）。下の EXIT 節の「記録更新のみ」は前回 EXIT ターン向けで、本セッション(再起動)は新規作業を進めてよい。

### 状況の正本（古い記述に惑わされない）
- `ANTHROPIC_API_KEY` は **valid**（2026-06-16 確認済）。「org disabled」は 06-13 の旧観測で**解消済** → API はブロッカーでない。
- `self_evolving_agents` staging = **(b) stopword 除去 + query 絞りで再生成**方針（ユーザー承認済）。ただし本実行は下記の人間ゲート。

### いま自律で進めてよい（人間ゲート不要・可逆）— この順で
1. **recurrent LM スレッド継続（現主戦場）**: RWKV-4 / GatedRNN / gpt の head-to-head を詰める。verdict は『定数状態メモリで動作・能力は学習予算に敏感で未収束』を**超えて強く主張しない**（strict unigram gate 未通過のため）。measured/projected 分離・artifact↔json 整合・drift テストの規律を維持。
2. **P1: held-out PPL 改善**（block_size 拡大 / `--config p1` / データ追加）。過学習は dropout で抑える既知教訓を踏襲。
3. **P1: 3D で歩く** — 学習済みモデルの clean-room 3D ビューア（`model_viz.json` を自前 Apache-2.0 ローダで。bbycroft コード非依存で再実装）。
4. テスト緑 / mypy strict / ruff の維持。honest disclosure（異常に良い数字は内訳を疑う）。

### 人間ゲート — 明示 GO が来るまで絶対にやらない
- ❌ `self_evolving_agents` / `verified_safe_learning` の staging→live **publish**
- ❌ `self_evolving_agents` precision rerun の**本実行**（本フェッチ / `papers/` 作り直し / 新規 output dir / rename / 削除）
- ❌ **push** 全般（no-push 既定）/ submodule 改変 / DB drop / force-push / `--no-verify`
- ❌ 再ログイン・認証要求が出たら**継続せず停止**（人間待ち）
  → 上記を進めてよいのは next_plan に「**C4=承認**」等の明示 GO が入った時のみ。迷ったら **fail-closed（やらない側）**。

### 衛生・可視化
- `feat/lm-recurrent` の LM 無関係 dirty（loop_ledger / *.svg / PROGRESS / next_plan / make_trajectory.py）と raptor 側差分は commit 時に**別件として分離**。
- 各ループ末に `py -3.11 tools/llterm_status.py` で自走ステータスを SVG 化（`docs/status/llterm_status.svg`・seed=`tools/llterm_status_seed.json`）すると進捗が一目で見える。

### 記事フィードバック（FullSense 記事側へ・重要）
- **article-worthy な発見**（数値・honest disclosure・教訓・新規性・落とし穴）は `docs/ARTICLE_SEEDS.md` に**正規形式で append**:
  `### N. タイトル` ／ `- **気付き**: …` ／ `- **根拠**: …（正本へのポインタ）` ／ `- **側面**: …（13 側面）`。
  過去観測を後日 supersede する場合も、旧 entry を削ったり書き換えたりせず、新しい numbered seed を append して上書き関係を明示する。
  現行 collector には「統合前提 cluster を機械的に読み飛ばす」マーカーが無いため、同一論点を 1〜2 本へ圧縮したい場合は **deposit 前**に `###` 単位を絞っておく。deposit 後の #21〜#28 のような束は、記事ドラフト化までは機械的に個別 seed として扱われる。
  ※ 2026-06-16 に `D:\projects\fullsense\tools\collect_research_seeds.py` を観測確認済み。collector が通す parser 最小条件は「日付セッション `## YYYY-MM-DD` 配下」かつ「`###` 見出し + `**気付き**` または `**側面**` の同一行非空値」で、同日複数セクションは date 単位で併存集約される。ただし producer 契約としては引き続き `気付き` / `根拠` / `側面` を揃えて書く。`ARTICLE_SEEDS.md` は append-only を原則とする。consumed 判定の実 regex は **観測メモ** として、`→ 記事化: #NN` と legacy shorthand `→ #...` / `→ 記事...` を拾い、裸の `記事化` / `published` は consumed 判定に使わない状態だった。観測対象は repo 外の local dirty 作業木なので、この記述は**内部仕様の snapshot 依存メモ**であり、契約として再利用する前に現物を再取得すること。散文では parser / consumer が実際に拾うフィールド記法 `**気付き**:` / `**側面**:` や、消費マーカー `→ 記事化: #NN` / `→ #...` / `→ 記事...` を本文用途で流用しない。記事ドラフトの小見出しを `###` で混ぜない。
- FullSense 記事側（ccr）が `fullsense/tools/collect_research_seeds.py` で全 project の seed を `fullsense/docs/articles/INBOX_research_seeds.md` に集約 → 記事化する。記事化されたら元エントリに `→ 記事化: #NN` を追記（INBOX で ☑ 化）。

---

## EXIT 時点の現在地

- **コア作業 (`corpus2skill` 修正 + 比較 rerun) は完了済み**。この EXIT 整形ターンでは **新規作業を開始せず、記録更新のみ** 実施
- `Source Query` 汚染是正、legacy summary resume 修正、最小 runtime 検証までは完了
- 次セッションの最初の判断点は **`verified_safe_learning` publish** と **`self_evolving_agents` precision rerun 本実行** のどちらの人間ゲートを先に回収するか
- 不可逆操作 (`publish` / rename / push / 削除) は引き続き人間承認なしに実行しない
- **(b) precision 改善 rerun は、準備(queries 確認・記録確認・既存成果の読取り)までは自律可だが、本フェッチ実行・`papers/` 作り直し・新規 output dir 生成は人間承認必須** とする

### 次セッションの最短一手

1. `★ユーザー判断 (2026-06-16, ccr 経由)` を開き、人間ゲートの未回答を確認
2. `verified_safe_learning` publish か `self_evolving_agents` precision rerun 本実行のどちらを先に進めるか、人間判断を回収
3. rerun 実行が承認された場合のみ、`queries_refined_candidate.txt` の SHA256 `2AB6A443E70D7A58DDDCFFE4213BF0156960C48E89109245CC9C34F74D6B7D73` を再確認して fetch / 再生成へ着手
   - 監査注記: この SHA は repo 外 `D:\docs\self_evolving_agents_corpus_v2.staging\_STAGING_META\queries_refined_candidate.txt` の実体に依存する。**llcore repo 単体では再現・強制できず、CI/レビューだけでは gate を検証できない**ため、判断時は **保存先パス + 取得日時 + SHA** を必ずセットで読み、外部ファイルの再取得を前提に扱う。現時点の再確認値は **2026-06-16 10:19:33 +09:00 取得時点**のもの
   - 意図メモ: これは collector 観測メモとは違い、precision rerun の**入力固定**として意図的に保持している SHA。旧 `0E6C...` から `2AB6...` への更新は query tightening を反映した rerun ゲート更新であり、drift ではない

## 今セッションまでに完了したこと (重複作業禁止)

### 1. verified_safe_learning 分野 — 完了・人間ゲート待ち

- 生成物: `D:\docs\verified_safe_learning_corpus_v2.staging`
  - 818 docs / 64 clusters / 72 SKILL.md
  - 判断記録: `_STAGING_META/DECISIONS.md`
- 検証済み: fallback 残 0 / frontmatter 破損 0 / Navigation リンク全有効
- live (`D:\docs\verified_safe_learning_corpus_v2`) は **未作成ではない**。2026-06-16 再確認で、`SKILL.md` frontmatter の `note_count: 97` と `Get-ChildItem D:\docs\verified_safe_learning_corpus_v2 -Filter *.md` 実測 98 files は、**97 ノート + `SKILL.md` = 98 files と整合**する v1/live flat corpus の存在を示す。したがって v2 staging の publish は「新規作成」ではなく、既存 live v1 をどう移行/置換/併存させるかの判断を含む
- 構造差の追加確認 (2026-06-16): 既存 live v1 は flat corpus (`SKILL.md` 起点, 98 md files) だが、v2 staging は hierarchical corpus (`INDEX.md` + `metadata.json` + 8 top-level `cluster_*`, 72 `SKILL.md`, publish tree md 実測 891) 。ここで **818 は corpus doc 数、891 は `INDEX.md` / cluster `SKILL.md` を含む md ファイル総数**。**top-level entrypoint が `SKILL.md` → `INDEX.md` に変わる**ため、publish は内容差だけでなく利用者/ツールの参照前提変更を伴う
- ツール互換性の追加確認 (2026-06-16): publish 対象 `D:\docs\<topic>_corpus_v2` を直接前提化しているのは `D:\tools\raptor\libexec\raptor-rad-ingest` で、`D:/docs/<topic>_corpus_v2/SKILL.md` を deposit / reindex 読取対象として使う。したがって v2 staging を live 名へ**単純置換すると、少なくとも rad-ingest 経由の `RAD_INDEX.md` 再生成で `(no SKILL.md)` 化する退行** が起こりうる。publish するなら、少なくとも top-level `SKILL.md` 互換導線をどう保つかまで判断が必要

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
  - `D:\docs\self_evolving_agents_corpus_v2.staging\_STAGING_META\DECISIONS.md` に semantic quality 低下、`OPENAI_API_KEY` 未設定、外部パス配置理由、当時の人間ゲート候補を追記
  - `docs/SESSION_SUMMARY.md` は正本ポインタのみに縮退
- 未着手:
  - `self_evolving_agents` の semantic quality 改善 rerun を staging 名で本実行
  - 2 つの staging の publish

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
  - 上記の編集・テスト・rerun 出力は **`D:\tools\raptor` 側の別リポジトリ管理物**。llcore 側 diff には含まれず、現時点の `git -C D:\tools\raptor status --short` では `_bazue_article.json`, `_bazue_body_numbered.txt`, `_bazue_patch.py` の削除差分が残っている
  - 上記 `_bazue_*` 差分は self_evolving_agents rerun とは無関係なので、次の rerun / commit 束には混ぜない
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

1. **Anthropic 要約器は直近セッションで疎通確認済み**
   - 2026-06-16 の指揮者セッション実測で `ANTHROPIC_API_KEY` は present かつ valid と確認済み
   - したがって現時点の主ブロッカーは API 復旧ではなく、rerun 本実行の人間判断待ち
2. **self_evolving_agents 実行時は `OPENAI_API_KEY` が未設定**
   - ただし corpus2skill の rerun 自体は Anthropic 経路で進められるため、OpenAI 未設定は補助情報に留まる
3. **self_evolving_agents は recall 優先で query を広く張っている**
   - 元 staging では少数 leaf どころかトップレベルから stopword 主導ラベル (`the / and / of` など) が残り、最大クラスタにも off-topic 混在がある
   - `self_evolving_agents_corpus_v2_stopwordcheck` では top-level label は改善したが、query 由来の off-topic 混入は依然残る
   - 決定論的補完は「辿れる導線」であって、意味的に信頼できる cluster summary ではない
   - publish 前に「この precision / semantic quality でよいか」を人間レビューする前提

## 次の具体的な一手 (優先順)

1. **【人間】verified_safe_learning staging の publish 判断**
   - `D:\docs\verified_safe_learning_corpus_v2.staging`
   - rename 手順は `..._STAGING_META/DECISIONS.md` の publish 節
   - ただし 2026-06-16 再確認で `D:\docs\verified_safe_learning_corpus_v2` は既存 live v1（`note_count: 97` と実測 98 files が **97 ノート + `SKILL.md` = 98 files と整合**する flat corpus）と判明。よって判断点は「publish するか」だけでなく、**既存 live v1 を残す / 置換する / 別名退避して v2 へ切替える** のどれを採るかも含む
   - 比較メモ:
     - v1/live = flat corpus, top-level `SKILL.md`, 98 md files
     - v2/staging = hierarchical corpus, top-level `INDEX.md`, 8 top-level clusters, 72 `SKILL.md`, publish tree md 実測 891
     - `818 docs` は corpus document 数、`891 md` は `INDEX.md` / cluster `SKILL.md` を含む md ファイル総数
     - `D:\tools\raptor\libexec\raptor-rad-ingest` は `D:\docs\<topic>_corpus_v2\SKILL.md` 前提で `RAD_INDEX.md` を再生成する
     - したがって「置換」は path の差し替えだけでなく **entrypoint 互換性と rad-ingest 側 reindex 契約の断絶** を受け入れる判断になる
   - 現時点の移行オプション整理:
     - 最小リスク: 既存 live v1 は維持し、v2 は別名のまま保持
     - 中間案: v2 を live 名へ採用するが、top-level `SKILL.md` 互換 shim を新設して `INDEX.md` へ案内し、`rad-ingest` / `RAD_INDEX.md` の契約だけは維持する
     - 最大変更: v2 をそのまま置換し、必要なら `rad-ingest` 側を `INDEX.md` 対応へ改修する
   - 比較軸メモ:
     - 最小リスク案 = 破壊半径は最小だが、利用者は v1/live と v2/staging の二重系を手で見分け続ける必要がある
     - 中間案 = live 名は v2 に寄せつつ、entrypoint だけ明示的 shim で後方互換に固定できる。`rad-ingest` が読めない場合は `(no SKILL.md)` として fail-closed に露出しやすく、監視もしやすい
     - 最大変更案 = 入口契約とツール契約を同時に動かすため、publish 時の判断コストも巻き戻しコストも最も高い
   - 現時点の推奨順:
     - 人間ゲートを最も通しやすいのは中間案。理由は、v2 の live 採用を前進させつつ、hacker corpus 側の既存教訓どおり「互換性は曖昧な fallback ではなく、明示的 entrypoint に解決して fail-closed に監視可能にする」ため
   - 中間案の shim 最小仕様:
     - frontmatter に少なくとも `name:` / top-level `description:` / `collected:` を持たせる (`collected:` は frontmatter 内なら `metadata:` 配下でも top-level でも read 可。実装はファイル全体を走査するため、shim 本文に偶発的な `collected:` 行を書かない)
     - 本文冒頭に「verified safe learning corpus は hierarchical v2 へ移行した」旨の短い説明を置く
     - `INDEX.md` への明示リンクと、必要なら top-level clusters の代表リンクだけを置く
     - `rad-ingest` が reindex で使うのは top-level `description:` と `collected:` (`description:` は列 0 必須、`collected:` は strip 後マッチ)。`collected:` は frontmatter 内の top-level / `metadata:` 配下どちらでもよいが、実装はファイル全体を走査するため本文中の偶発一致は避ける。本文の最初の非見出し段落は `description:` 欠落時のフォールバックに留まるため、shim では主に人間向け導線とみなす
   - 中間案の shim 草案（そのまま置ける最小骨子）:
     - frontmatter:
       `name: verified_safe_learning_corpus_v2` / top-level `description: verified safe learning の RAD コーパス (hierarchical v2; INDEX 起点)` / `metadata:` 配下 `collected: <publish日>`
     - `rad-ingest` 契約として必須なのは **列 0 の `description:`** と、**frontmatter 内のどこか(top-level でも `metadata:` 配下でも可)の `collected:`**。実装はファイル全体走査なので本文側に偶発的な `collected:` 行を置かない。`name:` は SKILL.md の慣習上は推奨だが、reindex 契約そのものには不要
     - H1: `# verified safe learning corpus`
     - 本文 1 段落目の例:
       `> FullSense 内部 RAD 知識源。verified safe learning corpus は hierarchical v2 へ移行したため、閲覧の起点は \`INDEX.md\`。`
     - その直後に `- [INDEX.md](./INDEX.md)` を置けば、人間向け導線を満たしつつ、`rad-ingest` 側は frontmatter の `description:` / `collected:` を安定して読める
   - 中間案を採る場合の最小チェックリスト:
     - publish 前: top-level `SKILL.md` shim を staging 側で先に作り、`description:` が行頭にあること、`collected:` が frontmatter 内に存在すること、`INDEX.md` への導線があることを静的に確認する。これは **事前フィルタ** であり、publish 判断の最終根拠ではない
     - publish 前: `INDEX.md` への相対リンクが live 名へ移っても壊れないことを確認する
     - publish 前: isolated copy に対して `py -3.11 D:\tools\raptor\libexec\raptor-rad-ingest --reindex --docs-root <temp_docs_root>` を 1 回流し、共有 `D:\docs\RAD_INDEX.md` を触らずに **実消費者の読取結果** を確認する。ここで `verified_safe_learning_corpus_v2` 行が `(no SKILL.md)` / `-` に退行しないことを publish 可否の最終根拠にする
     - publish 実行直前: 旧 live `D:\docs\verified_safe_learning_corpus_v2` を退避コピーまたは退避リネーム（例: `.bak-YYYYMMDD-HHMMSS`）し、共有 `D:\docs\RAD_INDEX.md` も同じ粒度で退避する。rollback 行で言う「直前退避」はこの時点で作成する
     - publish 実行: staging を live 名へ昇格し、その後に本番 `py -3.11 D:\tools\raptor\libexec\raptor-rad-ingest --reindex` を実行する。これは **共有 `D:\docs\RAD_INDEX.md` を上書きする副作用つき再生成** なので、上記 isolated dry-run 合格後に限って進む
     - publish 後: 本番 `reindex` の結果 `verified_safe_learning_corpus_v2` 行が `(no SKILL.md)` / `-` へ退行したら、退避した旧 live corpus ディレクトリを即座に書き戻し、退避した `RAD_INDEX.md` も戻すか、shim 修正後に corpus / index の両方へ再 `--reindex` して復旧する。fail 状態のまま放置しない
     - publish 後: 人間導線として top-level `SKILL.md` から `INDEX.md` へ 1 hop で辿れることだけを確認し、旧 v1 の 97 ノート一覧を再現していないことは仕様どおりとして扱う
   - static gate の pass / fail:
     - pass = frontmatter フェンスちょうど 2 本、frontmatter が 1 行目から開始、frontmatter 内 `description:`、frontmatter 内 `collected:`、本文側に実在する `INDEX.md` への 1 hop 導線、本文側の余計な `collected:` なし、かつ isolated copy に対する `py -3.11 D:\tools\raptor\libexec\raptor-rad-ingest --reindex --docs-root <temp_docs_root>` で `verified_safe_learning_corpus_v2` 行が `(no SKILL.md)` / `-` に退行しない
     - fail = 上記のどれか 1 つでも欠ける、リンク先抽出が空になる、または isolated dry-run で `RAD_INDEX.md` 行が退行する。fail 時は `--reindex` / publish に進まない
     - 補足: この gate は実消費者 `_read_collected()` より厳格で、`collected:` を frontmatter 内に限定して要求する。意図的に fail-closed 側へ寄せている
   - publish 前の隔離チェック例:
     - `$lines = Get-Content <staging>\\SKILL.md; $fence = @(); for ($i=0; $i -lt $lines.Count; $i++) { if ($lines[$i].Trim() -eq '---') { $fence += $i } }`
       frontmatter フェンス位置を先に確定する。`$fence.Count -ne 2` なら検査を通さず警告扱いにし、fail-closed で止める。加えて `$fence[0] -ne 0` や `$fence[0]` より前の非空行も fail 扱いにし、frontmatter が 1 行目から始まることを要求する
     - `if ($fence[1] -le ($fence[0] + 1)) { Write-Warning 'frontmatter body missing'; <チェック失敗扱い> }`
       frontmatter 区間が空なら不合格とし、PowerShell の降順 range で逆順走査しないよう fail-closed に止める
     - `$lines[($fence[0]+1)..($fence[1]-1)] | Select-String -Pattern '^description:'`
       frontmatter 区間に列 0 の `description:` があることを確認
     - `$lines[($fence[0]+1)..($fence[1]-1)] | Select-String -Pattern '^\\s*collected:'`
       frontmatter 区間に `collected:` があることを確認
     - `if ($fence[1] -ge $lines.Count-1) { Write-Warning 'body missing after frontmatter'; <チェック失敗扱い> }`
       frontmatter がファイル末尾までで本文ゼロなら不合格とし、PowerShell の降順 range で逆順本文を誤って拾わないよう fail-closed に止める
     - `$body = $lines[($fence[1]+1)..($lines.Count-1)]; $inCode = $false; $codeFence = ([string][char]96) * 3; $bodyNoCode = foreach ($line in $body) { if ($line.Trim().StartsWith($codeFence)) { $inCode = -not $inCode; continue }; if (-not $inCode) { $line } }`
       本文区間を先に切り出し、frontmatter 区間を除外したうえで、Markdown のコードフェンス内行も導線検査の対象から外す
     - `$m = $bodyNoCode | Select-String -Pattern '\\]\\(<?(?<target>(?:\\./)?INDEX\\.md(?:#[^)>\s]+)?)>?\\)' | Select-Object -First 1; $indexTarget = if ($m) { $m.Matches[0].Groups['target'].Value }`
       本文側に、相対形 (`INDEX.md` / `./INDEX.md`) の `INDEX.md` 1 hop 導線が少なくとも 1 本あることを確認し、リンク先文字列を取り出す（`#anchor` や `<...>` 囲みは許容）
     - `if (-not $indexTarget) { Write-Warning 'INDEX link target not found'; <チェック失敗扱い> } else { Test-Path (Join-Path <staging> (($indexTarget -split '#', 2)[0] -replace '/', '\\')) }`
       `SKILL.md` に書かれたリンク先そのものが実在することを確認する。リンク抽出に失敗した場合は pass させず fail-closed で止める
     - `$bodyNoCode | Select-String -Pattern '^\\s*collected:'`
       frontmatter 終了後の本文側に、想定外の `collected:` 行が紛れていないことを確認。実害が出るのは frontmatter 側 `collected:` が欠けたときに限られるが、gate としては過少検知より過検知を許す
     - `# 前提: staging 側に top-level SKILL.md shim を先に作成してから実行`
     - `$tempDocsRoot = Join-Path $env:TEMP ('rad-dryrun-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + (Get-Random -Maximum 100000)); try { New-Item -ItemType Directory -Path $tempDocsRoot -Force | Out-Null; Copy-Item <staging_dir> (Join-Path $tempDocsRoot 'verified_safe_learning_corpus_v2') -Recurse; py -3.11 D:\tools\raptor\libexec\raptor-rad-ingest --reindex --docs-root $tempDocsRoot; Select-String -Path (Join-Path $tempDocsRoot 'RAD_INDEX.md') -Pattern 'verified_safe_learning_corpus_v2' } finally { if (Test-Path $tempDocsRoot) { Remove-Item $tempDocsRoot -Recurse -Force } }`
       isolated copy を **空の `<temp_docs_root>` 直下**に `verified_safe_learning_corpus_v2` 名で置き、共有 `D:\docs\RAD_INDEX.md` を汚さず **temp 配下にのみ書き込んで** 実消費者 `rad-ingest` の読取結果を 1 回だけ確認する。`--docs-root` は corpus ディレクトリ自身ではなく **その親ディレクトリ** を指す点に注意する。文字列検査はこの dry-run 前の事前フィルタであり、publish 可否の最終根拠はここに置く
   - 中間案の残留リスク:
     - `rad-ingest` 契約は守れても、既存利用者が top-level `SKILL.md` 一枚物を期待していた場合は UX が変わる
     - したがって人間ゲートでは「完全後方互換」ではなく「entrypoint 互換 + 本体導線の変更」を受け入れる判断だと明示する
     - shim 本文には水平線 `---` を置かない。`raptor-rad-ingest` の `_read_description()` は `---` 行で frontmatter 判定を再トグルするため、本文内 `---` は静的検査と実装解釈の両方を不安定にする
   - 人間ゲートでの選び分け基準:
     - 最小リスク案を選ぶ条件 = live 名を当面変えたくない、既存参照者の UX 変更を避けたい、v2 は比較検証用に別名保持でよい
     - 中間案を選ぶ条件 = live 名を v2 へ前進させたいが、`rad-ingest` / `RAD_INDEX.md` の entrypoint 契約は壊したくない
     - 最大変更案を選ぶ条件 = `rad-ingest` 側の `INDEX.md` 対応改修まで同時に着手でき、巻き戻しより構造統一を優先する
     - 現在の前提だと、最も説明責任を果たしやすい default は中間案。理由は、内容本体は v2 へ寄せつつ、互換性は shim + static gate で fail-closed に監視できるため
   - 次に出す確認ダイアログの順序メモ:
     - 第1問は「今どちらの人間ゲートを先に処理するか」を聞く。選択肢は `verified_safe_learning publish` と `self_evolving_agents rerun 本実行`
     - `verified_safe_learning` が選ばれた場合だけ、第2問で `最小リスク / 中間案 / 最大変更` の 3 択を出す
     - recommended は中間案だが、UI 上の並びは比較しやすさを優先して `最小リスク / 中間案 / 最大変更` の順に固定し、recommended 表記だけを中間案へ付ける
   - 次回そのまま使う `LLTERM_CHOICE` 下書き:
     - 第1問:
       ⟦LLTERM_CHOICE multi=false question="どちらの人間ゲートを先に処理しますか?"⟧
       1) verified_safe_learning publish
       2) self_evolving_agents rerun
       ⟦/LLTERM_CHOICE⟧
     - 第2問（verified_safe_learning が選ばれた場合）:
       ⟦LLTERM_CHOICE multi=false question="verified_safe_learning の migration 方式を選んでください"⟧
       1) 最小リスク
       2) 中間案
       3) 最大変更
       ⟦/LLTERM_CHOICE⟧
2. **【人間】self_evolving_agents rerun 本実行の判断**
   - `D:\docs\self_evolving_agents_corpus_v2.staging`
   - 3 択のうち **(b) stopword 除去 + query を絞って staging 再生成** は既にユーザー承認済み
   - 残る判断は、`papers/` 作り直しと新規 output dir 生成を伴う **precision rerun 本実行に着手してよいか** の 1 点
   - 追加材料:
     - stopword 修正後の比較 rerun では top-level label quality は明確に改善
     - source query 起因の off-topic 混入は残るため、本実行の推奨方針は引き続き **(b)** 
3. **【Claude・次セッション】人間判断が (b) precision 改善 rerun の場合**
   - `D:\tools\raptor\packages\corpus2skill\embedder.py` / `clusterer.py` の stopword 修正は適用済みなので、そのまま使う
   - rerun 前提の最小 runtime 検証は完了済み。追加の確認を重複させず、まず `queries` / 記録 / 既存成果の読み合わせまで進めてよい
   - `_STAGING_META/queries_refined_candidate.txt` は準備済み。まずこれを採用候補として使い、必要なら title 制約や category 制約の微調整だけを追加する
   - 入力固定メモ: `queries_refined_candidate.txt` の現スナップショットは SHA256 `2AB6A443E70D7A58DDDCFFE4213BF0156960C48E89109245CC9C34F74D6B7D73`。対象は repo 外 `D:\docs\self_evolving_agents_corpus_v2.staging\_STAGING_META\queries_refined_candidate.txt` で、repo 単体では検証不能。**2026-06-16 10:19:33 +09:00 取得時点**の値として記録し、rerun 本実行前に **保存先パス + 取得日時 + SHA** を再確認する
   - 補足: この SHA は query tightening 後の rerun 入力を固定するための意図的な gate 値で、旧 `0E6C...` から `2AB6...` への変化自体が更新対象だった。もっと強い監査性が要る場合は、将来この query file 自体または hash 対象 snapshot を repo 内へ取り込む
   - rerun 本実行前の追加ゲート: lightweight probe により `ti:Reflexion` は完全クエリでも flagship `2303.11366` を回収できる一方、`AI Scientist` は派生研究が多く flagship が順位埋没しうると判明した。そのため candidate に `ti:"The AI Scientist"` と `ti:"The AI Scientist-v2"` の専用行を追加したが、確認できたのは各専用行を単独で投げたときに flagship 1 件を回収できることまでで、query file 全体としての最終 recall / precision 改善はまだ未検証。`ti:Reflexion` への tightening も recall 側副作用が未検証のまま扱う
   - **ここから先の本実行 (`papers/` 作り直し、別 output dir への fetch、`fetch_arxiv_topical.py` → `raptor-corpus2skill` 実行) は人間承認後**
   - rerun コマンドの骨子は既に固定できる:
     - `py -3.11 D:\tools\raptor\fetch_arxiv_topical.py --query-file D:\docs\self_evolving_agents_corpus_v2.staging\_STAGING_META\queries_refined_candidate.txt --output <new_papers_dir> --per-query 60 --since 2019-01-01`
   - 次に出す確認ダイアログの順序メモ:
     - `self_evolving_agents` 側は yes/no ではなく、`rerun 本実行へ進む / 現状維持で publish 判断へ送る / query 再調整を継続` の 3 択で聞く
     - recommended は「即 publish」ではなく、まず rerun 本実行の可否だけを決めること。publish 判断は rerun 結果を見るまで後段に置く
   - 次回そのまま使う `LLTERM_CHOICE` 下書き:
     - ⟦LLTERM_CHOICE multi=false question="self_evolving_agents を次にどう進めますか?"⟧
       1) rerun 本実行へ進む
       2) 現状維持で publish 判断へ送る
       3) query 再調整を継続
       ⟦/LLTERM_CHOICE⟧
     - その後 `py -3.11 D:\tools\raptor\raptor_corpus2skill.py --source <new_papers_dir> --name <new_staging_name> --max-depth 2 --min-cluster-size 5 --max-clusters 8`
   - broad query 由来の off-topic cluster を主に削る
   - 必要なら学術 stopword (`fig`, `et`, `al`) は rerun 前に小さく追加検証する。ただし過剰除去リスクがあるので後回し
4. **【Claude・次セッション】人間判断が (c) 手動除去の場合**
   - staging 内で off-topic docs / clusters の候補を列挙
   - 影響範囲を確認してから staging 側だけ編集

## 次セッション開始時の最短手順

1. `docs/next_plan.md` を開く
2. `docs/PROGRESS.md` で再開地点を確認する
3. 人間判断が未投入なら、新規実装ではなく `next_plan` の判断待ち項目から処理する
4. 人間判断が `(a)` なら各 staging の `_STAGING_META/DECISIONS.md` にある publish / rename 手順から開始する
5. 人間判断が `(b)` なら query 絞り込み rerun、`(c)` なら staging 側の手動除去へ進む
6. 人間判断なしで自律継続する場合でも、この EXIT 時点では **最小検証済みなのは query 汚染遮断まで**。`queries_refined_candidate.txt` の precision 改善効果は未検証なので、同じ確認を繰り返す必要はないが、本 rerun 後に before/after の混入率と recall 低下を必ず再評価する
7. `D:\tools\raptor` 側の `git status` は `_bazue_*` 3 件削除のみ。self_evolving_agents rerun とは無関係なので、次の rerun / commit 束には混ぜない

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

- llcore ブランチ: `feat/lm-recurrent` (本ファイル内の現在地は後段「LM recurrent 現在地」を正本とする)
- リポジトリの現状は記録更新のみ。query 候補の編集は repo 外 `D:\docs\...` で行っており、この repo の rerun 準備メモとは別管理。dirty の実体は都度 `git status` を正とする

## 統合修正指示の反映メモ (2026-06-13)

- 反映:
  - `fetch_arxiv_topical.py` の query 来歴は本文メタ行ではなく HTML comment 保存へ変更済み。loader 側でも `Authors/Date/arXiv/URL/Categories/Source Query` 行と source-query comment を TF-IDF 入力前に除去するよう反映済み
  - `runner._strip_skill_header()` は H1 (`# `) 限定に直し、`## Overview` 始まりの legacy/manual SKILL.md を落とさないよう補正
  - `DECISIONS.md` には source-query 追跡の制約 (`fp.exists()` skip で旧 papers に遡及しない / 複数 query 命中時は最初の保存分のみ残る) を追記
  - rerun 前の最小ランタイム確認として「1 query fetch → 1件目視 → 検索式語がラベルに出ないこと確認」を追加し、実施済み
- 来歴確認:
  - `D:\tools\raptor\libexec\raptor-rad-ingest` の `_ensure_utf8_io()` 差分は現 `git status` には存在しない。現在の外部 dirty は `_bazue_*` 3 件削除のみで、この rerun 準備メモからは対象外とする
  - `D:\tools\raptor` は llcore 外の別リポジトリで、ここで独立コミットまでは実施していない。次に raptor 側を触るときも、現存する `_bazue_*` 削除差分は self_evolving_agents rerun 束へ混ぜない
- 最小ランタイム確認:
  - temp dir `C:\Users\puruy\AppData\Local\Temp\rad_query_sanity` に 1 query だけ fetch して markdown 生成を確認
  - 生成物には `<!-- source-query: ... -->` comment が入る一方、loader 後の TF-IDF top terms から `ti` / `cat` / `source` / `query` / `authors` / `categories` は消えることを確認
  - 同じ 2 docs に対する `_make_label(...)` は `soundnessbench / soundness / atmospheric` となり、検索式語がラベルに残らないことを確認
- 見送り:
  - `_is_informative_label_term()` の `3d/2d/5g` まで落とし得る regex 指摘は妥当だが、この corpus の直近 blocker ではないため今回は未着手

## 承認待ちメモ (2026-06-16)

> ✅ 完了済み (`13bcc26`) — 本節は duplicate SVG canonical 化の**実施前に潰した論点を残す監査ログ**。現 HEAD では SVG 共有参照統一・test 改修・`git rm` まで完了している。

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
  - `docs/LM_RECURRENT_PLAN.md` の到達点整理を commit しようとした時点では `git commit` が `.git/index.lock` で停止したが、再確認時には lock は消滅しており、**lock 解除の手動削除**は不要になった（これは duplicate SVG 物理削除がまだ保留だった時点の監査ログ）
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
- 現状態:
  - duplicate SVG の **物理削除 (`git rm`)** は `13bcc26` で完了済み
  - `.llterm/loop_ledger.jsonl` の追跡解除は `3d1f6ab` で完了済み
  - `docs/status/` は `llterm_status.svg` の生成物と判断し `.gitignore` へ退避済み
- 自律継続の境界:
  - LM recurrent 本体では、verdict packet 完成以降に**承認なしで進めるべき必須タスクは残っていない**
  - 次に動くのは、追加 seed / 追加 budget / 比較基準変更の新要件が入ったとき
  - `.llterm/loop_ledger.jsonl` の tracking 方針変更は LM recurrent 本体とは別件の repo 衛生タスクとして扱われ、現在は実施済み
  - 2026-06-16 追記: 今後の状態報告では「コード変更ゼロ」と「planning/doc 更新は別途あり得る」を分けて書く。working tree に `.llterm/loop_ledger.jsonl` の自動追記 dirty がある場合は、それを明示して誤読を避ける

## EXIT 再開ポインタ (2026-06-16)

- 新セッションは `docs/SESSION_SUMMARY.md` と本節から再開する
- LM recurrent 本体では承認なしに進める必須タスクは残っていない
- 次の具体的な一手は **`verified_safe_learning` publish 判断または `self_evolving_agents` precision rerun 本実行の判断を回収すること**
- `loop_ledger` 追跡解除と duplicate SVG 物理削除は **どちらも完了済み** なので、この再開ポインタでは新たな承認対象ではない
- 直近 gate は `py -3.11 -m pytest tests/unit -k lm -q && py -3.11 -m mypy src/llcore/lm/ && py -3.11 -m ruff check src/llcore/lm/` で exit `0`（`91 passed, 401 deselected` / `mypy success` / `ruff success`）

## 再開メモ (2026-06-16, 現セッション)

- `docs/SESSION_SUMMARY.md` / 本ファイルを再読し、再開地点は **LM recurrent canonical 化まで完了済み、残る人間ゲートは `verified_safe_learning` / `self_evolving_agents` 側** と確認
- repo 直下には `CLAUDE.md` / `AGENTS.md` は存在しないため、再開時は `docs/next_plan.md` / `docs/PROGRESS.md` を主に参照する。ただし上位指示は引き続き優先し、global `C:\Users\puruy\.claude\CLAUDE.md` の規約も有効
- global `C:\Users\puruy\.claude\CLAUDE.md` の SESSION START 系規約を再確認し、報告構造・fail-closed・`py -3.11` / `rtk` 規約を本セッションでも継続適用すると確認
- RAD 研究接地として `D:\docs\self_evolving_agents_corpus_v2` / `D:\docs\hacker_corpus_v2` を再 grep し、既存差別化軸が引き続き memory / Reflexion / AI Scientist / model merging / recursive self-improvement にあることを再確認。`hacker_corpus_v2` は今回も query 精密化の直接材料は薄い
- `llcore` 作業木は再開時点で **clean ではなく**、記録更新の未コミット差分がある状態として扱う。固定的なファイル名列挙はせず、実体は都度 `git status` を正とする
- `self_evolving_agents` rerun 準備を追加で前進:
  - `queries_refined_candidate.txt` を見直し、既知ノイズに対応して `Reflexion` を `ti:` 条件へ tightening、`AI Scientist` に `scientific discovery` / `agentic tree search` 条件を追加
  - 目的は、既知の perovskite / domain-specific science agent 混入を query 段階で少しでも減らすこと。まだ fetch rerun 未実行なので **precision 改善は未検証**
  - 追加の lightweight probe として `fetch_arxiv_topical.py --query ... --count 5/20` を temp dir に対して実行し、`ti:Reflexion` は完全クエリでも flagship 本体を回収できる一方、`AI Scientist` は派生研究が多く `The AI Scientist` / `The AI Scientist-v2` が埋もれうることを確認。これに合わせて candidate query に flagship 専用行を追加したが、**専用行追加後の query file 全体としての改善は未検証**
  - 今回の query 設計教訓は `docs/ARTICLE_SEEDS.md` に article seed として追記済み。テーマは「flagship 回収 probe の必要性」と「query 1 行ではなく query file 全体を評価単位にすべき」
- `verified_safe_learning` 側の前提を再確認:
  - `D:\docs\verified_safe_learning_corpus_v2` は live 未作成ではなく、`SKILL.md` frontmatter 上 `note_count: 97` と `Get-ChildItem ... -Filter *.md` 実測 98 files は **97 ノート + `SKILL.md` = 98 files と整合**する v1/live flat corpus の存在を示す
  - 既存 live v1 は flat `SKILL.md` 起点、staging v2 は hierarchical `INDEX.md` 起点で、入口の型そのものが異なる
  - `D:\tools\raptor\libexec\raptor-rad-ingest` は live corpus の top-level `SKILL.md` を前提に `RAD_INDEX.md` を再生成するため、`INDEX.md` 起点の v2 をそのまま live 名へ置くと少なくとも rad-ingest 側で不整合になる
  - `rad-ingest` が reindex で実際に使うのは top-level `SKILL.md` の frontmatter (`description:` / `collected:`) で、本文側の導線は主に人間向けである。したがって完全置換より **`INDEX.md` へ案内する薄い `SKILL.md` shim** を併設する方が変更半径は小さい
  - したがって中間案は「旧 v1 の 97 ノート一覧を top-level に残す」ことではなく、**top-level だけ互換にして本体は `INDEX.md` 以下へ委譲する adapter** と捉えるのが正確
  - staging v2 (`818 docs / 64 clusters / 72 SKILL.md`) を publish する場合、単純 rename ではなく **既存 live v1 をどう扱うか** の人間判断が必要。なお `818` は corpus doc 数、`891` は `INDEX.md` / cluster `SKILL.md` を含む md 総数
  - 現時点で raptor 内部 live 相当の `D:\tools\raptor\.claude\skills\corpus\verified_safe_learning_corpus_v2` は未存在なので、移行対象は主に `D:\docs\...` 側
  - この「entrypoint 契約を壊さない migration」が論点だという教訓は `docs/ARTICLE_SEEDS.md` に seed #21 として追記済み
  - 現在の working tree には `(A) rerun query/SHA + dirty 記録更新` と `(B) article seed 追加 + 記事フィードバック節/再開メモ` が未コミット状態で混在している。具体的な dirty の実体は固定列挙せず、都度 `git status` を正とする
  - ただし commit 時の分離メモとして、`記事フィードバック` 節と `再開メモ (2026-06-16, 現セッション)` のような運用/再開メモ差分は、rerun query/SHA 追記と混ぜず **別件コミットに分離** する
  - 1 ファイル内に混在しているので、commit 時は `git add -p docs/next_plan.md docs/ARTICLE_SEEDS.md docs/SESSION_SUMMARY.md docs/PROGRESS.md docs/LM_RECURRENT_PLAN.md` で hunk 単位 staging を使う前提にする
  - 最低粒度の束分けは `(A) rerun query/SHA + dirty 記録更新` と `(B) article seed 追加 + 記事フィードバック節/再開メモ`。cherry-pick / 巻き戻し / 監査はこの単位で扱う
  - `docs/ARTICLE_SEEDS.md` については、append-only 追記 (#19〜#30) を他の再開メモ差分と論理上分離して扱う。コミット時は `git add -p` で束を意識して切る
  - #30 が示す append-only 方針（#17 の旧形式 supersede 注記は残したまま、以後は numbered seed の append に統一する）は、commit message にも明記して監査時の誤読を防ぐ
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
    - 2026-06-16 追記: この承認待ちフローはその後ユーザーが **選択肢 1** を選び、`13bcc26` で canonical 化まで完了した。以下の bullet は、実施前にどの論点を潰したかを残す監査ログとして保持する
    - 2026-06-16 追記: 現在の repo 状態は canonical 化・test 改修・LM gate 再確認まで反映済みであり、選択肢 2 / 3 の分岐は **履歴上の検討過程** である
    - 2026-06-16 追記: canonical shared SVG の候補は `lm_recurrent_pilot160.svg` に統一した。これは**削除または将来の共有参照化を行う場合の候補**であり、現 tracked state を即座に共有参照へ変えるものではない
    - 2026-06-16 追記: docs に書いた「全 block_size=64 SVG が byte 同一、`pilot256_40` のみ別」という**観測事実**は、drift 防止のため `tests/unit/test_lm_artifacts.py` の guard test で固定した。一方、seed / `max_iters` / `batch_size` / `eval_iters` が SVG に効かない理由は **renderer のコード読解にもとづく説明**であり、guard test が直接その因果まで証明しているわけではない
    - 2026-06-16 追記: `interim_summary.md` の保持理由は、旧版を「撤回」したというより、**より honest な文言へ整理した** と捉えるのが正確。現在は「tracked のまま残しつつ、将来 duplicate tracked SVG は canonical 化候補とする」という位置づけで、`verdict.md` が shared family reference、`interim_summary.md` が run ごとの artifact inventory という役割差を明記している
    - 2026-06-16 追記: 上の「docs-only / 選択肢 3 相当」は **`13bcc26` 実施前** の状態メモである。現 HEAD では canonical 共有参照への統一・test 改修・duplicate SVG 削除はすべて実施済み
    - 2026-06-16 追記: `py -3.11 -m pytest tests/unit/test_lm_artifacts.py -q` の `10 passed` は、canonical 名統一と test 改修を含む **削除後状態** でも再確認済み
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
  - 2026-06-16 追記: 現セッションの疎通確認で `ANTHROPIC_API_KEY` は **存在かつ有効**、`OPENAI_API_KEY` は未設定と確認した。したがって API 復旧は現時点の主ブロッカーではない。
  - **現ブロッカー**: rerun 方針そのものは承認済みだが、`papers/` 作り直しを伴う **precision rerun 本実行** は ccr 側バッファの保留項目に残っている。実行前にこの点の人間判断を明示的に回収する。
  - 手順: precision rerun 本実行の人間判断回収 → stopword/query 調整で corpus2skill 再生成 → off-topic 混入を再確認 → publish 判断。
- 他の人間ゲート（verified_safe_learning publish / precision rerun 本実行）は **ccr 側バッファで保留中**（ユーザー未回答）。API キー疎通自体は 2026-06-16 の指揮者セッション実測で確認済み。


> 訂正 (2026-06-16, ccr): 先の『self_evolving_agents 再生成は API キー復旧が先決』は誤り。ANTHROPIC_API_KEY は 06-16 に valid 確認済。唯一のゲートは rerun 本実行の人間承認。
