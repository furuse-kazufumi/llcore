# 記事素材: 強豪5社の手法 × FullSense/llcore の共通点(2026-06 解剖)

> **これは完成記事でなく濃い記事素材**(骨子+各社解説+共通点+比喩+honest 注記)。llterm が記事化する際の一次資源。
> 由来 = workflow `competitor-methods-article-material`(一次情報 deep-read → 共通点抽出 → honest 監査, 2026-06-18)。
> **honest 監査(verdict=採用可・条件付き)の必須修正を適用済み**(下記「★監査修正」参照)。
> llcore 側の全数値正本 = `docs/MEMORY_EFFICIENCY_FINDINGS.md` / `project_llcore_memory_efficiency_pivot`(memory)。
> 競合側の一次検証 = `docs/POSITIONING_VS_LLAMACPP.md` 脚注 + ARTICLE_SEEDS #46–50。

## ★監査修正(記事化時の絶対ルール)
1. **競合の数値は全て self-report / 二次情報 / 構造的帰結**。記事本文で裸の断定にせず必ず inline ラベル(`self-report`/`技報時点`/`二次・未照合`/`構造的帰結`)を付ける。**arXiv ID・日付は執筆前に arxiv MCP / WebFetch で実在確認**(モデル知識カットオフ後のため)。MangaFlow=arXiv:2605.28173 / PaddleOCR=arXiv:2606.03264 は第1検証で confirmed、**Cosmos=arXiv:2606.02800 は要再確認**。
2. **規模差 caveat を全フック直後に縫い込む**(§0 テーブルに依存させない)。芯フレーズ=「**同じ哲学に立つ / 手法を再導出した / 問題設定を独立に立てた、とは言える。でも『同等品質を出した』とは言わない**」。
3. **我田引水 2 件は格下げ済**: (a) Cosmos 2塔分業 ↔ llive オーケストラ = `shared_principle` ではなく **`loose_analogy`**(層内パラメータ分業 vs プロセス間分業=機構は無関係。対比として見せる方が誠実かつ面白い)。(b) MangaFlow 話者配置 ↔ bazue = **`shared_problem_framing`(未検証)**。「裏取りされた」は**禁止語**(bazue VLM 実機ベンチは未実行=我々はまだ解いても検証してもいない)。
4. **MangaFlow Count 100% は「決定的合成の構造的帰結であって生成精度ではない」**を同じ文に。**Gemma「26B 級」は dense/MoE 未確定**を併記。
5. **速度軸は我々ゼロ実証**(int8 は simulated quant=storage 圧縮のみ、速度未測)。「GPU で速度に化ける」は設計仮説と明記。**QAT のネガ側(2bit は QAT でも 97% gate 未達=82.9%)を成功談と対等の重みで**。

---

## 0. 射程と前提(記事冒頭の「正直な但し書き」= 信用の土台)

2026 年 6 月、近接時期に注目手法が出揃った — Google **Gemma 4 12B**、NVIDIA **Cosmos 3**、**MangaFlow**(東大+HKUST 広州)、Baidu **PaddleOCR-VL-1.6**、Nous Research **Hermes Agent**。解剖して自宅 PC の小実験群(FullSense の llcore/llive/llove/llmesh + 周辺 PoC)と照らすと、「**機能を足す**」でなく「**余計を削る・再利用する**」という同じ引き算の哲学が何度も現れた。

**規模が桁違いであることを最初に明示する**(これが記事の信用の土台 = honest disclosure の実演):

| 相手 | 規模 | 我々の対応物 | 規模 |
|---|---|---|---|
| Gemma 4 12B | 実 11.95B マルチモーダル LLM | llcore | tiny char-LM(1.36M–130M)PoC |
| Cosmos 3 | 16B–64B world model | llive 分業オーケストラ | プロセスレベル分業の PoC |
| PaddleOCR-VL-1.6 | 0.9B 文書解析 VLM(本番) | llcore 量子化アーク | char-LM の量子化 PoC |
| MangaFlow | 6 段エージェントパイプライン(本番) | manga-md-poc | L0 のみの低優先 spike |
| Hermes Agent | 約 19.6 万★・MIT・20+ プラットフォーム | llive 自己進化メモリ | 設計思想レベルの PoC |

---

## 1. フック案(feedback_articles_concept_hook 準拠)

### 案A:「巨人たちの引き算」(主フック推奨)
> 2026 年 6 月、巨大プレイヤーの最新の一手は揃って「機能を足す」ではなく「余計な前段を撤去してメモリを削る」だった。Google が 12B で 26B 級を半メモリで狙い(self-report)、NVIDIA が動画トークンを刈り、MangaFlow が再生成をやめて貼り込み、Hermes が会話をロードせず検索する。— そして自宅 CPU の小実験(llcore)も、いつのまにか同じ「引き算の設計哲学」の上に立っていた。**ただし相手は実 LLM、我々は tiny char-LM の PoC。同じ哲学に立つとは言えるが、同等品質ではない。**

### 案B:「自宅の物置で、マンション高層階の設計図を再導出する」(中盤の山)
> [reference_article_idea_inventory の「GPU 無し PC = 実家の物置 vs マンション高層階」]。大手が「QAT で int4 を bf16 並みに保つ」と一行で済ます部分を、我々は RTN→per-group→GPTQ→QAT と段階的に自前で再導出し、各段で「2bit はどこで壊れるか」を実測した。物置で高層階の図面を手描きしたら同じ結論に辿り着いた — **ただし図面は描けても、住める広さ(プロダクション品質)は出ない。再導出 ≠ 同等実装。**

### 案C:「ラングトンの蟻を眺めていたかもしれない、という疑い方」(締めの哲学)
> [reference_article_idea_inventory のラングトンの蟻 + 有限の猿定理]。強豪の派手な数字(半メモリで 26B 級、ベンチ首位、精度 100%)はほぼ全て self-report か自前ベンチ。我々は逆に、自分の好結果(「進化が実地形で 20/20 capability 勝利」)を疑い、meta-gate を足して **ARTIFACT(見かけの勝利)だと自己反証**した。見かけの創発が単純な正体に collapse する「ラングトンの蟻」を見抜く目を設計に組み込んだこと — それが巨人に勝てない我々の唯一の旗だ。

> 長尺構成: 案A を主フック → 案B を中盤の山 → 案C を締め。読者注意曲線(feedback_reader_attention_curve)= 驚き→技術の谷→哲学の着地。

---

## 2. 強豪の手法解説(技術者向け密度・用語グロス付き)

### 2-1. Gemma 4 12B — 「エンコーダを足さず撤去して LLM 本体に統合」する逆張り
**仕組み(かみくだき)**: 普通の VLM は画像理解に「画像専用翻訳機」= **ViT エンコーダ**(他サイズで約 550M)を前段に置く。音声も専用エンコーダ(約 300M)。Gemma 4 12B はこれを**丸ごと撤去**し、生パッチ・生波形を **lightweight linear layer**(単一行列積+位置埋め込み+正規化だけの薄い変換)で直接 LLM 埋め込み空間へ射影(encoder-free unified)。専用エンコーダ分のパラメータ・重みロード・前処理レイテンシが消える。
**用語グロス**: dense=全パラメータを使う構成(対 MoE)/ KV cache=過去トークンの Key/Value を貯める作業メモリ、文脈長に線形膨張 / hybrid 5:1 local/global attention=「近所だけ見る(窓 1024)」5 :「全体を見る」1、global 層だけ全長 KV / QAT=学習中に量子化を織り込み int4 でも bf16 近傍 / p-RoPE=次元一部だけ回転、local/global で theta 変える。
**メモリ実数(公式 docs/HF, `self-report`)**: BF16 26.7GB → SFP8 13.4GB → Q4_0(4bit) 6.7GB(GGUF 実測 ~6.98GB)。重み 6.7GB + 文脈 KV が 16GB 級に収まる。**「26B 級に迫る性能を半分以下のメモリで」(公式 blog, self-report・独立ベンチ未検証・比較対象 26B が dense/MoE か未確定)**。Apache 2.0。
**効く理由(3 層相乗)**: ①アーキ層=encoder-free で ~850M 削除 ②アテンション層=5:1+shared K/V+p-RoPE で長文 KV 構造圧縮 ③数値層=QAT で int4 を bf16 近傍のまま約 1/4。
**未確認(honest)**: 総学習トークン数・学習ハード・蒸留有無は公式 model card に not stated。音声エンコーダ完全撤去か lightweight 投影かは記述に揺れ。

### 2-2. NVIDIA Cosmos 3 — 「理解する塔」と「生成する塔」を 1 モデル内で分業
**仕組み**: 5 モダリティ(text/image/video/audio/action)を束ねる omnimodal world model。入力系列を ①reasoner tower(AR=次トークン予測の VLM、理解) ②generator tower(DM=拡散で雑音から彫り出す、生成)に割る。両塔は同じ骨格・joint attention・3D mRoPE を**共有**しつつ各層内は AR/DM が**別パラメータ**。繋がるのは joint attention のみ、条件付けは reasoner→generator の**一方向**。→「層内パラメータは別(専門化)/ attention・骨格は共有(統合)/ 条件付けは一方向(干渉防止)」の中間解。
**効率の鍵**: 推論専用なら reasoner だけ動かし生成塔を起動せず節約 / 量子化 BF16/FP8/**NVFP4(Blackwell で最大 2 倍速)** / **EVS**(冗長動画トークンを pruning)/ Qwen3-VL 重みから初期化(Nano 16B=8B+8B, Super 64B=32B+32B; Edge 2B は from scratch)。
**ベンチ**: Artificial Analysis で OSS の T2I/I2V 首位・RoboArena best policy(**技報執筆時点 self-report・自前ベンチ HUE 等**)。OpenMDW 1.1。
**未照合(honest)**: 技報 PDF(10MB 超)は取得失敗、損失関数・学習データ量・Edge 2B 数値は二次情報依拠で**一次未照合**。**arXiv:2606.02800 は要再確認**。

### 2-3. PaddleOCR-VL-1.6 — 「データを無差別に増やさず弱点を狙い撃ち」(★深掘りは API error で未完, 第1検証カード由来)
**仕組み(第1検証 confirmed, arXiv:2606.03264)**: 0.9B(NaViT 動的解像度エンコーダ + ERNIE-4.5-0.3B)。核心 = **region-aware data optimization**(学習データを無差別に増やすのでなく前版 1.5 の弱点領域 boundary-fragile/coverage-sparse/unreliable-supervision を特定して狙い撃ち補強)+ 段階的 post-training(**CPT→SFT→GRPO** 強化学習)。OmniDocBench v1.6=96.33%(**文書解析専用ベンチ・Baidu 自前測定**)。Apache-2.0。
**注**: 本 workflow の deep-read は API error で欠落。手法詳細を記事化する際は arXiv:2606.03264 の method 節を再 fetch して補完すること。

### 2-4. MangaFlow — 「不確実な生成」と「確実な構造処理」の境界を引き直す
**仕組み**: 「ストーリー→マンガ」を 1 枚生成でなく **6 段エージェントパイプライン**に分解(Planner→story section memory 構築→Layout agent[決定的射影 Π() で panel count/overlap を補正]→Panel agent[外部拡散 Gemini 2.5 Flash Image / FLUX.2 9B で描画]→ComposePage[決定的に貼り込み]→Text agent[吹き出し配置])。本体は**学習しない**オーケストレータ、描画は差し替え可能な外部 backbone に委任。
**2 つの効きどころ**: ①**決定的合成**=パネルを個別生成し機械的に貼り込むので Panel Count Acc/Layout IoU が**構造的に** 100%(Gemini)/97.1%(FLUX)。`構造的帰結であって生成精度ではない`(Direct baseline は Count Acc 27.94%/44.20% に崩壊、と対比)。②**story section memory**=一貫性を「長文プロンプトで毎回賭ける」から「参照を固定して再利用する明示条件」へ。ablation で memory を外すと Self-CSD 0.668→0.547。
**注(honest)**: ベンチは著者自作 meta-benchmark(本人「完全な manga dataset でない」明言)。**商用 mangaflow.studio は本論文と無関係の別物**。自動メトリクスは 1 回の生成ラン。

### 2-5. Hermes Agent — 「重みを触らず周辺の記憶を更新して育つ」
**仕組み**: 自己改善型 CLI エージェント。fine-tune せず**外部学習ループ**で育つ。①skill 自動生成(「5+ ツール呼び出し成功」「エラー回復」「ユーザー訂正」時に限り SKILL.md 保存=**無差別生成でなく明示トリガー**) ②skill 自己改善(`patch` 優先で token 効率) ③記憶永続化(MEMORY.md ~2,200字 + USER.md ~1,375字 の**固定バジェット**、超過時はサイレント破棄でなく**エラー返却で手動統合強制**) ④**FTS5 で過去会話をロードせず検索**(session_search 4,500×高速化, `release notes 由来`) ⑤Honcho user modeling(任意)。20+ ゲートウェイ、MIT、約 19.6 万★(**GitHub API 実測だが star の質=bot/campaign 比率は未検証**)。Desktop は既存 core の GUI ガワ。
**未読(honest)**: ソースコード本体・SQLite スキーマ・nudge プロンプト文言は未読。learning loop の有効性を示す独立ベンチ/査読はゼロ(汚染ループリスク指摘あり)。

---

## 3. 「我々との共通点」(記事の核・関係タイプと honest 併記)

### 3-A. 「引き算」グループ — メモリを削る設計
- **A-1 撤去・簡素化の哲学 [shared_principle]**: Gemma=エンコーダ撤去 ↔ llcore=量子化/mmap/定数状態で working-set を小さく。*honest: 共通項は「削る」という動詞で別レイヤ。方向性の一致であって Gemma 級統合アーキの実装ではない。*
- **A-2 量子化を「設計前提」に格上げ [shared_principle]**: Gemma/Cosmos=量子化を成立条件として内包 ↔ llcore=量子化を北極星中核に据え **capability-gate(top1 retention≥97%)を eval に fail-closed 配線**。*honest: 我々は「PPL-only gate は壊れた 2bit を PASS させる」危険を実証して gate 新設(11.9M の 2bit は top1 −13.5pp で半減破綻でも unigram gate PASS)。Gemma の int4=bf16 近傍のプロダクション品質には規模上届かない。*
- **A-3 量子化を自分の手で再導出 [we_rederive_theirs] ★山**: Gemma=「QAT で int4 を bf16 並み」を一言 ↔ llcore=RTN→per-group→GPTQ(自前 Hessian 誤差補償)→QAT(fake-quant+STE)を段階再導出。int8 約 3.9×圧縮・PPL 劣化 <0.1%。QAT capstone で multi_smoke 2bit が top1 retention **82.9%(QAT)vs GPTQ 12% / RTN 8%**。*honest(最重要・ネガ側を対等に): tiny char-LM の **2bit は QAT でも strict 97% cap-gate 未達(82.9%)= 同規模では制覇できないと自己反証**。手法を再導出して挙動を理解したのであって同等品質を出したのではない。3bit=PTQ 安全床。*
- **A-4 KV cache という共通の敵に別戦略 [we_diverge]**: Gemma=attention 工夫(5:1+shared K/V+p-RoPE)で線形膨張を**緩める** ↔ llcore=**定数状態 recurrent(RWKV/Mamba)**で膨張を**ゼロに**(harness 実測 T256→2048 で GPT peak ×2.65 vs recurrent ×1.00)。*honest: recurrent は capability で劣る可能性、我々自身 capability=NULL_TIE/NEGATIVE と認める。「定数状態が勝つ」のはメモリ軸限定。*
- **A-5 llama.cpp/GGUF 流 mmap という共通基盤 [shared_principle]**: Gemma=q4_0-gguf 等を用途別配布 ↔ llcore=int8 を src 推論パスへ配線(`Int8Linear`/`load_int8_model` mmap streaming)+ CLI で 5.5MB→1.5MB→mmap streaming で日本語生成。*honest: Gemma は成熟プロダクト、我々は PoC。我々の独自は「mmap load ΔRSS ~1.4MB 固定コスト=大モデルほど相対効果大」の実測知見。*
- **A-6 NVFP4 と「GPU で速度に化ける」二段構え [shared_principle]**: Cosmos=Blackwell NVFP4 で 2 倍速(実測)+ EVS ↔ llcore=設計指針「GPU で真の int8 GEMM(tensor core)で速度に化ける」。*honest(速度の非対称を hook に): Cosmos は実 GPU で実測、我々は **CPU PoC の設計仮説で真の int8 GEMM 速度は未測定(simulated quant・storage 圧縮しか実証なし)**。*

### 3-B. 「記憶は再生成でなく再利用」グループ
- **B-1 専門化しつつ統合(分業) [loose_analogy ★格下げ済]**: Cosmos=層内パラメータ分業(reasoner/generator、重み・KV 共有) ↔ llive=プロセス間分業(独立 AI CLI を Claude が指揮)。*honest: 機構は無関係(層内 vs プロセス間)。共有は抽象語のみ=無理に共通点化せず「対比」として見せる方が誠実かつ面白い。「同じ TRIZ 的分離思想」とは書かない。*
- **B-2 記憶を有界サイズで持ち越す [shared_principle(限定)]**: llcore 定数状態 recurrent(過去を固定長状態に lossy 圧縮畳み込み) ↔ MangaFlow story section memory は**別カテゴリ**(外部ファイルキャッシュ=サイズ非有界・lossless)。*honest: 「再利用」の意味が別物。原理として残すなら「定数/有界サイズで文脈を持ち越す」に限定。*
- **B-3 コンテキスト膨張を構造的に防ぐ規律 [shared_principle]**: Hermes=FTS5 でロードせず検索+固定バジェット超過エラー ↔ llcore=working-set を小さく予測可能に / llive=4層メモリ層別管理。*honest: 対象レイヤが違う(運用層 vs 推論層)。llive の管理機構が Hermes 並みに成熟かは未確認。*
- **B-4 決定的合成でレイアウトを構造保証 [shared_principle]**: MangaFlow=拡散にレイアウト遵守を期待せず決定的貼り込みで Count/IoU を構造保証 ↔ manga-md-poc=宣言的 DSL→決定的レンダラで自己完結 SVG(外部 http 参照ゼロ、Qiita/GitHub 投稿可)。*honest: MangaFlow は本番 6 段、manga-md-poc は L0 spike(ユーザー自身「お遊びレベル」)。Count 100% は構造的帰結。*
- **B-5 「話者≠中央の被写体」問題を独立に立てた [shared_problem_framing(未検証) ★格下げ済]**: MangaFlow Limitation「複雑パネル・stylized 顔の speaker localize は困難」 ↔ bazue index hard case=159(話者≠中央)。*honest: **「裏取りされた」は禁止語**。bazue VLM 実機ベンチは未実行=我々はまだ解いても検証してもいない。MangaFlow の Limitation はその問題設定の妥当性を傍証する程度。*

### 3-C. 「責任を持って育つ」グループ ★我々の差別化
- **C-1 重みを触らず周辺記憶を更新する自己改善 [our_differentiator]**: Hermes=skill/メモリ更新で育つ・skill 生成は明示トリガーで絞る ↔ llive=4層メモリ+派生集団進化、核心差別化は**成長を Approval Bus / HITL で責任化**(誤 skill 汚染を policy+SQLite ledger+@govern fail-closed deny で止める)+verified-plasticity gate。*honest(誇張しない): **Hermes も明示トリガーで責任化の意識を共有**=二項対立でなく**度合いの差**(承認を architecture に組み込むか vs トリガー条件で絞るか)。Hermes は約 19.6 万★の成熟プロダクト、llive は完成度・普及で大きく劣る。我々の強みは「成長の安全弁を architecture に組み込んだ」設計思想に限定(capability は llive も未実証)。*
- **C-2 honest disclosure を architecture/eval に配線 [our_differentiator] ★締め**: 全社=派手なベンチ主張の多くが self-report/自前ベンチ/技報時点 ↔ FullSense=「異常に良い結果は内訳を疑う」を運用規約に(llcore の進化 20/20 勝利を meta-gate で **ARTIFACT と自己反証**、verified-plasticity を NULL_TIE/NEGATIVE と honest 認定)。*honest(自分にも適用): **強豪の self-report が即「誇張・虚偽」ではない(再現待ち)**。大手も caveat を出す(MangaFlow「完全な benchmark でない」、Cosmos「技報時点」)。差は「honest disclosure を運用規約として配線したか」という**姿勢の差別化**であって「我々の数字が彼らより正しい」ではない。*

---

## 4. 13 側面タグ(feedback_daily_articles_policy)
特に厚い側面 = **技術設計・honest disclosure・業界比較・哲学・教訓**。
- Gemma encoder-free 撤去 → 技術設計 / 哲学(引き算) / 業界比較
- Gemma QAT を 4 段で再導出 → 技術設計 / 教訓 / honest disclosure
- KV cache 戦略の分岐(緩める vs ゼロ) → 技術設計 / 業界比較
- Cosmos 2塔分業 ↔ llive(対比) → 哲学 / TRIZ(★loose_analogy として)
- NVFP4「GPU で化ける」二段構え → 技術設計 / 未来予測
- MangaFlow 決定的合成 ↔ manga-md → 技術設計 / エコシステム
- 「話者≠中央」を独立に問題設定 → 教訓 / 認知科学(未検証注記必須)
- 記憶を有界サイズで持ち越す → 技術設計 / 認知科学
- Hermes ↔ llive 責任ある自己進化 → 業界比較 / 哲学(責任化) / 戦略
- honest disclosure を配線 → honest disclosure / 哲学 / 戦略
- メモリ効率が 2026 の主戦場 → 業界比較 / 戦略 / 未来予測
- 規模差を直視する → honest disclosure / 哲学

QIITA_GENERAL(非エンジニア向け)は「引き算の哲学」「責任を持って育つ」「規模差を正直に書く」の 3 点に絞ると刺さる。

---

## 5. 比喩候補(reference_article_idea_inventory 風・壊れる箇所を必ず添える)
| 比喩 | 使いどころ | どこで壊れるか |
|---|---|---|
| 実家の物置 vs マンション高層階 | 自宅 CPU で GPU の設計図を再導出。フックB | 図面は描けても住める広さ(プロダクション品質)は出ない。再導出 ≠ 同等実装。 |
| 引き算の彫刻(ミケランジェロ) | encoder 撤去/動画トークン刈り/再生成やめる=全社「削る」 | 削れば必ず良くなるわけでない。QAT 2bit のように削りすぎると壊れる床がある。 |
| ラングトンの蟻 | honest disclosure。見かけの創発が単純な正体に collapse。フックC・締め | self-report が必ず「嘘」ではない(再現待ち)。疑う対象は「数字」でなく「未検証であること」。 |
| 量子化 = 真空パック食品 | bf16→int4 は真空パック。GPU(電子レンジ)で真の int8 GEMM=温め直して速度に化ける | 限度超え(2bit)で中身が潰れる(cliff_then_flat)。 |
| オーケストラの指揮者と専門奏者 | Cosmos 2塔 ↔ llive オーケストラ(**対比**として) | Cosmos の奏者は同じ楽団のパート(層内)、llive は別の建物の別バンド(別プロセス)。距離が違う=だから loose_analogy。 |
| 図書館の蔵書検索 vs 全部持ち歩く | Hermes FTS5「ロードせず検索」 vs コンテキスト膨張 | 索引にない本(FTS5 ヒットしない文脈)は見つからない。検索品質依存。 |
| 下書きを使い回す漫画家 vs 毎回ゼロから | MangaFlow story section memory | LLM の参照キャッシュは drift しうる(ablation で 0.668→0.547 に留まる)。 |

> 1 記事に比喩は 3–5 個まで(feedback_article_break_points)。技術者向け=彫刻/真空パック/オーケストラ、非エンジニア向け=物置/図書館/下書き使い回し。

---

## 6. 横断テーマ(記事の背骨・どれか 1 本を主題に長編化可)
1. **メモリ効率が 2026 の主戦場**(ただし各社の「削る対象」は別レイヤ=量子化/エンコーダ撤去/決定的合成/検索。「最適化が主戦場」の同語反復に縮めず、各社の削る対象の違いで情報量を回復)。自宅 PC 制約が弱みでなく差別化軸になりうる時代背景。
2. **量子化は「後処理」から「設計前提」へ格上げ** — 大手の結論を鵜呑みにせず再導出する姿勢が honest disclosure と接続。
3. **「記憶は有界サイズで持ち越す/再利用」**(MangaFlow 外部キャッシュは lossless で別カテゴリと区別)。
4. **「不確実な生成 × 確実な構造処理」の境界設計** — FullSense は「責任所在を architecture に持ち込む」形で最も明示的に旗印化。
5. **自己改善は「重みを触らず周辺記憶を更新する外部学習ループ」へ収束** — FullSense は「育つ」を「責任を持って育つ」へ拡張(二項対立でなく度合いの差)。
6. **honest disclosure が最大の構造的差別化** — 「検証可能性と正直さを設計に組み込んだローカル・ファミリー OSS」は巨大プレイヤーが取りにくい空白地帯。
7. **規模差を直視することが信頼性の源** — 規模差の明記そのものが honest disclosure の記事上の実演。

---

## 7. 推奨記事構成(llterm の叩き台)
**技術者向け(QIITA_SUMMARY, 2–3 万字)**: フックA + §0 規模差テーブル → 強豪解説(§2) → 共通点 3 グループ(§3-A 引き算/§3-B 再利用/§3-C 責任化)→ 山=A-3(量子化 4 段再導出)・締め=C-2(honest 配線)→ honest 注記ブロック独立 → 比喩(彫刻/真空パック/オーケストラ)→ 結論「同じ哲学に立つとは言える。同等品質を出したとは言わない」。
**非エンジニア向け(QIITA_GENERAL)**: フックB(物置 vs 高層階)→「2026 年 6 月、巨人たちが揃って引き算を始めた」→「自宅の小実験が同じ発想に辿り着いた 3 場面(再導出/独立に問題設定/責任化)」→ 比喩(物置/図書館/下書き)→ 締め「正直に規模差を書くこと自体が差別化」。

---

参照: llcore 数値正本 `docs/MEMORY_EFFICIENCY_FINDINGS.md` / `docs/POSITIONING_VS_LLAMACPP.md` / 競合 seed `docs/ARTICLE_SEEDS.md` #46–50 / 比喩源 `reference_article_idea_inventory`(memory)。**bazue VLM ベンチは未実行(将来計画)を本文で必ず明記。**
