# 記事マスターシードバンク — 強豪 × llcore(2026-06)

> **llterm 用の記事ネタ集**。56 本生成 → 編集長が **26 本(S7 / A11 / B8)+ 連載構成案 2 案**に統合。career-grade(一次情報主義 + honest rigor)。
> 由来 = workflow `article-idea-bank`(2026-06-18, ユーザー指示「llterm が頑張れるよう記事ネタをいっぱい提供」)。honest 監査 verdict = **出版可(条件付き)**。
> 関連素材 = `docs/ARTICLE_MATERIAL_2026_06_competitor_methods.md`(強豪手法×共通点)/ `docs/POSITIONING_VS_LLAMACPP.md`(立ち位置)/ `docs/MEMORY_EFFICIENCY_FINDINGS.md`(数値正本)/ `docs/ARTICLE_SEEDS.md` #33–51。

## ★連載レベル恒久 caveat(全記事トップに固定・最優先 fix)
> **本連載の全 llcore 実測は自宅 CPU の極小モデル(0.81M〜130M char-LM)PoC であり、大手の実 LLM(12B〜64B)性能を直接反証・凌駕するものではない。比較は手法・思想・計測規律の次元で行う。**
> 量子化記事は「**footprint は実測だが simulated quant = 推論速度は一切未測**」をフック直後に太字で前置(「2bit/int8 で速くなった」誤読を構造的に防ぐ)。

## ★honest 監査の必須 fix(記事化時に適用)
1. **連載トップに上記 caveat バナーを固定**(個別注記の総和では「実 LLM を反証した」という誤読蓄積を防げない)。非エンジニア向け第1部で特に必須。
2. **S1 の過剰一般化を修正**: 「PPL が原理的に capability を隠す」→「今回 PPL と top1 は **lockstep で同時劣化**したが、gate 閾値(0.85×)が粗くて壊れた 2bit を PASS させた」に限定。事前予想「top1 先行劣化」が外れた事実を本文の主役に(誇張を削ると逆に強くなる)。
3. **thin 4本の救済/降格**: B7(家族OSS経済学=一次測定ゼロ→実 DL/star を出すか「設計意図の来歴記事」に降格)/ B5(母数3つ=実演 bazue が将来計画で空→S2 型4 + B6 に吸収)/ B8(fail-closed×キャリア=一般訓話→B1 に圧縮)/ A11(分業の言語ゲーム→A4/S6 のコラム化)。
4. **A10(認識=弁別記号)に線引き明記**: 「どこまで構造的同型でどこから別物か」を本文に。MangaFlow section memory(画素の知覚同定)と llcore recurrent(トークン分布の状態圧縮)は「過去を有限表現に畳む」点でのみ同型。MangaFlow 話者帰属困難と bazue hard case 159 は「**似た症状**」止まりで「**同じ原因**」と書かない(失敗メカニズムが別の可能性=要検証)。
5. **異分野アナロジーは 1 記事 1 本柱**。A9 の edge of chaos は蛇足→削除し、SPC × fail-closed の実測(0/200→200/200)に集中。各記事で「この接続が壊れる箇所」を必ず 1 段落明示。
6. **B6 の外挿を本文で太字宣言**: 「llcore 規模則を 12B〜64B 強豪に当てる」箇所は「以下は最大 130M での観測からの**外挿仮説であって測定ではない**。傾きの方向のみ literature 整合、絶対値は未保証」と本文側に太字。

## ★編集長の推し(S 級筆頭)
- **S2 cherry-pick 5型** = 連載の**看板**(時宜性最大・taxonomy 化で「読者の道具」に・最後に同じ 5 型を llcore 自身へ向ける自己監査章)。
- **A7 制御理論 = SSM** = クロスドメインの**白眉**(RWKV/Mamba=状態空間モデルは literature 正統で我田引水でない・「×1.00」実測 × 60 年の制御理論)。ρ≈1 の過剰接続だけ抑える。
- **S1 PPL trap** = honest disclosure 入口の最強フック(品質ゲート通過 vs top1 半減)。fix 2 適用で S 級確定。
- **B1 負けを見せる** = 連載の**着地点**(llcore の実敗北 QAT 82.9%/mmap 全載収束/streaming peak 減らず を信用の材料に)。

---

（以下、編集長整理版の全 26 本 + 連載構成案 2 案）

## 【S 層 — 今すぐ書く価値大(時宜性 × インパクト)】

時宜性=2026 年 6 月の業界アーキ収斂(Gemma4 / Cosmos3 / PaddleOCR-VL / Hermes)に正面で接続でき、かつ llcore 実測が最も強い 7 本。

---

### S1. PPL が PASS でもモデルは半分壊れていた — 単一スコアの罠を、自分のベンチで実演する
**統合元**: #10, #37(seed配列の "PPLが下がっても" 2本), #44系の core。**この命題の決定版を S1 に集約。**
**側面タグ**: honest disclosure / ベンチ / 認知科学 / 技術設計 / 教訓
**フック**: うちの 2bit 量子化モデルは品質ゲートを堂々と通過した。なのに次トークン正解率は 28.7%→15.2%、ほぼ半減していた。1 つの集計スコアは、能力の崩壊を平気で見逃す。
**angle**: 単一集計メトリクス(PPL)が capability の離散的崩壊を隠す普遍現象を、非エンジニアにも届く認知科学(Goodhart の法則・construct validity・代理目標の乖離)として独立させる。実演=realp1(11.9M)2bit が unigram PPL gate を PASS(101 < 0.85×215=183)するのに top1 -13.46pp。対策=新設 capability-gate(fp32 比 top-1 retention≥97%)=「第二の目」を fail-closed で配線。industry へ橋渡し:PaddleOCR の OmniDocBench 単一スコア 96.33 も「1 つの数で勝者を決める」同型構造。
**grounding**: `src/llcore/lm/eval.py` の held_out_top1_report + passes_capability_gate、`out/quant_bitwidth_sweep*.json`(realp1 2bit top1 -13.46pp なのに ppl-gate PASS、multi_smoke 3bit も同型)、docs/MEMORY_EFFICIENCY_FINDINGS.md (b')。PaddleOCR 単一スコア 96.33(arXiv:2606.03264)。
**honest 注記**: tiny char-LM・unigram baseline 前提の CPU PoC で実 LLM の評価系とは規模も指標も別。事前予想「top1 は PPL より先に劣化」は不成立(lockstep)— 誇張せず「同時劣化 + gate が粗い」と記録。PaddleOCR 数値は self-report で改竄ではない。97% 閾値は自前設定。
**想定読者・長さ**: 両方 / 18-22 分(7000-9000 字)

---

### S2. 「proprietary 超え」の解剖学 — ベンチ cherry-pick を見抜く 5 つの型と、自分の主張に同じ刃を向ける作法
**統合元**: #9(decisive), #15(self-report 段論), #44系の "巨大ベンチに勝つより" の横断整理部。**業界比較 honest disclosure の flagship。**
**側面タグ**: honest disclosure / 業界比較 / ベンチ / 教訓 / 戦略
**フック**: 「OSS が Gemini を超えた」という見出しを 3 回見たら、3 回とも内訳が違った。負け軸の省略・専用ベンチ・自前測定 — 派手な勝利は、決まって同じ 5 つの型のどれかで作られている。
**angle**: cherry-pick の分類学(taxonomy)を読者の道具にする。型1=負け軸の省略(Cosmos: Driving 79.3 vs 47.2 は出すが Robotics 57.8 vs 58.2 / General 73.7 vs 77.5 は伏せる)、型2=専用ベンチ(PaddleOCR OmniDocBench)、型3=自前測定(全数値が Baidu/NVIDIA 環境)、型4=母数小(Driving は 3 ベンチ)、型5=二次情報の拡大解釈(NVIDIA 公式は「open 内 1 位」限定)。最後に同じ 5 型の刃を llcore 自身の主張に当て直す自己監査章。「実在確認 < self-report < 第三者再現 < 独立査読」の信頼の梯子も組込。
**grounding**: Cosmos3 技報 Table 10、NVIDIA newsroom「open models 内 1 位」限定、PaddleOCR arXiv:2606.03264 Table 2。FullSense 規律 feedback_benchmark_honest_disclosure。llcore 再現用 scripts/ + out/*.json 公開・別プロセス隔離。
**honest 注記**: 5 型は「数値が嘘」でなく「主張の範囲が狭い」道具。各社数値は一次情報で実在確認済だが全て self-report・第三者再現未確認。llcore 自身も self-report 段で対象は tiny char-LM(0.81M-130M)CPU PoC — 自己監査章に必ず置く。
**想定読者・長さ**: 両方 / 20-25 分(8000-11000 字)

---

### S3. 「16GB で動く」の 16GB は RAM? VRAM? — open weights 時代の最大の不誠実は "必要メモリの曖昧さ" だ
**統合元**: #19(decisive), #14("26B 迫る" dense vs MoE 部分を内包), #21(bpw/4× 基準論を grounding に統合)。
**側面タグ**: 業界比較 / honest disclosure / エコシステム / 教訓
**フック**: Gemma4 12B は「16GB 級で動く」と紹介される。だがその 16GB は RAM か VRAM か、量子化前提か、誰も明記しない。fp16 の 12B は素で約 24GB だから、16GB という数字は静かに Q4 量子化を仮定している。
**angle**: open weights スペック表記の「必要メモリ基準の曖昧さ」という計測衛生問題を主題化。Gemma4=RAM/VRAM 曖昧 + Q4 前提、PaddleOCR/Cosmos=必要メモリ未測。さらに「dense 11.95B vs MoE のアクティブ param は意味が違う」「Q4 前提 16GB vs fp16 で footprint が変わる」「bpw と圧縮基準(int8 の 4× は fp32 比のみ・fp16 比なら約 1.9×)」を「揃えるべき軸チェックリスト」に。対比=llcore は working-set hard-max を実測し cap_set_ok と実測 peak を JSON にそのまま残す(522MB を 358MB 上限=68% で完走、logits checksum 完全一致)。読者が自分のモデル選定で「その GB 値は何基準か」を問える実務チェックリストに落とす。
**grounding**: POSITIONING [^1][^2][^3]、`out/mmap_ram_exceed_poc.json`(522MB を WS 上限 357.6MB で完走、cap_set_ok=true、checksum 一致 -215.1)、GGUF Q8_0=8.50bpw≈fp32 比 3.76×/fp16 比 1.88×。
**honest 注記**: llcore は「RAM 総量超」でなく「working-set 上限 < モデルサイズ」で同性質を実証。hard-max 強制可否は環境依存(cap_set_ok を残す)。tiny PoC vs 実 LLM の規模差を明記。各社数値は self-report。
**想定読者・長さ**: 両方 / 17-21 分(7000-9000 字)

---

### S4. 独立ベンチのない「使うほど賢くなる」は、反証できないから信頼もできない — Hermes の learning loop を解剖する
**統合元**: #11(falsifiability decisive), #8/#29/#35(汚染ループ × Approval Bus を grounding 層に内包), #41系("成長する vs 責任を持って成長する")。**Hermes 群 5 本を 1 本に集約。**
**側面タグ**: honest disclosure / 業界比較 / ベンチ / 哲学 / 自己進化
**フック**: stars 19 万のエージェントが「使うほど自己改善する」と謳う。だが効果を示す独立ベンチも査読論文も、arXiv 検索で 0 件だった。検証できない長所は、長所ではなく「願い」だ。
**angle**: 反証可能性(falsifiability)をベンチ honest disclosure の主題に据える。Hermes の learning loop 5 機構(skill 自動生成/自己改善/記憶永続/FTS5/Honcho)を一つずつ取り上げ、各機構が「誤 skill を生成・蓄積・再利用する汚染ループ」に転びうる経路を技術的に示す(二次情報 Pebblous 指摘)。効果が公式自称のみ=ポパー的「反証できない主張は科学的主張でない」に接地。対比=llive の Approval Bus + HITL は「成長を承認境界で観測・停止できる=反証可能な設計」、llcore の fail-closed capability-gate は「capability の自己改善にも fail-closed ゲート」の同型。ベンチの不在それ自体が一つの honest disclosure トピックになる珍しい角度。
**grounding**: GitHub NousResearch/hermes-agent(MIT)、learning loop 独立ベンチ/査読 arXiv 0 件、stars 196,554(API 実測)、Pebblous 汚染ループ指摘。llive Approval Bus/HITL(project_llive_9axis_skeleton)。llcore cap-gate(eval.py)。
**honest 注記**: 「ベンチがない=機能しない」ではない(まだ測られていない)。stars の質(bot/campaign 比率)未検証=脅威確度に留保。llive 側も Approval Bus の大規模独立ベンチ未整備=「設計上反証可能」と「実際に反証実験を回した」は別物。llcore cap-gate との接続は fail-closed の同型性であって機能の等価ではない。
**想定読者・長さ**: 両方 / 20-24 分(8000-10000 字)

---

### S5. ベンチに向けて学習する — PaddleOCR の弱点狙い撃ちと Goodhart の法則
**統合元**: #2(decisive: capability-gate × region-aware), #13(Goodhart 主題), #27/#36系("capability=NULL/データ支配" を grounding に内包)。
**側面タグ**: ベンチ / honest disclosure / 業界比較 / TRIZ / 戦略
**フック**: 0.9B のモデルが 235B を文書ベンチで超えた。種明かしは region-aware data optimization — ベンチが苦手とする箇所を狙い撃ちで学習データに足す手法。測られる数を最適化すると、その数は測定としての意味を失う。
**angle**: PaddleOCR-VL の region-aware data optimization + CPT→SFT→GRPO を題材に「評価指標に向けて学習する(training to the test)」と Goodhart を解説。これ自体は正当な最適化だが「汎化」としての意味は薄れる線引きを丁寧に。我々側の対比=llcore の識別力設計規律「floor(ベースライン)を仮説族に包含させてから最適化する」+ capability-gate≥97%。さらに「capability で勝とうとして null に当たり、データ・表現・計測へ撤退した」失敗からの学習(memory pivot)を「ボトルネックはデータ」命題の負け筋例として接続。GRPO がベンチ報酬を直接最適化しうる構造リスクも添える。
**grounding**: PaddleOCR-VL 0.9B(ERNIE-4.5-0.3B+NaViT)、OmniDocBench 96.33、region-aware + CPT→SFT→GRPO(arXiv:2606.03264)。llcore: capability=NULL/データ支配(memory:project_llcore_memory_efficiency_pivot)、floor 包含規律、cap-gate≥97%。
**honest 注記**: region-aware は標準的で正当な手法であり「ズル」ではない — 主張は「ベンチ特化は汎化主張を弱める」に限定。96.33 は self-report・文書専用ベンチ。「データ支配」結論は tiny char-LM の null 観測由来で実 VLM とは規模も手法も別物。
**想定読者・長さ**: 技術者寄り(両方可)/ 18-22 分(7000-9000 字)

---

### S6. エンコーダを「捨てる」設計と層を「捨てる」設計 — Gemma4 のモダリティ統合と llcore 量子化アークは同じ TRIZ 原理で動く
**統合元**: #1(decisive: エンコーダフリー × streaming-dequant), #24(GPTQ の中間→末端ずらしを内包), #22系(アーキ収斂地図を後半に組込)。
**側面タグ**: 技術設計 / TRIZ / 業界比較 / 哲学
**フック**: Google は Gemma4 で音声エンコーダを外し波形を直接投影した。我々は llcore で「重みを忠実に近似する」量子化を捨て、出力を正確にする方を選んだ。一見無関係なこの 2 つは、同じ発想に支えられている。
**angle**: Gemma4 12B のエンコーダフリー統合(別エンコーダの重み・中間特徴を持たない=中間物を作らない)を剪定設計と読み、llcore の Int8Linear streaming-dequant(層ごとに fp32 を作って即解放=中間物を最短寿命に / 常駐 539→149MB, 72% 減)と対置。さらに GPTQ の逆説(‖W−Ŵ‖² でなく ‖(W−Ŵ)X‖² を最小化=重み忠実度を捨て出力忠実度を買う / 2bit で weight 誤差 0.61→0.68 と悪化させつつ output 誤差 78.6→71.2 改善)を「最適化対象を中間から末端へずらす横断パターン」として核に。両者の限界も対称:Gemma4 は dense 11.95B で結局 16GB 級、llcore は圧力が無いと peak は減らない(963→882MB)。
**grounding**: Gemma4 エンコーダフリー音声波形直接投影・dense 11.95B(blog.google + HF model card)。llcore: int8 streaming `out/int8_streaming_infer.json`(539.6→148.9MB=0.285×、logits checksum -192.1)、GPTQ `scripts/gptq_compare.py`(weight 0.61→0.68 / output 78.6→71.2)。
**honest 注記**: llcore は 130M ランダム CharGPT の CPU PoC で simulated quant(真の int8 GEMM でない・速度未測)、Gemma4 は実 12B でモダリティ範囲も桁違い・統合効果は self-report(定量比較表未公開、「26B 迫る」も裏付けなし・相手 MoE)。「同じ原理」は思想レベルの類比で性能比較でない。TRIZ 対応付けは筆者解釈。
**想定読者・長さ**: 技術者 / 12000-15000 字

---

### S7. 大手が無料でモデルを配る時代、個人が勝てるのは「層」じゃなく「継ぎ目」だ
**統合元**: #40(decisive: 継ぎ目戦略), #23/#46系("モデルで勝てない"判断), #50系(ライセンス意思決定木を grounding に内包)。**戦略 flagship。**
**側面タグ**: 戦略 / 業界比較 / 未来予測 / エコシステム
**フック**: Google が Gemma4 12B を Apache2.0 で、NVIDIA が Cosmos3 を、Baidu が PaddleOCR-VL を出してくる。「モデルを作る」ことで個人が勝つ目はもう薄い。では、どこに座れば勝てるのか。
**angle**: 大手の Apache 無料配布で確定したのは「llcore はモデルで勝てない」。だから価値はモデルでなく再利用可能インフラ層=モデル間/責任境界/運用の「継ぎ目」(llmesh MCP ハブ / llive Approval Bus / llcore メモリ効率 gate)。大手は単一モデルの性能最適化に偏り、継ぎ目(統合・gate・計測)は彼らの主戦場でない。memory scaling 戦略の核(「ニューラルの賢さがメモリに指数で伸びる族は存在しない」)で「大モデルを無料で享受しつつ自分は計測と規律に集中する」ファネル設計を論じる。ライセンス意思決定木(Apache/MIT は乗れる船 / OpenMDW 非 OSI は将来制約が読めない)も継ぎ目選定の一部として組込。
**grounding**: Gemma4=Apache2.0/dense11.95B、Cosmos3=OpenMDW/2 塔、PaddleOCR=Apache2.0/0.9B、Hermes=MIT。llmesh=on-prem MCP、llive=Approval Bus、llcore=cap-gate≥97% + 5 実測点。MEMORY_SCALING_STRATEGY.md §1。
**honest 注記**: 強豪は実 LLM を本番品質で出荷、llcore は tiny char-LM PoC。「継ぎ目で勝つ」は戦略仮説で llmesh/llive が大手統合層(LangChain 等)に勝った定量比較はまだ無い。Cosmos は本体 64B で本質は大規模(小バリアントを持つだけ)。ライセンスは時点情報・法的助言でない。
**想定読者・長さ**: 両方 / 20-25 分(13000-16000 字)

---

## 【A 層 — 連載に組込(深掘り・技術詳細・体系性)】

実装報告・クロスドメイン・体系的整理。S 層の論点を技術的に裏打ちする 11 本。

---

### A1. 量子化アークを自前で歩く実装記録 — RTN→per-group→GPTQ→QAT、2bit の床はどこで動きどこで動かないか
**統合元**: #5(decisive), #25(業界 Q4 前提との答え合わせ部を内包), #44系(QAT 床), #45系(系譜学=来歴アングルを A8 に分離)。
**側面タグ**: 技術設計 / 実装報告 / honest disclosure / 業界比較 / ベンチ
**フック**: 低ビット量子化の手法を一つずつ自前実装して同条件でぶつけると、2bit の top1 保持率は RTN 8%→GPTQ 12%→QAT 30% と段階的に上がる。しかし strict capability-gate(97% 保持)は最後の QAT でも越えられない。
**angle**: 「手法は各ビット幅の damage を減らすが、実用床(3bit)は手法を変えても動かない — 床を動かすには質的に別の次元(学習時量子化)が要る」。RTN(素朴丸め)→per-group(scale を列群ごとに細かく)→GPTQ(入力 Hessian で出力誤差最小化)→QAT(fake-quant+STE)を一本の線で繋ぐ。極低ビット小モデルでは per-group32 RTN(-5.31pp)が GPTQ-per-channel(-6.38pp)を上回る=「粒度 > 誤差補償」も起きる(GGUF k-quant が小モデルで効く裏付け)。業界の量子化前提(Gemma4=Q4 前提 16GB 級)と実測床を答え合わせ。
**grounding**: `out/quant_bitwidth_sweep*.json` `quant_group_compare*.json` `gptq_compare*.json` `qat_train_2bit.json`(2bit top1 RTN 7.98→GPTQ 12.07→QAT 30.10%、retention 22/33/82.9%、全 cap-gate FAIL、realp1 3bit が PTQ 実用床)。手法的同根=GGUF k-quant、GPTQ(Frantar 2022)。
**honest 注記**: 全て 1-12M char-LM CPU PoC・weights-only・Linear のみ・simulated quant(速度未測)。実 LLM は大モデルほど低ビットに頑健で床が下がる=「tiny で 2bit が越えられない」は規模依存で大規模 LLM の量子化可能性を否定しない。閾値 97% は自前設定。
**想定読者・長さ**: 技術者 / 長(15000-18000 字)

---

### A2. mmap の「固定コスト」を実装で解剖する — なぜ 54MB モデルが load 時 1.42MB しか食わず、大きいモデルほど効くのか
**統合元**: #3(decisive), #32系(線形外挿バイアスの認知科学アングルを angle に吸収)。
**側面タグ**: 技術設計 / 実装報告 / 認知科学
**フック**: `torch.load(mmap=True)` + `load_state_dict(assign=True)` の 2 行で、54MB モデルの load 時 ΔRSS が 50.77MB から 1.42MB(×0.028)に落ちる。だがこの 1.42MB は "ほぼ定数" で、モデルが大きいほど相対効果が伸びる。
**angle**: 「mmap=省メモリ」は雑で、正しくは「load 時の固定コスト構造 → 大モデルほど相対効果大」という規模則。何が固定コストか(mmap セットアップ + state_dict メタの unpickling)、なぜ全 touch すると mmap も eager に収束するか(clean ページの on-demand fault-in)を別プロセス隔離計測の設計と共に解剖。さらに「人間の認知は入力に比例して出力が増える線形外挿をデフォルトにする」認知バイアスとして昇格させ、固定コスト規模則が直感と逆向きである点を TRIZ「事前作用/局所性質」と絡める。
**grounding**: `out/mmap_weights_poc.json`(realp1 53.91MB: eager 50.77MB / mmap 1.42MB=×0.028、touch 後両者~51.5MB、forward logits max|Δ|=0.0、2 回再測安定)。固定コスト~1.4-1.5MB はサイズ非依存=規模則の根拠。手法的同根=llama.cpp mmap weight loading。
**honest 注記**: 最大 53.91MB の CPU PoC。恩恵は部分 working set・ページキャッシュ共有・コールド起動遅延に限る。全 touch では最終 RSS は eager に近づく。真の RAM 超(モデル > 物理 RAM)はこの PoC では未検証。
**想定読者・長さ**: 技術者(両方可)/ 中(8000-12000 字)

---

### A3. 「使える RAM がモデルより小さくても動く」を Windows API で強制実証する — 522MB を 358MB の上限で完走させた手順
**統合元**: #4(decisive), #54系(体験記アングルを派生として A3 の "夜の作業ログ" 版に温存可)。
**側面タグ**: 技術設計 / 実装報告 / ユーザー体験
**フック**: 「仮想メモリで大きいモデルを回す」はよく言われるが実装でどう成立するかは曖昧なまま語られがちだ。`SetProcessWorkingSetSizeEx` で working-set 上限をモデルサイズ未満に強制し、522MB を 358MB の枠で forward 完走させ、出力が無制限実行と完全一致することまで実機確認した手順を全公開する。
**angle**: 「RAM 超で回る機構は read-only mmap ページの clean 性に依存する — 圧力をかけて初めて検証できる」。実装の勘所:(1)working-set hard-max を API で強制(cap_set_ok を JSON に正直に残す)、(2)read-only mmap ページは clean なので上限超過時に OS が pagefile 書込み無しで破棄→再 fault で disk 再読込、(3)logits checksum で機能正当性担保。途中の罠も正直に:int8 streaming で「層ごとに捨てれば peak も減る」と思ったら圧力が無いと torch caching allocator が解放メモリを OS に返さず peak ほぼ不変(963→882MB)。Gemma4/Cosmos が「Q4 で 16GB 級/Edge 2B」とサイズを下げる競合に対し、llcore は「枠を固定して中で回す」直交軸。
**grounding**: `out/mmap_ram_exceed_poc.json`(522MB を WS 上限 357.6MB で完走、peak 357.7MB≤上限、checksum -215.1、int8 保存 131MB=0.251×)。int8 streaming `(c)` 圧力なし 963→882MB・上限 368MB 完走。環境=Windows 11/Python 3.11/torch 2.12.0+cpu/avail~3.6GB。手法的同根=llama.cpp mmap+clean page eviction。
**honest 注記**: avail RAM 制約で「物理 RAM 総量超」でなく「working-set 上限 < モデルサイズ」で同性質を実証。hard-max 強制可否は環境依存(cap_set_ok=true は本環境結果)。130M ランダムモデル・int8 は disk/load まで(per-layer streaming dequant forward は将来課題)。
**想定読者・長さ**: 両方 / 長(13000-18000 字)

---

### A4. 定数状態 recurrent を実機 peak RSS で殴って確かめる — Cosmos の 2 塔も MangaFlow の section memory も「状態を太らせない」同じ戦い
**統合元**: #6(decisive), #28系(MangaFlow × 定数状態を grounding に内包), #26系(Cosmos 2 塔 vs llive 分業は A11 に分離)。
**側面タグ**: 技術設計 / 実装報告 / 業界比較 / 認知科学
**フック**: 文脈長を ×8 にすると GPT の peak working set は ×2.65 に膨らみ、recurrent/RWKV は ×1.00 で平坦。これを解析値でなく別プロセスの実機 peak RSS で裏取りした。同じ週、Cosmos3 は 2 塔を単一アーキに畳み、MangaFlow は外部 section memory でコマ間状態を持つ。
**angle**: 「長文脈・マルチモーダルのメモリ問題は、突き詰めると状態をどこにどう持つかの実装選択に収束する」。llcore の定数状態(RWKV/Mamba 系)を peak RSS スイープで実証(T256→2048 で GPT 229.8→607.9MB、recurrent 205.0→204.8MB 平坦、文脈コストは GPT で~25→403MB と超線形=attn O(T²))。Cosmos3 の 2 塔→単一アーキ(reasoner+generator を一塔=別々の重み常駐を畳む)、MangaFlow の story section memory(コマ間キャラ参照を外部記憶に逃がす)と並べ「定数/外部化された状態」を実装視点で対置。
**grounding**: `out/recurrent_runtime_rss.json` `out/mem_footprint.json`(T256→2048: GPT ×2.65 / Recurrent・RWKV ×1.00、state_bytes recurrent 2,048B 一定 vs GPT KV ×16・attn ×256)。Cosmos3: 2 塔単一アーキ(NVIDIA newsroom)。MangaFlow: section memory ablation CIDS 0.619→0.582 / CSD 0.668→0.547(arXiv:2605.28173)。
**honest 注記**: CPU char-LM の peak WS 実測(GPT.generate は block_size crop で実行上有界=本測は厳密長文脈想定)。Cosmos/MangaFlow 寄与は self-report/査読前。MangaFlow の作画は外部クラウド拡散モデル・Layout IoU 100% は自作幾何メトリクス。「同じ戦い」は設計思想の類比で性能の横並びでない。
**想定読者・長さ**: 両方 / 長(12000-18000 字)

---

### A5. footprint を「正直に」数える実装 — scale も bias も LayerNorm も計上して初めて「4× 圧縮」と言える
**統合元**: #7(decisive), #21系(bpw/基準明示の自己批判チェックリストを内包)。
**側面タグ**: 技術設計 / 実装報告 / honest disclosure
**フック**: 「int8 で 4× 圧縮」と書くのは簡単だが、量子化していない scale(fp32)・bias・LayerNorm・tied 重みを計上に入れると理想下限 0.25 に対し実値は 0.25-0.26 に膨らむ。低メモリ化の主張は、footprint の数え方を一行でも間違えると過大広告になる。
**angle**: 「圧縮率は何を分母分子に入れたかで簡単に盛れる — 計測コードの設計が honest disclosure の本体」。実装の勘所:(1)footprint に scale(fp32)と非量子化 1-D params 必ず加算、(2)tied wte/lm_head は同一 Parameter なので 1 度だけ計上(named_parameters は dedup で欠ける→state_dict 経由)、(3)因果マスク buffer は対象外、(4)per_channel は scale を行ごとに持つので per_tensor よりわずか増。さらに「4× は fp32 比のみ・GGUF 標準 fp16 比なら約 1.9×」「top-1 retention は llama.cpp に既出」「accuracy 併記は確立プロトコル」という自社過大主張を構造的に潰すチェックリストを公開。
**grounding**: `out/int8_quant_footprint*.json`(3 モデルで削減 73.8-74.8%・ΔPPL +0.00〜+0.10%、per_channel rel-RMSE 0.007 vs per_tensor 0.013-0.016、理想下限 0.25 vs 実値 0.25-0.26)、`src/llcore/lm/quant.py`(tied 重み state_dict 経由)。GGUF Q8_0=8.50bpw≈fp32 比 3.76×/fp16 比 1.88×。
**honest 注記**: weights-only(activation/KV メモリは別問題で未測)・simulated quant(速度未測)・1-12M char-LM CPU PoC。各社圧縮率も self-report で本記事の「検算」は方法論提示であって他社数値の反証でない。
**想定読者・長さ**: 技術者 / 中(8000-12000 字)

---

### A6. 「メモリで賢くなる」の嘘 — レジーム別に勝つプリミティブを選ぶ地図(SSM/MoE/RAG/量子化/Hopfield)
**統合元**: #23系(scaling 地図 decisive), #38系("AI は指数的に賢くなっていない"=emergence 誤読を angle に吸収)。
**側面タグ**: 業界比較 / エコシステム / 認知科学 / 教訓 / 哲学
**フック**: 「メモリを増やせば指数的に賢くなる」は業界の人気な誤読だ。8 アルゴリズム族を 26 エージェントで敵対検証した結論は身も蓋もない。ニューラルの賢さがメモリに指数で伸びる族は、ひとつも存在しなかった。
**angle**: 低メモリ波の各社手法を「どの regime でどのプリミティブが勝つか」の決定則として地図化。Transformer+KV(位置厳密照合だが実効窓 ≤ 容量窓)、Recurrent/SSM(O(1) 状態・長文/ストリーミング・recall 重は hybrid 必須)、RAG(≈log(corpus)・distractor 過多で負反転)、MoE(メモリ潤沢・compute 律速で勝つ・単機エッジ不向き)、量子化(bpw に厳密線形・cliff_then_flat)、Hopfield(容量が次元 d に指数だが条件付き・容量 ≠ 知能)。指数が本物なのは古典計算の 2 か所(メモ化/DP・modern Hopfield 容量軸)だけ。さらに「なぜ人間はそこに無い指数を見るのか」=(1)不連続評価指標が連続改善を断崖に見せる測定アーティファクト(Schaeffer 2023)、(2)指数の支出 × 線形の利得を指数の利得と混同、の二重構造を認知バイアス論へ接続。自社 overclaim(「大 RAM→大モデル常駐」は Beyond-Chinchilla で無条件には誤り)も正直に開示。
**grounding**: MEMORY_SCALING_STRATEGY.md §1-5(族別表・regime→primitive 決定則・指数は古典 2 か所のみ・Beyond-Chinchilla 訂正・Schaeffer 2023)。llcore 既測 3 プリミティブ(constant-state recurrent / mmap / int8)を地図上に配置。
**honest 注記**: 族別優位は regime 依存で普遍的勝者なし。llcore 実測は 3 プリミティブのみ(exact attention・MoE・Hopfield 層・offload 階層は未着手)。Hopfield 指数は次元軸限定・分離条件付き。batch・量子化は simulated。Schaeffer 2023 は学界で議論継続中。「指数はない」は「進歩がない」でなく「関数形の誤読を正す」主張。
**想定読者・長さ**: 両方 / 22-25 分(9000-11000 字)

---

### A7. 制御理論はとっくに「定数状態で過去を運ぶ」を解いていた — RWKV/Mamba を状態空間モデルとして読み直す
**統合元**: #45系(クロスドメイン decisive)。単独保持(クロスドメインの白眉)。
**側面タグ**: クロスドメイン / 教訓 / 認知科学
**フック**: 「文脈が伸びるほどメモリが膨らむ Transformer」と「文脈長によらず ×1.00 の recurrent」。我々の実機計測で出た差は、実は 60 年前の制御工学が「可観測・有界状態」として整理し尽くしていた問題の再演だった。
**angle**: RWKV/Mamba の「定数サイズの状態に過去を畳み込む」は状態空間モデル(SSM)そのものの語彙で書き直せる(SSM の S=State-Space は偶然でない)。観測 y を状態 x の更新に畳み込み続け過去全体を有限次元 x に圧縮=Kalman フィルタ以来の発想。Transformer の KV キャッシュは逆に「生の観測を全部とっておく」非圧縮表現で O(T) で膨らむ。recurrent の「なぜ平坦か」が config の偶然でなく構造的必然として読める。M2 の empirical_rho≈1 への張り付きは制御の「安定性境界(ρ<1 が収縮)」そのもので、表現力と安定性のトレードオフが制御理論の既知の緊張だと分かる。
**grounding**: harness T×16 で recurrent ×1.00 / GPT KV ×16 / attn ×256(MEMORY_EFFICIENCY (0))、runtime peak RSS GPT ×2.65 / recurrent ×1.00(同 (0'))。M2 empirical_rho≈1.000(ARTICLE_SEEDS #16)。制御理論側=状態空間モデル/可観測性/収縮写像 ρ<1。
**honest 注記**: 0.81M-130M tiny char-LM CPU 実測で実 LLM 規模を保証しない。ρ/状態空間の対応は構造的アナロジーで RWKV/Mamba が文字通り Kalman フィルタという主張でない(非線形ゲート・学習で乖離)。peak RSS は torch baseline 込みでクリーンな信号は増分トレンド。
**想定読者・長さ**: 技術者 / 12000-15000 字

---

### A8. 量子化は「信号処理の量子化雑音」を 40 年遅れで再発明している — GPTQ の誤差整形は ΔΣ 変調の親戚だ
**統合元**: #45系(クロスドメイン #2 decisive), #24/#33系(GPTQ 認識論アングルを grounding に内包), #46系(量子化系譜学=来歴を angle 後半に統合)。
**側面タグ**: クロスドメイン / 来歴 / 教訓 / 哲学
**フック**: GPTQ を自前実装して気付いた逆説 — 重みをわざと不正確(0.61→0.68)にして出力を正確(78.6→71.2)にする。この「誤差を未来のサンプルへ送って打ち消す」手口、オーディオ技術者なら即座にデジャヴを覚える。ΔΣ 変調のノイズシェーピングと同じ骨格だからだ。
**angle**: 信号処理は 1960 年代から「少ビット量子化で雑音が乗る」をノイズシェーピング・誤差拡散(Floyd-Steinberg)で解いてきた。GPTQ の「ある列の量子化誤差を未量子化列へ Hessian で伝播させ出力空間で打ち消す」は誤差拡散の重み版。「weight space で正確に近似する」でなく「function space で正確にする」指標の取り替えが核=信号処理が「波形を近似するな、知覚を近似しろ」と学んだ教訓の再演。校正データ(入力分布)が要る理由も「誤差整形は信号帯域に依存する」で説明。系譜学(RTN→per-group→GPTQ→QAT が「誤差を吸収する場所を推論時から学習時へ遡らせる前線移動」)も来歴アングルで組込。
**grounding**: 自前 GPTQ 2bit weight 0.61→0.68・output 78.6→71.2、‖(W−Ŵ)X‖² 最小化、Hessian H=Σxᵀx(MEMORY_EFFICIENCY (b''')、ARTICLE_SEEDS #42)、realp1 2bit GPTQ top1 -13.35→-6.38pp(52% 減)、校正 8,192 tokens。信号処理側=ΔΣ ノイズシェーピング/誤差拡散ディザ/perceptual coding。
**honest 注記**: 概念的アナロジーで GPTQ が文字通り ΔΣ 変調器でない(GPTQ は空間的伝播・ΔΣ は時間的フィードバック)。CPU・tiny char-LM・weights-only・simulated quant(速度未測)。GPTQ の優位も RTN per-group が上回る場合あり(#43)。GPTQ 原典 Frantar 2022 の完全再現は未保証。
**想定読者・長さ**: 技術者 / 13000-16000 字

---

### A9. 「良品だけ通す検査ライン」を進化計算に持ち込んだら空転した — SPC と fail-closed gate の交差点
**統合元**: #47系(クロスドメイン decisive), #43系("賢さは安定の崖っぷちで最大化"=empirical_rho≈1 を grounding に内包)。
**側面タグ**: クロスドメイン / 教訓 / TRIZ / honest disclosure
**フック**: 工場の検査ラインで「規格外を全部はじく」設定にしたら、最初の合格品が一個も出ずラインが止まった — 我々の MAP-Elites 進化探索で実際に起きたのがこれだ。random init から cert_inf gate の合格は 0/200。
**angle**: llmesh 内蔵 SPC(統計的工程管理)の発想を llcore の安全 gate 付き進化探索に重ねると同じ落とし穴を共有。SPC の管理限界を厳しくしすぎると歩留まりゼロ=fail-closed gate 空転と同型。解も同型(SPC が「まず工程能力のある初期条件を作る」ように W→0 反復縮小 fallback で known-safe な最初の合格者を保証=200/200)。gate の実効コストは「判定 1 回の速さ」でなく「reject 率 × resample」で決まる(SPC 検査コスト=検査単価 × 再検査回数)。さらに「fitness 最良個体の empirical_rho が安定境界 1.000 に張り付く=表現力は壊れる一歩手前で最大」を edge of chaos と接続。FullSense 哲学「責任所在を architecture に」が製造業品質管理と地続きであることを示す。
**grounding**: cert_inf admit 0/200 → W→0 fallback 200/200(ARTICLE_SEEDS #15)、gate 実効コスト=判定 × reject 率 × resample で smoke 13 分停滞(#13)、best gene empirical_rho=1.000・archive 94% 発散(#16)。llmesh SPC(Xbar-R/CUSUM)。cap-gate≥97%。SPC 側=管理限界/工程能力/歩留まり。
**honest 注記**: tiny char-LM CPU PoC(122 annotations 級小データ)で製造ラインの統計的厳密性とは規模もデータ量も別。SPC との対応は設計思想の共通性で llcore が SPC アルゴリズムを実装している主張でない。0/200→200/200 は単一 seed 観測を含む。edge of chaos 接続は比喩的接地。
**想定読者・長さ**: 両方 / 12000-14000 字

---

### A10. 認識とは「弁別的記号の照合」である — マンガ生成 AI と定数状態 RWKV が別ドメインで同じ認知テーゼに到達した
**統合元**: #30系(認知テーゼ decisive), #55系(manga-md 入れ子=話者帰属体験記を grounding/裏層に内包)。
**側面タグ**: 認知科学 / 哲学 / 技術設計 / クロスドメイン / ユーザー体験
**フック**: 東大+HKUST の MangaFlow はコマ間でキャラを一貫させるため story section memory という外部記憶を持つ。llcore の RWKV は任意長の過去を定数サイズの状態で運ぶ。一見無関係なこの二つは、実は「認識=identity を弁別する記号を文脈をまたいで照合し続けること」という一つの認知テーゼの別表現だ。
**angle**: 「同じキャラだと認識する」とは何かを問う。bazue 認知テーゼ「認識=弁別的記号の照合」を軸に、MangaFlow の story section memory(ablation で CIDS 0.619→0.582)と llcore の定数状態 recurrent(T×16 でも状態 ×1.00)を「一貫性とは記憶した弁別記号を照合し続けること」の共通原理として読む。MangaFlow 自認の限界「stylized 顔での話者帰属が困難」が bazue hard case 159(話者 ≠ 中央被写体)が突く弱点と一致=認知科学が予測する「弁別が崩れる境界」。裏層に manga-md(宣言的コマ→SVG)で「技術解説の道具(漫画)が同時に研究対象(認識の弁別性)になる入れ子構造」を体験記として組込。我田引水を避け構造的同型のみ主張。
**grounding**: MangaFlow arXiv:2605.28173 ablation(CIDS 0.619→0.582 / CSD 0.668→0.547)・自認の限界=話者帰属困難。`out/mem_footprint.json`(recurrent state T×16 で ×1.00 vs GPT KV×16)。bazue index 206 コマ・hard case 159(memory: project_bazue_vlm_benchmark)。manga-md(CLAUDE.md)。
**honest 注記**: MangaFlow は査読前 v1・引用ゼロ・作画は外部クラウド拡散モデル(Gemini 2.5 Flash Image/FLUX.2)。llcore は tiny char-LM CPU で実 LLM 規模の弁別性能は未検証。bazue VLM ベンチは「VLM 体制が整ったら」の将来計画で現時点は GT 整備段階。「認識=弁別的記号」はテーゼで実証済み定理でない。商用 mangaflow.studio は無関係の別製品。
**想定読者・長さ**: 両方 / 20-25 分(8000-10000 字)

---

### A11. 2 つの塔で分業する単一アーキと、複数 AI が役割を持つオーケストラ — Cosmos3 と llive の似て非なる「分業」
**統合元**: #26系(decisive)。単独保持(「分業」の多義性整理は他に代替なし)。
**側面タグ**: 技術設計 / 業界比較 / エコシステム / 哲学
**フック**: NVIDIA の Cosmos3 は「考える塔」と「生成する塔」を 1 つのアーキに同居させた。llive は複数の AI が役割を持って協調するオーケストラだ。どちらも「分業」だが、分けている境界がまるで違う。
**angle**: Cosmos3 の mixture-of-transformers 2 塔(reasoner+generator を単一アーキに同居、Qwen3-VL 流用)を「1 モデル内部のモジュール分業」、llive の分業オーケストラ(調査/真偽確認/レビュー等を役割分担、Approval Bus + HITL)を「プロセス境界を跨ぐエージェント分業」と整理。同じ「分業」の語が指す 2 つの設計境界を対比。Cosmos は重み/アーキ内部で reasoning と generation を分け効率を稼ぐ(単一デプロイ完結)、llive は独立プロセス間で責任と検証を分け「単独 AI で重要判断をしない」を architecture に落とす。核=「どこに分業境界を引くかが、最適化したい性質(推論効率 vs 責任所在の監査可能性)を決める」。MangaFlow のエージェント分割パイプラインも「プロセス分業」側の傍証。
**grounding**: Cosmos3 2 塔・5 モダリティ・Qwen3-VL 流用・Edge2B/Nano16B/Super64B(NVIDIA newsroom + HF)。MangaFlow エージェント分割(arXiv:2605.28173)。llive 分業オーケストラ・Approval Bus・memory:feedback_no_solo_ai_judgment。
**honest 注記**: Cosmos3 本体 64B(20 兆トークン)で本質は大規模。「Gemini 超え」は負け軸省略で成立。数値は全て NVIDIA self-report。llive の分業は設計・運用フレームで Cosmos の訓練済み単一モデルとは抽象レベルが違う=「分業」の語の多義性を整理する記事で性能比較でない。目的(効率 vs 責任)が異なり優劣の話でない。
**想定読者・長さ**: 両方 / 13000-16000 字

---

## 【B 層 — ストック(メタ方法論・キャリア・体験記)】

時宜性は低いが普遍的価値が高い、または S/A 層の素材を別読者向けに再構成したもの。8 本。

---

### B1. 勝った話より、負けを見せたほうが信用される — honest disclosure を個人ブランドの軸にする
**統合元**: #16系(decisive), #44系("内なる審判"), #41系/#39系(career-grade 自己批判)。**メタ flagship。キャリア群 4 本を集約。**
**側面タグ**: honest disclosure / 戦略 / キャリア / 哲学 / 教訓
**フック**: うちの量子化アークは 2bit で最後の壁を越えられなかった(QAT でも 82.9%、目標 97% に届かず)。それを隠さず書いたら、勝ち報告より反応が良かった。誠実な敗北報告は最強のブランディングだ。
**angle**: 業界が cherry-pick で勝利を演出する中、自分の限界・null・残課題を先に開示することが技術者個人のキャリア(一次情報主義 + honest rigor)に効く論。実演=llcore の敗北群(2bit は QAT でも cap-gate 97% 未達 / mmap は全載で eager に近づく / int8 streaming は圧力なしで peak 減らず)を「弱点の隠蔽でなく主張の強化材に変換する」実例として並べる。さらに「外を超えるベンチでなく自分を止める内なる審判(fail-closed cap-gate)を持つ」差を career-grade の核に。他社を分解する以上、自社数値も同じ厳しさで条件明記する義務=「批判は鏡として自分に返る」。
**grounding**: QAT 2bit top1 30.10%=fp32 の 82.9%(cap-gate 未達、MEMORY_EFFICIENCY (d))、mmap touch で eager 収束(同 (a))、int8 streaming 無圧力 963→882MB(同 (c))。feedback_benchmark_honest_disclosure。
**honest 注記**: 「誠実=必ず信頼される」は楽観で限界開示が評価される文化が前提・すべての場で有利と限らない。llcore の敗北は tiny char-LM CPU PoC スケールで実 LLM で同じ壁が同じ位置に立つ保証はない。「内なる審判が正しく強豪が不誠実」の二項対立に落とさない — 大手も技報詳細表には負け軸を載せており問題は主に二次拡散側。
**想定読者・長さ**: 両方 / 22-28 分(9000-12000 字)

---

### B2. 「最適化したのに効果ゼロ」の半分は、効果でなく計測の壊れ — 私が踏んだ 3 つの計測の罠
**統合元**: #53系(decisive)。単独保持(実装者向け実践集として独立価値)。
**側面タグ**: 教訓 / ユーザー体験 / 認知科学 / 技術設計
**フック**: 「ANN 化したのに速くならない(19.6→16.9ms)」「層ごとに捨てたのに peak メモリが減らない(963→882MB)」「2bit でも PPL は合格なのに精度は半減(top1 -13.5pp)」。これらは全部「手法が効かなかった」でなく「間違った指標を見ていた」だった。
**angle**: AI 駆動開発で最も時間を溶かす「効いてないように見えるが実は測り方が悪い」ケースを 3 つ解剖。(1)Amdahl の罠(ANN が速くないのは支配項が MiniLM encode~15ms で総当たりは数 ms=ボトルネックを測らず最適化)、(2)allocator の罠(int8 streaming で peak が減らないのは torch caching allocator が OS にメモリを返さないからで圧力を掛けて初めて顕在化)、(3)gate の粗さの罠(PPL-only gate は壊れた 2bit を PASS させる)。共通教訓=「計測の計測」(指標そのものの妥当性を改善を測る前に検証)。トップレベル ls が規模を 2.5 倍ずらした例(17.8k vs 44.8k docs)も添える。
**grounding**: ANN 19.6→16.9ms・支配項 encode~15ms(#1)、int8 streaming peak 963→882MB→上限で顕在化(#39)、PPL-only gate が realp1 2bit を PASS だが top1 -13.46pp→cap-gate 新設(#37)、ls 17.8k vs glob 44.8k docs(#3)。
**honest 注記**: 全て tiny char-LM CPU 自宅 PC 規模。「効果ゼロの半分は計測ミス」は経験則で定量的主張でない。各罠の数値は単発〜2 回再測(mmap は 2 回再測で安定、ANN recall@10 0.9825)。allocator 挙動は torch 2.12.0+cpu 固有。
**想定読者・長さ**: 技術者 / 14000-17000 字

---

### B3. 「失敗を消さなかった」ことが半年後に 2,000 万エッジの誤設計を救った — null 結果の複利
**統合元**: #52系(decisive)。単独保持(null 結果の複利という独自テーゼ)。
**側面タグ**: 教訓 / 哲学 / ユーザー体験
**フック**: 研究で「効果ゼロ」が出たとき、人はそれを記録から消したくなる。だが我々が M1 で「会話の連結性 hop は強い encoder では効果ゼロ〜微害」という null 結果を消さずに残していたおかげで、半年後に RAD 全量取込で約 2,000 万エッジを抱え込む誤設計を、躊躇なく回避できた。
**angle**: 「honest disclosure を研究の核に」を抽象論でなく「null 結果が後から配当を生んだ具体例の連鎖」として書く。失敗・null を消さない文化は道徳論でなく投資判断=記録された null には複利が付く。連鎖を辿る:(1)M1 で連結性 over-claim を honest に訂正→(2)その記録が M2 で「教師信号を外部事実(turn 構造)に接地する」circularity 回避を発想させ→(3)同じ記録が全量取込での group=None 選択を即決させた。メタ教訓「append-only ログでは supersede した観測も削らず新注記で残す」(ANTHROPIC org disabled が後日 valid に変わった例)も。
**grounding**: M1 null → 全量取込で group=None 即決・~2,000 万エッジ回避(#2)、circularity 回避=turn 構造接地(#5)、append-only supersede 規律(#30、#17)、CONNECTIVITY_BENCH_CORRECTION_2026_06_11.md。
**honest 注記**: llcore 研究内部の運用知見で tiny PoC 規模の判断履歴。「null が必ず後で効く」一般法則でなく特定連鎖で配当が出た事例(survivorship に注意)。エッジ数~2,000 万は設計時の見積もりで実走前の回避判断。
**想定読者・長さ**: 両方 / 12000-15000 字

---

### B4. 弱者の兵法としての TRIZ — 「自宅 CPU・少 RAM」という制約を北極星に変えた意思決定の解剖
**統合元**: #34系(decisive)。単独保持(pivot 意思決定の解剖は他に代替なし)。
**側面タグ**: TRIZ / 哲学 / 戦略 / キャリア
**フック**: GPU も大 RAM も持たない自宅 PC は LLM 研究では純粋な不利だ。だが 2026-06-16、llcore は北極星を「賢さ(capability)」から「メモリ効率(仮想メモリ含む)」へ転換した。負けていた土俵を降り、制約そのものを評価軸にした。
**angle**: 制約を価値へ反転する TRIZ 思想を一つの実意思決定として解剖。capability は進化 ≒ 勾配で NULL_TIE/NEGATIVE(負け筋)、memory は RWKV/Mamba 定数状態という構造的勝ち筋 — この「負け軸を捨て勝ち軸へ寄せる」判断手続きを孫子「勝ちやすきに勝つ」と TRIZ 矛盾解決を重ねて論じる。「制約があるからこそ業界が一斉に小型・低メモリ・ローカルへ収斂した 2026 年 6 月の正面に立てた」制約駆動のトレンド先取り構造も honest に検討(大手無料配布の脅威も併記)。
**grounding**: 北極星転換(memory: project_llcore_memory_efficiency_pivot、option A)。capability=NULL/負け筋、memory=実証済み勝ち筋。MEMORY_EFFICIENCY_FINDINGS.md 3 本柱。業界収斂=Gemma4/PaddleOCR-VL/Cosmos/Hermes。
**honest 注記**: 「pivot のタイミングが正しかった」は事後の傍証で因果証明でない。脅威も確定=大手が高性能小モデルを Apache/open で無料配布するため llcore はモデル本体では勝てず価値はメモリ効率の手法・計測・gate に限る。tiny PoC・CPU で実 LLM 規模の有効性は未検証。Cosmos は本体 64B で本質は大規模。
**想定読者・長さ**: 両方 / 20-22 分(8000-9000 字)

---

### B5. 母数 3 つで「勝った」と言えるか — ベンチの標本数と、bazue を 206 コマにした理由
**統合元**: #12系(decisive)。単独保持(標本設計という上流工程テーマ)。
**側面タグ**: ベンチ / honest disclosure / 技術設計 / 教訓
**フック**: Cosmos の「自動運転で Gemini 超え」は、たった 3 ベンチの平均だった。標本が少ないほど勝ちやすく、少ないほど何も言えない。では我々の bazue ベンチは、何コマあれば足りるのか。
**angle**: 業界の小標本ベンチ(Cosmos Driving=3 ベンチ母数)を入口に「勝敗を主張するのに標本がいくつ要るか」の統計的誠実さ。少数ベンチでは順位がノイズで容易に入れ替わる・平均だけで分散/信頼区間を出さない慣行を批判。我々の実践=bazue 話者帰属ベンチを 206 コマ・人間検証 GT で作り hard case 159(話者 ≠ 中央被写体)を意図的に含めた設計理由=「得意ケースだけ並べれば勝てる」を自分でやらないため。標本設計は honest disclosure の上流工程。
**grounding**: Cosmos3 技報 Table 10 Driving 79.3 vs 47.2 が 3 ベンチ母数(#47)。bazue index 206 コマ・hard case 159(memory: project_bazue_vlm_benchmark)。manga-md 話者帰属(尻尾=唯一の手がかり、docs/manga_grammar.md)。
**honest 注記**: bazue 206 コマも統計的には小規模でこれ単体で VLM の優劣を断ずる母数でない(VLM 体制が整い次第の検証用)。Cosmos の 3 ベンチ数値は self-report・実在確認済。主張は「大標本が常に正義」でなく「標本数と分散を開示せよ」で小標本にも探索的価値はある点を併記。
**想定読者・長さ**: 両方 / 16-20 分(6500-8500 字)

---

### B6. 0.3B のエンコーダと 16B の本体、64B の塔 — 強豪の「小型化」は規模則のどこを切り取っているか
**統合元**: #31系(decisive)。単独保持(規模則レンズで小型主張を読む独自性)。
**側面タグ**: 業界比較 / honest disclosure / 未来予測 / 認知科学
**フック**: PaddleOCR は 0.9B、Cosmos は Edge 2B から Super 64B まで。みな「小型・低メモリ」を謳うが、規模則のどこを切り取っているかは各社バラバラだ。
**angle**: 強豪各社の「小型化」が規模則のどの部分を切り取った主張かを llcore 実測規模則で読み解く。PaddleOCR 0.9B(専用タスクで構造的に有利)、Cosmos Edge2B/Nano16B/Super64B(本体 64B・20 兆トークンで本質は大規模、小バリアントを持つだけ)、Gemma4 12B(Q4 前提)を並べ「小型で勝てる条件」(専用タスク/量子化前提/小バリアント切り出し)の暗黙の前提を示す。llcore の規模則実測を当てる:mmap の load 時メモリはモデルサイズによらずほぼ固定(~1.4MB)→大きいほど相対効果大(7.73MB で ×0.218、53.91MB で ×0.028)、量子化 cliff はモデルが大きいほど低ビットに頑健(小 1.36M は 3bit で +11.6%、大 11.9M は 3bit で +4.8%)。核=「小型化の効果は規模に依存し、どの規模則を前提にするかで結論が逆転する」。
**grounding**: PaddleOCR 0.9B、Cosmos Edge2B/Nano16B/Super64B(本体 64B・20 兆トークン)、Gemma4 12B Q4 前提(各 HF/newsroom)。llcore: mmap 固定コスト~1.4MB・7.73MB→×0.218 / 53.91MB→×0.028、cliff モデルサイズ依存(#37)、emergence 誤読(#36)。
**honest 注記**: 強豪の規模諸元は self-report で直接ベンチしたものでない。llcore 規模則は最大 130M tiny char-LM CPU 実測で 12B〜64B への外挿保証なし — 「傾き」が同方向は literature 整合だが絶対値外挿は別途要実測。「大きいほど低ビットに頑健」は最大 11.9M での観測。
**想定読者・長さ**: 技術者 / 14000-17000 字

---

### B7. 「自宅で家族 OSS を作る」は趣味じゃなく戦略だ — llmesh/llive/llove/manga-md の設計の経済学
**統合元**: #42系(decisive)。単独保持(ファミリー設計の経済学は独立テーマ)。
**側面タグ**: 戦略 / エコシステム / 哲学 / キャリア
**フック**: 3 つの独立 OSS を自宅で作る。一見、個人の道楽だ。だが「単独でも価値が成立し、組み合わせると 1 つの世界観になる」という設計には、大手の単一プロダクト戦略にない経済的合理性がある。
**angle**: FullSense の「ファミリーで作る」を戦略論として分解。大手は単一の巨大プロダクトを出すが、個人/小チームは「小さく独立した部品を疎結合で束ねる」方が(1)単独でも採用される入口を増やせ(2)一つが失敗しても全体が死なず(3)組合せで差別化の堀を作れる。Unix 哲学/マイクロサービスの個人 OSS 版で TRIZ「分割原理」が効いている。キャリア面=「一つの大作に賭けず独立して価値を持つ小さな成果を連ねる」ポートフォリオ戦略。
**grounding**: FullSense=llmesh(MCP 統合)+llive(記憶)+llove(TUI)+manga-md(宣言的コマ→SVG)、各 PyPI 独立公開。Cosmos3=1 アーキ複数サイズ(分割だが単一ベンダ内)。差=大手は自社内分割、我々は独立 OSS として外部単独利用も成立。TRIZ 分割原理。
**honest 注記**: 「独立しても価値が成立」は設計目標で各プロジェクトが実際に単独採用された実績(DL/star/本番)で証明できていない。統合価値も F25 連携基盤は構想・部分実装段階。大手の単一プロダクトにも規模の経済の強みがあり「疎結合が常に勝つ」わけでない。個人/小チームに合った戦略で普遍解でない。
**想定読者・長さ**: 両方 / 20-24 分(13000-15000 字)

---

### B8. fail-closed という考え方は、AI の未来であると同時に「責任を引き受ける個人」のキャリア設計でもある
**統合元**: #43系(decisive: fail-closed × キャリア)。単独保持(技術原理 × 人生倫理の同型論)。
**側面タグ**: 未来予測 / 哲学 / キャリア / 技術設計
**フック**: llcore の量子化は品質が閾値(top1 retention≥97%)を割ったら通さない。llive は承認されない skill を成長に取り込まない。この「デフォルトで止める」設計思想は AI の未来の話であると同時に、個人がどう信用を積むかの話でもある。
**angle**: fail-closed(検証失敗時は通さない)を技術と人生の両方の原理として論じる。大手の競争は「性能を最大化する(fail-open 寄り、まず通す)」方向に傾きがちだが、ローカル/責任ある AI の未来は「信頼できなければ止める」を architecture に持ち込む方向(llcore cap-gate、llive Approval Bus)。これは「異常値が出たら勝った気にならず内訳を疑う」honest disclosure と同根。キャリア論=個人が信用を築く道も同じ「できると言い切る前に検証が通らなければ止める」fail-closed な誠実さが長期の個人ブランドを作る。AI 設計原理と個人の倫理が同型という主張。
**grounding**: llcore=fail-closed cap-gate(retention≥97% でないと通さない)、量子化アーク RTN→QAT、2bit=QAT 領域 82.9%。llive=Approval Bus+HITL。FullSense 哲学=「責任所在を architecture に」。Hermes の learning loop(無条件受容)との対比。
**honest 注記**: fail-closed は安全側だが止めすぎれば使い物にならないトレードオフ(過剰拒否の偽陽性)。gate 97% も tiny char-LM 閾値で実 LLM の最適閾値は別問題。「技術原理と人生倫理が同型」は比喩でどこで壊れるか(AI に人間の倫理を素朴に投影する危うさ)も明示。比喩の美しさで論を盛らない。
**想定読者・長さ**: 両方 / 22-26 分(14000-17000 字)

---

## 統合で吸収・降格した主要シード一覧(透明性のため)

- **#17/#18 (POSITIONING 系: 品質ゲート空欄 / 既知再導出)** → S2(cherry-pick 分類学)+ A5(footprint 検算)に吸収。単独だと「llama.cpp と競合しない」自己卑下が前面に出すぎるため命題化して再配置。
- **#20 (bpw/4× 基準)** → A5 + S3 に分割吸収。
- **#22 (アーキ収斂地図)** → S6 後半 + A11 に分割。
- **#48-51, #56 (ライセンス/キャリア/未来予測の重複群)** → S7 / B7 / B8 に集約。ライセンス意思決定木は S7 の grounding に格納。
- **#32/#33/#35-39 (認知科学 × 技術の二重出し)** → 各技術記事(A2/A6/A8/B2)の angle に認知科学レンズを織り込む形で一体化(別記事に割らない)。
- **#54/#55 (体験記)** → A3 / A10 の「夜の作業ログ」「入れ子体験記」セクションとして内包(独立 B 記事に割ると S/A と grounding が丸かぶりするため)。

---

# 連載構成案

## 連載案 1 ──【主軸推奨】「自宅 CPU から見た 2026 年 6 月の LLM 業界」全 3 部 12 本
2026 年 6 月の業界アーキ収斂(Gemma4/Cosmos3/PaddleOCR-VL/Hermes)を縦糸に、llcore 実測を横糸にする。**時宜性が最大の今、これを最優先で走らせる。**

- **第 1 部「測り方を疑う」**(honest disclosure 入口・非エンジニアも読める)
  S1(PPL の罠)→ S2(cherry-pick 5 型)→ S3(16GB は RAM? VRAM?)→ B5(母数 3 つ)
- **第 2 部「実装で掘る」**(技術者向け・llcore 実測の本丸)
  S6(エンコーダ/層を捨てる)→ A1(量子化アーク)→ A2(mmap 固定コスト)→ A3(522MB を 358MB で)→ A4(定数状態 peak RSS)
- **第 3 部「どこに座るか」**(戦略・締め)
  A6(メモリで賢くなるの嘘)→ S7(継ぎ目で勝つ)→ B1(負けを見せる)

各回の冒頭に前回の honest_note を 1 行引いて「ここまでの留保」を積み上げる構成。第 3 部 B1 で全留保を回収して個人ブランド論に着地。

## 連載案 2 ──【テーマ別・ストック向け】「クロスドメインで読む低メモリ AI」全 4 本
時事性に依存せず、いつでも出せる体系シリーズ。career-grade の「異分野で同じ問題が既に解かれていた」を主題に、A 層クロスドメインを核にする。

- A7(制御理論 = SSM)→ A8(信号処理 = GPTQ/ΔΣ)→ A9(SPC = fail-closed gate)→ A10(認知科学 = 弁別的記号の照合)
- 締めに B4(弱者の兵法 TRIZ)を「なぜ異分野に手を伸ばすのが弱者の最適戦略か」のメタ回として追加し全 5 本化も可。

**運用メモ(llterm 向け)**: S 層 7 本は相互に grounding が重なる(特に量子化・mmap)ので、同一週に S1→S2→S3 を出すと重複感が出る。連載案 1 の部構成で 1 部 = 1 週ペースが安全。B 層は S/A の合間の「箸休め」として時系列を気にせず差し込める。
