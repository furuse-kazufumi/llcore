# AnnotationStore 差別化判定 (2026-06-11, 5-agent workflow + 敵対検証)

**総合判定 = partially-novel.** 単体機能はすべて既踏。差別化の核は機能でなく「CLIP の弱点を
**再訓練せず構造で補償**する短句二層メモリ」という統合の仕方に限定される。最も野心的な
「verified-evolution との結合」は**現状コードに未配線**(設計意図)。この分離を明示しない限り over-claim。

## 既踏 (単体機能 — over-claim 禁止)
- **dedup→encode-once→cache** (実測 encode_saved_ratio 46.3%) = LangChain CacheBackedEmbeddings /
  RedisVL / content-hash dedup と実質同一。**新規性ゼロ** (cache hit 率を測っているだけ)。
- **CLIP/SigLIP text encoder をテキストメモリに流用** = Jina CLIP (arXiv:2405.20204) が直接測定済で
  **結論は否定的** (vanilla CLIP は MTEB retrieval で SBERT 系の約 1/3)。naive value prop は反証済み。
- **surface bigram Jaccard で誤字接続** = pg_trgm / Taxamatch (枯れた技術)。
- **ambiguity = 近傍非類似度** = structural holes / bridge centrality (Burt) の標準定義。
- **多義性を複数エッジ保持** = Mem0g の閾値超え multi-edge / multi-sense embeddings と発想重複。

## 未踏 (交差点 — ただし質は限定的)
1. CLIP の「短句が強み・長文/binding が弱み」特性を**意図的に短句アノテーション粒度に限定**して活かす。
2. retrieval 弱点を**再訓練でなく** surface(bigram)層で**構造補償** (brittleness 論文 arXiv:2511.04247 への mitigation)。
3. argmax で潰さず多義性を**エッジ多重度**として温存。
4. これら + dedup-cache + ambiguity を**一つの on-prem/CPU/Apache-2.0 PoC に統合**。

## 差別化になる最小の主張 (over-claim 回避)
> 「CLIP/SigLIP text encoder を**短句アノテーション粒度に限定**し、その既知の脆弱性
> (typo brittleness / 多義性) を**再訓練でなく** surface-bigram 層と多エッジ保持で
> **構造的に補償**した、dedup-cache 付き二層リンクメモリ。」

これ以上の主張 (「CLIP をテキスト検索に使える」「verified-gated に進化するメモリ」) はいずれも
over-claim (順に反証済み / 未実装)。

## 次の検証実験 (差別化を実証 or 反証する最小 PoC)
1. **【最優先】retrieval 品質 head-to-head**: AnnotationStore(CLIP-text) vs E5/BGE/SBERT を recall@k /
   nDCG で比較。短句限定で CLIP が拮抗すれば主張①が立つ。負ければ差別化は surface 補償のみに縮退 (falsify ライン)。
2. **surface 層 ablation**: 人工 typo 注入で semantic-only vs semantic+surface の「壊れた近傍回復率」を測定。
3. **多義性温存の効用**: top-k multi-edge vs argmax 1-edge を下流 (文脈注入正答率/bridge 検出 F1) で比較。

## 配線の本命 (未実装)
角度D の「cert gate × 会話アノテーション教師信号」結合 = novelty の本命だが**未配線**。
上記 3 つを固めてから「会話アノテーション教師下で sound gate が経験 gate を上回る」を実 LLM hidden で測る筋。

出典: 5-agent workflow (semantic-cache / clip-as-text-memory / typed-graph-memory / verified-evolution-tie)
+ completeness 検証。詳細は session transcript。
