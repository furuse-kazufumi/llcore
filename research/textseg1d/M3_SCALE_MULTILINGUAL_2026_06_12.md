# M3 大規模化 + 多言語 encoder head-to-head (2026-06-12)

> ROADMAP M3 の検証 (i)(ii)。M3.0 取込 PoC (39 docs, 世界知識 MRR 0→0.639, 会話干渉ゼロ —
> `M3_RAD_INGEST_POC_2026_06_12.md`) の続編として、(i) RAD corpus 3 分野 ~2,000 docs を
> 取り込んだ大規模 store (約 56 倍) での干渉・埋もれの実測、(ii) MiniLM vs
> multilingual-e5-small (prefix あり/なし) の head-to-head を行った。
> 正本 = `out/rad_scale_poc_scale.json` (検証 i) + `out/rad_scale_poc_ml.json` (検証 ii)、
> スクリプト = `scripts/rad_scale_poc.py`。
>
> **実行注記 (honest)**: 当初は単一実行 (`out/rad_scale_poc.json`) を予定したが、harness
> background のプロセスツリーがターン境界で silent kill される環境問題で 2 回消滅したため、
> `--skip-multilingual` / `--skip-scale` + `--out` 分割の foreground 実行で完走した
> (i: 509.8s / ii: 84.2s、いずれも exit 0)。スクリプト・probe・cap 仕様は分割前と同一。

## 検証 (i) — 大規模化: 3 corpus ~2,000 docs / 59,971 annotations

### 取込規模 (annotations 上限 60,000、超過時は doc 単位サブサンプル + 必ず log)

| corpus | docs 取込/総数 | cap 除外 | annotation instances | 取込時間 |
|---|---|---|---|---|
| language_corpus_src | 64/64 | 0 | 1,718 | 16.9s |
| evolution_corpus_v2 | 760/764 | 4 | 24,888 | 168.0s |
| agents_corpus_v2 | 1,165/1,170 | 5 | 45,233 | 278.2s |
| **計** | **1,989/1,998** | **9** | 71,839 | **463.1s** |

- cap 除外 9 docs は doc 単位の等間隔サブサンプル (deterministic, 乱数なし) で、
  ログ・JSON 両方に件数を明示 (silent cap なし)。simulated unique = 59,971 ≤ 60,000。
- store 最終規模: **59,971 annotations** (M3.0 の 1,071 から **約 56 倍**)、共起エッジ
  1,334,998、embedding matrix 87.8MB (59,971×384)、process RSS 1,008MB。
- encode_saved_ratio 0.178 (M3.0 の 0.0395 より上昇 — 分野横断で術語 dedup が効き始めた)。

### 結果 (確定値): 両 probe 系列とも劣化 — ただし様相が非対称

| 条件 | probe | R@1 | R@3 | MRR | 参照値 (M3.0 store) |
|---|---|---|---|---|---|
| 大規模 store | 会話 22 probe | 0.818 | 0.955 | **0.8902** | 0.909 / 1.000 / 0.9470 |
| 大規模 store | loop 世界知識 18 probe | 0.222 | 0.389 | **0.3056** | 0.611 / 0.611 / 0.6389 |

- **会話 22 probe: 軽微な劣化** (MRR −0.057)。rank 変化は 2 probe のみ
  ("where do i live" 1→2、"name a famous novel" 1→4)。残り 20 probe は rank 不変。
  56 倍 store でも R@3 0.955 — 会話 retrieval は実用水準を保った。
- **loop 18 probe: 大幅な埋もれ** (MRR −0.333、R@1 0.611→0.222)。rank 1 維持は 4 probe
  (watchdog / MAPE-K / OODA / CoVe) のみ。9 probe が劣化し、うち 4 probe が圏外へ
  (canary 1→0、chaos 4→0、value iteration 1→0 など)、active learning 1→8、
  gitops 4→8、alphazero 1→4、EWC 1→3、reflexion 1→3、circuit breaker 1→3。

### 解釈 (honest) — 「干渉ゼロ」は条件付きだったと down-claim する

1. **M3.0 の「干渉ゼロ」はトピック非重複の条件付きだった**。今回劣化した loop probe は
   ほぼすべて新規 corpus とトピックが重なるもの (alphazero/EWC/active learning/reflexion =
   evolution・agents corpus の頻出主題)。逆に会話 22 probe (名前・地理・料理・算数) は
   新規 corpus と重ならず軽微な劣化で済んだ。**埋もれの主因は store の絶対規模ではなく
   トピック重複**と解釈できる。これは検証 (iii) (会話トピック重複 corpus での干渉測定) の
   問いに対する部分的な先行回答でもある: 重複すれば会話 probe も同様に埋もれる可能性が高い。
2. **「正解 doc が消えた」のではなく「同義の別 doc に置き換わった」ケースが混在**。
   loop probe の gold は loop_engineering corpus の verbatim 文字列に事前登録で固定して
   いるため、agents corpus 側の同主題 annotation (gold 文字列を含まない) が上位を占めると
   undercount される。M3.0 の失敗分析と同じ「gold 判定基準の取りこぼし」が大規模化で
   増幅された側面があり、0.3056 は下限値。ただし事前登録 probe の変更はしない (数値確定)。
3. **実用上の含意**: 分野横断の単一フラット store は同主題間の食い合いを起こす。
   M3 続行時の改善候補: (a) corpus/scope メタデータによる検索スコープ絞り込み
   (AnnotationStore は group/role を既に持つ)、(b) doc レベル gold 判定の併記、
   (c) 隣接 annotation 1-hop 展開。cap 60k での取込自体は 463s / RSS 1GB で
   実用圏 — 取込コストは問題にならない。
4. **レイテンシ**: query 平均 19.2ms (M3.0 比 +7ms)。59,971×384 の総当たり cosine
   としては許容圏。10 万超で ANN 化を検討する水準。

## 検証 (ii) — MiniLM vs multilingual-e5-small (store = M3.0 と同構成 1,071)

probe は既存 (会話 22 + loop 18) を変更なし。E5 は "query: "/"passage: " prefix
あり/なし両方 (公式推奨は prefix あり)。

| encoder | world R@1 | world R@3 | world MRR | conv R@1 | conv R@3 | conv MRR | lat (world/conv) |
|---|---|---|---|---|---|---|---|
| MiniLM (基準) | 0.611 | 0.611 | **0.6389** | 0.909 | 1.000 | **0.9470** | 12.3 / 12.1ms |
| e5 prefix なし | 0.611 | 0.722 | **0.6731** | 0.864 | 1.000 | 0.9318 | 24.9 / 22.0ms |
| e5 prefix あり | 0.500 | 0.667 | 0.5972 | 0.864 | 1.000 | 0.9318 | 27.2 / 19.8ms |

- **e5 (prefix なし) は world +0.034**。M3.0 の失敗 5 問のうち **3 問を rank 1 に救出**
  (PID setpoint / RLHF reward model / Voyager skill library — いずれも M3.0 分析で
  「正解 doc は取れているが gold が日本語側」とした問)。**M3.0 の解釈 3 (日英混在
  undercount 説) を多言語 encoder が部分的に実証した**。残り 2 問 (receding horizon /
  ToT backtracking) は e5 でも圏外。
- **ただし conv は −0.015** (R@1 0.909→0.864) かつ**レイテンシ約 2 倍** (次元 384 同一
  だがモデルが深い)。
- **prefix あり (公式推奨) はむしろ悪化** (world 0.5972 < MiniLM)。OODA probe が 1→6 に
  崩れるなど。短句 annotation (1 文単位) を passage 扱いする本ユースケースでは
  非対称 prefix が裏目に出る。
- **結論: MiniLM 続投**。e5 の world +0.034 は conv −0.015 + レイテンシ 2 倍の対価に
  見合わない。多言語 gold の undercount は encoder 交換ではなく gold 判定の多言語化
  (doc レベル判定の併記) で解消する方が筋が良い。

## 限界

- (i) の loop probe 劣化幅は gold 判定の undercount を含む (解釈 2)。「retrieval 品質の
  劣化」と「測定系の取りこぼし」の分離には doc レベル gold 判定が必要 (未実装)。
- 検証 (ii) は M3.0 規模 (1,071) のみ。大規模 store での encoder 差は未測定。
- corpus は RAD 21 分野中 3 分野。cap 60k は手元 RSS 制約由来で、~48 分野全量
  (annotations 数十万級) は ANN / スコープ絞りなしには現実的でない。
- probe は引き続き自作・小規模 (22 + 18)。M3.0 と同じ設計者バイアスの限界を引き継ぐ。

## 再現

```
cd D:/projects/llcore
py -3.11 scripts/rad_scale_poc.py --skip-multilingual --out out/rad_scale_poc_scale.json  # (i) 509.8s
py -3.11 scripts/rad_scale_poc.py --skip-scale --out out/rad_scale_poc_ml.json            # (ii) 84.2s
py -3.11 -m pytest tests/unit -q   # src 変更なし
py -3.11 -m ruff check scripts/rad_scale_poc.py
```
