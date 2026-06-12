# M3 会話トピック重複 corpus の干渉測定 (検証 iii, 2026-06-12)

> ROADMAP M3 の検証 (iii)。M3.1 検証 (i) (`M3_SCALE_MULTILINGUAL_2026_06_12.md`) で
> 「埋もれの主因は store の絶対規模ではなくトピック重複」と解釈した仮説を、**会話 probe 側で
> 直接検証**する。会話の天文ターン (mars が reddish / eight planets) とトピックが重なる
> astrophysics corpus (arXiv abstracts md) を段階注入し、会話 22 probe — 特に天文 3 probe —
> が選択的に埋もれるかを実測した。
> 正本 = `out/rad_topic_overlap_poc.json`、スクリプト = `scripts/rad_topic_overlap_poc.py`。

## 設計

- **store ベース**: 会話 + loop_engineering corpus (M3.0 と同構成、1,071 annotations)。
  encoder = MiniLM 固定 (検証 (ii) で続投確定)。
- **重複 corpus**: `D:\docs\astrophysics_corpus_v2` (3,757 md, arXiv abstract 形式)。
- **段階注入**: 100 / 400 / 800 docs。**nested 等間隔サブサンプル** (800 の選択列から 400 を、
  400 から 100 を等間隔抽出 — deterministic・乱数なし) + incremental ingest (encode 重複なし)。
- **probe**: 会話 22 + loop 世界知識 18 (いずれも事前登録済み・変更なし)。loop probe は
  「トピック非重複側の対照群」のつもりで置いた (後述の通りこの想定は部分的に崩れた — honest)。
- 実行は foreground (background silent kill 回避)。191.8s / exit 0。

## 結果 (確定値)

| store | annotations | conv R@1 | conv R@3 | conv MRR | loop R@1 | loop R@3 | loop MRR |
|---|---|---|---|---|---|---|---|
| base (会話+loop) | 1,071 | 0.909 | 1.000 | **0.9470** | 0.611 | 0.611 | **0.6389** |
| +100 docs | 4,017 | 0.864 | 1.000 | 0.9242 | 0.500 | 0.611 | 0.5694 |
| +400 docs | 12,292 | 0.773 | 1.000 | 0.8712 | 0.500 | 0.611 | 0.5602 |
| +800 docs | 23,169 | 0.727 | **1.000** | **0.8485** | 0.444 | 0.444 | **0.4895** |

天文 3 probe (会話 PROBES のサブセット参照、変更なし) の rank 推移:

| probe | base | +100 | +400 | +800 |
|---|---|---|---|---|
| which planet is reddish | 1 | 1 | 2 | 2 |
| how many planets in the solar system | 1 | 1 | 1 | 1 |
| what is the reddish planet known for | 1 | 1 | 2 | 2 |

+800 docs 時の rank 変化の全量 (これ以外の probe は base から不変):

- 会話 (5/22 が変化): reddish 系 2 probe 1→2、describe rain 1→2、
  what is carbonara made with 2→3、name a famous novel 1→2。
- loop (4/18 が変化): canary deployment **1→9**、chaos engineering **4→0 (圏外)**、
  active learning 1→5、CoVe verification 1→4。

## post-hoc 実ヒット分析 (解釈のみ — 測定値は変更しない)

+800 store を再構築し劣化 probe の top-10 を目視確認した:

- **reddish 2 probe (1→2)**: rank 1 を取ったのは「Jupiter の Great Red Spot の red
  colouration」abstract — まさにトピック重複。ただし会話の正解
  「mars is known for its reddish hue…」は rank 2 (スコア 0.61) で残存。**壊滅ではなく
  1 ランクの押し下げ**。
- **describe rain (1→2)**: rank 1 は corpus 由来の 1 語 annotation「rains」(系外惑星大気の
  降雨)。会話の正解 (clouds/moisture/condense を含む文) は rank 2 で残存。
- **canary deployment (1→9)**: 押し下げ犯は astrophysics 論文の deployment/migration/upgrade
  語彙の annotation 群 (衛星 payload deployment、モデル deployment 等) — **トピックではなく
  語彙の重複**。なお corpus 中の "canary" は 1 件のみで Canary 諸島天文台説は棄却 (grep 確認)。
  この probe は base でもスコア 0.42 帯の弱いマッチで、弱マッチ probe ほど語彙ノイズに脆い。
- **chaos engineering (4→0)**: top-1/2 は**実は正解 doc の annotation** (「chaos engineering
  experiment loop steady-state hypothesis → inject → observe → learn」) だが、gold
  "steady state" (スペース区切り) と annotation 内の "steady-state" (ハイフン) が不一致で
  カウント外。rank 3 以降は軌道カオス (Lyapunov 指数等) の語彙重複。つまりこの probe の
  「圏外」は **gold 表記揺れの undercount が干渉で顕在化**したもの。doc レベルでは正解が
  rank 1 を維持している。

## 解釈 (honest)

1. **天文 probe の選択的壊滅は起きなかった**。重複 corpus 800 docs (corpus annotation 比
  約 20 倍) でも天文 probe は rank 2 止まり・conv R@3 は 1.000 を維持。会話の宣言的
  annotation は query との表層マッチが強く (スコア 0.59-0.61)、abstract 系の術語 annotation
  に対して最後の 1 ランクを守った。**fail モードは「壊滅」ではなく「R@1 → R@2-3 への漸進的
  押し下げ」**。
2. **劣化はトピック/語彙重複 probe に集中** — 仮説を会話側でも確認。+800 で変化した
  9 probe (conv 5 + loop 4) のうち 7 probe は重複で説明でき (reddish×2 / rain / canary 語彙 /
  chaos 語彙 / active learning / CoVe)、トピック直交な probe (名前・地理・算数・OODA・
  MAPE-K 等) は 1 ランクも動かなかった。carbonara 2→3 と novel 1→2 のみ重複で説明できない
  規模ノイズ。
3. **「対照群 = loop probe」の想定は部分的に崩れた** (honest): astrophysics は arXiv 論文で
  あり、deployment (装置)、chaos (軌道力学)、learning (cs.LG cross-list)、verification の
  語彙を普通に含む。loop MRR −0.149 は「非重複でも規模で劣化する」証拠ではなく、
  **probe 単位の語彙重複に分解できる** (劣化 4 probe 全てに重複の実ヒットを確認、
  直交 14 probe は不変)。
4. **検証 (i) との整合**: conv probe は非重複 corpus 60k annotations で MRR 0.890 (検証 i)、
  重複 corpus 23k で 0.849 (本測定) — **より小さい store でより大きい劣化** = annotation
  あたりの干渉力はトピック重複 corpus の方が強い。規模それ自体より「何を入れるか」が効く。
5. **実用上の含意**: (a) R@3 が保たれる限り上位 k 件をプロンプトに入れる用途では実害が
  小さい — fail モードが漸進的なのは設計上の朗報。(b) gold 表記揺れ (steady state vs
  steady-state) の undercount が干渉測定を歪める — doc レベル gold 判定の併記は (i) に続き
  本測定でも必要性が確認された。(c) スコープ絞り込み (group/role メタデータ) は重複 corpus
  混在時の R@1 防衛策として有効なはず — corpus 由来 annotation が rank 1 を取った事例は
  全て role="corpus" であり、role フィルタだけで会話 probe の押し下げは全て解消できる構造。

## 限界

- 天文の「重複」は 2 トピック (赤い惑星・太陽系) のみ。会話と corpus の重複密度がもっと
  高い場合 (例: 料理 corpus と料理の会話) の挙動は未測定。
- 800 docs は corpus の 21% (3,757 中)。全量での外挿は未検証 (annotations ~10 万級)。
- gold はキーワード包含 (verbatim) のまま。chaos probe の例の通り表記揺れで undercount する
  — rank 変化の解釈は実ヒット目視で補ったが、測定値自体は補正していない。
- corpus annotation が会話 gold 文字列を偶然含んで「正解扱い」になる方向の誤差は未補正
  (本測定の劣化幅はその分過小評価の可能性がある)。

## 再現

```
cd D:/projects/llcore
py -3.11 scripts/rad_topic_overlap_poc.py    # -> out/rad_topic_overlap_poc.json (191.8s)
py -3.11 -m pytest tests/unit -q             # src 変更なし
py -3.11 -m ruff check scripts/rad_topic_overlap_poc.py
```
