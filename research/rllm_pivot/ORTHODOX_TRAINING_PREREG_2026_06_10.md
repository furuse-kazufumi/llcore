# 事前登録: 正統派の学習(データ+勾配)で実 LM の能力が「進化」するか — Kaggle 長期テスト

**作成**: 2026-06-10 / **方針決定**: ユーザー「正統派の手法(データ+勾配で能力を育てる)で進化できるか試したい」。
**位置づけ**: llcore の検証ニッチ(verified 構造進化=capability NEGATIVE 確定)を**根源から置き換え**、
**正統派の SFT/継続学習**で「データを与えると能力が育つ(=ユーザーの言う"進化")」かを実 LM・実データで測る honest ベースライン。
**規律**: 事前登録 = 後付け解釈禁止 / 強 baseline 比較 / honest negative 許容 / 「変に良い」結果は内訳(汚染)を疑う([[feedback_benchmark_honest_disclosure]])。

---

## 0. 問い(falsifiable)
**正統派の学習(標準 SFT / 継続学習, AdamW 勾配)で、実小型 LM の能力(なぞなぞ正答・日本語整合性)は、データ量とともに育つか?**
- llcore の実測(2026-06-10): base SmolLM2-135M はなぞなぞ不可・日本語崩壊。進化(構造)は capability NEGATIVE。
- 本テストは「では正統ルート(データ+勾配)なら育つのか」を分離測定する。

## 1. 事前登録仮説
- **H1(データは効くか)**: SFT 後の held-out 能力が、未学習 base を**事前設定マージン**以上で上回る。
  - 指標: (a) held-out なぞなぞ/QA 正答率 +≥10pt、(b) 日本語整合性(held-out JA perplexity 改善 かつ LLM-judge coherence +≥1段階)。
- **H2(データ規模曲線=「進化っぽさ」)**: 能力が**学習データ量とともに単調増加**する(0 / 10k / 100k / 1M examples の 4 点で曲線を描く)。
  - これがユーザーの「データを大量に与える方が進化っぽい」の直接検証。単調増加なら "data = 進化" 支持。
- **H3(スケール交互作用, Stage B 任意)**: 大きい base ほどデータの恩恵が大きい。

## 2. 強 baseline / honest 統制(必須)
- **未学習 base**(現状能力。SmolLM2 はなぞなぞ不可=床)。
- **既製 instruct モデル参照**(例 SmolLM2-360M-Instruct / 1.7B-Instruct):自前 SFT を過大評価しないための「正統派が既に到達している水準」。自前 < 既製 なら "我々の SFT でなくスケール/データ量が本質" と honest 開示。
- **★汚染チェック**: eval セット(なぞなぞ/QA)が学習データに含まれないことを n-gram/ハッシュで検査。**held-out 厳守**。「変に高い正答率」→ まず汚染を疑う。
- seed 分散(≥3 seed)、cherry-pick 禁止、全 raw 出力を保存。

## 3. 被験設定
| 軸 | Stage A(feasibility, 無料 Kaggle) | Stage B(scale, 課金可) |
|---|---|---|
| base | SmolLM2-360M(Apache-2.0, base) | + SmolLM2-1.7B(Apache-2.0) |
| 学習 | 標準 SFT(full or LoRA), AdamW, cosine LR | 継続学習 + データ規模 sweep(H2 曲線) |
| データ | open-license instruction+QA+JA(eval は hold-out) | 規模拡大(10k→1M)、JA 比率 sweep |
| 計算 | T4/P100 16GB, 数時間, resumable ckpt | cloud spot(4090/A100), 長時間, 上限キャップ |
| 予算 | 0(無料 Kaggle) | 要見積り(上限を先に固定) |

※Qwen は商用障壁([[feedback_qwen_commercial_barrier]])ゆえ base から除外、Apache-2.0 系(SmolLM2/Pythia/Llama 系)に限定。

## 4. decision gate(honest negative 許容)
- **Gate A**: H1 PASS(SFT が未学習 base を有意に上回る)→ Stage B(H2 曲線+スケール)へ。**FAIL → 「正統学習でも Kaggle 規模では能力に届かない=capability には実スケールが要る」を第一級 negative として記録し停止**(課金前に止める)。
- **Gate B**: H2 で単調増加かつ既製 instruct 水準に近づく → 「データ方向は本物の進化」を data で支持。頭打ち/既製に遠い → 「小規模では限界、スケールが律速」を honest 開示。
- いずれも**使い切るまで回さない**。中間 gate で判断。

## 5. measurements(ログ)
train/val loss、held-out perplexity、eval 正答率(データ量点ごと)、LLM-judge coherence、seed 分散、wall-clock、コスト、汚染チェック結果。

## 6. honest 留保(先出し)
- 小規模 SFT の「能力向上」は**既製 instruct 化の再演**になりがち=新規性は無い(我々の貢献でなく正統手法の確認)。本テストの価値は「**FullSense の capability は正統学習由来であり、llcore の進化ニッチ由来でない**」を data で確定し、**llcore を guarantee/安全層に正式に位置づける**判断材料を作ること。
- 「データを与えれば育つ」が YES でも、それは **on-prem 個人開発が規模で勝てる土俵ではない**([[feedback_gpu_rent_over_buy]])。勝ち筋は規模でなく(a)私的な手元データ (b)責任/保証。
- 研究 `w1552o5vu` の示唆(効くのは流暢言語でなく離散・命名・合成可能な記号層=llrepr 的 typed annotation)は、本ベースラインの**次**の実験(言語接地目的)で検証する別レバー。本 preg はまず正統学習ベースラインに集中。

## 7. 次アクション
1. ユーザー go → Stage A を無料 Kaggle で実装(base/data/eval/汚染チェック/SFT/judge harness、resumable)。
2. Gate A 判定 → PASS なら Stage B 予算上限を相談して scale。
3. 全工程 honest-disclosure(汚染・既製比較・seed 分散を必ず併記)。

正本データ予定 = `research/rllm_pivot/orthodox_training_*` + 本 preg。関連: [[project_llcore_evolvable_llm_replan_2026_06_09]] [[feedback_llcore_must_become_llm_relevant]] [[feedback_poc_feasibility_first]]
