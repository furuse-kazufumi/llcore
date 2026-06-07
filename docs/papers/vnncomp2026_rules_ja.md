<!--
非公式・参考訳 (unofficial translation)。
- 原文 (binding): https://github.com/VNN-COMP/vnncomp2026/blob/main/rules.md
- 取得日: 2026-05-31
- 正文は英語原文。本訳は理解補助であり、解釈に差異がある場合は英語原文が優先する。
- 専門用語は原語温存 (日本語(English) 形式)。数値・スクリプト名・リンクは原文を改変しない。
-->

# VNN-COMP 2026 ルール（参考訳）

> ⚠️ 本ファイルは**非公式の参考訳**です。**正本は英語原文**（上記コメント参照）。スコアや締切など重要事項は必ず原文で確認してください。

**Website**: [https://vnn-comp.github.io/#vnncomp2026](https://vnn-comp.github.io/#vnncomp2026)

**VNN-LIB Website and Details**: [https://www.vnnlib.org/](https://www.vnnlib.org/)

**2026 参照先:**

- ルール議論リポジトリ (Rules discussion repository): [https://github.com/VNN-COMP/vnncomp2026/issues](https://github.com/VNN-COMP/vnncomp2026/issues)

**参考: 2025 年の過去 benchmark / 議論 / 結果:**

- ルール議論リポジトリ: [https://github.com/VNN-COMP/vnncomp2025/issues](https://github.com/VNN-COMP/vnncomp2025/issues)
- Benchmark リポジトリ: [https://github.com/VNN-COMP/vnncomp2025_benchmarks](https://github.com/VNN-COMP/vnncomp2025_benchmarks)
- 結果リポジトリ (Results repository): [https://github.com/VNN-COMP/vnncomp2025_results](https://github.com/VNN-COMP/vnncomp2025_results)
- レポート (Report): TBD（未定）

**変更履歴 (Changes and History):**

- 2026: 2025 年版をベースに初稿。**SAIV 2026 @ FLoC 2026**（2026 年 7 月 24–25 日、ポルトガル・リスボン）向けに更新。
  - **Competition Contribution short papers（4–6 ページ）は LNCS の SAIV proceedings に収録される**
- 2025: [2025 年ルール](https://github.com/VNN-COMP/vnncomp2025) 参照
- 2024: 2 トラック（regular と extended）を導入、meta-solver の定義を廃止

---

## 概要 (Overview)

VNN-COMP は、ニューラルネット検証手法を benchmark 上で評価する。評価軸は (1) benchmark 群にわたって**いくつの property を verify / falsify できるか**、(2) **与えられた計算資源の範囲内で**、の 2 点。スコアの仕組みは後述するが、本質は「**与えられた計算予算内でできるだけ多くの property を証明（prove）または反証（falsify）すること**」に報酬を与える。これにより**精度（precision）とスケーラビリティ（scalability）の両方**を評価し、コミュニティの課題を浮き彫りにする。

## 用語 (Terminology)

用語は SAT/SMT コンペで使われるものに準ずる（VNN-LIB 言語が一部 SMT-LIB に基づくため）。

**Instance（インスタンス）**: 入力 (input) + property + ニューラルネットモデル、および timeout からなる。
- 例: MNIST 分類器 + 入力画像 1 枚 + ある局所 robustness しきい値 ε

**Benchmark（ベンチマーク）**: instance の集合。
- 例: ある特定の MNIST 分類器 + 入力画像 100 枚 + ある robustness しきい値 ε

**各 instance にツールを走らせた結果として起こりうるもの:**

- **unsat**: `.vnnlib` 内の全制約を同時には満たせない（`.vnnlib` は通常 counterexample、すなわち「望ましい property の否定」をエンコードする。counterexample が存在しなければ望ましい property が成立し、`unsat` を出力すべき）。直感: 「unsafe な出力」はあり得ない。
- **sat**: `.vnnlib` の全制約を同時に満たせる。sat の witness とは、全制約を真にする `.vnnlib` 内変数への割り当て。直感: 許容入力から「unsafe な出力」があり得る。
- **timeout**: 実行時間が instance ごとの timeout を超過。
- **error**: instance 実行がエラー／クラッシュ。
- **unknown**: その他。

| 返した結果 \ 正解 (GT) | Unsat（成立） | Sat（違反） | Timeout | Error | Unknown |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Unsat（成立） | Correct（正） | Incorrect（誤） |  |  |  |
| Sat（違反） | Incorrect（誤） | Correct（正） |  |  |  |

## 競技フェーズ (Competition Phases)

1. **Pre-Competition Phase（事前準備）**: ツール作者は自分のツール用に 3 つのスクリプトを用意する。各スクリプトの役割は前年リポジトリの README 参照: [https://github.com/stanleybak/vnncomp2021](https://github.com/stanleybak/vnncomp2021)。

2. **Rules Discussion and Benchmark Solicitation Phase（ルール議論・benchmark 募集）**: 参加者とルールを議論・確定する。ツール作者やその他の参加者が benchmark を提案し、正しい形式かを website 上で検証する。benchmark instance は `.onnx` ネットワーク、`.vnnlib` 仕様ファイルを持ち、**random seed に基づいて instance の `.csv` を生成するスクリプト**を含むこと。

3. **Measurement Phase (unofficial)（非公式計測）**: ツール作者が website 上で、既知の seed を使い全 benchmark に自ツールを走らせる。エラー修正のため複数回実施可。website: https://vnn.repeatability.cps.cit.tum.de/

4. **Measurement Phase (official)（公式計測）**: 主催者が website 上で、**新しい random seed** を使って benchmark instance を選び、全ツールを全 benchmark に走らせる。

5. **Reporting Phase（報告）**: 公式計測の結果発表後、参加者は summary report 用テキスト（自ツールや benchmark の説明）の寄稿を求められる。レポートは arXiv に掲載され、著者は主催者。結果の不一致は主催者へ報告でき、arXiv レポートで更新される。

## 入出力形式 (Input and Output Formats)

**入力形式**: `.onnx` と `.vnnlib`

**実行結果の出力形式**: 結果文字列（"sat", "unsat", "timeout", "error", "unknown"）を含む単一のプレーンテキストファイル。"sat" の場合、ファイルの 2 行目に各変数への割り当て（witness）を記す。"sat" 出力時に witness が無いとペナルティが科される。**ただし** onnxruntime によるネットワーク実行が witness と小さな数値精度内（相対誤差 1e-3 未満のミスマッチ）で一致し、かつ制約が絶対誤差 1e-4 以内で満たされていればペナルティは無い。witness 形式の出力例:

```
sat
((X_0 0.02500000074505806)
 (X_1 0.97500000000000000)
 (Y_0 -0.03500000023705806)
 (Y_1  0.32500000072225301)
 (Y_2 0.02500000094505020))
```

**検証 instance の実行プロセス:** 検証 instance は 1 件ずつ実行され、instance 準備スクリプトと、実際に spec をチェックする instance 実行スクリプトを交互に呼び出す。

## 評価ベンチマーク計画と実行インスタンス (Evaluation Benchmarking Plan and Execution Instances)

* 実行は **AWS** 上で行う。
* 注: 本節は暫定。ここで挙げる instance を確保できるか確認中で、他の instance タイプもルール議論フェーズで提案可能。
* クラウド評価プラットフォームは 3 種（CPU 重視・GPU 重視・"balanced"）。各々ほぼ同じ コスト/時間。各ツール作者は全 benchmark で参加する 1 プラットフォームを選ぶ。instance は以下（暫定・在庫次第。他を希望なら github issues にコメント）:
  - **CPU:** m5.16xlarge, $3.072 / hour, 64 vCPU, 256 GB memory
  - **GPU:** p3.2xlarge, $3.06/hour, 8 vCPUs, 61 GB memory, 1x V100 GPU
  - **Balanced:** g5.8xlarge, $2.44 per hour, 32 vCPUs, 128 GB memory
* 理由: CPU のみのツールと GPU を使うツールがあるため、おおよそコスト等価の instance を比較基準とした。本方式には一長一短がある。

## ランタイム上限 (Runtime Caps)

* **instance ごと**: 各検証 instance は最大 X 分（benchmark 提案者が決定）で timeout。instance ごとに異なる値でよい。
* **benchmark ごと**: benchmark 内の全検証 instance の実行時間合計は**最大 6 時間**。

例: ある benchmark 提案は「1 時間 timeout の instance 6 件」でも「3.6 分 timeout の instance 100 件」でもよい。

報告・表彰のため、主催者が benchmark をさらにグループ分け（画像分類、制御、新規 vs 既存 benchmark 等）することがあるが、これはスコアに影響しない。

## スコアリング (Scoring)

各研究グループは複数ツールを提案してよい。ただし提案する各ツールは他と**実質的に異なる**こと（例: 同一ツールのパラメータ違いは不可）。ツールは実際に異なる必要があり（例: 単にヒューリスティクスのパラメータが違うだけの同一ツールは不可）、際どいケースは主催者が参加者と議論して判断する。理由: 参加者が同一ツールのヒューリスティクス違いを多数提出しうるため。

**Meta-solvers**: 2024 年に定義を廃止（実際には現れず、コード差分の割合に基づく判定は実行困難なため）。他のライブラリを再利用し他者の研究の上に構築する場合は、標準的な学術的礼節として、レポート・ツール説明・発表概要で依存関係を明記すること。依存関係は benchmark 投票の提出時に収集する。

## Benchmark と Benchmark 選定 (Benchmark and Benchmark Selection)

スコアリングには 2 トラックが存在する: **Regular track** と **Extended Track**。
- **Regular track**: ツール参加者の **50% 以上**がスコア対象として投票した benchmark。
- **Extended track**: **1 票以上**を得てスコア対象とされ、かつ Regular track に入らない benchmark。
- 無投票の benchmark: スコア対象外だが、レポートには記述される。

理由: ツール参加者には汎用的なアーキテクチャ対応を奨励し、できるだけ多くの benchmark を扱う動機づけをする（ゆえに Extended track 入りは 1 票で足りる）。一方、ある benchmark を 1 つかごく少数のツールしか扱えない場合、それは competition というより challenge であり、同じ benchmark を解析できる他ツールと比較した「最良性能」を必ずしも評価できないため。

誰でも benchmark を提案でき、提案数に上限はない。benchmark 提案者とツール参加者の間の協力関係はすべて開示すること（benchmark 投票プロセスで収集）。

非ツール参加者（検証ツール利用に関心のある産業グループ等）も benchmark を提案できる。

標準データセット（MNIST や CIFAR 等）の分類 benchmark では、特定画像サブセットに合わせ込むのでなく、**データセット内の任意画像で動く汎用形**にすること。競技計測ではこの場合、主催者が random seed で対象サブセットを選ぶ。非画像 benchmark では、提案者が **seed に基づき benchmark をランダム選択／変異させる手段**も提供すること。全 benchmark は、渡された random seed に基づき各 benchmark instance を列挙する `.csv` を生成するスクリプトを含むこと。過去の seed 選択方法は[このファイル](https://github.com/stanleybak/vnncomp2021/blob/main/benchmarks/generate_random_instances.sh)参照。

## Benchmark 提案 (Benchmark Proposals)

新しい benchmark を提案するには、必要なコードを全て含む**公開 git リポジトリ**を作成し、github issues に benchmark 詳細とリンクを投稿する。

リポジトリは以下の構成にすること:

* `generate_properties.py` ファイルを含み、**seed を唯一のコマンドライン引数として受け取る**こと。
* 全 `.vnnlib` ファイルを置くフォルダがあること（`generate_properties.py` と同一フォルダでも可）。
* 全 `.onnx` ファイルを置くフォルダがあること（`generate_properties.py` と同一フォルダでも可）。
* `generate_properties.py` は **Python 3.8** で **t2.large** AWS instance 上で実行される。

## スコアリング詳細 (Scoring Details)

各 instance のスコア:

* 正しい unsat: **10 点**
* 正しい sat: **10 点**
* 誤った結果 or ペナルティ: **-150 点**
* Timeout / Error / Unknown: **0 点**

**Time bonus（時間ボーナス）:** **無し**。過去の分析では総合スコアに差が出なかったため。とはいえ、instance timeout 到達前に解析を完了させるためコード最適化の動機は残る。

各カテゴリの benchmark スコアはパーセント表示で、「**そのカテゴリでの個別 instance スコア合計 × 100 ÷ 任意ツールの当該カテゴリ instance スコア合計の最大値**」で計算（例: あるカテゴリで instance スコア合計が最大のツールが 100% を得る）。

ツールの総合スコアは benchmark スコアの合計（パーセントの和）。よって各 benchmark は等しい重みを持つ。

## ルール変更 (Rule Changes)

benchmark 確定前は、参加ツールが本文書のルール変更を提案できる。これは "rules discussion" github issue 上で行う。チームは**具体的な**変更案を示す "motion"（動議）を投稿する（例: 「"xyz" の文を削除し "ijk" に置換」）。別チームが（github issue で返信して）second すれば、主催者は google form 等の投票機構を作り、参加ツールごとに 1 票を割り当てる。結果は公開。変更が承認されるには参加ツールの**過半数が賛成投票**する必要がある。議論・投票で合意に至らない場合は主催者が裁定する。

## 賞金・表彰 (Cash Prizes & Awards)

スポンサーの有無・法的/税務制約を条件に、総合スコア最上位ツール（benchmark スコア合計が最大のツール）に賞金を提供。各 benchmark / カテゴリの最上位ツールには award 証書。同点は全解決 instance の総実行時間で決する。**VNN-COMP 主催者や財務スポンサーを開発者/著者に含むツールは賞金を受け取れない**（カテゴリ/総合優勝や証書は可）。「best falsifier」賞は、あるツールがこの点で突出していれば主催者が検討。その他の賞は主催者裁量。

## 名誉規定と主催者裁量 (Honor Code and Organizer Discretion)

不正検出の対策は講じるが、最終的には参加者がフレンドリーな精神を守り、不当な優位を得ようとしないことを望む。例: 実際に property の成立を証明していない限り instance に "unsat" を出力すべきでない。名誉規定違反は主催者がケースバイケースで裁定する。

## タイムライン (Timeline)

最新タイムラインは今年の website 参照: [https://vnn-comp.github.io/#vnncomp2026](https://vnn-comp.github.io/#vnncomp2026)

## 免責 (Disclaimers)

主催者は、円滑な運営・想定外の問題への対処・参加促進・曖昧さの解消のためにルールを変更できる。今年の主催者: Taylor T. Johnson, Stanley Bak, Christopher Brix, Tobias Ladner, Lukas Koller, Konstantin Kaulen, Edoardo Manino, Thomas Flinkow, Haoze Wu, Hai Duong, ThanhVu H. Nguyen。主催者もツールを提出し競技参加できるが、本人もチームも賞金は受け取れない。

---

## ツール作者向け手順 (Instructions for Tool Authors)

クラウド上での自動評価のため、ツールを標準形式で、セットアップ・実行用の bash スクリプトと共に用意する必要がある。スクリプトは web からの資源取得（git リポジトリの clone 等）に使える。将来の再現性のため、**特定 commit / tag を clone する**のが望ましい。

### ワークフロー (Workflow)

VNN-COMP への提出は website 経由: https://vnn.repeatability.cps.cit.tum.de/

ネットワークは VNNLIB 標準準拠の `.onnx` 形式で提供。property は `.vnnlib` ファイルで提供。`.onnx` と `.vnnlib` は benchmark リポジトリの clone により home フォルダにダウンロードされる。例: テスト用ネット/property は `~/vnncomp_repo/benchmarks/test/test.onnx` と `~/vnncomp_repo/test/test.vnnlib`。

各ツール作者は、選択した amazon AMI（OS）が動く amazon クラウド instance への（間接）アクセスを与えられ、そこでツール用ライセンス等もダウンロードできる。website に無い別 AMI を追加したい場合は evaluation chairs に連絡。

その後ツールのスクリプトが home フォルダにコピーされる。評価プラットフォームは `install_tool.sh` を実行してツールを導入し、各 benchmark instance ごとに `prepare_instance.sh` → `run_instance.sh` を呼び、結果ファイルを生成させる。

benchmark カテゴリ全体をスキップしたい場合、`prepare_instance.sh` が非ゼロ値を返すようにできる（カテゴリはコマンドライン引数で渡される）。

### スクリプト (Scripts)

各ツールにつき 3 つのスクリプトを用意する。

* **install_tool.sh**: 引数 1 つ "v1"（バージョン文字列）。一度だけ実行され、依存関係のダウンロード、ファイルのコンパイル、必要ライセンスのセットアップ（自動化可能なら）等を行う。自動取得できないライセンスもあるため、その場合はスクリプト実行前にツール作者が手動でライセンス取得する責任を負う。

* **prepare_instance.sh**: 引数 4 つ。第 1 = "v1"、第 2 = benchmark 識別子文字列（例 "acasxu"）、第 3 = `.onnx` ファイルへのパス、第 4 = `.vnnlib` ファイルへのパス。benchmark を評価用に準備する（例: onnx を pytorch に変換、vnnlib を読んで当該 property の C++ ソースを生成し gcc でコンパイル）。次の benchmark 計測に向けシステムを良好な状態にする用途にも使える（前回実行のゾンビプロセスが無い、GPU が利用可能、等）。**このスクリプトで解析を行ってはならない**。妥当な timeout（10 分）が課され、超過すると当該 instance の結果は "unknown" 扱い。benchmark 名は渡される（benchmark 単位のチューニング/設定は可。**instance 単位の設定は不可** → ツール設定のカスタマイズに onnx/vnnlib のファイル名を使ってはならない）。カテゴリ全体スキップは非ゼロ値の返却で可。

* **run_instance.sh**: 引数 6 つ。第 1 = "v1"、第 2 = benchmark 識別子文字列（例 "acasxu"）、第 3 = `.onnx` パス、第 4 = `.vnnlib` パス、第 5 = 結果ファイルへのパス、第 6 = timeout（秒）。timeout を大きく超えるとスクリプトは kill される（GPU 等の資源を綺麗に解放したいなら自発的に終了するのが望ましいこともある）。結果ファイルはスクリプト実行後に作成され、1 行に 1 単語を含む単純テキスト: **holds, violated, timeout, error, unknown** のいずれか。

問題があれば本リポジトリの対応する github issue に投稿: [https://github.com/VNN-COMP/vnncomp2026/issues](https://github.com/VNN-COMP/vnncomp2026/issues)

---

## 我々（llcore / online-arch-evo 提案）にとっての要点メモ

> 訳者補足（原文には無い・理解用）。Path B 検討の足がかり。

- **benchmark 提案は誰でも可・上限なし**。**非ツール参加者でも提案できる**（[Benchmark Selection] 節）。= `online-arch-evo` のカテゴリ/benchmark 提案を出す資格は問題なくある。
- 提案の実体要件 = **公開 git repo** に `generate_properties.py`（seed 引数）+ `.onnx` フォルダ + `.vnnlib` フォルダ。**Python 3.8 / t2.large で動くこと**。→ これが [[project_llcore_init_2026_05_29]] の §9 item 10（`.onnx`/`.vnnlib` パーサ未実装）を埋める必要がある所以。
- ただし現行ルールは **「固定ネット 1 件を解く」前提**（入力 = `.onnx` + `.vnnlib`、出力 = holds/violated/…）。我々の「**ChangeOp の流れを 1 手ごと差分検証**」は、この入出力契約に**そのままは載らない**（= 提案論文の contract-mismatch 論点と一致）。よって Path B は「既存ルールに benchmark を 1 つ足す」のでなく、**rules discussion issue で新カテゴリの I/O 契約自体を提案・議論する**動きになる（[Rule Changes] の motion 機構、または organizers への新カテゴリ相談）。
- **Competition Contribution short paper（4–6 ページ、LNCS、SAIV proceedings 収録）**という発表枠の存在も判明 = 将来エディションで「カテゴリ提案 + benchmark」を short paper として出す経路がある。
- 主催者（裁定者）: Taylor T. Johnson, Stanley Bak, Christopher Brix ほか。**コミュニティ駆動でルール変更は motion → second → 過半数投票**。
