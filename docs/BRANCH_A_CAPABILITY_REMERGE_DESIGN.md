No branch-A reference in next_plan; I'll treat "branch A (capability 再合流)" as the orchestrator's framing — the prospective integration item (c) from POSITIONING_VS_LLAMACPP.md §2. I have everything needed. Producing the integration design now.

---

# 統合設計: branch A — メモリ効率指標 × verified-plasticity gate の進化探索統合

> 種別: 設計文書(コード非実装)。正本数値参照先 = `docs/MEMORY_EFFICIENCY_FINDINGS.md` / `docs/POSITIONING_VS_LLAMACPP.md` §2(c)。
> honest 規律は POSITIONING_VS_LLAMACPP.md の作法(再導出/独自の明示分離・過大主張の構造的防止)を全面踏襲する。
> 配置案: `D:/projects/llcore/docs/BRANCH_A_MEMORY_FITNESS_GATE_DESIGN.md`

## 0. 一行要旨と立ち位置の固定

branch A は **「capability を取り戻す」ものではない**。capability の進化探索は `evolution_vs_random().passes=False`(進化≒random)が確定済みの**負け筋**であり、本統合はそこを再挑戦しない。branch A が再合流させるのは、**capability アークで作った機構資産(GA + verified-plasticity gate + falsification harness)を、勝ち筋である「メモリ効率」北極星に転用する**ことである。

一行で: **「メモリ効率指標(footprint / cliff / retention)を進化の適応度にし、その探索を verified-plasticity gate で fail-closed に選別する統合層」。** 新規アルゴリズムではなく「既存資産の北極星への再配線 + honest な統合実証」。

---

## 1. 統合アーキ

### 1.1 全体図(探索ループのどこに何を噛ませるか)

```
[初期集団 init] ──► [子生成 _make_child: tournament+crossover+mutate]
                          │
                          ▼
                ┌─────────────────────────────┐
                │ verified-plasticity gate     │  ← (A) fail-closed フィルタ
                │ _gate_admits(child, mode)    │     既存 evolve() に配線済み
                │ admit?  no → resample(cap)    │     ("contraction"/"trajectory_tube")
                │         cap到達 → _FALLBACK   │
                └─────────────────────────────┘
                          │ admit された子のみ
                          ▼
                ┌─────────────────────────────┐
                │ メモリ効率 fitness           │  ← (B) 新 fitness 関数(本設計の主役)
                │ memory_fitness(gene)         │     accuracy×memory を多目的→スカラ化
                │  = w_a·retention + w_m·(1-foot)│    or Pareto rank
                └─────────────────────────────┘
                          │ 適応度
                          ▼
                [選択 → 次世代] ──► … ──► [falsification: evolution_vs_random]
                                                  ← (C) 進化 vs random 同予算検証
```

要点: **gate は子生成直後(fitness 評価の前)**、**memory fitness は admit 後の評価関数**。この順序は既存 `evolve()` の構造(L565-589: `_gate_admits` → `evaluate_population`)に一切手を入れず成立する。これが最小改変の核心。

### 1.2 (A) gate の配線 — 既存資産を「そのまま使う」

`evolve(..., gate_mode="contraction" | "trajectory_tube", resample_cap=50, w_bar=..., r_max=...)` は**既に存在し配線済み**(`src/llcore/evolution/minimal_ga.py` L390-614)。branch A は gate に対して **新規実装を要しない**。やることは:

- メモリ fitness 探索を回すとき `gate_mode="none"`(control)と `gate_mode="contraction"`(gated)の **2 本を必ず対で回す**(下記 PoC G2)。
- gate の意味づけ: メモリ効率を最大化しようとする進化圧は、表現力を上げる方向 = **収縮境界 ρ→1 に張り付く方向**に gene を押す(ARTICLE_SEEDS #16「進化は gate があってもなくても境界を目指す」)。gate は「メモリ的に有利だが力学的に発散する gene」を admit しないための安全弁として働く。

> honest 注記(POSITIONING §2(c)準拠): gate の構成要素 — sound contraction 証明 / fail-closed resample / known-safe fallback — は **すべて prior-art**(CEGIS / Deb 2000 feasibility rule / shielding / Simplex)。branch A は gate を発明しない。

### 1.3 (B) メモリ効率 fitness — 本設計で新規に作る唯一の実質コンポーネント

#### 多目的の定式化(2 案、PoC は案 1 から)

メモリ効率と capability(retention)は **トレードオフ**(低 footprint ほど retention が崩れる: `MEMORY_EFFICIENCY_FINDINGS.md` (b')cliff)。これは多目的問題そのもの。

**案 1(PoC 既定): スカラ化(重み付き和)** — MnasNet 流。既存 `evolve()` がスカラ fitness 前提なので**改変ゼロで載る**。

```
memory_fitness(gene) = w_acc · retention(gene) + w_mem · (1 − norm_footprint(gene))
   retention(gene)     ∈ [0,1]  = top-1 retention vs fp32 ref(eval.py の既存量)
   norm_footprint(gene)∈ [0,1]  = footprint_bytes(gene) / footprint_fp32
   既定 w_acc=0.7, w_mem=0.3(MnasNet の reward 重み感に倣う)
```

**案 2(将来): Pareto front(NSGA-II 風)** — LEMONADE / OFA² 流。1 run で accuracy×memory の front を出す。ただし `evolve()` のスカラ選択を多目的選択へ拡張する必要があり**改変が大きい**ため PoC スコープ外。案 1 で「統合が機能する」を先に示してから検討。

#### gene と「メモリ指標」の接続 — ここが設計の肝(かつ honest 留保の最大点)

現行 GA の gene は `StateUpdateGene(decay, mix, gate_str)` の **3 次元 RWKV 状態更新遺伝子**であって、量子化ビット幅や層構成ではない。よって「footprint をそのまま gene が決める」わけではない。**2 つの接続経路**を分けて設計する:

- **経路 P1(PoC 推奨・安全): gene → state footprint 直結。** 状態更新 gene は recurrent state の力学を決める。「定数状態 vs 文脈線形」(`(0)`/`(0')`)の構造的勝ち筋は gene の収縮性に依存する。fitness の memory 項を **「証明可能に有界な状態を保ちつつ task retention を保持できるか」** に置く。norm_footprint は gene の収縮率(`empirical_lipschitz` / cert backend)由来の **state-boundedness proxy** とし、retention は fixed-readout probe fitness(`tasks.py`)を流用。= **既存 fitness 資産だけで閉じる**。
- **経路 P2(将来・拡張): 探索空間を量子化ビット幅へ拡張。** HAQ/APQ 流に「層ごとビット幅」を gene に含め、footprint=`int8_footprint_bytes`、retention=`passes_capability_gate` を直接 fitness 化。これは LM 本体(`lm/quant.py`/`lm/eval.py`)と GA を結合する大改変。**PoC スコープ外**だが branch A の本来の射程(POSITIONING §2(c)が指す統合)。

> **最重要 honest 留保**: 現状 PoC で測れるのは **P1(合成タスク × state 力学)** であって、`MEMORY_EFFICIENCY_FINDINGS.md` の実 char-LM footprint(539→149MB 等)を **直接 fitness にした探索ではない**。「メモリ実測を進化適応度にした」と書けるのは P2 到達後。PoC は「**メモリ効率方向の代理指標を fitness にした gated 探索が機能するか**」までを誠実なスコープとする。

### 1.4 既存テストを壊さない最小改変

| 改変 | 方式 | 既存テストへの影響 |
|---|---|---|
| メモリ fitness 関数 | `src/llcore/fitness/` に **新ファイル**(例 `memory_objective.py`)で `SyntheticTask` Protocol に準拠した `MemoryEfficiencyObjective` を additive 追加 | ゼロ(既存 `tasks.py` 不変) |
| gate 配線 | **改変なし**(`evolve()` の `gate_mode` を呼び出し側で渡すだけ) | ゼロ(`gate_mode="none"` byte-identical 保証は既存) |
| falsification | **改変なし**(`evolution_vs_random(eval_once=memory_objective)` に渡すだけ) | ゼロ |
| PoC スクリプト | `scripts/poc_branch_a_memory_fitness.py` を新規追加 | ゼロ |
| 多目的スカラ化重み | fitness 関数の引数(default 固定)で外出し | ゼロ |

設計原則(`evolve()` の既存規律と一致): **新フラグ・新関数・新ファイルの additive 追加のみ。既存シンボルの signature と挙動を 1 bit も変えない。** これは ARTICLE_SEEDS #15「安全機構は規律であって最適化ではない=新しい探索器に載せ替えるたびに規律ごと移植」の逆運用 — gate 規律が既に `evolve()` に載っているので、fitness を差し替えるだけで規律は自動的に効く。

---

## 2. 再導出 vs 独自(honest・最重要)

POSITIONING_VS_LLAMACPP.md の作法に従い、**prior-art に照らして既知のものは独自と書かない**。

### 2.1 既知(再導出)— 独自と書いてはいけないもの

| 構成要素 | 一次情報(prior-art) | 判定 |
|---|---|---|
| メモリ footprint を適応度/目的に組み込む | HW-NAS の定義要素。MnasNet(2018, reward=ACC×[LAT/T]^w)起点、survey arXiv:2101.09336 が latency/energy/memory を標準カテゴリ化。MicroNAS/μNAS/MicroNets が peak memory を MCU 向けに明示目的化 | **再導出(既知パラダイム・新規性ゼロ)** |
| accuracy×memory の多目的/Pareto 探索 | NSGA-II が事実上標準(LEMONADE 2018 / OFA² 2023)。SMS-EMOA / qEHVI / scalarization(MnasNet 流)も定番。HV/IGD 評価も定石 | **再導出(既知)** |
| 量子化耐性/retention を探索目的に | HAQ(2018, RL×hardware sim reward)/ APQ(2020, quantization-aware accuracy predictor を fitness)/ robustness を zero-cost proxy で joint 予測(arXiv:2307.09365) | **再導出(既知)** |
| fail-closed で実行不能解を棄却する制約付き進化 | Deb 2000 feasibility rules / death-penalty(PyGMO)。出発点であり新規性ゼロ | **再導出(既知)** |
| 検証器を探索ループの gate にする | CEGIS(arXiv:1505.03953)/ Neural Lyapunov + Fossil / shielding / Simplex runtime-assurance。ローカル `verified_safe_learning_corpus_v2`(97ノート)に既収録 | **再導出(既知)** |
| 収縮 ρ<1 / Lyapunov 証明 | safe-control の常套 | **再導出(既知)** |

→ つまり **「メモリ指標を適応度に」「制約 fail-closed」「検証器を gate に」は、それ単体ではすべて既知**。記事・対外発信で「独自」「新パラダイム」と書いてはならない(POSITIONING チェックリスト準拠)。

### 2.2 残る差別化(控えめに)— llcore 固有と言えるもの

prior-art 検証を経ても単一先行研究に未見の **狭い結合**のみを、**「実証前/未踏点」と明記して控えめに**主張する:

- **(i) 三者結合の特定文脈:** 「**進化的探索 × sound 収縮(verified-plasticity)gate × メモリ効率北極星 × LLM/recurrent hidden-state 力学**」の四点交差の具体実装。従来の verified-control は手設計制御器・固定構造が主で、進化探索の探索子棄却に sound 収縮 cert を据え、かつ目的をメモリ効率に取る組合せは prior-art(HW-NAS は経験的 proxy gate、safe-control は固定構造)に**完全一致する先行が未確認**。ただし隣接領域は濃いため「**新規原理ではなく、既知レンガの未踏な結合**」と書く。
- **(ii) gate 判別力を第一級指標にする評価枠組み:** 経験 gate(STABLE 風 ~84% false-admit)vs sound cert(0% false-admit)の判別力を、メモリ fitness 探索の文脈で測る(ARTICLE_SEEDS #16/#14 の延長)。ただし shielding 評価・Simplex 切替評価の延長であり**原理的新規ではない**。
- **(iii) 自宅 CPU 再現性:** GPU 不要・少 RAM で「メモリ効率 fitness × gated 進化」を誰でも再現できる教育/自己検証レイヤ(POSITIONING §1 の再導出価値 3 点と同じ位置づけ)。

> **総括の honest 一文**(記事・doc 冒頭に固定): 「branch A は **既知の HW-NAS / 多目的 NAS / 制約付き進化 / verified-control を、llcore のメモリ効率北極星 × recurrent 力学に転用した統合**である。新規圧縮アルゴリズムでも新規探索原理でもない。誠実に主張できるのは『**狭い四点結合の実証 + gate 判別力の計測 + 自宅 CPU 再現性**』に限る。」

---

## 3. CPU 小実証計画(PoC)

### 3.1 目的と「機能した」の定義

PoC が示すべきは「**メモリ効率方向の fitness で進化を回したとき、verified-plasticity gate が feasible(安全)解の質を有意に改善するか**」。capability アークが `passes=False` だった事実を踏まえ、**「進化が random に勝つ(capability)」は主目的にしない**。主目的は **「gate あり/なしで、admit された解の力学的安全性(ρ<1 保持率)と、その安全制約下でのメモリ効率の到達点が改善するか」**。

### 3.2 実験デザイン(2×2 + 予算統制)

`scripts/poc_branch_a_memory_fitness.py`(新規)で以下を回す。各セルは `base_seed` を ≥15 seed で振る:

| 軸 | 水準 |
|---|---|
| gate | `none`(control)/ `contraction`(gated, sound) |
| fitness | `memory_objective`(w_acc=0.7, w_mem=0.3) / `retention_only`(w_mem=0 = capability ベースライン) |

各 run で記録(別プロセス隔離は不要 — 量子化実測でなく合成タスクのため。P2 で実 footprint を測るときは `MEMORY_EFFICIENCY_FINDINGS` 流の別プロセス隔離を踏襲):
- 採用 gene の `empirical_lipschitz`(ρ)と cert backend 判定(発散境界からの距離)
- norm_footprint / retention の内訳(スカラ化前の 2 値を必ず両方残す = Pareto 観察用)
- `GateStats`(n_rejections / n_resamples / fallback_count)= gate のコスト構造(ARTICLE_SEEDS #13)
- `evolution_vs_random()` の `FalsificationResult`(diff / wilcoxon_p / paired_sign_delta / passes)

### 3.3 何を測れば「統合が機能する」と言えるか(合否ゲート G1–G6)

| ゲート | 判定基準 | 何を意味するか |
|---|---|---|
| **G1 後方互換** | `gate_mode="none"` の memory_fitness 探索が既存 evolve と RNG byte-identical | 最小改変の証明(既存テスト不破壊) |
| **G2 gate 効力** | gated 採用 gene の ρ<1 保持率 ≫ control(例 control ~6% safe vs gated ~100% safe、ARTICLE_SEEDS #14/#16 の memory 文脈再現) | verified-plasticity gate がメモリ圧の下でも発散個体を弾く |
| **G3 fail-closed 健全性** | gate 不在 admit が 0(`_FALLBACK_GENE` 採用で空転しない、ARTICLE_SEEDS #15) | 「最初の admit」設計が memory fitness でも空転しない |
| **G4 トレードオフ可視化** | memory_objective が retention_only より norm_footprint を下げ、かつ retention 劣化が cliff 前で止まる | accuracy×memory トレードオフが探索で動く(Pareto の片鱗) |
| **G5 honest 反証** | `evolution_vs_random(memory_objective).passes` の真偽を**そのまま記録**(False でも隠さない) | capability アークの教訓を継承。memory fitness でも進化が random を有意に上回らない可能性を事前開示 |
| **G6 gate コスト構造** | reject 率 × resample で判定回数が膨張しないこと(`contraction` は µs 判定。ARTICLE_SEEDS #13 の cert_sdp 停滞を回避) | gated 探索の実効コストが現実的 |

**「機能した」の最小定義 = G1 ∧ G2 ∧ G3 ∧ G6。** G4 は「統合に価値の芽がある」、G5 は「過大主張しない」ための honest ゲート。**G5 が False(進化≒random)でも PoC は失敗ではない** — 「gate がメモリ探索を sound にする」という guarantee 側の主張は capability 側の勝敗と独立だから(ARTICLE_SEEDS #14「capability と guarantee の直交性」)。

### 3.4 TDD テスト項目(`tests/unit/test_branch_a_memory_fitness.py`、additive)

- `test_memory_objective_in_unit_range` — fitness ∈ [0,1]、w_acc+w_mem 正規化、retention/footprint 両端(完全保持/完全圧縮)で期待値。
- `test_memory_objective_satisfies_synthetic_task_protocol` — `SyntheticTask` Protocol 準拠(`evaluate_gene`/`evolution_vs_random` に渡せる duck typing)。
- `test_evolve_none_byte_identical_with_memory_objective` — **G1**: `gate_mode="none"` で memory_objective を使っても、同 seed で RNG draw 順が既存 evolve と一致(既存 `test_kernel_ga_generalization` の byte-identity 規律を踏襲)。
- `test_gated_admits_only_contracting_genes` — **G2**: gated 採用 gene が全て `verify_lipschitz_contraction(...).contraction is True`。
- `test_gated_fallback_when_no_admit` — **G3**: admit 不能集団で `_FALLBACK_GENE` が採用され `fallback_count>0`、空転しない。
- `test_falsification_result_recorded_not_asserted` — **G5**: `passes` の値を assert で固定せず、**記録される**ことだけ検証(honest 反証は値を強制しない)。
- `test_weight_zero_mem_reduces_to_retention_baseline` — w_mem=0 で retention_only と一致(ベースライン包含、ARTICLE_SEEDS #11「floor を族に包含」)。
- `test_gate_cost_bounded` — **G6**: `contraction` mode の n_resamples が resample_cap×children を超えない(コスト爆発しない)。

### 3.5 honest 留保(PoC スコープの境界)

- PoC は **P1(合成タスク × state 力学 proxy)**。実 char-LM の量子化 footprint(`int8_footprint_bytes`)を直接 fitness にした探索は **P2 = 将来**。
- norm_footprint は state-boundedness proxy であって RSS 実測ではない。「実機メモリを進化適応度にした」とは **書かない**。
- 速度・真の int8 GEMM は未測(`MEMORY_EFFICIENCY_FINDINGS` (b)留保と同じ)。

---

## 4. honest 見積もり(過大期待の構造的排除)

POSITIONING §2(c)・チェックリストに従い、**測っていない利得を捏造しない**。

- **capability を取り戻すか? → No(これは取り戻さない)。** capability の進化探索は `passes=False` が確定済み。branch A は memory fitness に切り替えても、**進化が random を capability で上回ることを目標にしない**。G5 で `evolution_vs_random.passes` が False に出る可能性が高い(memory 項を混ぜても探索の本質は変わらない)ことを**事前に開示**する。
- **この統合は本質的に何をするか? → 「メモリ効率方向の探索を sound にする(guarantee 側)」だけ。** 価値は capability 勝利ではなく、**「メモリ的に有利だが力学的に発散する gene を fail-closed で排除しながら探索できる」という安全性**に置かれる。これは ARTICLE_SEEDS #14「capability と guarantee の直交性」の memory 文脈版。
- **「進化 × メモリ fitness」が prior-art を超えるか? → No。** 大半が HW-NAS / 多目的 NAS / 制約付き進化の再導出(§2.1)。新規は「四点結合の実証 + 判別力計測 + 自宅 CPU 再現」という**統合的・実証的貢献に限る**(§2.2)。
- **期待値の上限(誠実な天井):** PoC が全ゲート(G1–G3,G6)通過しても、言えるのは「**自宅 CPU で、メモリ効率 fitness の gated 進化が空転せず・発散個体を弾き・現実的コストで回ることを実証した**」まで。「メモリ効率を進化で最適化できる」「capability を回復した」は**書けない**。
- **失敗を消さない:** G5 が False、G4 でトレードオフが動かない等の null 結果も、`docs/poc/` の verdict として残す(capability アークの `passes=False` を消さなかったのと同じ規律。ARTICLE_SEEDS #2「過去の null が将来の設計判断を救う」)。

> **一行見積もり:** branch A は **capability を取り戻さない。メモリ効率方向の探索を verified-plasticity gate で sound にする(guarantee を付ける)だけ**であり、その価値と新規性は控えめ(既知レンガの未踏結合 + 計測規律 + 再現性)である。

---

## 5. 記事接続(ARTICLE_SEEDS deposit 先)

`docs/ARTICLE_SEEDS.md` は append-only。本統合の気付きは **2026-06-18 セッション見出し** の下に numbered seed として deposit する(collector 受理条件 = `## YYYY-MM-DD` date key + `### ` 配下に `**気付き**`/`**側面**` 同一行非空、ARTICLE_SEEDS #29)。各 seed は `気付き` + `根拠`(正本ポインタ)+ `側面`(13 側面)を必須とする。

| deposit する seed(side) | 気付きの核 | 効く記事側面 |
|---|---|---|
| **#41 capability を捨てた資産は guarantee 側に再合流できる** | 負け筋(capability 進化)で作った GA+gate+falsification 資産を、勝ち筋(メモリ効率)の「探索を sound にする」用途へ転用。機構は捨てず北極星だけ載せ替える(#35「構造プロット→実測」の pivot 作法の延長) | 戦略 / 哲学 / 教訓 / エコシステム |
| **#42 メモリ fitness を足しても進化は安全境界を目指す(gate の意味が出る)** | memory 圧は表現力↑=収縮境界 ρ→1 へ gene を押す。gate あり/なしで「境界のどちら側か」が分かれる(#16 の memory 文脈再現)。capability と guarantee の直交性(#14)が memory でも成立 | honest disclosure / ベンチ / 哲学 / 認知科学 |
| **#43 「メモリ指標を適応度に」は HW-NAS の再導出 — 独自は四点結合だけ** | MnasNet/HAQ/NSGA-II/Deb/CEGIS 全部既知。誠実な novelty は「進化×sound gate×メモリ北極星×recurrent 力学」の狭い結合 + 自宅 CPU 再現に限る(POSITIONING §2(c)準拠) | 業界比較 / honest disclosure / 戦略 |
| **#44 gated メモリ探索の 2 大落とし穴 = 空転と判別力** | 「最初の admit」設計(#15)と reject率×resample のコスト構造(#13)を memory fitness 文脈でも踏む。経験 gate vs sound cert の判別力を第一級指標に | 技術設計 / 教訓 / TRIZ / 実装報告 |

これらは `feedback_daily_articles_policy` の 13 側面のうち主に **戦略 / honest disclosure / 業界比較 / 技術設計 / 哲学** に効く。技術者向け(QIITA_SUMMARY 系)と非エンジニア向け(QIITA_GENERAL 系)を並走させる際、本 branch は「**負けた研究の機構を勝ち筋に再利用する**」という物語性(#41)が非エンジニア向けフックになり、「**メモリ指標を適応度にするのは NAS の常識・独自は狭い**」という honest 開示(#43)が技術者向けの信頼を作る。

---

## 関連ファイル(絶対パス)

- 設計配置先(本文書): `D:/projects/llcore/docs/BRANCH_A_MEMORY_FITNESS_GATE_DESIGN.md`
- gate 配線済み GA(改変不要): `D:/projects/llcore/src/llcore/evolution/minimal_ga.py`(`evolve()` L390-614, `_gate_admits` L232-308, `_FALLBACK_GENE` L229)
- 既存 fitness(不変・流用): `D:/projects/llcore/src/llcore/fitness/tasks.py`(`SyntheticTask` Protocol, `evaluate_gene`, `FixedReadout`)
- falsification harness(改変不要・eval_once 差し替えのみ): `D:/projects/llcore/src/llcore/evolution/honest_eval.py`(`evolution_vs_random`)
- 新規 fitness(additive 追加先): `D:/projects/llcore/src/llcore/fitness/memory_objective.py`(本設計で新規)
- 新規 PoC: `D:/projects/llcore/scripts/poc_branch_a_memory_fitness.py`(本設計で新規)
- 新規テスト: `D:/projects/llcore/tests/unit/test_branch_a_memory_fitness.py`(本設計で新規)
- メモリ実測正本(P2 接続先): `D:/projects/llcore/docs/MEMORY_EFFICIENCY_FINDINGS.md`、`D:/projects/llcore/src/llcore/lm/quant.py`(`int8_footprint_bytes`)、`D:/projects/llcore/src/llcore/lm/eval.py`(`passes_capability_gate`)
- honest 規律の作法(全面踏襲元): `D:/projects/llcore/docs/POSITIONING_VS_LLAMACPP.md` §2(c) + 付録チェックリスト
- 記事 deposit 先: `D:/projects/llcore/docs/ARTICLE_SEEDS.md`(2026-06-18 見出し下に #41–#44 を append)
