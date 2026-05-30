# 梯子段1 VERDICT — 複数 reservoir 結合で parity の床は外れるか

> 2026-05-30。research/ladder1_multi_reservoir/ 隔離、src 非変更、push 未。
> 設計=`LADDER1_DESIGN_multi_reservoir.md`。Step C verdict「次(CPU安価2本柱)(1)」の決着。

## 問い

Step C verdict: 単一 leaky reservoir + ridge readout は 5-bit parity (delayed_parity,
window=5) を解けない = **基質の床** (Minsky-Papert、全 method held-out R²≈0.016)。
**複数 reservoir 結合 (多細胞) で床が外れ、③ (分離が選択を可能にする) をクリーンに検定
できる土俵ができるか。**

## 方法 (2 系統 + 反証)

1. **deep 機構** (DeepESN 深さ): random search 到達天井を N_SEEDS=15 の strict gate
   (片側 Wilcoxon p<0.05 ∧ |paired_sign_delta|≥0.147) で測定。total_taps を揃えた
   公平比較で「深さ」と「規模」を分離。
2. **5 機構ワークフロー** (ladder1-floor-break): 原理の異なる床外し機構を並列実装・
   quick 測定 (held-out max R² > 0.5 で床外れと判定) し、各々を独立 agent が adversarial
   verify。`parallel_gated` (乗法結合=多細胞分業) / `quadratic_readout` (明示2次=readout
   対照) / `evolved_search` (進化で真天井) / `wide_single` (幅) / `hybrid_max` (全部入り anchor)。

全経路 held-out 厳守 (train/eval 別 draw、データリークなし)、各機構 pytest pass。

## 結果

### deep (DeepESN, N_SEEDS=15, exp_l1_results.json)

| 構成 | total_taps | max R² (mean) | vs 1L-8 | 判定 |
|---|---|---|---|---|
| 1L-8 (床) | 8 | 0.021 | — | Step C 床を再現 |
| 1L-16 (taps増のみ) | 16 | 0.019 | δ=−0.33 | 無効 |
| 2L-8×8 (深さ2・同規模16) | 16 | 0.051 | δ=+0.47 p=0.010 | **strict PASS** |
| 3L-8×8×8 (深さ3) | 24 | 0.096 | δ=+0.60 p=0.004 | **strict PASS** |

→ **深さ (層間非線形合成) は統計的に有意に床を持ち上げる** (同規模 taps 増は無効)。
**だが絶対値 0.05–0.10 で parity 未解決** (完全解 R²=1 に程遠い)。

### 5 機構ワークフロー (全 floor_lifted=false, genuinely_lifted=0)

| 機構 | best held-out max R² | baseline 1L-8 | attribution | 核心 |
|---|---|---|---|---|
| parallel_gated (乗法結合) | 0.012 | 0.130 | reservoir表現力 | 積項はノイズ。substrate が clean な per-bit 表現を持たず h^i⊙h^j が a·b を復元しない。データ飢餓仮説は n_train=400 でも棄却 |
| quadratic_readout (明示2次) | 0.055 | 0.130 | readout | **degree-2 は 2-bit XOR のみ**。理想 per-bit positive control (`exp_quad_positive_control.py`, raw held-out R²) で **window=2→R²=+1.0000 / window=3→−0.064 / window=4→−0.052 / window=5→−0.086** = degree-2 readout は degree-2 単項式のみ可解、degree≥3 は不能を実証 |
| evolved_search (進化) | 0.018 (3L-evolved honest) | 0.045 (1L-8) | reservoir表現力 | **探索不足でなく表現力限界**。3L-evolved の fresh-seed honest 再評価天井 0.018 が baseline (1L-8 random max 0.045) 以下 = 探索強化は無効。3L-random の見かけ天井 0.135 は random search の **単一 eval-seed への lucky-draw (selection-on-noise)** で別物 (elitism ではない、random に elite 持越しは無い) |
| wide_single (幅) | 0.045 | 0.045 | readout | 幅 8→64 で悪化。床は readout 側 (Minsky-Papert)、幅では超えられない |
| hybrid_max (全部入り) | 0.282 | 0.075 | reservoir表現力 | 最大表現力でも未解決 (held-out clip R²、保存値は clip 後)。HM-quad/HM-full の mean=0.000 = 2 次特徴は parity を解かない。clip後の単一 lucky-seed 値 0.282 は探索分散で解でない (degree≥3 が degree-2 readout で不能なことは positive control が raw R² で実証) |

## 最重要発見 (梯子段1 の核心)

**5-bit parity = degree-5 単項式 (b₁·b₂·b₃·b₄·b₅)。これを CPU reservoir + 線形/2次 ridge
readout で解くのは原理的に困難。**

- 「二次特徴で XOR が線形分離可能」は **2-bit XOR (degree-2) にのみ成立する古典結果**。
  理想 per-bit positive control (`exp_quad_positive_control.py`、完全記憶を仮定し reservoir
  ダイナミクスを切り離した上限テスト、raw held-out R²) が **window=2→+1.0000、window=3/4/5→
  −0.064/−0.052/−0.086** を出したのが決定的証拠。target は `bits[:window]` の積 = degree-window
  単項式なので、degree-2 readout は degree-2 (window=2) のみ表現でき degree≥3 は **完全記憶でも
  原理的に不能**。reservoir 上の見かけ 0.055 は探索分散の小正値で解ではない。
- 探索を random→進化に強化しても床は動かない (evolved_search)。**3L-evolved の fresh-seed
  honest 再評価天井は 0.018** で baseline (1L-8 random max 0.045) 以下 = 進化は床を外さない。
  (注: 3L-random の見かけ天井 0.135 は random search が単一 eval-seed 上で 300 本中 max を取る
  **selection-on-noise (lucky-draw)** による水増しで、進化の elitism 凍結持越し artifact とは
  別物 — random search に世代も elite 持越しも無い。honest 再評価は ES 経路のみに実装。)
- 幅・乗法結合・全部入りも床を外さない。共通原因 = **random/進化で引いた reservoir 最終状態が
  過去ビットの分離した線形表現を保持しない** (tanh 飽和混合) + **parity の高次数**。

## 結論 (honest)

**複数 reservoir 結合 (深さ/並列乗法/2次readout/進化/超ワイド/全部入り) では 5-bit parity の
床は外れない。** 深さは表現力をわずかに有意に上げるが degree-5 の壁を越えない。

「③ が無力」ではなく **「parity を解ける基質が梯子段1 では作れず、③ をクリーンに検定する土俵が
立たなかった」** — Step C と同型の結論だが、今回は **5機構 × 独立反証で「degree-5 の原理的
限界」として確定**した点が前進。CPU reservoir+ridge パラダイムの根本的天井を切り分けた。

## 次 (③ 検定の土俵を作る 3 つの道)

1. **window↓ (2–3 bit XOR)**: quadratic_readout が解けることを実証済 (R²=1.0)。そこで ③ を
   検定できるが、easy 化で**欺瞞性が薄れる**懸念 (Step C flip_flop の too-easy 罠と同型) → 単独では弱い。
2. **reservoir を勾配学習 (backprop)** で per-bit 分離保持: GPU/学習フェーズ (実 LLM landscape
   測定と同じ投資判断、現時点では保留)。
3. **Step C verdict (2) E-A = 多タスク分布** (風/天候、hold-out 汎化で③寄与、③ablation=MAP-E
   vs MAP-E_randselect): **parity の degree-5 床に縛られない** → **最有力** (CPU 可、欺瞞性を
   別構造で確保、③ 寄与を汎化で測れる)。

## §6(g) 反証打ち切りへの含意

Step C verdict §6(g): 「結合 reservoir + lexicase + 軌跡 fitness を全部入れても C3 不通過なら
③ 非 load-bearing 確定」。**梯子段1 は結合 reservoir が C3 検定の前提 (parity を解ける基質) に
到達しないことを示した** → parity 経路での ③ 判定は不能。lexicase baseline の投入は parity が
解ける土俵を前提とするため、**③ の load-bearing 判定は parity 経路を捨て E-A (多タスク汎化)
経路に移すのが筋**。無限後退は「parity に固執しない」ことで止める。

## 規律・成果物

- 全機構 held-out 厳守・リークなし・pytest pass・honest 内訳 (positive control / ablation /
  confound 調査 / 水増し排除) を各 honest_notes に保持。
- ワークフロー: 5 agent / 264 tool uses / genuinely_lifted=0。
- 実装: `research/ladder1_multi_reservoir/` (multi_reservoir.py + mech_*.py 5種 + tests + exp_*)。
- 関連: [[project_llcore_init_2026_05_29]] / [[feedback_codex_pair_review_for_llcore]] /
  [[feedback_benchmark_honest_disclosure]] (水増し排除) / Step C verdict。
- push 未 (ローカル保持)。Codex pair-review 後に commit。
