# Step C 適用条件 設計ノート — ③ (behavioral niching) が load-bearing になる欺瞞性閾値の特性化

> **凡例 — 進化4要素 (Darwin/Mayr)**: ①変異 / ②遺伝 / **③適者生存・選択** (= 本書の主役。「③」はこの 3 番目) / ④過剰繁殖。
> 状態: research/step_c_applicability/ 隔離・src 非変更・push 未。selection_lab は read-only import。

## 1. 動機 — 二値結果を特性化曲線に変える

確定済みの一次情報 (再導出しない):

- `STEP4_SELECTION_VERDICT.md` — **合成 deceptive corridor (exp4)** で③が decisively load-bearing
  (MAP-Elites が random/panmictic-GA/RR-hillclimb の 3 baseline 全勝, p=1.9e-6, δ=+1.00)。
  **smooth corridor (exp5)** では優位が消失。
- `E_A_VERDICT.md` — 多タスク汎化 (smooth landscape) で③は load-bearing でない (honest negative)。
- `STEP_C_VERDICT.md` — 記憶タスクは床/天井で C3 が非診断 (③をクリーンに検定できず N/A)。

問題: 「exp4 では③が立つ / smooth では立たない」という **二値 (binary)** の結果しかない。
科学的空白 = **「滑らか↔欺瞞」を連続パラメータで sweep し、③の優位を knob の関数として測る
制御された特性化 (parametric characterization)**。これにより過去の binary negative を
**特性化曲線**に変換し、「③が立ち始める欺瞞性の閾値 d*」を特定する。

> **⚠ 訂正 (Codex pair-review, 結果後)**: 本 knob は exp4/exp5 の eval 関数そのものの**厳密内挿ではない**
> (本 family は全 d で `max(local, glob, ramp*(1-dip))` を使い、exp4=`max(local,glob)`/exp5=広 Gaussian
> とは別)。正確には **exp4/exp5 に着想を得た ramp-with-dip の新 toy family** で、d=0=monotone smooth
> control・d=1=deep-dip deceptive control。得られる d* は **この family 内の閾値**。また `random` は
> 「d 非依存の定数敗北」ではなく reach が d 依存 (高 d で最強 baseline 化)。詳細は VERDICT §3 item 7-8。

## 2. Falsifiable な仮説

> **H1 (特性化仮説)**: corridor の欺瞞性 (= 局所峰と大域峰の間の谷 dip の深さ d) を smooth (d=0)
> から deceptive (d=1) に sweep すると、MAP-Elites の strict-gate 優位は **ある閾値 d* を境に
> 0 → 正に転じる**。d < d* では③不要 (hill-climbing で十分)、d ≥ d* で③ load-bearing。

**反証条件 (falsification)**: 以下のいずれかなら H1 は棄却または「閾値なし」と報告する。
- (a) 全 d で③が load-bearing でない (合成 corridor ですら③が立たない → 機構自体の否定)。
- (b) 全 d で③が load-bearing (d=0 の smooth ですら③が勝つ → 知見「smooth では不要」と矛盾、
  knob 設計の欠陥を疑う)。
- (c) advantage が d に対し非単調・ノイズだらけで閾値が定義できない (特性化失敗)。

「**clean な閾値が見つからない**」「**③は人工的に狭い regime でしか効かない**」も valid な結果。
positive を捏造しない (`feedback_benchmark_honest_disclosure`)。

## 3. 選んだ deceptiveness knob (primary を 1 つ選定 + 正当化)

候補は 3 つあった: (i) **dip 深さ**, (ii) behavior descriptor と fitness 勾配の **misalignment**,
(iii) corridor の **narrowness**。**(i) dip 深さ d を primary に選ぶ**。

### knob 定義

exp4 の genotypic corridor (behavior = mean(gene), 1D ∈[0,1]) を保ち、局所峰 (b=0.4, 高さ0.60) と
大域峰 (b=0.9, 高さ1.0) を **固定**したまま、両者を結ぶ単調登坂 ramp に **深さ d の谷 (dip)** を彫る:

```
local(b)  = 0.60 * exp(-(b-0.40)^2 / (2*0.08^2))
glob(b)   = 1.00 * exp(-(b-0.90)^2 / (2*0.06^2))
t(b)      = clip((b-0.40)/(0.90-0.40), 0, 1)
ramp(b)   = 0.60 + t(b)*(1.00-0.60)                 # 局所峰→大域峰を結ぶ単調登坂
dip(b)    = d * exp(-(b-0.65)^2 / (2*0.07^2))        # 谷の中央 b=0.65 に深さ d を彫る
f(b)      = max(local(b), glob(b), ramp(b)*(1-dip(b))) + N(0, 0.008)   (b∈[0.40,0.90])
```

- **d=0.0**: 谷無し → b: 0.4→0.9 が **厳密に単調増加** (downhill 0 step・正の勾配が常に存在)。
  = monotone smooth **control** (exp5 着想)。hill-climbing は連続した上り勾配で大域へ登れる → ③不要のはず。
- **d=1.0**: 谷の床 ≈ 0 → 深い dip = deep-dip deceptive **control** (exp4 着想)。hill-climb は downhill 拒否で罠。
- **中間 d**: dip の深さが連続変化。**唯一の自由度が「欺瞞 (dip) の深さ」**。

### なぜ dip 深さを primary に選んだか

1. exp4/exp5 の **本質的な差**が dip の有無。dip 深さはその差を連続化した最小の 1 パラメータ
   → exp4/exp5 を **着想元**とし d=0 (smooth control)/d=1 (deep-dip control) で挟む
   (eval 関数の厳密内挿ではない = 上記訂正ボックス)。
2. behavior=mean の **genotypic corridor 構造を全 d で不変**に保つ (corridor 幅の交絡を排除)。
   ただし **`random` の reach は d 依存** (高 d で最強 baseline 化、当初の「d 非依存定数」は誤り→訂正)。
   MAP-E 優位の変化は「dip 越え (= ③ の本質効果) の難易/不要化」に帰属し、load_bearing は 3 baseline
   全勝で保守的に判定する。
3. 峰の高さ・位置を固定するので「大域最適の価値」も d 非依存 → 優位の大小が**欺瞞性のみの関数**。
4. **ramp 基線**にすることで d=0 で hill-climb に **連続した上り勾配**を与え「dip が無ければ登れる」を
   保証 → 閾値が「勾配の有無」でなく「dip の深さ」の純粋な関数になる
   (単純な flat floor だと勾配 0 で hill-climb の登坂信号が消え、谷が浅くても smooth にならない=
   平床は弱い罠。これを避けるため床でなく正勾配 ramp を採用)。

(ii) misalignment / (iii) narrowness は behavior↔fitness の関係や corridor 構造自体を動かすため
random の失敗率も同時に変わり、「欺瞞性のみの関数」という清潔さが崩れる。dip 深さは最も交絡が少ない。
これらは将来の二次 knob として残す (本ノートでは primary 1 本に絞る)。

## 4. sweep 範囲・メトリック・統計

- **sweep**: d ∈ {0.0, 0.05, 0.10, 0.13, 0.16, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.0}
  (13 levels ≥ 要求の 6)。閾値が [0.10,0.20] にあると分かったので近傍を密に取り transition の
  sharp/gradual を解像。
- **メトリック**:
  - **MAP-E advantage** = MAP-Elites の honest 再評価 mean − best baseline の mean。
  - **strict gate pass** (honest_eval §5 完全版): `diff>0 ∧ 片側 Wilcoxon p<0.05 ∧ n_seeds≥15
    ∧ |paired_sign_delta|≥0.147`。selection_lab.compare は前 2 条件のみなので本実験で完全 gate を
    再実装 (n と効果量も課す)。
  - **load_bearing (厳格)** = MAP-E が **3 baseline 全て**に strict gate 勝利 (exp4 の③成立定義と同一)。
  - **n_baselines_beaten** (0-3) と **reach_rate** (honest fitness>0.8 = 大域峰 proxy) も記録。
- **規律** (一次情報の方針を継承):
  - seed = ea_lab.py 踏襲: 進化 RNG = `SeedSequence([base, method_idx, s])` で一意化、honest 再評価は
    index s で **全 method 共通 (common random numbers)** → paired Wilcoxon の前提充足。
  - equal budget (全 method 同一 n_evals=6000)、fresh-seed honest 再評価 (n_trials=30)、CPU・numpy のみ。
  - robustness: base_seed を 20260530 / 777 / 31337 で確認。

## 5. 「特性化 成功 / 失敗」の定義

- **成功**: advantage(d) が d=0 近傍で ≈0 (or 負) → ある d* を境に正に転じ、d≥d* で strict-gate
  load-bearing が安定継続する **単調な閾値構造**が出る。d* と transition の幅 (sharp/gradual) を報告。
- **部分成功**: 閾値は出るが baseline ごとに d* が違う (例: RR-hillclimb と panmictic-GA で dip 耐性が
  異なる) → 「閾値帯 (band)」として報告。これも valid な特性化。
- **失敗**: §2 反証条件 (a)/(b)/(c) のいずれか。「閾値なし/③は狭い regime のみ」と honest に報告。

## 6. 実 task の honest 配置 (この軸のどこに落ちるか)

最重要の honest 課題: 過去の negative (E-A multitask / Step C memory) はこの欺瞞性軸の **どこ**に
落ちるのか。**厳密な numeric 配置は不可能**であることを先に宣言する (理由は verdict §4):
- 本 knob は behavior=mean の合成 corridor 上の dip 深さ。実 task の fitness は behavior 記述子も
  fitness も別物 (ESN reservoir パラメータ / 多タスク汎化 R²) で、同じ d スケールに射影する厳密な
  写像は無い。
- できるのは **定性的配置**: 実 task は「baseline がほぼ大域に到達 (reach≈1) で MAP-E が baseline に
  勝てない」という観測パターンを示した → これは本 sweep の **d < d* (smooth 側)** の挙動と同型。
  この同型性に基づく定性配置に留め、numeric な d 値は付けない (verdict §4 で明記)。

## 7. 成果物

1. 本ノート `docs/poc/STEP_C_APPLICABILITY_DESIGN.md`。
2. `research/step_c_applicability/exp_knob_sweep.py` (sweep 実行・JSON 出力)。
3. `research/step_c_applicability/exp_knob_sweep_results.json` (結果)。
4. `docs/poc/STEP_C_APPLICABILITY_VERDICT.md` (特性化曲線・閾値・実 task 配置・honest 留保)。

## 参照

- `STEP4_SELECTION_VERDICT.md` (exp4 deceptive / exp5 smooth, ③存在証明と境界)。
- `E_A_VERDICT.md` / `STEP_C_VERDICT.md` (実 proxy / 記憶タスクの honest negative)。
- `research/step4_selection/selection_lab.py` (MAP-Elites + baselines + 比較ハーネス, read-only)。
- `research/ea_multitask/ea_lab.py` (SeedSequence + CRN seed 設計の参照実装)。
- `src/llcore/evolution/honest_eval.py` (strict gate / honest_reevaluate)。
