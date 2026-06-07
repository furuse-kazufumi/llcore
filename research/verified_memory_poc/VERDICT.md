# Phase 2a — trajectory_tube gate PoC VERDICT

作成日: 2026-06-06
設計の正本は別途管理 (本 VERDICT が公開向けの一次成果物)。
規律: honest disclosure 厳守 (`feedback_benchmark_honest_disclosure`)。誇張禁止。
ステータス: **実装完了 + falsifiable 検証完走**。

このドキュメントは「やったこと」「結果」「honest 留保」をまとめる Phase 2a の一次成果物。

---

## 1. 何を実装したか (additive 性の担保)

設計 doc の Case B 主軸ハイブリッドに従い、**進化ループ内 prove-then-reject ゲート**に
`gate_mode="trajectory_tube"` を additive 追加した。

src への変更は設計 doc §4.4 の制約どおり最小限:

| ファイル | 変更 | additive 性 |
|---|---|---|
| `src/llcore/evolution/minimal_ga.py` | `_gate_admits` に `trajectory_tube` **1 分岐** + `evolve()` に `w_bar`/`r_max` kwarg | `gate_mode="none"` は **byte-identical** を維持 |
| それ以外の src | **変更なし** | `tracking_tube` / `eval_step` / `run_sequence` / `verify_lipschitz_contraction` はそのまま呼ぶだけ |

research/ への追加 (一次成果物):
- `research/verified_memory_poc/disturbance_checker.py` — tube 定理の 64-seed cross-check
  (`poc_target_trajectory.py:109` を scalar `run_sequence` 基板に **port**)。
- `research/verified_memory_poc/run_3arm_ab.py` — 3-arm A/B + falsifiable 検証。
- `research/verified_memory_poc/results_3arm.json` — 生結果。

### additive 性の回帰テスト (byte-identical 担保)

- `tests/unit/test_minimal_ga_gate.py::test_gate_none_byte_identical_with_wbar_rmax_kwargs`
  — `gate_mode="none"` は `w_bar`/`r_max` を渡しても旧挙動 byte-identical (best_fitness_curve /
  diversity_curve 完全一致)。
- 既存 `test_gate_none_byte_identical_to_default` / `test_src_evolve_matches_research_gated_evolve`
  は無改変で全 green。
- **全テスト 312 passed** (baseline 294 + 新規 18)。回帰ゼロ。

---

## 2. fail-closed 規律の実装 (設計 doc §2.1 地雷リスト遵守)

新 glue (`_gate_admits` の `trajectory_tube` 分岐) は以下を **全て reject** にする:

- `tracking_tube(...).admits == False` (= `contraction_ok=False` ∨ tube=∞ ∨ r>r_max)。
- `tracking_tube` が **例外**を投げた場合 (`try/except` で握り潰して reject)。
- `w_bar=None` で trajectory_tube を呼ぶ誤用は **fail-loud** (`ValueError`; 黙って admit しない)。

設計 doc §2.1 の fail-OPEN trap (refinement.py:209-218 / invariants.py:392-405) には**触れていない**:
trajectory_tube gate は `tracking_tube` の閉形式判定のみを使い、`refinement` も
z3 不在で None を返す `verify_lipschitz_contraction` も使わないため、当該 trap には非接触。
(`contraction` arm は従来どおり `verify_lipschitz_contraction(...).contraction is True` =
hard True only で fail-closed。)

---

## 3. gate パラメータの選定理由

`w_bar=0.1`, `r_max=0.05`。

設計 doc §3.1 の「contracting 中 tube 半径 p50≈0.031」を出発点に、**scalar gene 空間で
実 sweep** (N=20000, seed=2026, w_bar=0.1) して binding 値を確定した:

| 指標 | 値 |
|---|---|
| closed-form contracting (tube gate L<1) | 14018 / 20000 (70.1 %) |
| Z3 free-t contracting (contraction gate) | 14018 / 20000 (70.1 %) — **scalar では同集合** |
| contracting 中の tube 半径 | min=0.0, p25=0.030, p50=0.063, p75=0.100, max=636 |
| `r_max=0.05` で admit | 5691 / 14018 (40.6 % of contracting) |
| `r_max=0.05` で reject (contracting なのに) | **8327 件** |

→ `r_max=0.05` は **strongly binding** (admit も十分 / reject も大量) で、
never-bind 退化 (反証条件 b) でも degenerate でもない。よって (P2) が成立する値として採用。

### 重要な実コードとの齟齬 (設計 doc に対する注記)

設計 doc §4.3 訂正2 は「free-t L (Z3 gate) ≥ achievable-t L (tube gate) なので 2 gate は
non-comparable で、box mismatch を混入させるな」と警告していた。
**実コードで確認した結果、scalar gene では両 gate の L 定義が一致する**:

- `tracking_tube` の `contraction_ok` = `state_lipschitz_inf` の achievable-t box 閉形式 L < 1。
- `verify_lipschitz_contraction` の `L_upper_bound` = `_lipschitz_upper_bound(decay, gate_str)`
  = `max(|decay|, |decay+(1−decay)·gate_str|)` (free-t 端点)。
- scalar gene (`W=[[gate_str]]`, V=[[mix]]) では `_infnorm_sup` の端点列挙が
  `max(|decay|, |decay+(1−decay)·t·gate_str|)` を t∈{t_min,1} で取り、結果として
  **両者とも同じ contraction 判定 (14018 件で完全一致, witness で L=0.5 一致を確認)** になる。

含意: 設計 doc が懸念した box mismatch は **scalar gene では発生しない**。
よって (P2) の「真部分集合」主張は、L を揃えるための追加工作なしに、**そのまま clean に成立する**
(reject は L 差ではなく tube 半径 = G·w̄ 由来の差)。これは doc の前提より**有利**な発見。
ただし coupled gene に拡張する場合は doc §4.3 訂正2 の警告が再び効くため、その時は同一 L 定義で
揃える必要がある (本 PoC のスコープ外)。

---

## 4. P1 / P2 / P3 の結果

| 命題 | 内容 | 判定 | 根拠 |
|---|---|---|---|
| **P1** | trajectory_tube が admit した全カーネルで実測 limsup‖e_t‖∞ ≤ certified tube r (N≥64 seed, 0 違反) | **PASS** | 3-arm の最終集団から admit gene **180 件**を 64-seed 外乱 driver で cross-check → tube 違反 **0 件** |
| **P2** | contraction=True だが r>r_max のカーネルを reject (n_rejections>0) = trajectory gate は contraction gate の真部分集合に絞る | **PASS** | trajectory_tube arm の tube reject 総数 **640 件**。構成的 witness gene `[0.5,1.0,0.0]` で same_L=True (closed-form L = Z3 L_upper_bound = 0.5), tube=0.1 > r_max=0.05, witness_holds=True |
| **P3** | named-slot write (= eval_step) が数値的に既存 eval_step と恒等 (bridge が虚構でない) | **PASS** | `test_verified_memory_poc.py::test_p3_named_slot_write_identical_to_eval_step` (3 gene × 20 (s,x) で bit-for-bit 一致) + `test_p3_bridge_not_a_fork_of_run_sequence` (逐次適用 == run_sequence) |

---

## 5. 3-arm 比較表

mean test-fit (held-out, TEST_N_TRIALS=20, 独立 RNG) over 3 seeds {1000,1001,1002}:

| task (delay) | none | contraction | trajectory_tube | tube rej (con / tt) |
|---|---|---|---|---|
| copy_d0 (delay=0) | 0.1681 | 0.1695 | **0.1731** | 63 / 202 |
| copy_d4 (delay=4) | 0.2016 | 0.1991 | **0.2040** | 80 / 186 |
| copy_d8 (delay=8) | 0.1832 | 0.1829 | **0.2003** | 100 / 252 |

trajectory_tube が全 delay で僅差で最良。だが下記 (c) の通り **fitness 優位は統計的に主張できない**。

### per-seed paired delta (trajectory_tube − contraction)

| task | seed1000 | seed1001 | seed1002 | mean |
|---|---|---|---|---|
| copy_d0 | +0.0122 | +0.0016 | −0.0031 | +0.0036 |
| copy_d4 | +0.0012 | +0.0112 | +0.0023 | +0.0049 |
| copy_d8 | **−0.0171** | **+0.0687** | +0.0006 | +0.0174 |

→ d8 の平均優位 (+0.017) は **seed 1001 単独 (+0.069) に駆動**され、seed 1000 では負 (−0.017)。
符号不一致 + n=3 = **統計的有意性なし**。

---

## 6. 反証条件 (a)(b)(c) の判定

| 反証条件 | 内容 | 判定 |
|---|---|---|
| **(a)** | admit したカーネルの実測軌道誤差 > certified tube r → 定理/実装の破綻 | **not triggered** (P1 violations = 0 / 180) |
| **(b)** | trajectory_tube と contraction gate が同一集合を admit → r_max 緩すぎ degenerate (re-skin) | **not triggered** (tube rejects = 640 > 0; r_max strongly binding) |
| **(c)** | delay>0 memory タスクで gated 集団の best fitness が single-step L<1 gate と統計的に区別不能 → trajectory lift が memory 保持に何も買っていない | **INCONCLUSIVE (honest red flag)** — n=3 seed では区別不能。下記参照 |

### (c) の honest 評価 (最重要)

**trajectory_tube の fitness 優位は、本 PoC の規模 (n=3 seed) では統計的に主張できない。**
per-seed delta の符号が一致せず、d8 の見かけの優位は 1 seed の outlier に依存している。

これは設計 doc §5.2 の honest 留保および `feedback_benchmark_honest_disclosure` が予期した通りの
結果である。**Phase 2a が demonstrable に確立したのは P1 (soundness) と P2 (strict additive
discriminating power) であって、(c) の "trajectory lift が memory horizon に効く" ではない。**
(c) を主張するには seed 数を大幅に増やす (n≥20, exp_b_runner.py 規模) 必要があり、それは
次フェーズの課題。現時点で fitness benefit を謳うことは over-claim にあたるため**しない**。

### 「綺麗すぎる」結果のチェック (設計 doc §4.5-7)

P1=0 違反 / P2=640 reject は「綺麗」に見えるが、per-gene verdict (`results_3arm.json` の
`p1_cross_check` 配列 + tube reject 数) を dump して確認した結果:
- **r_max は never-bind に退化していない** (tube reject が contraction reject を大きく上回る:
  202>63, 186>80, 252>100)。trajectory gate は contraction gate より厳しく絞っている。
- P1=0 違反は vacuous ではない (admit gene 180 件を実測 cross-check した上での 0 違反;
  certified tube に対し実測誤差は steady-state で確実に内側)。
- 非契約 gene は tube=∞ で vacuous holds にしない設計 (`contraction_ok=False → tube_holds=False`)。

よって P1/P2 の「綺麗さ」は退化由来ではなく**本物**。一方で (c) の見かけの優位は
outlier 由来であり、**そちらは綺麗に見えても信用しない**。

---

## 7. honest 留保 (設計 doc §5.1 / §5.2 を本 PoC に即して再掲)

言ってよいこと:
- **soundness**: tube 不等式 limsup‖e_t‖∞ ≤ G·w̄/(1−L) は Banach + Lipschitz 合成の本物定理
  (SSGM の Proof Sketch と差がつく軸)。参照が系自身の解なので ρ_feas=0。
- **動く実装**: gate in loop (640 件の実 reject) vs SSGM の実装ゼロ。
- **param → state 軌道の bridge** (P3 で eval_step 恒等を確認)。
- **strict additive discriminating power** (P2): trajectory_tube は contraction gate の真部分集合に絞る。

言ってはいけないこと (本 PoC で守った 4 点 + 1):
- (a) memory write gate / verified memory evolution の**応用アイデアの新規性** (SSGM が 2026-03-12 先取り)。
- (b) tube gate が **「Z3-exact contraction」** — 実体は achievable-t box 上の閉形式 numpy 比較。
  本 PoC のコメント・docstring でも「閉形式 (Banach 系) tube」と明記し「Z3-exact tube」とは書いていない。
- (c) trajectory_tube が無条件に contraction gate より strictly stronger — scalar では同 L 定義で
  真部分集合だが、coupled では box mismatch で non-comparable になりうる。
- (d) **「external memory bank」** — bank/retrieval 不在。本 PoC は「bank/external/cross-slot/retrieval」
  の語を一切使わず、bridge anchor は "named-slot write" 解釈に留めた。
- (e) **fitness benefit on memory tasks** — n=3 では統計的に区別不能 (§6 (c))。

残る本質的弱点 (honest):
- **bridge は半分**: tube の L も G もカーネル param の閉形式端点量から計算され、実現された軌道に対する
  新規 Z3 query ではない。64-seed 外乱 run は**定理の cross-check** であって gate soundness の一部ではない。
  防衛的に正確な framing = 「param-contraction → param 由来の trajectory-tube 境界」。
- **CopyTask の正直注記**: `CopyTask` の fitness は "fixed-readout probe-based fitness" であり
  gene-pure fitness ではない。memory probe としては妥当だが "real memory horizon" を過剰主張しない。
- **w̄ は離散入力では弱い**: CopyTask の入力は離散 sample であり、連続外乱 bound w̄ の worst-case は
  実タスクの摂動と完全には一致しない (cross-check は uniform d∈[−w̄,w̄] で worst をなぞる近似)。

---

## 8. 再現コマンド

```powershell
# 全テスト (additive 回帰含む)
py -3.11 -m pytest tests/ -v

# trajectory_tube gate / 外乱チェッカ / bridge anchor のみ
py -3.11 -m pytest tests/unit/test_minimal_ga_gate.py tests/unit/test_verified_memory_poc.py -v

# 3-arm A/B + falsifiable 検証 (CPU 完結, 約 4 分)
py -3.11 research/verified_memory_poc/run_3arm_ab.py
```

---

## 9. 結論 (一行)

**Phase 2a の唯一の目的 (param → memory-update ギャップを最小コードで over-claim せず埋める) は達成。**
P1 (soundness) と P2 (strict additive discriminating power) は demonstrable に確立。
(c) (trajectory lift が memory horizon に効く) は n=3 では統計的に区別不能 = honest red flag として
明示し、fitness benefit は主張しない。additive 性 (gate_mode="none" byte-identical) は回帰テストで担保。

---

## 10. (c) の決着 — n=20 事前登録 run (2026-06-07 追補)

§6 (c) の INCONCLUSIVE を `run_c_decision.py` で決着させた。**結果取得前に判定基準を commit**
(4722095: 主仮説 = copy_d8 confirmatory / sign-flip permutation 両側 / α=0.05 / seed 2000-2019 /
d0・d4 は exploratory)。生結果 = `results_c_decision.json` (wall 266s)。

### 結果: **H1 supported — (c) は棄却されない**

| task | mean Δ (tube−contraction) | p (sign-flip) | 符号 | 役割 |
|---|---|---|---|---|
| copy_d0 | −0.0002 | 0.957 | +7/−11 | exploratory |
| copy_d4 | +0.0098 | 0.104 | +10/−10 | exploratory |
| copy_d8 | **+0.0152** | **0.0056** | **+16/−4** | **confirmatory** |

delay 0 → 4 → 8 で単調に効果が立ち上がる dose-response パターン =
「trajectory tube bound は memory horizon が長いほど効く」と整合。

### 内訳を疑った結果 (honest discipline)

- **outlier 駆動ではない**: d8 で top1 (+0.0905) を除いた mean +0.0112 / trimmed mean (両端 2 除外)
  +0.0107 / median +0.0054 — いずれも正。pilot n=3 の「seed1001 単独駆動」とは質的に異なる。
- **分布は右裾** (mean 0.0152 > median 0.0054): 効果は「多数 seed で小さい正 + 数 seed で大きい正」。
  一様な +0.015 ではない、と書く。
- **GA は壊れていない**: d8 で tube arm 2224 rejects vs contraction 741 (3 倍絞る) なのに
  fallback 0 (全世代 admit 子を見つけられた)。絞った上で fitness が高い =
  navigability (検問が良い領域へ誘導) の追加証拠。

### 言ってよいこと (更新)

- §7 の禁止事項 (e) を更新: **delay=8 では事前登録した confirmatory 検定で fitness 優位が
  統計的に検出された** (n=20, p=0.0056)。ただし効果量は小さく (mean Δ≈+0.015)、
  probe-based fitness / scalar gene / 小規模 GA (pop=20×gen=20) のスコープ内に限る。
- d0/d4 は依然主張しない (exploratory, n.s.)。「memory を要しないタスクでは差が消える」は
  むしろ仕様どおり (tube は memory 保持の保証であって万能の性能向上ではない)。

---

## 11. 第二独立軸 (外乱振幅 w̄) の決着 — 用量反応は horizon 固有 (2026-06-07 追補)

(c) の用量反応 (delay 0→4→8 で効果が立つ) が「一般的な不変量負荷の法則」か「memory horizon
固有」かを、**機構的に独立な第二軸**で事前登録検証した。`run_wbar_decision.py` (事前登録 commit
ebdf703 = 結果取得前)。生結果 = `results_wbar_decision.json` (wall 316s)。

### 動機と軸選定 (seq_len の機構的棄却)

最初に第二軸候補とした **`seq_len` (入力列長) は無効**である: CopyTask の
`target_idx = seq_len-1-delay` ゆえ target 入力→readout の保持ステップ数は**常に delay (=8) 固定**。
seq_len を増やしても増えるのは target *前*のステップだけで、contraction (L<1) 下で古い入力は L^k
で減衰し数ステップで飽和 = **保持 horizon を増やさない**。よって seq_len は負荷軸として使えない。

正しい第二軸 = **task 外乱振幅 w̄_task** (tube 公式 r=G·w̄/(1−L) の**分子** G·w̄ 側; delay は**分母**
1/(1−L) 側を動かすので機構独立)。`DisturbedCopyTask` は入力に d~U[−w̄_task,w̄_task] を注入し
**クリーンな** target を復元させる (保持には外乱除去=tight contraction=小 tube が要る)。
**交絡回避**: gate の w̄_gate=0.1/r_max=0.05 は固定 (admit 率不変=binding 交絡なし)、task 外乱のみ動かす。

### 結果: **NEGATIVE — w̄ 軸では用量反応が立たない**

| w̄_task | mean Δ (tube−contraction) | p (sign-flip) | 符号 | median | 役割 |
|---|---|---|---|---|---|
| 0.00 | +0.0116 | 0.093 | +13/−7 | +0.0091 | exploratory (= d8 再現) |
| 0.05 | +0.0033 | 0.369 | +10/−10 | +0.0001 | exploratory |
| 0.10 | +0.0070 | 0.082 | +15/−5 | +0.0065 | exploratory |
| **0.20** | **+0.0026** | **0.594** | +11/−9 | +0.0009 | **confirmatory** |

**monotone non-decreasing = False**。confirmatory (w̄=0.20) で p=0.594 → 反証条件 **F1 発火**:
外乱軸では用量反応が立たない。gate は全 w̄ で binding (tube reject ~2500 vs contraction ~800,
fallback 0) = 退化由来でない健全な NEGATIVE (F5/F8 不発)。

### 解釈: 用量反応は分母 (保持余裕) 駆動、分子 (外乱) 駆動でない — **理論と整合**

この NEGATIVE は ESN 記憶容量との同型理論 (tube r=G·w̄/(1−L) ↔ memory capacity ~1/(1−ρ)) と
**整合する**。記憶容量の発散は**分母** (1−L) 現象であり、効果が住むのは保持余裕=horizon (delay) 軸。
**分子** G·w̄ (外乱) を動かしても gate 価値は増えない — この memory タスクでは外乱除去 (tight
contraction = 小 1/(1−L)) と保持 (L を小さくしすぎない) が**トレードオフ** (memory-nonlinearity
trade-off) するため。→ 用量反応の正しい言明は「**verified gate の価値は保持余裕の負荷 (= memory
horizon) に比例する**」であって「一般的な不変量負荷に比例する」ではない。**法則を分母軸に narrow**。

### 副次: d8 効果の独立再現 + pooled 強化 (honest 内訳)

w̄_task=0.00 セル (delay=8/seq_len=32/外乱なし) は **§10 の (c) d8 の fresh-seed 独立再現**
(seeds 3000-3019 vs 原 2000-2019)。

| run | n | mean Δ | median | 符号 | p |
|---|---|---|---|---|---|
| §10 d8 (s2000-19) | 20 | +0.0152 | +0.0054 | +16/−4 | 0.0056 |
| 再現 w0 (s3000-19) | 20 | +0.0116 | +0.0091 | +13/−7 | **0.093** |
| **pooled** | **40** | **+0.0134** | +0.0086 | +29/−11 | **0.0021** |

(pooled: trimmed10%=+0.0108 / drop-max|Δ| p=0.0045 = outlier 非駆動)。

honest 評価: **効果の方向は再現** (2/2 run 正、pooled +29/−11)。**pooled n=40 では robust に有意**
(p=0.0021)。ただし **§10 の single-run p=0.0056 は seed 有利端**であり、fresh n=20 単独では
p=0.093 (borderline)。→ d8 効果は real かつ pooled で強化されるが、「single n=20 で p=0.0056」を
唯一の証拠として強く出すのは over-claim 気味。**pooled n=40 (p=0.0021) + single-run の seed
感受性を併記する**のが正しい開示 (feedback_benchmark_honest_disclosure: 異常に低い p は内訳を疑う)。

### §9/論文への含意 (要反映)

論文 §9 の (c) 記述を更新する: (1) 証拠を pooled n=40 (p=0.0021, robust) に格上げしつつ
single-run の seed 感受性を honest caveat として併記。(2) **dose-response は horizon (delay/分母)
軸固有で、外乱 (w̄/分子) 軸では立たない (事前登録 NEGATIVE, p=0.594)** を明記 = 法則の適用範囲を
正しく narrow。(3) ESN-capacity 同型 (1/(1−ρ)) は分母現象だから horizon 軸で立ち外乱軸で立たない
ことを**予測する** = 理論と経験の整合を 1 段強める。
