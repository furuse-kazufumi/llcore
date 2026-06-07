# llcore 進化検出マシナリ — 統計的検出力 自己監査 VERDICT

> 2026-06-01。隔離ディレクトリ `research/statistical_power_audit/`。**src 無改変・git 非実行**。
> 命題 `H0_suppress` を 4 方向 (A 校正 / B 検出力逆算 / C ablation / D Type I guard) で falsifiable に検証。
> 生数値: `results_power_audit.json` (統合) + `calibrate_known_positive_results.json` /
> `repower_real_negatives_results.json` / `ablate_suppression_knobs_results.json` /
> `type1_guard_sweep_results.json`。3 独立 VERIFY lens (type1_tradeoff / power_calc / calibration_circularity)
> の指摘を §9 に honest に反映済み。
>
> **これは「我々の統計が厳しすぎて進化を見落としていないか」の自己監査である。** 結論を緩める方向の発見は
> 必ず「緩和が偽陽性をどれだけ増やすか (Type I 代償)」と対にして読む。「緩めれば進化が見える」だけでは
> 不十分という規律 (honest disclosure) を全節で守る。

---

## 1. 命題と方法

### falsifiable 命題 `H0_suppress`

> 「llcore の進化検出マシナリ — K1 fresh-seed 再評価 + K2/exp_knob_sweep の strict gate 連言
> {n_seeds≥15 ∧ 片側 Wilcoxon p<0.05 ∧ |paired_sign_delta|≥0.147 ∧ diff>0} + K3 equal-budget/global-best/HONEST_N
> + K4 ridge clip=True — は、効果量が中程度 (Cliff δ 相当 0.15–0.40) の真の進化優位を、現行の n と閾値の下で
> 80% 検出力に満たない確率で取り逃がす (= Type II 偏向)。」

### 判定基準 (機械評価、AND 連言)

- **suppression 認定 (S1∧S2∧S3∧S4)**: S1 校正で d∈[0.16, d_fn(15)] の中効果取り逃がし域が存在 / S2 実 negative が
  underpowered かつ n80≥30 / S3 ablation で verdict 反転 / **S4 その反転が null FPR をほぼ上げない sweet spot を持つ**。
  S3 を満たすが S4 を満たさない (偽陽性も同率で上がる) なら「抑制でなく厳しさ⇄緩さのトレードオフ点」とし **suppression 非認定**。
- **健全認定 (R1∧R2∧R3)**: R1 既知真陽性を n≥15 で ≥80% 検出 / R2 diff<0 対照が真に効果無し / R3 緩和は信号でなくノイズを拾う。

### 方法

既存 lab を **read-only import 再利用** (改造禁止): `exp_knob_sweep.make_corridor_eval(d)/run_methods_crn/strict_gate`、
`honest_eval._paired_p/_paired_sign_delta` (exact `scipy.wilcoxon`)、`ea_lab._map_elites_core`、`ridge_readout.ridge_fitness`。
Python = `py -3.11`、numpy+scipy、CPU 完結。power シミュレーションは正規近似 (tie+連続性補正) の vectorized 版で実施し
exact scipy で G3 校正 (近似誤差 n=15 で max|Δp|=0.0055)。

### 破綻ゲート (全通過)

| gate | 内容 | 結果 |
|---|---|---|
| G1 CPU 完走 | 4 script 各 <900s, exit 0 | calib **575s** / repower **332s** / ablate **204s** / type1 **294s** = 全 <900s |
| G2 再現性 | 同 seed 2 回で per-seed 配列完全一致 | d=0.16 n=15 で `np.array_equal=True`、d*_strict は 3 base_seed {20260530,777,31337} で 0.16 に一致 |
| G3 power 計算器妥当 | null≤0.06 / 大効果≥0.95 / C-gen4b 模擬一致 | null=0.023(boot)/0.050(param), big=1.00, cg4b 模擬 n15=0.582 / n30=0.834 = **VALID** |
| G4 src 不変 + git なし | src/ 差分ゼロ, 書込は research 配下のみ, git 呼び出しゼロ | 全 `_meta.src_unchanged=true`、`src_changed_files=[]` |

---

## 2. 偽陰性境界地図 (効果量 d × 標本数 n) — (A) 校正

`calibrate_known_positive_results.json`。既知真陽性 corridor (`make_corridor_eval(d)` = ramp に深さ d の dip を彫った
欺瞞回廊。③ behavioral niching が脱出機構) に full strict gate をかけ、「③ が真に勝つのに no-effect 判定になる最小 d」
(= false-negative onset `d_fn(n)`) を n ごとに測定。3 base_seed で robust。

| n_seeds | d_fn (load-bearing onset) | d*=0.16 を検出? | d=0.16 の textbook Cliff δ |
|---|---|---|---|
| **10** | **到達せず (全 d で盲点)** | **No** | +0.44 |
| **15 (実験標準)** | 0.15 | **Yes** | +1.00 |
| 30 | 0.15 | Yes | +0.59 |
| 60 | 0.14 | Yes | +1.00 |

**読み:**

- **n=15/30/60 では gate は健全** — d_fn ≤ 0.16 で既知真陽性 d*=0.16 を取り逃がさない (むしろ d=0.15 まで検出)。**S1 は n=15 で不成立 / R1 成立。**
- **n≤10 は構造的盲点** — d=0.16/0.18/0.20 が Cliff δ=+0.44〜+1.0・p<0.001 でも **n≥15 床で全て no-effect 判定**。
  → **step6 exp7 は n=8/6 で実施 = まさにこの盲点域。強効果すら gate が見えない。S1 は n≤10 で成立。**
- 律速は一貫して **片側 Wilcoxon p<0.05 + 小 n**。容疑だった min_effect=0.147 床ではない (観測 n 域では)。
- **scope 限定 (VERIFY: calibration_circularity, refuted=true/medium)**: この境界は ramp-with-dip family **内部**の境界であり、
  behavior=mean (③-favoring 軸) を彫った合成 landscape 上で測っている。`d_fn(n)` の magnitude は実 ESN/text proxy substrate へ
  そのまま転送しない。ただし「③ load-bearing」の判定は監査対象 gate ではなく **独立 oracle (per-method reach rate, 大域峰到達率)**
  で確認しており (d=0.14 n=30 で reach: MAP-E=1.00 vs RR=0.33/GA=0.57/rnd=0.10、d=0 smooth では全 method reach=1.00 で MAP-E 僅か劣位)、
  境界は gate の自己充足トートロジーではない。詳細 §9。

---

## 3. 実 negative の検出力逆算表 — (B) repower

`repower_real_negatives_results.json`。実 per-seed paired delta を `exp_ea3_results.json` / `exp_c2c3_results.json` から
**復元** (仮定値でない)。ブートストラップ B=5000。

| case | diff | Cohen dz | psd | textbook Cliff δ | observed power (n=15) | n80 (param / boot) | underpowered | 律速 (@n=15) |
|---|---|---|---|---|---|---|---|---|
| **C-gen4b** MAP-E vs random | +0.063 | +0.35 | +0.20 | +0.396 | **0.31** | ~64 / **~255** | **Yes** | p_only |
| **flip_flop** MAP-E vs random | +0.004 | +0.31 | +0.33 | +0.271 | **0.27** | ~89 / ~82 | **Yes** | p_only |
| flip_flop vs RR-hillclimb | +0.004 | +0.27 | +0.07 | +0.280 | 0.15 | ~169 / 未到達 | Yes | p_only |
| C-gen4a vs panmictic (diff<0 対照) | −0.019 | −0.13 | −0.07 | +0.004 | **0.03** | 未到達 | **No (健全)** | p_only |
| delayed_parity vs random (基質床 R²≈0) | −0.001 | −0.20 | −0.20 | −0.156 | 0.004 | 未到達 | No (健全) | p_only |
| C-gen3 vs randselect (PASS 対照) | +0.126 | +0.68 | +0.60 | +0.618 | 0.74 | ~16 / ~18 | (borderline) | p_only |

**読み:**

- diff>0 符号一貫の **C-gen4b (power 0.31) と flip_flop (power 0.27) は underpowered** — 80% power に現行 n=15 の
  **4–6 倍 (n~64–89, 保守側 bootstrap で C-gen4b は ~255)** を要す。**S2 成立。**
  → これらの honest-negative は「③不在の証拠」ではなく **inconclusive**。STEP_C_VERDICT 自身の自認 (underpowered=null 受容でない) と一致。
- 対照 **C-gen4a (diff<0) は power 0.03 で真に効果無し = 健全 negative (R2 成立)。** delayed_parity も同様 (基質床)。
- **PASS 対照 C-gen3 ですら n80~16–18** = 現行 n=15 は「勝つ case でギリギリ」。マシナリ全体が小 n で運用されている。

---

## 4. suppression ablation 表 — (C) + Type I コスト (D)

`ablate_suppression_knobs_results.json` + `type1_guard_sweep_results.json`。OFAT (他 knob ON 固定)。
**各 toggle は対応する null FPR と対で読む (honest disclosure の核)。**

| knob / level | verdict 反転? | borderline TPR | shuffle-null FPR | net gain | 解釈 |
|---|---|---|---|---|---|
| baseline (全 ON) | — | 0.20 | 0.047 (≈名目 α) | +0.153 | gate は適切に校正・むしろ保守的 |
| **K2_alpha 0.05→0.20** | **Yes** (C-gen4b, flip_flop) | 0.20 (不変) | **0.172 (>2×α)** | +0.028 | **純 Type I コスト。p で落ちる underpowered case を救うが null FPR も等価に増。sweet spot でない** |
| K2_alpha 0.05→0.10 | No | 0.20 | 0.090 | +0.110 | FPR が α の倍に。TPR 不変 |
| K2_min_effect 0.147→0.05 | No | 0.20 | 0.046 | +0.154 | **binding 制約でない** (緩めても TPR/FPR 不動)。sweet_spot 認定だが baseline と実質同等 |
| K2_min_effect →0.0 | No | 0.20 | 0.049 | +0.151 | 同上 |
| K2_min_seeds 15→{10,5} | No | 0.20 | 0.048–0.053 | +0.147–0.152 | 実 negative は n=15 固定で無反転 |
| **K1 fresh-seed 再評価 OFF** | No (corridor d=0.13) | — | pure-null 増は限定的 | — | この null/budget では FPR 増を強く示せず (§9 caveat 2) |
| **K3 archive-max vs global-best** | No (両読み出しとも③勝利) | — | — | — | **archive-max は diff を ~50x 水増し** (archive diff=+0.081/δ=+0.60 vs global diff=+0.0016/δ=+0.40)。Codex F2 の global-best 化は③過大評価を正しく除去 = suppression でなく **Type I 防止** |
| **K4 ridge clip True→False** | **Yes (ridge 系)** | — | (null-ridge FPR は未測, §9) | — | clip が ridge fitness の spread を最大 **13x 平坦化** (addition: clip_true_std=0.028 vs clip_false_std=0.380, floored 92.5%)。floored gene の raw R² は spread 0.37・符号一貫 (全負) = **clip が真の landscape 構造を隠蔽** = **唯一の能動的 suppression 機序** |

**反転をもたらす唯一の gate 緩和 K2_alpha=0.20 は純 Type I コスト** (FPR 0.047→0.172、TPR 0.20 のまま不変)。
**S4 不成立** = 反転は sweet spot を持たない。min_effect/min_seeds 緩和は TPR を 1 件も増やさない。

---

## 5. 最終判定

> ## **(c) 部分的に underpowered。** 統計マシナリは「一律に進化を抑えている」のではない (`suppress_all_S1∧S2∧S3∧S4 = False`)。

`criteria_evaluation` (results_power_audit.json):

```
S1 Type II @ n=15  = False   (実験標準 n では gate 健全、既知真陽性 d*=0.16 を検出)
S1 Type II @ n≤10  = True    (小 n では強効果すら n 床で盲点)
S2 underpowered     = True   (実 negative の一部 C-gen4b/flip_flop は inconclusive)
S3 flip exists      = True   (K2_alpha=0.20 で 2 件反転)
S4 flip is sweet    = False  (反転緩和は Type I を等価に増やす → 抑制でなくトレードオフ点)
suppress(S1∧S2∧S3∧S4) = False
R1 detect known+ @n15 = True / R2 diff<0 control healthy = True / R3 relaxation inflates FPR = True
```

**(a) 偏って進化を抑えている — 否 (一律には)。** 実験標準 n=15 では校正は健全で、既知真陽性 (Cliff δ=+1.0) を取り逃がさない。
gate 閾値の連言 (p / min_effect / n) 自体に Type II 偏向の証拠はなく、緩めると Type I が代償として増える。

**(b) 完全に健全＝進化は本当に無い — 否。** 以下 3 点で部分的 underpower が実在する:

1. **実 negative の C-gen4b / flip_flop は observed power 0.27–0.31 で underpowered** = 「③不在の証拠」でなく inconclusive。
2. **n≤10 域 (step6 exp7 の n=8/6) は強効果すら検出不能な構造的盲点**。
3. **ridge clip=True は ridge landscape を最大 13x 平坦化し floored gene (92.5%) の符号一貫な raw 構造を隠蔽** (K4 = 唯一の能動的 suppression 機序)。

**∴ 結論は (c):** マシナリは一律抑制ではないが、**特定の運用条件 (小 n, ridge clip) で Type II 側に偏る**。
律速は (観測 n 域では) min_effect 床でなく **片側 Wilcoxon p<0.05 + 小 n**。
**生き残った refutation (§9 の power_calc/medium) により、n80 域では C-gen4b で psd≈床が binding に転じる** —
これは (c) を **むしろ補強**する方向の発見であり (中効果 psd∈[0.15,0.20] 帯を psd 床が構造的に律速)、§9 で訂正開示する。

---

## 6. ③ 研究 / E-A / flip_flop 結論への含意

| 既存結論 | 本監査の含意 | アクション |
|---|---|---|
| **E-A C-gen4b** (MAP-E vs random, ③無効と読まれうる) | underpowered (power 0.31)。inconclusive | **n~64 (保守 ~255) で再測**。再測で p≥0.05 なら初めて③不在と言える |
| **E-A C-gen4a** (MAP-E vs panmictic, diff<0) | power 0.03、真に効果無し = **健全 negative** | 再測不要。③不在の正当な証拠 |
| **flip_flop** (δ=+0.33, p=0.15) | underpowered (power 0.27)。STEP_C の「inconclusive」自認は正しい | **n~82–89 で再測** |
| **delayed_parity** (基質床 R²≈0) | power 0.004、真に null (基質が信号を持たない) | 再測不要。基質側の問題で③の問題でない |
| **E-A C-gen3** (MAP-E vs randselect, PASS) | power 0.74 で n80~16–18 = **n=15 では borderline PASS** | 結論は維持されるが marginal。確証に n≥20 推奨 |
| **step6 exp7** (実 ESN proxy, n=8/6, ③ negative) | n≤10 盲点域 = 強効果すら検出不能 | **n≥15 で再測必須**。現 n では③不在を主張できない |

**③ (behavioral niching/selection) は欺瞞回廊 (corridor) で堅牢に load-bearing** (d*=0.16 で 3 baseline 全勝、Cliff δ=+1.0)。
③不要に見える既存 negative の多くは **(i) landscape が滑らか (真に③不要)** か **(ii) underpowered (inconclusive)** の
いずれかで、両者は再測で分離できる。**現時点で「③不在」と確定できるのは diff<0 符号の C-gen4a / delayed_parity のみ。**

---

## 7. 較正済の推奨閾値 (Type I を保ちつつ検出力を上げる)

**gate 閾値は緩めない。検出力不足は閾値でなく n で解く** (power は n の単調増加関数、閾値緩和は Type I を等価に増やす)。

| 項目 | 現行 | 推奨 | 根拠 |
|---|---|---|---|
| 片側 Wilcoxon α | 0.05 | **0.05 維持** | α↑ は null FPR を等価に膨張 (0.05→0.20 で FPR 0.047→0.172)、TPR 不変。sweet spot なし |
| min_effect (paired_sign_delta 床) | 0.147 | **0.147 維持** (中効果帯は n で救う) | 観測 n 域では binding でない。**ただし n80 域では psd≈床の case で binding に転じる (§9)** — 床自体が中効果 psd∈[0.15,0.20] 検出を構造的に律速する点に留意 |
| min_seeds (中効果検出) | 15 | **≥30** (中効果 δ∈[0.15,0.40] を狙うとき) | n=15 は PASS case でギリギリ (C-gen3 n80~17)、中効果は n80~64–89 |
| min_seeds (強効果のみ) | 15 | **≥15 厳守 (10 不可)** | n≤10 では Cliff δ=+1.0 すら盲点 (A 校正) |
| ridge fitness | clip=True (選択圧) | **選択圧は clip=True 維持、③判定/診断時は clip=False の raw R² で spread・符号を併報** | clip は floor 域の構造を隠す (K4) が、選択圧としての [0,1] 制約は妥当 |
| HONEST_N (fresh-seed 再評価試行) | 30 | **30 維持 (K1 を外さない)** | K1 OFF は elitism 持越しで偽 best を水増し (+0.29 実測, EVOLUTION_SOUNDNESS_AUDIT)。fresh-seed 再評価は外してはならない |

**運用ルール:** 中効果 (Cliff δ 0.15–0.40) を狙う実験は **設計時に n≥30–90 を確保** (effect size から事前 power 計算)。
強効果スクリーニングのみなら n≥15。**n<15 の実験 (step6) は結論保留。**

---

## 8. ユーザー懸念「統計的に進化を抑えているのではないか」への直接回答

**結論: 「一律に抑えている」わけではないが、「特定条件で取りこぼしている」のは事実です。** 内訳:

1. **gate の閾値設計 (p<0.05 ∧ |psd|≥0.147 ∧ n≥15 ∧ diff>0) 自体は健全。** 実験標準 n=15 では既知の真の進化 (欺瞞回廊で
   ③ が Cliff δ=+1.0 で全勝する case) を正しく検出し、偽陰性に偏っていません。閾値を緩めると進化が「見える」ように
   なりますが、その唯一の道 (α を上げる) は偽陽性を等価に増やすだけで、真の信号は 1 件も増えませんでした。
   **つまり閾値は「厳しすぎる」のではなく、Type I と Type II のトレードオフの妥当な点に居ます。**

2. **ただし「進化が無い」と読まれていた既存結果のいくつかは、実は『判定不能 (underpowered)』でした。**
   E-A C-gen4b (検出力 31%) と flip_flop (27%) は、効果が本物だとしても現行の標本数 (15) では 27–31% の確率でしか
   検出できません。これらは「③不在の証拠」ではなく、**標本を 4–6 倍 (60–90) に増やして測り直すべき宙づりの結果**です。

3. **真の抑制機序が 1 つだけ見つかりました: ridge fitness の `clip=True`。** 性能を [0,1] に丸める処理が、ある landscape では
   遺伝子の 92.5% を 0 に潰し、その下に隠れた符号一貫な構造 (spread 0.37) を完全に見えなくしていました。選択圧としては
   妥当ですが、**「③ が要るか」を診断するときは clip=False で生の値を見る**べきです。

4. **小標本 (n≤10) は強い進化すら見えない盲点です。** step6 exp7 は n=8/6 で実施されており、ここでは Cliff δ=+1.0 の
   強効果すら gate を通りません。**この実験での③ negative 結論は撤回し n≥15 で再測すべき**です。

**要するに:** あなたの直感は部分的に正しい。統計は「進化全般を握りつぶしている」のではないが、**(小 n の実験 + ridge clip の
2 条件下で) 本当はあるかもしれない中程度の進化を取りこぼしている**。直し方は「閾値を緩める」ではなく「標本数を増やす +
診断時は clip を外す」です。緩めると偽陽性が増えるだけで、進化研究の信頼性をむしろ損ないます。

---

## 9. honest 留保 (surviving refutation を含む)

3 独立 VERIFY lens のうち 2 件が medium 重大度で生き残った。**いずれも (c) の結論を覆さないが、限定・訂正する。**

### 9.1 [power_calc, refuted=true, medium] 律速条件の帰属が n80 域で逆転 — **headline 訂正**

- 検出力計算「器」(片側方向・効果量注入・正規近似) は exact scipy で独立再現され **正しい** (max|Δp|=0.0055、underpowered 判定堅牢)。
- **しかし** RUN summary の headline「律速は全 case で片側 Wilcoxon p<0.05 + 小 n であり、容疑の min_effect=0.147 床ではない」は
  **n=15 でのみ算出した `limiting_condition` を大域に外挿した誤帰属**。
- **訂正**: C-gen4b の母集団 psd=0.20 は床 0.147 のすぐ上。bootstrap で n を増やすと p<0.05 は飽和する一方、|psd|≥0.147 は
  頭打ちになり、**bootstrap n80~255 近傍の binding 条件は psd-floor であって p-value ではない**。これが param n80(64) と
  bootstrap n80(255) の 4x 乖離の機序 (parametric は Gaussian 仮定で psd→1、bootstrap は経験符号分布を保持し psd 床が binding)。
- **含意**: 保守的 (bootstrap) n80 を採るなら「n を増やせば p<0.05 で必ず解ける」は誤り。**psd 床の存在自体が中効果
  (psd 0.15–0.20 帯) の検出を構造的に律速** = これは **(c) 部分 underpower をむしろ補強する**。§7 の min_effect 行に反映済み。
- caveat 訂正: param/boot n80 乖離は「歪み/外れ値由来」(旧 caveat) ではなく **psd 床の機序差**が正確な説明。

### 9.2 [calibration_circularity, refuted=true, medium] 校正の family-internal 限定

- 校正は behavior=mean (③-favoring 軸) を彫った ramp-with-dip family 上で走り、`d_fn(n)` は **この family 内部の境界**。
  実 ESN/text proxy substrate への magnitude 転送は audit 自身が否定 (caveat 5)。
- 循環は完全には無い: 「③ load-bearing」は監査対象 gate でなく **独立 oracle (reach rate)** でアンカーされ、gate と実際に
  不一致する点 (d=0.14 n=30 で reach 分離だが gate FAIL) を確認済み = 境界は gate の自己充足トートロジーではない。
- **残る partial circularity**: behavior=mean は d=0 で fitness と corr+0.96 (collinear)、reach proxy も同 corridor 構造由来。
  → `d_fn(n)` の **一般性は実 substrate へ転送しない**。**推奨**: collinear でない合成 behavior でも再測、basin membership 直接 oracle も併報、実 substrate での偽陰性境界を別途測定。

### 9.3 [type1_tradeoff, refuted=false, medium scope gap] K3/K4 flip の null FPR 未検証

- 主対象「緩めたら見えた flip」(=K2_alpha=0.20) は null 再チェック済みで **disguised FPR inflation と判明** (FPR 0.043→0.184)、
  audit は正しく suppression 非認定 → この lens は refute されない。
- **scope gap (medium)**: Type I guard の null FPR sweep は **K2 のみ**。K4 (clip=False) と K3 (archive-max) の flip 系は
  null-ridge landscape での FPR 対照が未実施。K4 が null-ridge 上で単にノイズ分散を見かけの構造に膨らませる可能性は
  未 guard。ただし K4 は実 negative で verdict flip を起こしておらず、機序論 (floor 率 92.5%・raw spread 0.37 実測) は妥当。

### 9.4 その他の既開示 caveat

- **K3/K4 (ridge_fitness ~35ms/call) は G1 (<900s) 遵守のため予算縮小** (K3: n_seeds=10/n_evals=120/honest_n=12、K4: n_genes=120)。
  `_meta.budget_reduction_disclosure` に明記 (silent truncation なし)。縮小で K3 の③ gap は両読み出しとも PASS 未達だが
  archive-max が diff を ~50x 水増しする符号は安定。
- **K1 OFF の Type I 危険性は pure-null (gene 非依存 → method 間に構造差なし) では限定的にしか現れない**。「K1 を外すな」
  結論は本 harness でなく外部 EVOLUTION_SOUNDNESS_AUDIT の +0.29 水増し実測に依拠。理想は「構造あり-だが-③無効」null での再測。
- power シミュレーションは正規近似 (G1 遵守)、観測値の単発判定は src の exact `scipy.wilcoxon` を使用。

---

*生成物は全て `research/statistical_power_audit/` 配下。src/llcore/ は import 再利用のみで無改変。git 操作なし (orchestrator が一括 commit)。*
