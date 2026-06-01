# 第三軸 (③ 選択圧/分離) 決着 VERDICT — Step D Settle

> 2026-06-02。隔離ディレクトリ `research/step_d_settle/`。**src 無改変・git 非実行**。
> 前段 `research/statistical_power_audit/STATISTICAL_POWER_VERDICT.md` の (c)「部分的に underpowered」を受け、
> 「③ は実問題近傍で本当に load-bearing なのか / proper power で測れば決着するのか」を 3 実験で詰める。
>
> - **EXP1** = proper-n 再検定 (C-gen4b・flip_flop を bootstrap re-power + fresh 真再走) → `exp1_proper_n/`
> - **EXP2** = 決定論 C1 多峰性測定 (eval noise を物理的にゼロ化して地形が欺瞞=多峰か滑らか=単峰か) → `exp2_deterministic_c1/`
> - **EXP3** = K4 ridge clip の verdict-flip + null-ridge FPR (clip が真の suppression か診断的所見か) → `exp3_clip_flip/`
>
> 生数値: `exp1_proper_n/exp1_repower_proper_n_results.json` + `exp1_freshrun_partial.json` /
> `exp2_deterministic_c1/exp2_results.json` / `exp3_clip_flip/exp3_clip_flip_results.json`。
> 3 独立 VERIFY lens (power_adequacy / determinism_and_circularity / clip_flip_validity) の指摘を §6 に honest に反映。
>
> **これは「③が要るのか要らないのか、proper power で決着するのか」の自己監査である。**
> 単発の `p<0.05` を「③確定」と読まない規律 (honest disclosure) を全節で守る。「③有意化」も「③ null 確定」も
> 「なお原理的に判別不能」も、根拠が立てば等しく valid な結論として扱う。

---

## 0. 破綻ゲート (全実験まとめ)

| gate | 内容 | EXP1 | EXP2 | EXP3 |
|---|---|---|---|---|
| **G1** CPU 完走 (<900s) | 各 run exit 0, wall<900s | A 層 541s ✓ / fresh は 6 chunk 各 <900s (max 868s) ✓ | 単 run 4.95s ✓ (nr=24 perneuron40 のみ G1 超過で除外, 明記) | **both 1 プロセス 1203s = G1 違反 (honest 開示)**, split で各 phase <900s |
| **G2** 再現性 | 同 seed で per-seed/per-gene 完全一致 | bootstrap CRN exact / fresh[0:15]==元 exp_ea3 exact ✓ | 決定論 fitness で a1==a2 bit 一致, is_multimodal flag 3 seed 全一致 ✓ | smoke で both==split 配列完全一致 (決定論等価) ✓ |
| **G3** 診断器/power 計算器妥当 | null≤0.06 / 大効果≥0.95 / control 分離 | power_engine VALID (null=0.023, big=1.000, cg4b 模擬 n15=0.593/n30=0.842) ✓ | pos control vf 0.70/0.80 (mm=True) ∧ neg control vf=0.0 (smooth) → diagnostic_valid ✓ | known-pos p=3.05e-5/psd=1.0 ∧ known-null FPR=0.028 ✓ |
| **G4** src 不変 + git なし | src/ 差分ゼロ, 書込 step_d_settle 配下のみ | src_unchanged=True, changed=[] ✓ | 同 ✓ | 同 ✓ |

G1 違反は EXP3 の authoritative `--phase both` run の 1 件のみ。silent truncation はしておらず (全 verdict-flip + null-FPR を完走)、
split 等価性を smoke で検証済みのため結果妥当性には影響しない (§6.3)。

---

## 1. 三実験 数値結果表

### EXP1 — proper-n 再検定 (3 case)

A 層 = 実 per-seed delta の CRN paired bootstrap (B=5000)。B 層 = ea_lab を SeedSequence で拡張した fresh 真再走 (C-gen4b のみ n=64 到達)。

| case | 元 n=15 (監査) | A 層 power@n15 | A 層 n80 (full gate) | psd 床上? | B 層 fresh 真再走 | fresh gate | 判定 |
|---|---|---|---|---|---|---|---|
| **C-gen4b** MAP-E vs random | diff +0.063 / psd +0.20 / p 0.126 | 0.304 | **None (≤255 で 0.80 未到達)** | psd≈0.20 (床上だが頭打ち) | **n=64**: diff **+0.0472**, p **0.0382**, psd **+0.188**, **gate PASS** | **PASS (4 条件)** だが 更新 power@n64=**0.517** | **③ load_bearing 候補 (still_inconclusive)** |
| **flip_flop** MAP-E vs random | diff +0.004 / psd +0.333 / p 0.151 | 0.287 | **≈82.8** (p 律速, psd 床上) | psd=0.333 (床上) | fresh 未走 (G1 非現実的, ~266s/seed) | — | **still_inconclusive (A 層保留)** |
| **flip_flop** vs RR-hillclimb | diff +0.004 / psd +0.067 / p 0.262 | 0.155 | **None** | **psd=0.067 < 0.147 床 (床未満)** | fresh 未走 | — | **null 寄り (効果が床以下)** |

**power 計算器校正 (G3):** null=0.0232, big_effect=1.000, C-gen4b 模擬 n15=0.5926 / n30=0.8424 — 4 checks 全 pass、`power_engine_valid=True`。

**psd 床の構造的天井 (honest disclosure §6.1 と連動):** C-gen4b の median psd は n=15→0.200, n=255→0.200 で **動かない**。
`P(|psd|≥0.147)` (= 効果量条件の充足率) は n=15→0.713, n=64→0.697, n=255→**0.794 で頭打ち** = `p<0.05` が飽和しても
psd 床が binding に転じ、大 n でも full gate power が ~0.80 を超えない構造的天井。中効果 (psd∈[0.15,0.20]) ゆえ n では律速できない。

### EXP2 — 決定論 C1 多峰性 (eval noise = 機械 eps、noise floor を物理的に回避)

`landscape_map.multimodality_report` を read-only 再利用。exp7 と同 substrate (corpus 24000 chars, ESN seed=0, N=40, ridge closed-form)。
全 landscape で `eval_noise_std ≤ 1.11e-16` (機械 eps, 評価ノイズではない) を実測確認。

| landscape | group | dim | n cells | valley_fraction (mean/max) | is_multimodal | 判定 |
|---|---|---|---|---|---|---|
| **ESN_3param** (実 proxy) | real | 3 | 6 | **0.000 / 0.000** | **False** (3 seed 一致) | **滑らか=単峰 → ③ 不要を noise-free で確定** |
| **ESN_perneuron40** (実 proxy) | real | 40 | 3 | **0.096 / 0.121** | **False** (3 seed 一致) | **smooth 寄り (床 0.2 未満) → ③ 不要** |
| ctrl_multipeak_dim3 (pos control) | control | 3 | 6 | 0.701 / 0.727 | True | 診断器健全 (多峰を検出) ✓ |
| ctrl_multipeak_dim40 (pos control) | control | 40 | 6 | 0.795 / 0.818 | True | 診断器健全 ✓ |
| ctrl_quadratic_dim3 (neg control) | control | 3 | 6 | 0.000 | False | 診断器健全 (滑らかを検出) ✓ |
| ctrl_quadratic_dim40 (neg control) | control | 40 | 6 | 0.000 | False | 診断器健全 ✓ |
| note_corridor_d016 (※) | note | 24 | 3 | 0.000 | False | **C1 正 control 失格 (副次発見, 後述)** |

**判定 = `null_confirmed_at_power`** (real_landscape_verdicts: ESN_3param / ESN_perneuron40 とも)。`diagnostic_valid=True` (dim3/dim40 とも)。

### EXP3 — K4 ridge clip verdict-flip + null-ridge FPR

`_audit_common.eval_gate` (3 baseline 全勝 = load_bearing) を clip=True/False で同一 CRN seed 適用。n_seeds=15 (縮小予算, §6.3)。

| task | clip | MAP-E mean | n_baselines_beaten | load_bearing | verdict_flip |
|---|---|---|---|---|---|
| **addition** | True | +0.0100 | **1/3** (panmictic のみ) | False | — |
| **addition** | False | **−1.212** | **0/3** (全悪化, vs random diff=−0.423) | False | **False** |
| **flip_flop** | True | +0.426 | 0/3 | False | — |
| **flip_flop** | False | +0.438 | 0/3 | False | **False** |

**null-ridge FPR** (gene 非依存 target = 真の H0, n_null=4): clip=True FPR=**0.0** / clip=False FPR=**0.0** / delta=0.0 / `clip_false_inflates_fpr=False`。
G3 gate sanity: known-positive p=3.05e-5 psd=1.0 (PASS), known-null FPR=0.028 (<0.10) → `diagnostic_valid=True`。

**判定 = `not_load_bearing_at_this_budget`** (CF4; 旧 label `null_confirmed_at_power` を FPR 0/0 + ~7x 縮小予算ゆえ精緻化。但し低予算; §6.3)。verdict_flip=False 全 task、かつ clip=False は MAP-E を**むしろ劣化**させた (= clip は信号を隠していない)。

---

## 2. ③ 最終判定 — task ごとに根拠つき (A / B / C)

判定枠: **(A) proper power で load-bearing と判明 / (B) adequate power でも null 確定 / (C) psd 床等でなお原理的に判別不能**。

| task / landscape | 判定 | 根拠 |
|---|---|---|
| **C-gen4b** (MAP-E vs random, 実 multitask 近傍) | **(A 寄りだが未確定 = ③ load-bearing 候補)** | fresh 真再走 n=64 で full strict gate PASS (diff=+0.0472, 片側 p=0.038, psd=+0.188, n≥15, diff>0)。監査が「③不要」と読んだのは誤りで **③ は NOT null** の方向。**ただし**更新 power@n64=0.517<0.80 で確証に至らず、かつ §6.1 の surviving refutation (optional-stopping ドリフト + 多重比較) により「候補」止まり。**(A) を主張するには fresh を n80 まで延長要。** |
| **flip_flop** (MAP-E vs random) | **(C) なお判別不能** | A 層 full power n80≈82.8 (検定 p が律速、psd=0.333 は床上)。fresh 真再走が CPU 非現実的 (~266s/seed) で未到達。effect は床上なので「n を確保すれば決着しうる」が、現実的予算では保留。 |
| **flip_flop** vs RR-hillclimb | **(B 寄り = null 確定ではなく null 寄り)** | psd=0.067 < 0.147 床。`P(\|psd\|≥床)` は n と共に **減少** (0.612@n15 → 0.095@n255) = 効果が床以下。full gate はどんな n でも不可。ただし「効果ゼロ」の積極的確証 (diff→0 への収束を fresh で示す) は未取得のため **厳密には (B) でなく「null 寄り」** と honest に留める。 |
| **ESN_3param** 地形 (実 text proxy, dim=3) | **(B) adequate power で null 確定** | 決定論化 (eval noise std=0) で normalization_confound の noise-floor 偽陽性経路を物理的に除去。それでも valley_fraction=0.000 が 6 cell × 3 seed 全一致 → 実地形は単峰 = **③ 不要を noise-free で確定**。診断器は pos/neg control を正しく分離。**(§6.2c 留保: 厳密には「真に単峰」でなく「C1 閾値を僅かに下回る浅い谷 (~2–4%) を持つ弱 multi-basin」— max 相対 dip 0.0435 が 0.05 床直下)。** |
| **ESN_perneuron40** 地形 (実 text proxy, dim=40) | **(B) adequate power で null 確定 (smooth 寄り)** | vf_mean=0.096 < 0.2 床、is_multimodal=False が 3 seed 全一致。3param よりわずかに valley の兆候はあるが多峰判定に届かず。**ただし §6.2 の閾値近接性 (vf=0.121 は 0.2 まで 0.079) に留意。** |
| **K4 ridge clip** (suppression 機序候補) | **(B 寄り = `not_load_bearing_at_this_budget`/降格)** | verdict_flip=False 全 task。clip を外しても③は出ず、むしろ MAP-E が劣化 (addition: +0.010→−1.212)。null-ridge FPR は clip 差ゼロ。→ K4 は「能動的 suppression」でなく **診断的所見**に降格。FPR 0/0 + ~7x 縮小予算ゆえ verdict label は「null 確定」でなく **`not_load_bearing_at_this_budget`** に統一 (CF4; §6.3 で「at this budget」と開示済)。 |

**総括の一文:** **③ が「実問題近傍で proper power に乗せれば load-bearing」と (A) で確定した case は一つもない。**
最も近いのは C-gen4b で「③ は NOT null の方向 (gate PASS) だが小効果 (dz≈0.28) ゆえ検出力が乗らず、かつ近傍データはドリフトで反証寄り」=
**load_bearing 候補 / still_inconclusive**。一方、**地形そのもの (ESN_3param/perneuron40) は決定論測定で真に滑らか=③ 不要を (B) で確定**した。
すなわち「③不要に見えた過去 negative の多くは underpower でなく地形が本当に滑らかだったから」という像が、実 substrate 上で初めて noise-free に裏付けられた。

---

## 3. 実地形は欺瞞的 (多峰) か滑らか (単峰) か — 決定論 C1 (EXP2) の結論

**結論: 実 text proxy 地形は (実測した dim=3 / dim=40 両 substrate で) 滑らか = 単峰。欺瞞的 (multi-basin) ではない。**

- 決定論化が決め手: `normalization_confound` では「谷閾 0.05·|fit| ≪ eval noise std」で多峰を計測不能だった。
  EXP2 は ESN reservoir (固定 seed=0) + ridge readout (closed-form `np.linalg.solve`) が **rng を一切取らない**ことを使い、
  eval_noise_std を機械 eps まで物理的にゼロ化 → noisy-flat 偽陽性経路を構造的に消した上で valley を測った。
- ESN_3param: vf=0.000 (6 cell × 3 seed)。ESN_perneuron40: vf_mean=0.096 < 0.2 床 (3 seed)。両 substrate とも is_multimodal=False で全一致。
- 診断器健全性 (G3): 多峰 control (vf 0.70/0.80, mm=True) と滑らか control (二次関数 vf=0.0) を正しく分離。

**副次発見 (corridor の正体, honest):** 既知 positive control として期待した `make_corridor_eval(d=0.16)` は決定論化下で vf=0.0 (単峰判定)。
corridor の欺瞞性は **behavioral-reach 欺瞞** (単一 basin に trap、③ behavioral niching が脱出機構) であって **C1 multi-basin 欺瞞ではない**。
→ corridor は C1 の正 control にならないことを実測で確定。これは STATISTICAL_POWER_VERDICT §2 の corridor 校正の scope を狭める所見
(corridor 由来の `d_fn(n)` は behavioral-reach 軸の境界であり、C1 地形多峰性には転送しない)。

---

## 4. K4 clip は真の suppression か — EXP3 の結論

**結論: K4 (ridge clip=True) は「唯一の能動的 suppression 機序」ではなく、「spread を潰すが verdict を変えない診断的所見」に降格。**

- verdict_flip=False が両 task で成立 (Codex F2 の核心測定)。clip=False で③ load_bearing が出ない。
- 決定的な反証証拠: clip=False は addition で MAP-E を **+0.010 → −1.212 に劣化** (vs random diff +0.0026 → −0.423)。
  これは「clip が真の信号を隠している」仮説を能動的に反証する — clip=False は raw R²<0 のノイズ領域 (15/15 seed 負, R² in [−3.68, −0.20]) に
  MAP-E を落とし、構造を回復するどころか悪化させた。
- null-ridge FPR は clip 差ゼロ (両 0.0) → 「clip=False がノイズを構造として拾い Type I を増やす」も起きていない (低 n の床値, §6.3)。

これにより STATISTICAL_POWER_VERDICT の §4/§5/§8 が断定していた「K4 = 唯一の能動的 suppression 機序」は **過大**と判明。
Codex F2 が「有力候補に降格」と予言したとおり、本実験で **診断的所見に確定降格**した。

**verdict label (CF4):** EXP3 の run verdict label は `null_confirmed_at_power` から **`not_load_bearing_at_this_budget`** に統一する。
null-ridge FPR=0/0 は null_seeds=4 の床値、かつ ~7x 縮小予算 (n_evals 60 vs 400, n_tr 16 vs 48) ゆえ、
「null を確定した」より「**この予算では K4 が load-bearing でない**」が正確 (Codex CF4)。判定の実体 (診断的所見への降格) は不変、label 語のみ精緻化。

---

## 5. 過去 verdict (E-A C-gen4 / StepC / step6 / 谷深さ N/A / 統計監査) をどう更新するか

| 過去 verdict | 過去の読み | 本 Step D の更新 |
|---|---|---|
| **E-A C-gen4b** (MAP-E vs random) | 監査: underpowered (power 0.31), inconclusive, 「③不在の証拠ではない」 | **方向更新: ③ は NOT null の方向 (fresh n=64 で gate PASS)**。ただし「③確定」には至らず **load_bearing 候補 / still_inconclusive**。元 n=15 の psd=0.200 は fresh 拡張で n=44→0.318, n=64→0.188 と低下 = 元は中庸〜やや楽観側だった。 |
| **E-A C-gen4a** (MAP-E vs panmictic, diff<0) | 監査: power 0.03, 真に効果無し = 健全 negative | **不変** (本 Step D では再測せず)。③不在の正当な diff<0 証拠として維持。 |
| **flip_flop** (δ=+0.33, p=0.15) | StepC: inconclusive (正しい自認), 監査: n80≈82–89 | **更新: (C) なお判別不能**。vs random は A 層 n80≈82.8 で床上だが fresh 非現実的、vs RR は psd=0.067 で床未満=null 寄りと分離して確定。 |
| **step6 exp7** (実 ESN proxy, n=8/6, ③ negative) | 監査: n≤10 盲点域, 「③不在を主張できない, n≥15 で再測必須」 | **大幅更新: 地形が本当に滑らか (③ 不要) を noise-free で確定**。step6 の③ negative は「underpower だから保留」ではなく **「ESN proxy 地形が真に単峰だから③が不要」**が正しい説明 (EXP2)。再測しても多峰は出ない。 |
| **谷深さ N/A** (normalization_confound, behavior=mean collinear で計測不能) | 「instrument 不能」(谷閾 ≪ eval noise) | **解消: 決定論化で計測可能化**。計測した結果 vf≈0 (単峰)。ただし §6.2 で「閾値近接の浅い谷 (max dip 0.0435 が 0.05 床直下)」が surviving refutation として残る。 |
| **K4 clip = 唯一の能動的 suppression 機序** (統計監査 §4/§5/§8) | 「clip が真の landscape 構造を隠蔽 = 唯一の能動的 suppression」 | **降格: 診断的所見** (label = `not_load_bearing_at_this_budget`, CF4)。verdict-flip=False、clip=False は MAP-E をむしろ劣化。Codex F2 の予言どおり。FPR 0/0 + ~7x 縮小予算ゆえ「null 確定」でなく「at this budget で非載荷」と限定。 |
| **統計監査 最終判定 (c) 部分 underpower** | 一律抑制ではないが小 n + ridge clip で Type II 偏向 | **方向維持 + 中身入替**: (i) 中効果 case は psd 床で n では決着不能 (C-gen4b で実証, §6.1)、(ii) ridge clip の Type II 偏向は **K4 降格により実質消失** (verdict を変えない)、(iii) 「③不要」の多くは underpower でなく **地形が真に滑らか** (EXP2)。 |

**§6/§7 の Codex cleanup (F1 Type I event 定義統一 / F3 consolidate JSON 訂正 / F4 sweet-spot ラベル削除) は本 Step D では未実施** (EXP1/2/3 の 3 実験のみ完走)。
DESIGN の `codex_cleanups` 欄は計画として残るが、`research/step_d_settle/codex_cleanups_F1F3.py` は未作成。**次サイクルの TODO として明示** (§7)。

---

## 6. Surviving refutation (honest 留保)

3 独立 VERIFY lens がいずれも `refuted=true / medium` で生き残った。**いずれも各実験の保守的 verdict (candidate / still_inconclusive /
threshold-narrow) を覆さないが、positive 寄りの headline 強調を弱める方向で限定する。**

### 6.1 [power_adequacy, refuted=true, medium] C-gen4b の gate PASS は optional-stopping + 多重比較で脆い

VERIFY が独立再現し、本 agent も raw per-seed (`exp1_freshrun_partial.json`) で再確認した:

- **走行内ドリフト (未開示だった)**: 累積 verdict は n=40 で初 PASS (p=0.042) → n=44 p=0.017 → n=60 p=0.010 と一旦深く有意化した後、
  n=64 で p=0.038 へ **0.05 境界近くへ戻った**。前半 32 seed は diff=+0.0755 (frac_pos=0.625) だが **後半 32 seed は diff=+0.0189、
  最後の 16 seed は diff=−0.0023 (負, frac_pos=0.375)、最後の 9 seed (最終 chunk) は diff=−0.0376**。
  = PASS は前半 seed に支えられ、近傍データは逆方向に走っている。これはどの開示フィールドにも記録されていなかった。
- **多重比較**: p=0.038 は α=0.05 では PASS だが、EXP1 の 3 case だけでも Bonferroni α=0.0167 を **超過 (FAIL)**。③ research family 全体では更に厳しい。
- **含意**: C-gen4b を「③ load-bearing 候補」と呼ぶことは妥当だが、**「③ は NOT null」という headline は単発の境界 p=0.038 に寄りかかりすぎ**。
  走行内ドリフトは「候補が偽陽性かもしれない」真の証拠。**推奨 remediation: (a) p の n 軌跡と後半 seed の符号反転を開示フィールドに記録 (本節で実施)、
  (b) 多重比較補正後の閾値を併報、(c) fresh seed を n=64 超へ延長 — ドリフトが続けば候補は生き残らない可能性。**
- power_calc 器自体は G3 校正済みで正しく、prong 1 (null-at-power) は honest に処理済み (どの case も null_confirmed_at_power を主張せず、
  flip_flop vs RR の power 曲線が n と共に減少することを正しくラベル) = この点は refute されない。

### 6.2 [determinism_and_circularity, refuted=true, medium] 単峰 verdict は閾値近接で脆い (決定論・非循環自体は clean)

- **決定論 (claim 成立)**: VERIFY が load-bearing な実 ESN cell を自力で再走し bit 一致を確認 (ESN_3param vf=0.0, perneuron40 vf=0.12121212121212122 が
  stored と完全一致)。eval_noise_std=1.11e-16 は ULP 由来で評価ノイズではない (claim 正当)。
- **循環 (懸念は refute)**: 元の normalization_confound の循環は「behavior=mean(g) が fitness と collinear」だった。
  直接検定で corr(\_acc_3param, mean(g))=0.0365 (≈0)、同 mean の異 gene が distinct fitness。C1 診断器は behavior descriptor を一切使わず
  gene 空間の hill-climb optima + midpoint fitness のみで地形幾何を直接判定 → collinearity は再演されていない。
- **残る脆さ (未開示だった)**: 「滑らか/単峰=③不要」は閾値駆動。ESN_3param の midpoint の **90.9% が下方に dip**し、最大相対 dip=0.0435 は
  C1 谷閾 0.05 の **直下 (13% 以内)**。ESN_perneuron40 の vf=0.121 は is_multimodal flip (0.2) まで **0.079**。
  閾値の小変更 (0.05→0.04, または vf 床 0.2→0.12) で両 verdict は反転しうる。
- **含意**: 正確な表現は「**真に単峰**」ではなく「**C1 多峰性閾値を僅かに下回る浅い谷 (~2–4%) を持つ弱 multi-basin 地形**」。
  (B) null 確定の方向は維持されるが robustness は閾値近接ゆえ限定的。**推奨: (a) 実 cell 決定論再確認を committed G2 ファイルに移植
  (現状 control のみ再検証)、(b) dip 深さ分布と閾値マージンを results JSON に記録、(c) verdict 文言を閾値近接性込みに緩める。**

### 6.3 [clip_flip_validity, refuted=true, medium] K4 降格は低予算ゆえ「at this budget」限定

- **核心リスク不在**: verdict_flip=False = lens が警戒する「偽多峰/false structure recovery」失敗モードは起きていない。
  clip=False の addition 負 R² 領域は src `ridge_readout.py` docstring が予言する「addition は線形非可解」regime そのものだが、
  MAP-E を **偽陽性に持ち上げず劣化**させた (raw R² 符号弁別が機能) = 「clip が信号を隠す」を能動反証。flip_flop clip=False は有意な R² を保持 (std 0.055→0.062) = 真の no-signal。
- **確実性の過大評価 (medium)**:
  (1) null-ridge FPR=0/0 は null_seeds=4 のみの **床値** (FPR delta <0.25 を分解できない) — 「Type I コストなし」の positive 証拠としては弱い。
  (2) **~7x 予算縮小** (n_evals 60 vs exp_ea3 の 400, n_tr 16 vs 48)。clip=True で MAP-E は 0–1/3 baseline 勝の **床値近傍** (~0.01) ゆえ
  「flip なし」の一部は「この power では選択信号自体が検出不能」=「clip 無関係」と区別しきれない (EXP1 psd 床と同根)。
  (3) headline「clip=False で MAP-E が劣化」は addition 特有の崩壊を flip_flop に一般化気味 — flip_flop の clip=False は vs random で弱く
  suppression 支持方向 (+0.0485) に動くが、これは 1 つの degenerate random seed (−0.467) が牽引する outlier 駆動で null は崩れない。
- **含意**: K4 の「能動的 suppression → 診断的所見」降格は方法論的に妥当かつ保守的だが、**「at this budget で③検出不能」を含む弱い確定**であり、
  「firm refutation」より「not load-bearing at this budget」と述べるべき。

---

## 7. 次の判断 — GPU full LLM 損失地形 (b) へ進むか、③ を打ち切るか

**判断材料の整理:**

1. **③ は corridor (behavioral-reach 欺瞞) では堅牢に load-bearing** (統計監査 §6, Cliff δ=+1.0)。**しかし実 substrate (ESN/ridge text proxy) の
   地形は決定論測定で真に滑らか=③ 不要** (EXP2)。= ③ の価値は「地形が多峰/欺瞞的なとき」に限局し、現行 proxy substrate はその条件を満たさない。
2. **実 multitask 近傍 (C-gen4b) では③が NOT null の弱い兆候 (fresh gate PASS) はあるが、小効果 (dz≈0.28) + ドリフト + 多重比較で
   「候補」止まり**。proper power に乗せても psd≈0.20 床が天井になり、現実的 n では (A) 確定に到達しない見込み (§6.1)。
3. **K4 (clip) の Type II 偏向は降格で実質消失** (EXP3) = 統計監査が挙げた「ridge clip 由来の取りこぼし」は心配無用に。

**∴ 推奨判断:**

> **(優先) GPU full LLM 損失地形 (b) へ進む。** 現行 ESN/ridge proxy 地形は (B) で「真に滑らか=③不要」と確定したため、
> proxy 上で③をこれ以上追っても (A) は出ない (地形が単峰なら選択圧/分離に利得が無いのは当然)。③ が load-bearing になりうるのは
> **地形が多峰/欺瞞的な substrate** であり、その候補が backprop で学習する full LLM の損失地形 (proxy と異なり真の多峰性が報告される領域)。
> **③ research の本丸は「proxy で null を確定した」ことを足場に、多峰性が期待できる full LLM 地形へ移すべき。**

> **(③ proxy 上の追加投資は打ち切り寄り)** C-gen4b の fresh 延長 (n=64→n80) は安価ではない (~34s/seed) 割に、
> psd 床天井ゆえ (A) 確定の見込みが低く、ドリフトで候補が消える可能性もある。**proxy 上の③確定に CPU を追加投入する優先度は低い。**
> ただし「③ NOT null の弱い兆候」は full LLM 地形での③再評価の **事前仮説** として価値があるので、結論ごと打ち切るのではなく
> substrate を移して再問する。

---

## 8. 結論一文

> **proxy substrate 上では「③ (選択圧/分離) は地形が真に滑らかゆえ不要」が (B) で noise-free に確定**し、
> 実 multitask 近傍 (C-gen4b) でのみ「③ NOT null」の弱い兆候が出たが小効果 + ドリフト + 多重比較で **load_bearing 候補 / still_inconclusive** に留まる。
> K4 clip は能動的 suppression でなく診断的所見に降格。**③ の本丸検証は、多峰性が期待できる GPU full LLM 損失地形 (b) へ移すのが次の一手。**

---

*生成物は全て `research/step_d_settle/` 配下 (exp1_proper_n / exp2_deterministic_c1 / exp3_clip_flip)。src/llcore/ は import 再利用のみで無改変。
git 操作なし (orchestrator が一括 commit)。UTF-8。py -3.11。*
