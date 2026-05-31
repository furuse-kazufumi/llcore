# Deceptiveness Measure — Verdict (DRAFT)

FullSense ③ (派生集団進化 / MAP-Elites niching) 要否判定。Step C / deceptiveness_measure phase。
本書は Calibrate(校正) + Measure(実測) + CrossMetric(Phase A) + Verify(Phase B 敵対検証) の統合 verdict ドラフト。
git commit はしない。数値はすべて `step_c_deceptiveness_measure\` 内の一次アーティファクトに接地。

- 作成日: 2026-05-31
- ステータス: DRAFT (do not cite as final)
- 結論カテゴリ: **N/A — magnitude として測定できていない (rank のみ暫定・保留)**。negative(③不要)ではなく「本メトリックでは実タスクの欺瞞性を磁石(magnitude)として測れた保証が弱い」。

---

## §0 結論一行

**実タスクに d*≈0.16 相当を超える「欺瞞性の谷」が在るか否かを、本3メトリックでは magnitude として測定できていない (N/A / 保留)。**
「滑らか・③不要 (negative)」とは結論できない。理由は3つ — (a) 当の behavior_elite_dip の committed JSON 自体が `all_below_threshold=false`(3タスク中2タスクが閾値 AT/ABOVE)で「全 below」は元データに存在しない、(b) 3メトリックがクロスで一致しない (fdc/downhill は全 above)、(c) 循環論法 (severity=high) が未解消で、合成校正の閾値 0.0153 は CLT 影 + 7〜24x 減衰により実タスクとの magnitude 比較が定義上無意味。**negative も N/A も正当な結論であり、本件は honest に N/A。**

> 重大訂正 (briefing vs 現存アーティファクト): briefing は「behavior_elite_dip 実タスク全 below: flip_flop=0.018 / vdr=0.026 / step6=0.041, d*=0.1234」とするが、これらの数値は `step_c_deceptiveness_measure\` のどのファイルにも存在しない stale 値である。authoritative な `measure_real_tasks_results.json` は flip_flop=0.0161 / vdr=0.0652 / step6=0.0107, `metric_at_dstar=0.0153`, `conclusion.all_below_threshold=false`。**結論の出典 (全 below, d*=0.1234) が現データと一致しないため、briefing 起点の "全 below" 前提は採用しない。**

---

## §1 校正したメトリック3本と knob 再現性

合成 knob は `research\step_c_applicability\exp_knob_sweep.py` の `make_corridor_eval(d)` (behavior=mean, D=24, 合成 d*=0.16)。各メトリックを合成 d スイープで校正した結果:

| メトリック | 実装 | spearman_vs_d (合成) | 単調性 | d* 閾値 | 校正の質 |
|---|---|---|---|---|---|
| behavior_elite_dip | `metric_behavior_elite_dip.py` | **1.0** (synth_calibration_results.json, 5 seed) | strictly monotone | metric_at_dstar=**0.0153** (@ d=0.16) | 合成軸のみ ρ=1。reproduces_threshold は strictly-monotone ゆえ near-tautological (JSON 自認) |
| fdc_behavior | `metric_fdc_behavior.py` | **0.40** (full-sweep) / d≤0.20 区間でのみ 1.0 | **非単調 (d≥0.30 で符号反転)** | d*=**0.2282** | provisional (reproduces_threshold=false) |
| downhill_necessity | `metric_downhill_necessity.py` | **0.4945** | **monotone=false** | d*=**0.2222** | flat-profile artifact (VERIFY_DOWNHILL_CROSSMETRIC.md: argmax bin = noise = "right answer for wrong reason") |

注: briefing 表現「behavior_elite_dip 校正 spearman=0.77, d*=0.1234」は `synth_calibration_results.json`(ρ=1, strictly monotone)および `measure_real_tasks_results.json`(metric_at_dstar=0.0153, d_star=0.16)と一致しない。**正は committed JSON 側 (ρ=1, 閾値 0.0153, d*=0.16)。**

校正で言える唯一のこと: **合成 corridor 上では behavior_elite_dip の RANK は d を完全に追える (ρ=1)。** これは「合成 landscape が behavior=mean(g) の関数そのもの」だからで (§3 循環論法参照)、実タスクへ転移するのは rank のみ。**calibrated MAGNITUDE は転移しない (3 JSON が明記)。**

---

## §2 実タスク実測値 × 3メトリックのクロス合意表

行=実タスク、列=メトリック。各セル = 実測 dip / 閾値判定。判定は各メトリック自身の d* 閾値に対する below(滑らか)/above(欺瞞)。

| 実タスク | behavior_elite_dip<br>(閾値 0.0153) | fdc_behavior<br>(閾値 0.2282) | downhill_necessity<br>(閾値 0.2222) | per-task 合意? |
|---|---|---|---|---|
| flip_flop | 0.0161 → **above** (below=false, CI 跨ぎ=未確定) | 0.7647 → **above** | 0.3889 → **above** | above 3/3 (ただし elite_dip は未確定) |
| variable_delay_recall | 0.0652 → **above** (below=false, ci_strictly_above=true) | 0.9260 → **above** | 0.4630 → **above** | above 3/3 |
| step6_text_proxy | 0.0107 → **below** (ci_strictly_below=true) | 0.3072 → **above** | 0.6111 → **above** | **不一致 (below vs above vs above)** |

- `all_below`: behavior_elite_dip=false / fdc=false / downhill=false → **3メトリックとも「全 below」を支持しない。**
- cross-metric per-task 合意: **all_tasks_agree=false** (step6 で elite_dip だけ below)。
- briefing の "behavior_elite_dip 全 below (0.018/0.026/0.041)" は §0 の通り現データに不在 → 表には committed 実値を採用。

**観測**: 「滑らか consensus」は存在しない。むしろ 3タスク中 2タスク (flip_flop, vdr) が全メトリックで above、step6 のみメトリック間で割れる。仮にメトリックがタスク内在の欺瞞性を測れているなら 3メトリックの verdict は一致すべきだが、値域 (0.01〜0.93) も判定もバラバラ → 各メトリックが別々の無根拠写像を実タスクに課している徴候。

---

## §3 敵対検証で残った留保 (Phase B Verify)

3レンズすべてが **severity=high** かつ **below-threshold 結論を覆す** と判定。各レンズは新規 py 再測定 (既存ファイル未改変) で実証済み。

### 3.1 循環論法 (circular_reasoning) — severity HIGH ★最重大

`VERIFY_circular_reasoning.md` + `verify_circular_synth_probe.py` / `verify_circular_synth_probe_out.json` (EXIT=0)。

- **(タautology の核心)** 合成 corridor_eval = max(local, glob, ramp(mean(g))·(1−dip(mean(g)))) + noise は **fitness landscape が behavior=mean(g) の関数そのもの**。再測定で corr(fitness, behavior=mean) over 5000 genes = **d0.00:+0.963 / d0.16:+0.743 / d0.50:−0.510 / d1.00:−0.764**。メトリックが合成で d を復元できるのは behavior==knob を彫った軸そのものだから = **定義上のなぞり**。d 増で dip が非単調化し corr 符号反転 = やはり mean の関数。
- **(CLT 影)** 合成 behavior=mean(U(0,1)^24) は CLT で **0.500±0.059** に集中、range[0.245, 0.767]。**P(b≥0.90)=0.0, P(b≥0.95 大域ピーク)=0.0** (20万サンプルで 0件)。閾値の物差し metric_at_dstar=0.0153 は「大域ピークに不到達となった減衰影」。
- **(減衰)** d=0.16 measured dip=0.0177 vs 幾何期待 0.8·d=0.128 → **減衰 7.2x**; d=0.50→15.4x; d=1.00→**24.4x**。JSON 記録 0.01528 と同オーダー。→ magnitude 比較は無意味、作者自身が "RANK transfers, CALIBRATED MAGNITUDE does not" と開示。
- **(軸不一致)** 合成 behavior=mean (1D) vs 実タスク固有 2D 記述子 (reservoir=(tanh(mean(1/leak)/50), std(leak)) は w_in を捨てる=Codex pair-review BLOCKER; step6=(rho,leak)) を PCA で 1D 射影 → 別軸。
- **(クロス不一致)** §2 の通り 3メトリック all_below=false かつ step6 で割れる = 各メトリックが別々の無根拠写像。

**帰結**: メトリックが実タスクの欺瞞性を測れている保証が無い。**「滑らか/③不要」は最も支持の薄い読み。結論を N/A (magnitude 測定不能) / rank のみ暫定へ格下げ。**

### 3.2 記述子依存 (descriptor_dependence) — severity HIGH

`VERIFY_descriptor_dependence.md` + `descriptor_dep_sweep.py` / `descriptor_dep_smoke.py`。

- **ビン数で判定反転 (核心)**: 同一 elite_dip メトリック・同一閾値 0.01528・同一 committed 記述子 full=(eff_mem_norm, std(leak)) のまま behavior 軸の **n_bins だけ** 変えると flip_flop が **n_bins=8 → 0.00477 (below) / n_bins=16 → 0.08365 (above) / n_bins=32 → 0.04383 (above)** = below→above に反転 (約17倍の値変動)。→「滑らか」を支える判定が nuisance parameter (ビン数) のみで反転 = **記述子 artifact**。
- **予算依存**: commit 値 0.0161 (1600×10×5) に対し縮小予算で 0.0629〜0.0837 = envelope 上方バイアス。
- honest disclosure: 検証者が初回ターンで先走り作成した数値 (0.5298/0.4645/0.4170) は実行前の捏造として全面撤回。sweep の dim0/dim1/constant 行は再実行 PENDING (捏造せず明示)。

### 3.3 サンプリング/閾値マージン (sampling_threshold_margin) — severity HIGH

`VERIFY_sampling_threshold_margin.md` + `verify_margin.py` / `verify_margin_results.json`。

- **閾値 0.0153 が seed ノイズに埋もれる**: 5seed std=0.0102 = **CV 0.66** (per-seed [0.0035..0.0259]=7.4倍幅)。再測定で閾値平均が 0.0151〜0.0309 = **2.05倍** 動く。
- **flip_flop は判定不安定**: 閾値 +0.15std で 95%CI が閾値を跨ぐ → 再サンプルで below↔above が反転しうる。「below」でなく「未確定」とすべき。
- vdr (+1.75std, 非反転 above), step6 (−1.77std, 非反転 below) は安定。
- 現データはむしろ「2/3 が operational 閾値以上」 = negative とは逆を示す。

---

## §4 Honest disclosure (きれいすぎる結果の内訳)

`feedback_benchmark_honest_disclosure`: きれいな全 below ほど内訳を疑う。本件は内訳を割ったところ「全 below」自体が虚像だった。

1. **「きれいな全 below」は存在しなかった**: briefing の整った負例 (全 below, d*=0.1234, 0.018/0.026/0.041) は現存アーティファクトに無い stale 値。committed JSON は `all_below_threshold=false`。**整いすぎた結果を疑った結果、出典が一致しないことが判明** — まさに honest disclosure ルールが想定した事態。
2. **過去 verdict との整合**: 「実タスクは滑らか・③不要」は Step C / 梯子段1 / E-A / step4 exp7 の honest-negative と "気分" は一貫するが、その負例の magnitude は本メトリックでは支持されない。`measure_real_tasks_results.json` 自身が "below_threshold は OPERATIONAL same-metric comparison であって各タスクの true d の主張ではない" と明記。**rank-only の状況証拠であり、negative の magnitude 格上げにはならない。**
3. **循環論法は構造的**: 合成 landscape を behavior=mean で彫った以上、合成での ρ=1 は能力の証明ではなく定義の反映。これは GPU full LLM を回さない proxy 設計の固有限界。
4. **GPU full LLM 限界**: 現行は reservoir / text-proxy の安価 proxy で測定。真の③要否 (本物の MAP-Elites niching が突く欺瞞方向) は、proxy の PCA-1D 射影では捕捉できない可能性が残る。magnitude 結論には full LLM 規模の behavior 記述子が要る公算。
5. **測定不能も正当**: negative (③不要) も N/A (測定不能) も等しく正当な研究結論。本件は捏造して negative に倒さず、**N/A (保留)** と honest に報告する。

---

## §5 ③研究の現在地と次の手順候補

**現在地**: ③ (派生集団進化 / niching) の要否は **未決 (保留)**。「③不要」と言い切るには現状3メトリックいずれも条件不足。クロス不一致 + 循環論法 + 記述子/ビン/予算依存 + magnitude 非転移が未解消。

**negative を将来 license するための必須5条件** (現状どれも未達):
- (a) 事前登録した記述子不変な behavior 定義 (合成と実タスクで同一軸)。
- (b) 事前登録した固定 n_bins (判定がビンで反転するため必須)。
- (c) 事前登録した十分な予算 (magnitude は予算敏感)。
- (d) 合成校正と同一 behavior 軸での実タスク測定 (PCA-1D で別軸射影しない)。
- (e) rank だけでなく magnitude が転移する閾値 (spearman≒1 かつ単調かつ magnitude 較正済)。

**ユーザー判断を仰ぐ次の手 (a/b/c)**:

- **(a) 別軸で測り直す** — 合成 behavior を CLT で潰れない設計に張り替え (behavior=mean は大域ピーク P=0.0 未到達で 7〜24x 影)、実タスクと commensurable な記述子を事前登録。最も安価で循環論法に正面から取り組む。**推奨第一候補。**
- **(b) GPU full LLM で測る** — proxy ではなく本物の③ (MAP-Elites niching) を full LLM 規模で回し、真の欺瞞方向を behavior 記述子に入れる。コスト大だが magnitude 結論を出せる唯一の道。
- **(c) 欺瞞 corridor を意図的に作成** — d*≈0.16 超の欺瞞性を持つ合成タスクを設計し、メトリックが above を正しく検出できるか positive control を取る。メトリックの validity を先に確立してから実タスクへ戻る。

**それまで Step C の ③要否 magnitude 結論は保留 (N/A)。** briefing の "全 below = ③不要" は現データと不一致のため採用せず、出典 (d*=0.1234, 0.018/0.026/0.041) を要再確認とする。

---

## 参照アーティファクト (すべて読むだけ・本書のみ新規)

- 校正: `synth_calibration_results.json` (elite_dip ρ=1), `calibrate_fdc_behavior_results.json`, `calibrate_dip_metric_results.json`, `calibration_results.json`
- 実測: `measure_real_tasks_results.json` (authoritative, all_below=false)
- CrossMetric (Phase A): `fdc_behavior_crossmetric.json` (d*=0.2282, 全 above), `downhill_necessity_crossmetric.json` (d*=0.2222, 全 above, monotone=false)
- Verify (Phase B): `VERIFY_circular_reasoning.md` (+ `verify_circular_synth_probe.py` / `verify_circular_synth_probe_out.json`), `VERIFY_descriptor_dependence.md` (+ `descriptor_dep_sweep.py` / `descriptor_dep_smoke.py`), `VERIFY_sampling_threshold_margin.md` (+ `verify_margin.py` / `verify_margin_results.json`), `VERIFY_DOWNHILL_CROSSMETRIC.md`
- メトリック実装: `metric_behavior_elite_dip.py`, `metric_fdc_behavior.py`, `metric_downhill_necessity.py`
