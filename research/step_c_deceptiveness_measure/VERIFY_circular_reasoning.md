# VERIFY — adversarial lens: circular_reasoning

Date: 2026-05-31 / Lens: 循環論法 (メトリックが合成 knob の d を「定義上なぞる」だけで、
実タスクの欺瞞性を測れている保証が無い可能性)。デフォルトは結論を疑う。
Target conclusion: 「実タスクは全て below_threshold = 滑らか・③不要」。
方法: 既存 .py / .json を全文一次読取 + 合成側の循環性を `verify_circular_synth_probe.py`
(新規) で py-3.11 再測定 (出力 `verify_circular_synth_probe_out.json`)。既存ファイルは未改変。

判定: **結論を覆す (severity = high)**。実タスク測定値は magnitude として
**測定できていない (N/A) / rank のみ暫定** に格下げすべき。

---

## 0. 重大: briefing の前提が現存アーティファクトと矛盾する (最優先)

briefing は「behavior_elite_dip: flip_flop=0.018 / vdr=0.026 / step6=0.041, d*=0.1234 → 全 below」。
だが現物 `measure_real_tasks_results.json` 全文は別物:

| task | 実 JSON dip mean | 95%CI | below_threshold |
|---|---|---|---|
| variable_delay_recall | **0.0652** | [0.0298, 0.1006] | **false (ci_strictly_above=true)** |
| flip_flop | **0.0161** | [0.0094, 0.0227] | **false** |
| step6_text_proxy | **0.0107** | [0.0075, 0.0139] | true (ci_strictly_below) |

`metric_at_dstar = 0.015281642817850516` (briefing の 0.1234 ではない)、`d_star=0.16`、
`conclusion.all_below_threshold = **false**`。
→ behavior_elite_dip 自身で「全 below = 滑らか」は **既に偽**。

## 1. 循環論法の構造核心 (合成側を再測定で実証)

合成 corridor の定義 (`exp_knob_sweep.py`): genotype g∈[0,1]^24、`behavior_mean(g)=mean(g)`、
`corridor_eval = max(local(mean), glob(mean), ramp(mean)·(1-dip(mean))) + noise(0.008)`。
**fitness landscape は behavior=mean の関数そのもの**。dip 深さ d は mean 軸の上に彫られている。

`verify_circular_synth_probe.py` P1 (再測定, 5000 genes):

| d | corr(fitness, behavior=mean) |
|---|---|
| 0.00 | **+0.9629** |
| 0.16 | +0.7431 |
| 0.50 | -0.5101 |
| 1.00 | -0.7639 |

→ smooth (d=0) では fitness は behavior=mean にほぼ完全従属 (corr+0.96)。メトリックが合成で
「d を復元」できるのは **behavior == knob を彫った軸そのもの**だから = 定義上のなぞり (循環)。
d を上げると dip が mean→fitness 関係を非単調化し corr の符号が反転する (谷の効果)。
いずれも「fitness が behavior=mean の関数」を裏付ける。

## 2. 比較の物差し d* は合成記述子の CLT 劣化 shadow (再測定で 7.2x を実証)

P2 (再測定, 20万サンプル): `mean(U(0,1)^24)` は CLT で **0.5003±0.0590** に集中 (range [0.245,0.767])。
- P(b ∈ well[0.60,0.70]) = **0.0451**
- P(b ≥ 0.90) = **0.0**、P(b ≥ 0.95 大域ピーク) = **0.0** (20万サンプルで 1 件も無し)

→ 大域ピーク (高 behavior 域) は**サンプル不到達**。behavior-elite envelope は大域ピーク bin に
届かず、測る dip は真の d の減衰影。

P3 (再測定, n_seeds=3): measured synthetic dip vs 幾何期待 0.8·d:

| d | measured dip (95%CI) | 幾何期待 0.8·d | 減衰 |
|---|---|---|---|
| 0.16 | **0.0177** ([-0.0093,0.0447]) | 0.128 | **7.2x** |
| 0.50 | 0.0259 ([-0.0374,0.0892]) | 0.400 | 15.4x |
| 1.00 | 0.0328 ([-0.0660,0.1317]) | 0.800 | 24.4x |

実タスクの raw dip を照らす閾値 metric_at_dstar=0.0153 (JSON 記録; 本再測定 d=0.16 でも同オーダー
0.0177) は、合成 behavior=mean の CLT 潰れ + 大域ピーク不到達による **約 7-24x の減衰影**。
**magnitude 比較は意味を持たない**。コード作者も `measure_real_tasks.py` L22-27 +
honest_disclosure で「約8x減衰した shadow」「RANK は転送 (ρ=1) するが CALIBRATED MAGNITUDE は
転送しない」と自白。なお CI が 0 をまたぐ (再測定の n_seeds=3 では負値も含む) ことも、この物差しが
ノイズフロアと区別しにくい弱い量であることを示す。

## 3. 軸不一致 (作者開示) — 同じ意味の量を測っていない

- 合成: `behavior=mean(gene)` (1D)。
- 実タスク: task 固有 2D 記述子 → PCA 第1主成分で 1D 射影。
  - reservoir (vdr/flip_flop): `make_behavior` (reservoir.py L182-209) =
    `(eff_mem_norm = tanh(mean(1/leak)/50), std(leak))`。これは **leak のみ使い w_in を捨てる**
    (Codex pair-review が 2026-05-30 に BLOCKER 指摘、コメントに明記)。実ダイナミクスは
    leak+w_in 依存なので記述子は landscape の一部しか見ていない。
  - step6: `behavior=(rho, leak)` = 正規化 gene 成分。
- `honest_disclosure.axis_mismatch`: 「Real-task behavior axes differ from synthetic
  behavior=mean ... NOT a claim about each task's true d. RANK transfers, MAGNITUDE does not.」

→ 「合成と実タスクで同じ意味の量を測っているか」という当レンズの問いに、作者自身が「No (軸が
違う、magnitude 非転送)」と開示。循環論法の懸念は否定されるどころか前提として開示されている。

## 4. cross-metric 不一致 = 循環論法の決定的症状

| task | elite_dip (d*=0.0153) | fdc (d*=0.2282) | downhill (d*=0.2222) |
|---|---|---|---|
| vdr | 0.0652 → above | 0.926 → above | 0.463 → above |
| flip_flop | 0.0161 → above | 0.765 → above | 0.389 → above |
| step6 | 0.0107 → **below** | 0.307 → **above** | 0.611 → **above** |

- 3 メトリック全て `all_below_threshold = false`。
- per-task で step6 が割れる (`per_task_agreement` step6=false, `all_tasks_agree=false`)。
- fdc 校正: spearman_full=0.40, =1.0 のみ d<=0.20, d>=0.30 で **FDC 反転**, provisional
  (`reproduces_threshold=false`)。
- downhill 校正: spearman=0.4945, **monotone=false**, per-seed [0.0,0.0,0.667] 等の jumpy。
- prior `VERIFY_DOWNHILL_CROSSMETRIC.md`: 「NOT a clean 3/3 'smooth' consensus」、downhill は
  flat profile で「reads HIGH deceptiveness ... for the wrong reason (argmax bin is noise)」=
  欺瞞性の意味と逆を測定。

3 メトリックが「タスク内在の欺瞞性」を測れているなら一致すべき。値域も verdict もバラバラ =
各メトリックが別々の射影/正規化/校正という無根拠写像を課している (§3 の軸不一致) 直接の帰結。

## 5. 結論 / severity / 推奨

「実タスクは全て below_threshold = 滑らか・③不要」は **支持されない**:
(a) behavior_elite_dip 自身 all_below=false、(b) fdc/downhill も all_below=false、
(c) 合成 behavior=mean は fitness と corr≈+0.96 (循環)、(d) 物差し 0.0153 は CLT 7-24x 減衰影で
magnitude 非転送、(e) 軸不一致を作者が開示、(f) 3 メトリック verdict 不一致。

**循環論法レンズ判定: メトリックが実タスクの欺瞞性を「測れている保証」は無い (軸不一致 +
magnitude 非転送 + cross-metric 不一致)。severity = HIGH。**
正しい状態は negative(③不要) でも positive でもなく、**「magnitude として測定できていない
(N/A) / rank のみ暫定」**。「全 below = 滑らか」は false consensus。

推奨:
1. 合成と実タスクで**同一定義の behavior 記述子**を採るか、commensurability を corr/rank で
   **実証**する (assumed を排除)。
2. 物差しを CLT で潰れない合成 behavior に張り替える (behavior=mean は大域ピーク P=0.0 未到達で
   7-24x shadow 化、magnitude 比較を無意味にしている)。
3. cross-metric の per-task 不一致 (特に step6) と downhill の非単調校正・flat-profile artifact、
   fdc の d>=0.30 反転が解消するまで「滑らか」を主張しない。
4. それまで Step C の③要否 **magnitude** 結論は保留。briefing が引いた数値 (全 below, d*=0.1234)
   は現存アーティファクトと不一致 = 結論の出典自体を要再確認。

## Artifacts (新規, 既存ファイル未改変)
- `verify_circular_synth_probe.py` — 合成側循環性の再測定 (P1 corr / P2 CLT / P3 attenuation)。
- `verify_circular_synth_probe_out.json` — 再測定の機械可読出力 (authoritative)。
- 本メモ `VERIFY_circular_reasoning.md`。
