# BG9 VERDICT — ③ 欺瞞地形 in KernelGenome 空間

> 2026-06-02。事前登録 `BG9_PREREGISTRATION.md` に対する 3 値判定。隔離 `research/kernel_diversification/`、
> src/llcore 無改変、git は orchestrator 一括。本 verdict は **substrate validity (BG9-3) + harness validity
> (BG9-4) + 敵対検証 red-team** の 3 段から確定した。

---

## 0. 一文結論

> **BG9 = N/A (形式) → 実質「決定的な構造的 negative」**: kernel 多様化空間では ③ (MAP-Elites の
> kernel_id niching) を **強 baseline RR-hillclimb から分離できない**。kernel 選択は 4 離散の単一低次元座標で、
> RR は restart で kernel_id を直接サンプルして欺瞞 corridor を回避する。④の優位 (behavioral stepping-stone)
> が成立するには **高次元 behavior 空間** が要る (Step4 の mean(24) は CLT 不到達ゆえ RR を排除できた)。
> **= CPU kernel-union 路は③検出について構造的に閉じている。** real task smoke でも③は勝たない。
> ③ の残り路は **高次元の GPU full-LLM 損失地形のみ**。

これは「谷深さ N/A (計測器が circular で magnitude 測定不能)」とは質的に異なる: **計測器 (harness) は健全**
(negative control 正しく null、GA/random は明瞭に検出)。N/A の原因は **substrate が構造的に③欺瞞を host できない**こと。

---

## 1. 三段の証拠

### 1.1 substrate validity (BG9-3) — PASS だが弱い
kernel-favoring task 群を第一原理設計し強 BG6 で validity 検定 (`bg6_strong_results.json`):
- 写像は **非定数** (PASS): probe で mamba/linear_attn/rwkv が別 task で best。
- **但し弱い**: **hopfield はどの task でも勝たず** (対角スカラ mock で tanh アトラクタが機能不全、per-seed
  R² が 0/0.99/0 と二極化) = 実質「4 kernel union」でなく **3 kernel**。clean な専門化は 2 軸のみ
  (selective_copy↔mamba margin +0.16 / weighted_accum↔linear_attn +0.10〜0.33)。bistable_denoise は
  仮説 (hopfield) 外で rwkv が拾い、leaky_tracking は margin +0.02〜0.03 で fragile。
- 採用 real suite = `{selective_copy, bistable_denoise, weighted_accum}` (leaky 除外)。
- **弁別の存在 ≠ 多峰/障壁**。non-inert 化は成功 (BG6 の memory_tasks 中立の轍は回避) だが、欺瞞地形を保証しない。

### 1.2 harness validity (BG9-4) — positive control が validate しない (構造的)
固定 param (pre-reg §1: behavior=(kernel_id, theta L1)/n_bins=(4,8)/init_batch=64) で MAP-E vs
RR-hillclimb / panmictic-GA / random を CRN paired・honest 再評価で比較 (`bg9_smoke_results.json`):

| 基質 | 結果 |
|---|---|
| **positive control** (synthetic kernel-barrier) | MAP-E が panmictic (+0.423) と random (+0.208) は撃破するが **RR には勝てない (+0.051, p=0.31, δ=+0.20 → FAIL)** → 3 baseline 全勝に届かず = **harness validity N/A** |
| **negative control** (delayed_recall, kernel 中立) | 全 method R²≈1.0 飽和、MAP-E 優位なし = **正しく null** (false-positive なし、harness 健全) |
| **real** (kernel-favoring multi-task) smoke | MAP-E beaten 0/3、panmictic が逆に MAP-E を上回る = **③ 勝たず** (honest negative 一次像) |

3 geometry (kid-ramp+theta-corridor / product-of-theta / per-kid theta-corridor) を試すも、RR を締めると
MAP-E も target に届かず分離崩壊。**RR が kernel_id∈[0,4) を直接サンプルする**のが根因。

### 1.3 敵対検証 red-team — 構造的 N/A は robust (反証できず・強化)
独立 red-team が別角度で BG9-4 主張を反証 or 確証 (`red_team_*.py` + JSON):
- **機構 evidence**: instrumented RR が素 `selection_lab.random_restart_hillclimb` と bit 一致。positive
  control 上で restart kid が 4 basin に [12,18,16,18] とほぼ一様分散、target 到達 88%、best_origin =
  restart→in-basin climb が 6/8 seed → **「RR は restart で kernel_id 直接サンプル→谷回避」を数値確証**。
- **faithful 反証 4 構成 (高次元 theta corridor[pre-reg 逸脱 probe] / sequential-kernel / in-basin L1
  corridor / deceptive multi-basin) すべて `beats_rr=False`**。緩めると RR も同等到達、締めると MAP-E が
  先に starve。**RR だけ落ちる窓が構造的に空**。
- **境界**: theta corridor 次元 D=0→3 sweep で締めるほど MAP-E が RR より速く starve (D=3: MAP-E reach
  0.08 vs RR 0.42)。base_seed 3 通りで同一 → **「RR を排除して③が通る behavior 次元は kernel 空間に存在しない」**を定量確証。
- **総合**: BG9-4 の構造主張は反証できず、むしろ 3 タスクで強化された。

---

## 2. 構造的洞察 (なぜ kernel 路は閉じているのか) — 本 verdict の payoff

> **③ (MAP-Elites の behavioral niching) が強 baseline を上回るのは、「難所」が
> 高次元 behavior 空間にあって直接サンプリング (random restart) で到達できないときだけ。**

- Step4 の欺瞞 corridor: `behavior = mean(24 次元 gene)`。CLT で平均が 0.5 に集中 → 大域ピーク (mean≈0.9) は
  measure-zero 域 = random/RR が直接サンプル不能。だから MAP-E の archive stepping-stone が load-bearing だった。
- kernel 選択: kernel_id は **4 離散の単一座標**。RR の restart は kernel_id を一様サンプルし全 4 kernel を直撃 →
  「best kernel を探す」のに谷を跨ぐ必要がない (teleport)。よって ③ の niching 優位が原理的に出ない。
- theta 空間に欺瞞を移しても (red-team C-A〜C-D)、RR は restart 後に in-basin で greedy climb するため、
  corridor を RR が抜けられない程度に締めると MAP-E も同程度に starve する。**RR fail ∧ MAP-E succeed の窓が無い。**

→ **「探索空間を kernel 多様化で拡張すれば③が unlock するか」(Step4 §7 の問い) の答えは NO (CPU では構造的に)。**
拡張が ③ を unlock するには、追加した自由度が **高次元で直接サンプル困難** な behavior を生む必要がある。
kernel 選択 (低次元・離散) はその条件を満たさない。

---

## 3. ③ 研究 arc 全体との整合

| フェーズ | 結論 |
|---|---|
| Step4 (合成 corridor) | ③ は欺瞞 corridor で decisively load-bearing (本物の機構) |
| Step C / 梯子段1 / E-A | 実 proxy・多タスク・記憶タスクで③不要 (滑らか/中立) |
| Step D (決定論 C1) | 実 text proxy 地形は noise-free で真に滑らか=単峰、③不要を (B) 確定 |
| **BG9 (kernel 多様化)** | **kernel-union でも③は load-bearing になれない (構造的: 低次元選択は RR を排除不能)** |

BG9 は arc に **「なぜ CPU で③が立たないか」の構造的説明**を追加する: 実 CPU 基質の「難所」(leak 率・
kernel 選択等) は低次元で、強 baseline が直接解ける。③ の優位は高次元 behavior を要する。

---

## 4. GPU 判断への含意 (更新 — 重要)

- **CPU 出し切りゲートが CLEAR**: BG9 が最後の CPU 路 (kernel-union) を構造的に閉じた。③ の残り路は
  **高次元の GPU full-LLM 損失地形のみ**。
- **構造的洞察が GPU の賭けを better-motivated にする (但し依然 bet)**: ③ は高次元 behavior で初めて意味を持つ。
  full-LLM パラメータ空間は数百万次元 = まさに高次元。だから GPU 検定は「full-LLM だけが例外かも」という弱い賭けでなく
  「③ は高次元を要し full-LLM が高次元域」という原理に沿う。**ただし依然 bet**: 実 LLM 地形が backprop 系の強 baseline で
  ナビゲート可能なら③不要 (BG9 の RR と同型のリスク = GPU でも「強 baseline が直接解く」可能性)。
- **タイミング (前回フレーム更新)**: GPU は「③のため単独」でなく **ポートフォリオ判断** (llive 実 LLM fitness 等と相乗り) +
  **クラウド借りで事前登録1本** (資本コミット前) が適正。BG9 の構造的洞察を **GPU 事前仮説** とする:
  「③が full-LLM で load-bearing なら、その難所は高次元 behavior 空間にあり直接サンプル/backprop で到達困難なはず」。
  これが GPU 実験の falsifiable な go/no-go 基準になる。

---

## 5. pre-reg §4 との対応 (3 値判定の根拠)

pre-reg §4: **N/A = positive control すら③不成立 (harness が③を検出できない)**。本件は **まさにこの行**。
- ただし「測定不能」の質が谷深さ N/A と異なる: あちらは計測器が circular (behavior=fitness 定義のなぞり)。
  こちらは **計測器健全・substrate が構造的に③欺瞞を host 不能**。よって N/A だが **「kernel 路は閉じている」という
  決定的 negative 情報**を持つ (情報量のある N/A)。
- **③成立は出ていない** (整いすぎた③成立を疑う規律は不要だった = honest prior 通り)。
- real の full ≥15-seed 本検定は **実施しない**: positive control validity が構造的に立たない以上、real で③不要が出ても
  「③不要 vs 検出器盲」を分離できない — そして red-team が「検出器盲は kernel 空間の構造」と確定したので、full run は
  結論を変えない (smoke で既に③勝たず + 構造で説明済)。CPU を 7.5h 投じる正当性なし。

---

## 6. honest 留保

- harness/red-team は smoke 規模 (5-12 seed)。本検定 15 seed では数値が動くが、**構造 (締めると MAP-E が先に
  starve / RR が kernel_id 直接サンプル) は seed 非依存で頑健**。
- substrate は弱い (実質 3 kernel、hopfield 機能不全)。より強い kernel 弁別 (full kernel 実装、非対角) なら
  別結論の余地は **理論上**あるが、③ の構造的障壁 (低次元選択 → RR 直接サンプル) は kernel 実装の質と独立。
- 対角 mock の限界は kernels.py スコープ宣言どおり (mechanism feasibility のみ主張、full kernel 性能非主張)。
- C-A は pre-reg behavior 逸脱の探索的 probe (JSON に `pre_reg_deviation: true`)。それでも RR 排除失敗 = N/A を強める。

---

## 7. 成果物

- substrate: `kernel_favoring_tasks.py` / `bg6_strong.py` / `bg6_strong_results.json`
- harness: `bg9_driver.py` / `bg9_smoke_results.json`
- 敵対検証: `red_team_mechanism.py` / `red_team_disprove.py` / `red_team_boundary.py` + JSON
- 事前登録: `BG9_PREREGISTRATION.md`

*UTF-8 / py -3.11 / src 無改変 / git は orchestrator 一括。*
