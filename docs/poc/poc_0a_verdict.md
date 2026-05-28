# PoC 0a Verdict — state update 数式遺伝子 (v1 → v2 redesign)

調査日: 2026-05-29  
ファイル: `scripts/poc_0a_state_update_gene.py` + `src/llcore/state_update/genes.py`  
Test: `tests/unit/test_poc_0a_state_update.py`

---

## v1 失敗 + reviewer verdict + v2 redesign の経緯 (記録)

### v1 数式 (失敗)
```
state[t+1] = decay * state[t] + mix * x[t] * tanh(gate_str * state[t])
```
- 5 ゲート G1-G5 形式 PASS だが **state が永久に 0** (zero attractor)
- 原因: gate 引数が state のみ → 初期 state=0 で `tanh(0)=0` → 永久 0

### 2 Reviewer 独立 verdict (2026-05-29)

| reviewer | 単純バグか? | 推奨数式 | 追加ゲート | 他の盲点 |
|---|---|---|---|---|
| **gem-critic** (Claude adversarial) | **設計問題** | Mamba-style or RWKV-style | G6/G7/G8 | clip 範囲 (mix 非負) / 3 パラ妥当性 / 決定論強すぎ |
| **gpt-5.4** (Codex, ChatGPT account) | **設計問題** (representation + falsification 両方甘い) | **RWKV-style が最 defensible** (parameter identifiability) | G6/G7/G8 + **G9 zero-state escape + G10 parameter sensitivity** | clip 範囲 (mix/gate_str 負許容) / K=100 と「K=f(decay,gate_str)」の不整合 / RNN-like 定義不明 |

両者共通: 単純バグでなく **設計問題**。**Fix A (Claude 当初推奨) は不採用**。

### v2 採択
- **数式**: RWKV-style leak integrator (両 reviewer 推奨に含まれる、gpt-5.4 が最 defensible 認定)
  ```
  state[t+1] = decay * state[t] + (1 - decay) * tanh(mix * x[t] + gate_str * state[t])
  ```
- **clip 範囲拡張** (reviewer 指摘):
  - decay ∈ [0, 1] (memory timescale)
  - mix ∈ **[-1, 1]** (負入力許容)
  - gate_str ∈ **[-2, 2]** (抑制性 recurrent 許容)
- **追加ゲート G6-G10** (両 reviewer 指摘の合計)

---

## falsifiable 命題 (v2)

> decay/mix/gate_str の 3 パラメータで RNN-like leak integrator + recurrent
> nonlinear coupling を表現でき、入力長 L=256, dim=8 の有界入力に対し
> (a) state が NaN/Inf にならず、
> (b) state_norm が K=10 * input_norm 以下で抑えられ、
> (c) 非ゼロ入力で state が非自明 (variance > 0) に動き、
> (d) 異なる入力列で state 軌跡が区別できる。

## 破綻ゲート (PASS/FAIL)

- [x] G1: 単一 step finite
- [x] G2: L=256 bounded norm (K=10)
- [x] G3: determinism (seed=42)
- [x] G4: degenerate values (decay=0/1, mix=0/-, gate_str=0/- 全 finite)
- [x] G5: random population N=20 (v2 拡張)
- [x] **G6: 非自明性** (mean > 0.01·input_norm かつ var > 1e-6)
- [x] **G7: 入力区別性** (異なる入力列で rel_dist > 0.1)
- [x] **G8: 記憶持続性** (decay=0.95, zero-input phase で norm 維持)
- [x] **G9: zero-state escape** (state=0 初期で 5 step 以内に norm > 1e-3)
- [x] **G10: parameter sensitivity** (各 gene perturbation で dist > 0.01)

## 実行結果 (2026-05-29)

```
PoC 0a v2 verdict: PASS — RWKV-style state update gene は falsifiable
                   命題 (有界 ∧ 非自明 ∧ 情報伝達 ∧ 記憶) を全て満たす.
```

| ゲート | 結果 |
|---|---|
| G2 max_state_norm | 0.216 (vs input_norm 1.583, K=10) |
| G6 mean_norm / var | 0.4171 / 7.37e-03 (threshold 0.016 / 1e-6) |
| G7 rel_distance | 0.629 > 0.1 |
| G8 norm[100]/norm[50] | 0.282 > 0.01 |
| G9 escape step | 1 (within 5) |
| G10 sensitivity dist | 0.16 - 0.56 |

## 実行方法

```powershell
cd D:/projects/llcore
py -3.11 -m pip install -e .[dev]
py -3.11 scripts/poc_0a_state_update_gene.py
py -3.11 -m pytest tests/unit/test_poc_0a_state_update.py -v
```

## 次段 (PoC 0b)

→ 合成 sequence fitness (copy / addition task) を `scripts/poc_0b_synthetic_fitness.py` に
   - StateUpdateGene + run_sequence で生成した state 軌跡から readout
   - copy task: 入力 sequence の遅延再現
   - addition task: 数値累積
   - fitness の決定論性 + 非 degenerate + 範囲合理性を G1-G4 で検証

## honest 留保 (v2)

- これは「数式表現が動く」レベルの mechanism feasibility のみ
- 実 task fitness は PoC 0b、進化は PoC 0c (自前 minimal GA、llive 非依存)
- 入力範囲は人為的 (input ∈ [-1, 1])、実 LLM scale ではない
- K=10 は **緩い実装上界**。RWKV-style convex combination の理論上界 sqrt(dim) より緩い
- **K=f(decay, gate_str)** という当初命題と固定 K の不整合は gpt-5.4 reviewer 指摘済、PoC 0b 以降で再検討
- G6 の variance threshold 1e-6 / G8 の ratio 0.01 は経験則、Stage 1 で再較正
- Codex (gpt-5.4) と Claude の相互 review ルールは **本 PoC から定常運用**
  (memory `feedback_codex_pair_review_for_llcore` 参照)

## 関連 memory

- [[project_core_evolution_survey_2026_05_28]]
- [[feedback_benchmark_honest_disclosure]] — 異常に良い結果は内訳を疑う (v1 zero attractor 検出)
- [[feedback_external_ai_verify]] — 外部 AI finding は実コード検証
- [[reference_codex_two_pillar]] — codex CLI default を gpt-5.4 に修復済 (2026-05-29)
