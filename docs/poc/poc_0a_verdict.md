# PoC 0a Verdict — state update 数式遺伝子

調査日: 2026-05-29  
ファイル: `scripts/poc_0a_state_update_gene.py` + `src/llcore/state_update/genes.py`  
Test: `tests/unit/test_poc_0a_state_update.py`

## falsifiable 命題

> decay/mix/gate_str の 3 パラメータで RNN-like state update を表現でき、
> 入力長 L=256, dim=8 の有界入力に対し state が NaN/Inf にならず、
> state_norm が `K * input_norm` で抑えられる (K は decay/gate_str の関数)。

## 破綻ゲート (PASS/FAIL)

- [ ] G1: 単一 step で NaN/Inf 混入なし
- [ ] G2: L=256 sequence で state norm が有界 (norm < 100 * input_norm)
- [ ] G3: 同 gene / seed で run 2 回の結果が完全一致 (決定論性)
- [ ] G4: 極端値 (decay=0/mix=0/gate_str=0 等) で degenerate せず
- [ ] G5: random 個体集団 N=5 全てが G1-G2 をパス

## 実行方法

```powershell
cd D:/projects/llcore
py -3.11 -m pip install -e .[dev]
py -3.11 scripts/poc_0a_state_update_gene.py
py -3.11 -m pytest tests/unit/test_poc_0a_state_update.py -v
```

## 結果 (要記入)

実行日:  
verdict: PASS / FAIL  
詳細:  
- G1: ?  
- G2: ?  
- G3: ?  
- G4: ?  
- G5: ?  

## 次段 (G1-G5 全 PASS なら)

→ PoC 0b: 合成 sequence fitness (copy / addition task)

## 破綻時の対応 (G1-G5 のいずれか FAIL)

- gene 表現 (decay/mix/gate_str) を見直す
- 数値範囲・clip の閾値を再設計
- gate 関数を tanh から別関数に変更検討
- PoC 0b に進まず本 PoC を再設計

## honest 留保

- これは「数式表現が動く」レベルの mechanism feasibility のみ
- 実 task fitness は PoC 0b、進化は PoC 0c
- 数値範囲は人為的 (input ∈ [-1, 1])、実 LLM scale ではない
- K=100 は緩い上界。理論的に decay/gate_str から狭く計算可能だが PoC は緩く
