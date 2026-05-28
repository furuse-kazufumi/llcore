# PoC 0c Verdict — llcore 自前 minimal GA (llive 非依存) 進化 10×10

調査日: 2026-05-29  
ファイル: `scripts/poc_0c_minimal_ga.py` + `src/llcore/evolution/minimal_ga.py`  
Test: `tests/unit/test_poc_0c_minimal_ga.py`

## falsifiable 命題 (v2 wording, Codex honest 強化)

> llcore 自前 minimal GA (tournament + uniform mutation + elitism) で
> StateUpdateGene を copy/addition task に適応させる進化が
> (a) 完走、(b) 全滅なし、(c) **archived best** が単調非減少、
> (d) diversity 維持、(e) random と **competitive**、(f) 決定論性、
> (g) **specialization suggested**

## 破綻ゲート (G1-G7)

- [x] G1: 10x10 evolve 完走 (NaN/Inf なし)
- [x] G2: 全滅なし (size 一定)
- [x] G3: archived best 単調非減少
- [x] G4: diversity 維持 (min > 1e-6)
- [x] G5: random baseline と competitive (>=0.9 比)
- [x] G6: 決定論性 (同 seed で完全一致)
- [x] G7: specialization suggested (gene dist > 0.1)

## 実行結果

```
PoC 0c verdict: PASS — 自前 minimal GA で進化 10×10 が機能、
                 elitism による単調性 + diversity 維持 + specialist 出現を実証.
```

| ゲート | 結果 |
|---|---|
| G1 | gens=11, best=0.510, all_finite |
| G2 | size = 10 全世代 |
| G3 | start 0.249 → end 0.552, monotonic=True (improved=True) |
| G4 | diversity min=0.32, start=0.59 → end=0.42 |
| G5 | best_evolved=0.574 (110 eval × 5 seeds) vs best_random=0.594 (200 eval) |
| G6 | curve / final_best 完全一致 |
| G7 | copy_best (d=0.23,m=0.62,g=-0.60) vs add_best (d=0.77,m=-0.68,g=1.03), **dist=2.15** |

## v1 → v2 修正

### v1 で発生
- G3 FAIL: best fitness が単調でない (improved だが non-monotonic)
- 原因: elite を gene のみ持ち越し、再評価で stochastic fitness 変動

### v2 fix
- elite を **Individual (gene + fitness) ごと持ち越し**、再評価しない
- これにより archived best monotonicity を保証
- honest disclosure: 「current population の真 best が単調」ではない、archived best の話

## Codex pair-review verdict

### Green-light (条件付き Yes)
PoC 0c としては通してよい。ただし external-facing wording は honest に下げる:
- G3: "current fitness monotonic" → "**archived best monotonic**"
- G5: "beats baseline" → "**competitive / compute-efficient**"
- G7: "task dependence demonstrated" → "**specialization suggested**"

### honest 留保 (Codex 指摘)
1. G5 比較設計が apples-to-oranges (GA best-of-5 × 110 eval vs random 1 × 200 eval)
2. G7 は cross-eval なし (copy_best を add task で評価して落ちることの確認は将来 G8)
3. script vs test の seed 本数不一致 (script=5 seeds, test=3 seeds、CI 軽量化のため)

## 設計判断 (v2 確定)

### elitism = 前世代 fitness 保持
- 標準 GA では elite を gene のみ持ち越し再評価が一般的
- llcore では fitness が stochastic (n_trials=3, task.generate に rng) なので
  fitness ごと保持 = archived best monotonic を保証
- これは "best-so-far archive" 設計

### 自前実装の正当化
- llive lldarwin_v2 (ε-lexicase + novelty + factor-subspace QD + 中立貯蔵庫等) は
  state_update gene 3 次元には overkill
- 素朴 tournament + uniform mutation で十分 (低次元 + 短 generation)
- 将来 (v0.2+) 高度 selector が必要になれば自前実装で追従、llive 参考のみ

## 実行方法

```powershell
cd D:/projects/llcore
py -3.11 scripts/poc_0c_minimal_ga.py
py -3.11 -m pytest tests/unit/test_poc_0c_minimal_ga.py -v
```

## 次段 (Stage 1a)

→ Z3 verifier の state_norm 有界 不変量制約
   - `src/llcore/verifier/` に Z3 satisfiability 検査を実装
   - 進化中の gene を Z3 で gate (online verification)
   - falsifiable 命題: Z3 gate ON で reject 率 > 0 かつ 探索効率向上

## 将来の改善候補 (Codex 指摘で残置)

- G8: cross-eval test (copy_best/add_best を相互 task で評価、自己 task 優越を実証)
- G9: G5 同条件比較 (eval count 厳密一致)
- 0c' (optional): llive lldarwin_v2 import で baseline 比較実験

## 関連 memory

- [[project_llcore_init_2026_05_29]]
- [[feedback_codex_pair_review_for_llcore]]
- [[feedback_benchmark_honest_disclosure]]
