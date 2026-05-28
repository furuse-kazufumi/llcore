# PoC 1a Verdict — Z3 verifier state_norm 有界 invariant (Stage 1a)

調査日: 2026-05-29  
ファイル: `scripts/poc_1a_z3_invariant.py` + `src/llcore/verifier/invariants.py`  
Test: `tests/unit/test_poc_1a_z3_invariant.py`

## falsifiable 命題

> llcore 自前 Z3 verifier が
> (a) z3-solver を import / 動作確認、(b) clip 範囲全体で |state|<=1 を symbolic 証明 (unsat)、
> (c) illegal range で反例検出 (sat)、(d) 単一 gene verify_gene_safe 機能、
> (e) online gate reject 率測定、(f) timeout < 1 sec、(g) 決定論性、
> (h) 厳しい state_bound で reject 出る (filtering 機能)。

## 破綻ゲート (G1-G8, v2 拡張)

- [x] G1: z3-solver 利用可能
- [x] G2: clip 範囲下 unsat (invariant proof)
- [x] G3: illegal decay=2 sat (sound 反例検出)
- [x] G4: 5 random gene 全 admit
- [x] G5: 30 random gene reject 率 0% (clip 範囲下では期待値)
- [x] G6: **elapsed=5.8ms** (< 1 sec)
- [x] G7: 決定論性
- [x] **G8: extended state_bound=0.3 で sat (filtering 機能実証)** ← Codex 指摘で追加

## v1 → v2 修正 (Codex pair-review 反映)

### v1 bug (Codex 発見)
`verify_gene_safe` で `tanh_val` を free in [-1, 1] にしていたため、**mix と gate_str が
式に効いていなかった** = gene-specific reasoning でなく decay のみで判定。

### v2 fix
tanh の引数 `pre = mix*x + gate_str*s` を明示計算し、tighter sound bound:
- `tanh_val^2 <= pre^2` (tanh は 1-Lipschitz, tanh(0)=0)
- `|tanh_val| <= 1`
- `tanh_val * pre >= 0` (符号一致, 奇関数性)

これで verify_gene_safe が真に gene-specific になる。

### G8 追加 (Codex 指摘)
filtering 機能の実証として `state_bound=0.3` で全域検査 → sat (反例検出) を要求。
verifier が「常に admit」でなく「条件次第で reject する」ことを機械的に示す。

## 実行結果

```
PoC 1a verdict: PASS — Z3 verifier で clip 範囲下の有界性を symbolic 証明.
                 online gate として実用可、決定論性 + timeout OK.
                 次段 Stage 2a (factor_hook × RWKV mock) に進めます.
```

| ゲート | 結果 |
|---|---|
| G2 | unsat: \|state\|<=1.0 holds for all clipped gene with \|x\|<=1.0 |
| G3 | illegal decay=2 sat (sound 確認) |
| G6 | **5.8ms** (極めて高速、online gate 実用可) |
| G8 | state_bound=0.3 で sat counterexample あり = filtering 機能 |

## Codex pair-review verdict

### Green-light (条件付き Yes)
- "PoC 1a: clip 範囲下で `|state|<=1` を機械的確認" は通してよい
- v1 で発見された **verify_gene_safe の gene-non-specific bug は v2 で修正済**
- 主張は **proof-oriented** に絞るのが honest:
  - "UNSAT proof" の主張は妥当
  - "online gate" の主張は **G8 追加で実用例示**

### honest 留保 (Codex 指摘で記録)
1. **論文化への距離**: 1-step scalar invariant の over-approx proof 段階
2. **不足**:
   - 実 NN 意味論との対応付け (現状 scalar 数式)
   - 近似導入時の soundness theorem (現状 informal)
   - 多次元・多 step・層構造への一般化
   - baseline verifier との比較
   - **Marabou Incremental の "異なる構造の refinement relation" の sound 拡張** (本研究核独自軸)
3. これらは Stage 後期 / 論文執筆段階で対応

## 設計判断

### tanh 近似 (sound over-approx)
全域証明 (`verify_state_norm_invariant`):
- `tanh_val ∈ [-1, 1]` の free 変数 = 上界保守的だが sound
- "全 gene + 全状態 + 全 tanh 値" の最悪ケースで invariant 保証

単一 gene 検査 (`verify_gene_safe`, v2 fix):
- tighter bound: `tanh_val^2 <= pre^2 ∧ |tanh_val| <= 1 ∧ tanh_val * pre >= 0`
- gene-specific な mix/gate_str の効果が式に乗る

### Lyapunov-style → Z3 化の意義
数式証明 (人間にとって明白) を Z3 化することで:
1. 仮定が API/コードに正しく落ちているか保証
2. 実装が証明したい命題を取り違えていないか機械検査
3. 将来の式変更で性質が壊れた時に自動検出

→ **進化ループで gene が clip 範囲を逸脱しないことを毎世代 Z3 で確認可能** = llcore 独自軸の動作実証。

## 実行方法

```powershell
cd D:/projects/llcore
py -3.11 -m pip install -e .[z3]
py -3.11 scripts/poc_1a_z3_invariant.py
py -3.11 -m pytest tests/unit/test_poc_1a_z3_invariant.py -v
```

## 次段 (Stage 2a)

→ factor_hook × RWKV mock 接続
   - llive `factor_hook.py` の Protocol を llcore 側で受け取る薄い consumer
   - 10 思考因子 → state update 係数の動的調整
   - mock 環境 (実 RWKV weight なし) で接続検証

## 将来 (post-llcore-完成)

- 多次元 state への Z3 拡張 (現在 scalar)
- 多 step invariant (現在 1-step)
- Marabou Incremental NN Verification との bridge
- PrediPrune (ML pruning) と Quokka (LLM invariant synthesis) との統合

## 関連 memory

- [[project_llcore_init_2026_05_29]]
- [[feedback_codex_pair_review_for_llcore]]
- [[project_core_evolution_survey_2026_05_28]]
