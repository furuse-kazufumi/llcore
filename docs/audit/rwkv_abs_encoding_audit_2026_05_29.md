# llcore RWKV side Z3 abs encoding audit (2026-05-29)

**動機**: research/other_archs/neural_ode/ Stage 2.3 (Codex review) で発覚した
**Z3 abs encoding バグ**が llcore 本流 (Stage 0-3 PoC 1a/2a/3a の verifier) にも
存在しないか体系的に audit する.

## Neural ODE で発覚したバグ pattern (Codex finding)

```python
# 旧実装 (unsound)
abs_A = z3.Real("abs_A")
solver.add(abs_A >= A)
solver.add(abs_A >= -A)
# ↑ これは |A| <= abs_A (下界) しか保証せず、abs_A = |A| (等式) を保証しない
# → abs_A が真の |A| より大きい値を取る Z3 model が許容され、偽 sat が出る可能性
```

正しい encoding:
```python
abs_A = z3.If(A >= 0, A, -A)  # 式として真の |A| を表現
```

詳細: `D:/projects/llcore/research/other_archs/neural_ode/verdict.md` Codex review record.

## Audit 対象

llcore 本流の Z3 制約を扱う全ファイル:

| ファイル | scope |
|---|---|
| `src/llcore/verifier/invariants.py` | PoC 1a state_norm invariant (RWKV-style) |
| `src/llcore/verifier/refinement.py` | PoC 3a refinement relation R + composition |

(ChangeOp / curriculum モジュールは Z3 不使用、本 audit 対象外)

## Audit 手法

1. Grep で `abs|Abs|>= -|<= -` pattern を全 verifier ファイルから抽出
2. `z3.Solver` / `z3.Real` / `solver.add` の使用箇所を全 enumerate
3. **`abs_X >= X ∧ abs_X >= -X` pattern** (= Neural ODE bug) の検出
4. `|X| <= bound` 条件が **box 形式 (`X >= -bound ∧ X <= bound`)** で表現されているか確認

## 結果: **CLEAN** (Neural ODE bug pattern 不在)

### invariants.py (PoC 1a state_norm)

| 制約 | encoding | 評価 |
|---|---|---|
| `decay ∈ [0,1]` | `solver.add(decay >= 0, decay <= 1)` | box ✓ |
| `mix ∈ [-1,1]` | `solver.add(mix >= -1, mix <= 1)` | box ✓ |
| `gate_str ∈ [-2,2]` | `solver.add(gate_str >= -2, gate_str <= 2)` | box ✓ |
| `|s| <= state_bound` | `solver.add(s >= -state_bound, s <= state_bound)` | **box ✓** (Neural ODE bug pattern 不使用) |
| `|x| <= max_input_abs` | `solver.add(x >= -max_input_abs, x <= max_input_abs)` | **box ✓** |
| `|tanh(z)| <= 1` 近似 | `solver.add(tanh_val >= -1, tanh_val <= 1)` | **box sound 近似** (tanh の上界保守) |
| violation 探索 | `solver.add(z3.Or(s_next > state_bound, s_next < -state_bound))` | box ✓ |

**結論 (invariants.py)**: 全 Z3 制約が **box 形式 (`X >= -bound ∧ X <= bound`)** で
encoding されており、補助変数 `abs_X` を導入する pattern が存在しない. Neural ODE で
発覚した「補助変数下界のみ」bug は **不在**.

### refinement.py (PoC 3a sound 拡張)

| 制約 | encoding | 評価 |
|---|---|---|
| `|s| <= bound` | `solver.add(s >= -bound, s <= bound)` | box ✓ |
| `|x| <= max_input_abs` | `solver.add(x >= -max_input_abs, x <= max_input_abs)` | box ✓ |
| `tanh_after, tanh1, tanh2 ∈ [-1,1]` | 各 `solver.add(tanh_X >= -1, tanh_X <= 1)` | box sound 近似 ✓ |
| violation (state_bound 超過) | `solver.add(z3.Or(s_next > threshold, s_next < -threshold))` | box ✓ |
| composition: `s2 > threshold ∨ s2 < -threshold` | `solver.add(z3.Or(s2 > threshold, s2 < -threshold))` | box ✓ |

**結論 (refinement.py)**: 同じく全 box 形式、abs encoding bug pattern 不在.

## 横断的考察

llcore RWKV side では `|X|` 条件を **常に `X ∈ [-bound, bound]` の box 形式**で表現
しており、Neural ODE で発覚した「補助変数 abs encoding が下界のみ保証」bug は
構造的に発生していない. これは:

1. **設計の sound 性**: PoC 0a/1a/2a/3a 開発時に意識的か無意識的か box 形式を採用
2. **tanh の上界保守**: tanh は `|tanh(z)| <= 1` の box で抑える保守的 sound 近似で
   `z3.If` 不要 (もし strict equality `tanh_val = tanh(z)` を試みれば Z3 で扱えない
   transcendental → Z3 が解けない、box 近似は妥当)
3. **Neural ODE bug の特殊性**: research/other_archs/neural_ode/ode_verifier.py は
   Lipschitz 上界 `|A| + |W|*|b|` を Z3 で構築する際に `Abs` 補助変数を試したため
   bug 発覚 (RWKV では Lipschitz でなく state_norm 直接 box bound のため不要)

## 結論

llcore RWKV 本流 verifier (`src/llcore/verifier/`) は **Neural ODE で発覚した abs
encoding bug pattern を含まない**. 全 Z3 制約が box 形式 (`X ∈ [-bound, bound]`)
で encoding されており構造的に sound.

Stage 2.1 の honest 訂正規律 ([[feedback_codex_pair_review_for_llcore]]) に従い、
本 audit は read-only 検査として記録. 修正は不要.

## 関連 commit / memory

- `e6d91ab` research(neural_ode): Z3 abs encoding バグ発見 + 修正
- `837d335` fix(snn-stage-2.1): honest 訂正 (Edit log success ≠ file 反映)
- [[project_llcore_init_2026_05_29]] — Stage 0-3 完成宣言 + research phase + audit 結果
- [[feedback_codex_pair_review_for_llcore]] — pair-review 規律
- [[feedback_external_ai_verify]] — codex finding を実コード検証 (本 audit で実践)

## audit 完了日

2026-05-29
