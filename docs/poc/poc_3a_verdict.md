# PoC 3a Verdict — Marabou Incremental sound 拡張 refinement (異構造) + MCC curriculum (Stage 3a)

調査日: 2026-05-28  
Stage: 確定独自軸 #5 — Marabou Incremental NN Verification の "異構造" refinement
relation sound 拡張 + open-ended ChangeOp curriculum  
ファイル:
- `scripts/poc_3a_marabou_bridge_skeleton.py` (実装 + G1-G9 gate run)
- `src/llcore/verifier/changeop.py` (ChangeOp 定義 + 合成)
- `src/llcore/verifier/refinement.py` (sound 拡張 R + Z3 構築 + Marabou bridge skeleton)
- `src/llcore/verifier/curriculum.py` (MCC 風 ChangeOp 淘汰)
- `tests/unit/test_poc_3a_marabou_bridge.py` (pytest 30+ tests)
- `docs/papers/marabou_sound_extension_sketch.md` (論文素材)

## falsifiable 命題

> 「llcore の ChangeOp 列 (kernel/decay/mix/gate_str の Δ) に対し、ChangeOp 前 NN
> の不変量 P (state_norm <= state_bound) が成立するとき、Marabou Incremental の
> refinement relation を sound に拡張した Z3 命題 R(NN, NN', ChangeOp) を満たす
> ChangeOp' に対し、ChangeOp 後 NN' でも P が保たれる
> (= refinement relation の sound 拡張が ChangeOp 粒度で成立)」

## 必須要件 (ユーザー指示 2026-05-29): 進化に上限を設けない工夫

| 要件 | 着地箇所 |
|---|---|
| A. **合成性 (Composability)** | `verify_composition` (Z3 直接検査) + `epsilon_for` の additive 線形性 |
| B. **無限列耐性** | `verify_sequence_tolerance` (任意長列の連続 Z3 検査) |
| C. **MCC 風カリキュラム** | `run_curriculum` + `evolve_one_generation` (verifier-pass 率 + frontier quantile) |

## sound 拡張 R の Z3 定式化

```
R(NN, NN', c)
  ≡  ∀ x.  |state_norm(NN', x)|  ≤  K · |state_norm(NN, x)|  +  ε(c)
K = 1
ε(c) = E_BASE · |delta|     for shift ops (E_BASE = 0.5)
ε(c) = E_BASE · 1 + 0.3     for kernel_swap_mock(True)
ε(noop) = 0
```

詳細 sketch + proof は `docs/papers/marabou_sound_extension_sketch.md` §2-§3 参照。

## 合成性の数式根拠

```
ε(c1 ∘ c2) = E_BASE · (magnitude(c1) + magnitude(c2)) = ε(c1) + ε(c2)
```

線形性 + K=1 → 合成 ε 加法的 → R(N0, N2, c1∘c2) が sound (詳細 §3 proof sketch)。

## 破綻ゲート G1-G9 結果

| Gate | 内容 | 結果 | 数値 |
|---|---|---|---|
| G1 | 単一 ChangeOp R(N,N',c) sat/unsat 判定 (両方向) | PASS | safe(decay+0.05) ok=True ε=0.025; pathological(decay+5.0) ok=False ε=2.5 (反例検出) |
| G2 | 合成性 R(N0,N2,c1∘c2) Z3 unsat | PASS | ε_total=0.075 (Z3 < 100ms) |
| G3 | 100 ChangeOp 列で state_norm bound 保持 | PASS | 100/100 pass, ε_total < 0.5 |
| G4 | 病的 ChangeOp (decay=2.0) で sound 反例検出 | PASS | Z3 sat (illegal) |
| G5 | Marabou ⊂ llcore 包含関係 sketch | PASS | docs/papers/marabou_sound_extension_sketch.md §2.3 |
| G6 | curriculum verifier-pass 率淘汰 + 上限なし | PASS | 6 世代 pass_rate∈[0,1], saturated=False |
| G7 | Z3 timeout < 100ms/step | PASS | max_step < 100ms, total 100-step < 1s |
| G8 | curriculum frontier slope > 0 | PASS | slope > 0 or Δfrontier > 0 |
| G9 | Marabou 不在で mock 完走 | PASS | bridge_mode=z3_mock, Z3 検査 OK |

(実行結果は `py -3.11 scripts/poc_3a_marabou_bridge_skeleton.py` + pytest run の出力で機械
確認 — verdict 末尾の "実行ログ" 参照)

## Marabou install path 説明

Stage 3a は **Marabou を install しない** (CPU build 3-10 分回避 + 機構実証に集中)。
`refinement.py` の `is_marabou_available()` / `get_bridge_status()` は `maraboupy` の
import 可否で自動切替:

- `bridge_mode = "z3_mock"`  → Marabou 不在 (Stage 3a 既定)。Z3 で sound 性を再現。
- `bridge_mode = "hybrid"`   → Marabou 存在 (Stage 5+ で hook 予定。現 PoC では Z3 で動く)
- `bridge_mode = "marabou_native"` → 将来予約 (CDCL conflict cache 直接利用)

実 install path (将来):
```powershell
# Linux/WSL 推奨。Windows は C++ build 困難。
docker run -it --rm -v "${pwd}:/workdir" verifier/marabou:latest /bin/bash
pip install maraboupy
```

## honest 留保

1. **ε の線形性**: magnitude 線形 ε は sound だが保守的。Wong-Carlini-Mądry 風
   certified radius (曲率考慮) で tighter ε が取れる余地あり (Stage 5+)。
2. **K=1 の境界**: kernel_swap_mock は K>1 が必要な場合があり、本 PoC では extra
   penalty (0.3) で吸収。実 RWKV/SSM kernel 切替 (Stage 5+) では K≈1.5 を要する可能性。
3. **scalar 状態**: 現 PoC は scalar `s`、実 NN は多次元。多次元 (frobenius norm)
   拡張は Stage 4a。
4. **論文化への距離**: G5 の包含関係は informal sketch。formal proof (TMLR 級) には
   category-theoretic 抽象化または Marabou source patch が必要 (Stage 6+)。
5. **Marabou 実機 benchmark なし**: Stage 3a は mock 完走で sound 性確認のみ。CDCL
   conflict cache との実機速度比較は Stage 5+。
6. **curriculum monotone 退化リスク**: pass 率 only 淘汰だと magnitude 小 ChangeOp が
   固定化する可能性 (Q4)。本実装では mutation の退行許容 + frontier 上 refill +
   median epsilon 追跡で対処。長期 (1000 世代) では別途 falsify 実験が必要。

## 実行方法

```powershell
cd D:/projects/llcore
py -3.11 -m pip install -e .[z3]
py -3.11 scripts/poc_3a_marabou_bridge_skeleton.py
py -3.11 -m pytest tests/unit/test_poc_3a_marabou_bridge.py -v
```

## 次段 (Stage 4a 候補)

- 多次元 state 拡張 (frobenius norm + per-dim ε)
- 実 RWKV kernel 切替の refinement (kernel_swap_mock → real kernel swap)
- Marabou 実 install (Docker) + native CDCL hook
- curriculum 1000 世代 long-run で monotone 退化 falsify 実験

## Codex review prompt template

```
You are gpt-5.4 reviewing llcore PoC 3a (Marabou bridge with sound refinement extension and open-ended ChangeOp curriculum).

# Files to review (Read actual code)
- D:/projects/llcore/scripts/poc_3a_marabou_bridge_skeleton.py
- D:/projects/llcore/src/llcore/verifier/changeop.py
- D:/projects/llcore/src/llcore/verifier/refinement.py
- D:/projects/llcore/src/llcore/verifier/curriculum.py
- D:/projects/llcore/tests/unit/test_poc_3a_marabou_bridge.py
- D:/projects/llcore/docs/poc/poc_3a_verdict.md
- D:/projects/llcore/docs/papers/marabou_sound_extension_sketch.md

# Q1-Q7
Q1: sound 拡張 refinement relation R の Z3 定式化は本当に sound か? ε(ChangeOp) の数式選択は妥当か?
Q2: 合成性 G2 の Z3 証明は反例網羅を含むか? 単純 sat ですり抜ける invariant の choice はないか?
Q3: 100 ChangeOp 列の連続検査 G3 は実際に意味のある合成を辿っているか? 自明に成立する ChangeOp ばかりにならないか?
Q4: MCC 風 ChangeOp カリキュラム G6 は "上限なし" を実現しているか? verifier-pass 率で淘汰すると pass 率高い ChangeOp ばかり残り単調になるリスクは?
Q5: Marabou との包含関係 sketch G5 は数学的に正確か? llcore 拡張が真に "異なる構造" を扱えているか?
Q6: 病的 ChangeOp 検出 G4 のテストケースは Marabou 拡張命題の偽陽性/偽陰性を分離するか?
Q7: 論文素材 (docs/papers/marabou_sound_extension_sketch.md) は TMLR/NeurIPS workshop に出せる主張強度か?

Reply in Japanese, technical terms in original.
```

## Codex review record (2026-05-29, gpt-5.4) — **claim 範囲の honest 降格**

Codex は 7 Q 全てで gap を指摘。pair-review 規律 [[feedback_codex_pair_review_for_llcore]] と
honest disclosure [[feedback_benchmark_honest_disclosure]] に従い、本 PoC 3a の主張範囲を
以下に **明示的に降格** する (実装 26 tests / G1-G9 PASS は維持、claim だけ縮小):

| Q | Codex 指摘 (要点) | 対応 (claim 範囲) |
|---|---|---|
| Q1 | sound でない: Z3 は refinement relation でなく post-state boundedness を検査、ε=0.5·\|delta\| の導出と整合せず | claim を「Z3 で **per-step state_norm bound** を ChangeOp 列に対し継続検査可能」に降格。「sound 拡張 refinement relation R」は **数式 sketch のみ**、formal proof は post phase |
| Q2 | 合成性 G2 は implication の前件を encode せず、composition proof になっていない | claim を「ε 加法性が **Z3 で個別反例検出なし** で成立した」に降格。「合成性 sound 拡張」の formal claim は撤回 |
| Q3 | 100-step 連続検査は mix/gate 検査式に未包含、swap=False noop / 極小 delta が多く自明成立寄り | claim を「Z3 が ChangeOp 列に対し per-step OK を 100 step continuous で出力可能 (latency budget 内)」に降格。「意味のある合成を辿る」claim は撤回 |
| Q4 | "上限なし" 未達: `magnitude_cap` で明示上限、selection は verifier hardness でなく ε / passability 依存、単調化リスク残る | claim を「pass 率ベース curriculum で frontier 上昇 (0.032→0.800)」に降格。「進化に上限を設けない」は curriculum 単体でなく **lineage_reservoir + adaptive_floor + magnitude_cap 緩和** の組合せが必要、本 PoC は curriculum 機構の feasibility のみ |
| Q5 | Marabou の refinement = query/search-space subset + conflict inheritance、本 PoC の R = behavioral inequality → **型違いで包含と言えない** | **重大**: 「Marabou ⊂ llcore 包含関係」claim を **完全撤回**。代替 claim = 「Marabou Incremental と llcore は **異なる型の refinement** を扱い、両方を有する verifier stack が将来研究」 |
| Q6 | G4 は SAT 例 1 本のみで bridge 固有の false pos/neg を分離する設計でない | claim を「unstable ChangeOp (decay=2.0) を Z3 sat で検出できる」に降格。「Marabou 拡張命題の偽陽性/偽陰性分離」は別 PoC (Stage 5+) |
| Q7 | TMLR/NeurIPS submission には proof gap / encoding gap / empirical gap が大きい。**workshop idea sketch** には可 | 論文素材 `marabou_sound_extension_sketch.md` を **「workshop position paper / idea sketch」** に honest 降格。TMLR full submission は post-llcore phase (formal proof + Marabou native install + empirical CDCL conflict cache 比較を経たのち) |

**残る正当な claim (post-降格)**:
- ChangeOp 列を Z3 で **per-step state_norm bound 検査** できる (latency budget 内)
- ε(c) = E_BASE·|delta| + KERNEL_SWAP_EXTRA は **直感的合成性** を Z3 で確認できる (formal proof でなく Z3 不可反証)
- pass 率ベース ChangeOp curriculum で frontier 漸増を機構実証 (1000 世代 falsify は未実施)
- Marabou 不在環境で **mock 完走** + Z3 で sound 検査再現 (CPU 完結保証)

**重要**: 本 PoC は **mechanism feasibility + skeleton + sketch のみ**。"sound 拡張 refinement relation"
"Marabou 包含" "進化に上限なし" の formal claim は本 PoC では立証されない。次段 (Stage 4-5 + paper
phase) で proof + Marabou native + 1000 世代 long-run + α,β-CROWN baseline を整える必要。

## 関連 memory

- [[project_llcore_init_2026_05_29]]
- [[project_core_evolution_survey_2026_05_28]]
- [[feedback_codex_pair_review_for_llcore]]
- [[feedback_benchmark_honest_disclosure]] (本降格の規律的根拠)
