# R-endo — 内的検証器 (endogenous verifier) PoC VERDICT (2026-06-07)

ユーザー提案「検証器を llcore 自身が持ってはどうか」(entity が自分の sound verifier を持ち、自律的に
自己判断する) の価値を、現構造を壊さない additive PoC + 事前登録 A/B で検証した決着。

正本コード: `run_d_internal_ab.py` (事前登録 commit 2bb5883 = 結果取得前) /
`src/llcore/state_update/genes.py::StateUpdateGene.is_verified_trajectory_tube` (read-only method) /
`tests/unit/test_internal_self_verify.py` (回帰テスト) / `results_d_internal_ab.json` (生結果, wall 156s)。

## 結論一行

**内的検証器は feasible かつ sound だが、本基質 (有界 CopyTask) では autonomy 上の優位を生まない
= location shift にすぎない。価値判定 = CONDITIONAL (advisory-only park)。** 内的化が効くのは
「環境変化が viability を脅かす (発散しうる) 基質」に限る、という方向を指す honest negative。

## 何を実装したか (additive 規律)

`StateUpdateGene` に read-only メソッド 1 個を追加しただけ:
```python
def is_verified_trajectory_tube(self, w_bar, r_max=None) -> bool:
    from llcore.verifier.tracking_tube import tracking_tube
    return bool(tracking_tube(self, w_bar=w_bar, r_max=r_max).admits)
```
- production `evolve()` / `_gate_admits` / verifier モジュールは **一切 untouched**。
- 既存 111 tests + 新規 6 tests green。method は frozen dataclass の read-only。
- local import で循環依存 (verifier → state_update) を回避。コスト = 閉形式 O(n²) ~100µs。

## 実験設計 (環境ステップ下の再適応 A/B)

内的化 ≡ **環境結合適応 gating** (location shift 自体は無価値という honest 前提)。`DisturbedCopyTask`
(delay=8, seq_len=32) の外乱 w_env を mid-run でステップ変化させ、再適応を測る:
- phase 1 (10 世代): w_env = W_LOW=0.05。 phase 2 (10 世代): w_env = W_HIGH=0.20 (環境悪化)。
  `evolve` の `initial_pop` で集団継続 + 同一 rng (segmented evolution)。
- **NONE**: gate なし。 **EXO_fixed**: gate w̄=W_LOW 固定 (外部設計時 gate, 環境に盲目)。
  **ENDO**: gate w̄=現 w_env (entity が現環境に結合し W_HIGH に自己再 gate)。
- phase 1 では ENDO=EXO_fixed (同 w̄) ので分岐は phase 2 のみ = 効果を環境変化に isolate。
- n=20 seeds (3000-3019)。GA_KW/R_MAX/STATE_DIM は run_3arm_ab から import (Phase 2a 同一構成)。

## 事前登録仮説の判定

| 仮説 | 内容 | 結果 | 判定 |
|---|---|---|---|
| **H3** (safety/correctness) | entity self-verdict == 外部 gate verdict | random 5000 gene で **0 disagreement** | **PASS** — 内的判定は sound・外部 gate と構成的一致 |
| **H1** (non-inferiority) | phase2 final test (W_HIGH) ENDO ≥ EXO_fixed (margin −δ=−0.0067) | Δ=+0.0060, p=0.205, +12/−8 | **PASS (非劣)** — fitness tax なし |
| **H2** (autonomy 本丸) | phase2 再適応 AUC で ENDO > EXO_fixed | Δ=**−0.0401**, p=0.666, +9/−11 | **NULL** — ENDO は速くない (むしろ僅かに遅い, n.s.) |

補助 (exploratory):
- **diversity**: ENDO−EXO Δ=−0.069 (p=0.223) — ENDO は env-coupled tight 再 gate (rej 116 vs EXO 24)
  で集団をやや狭める (final diversity 0.439 vs 0.508 vs NONE 0.602)。有意でないが方向はコスト側。
- **gate の fitness 無効性**: NONE−EXO Δ=+0.0050 (p=0.212) — 無 gate が EXO とほぼ同等 (むしろ僅か上)。
  = この有界基質では gate 自体が fitness に効かない (M3 NEGATIVE / §11 と整合)。fallback 全 arm 0。

## なぜ NULL か (機構, honest)

1. **基質が有界**: CopyTask は `decay·s + (1−decay)·tanh(...)` の convex combination で state が構造的に
   有界。環境悪化 (外乱 ↑) は fitness を下げるが **viability を脅かさない (発散しない)**。よって内的 gate が
   防ぐべき破綻が存在せず、「entity が自分を守る」自己検証に仕事がない。
2. **外乱除去↔保持のトレードオフ** (§11 と同根): ENDO が W_HIGH に tight 再 gate すると tube
   r=G·w̄/(1−L)≤r_max が小 L (速い忘却) を要求 → memory 保持を犠牲。disturbance-bound を締めても
   memory タスクの fitness は買えない (§11 の w̄ NEGATIVE が予測した通り)。
3. **M3 の含意**: smooth/有界 地形では「誰が・どこで選択するか」(location/density) でなく「何を選択するか」
   が効く。判定主体を GA→gene に移す location shift は dynamics を変えない。

## 価値判定と先行研究 (over-claim 排除)

- **概念は Gödel Machine (Schmidhuber 2003) の rediscovery**。「証明してから自己改変」「entity が
  proof checker を内蔵」は既出 (論文も SS-GM/SGM を cite)。「最初の自己進化検証器」「自律エージェントが
  欺瞞しない」は over-claim。
- **本 PoC が demonstrable に確立したのは (a) feasibility (entity が sound verifier を持てる) と
  (b) soundness/correctness (H3: self-verdict は外部 gate と 0 乖離・gameable でない) のみ**。
  「自律の優位」(H2) は本基質では確立できなかった。
- 真の差別化軸 (certificate が箱上多項式不等式=捏造不能、DGM/SEAL の性能 gate と質的に違う) は H3 で
  裏付くが、それは「内的化」でなく「健全 certificate を gate にする」という既存 llcore 貢献の再確認。

## 次の方向 (この negative が指す先)

内的化が autonomy 優位を生む条件 = **環境変化が viability を脅かす (発散しうる) 基質**。
llcore には既に該当例がある: **HD-1 GPU 実験 (research/highdim_evolution/) で ungated gradient は
contractive region を出る (19/20 seeds, ρ→1.95@n=256)**。このような「放置すると発散する」基質でこそ、
entity が現環境に結合して自己 gate する内的化が、外部固定 gate の見逃す破綻を防ぎうる。
→ R-endo を park し、内的化の再評価は **発散しうる基質 (HD-1 系 / 非有界 recurrence)** で行うのが筋。
本 PoC の method・runner・回帰テストはその時の踏み石として保持。

## 言ってよいこと / いけないこと

言ってよい: 内的化は **additive に実装可能・sound (H3=0)・fitness tax なし (H1 非劣)**。
言ってはいけない: 内的化が **自律適応を速める** (H2 NULL)。**多様性を上げる** (むしろ僅かに下げる)。
**本基質で外部 gate より優位** (location shift にすぎない)。
honest 留保: scalar gene / n≤8 / 有界 CopyTask / 単一環境ステップのスコープ。null も
feedback_benchmark_honest_disclosure に従い削除せず教訓として残す。
