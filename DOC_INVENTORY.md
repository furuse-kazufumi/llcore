# 📚 ドキュメント目録 — llcore

> 自動生成 (`py -3.11 D:\tools\gen_doc_inventory.py <repo>`)。ファイル追加後に再実行で更新。
> **公開/内部フラグはヒューリスティックの仮判定**。公開前に必ず人手で確認すること。

- 総ドキュメント数: **31** （🌐 公開候補 2 / 🔒 内部? 20 / ❓ 要判断 9）
- コーパス・依存・仮想環境・.git は除外。

## 目次

- [(ルート)](#g0) (1)
- [docs](#g1) (2)
- [docs/audit](#g2) (1)
- [docs/design](#g3) (1)
- [docs/eval](#g4) (1)
- [docs/papers](#g5) (4)
- [docs/poc](#g6) (14)
- [research/other_archs](#g7) (2)
- [research/other_archs/gnn](#g8) (1)
- [research/other_archs/gnn/dynamic_graph](#g9) (1)
- [research/other_archs/neural_ode](#g10) (1)
- [research/other_archs/snn](#g11) (1)
- [research/other_archs/snn/izhikevich](#g12) (1)

<a id="g0"></a>

## (ルート) (1)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [README.md](README.md) | llcore — Verified Neural Architecture Evolution on CPU | Transformer のコアアルゴリズム (state update / 学習則 / 認知駆動 Δ) に進化形態を与え、Z3 verifier で破綻させずに 異アルゴリズムへ進化させる研究フレームワーク。CPU 完結。 | 2026-05-28 | 🌐 公開候補 |

<a id="g1"></a>

## docs (2)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [ARCHITECTURE_LANDSCAPE.md](docs/ARCHITECTURE_LANDSCAPE.md) | llcore Architecture Landscape — アーキ体系俯瞰 | llcore は Transformer のコアアルゴリズム (state update / 学習則 / 認知駆動 Δ) に進化形態を与え、Z3 verifier で破綻させずに異アルゴリズムへ進化させる研究フレームワーク。CPU 完結。「同じ design pattern (gene 化 + Z3 invariant gate + 進化 + open-ended 機構) が複数アーキで成立するか | 2026-05-29 | ❓ 要判断 |
| [SESSION_SUMMARY.md](docs/SESSION_SUMMARY.md) | Session Summary (auto-generated) | f8c5fac feat(step6): 実substrate proxy(ESN×実テキスト)は欺瞞的でない — ③/GPU投資の根拠は弱い + 用語集 | 2026-05-30 | 🔒 内部? |

<a id="g2"></a>

## docs/audit (1)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [rwkv_abs_encoding_audit_2026_05_29.md](docs/audit/rwkv_abs_encoding_audit_2026_05_29.md) | llcore RWKV side Z3 abs encoding audit (2026-05-29) | Z3 abs encoding バグが llcore 本流 (Stage 0-3 PoC 1a/2a/3a の verifier) にも | 2026-05-29 | 🔒 内部? |

<a id="g3"></a>

## docs/design (1)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [kernel_plugin_0_2_0a0.md](docs/design/kernel_plugin_0_2_0a0.md) | llcore 0.2.0a0 — Kernel Plugin アーキテクチャ設計 | 本流 src/llcore/ に 構造破綻なく additive 取り込みするための plugin 境界を formal 化する。 | 2026-05-29 | ❓ 要判断 |

<a id="g4"></a>

## docs/eval (1)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [STAGE_3_EVAL_HARNESS.md](docs/eval/STAGE_3_EVAL_HARNESS.md) | Stage 3 評価 Harness — 残 3 独自軸 PoC の横断評価設計 | - MODES Anew (新規行動採用件数 / 世代) が 90% 世代で 0 (PoC 2b G6) | 2026-05-28 | ❓ 要判断 |

<a id="g5"></a>

## docs/papers (4)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [marabou_sound_extension_sketch.md](docs/papers/marabou_sound_extension_sketch.md) | Marabou Incremental NN Verification の "異構造" Refinement Relation Sound 拡張 — Sketch | Wu et al. (2026-03, arxiv 2603.12232) "Incremental NN Verification via Learned Conflicts" は | 2026-05-28 | ❓ 要判断 |
| [vnn_comp_benchmark_spec.md](docs/papers/vnn_comp_benchmark_spec.md) | online-arch-evo — Benchmark Specification, v0.1 | This document is normative. The proposal paper is descriptive; if the two disagree, this file wins. | 2026-05-28 | ❓ 要判断 |
| [vnn_comp_online_arch_evolution_proposal.md](docs/papers/vnn_comp_online_arch_evolution_proposal.md) | Online Architecture Evolution Verification: A New VNN-COMP Category for Continuously Mutating Neural Networks | Draft target venues: TMLR (full paper, primary) / GECCO 2027 short paper (Evolutionary Computation track) / NeurIPS 2026 workshop on Verification × ML. | 2026-05-28 | ❓ 要判断 |
| [vnn_comp_reference_impl_spec.md](docs/papers/vnn_comp_reference_impl_spec.md) | Reference Implementation Spec — llcore PoC 1a wrapper | This document describes the reference online-arch-evo submission, built by wrapping llcore PoC 1a (scripts/poc1az3invariant.py + src/llcore/verifier/invariants.py) in the stdin/stdout protocol of the | 2026-05-28 | ❓ 要判断 |

<a id="g6"></a>

## docs/poc (14)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [COMPLETION_VERDICT.md](docs/poc/COMPLETION_VERDICT.md) | llcore CPU PoC Battery 完成 — Final Verdict | → Stop hook condition "llcore完成。" 満足。 | 2026-05-28 | 🔒 内部? |
| [EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md](docs/poc/EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md) | llcore 進化機構 健全性監査 (Evolution Soundness Audit) | 後者を falsifiable に検証する。 | 2026-05-30 | 🔒 内部? |
| [poc_0a_verdict.md](docs/poc/poc_0a_verdict.md) | PoC 0a Verdict — state update 数式遺伝子 (v1 → v2 redesign) | statet+1 = decay  statet + mix  xt  tanh(gatestr  statet) | 2026-05-28 | 🔒 内部? |
| [poc_0b_verdict.md](docs/poc/poc_0b_verdict.md) | PoC 0b Verdict — Synthetic sequence fitness (copy / addition) | 次段 PoC 0c (自前 minimal GA) に進めます. | 2026-05-28 | 🔒 内部? |
| [poc_0c_verdict.md](docs/poc/poc_0c_verdict.md) | PoC 0c Verdict — llcore 自前 minimal GA (llive 非依存) 進化 10×10 | elitism による単調性 + diversity 維持 + specialist 出現を実証. | 2026-05-28 | 🔒 内部? |
| [poc_1a_verdict.md](docs/poc/poc_1a_verdict.md) | PoC 1a Verdict — Z3 verifier statenorm 有界 invariant (Stage 1a) | verifygenesafe で tanhval を free in -1, 1 にしていたため、mix と gatestr が | 2026-05-28 | 🔒 内部? |
| [poc_2a_verdict.md](docs/poc/poc_2a_verdict.md) | PoC 2a Verdict — factorhook × state update kernel (mock) | llcore 自前 factorhook protocol が (a) 10 因子保持 + clamp、(b) Noop で Δ=1.0、 | 2026-05-28 | 🔒 内部? |
| [poc_2b_verdict.md](docs/poc/poc_2b_verdict.md) | PoC 2b Verdict — persona-indexed specialist × Z3 verifier × 開放端進化機構 | - scripts/poc2bpersonaindexedverifiedevolution.py | 2026-05-28 | 🔒 内部? |
| [poc_3a_verdict.md](docs/poc/poc_3a_verdict.md) | PoC 3a Verdict — Marabou Incremental sound 拡張 refinement (異構造) + MCC curriculum (Stage 3a) | relation sound 拡張 + open-ended ChangeOp curriculum | 2026-05-28 | 🔒 内部? |
| [poc_7a_verdict.md](docs/poc/poc_7a_verdict.md) | PoC 7a Verdict — VNN-COMP online-arch-evo 新カテゴリ提案 (Stage 7a) | tests/unit/testpoc7avnncompreference.py ............. 17 passed in 1.04s | 2026-05-28 | 🔒 内部? |
| [STAGE_3_VERDICT.md](docs/poc/STAGE_3_VERDICT.md) | Stage 3 統合 Verdict — 残 3 独自軸 PoC + Codex pair-review 完走 | - critical gate (PoC 2b G3/G7 + PoC 3a G1/G2/G3 + PoC 7a G3) すべて PASS、PASS 率 100% | 2026-05-28 | 🔒 内部? |
| [STEP4_DESIGN_space_expansion_niching.md](docs/poc/STEP4_DESIGN_space_expansion_niching.md) | CPU 手順 4 設計ノート — 空間拡張 + 分離機構で ③ を立てる | commit 5ee1c13) / 手順 2 (per-gene ridge readout un-flatten, commit c578f6f) の上に立つ。 | 2026-05-30 | ❓ 要判断 |
| [STEP4_SELECTION_VERDICT.md](docs/poc/STEP4_SELECTION_VERDICT.md) | 手順 4 verdict — ③ が立つ状態を発見 (deceptive corridor + behavioral niching) | Goal (ユーザー 2026-05-30): 「推奨の方法 (MAP-Elites) で進め、③ が立つという状態を探す」。 | 2026-05-30 | 🔒 内部? |
| [YOUGO_平易版.md](docs/poc/YOUGO_平易版.md) | llcore 進化研究 用語集（平易版・図入り） | この研究で使う英語・専門用語を「山登り」のたとえで説明します。 | 2026-05-30 | ❓ 要判断 |

<a id="g7"></a>

## research/other_archs (2)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [OTHERARCH_VERDICT.md](research/other_archs/OTHERARCH_VERDICT.md) | Other-Architecture 統合 Verdict (完成版, 2026-05-29) | 3 PoC で llcore.evolution.{adaptivefloor, lineagereservoir, modesmeter} 4 機構のうち 3 機構 (適応難易度 + 中立貯蔵庫 + MODES) + Z3 per-gene invariant + 自前 minimal GA が動作。MCC curriculum は Neural ODE で dt anneal、GNN で L | 2026-05-29 | 🔒 内部? |
| [README.md](research/other_archs/README.md) | llcore research — Other Architectures (Transformer 以外への移植 PoC) | llcore approach が他アーキで成立するための必要条件: | 2026-05-29 | 🌐 公開候補 |

<a id="g8"></a>

## research/other_archs/gnn (1)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [verdict.md](research/other_archs/gnn/verdict.md) | GNN PoC Verdict — llcore approach の GNN message passing op への移植 | - research/otherarchs/gnn/gnngene.py (GnnGene + aggregate + update + forward) | 2026-05-29 | 🔒 内部? |

<a id="g9"></a>

## research/other_archs/gnn/dynamic_graph (1)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [verdict.md](research/other_archs/gnn/dynamic_graph/verdict.md) | Dynamic GNN Stage 2 PoC Verdict — 動的 graph + ChangeOp の本格実証 | - research/otherarchs/gnn/dynamicgraph/dgnngene.py (DynamicGraph + GraphChangeOp 4 種 + DynamicGnnGene) | 2026-05-29 | 🔒 内部? |

<a id="g10"></a>

## research/other_archs/neural_ode (1)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [verdict.md](research/other_archs/neural_ode/verdict.md) | PoC Verdict — Neural ODE / LTC への llcore approach 移植 | - odegene.py (NeuralODEGene dataclass + vectorfield + forwardeuler + empiricallipschitz) | 2026-05-29 | 🔒 内部? |

<a id="g11"></a>

## research/other_archs/snn (1)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [verdict.md](research/other_archs/snn/verdict.md) | PoC SNN — LIF への llcore approach 移植 verdict (2026-05-29) | taum  dV/dt = -(V - Vrest) + R  I(t) | 2026-05-29 | 🔒 内部? |

<a id="g12"></a>

## research/other_archs/snn/izhikevich (1)

| ファイル | タイトル | 説明 | 更新 | 区分 |
|---|---|---|---|---|
| [verdict.md](research/other_archs/snn/izhikevich/verdict.md) | PoC SNN Izhikevich — LIF への llcore approach 移植 一般化 verdict (2026-05-29) | dv/dt = 0.04 v^2 + 5 v + 140 - u + I | 2026-05-29 | 🔒 内部? |
