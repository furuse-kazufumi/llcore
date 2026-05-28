# SPDX-License-Identifier: Apache-2.0
"""llcore — Verified Neural Architecture Evolution on CPU.

FullSense ll- family の 4 番目 (llmesh / llive / llove に続く)。
Transformer のコアアルゴリズム (state update / 学習則 / 認知駆動 Δ) を
**Z3 verifier で破綻させずに** 進化させる研究フレームワーク。

設計原則:
- CPU 完結 (個人 compute 制約下で論文化価値を出す)
- 段階的 PoC (各段 falsifiable + 破綻ゲート)
- 既存 llive 資産は import 再利用、改造禁止
- llive と将来融合する流れに合わせ、API は llive から import 容易に保つ

詳細は README.md 段階的 PoC レダー、および
``llive/docs/papers/2026-05-29_research_plan_core_evolution.md`` 参照。
"""

__version__ = "0.1.0a0"
