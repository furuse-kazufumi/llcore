# SPDX-License-Identifier: Apache-2.0
"""llcore 自前 minimal 進化エンジン (llive 非依存).

設計判断 (ユーザー指示 2026-05-29):
- llive の lldarwin_v2 / Genome3D / persona_evolution には依存しない
- 自前 minimal GA で進化ループを回す (state update 数式遺伝子に特化)
- llive 既存資産は **baseline 比較実験** (PoC 0c') 用に optional import のみ

最小 API (Stage 0c で実装):
- :class:`Population` — 個体集団管理
- :class:`MinimalGA` — tournament + uniform mutation の素朴 GA
- :func:`evolve` — 進化 main loop (10 個体 × 10 世代から)

将来拡張 (Stage 1+):
- Z3 verifier-gate 統合 (`llcore.verifier`)
- ε-lexicase / novelty 等の高度 selector を自前実装 (llive 参考だが import なし)
- factor_hook 注入経路 (`llcore.factor_hook`)

honest 留保:
- Stage 0c はまだ未着手 (本 module は package skeleton のみ)
- llive lldarwin_v2 の優秀な機能 (ε-lexicase / QD / 中立貯蔵庫) は自前実装で追従する
  が、初期は素朴 tournament で十分 (state update 数式は genome 次元 3 と小さい)
"""

from .adaptive_floor import AdaptiveFloorGate
from .lineage_reservoir import LineageReservoir
from .minimal_ga import (
    EvolutionResult,
    FitnessFunc,
    Individual,
    Population,
    crossover_uniform,
    evaluate_population,
    evolve,
    initialize_random_population,
    tournament_select,
    uniform_mutate,
)
from .modes_meter import ModesMeter, pairwise_l2_diversity

__all__ = [
    "AdaptiveFloorGate",
    "EvolutionResult",
    "FitnessFunc",
    "Individual",
    "LineageReservoir",
    "ModesMeter",
    "Population",
    "crossover_uniform",
    "evaluate_population",
    "evolve",
    "initialize_random_population",
    "pairwise_l2_diversity",
    "tournament_select",
    "uniform_mutate",
]
