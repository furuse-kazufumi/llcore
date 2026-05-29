# SPDX-License-Identifier: Apache-2.0
"""GNN Stage 2 — 動的 graph + ChangeOp PoC (llcore 独自軸 #5 接続).

Codex F3 で指摘された「固定 ring topology のため真の構造変化 ChangeOp 未実証」を
解消するための 隔離 PoC。既存 ``research/other_archs/gnn/`` (固定 ring) には触れず、
本 dir 内に dynamic graph + node/edge ChangeOp + refinement chain Z3 検査を実装。
"""
