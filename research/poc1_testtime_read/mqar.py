# SPDX-License-Identifier: Apache-2.0
"""PoC-1 の合成 MQAR (multi-query associative recall) データ生成。

pre-reg (`docs/research/preregistration/prereg_poc1_testtime_read_2026-06-29.md`) §3.1 の
「合成 MQAR + S-NIAH passkey で学習 → 状態を凍結 → read 側 test-time を比較」の学習タスク。

系列フォーマット (1 系列 = KV フェーズ + Query フェーズ):
    k1 v1 k2 v2 ... kN vN   kq1 BLANK kq2 BLANK ... kqQ BLANK
- key   token: [1 .. D]
- value token: [D+1 .. 2D]
- BLANK/pad  : 0
学習ターゲット: 各 query-key 位置 kqj の **次トークン予測**で、束縛値 vqj を当てる (CE)。
それ以外の位置は ignore_index=-1。→ recall = query-key 位置の argmax が束縛値と一致する割合。

非直交 key: key/value 埋め込みはモデルが学習する (designed codebook 無し) = pre-reg の
novelty regime (学習・非直交 key の凍結状態) を満たす。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MQARConfig:
    num_keys: int = 16       # D: 異なる key 種
    num_pairs: int = 8       # N: 1 系列あたりの KV ペア数 (<= num_keys)
    num_queries: int = 8     # Q: 1 系列あたりのクエリ数
    seed: int = 0
    unique_values: bool = False  # True=系列内 value も非復元 (binding 混入なし)。
    #   既定 False = 復元抽選 (現行動作・byte-identical)。敵対レビュー由来の binding-artifact 検証用:
    #   value 重複があると「頻出値スナップ」で recall が水増しされうる (2026-07-11)。

    def __post_init__(self) -> None:
        if not (0 < self.num_pairs <= self.num_keys):
            raise ValueError("num_pairs は 1..num_keys の範囲")
        if self.num_queries <= 0:
            raise ValueError("num_queries > 0")

    @property
    def vocab_size(self) -> int:
        return 2 * self.num_keys + 1  # 0=BLANK, key [1..D], value [D+1..2D]

    @property
    def seq_len(self) -> int:
        return 2 * self.num_pairs + 2 * self.num_queries


def make_batch(cfg: MQARConfig, batch_size: int, generator: torch.Generator
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(inputs[B,T], targets[B,T], query_pos[B,Q]) を返す。

    targets は query-key 位置のみ束縛値、他は -1。query_pos は各系列のクエリ位置 (評価用)。
    """
    D, N, Q = cfg.num_keys, cfg.num_pairs, cfg.num_queries
    T = cfg.seq_len
    inp = torch.zeros(batch_size, T, dtype=torch.long)
    tgt = torch.full((batch_size, T), -1, dtype=torch.long)
    qpos = torch.zeros(batch_size, Q, dtype=torch.long)
    for b in range(batch_size):
        # N 個の異なる key を選び、各に random value を束縛する
        keys = torch.randperm(D, generator=generator)[:N] + 1              # [1..D]
        if cfg.unique_values:
            vals = torch.randperm(D, generator=generator)[:N] + (D + 1)    # 非復元=系列内 value 一意
        else:
            vals = torch.randint(0, D, (N,), generator=generator) + (D + 1)  # 復元 (現行・重複可)
        # KV フェーズ
        for i in range(N):
            inp[b, 2 * i] = keys[i]
            inp[b, 2 * i + 1] = vals[i]
        # Query フェーズ: N 個から Q 個を復元 (重複可)
        base = 2 * N
        qidx = torch.randint(0, N, (Q,), generator=generator)
        for j in range(Q):
            pos = base + 2 * j
            inp[b, pos] = keys[qidx[j]]
            tgt[b, pos] = vals[qidx[j]]  # 次トークン = 束縛値
            qpos[b, j] = pos
    return inp, tgt, qpos
