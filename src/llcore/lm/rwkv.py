# SPDX-License-Identifier: Apache-2.0
"""Minimal RWKV-4 style recurrent char LM with stable running-max WKV."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, cast

import torch
from torch import nn
from torch.nn import functional as F

from llcore.lm.activations import K3_BETA_GATE, K3_BETA_UP, situ_glu, softcap

_FFN_ACTIVATIONS = frozenset({"sq_relu", "swiglu", "situ_glu"})


class RWKVLayerState(NamedTuple):
    prev_tm_x: torch.Tensor
    prev_cm_x: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    p: torch.Tensor


def bounded_log_decay(raw: torch.Tensor, *, log_floor: float) -> torch.Tensor:
    """Kimi K3 式の下限付き log-decay ``g = log_floor * sigmoid(raw)``.

    ``sigmoid`` は開区間 (0, 1) を返すので ``g ∈ (log_floor, 0)``、したがって保持率
    ``α = exp(g) ∈ (exp(log_floor), 1)`` となり **1 ステップあたりの忘却速度に床が付く**。
    従来の ``-exp(raw)`` は下に非有界で ``α → 0`` (チャネル死 = 1 ステップで全忘却) を
    許してしまう。

    Parameters
    ----------
    raw : torch.Tensor
        生パラメータ (``time_decay``)。値域の制約なし。
    log_floor : float
        log 空間の下限。負値のみ (例: K3 の ``g_min = -5``)。

    Returns
    -------
    torch.Tensor
        ``(log_floor, 0)`` に収まる log-decay。

    Raises
    ------
    ValueError
        ``log_floor >= 0`` の場合 (減衰にならない)。
    """
    if log_floor >= 0.0:
        raise ValueError(f"log_floor must be < 0 (decay only), got {log_floor}")
    return log_floor * torch.sigmoid(raw)


def init_raw_for_log_decay(target_log_decay: float, *, log_floor: float | None) -> float:
    """目標 log-decay を実現する ``time_decay`` の raw 値を返す (逆写像).

    パラメータ化を切り替えると同じ raw 値でも保持率が変わる (zeros 初期化だと
    ``-exp(0) = -1`` に対し ``-5*sigmoid(0) = -2.5``)。**初期保持率を揃えずに
    比較すると、測っているのはパラメータ化ではなく初期値の差**になるので、
    統制実験ではこの逆写像で両者の出発点を合わせる。

    Parameters
    ----------
    target_log_decay : float
        揃えたい log-decay (負値)。保持率は ``exp(target_log_decay)``。
    log_floor : float | None
        None = ``-exp(w)`` (従来)、負値 = ``log_floor * sigmoid(w)`` (有界)。

    Returns
    -------
    float
        その parameterization で ``target_log_decay`` を再現する raw 値。

    Raises
    ------
    ValueError
        ``target_log_decay >= 0``、または有界版で床より下を要求した場合
        (その値は表現できない)。
    """
    if target_log_decay >= 0.0:
        raise ValueError(
            f"target_log_decay must be < 0 (decay only), got {target_log_decay}"
        )
    if log_floor is None:
        # -exp(w) = target  ->  w = log(-target)
        return math.log(-target_log_decay)
    if log_floor >= 0.0:
        raise ValueError(f"log_floor must be < 0 or None, got {log_floor}")
    if target_log_decay <= log_floor:
        raise ValueError(
            f"target_log_decay {target_log_decay} is not representable above "
            f"log_floor {log_floor} (有界版は床より下を表現できない)"
        )
    # log_floor * sigmoid(w) = target  ->  sigmoid(w) = target/log_floor
    ratio = target_log_decay / log_floor  # ∈ (0, 1)
    return math.log(ratio / (1.0 - ratio))


@dataclass
class RWKVConfig:
    """Configuration for :class:`RWKVLM`."""

    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    model_type: str = "rwkv-4"
    decay_log_floor: float | None = None
    """None = 従来の ``-exp(w)`` (下に非有界)。負値を与えると K3 式の有界 log-decay。"""
    ffn_activation: str = "sq_relu"
    """``"sq_relu"`` = 従来の squared-ReLU (非有界・二次)。``"swiglu"`` = 非有界 GLU
    (situ_glu と同一構造の対照)。``"situ_glu"`` = K3 の有界 GLU。"""
    situ_beta_gate: float = K3_BETA_GATE
    situ_beta_up: float = K3_BETA_UP
    ffn_softcap: float | None = None
    """None = 掛けない。正値を与えると channel-mix の hidden に ``softcap(a, β)`` を
    適用して ``|a| <= β`` に有界化する。**活性化の種類に依らず効く直交した knob** で、
    SiTU-GLU と違い GLU 構造 (up 射影) を要求しないためパラメータ数が変わらない。"""
    ffn_hidden_mult: float = 4.0
    """channel-mix の hidden 幅倍率。SiTU-GLU は枝が 1 本増えるので、パラメータ数を
    sq_relu と揃えて比較したい場合は 8/3 にする (SwiGLU 論文と同じ 2/3 則)。"""

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be > 0, got {self.n_layer}")
        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be > 0, got {self.n_embd}")
        if self.decay_log_floor is not None and self.decay_log_floor >= 0.0:
            raise ValueError(
                f"decay_log_floor must be < 0 or None, got {self.decay_log_floor}"
            )
        if self.ffn_softcap is not None and self.ffn_softcap <= 0.0:
            raise ValueError(f"ffn_softcap must be > 0 or None, got {self.ffn_softcap}")
        if self.ffn_hidden_mult <= 0.0:
            raise ValueError(
                f"ffn_hidden_mult must be > 0, got {self.ffn_hidden_mult}"
            )
        if self.ffn_activation not in _FFN_ACTIVATIONS:
            raise ValueError(
                f"ffn_activation must be one of {sorted(_FFN_ACTIVATIONS)}, "
                f"got {self.ffn_activation!r}"
            )


def _mix(cur: torch.Tensor, prev: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    return cur * mix + prev * (1.0 - mix)


class RWKVTimeMix(nn.Module):
    """RWKV time-mix with stable running-max WKV state."""

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        d = config.n_embd
        self.mix_k = nn.Parameter(torch.rand(d))
        self.mix_v = nn.Parameter(torch.rand(d))
        self.mix_r = nn.Parameter(torch.rand(d))
        self.time_decay = nn.Parameter(torch.zeros(d))
        self.time_first = nn.Parameter(torch.zeros(d))
        self.key = nn.Linear(d, d, bias=config.bias)
        self.value = nn.Linear(d, d, bias=config.bias)
        self.receptance = nn.Linear(d, d, bias=config.bias)
        self.output = nn.Linear(d, d, bias=config.bias)
        self.decay_log_floor = config.decay_log_floor

    def log_decay(self) -> torch.Tensor:
        """1 ステップ分の log-decay を返す (``α = exp(log_decay)`` が保持率)."""
        if self.decay_log_floor is None:
            return -torch.exp(self.time_decay)
        return bounded_log_decay(self.time_decay, log_floor=self.decay_log_floor)

    def step(
        self, x: torch.Tensor, prev_x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xk = _mix(x, prev_x, self.mix_k)
        xv = _mix(x, prev_x, self.mix_v)
        xr = _mix(x, prev_x, self.mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        decay = self.log_decay()

        q = torch.maximum(p, self.time_first + k)
        e1 = torch.exp(p - q)
        e2 = torch.exp(self.time_first + k - q)
        wkv = (e1 * a + e2 * v) / (e1 * b + e2)

        q2 = torch.maximum(p + decay, k)
        e1n = torch.exp(p + decay - q2)
        e2n = torch.exp(k - q2)
        next_a = e1n * a + e2n * v
        next_b = e1n * b + e2n
        next_p = q2
        out = self.output(r * wkv)
        return out, next_a, next_b, next_p


class RWKVChannelMix(nn.Module):
    """RWKV channel-mix FFN.

    既定は RWKV-4 本来の squared-ReLU (``value(relu(key(x))**2)``)。これは**非有界かつ
    二次**なので、低ビット化・形式検証の観点では最も扱いにくい。``ffn_activation``
    を ``"situ_glu"`` にすると Kimi K3 の有界 GLU に切り替わり、value へ渡る中間活性が
    ``|a| <= β_gate·β_up`` に有界化される (既定 4.0 × 25.0 = 100)。
    """

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        d = config.n_embd
        hidden = max(1, round(config.ffn_hidden_mult * d))
        self.mix_k = nn.Parameter(torch.rand(d))
        self.mix_r = nn.Parameter(torch.rand(d))
        self.key = nn.Linear(d, hidden, bias=config.bias)
        self.value = nn.Linear(hidden, d, bias=config.bias)
        self.receptance = nn.Linear(d, d, bias=config.bias)
        self.ffn_activation = config.ffn_activation
        self.situ_beta_gate = config.situ_beta_gate
        self.situ_beta_up = config.situ_beta_up
        self.ffn_softcap = config.ffn_softcap
        if self.ffn_activation in ("situ_glu", "swiglu"):
            # GLU は gate 枝と up 枝の 2 本が要る。key を gate 枝に流用し up を足す。
            self.up = nn.Linear(d, hidden, bias=config.bias)

    def ffn_hidden(self, x: torch.Tensor, prev_x: torch.Tensor) -> torch.Tensor:
        """``value`` 射影に渡る直前の中間活性を返す (有界性の検査点)."""
        hidden = self._ffn_hidden_raw(x, prev_x)
        if self.ffn_softcap is not None:
            hidden = softcap(hidden, beta=self.ffn_softcap)
        return hidden

    def _ffn_hidden_raw(self, x: torch.Tensor, prev_x: torch.Tensor) -> torch.Tensor:
        xk = _mix(x, prev_x, self.mix_k)
        if self.ffn_activation == "situ_glu":
            return situ_glu(
                self.key(xk),
                self.up(xk),
                beta_gate=self.situ_beta_gate,
                beta_up=self.situ_beta_up,
            )
        if self.ffn_activation == "swiglu":
            # situ_glu と同一構造・同一パラメータ数で **有界化だけを外した**対照。
            # これとの差が「有界化のコスト」を切り分ける。
            return cast(torch.Tensor, F.silu(self.key(xk)) * self.up(xk))
        k = F.relu(self.key(xk))
        return cast(torch.Tensor, k * k)

    def step(self, x: torch.Tensor, prev_x: torch.Tensor) -> torch.Tensor:
        xr = _mix(x, prev_x, self.mix_r)
        kv = self.value(self.ffn_hidden(x, prev_x))
        r = torch.sigmoid(self.receptance(xr))
        return cast(torch.Tensor, r * kv)


class RWKVBlock(nn.Module):
    """Pre-LN RWKV block with time-mix then channel-mix."""

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.time_mixer = RWKVTimeMix(config)
        self.channel_mixer = RWKVChannelMix(config)

    def step(self, x: torch.Tensor, state: RWKVLayerState) -> tuple[torch.Tensor, RWKVLayerState]:
        x_ln1 = self.ln1(x)
        att, next_a, next_b, next_p = self.time_mixer.step(
            x_ln1, state.prev_tm_x, state.a, state.b, state.p
        )
        x = x + att
        x_ln2 = self.ln2(x)
        ffn = self.channel_mixer.step(x_ln2, state.prev_cm_x)
        x = x + ffn
        next_state = RWKVLayerState(
            prev_tm_x=x_ln1,
            prev_cm_x=x_ln2,
            a=next_a,
            b=next_b,
            p=next_p,
        )
        return x, next_state


class RWKVLM(nn.Module):
    """RWKV-4 style char LM with O(1) generation state per layer."""

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.ln0 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([RWKVBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def init_state(
        self, batch_size: int, *, device: torch.device | None = None
    ) -> list[RWKVLayerState]:
        dev = device if device is not None else self.lm_head.weight.device
        zeros = torch.zeros(batch_size, self.config.n_embd, device=dev)
        neg_inf = torch.full((batch_size, self.config.n_embd), -1e30, device=dev)
        return [
            RWKVLayerState(
                prev_tm_x=zeros.clone(),
                prev_cm_x=zeros.clone(),
                a=zeros.clone(),
                b=zeros.clone(),
                p=neg_inf.clone(),
            )
            for _ in range(self.config.n_layer)
        ]

    def state_bytes(self, state: list[RWKVLayerState]) -> int:
        total = 0
        for layer_state in state:
            for tensor in layer_state:
                total += int(tensor.numel() * tensor.element_size())
        return total

    def step(
        self, idx_t: torch.Tensor, state: list[RWKVLayerState] | None = None
    ) -> tuple[torch.Tensor, list[RWKVLayerState]]:
        if idx_t.ndim != 1:
            raise ValueError(f"idx_t must be 1-D [B], got shape {tuple(idx_t.shape)}")
        cur_state = self.init_state(idx_t.size(0), device=idx_t.device) if state is None else state
        x = self.drop(self.ln0(self.emb(idx_t)))
        next_state: list[RWKVLayerState] = []
        for li, block in enumerate(self.blocks):
            assert isinstance(block, RWKVBlock)
            x, layer_state = block.step(x, cur_state[li])
            next_state.append(layer_state)
        x = self.ln_f(x)
        logits = cast(torch.Tensor, self.lm_head(x))
        return logits, next_state

    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self(idx)[0])

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError(f"idx must be 2-D [B,T], got shape {tuple(idx.shape)}")
        _, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(
                f"sequence length {t} exceeds block_size {self.config.block_size}"
            )
        state = self.init_state(idx.size(0), device=idx.device)
        logits_steps: list[torch.Tensor] = []
        for pos in range(t):
            logits_t, state = self.step(idx[:, pos], state)
            logits_steps.append(logits_t.unsqueeze(1))
        logits = torch.cat(logits_steps, dim=1)
        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def streaming_nll(self, ids: torch.Tensor, chunk_size: int = 256) -> tuple[float, int]:
        """Mean next-token cross-entropy (nats) over a 1-D sequence of *any* length.

        The O(1)-per-layer WKV state has no architectural context limit, so this scores
        sequences longer than ``block_size`` (which only caps the batched :meth:`forward`)
        and never materializes O(T) logits: it steps through the sequence accumulating the
        loss, so peak activation memory is O(``chunk_size``) in T while the state stays
        O(1). Predicts ``ids[1:]`` from ``ids[:-1]``; returns ``(mean_nll, n_predicted)``.
        A GPT cannot do this — its attention is O(T²) and ``block_size``-bounded.
        """
        if ids.ndim != 1:
            raise ValueError(f"ids must be 1-D [T], got shape {tuple(ids.shape)}")
        n = int(ids.size(0))
        if n < 2:
            raise ValueError(f"streaming_nll needs >= 2 tokens, got {n}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        was_training = self.training
        self.eval()
        state: list[RWKVLayerState] | None = None
        inputs, targets = ids[:-1], ids[1:]
        total = 0.0
        for start in range(0, n - 1, chunk_size):
            stop = min(start + chunk_size, n - 1)
            logits_chunk: list[torch.Tensor] = []
            for i in range(start, stop):
                logits, state = self.step(inputs[i : i + 1], state)
                logits_chunk.append(logits)
            chunk = torch.cat(logits_chunk, dim=0)
            total += float(F.cross_entropy(chunk, targets[start:stop], reduction="sum").item())
        if was_training:
            self.train()
        return total / (n - 1), n - 1

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if idx.ndim != 2:
            raise ValueError(f"idx must be 2-D [B,T], got shape {tuple(idx.shape)}")
        if idx.size(1) == 0:
            raise ValueError("idx must contain at least one prompt token")
        was_training = self.training
        self.eval()
        state = self.init_state(idx.size(0), device=idx.device)
        last_logits: torch.Tensor | None = None
        for pos in range(idx.size(1)):
            last_logits, state = self.step(idx[:, pos], state)
        assert last_logits is not None
        out = idx
        for _ in range(max_new_tokens):
            logits = last_logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            out = torch.cat((out, idx_next), dim=1)
            last_logits, state = self.step(idx_next[:, 0], state)
        if was_training:
            self.train()
        return out
