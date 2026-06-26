# SPDX-License-Identifier: Apache-2.0
"""Delta-rule fast-weight constant-state char LM — L1 plateau-null experiment.

⚠ HONEST NAMING CORRECTION (2026-06-26, cross-review by Codex + faithfulness deep-reads of
arXiv:2407.04620 / 2412.06464 / 2406.06484): this cell is a **Gated DeltaNet**, NOT TTT-Linear.
As of the 2026-06-26 patch the core follows the canonical Gated DeltaNet update (Eq.8 of
arXiv:2412.06464): ``S_t = α_t·S_{t-1} + β_t·(v_t − α_t·S_{t-1}k̂_t)k̂_tᵀ`` with **data-dependent**
scalar decay ``α_t = exp(-exp(A_log)·softplus(a_proj(h)+dt_bias))`` (Mamba2 parameterization) and
write strength ``β_t = sigmoid(b_proj(h))``, L2-normalized q,k, and a gated-RMSNorm output. This
replaced the earlier static per-channel ``decay/eta`` (which made it merely a constant-decay
DeltaNet). Remaining gap vs the canonical block: single-head (no multi-head), no short depthwise
conv. The class is still named ``TTTLinearLM`` pending a mechanical rename to ``GatedDeltaNetLM``
— read it as the latter. A faithful TTT-Linear variant (inner LN, η(x), mini-batch dual form) is
separate future work; the running ``out/ttt_plateau`` result used the pre-patch *static-gate*
cell and is the static-gate baseline.

KNOWN LIMITATION (Codex + faithfulness review): to actually move the effective-context plateau the
OUTER training must carry the state across blocks (truncated BPTT with state-carry, ``tbptt.py``);
with the stock ``Trainer`` (state reset every ``block_size``) this model never sees >block_size
dependencies during training, so ``context_length_curve`` past ``block_size`` measures the
recurrence's *extrapolation inductive bias*, not learned long-context. Interpret accordingly.

----- original motivation (kept) -----

Motivation (docs/MODEL_LANDSCAPE_2026_06.md §10): llcore's gated-RNN (``recurrent.py``) plateaus
at an effective context ≈ ``block_size`` because its mixing matrix ``W`` is *learned once* and the
state update is a fixed linear recurrence — what to remember is decided only by BPTT through the
training window. TTT (Sun et al. 2024, arXiv:2407.04620) instead makes the per-layer state a
**fast-weight matrix updated by one inner gradient step on a self-supervised loss** each token, so
"what to store" is decided online and is not bound to the BPTT horizon. The TTT paper shows TTT
keeps improving with context where Mamba stalls — exactly the plateau llcore is stuck on.

The cell here is the TTT-Linear / delta-rule update (constant O(d²) state, O(1) per step):

    k_t, v_t, q_t = K(h_t), V(h_t), Q(h_t)          # slow weights (learned by outer BPTT)
    k̂_t          = k_t / ‖k_t‖                       # L2-normalize for a bounded outer product
    grad         = (S_{t-1} k̂_t − v_t) ⊗ k̂_t         # ∇_S ½‖S k̂ − v‖²  (the delta rule)
    S_t          = decay ⊙ S_{t-1} − η · grad         # one inner SGD step, learnable decay/η
    o_t          = S_t q_t                            # query the freshly-updated memory

``decay = sigmoid(raw_decay)`` (per state-row forget gate) and ``η = sigmoid(raw_eta)`` (bounded
inner learning rate) are slow parameters. The outer training still uses BPTT, but the *inductive
bias* (gradient-based associative memory) is what we test against the gated-RNN baseline and a
capacity-only "StateX-style" wider gated-RNN, via ``longctx_eval.context_length_curve``.

CPU / float32 only. Mirrors :class:`llcore.lm.recurrent.RecurrentLM`'s public interface (``step`` /
``forward`` / ``streaming_nll`` / ``generate`` / ``init_state``) so the existing trainer and
long-context evaluators consume it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TTTLinearConfig:
    """Configuration for :class:`TTTLinearLM`."""

    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_embd: int = 128
    state_dim: int = 64
    dropout: float = 0.0
    bias: bool = True
    model_type: str = "ttt-linear"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be > 0, got {self.n_layer}")
        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be > 0, got {self.n_embd}")
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be > 0, got {self.state_dim}")


class TTTLinearCore(nn.Module):
    """One TTT-Linear layer: a delta-rule fast-weight state queried back to the token width.

    The per-layer state is a fast-weight matrix ``S`` of shape ``[B, state_dim, state_dim]``
    (constant in sequence length); ``step`` performs one inner gradient step on the per-token
    reconstruction loss and returns the residual-updated hidden plus the new state.
    """

    def __init__(self, config: TTTLinearConfig) -> None:
        super().__init__()
        d, sd = config.n_embd, config.state_dim
        self.state_dim = sd
        self.k_proj = nn.Linear(d, sd, bias=config.bias)
        self.v_proj = nn.Linear(d, sd, bias=config.bias)
        self.q_proj = nn.Linear(d, sd, bias=config.bias)
        self.out_proj = nn.Linear(sd, d, bias=config.bias)
        # data-dependent decay α_t = exp(-exp(A_log)·softplus(a_proj(h)+dt_bias)) ∈ (0,1)
        # (Mamba2 parameterization; faithful Gated DeltaNet, arXiv:2412.06464 §3.3) — per-sample scalar
        self.a_proj = nn.Linear(d, 1, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.empty(1).uniform_(1.0, 16.0)))
        dt = torch.tensor([0.05])
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(dt)))  # inverse-softplus(0.05)
        # data-dependent write strength β_t = sigmoid(b_proj(h)) ∈ (0,1) — per-sample scalar
        self.b_proj = nn.Linear(d, 1, bias=True)
        # output gate + gated RMSNorm (Mamba2/fla output normalization)
        self.g_proj = nn.Linear(d, sd, bias=config.bias)
        self.o_norm_w = nn.Parameter(torch.ones(sd))
        self.norm = nn.LayerNorm(d, bias=config.bias)

    def step(self, h: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # state: [B, sd, sd] fast weight; h: [B, d]. Faithful Gated DeltaNet update (Eq.8):
        #   S_t = α_t·S_{t-1} + β_t·(v_t − α_t·S_{t-1}k̂_t)·k̂_tᵀ   (prediction taken AFTER decay)
        k = self.k_proj(h)
        v = self.v_proj(h)
        q = self.q_proj(h)
        khat = F.normalize(k, p=2.0, dim=-1, eps=1e-6)  # [B, sd]
        qhat = F.normalize(q, p=2.0, dim=-1, eps=1e-6)  # [B, sd] (canon: q also L2-normalized)
        alpha = torch.exp(-self.A_log.exp() * F.softplus(self.a_proj(h) + self.dt_bias))  # [B,1]
        beta = torch.sigmoid(self.b_proj(h))  # [B,1]
        pred = torch.einsum("bij,bj->bi", state, khat)  # S k̂  -> [B, sd]
        err = v - alpha * pred  # v − α·S k̂  (decayed prediction) -> [B, sd]
        write = torch.einsum("bi,bj->bij", err, khat)  # (·) ⊗ k̂  -> [B, sd, sd]
        next_state = alpha[:, :, None] * state + beta[:, :, None] * write
        out = torch.einsum("bij,bj->bi", next_state, qhat)  # S' q̂ -> [B, sd]
        gated = out * F.silu(self.g_proj(h))  # output gate
        gated = gated * torch.rsqrt(gated.pow(2).mean(-1, keepdim=True) + 1e-6) * self.o_norm_w
        next_h = self.norm(h + self.out_proj(gated))
        return next_h, next_state


class TTTLinearLM(nn.Module):
    """Char-level TTT-Linear LM with a constant-size per-layer fast-weight state."""

    def __init__(self, config: TTTLinearConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([TTTLinearCore(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
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

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_state(
        self, batch_size: int, *, device: torch.device | None = None
    ) -> list[torch.Tensor]:
        dev = device if device is not None else self.lm_head.weight.device
        sd = self.config.state_dim
        return [
            torch.zeros(batch_size, sd, sd, device=dev) for _ in range(self.config.n_layer)
        ]

    def state_bytes(self, state: list[torch.Tensor]) -> int:
        return sum(int(s.numel() * s.element_size()) for s in state)

    def step(
        self, idx_t: torch.Tensor, state: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if idx_t.ndim != 1:
            raise ValueError(f"idx_t must be 1-D [B], got shape {tuple(idx_t.shape)}")
        batch = idx_t.size(0)
        cur_state = self.init_state(batch, device=idx_t.device) if state is None else state
        h = self.transformer["drop"](self.transformer["wte"](idx_t))
        next_state: list[torch.Tensor] = []
        for li, layer in enumerate(cast(nn.ModuleList, self.transformer["h"])):
            assert isinstance(layer, TTTLinearCore)
            h, s = layer.step(h, cur_state[li])
            next_state.append(s)
        h = self.transformer["ln_f"](h)
        logits = cast(torch.Tensor, self.lm_head(h))
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
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")
        state = self.init_state(idx.size(0), device=idx.device)
        logits_steps: list[torch.Tensor] = []
        for pos in range(t):
            step_logits, state = self.step(idx[:, pos], state)
            logits_steps.append(step_logits.unsqueeze(1))
        logits = torch.cat(logits_steps, dim=1)
        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def streaming_nll(self, ids: torch.Tensor, chunk_size: int = 256) -> tuple[float, int]:
        """Mean next-token cross-entropy (nats) over a 1-D sequence of any length (O(1) state)."""
        if ids.ndim != 1:
            raise ValueError(f"ids must be 1-D [T], got shape {tuple(ids.shape)}")
        n = int(ids.size(0))
        if n < 2:
            raise ValueError(f"streaming_nll needs >= 2 tokens, got {n}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        was_training = self.training
        self.eval()
        state: list[torch.Tensor] | None = None
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
        batch, prompt_len = idx.shape
        state = self.init_state(batch, device=idx.device)
        last_logits: torch.Tensor | None = None
        for pos in range(prompt_len):
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
