# SPDX-License-Identifier: Apache-2.0
"""Head-to-head utilities for GPT vs recurrent constant-state LMs."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import torch

from llcore.lm.data import encode_corpus, train_val_split
from llcore.lm.eval import held_out_report_any, passes_gate
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import TrainConfig, Trainer


ConstantStateLM: TypeAlias = RecurrentLM | RWKVLM


@dataclass
class CompareConfig:
    """Small comparison recipe intended for CPU smoke runs."""

    block_size: int = 64
    n_layer: int = 2
    n_embd: int = 64
    state_size: int = 64
    max_iters: int = 120
    batch_size: int = 12
    eval_iters: int = 4
    seed: int = 1337


def build_models(vocab_size: int, cfg: CompareConfig) -> tuple[CharGPT, RecurrentLM, RWKVLM]:
    gpt = CharGPT(
        GPTConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=max(1, cfg.n_embd // 32),
            n_embd=cfg.n_embd,
        )
    )
    recurrent = RecurrentLM(
        RecurrentConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_embd=cfg.n_embd,
            state_size=cfg.state_size,
        )
    )
    rwkv = RWKVLM(
        RWKVConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_embd=cfg.n_embd,
        )
    )
    return gpt, recurrent, rwkv


def gpt_kv_bytes(model: CharGPT, prompt_len: int) -> int:
    return 2 * model.config.n_layer * prompt_len * model.config.n_embd * 4


def constant_state_bytes(model: ConstantStateLM) -> int:
    if isinstance(model, RecurrentLM):
        recurrent_state = model.init_state(batch_size=1)
        return model.state_bytes(recurrent_state)
    rwkv_state = model.init_state(batch_size=1)
    return model.state_bytes(rwkv_state)


def compare_on_text(
    text: str,
    *,
    cfg: CompareConfig | None = None,
    out_path: Path | None = None,
) -> dict[str, object]:
    recipe = cfg or CompareConfig()
    torch.manual_seed(recipe.seed)
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=0.1)
    gpt, recurrent, rwkv = build_models(tok.vocab_size, recipe)
    train_cfg = TrainConfig(
        max_iters=recipe.max_iters,
        warmup_iters=max(1, recipe.max_iters // 10),
        lr_decay_iters=recipe.max_iters,
        batch_size=recipe.batch_size,
        eval_interval=recipe.max_iters,
        eval_iters=recipe.eval_iters,
        seed=recipe.seed,
    )

    Trainer(gpt, train_cfg).train(train_ids, val_ids)
    Trainer(recurrent, train_cfg).train(train_ids, val_ids)
    Trainer(rwkv, train_cfg).train(train_ids, val_ids)
    reports = {
        "gpt": held_out_report_any(gpt, train_ids, val_ids, tok.vocab_size, block_size=recipe.block_size),
        "recurrent": held_out_report_any(
            recurrent, train_ids, val_ids, tok.vocab_size, block_size=recipe.block_size
        ),
        "rwkv": held_out_report_any(rwkv, train_ids, val_ids, tok.vocab_size, block_size=recipe.block_size),
    }

    lengths = [1, 16, recipe.block_size, recipe.block_size * 4]
    memory = {
        "gpt_kv_bytes": {str(t): gpt_kv_bytes(gpt, t) for t in lengths},
        "recurrent_state_bytes": constant_state_bytes(recurrent),
        "rwkv_state_bytes": constant_state_bytes(rwkv),
    }
    result: dict[str, object] = {
        "config": asdict(recipe),
        "reports": reports,
        "memory": memory,
        "verdict": {
            name: {
                "passes_unigram_gate": passes_gate(report["model_ppl"], report["unigram_ppl"]),
                "ppl_ratio_vs_gpt": report["model_ppl"] / reports["gpt"]["model_ppl"],
            }
            for name, report in reports.items()
        },
    }
    if out_path is not None:
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
