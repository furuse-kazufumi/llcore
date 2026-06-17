# SPDX-License-Identifier: Apache-2.0
"""llcore.lm — CPU char-level Transformer language model (capability-first P0).

A small, from-scratch GPT-2 "nano" char-LM with a trainer, evaluation against a
char unigram baseline, sampling, and an exporter that produces bbycroft/llm-viz
compatible model JSON. CPU / ``torch.float32`` only. See ``docs/LM_P0_PLAN.md``.
"""
from __future__ import annotations

from llcore.lm.data import (
    encode_corpus,
    fetch_aozora_text,
    fetch_tinyshakespeare,
    get_batch,
    train_val_split,
)
from llcore.lm.compare import compare_on_text, gpt_kv_bytes
from llcore.lm.eval import (
    held_out_nll,
    held_out_perplexity,
    held_out_report,
    held_out_report_any,
    passes_gate,
    unigram_nll,
    unigram_perplexity,
)
from llcore.lm.export import save_viz_json, to_viz_dict
from llcore.lm.generation import generate_text, is_degenerate
from llcore.lm.model import (
    Block,
    CausalSelfAttention,
    CharGPT,
    GPTConfig,
    NewGELU,
)
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM
from llcore.lm_compare_config import CompareConfig
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import TrainConfig, Trainer

__all__ = [
    "Block",
    "CausalSelfAttention",
    "CharGPT",
    "CharTokenizer",
    "CompareConfig",
    "GPTConfig",
    "NewGELU",
    "RecurrentConfig",
    "RecurrentLM",
    "RWKVConfig",
    "RWKVLM",
    "TrainConfig",
    "Trainer",
    "encode_corpus",
    "fetch_aozora_text",
    "fetch_tinyshakespeare",
    "generate_text",
    "compare_on_text",
    "get_batch",
    "gpt_kv_bytes",
    "held_out_nll",
    "held_out_perplexity",
    "held_out_report",
    "held_out_report_any",
    "is_degenerate",
    "passes_gate",
    "save_viz_json",
    "to_viz_dict",
    "train_val_split",
    "unigram_nll",
    "unigram_perplexity",
]
