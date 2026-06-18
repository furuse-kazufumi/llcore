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
    held_out_top1_report,
    passes_capability_gate,
    passes_gate,
    unigram_nll,
    unigram_perplexity,
)
from llcore.lm.export import save_viz_json, to_viz_dict
from llcore.lm.quant import (
    Int8Linear,
    convert_linears_to_int8,
    int8_footprint_bytes,
    load_int8_model,
    quantize_per_channel_int8,
    save_int8_checkpoint,
)
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
    "Int8Linear",
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
    "convert_linears_to_int8",
    "get_batch",
    "gpt_kv_bytes",
    "held_out_nll",
    "held_out_perplexity",
    "held_out_report",
    "held_out_report_any",
    "held_out_top1_report",
    "int8_footprint_bytes",
    "is_degenerate",
    "load_int8_model",
    "passes_capability_gate",
    "passes_gate",
    "quantize_per_channel_int8",
    "save_int8_checkpoint",
    "save_viz_json",
    "to_viz_dict",
    "train_val_split",
    "unigram_nll",
    "unigram_perplexity",
]
