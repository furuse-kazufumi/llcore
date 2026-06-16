# SPDX-License-Identifier: Apache-2.0
"""Tests for legacy log parsing in :mod:`scripts.p1_compare`."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from contextlib import redirect_stdout
import io
import torch

p1_compare = import_module("scripts.p1_compare")


def test_parse_log_prefers_new_arch_line(tmp_path: Path) -> None:
    log_path = tmp_path / "new_run.log"
    log_path.write_text(
        "[model] p1: 1,234 params L6 H4 D384 ctx256 dropout0.1\n"
        "[train] iter 10 train_loss 1.23 val_loss 1.11\n",
        encoding="utf-8",
    )

    parsed = p1_compare._parse_log(log_path)

    assert parsed["arch"] == {
        "config": "p1",
        "params": 1234,
        "L": 6,
        "H": 4,
        "D": 384,
        "ctx": 256,
        "dropout": 0.1,
    }
    assert parsed["traj"] == [{"iter": 10, "train": 1.23, "val": 1.11}]


def test_parse_log_falls_back_to_legacy_cfg_in_any_key_order(tmp_path: Path) -> None:
    log_path = tmp_path / "legacy_run.log"
    log_path.write_text(
        "[model] smoke: 12,345 params  cfg={'dropout': 0, 'block_size': 64, "
        "'n_embd': 128, 'n_layer': 4, 'n_head': 4}\n",
        encoding="utf-8",
    )

    parsed = p1_compare._parse_log(log_path)

    assert parsed["arch"] == {
        "config": "smoke",
        "params": 12345,
        "L": 4,
        "H": 4,
        "D": 128,
        "ctx": 64,
        "dropout": 0.0,
    }


def test_parse_log_returns_arch_none_for_eval_log_without_model_line(tmp_path: Path) -> None:
    log_path = tmp_path / "eval_only.log"
    log_path.write_text(
        "[train] iter 100 train_loss 1.50 val_loss 1.40\n",
        encoding="utf-8",
    )

    parsed = p1_compare._parse_log(log_path)

    assert parsed["arch"] is None
    assert parsed["traj"] == [{"iter": 100, "train": 1.5, "val": 1.4}]


def test_parse_log_legacy_cfg_accepts_decimal_dropout(tmp_path: Path) -> None:
    log_path = tmp_path / "legacy_dropout.log"
    log_path.write_text(
        "[model] smoke: 12,345 params cfg={'n_head': 4, 'dropout': 0.0, "
        "'n_layer': 4, 'block_size': 64, 'n_embd': 128}\n",
        encoding="utf-8",
    )

    parsed = p1_compare._parse_log(log_path)

    assert parsed["arch"]["dropout"] == 0.0


def test_parse_log_legacy_cfg_allows_tab_after_model_prefix(tmp_path: Path) -> None:
    log_path = tmp_path / "legacy_tab.log"
    log_path.write_text(
        "[model]\tsmoke: 12,345 params cfg={'n_head': 4, 'dropout': 0.0, "
        "'n_layer': 4, 'block_size': 64, 'n_embd': 128}\n",
        encoding="utf-8",
    )

    parsed = p1_compare._parse_log(log_path)

    assert parsed["arch"] == {
        "config": "smoke",
        "params": 12345,
        "L": 4,
        "H": 4,
        "D": 128,
        "ctx": 64,
        "dropout": 0.0,
    }


def test_parse_log_legacy_cfg_requires_all_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "legacy_partial.log"
    log_path.write_text(
        "[model] smoke: 12,345 params cfg={'n_head': 4, 'dropout': 0.0, "
        "'n_layer': 4, 'block_size': 64}\n",
        encoding="utf-8",
    )

    parsed = p1_compare._parse_log(log_path)

    assert parsed["arch"] == {
        "config": "smoke",
        "params": 12345,
        "L": None,
        "H": None,
        "D": None,
        "ctx": None,
        "dropout": None,
    }


def test_parse_log_does_not_use_cfg_without_model_line(tmp_path: Path) -> None:
    log_path = tmp_path / "cfg_without_model.log"
    log_path.write_text(
        "cfg={'n_head': 4, 'dropout': 0.0, 'n_layer': 4, "
        "'block_size': 64, 'n_embd': 128}\n",
        encoding="utf-8",
    )

    parsed = p1_compare._parse_log(log_path)

    assert parsed["arch"] is None


def test_parse_log_nested_cfg_degrades_to_arch_na(tmp_path: Path) -> None:
    log_path = tmp_path / "legacy_nested_cfg.log"
    log_path.write_text(
        "[model] smoke: 12,345 params cfg={'n_layer': 4, 'n_head': 4, "
        "'extra': {'a': 1}, 'n_embd': 128, 'block_size': 64, 'dropout': 0.0}\n",
        encoding="utf-8",
    )

    parsed = p1_compare._parse_log(log_path)

    assert parsed["arch"] == {
        "config": "smoke",
        "params": 12345,
        "L": None,
        "H": None,
        "D": None,
        "ctx": None,
        "dropout": None,
    }


def test_markdown_table_handles_missing_params() -> None:
    table = p1_compare._markdown_table(
        [
            {
                "run": "eval_only",
                "arch": None,
                "verdict": {
                    "config": "eval",
                    "extra_corpus_files": [],
                    "max_iters": 0,
                    "n_eval_tokens": 128,
                    "unigram_ppl": 10.0,
                    "model_ppl": 9.0,
                    "ratio_model_over_unigram": 0.9,
                    "ppl_gate_pass": True,
                    "degenerate_sample": False,
                },
                "best_val": None,
                "final_val": None,
                "overfit_gap": None,
                "traj": [],
            }
        ]
    )

    assert "| ? |" in table
    assert "None" not in table


def test_markdown_table_handles_missing_optional_verdict_cells() -> None:
    table = p1_compare._markdown_table(
        [
            {
                "run": "missing_optional",
                "arch": None,
                "verdict": {
                    "config": "eval",
                    "extra_corpus_files": [],
                    "max_iters": 0,
                    "n_eval_tokens": 128,
                    "unigram_ppl": 10.0,
                },
                "best_val": None,
                "final_val": None,
                "overfit_gap": None,
                "traj": [],
            }
        ]
    )

    assert "| ? | ? | ? | ? | ? |" in table
    assert "FAIL" not in table


def test_main_skips_headline_when_ratio_missing(tmp_path: Path) -> None:
    run_base = tmp_path / "lm_aozora_drop"
    run_p1 = tmp_path / "lm_aozora_realp1"
    run_base.mkdir()
    run_p1.mkdir()

    (run_base / "verdict.json").write_text(
        '{"config":"smoke","max_iters":1,"n_eval_tokens":8,"unigram_ppl":10.0,"model_ppl":9.0}',
        encoding="utf-8",
    )
    (run_p1 / "verdict.json").write_text(
        '{"config":"p1","max_iters":1,"n_eval_tokens":8,"unigram_ppl":10.0,"model_ppl":8.5}',
        encoding="utf-8",
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = p1_compare.main([str(run_base), str(run_p1)])

    assert rc == 0
    assert "skip headline comparison" in buf.getvalue()


def test_main_writes_json_even_when_headline_ratio_missing(tmp_path: Path) -> None:
    run_base = tmp_path / "lm_aozora_drop"
    run_p1 = tmp_path / "lm_aozora_realp1"
    out_json = tmp_path / "out.json"
    run_base.mkdir()
    run_p1.mkdir()

    (run_base / "verdict.json").write_text(
        '{"config":"smoke","max_iters":1,"n_eval_tokens":8,"unigram_ppl":10.0,"model_ppl":9.0}',
        encoding="utf-8",
    )
    (run_p1 / "verdict.json").write_text(
        '{"config":"p1","max_iters":1,"n_eval_tokens":8,"unigram_ppl":10.0,"model_ppl":8.5}',
        encoding="utf-8",
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = p1_compare.main([str(run_base), str(run_p1), "--json", str(out_json)])

    assert rc == 0
    assert "skip headline comparison" in buf.getvalue()
    assert out_json.exists()


def test_gather_prefers_checkpoint_arch_over_stale_legacy_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "lm_aozora_drop"
    run_dir.mkdir()
    (run_dir / "verdict.json").write_text(
        '{"config":"smoke","max_iters":3500,"n_eval_tokens":128,"unigram_ppl":10.0,"model_ppl":9.0}',
        encoding="utf-8",
    )
    torch.save(
        {
            "config": {
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 128,
                "block_size": 64,
                "dropout": 0.1,
            },
            "model_state": {},
            "itos": ["a", "b"],
        },
        run_dir / "model.pt",
    )
    (tmp_path / "lm_aozora_drop_run.log").write_text(
        "[model] smoke: 12,345 params cfg={'n_layer': 4, 'n_head': 4, "
        "'n_embd': 128, 'block_size': 64, 'dropout': 0.0}\n"
        "[train] iter 10 train_loss 1.23 val_loss 1.11\n",
        encoding="utf-8",
    )

    gathered = p1_compare._gather(run_dir)

    assert gathered is not None
    assert gathered["arch"] == {
        "config": "smoke",
        "params": 12345,
        "L": 4,
        "H": 4,
        "D": 128,
        "ctx": 64,
        "dropout": 0.1,
    }


def test_gather_gracefully_falls_back_when_checkpoint_is_unreadable(tmp_path: Path) -> None:
    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    (run_dir / "verdict.json").write_text(
        '{"config":"smoke","extra_corpus_files":[],"max_iters":1,"n_eval_tokens":8,"unigram_ppl":10.0,"model_ppl":9.0}',
        encoding="utf-8",
    )
    (run_dir / "model.pt").write_bytes(b"not a torch checkpoint")
    (tmp_path / "broken_run.log").write_text(
        "[model] smoke: 12,345 params L4 H4 D128 ctx64 dropout0.1\n",
        encoding="utf-8",
    )

    gathered = p1_compare._gather(run_dir)

    assert gathered is not None
    assert gathered["arch"]["config"] == "smoke"
    assert gathered["arch"]["dropout"] == 0.1


def test_markdown_table_shows_extra_corpus_count() -> None:
    table = p1_compare._markdown_table(
        [
            {
                "run": "with_extra",
                "arch": {
                    "config": "smoke",
                    "params": 12345,
                    "L": 4,
                    "H": 4,
                    "D": 128,
                    "ctx": 64,
                    "dropout": 0.1,
                },
                "verdict": {
                    "config": "smoke",
                    "extra_corpus_files": ["a.txt", "b.txt"],
                    "max_iters": 2,
                    "n_eval_tokens": 8,
                    "unigram_ppl": 10.0,
                    "model_ppl": 9.0,
                    "ratio_model_over_unigram": 0.9,
                    "ppl_gate_pass": True,
                    "degenerate_sample": False,
                },
                "best_val": 1.0,
                "final_val": 1.1,
                "overfit_gap": 0.1,
                "traj": [],
            }
        ]
    )

    assert "| with_extra | 2 | smoke L4H4D128 ctx64 drop0.1 |" in table
