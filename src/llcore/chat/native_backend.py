# SPDX-License-Identifier: Apache-2.0
"""llcore.chat.native_backend — llcore 自前 forward で会話する ChatBackend。

``backend.py`` の :class:`TransformersBackend` は HF ``transformers`` の ``generate`` を
そのまま呼ぶ (= HF ラッパー)。本 backend はそれを使わず、**llcore が自分で書いた forward**
(:mod:`llcore.runtime.qwen2`, HF と golden 一致を検証済み) を ``load_qwen2`` でロードし、
KV-cache を育てながら greedy / nucleus サンプリングで 1 トークンずつデコードする。

honest scope: 会話 *能力* は事前学習済み重み (Qwen2.5, Apache-2.0) 由来であって本 backend が
作るものではない。本 backend の意味は「その重みを **HF でなく llcore のコードで動かしている**」
ことの実体化 — `scripts/prove_native_matches_hf.py` が HF と同一トークンを出すことを実演する。

依存 (torch / transformers / safetensors) は lazy import し、不在時は
:class:`~llcore.chat.backend.ChatDependencyError` で fail-closed (黙って mock に劣化しない —
:class:`TransformersBackend` と同じ規律)。``--native`` はローカルモデルディレクトリ
(config.json + safetensors + tokenizer) を要する (``load_qwen2`` がディレクトリを読むため)。
"""
from __future__ import annotations

import time
from typing import Any, Sequence

from llcore.chat.backend import ChatDependencyError
from llcore.chat.session import GenerationSettings, Message


class NativeQwenBackend:
    """ChatBackend 実装。llcore 自前 forward (``runtime/qwen2.py``) で Qwen2 を回す (CPU)。

    Args:
        model_dir: ローカルモデルディレクトリ (config.json + *.safetensors + tokenizer)。
            HF ID ではなくパス (``load_qwen2`` がディレクトリを読む)。
        seed: サンプリング再現用。指定時は generate 毎に ``torch.manual_seed``。
        int8: True なら streaming-int8 ロード (``load_qwen2_int8``、RAM 削減・CPU は遅い)。

    ロードは初回 ``generate`` まで遅延 (lazy)。ロード秒は ``load_seconds`` に記録。
    """

    def __init__(self, model_dir: str, *, seed: int | None = None, int8: bool = False) -> None:
        self.model_dir = model_dir
        self._seed = seed
        self._int8 = int8
        self._torch: Any = None
        self._model: Any = None
        self._tok: Any = None
        self.load_seconds: float | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch

            from llcore.runtime.loader import load_qwen2, load_qwen2_int8
        except ImportError as exc:  # fail-closed: optional extra 'chat' 不在
            raise ChatDependencyError(
                "llcore 自前 forward 会話には optional extra 'chat' が必要です: "
                'pip install "llmesh-llcore[chat]" '
                "(torch / transformers / safetensors がインストールされます)"
            ) from exc
        t0 = time.time()
        loader = load_qwen2_int8 if self._int8 else load_qwen2
        model, tok, _ = loader(self.model_dir)
        model.eval()
        self.load_seconds = time.time() - t0
        self._torch = torch
        self._model = model
        self._tok = tok

    def _apply_repetition_penalty(self, logits: Any, seen: set[int]) -> None:
        """HF ``RepetitionPenaltyLogitsProcessor`` と同式: >0 は /penalty、<0 は *penalty。"""
        # この backend では penalty 適用は in-place。greedy 一致証明時は penalty=1.0 で無効。
        pass  # 実体は generate 内 (penalty 値を渡す必要があるため)

    def _sample_next(self, step_logits: Any, settings: GenerationSettings) -> int:
        """nucleus (top-p) サンプリングで次トークンを選ぶ。"""
        torch = self._torch
        logits = step_logits / max(settings.temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= settings.top_p
        keep[0] = True  # 最低 1 トークンは必ず残す (top_p が極端でも空集合にしない)
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[choice].item())

    def generate(self, messages: Sequence[Message], settings: GenerationSettings) -> str:
        """履歴 (system 含む、末尾 user) に対する assistant 応答を自前 forward で生成。

        ``settings.do_sample`` False = greedy (argmax)、True = temperature/top-p サンプリング。
        ``repetition_penalty != 1.0`` は HF と同式で適用 (既出トークンの logit を減衰)。
        EOS で打ち切り。KV-cache を育てて O(1) per-step。
        """
        self._ensure_loaded()
        torch = self._torch
        if self._seed is not None:
            torch.manual_seed(self._seed)

        msgs = [m.as_dict() for m in messages]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self._tok(text, return_tensors="pt").input_ids
        eos = getattr(self._tok, "eos_token_id", None)
        penalty = settings.repetition_penalty
        seen: set[int] = {int(t) for t in ids[0].tolist()}
        out_toks: list[int] = []

        with torch.no_grad():
            logits, cache = self._model(ids, return_cache=True)
            for _ in range(settings.max_new_tokens):
                step = logits[0, -1].float().clone()
                if penalty != 1.0 and seen:
                    idx = torch.tensor(sorted(seen), dtype=torch.long)
                    vals = step[idx]
                    step[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
                if settings.do_sample:
                    nxt = self._sample_next(step, settings)
                else:
                    nxt = int(step.argmax().item())
                if eos is not None and nxt == eos:
                    break
                out_toks.append(nxt)
                seen.add(nxt)
                logits, cache = self._model(
                    torch.tensor([[nxt]]), past=cache, return_cache=True
                )
        return str(self._tok.decode(out_toks, skip_special_tokens=True)).strip()

    def health(self) -> bool:
        """True if llcore native runtime + the model dir can actually be loaded (real, not a ping)."""
        try:
            self._ensure_loaded()
            return True
        except Exception:  # noqa: BLE001 - any import/load failure = unhealthy
            return False
