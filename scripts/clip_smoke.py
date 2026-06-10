# SPDX-License-Identifier: Apache-2.0
"""llcore.clip 実機スモーク — 合成画像 zero-shot + テキスト検索 (CPU, on-prem)。

検証項目:
  1. text 検索: クエリと意味的に近い文が cosine 順位で先頭に来るか
  2. zero-shot: PIL で合成した単純図形 (赤い正方形 / 青い円 / 緑の三角) を
     正しいラベルに割り当てられるか

honest 留保:
- 合成図形は自然画像と分布が違うため、CLIP/SigLIP が誤ることがある。配管検証が主目的で、
  失敗も verbatim 記録する (capability 主張ではない)。
- score は cosine 類似度 (較正確率ではない)。

使い方::

    py -3.11 scripts/clip_smoke.py [--model ID] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import ClipBackend, zero_shot  # noqa: E402

_ensure_utf8_stdout()


def make_shapes(tmp_dir: Path) -> dict[str, Path]:
    """単純図形の合成画像を生成 (224x224, 白背景)。"""
    from PIL import Image, ImageDraw

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    img = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(img).rectangle([56, 56, 168, 168], fill="red")
    out["a red square"] = tmp_dir / "red_square.png"
    img.save(out["a red square"])

    img = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(img).ellipse([56, 56, 168, 168], fill="blue")
    out["a blue circle"] = tmp_dir / "blue_circle.png"
    img.save(out["a blue circle"])

    img = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(img).polygon([(112, 48), (48, 176), (176, 176)], fill="green")
    out["a green triangle"] = tmp_dir / "green_triangle.png"
    img.save(out["a green triangle"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "clip_smoke_results.json",
    )
    args = parser.parse_args()

    backend = ClipBackend(model_id=args.model)
    results: dict[str, object] = {"model": backend.model_id}
    t_start = time.time()

    # 1. text 検索 sanity
    texts = ["a cat", "a dog", "an airplane", "a bowl of soup"]
    query = "a sleeping kitten"
    T = backend.encode_texts(texts)
    q = backend.encode_texts([query])
    sims = (q @ T.T)[0]
    order = sims.argsort()[::-1]
    ranking = [(texts[int(i)], float(sims[int(i)])) for i in order]
    text_ok = ranking[0][0] == "a cat"
    print(f'text retrieval (query="{query}"):', flush=True)
    for t, s in ranking:
        print(f"  {s:+.4f}  {t}", flush=True)
    print(f"  → top1='{ranking[0][0]}' ({'expected' if text_ok else 'unexpected'})", flush=True)
    results["text_retrieval"] = {"query": query, "ranking": ranking, "top1_expected": text_ok}

    # 2. 合成図形 zero-shot (3 画像 × 3 ラベル)
    labels = ["a red square", "a blue circle", "a green triangle"]
    shapes = make_shapes(Path(__file__).resolve().parents[1] / "out" / "clip_smoke_shapes")
    zs_results = []
    n_correct = 0
    for true_label, path in shapes.items():
        ranking = zero_shot(backend, path, labels, template="{}")
        top1 = ranking[0][0]
        ok = top1 == true_label
        n_correct += int(ok)
        print(f"zero-shot {path.name}: top1='{top1}' ({'correct' if ok else 'WRONG'})  "
              f"{[(lab, round(s, 4)) for lab, s in ranking]}", flush=True)
        zs_results.append({"true": true_label, "ranking": ranking, "correct": ok})
    results["zero_shot_shapes"] = {"n_correct": n_correct, "n_total": len(shapes), "cases": zs_results}
    results["load_seconds"] = backend.load_seconds
    results["total_seconds"] = round(time.time() - t_start, 1)
    results["note"] = (
        "合成図形は自然画像と分布が違うため誤りうる (配管検証が主目的、verbatim 記録が正)。"
        "score は cosine (較正確率でない)。"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary: text_top1={'ok' if text_ok else 'ng'}, shapes={n_correct}/{len(shapes)} "
          f"(load {backend.load_seconds:.1f}s)", flush=True)
    print(f"results: {args.out}", flush=True)
    # 配管が動いていれば 0 (text 検索 or 図形 1 つ以上正解)
    return 0 if (text_ok or n_correct > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
