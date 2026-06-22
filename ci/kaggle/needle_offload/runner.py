"""Kaggle CPU offload — NAS proxy-v2 long-context needle/passkey + 2048 sweep.

Why Kaggle CPU (not GPU): llcore's model loader pins the model to CPU
(`loader.py: model.to_empty(device="cpu")`, custom Int8Linear path), so a GPU
instance would sit idle. The bottleneck for the 2048/4096 full-attention forward
is RAM, not compute: the author's local box (3.6 GB) thrashes and the GitHub
Actions runner (7 GB, 350-min job cap) timed out mid-needle on run-2
(27918958686). A Kaggle CPU notebook has ~30 GB RAM and a 12 h runtime, which
clears the ~3.9 GB working set with huge margin and has time for the FULL needle
sweep (2048 AND 4096) that the GH lite contingency had to drop.

Faithful port of `.github/workflows/nas-needle-offload.yml`:
  resume from the committed eval_cache snapshot (no 6.6 h GA rerun) → rigorous
  tier + 2048 context sweep + needle at 2048,4096 → emit nas_pareto.json.

Output: /kaggle/working/nas_pareto.json (+ report + log) for `kaggle kernels output`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO = "https://github.com/furuse-kazufumi/llcore.git"
BRANCH = "feat/lm-recurrent"
WORK = "/kaggle/working"


def run(cmd: list[str], **kw: object) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)  # type: ignore[arg-type]


def main() -> int:
    os.chdir(WORK)
    # 1. Get the code + fixtures (shallow clone of the branch that ran on GH).
    if not os.path.isdir("llcore"):
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, "llcore"])
    repo = os.path.join(WORK, "llcore")
    os.chdir(repo)

    # 2. CPU torch + llcore + transformers (mirrors the GH workflow exactly).
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cpu"])
    run([sys.executable, "-m", "pip", "install", "-e", "."])
    run([sys.executable, "-m", "pip", "install",
         "transformers>=4.46", "safetensors", "huggingface_hub[cli]"])

    # 3. Model basename must match the eval_cache meta for cross-machine resume.
    run(["hf", "download", "Qwen/Qwen2.5-0.5B-Instruct",
         "--local-dir", "model/Qwen2.5-0.5B-Instruct"])

    # 4. Stage corpus prefix + resumable eval_cache snapshot.
    os.makedirs("out/nas_pareto_v2full", exist_ok=True)
    shutil.copy("ci/fixtures/corpus_aozora_multi.txt", "out/corpus_aozora_multi.txt")
    shutil.copy("ci/fixtures/eval_cache.json", "out/nas_pareto_v2full/eval_cache.json")

    # 5. Rigorous tier + needle (2048 AND 4096) + 2048 sweep, GA resumed.
    log = os.path.join(WORK, "run_offload.log")
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [sys.executable, "scripts/nas_pareto.py",
             "--proxy-v2",
             "--model-dir", "model/Qwen2.5-0.5B-Instruct",
             "--text-file", "out/corpus_aozora_multi.txt",
             "--out", "out/nas_pareto_v2full",
             "--context-sweep", "256,512,1024,2048",
             "--needle", "--needle-lengths", "2048,4096"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        rc = proc.wait()
    if rc != 0:
        print(f"::error:: nas_pareto.py exited {rc}", flush=True)

    # 6. Assert GA was resumed (never silently rerun the 26 h search).
    with open(log, encoding="utf-8") as fh:
        resumed = "[resume]" in fh.read()
    if not resumed:
        print("::error:: eval_cache did NOT resume — meta mismatch.", flush=True)

    # 7. Honest-disclosure report (best-effort).
    try:
        run([sys.executable, "scripts/nas_pareto_report.py",
             "out/nas_pareto_v2full/nas_pareto.json",
             "-o", "out/nas_pareto_v2full/nas_pareto_report.md"])
    except Exception as exc:  # noqa: BLE001
        print(f"report skipped: {exc}", flush=True)

    # 8. Surface artifacts to the kernel output root.
    for src in ("out/nas_pareto_v2full/nas_pareto.json",
                "out/nas_pareto_v2full/nas_pareto_report.md"):
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(WORK, os.path.basename(src)))

    return rc if rc != 0 else (0 if resumed else 2)


if __name__ == "__main__":
    sys.exit(main())
