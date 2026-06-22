"""Fallback offload -- 2048 context sweep ONLY (needle dropped entirely).

Escalation target for EXIT(53): if even the v3 retry (needle 2048-only +
sweep) zombies past the 12 h cap, the single remaining heavy leg is the
2048 full-attention *needle* forward. This variant drops `--needle`
altogether and keeps just the context sweep up to 2048, which alone still
anchors b2's long-context decay claim (Delta-nll 256->512->1024->2048,
L115/L137-138). needle stays honestly UNTESTED in the article.

To deploy: copy this over runner.py (or point kernel-metadata code_file
here) and `kaggle kernels push -p ci/kaggle/needle_offload`. Computation
offload is autonomously permitted; this is not a git push, so no human gate.

Faithful port of runner.py minus the needle leg. Output:
/kaggle/working/nas_pareto.json (+ report + log).
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

    # 5. Rigorous tier + 2048 sweep, GA resumed. NO needle (dropped to fit the
    #    12 h budget after the needle 2048 leg itself zombied).
    log = os.path.join(WORK, "run_offload.log")
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [sys.executable, "scripts/nas_pareto.py",
             "--proxy-v2",
             "--model-dir", "model/Qwen2.5-0.5B-Instruct",
             "--text-file", "out/corpus_aozora_multi.txt",
             "--out", "out/nas_pareto_v2full",
             "--context-sweep", "256,512,1024,2048"],
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
        print("::error:: eval_cache did NOT resume -- meta mismatch.", flush=True)

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
