# Session Summary (auto-generated)

> 自動生成: `libexec/raptor-auto-summary` (Stop hook)
> 次回 ccr 起動時に CLAUDE.md SESSION START で自動的に読み取られる。

- **最終更新**: 2026-06-17 13:27:37
- **プロジェクト**: `D:/projects/llcore`
- **ブランチ**: `feat/lm-recurrent`

## 直近の git log

```
e433a26 Record Kaggle dataset version result
0ab944b Harden Kaggle bundle root publish safety
5d9f57d Harden Kaggle publish safety scans
f1ea165 Record live Kaggle publish gate evidence
```

## 現在の git status

```
M docs/SESSION_SUMMARY.md
 M docs/next_plan.md
 M scripts/build_kaggle_lm_compare_bundle.py
 M scripts/kaggle_bundle_preflight.py
 M tests/unit/test_build_kaggle_lm_compare_bundle.py
```

## 監査メモ

- remote dataset publish は `kaggle datasets version ... --dir-mode zip -m "update dataset payload"` まで完了。
- publish 後 remote file list は `src_llcore/src/llcore/...` / `pkg_llcore/llcore/...` の **展開済みツリー**で、`.zip` 実体ではない。
- 新規 hardening:
  - `runner.py` は zip 実体と extracted tree の **dual-path** をサポート
  - extracted tree は `source_sha256` で検証し、`pkg_llcore` / `src_llcore/src` を直接 `sys.path` に追加
  - `preflight --run-runner` は dataset mode で remote mount 風 temp root を作って smoke
- actual remote download smoke:
  - dataset を `D:/projects/llcore_kaggle_remote_dataset_20260617g` へ `kaggle datasets download --unzip`
  - `LLCORE_KAGGLE_DATA_ROOT=D:/projects/llcore_kaggle_remote_dataset_20260617g py -3.11 D:/projects/llcore_kaggle_livecheck_20260617g/runner.py`
  - 成功ログ: `D:/projects/llcore_kaggle_livecheck_20260617g_remote_download_smoke.txt`
- fresh evidence:
  - `D:/projects/llcore_kaggle_livecheck_20260617g/prepare_report.json`
  - `D:/projects/llcore_kaggle_livecheck_20260617g/preflight_report.json`
  - `D:/projects/llcore_kaggle_livecheck_20260617g_preflight_stdout.txt`
  - `D:/projects/llcore_kaggle_livecheck_20260617g_readiness_stdout.txt`
  - `D:/projects/llcore_kaggle_livecheck_20260617g_dataset_status_post_version.txt`
  - `D:/projects/llcore_kaggle_livecheck_20260617g_dataset_files_post_version.csv`
- 検証:
  - `py -3.11 -m pytest tests/unit/test_build_kaggle_lm_compare_bundle.py tests/unit/test_kaggle_bundle_preflight.py tests/unit/test_prepare_kaggle_lm_compare_bundle.py tests/unit/test_kaggle_push_readiness.py -q` → `120 passed`
  - `py -3.11 -m ruff check scripts/build_kaggle_lm_compare_bundle.py scripts/kaggle_bundle_preflight.py tests/unit/test_build_kaggle_lm_compare_bundle.py tests/unit/test_kaggle_bundle_preflight.py tests/unit/test_prepare_kaggle_lm_compare_bundle.py`
  - `$env:MYPYPATH='D:\projects\llcore\src'; py -3.11 -m mypy scripts/build_kaggle_lm_compare_bundle.py scripts/kaggle_bundle_preflight.py tests/unit/test_build_kaggle_lm_compare_bundle.py tests/unit/test_kaggle_bundle_preflight.py tests/unit/test_prepare_kaggle_lm_compare_bundle.py`
- 次の不可逆操作候補:
  - `kaggle kernels push -p "D:\projects\llcore_kaggle_livecheck_20260617g"`
  - ただし human gate 必須

---

> このファイルは毎ターン自動上書きされます。**手動で書いた内容は失われます。**
> 永続化したいメモは `docs/PROGRESS.md` または `docs/NOTES.md` を使ってください。
