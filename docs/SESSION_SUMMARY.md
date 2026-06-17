# Session Summary (auto-generated)

> 自動生成: `libexec/raptor-auto-summary` (Stop hook)
> 次回 ccr 起動時に CLAUDE.md SESSION START で自動的に読み取られる。

- **最終更新**: 2026-06-17 12:29:41
- **プロジェクト**: `D:/projects/llcore`
- **ブランチ**: `feat/lm-recurrent`

## 直近の git log

```
5d9f57d Harden Kaggle publish safety scans
f1ea165 Record live Kaggle publish gate evidence
445b503 Clarify canonical Kaggle publish candidate
84fd9cb Record latest Kaggle payload provenance
0e1c655 Fail closed on unknown Kaggle payload entries
d2974bc Harden Kaggle preflight report provenance
```

## 現在の git status

```
M docs/SESSION_SUMMARY.md
 M docs/next_plan.md
 M scripts/build_kaggle_lm_compare_bundle.py
 M scripts/kaggle_bundle_preflight.py
 M scripts/prepare_kaggle_lm_compare_bundle.py
 M tests/unit/test_build_kaggle_lm_compare_bundle.py
 M tests/unit/test_kaggle_bundle_preflight.py
 M tests/unit/test_prepare_kaggle_lm_compare_bundle.py
```

## 監査メモ

- 現行 canonical Kaggle candidate は `D:/projects/llcore_kaggle_livecheck_20260617g`。
- 最新 hardening の未コミット差分:
  - `.kaggleignore` へ `preflight_report.json` / `prepare_report.json` を追加
  - root report の `bundle_dir` / publish command / runner stdout を `<bundle_dir>` ベースへ相対化
  - `.kaggleignore` negation を glob 含めて fail-closed 化
  - bundle root allowlist を name-only から type/no-symlink まで固定
  - root text file も publish safety scan に含める
- fresh evidence:
  - `D:/projects/llcore_kaggle_livecheck_20260617g/prepare_report.json`
  - `D:/projects/llcore_kaggle_livecheck_20260617g/preflight_report.json`
  - `D:/projects/llcore_kaggle_livecheck_20260617g_preflight_stdout.txt`
  - `D:/projects/llcore_kaggle_livecheck_20260617g_readiness_stdout.txt`
  - `D:/projects/llcore_kaggle_livecheck_20260617g_dataset_status.txt`
  - `D:/projects/llcore_kaggle_livecheck_20260617g_dataset_files.csv`
- live remote state:
  - `kaggle datasets status furusekazufumi/llcore-lm-compare-support` → `ready`
  - `kaggle datasets files ... --csv` は現 remote がまだ `LICENSE` / `NOTICE` / `config.json` / `dataset_payload_manifest.json` / `input_corpus.txt` の partial publish 状態であることを再確認
- 検証:
  - `py -3.11 -m pytest tests/unit/test_kaggle_bundle_preflight.py tests/unit/test_prepare_kaggle_lm_compare_bundle.py -q` → `55 passed`
  - `py -3.11 -m pytest tests/unit/test_build_kaggle_lm_compare_bundle.py tests/unit/test_kaggle_bundle_preflight.py tests/unit/test_prepare_kaggle_lm_compare_bundle.py tests/unit/test_kaggle_push_readiness.py -q` → `119 passed`
  - `py -3.11 -m ruff check scripts/build_kaggle_lm_compare_bundle.py scripts/kaggle_bundle_preflight.py scripts/prepare_kaggle_lm_compare_bundle.py tests/unit/test_build_kaggle_lm_compare_bundle.py tests/unit/test_kaggle_bundle_preflight.py tests/unit/test_prepare_kaggle_lm_compare_bundle.py`
  - `$env:MYPYPATH='D:\projects\llcore\src'; py -3.11 -m mypy scripts/build_kaggle_lm_compare_bundle.py scripts/kaggle_bundle_preflight.py scripts/prepare_kaggle_lm_compare_bundle.py tests/unit/test_build_kaggle_lm_compare_bundle.py tests/unit/test_kaggle_bundle_preflight.py tests/unit/test_prepare_kaggle_lm_compare_bundle.py`
- publish gate の次手:
  - 実行コマンドは `kaggle datasets version -p "D:\projects\llcore_kaggle_livecheck_20260617g\dataset_payload" --dir-mode zip -m "update dataset payload (llcore_kaggle_livecheck_20260617g)"`
  - ただし external publish は未実行で、人間ゲート必須

---

> このファイルは毎ターン自動上書きされます。**手動で書いた内容は失われます。**
> 永続化したいメモは `docs/PROGRESS.md` または `docs/NOTES.md` を使ってください。
