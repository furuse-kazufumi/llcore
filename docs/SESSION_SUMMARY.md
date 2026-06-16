# Session Summary

- **更新時刻**: 2026-06-17
- **プロジェクト**: `D:/projects/llcore`
- **ブランチ**: `feat/lm-recurrent`

## 現在地点

- 主作業は 2 束:
  - `scripts/memory_footprint_harness.py` と [tests/unit/test_memory_footprint_harness.py](D:/projects/llcore/tests/unit/test_memory_footprint_harness.py)
  - Kaggle local/offload 導線 4 本:
    - [scripts/build_kaggle_lm_compare_bundle.py](D:/projects/llcore/scripts/build_kaggle_lm_compare_bundle.py)
    - [scripts/kaggle_bundle_preflight.py](D:/projects/llcore/scripts/kaggle_bundle_preflight.py)
    - [scripts/prepare_kaggle_lm_compare_bundle.py](D:/projects/llcore/scripts/prepare_kaggle_lm_compare_bundle.py)
    - [scripts/kaggle_push_readiness.py](D:/projects/llcore/scripts/kaggle_push_readiness.py)
- Kaggle focused 回帰の最新:
  - builder / preflight / prepare / readiness の **4 ファイル集合**で **`50 passed`**
- Kaggle を含む broad gate の最新:
  - `py -3.11 -m pytest tests/unit -k "lm or corpus or probe or p1_compare or p1_prepare or p1_manifest_reconcile or memory_footprint or kaggle_lm_compare_bundle or kaggle_bundle_preflight or prepare_kaggle_lm_compare_bundle or kaggle_push_readiness" -q`
  - **`262 passed, 401 deselected`**
- live 実測は **3 本**ある。旧 auth failure path (`rc=3`) は OAuth 依存実装時の記録で、現行実装では **CPU bundle の ready path (`rc=0`)** も `C:\Users\puruy\AppData\Local\Temp\llcore_kaggle_livecheck_20260617b` で確認済み。さらに GPU bundle `C:\Users\puruy\AppData\Local\Temp\llcore_kaggle_gpu_livecheck_20260617` でも readiness を試したが、ローカル **Kaggle CLI 2.2.1 の `quota` サブコマンド自体が `not enough values to unpack (expected 2, got 1)` で失敗**し、`rc=4` へ落ちた。現行 auth は OAuth ではなく **push credential (`kaggle.json` または `KAGGLE_USERNAME`+`KAGGLE_KEY`) + `kaggle kernels list -m --page-size 1 --csv` 疎通**で見る。したがって GPU quota live path は「未実施」ではなく **local CLI failure で確認不能**、実 `kaggle kernels push` は未実施。
- 外部 publish は **0 件**。

## 現在の git status

```text
(clean after local commits `af90dd6` and `1d1234a`)
```

- 再開時点では `docs/LM_RECURRENT_PLAN.md` も **dirty** だった。もし他の記録でこのファイルが脱落していたら、その記述は stale とみなす。現在は local commit 済み。

## Kaggle 導線の現契約

- builder の safe default は **private + internet disabled + GPU disabled + TPU disabled**
- preflight は corpus / `src/llcore` / `runner.py` / `config.json` の hash 再計算、metadata↔manifest 整合、`copied_files`、runner sidecar 再生成まで見る
- readiness は:
  - `rc=2` = local validation / preflight failure / TPU unsupported
  - `rc=3` = auth failure
  - `rc=4` = quota failure
- CPU bundle は **quota check 自体を skip** する
- GPU bundle は **GPU-like row (`"gpu"` substring match) の残量のみ**で判定
- `enable_tpu=true` bundle は readiness 未対応なので **clean `rc=2` reject**

## 次の具体的な一手

1. ローカル整理は完了。
   - `af90dd6` = memory harness
   - `1d1234a` = Kaggle 導線 + docs
2. `docs/next_plan.md` の再開追記を正本として、CPU ready path `rc=0` と GPU quota live path の CLI 2.2.1 failure を前提に次の人間ゲート準備へ進む。
3. ここまでは自律で可。`kaggle kernels push` に進む段になったら、直前に `docs/next_plan.md` を更新してから **`⟦LLTERM_CHOICE⟧`** で人間確認へ切り替える。

## 補足

- `docs/SESSION_SUMMARY.md` は本来 hook 更新物なので、**再開の正本は `docs/next_plan.md`**。
- 旧 `out/kaggle_lm_compare_smoke` は `source_sha256` 追加前の古い manifest 世代で、current preflight の正本には使わない。
- 再開後の focused gate 再実行は **memory harness を含む 5 test ファイル集合で `57 passed`**、対応 `mypy` / `ruff` も通過。
- fresh bundle は `C:\Users\puruy\AppData\Local\Temp\llcore_kaggle_livecheck_20260617b` に再生成し、前回と同じ manifest hash 群で `kaggle_push_readiness.py` の **CPU ready path `rc=0`** を確認した。CPU bundle なので live quota は skip され、`quota_rows=0 quota_checked=cpu` を表示した。別途 GPU bundle `C:\Users\puruy\AppData\Local\Temp\llcore_kaggle_gpu_livecheck_20260617` では readiness を試したが、`kaggle quota -v` が **CLI 2.2.1 側で `not enough values to unpack (expected 2, got 1)`** となり `rc=4` で停止した。実 `kaggle kernels push` 自体は未実施。
