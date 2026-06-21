# CI fixtures — NAS needle/2048 offload

`.github/workflows/nas-needle-offload.yml` 用の最小 fixture。

- **`corpus_aozora_multi.txt`** — 本物の `out/corpus_aozora_multi.txt`(青空文庫 PD マルチ作品, 9.8MB)の**先頭 200,000 文字プレフィックス**(~580KB)。これだけで (1) `base_nll` を厳密再現(base は `text[50000:][:256 tok]`)し、(2) rigorous tier + 2048 sweep + needle が使う全ウィンドウ(最大トークン index ~32,768; プレフィックスは 50k skip 後 ~171,918 トークン)を切り出せる。BPE は左から決定的なので、当該位置のトークンは full corpus と一致する。
- **`eval_cache.json`** — overnight 走(386 evals, commit 2957d1a 系)の GA スナップショット(scalar 200 + vector 200… 復元時 386/386)。これを resume することで CI は GA 6.6h をスキップし rigorous tier + needle のみ実行する。`meta.base_nll=4.415451`、`model_dir`/`text_file` は basename 比較・base_nll は 1e-3 tolerance で cross-platform 一致(`eval_cache_io._meta_matches`)。

provenance: 2026-06-21, `D:/models/Qwen2.5-0.5B-Instruct`(= HF `Qwen/Qwen2.5-0.5B-Instruct`)。
