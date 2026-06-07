# SPDX-License-Identifier: Apache-2.0
"""RAD (Research Aggregation Directory) lookup — llcore standalone path.

llive / raptor が管理する RAD コーパス (`<rad_dir>/<domain>_corpus_v2/`,
21+ 分野 / 約 48,800 documents) を **llive 非依存**で参照する薄い helper。
これにより llcore は llmesh-llive optional-dep なしで「進化中個体に先行研究
hint を注入する」研究フレームワークとして単独運用できる。

設計判断:
- stdlib のみ依存 (依存爆発回避)
- 環境変数 ``LLCORE_RAD_DIR`` で RAD 根ディレクトリを設定 (default ``~/.llcore/rad``; 不在なら graceful degrade)
- corpus2skill 階層 (cluster_*/SKILL.md) を理解せず、markdown ファイルを path/regex
  で直接 grep する素朴な API。階層理解は将来 (PoC 3 以降) の課題。
- 結果は :class:`RADHit` (path + snippet) のリスト。実コード時は ``read_doc()`` で本文取得。

将来 (PoC 3+):
- corpus2skill SKILL.md の階層を辿る higher-quality hint
- 進化中個体 fitness に「RAD 先行研究との overlap 度」を 1 軸として追加

References:
- llive ``project_corpus2skill`` / ``project_rad_expansion_2026_05``
- raptor ``.claude/skills/rad-research/SKILL.md`` (auto-trigger 規約)
"""

from .lookup import RADHit, list_domains, read_doc, search

__all__ = ["RADHit", "list_domains", "read_doc", "search"]
