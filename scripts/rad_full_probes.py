# SPDX-License-Identifier: Apache-2.0
"""RAD 全量取込後の世界知識 retrieval 測定 — 事前登録 probe (2026-06-12 固定)。

ROADMAP M3 残項目「世界知識注入が retrieval/grounding を改善するか実測」用。
**事前登録の規律**: 本ファイルのコミットが登録時点。全量取込の測定結果を見た後の
probe 追加・変更・削除は禁止 (cherry-pick 防止)。M3.0 の WORLD_PROBES (loop 18) /
会話 22 probe (connectivity_bench.PROBES) と合わせて使う。

作成方法 (honest 開示):
- Explore agent が各 corpus の実 doc をサンプル読みし、doc に verbatim 存在する
  キーワードを gold に、**doc の文の丸写しでない言い換え疑問文**を probe にした
  (表層一致での正解を防ぐ M3.0 の流儀)。
- gold の verbatim 存在は agent が grep で全件検証済み。
- gold は小文字化して保持 (connectivity_bench.rank_of は hit.lower() に対する
  substring 判定のため)。
- 5 分野 × 6 probe = 30。分野は docs 数と probe の作りやすさ (明確な技術キーワード)
  で選定 — 全 51 分野の網羅でない点は限界として開示する。

出典 doc は各 probe のコメントに記録 (検証可能性)。
"""
from __future__ import annotations

FULL_PROBES: dict[str, list[tuple[str, list[str]]]] = {
    "astrophysics": [
        ("How can astronomers improve measurements of exoplanet atmospheres by "
         "reducing interference from the host star's light and activity?",
         ["stellar contamination", "transmission spectra"]),  # 2602.10330
        ("What force exerted by photons can significantly alter the orbital dynamics "
         "and design of spacecraft operating in deep space?",
         ["radiation pressure"]),  # 2602.10712
        ("What observational technique uses the periodic shifts in a star's position "
         "due to orbiting bodies to detect planets around other stars?",
         ["giant planet", "radial-velocity data"]),  # 2602.10919
        ("What periodic pattern in solar activity with a period near 1.7 years "
         "appears connected to gravitational influences from planets?",
         ["quasi-biennial oscillation"]),  # 2602.11227
        ("What key demographic feature distinguishes the population of small rocky "
         "worlds from larger gas-dominated exoplanets?",
         ["radius valley"]),  # 2602.11923
        ("What happens to planetary-mass objects that drift through space without "
         "orbiting a star as they encounter stellar gravitational fields?",
         ["free-floating planets", "gravitational scattering"]),  # 2602.12017
    ],
    "cryptography": [
        ("What security mechanism verifies that software changes in repositories "
         "were created by developers with authorized cryptographic keys?",
         ["commit signing"]),  # 2604.14014
        ("What infrastructure protections ensure that source code retains its "
         "integrity and authenticity as it moves through development and deployment?",
         ["software supply chains"]),  # 2604.14014
        ("What cryptographic protocol enables both servers and clients to "
         "authenticate each other using digital certificates?",
         ["mutual tls"]),  # 2604.14330
        ("What digital artifact must users install and configure on their devices "
         "to prove identity when accessing TLS-protected services?",
         ["client certificate"]),  # 2604.14330
        ("What static analysis technique tracks how confidential data flows through "
         "a program to detect improper information leakage?",
         ["information-flow control"]),  # 2604.14357
        ("What secure computation protocol allows two parties to jointly determine "
         "which items they both possess without revealing their full lists?",
         ["private set intersection"]),  # 2604.14909
    ],
    "compiler": [
        ("What programming language technique maintains type safety guarantees "
         "throughout the entire compilation process into low-level assembly?",
         ["typed assembly language"]),  # 2509.08727
        ("How must cryptographic code be written to avoid leaking secrets through "
         "measurable differences in execution duration?",
         ["constant-time"]),  # 2509.08727
        ("What compiler techniques speed up mathematical computations while "
         "preserving numerical accuracy in scientific simulations?",
         ["floating-point optimization"]),  # 2509.09019
        ("What pattern of nested loops in numerical code can be automatically "
         "identified and lifted into domain-specific language abstractions?",
         ["stencil kernel"]),  # 2509.10236
        ("What approach preserves source-level type information through compilation "
         "stages to enable verification at the assembly level?",
         ["type-preserving compilation"]),  # 2509.09059
        ("What properties must zero-knowledge virtual machines satisfy to prevent "
         "them from accepting invalid computations or rejecting valid ones?",
         ["soundness and completeness"]),  # 2509.10819
    ],
    "distributed_systems": [
        ("How can systems efficiently generate multiple media types like text, "
         "speech, and video simultaneously within strict latency bounds?",
         ["multi-modal", "real-time"]),  # 2603.05800
        ("What platform-level technique eliminates redundant function invocations "
         "by combining multiple independent functions into a single execution?",
         ["function fusion"]),  # 2603.06170
        ("How should computational resources be allocated across a datacenter to "
         "minimize communication latency in distributed machine learning?",
         ["device placement", "distributed training"]),  # 2603.06798
        ("What architectural approach allows a company to reduce duplicate "
         "infrastructure while maintaining service reliability during failures?",
         ["failover"]),  # 2603.07345
        ("What decentralized protocol allows distributed systems to maintain "
         "service discovery even when networks become partitioned?",
         ["gossip"]),  # 2603.07750
        ("How can neural networks with conditional layer selection be deployed "
         "across multiple accelerators without creating severe performance "
         "bottlenecks?",
         ["mixture-of-experts", "expert parallelism"]),  # 2603.06350
    ],
    "language": [
        ("What natural language processing task involves identifying when different "
         "expressions in a text refer to the same entity or individual?",
         ["coreference"]),  # semantics-pragmatics/coreference-resolution.md
        ("What linguistic phenomenon where subjects or objects can be entirely "
         "omitted is particularly common in East Asian languages?",
         ["zero pronoun"]),  # semantics-pragmatics/coreference-resolution.md
        ("What syntactic model represents sentence structure as directed connections "
         "between individual words rather than hierarchical phrase groups?",
         ["dependency grammar"]),  # syntax-grammar/dependency-grammar.md
        ("What acoustic feature of vowels corresponds to resonance peaks in the "
         "frequency spectrum produced by vocal tract shape?",
         ["formant"]),  # phonetics-phonology/acoustic-phonetics.md
        ("What classic dynamic programming algorithm efficiently selects the optimal "
         "word segmentation when multiple boundary options exist?",
         ["viterbi"]),  # morphology-lexicon/japanese-morphological-analysis.md
        ("What linguistic theory proposes that humans have an innate biological "
         "capacity to produce and understand infinite new sentences?",
         ["generative grammar"]),  # syntax-grammar/generative-grammar-xbar-minimalism.md
    ],
}

ALL_FULL_PROBES: list[tuple[str, list[str]]] = [
    p for probes in FULL_PROBES.values() for p in probes
]
