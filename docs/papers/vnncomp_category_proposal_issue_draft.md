<!--
B1: VNN-COMP rules-discussion 投稿用ドラフト (online-arch-evo カテゴリ提案)。
- 投稿先候補: https://github.com/VNN-COMP/vnncomp2026/issues (rules discussion)
- 投稿者: ユーザー (リクエスト承認後)。英語で投稿。
- 方針: honest framing (提案・未査読・骨組み実装・契約拡張の議論。"うちが最強"主張はしない)。
- llcore repo が未公開のため URL は [TO ADD] のまま。公開後に差し替え。
- 下部に日本語ガイドと投稿チェックリスト。
-->

# 投稿用本文（英語・そのままコピペ可）

---

**Title:** RFC: a new category for *online architecture-evolution* verification (verifying a stream of `ChangeOp` mutations against a fixed invariant)

Hi all,

I'd like to open a discussion (not a formal motion yet) about a possible **new benchmark category** for a future edition, tentatively `online-arch-evo`. I'm a benchmark proposer, not a tool participant, and I'd value the community's view on whether this fits VNN-COMP's scope.

**The gap.** Every current VNN-COMP instance is *fixed-network*: one `.onnx` + one `.vnnlib`, asking "does this network satisfy this property?". A growing line of work mutates a network's *topology* many times during a single run (NAS, AutoML-Zero, weight-agnostic nets, and small-compute architecture-evolution systems). For these, the question shifts from "is *this* network safe?" to **"as a stream of mutations is applied, can each mutation be *incrementally* certified to preserve a fixed safety invariant, within a bounded per-step time budget, with a refuse/rollback option when the proof fails?"**

This is close to incremental NN verification (e.g., IVAN, PLDI 2023) but generalizes it from a *single* network update to an *unbounded* mutation stream. Three concrete queries motivate it:

- **A (commit gate):** is one mutation safe to apply *before* committing it?
- **B (survival curve):** over a mutation budget, at which step does the invariant first break?
- **C (family soundness):** does a whole *family* of mutations preserve the invariant?

**Why I think this is a category, not just a benchmark.** These queries don't seem expressible under the current one-network-per-call I/O contract — it's a *contract mismatch*, not a limitation of any tool (α,β-CROWN and others are excellent at the fixed-network task). So rather than dropping `.onnx`/`.vnnlib` pairs into an existing category, I think it needs a small extension to the input contract (a per-step `ChangeOp` stream + per-step verdicts under a wall-clock budget). I'd like feedback on whether the organizers/community see this as in-scope.

**What exists today (honest status).** I have a **draft specification** (file formats, scoring, per-step budget, plus mechanisms intended to keep the category open-ended across years) and a **minimal reference prototype** (a scalar-state Z3 invariant verifier, ~6 ms/step, that emits independently-auditable `unsat` witnesses). I want to be upfront that it is an **early, non-peer-reviewed proposal**: the prototype is a skeleton — it does **not yet parse real `.onnx`/`.vnnlib`** (I'm working on that now), `sat`-witness emission is not done yet, and there are no production benchmark instances yet. So this is a *call for discussion*, not a ready-to-vote benchmark.

**What I'm asking.**
1. Is "online architecture-evolution verification" something the community considers in-scope for a future edition?
2. If so, what's the right path — a new category, an extended track, or a contribution short paper to SAIV first?
3. Any prior art or concerns I should fold in before drafting a concrete benchmark repo (`generate_properties.py` + `.onnx` + `.vnnlib`)?

Happy to share the full draft proposal and the reference implementation. Thanks for considering it!

— [氏名 / 連絡先]
Draft proposal & reference implementation: [LINKS TO ADD ONCE THE REPO IS PUBLIC]

---

# 日本語ガイド（投稿前に読む）

## これは何
VNN-COMP の **rules discussion**（GitHub issue）に投げる、`online-arch-evo` カテゴリの **RFC（議論の呼びかけ）**。いきなり正式な motion（ルール変更動議）ではなく、「**そもそもこれは VNN-COMP の範囲に合うか**」をコミュニティに問う柔らかい入口にしている。

## なぜこのトーンか（honest framing）
- 「うちのツールが最強」とは**言っていない**。α,β-CROWN 等を立てつつ「**問題の出し方（契約）に枠が無い**」という論点に絞っている。
- **未査読・骨組み実装・benchmark 未完成**を正直に明記（VNN-COMP は名誉規定が厳しく、誇張は逆効果。正直な早期提案は歓迎される文化）。
- 「新カテゴリか / 拡張トラックか / まず SAIV short paper か」を**相手に委ねる**形にして、コミュニティ駆動の流儀に合わせた。

## 投稿前チェックリスト
1. **`— [氏名 / 連絡先]`** を埋める（前回の署名テンプレ参照。例: `Kazufumi Furuse / kazufumi@furuse.work`）。
2. **`[LINKS TO ADD ...]`**: llcore repo は未公開。**公開してから**論文(`vnn_comp_online_arch_evolution_proposal.md`)と reference impl(`scripts/poc_7a_vnn_comp_reference_impl.py`)のリンクを貼る。公開前に投稿するなら「論文ドラフトは要請があれば共有します」に留める（現文面は既にそうなっている）。
3. **投稿先**: まず議論なので `vnncomp2026/issues` の rules-discussion に新規 issue。タイトルは上記 `RFC: ...` のまま可。
4. **タイミング**: 2026 分の benchmark 締切は終了済み。これは **2027 サイクルに向けた地ならし**。焦らず、ML リスト参加が通って空気を読んでから投げてよい。
5. **motion ではない**: 文中「not a formal motion yet」と明記済み。正式なルール変更が必要になったら、後で具体的 motion（「この文をこう変える」）を別途出す。

## 投稿後の想定
- 反応（in-scope か / 既存 track で十分では / prior art の指摘）に応じて、(a) 具体 benchmark repo の準備（B2: パーサ実装が前提）、(b) SAIV short paper 化、へ分岐。
- 関連: [[project_llcore_init_2026_05_29]] の §7a、`vnncomp2026_rules_かみくだき.md`（誰でも提案可・契約不一致でルール議論経路）。
