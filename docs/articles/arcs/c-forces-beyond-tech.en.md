# The Dominant Terms Beyond Technology — License, Growth, and Memory

When we talk about LLMs, our attention is inevitably drawn to technical metrics like "intelligence" and "speed." Yet watching the industry through the first half of 2026 from a home CPU, what I was reminded of again and again was something else entirely. **What decides a product's fate is often not the technology itself, but the dominant term that sits "outside" the technology** — that was the lesson.

A dominant term is the term that most strongly governs the outcome. You can double performance, but if the commercial license doesn't fit, you can't adopt it. You can speed up the learning loop, but without a mechanism to stop errors, the thing "degrades the more you use it." You can refine the memory machinery, but if you mistake which domain to fold what into, you won't get consistency. This piece (part of the "The LLM Industry of June 2026, Seen from a Home CPU" series, ecosystem edition) tours those "dominant terms beyond technology" in three stages: **license → growth → memory**.

---

> ### Shared Disclaimer (a banner that applies to the entire series)
>
> **The measured figures shown for the series' centerpiece (llcore) are all honest disclosures of a tiny-model PoC running on a home CPU.** This is not a story about directly refuting or surpassing the major players' real LLMs. There are moments in this piece where I compare against major players, other companies, and other projects, but what I am comparing is **the dimension of method, philosophy, and measurement discipline** — not the superiority or inferiority of absolute performance. And this piece is **neither legal advice nor investment advice.** All references to other companies (Google / NVIDIA / Baidu / Nous Research / the MangaFlow authors) reflect my understanding as of June 2026 from public information, and figures come with their sources cited. As is this series' practice, **the blade I turn on others I always, in the end, turn on myself (FullSense / llcore / llive) as well.**

Let me define, just once here, the shared terms that recur in this piece (I won't repeat them in each chapter).

- **honest disclosure**: the discipline of, before claiming "it worked," doubting the breakdown and disclosing — without hiding — the unproven parts, the conditions, and the counterexamples. It is the backbone of this series. When an abnormally good number appears, you must decompose it before letting yourself feel victorious.
- **OSI-certified OSS**: a license that satisfies the Open Source Initiative's definition (free use, modification, and redistribution, no discrimination against fields of endeavor, etc.). Apache 2.0 and MIT are representative. Among those that call themselves an "open model license," some **do not** satisfy it.
- **constant state**: a scheme that, no matter how much of the past you read, folds it into a fixed-size "summary" to carry forward. It is the recurrent / SSM-family idea where the size of the state does not grow even as the text gets longer. I treat this in detail in Chapter 3.

---

# Chapter 1 — Open-Weights Licenses Split into Three Kinds

## A single word, "open," does not settle whether commercial use is allowed

In the first half of 2026, releases of notable open models came one after another (Gemma 4 in April, Cosmos 3 / PaddleOCR in June). What's interesting is that **the licenses split into three kinds.** Lumping them together as "open (public weights)" makes you miss this difference.

"open weights" means nothing more than "**the weights can be downloaded.**" "Usable commercially," "you can freely ship derivatives," "it's classic OSI-certified OSS" are **separate matters.** Even if the weights can be downloaded, some have conditions attached to commercial use, and some have constraints on derivatives. So instead of "open, so I'm safe," you need to discern **which kind of open** it is.

### The rental-goods analogy

Even a tool marked "we'll lend it to you for free," when you look closely, comes with conditions that vary wildly.

- **A: Truly free** (modification, redistribution, and commercial use all OK)
- **B: You can borrow it, but commercial use has separate conditions** (you can't tell without reading the contract)
- **C: The body is free, but the parts inside belong to a different lender** (you need to check the parts' conditions too)

If you look only at the "free rental" sign and assume everything is A, you'll be in trouble later. AI releases were exactly the same.

## The actual three-way split (first half of 2026)

Let me line up the three that appeared in this period along the license axis.

| Model | License | Kind | Outlook for commercial local use |
|---|---|---|---|
| **Gemma 4** (Google) | **Apache 2.0** | OSI-certified OSS | Zero barriers (the "change" discussed below is significant) |
| **Cosmos 3** (NVIDIA) | **OpenMDW 1.1** | open model license (not classic OSI-certified OSS) | Only after **reading the actual clauses** for commercial conditions and derivative provisions |
| **PaddleOCR-VL** (Baidu) | **Apache 2.0** (but mind the base) | OSI-certified OSS | Apache, but the **derived license of the ERNIE-4.5 base needs checking** |

![A map sorting the three models of early 2026 along the license axis into three kinds (OSI-certified OSS / open model / proprietary terms or conditional). Gemma 4 is Apache 2.0, Cosmos 3 is OpenMDW 1.1 (non-OSS = the actual clauses must be read), PaddleOCR-VL is Apache but ERNIE-4.5-based and needs checking. The point: "the weights being downloadable (technology) and the model being usable commercially (license) are separate acceptance checks," and before adoption you check three things separately — kind, commercial conditions, and base inheritance — captured in a single map.](../../../assets/articles/llcore_c1_license_map.svg)

_A single diagram capturing this chapter's point: even "free releases" split into three kinds of conditions._

### (a) Gemma 4 = the "change" to Apache 2.0 is real progress

Gemma, through 1/2/3, was distributed under Google's own **Gemma Terms of Use**. The big deal is that this **changed to Apache 2.0 in Gemma 4.** Proprietary terms tend to come with usage restrictions and redistribution conditions, but Apache 2.0 is straightforward, OSI-certified OSS. **The barrier to commercial local use became effectively zero.** This is not a "minor update" — on the license front it is **qualitative progress.** It looks unassuming, but it was a big step that all but erased the barrier to using it for work.

### (b) Cosmos 3 = OpenMDW 1.1 is "open, but not OSS"

NVIDIA's Cosmos 3 is **OpenMDW 1.1** (a model-centric license in the Linux Foundation family). This calls itself an "open model," but it is **a different thing from classic OSI-certified OSS.** The treatment of commercial conditions and derivatives is not uniformly "free" the way Apache is. **Judging "open, so commercial is OK" without reading the clauses is a dangerous pattern.** In the rental-goods terms from earlier, this is "B: you can borrow it, but commercial use has separate conditions."

### (c) PaddleOCR = Apache 2.0, but mind the base

PaddleOCR-VL is Apache 2.0. It looks straightforward, but because it is **based on ERNIE-4.5**, you need to check the license inheritance on the base-model side. The pattern "the wrapper is Apache, but the inheritance of the contents is separate" is a common pitfall with derived models. In rental-goods terms, this is "C: the body is free, but the parts inside belong to a different lender."

## Why this matters to the FullSense / llmesh strategy

Here is the "so what" that is the real point. The FullSense I'm involved with is **Apache-2.0 + Commercial dual-license**, and within it, **llmesh** is "a hub that integrates and serves multiple LLMs on-prem (locally)." For llmesh, **an increase in backend models that can be used commercially locally** directly means an expansion of options = a tailwind.

- **Apache 2.0 models that can run locally have increased** (Gemma 4, etc.) → llmesh's on-prem backend candidates increase with no commercial barriers.
- On the other hand, **not depending too heavily on models with commercial barriers** (proprietary terms, conditional open model licenses, or commercial conditions like the Qwen family) is the safe side for a dual-license product. From the standpoint that "depending on Qwen could become a commercial barrier," FullSense places weight on backend portability.

And one more thing: it reinforces **the positioning of llcore (my homemade tiny-model research).** In a world where "excellent Apache 2.0 local models have increased," **competing on intelligence with a homemade model makes ever less sense.** If you need conversational quality, just run Gemma and the like via llmesh. **llcore should not compete on the model and should commit entirely to being a research vehicle for memory efficiency** — this license map supports the correctness of that division of labor from behind (this bet on "memory efficiency" comes back again in Chapter 3).

## Chapter 1 takeaway — decompose "open" into three questions

When you make a public model an adoption candidate, instead of "is it open," confirm the following three **separately.**

1. **The kind of license**: OSI-certified OSS (Apache/MIT), proprietary terms, or an "open model" license (OpenMDW, etc.)?
2. **The conditions of commercial use**: are there additional conditions or restrictions on commercial use (revenue thresholds, usage restrictions, the scope of attribution obligations)?
3. **Derivatives and base inheritance**: can you ship derived models? Does it not inherit the base model's license?

> **Detection signal**: don't judge commercial viability on the single phrase "open weights!" Check the license name, and **if it is not OSI-certified OSS, always read the actual clauses.** Even if it says "Apache," trace the base model's inheritance one level down.

### Honest reservations (Chapter 1)

- This chapter is **not legal advice.** The license classification is a directional map, and **what carries binding force is each model's license clauses themselves.** Read the originals before making an adoption decision.
- The descriptions reflect my **understanding as of June 2026**, and licenses can be revised (as Gemma changed from 3→4).
- For each model's particulars (OpenMDW 1.1's commercial conditions and derivative provisions, the inheritance scope of the ERNIE-4.5 base), this chapter **has not done a clause-by-clause confirmation.** The places I explicitly marked "actual clauses must be read" are precisely the confirmation points to check before adoption.

License was the first example of a "dominant term beyond technology." However good the performance, if the commercial conditions don't fit you can't adopt it — **acceptance-check the weights being downloadable (technology) and being usable commercially (license) separately.** This is the practice of applying honest disclosure to the license side. The next chapter handles another "dominant term beyond technology" — an agent's "growth."

---

# Chapter 2 — "Growing" and "Growing Responsibly" Are Different Things

## "It grows the more you use it" does not necessarily mean "it gets better"

If license is the dominant term for "can you adopt it," then for a self-growing agent the dominant term is "**the direction it grows.**" Let me dissect the hottest tagline of the moment, "self-growing agent."

I'll state the conclusion first. **"It grows the more you use it" is an attractive phrase, but "grows" does not mean "gets better."** If it accumulates wrong learning, the agent can also "degrade the more you use it." This chapter recasts that branching point through the design philosophy of "growing responsibly."

## What the Hermes Agent is (fairly — first, the strengths)

Nous Research's **Hermes Agent** (OSS, MIT license, **released in early 2026**) is an agent that touts a **built-in learning loop** as its selling point. According to public information, the skeleton is the following five points:

1. **Automatically generates skills (reusable procedures)** after completing a task
2. **Self-improves skills** during use
3. **Persists memory**
4. Retrieves the past via **full-text cross-search with FTS5**
5. **User modeling via Honcho** (learning a picture of the user)

An agent that "grows into your own as you use it." This is genuine appeal, and the GitHub stars are very numerous (discussed below, but with reservations about the quality of stars). And this **competes head-on** with **llive** (the cognitive OS I'm involved with that touts self-evolution, four-layer memory, and derivative-population evolution). For exactly that reason, let me look at it fairly, but honestly.

## ★The core — "growth" has an implicit premise

The claim "it grows the more you use it" has an **unspoken premise.**

> **What it learned is not always correct.**

When an agent "auto-generates skills after a task and reuses them," what happens if that skill is **wrong**? The wrong procedure accumulates in memory, gets reused in the next task, and becomes the foundation for yet another skill. **The error self-propagates** — I call this a "contamination loop" (secondary sources point out the same kind of risk).

Concretely, it goes like this. Suppose a procedure the AI once committed to memory as "for this kind of work, use this procedure" contains a small error. Then in the next similar task that erroneous procedure is invoked as-is, and on top of it "a larger procedure that uses that procedure" gets stacked. Once this happens, it's hard to later extract the original small error, and the AI "grows" with the foundation itself contaminated.

To use an analogy, it's like **an athlete who keeps practicing with a strange self-taught habit baked in.** The more you practice, the better you get — that's not guaranteed. Repeating a wrong form can also mean **the more you practice, the more your poor form sets in.** Treating "growth" as unconditionally good makes this contamination loop invisible. **Growth = improvement is false.** A learning loop improves if you learn correctly and degrades if you learn wrongly. It is a blade open in both directions.

![A diagram where the same built-in learning loop (skill auto-generation / improvement / memory persistence / full-text search / user modeling) branches in two directions depending on whether there is a gate. Without a gate ("it grows"), it heads toward a "contamination loop" where errors self-propagate; with an approval gate (Approval Bus / HITL / honest disclosure, fail-closed) wedged in, "it grows responsibly" heads toward silently refusing to pass — stopping — erroneous skills. The right column shows that neither Hermes nor llive has independent benchmarks or peer review, so both are unproven. The branching point is not "how much you can learn" but "whether you can stop it when you learn wrongly" — captured in a single diagram.](../../../assets/articles/llcore_c2_responsible_growth.svg)

_Even with the same learning loop, whether the mechanism to stop is in the architecture decides whether it "grows" or "degrades."_

## So the question to ask is not "does it grow" but "does it grow responsibly"

Here is the branching point of llive's design philosophy. llive goes one step beyond "growing" and places at its core **"growing with a mechanism to stop wrong learning."** The concrete machinery is:

- **Approval Bus**: passing state changes (skill adoption, promotion to memory) through a checkpoint that requires approval rather than passing them unconditionally.
- **HITL (Human-in-the-Loop)**: inserting a human (or a higher-level verification) into the accept/reject decision for important learning.
- **honest disclosure**: before claiming "it grew," doubting whether that learning is truly an improvement and disclosing the breakdown (that discipline defined at the start of this piece).

In short, these are a design that **"wedges a fail-closed gate into the learning loop."** fail-closed is the default behavior of "when in doubt, don't pass — stop." The idea, in essence, is to **build in from the start** a mechanism where a coach stands beside you and stops you with "that form is wrong." Rather than being careful through willpower, you **stop wrong learning as architecture.**

> A growing agent = runs a learning loop.
> A responsibly growing agent = wedges an **approval gate** into the learning loop and stops contamination.

FullSense's design philosophy (bringing the locus of responsibility to the architecture level) takes concrete form here.

## ★Here I turn the blade on myself (don't only cut the competitor)

Left here, this reads as "llive is great / Hermes is dangerous." That is the pattern this series most despises. Fairly, I apply the same yardstick to both.

**A fair evaluation of the Hermes side:**

- Hermes is real, published under MIT, and a genuine, widely-used OSS. Branding it "dangerous" is unjust.
- That said, looking honestly, **I cannot confirm any independent benchmark or peer-reviewed paper demonstrating the effectiveness of the learning loop** (0 hits on an arXiv search; the effect is mainly an official self-claim). "It gets better as it grows" is **a claim not yet third-party-verified.**
- The GitHub stars are **on the order of tens of thousands** (multiple independent reports put it at roughly 47,000–57,000 as of April 2026, a sharp rise within two months of the February 2026 release), and it is certainly popular. However, **the star figures and their quality (the proportion from bots / campaigns) disagree across sources**, and the value of "roughly 196,554" I originally referenced could not be corroborated by other sources (suspected to be inflated). **The threat of "mindshare leading the way" is real, but the precise figure for its scale is source-dependent and requires a reservation on confidence.**
- The most recent release is, per public information, **more of a GUI shell over the existing core**, not the addition of a new model or new framework — I note this too, to avoid exaggeration.

**The same blade on the llive side (myself):**

- **llive, too, is likewise unproven.** I have not yet produced an independent benchmark showing that Approval Bus + HITL "actually stops the contamination loop." "Responsible growth" is at present **a design philosophy, not a demonstrated advantage.**
- If I say Hermes "has no independent benchmark," the same tag is pinned on llive. The only difference is the presence or absence of the design choice "**whether it holds a mechanism to stop contamination in the architecture**," and the effectiveness of that mechanism is, **for both, awaiting future verification.**
- As long as I hold up "growth ≠ improvement," llive's own "evolution" is also a target to doubt the breakdown of, without feeling victorious.

So the honest conclusion is not "llive is superior" but rather **"both are unproven, but llive is betting on the side that builds the mechanism to stop wrong learning into the architecture from the start"** — a statement of a design choice.

## Chapter 2 takeaway — when you see "self-growth," ask three things

When you see the tagline "AI that grows the more you use it," confirm the following three.

1. **Who guarantees the correctness of the learning**: are auto-generated skills / memories adopted unconditionally? Is there a gate for approval and verification?
2. **Is there a countermeasure against the contamination loop**: is there a mechanism to stop the path by which wrong learning accumulates and gets reused?
3. **Is "grows means gets better" verified**: is there an independent benchmark or peer review, or is it a self-claim? (Ask this of the competitor and of yourself.)

> **Detection signal**: be wary if "gets smarter the more you use it" is spoken of as an unconditional good. Growth is double-edged, and **the presence or absence of a mechanism to stop** is the dividing line of responsibility.

### Honest reservations (Chapter 2)

The competition among self-growing agents tends to be discussed in terms of "how much it can learn." But the true branching point is **"when it learns wrongly, can it be stopped?"** A learning loop, without an approval gate, inverts into a contamination loop. However — as is this series' practice, let me say it once more at the end — **that the mechanism actually stops contamination, llive itself has not yet proven either.** Hermes and llive both stand on the same "unproven" rung. What differs is only the design being bet on. "Responsible growth" is a **promise** to be proven by measurement from here, not a victory already achieved.

Placing responsibility **in the architecture** rather than "being careful in operation" — here, the shared mechanism that appeared in Chapter 2 of "persisting memory and retrieving it later" is in fact the protagonist of Chapter 3. Both Hermes and llive were using that skeleton. The final chapter sees that "memory" replayed across domains.

---

# Chapter 3 — Consistency Through Memory Is a Universal Component Across Domains

## "Fold the past into memory, retrieve it later"

In Chapter 2 the agent had a mechanism to "persist memory and retrieve it later." In fact this idea of "**folding the past into memory to maintain consistency**" works with the same skeleton not only for LLM long contexts and agents but **also in multimodal generation (manga generation).** This chapter gathers that corroborating evidence.

Here, recall the **constant state** defined at the start. When a text AI handles long text, no matter how much of the past it reads, it folds it into a fixed-size "summary (state)" to carry. You compress that day's diary into a single summary card within the day, and the next day it's enough to look only at the card — that idea. The recurrent / SSM family uses this, and the size of the state does not grow even as the text gets longer.

## What MangaFlow is (fairly — first, the key points)

**MangaFlow** (University of Tokyo + HKUST Guangzhou, arXiv:2605.28173) is research that generates manga. One of the hard parts of manga generation is **character consistency across panels** — it's a problem if the same character is drawn like a different person in the next panel. MangaFlow secures this with **story section memory.**

The mechanism is **external memory that ties references to characters, scenes, and objects to each section and reuses them across panels.** It holds a reference of "this character looks like this" in memory and pulls it in for later panels. The ablation (an experiment that removes that mechanism) shows the contribution (discussed below; self-report).

## ★The skeleton is the same — "fold the past into memory, retrieve it later"

Here it connects to the main thread of the series. MangaFlow's story section memory is, structurally, **the same skeleton as the "maintain state with memory" idea I've repeated in llcore / llive.**

- **LLM long context**: constant-state recurrent / SSM folds the past into a fixed-size state to carry, and uses it for later prediction.
- **Manga generation** (MangaFlow): folds character references into external memory to hold, and pulls them in for later panel generation.
- **Cognitive OS** (llive's four-layer memory): folds experience into hierarchized memory, and pulls it in for later tasks.

The domains differ (text prediction / image generation / agent), yet they share the skeleton **"consistency is obtained by retaining the past in memory and reusing it."** This becomes **corroborating evidence from other domains** for the claim that "the memory mechanism is not an LLM-specific trick but a universal component that underpins the consistency of generation and inference."

Let me look at the contribution through the ablation numbers (all are paper self-report):

| Metric | With memory | Without memory (ablation) |
|---|---:|---:|
| CIDS (character consistency family) | 0.619 | 0.582 |
| CSD (character similarity family) | 0.668 | 0.547 |

Remove memory and the consistency metrics drop. **"Memory is contributing to consistency"** — that another domain showed through its own experiment is the substance of the corroboration.

![A diagram showing that the universal component (skeleton) of "fold the past into memory and retrieve it later" is replayed in the same form across three domains — LLM long context, manga generation (MangaFlow), and cognitive OS (llive four-layer memory). At the bottom it adds MangaFlow's ablation self-report values (CIDS 0.619→0.582 / CSD 0.668→0.547, dropping when memory is removed), pinning down that what can be used as corroboration goes only as far as "the direction that memory contributes to consistency" — a single diagram.](../../../assets/articles/llcore_c3_memory_universal.svg)

_The same "fold into memory and retrieve later" skeleton is replayed across three domains: text prediction, image generation, and agents._

## ★An essential honest note (so as not to overvalue MangaFlow)

Reading this corroboration as "therefore manga generation is SOTA" is wrong. To treat the paper honestly, I make four notes explicit.

1. **The "drawing" is not MangaFlow itself.** What generates the pixels is an external cloud diffusion model (Gemini 2.5 Flash Image / FLUX.2 9B), and what MangaFlow handles is **its control layer** (instructions for layout and character references). Reading it as "MangaFlow draws the picture" is wrong.
2. **Layout IoU 100% / Coverage 99.98% are a different arena.** These are **self-made metrics that are nearly self-evident by design, since placement is made explicit with geometric coordinates** — they are not numbers measured in the same arena as a baseline that generates from pixels. Moreover, since it's a test designed by oneself, note too that favorable numbers tend to come out.
3. **The commercial site mangaflow.studio is a separate product unrelated to this paper.** Don't confuse them just because the name is the same.
4. **Pre-review v1, zero citations, no GitHub-public listing.** That is, third-party verification is incomplete (on the ladder of trust, the self-report rung).

What can be used as corroboration goes only as far as "the direction that memory contributes to consistency," and **it is not an absolute performance claim** — drawing this line is honest disclosure.

## One more connection — MangaFlow's "weakness" backs up the reason another effort exists

MangaFlow also writes its own limitations. **"Speaker attribution and speech-bubble placement on stylized faces is difficult."** With a face deformed in a manga-like way, "who is speaking" and "where to put the bubble" are hard, it says.

This is **exactly the weakness that another effort of mine is targeting** (a benchmark for understanding manga panels, especially one that pokes at hard cases like "speaker ≠ the subject at the center"). In other words, MangaFlow's self-acknowledged limitation becomes an external backing for the benchmark's reason to exist — namely that **"speaker and utterance attribution in stylized art is an unsolved hard problem."** The place where the generation side (MangaFlow) says "this is hard" coincides with the place the understanding-side benchmark should evaluate. The confession of a weakness ended up backing up the reason for another effort to exist.

## Chapter 3 takeaway — when you see a "memory mechanism," read the skeleton and the domain separately

When a new generative AI touts "memory that maintains consistency," look at the following separately.

1. **What is the skeleton**: is it the universal component of folding the past into memory and retrieving it later (often, yes)?
2. **Consistency in which domain**: text context, image characters, or agent experience? Even with the same skeleton, the metrics it affects and the cost differ.
3. **The arena of the numbers**: is that consistency metric measured in the same arena as the baseline, or a self-made metric that is self-evident by design?

> **Detection signal**: "maintain consistency with memory" is a powerful but universal skeleton. The novelty lies not in the "skeleton" but in "**in which domain and how it folds.**" Doubt the arena when a self-made metric reads 100%.

### Honest reservations (Chapter 3)

The memory mechanism is not an LLM trick. It is **a universal, cross-domain component that underpins the consistency of generation and inference.** That is exactly why FullSense "places memory at the center of the architecture" (llive's four-layer memory). However — as is this series' practice — MangaFlow's numbers are pre-review self-report, and what can be used as corroboration goes only as far as **the direction.** I keep the line that even if the universality of the skeleton can be supported, individual performance claims await third-party verification.

---

# Landing — Extending Honest Disclosure to the Axes of Industry and Ecosystem

The three chapters look like separate stories, but they are tied by a single thread. **The dominant term is often outside the technology.**

- **License (Chapter 1)**: however good the performance, if the commercial conditions don't fit you can't adopt it. Decompose the single word "open" into three questions.
- **Growth (Chapter 2)**: however much it can learn, without a mechanism to stop errors it "degrades the more you use it." Place responsibility in the architecture.
- **Memory (Chapter 3)**: the skeleton of memory is universal, but the novelty lies in "in which domain and how it folds." Doubt the arena when a self-made metric reads 100%.

And to all three, I applied the same discipline — honest disclosure. This series originally began from "with a tiny model on a home CPU, doubt the way of measuring, and show the loss." This piece extends that discipline **beyond my own measured figures to the industry's numbers, other companies' claims, and corroborating evidence from other domains.**

- Even when citing another company's number (Hermes's stars), I doubted my own cited value and retracted the inflated one.
- Even when citing another domain's numbers (MangaFlow's ablation), I made explicit the difference in arena and that it is at the self-report rung.
- The blade I use to cut the competitor I turned, in the end, always on myself (llive is unproven too).

"Open, so I'm safe," "gets smarter the more you use it," "memory yields consistency" — every one is an attractive phrase, but instead of believing the single word outright, inspect the premises one by one. Beyond the numbers, run the same honest disclosure through license, design philosophy, and cross-domain corroboration. That was the practice for not overlooking the dominant terms beyond technology.

---

_Sources and confirmation status (as of 2026-06). **This piece is not legal or investment advice.**_

- **Chapter 1**: Gemma 4 (Apache 2.0, released 2026-04-02; on Google's official blog "Expanding the Gemmaverse with Apache 2.0," the change from proprietary terms → Apache 2.0 = removal of commercial constraints is confirmed. Sizes: E2B/E4B/26B MoE/31B Dense) / Cosmos 3 (arXiv:2606.02800, NVIDIA 2026-06-01, OpenMDW-1.1 License = Linux Foundation) / PaddleOCR-VL (arXiv:2606.03264, Apache 2.0, NaViT+ERNIE-4.5-0.3B, OmniDocBench v1.6 96.33%). All reflect the understanding as of 2026-06, and **clause-by-clause confirmation is required by each party before adoption.** Context on the FullSense side = Apache-2.0 + Commercial dual-license / llmesh on-prem backend strategy / the policy of avoiding the Qwen commercial barrier.
- **Chapter 2**: Hermes Agent = GitHub `NousResearch/hermes-agent` (MIT, **released February 2026**). The learning loop (skill auto-generation / improvement during use / memory persistence / past search / user modeling) is confirmed. No independent benchmark or peer-reviewed paper showing effectiveness could be confirmed (= mainly official self-claim). **The stars are, per multiple reports, roughly 47k–57k as of 2026/4 (on the order of tens of thousands). The "roughly 196,554" originally referenced could not be corroborated by other sources and is retracted as suspected inflation.** The contamination-loop risk is pointed out in secondary sources. **Correction history: the first draft, derived from the seed, stated "created 2025-07 / 196,554 stars," but on web re-collation this was corrected to "released 2026-02 / tens of thousands of stars."** llive side = FullSense's Approval Bus + HITL + honest disclosure (as of this piece, no independent benchmark presented = a design philosophy, not a demonstrated advantage).
- **Chapter 3**: MangaFlow = arXiv:2605.28173 (University of Tokyo + HKUST Guangzhou). Table1/Table2 + ablation (CIDS 0.619→0.582 / CSD 0.668→0.547, all paper self-report). **Essential notes**: (1) the drawing is by external cloud diffusion models (Gemini 2.5 Flash Image / FLUX.2 9B) and MangaFlow is the control layer, (2) Layout IoU 100%/Coverage 99.98% are self-made metrics of explicit geometric placement and a different arena from a generation baseline, (3) the commercial mangaflow.studio is an unrelated separate product, (4) pre-review v1, zero citations, no GitHub-public listing (= self-report rung). Connection points = llcore constant-state recurrent / llive four-layer memory / manga-panel understanding benchmark (the hard case of speaker ≠ center subject). This piece is not a performance-SOTA claim but directional corroboration._
