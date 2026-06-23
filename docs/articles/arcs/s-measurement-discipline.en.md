# Distrust the Measure — The Trap of the Single Score and an Anatomy of "We Beat Them"

> Series "The June 2026 LLM Industry, as Seen from a Home CPU," Part 1.

A green checkmark appeared. "This model is usable," the system ruled. And yet, when I measured the same model's "accuracy at nailing the next character exactly," it had nearly halved. The capability had collapsed, yet the inspection said "PASS." That is the starting point of this series.

The theme of this arc is not some flashy new technique. It is this: **"When you trust a single number too much, what do you fail to see?"** First, I dissect this honestly through a failure of my own (Part I); next, I broaden it into a taxonomy of how the industry's "we beat so-and-so" headlines exploit the very same hole (Part II); and finally, I turn that blade back on myself (self-audit). **The discipline of distrusting how we measure is the very foundation of this entire series.** That is precisely why, in the next arc, we will be able to "show our losses" with our heads held high.

---

> **A caveat for this series (please read this first)**
>
> Every llcore measurement in this series comes from a **PoC of tiny models on a home CPU (char-LMs—character-level language models—of 0.81M to 130M parameters)**, and it does not directly disprove or surpass the performance of the large players' real LLMs (12B–64B). The comparisons are made **at the dimension of method, philosophy, and measurement discipline**. The quantization figures for footprint (memory usage, resident bytes) are actual measurements, but **they are simulated quant (pseudo-quantization), and inference speed was never measured at all.**
>
> I place this disclaimer at the very top because, later, I will line these up against the published scores of the large players' models as "traps of the same structure." **I am not comparing the magnitude of the numbers; I am comparing the shape (the form) of the way they are evaluated**—keep hold of that one point until you finish reading.

---

## Part I — The PPL PASSed, but the Model Was Half Broken

## 1. The Trigger: Chasing Memory Efficiency, I Saw a Different Cliff

In llcore (my experiment box for ultra-tiny LMs running on a home CPU), the north star I am currently chasing is **memory efficiency**. The same smarts, with less memory. The royal road to that end is **quantization**—representing the model's weights with fewer bits, dropping from fp32 (32-bit floating point) down to 8-bit / 4-bit / 3-bit / 2-bit.

Halve the bit width and the memory roughly halves too. Look at footprint alone, and it is a straight-line "bargain." But of course, cut too far and accuracy falls. I wanted to judge **how far I could cut** mechanically. So I set up a "quality gate."

The first gate I put in place was simple.

> **PASS if Perplexity (PPL) falls below 0.85× the unigram baseline (the weakest model, which predicts using only word-occurrence frequency).**

Here, let me carefully explain **Perplexity (PPL)**, which appears again and again throughout this series, just this once. PPL is a metric for "how surprised the model is by the next word," and lower is better. Intuitively, it is close to "on average, among how many choices is it hesitating?" The unigram PPL is the number for the weakest predictor, which uses no context at all, so **a quantized model that cannot fall well below it is deemed to have "lost the ability to read context"** and is dropped—the logic holds.

I ran the quantized models at each bit width through this gate. The subject is a char-LM of 11.9M parameters that I call realp1. And when 2-bit's turn came, **my prior expectation was wrong.**

---

## 2. The Prior Expectation, and How It Was Wrong

Before the experiment, I had a hypothesis in my head, something like this.

> "Even if PPL (average surprise) is held reasonably steady, the **top-1 hit rate (the rate at which the most likely candidate hits dead-on) will collapse first.** Because PPL looks at the overall smoothness of the distribution and is therefore dull, while top-1 is the sharp judgment of 'is the #1 candidate correct?', so the breakage should appear earlier."

Here let me pin down **top-1** (another foundational term used throughout this series). The top-1 hit rate is "the proportion of cases where the candidate the model output with the most confidence matches the actual correct answer dead-on." When we say it was hitting close to 3 out of 10 characters, that "close to 3 characters" is this.

In other words, I expected **PPL to look unscathed while top-1 dies first**—I expected the two to **diverge**. If that were so, I could write a tidy lesson-learned article: "the PPL gate hides top-1's degradation."

The result was this (realp1, comparing each bit width against the fp32 baseline. **Displayed values are rounded; Δ is the raw measured value**).

| Bit width | PPL | PPL degradation | top-1 hit rate | top-1 degradation (Δ) |
|---|---|---|---|---|
| fp32 (baseline) | 38.32 | — | 28.66% | — |
| 8bit | about the same | slight | about the same | slight |
| 4bit | mildly worse | small | mildly lower | small |
| 3bit | worse | moderate | lower | moderate |
| **2bit** | **101.114** | **+163.90%** | **15.21%** | **−13.46pp** |

> Footnote: Subtracting the rounded displayed values 28.66% → 15.21% by hand gives −13.45pp, but the Δ = −13.46pp in the table is computed from the **raw measured values before rounding**. I note both for the sake of arithmetic audit.

My expectation was wrong. **PPL and top-1 did not diverge.** At 2-bit, both collapsed simultaneously, in lockstep—in this article I call this **lockstep (simultaneous degradation)**. PPL worsened by a full +163.90%, and top-1 nearly halved from 28.66% → 15.21%. What had been hitting close to 3 out of 10 characters now hits only 1.5. **The situation where only one of them stays unscathed did not occur.**

We need to pause here. What was wrong was the "expectation," not that the lesson vanished. On the contrary, **the very fact that I was wrong is the protagonist of this article.**

---

## 3. So What Was the Actual Problem—Narrowing the Claim to Two Lines

"If it was lockstep, isn't the PPL gate enough? If top-1 falls together, you should be able to reject it the moment PPL falls"—a reasonable objection. That is exactly why I will write down, accurately and without exaggeration, **what actually broke**. My claim is only the following two lines.

> **(1) This time, PPL and top-1 were in lockstep (simultaneous degradation), and my prior expectation that one would die first was wrong.**
> **(2) And yet, 2-bit PASSed the "quality gate." The cause was that the gate's threshold (0.85×) was too coarse, letting through a 2-bit that was clearly broken.**

Let me follow the numbers. The unigram baseline PPL is about 215. The pass line of the gate is **0.85 × 215 ≈ 183**. The 2-bit PPL is **101.114**.

**101.114 < 183. Therefore PASS.**

The inspection did not lie. It merely operated as configured. Even if top-1 had halved, the gate has only the single yardstick of PPL. On that yardstick, it looks "comfortably better than unigram." **Even when both fall in lockstep, if the threshold is loose, a half-broken state—broken but not yet bottomed out—gets passed**—that is the whole of what happened.

Not mistaking this point is the core of this article's honesty. I do **not claim that "PPL inherently hides capability."** This time it did not hide it. The two fell together in lockstep. The problem was not the **choice of metric** but the **coarseness of the threshold**. The moment I inflate this into a generality, this record becomes a lie.

PPL and top-1 are not, to begin with, **independent and separate things**. Both are merely peering at the same output distribution from different angles. PPL smooths out the "surprise" of the whole distribution; top-1 carves out "did #1 hit?" They are **a wide shot and a close-up of the same scenery.** So that they move in lockstep is, on reflection, not mysterious at all. My expectation was just sloppy.

![A contrast of two judgments on the same 2-bit quantized model. The bar chart on the left shows that the top-1 accuracy fell from 28.7% before compression to 15.2% after 2-bit (−13.5pp, nearly halved), while the right contrasts how the loose PPL gate (101.1 < 183) issues a PASS, whereas the 97%-capability-retention gate gives a FAIL at 53% remaining. The point is that even with the same model and the same numbers, the verdict flips with a single choice of how to draw the gate—what is asked is not whether a good point came out, but whether the inspection can drop the bad ones.](../../../assets/articles/llcore_s1_single_score.svg)

_Even with the same model and the same numbers, the verdict flips with a single choice of how to draw the gate._

---

## 4. The Lens of Cognitive Science: Goodhart's Law and Construct Validity

Why be so wary just because "a single score issued a PASS"? Here are three concepts this series returns to again and again.

### Goodhart's Law

An empirical rule originating with the economist Charles Goodhart. Put plainly—

> **"The moment a metric becomes the target, it ceases to be a good metric."**

It was originally about monetary policy. When a central bank decides "if we control this monetary indicator, the economy will be stable" and starts aiming at the indicator, people's behavior optimizes toward that indicator, and the relationship between the indicator and reality breaks down. **The thing you were supposed to be measuring becomes unmeasurable.**

Transposed to the quantization gate—the moment I decide "PASS if PPL drops below 183," the object of judgment becomes not what I really wanted, **"is the model genuinely smart,"** but only **"does the number called PPL drop below 183."** This time, because it was lockstep, the breakdown stayed shallow, but the structure is exactly Goodhart's: **the moment you optimize for a proxy, you can drift away from the original goal.**

### Construct Validity

A term from psychometrics. It is the idea of asking whether "the abstract thing you want to measure (the construct)" and "the operational metric you actually measure" **properly correspond.**

- The construct you want to measure: **the model's language ability**
- The actual metric: **a single Perplexity**

PPL certainly captures one facet of language ability. But "the ability to nail the next token at #1," and the felt quality of generation, are also other faces of the construct. **When you let one metric represent the whole construct, the faces the metric does not pick up pass through unchecked.** This time top-1 was in lockstep, so it fell together and was conspicuous; but had there been a different kind of breakage—say, "PPL stays fine while only a specific syntax collapses"—the PPL gate would never see it.

### Proxy Divergence — The Health-Checkup Analogy

Compressing the two above into a single phrase, it is the danger of **"thinking you've measured the real thing via a proxy."**

Imagine settling a health checkup with a bathroom scale alone. Weight is a good proxy for health. But **a person whose weight alone is normal while their blood pressure is off the chart** is judged healthy by the scale. My quality check was the same: looking only at one score (the scale) and declaring "PASS." The faces that score did not pick up (the capability collapse) passed through unchecked.

Let me also write down where this metaphor breaks. Weight and blood pressure are metrics that **can move independently.** This time's PPL and top-1, by contrast—as stated above—are **merely the same output distribution seen from different angles; they are not independent.** So the health-checkup metaphor correctly carries the lesson "look at multiple metrics," but if you read it in the direction of "PPL and top-1 are independent, so measure both and you're safe," that is **overreach.** This time's real prescription was not "add more metrics" per se, but **tighten the threshold and, moreover, change it to a relative baseline against fp32.** Continued in the next section.

---

## 5. The Countermeasure: Wiring a Retention-Ratio Capability-Gate, Fail-Closed

If you end a lesson at "let's be careful," you will inevitably fall off the same cliff. So I **put it into code.**

What I newly built is the **capability-gate.** The idea is this.

- Old gate: "PASS if PPL drops below 0.85× the unigram" (**a coarse threshold on absolute PPL**)
- New gate: "FAIL unless **top-1 retention relative to fp32 is at least 97%**" (**the relative retention rate of capability itself**)

Retention is "post-quantization top-1 ÷ fp32 top-1." Taking fp32 as a perfect 100%, it measures what fraction of capability survives quantization.

Applying it to 2-bit—top-1 goes 28.66% → 15.21%.

> retention = 15.21 / 28.66 ≈ **53.1%** (= about **47% (46.9%) capability loss**)

The threshold is 97%. **53% falls far short of 97%, so FAIL.** The new gate cleanly drops 2-bit. The very thing the old gate PASSed.

The crux of the implementation is **fail-closed** (`src/llcore/lm/eval.py`). `passes_capability_gate(min_retention=0.97)` returns a pass only when retention is **at or above** (`>=`) the threshold. When an exception is thrown, when data is missing, when the comparison baseline (fp32) cannot be obtained—**every undecidable case falls to FAIL.** Making "do not let through what you cannot vouch for" the default—this is the same discipline shared with FullSense's MCP-family projects. `held_out_top1_report` separately reports top-1 on held-out (validation data not used in training), measured with teacher-forcing (a measurement method that feeds the correct answer one character at a time and has the model predict the next character).

Let me make one design decision explicit here. **The new gate does not discard PPL.** I keep the PPL gate as a cutoff for "does it have the basic ability to read context," and on top of it I **layered** the retention gate for "how much capability was retained." I did not replace one yardstick with another; I **added a face to measure while tightening up with a relative baseline and a strict threshold.** Exactly the textbook countermeasure against Goodhart.

> What I used to reproduce: `scripts/quant_bitwidth_sweep.py` (quantizes and evaluates 8/4/3/2bit in order), output is `out/quant_bitwidth_sweep*.json`, and the canonical record for the design and numbers is `docs/MEMORY_EFFICIENCY_FINDINGS.md`. All code references are relative paths.

Trust in a health checkup is decided not by "everything was normal" but by "is it designed to properly catch abnormalities if there are any." AI quality checks are the same. Not **that a good point came out**, but **whether the inspection can properly drop the bad ones**—this is the core of Part I.

---

## Part II — An Anatomy of "Beating Proprietary": The 5 Types of Cherry-Pick

In Part I, I dissected a concrete example of my own gate falling into the trap of the "single score." The same trap is not exclusive to a home tiny model. **The industry's "we beat so-and-so" headlines use exactly this form of "telling the winner with a single number."** From here, I classify that technique into five "types" and turn it into a checklist readers can use themselves.

## 6. The Hook — I Saw the Same Headline Three Times, and the Breakdown Differed All Three Times

I saw the headline "OSS beat the big players" three times in the single month of June 2026. Google's Gemma 4, NVIDIA's Cosmos 3, Baidu's PaddleOCR-VL. All are real news, and all have primary sources (official blogs or arXiv papers). The numbers were not fabricated.

And yet, all three times, the substance of "beat" was different.

- The first time, they put **only the axis they won** in the headline (the axis they lost was tucked deep in a table of the tech report, in a hushed voice).
- The second time, it was first place on **a bespoke benchmark they happen to excel at** (it was not about general ability).
- The third time, **the official statement only said "#1 among open models,"** yet the secondary-media headline alone grew into "beats Gemini."

Flashy victories are, without exception, manufactured by one of the same five "types." The purpose of this article is to hand you those 5 types as **a taxonomy (a toolset for sorting things into types) that readers can use themselves.** The goal is not the gut-feel slogan "distrust numbers," but for you to take home **a procedure for guessing which type was used to stage a given victory.** And finally, I turn those five blades back on llcore, my own home PoC. Criticism is meaningless unless it returns to oneself as a mirror.

> ☕ Take a breather here. The numbers about to appear are all "real primary sources." There are no villains. The problem lies not in "the numbers" but in "the reach of the claim that comes attached to the numbers (how far you can actually say)."

> **The breakdown of the scale gap (precisely, in one line)**: "about a 15,000× scale gap" is the maximum gap when comparing the smallest char-LM (0.81M) with Gemma 4 (11.95B). Even within the same PoC, the demonstration I describe later of "running with RAM exceeded" used a 130M model, and that one is only **about 92×** relative to Gemma. **The scale gap differs per demo,** so do not read it as "15,000× in any demo" (applying the maximum gap uniformly is itself Type 1, described later).

---

## 7. First, Just Two Terms — The Foundation of This Article

PPL and top-1 were explained in Part I, so here I add only the two new words I now need. **Benchmark** (a "common test" for measuring a model's ability. Close to a school mock exam, but **who wrote the questions and who graded them** greatly changes the meaning of the result) is intuitive enough, so—

**self-report** — the state where the number was **measured by the subject of measurement (= the company that built the model) itself and announced by itself.** In mock-exam terms, the state of "writing your own questions, solving them yourself, grading them yourself, and announcing yourself that you got a perfect score." It is not necessarily a lie. But neither has a third party measured it again under the same conditions to verify it (= reproduced it). **Every other company's number that appears in this article is, unless otherwise noted, a self-report.** This does not mean "therefore distrust it," but rather "let's tag it as still awaiting reproduction."

**cherry-pick** — selecting and showing **only the results convenient to oneself** from among the many results that exist. The image is plucking only the ripe red fruit from a tree laden with cherries. What matters is that **the plucked fruit is real.** They simply do not show the green fruit (= the axis they lost). So cherry-pick is more accurately grasped not as "a lie" but as the phenomenon that "**the reach of the claim is narrower than the headline insinuates.**"

This line-drawing of "not a lie, but narrow in reach" is the tone running through this whole article. This is not an article that indicts anyone. It is **a practice problem in how to read numbers.**

---

## 8. The 5 Types of Cherry-Pick (a Taxonomy)

Here is the main body. Using real cases from June 2026, I dissect the five types one by one. Each type comes with a "**detection signal (suspect when you see this),**" so once you finish reading, try applying them to the "OSS beats so-and-so" that flows across your own timeline.

### Type 1 — Loss-Axis Omission

**Definition**: Of multiple evaluation axes (categories), put **only the axis you won** front and center, and conceal the axes you lost (or place them small, deep in the tech report).

**Example: NVIDIA Cosmos 3.** Cosmos 3 is a large-scale world model for physical AI, announced as "Cosmos 3: Omnimodal World Models for Physical AI" (NVIDIA, 2026-06-01) (tech report, self-report, primary-source not cross-checked = tech-report PDF not obtained). Looking at Table 10 of the tech report, it wins by a large margin in the Driving category at **79.3 vs Gemini 47.2** (as of the tech report, self-report, primary-source not cross-checked). Looking at this alone, it "overwhelms Gemini."

And yet, **the same Table 10 is composed of multiple categories,** and another row reads as follows (all as of the tech report, self-report, primary-source not cross-checked). I use the word "axis (= category)" here, and this refers to the **number of evaluation categories** in Table 10 (it is distinct from the "number of benchmarks" in Type 4 described later, so I separate the terms to avoid confusion).

| Category (axis) | Cosmos 3 | Gemini | Outcome |
|---|---|---|---|
| Driving | 79.3 | 47.2 | Cosmos wins (large margin) |
| Robotics | 57.8 | 58.2 | **Gemini wins** |
| General | 73.7 | 77.5 | **Gemini wins** |
| SmartInfra | value not obtained | value not obtained | **unknown (needs tech-report verification)** |

Table 10 is composed of the **category-average of 4 categories (Driving / Robotics / General / SmartInfra).** Because I have not been able to obtain the primary tech-report PDF, the **SmartInfra Cosmos / Gemini scores are not obtained**—so I have honestly left that blank (filling in numbers I cannot obtain is exactly the practice this article criticizes).

What can be said within the known range is that **Cosmos wins only on the single Driving axis, loses to Gemini on Robotics and General, and SmartInfra is not obtained and unknown.** In other words, the summary "beats Gemini" compresses into the single Driving axis a picture that is **1 win and 2 losses among the 3 known categories of 4, with the remaining 1 unknown.**

Let me write down a caution to myself first. **I have just now left the 4th axis (SmartInfra) blank as "not obtained."** If an article that indicts loss-axis omission silently drops the 4th axis, then I myself would be committing Type 1. So I do not shrink it to "a 3-axis story"; I make explicit that **there are 4 categories, and that one is blank due to my own failure to obtain it.**

And in defense of NVIDIA, **the tech report does list all categories (within what I could read).** They did not erase the numbers they lost. The big problems arise mainly on the secondary-diffusion side (Type 5, described later); the tech report itself is honest. Type 1 should be grasped as a problem of "the craft of summarizing" rather than "malice." Summarizing necessarily discards information, so **the claim's color changes depending on what was discarded.**

> **Detection signal**: When the "so-and-so" in "beat so-and-so" is a single task name or single score. If an ability that should have multiple categories (driving, manipulation, general, infrastructure…) has been flattened into one line, go find the table before it was flattened. And count **whether all the categories of the table are visible** (whether someone—including yourself—is concealing any).

### Type 2 — Bespoke-Benchmark

**Definition**: Tell a victory on **a benchmark specialized for a task you excel at** as if it were a victory in general ability.

**Example: Baidu PaddleOCR-VL-1.6** (arXiv:2606.03264, Apache-2.0; the arXiv ID is primary-source confirmed). A model of 0.9B (NaViT dynamic-resolution encoder + ERNIE-4.5-0.3B)—a mere 900 million parameters. This scored **96.33% on OmniDocBench v1.6** (self-report) and was talked about as "surpassing the 235B class and Gemini." A 0.9B beating a 235B—looking at the number alone, it is stunning.

But **OmniDocBench is a benchmark dedicated to "document parsing."** It measures how accurately you can read characters, tables, and layout from scanned documents and PDFs. It is the score for **the single task of reading documents,** not general-purpose "smartness." Moreover, who measured it was Baidu itself (self-report).

This is not "cheating." A 0.9B specialized in document parsing surpassing a giant general-purpose model on a document-parsing-dedicated benchmark is **rather the correct fruit of specialization.** The problem is not the number, but that when "**won at document parsing**" is translated into "**surpassed Gemini,**" the reach quietly widens. It is like introducing an athlete who won a gold medal in the 100m as "the world's strongest athlete"—the 100m may truly be world-best, but no one is talking about the marathon or the shot put.

Note—just as my PPL gate in Part I missed "the ability to nail the next token at #1 (top-1)," **a single aggregate score can paper over, by the average, the fact that some part of its breakdown has sunk.** OmniDocBench 96.33 is "the average goodness aggregated across document parsing." How the breakdown of tables, the degradation of specific languages, and weakness on subsets like formulas, vertical writing, and low resolution are distributed beneath the average of 96.33 cannot be read from a single number (this is not to denigrate PaddleOCR—rather, an official card that explicitly states self-report is honest). **The single-score problem of Part I is recurring, at a changed scale.**

> **Detection signal**: Whether the target task is written into the benchmark name (OmniDoc**Bench** = Doc = document). If a victory on a bespoke benchmark is translated into "surpassing general ability," investigate what that benchmark measures.

### Type 3 — Self-Measured / Home-Court

**Definition**: **All the numbers used in the comparison were measured in the announcer's own environment, on their own benchmark.** A third party has not reproduced it under the same conditions.

This is a more fundamental type underlying Type 1 and Type 2. Cosmos 3's Table 10, and PaddleOCR's 96.33%, are both **self-reports measured by the announcer (NVIDIA / Baidu) in their own environment.** There are even cases where **the winning side runs the opponent's model (Gemini, etc.) in its own setup and grades it.**

In baseball terms, it is "**the result of a game played on your own home ground, with you as the umpire.**" The home advantage (you know the quirks of the field, the crowd is on your side) and the umpire's interpretation are all on your side. That does not necessarily make the game result fake, but **its credibility is a notch different from reproduction at a neutral field with a neutral umpire.**

If I take up Gemma 4's "approaching the 26B class" here, **it actually does not become a pure example of Type 3.** Because Gemma 4's "26B class" claim has **no quantitative benchmark table in the primary source itself** (it is general prose in the official blog, and whether the 26B comparison target is dense or MoE [= a sparse configuration that activates only some parameters; active params about 4B] is not even stated). In other words, this is not even "a number the announcer measured in-house"; it is a state where **a strong claim is made in general prose, with no number presented.** So this is more accurately treated not as Type 3 but as an example **closer to Type 1 (concealing the convenient choice of comparison target) or Type 5 (general prose turning into an assertion in the headline)**—because "a self-measured number" and "general prose with no number at all" are different problems, I separate them here.

What makes Type 3 troublesome is that it is **harder to see than Type 1 or Type 2.** Loss-axis omission (Type 1) you can notice by reading the tech report. A bespoke benchmark (Type 2) you can notice from the benchmark name. But "this is a self-report" you will overlook unless it is stated. That is exactly why the habit of **tagging, by default, "an unstated number is a self-report"** is effective.

> **Detection signal**: A number that does not state "who measured it," assume first to be a self-report. Look for whether there is a mention of third-party reproduction (an independent research institution or community re-measuring under the same conditions). And in the case of "a strong claim, yet no quantitative benchmark table even in the primary source," be wary as a problem prior to self-report.

### Type 4 — Small-N

**Definition**: **The sample (the number of benchmarks, the number of test items) that grounds the win/loss is small.** The fewer, the more easily the ranking swaps due to noise; the fewer, the less you can actually say.

Let me dig one level deeper into Cosmos's Driving win (79.3 vs 47.2) seen in Type 1. **The "number" here is distinct from Type 1's "number of categories (axes)."** In Type 1, I counted the number of axes—"Table 10 has 4 categories." What Type 4 counts is **how many benchmarks there are inside the single category of Driving.** And the Driving score was **the average of merely 3 benchmarks** (as of the tech report, self-report, primary-source not cross-checked).

For the avoidance of doubt: this "3" is not the axis count of Table 10 (= 4) but **the number of benchmarks composing the single Driving axis.** Because both happen to be single-digit numbers, they tend to get confused, but "there are 4 categories (Type 1)" and "Driving is the average of 3 benchmarks (Type 4)" point to different things.

What happens when the sample is only 3? Just as you cannot say "this die favors 6" even if you roll a die 3 times and get all 6s, **the average of 3 benchmarks cannot rule out chance bias.** One benchmark happening to favor that model is enough to move the average greatly. In statistical terms, it is a state of "**large variance, wide confidence interval.**"

Moreover, in such small-sample claims, **the variance or confidence interval itself is mostly not reported.** Only the average value 79.3 walks off on its own. Properly, you need to attach a width—"79.3 ± something"—or you cannot judge whether the gap from 47.2 is real or a fluke.

Here too, dismissing it as "3 benchmarks are worthless" is overreach. **A small number of benchmarks also has exploratory value** (as a first clue to a new task). What should be claimed is not "large samples are justice" but "**disclose the sample size and variance.**" If it's 3 benchmarks, just write honestly, "the average of 3 benchmarks, and the variance is about this much."

> **Detection signal**: Count how many items (the number of benchmarks) lie behind "average" or "aggregate score." If N (the sample size) is single-digit, suspect that the ranking swaps easily. An average with no ± attached—suspect the width first. Do not confuse "the number of categories" with "the number of benchmarks within a category."

### Type 5 — Second-Hand Inflation

**Definition**: **The official's cautious claim inflates into a strong claim** as it passes through secondary media and social networks. The announcer qualified it, but the qualification falls off in the game of telephone.

This is the type closest to the true nature of this article's hook, "the breakdown differed all three times."

**Example: Cosmos 3's "beats Gemini."** Reading NVIDIA's official newsroom (press release), there is a careful qualification, "**#1 among open models**" (self-report). Not "#1 among all models," not "surpassed Gemini." A narrow-reach claim of **first place within the group of open (open-weight) models.**

And yet, in the process of this flowing into secondary media and timelines, the "within open" of "#1 within open" **slips off,** and only "beats Gemini" remains. The announcer qualified it, but in the game of telephone the qualification is stripped. It is also important in this context that Cosmos's license is OpenMDW 1.1 (an "open model" license that is not OSI-approved OSS)—the very setting of the arena, "within open," requires an annotation to begin with.

What makes Type 5 the nastiest is that **the announcer is not at fault.** NVIDIA qualified it correctly. The one who inflated it is the receiving side. So reading it as "NVIDIA lied" is wrong; correctly, it is "**trace back to the primary source and the claim was much narrower.**" This is a problem of the distribution structure of information, not a problem of technology.

> **Detection signal**: When you see a strong headline ("beats Gemini"), **always trace back to the primary source (official blog, paper, newsroom).** Cross-check whether the qualifying words attached to the primary claim ("within open," "on a document benchmark," "on a specific task") have disappeared from the headline.

---

## 9. Summarizing the 5 Types on a Single Sheet

Let me compress the above into a take-home checklist. When "OSS beats so-and-so" flows across your timeline, try applying these from the top.

| Type | What it is doing | Detection signal | June 2026 example |
|---|---|---|---|
| Type 1 Loss-axis omission | Front only the axis (category) won | "Beat" is a single score / single task | Cosmos: shows only Driving, conceals Robotics/General (+SmartInfra) |
| Type 2 Bespoke benchmark | A win on a task-specialized benchmark told as a general win | The task is in the benchmark name | PaddleOCR: OmniDoc**Bench** 96.33% (document-only) |
| Type 3 Self-measured | All numbers in announcer's environment (or general prose with no number at all) | "Who measured it" is not written | Cosmos/PaddleOCR are self-report; Gemma "26B class" has no quantitative table even in primary source |
| Type 4 Small-N | Sample (number of benchmarks) is small, ranking is noise | N behind the average is single-digit, no ± | Cosmos Driving is the average of **3 benchmarks** (distinct from the category count of 4) |
| Type 5 Second-hand inflation | The official qualification is stripped in telephone | The headline is stronger than the primary source | NVIDIA "#1 within open" → secondary "beats Gemini" |

> 🎯 These 5 types are **not mutually exclusive.** It is entirely ordinary for a single "victory" to use Type 1 + Type 3 + Type 4 simultaneously (Cosmos's Driving win is exactly that). So rather than narrowing "which type" to one, it is more practical to count "**how many apply.**" The more types apply, the narrower the claim's reach.

![A table laying out the 5 types of cherry-pick in 3 columns—on the left, Types 1–5 (loss-axis omission / bespoke benchmark / self-measured / small-N / second-hand inflation); in the center, June 2026 other-company examples (Cosmos's Driving 79.3 vs 47.2, PaddleOCR's OmniDocBench 96.33%, etc.); on the right, the self-audit turning the same blade on llcore itself (alongside the fp32-based 3.9×, the fp16-based 1.9× is noted; about 15,000× smaller than Gemma; Driving is a 3-benchmark average; keep attaching qualifying words to the RAM-exceeded demo oneself). At the bottom, as "losses offered up," 2-bit QAT falls short of the in-house strict gate 97% at top-1 retention 82.9%, and speed is unmeasured. The point is that the 5 types are not a weapon to beat others but a mirror that returns to oneself, and that everyone is at self-report (rung 2).](../../../assets/articles/llcore_s2_cherrypick.svg)

*A single sheet applying the same 5 types to other companies' examples (center) and to llcore itself (right). Criticism returns to oneself as a mirror.*

---

## 10. The Metaphor — And the Place Where That Metaphor Breaks

As a metaphor that gathers the 5 types into a single image, **"a sports highlight reel"** is convenient.

A highlight reel splices together **only the plays the player landed.** Whiffs and weak grounders do not appear (Type 1). And even within that sport, **the scenes they excel at** are chosen (Type 2). Who filmed it was the team's PR department (Type 3). If the number of games is small, even "in peak form" may be coincidence (Type 4). And the comment the PR added, "one of the league's best performances," grows among fans into "the greatest of all time" (Type 5). **The highlight footage is all real.** Not one is fabricated. But if you watch it and think "this player is invincible," that is not the footage's fault—you were done in by **the nature of the highlight format itself.**

Let me honestly state **the place where this metaphor breaks** too. A highlight reel carries the strong nuance of "deliberately selecting good scenes." But the 5 types of cherry-pick are **not necessarily intentional.** Type 5 (second-hand inflation) has no intent on the announcer's part; it inflates on its own on the receiving side. Type 1, too, has all categories in the tech report and merely drops them in summary, so it is not necessarily malice. So if you hold too strongly to the association "highlight reel = sneaky PR," you slip into **a conspiracy-leaning reading that "companies are deliberately deceiving."** Much of the actual reality is **the unavoidable side effect of information processing called summarizing and telephone.** The metaphor is good for carrying the skeleton "the format narrows the reach," but it cannot carry as far as "**the presence or absence of intent**"—get off here.

> ☕ Break point. Up to here has been "tools for reading other people's benchmarks." From the next chapter, I drive these 5 blades **into myself.** This is the main event of this article, and the part that most strongly embodies FullSense's operational discipline `feedback_benchmark_honest_disclosure` (when an abnormally good result comes out, always suspect the breakdown before getting a winning feeling).

---

## Self-Audit — Reapplying the Same 5 Types to llcore Itself

On my home PC with no GPU, I run a cluster of experiments with a tiny model called llcore. The north star is not "smarts" but "memory efficiency." From those measured values, I make several claims. For example, "about 3.9× compression with int8 quantization," and "it runs even when the usable RAM is smaller than the model."

Having dissected other companies' benchmarks with the 5 types, **it is only proper to drive the same 5 blades into my own claims too.** If I do not, this article becomes a mere boomerang.

### Against Type 3 (Self-Measured) — All Self-Report to Begin With, and Tiny

First, the heaviest blow. **All of llcore's numbers are self-reports I measured myself on a home CPU.** Far from third-party reproduction, the subject is **a CPU PoC of tiny char-LMs (0.81M to 130M parameters).** Considering that Gemma 4 is 11.95B (about 12 billion), the ratio against the smallest model (0.81M) is **about 1/15,000.** However—and I separate this honestly too—the demonstration of "running with RAM exceeded" described later used a 130M model, and that one is **about 1/92** relative to Gemma. **Writing uniformly that "every llcore is 1/15,000" would apply an excessive (= too harsh on myself, and moreover inaccurate) scale gap to the 130M demo,** so I write it separately per demo. In any case, if I tagged other companies' self-reports as "awaiting reproduction," it would be unfair not to attach **an even thicker tag** to my own self-reports. So llcore's claims are never "surpassed real LLMs"; they are limited to "**what can be said at the level of method, philosophy, and measurement discipline**" (this is the meaning of the series caveat).

### Against Type 1 (Loss-Axis Omission) — "3.9× with int8" Can Be Inflated by Hiding the Baseline

I have measured "about 3.9× compression with int8 weight-only quantization (74–75% reduction), PPL degradation under 0.1%" (consistent across 3 models, `src/llcore/lm/quant.py`). The number is real. But **this "about 3.9×" is the value when fp32 (32-bit floating point) is the baseline.**

The conventional baseline of industry-standard quantization tools (llama.cpp / GGUF) is **fp16 (16-bit).** With fp16 as the baseline, int8 (8-bit) becomes only **about 1.9×** (GGUF's Q8_0 = 8.50 bpw is about 1.88× vs fp16, about 3.76× vs fp32). So if I write "3.9×" without **explicitly stating it is fp32-based, I am implicitly choosing a favorable baseline and inflating.** If I were to round to a clean "about 4×" and hide the baseline, that would be exactly the **Type 1 (showing only the convenient baseline axis)** I myself could fall into. So both in the body and in tables, I make it my discipline to **unify the headline number to "about 3.9× (fp32 baseline)"** and always note that on an fp16 baseline it is 1.9×.

### Against Type 2 (Bespoke Benchmark) / Type 4 (Small-N) — The Super-Niche of char-LM, a Tiny Corpus

If I criticize others for "measuring on a document-dedicated benchmark (Type 2)," I must admit that **char-LM (character-level language model), my own measurement subject, is also an extremely peculiar niche.** It is a different world in both scale and metric from the evaluation systems of real LLMs (general-purpose benchmarks like MMLU or HellaSwag). My "PPL dropped" is **a story on this tiny, peculiar arena,** not a story about general ability.

The sample size (Type 4), honestly too. llcore's corpus is small-scale, and the quantization sweep is an observation on a limited number of models (0.81M / 1.36M / 11.9M / 130M). The trend that "larger models are more robust to low bits" agrees in direction with the literature (Dr. Dettmers et al. (2023), etc.), but it is **an observation up to 11.9M at most,** and extrapolation to 12B–64B cannot be guaranteed. The direction of the slope is consistent with the literature; the absolute values are unguaranteed—writing this in bold in the body is my Type 4 countermeasure.

### Against Type 5 (Second-Hand Inflation) — So as Not to Become the Source of Inflation Myself

What I must be most careful of is that **I myself could become the "source" of Type 5.** For example, the claim "it runs even when the usable RAM is smaller than the model." This is measured (a fp32 522MB model completed a forward pass with a working-set cap of 358MB = 68% of the model, with the output checksum an exact match -215.1. The model of this demo is 130M = about 92× vs Gemma).

But **because my machine had limited avail RAM in implementation (about 3.6GB at runtime),** what I demonstrated was not "**a giant model that literally exceeds the total physical RAM**" but a story under the condition "**working-set cap < model size,**" which has the same property. If I wrote here "a model that exceeds RAM ran!" and stripped the qualification, that would be me doing **Type 5 (stripping the qualification)** myself. So I leave `cap_set_ok` (whether the cap enforcement succeeded) and the measured peak as-is in the JSON output, and do not fake success. **Keep attaching qualifying words to my own claims myself**—this is the only way not to become the source.

### Two More Losses I Offer Up Myself

Beyond the blades of the 5 types, I disclose, before being asked, of my own accord (this too is the craft of honest disclosure).

- **Speed was never measured at all.** llcore's quantization is **simulated quant (pseudo-quantization)**—the saved int8 is restored to fp32 at inference time for the computation. In other words, "memory got smaller (footprint = measured)," but I have **never once been able to say** "it got faster" (speed unmeasured). To structurally prevent the misreading "it got faster with 2bit / int8," this is the most important reservation, also written in the caveat banner.
- **2-bit could not clear my own quality gate even with the best effort.** Even implementing QAT (quantization-aware training), which weaves quantization into training, the 2-bit top-1 retention is **82.9%.** This is a large advance over PTQ (post-training quantization)'s RTN 22% / GPTQ 33%, but it **did not reach** the strict gate newly built in Part I (97% retention vs fp32). This is a loss I would like to hide, but hiding it would be Type 1 itself.

---

## 11. The Ladder of Trust — Self-Report Is Not the Bottom Rung, but Neither Is It the Top Rung

As a single axis running through the 5 types, I use an organizing scheme called the "**ladder of trust.**" It is the habit of, when you see a number, guessing which rung of the ladder it is on.

```
Rung 4  Independent peer review  ← verified by a third party as a peer-reviewed paper (most trustworthy)
   ↑
Rung 3  Third-party reproduction ← someone else re-measured under the same conditions and matched
   ↑
Rung 2  self-report              ← the subject measured and the subject announced (awaiting reproduction)
   ↑
Rung 1  Existence confirmed      ← only that the number / the paper exists has been verified
```

Rephrased in familiar terms—the very bottom is **self-grading by the subject** (the state of writing your own mock exam and grading it yourself). Above it is the rung where **a friend re-solved the same questions and got the same score.** Higher still is the rung where **a third party graded an official mock exam.** The very top is the rung where **an expert properly peer-reviewed it.**

What matters is that **self-report (rung 2) is not "a lie."** It is above rung 1 (does not even exist = fabrication). **The problem is the misreading of "treating rung 2 as if it were rung 3 or rung 4."** The moment you re-read "the subject is saying so" as "everyone has verified it," you have skipped two rungs.

Here, I disclose without dodging the breakdown of this article's own "rung 1 (existence confirmed)"—because, having raised the banner of honest disclosure, I must not manufacture a false rung in my own verification claims.

- **PaddleOCR-VL-1.6 (arXiv:2606.03264)**: arXiv ID confirmed in the primary source (confirmed).
- **Gemma 4 12B**: official blog / HF model card confirmed in the primary source (but "26B class" is general prose with no quantitative benchmark table in the primary source).
- **NVIDIA Cosmos 3 (arXiv:2606.02800)**: the **existence of the arXiv ID was confirmed as of posting** ("Cosmos 3: Omnimodal World Models for Physical AI" exists on arXiv). **However, the tech-report PDF itself has not been obtained on my end, and each score of Table 10 (79.3 / 47.2 / 57.8 / 58.2 / 73.7 / 77.5; SmartInfra not obtained) relies on secondary information = primary-source not cross-checked.** That the numbers exist and that I cross-checked those numbers with my own eyes in the primary source are different rungs, so I must not claim the latter. Read all of Cosmos's numbers with the triple tag of "as of the tech report, self-report, primary-source not cross-checked."

In other words, the rung of other companies' numbers is not uniform. **A number that reached existence-of-ID (part of rung 1) and a number I could cross-check myself against the primary data table have different certainties.** Writing them all together as "all cross-checked" would itself be—the act this article was wary of in Type 3—"making it look as if you verified when you did not."

And llcore is at exactly the same rung 2. My numbers, too, have completed existence confirmation (rung 1) (the JSON and reproduction scripts are intended to be published), but they have **not yet received third-party reproduction (rung 3) or independent peer review (rung 4).** So other companies and llcore stand on the **same rung** of the ladder of trust. What differs is only the scale (15,000× at the smallest, 92× in the RAM demo) and the posture of "**whether I keep attaching qualifying words myself.**"

It is not "my numbers are correct and other companies' are dubious." **Everyone is at rung 2, and everyone should aim for rung 3.** Whether one has that awareness is the core of what I wanted to say in this arc.

---

## 12. The Landing of This Arc — Measurement Discipline Is the Foundation of the Series, So Next We "Show Our Losses"

The 5 types of cherry-pick (loss-axis omission, bespoke benchmark, self-measured, small-N, second-hand inflation) are not weapons for indicting others. They are **a mirror for noticing, oneself, that one's own claim has a reach narrower than the headline insinuates.**

In Part I, I dissected the moment my loose PPL gate issued a certificate of passage to a half-broken 2-bit; in Part II, I dissected other companies' benchmarks with the 5 types and reapplied those same 5 blades to llcore. In that process, I myself left the 4th axis (SmartInfra) blank, was about to write the scale gap uniformly as 15,000×, and was about to round to "about 4×"—in each case, had I not stopped, I would have committed the very types this article criticized. The conclusion is this—**llcore is not putting out more honest numbers than other companies; it stands on the same rung 2 (self-report) as other companies, at a scale of 1/15,000 at the smallest (1/92 in the RAM demo).** The only difference is "whether I keep attaching qualifying words myself." And even that becomes a Type 5 source the moment I forget to attach them.

So if I were to name a single lesson to take home from this arc, it is this.

> **When you see a flashy victory, first trace back to the primary source and apply the 5 types. And then, apply the same 5 types to your own claims. Do only the former and forget the latter, and the criticism comes back as a boomerang and illuminates the narrowness of your own reach.**

When an abnormally good result comes out, suspect the breakdown before getting a winning feeling. That applies not only to others' numbers but **first and foremost to your own numbers**—that was the view of June 2026 as seen from a tiny model on a home CPU.

And this **measurement discipline is the very foundation of the entire series.** Only after distrusting how we measure, knowing the trap of the single score, and becoming able to attach qualifying words to our own numbers ourselves can we, with our heads held high, "show our losses." So in the next arc, instead of a story about winning, we exhibit head-on **the very cliff I could not climb.**

---

### Primary Sources (sources of the numbers in this arc)

**Because the rung of verification differs per number, I do not write "cross-checked" all together.** All other companies' numbers are self-report (ladder of trust, rung 2).

- **NVIDIA Cosmos 3**: "Cosmos 3: Omnimodal World Models for Physical AI" (NVIDIA, 2026-06-01, arXiv:2606.02800). **The existence of the arXiv ID was confirmed as of posting. However, the tech-report PDF itself has not been obtained = the numbers are primary-source not cross-checked (relying on secondary information).** Tech-report Table 10 is the category-average of 4 categories (Driving / Robotics / General / SmartInfra) (as of the tech report, self-report, primary-source not cross-checked): Driving 79.3 vs Gemini 47.2 / Robotics 57.8 vs 58.2 / General 73.7 vs 77.5 / **SmartInfra value not obtained (needs tech-report verification)**. Driving is **the average of 3 benchmarks** (= Type 4 small-N. A number distinct from Table 10's category count of 4). The official newsroom qualifies it to "#1 within open models." License OpenMDW 1.1 (an open model license, not OSI-approved OSS).
- **Baidu PaddleOCR-VL-1.6**: arXiv:2606.03264, Apache-2.0 (arXiv ID primary-source confirmed). 0.9B (NaViT + ERNIE-4.5-0.3B). OmniDocBench v1.6 = 96.33% (document-parsing-dedicated benchmark, Baidu in-house measurement, self-report).
- **Google Gemma 4 12B**: Apache 2.0, dense 11.95B (official blog / HF model card primary-source confirmed). "Approaching the 26B class" is **general prose with no quantitative benchmark table in the primary source**, the number itself is not presented, and whether the 26B comparison target is dense or MoE (active ~4B) is undetermined (= apples-to-oranges). Note that there is no data table even prior to self-report.
- **llama.cpp / GGUF**: Q8_0 = 8.50 bpw ≈ 1.88× vs fp16 / 3.76× vs fp32 (official quantize README, k-quants PR #1684, primary-source confirmed).

### Sources of llcore's Numbers (all home CPU, char-LM, self-report)

Canonical = `docs/MEMORY_EFFICIENCY_FINDINGS.md` / `docs/POSITIONING_VS_LLAMACPP.md`. Evaluation implementation = `src/llcore/lm/eval.py`, quantization = `src/llcore/lm/quant.py`, reproduction = `scripts/quant_bitwidth_sweep.py` → `out/quant_bitwidth_sweep*.json`.

- Quantization bit-width sweep (realp1, 11.9M char-LM): fp32-baseline PPL 38.32 / top-1 28.66%. At 2-bit, PPL 101.114 (+163.90%) / top-1 15.21% (−13.46pp, retention ≈ 53.1%, capability loss about 47%). The old PPL gate (0.85 × unigram about 215 ≈ 183) PASSes; the new capability-gate (fp32-based top-1 retention ≥ 97%, fail-closed) FAILs.
- int8 weight-only quantization: weight resident **about 3.9× (74–75% reduction, fp32 baseline. about 1.9× on fp16 baseline)** / held-out PPL degradation under 0.1% (consistent across 3 models).
- int8 streaming-dequant inference: 72% reduction in resident model (dense fp32 about 539MB → stream int8 about 149MB). Via the `src/llcore/lm/eval.py` path.
- Working-set cap demonstration (model 130M = about 92× vs Gemma): completed a forward pass on a fp32 522MB model with a WS cap of 358MB (68% of the model), with an exact match of the capped/uncapped logits checksum (-215.1). `cap_set_ok=true` is the result in this environment.
- 2-bit QAT capstone: top-1 retention 82.9% (falls short of the strict cap-gate 97%). Greatly exceeds PTQ RTN 22% / GPTQ 33%, but did not reach full conquest of 2-bit on a tiny model.
- capability-gate: applies fp32-based top-1 retention ≥ 97% fail-closed (`src/llcore/lm/eval.py`). Passage at the 8B class is unproven.
- All numbers: tiny char-LM (0.81M to 130M), CPU, simulated quant (**inference speed unmeasured**). The scale gap from real LLMs (12B–64B) is about 15,000× for the smallest model, about 92× for the 130M model of the RAM-exceeded demonstration.

> Operational discipline: `feedback_benchmark_honest_disclosure` (when an abnormally good result comes out, always suspect the breakdown before getting a winning feeling). This arc is written as a demonstration of this discipline.
