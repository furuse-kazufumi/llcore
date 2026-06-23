# Show Your Losses — A Story of Honest Disclosure (The 2026 June LLM Industry as Seen From a Home CPU, Part 2)

The climax of a technical article is usually "We beat competitor X by Y%." It feels good to read, and it feels good to write.

But that very "good feeling" is, in my view, the single greatest poison eroding the credibility of technical writing. **Numbers that claim a win can be manufactured at will.** Show only the cases you're good at, erase the axes where you lose from the table, measure only in your own environment — none of these are fraud; they are routinely practiced as "industry convention."

So I bet part of this series on **"showing the losses."** Don't hide them, don't water them down, don't dissolve them into moralizing — always pin them to a number and show them. Why do such a thing? **Because showing the losses earns more trust than telling the wins.** This is not mere idealism. By the end of this part, I will line up several moments where I "negated my own achievement by the rules I myself made." That accumulation of "losses" is, I believe, the strongest calling card I can offer as an individual.

This article folds the 7 "losses (and the mechanisms that discipline the losses)" — written separately across the series — into a single story. The order has meaning: it starts with **b1**, declaring the ethics of the series; moves to **b5**, where the capability ceiling of a homemade LLM is shown in actual output and the north star is switched; **b3**, where a review AI's mistaken finding is overturned with primary evidence; **b4**, a suggestive loss where a safety gate turned out to be redundant; **b7**, transplanting the tools of a losing line of research onto a winnable arena; **b2**, suppressing a victory declaration with my own noise floor; and finally **b6**, folding a winning line into a tool that anyone can accept in one line.

---

> **The reach of all llcore measurements in this series (a disclaimer, raised large here just once)**
>
> Every figure is a measurement from a **home-CPU tiny-model (0.81M–130M char-LM; only one part of this series CPU-converts a 0.5B off-the-shelf model) PoC.** **It does not directly refute or surpass the performance of the big players' real LLMs (12B–64B).** The comparison is made on the dimension of **method, philosophy, and measurement discipline.** The quantization footprint is a measured value; inference speed is unmeasured because it uses simulated quant. The score is a next-token-nll proxy (a perplexity proxy for next-token prediction), and it contains no conversational-quality claims. The reproduction code and output JSON (`scripts/*.py` / `out/*.json` / `src/llcore/lm/eval.py` and others) are published on GitHub.

I raise this single sentence large before entering the main text. Normally the "disclaimer" is a small note appended to the end of an article; I deliberately place it at the top because most of what I'm doing in this part is **"a story of what didn't work."** And this series puts that body of defeats squarely in the leading role.

---

## A Shared Yardstick — PPL, top-1, honest disclosure, constant state (just once, up front)

Here, just once, I'll break down the terms that recur in each chapter. The chapters that follow proceed on the assumption of these definitions.

**PPL (perplexity).** A metric that measures "how hard the model seems to find it to guess the next single character," using the **entire probability distribution.** It averages the overall quality of the whole distribution — "next is a at 40%, b at 30%, c at 20%…". Lower is better. There is one weakness here. **Even when top-1 (the candidate with the highest probability) breaks and gets swapped out, if the tail of the distribution (the distribution from 2nd place down) remains reasonably intact, the average can be salvaged somewhat.**

**top-1 accuracy.** The fraction of cases where "the character predicted as most probable" actually matches the true answer. The lower it is, the more the "hit-it-dead-on power" has fallen. If PPL is the average over the whole distribution, top-1 is the dead-on, single-shot gamble. As we'll see later, these two do not always move together.

**honest disclosure.** The discipline at the foundation of this entire series: "If an abnormally good result comes out, always doubt the breakdown before you feel like you've won. Don't erase failures — keep them as lessons." It is aimed not at others but, first, at oneself. The series' other rule — **a finding from an external AI (even a review AI) is adopted only after verifying it one item at a time against actual code and primary sources** — is an extension of this (it actually pays off in chapter b3).

**constant-state recurrent.** A mechanism that keeps memory **flat** without growing even as the context (the conversation so far) is lengthened. The north star of "memory efficiency" that comes up later bets almost entirely on this property. By analogy, it's like continuously holding a single summary memo into which old talk has been re-condensed. That said, it has a weakness — "it stops working as well the longer the text" — and we measure that without hiding it too (chapter b2).

Let me leave one health-checkup analogy. Even if every item in a full medical exam is within the reference range, you can rest easy only **when that exam is designed to actually catch abnormalities.** If the blood-pressure cuff is broken and always reads "120/80," the output "normal" has no value. The reliability of an exam is determined not by "a good result came out" but by **"whether it can produce a bad result when a bad result ought to come out."** This series is the story of building that "mechanism that can produce a bad result."

---

## b1. What I Built Is Not a "Bench to Surpass the Outside" but a "Referee That Stops Myself"

Normally, the climax of a technical article is "We beat competitor X by Y%." As I wrote at the start, that good feeling can become poison. So in llcore (research on a tiny char-LM running on a home CPU), I inverted the policy. **I quit building a bench to surpass the outside, and poured the effort into building a referee that stops myself.**

A soccer referee is normally neutral. But what I built was **a referee who blows the whistle hardest on its own team's fouls.** Concretely, the evaluation code `src/llcore/lm/eval.py` places a two-stage pass/fail gate.

```python
def passes_gate(model_ppl: float, unigram_ppl: float, margin: float = 0.85) -> bool:
    # PPL gate: PASS if model PPL is at most 0.85x the unigram baseline
    return model_ppl <= margin * unigram_ppl

def passes_capability_gate(
    model_top1: float, reference_top1: float, min_retention: float = 0.97
) -> bool:
    # Capability gate: FAIL unless the reference (fp32) top-1 is retained at 97% or above
    if reference_top1 <= 0.0:
        return model_top1 >= 0.0
    return model_top1 >= min_retention * reference_top1
```

`passes_gate` checks "is the PPL clearly below the unigram (a baseline that ignores order and uses only character occurrence frequency)." `passes_capability_gate` (hereafter the **cap-gate**) checks "is the fp32 model's top-1 retained at 97% or above."

The crux is that I deliberately set the latter **strict (97% retention).** Unless the top-1 retention rate against the reference model is 97% or above, it FAILs — a **checkpoint to stop myself.** Anything that merely looks a little promising will absolutely not pass.

### Why "Two Gates" Are Needed, in Real Data

The direct trigger for building the cap-gate was a bit-width sweep of quantization. I crushed the weights step by step from 8bit down to 2bit and measured both PPL and top-1.

Per-channel quantization of **multi_smoke (1.36M params, vocab 4358, Japanese multi-corpus. fp32 PPL 24.88 / top1 36.28%):**

| bits | reduction | PPL | ΔPPL% | top1 | Δtop1 (pp) | PPL gate |
|---|---|---|---|---|---|---|
| 8 | 74.0% | 24.886 | +0.01% | 36.28% | −0.00 | PASS |
| 5 | 83.3% | 24.935 | +0.21% | 36.25% | −0.04 | PASS |
| 4 | 86.4% | 25.295 | +1.66% | 36.08% | −0.20 | PASS |
| 3 | 89.5% | 27.761 | +11.57% | 34.30% | −1.98 | PASS |
| 2 | 92.6% | 269.716 | **+983.95%** | 7.49% | **−28.80** | **FAIL** |

multi_smoke collapsed at 2bit in both PPL and top1 together, so the PPL gate correctly issued a FAIL. The problem was the larger **realp1 (11.9M params, vocab 3044, single Japanese book. fp32 PPL 38.32 / top1 28.66%).**

| bits | reduction | PPL | ΔPPL% | top1 | Δtop1 (pp) | PPL gate |
|---|---|---|---|---|---|---|
| 8 | 74.6% | 38.316 | +0.00% | 28.66% | −0.02 | PASS |
| 4 | 87.1% | 38.599 | +0.74% | 28.42% | −0.25 | PASS |
| 3 | 90.2% | 40.154 | +4.80% | 27.97% | −0.70 | PASS |
| 2 | 93.3% | 101.114 | **+163.90%** | 15.21% | **−13.46** | **PASS** |

**Look at realp1's 2bit. top1 goes from 28.66% → 15.21%, −13.46pp, nearly halving. Capability is clearly broken. And yet the PPL gate PASSed.** This is because the PPL was 101, below 0.85x (= about 183) of the unigram baseline (215). Precisely because I was looking at only one lax score (PPL), I let through a model whose capability had halved.

Here, let me correct, myself, the exaggeration I first wanted to write. One is tempted to say, "PPL is, **in principle,** a metric that hides capability." But the real data doesn't say anything that strong. **Even at realp1's 2bit, PPL degraded greatly, by +163.9%, and top1 and PPL collapsed almost in lockstep.** The accurate observation is this — **it's not that "PPL hid capability," but that "the pass/fail threshold of 0.85x was too coarse and PASSed a broken 2bit."** The metric captured the degradation on both counts. It was merely that my gate was drawn too loosely.

The hypothesis I had set beforehand ("top1 should degrade before PPL") also **did not hold** in this data. I record this too, without erasing it. The very fact that the hypothesis was wrong became the grounds for redesigning the cap-gate not as "a check that gets ahead of PPL" but as "a check that **re-tightens the threshold.**"

In response to this finding, I added the cap-gate to eval and wired it into the sweep, and as a result I could actually catch the **bit-widths where the PPL gate PASSes but the cap-gate stops them = multi_smoke 3bit / realp1 2bit.** My own evaluation infrastructure, by its own hand, plugged one notch of my own laxness.

### Defeat Log (1) — Self-refuting "2bit cannot clear the cap-gate even with QAT/LSQ"

From here begins the body of defeats. The first is the story I persisted on the hardest and still lost.

"If I could quantize to 2bit while barely dropping top-1, the memory efficiency of a home-CPU char-LM would improve dramatically." If I could realize this, it would be the centerpiece of the series. So I strengthened the PTQ (post-training quantization) methods in sequence.

**multi_smoke 2bit (fp32 ref: PPL 24.88 / top1 36.28%), top-1 retention as I raised the methods:**

| method | PPL | top1 | retention | cap-gate |
|---|---|---|---|---|
| PTQ RTN (naive rounding) | 236.9 | 7.98% | 22% | FAIL |
| PTQ GPTQ (error compensation) | 138.7 | 12.07% | 33% | FAIL |
| QAT (train with fixed scale) | 38.15 | 30.10% | 82.9% | FAIL |
| **QAT + LSQ (learnable scale)** | **36.79** | **30.48%** | **84.0%** | **FAIL** |

Each time I raised the method, the damage did indeed shrink. RTN's 22% → GPTQ's 33% → QAT's 82.9%. QAT (quantization-aware training — retrain on the premise of crushing to 2bit) recovered top-1 to 30.10%, retaining about 3.8x RTN and about 2.5x GPTQ.

So I persisted one more notch, implementing **LSQ (Learned Step Size Quantization)** — a method that **moves the quantization step size (scale) by learning** rather than fixing it (Esser et al., ICLR 2020) — myself, exactly per the paper's equations (gradient balancing g=1/√(N·Q_P), initialization s=2·mean(|w|)/√Q_P). The result was **top-1 30.48%, retention 84.0%.** It did beat fixed-scale QAT (82.9%), but **the difference is +1.1pp.** The lineage RTN 22% → GPTQ 33% → QAT 82.9% → LSQ 84.0% keeps improving monotonically to the end, but the gains have completely plateaued, and **even with LSQ, it FAILs the 97% checkpoint.**

Here lies the most important implication of this defeat. **The plateau came not because methods were lacking, but because scale was lacking.** LSQ's own paper reports that 2bit collapses by −14pt on a small model (SqueezeNext) with parameters trimmed to the extreme (whereas on the larger ResNet it's −2.9pt). The scaling laws of quantization (Dettmers et al. 2023, QiD 2024) say in unison that "small models have no redundancy and cannot absorb the quantization noise of 2bit." Indeed, retention in the high-90% range at 2bit is reported only when VQ codebooks or long QAT are combined on large models of 7B and above (in EfficientQAT, 7B=92.7%, 70B=95.9%). **That a 1.36M char-LM, no matter how much I raise the method, cannot reach 97% at 2bit was a loss foretold from the start.** So I do not make this "+1.1pp" a win report. I record it as a **confirmed loss: "even with a learnable scale, I could not cross the wall of scale."**

(Note on reach: this 82.9% is the figure for **multi_smoke (1.36M).** For realp1, GPTQ gave a top1 retention rate of 77.7%. Either way, 2bit does not reach 97%.)

If I had not built the cap-gate, I would probably have written the article with a chipper headline like "82.9% 2bit top-1 retention with QAT! About 3.8x RTN!" That headline is not a lie. But I would have **failed to convey to readers the crucial defeat: "it does not reach 97% = at this scale, 2bit has not been conquered."** The referee I built myself stopped my own optimism.

### Defeat Log (2) — "mmap is always memory-saving" is not true

Second. This is a defeat that lodges at the very root of me, the one who set "memory efficiency" as the north star.

If I use `mmap` (a mechanism that maps a file directly into memory and loads only the pages it needs; used by llama.cpp and others), then even a huge model would put only "as much as it used" into memory — that's what I expected.

**realp1 (11.9M params, model.pt 49.3MB, param 53.91MB):**

| mode | ΔRSS at load | ΔRSS after touch |
|---|---|---|
| eager (`mmap=False`) | **50.77MB** (≈ whole model loaded) | 51.64MB |
| mmap (`mmap=True`) | **1.42MB (×0.028)** | 51.54MB |

Right after load, eager immediately reads all weights (about 51MB) into memory, while mmap loads only **1.42MB, a mere 2.8%.** "This is it," I thought. But look at the **ΔRSS after touch** column. Even with mmap, once all weights are actually accessed (all of them used in forward inference), it eventually swells to **51.54MB.** Almost the same as eager (51.64MB).

In other words, **for a workload that necessarily uses all weights at least once (= ordinary inference), mmap's final memory converges to eager.** The accurate claim is not "**always memory-saving**" but "**loaded lazily, only as much as needed.**" The benefit is genuine only when (a) you use only part of the working set, (b) you share the page cache across multiple models, or (c) you can tolerate the latency of a cold start.

(That said, this property has another use. A read-only mmap page is "clean," so under memory pressure the OS can discard it without writing it back to the pagefile, then re-read it from disk on re-access. Using this, you can run a forward to completion even when "available physical RAM < model size" — in fact I ran a 522MB model to completion under a 358MB working-set cap, and the logits checksum matched exactly. I deny "always memory-saving" while affirming "runs over RAM" as a separate mechanism. I draw the line on both with numbers.)

### Defeat Log (3) — int8 streaming, with no pressure the peak barely moves

Third. This is a textbook case of "the logic is correct partway, but when struck on real hardware, the expectation was betrayed." The idea goes like this — keep the weights resident in int8, and within the forward, restore (dequant) to fp32 **layer by layer** and release immediately. That should lower both the resident memory and the peak memory during inference.

**130M params (n_embd=1024 / L=10), dense (restore all layers at once) vs stream (restore layer by layer):**

| mode | resident | peak WS | logits checksum |
|---|---|---|---|
| dense | 538.6MB (fp32 fully loaded) | 963.8MB | −192.1 |
| stream | **148.9MB (int8, ×0.285)** | 882.2MB | −192.1 |

**Resident memory won robustly. 538.6MB → 148.9MB, a 72% reduction.** This is genuine. But look at the **peak WS** column. 963.8MB → 882.2MB. **It has barely moved.**

Even though I'm discarding layer by layer, why doesn't the peak come down? The reason lies in torch's caching allocator. **Once it secures fp32 memory, even after releasing it, it does not return it to the OS but holds it inside its own pool.** On top of that, temporary activations during inference also load. So "in peacetime, with no pressure applied," the peak was nearly unchanged. Precisely, **"resident drops by 72%, but the peacetime peak is nearly unchanged."** The reduction materializes only **when memory pressure is applied** (forcing a working-set cap of 368MB, dense's resident 539MB doesn't fit and won't run, but stream completes at 368MB).

The logits checksums of all three results (dense / stream / stream-capped) match exactly at −192.1. Memory optimization did not change the result. **Lining up "the correct part (resident −72%)" and "the disappointing part (peacetime peak unchanged)" in the same table** — that is my way of honest disclosure.

### Defeat Log (4) — "Evolution won 20/20," judged ARTIFACT by my own meta-gate

The fourth is the one of the four that made me feel the most like I'd won. And precisely for that reason, it was the biggest harvest.

In a separate line of research (a framework that retrofits a small recurrent adapter onto a frozen small LLM and evolves its structure online), the following result came out. **On a cross-entropy landscape derived from the real SmolLM2-135M, evolution (MAP-Elites) beat finite-difference gradient 20/20.** 20 tries, 20 wins. The p-value was also 9.5e-7. "Evolution beats gradient" — a headline full of dreams.

But my framework had, at the very moment of winning, a built-in **meta-gate (strong-gradient meta-gate) that automatically summons a stronger opponent.** I threw not finite-diff but a **strong analytic gradient via backprop (torch Adam)** back at it.

**held-out (unseen data) mean fitness (= −CE, higher is better):**

| method | held-out mean | note |
|---|---|---|
| **strong analytic gradient (torch Adam)** | **−1.446** | best of all methods |
| MAP-Elites (evolution) | −1.454 | 2nd |
| random | −1.473 | |
| finite-diff gradient | −1.483 | last |

The strong gradient (−1.446) pulled back ahead of evolution (−1.454). **In evolution vs strong analytic gradient, the gradient reversed 19/20 (p=3.5e-4).** The pre-registered 4-condition AND did not hold. The verdict was fixed at **ARTIFACT (apparent win) + NEGATIVE (a loss on capability).**

Here, let me write the mechanism of this win/loss accurately. It is the sharpest point pointed out by the critic. **The "equal budget" of this comparison means an equal number of forward CE evaluations, not an equal number of effective update steps.** torch Adam can do 2000 step updates, whereas finite-diff / evolution can advance only about 95 steps within budget (finite-diff advances only 1 step per dimension+1 evaluations). **This "asymmetry in update counts" is the true nature of the ARTIFACT.** Evolution beat finite-diff because finite-diff bore the handicap of advancing only about 95 steps from a cold start. With the same evaluation budget, if you use the gradient "correctly" (exactly, via backprop), you can advance 2000 steps, and the win/loss flips. In other words, evolution's apparent win was an **artifact of a weak baseline.**

If there had been no meta-gate, I would have **proudly published the false-positive** "evolution wins 20/20 on capability on a real landscape." My own framework actually stopped one such thing before publish. This is not a "loss report." It is **"a report that the discipline of self-skepticism functioned on top of the data."** I implemented "doubt the breakdown before you feel like you've won" not as idealism but as an apparatus, and it actually knocked down one of my own achievements — this is the strongest evidence I can offer.

![A diagram of a two-stage referee: evolution (MAP-Elites) wins 20/20 against the weak opponent finite-diff, but when the meta-gate automatically summons a strong backprop gradient, it reverses 19/20 and the verdict becomes ARTIFACT. The true nature of the win/loss is the asymmetry in update counts (gradient 2000 step vs evolution about 95 step), and it shows that an apparent win of margin 0.008 was stopped by my own framework before publish.](../../../assets/articles/llcore_b1_show_losses.svg)

*Putting the same experiment through a two-stage referee — the "20/20 victory" against a weak opponent is judged ARTIFACT by an automatic rematch against a strong opponent.*

### Turning the blade aimed at competitors fairly upon myself, too

Here, let me write one admonition to myself. Up to this point I cited "the industry convention of erasing the axes where you lose" with a critical nuance. As a representative example, let me cite the figures of a certain big player's autonomous-driving-oriented model (Cosmos, at the time of self-report / tech report). While the tech report foregrounds the winning axis of **Driving 79.3 vs 47.2 (own vs comparison target)**, the losing axes of **Robotics 57.8 vs 58.2 / General 73.7 vs 77.5** tend to be hidden from the table (the figures are the tech report's, labeled as of the tech report, third-party unverified).

But to be fair, this blade must be turned upon myself as well. **Robotics' 57.8 vs 58.2 is a difference of 0.4. There is ample possibility this falls within measurement error.** The criticism "they hid the losing axis" is valid, but it's fair to also note alongside it the fact that "that loss is by a hair, and whether it's significant is dubious." Erasing the losing axis is exaggeration, and writing up a hairsbreadth loss grandly as a "defeat" is also exaggeration. 82.9% is both a win — "improved 3.8x with QAT" — and a loss — "does not reach 97%." 963.8 → 882.2MB is the win of "resident −72%" and the loss of "peacetime peak unchanged." **Keeping the blade aimed at my own numbers and the blade aimed at competitors' numbers equally sharp** — only when that's achieved does the disclosure of defeat become "material for trust."

---

## b5. I Talked to My Homemade LLM, and Sōseki's Ghost Appeared — Cleverness Is a Structural Ceiling

So far I've been telling the story of "a referee that stops myself." But then why do I not compete on "cleverness" in the first place? I'll show the reason not with an impression but with **the result of actually talking to it.**

I'm often asked, "Given a good enough environment, surely llcore could converse about as well as an ordinary LLM?" I used the best model I had on hand (`realp1`: an 11.9M-parameter character-level LM trained on Aozora Bunko, CPU). I tried talking to it (unedited).

> **Me**: Hello.
> **llcore**: "Such-na, what a long one." "Oh? My, such-sa, what is it. Sensei is chai-good-good"
>
> **Me**: The highest mountain in Japan is
> **llcore**: "It is long. After that, when I saw kunshi, it's number one… Kangetsu-kun… Imo's uncle…"

…Mount Fuji never came out.

What I want you to note is that **only the "Japanese-likeness" is oddly present.** "Sensei," "Kangetsu-kun," "uncle" — these are the **prose style and character names** that appear in Aozora Bunko (Natsume Sōseki's *I Am a Cat* and others). The model can spit out **fragments that imitate** Sōseki's vocabulary and tone. But the meaning doesn't track, and it can't answer the question at all. It has neither the vocabulary nor the knowledge. It doesn't hold the fact "Mount Fuji," so naturally it can't answer.

> **You can summon Sōseki's "ghost." But you cannot "converse."**

Why does it come out this way? This model predicts "the next single character" at the **character level (char-level).** Its top-1 is **about 29% (0.2866).** It's roughly right about 1 in every 3 to 4 characters (the same figure as realp1's fp32 top1 28.66% seen in chapter b1). At this accuracy, "plausible-looking strings of characters" may connect, but a sentence that makes sense, or an answer to a question, will not structurally come out.

> **"It'll reach conversation level depending on the environment" — it won't.** The constraint is not the environment but the **essence** (character level, tiny parameters, tiny data).

What about good hardware? That just means "you can train a larger model on more data," and the moment you do that, llcore becomes a different thing (a cloud-scale LLM). The path to making it clever while remaining a home-CPU char-LM is structurally closed.

### So I Moved the North Star From "Cleverness" to "Memory Efficiency"

This is the turning point of the whole series. Chasing **capability (cleverness) with a CPU character-level tiny LM is a structurally capped losing line.** Acknowledging this was the starting point of honest disclosure. Rather than keep challenging with cleverness something I can't win on cleverness, **I switched to an arena I can win on.**

So I switched the north star to **memory efficiency.** "On a small machine at home, keep the working set small and predictable" — on this arena, even a character-level tiny LM can do meaningful, research-grade measurement and design (the talk of mmap, int8 streaming, and constant state seen in chapter b1 is the fruit of this).

And the division of roles became clear.

- **If you want a conversational LLM:** in FullSense's design, run an existing excellent LLM (Gemma, etc.) **on-prem (at hand)** via `llmesh`.
- **What is llcore:** not a chatbot, but a **research vehicle** (a testbed for evolutionary search, verification discipline, and memory efficiency).

Not "take on ChatGPT with a homemade LLM," but "make a small homemade model into a test stand aimed at a winnable question (memory efficiency)." That only a ghost came out when I talked to it is what made me settle on that resignation.

![A diagram showing that the next-character top-1=28.66% of a character-level tiny LM (realp1 / 11.9M params) determines the capability ceiling. The path of chasing cleverness is closed "by structure, not by environment" (Sōseki's ghost appears but conversation doesn't hold), while the path of betting the same model on memory efficiency is a winnable research arena — hence the branch where the north star is switched.](../../../assets/articles/llcore_b5_capability_ceiling.svg)

*The ceiling is essence, not environment. So I moved the north star from "cleverness" to "memory efficiency."*

When a homemade model's output is incoherent, you normally don't want to show it. But I thought **this "Sōseki's ghost" is precisely what should be shown.** Showing the capability ceiling honestly is both a confession of defeat and **a declaration of strategy.** I didn't give up on cleverness; I left cleverness to the existing LLMs and bet myself on a different value, "small and predictable" — I record that branch point, with actual output attached.

---

## b3. An AI Reviewer Hit Me With a CRITICAL: "That Number Is Fabricated"

Since I moved the north star to "memory efficiency," from here it becomes a story of how to handle the **numbers** of memory. The flavor changes; this time is **the round where the other party (an AI reviewer) was wrong.** But the lead role is not "the AI was wrong"; it is **the verification process itself: "I checked a plausible finding against primary evidence rather than swallowing it whole, and it was wrong."**

It started when I ran an adversarial review on my homemade memory-measurement toolkit. The AI critic (I call it `gem-critic`) declared the following.

> **"realp1's footprint 49.23MB → 13.97MB is fabricated. The correct value is 47.66MB → 12.10MB. This is a CRITICAL inaccuracy."**

Fabricated. CRITICAL. Strong words. Normally one would rush to rewrite the number. But I did not rewrite it. Following the rule touched on in the Shared Yardstick chapter — **a finding from an external AI is adopted only after verifying it one item at a time against actual code and primary sources** — I did not swallow the critic's "correct number" whole, but traced **where both numbers came from** back to the code.

### Chasing a 1.57MB Difference, Byte by Byte

The difference between my tool's value and the critic's "correct value" is, in fp32, **49.23MB − 47.66MB = 1.57MB.** Converted to bytes:

- My value: **49,231,872 B** (= 49.23MB)
- Critic's "correct": **47,659,008 B** (= 11,914,752 parameters × 4 bytes)
- Difference: **1,572,864 B**

Factoring this 1,572,864 into primes —— **6 × 256 × 256 × 4.** `realp1` is a 6-layer model with context length 256. Each layer holds a `256 × 256` **causal-mask buffer** in fp32 (4 bytes). For 6 layers, that's exactly 1,572,864 bytes. **The true nature of the difference was the causal-mask buffer.**

A suitcase analogy may be easy to grasp. Two people measured the weight of the same suitcase; A measured "just the contents," B measured "the whole thing including the suitcase body." **Neither is lying.** They differ only in "what they included when measuring." But unlike the contents, this body has a part that doesn't fold small (the accessory = mask) — and that connects to the later story.

- My tool `int8_footprint_bytes` is a **"resident-weight-byte accounting"** that scans the model's `parameters()` **plus** `buffers()`. Because it counts the weight-system bytes actually resident in RAM, it includes the causal-mask buffer too.
- The number from the old script that the critic deemed "correct" is **params-only accounting.** It counts only the learnable parameters and does not count buffers.

Both numbers are correct under their respective accounting standards. It was **not "fabrication" but "a mismatch of accounting standards."**

![A diagram decomposing the same realp1 model's fp32 footprint with a stacked bar. On top of the green params 47.66MB rides the amber causal-mask buffer +1.57MB, totaling 49.23MB. Factoring the difference 1,572,864B into primes gives 6×256×256×4 = matching 6 layers of mask. It shows in one image the true nature of the accounting difference: the critic's params-only accounting versus my params+buffers accounting, not fabrication.](../../../assets/articles/llcore_b3_accounting_diff.svg)

*Factoring the 1.57MB difference into primes byte by byte exactly matches 6 layers of the causal-mask buffer.*

What's interesting, moreover, is the direction. Including the un-quantizable mask as fp32 makes the reduction rate to int8 come out **conservative, at 71.6%** (since the mask can't be int8, it drags down the reduction). In other words, my accounting, far from over-claiming the reduction effect, actually **shows it conservatively small (underclaim).** Far from looking good through fabrication, the number was strict in the honest direction.

### Lesson — State the "accounting" for memory figures, and meet AI's findings with primary evidence

The memory story has at least 4 different "ways of counting." **params-only** (just the learnable parameters) / **params + buffers** (resident-weight, including fixed buffers such as the mask) / **peak RSS / working set** (the maximum the process held in physical memory at runtime) / **on-disk** (file size). These diverge by hundreds of MB without batting an eye. For the same model, unless you write "which one you mean," readers will compare someone else's number. This "fabrication" uproar happened, in short, **because params-only and params+buffers were pitted against each other.**

And if I had swallowed the critic's "CRITICAL, fabricated" whole, I would have **rewritten my own correct (more conservative, more honest) number into params-only.** Verification stopped that degradation.

> The AI's "CRITICAL" is not a conclusion but **the starting point of an investigation.**
> Until I factored 1.57MB into primes, I couldn't tell which was right.

This is not an article of flashy results. But "doubt your own numbers, doubt others' findings too, and settle it at the end with byte-level primary evidence" — this plain verification discipline is the backbone that makes a small home-CPU experiment trustworthy.

---

## b4. I Built a Safety Gate, and My Own Objective Function Made It Redundant

Back to the "show your losses" series. This time the loss has a slightly different character. **It's not a bug, nor a performance shortfall, but the loss that "the safety device I prepared had no scene where it took effect."**

I bolted a **mathematically sound safety gate** onto architecture search (searching the structure of an AI with evolutionary computation). Concretely, I **prove with Z3 (a theorem prover)** "is this structure stable (does the state not diverge)," and reject, fail-closed, anything that can't be proven. In theory it's a powerful contraption. The claim at design time (I called this G2) was the following.

> **"Without the gate, only about 6% of individuals turn out safe. With the gate, about 100% become safe. Therefore the gate is essential."**

The "about 6%" here is a **prior estimate** placed to motivate the gate (an assumed value with a different setting in mind), not a measured safety rate from this experiment. When measured, this claim was **refuted.**

When I ran a 2×2 experiment with and without the gate and measured the safety rate (safe_rate) —— **even without the gate, the safe_rate was already 95–100%.** Far from "6% → 100%," adding the gate only **nudges safe_rate up slightly from 0.95 → 1.00** (the safe_rate of the retention-focused arm goes from none 0.95 → gate 1.00). The safety rate was high to begin with. The safety gate I believed "ought to take effect" **had, in this setting, almost no job.**

![A bar chart placing the design-time claim "6%→100%" and the measured "95%→100%" side by side. Left = the prior assumption that estimated without-gate at 6%; right = in measurement, the retention arm's without-gate is already 0.95 and with-gate is 1.00, showing the improvement margin is slight and the gate was nearly redundant.](../../../assets/articles/llcore_b4_safety_gate.svg)

_The left "6%→100%" is the prior estimate (the expected value of a different setting), the right is measured. In measurement, even without the gate the safety rate is already high, and the gate had almost no turn to play._

### Why the Gate Became Redundant

You put a guardrail on a clifftop road, and **it turned out the cars all had a habit of driving on the mountain side, opposite the cliff** — since no one veers toward the cliff, the guardrail stands proudly but not a single car hits it. Tracing the reason, it was exactly this structure.

My search's **objective function (reward)** has two parts — **memory efficiency** (lower the state's contraction rate = prefer structures that don't eat memory) and **retention** (prefer structures that reproduce the past well). But **both of these rewards prefer "bounded (non-wild) structures."** A structure that doesn't eat memory and a structure that correctly reproduces the past both lean toward the "state doesn't diverge" side.

> **The objective function was already rewarding the safe side.** So the after-the-fact safety gate merely approved, once more, as "safe," the safe individuals that had already been selected.
>
> **The value of a safety gate materializes only "when the objective function prefers the dangerous (divergent) side."** If the objective already rewards safety, the gate becomes redundant.

This is a boundary condition of design judgment: "Before building a safety mechanism, look at which side the objective function is rewarding." The very fact of learning it doesn't take effect advanced my understanding of the design more than its taking effect would have.

### A Bonus Finding, and a Caveat I Don't Hide

One more, an unanticipated cost came into view. The memory-efficiency reward **nearly halved footprint (an occupancy metric) from 0.375 → 0.149.** As aimed. But the cost was not just a slight drop in retention. **A retention-focused individual** exceeds random initialization on capability by **+0.012** (passes=True), whereas **a memory-focused individual** has a small footprint but its margin over random on capability **shrinks to +0.004, with the verdict passes=False.** In other words, push memory too hard and you enter a region where you lose even **the minimal edge of "cleverer than random" (capability edge).** I thought the trade-off was "memory vs retention," but in reality it had surfaced in a harsher place: "memory vs the very capability of beating random."

I place the caveats without hiding them. **footprint is a proxy for state-boundedness,** not real RSS (real memory) (P1 reservation). And it's a **tiny PoC.** With a different objective function (one that tolerates / prefers divergence), there's ample possibility the same gate decisively takes effect. Z3 actually ran (rejecting 71 of 196 individuals, 0 fallbacks), and the gate mechanism itself functioned soundly. **It's not that "the gate was broken" but that "the gate had no turn to play"** —— this distinction is the crux of this chapter.

Normally, it feels good if you can build a safety mechanism and write "it took effect 100%." But what I got this time was the result "when I built it, with this objective function, it didn't take effect." Rather than boasting of a gate that takes effect 100%, being able to draw the line by measurement on "when it takes effect and when it's redundant" is an asset that pays off for the next design. A loss is a loss. But this loss is worth recording.

---

## b7. Re-wiring the Tools of a Losing Line of Research Onto a Winnable Arena

In chapter b5 I wrote, "Chasing capability (cleverness) is a structural losing line, so I pivoted the north star to memory efficiency." This time is the **practice** of that pivot. "Pivot" conjures the image of "throw away the failure and rebuild," but there's a more advantageous move. **Don't throw away the tools (mechanisms) built in the losing research; just swap the north star onto them.**

**What was lost:** the evolutionary search for capability (cleverness). With "evolved structure ≈ random structure" (`evolution_vs_random().passes = False`), there was no gain in cleverness (the same confirmed losing line as the capability-edge talk in chapter b4).

**What remained (mechanism assets):** the whole set of tools built in the course of running that losing line remained intact. **minimal_ga** (a minimal evolutionary-search engine), the **verified-plasticity gate** (a gate that mathematically proves "the structure is stable = contracting" via Z3 / SDP), and the **falsification harness** (a verification stand that goes after refuting claims). These were built for "cleverness search," but **as tools they don't depend on the north star.**

A fishing analogy works well. You aimed for a certain fish (cleverness) and assembled a fine rod, net, and fish finder, but no matter how hard you tried, that fish wouldn't bite. To throw away even the tools here is wasteful. The rod doesn't know "which fish you're catching." Cast the line, take the bite, reel it in — this function doesn't change no matter what quarry you target. **The tools don't depend on the goal.** So you only need to change the fish you target (the north star).

So I transplanted this whole mechanism into **memory-efficiency search.** The evolution engine, as is, **swaps its fitness (reward) to a memory metric**; the proof gate, as is, **serves as a gate that soundly guarantees "is the search result stable"**; the falsification stand, as is, **serves to doubt memory-efficiency claims.** Swap only the north star (capability → memory), and throw away not a single mechanism.

![A diagram showing that the 3 mechanisms built in the lost cleverness search (evolution engine, contraction-proof gate, falsification stand) are kept as is, swapping only the north star from capability→memory. As the figure that took effect after transplant, false-admit is 84% for the empirical gate versus 0% for the sound-proof gate, showing the over-80% miss rate goes to zero.](../../../assets/articles/llcore_b7_rewire.svg)

*Swap only the goal (north star); throw away not a single tool. Re-place the value on the guarantee side.*

### An Honest Stance — Most of It Is "Re-derivation"

If I fudge this I lose trust, so I state it clearly. **Most of this design is a re-derivation of known methods.** "Make a memory metric the fitness" → common sense of HW-NAS (MnasNet) / HAQ. "Scalarize accuracy × memory" → the standard of multi-objective optimization (Deb 2000). "fail-closed constrained evolution" → the standard of constrained NSGA-II. "Make the verifier a gate" → the framework of CEGIS (counterexample-guided synthesis). None of these are my inventions (the prior-art confidence is high).

So what is the honest originality? **Only the conjunction of four narrow points** — evolution × a sound contraction gate × a memory north star × recurrent dynamics, running this particular combination reproducibly on a home CPU. And there is one more measurable original contribution.

### The Figure That Took Effect — The Difference in Discriminative Power Between the "Empirical Gate" and "Sound Proof"

There are two kinds of gate. The **empirical gate** (judging "roughly seems stable" by rule of thumb) and the **sound proof gate** (proving "it is stable" mathematically with Z3/SDP). When I measured the **discriminative power** of these two, a clear difference came out.

> **The empirical gate's false-admit (mistaking a dangerous one for safe) is 84%.**
> **The sound proof gate's false-admit is 0%.**

The false-admit here is **an error in the direction of passing a dangerous one as "safe,"** the opposite direction from the error of stopping a safe one just in case (over-cautious). This one connects directly to accidents. Over 80% of what passed as "roughly seems stable" was in fact not safe. Meanwhile, what was mathematically proven had zero misses. An empirical check guesses "this resembles one I've seen before, so it's probably fine," so dangerous ones that don't resemble anything slip through. By contrast, a proving apparatus rewrites the structure into equations and **logically nails down that "not a single path toward runaway exists"** — so "happened to miss it" cannot occur in principle. This is the true nature of the 0% miss. **"Plausible safety" and "proven safety" are different things** — that came out in numbers.

### A Caveat I Don't Hide — branch A Does Not Recover Cleverness

This is the most important honest disclosure. **This transplant does not recover capability (cleverness).** The value lies solely on the **guarantee side.** Not "becomes clever" but "**stability can be proven.**" Capability and guarantee are orthogonal; losing on cleverness does not mean losing even the value of guarantee. To turn the tools of a losing line onto a winning line meant **re-placing the value on a winnable axis (guarantee).** Reservation: footprint is a proxy for state-boundedness, not real RSS (P1). Making real footprint the fitness is future work (P2). prior-art = MnasNet / HAQ / NSGA-II / Deb 2000 / CEGIS / Dohare 2024.

> What was lost is the "goal (cleverness)," not the "tools (evolution, proof, falsification)."
> Swap only the goal, and the tools come back to life on a winnable arena.

The proof gate built for cleverness search came back to life as the apparatus that "guarantees safety at 0% false-admit" in memory-efficiency search. **Failure becomes a double loss if you throw it away mechanism and all.** Swap only the north star —— that was the most realistic craft for turning a loss into an asset.

---

## b2. I Silenced "We Won by +15.3%" With My Own Noise Floor

Up to here it was mainly the story of "showing the losses." This time is one step beyond —— **the story of starting to say "we won," then retracting that victory declaration with the noise yardstick I built myself.**

The architecture search (NAS) I ran on my home CPU overnight (about 6.6 hours, 386 real evaluations) first shouted this.

> `memetic frontier dominates greedy: hypervolume +15.3%`

The design group searched by evolutionary computation (NSGA-II) **beat the naive greedy baseline by +15.3% in hypervolume.** A victory declaration. But the **headline verdict** of the same run was this.

> **verdict: suppressed — selection optimism exceeds noise floor**

The same run says, out of one mouth, "we won by +15.3%," and out of the other, "don't release that win." **This very contradiction is the mechanism I deliberately rigged this time.**

The subject is **a Pareto search of memory vs quality.** For each layer of the Transformer, do you keep the heavy softmax attention mixer, or replace it with a lighter sliding window or linear attention — the more you replace, the more memory you save (the KV cache shrinks), but the quality degrades (the next token becomes harder to hit). The target is an off-the-shelf `Qwen2.5-0.5B-Instruct` (24 layers) converted on the CPU. The baseline (all-layer softmax) nll is 4.4155 (ppl ≈ 82.72). I compare two ways of searching. **greedy** (a naive method that adds "the most advantageous replacement right now" one layer at a time) and **memetic** (the elaborate method of NSGA-II + local search). The question is, "Does the elaborate memetic truly beat the naive greedy?"

### Why You Must Not Believe "We Won" Instantly

There are two pitfalls. Not my own bright idea, but established problems already warned of by prior NAS research.

**Pitfall A: the proxy is noisy in the first place.** Evaluating candidates for real is too expensive, so you use a proxy (a surrogate metric, here "the perplexity of next-token prediction"). It's cheap but **noisy** and often diverges from real performance. MTF-PDNS (Pareto Dominance-based Novelty Search with Multiple Training-Free metrics, Vo & Luong et al., arXiv:2407.20656, 2024) itself writes "performance objectives do not fully align with the actual performance ... as is often the case with training-free metrics." To make it clear this is not reinventing the wheel — MTF-PDNS's countermeasure to this problem is "keep the search space broad with a novelty/diversity score." **This work's countermeasure is in an entirely different direction: "quantify the uncertainty of the proxy itself and suppress the claim of a win."** One copes with diversity, the other measures uncertainty to discipline the claim. The lines of solution differ, so it's not a rehash of an established area.

**Pitfall B: winner's curse.** When you evaluate many candidates and pick the one that came out best, the picked value mixes in "the true ability + the luck that happened to be good in that evaluation window." The win margin measured in the same window is structurally an **overestimate.** It's the same phenomenon as the famous "boast about the selected one with the same data and it inflates" in A/B testing and GWAS. The zero-shot `+15.3%` was precisely a win measured in the very window used for selection.

In a mock-exam analogy — your new study method raised your score by 15%, but that mock exam used the very problems you'd practiced many times to perfect the new study method. To confirm your true ability, the only way is to re-measure on a different set of problems you've never seen before.

### The Answer I Rigged — Physically Separate the Search Window From the Referee Window

So the core this time. **I separated the window used for search (selection) from the window used for final judgment so they don't collide in token position.** The search pool (fast pool) is 8 windows; the judgment pool (holdout) is a separate set of 12 non-overlapping windows shifted by 8,192 tokens from there. When you re-measure a design that won at selection in a brand-new window never used for selection, you can flush out the **optimism bias (optimism_gap = selection − holdout)** (lower Δnll is better).

| memory saving | Δnll (selection window) | Δnll (**holdout**) | optimism gap | 95% CI |
|---:|---:|---:|---:|:--:|
| 11.5% | 0.0184 | 0.0170 | 0.0014 | 0.0144..0.0200 |
| 27.1% | 0.0754 | 0.0696 | 0.0058 | 0.0632..0.0767 |
| 45.9% | 0.3675 | 0.3404 | 0.0271 | 0.3216..0.3587 |
| 61.3% | 0.5962 | 0.5419 | 0.0543 | 0.5203..0.5653 |
| **68.5%** | 0.6667 | 0.6015 | **0.0652** | 0.5774..0.6232 |
| 72.2% | 1.1744 | 1.1107 | 0.0637 | 1.0776..1.1422 |
| 83.9% | 1.2436 | 1.1818 | 0.0618 | 1.1500..1.2109 |

The selection-window values (2nd column) **look better than the holdout (3rd column) in nearly every row.** That difference is the optimism gap. The maximum is **0.0652** at the 68.5%-saving point. The judgment logic is this.

> **The maximum optimism_gap (0.0652) exceeds the floor of half the bootstrap CI width (0.0204) → suppress the victory claim of the individual frontier point.**

If the optimism bias is larger than the noise floor, I won't let it say "we won at this point." **I silenced my own win with my own noise yardstick.**

### But I Don't Silence Everything — Count Confidence by Granularity

This is the design I like best this time. **Don't treat "the win" as monolithic.** The win claim of an **individual point** on the frontier was suppressed as above. But for the **hypervolume (HV, the area the frontier covers = overall goodness)** that folds the whole frontier into one, separately came out the following.

> **HV gain (memetic − greedy, holdout) = +16.8% (95% CI 16.2..17.7%, p_memetic_wins 1.000)**

Moreover, this "win" is designed to **fire only when the lower bound of the CI exceeds 0** (never fire on a point estimate). This time the CI lower bound is 16.2% > 0, so I kept **only** the memetic advantage in the HV dimension.

You may snag on "you re-measured the selection-window +15.3% in a brand-new window, yet the holdout HV gain is +16.8% — **the win margin actually grew.**" This is not a contradiction. **The individual-point optimism_gap (selection − holdout) does indeed shrink in a brand-new window,** while **HV is a quantity on a different basis.** +15.3% is the area ratio the selection-window frontier covers; +16.8% is the area ratio the holdout-window frontier covers, and both the placement of the points composing the area and the choice of reference point change per window. **It is not a yardstick on which area quantities (HV) can be directly compared as "shrink/grow."** That's exactly why the HV win is kept not because "it's bigger than +15.3% so it's strong," but **only on the independent grounds that the CI lower bound exceeds 0.**

![A diagram re-measuring the selection-window +15.3% on a fresh holdout, taking and discarding by granularity: individual points are SUPPRESSED, the overall HV is KEPT, and the needle is UNTESTED.](../../../assets/articles/llcore_suppress_win.svg)

*Figure: re-measuring the "+15.3% win" from the selection window on a fresh holdout. Since the optimism bias (max 0.0652) exceeds the noise floor (0.0204), the win of individual frontier points is suppressed (SUPPRESSED); only the overall HV gain +16.8% (CI_lo>0) is kept (KEPT); long-distance retrieval is disclosed as untested (UNTESTED).*

The gateway that applies suppression has 5 stages. (1) **paired multi-window bootstrap CI** — attach a confidence interval of window-pair resampling to the proxy itself, and don't let it speak with a point estimate. (2) **remove winner's curse with fresh holdout** — if it exceeds the floor, suppress the verdict. (3) **proxy-vs-judge Kendall τ** — if τ<0.7, demote a positive verdict to 'suggestive' (this time **τ=1.00** so no demotion). (4) **HV gain fires only at CI_lo>0.** (5) **memetic ≈ greedy is stated explicitly as an honest negative.** Where prior research admits "the proxy is noisy," this work advances to the stage of "actually measuring that noisiness and suppressing the claim instead." Rather than inventing a new search operator, I place the value on having added a **disclosure layer that quantifies uncertainty and disciplines the verdict.**

### An Unverified Gap I Don't Hide (honest gap)

Honesty is determined not only by "where I measured" but also by "declaring where I didn't measure."

- **The context-length sweep goes up to 256/512/1024. Long-distance retrieval (needle / passkey) above 2048 is also unmeasured.** Sweeping the context length on the most aggressive design (83.9% saving), Δnll increased the more degradation the longer the text: **256:0.761 → 512:1.012 → 1024:1.182** (a sign of the constant-state weakness growing). This also agrees with the theory side — prior research that reduced the long-distance memory degradation of linear recurrence models to a decay spectrum (*Optimal Decay Spectra for Linear Recurrences*, arXiv:2604.07658) proves that with random initialization the minimal spectral gap collapses to O(N⁻²), and the error **degrades effectively algebraically (sub-exponentially)** at long contexts. Moreover, even if I could measure the needle, **I won't settle it on a perfect score alone** — prior research tracking the learning dynamics of long-context continual pre-training (*Revealing the Learning Dynamics of Long-Context Continual Pre-training*, arXiv:2604.02650) reports that the NIAH (needle-in-a-haystack) score shows "deceptive saturation" earlier than the intrinsic convergence, and that PPL-based analysis can more correctly track the actual improvement. The reading "needle is perfect, so long-distance memory is fine" is itself one form of the "feeling like you've won" that this article warns against.
- **In trying to measure that, my own machine's memory broke first (an honest punchline).** In the original draft I wrote "2048 structurally won't come out because the inner-loop length is 1024," but this was **wrong** (the sweep process can cut a window of arbitrary length independent of the inner-loop length, and the corpus is sufficiently long at 2.3 million tokens). After correcting it, I actually re-ran the 2048 sweep + needle retrieval, and —— **on the home CPU (physical RAM 3.6GB), the 2048-token full-attention forward swelled the working set to 3.9GB, exceeded RAM, and thrashed in swap.** I tried twice and twice failed to complete. There is a laughable irony here —— **the theme of this chapter is the failure mode "constant-state models overflow memory at long contexts," yet my machine that tried to measure it overflowed memory at, precisely, long contexts.** The correct next move is offload to GPU; until then, I honestly leave **2048+ long-context retrieval as "unverified."** I **do not say** "therefore long-distance retrieval is fine."
- **attention-KL is diagnostic-only, not wired into fitness.** The per-layer forward KL (softmax‖student) is mean 3.68 / max 7.67 (layer 9). It is a diagnostic for seeing where degradation lies, and is not included in the search's objective function.

The industry leaderboard is filled with "we won." But there are few examples that disclose, with the same precision, whether that win is the optimism bias of the selection window or true ability that survives in a brand-new window.

> Before you feel like you've won, doubt the breakdown. Make that "doubt" not an impression but a **threshold.**

After "showing the losses," next is "being able to suppress the win yourself." Honesty moves from something you show to something you **engineer.**

---

## b6. Folding a Winning Line Into a Form Anyone Can Accept in One Line

Having stacked up the "show your losses" rounds (the defeat of 2bit, the swing-and-miss of the safety gate, the AI review's mistaken finding, the homemade LLM that can't converse), in b2 we saw "suppressing the win yourself." Last is its **paired round.** Not a loss, but **the story of folding a winning line into a tool.** But the lead role is not a new invention; it is **the plain honesty of "making it a form anyone can accept in one line."**

llcore had several **verified primitives** of memory efficiency — int8 quantization (weights to a quarter), mmap streaming load, constant-state recurrent (memory stays flat even at long contexts), the cap-gate (a gate that accepts whether capability is not broken). The problem was that these were **scattered piecemeal** across `scripts/` and `lm/`. The kitchen tools were stuffed here and there in the drawers, and "which one to call to learn what" was invisible from outside. The winning line exists, but it can't be used.

So I built one **facade (reception counter)** called `llcore.memory` and added `measure_memory()`. **Hand it one fp32 model, and it returns in one go "how much smaller it gets, and what it loses in exchange."** The returned `MemoryReport` folds three axes into one sheet. (1) **Quantization footprint (static)** how much smaller it gets at int8, (2) **capability retention + cap-gate** whether capability is preserved (if broken, stop at acceptance), (3) **context-length KV growth (dynamic)** how memory grows as the context lengthens.

![A diagram folding the scattered int8 / mmap / constant-state / cap-gate primitives into a facade (llcore.memory), where measure_memory() gathers the 3 axes of footprint, capability, and KV growth onto one sheet. It shows the structure where the only originality is the single acceptance operation: write out int8 only when the cap-gate PASSes, and refuse fail-closed when FAIL / no corpus.](../../../assets/articles/llcore_b6_packaging.svg)

_The whole picture of folding the re-derivation of existing methods into one reception counter, with the only originality — cap-gate acceptance — built in as a fail-closed branch._

### Honest Numbers From Real Hardware

Measurements on actually-trained models (footprint is resident-weight accounting of params+buffers, a different thing from on-disk — this connects directly to the accounting story in chapter b3):

| model | fp32 | int8 | reduction |
|---|---:|---:|---:|
| realp1 | 49.23 MB | 13.97 MB | **71.6% reduction** |
| multi_smoke | 5.50 MB | 1.51 MB | **72.6% reduction** |

On the capability side, on the full Aozora Bunko multi text (330,368 tokens), **top-1 retention 100.0%** (fp32 0.3629 → int8 0.3631), cap-gate PASS. Here, an **important honest note**:

> That int8 "beat" fp32 by just 0.0002 is **not an improvement.** It is the measurement noise of tied argmax.

At this scale the capability cost of int8 quantization is effectively zero —— a corroboration of "per-channel int8 degradation <0.1%" from a separate article, and by no means a claim that "int8 became cleverer." The better the number that comes out, the more you doubt the breakdown and write it honestly. This too is part of packaging.

The 3rd axis (context-length KV growth) also comes out in the same report: for realp1, T=256's 4.72MB → T=2048's 37.75MB, a **×8.0 linear growth** (while `constant_state_bytes` is constant, independent of context length). The "term that grows vs the term that stays flat" touched on in the Shared Yardstick chapter is now reproducible by anyone with one tool.

### Making Acceptance fail-closed — This Is the Only Honest Originality

I'll write the stance honestly. **The primitives themselves are not new inventions.** int8 / mmap / constant state are **re-derivations** of methods already established by llama.cpp and GGUF. So what is my honest originality? **"The operation of accepting a footprint win, fail-closed, through a capability gate"** —— just that one point.

Concretely, the CLI's `--save-int8` writes out an int8 checkpoint **only when the cap-gate PASSes.** If capability acceptance fails (FAIL), or capability can't be measured because no corpus is specified, it **refuses the write fail-closed** (an operator can explicitly override with `--force`). `measure_memory` returns capability as `None` if there is no evaluation data, and **"does not fabricate what doesn't exist."**

Why is this needed? As we saw in chapter b1, **a PPL-only gate happily PASSes a broken low-bit model whose top-1 has halved.** Structurally stopping, at acceptance, the accident of mistaking "got smaller" for "usable" —— this is the only claim built into packaging. In a cooking-recipe analogy: even if one person alone discovers "an amazing trick that's super tasty," it can't be reproduced while the amounts and steps stay vague. Only when it's written up into a recipe with inspection steps — "this many grams of ingredients, the heat is here, taste-check and confirm here" — can others make the same flavor, and moreover prevent accidents like "this is burnt, don't serve it."

I place the caveats without hiding them. footprint is **resident-weight accounting (params + buffers)** that even includes the fp32 of the un-quantizable causal-mask, the conservative side, a different thing from the on-disk file size. The "the better the HW, the more it pays off" design guidance (int8 → true int8 GEMM on GPU / mmap → shared page cache on large RAM / constant state → long contexts) are all **design hypotheses, unmeasured**; speed is not measured in this series. I assembled the facade in a state where 870 unit tests pass, but this is a guarantee that "the wiring is correct," not a guarantee that "the algorithm is new."

> The value of research lies not only in the flashiness of novelty but also in **"turning an existing winning line into a tool that anyone can reproduce and accept."**

Re-derive the frontier on a home CPU, and turn it into a tool that answers "did it get smaller? is it not broken?" in one command. It's plain. But what should be placed at the end of a series that honestly shows its losses is, I think, this stance of **"leaving the winning line as an acceptable tool."**

---

## Landing — Why "Losses" Become the Core of an Individual's Career

Through 7 chapters, I had my own achievements stopped by my own referee, again and again. 2bit QAT, the over-generalization of mmap, the peak expectation of int8 streaming, evolution's 20/20, NAS's +15.3%. The safety gate had no turn to play, and the ceiling of cleverness was structurally closed. The AI reviewer's "it's fabricated" was overturned with primary evidence, the tools of a losing line of research came back to life on a different arena, and the winning line was folded into a tool that can be accepted in one line.

**First, victory claims can be manufactured infinitely, but the disclosure of losses cannot lie.** "It didn't reach 97%," "the peak was nearly unchanged," "I negated my own 20/20 myself," "I silenced +15.3% with my own noise floor" — there's no gain in padding numbers like these. If anything, padding makes you lose. That's exactly why a person who puts these forward of their own accord is presumed to be "this person is probably putting forward the good numbers with the same honesty too." **The disclosure of defeat is the security deposit that guarantees the credibility of the victory numbers.**

**Second, what I built was not a "clever model" but a "mechanism that stops myself."** The cap-gate (97% retention), the strong-gradient meta-gate, the holdout window for removing winner's curse, the cap-gate fail-closed acceptance — none of these are apparatuses for surpassing someone on the outside. They are **apparatuses for stopping my own optimism myself, before publication.** And they actually stopped it. Rather than building one bench to surpass the outside, **building one inner referee that stops myself is far harder, and far more trusted.**

**Third, this was possible precisely because it's a home-CPU tiny model.** As in the disclaimer at the start, the numbers shown here are a char-LM PoC of 0.81M–130M (only chapter b2 CPU-converts a 0.5B off-the-shelf model), and do not refute the performance of a 12B–64B real LLM. Trading blows with the big players on "cleverness," I have no chance of winning. **So I stepped off the arena of cleverness and shifted my stance to the arena of measurement discipline.** I draw the line, without hiding it, on "which numbers can be trusted and which cannot," on top of my own small experiment, and show it. With this, even without owning a 12B model, it becomes a value an individual can offer.

Stories of wins are forgotten once read. **The story of pinning a loss to a number and honestly opening it up remains as trust.** Turning the defeat log of a home PoC into material for trust — that is the true nature of what I've been doing throughout this part. The more abnormally good a result comes out, the more you doubt the breakdown before you feel like you've won — not at others, but at yourself first. That is the foundation of this entire series.

---

> **Restated (the reach of this article)**: Every figure is a measurement from a home-CPU tiny char-LM (0.81M–130M; only chapter b2 CPU-converts a 0.5B off-the-shelf model) PoC. The quantization footprint is a measured value; inference speed is unmeasured because it uses simulated quant. The score is a next-token-nll proxy and contains no conversational-quality claims. It is not a claim to directly refute or surpass the big players' real LLMs (12B–64B) on performance, but **a comparison on the dimension of method, philosophy, and measurement discipline.** The reproduction code and output JSON (`scripts/*.py` / `out/*.json` / `src/llcore/lm/eval.py` / `src/llcore/lm/quant.py` / `src/llcore/runtime/eval_proxy.py` / `src/llcore/fitness/memory_objective.py` / `src/llcore/memory.py` and others) are published on GitHub.
