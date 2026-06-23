# A Win by Structure — Constant State, Dominant Terms, and Design Smells

> Consolidated edition of the series "The June 2026 LLM Industry, Seen from a Home CPU" — **Arc A**.
> Four articles (control theory × SSM / context-memory blowup / static dominant term / test smells) re-woven into one.
> A single theme runs through all of them: "**Don't lower the bits — go after the architecture and the dominant term.**"
> The thread I follow is my own chain of failures: while I was fixated on shaving a few MB off the weights via quantization, I was overlooking the terms that actually matter (the quadratic term that swells with context, the scaffolding of the language runtime). I knock them down one by one against measured curves on a home CPU.

---

> **[Series-wide caveat (pinned once, here, for this arc too)]**
> Every llcore measurement in this series is a **PoC on tiny models on a home CPU (char-LMs of 0.81M–130M parameters, n_embd 256 / 4-layer class)**, and it does **not** directly refute or surpass the performance of the large vendors' real LLMs (12B–64B). Comparisons are made at the level of **method, philosophy, and measurement discipline**. Quantization figures are footprint (occupied bytes) = actually measured, but **simulated quant = inference speed is not measured**.
>
> The "×1.00," "×2.65," "×7.53," "142×," and so on that appear in this arc are all **CPU measurements of tiny models**. What you should read is **not the absolute values but the trend — the shape of the growth, which term dominates**. Reading recurrent (RWKV/Mamba family) **as a state-space model** is a legitimate connection in the literature, but **I do NOT say "because control theory solved it, recurrent comprehensively beats Transformer."** Rather, I write it honestly, as llcore's own null result: it **wins structurally on the memory axis, but on the capability (smartness) axis recurrent may be inferior to Transformer**.

---

## Introduction — Not "Lower the Bits" but "Go After the Structure"

When people try to make AI lighter, the first thing they think of is "compress the contents and thin them out." Weights from fp32 to int8, then down to 2-bit. I was the same. I was diligently shaving the model's weights.

But as I piled up measurements on my GPU-less home PC, I found, one after another, that the terms I was shaving **were not the dominant terms in the first place**. What actually matters is not "lowering the bits" but the more structural choices: "**choose an architecture that simply does not have the term that swells as length grows**" and "**measure and attack the biggest piece of scaffolding**."

This article hands over that realization in four steps.

1. **a7 — Build the bridge:** Reading recurrent (RWKV/Mamba) as a state-space model that "carries the past in a constant state" reveals it to be a re-performance of 60-year-old control engineering.
2. **a8 — Dynamic dominant term:** Stretch the context length, and Transformer swells super-linearly in both memory and time, while recurrent stays flat (measured on real hardware).
3. **a9 — Static dominant term:** Right next to where int8 shaved a few MB, the torch runtime was eating 184MB. Measure the dominant term before rewriting in Rust.
4. **a11 — Universal lesson:** A flaky test is a design smell. Separate the decision into a pure function.

Running through it all: "**Don't go after bits — go after the architecture / the dominant term.**" Now, let's cross the bridges in order.

---

## Part 1 (a7) — Control Theory Had Long Since Solved "Carry the Past in a Constant State"

## Hook — My measured values were in a 60-year-old textbook

One night, on my GPU-less home PC, I was running an experiment like this.

"If I stretch the context length T (the number of tokens the model sees at once) from 256 to 2048 — 8×  — how does the actual memory usage during inference (peak working set) change?"

I isolated it in a separate process and measured the real RSS (Resident Set Size) reported by the OS. The results were these.

| Context length T | GPT (Transformer) peak WS | Recurrent peak WS | RWKV peak WS |
|---|---|---|---|
| 256 | 229.8 MB | 205.0 MB | 215.3 MB |
| 512 | 247.3 MB | 205.1 MB | 216.0 MB |
| 1024 | 330.5 MB | 204.8 MB | 215.7 MB |
| 2048 | **607.9 MB** | 204.8 MB | 215.2 MB |

(`scripts/recurrent_runtime_rss.py` / `out/recurrent_runtime_rss.json`. Windows 11 / Python 3.11 / torch 2.12.0+cpu.)

Make the context 8× larger, and **Transformer's peak swells to ×2.65, while Recurrent and RWKV are ×1.00 — that is, flat**. Subtract the fixed baseline (~205MB = torch runtime + weights), and only Transformer's "cost that grows with context" stretches from ~25MB to ~403MB — **super-linearly** (+277MB on the 1024→2048 step alone).

> **In plain terms (what is a working set):** It is the amount of "pages the process is actually keeping in physical RAM right now." Memory that merely sits on disk or has merely been reserved is not included. Think of it as "the real quantity of memory it is actually touching." The peak WS is its maximum. Whether you run out of memory when processing long text turns on this **actual-usage peak**, not on the reserved amount. So rather than "analytically it should be this," I deliberately backed it up with the RSS the OS reports.

> **Honest reservation on the measurement (no term decomposition):** Reading this "super-linear" growth as "the quadratic term O(T²) starting to dominate at large T" is **an interpretation from the analytic model in which the KV cache is linear and the attention matrix is quadratic**. What I measured is **only the combined peak RSS** of the process; I did **not** decompose that increment into "the KV-linear part," "the attention-quadratic part," and "the intermediate-buffer part" and measure them separately. The observed fact only goes as far as "Transformer's context-dependent cost grew super-linearly"; that "that super-linearity is the quadratic term becoming manifest" is an interpretation based on the analytic model — I keep the two distinct.

### The same direction appears on the time axis too (scaling exponent)

I measured not only memory but also **the time inference takes** over the same T sweep (`scripts/recurrent_latency_sweep.py` / `out/recurrent_latency_sweep.json`, 11 measurements per point, `torch.set_num_threads(1)`). When T is stretched 128→2048 (×16), the **scaling exponent p** (estimating time ∝ Tᵖ by log-log least squares; min-based, robust to contention) is: **GPT p≈1.37 (clearly super-linear, leaning toward O(T²); ×45.6 for ×16 context) / Recurrent p≈1.00 / RWKV p≈0.99 (both nearly linear, O(T), ×16)**. The pattern seen in memory — "Transformer swells with context / recurrent and RWKV are structurally light" — reproduced in **the same direction on the time axis**.

> **Honest reservation (cross-mode absolute ms are not comparable):** Recurrent/RWKV are measured here with a **per-step loop in Python** (T function calls), where the interpreter-call overhead dominates — it is meaningless to speak of "recurrent is faster/slower" in absolute terms; the only thing to read is **the growth within each mode (the scaling exponent)**. Note that at repeats=7 initially, RWKV's T=128 became an outlier from startup noise and fouled the slope (p≈0.5), but **increasing to repeats=11 made RWKV converge to p≈0.99 too** — I keep the whole episode rather than erasing the noisy point, raising the repeats to knock it down instead.

![Scaling of inference latency (log-log). Each mode is normalized at T=128 to compare only the "shape" of the growth. GPT (red) deviates above the ideal linear line (dashed, p=1) to ×45.6 at ×16 context — super-linear. Recurrent and RWKV (green, nearly overlapping) follow the dashed line at ×16 for ×16 context (p≈0.99–1.00) — linear. Takeaway: not a contest of speed, but how the structure differs in each method's tendency to get suddenly heavier as you lengthen things.](../../../assets/articles/llcore_latency_scaling.svg)
> Since each mode is normalized to its own T=128, all you look at is "how steep the rise is." Only GPT deviates above the ideal-linear (dashed p=1) line = super-linear. Recurrent and RWKV nearly overlap and follow the dashed line = linear.

### A sharper axis still — decode (generating one token at a time) and amortization

Up to here we covered the prefill/batch cost of "forwarding length T **all at once**." But what dominates the felt latency of a chat is the decode cost: **with a context of T tokens already in hand, producing the next 1 token**. In the same process, the same model, and the same conditions, I timed **both prefill (build the state / read the entire context) and decode (the next 1 token)** (`scripts/decode_latency_sweep.py` / `out/decode_latency_sweep.json`. **Measurement done in a clean runner with no memory pressure**).

The measurements (n_embd=256/L4, T=128→2048, ×16) are these: **Recurrent and RWKV grow at O(T) in prefill (×15.6 / ×17.5, exponent p≈1.0), but drop to flat O(1) in decode (×0.98 / ×1.10, p≈-0.002 / 0.03)**. Meanwhile **GPT has prefill and decode essentially coincide at every T** (prefill≈decode at each T, e.g. 99≈96ms / 958≈957ms), and both **grow at the same exponent p≈1.37**. That **GPT's decode exponent (1.365) matches its prefill exponent (1.372) exactly** is the best possible corroboration that "for a cache-less GPT, decode is the same computation as prefill."

![Per-token cost of prefill (state-building O(T)) vs decode (next 1 token O(1)) (log-log). Measured within the same run. Recurrent / RWKV rise in prefill (p≈1.0, O(T)) but drop to flat in decode (p≈0, O(1)) = amortization. GPT (no KV cache) has prefill and decode overlap (p≈1.37, inseparable) = re-forwards the entire context every token. Takeaway: the longer the generation, the more recurrent's "just add one move to the state" pays off.](../../../assets/articles/llcore_decode_latency.svg)

**The crux is recurrent's amortization.** Think of the "state" here as **a fixed-size notebook that summarizes everything so far**. Recurrent can **pay separately** for "building the context (prefill: update the state T times over T tokens = O(T))" and "adding one move to the built state (decode, O(1))." In long generation, prefill happens only once at the start, and every token after that is O(1) — this is what determines the felt experience of streaming. GPT (cache-less) cannot make this separation, and **one decode step ends up being the same "re-forward of the entire context" as prefill**. **Because I measured this in a single run, this "only recurrent drops O(T)→O(1) while GPT does not" comes out as a direct contrast, not a cross-run comparison.**

> **Honest reservations:** (1) As with prefill, **cross-mode absolute ms are not comparable** (recurrent/RWKV are measured with a Python per-step loop). (2) **This GPT has no KV cache.** Production LLM serving drops decode to O(T)/token with a KV cache, but **even with a cache, GPT's decode keeps growing with T, qualitatively distinct from recurrent's flat O(1)**. (3) That GPT's exponent stays at ~1.37 rather than the theoretical 2 (O(T²)) reflects a regime where, in this small model (n_embd=256), the quadratic term has not entered full dominance.

When I saw this gap, for a moment I thought it was a "new discovery." But it was not. It is a re-performance of how **1960s control engineering had thoroughly organized the problem of "carrying an observable system in a bounded state."** The "S" of the **State-Space Model (SSM)** to which RWKV and Mamba belong is the S of State. That is no coincidence.

## First, the Transformer Side — Why It Swells with Context (the KV Cache, a "Keep Everything" Design)

The core of Transformer is **self-attention**. When processing a new token, it measures, by inner product, "how related" it is to every previous token, and adds in the strongly related ones with weights. The strength is that it can **match exactly regardless of position**. It can directly access "a proper noun that appeared 3000 tokens ago" with no decay. The cost is that it **cannot match unless it keeps the entire past**. It holds onto each token's Key and Value as a **KV cache**.

> **In plain terms (what is a KV cache):** Attention makes three things from each token — Query, Key, and Value. When advancing generation one token at a time, remaking every past Key and Value each time would be wasteful, so the once-made Keys/Values are stockpiled. That is the KV cache. **It is a handy working memory, but it keeps growing linearly as tokens increase.** Think of it as "an uncompressed memory that stores all past observations raw."

If we plot how memory grows against context length T with a structural plot (theoretical values computed from the config), it comes out like this.

| T (context length) | Recurrent state (measured) | RWKV state (measured) | GPT KV cache (analytic) | GPT attention matrix (analytic) |
|---|---|---|---|---|
| 64 | 2,048 B | 10,240 B | 262,144 B | 65,536 B |
| 1024 | 2,048 B | 10,240 B | 4,194,304 B | 16,777,216 B |
| Ratio (context ×16) | **×1.00** | **×1.00** | **×16 (linear)** | **×256 (quadratic)** |

(`scripts/memory_footprint_harness.py` / `out/mem_footprint.json`. state_bytes is a real byte measurement; KV/attn are analytic values derived from the config.)

Transformer has two things that grow. The **KV cache** is **linear** in T (×16 for ×16 context); the score matrix of exact long-context attention is T×T, so it is **quadratic** in T (×256 for ×16 context). That the hook's real-hardware peak RSS showed "context cost going super-linearly from ~25MB→~403MB" can be interpreted as **this quadratic term (O(T²)) starting to dominate** at large T (though, as noted above, this is a reading from the analytic model, not a measured term decomposition).

> **Honest reservation:** In implementation, GPT.generate crops the context to block_size internally, so merely running it is bounded. The "linear / quadratic" above is the quantity required if you **stretch block_size and attend over genuinely long context**. To be precise, it is not "Transformer must explode," but rather "**unless you give up exact long context**, it is a structure that pays cost linearly/quadratically in context length."

## The Recurrent Side — Why It Stays Flat ("Folding" the Past into a Finite State)

A recurrent model does not keep the entire past. Instead, it holds a single **fixed-size state `s`**, and updates `s` each time a new observation arrives.

```
s_t = f(s_{t-1}, x_t)      # update the state from the previous state and new input
y_t = g(s_t)               # produce the output from the current state
```

The point is that the size of `s_t` is **constant regardless of T**. Whether it processes 10 tokens or 10,000 tokens, the byte count of the state tensor is the same. The reason the recurrent state in the table above is always 2,048 B and the RWKV state is constant at 10,240 B is exactly this.

> **In plain terms (a cooking analogy):** Transformer is the "keep all the ingredients you've ever used in the fridge" approach; recurrent is the "keep dropping ingredients into one pot and simmering, passing on only the pot's contents (= the state)" approach. The pot's size is fixed, so no matter how long you simmer, the storage space does not grow. **But the longer you simmer, the more the flavor of the ingredients you added first fades** (= loss of old information). This "fading" connects to where things break in the latter half.

## Here We Cross the Bridge: State-Space Models (SSM) and Control Theory

This is the core of Part 1. Recurrent's "fold the past into a constant state" can be rewritten in the **vocabulary of the control-engineering state-space model (SSM)**. This is a legitimate connection in the literature. "Structured State Spaces" is the title of the **S4 paper** (Gu et al., 2022), the predecessor of SSM-family LLMs, and **Mamba** (Gu & Dao, 2023) is its direct descendant. In all of these, **the state-space model (SSM) is a standard concept of signal processing / control**.

Note that the "can be rewritten" of this section is a **structural analogy**, not a claim of mathematical identity. The bridge holds at the level of the skeleton — "fold the past into a finite representation via sequential state updates" — and does not mean RWKV/Mamba literally are linear time-invariant state-space models or Kalman filters.

### What is a State-Space Model (the 60-year standard form of control engineering)

A linear time-invariant state-space model is written, textbook-style, like this.

```
x_{t+1} = A x_t + B u_t      # state equation: next state = A×current state + B×input
y_t     = C x_t + D u_t      # observation equation: output = C×state (+ D×input)
```

- `x` = the **state**. The system's "finite-dimensional vector summarizing all history so far."
- `A` = the state-transition matrix. It decides "how much of the past state carries over into the next."

The decisive thing is that **`x` is finite-dimensional and fixed-size**. Control engineering is built from the outset on the premise of "compressing an infinitely continuing input history into a finite state and carrying it." RWKV/Mamba's `s_t = f(s_{t-1}, x_t)` is precisely a (nonlinearly gated) relative of this state equation.

> **In plain terms (why control engineering takes this form):** In a factory plant or rocket attitude control, there is no room to store every past sensor value and consult them all each time. You need a form where "as long as you know the current state, you can decide how to move next." So control engineering started from "**fold the entire past into a finite state**" from the very beginning. What recurrent rediscovered in the LLM long-context problem was this 60-year-old starting point.

### Observability — "Is folding the past into a finite state really enough?"

Control engineering has a concept called **observability**. Roughly, it is the property "Can the internal state `x` be uniquely reconstructed by looking at the history of the output `y`?" This corresponds directly to the worry over recurrent's "is a constant state really okay?" **If the state dimension is insufficient for the distinctions among pasts you want to tell apart**, two pasts that should be distinguished get crushed into the same state (it becomes unobservable). Conversely, if the state dimension is sufficient for the needed distinctions, the necessary distinctions are preserved even with a constant state. Transformer's "keep the entire past" can also be read as **guaranteeing observability at all times by brute force**.

### The Kalman Filter — the Origin of "Keep Folding Observations into the State Update"

In 1960, Dr. Rudolf E. Kálmán formulated the **Kalman filter** (*A New Approach to Linear Filtering and Prediction Problems*, 1960), an algorithm that does exactly "keep folding successively arriving observations into a fixed-size state estimate (and covariance)." Each time a new observation arrives, it does not recompute the entire past; it merely updates the current state estimate. RWKV/Mamba's "throw away the KV cache and keep updating a fixed state" is, structurally, the same skeleton as the "sequential state update" idea that goes back to Kalman. Transformer's KV cache is the opposite pole — **an uncompressed representation that piles up observations raw without folding them**.

> **★ Here is where "the bridge breaks," part 1 (not literally Kalman):** Declaring RWKV/Mamba to "be Kalman filters" is **wrong**. The Kalman filter is the optimal estimator for a linear-Gaussian system, with an explicit observation model, noise covariance, and a proof of optimality. RWKV/Mamba have **nonlinear gates** (tanh / sigmoid / input-dependent transitions), their matrices are learned from data, and there is no optimality guarantee. What the two share is only the structural skeleton of "**fold the past into a finite representation via sequential state updates**." This article's claim is throughout a **structural analogy**, not mathematical identity.

### So recurrent's "flatness" is not a config coincidence but a structural necessity

Once you get here, the hook's "recurrent peak WS is ×1.00 regardless of T" is seen to be no coincidence of my experimental setup. The state-space model is, **by definition**, fixed at a finite-dimensional state. No matter how much you stretch the context, all it carries is the same-size `x` (state). **The difference between the two is not an implementation tuning margin but the fundamental architectural choice of how to represent the past (fold it / pile it up).**

## ★ Most Important Fix — Do Not "Over-Connect" ρ (Contractivity)

Let me preempt and knock down the confusion this kind of cross-domain article is most prone to commit. Bring in control engineering, and you are tempted to **lump "stability (contractivity)" together with "learning clinging to the stability boundary" and turn it into a tidy story**. This is wrong.

**(A) Contraction mapping ρ<1 = a proof of stability (something to require structurally)**
In another llcore arc (Verified Plasticity), I retrofit a small recurrent adapter `s' = decay⊙s + (1−decay)⊙tanh(Ws + x)` onto the hidden state of a frozen LLM and guarantee its stability with a proof. If the spectral radius `ρ(J)<1` of the Jacobian `J`, the state update becomes a **contraction mapping**, and small perturbations decay over time (the echo-state property). llcore's certifiers (`cert_inf`: ‖J‖_∞<1, `cert_two`: σ_max<1 at all vertices, `cert_sdp`: a common Lyapunov LMI) **pass only those satisfying `ρ(J)<1`, fail-closed**. `empirical_rho` (an independent oracle that corners the eigenvalue from below) checks the soundness of the proof.

> **In plain terms (what is a contraction mapping):** A mapping where, when you shift the input a little, the shift in the output **is always smaller than the original shift**. No matter how many times you repeat it, the shift does not grow, so the system does not run away and settles to a point. `ρ<1` is its mathematical passing line.

**(B) Learning clings to the stability boundary ρ≈1 (a tension with expressiveness, an observed phenomenon)**
In a different experiment (M2), an observation emerged that, under evolutionary search, **the `empirical_rho` of the best-fitness individual clings to 1.000**. This is a manifestation of the **expressiveness vs. stability trade-off** known in reservoir computing and recurrent learning: "as you try to maximize expressiveness, the state update edges toward the **cliff edge of stability (ρ→1)**."

Here it is tempting to **stitch it all into one beautiful story**: "Control theory says `ρ<1` is the condition for stability, and learning also goes to `ρ≈1`. So control theory even predicted the edge of chaos!" This is over-connection. The two are **opposite in direction**. (A) is "**keep it** at `ρ<1`" (an upper-bound norm we impose). (B) is "`ρ` **drifts** toward 1" (a behavior the optimization exhibits). Because they use the same symbol `ρ`, they appear connected at a glance, but **they are logically independent**.

> **So the correct way to put it in this article:** Control theory had, 60 years ago, solved **(A) "carry the past in a constant state + characterize its stability with `ρ<1`."** Meanwhile, **(B) "expressiveness clings to the stability boundary `ρ≈1`" is a known trade-off — the tension between stability and expressiveness — that control and machine learning each know separately**, and it does not follow automatically from (A). Distinguish the two and then place them side by side — that is the honest connection.

## ★ Another Place Where the Bridge Breaks — Recurrent "may" Be Inferior to Transformer in Capability

So far I have written about recurrent's **structural win on the memory axis**. But closing with "therefore recurrent beats Transformer" would be special pleading and would violate the caveat at the top of the series. Let me write honestly.

In llcore's evolutionary-search arc, when I measured the **capability (smartness — does it surpass the gradient in perplexity / cross-entropy?)** of recurrent-family adapters, the landscape was **NULL_TIE / NEGATIVE** (evolution shows no significant difference against baseline — if anything, slightly harmful). That is, llcore itself acknowledges that "**in this particular experimental setup, no winning path in capability could be confirmed for recurrent evolution**" (`project_llcore_memory_efficiency_pivot`, the direct reason for shifting the North Star from capability to memory).

> **Honest disclosure / source boundaries:** The general literature trend I am about to describe and the llcore null result I just described have **different origins**. The former is **external knowledge generally pointed out in the SSM literature** = not directly verified by llcore measurement in this article. The latter (NULL_TIE / NEGATIVE) is **llcore's own null result in a particular experimental setup** = the source is llcore measurement.

That general literature trend is this — the KV cache's property of "exactly retaining all observations" tends to beat recurrent's lossy compressed state at **position-agnostic exact long-range recall**, as is generally pointed out in the SSM literature. On recall-heavy tasks, pure recurrent is said to often fall short of Transformer, and that is exactly why many practical SSMs head toward **hybrids** that interleave attention layers — a framing commonly seen in the literature (e.g. Jamba, a hybrid stacking Mamba and Transformer alternately).

The "the more you simmer, the more the flavor of the first ingredients fades" I touched on in the cooking analogy — that is the symptom that bites on the capability side. **When the state dimension is insufficient for the needed distinctions**, somewhere in the process of folding the past into a fixed-size state, a past that should be distinguished gets crushed. The mechanism that wins on memory can, when the state dimension is insufficient, also become a source of weakness in capability.

**So this article's claim is "limited to the memory axis."** "Recurrent runs in constant memory regardless of context length" is structurally correct (as my measured ×1.00 shows). But "recurrent is comprehensively superior to Transformer" is something I do **not** say. Rather, llcore acknowledged that it could not confirm a winning path in capability in this setup, and then moved its North Star to an axis where it can win (memory efficiency) — this article stands on that honest defeat.

## Part 1 Analogy — "The Person Who Archives the Diary Verbatim" and "The Person Who Only Updates a Summary"

Two people keep a diary every day.

- **Person A (Transformer)** writes each day's events into the notebook **verbatim, exactly as they were**. No matter how many years ago, flipping the pages recalls it word for word. In exchange, the notebook grows thicker year by year, and one day it finally no longer fits on the bookshelf (the linear blowup of the KV cache). Furthermore, since A has the habit of "cross-checking all past pages against one another," the effort grows with the square of the number of pages (attention's O(T²)).
- **Person B (recurrent / SSM)** merely, each night, **folds the day's events into a one-sentence summary and updates a single "current summary card."** The card stays one card no matter how many years pass. The bookshelf never grows at all (constant state ×1.00). It is the way control engineers and Dr. Kálmán organized 60 years ago as "this is often enough."

**Where this analogy breaks:** (1) B's card is a **summary**, so when asked, word for word, "on what month and day three years ago did who say what," B loses to A (lossy compression = capability weakness). (2) If the summary field (= the state dimension) is too narrow for the needed distinctions, two days that should be distinguished get crushed into the same summary (loss of observability). Cramming in too much information brings the card update closer to the cliff edge of stability (ρ≈1) — but this is a separate matter from the "keep the card to one sheet" design and must not be mixed in. (3) In the first place, B is not "doing the same thing as Dr. Kálmán"; the skeleton merely resembles it, and it is nonlinear with no optimality guarantee.

An analogy is a scaffold for understanding, not a proof. Use it after showing where the scaffold collapses — that is this series' practice.

---

## Part 2 (a8) — Quadruple the Context and Transformer's Memory Swelled 5×, While Recurrent Didn't Budge a Millimeter

In Part 1, I built the bridge "control theory had long since solved 'carry the past in a constant state.'" Part 2 is its **consequence in real-hardware memory**. The **dynamic dominant term** — the term that grows with context length — shown as a curve.

In another article, I wrote about how, when I tried to measure long-context retrieval performance, **my home machine's memory (3.6GB RAM) cried uncle first**. A 2048-token full-attention forward swelled the working set to 3.9GB and swapped. That looked like an accident, but it was in fact **a necessity the structure had foretold**.

> **Conclusion first:** When the context length was taken from 1024 → 4096 tokens (4×), Transformer's (GPT's) peak memory swelled from 331.8MB → 1673.0MB, about **5.04×**. Under the same conditions, recurrent was 205.4MB → 205.6MB — that is, **it didn't budge a millimeter**.

## All the Numbers, Fully Disclosed — Read the Curve, Not a Point

I prepared lightweight models in 3 architectures (GPT / recurrent / RWKV), and while varying the context length **measured the peak working set (WinAPI's `PeakWorkingSetSize`) in an isolated separate process**. `scripts/recurrent_runtime_rss.py --lengths 128,256,512,1024,2048,4096` (real hardware, 2026-06, tiny model n_embd 256 / 4 layers / 8 heads). Look not at a single point but at a **32×-range curve**.

| Context length L | GPT peak WS | recurrent peak WS | RWKV peak WS |
|---:|---:|---:|---:|
| 128 | 222.3 MB | 205.5 MB | 215.4 MB |
| 256 | 230.7 MB | 205.2 MB | 215.7 MB |
| 512 | 247.7 MB | 205.3 MB | 216.1 MB |
| 1024 | 331.8 MB | 205.4 MB | 216.0 MB |
| 2048 | 607.9 MB | 205.6 MB | 216.0 MB |
| 4096 | **1673.0 MB** | **205.6 MB** | 215.9 MB |
| **Ratio for 128→4096 (×32)** | **×7.53** | **×1.00** | **×1.00** |

**The shape of the curve is the strongest evidence.** GPT's swell rate depends on the measurement range: at 128→512 (×4) it is **×1.11**, nearly flat — in this region the fixed weights (the model body) dominate the peak, and attention's O(L²) term is still buried in the noise. But at 512→4096 (×8) it is **×6.75** — here the quadratic term overtakes the fixed floor and the curve shoots up. A single headline of "context ×N, memory ×M" varies wildly in M **depending on where you started measuring** (on the same setup it becomes both ×1.11 and ×6.75). So I present a curve, not a single point. Recurrent / RWKV stick to 205 / 216 MB across the whole range (×1.00), backing up over a 32× range that the constant state is independent of context length.

> Supplement: The 1024 and 2048 points match an independent run on a different day (`out/recurrent_runtime_rss_long.json`) to within 0.1 MB, confirming cross-run reproducibility as well.

![A line graph showing that over context length 128→4096 (log axis), GPT's peak WS is nearly flat at short context (128→512), 222→248MB (the fixed-weight floor dominates, ×1.11), but surges at long context, 608→1673MB (the quadratic term becomes manifest, ×6.75), while recurrent/RWKV stay flat across the whole range at 205MB (×1.00). Takeaway: the swelling is not constant — the ratio varies wildly depending on where you start measuring = read a curve, not a point.](../../../assets/articles/llcore_context_memory.svg)

*Figure: the horizontal axis is logarithmic. GPT (red) is nearly flat at short context, hidden under the fixed-weight floor, and surges at long context as the quadratic term shows its face (×7.53 across the full 128→4096 range). Recurrent/RWKV (green) are flat over the whole range with a constant state (×1.00). At T=4096, GPT is 8.1× recurrent.*

There are three things to read.

1. **GPT swells super-linearly.** At context 4× (1024→4096), **memory 5.04×**; over the full 32× range, **7.53×**. Because attention's score matrix grows as L×L, i.e. **O(L²)**. The growth is not constant — at short context it is flat, hidden by the fixed weights; the longer it gets, the more the quadratic term shows its face and sharpens.
2. **Recurrent and RWKV do not move.** Even at context 4×, **×1.00**. Because they fold the past into a fixed-size state, there is in principle no term that depends on context length.
3. **At T=4096, GPT's 1673MB is 8.1× recurrent's 205MB.** Extrapolating, **at T=8192 GPT is about 6.5GB** — a **wall that physically does not fit** on my home machine (3.6GB RAM). Recurrent, meanwhile, stays at 205MB and does not flinch.

The 3.9GB my machine swapped to was exactly a point on this curve. Not an accident, but **exactly as the O(L²) foretold**.

> **In plain terms (a meeting analogy):** In a 10-person meeting, if everyone speaks while checking "everyone else's remarks" one by one, the combinations of checks grow with the **square** of the number of people (5 people = 25 ways, 10 people = 100 ways). This is Transformer's O(L²). Recurrent, by contrast, is the "pass around a single notebook, each person writing only the key points and handing it on" approach — the notebook's thickness does not change as the number of people grows. But this "summary notebook" is lossy and cannot reconstruct every word (the price of flat memory).

## Why It Comes Out This Way (the Structural Story)

GPT's attention is an all-pairs "every token looks at every token." With L tokens, the combinations to look at are L×L, so memory and compute are both **O(L²)**. On top of that, the KV cache keeps eating memory at **O(L)**. Recurrent / RWKV have the opposite idea — read the past one at a time while **folding it into a fixed-size state vector**, and carry that to the next. The state's size does not depend on context length (its contents are a constant of a few KB). This is the "control-theory state-space model" of Part 1 itself. The story is that what the theory said — "you can carry the past in a constant state" — appeared as a **flat straight line** in the real-hardware peak WS.

## The Caveats I Don't Hide (the Limits of These Numbers)

- **These are tiny, untrained models.** Randomly initialized models with n_embd 256 / 4 layers. Memory behavior is determined by the architecture and does not depend on the training state (so it is valid for this measurement), but do not apply the **absolute values** as-is to the large vendors' LLMs. What you should read is the **trend of the growth**.
- **The peak WS is the sum of "the bare torch runtime + the fixed weights + the context-dependent buffers."** Most of recurrent's "flat 205MB" is the torch baseline (the dominant term measured in Part 3); the constant state itself is on the order of KB. That is, the 205MB shows not "the constant state is heavy" but "**there is no growing term**."
- **GPT, in real operation, crops the context with block_size**, so it does not swell this naively. This measurement looks at the necessary amount — "the minimum required if you retain exact long context."

## Part 2 Lesson — "Predictability" over "Speed"

What matters in the age of long context is, more than peak performance, **"being able to predict how memory behaves with respect to context length."** GPT is fast and strong. But stretch the context and the memory heads toward an **unpredictable cliff at O(L²)**. Recurrent, even if slow, **stays flat at 205MB however many tokens come** — predictable.

> Quantization "thins down what already exists."
> Architecture choice "doesn't have the growing term to begin with."
> The longer the context, the more the latter pulls ahead of the former by structure.

This conclusion — "quantization (shaving bits) has a ceiling; the real winner is choosing an architecture that does not have the dynamic dominant term" — connects directly to the next, Part 3 — the **static dominant term**.

---

## Part 3 (a9) — I Was Optimizing the Wrong Term: Right Next to int8 Shaving 4MB, the Language Runtime Was Eating 184MB

Part 2 was the story of the **dynamic dominant term** — "make the context longer and Transformer's memory swells with the square." Part 3 is the flip side, the story of the **static dominant term**.

The trigger was a common offhand remark.

> "If you're going to be thorough about memory efficiency, why not just rewrite it in Rust? That'd be more efficient, right?"

Plausible. But I make it a rule **not to automatically believe** this kind of "switch to ○○ and efficiency goes up." So I measured. The result overturned the premise of my optimization.

> **Right next to where I was shaving the model's weights to about a quarter (a few MB) with int8 quantization, the Python+torch language runtime was eating 184MB while doing nothing.** The term I was shaving was not the dominant term in the first place.

## All the Numbers, Fully Disclosed, and the Shocking Ratio

I called Windows's `GetProcessMemoryInfo` (WorkingSetSize) directly from `ctypes` and measured the process's RSS in stages.

| Stage | Process RSS |
|---|---:|
| Python interpreter baseline | 13.4 MB |
| After `import torch` | 197.3 MB (**+183.9 MB of "torch tax"**) |
| After loading the model | 213.6 MB |
| **The actual int8 model weights** | **1.51 MB** |

The ratio that comes out of this bites.

> **The process RSS of 213.6MB is about 142× the model body of 1.51MB.**

I was focused on shaving "the weights to a quarter!" — a few MB — with int8 quantization. But the whole picture was a state in which **212MB of "scaffolding" was erected around a 1.51MB body**. A few-MB saving is close to noise in front of 142× the scaffolding.

> **In plain terms (a luggage-and-cart analogy):** For a move, you worked hard reviewing the packing to make the luggage 4kg lighter. Admirable. But what if **the cart carrying that luggage weighed 184kg**? A 4kg saving is almost meaningless in front of a 184kg cart. **What should have been lightened was not the luggage but the cart.** That is exactly what was happening with my AI.

![A stacked bar chart with the measured breakdown of the 213.6MB process RSS. The torch tax of 183.9MB (the language-runtime scaffolding) accounts for the bulk, and the int8 weight body of 1.51MB is nearly a line at the same horizontal scale. Reproduced in another run too at 197.8MB after torch / 179.7MB tax, a ~2% difference. Takeaway: the dominant term was not the body I was shaving but the language-runtime scaffolding (about 142× the body).](../../../assets/articles/llcore_a9_static_baseline.svg)

*The term I was shaving (the int8 body, 1.51MB) becomes only a thin line at the same horizontal scale. The dominant term was the 183.9MB torch-tax scaffolding.*

> **Reproducibility (backed up with a committed harness):** I later re-measured this one-off measurement by hardening it into a **re-runnable harness** (`scripts/runtime_floor_rss.py` / `out/runtime_floor_rss.json`, median of 3 measurements per stage in isolated separate processes). The result was **197.8 MB after `import torch` (nearly an exact match to the initial 197.3) / a torch tax of 179.7 MB (a ~2% difference from the initial +183.9)** — that the lead actor, **the torch-runtime tax, is ~180MB, holds firm in another run too**. The baseline, on the other hand, rose slightly from 13.4→18.1 MB, and the scaffolding ratio **changes with the model size**: 142× for the 1.51MB model in the main text, 73× for the harness-default 2.8MB model (n_embd=176). The absolute value of the ratio is config-dependent, but **the structure that "the scaffolding is an order of magnitude larger than the body" is invariant** — that is what is load-bearing.

## "Go After the Dominant Term" — Which Term You Shave Is Everything

Optimization has an iron rule. **Go after the biggest term (the dominant term).** No matter how much you polish a small term, the whole barely moves.

- The **dynamic dominant term** is the attention/KV that grows with context length (the O(L²) measured in Part 2).
- The **static dominant term** is this Part 3's **language-runtime baseline (+184MB for torch alone)**.

At bean-model scale, the latter (the 184MB torch tax) was overwhelmingly dominant. If so, **what actually matters is not the few MB of int8 but how to reduce this 184MB of scaffolding**.

And **this is the true sweet spot of "why not Rust?"** A lean native binary (C++'s llama.cpp, Rust's candle / mistral.rs, etc.) has an order-of-magnitude smaller runtime baseline (typically a few MB to a dozen-odd MB). A **reduction on the order of 184MB** is incomparably larger than the few MB of int8, and connects directly to the North Star of "keep the working set small and predictable." Furthermore, with Rust you can hold the allocator yourself, so the other perennial problem — "torch's caching allocator does not return freed fp32 to the OS, so the normal-time peak does not come down" — can also be knocked down head-on with `madvise(MADV_DONTNEED)` or explicit freeing.

## The Other Side I Don't Hide ("Rust = efficiency↑" Is Not Automatic Either)

Stop here and it becomes "Rust-is-a-panacea-ism." As honest disclosure, let me make the other side explicit.

- **Rust does not make int8 smaller, nor mmap better, nor a constant state constant.** These wins derive from representation, OS, and algorithm and are **language-independent**. Moving to Rust does not improve them by a single byte. Where Rust matters is only the **baseline (scaffolding)** term.
- **Bare Rust's matrix multiply is slower than torch's BLAS (MKL/OpenBLAS).** "Rust = fast" is not automatic either. To get speed you need candle or BLAS bindings.
- **Scale dependence is the crux.** The 184MB tax is dominant **because it is bean-model scale**. With GB-class weights, the baseline becomes relatively negligible. That is, **Rust's baseline advantage is greatest in the region llcore actually lives in (small models, small RAM) and smallest at the large scale where weights dominate**.
- **This is a move llama.cpp has already demonstrated.** As a position, it is a **re-derivation**, and the honest originality is not there (the language) but in the operation of the cap-gate (capability gate) I wrote about in another article.
- **Unconfirmed (honest gap).** The baseline RSS of a Rust/candle version is **not yet measured**. "A native version has an order-of-magnitude smaller baseline" is a reasonable estimate from the literature and existing implementations; final confirmation requires a candle measurement.

## Part 3 Lesson — Measure the Baseline Before Rewriting in Rust

> When you hear "switch to ○○ and efficiency goes up," **measure the current dominant term** before rewriting.
> My dominant term was not the weights int8 can shave, but the 184MB the moment I imported torch.

Had I run off to "fully port to Rust for memory efficiency" without measuring, most of the effort would have evaporated into "just **rewriting** the language-independent wins (int8/mmap/constant state) in Rust," and the true main objective — the 184MB baseline reduction that actually matters — would have been buried. **Narrow the aim to 'make the inference path native and erase the interpreter tax'** — that was the correct porting policy, visible only after measuring (aimed at the scaffolding, not at speed).

Optimization measures "which term is large" before polishing. It sounds obvious, yet I myself, shaving 4MB while overlooking 184MB, was the one who most forgot it. This discipline of "**measure, then move your hands**" connects to the universal lesson of the final Part 4.

---

## Part 4 (a11) — A Flaky Test Was Not the Test's Fault but a Sign of the Design

So far it has been about the dominant term in AI architecture. Lastly, changing the tone a bit, this is a scene of universal engineering: "**the difficulty of writing a test taught me a design smell**." Not AI-specific, a small piece that works across making things in general — but, just like "measure the dominant term," it is the same discipline of **don't mistake the source of the symptom**.

## What I Wanted to Do, and the Test That Wavered

I tried to write, in TDD (test-driven development), a promotion gate that "**saves the quantized model only when it PASSes capability acceptance (cap-gate)**." The spec is simple: gate PASS → write, FAIL → don't write.

But the test became **flaky (the result wavers each run)**. Sometimes pass, sometimes fail. With that, I cannot assert "it works properly."

The cause was the **untrained (randomly initialized) model** I used as the test target. A random model's "next-token prediction" is nearly a uniform distribution. Add the slight noise of int8 quantization on top, and **the top-1 (most likely candidate) swaps across the boundary**. As a result, the retention (capability-retention rate) wobbles above and below the passing line, and when "write on PASS" is tested wholesale via the CLI, PASS/FAIL changes from run to run.

> **In plain terms (what is top-1):** The choice the model puts at rank 1 as the "most likely next candidate." When the candidates are nearly level (uniform distribution), the slightest noise swaps rank 1 and rank 2. This was the source of the wobble in the decision.

Here, "fudging it with a retry in the test" is a bad move. A flaky test is usually **not the test's problem but the design's problem**.

## The TDD Maxim — "Hard to Test = Unclear Design"

In test-driven development, there is a rule of thumb like this.

> **When a test is hard to write, it is not the test that is bad, but the design that is unclear, or the coupling that is too strong.**

Looking at my own code through this lens, the cause came into view. **"The decision logic of whether to promote" and "the input/output (I/O) of writing a file" were mixed into a single function.** Because the decision and the I/O were fused, I could not check the decision alone in isolation, and had no choice but to go through the I/O-laden nondeterministic path every time.

> **In plain terms (a cooking analogy):** Picture a cook who, wanting to taste-test, does "seasoning" and "plating" in the same single motion. You only want to check the taste, but you cannot know the taste each time without going all the way to plating. And because the quality of the plating wobbles day to day, even the taste judgment wobbles. My program was exactly this.

## The Fix — Carve Out the Decision into a Pure Function

So I carved out only the decision into a **pure function** (a function that always returns the same output for the same input and has no side effects).

```
_should_promote(report, force) -> (bool, reason)
```

Pass `report` (the result of capability acceptance) and `force` (the force flag), and it just returns "whether to promote" and "the reason for it." It does not write a file. Then —

- The **decision logic** can be tested **deterministically** by passing hand-crafted acceptance results (gate = PASS / FAIL / not-measured, 3 ways × with/without force).
- The **CLI integration test** avoids nondeterminism and is **limited to the deterministic path only** ("set the retention floor to 0.0 so it always PASSes," "no corpus specified so it always rejects").

The flakiness vanished. Not because I added a retry to the test, but because I **separated the decision and the I/O**.

![A BEFORE/AFTER comparison figure. Left (BEFORE) has "decision" and "file write" fused into one function, and int8 noise makes top-1 swap across the boundary, so a wholesale CLI test wobbles PASS/FAIL. Right (AFTER) carves out only the decision into a pure function _should_promote → (bool, reason), separating it into a unit test of the decision (gate=PASS/FAIL/not-measured × force) and the CLI's deterministic path (--min-retention 0.0 → PASS / no corpus → reject), and the flakiness vanishes. Takeaway: the test's wobble is a free reviewer saying "separate the design."](../../../assets/articles/llcore_a11_test_smell.svg)

_A before/after comparison: the fusion of decision and I/O (left) separated by making the decision a pure function (right)._

## Part 4 Lesson — Don't Dismiss Flakiness as "the Test's Bad Luck"

When you see a flaky test, you are tempted to dismiss it as "the environment's fault" or "just a fluke." But in many cases it is **a sign that the design is mixing something**.

> A test that is hard to write or wavers —
> that is not "fix the test" but a signal saying "**separate the design**."

This time "decision" and "input/output" were fused. Once separated, the test became deterministic and the code became easier to read. **The difficulty of a test is a free reviewer pointing out where the design can be improved.** Even in making a small AI tool, this classic principle worked as-is.

---

## Integrated Self-Audit / Honest Disclosure

As career-grade discipline, I turn the blade on the claims of this whole arc myself.

- **Scale:** All measured values (×1.00 / ×2.65 / ×7.53 / 142× / state 2,048–10,240 B / torch tax 184MB) were measured **on a CPU with tiny char-LMs of 0.81M–130M**. There is no guarantee that the same flatness or the same swelling comes out **at the same ratio** on real LLMs of 12B–64B. The **direction** of the slope is consistent with the literature, but **extrapolation of the absolute values is unguaranteed**.
- **Cleanliness of measurement (no term decomposition):** The peak WS is the **sum** of the torch runtime + fixed weights + context-dependent buffers, and the clean signal is the **increment trend** after subtracting the fixed baseline (~205MB). I did not decompose that increment into "the KV-linear part" and "the attention-quadratic part" and measure them; "super-linear = the quadratic term becoming manifest" is an interpretation from the analytic model.
- **Reservation on the analogy:** The correspondence with SSM / observability / contraction mapping / Kalman filter is a **structural analogy**. I do not claim that RWKV/Mamba literally are state-space models or Kalman filters (nonlinear gates, learned matrices, no optimality).
- **Non-confusion of ρ:** I distinguished "`ρ<1` = a proof of stability (a constraint I impose)" from "clinging to `ρ≈1` = the expressiveness vs. stability trade-off (an observed phenomenon)." The latter does not follow automatically from the former.
- **Source boundaries:** The capability conclusion's **NULL_TIE / NEGATIVE is llcore's own null measurement**, while **"pure recurrent tends to be inferior at recall / practical SSMs head toward hybrids" is a general trend of the SSM literature (external knowledge unverified in this article)**.
- **The honest defeat in capability:** Recurrent's capability landscape is already certified by llcore itself as NULL_TIE / NEGATIVE. This arc's superiority claim is **limited to the memory axis** and is not a comprehensive capability superiority.
- **Rust unconfirmed:** The baseline RSS of Rust/candle is unmeasured. "A native version has an order-of-magnitude smaller baseline" is an estimate from the literature and existing implementations; final confirmation requires a candle measurement.
- **The analogy is a single pillar:** I narrowed it to the single pillar of SSM / control theory and did not add other analogies like edge of chaos or reservoir computing (avoiding over-connection).

---

## Closing — Don't Be Ashamed of "Re-performance," Don't Exaggerate "Identity," and Measure the Dominant Term

What I measured on my home CPU in this arc — "context length ×8 gives Transformer ×2.65 / recurrent ×1.00," "×32 gives ×7.53 vs ×1.00," "next to the 1.51MB int8 body, a 184MB torch tax" — none of these are new discoveries. They are a **re-performance, at char-LM scale, of the problem 60-year-old control engineering had thoroughly organized — "carry the past in a constant state + characterize its stability with `ρ<1`"** — and a straightforward application of the iron rule of optimization, "go after the dominant term."

But a re-performance has value.

1. **The structure falls into place.** Recurrent's flatness is seen to be not a config coincidence but a structural necessity coming from the definition of the state-space model. So one can say "**this family wins structurally on the memory axis**."
2. **It guards against overclaiming.** Borrowing the vocabulary of control theory, one also sees **where that vocabulary breaks** (not literally Kalman / it can be inferior in capability / `ρ<1` and `ρ≈1` are different / (A) is solved but (B) was not predicted). One avoids puffing up the argument with borrowed authority.

And the theme running through all four is consistent — "**not 'lower the bits' but 'go after the architecture and the dominant term.'**"

- Rather than shaving the few MB of int8 with quantization, **choosing an architecture that does not have the dynamic dominant term (O(L²)) that grows with context** is an order of magnitude more effective in long context (Part 2).
- Before rewriting in Rust, **measuring the static dominant term (the 184MB language-runtime scaffolding)** keeps you from mistaking the term to shave (Part 3).
- And when a test wavers, **measure the source of the symptom (the fusion in the design) and separate it** — this too is the same discipline of "measure where it matters most before moving your hands" (Part 4).

The competition for long context and low memory in LLMs, pushed to its limit, converges on the very questions that the classics of control engineering and optimization faced half a century ago — "where and how to hold the past" and "which term is dominant." Transformer buys positional exactness with "keep everything," and pays in memory. Recurrent/SSM saves memory with "fold into a finite state," and pays in the sharpness of distinction. **It is not which is right but a choice of which axis to pay on**, and the (A) part of that map was something control engineering had drawn long ago — that is the end of the crossing of this arc, "A Win by Structure."

In the next arc, I will extend this "win by structure" thinking into measurement discipline itself (the honest disclosure of doubting your own win) and into FullSense's contextual connection of local / on-prem.

---

### References (replace with commit / file references when posting to Qiita)

- llcore measurement master: `docs/MEMORY_EFFICIENCY_FINDINGS.md`
- harnesses: `scripts/memory_footprint_harness.py` (`out/mem_footprint.json`), `scripts/recurrent_runtime_rss.py` (`out/recurrent_runtime_rss.json` / 32× curve `out/recurrent_runtime_rss_curve32.json`), `scripts/recurrent_latency_sweep.py` (`out/recurrent_latency_sweep.json`), `scripts/decode_latency_sweep.py` (`out/decode_latency_sweep.json`), `scripts/runtime_floor_rss.py` (`out/runtime_floor_rss.json`)
- `ρ<1` certifiers and `empirical_rho`: the Verified Plasticity arc (`cert_inf` / `cert_two` / `cert_sdp`, `empirical_rho` independent oracle)
- a11 implementation: `_should_promote` (pure function) in `src/llcore/memory.py`, `tests/unit/test_memory_facade.py`
- Control-theory side (primary concepts): state-space model (SSM) / observability / contraction mapping `ρ<1` / Kalman filter (R. E. Kálmán, 1960, *A New Approach to Linear Filtering and Prediction Problems*)
- SSM-family LLMs: S4 (Gu et al., 2022, *Efficiently Modeling Long Sequences with Structured State Spaces*) / Mamba (Gu & Dao, 2023, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*) / RWKV / hybrid example Jamba (Mamba × Transformer alternating)
- capability reservation: `project_llcore_memory_efficiency_pivot` (shifted the North Star from capability → memory; recurrent capability = NULL_TIE / NEGATIVE = llcore's own null measurement)
