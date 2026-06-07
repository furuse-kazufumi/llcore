# Biological Grounding for the Third-Axis (Selection / Diversity-Maintenance) Finding

*Companion artifact to `third_axis_selection_arc_2026-06-02.md`. Verified 2026-06-02.*

> **Scope and honesty contract.** This document supplies *biological grounding* — an illustrative
> precedent from evolutionary biology — for our computational finding that diversity-maintaining
> selection (operationalized as MAP-Elites behavioral niching, our factor ③) is load-bearing
> **only** in high-dimensional, deceptive fitness landscapes, and is unnecessary on the realistic
> low-dimensional / smooth CPU substrates we tested, where a strong random-restart hill-climbing
> baseline solves them directly. **The biology is grounding, not proof.** Sewall Wright's shifting
> balance theory and the Wright–Fisher debate are *structural analogies* that explain *why* ③
> should help when it helps and *when* it should be expected to add nothing. They do not
> demonstrate our computational result, and several junctures of the mapping are loose; every such
> looseness is flagged inline. We cite only references whose existence, attribution, and load-bearing
> content were adversarially verified; any unverified mapping is presented explicitly as an analogy.

---

## Part 1 — Biological Grounding (paper-grade, English)

### 1.1 Why reach for biology at all

Our central claim is a *boundary condition*: diversity-maintaining selection earns its keep in a
narrow regime (high-dimensional deception) and is redundant outside it. Evolutionary biology
contains an unusually clean precedent for exactly this boundary condition — the debate, running for
most of the 20th century, over whether ordinary mass selection in a single large population suffices
to produce adaptation, or whether a *population subdivided into many partially isolated demes* (a
diversity-maintaining structure) is needed to reach adaptive peaks that mass selection cannot. That
debate is the Wright–Fisher controversy, and Wright's positive proposal is the **shifting balance
theory**. The structural correspondence to MAP-Elites / Quality-Diversity is strong enough to be
worth stating carefully, and — importantly — the biologists' *empirical verdict* on shifting balance
echoes our own negative result.

### 1.2 Wright's shifting balance theory as the precedent for ③

Sewall Wright's shifting balance theory rests on two foundational papers: Wright (1931),
*Evolution in Mendelian Populations* (*Genetics* 16:97–159), which introduced random genetic drift
and effective population size; and Wright (1932), *The roles of mutation, inbreeding, crossbreeding
and selection in evolution* (*Proc. 6th Int. Congr. Genetics* 1:356–366), which introduced the
adaptive-landscape metaphor and the subdivided-population argument.

> **Dating caveat (verified).** The *name* "shifting balance" and the crisp, numbered three-phase
> articulation were elaborated by Wright across later writings (notably the 1970s–80s), not stated
> as a numbered process in the 1931/32 papers. "1931–1932" is correct for the *foundations*; the
> three-phase framing is a later codification.

The canonical three phases are:

- **Phase I — drift in small, partially isolated demes.** Random genetic drift moves a deme off its
  local fitness peak, down into and across a fitness valley.
- **Phase II — mass (individual) selection within a deme.** Once drift has carried a deme into the
  domain of attraction of a new, higher peak, ordinary within-deme selection drives it up that peak.
- **Phase III — interdeme selection.** Demes sitting on higher peaks export disproportionately many
  migrants (differential dispersion/productivity), so superior gene combinations spread and shift the
  whole species to the higher peak.

> **Phase-ordering caveat (verified).** Interdeme selection is **Phase III**, not Phase II — one
> secondary source mislabels it; do not conflate within-deme *mass selection* (Phase II) with the
> metapopulation-level *interdeme selection* of Phase III.

The core logic is precisely the one our paper turns on. A single large *panmictic* population under
deterministic mass selection is trapped on its local peak, because crossing a valley requires
transiently *lowering* mean fitness — which deterministic selection will not do. Subdivision plus
drift lets *some* demes pay that cost stochastically, while the metapopulation as a whole retains the
gains. This is the biological statement of "stepping-stones across a deceptive valley."

**The mapping to ③ / MAP-Elites (an analogy, not an attribution).** Each archive cell / niche is a
quasi-isolated deme; local elitism within a cell is within-deme mass selection (Phase II);
crossover or migration across cells is interdeme spread (Phase III); and the *archive collectively*
crosses low-performance valleys that a single converging panmictic EA cannot. This correspondence is
real and is frequently drawn informally in the QD literature, but two honesty points must be made:

1. **It is a commentators' framing, not Wright's claim nor MAP-Elites' stated lineage.** The
   foundational MAP-Elites paper (Mouret & Clune 2015, *arXiv*:1504.04909) and the core QD /
   illumination literature **do not cite Wright or "shifting balance"**; they frame the benefit as
   *illuminating* a feature space and escaping deception by maintaining diversity. We therefore cite
   Wright (1931/1932) as **inspiration / analogy**, never as the source MAP-Elites builds on.
2. **The mechanism is structural, not identical.** MAP-Elites valley-crossing is mediated by
   *variation operators* producing offspring that land in new cells, **not by genetic drift**. The
   analogy is structural — diversity + parallel local hill-climbing + cross-niche transfer — and not
   mechanistically the same thing as drift. Equally, it is the *archive* (≈ metapopulation) that
   crosses the valley, not any single cell — the same correction Wright's defenders insist on for
   the biology (the metapopulation, not any one deme, reliably finds the higher peak).

**Why this is the right precedent for our finding.** Wright's whole motivation was that a single
large panmictic population under pure mass selection gets *stuck on a local peak* and cannot cross
adaptive valleys — so he introduced subdivided demes (≈ niches/islands) plus drift to maintain and
explore diversity. That is the same regime in which ③ is load-bearing in our experiments: a
high-dimensional, deceptive landscape with valleys between peaks. In a **low-dimensional discrete
choice problem there is no deceptive valley structure to cross**, so a strong random-restart /
multi-start baseline reaches the global optimum directly and the niching machinery is not
load-bearing. Shifting balance therefore clarifies both *why* ③ helps when it helps and *when* it
should be expected to add nothing.

### 1.3 The Wright–Fisher debate and the epistasis / landscape-shape crux

The headline contrast is accurate: R. A. Fisher (*The Genetical Theory of Natural Selection*, 1930)
held that large, effectively panmictic populations under mass/directional selection on additive
genetic variance were the dominant engine of adaptation; Wright emphasized subdivision, drift as a
*creative* (not merely degradative) force, and a multi-peaked adaptive landscape. Two precision
points should be made in a research-grade treatment:

> **Overstatement caveat (verified).** "Fisher thought drift unimportant" is a popular-science
> compression. Fisher *conceded* drift occurred but judged it quantitatively negligible relative to
> selection in realistically large populations; he did not *deny* it. State it as "drift
> quantitatively negligible by Fisher's estimate," not "rejected." Symmetrically, Fisher's charge
> that Wright nearly rejected selection is regarded by Wright's defenders as a misreading — Wright
> always retained mass selection as Phase II.

> **The deepest axis (verified, and often omitted).** The core of the disagreement was **epistasis /
> gene interaction and the *shape* of the fitness landscape**, not merely population size and drift.
> Wright assumed a *rugged, multi-peaked* landscape arising from non-additive interactions among
> loci (hence the need for drift to cross valleys); Fisher granted that non-additive interactions
> exist but argued they were unimportant to the evolutionary process, implicitly favoring a
> single-peak / smoothly-ascending landscape where mass selection suffices.

This epistasis/ruggedness axis is exactly the dimension our finding lives on. **Landscape topology
is the whole question.** When the landscape is genuinely rugged and high-dimensional (Wright's
regime), diversity-maintenance is what lets search cross valleys; when it is smooth or its hard
coordinate is low-dimensional (Fisher's regime), mass selection — the biological analogue of a
strong random-restart hill-climber — already suffices. Our deterministic, noise-free landscape
measurement found the realistic echo-state-reservoir text-proxy landscapes to be *genuinely smooth*
(single-basin in `dim=3` and effectively smooth in `dim=40`), and the four-kernel-union search space
to have a hard coordinate that is **low-dimensional and discrete** (kernel choice over four
options) — both squarely the Fisher regime, where ③ should not and did not help.

### 1.4 Our negative result as a computational echo of the Coyne–Barton–Turelli critique

The most load-bearing biological correspondence is not to Wright's *proposal* but to the field's
*empirical verdict* on it. Coyne, Barton & Turelli (1997, *Evolution* 51(3):643–671, "Perspective: A
critique of Sewall Wright's shifting balance theory of evolution") evaluated the three-phase theory
on theoretical and empirical grounds and concluded — verified verbatim against the full text — that:

- **Mass selection usually suffices.** There are "few empirical observations explained better by
  Wright's three-phase mechanism than by simple mass selection"; nearly all are "equally well
  explained by Fisherian mass selection," and artificial-selection experiments "fail to show that
  selection in subdivided populations produces greater response than does mass selection in large
  populations."
- **Shifting balance works only under restrictive, rarely-met conditions.** "Theory shows that the
  [shifting balance] can sometimes be an efficient mechanism of adaptation, but only under
  restrictive conditions"; empirical population-structure estimates imply "in most cases, random
  drift would be strong enough to cause shifts only between peaks separated by *shallow* valleys"
  (deep valleys are rarely crossed by drift), and most adaptations need no valley-crossing at all.

This is a remarkably exact biological mirror of our finding. Translating their verdict into our
terms: **when the landscape is not genuinely deceptive/high-dimensional, plain mass selection (≈ a
strong random-restart / hill-climbing baseline) already solves the problem, and the
diversity-maintaining apparatus buys little.** Their quantitative point — drift crosses only shallow
valleys, most adaptations require no valley-crossing — is the biological statement of our claim that
*real landscapes are usually simple*, so niching is redundant. The shifting-balance debate thus
supplies both the *theoretical rationale* for ③ (Wright) and a *documented cautionary case* that its
empirical necessity must be demonstrated, not assumed (Coyne–Barton–Turelli) — precisely the
discipline our paper adopts.

> **Three honesty caveats on the Coyne–Barton–Turelli mapping (verified).**
> 1. **They did not "disprove" shifting balance.** They explicitly grant that Phases I/II can occur
>    and document six empirical cases showing at least one shifting-balance component (*Wolbachia* in
>    *D. simulans* the closest analog). Their claim is the *narrower probabilistic* one: shifting
>    balance is unlikely to be a *general / important* cause of adaptation versus mass selection, and
>    is essentially untestable as a *common* mechanism. Saying they "refuted" it overstates them.
> 2. **The controversy was live, not settled.** Wade & Goodnight (1998) and Peck, Ellner & Gould
>    (1998, whose title literally argues the process is "feasible") pushed back; Coyne, Barton &
>    Turelli (2000, *Evolution* 54(1):306–317) is a *defense* of the 1997 critique, and Goodnight &
>    Wade's same-issue reply ("The Ongoing Synthesis," 54(1):317–324) kept the debate open. We must
>    not cite the 1997 critique as the closing word.
> 3. **One biological mechanism has no clean computational counterpart — and is a *stronger* claim
>    than ours.** Phase III shows diversity-maintenance can actively *hurt*: the gene-flow barriers
>    that preserve diversity (enabling Phase I) also trap good solutions in marginal demes and block
>    their species-wide spread. So in biology niching is not merely "unnecessary" off the deceptive
>    regime — it can be *counterproductive*. Our stateless discrete-choice setting has no exact
>    analogue of this gene-flow/recombination cost, so this asymmetry should **not** be over-mapped
>    onto our finding; we note it as a place the analogy is loose, and a respect in which biology
>    makes a stronger claim than we do.

### 1.5 The dimensionality axis, grounded in two real-world cases

Our computational claim has two poles — a low-dimensional "control" pole where ③ is unnecessary, and
a high-dimensional "contingent valley" pole where diversity-heavy search matters. Evolutionary
biology supplies a clean real-world exemplar for each.

#### 1.5.1 Low-dimensional pole — peppered-moth industrial melanism (the BG9-kernel analogue)

The classic textbook case of a **low-dimensional, few-allele, near-discrete trait under strong
directional selection** is industrial melanism in the peppered moth (*Biston betularia*):

- **Single Mendelian locus, few alleles.** The *carbonaria* (black) vs *typica* (pale) polymorphism
  is controlled by a single locus; melanism is caused by a *dominant* allele over recessive *typica*,
  with intermediate *insularia* alleles — a small, discrete allele set, so the trait is genuinely
  low-dimensional (Cook & Saccheri 2013).
- **Known molecular mechanism.** The causal mutation is the insertion of a large transposable element
  into the first intron of the *cortex* gene, raising *cortex* transcript abundance during early
  wing-disc development; dated to ~1819, matching the historical record (van't Hof et al. 2016,
  *Nature*; singular recent origin shown in van't Hof et al. 2011, *Science*).
- **Strong directional selection.** Selection coefficients against the disfavored morph are strong
  and broadly consistent across independent estimates — roughly *s* ≈ 0.1–0.2 (historical surveys
  8–35 %); the cline/dispersal model of Saccheri et al. (2008, *PNAS*) gives *s* ≈ 0.2 against
  *carbonaria*; Majerus's predation experiment, published in Cook, Grant, Saccheri & Mallet (2012,
  *Biology Letters*; 4864 moths), reaffirmed selective bird predation by camouflage.

Here simple directional selection — the biological analogue of greedy hill-climbing / strong
random-restart — directly fixes the fitter morph as the environment (soot, lichen) shifts the
optimum back and forth. The optimum is unimodal at any given time; there is no rugged, deceptive
landscape; **no diversity-maintaining / niching mechanism is needed, and none is invoked in the
mainstream account.** This is exactly our BG9-kernel "kernel-union" result: kernel choice is a
single low-dimensional discrete coordinate (four options), so a random-restart hill-climber samples
all kernels directly and ③ is structurally unable to separate from it. The peppered moth is the
biological face of *the BG9 kernel case*.

> **Honesty caveats on the peppered-moth mapping (verified).**
> - The polymorphism *is* transiently maintained during cline/transition periods — but by **spatial
>   environmental heterogeneity + gene flow (migration–selection balance)**, *not* by an intrinsic
>   diversity-preservation mechanism — so it does not contradict the finding.
> - There is a genuine, peer-reviewed historical debate over whether the ~1970 North Wales cline
>   required a **non-visual heterozygote advantage** (a balancing/diversity component) in addition to
>   predation. Later modelling (Saccheri et al. 2008; Cook & Saccheri 2013) showed higher dispersal
>   plus modest rural selection can "obviate the need" for the non-visual component, **but it remains
>   contested, not closed.** A flat claim that *no* non-directional mechanism is involved would
>   overstate current certainty; the safe claim is that the *dominant and sufficient* explanation is
>   directional predation-driven selection.
> - *cortex* is a cell-cycle regulator (cyclin/fizzy-family), so calling it a "melanization gene" is
>   loose — it is the *switch* locus, acting via developmental timing. (Does not affect the
>   low-dimensionality point.)
> - Cite the 2012 Majerus vindication, not just Kettlewell (1955), to avoid the
>   discredited-experiment objection raised against Kettlewell's original mark–recapture work; and
>   cite a *range* for *s*, not a single canonical value.

#### 1.5.2 High-dimensional contingent pole — Lenski's Cit+ innovation (the ③ regime)

The complementary real-world case is the evolution of aerobic citrate utilization (Cit+) in Richard
Lenski's *E. coli* Long-Term Evolution Experiment (LTEE), the textbook example of
**high-dimensional, historically-contingent fitness-landscape evolution**:

- Twelve initially identical populations were founded in 1988 in a glucose-limited medium that also
  contains citrate (which *E. coli* cannot use aerobically). Aerobic Cit+ evolved in **exactly 1 of
  12** populations (Ara-3), first detected at generation **31,500** (Blount, Borland & Lenski 2008,
  *PNAS* 105(23):7899–7906); no Cit+ in the other 11 (220 cultures, >3,500 clones screened).
- **Replay experiments** showed clones from *later* Ara-3 generations re-evolved Cit+ more readily
  than earlier clones, distinguishing **historical contingency / potentiating prior mutations** from
  a simple rare-but-constant-rate mutation.
- The genomic follow-up (Blount et al. 2012, *Nature* 489:513–518) decomposed the innovation into
  **potentiation** (≥2 prior mutations creating a permissive background, at least one arising before
  three clades C1/C2/C3 diverged and coexisted >10,000 generations), **actualization** (a tandem
  duplication of a 2933-bp segment capturing an aerobically expressed promoter to express the
  normally silent anaerobic citrate transporter gene *citT* — amplification-mediated promoter
  capture / exaptation), and **refinement**.

This case genuinely exemplifies contingency, epistasis, and the value of running many independent
lineages over a vast mutational space — a fair real-world analogue for *why* diversity-maintaining,
exploration-heavy search over a high-dimensional rugged landscape can matter: the innovation required
a specific *ordered* accumulation of permissive mutations (an access-by-mutation barrier),
reproducibly potentiated only in the right genetic background. It sits firmly on the high-dimensional
/ contingent side, supporting the *antecedent* of our conditional (the regime in which ③ should
help).

> **Honesty caveats on the Lenski mapping (verified) — this maps to our conditional's antecedent,
> not its headline.**
> - **The LTEE used no diversity-maintaining *algorithm*.** It is natural selection on a single
>   carbon source; the within-population clade coexistence (C1/C2/C3) arose via
>   negative-frequency-dependent *ecological* dynamics, not an engineered niching operator. It is an
>   *existence proof* that contingency + standing diversity enable rare innovations — **not** a
>   controlled demonstration that a niching algorithm beats random restart.
> - **It cannot, by itself, distinguish "diversity-maintaining selection was necessary" from "enough
>   independent replicates + time were necessary."** The 12 parallel populations are themselves a
>   random-restart-like design, and Cit+ still required ~31,500 generations in just one. Use it as
>   illustrative motivation for the high-dimensional regime, **not** as evidence that niching
>   outperforms a strong restart baseline.
> - Popular summaries that say *E. coli* "evolved a new ability to eat citrate from scratch"
>   overstate it: the innovation was *regulatory* (aerobic expression of an existing transporter via
>   promoter capture), an exaptation — not a brand-new gene or biochemical activity. The
>   "valley-crossing / epistasis landscape" framing is interpretive scaffolding layered on the
>   empirical results, not a direct quote from the papers.
> - A later result (Van Hofwegen, Hovde & Maier 2016, *J. Bacteriol.*) showed Cit+ can be selected
>   far faster under *direct* citrate selection, used by some to dispute the "rare/contingent"
>   framing; Lenski's group rebutted that this does not contradict potentiation under LTEE
>   conditions. If we lean on the "extraordinarily rare / long-delayed" narrative, we should
>   acknowledge this contested follow-up rather than presenting rarity as uncontested.

### 1.6 The accepted computational rationale (MAP-Elites / QD), with attribution tightened

For completeness, the computer-science side of the analogy is itself precise. MAP-Elites (Mouret &
Clune 2015, *arXiv*:1504.04909, "Illuminating search spaces by mapping elites") maintains a
structured archive holding the single highest-performing "elite" per cell of a discretized,
user-chosen feature/behavior space; its abstract foregrounds **illumination** and diversity of
high-performers. Novelty search (Lehman & Stanley 2011, "Abandoning Objectives," *Evolutionary
Computation* 19(2):189–223) rewards behavioral novelty while *ignoring* the objective, explicitly to
**circumvent deception**. Both are foundational to Quality-Diversity (term coined by Pugh, Soros &
Stanley 2016).

> **Attribution caveat (verified).** The "stepping-stones across deceptive valleys" rationale
> originates with **novelty search and the broader QD/open-endedness program**, *not* with the
> MAP-Elites 2015 abstract itself, which does not use the words "stepping stones," "deception," or
> "open-endedness." Conflating the two is the common oversimplification; they are complementary
> (novelty = behavior characterization / selection pressure; MAP-Elites = archive structure).
> MAP-Elites keeps *one* elite per cell, so it is not pure novelty search; it couples local quality
> with global behavioral coverage. There is **no theorem** that niching crosses arbitrary deceptive
> valleys — gains are demonstrated on specific benchmarks (maze navigation, biped walking,
> robot-morphology spaces), and open-endedness remains genuinely unsolved.

Crucially, this accepted rationale is *explicitly conditioned on deceptive landscapes* — Lehman &
Stanley motivate novelty search precisely by deception — which corroborates the *scope* of ③ rather
than universalizing it. Nothing in this literature claims niching is necessary for low-dimensional
discrete choices where the fitness gradient is informative; there, a strong random-restart baseline
solves directly. The CS literature and the biology agree on the same boundary condition.

### 1.7 Summary of the grounding

| Pole | Biology | Landscape | ③ load-bearing? | Our substrate |
|---|---|---|---|---|
| Low-dim / smooth | Peppered moth (single locus, *s*≈0.1–0.2, directional) | unimodal, shifting | **No** — mass selection suffices | BG9 kernel-union; ESN/ridge text proxy (deterministic, smooth) |
| High-dim / contingent | Lenski Cit+ (potentiation→actualization→refinement) | rugged, valley-by-mutation | **Yes** (regime where it *can* matter) | synthetic deceptive corridor (behavior = mean of 24-D genotype) |
| Empirical verdict | Coyne–Barton–Turelli: mass selection usually suffices, shifting balance rarely decisive | real landscapes usually simple | mirrors our **negative result** | every realistic CPU substrate tested |

**Bottom line for the paper.** Wright's shifting balance is the right biological precedent for *why*
③ helps when it helps; the Wright–Fisher epistasis/ruggedness axis is the right framing for the
*dimensionality* condition; the peppered moth and Lenski Cit+ are the clean low- and high-dimensional
poles; and Coyne–Barton–Turelli is the biological precedent for our *negative* result — that
diversity-maintenance is rarely decisive because real landscapes are usually simple. **None of this
proves the computational result; it grounds it.** The analogy is structural, not mechanistic, and is
loosest where biology adds a cost (Phase III gene-flow trapping) that our stateless setting lacks.

---

## Part 2 — 平易版（中学生レベルの説明）

> ここでは同じ生物学の話を、専門用語を使わずに説明する。私たちの研究結果（BG9 や「第三軸」の話）と、
> 「わざわざ GPU を借りるべきか？」という問いに、どうつながるのかも書く。**注意：生物学はあくまで「たとえ話」
> であって、私たちのコンピュータ実験の結果を証明するものではない。** たとえ話がぴったり合わない場所は正直に書く。

### 2.1 まず、私たちが見つけたことの一言まとめ

進化のアルゴリズムには「いろんなタイプの答えを同時にキープしておく」しくみ（私たちはこれを **③** と呼ぶ。
MAP-Elites という方法がそれ）がある。私たちが調べたのは「このしくみは本当に役に立つの？」という問い。

答えは **「役に立つ場面はすごく限られている」**。

- **デコボコがいっぱいで、谷を越えないと一番高い山に行けないような、複雑（＝高次元）な地形**のときだけ役立つ。
- ふつうの、なだらかで山が一つだけの地形（＝低次元・単純）では、**もっと単純なやり方（ランダムに何度もやり直す
  山登り＝「ランダムリスタート山登り法」）で十分**。③ のしくみは要らない。

これとそっくりな話が、実は 100 年近く前の生物学の論争にある。それを紹介する。

### 2.2 シフティング・バランス＝「みんなで散らばって谷を渡る」作戦

生物学者のシューアル・ライト（1931・1932 年）は、こう考えた。

ある生き物の集団が大きな「一つの群れ」のままだと、目の前の小さな丘（そこそこ良いけれど一番じゃない）に
登りきってしまうと、そこから動けなくなる。なぜなら、もっと高い山に行くには、いったん「谷」（条件が悪くなる場所）
を下らないといけないのに、ふつうの自然淘汰は「下る」ことを許さないから。

ライトのアイデアは **「群れを小さなグループにバラバラに分ける」** こと。

1. **第1段階**：小さなグループは偶然（ドリフト）でフラフラ動き、たまたま谷を渡って別の山のふもとにたどり着く。
2. **第2段階**：そこからふつうの自然淘汰でその山を登る。
3. **第3段階**：高い山に登れたグループはたくさん子孫を出し、その良い遺伝子が群れ全体に広がる。

これが **シフティング・バランス（移り変わるバランス）** という考え方。
ポイントは「**いろんなグループに散らばっておくと、誰かが谷を渡って高い山にたどり着ける**」。

これは私たちの ③（MAP-Elites）とそっくり。MAP-Elites は「いろんなタイプの答え」を箱に分けてキープしておく。
箱の一つひとつが、ライトの「小さなグループ」にあたる。**ただし、これは「似ている」という *たとえ話* であって、
MAP-Elites を作った人がライトを真似たわけではない**（MAP-Elites の論文はライトを引用していない）。
そこは正直に書いておく。

### 2.3 でも、その作戦は「いつも必要」ではなかった——ライト対フィッシャー、そして批判

ライトと同時代の **フィッシャー**（1930 年）は、逆のことを言った。「大きな群れのままで、ふつうの自然淘汰だけで
十分に進化できる。わざわざバラバラに散らばる必要はない」。

二人の一番深い対立は、実は **「地形がデコボコ（山がたくさん）か、なだらか（山が一つ）か」** だった。
ライトは「デコボコだから谷を渡る作戦がいる」と考え、フィッシャーは「だいたいなだらかだから、ふつうの淘汰で
登っていける」と考えた。

> 細かい注意（正直に）：「フィッシャーは偶然（ドリフト）を無視した」とよく言われるが、正確には「あることは
> 認めたが、大きな群れでは効果が小さいと考えた」。完全に否定したわけではない。

そして後の生物学者 **コイン・バートン・トゥレリ（1997 年）** が、ライトの作戦を本気で検証してこう結論した。

- **ふつうの自然淘汰だけでたいてい説明できる。** ライトの三段階作戦でしか説明できない実例は、ほとんど見つからない。
- **ライトの作戦が効くのは、すごく条件がそろったとき（深い谷があるとき）だけ。** でも現実の谷はたいてい浅くて、
  そもそも谷を渡らなくても進化できることが多い。

これがまさに **私たちの結果とそっくり**。私たちも「地形が本当はなだらかなら、③ のしくみは要らない。
単純なやり方で十分」と分かった。コイン・バートンたちの「現実の地形はたいてい単純だから、散らばる作戦は
めったに決め手にならない」は、私たちの **負の結果（③ は要らなかった）の生物学版**。

> 正直な注意：コイン・バートンたちは「ライトの作戦は絶対あり得ない」と言ったのではない。「一般的・重要な
> 仕組みとは言えない」と言っただけ。しかもこの論争はまだ決着していない（反論もある）。だから「ライトは間違い」
> と書いてはいけない。

### 2.4 たとえ話①：暗くなった蛾（ひくい次元＝ふつうの淘汰で十分）

イギリスの **オオシモフリエダシャク** という蛾の話。工場の煙で木が黒くなった時代、白い蛾は鳥に食べられやすく、
黒い蛾が増えた。空気がきれいになると、また白い蛾が増えた。

この「黒くなる／白くなる」は **たった一つの遺伝子のスイッチ**（黒が優性）で決まる。選べる色は実質 2〜3 種類だけ＝
**とても単純（低次元）**。鳥に食べられにくい色がそのまま生き残るだけ（＝ふつうの強い自然淘汰）。
**散らばる作戦（③）は要らないし、実際だれも使っていない。**

これは私たちの **BG9 の「カーネル選択」の話** とまったく同じ。カーネルは 4 種類から選ぶだけ＝低次元の単純な選択。
だから「ランダムに何度もやり直す山登り法」が 4 種類を全部直接ためせてしまう。③ のしくみは出番がない。
**暗くなった蛾＝BG9 のカーネルの話の生き物版。**

> 正直な注意：色がしばらく混ざっている時期もあるが、それは「場所によって環境が違う＋移動」のせいで、
> ③ のような「多様性を守るしくみ」のおかげではない。たとえ話がちょっとずれる所。

### 2.5 たとえ話②：大腸菌が新しい力を手に入れた話（高い次元＝歴史と多様性が効く）

レンスキーという研究者の **大腸菌の超長期実験**。同じ大腸菌を 12 グループに分けて 1988 年からずっと育てた。
あるとき **12 グループのうち 1 つだけ**が、それまで使えなかった「クエン酸」を酸素のある環境で食べる新しい力を
手に入れた（3 万 1500 世代目）。

大事なのは、それが **「いきなり」ではなく「前もって別の変化が積み重なっていた特定のグループでだけ」起きた**こと。
順番に複数の変化が積み重ならないと、その新しい力にたどり着けなかった。これが **高次元で歴史に依存する地形**
（＝谷を順番に越えていく必要がある複雑な地形）の本物の例。**③ が効きうる側のたとえ話。**

> 正直な注意：これは「③ というアルゴリズムが勝った」証明ではない。ただの自然の実験で、③ のしくみは使っていない。
> しかも 12 グループに分けたこと自体が「ランダムに何度もやり直す」のに似ている。だから「散らばる作戦が
> 一番だった」とまでは言えない。あくまで「複雑な地形では多様性が効きうる」というイメージのたとえ。

### 2.6 BG9 と「GPU を借りるべきか」への結論

ここまでをまとめると：

- **私たちが試した CPU の地形は、ぜんぶ「なだらか」か「低次元の単純な選択」だった。** だから ③ は要らなかった
  （＝暗くなった蛾、フィッシャー、コイン・バートンの側）。BG9 のカーネル選択もまさにこれで、**構造的に ③ は
  単純なやり方に勝てない**ことが分かった。
- **③ が本当に効くのは「デコボコで高次元の地形」だけ**（＝ライトのシフティング・バランス、レンスキーの大腸菌の側）。
- では、その「デコボコで高次元の地形」はどこにある？ → **GPU で動かす本物の大規模 LLM の損失地形**くらいしか
  残っていない。そこは何百万次元もある＝まさに高次元。

だから「GPU を借りて本物の LLM で ③ を試す」のは、**ヤマ勘ではなく、ちゃんとした理由（高次元の地形でだけ ③ は
意味を持つ）に沿った賭け**になる。ただし **やっぱり賭け**：本物の LLM の地形も、勾配を使う強いやり方（backprop）で
スイスイ進めてしまうなら、結局 ③ は要らない可能性がある（BG9 で「ランダムリスタート山登り法」に勝てなかったのと
同じリスク）。

> **最後に正直なこと**：生物学の話は「なるほど、そういう仕組みか」と納得するための *たとえ話* であって、
> 私たちのコンピュータ実験の結果を *証明* するものではない。とくに、生物では「散らばる作戦」がときに *逆効果*
> になる（良い遺伝子が小さなグループに閉じ込められて広がらない＝シフティング・バランスの第3段階の問題）が、
> 私たちのコンピュータの設定にはそれにあたるものがない。たとえ話がぴったり合わない場所がある、ということは
> はっきり覚えておく。

---

## Appendix — Verified References (cite only these)

All references below had their existence, attribution, and load-bearing content adversarially
verified. Items marked **[needs manual confirmation before submission]** were verified at the
citation/abstract level only or carry a precision flag; confirm page numbers / exact wording against
the primary source before paper submission.

**Wright / shifting balance:**

- Wright, S. 1931. Evolution in Mendelian Populations. *Genetics* 16(2):97–159.
- Wright, S. 1932. The roles of mutation, inbreeding, crossbreeding and selection in evolution.
  *Proceedings of the Sixth International Congress of Genetics* 1:356–366. *(Conference proceedings,
  not a journal article — cite precisely.)*
- Fisher, R. A. 1930. *The Genetical Theory of Natural Selection.* Clarendon Press, Oxford.
- Provine, W. B. 1986. *Sewall Wright and Evolutionary Biology.* University of Chicago Press.
  *(Historical analysis of the controversy; source for the genotype-space vs allele-frequency-space
  ambiguity of the landscape metaphor.)*

**Critique / debate:**

- Coyne, J. A., Barton, N. H., Turelli, M. 1997. Perspective: A critique of Sewall Wright's shifting
  balance theory of evolution. *Evolution* 51(3):643–671. *(Verified verbatim from full text via
  JSTOR 2411143; abstract cross-confirmed OUP + PubMed PMID 28568586.)*
- Coyne, J. A., Barton, N. H., Turelli, M. 2000. Is Wright's shifting balance process important in
  evolution? *Evolution* 54(1):306–317. *(Reply to critics; verified via PubMed PMID 10937209.)*
- Goodnight, C. J. & Wade, M. J. 2000. The ongoing synthesis: a reply to Coyne, Barton, and Turelli.
  *Evolution* 54(1):317–324. **[needs manual confirmation before submission — verified via OUP/BioOne
  at citation level.]**
- Wade, M. J. & Goodnight, C. J. 1998. The theories of Fisher and Wright in the context of
  metapopulations. *Evolution* 52(6):1537–1553. **[needs manual confirmation before submission.]**
- Peck, S. L., Ellner, S. P., Gould, F. 1998. A spatially explicit stochastic model demonstrates the
  feasibility of Wright's shifting balance theory. *Evolution* 52(6):1834–1839. **[needs manual
  confirmation before submission.]**

**Peppered moth (low-dimensional pole):**

- van't Hof, A. E., Campagne, P., Rigden, D. J., et al. 2016. The industrial melanism mutation in
  British peppered moths is a transposable element. *Nature* 534(7605):102–105. doi:10.1038/nature17951.
- van't Hof, A. E., Edmonds, N., Dalíková, M., Marec, F., Saccheri, I. J. 2011. Industrial melanism in
  British peppered moths has a singular and recent mutational origin. *Science* 332(6032):958–960.
  doi:10.1126/science.1203043.
- Cook, L. M. & Saccheri, I. J. 2013. The peppered moth and industrial melanism: evolution of a
  natural selection case study. *Heredity* 110(3):207–212. doi:10.1038/hdy.2012.92.
- Saccheri, I. J., Rousset, F., Watts, P. C., Brakefield, P. M., Cook, L. M. 2008. Selection and gene
  flow on a diminishing cline of melanic peppered moths. *PNAS* 105(42):16212–16217.
  doi:10.1073/pnas.0803785105.
- Cook, L. M., Grant, B. S., Saccheri, I. J., Mallet, J. 2012. Selective bird predation on the
  peppered moth: the last experiment of Michael Majerus. *Biology Letters* 8(4):609–612.
  doi:10.1098/rsbl.2011.1136.

**Lenski Cit+ (high-dimensional contingent pole):**

- Blount, Z. D., Borland, C. Z., Lenski, R. E. 2008. Historical contingency and the evolution of a
  key innovation in an experimental population of *Escherichia coli*. *PNAS* 105(23):7899–7906.
  doi:10.1073/pnas.0803151105.
- Blount, Z. D., Barrick, J. E., Davidson, C. J., Lenski, R. E. 2012. Genomic analysis of a key
  innovation in an experimental *Escherichia coli* population. *Nature* 489(7417):513–518.
  doi:10.1038/nature11514.
- Van Hofwegen, D. J., Hovde, C. J., Minnich, S. A. 2016. Rapid evolution of citrate utilization by
  *Escherichia coli* by direct selection requires *citT* and *dctA*. *J. Bacteriol.* 198(7):1022–1034.
  doi:10.1128/JB.00831-15. **[needs manual confirmation before submission — author list / title
  verified at abstract level only; cited as the *contested* rapid-selection follow-up, not as
  support.]**

**MAP-Elites / Quality-Diversity (the computational operationalization of ③):**

- Mouret, J.-B. & Clune, J. 2015. Illuminating search spaces by mapping elites. *arXiv*:1504.04909.
  *(Does NOT cite Wright / "shifting balance" — Wright is our analogy, not MAP-Elites' lineage.)*
- Lehman, J. & Stanley, K. O. 2011. Abandoning objectives: evolution through the search for novelty
  alone. *Evolutionary Computation* 19(2):189–223.
- Pugh, J. K., Soros, L. B., Stanley, K. O. 2016. Quality diversity: a new frontier for evolutionary
  computation. *Frontiers in Robotics and AI* 3:40.
- Stanley, K. O., Lehman, J., Soros, L. 2017. Open-endedness: the last grand challenge you've never
  heard of. O'Reilly. **[needs manual confirmation before submission — non-archival web essay; verify
  citable form.]**
- Nordmoen, J., et al. 2021. MAP-Elites enables powerful stepping stones and diversity for modular
  robotics. *Frontiers in Robotics and AI* 8:639173. **[needs manual confirmation before
  submission.]**

---

*Generated 2026-06-02. UTF-8. No git operations performed. Biology is illustrative grounding for the
computational finding in `third_axis_selection_arc_2026-06-02.md`, not proof of it; loose junctures of
the analogy are flagged inline.*
