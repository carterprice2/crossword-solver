# Evaluation methodology and results

This is the part of the take-home I'd want to be asked about. The agent is
worth building only if it can be shown to beat the obvious alternatives, and
"shown" means a design that could have found the answer *no* as easily as yes.

---

## 1. What is being measured

A crossword solver can fail in several different ways, and a single accuracy
number blurs them together:

- it can put the wrong letter in a square,
- it can leave a square blank,
- it can produce a grid that contradicts itself,
- it can get 95% of a puzzle right and still not have solved the puzzle.

So the harness reports six things per puzzle.

| Metric | Definition | Why it is here |
|---|---|---|
| **WCR** | correct entries / total entries | Primary metric. Continuous, so it has usable statistical power at small n. |
| **LCR** | correct letters / fillable squares | Partial credit within an entry; the metric published solvers report. |
| **ICR** | crossings where across and down agree | **Needs no answer key.** Computable during a solve, so it doubles as the stopping signal. |
| **Exact** | every square correct | The only metric a human solver would recognize as "solved". |
| **Cell precision / recall** | of squares filled, how many right / of all squares, how many right | Separates *declining* from *guessing wrong*. |
| **ECE / Brier** | calibration of self-reported confidence | The search consumes confidence as a probability, so miscalibration is a direct harm, not a cosmetic one. |

WCR, LCR and ICR follow [CrossWordBench](https://arxiv.org/abs/2504.00043).
Exact-puzzle accuracy follows the [Berkeley Crossword
Solver](https://arxiv.org/abs/2205.09665) (82% exact, 99.9% letter accuracy on
NYT), which is the reference point for what "good" means on real puzzles.

**Why precision and recall earn their place.** An agent that leaves a square
blank and one that fills it wrongly score identically on LCR. They are not the
same system. The wildcard mechanism (§6) was built to trade the second failure
for the first, so the evaluation needed to be able to see the difference —
which is how it was able to show that the mechanism does not in fact deliver it.

---

## 2. Design of the comparison

### Arms

| Arm | What it is | Isolates |
|---|---|---|
| `a0` | One prompt: whole grid, whole clue list | the baseline |
| `a1` | Per-clue queries, top answer each, crossings ignored | decomposition |
| `a2` | `a1` + constraint propagation and weighted search | **constraint propagation** |
| `a3` | `a2` + repair rounds — **the full agent** | **the agentic loop** |
| `a4` | `a3` + a second model family in round 0 | ensembling |
| `a5` | `a3` with the reasoning model at every stage | the cost ceiling |
| `a6` | `a3` with the wildcard disabled — must guess | **the option to decline** |

Each adjacent pair differs by exactly one mechanism, so `a3 − a2` is the value
of the re-query loop and nothing else.

### Pairing

Every arm runs the identical puzzles at the identical seeds. This matters more
than the sample size does. Between-puzzle variance dominates — a 7×7 of common
words and an 11×11 of Webster obscurities are not the same task — and pairing
removes that variance instead of averaging over it. All reported differences are
paired bootstrap over puzzles (10,000 resamples) on the *difference*, not two
independent means with overlapping error bars.

### Two variances, kept apart

Sampling temperature makes the same arm score differently on the same puzzle.
That is not the same uncertainty as "how would this arm do on a *new* puzzle",
and averaging them into one error bar overstates confidence. The harness runs
the headline arm at three seeds and reports seed SD separately from the
across-puzzle bootstrap CI.

### Exact-solve is reported, but not leaned on

Exact-solve is one Bernoulli trial per puzzle. At n=40 a Wilson interval spans
roughly ±15 points; separating 80% from 70% at 80% power needs about **300
puzzles**, which `stats.required_n` computes and the generated summary prints.
So exact-solve appears as a descriptive rate with its interval, and arm-vs-arm
claims about it use **McNemar's exact test** on the discordant pairs — paired,
and far more sensitive than comparing two proportions.

### Difficulty control

`prefill_ratio` reveals a fraction of the grid before the solve, following
CrossWordBench. Revealed cells are spread rather than sampled independently: a
random draw clusters, and a clustered reveal hands over whole answers instead of
uniformly reducing difficulty.

Results are also stratified by grid size, answer length, clue type
(definition / fill-in-the-blank / abbreviation / wordplay / proper noun /
crosswordese) and provenance.

---

## 3. Results: the offline sweep

The agent's real input is not a puzzle — it is a **noisy candidate list**. That
input can be simulated exactly, which makes the architecture measurable without
an API key: `OracleClient` builds candidate lists from a known solution with a
controlled probability that the correct answer is absent (`recall`) and that it
is not ranked first (`top1_error`). Sweeping those knobs answers the question
the ablation is really asking — *how much model error does this machinery
absorb?*

12 generated puzzles × 3 seeds per cell. Reproduce with `make sweep`; raw
numbers in [`results/synthetic-sweep.json`](results/synthetic-sweep.json).

### WCR by arm and candidate quality

<!-- BEGIN SWEEP_TABLE -->
| candidate recall | top-1 error | a0 one-shot | a1 per-clue | a2 +constraints | a3 full agent | a6 no wildcard |
|------------------|-------------|-------------|-------------|-----------------|---------------|----------------|
| 0.95             | 0.20        | 0.773       | 0.725       | 0.975           | 0.997         | 0.997          |
| 0.80             | 0.35        | 0.633       | 0.576       | 0.913           | 0.990         | 0.987          |
| 0.65             | 0.45        | 0.443       | 0.488       | 0.714           | 0.946         | 0.952          |
| 0.50             | 0.55        | 0.341       | 0.410       | 0.601           | 0.877         | 0.898          |
| 0.35             | 0.65        | 0.270       | 0.274       | 0.450           | 0.731         | 0.717          |

Mean WCR over 12 puzzles x 3 seeds. `recall` is the probability the correct answer appears among the candidates at all; `top-1 error` the probability it is not ranked first. Lower rows are weaker models.
<!-- END SWEEP_TABLE -->

### The finding

**The repair loop's value grows as the base model gets worse.** When candidates
are good it adds little, because constraint propagation alone already resolves
the grid. When candidates are poor it is worth several times more.

<!-- BEGIN REPAIR_DELTA -->
| candidate recall | a2 WCR | a3 WCR | repair gain | 95% CI           | calls a2 -> a3 |
|------------------|--------|--------|-------------|------------------|----------------|
| 0.95             | 0.975  | 0.997  | +0.022      | [+0.007, +0.039] | 4.5 -> 9.8     |
| 0.80             | 0.913  | 0.990  | +0.078      | [+0.051, +0.109] | 4.5 -> 11.6    |
| 0.65             | 0.714  | 0.946  | +0.232      | [+0.186, +0.278] | 4.5 -> 12.4    |
| 0.50             | 0.601  | 0.877  | +0.277      | [+0.247, +0.306] | 4.5 -> 12.6    |
| 0.35             | 0.450  | 0.731  | +0.282      | [+0.258, +0.307] | 4.5 -> 13.0    |

Paired bootstrap over puzzles on the difference, 4,000 resamples. Every interval excludes zero.
<!-- END REPAIR_DELTA -->

This is the practically useful result, because it says *when to spend the extra
tokens*: with a strong model on easy puzzles, one propagation pass is nearly all
of the benefit and the repair rounds roughly double the call count for a small
gain. With a weaker or cheaper model, the loop is what makes the system work at
all — which is an argument for pairing a cheap bulk model with an agentic loop
rather than paying for a large model on every clue.

### The mechanism that did not work

The wildcard — letting the solver decline a slot rather than guess — was
expected to buy cell precision. It buys nothing, and at one noise level it is
significantly *harmful*. The numbers and the reason are in §6; it is recorded
here rather than quietly dropped, because an ablation suite that only ever
confirms its own design is not measuring anything.

### What the sweep cannot tell you

The oracle answers each slot independently. It therefore **cannot** model the
way a real model degrades on a long whole-grid prompt, which is precisely the
weakness `a0` is supposed to exhibit. The `a1 − a0` contrast offline is a
mechanism check, not a result; only a live run measures it. This caveat is
recorded in the sweep output itself so the number cannot be quoted out of
context.

Equally, the oracle's confidences are drawn from a tunable distribution rather
than produced by a model that might be systematically overconfident. Calibration
(ECE, Brier) is therefore only meaningful on a live run.

---

## 4. Corpus, and the contamination problem

Crossword evaluation has a contamination problem that is easy to overlook: the
puzzles worth testing on are famous, published, and in every model's training
data. A model may be recalling the 1955 NYT grid rather than solving it.

Two corpora, deliberately:

**Generated (committed, 12 puzzles, 7×7–11×11).** Built by `crossword/gen/` from
a word bank derived by intersecting a public-domain word list with Webster's
1913. These puzzles **have never been published**, so no model can have
memorized them. They are the contamination-free slice.

**Real (fetched, not committed).** 6,000+ pre-1965 NYT puzzles from xd.saul.pw,
the era now in the public domain — and the only source of real 15×15 grids here.

### Two probes

1. **Generated vs published, side by side.** The generated report prints WCR by
   provenance. A large positive gap on published puzzles is a memorization
   signal rather than a capability signal.
2. **Blank-clue probe.** Ask for the answers giving only the puzzle title, date,
   slot ids and lengths — *no clue text*. Accuracy meaningfully above a
   length-and-letter-frequency prior is direct evidence of recall. Puzzles above
   threshold are flagged and excluded from headline numbers.

### Honest limitations of the generated corpus

I would rather state these than have them found:

- **The clues are dictionary definitions, not crossword clues.** Condensed
  Webster 1913 entries are definitional and often archaic. Real NYT clues use
  wordplay, misdirection and cultural reference. The generated suite is a
  **control set**, not a difficulty proxy — it establishes a contamination-free
  floor, and the published corpus establishes realism. Neither alone is enough.
- **The vocabulary skews old.** Webster 1913 has no modern entries, and the
  filler will reach for obscurities (UVA, LEA, ALA) when a grid is tight.
  Commonness ordering (§6) keeps most of the fill ordinary, but not all of it.
- **No 15×15, no themes, no rebus.** A constructor-grade word list would be
  needed for 15×15; themed puzzles need interdependent long answers the
  generator does not model.
- **Clue-type labels are regex heuristics.** Their agreement with human labels
  should be spot-checked on ~100 clues before the per-type table is leaned on.

---

## 5. What a live run should report

Everything above runs offline. A live run on Nebius Token Factory adds the
model comparison matrix. Produce it in stages with `make screen-arms`, then
`make screen-models FROM=...`, then `make final-grid FROM=...` — each stage
writes a per-cell grid (WCR/LCR/ICR, tokens, USD, turns, time) and a
`winners.json` you can edit before the next.

1. **The model comparison matrix** — `Qwen3-30B-A3B-Instruct-2507`,
   `Qwen3-235B-A22B`, `gpt-oss-120b`, `Llama-3.3-70B-Instruct`,
   `DeepSeek-V4-Pro`, `Kimi-K2.5`, `GLM-5.2` × WCR, LCR, exact, ECE, tokens,
   USD, p95 latency. n=10 per cell for cost; n=40 on the headline arm.
2. **Calibration per model** — a reliability diagram. The prediction worth
   testing: models differ more in *calibration* than in raw accuracy, and for
   this architecture calibration matters more, because the search weights
   candidates by their stated confidence.
3. **`a0` versus the rest**, which the oracle cannot measure (§3).
4. **Structured-output support** — which rung of the degradation ladder each
   model accepted. Recorded automatically and printed in every summary.
5. **Cost per solved puzzle**, the Pareto frontier over arms and models.

The recommended first run:

```bash
export NEBIUS_API_KEY=...
python3 -m crossword models ping
python3 -m crossword eval --suite mini --arms a0,a1,a2,a3 --seeds 3
python3 scripts/fetch_xd.py --what puzzles
python3 -m crossword eval --suite xd --arms a2,a3 --limit 40 --seeds 3
```

---

## 6. Design decisions worth defending

**Soft AC-3 rather than AC-3.** Arc consistency assumes domains contain every
legal value. Here they contain whatever the model said, so a correct answer
often has no supporting candidate on the crossing slot, and strict propagation
deletes it. A prune that would empty a domain is skipped and the crossing is
recorded as a conflict site instead — which then becomes the list of places
worth another model call. This is the single most important implementation
detail in the constraint layer.

**CS50 search heuristics, adapted for scored LLM domains.** The fill is a
weighted CSP, not "find any legal grid." Variable order is dynamic **MRV**
(fewest candidates that still fit the current letters) with the **degree**
heuristic as a tie-break. Values stay ordered by score; **least-constraining
value** only breaks ties, because trying the 0.9 answer before the 0.2 one is
what makes branch-and-bound prune. Contradictory partial assignments are
cached as nogoods so a restart does not replay them. The CS50 unique-word
rule is *not* used: real clues can independently produce the same short
answer (the 3×3 fixture has ARE across and ARE down), and uniqueness is a
constructor constraint, not a solver one.

**A wildcard value — and the ablation says it does not pay off.** Any slot can
be left blank at a fixed probability cost, the intent being that declining beats
committing to a wrong answer that corrupts four crossing slots. Arm `a6`
disables it. The measured result is a **negative one**:

| recall | a3 (can decline) | a6 (must guess) | a3 − a6 | a3 cell precision | a6 cell precision |
|---|---|---|---|---|---|
| 0.95 | 0.997 | 0.997 | +0.000 | 0.999 | 0.999 |
| 0.80 | 0.990 | 0.987 | +0.003 | 0.998 | 0.997 |
| 0.65 | 0.946 | 0.952 | −0.007 | 0.987 | 0.988 |
| 0.50 | 0.877 | 0.898 | **−0.021** (CI [−0.04, −0.01]) | 0.968 | 0.974 |
| 0.35 | 0.731 | 0.717 | +0.014 | 0.928 | 0.924 |

At recall 0.50 the difference is significant *against* the wildcard, and the
precision benefit I expected does not appear at all — the two arms are within
0.006 of each other everywhere.

The explanation is that the endgame fill undoes the mechanism. A declined slot
does not stay blank; it gets filled from the per-cell letter marginals, which is
a *weaker* inference than the search would have made. So the wildcard does not
buy precision, it just defers the decision to a worse estimator. Forcing a guess
that at least satisfies the crossing constraints turns out to be the better bet.

Two honest conclusions: the wildcard should either be removed or paired with a
solver that genuinely reports blanks, and — more importantly — this is what the
ablation is *for*. A mechanism that sounded compelling in design turned out to
be worth nothing, and the design would not have revealed that if `a6` had been
left out because the answer seemed obvious.

**Locking needs two directions to agree.** A cell is fixed only when a confident
across *and* a confident down produce the same letter. One model can be
confidently wrong; two entries agreeing is a much stronger claim. Locks are
never released, which is also what makes the loop provably terminate.

**Two patterns in the repair prompt.** The confirmed pattern (locked cells) is a
hard constraint; the likely pattern (including unconfirmed crossing answers) is
a hint, labelled as one. Domain filtering uses only the confirmed pattern — a
tentative letter must never delete a correct candidate. In a representative run
the confirmed pattern covers 20% of the letters and the likely pattern 97%, so
conflating them would have been either far too weak or actively misleading.

**Frequency orders the filler, it does not gate the bank.** A small word bank
cannot fill a dense grid; a large unordered one fills it with obscurities.
Ordering candidate words by commonness gets both. Commonness is estimated from
Webster itself — how many definitions use a word — which avoids depending on a
frequency list under a licence that would contaminate the repo.

**No dependencies.** The Nebius API is OpenAI-compatible REST; the client is
~250 lines of `urllib`. The payoff is that `git clone && make test && make demo`
reproduces the offline results with no install step and no network, which for a
project whose main claim is about evaluation is worth more than SDK ergonomics.

---

## 7. What I would do next

In order of expected value:

1. **Retrieval over the 6M-clue database.** Crossword clues repeat heavily; an
   exact normalized-clue lookup is nearly free and would likely beat every
   arm here. It must be reported as a *separate* arm with leave-one-puzzle-out,
   because its lift is a retrieval result, not a reasoning result.
2. **Calibration-aware confidence.** Fit a per-model mapping from stated
   confidence to empirical accuracy and feed the corrected value to the search.
   Given how directly the search consumes these numbers, this is probably the
   cheapest real accuracy gain available.
3. **Theme detection.** Themed puzzles have interdependent long answers; solving
   them one slot at a time leaves information on the table.
4. **Rebus support.** The data model already handles multi-letter squares (it
   parses the 1955 NYT rebus puzzle correctly); the agent does not propose them.
5. **A 15×15 generator**, which needs a constructor-grade word list.
