# Evaluation methodology and results

This is the part of the take-home I'd want to be asked about. The agent is
worth building only if it can be shown to beat the obvious alternatives, and
"shown" means a design that could have found the answer *no* as easily as yes.

Two layers of evidence, on purpose:

1. **Offline sweep** (§3) — 12 puzzles × 3 seeds × five candidate-quality
   levels, no API key. This is the statistical backbone: paired, powered, and
   cheap enough to be honest about negative results.
2. **Live Token Factory runs** (§5) — real models, real tokens, real cost.
   Smaller n. I quote them as what they are, not as a substitute for the sweep.

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

### Live tournament (how the model matrix is produced)

A full 8-model × 7-arm × 3-size cube is the wrong spend. The harness instead
runs three pause-gated recipes (`make screen-arms`, `make screen-models FROM=…`,
`make final-grid FROM=…`). Each stage writes a per-cell grid (WCR/LCR/ICR,
tokens, USD, turns, time) and a `winners.json` you can edit before the next.
Rank is **mean WCR**; cost is a column and a tie-break, not a filter. Nothing
auto-starts the next stage. What actually ran, and what did not, is §5.

---

## 3. Results: the offline sweep

The agent's real input is not a puzzle — it is a **noisy candidate list**. That
input can be simulated exactly, which makes the architecture measurable without
an API key: `OracleClient` builds candidate lists from a known solution with a
controlled probability that the correct answer is absent (`recall`) and that it
is not ranked first (`top1_error`). Sweeping those knobs answers the question
the ablation is really asking — *how much model error does this machinery
absorb?*

Reproduce with `make sweep`; raw numbers in
[`results/synthetic-sweep.json`](results/synthetic-sweep.json). The committed
file is 12 puzzles × 3 seeds. The current `corpus/mini` suite is 7 unique grids
(§4); I have not re-run the sweep against the smaller set, and the tables below
are from that file.

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

The live screen in §5 is the other side of that argument: once the model is
strong enough to exact-solve an 11×11, the interesting question is no longer
"does repair help" but "do you still need the expensive model on every round."
That contrast (`a5 − a3`) is **not** in the live numbers yet.

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

**Generated (committed, 7 unique puzzles, 7×7–11×11).** Built by `crossword/gen/`
from a word bank derived by intersecting a public-domain word list with
Webster's 1913. These puzzles **have never been published**, so no model can
have memorized them. They are the contamination-free slice. The live 11×11 in
§5 is `mini-11-04-0` from this set.

**Real (fetched, not committed).** 6,000+ pre-1965 NYT puzzles from xd.saul.pw,
the era now in the public domain — and the only source of real 15×15 grids here.
I have not yet run the live harness on that stratum.

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

## 5. Results: live, on Token Factory

The offline sweep measures the architecture against *simulated* candidate
noise. A live run measures it against real models: cost, structured-output
support, and the whole-grid prompt the oracle cannot fake.

I did not run the 8 × 7 × 3 cube. Stages, one seed, rank by WCR. Cost is
visible on every row and decides ties.

### What ran, and what did not

| Run | What it is | Quote it? |
|---|---|---|
| [`live-screen-models-11`](results/live-screen-models-11/summary.md) | 7 models × arm `a5` × one 11×11 (`mini-11-04-0`) | **Yes. This is the live number.** |
| [`live-max-correct-arms-11`](results/live-max-correct-arms-11/summary.md) | `a5` vs `a6` on Qwen 397B, same 11×11 | Yes, as a failure mode, not as an arm ranking |
| `live-screen-arms` / `run-20260901-220639` | Qwen 30B 7×7, aborted after connection drops on `a1`/`a2` | No. Those WCR numbers are not a mechanism contrast |
| `final-grid` (7 / 9 / 11 × top 3 × top 3) | Not run | — |
| `zai-org/GLM-5.2` | In `KNOWN_MODELS`, no cell | — |
| Published NYT / xd 15×15 | Not run | — |

USD uses the frozen Token Factory rates in `crossword/eval/pricing.py`
(observed 2026-08-28). Unknown ids would print `?`; every model below is in
the table.

### Model screen — one 11×11, arm `a5`, seed 0

Every cell used **the screened model at every stage** (`a5`). That is the
cost ceiling, not the default agent (`a3`, cheap bulk + expensive repair).
Read the table as a *model* ranking on a hard generated puzzle, not as an
ablation of the loop.

<!-- BEGIN LIVE_MODEL_GRID -->
| model                         | WCR   | LCR   | exact | tokens | USD   | turns | calls | sec   | rung          |
|-------------------------------|-------|-------|-------|--------|-------|-------|-------|-------|---------------|
| MiniMax-M3                    | 1.000 | 1.000 | 1     | 23251  | 0.011 | 5     | 15    | 18.6  | strict_schema |
| DeepSeek-V4-Pro               | 1.000 | 1.000 | 1     | 14012  | 0.034 | 3     | 9     | 24.3  | strict_schema |
| Qwen3.5-397B-A17B             | 0.952 | 0.989 | 0     | 48364  | 0.143 | 3     | 10    | 554.6 | strict_schema |
| gpt-oss-120b                  | 0.881 | 0.957 | 0     | 51634  | 0.026 | 3     | 11    | 171.8 | json_object   |
| Qwen3-235B-A22B-Instruct-2507 | 0.833 | 0.946 | 0     | 28044  | 0.010 | 5     | 14    | 122.7 | strict_schema |
| Llama-3.3-70B-Instruct        | 0.619 | 0.806 | 0     | 28645  | 0.006 | 5     | 19    | 146.2 | strict_schema |
| Qwen3-30B-A3B-Instruct-2507   | 0.524 | 0.763 | 0     | 51891  | 0.011 | 5     | 21    | 300.9 | strict_schema |

Arm `a5` (the screened model at every stage), puzzle `mini-11-04-0` (11×11, 42 slots), seed 0, prefill 0. Ranked by WCR, then lower USD. Raw cells: `results/live-screen-models-11/cells.jsonl`.
<!-- END LIVE_MODEL_GRID -->

**MiniMax-M3 is the pick.** It tied DeepSeek-V4-Pro at exact 1.000 WCR / LCR
and won the tie-break: **$0.011 vs $0.034**, 18.6s vs 24.3s. DeepSeek used
fewer tokens (14k vs 23k) and fewer calls (9 vs 15); MiniMax was still the
cheaper wall-clock and dollar win. `winners.json` advances MiniMax, DeepSeek,
and Qwen 397B (0.952 WCR, not exact).

The cheap default, Qwen 30B, landed at **0.524 WCR** with 13 slots declined.
The loop cannot invent a word the model never offered. Llama-3.3-70B (0.619,
12 open) is the same shape. gpt-oss-120b (0.881) and Qwen 235B (0.833) sit
in the middle: high LCR, not exact.

Two models exact-solving one 11×11 is a descriptive rate, not a published
solver number. Wilson on 2/7 of *models* is the wrong interval; the right
caveat is **n = 1 puzzle, n = 1 seed**.

### What this does and does not isolate

- **It isolates model quality**, holding arm and puzzle fixed. The spread
  from 0.524 to 1.000 on the same 42 slots is the live result.
- **It does not isolate `a3 − a2`.** That claim still belongs to the offline
  sweep. Every live cell here is `a5`.
- **It does not isolate `a5 − a3`.** I do not know whether MiniMax would
  exact-solve this grid with cheap bulk candidates and expensive repair only.
  That is the next cell I would buy (§7).
- **It does not measure `a0`.** A 7×7 one-shot on Qwen 30B scored 0.25 WCR
  before the arm screen died on dropped connections. I am not treating that
  as the `a0` vs `a1` result the sweep said only live can provide.

### Structured output

Six of seven models accepted `strict_schema`. **gpt-oss-120b fell back to
`json_object`.** The degradation ladder did its job: the cell completed
(0.881 WCR) instead of dying on HTTP 400. Which rung a model settles on is
itself a finding, and it is recorded on every cell.

### A run that failed, and why it is in the write-up

The same Qwen 397B, same arm `a5`, same `mini-11-04-0`, about an hour apart:

<!-- BEGIN LIVE_QWEN397_ARMS -->
| arm | WCR   | LCR   | Prec  | Rec   | open | USD   | JSON misses |
|-----|-------|-------|-------|-------|------|-------|-------------|
| a5  | 0.095 | 0.151 | 1.000 | 0.151 | 38   | 0.165 | 8           |
| a6  | 0.000 | 0.000 | 0.000 | 0.000 | 42   | 0.168 | 10          |

`Qwen/Qwen3.5-397B-A17B` on `mini-11-04-0`, seed 0. `JSON misses` counts `no JSON object found in response` warnings. Raw cells: `results/live-max-correct-arms-11/cells.jsonl`.
<!-- END LIVE_QWEN397_ARMS -->

The later screen-models cell for that model is 0.952 WCR, `strict_schema`,
one dropped connection, no JSON-miss pile-up. I am not going to pretend
0.095 and 0.952 are the same measurement with noise bars. They are two
draws from a flaky live API, and n=1 cannot tell them apart from a parser
or prompt change in between.

The thing I *will* take from the failed pair: when the model returns no
candidates, **forcing a guess (`a6`) cannot beat declining (`a5`)**. There
is nothing to guess from. `a6` scored 0.000 with 42 open slots; `a5` scored
0.095 with 38 open and precision 1.000 on the few cells it filled. That is
a live illustration of why precision/recall sit next to WCR, and it is the
opposite of the offline wildcard finding (there the oracle always produces
*some* list). Both can be true.

### Reproduce

The live numbers above are generated from the committed jsonl, not typed in
by hand. From the repo root, with no API key:

```bash
python3 scripts/update_report.py
python3 -m crossword report results/live-screen-models-11/
```

To spend another Token Factory run, resume is safe for the seven finished
cells and **will start `GLM-5.2`**, which has no line in that jsonl:

```bash
python3 -m crossword eval --recipe screen-models \
    --arms a5 --puzzles mini-11-04-0 \
    --run-id live-screen-models-11
```

The recommended next spend, if I were continuing the tournament:

```bash
make final-grid FROM=results/live-screen-models-11
```

That would put MiniMax, DeepSeek, and Qwen 397B on one 7×7, one 9×9, and
this 11×11. I have not paid for it yet. Override `--arms a3,a5` if the
question is whether the cost ceiling is still doing work once the model is
strong.

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
disables it. The measured result, offline, is a **negative one**:

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
that at least satisfies the crossing constraints turns out to be the better bet
— **when there is a list to guess from.** The live Qwen 397B pair in §5 is the
other branch: empty lists, `a6` scores zero.

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

In order of expected value, given what is now on disk:

1. **`a3` vs `a5` on MiniMax, same 11×11.** The live screen used the cost
   ceiling. If MiniMax exact-solves as `a3`, the default routing (cheap bulk,
   expensive repair) is enough and `a5` is a luxury. If it does not, the
   ceiling is load-bearing. One cell. Highest information per dollar.
2. **`final-grid`** from `results/live-screen-models-11` — MiniMax, DeepSeek,
   Qwen 397B × one 7×7, one 9×9, this 11×11. Optionally `--arms a3,a5`. That
   is the published matrix the tournament was designed to produce.
3. **Live `a2` vs `a3` on MiniMax**, several minis, one seed. The offline
   sweep owns the repair-loop claim; a live confirmation on a model that can
   exact-solve would show whether the loop still pays once candidates are
   good, which is exactly the cheap end of the sweep curve.
4. **Calibration-aware confidence.** A MiniMax-only re-run of the exact
   cell logged ECE 0.064 on a perfect fill (overconfident on cells it got
   right). Fit a per-model map from stated confidence to empirical accuracy
   and feed the corrected value to the search. Still probably the cheapest
   real accuracy gain on weaker models, where the search is actually using
   those numbers to choose.
5. **Retrieval over the 6M-clue database.** Crossword clues repeat heavily; an
   exact normalized-clue lookup is nearly free and would likely beat every
   arm here. It must be reported as a *separate* arm with leave-one-puzzle-out,
   because its lift is a retrieval result, not a reasoning result.
6. **Published 15×15.** `scripts/write_nyt_2021_05_28.py` and `fetch_xd.py`
   exist. The generated suite is a control; it is not NYT. Theme detection and
   rebus remain out of the agent even though the data model already parses a
   1955 rebus puzzle.
