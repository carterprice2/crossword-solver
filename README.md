# Crossword Agent

An AI agent that solves crossword puzzles, running on [Nebius Token
Factory](https://tokenfactory.nebius.com/), together with the evaluation
harness used to measure how good its solutions are.

A crossword is not a list of trivia questions. It is a constraint-satisfaction
problem where the candidate answers happen to come from a language model, and
where every answer is checked by four or five others crossing it. The agent is
built around that fact: it proposes, propagates the crossing constraints,
searches for the best consistent grid, and then **goes back to the model with
the letters it has since learned**. That last step is the interesting one, and
the evaluation is designed to measure exactly what it is worth.

---

## Quick start

No installation, no dependencies, no API key:

```bash
git clone <this repo>
cd crossword-solver
make test     # 249 tests, offline
make demo     # watch a solve, offline
```

`make demo` runs the full agent loop against synthetic candidate lists in which
40% of the correct answers are missing, so you can watch the repair rounds
recover them. Membership checks use `/usr/share/dict/words` (present on macOS;
on Debian/Ubuntu, `apt install wamerican`).

To run it for real, get a key from
[tokenfactory.nebius.com](https://tokenfactory.nebius.com/) (free signup) and
either `export NEBIUS_API_KEY=...` or put it in a gitignored `.env` in this
directory:

```bash
export NEBIUS_API_KEY=...
make models                                  # check the key, list models
python3 -m crossword solve corpus/mini/mini-09-00-0.xd --live
make eval                                    # the ablation matrix
```

Live verification is staged so a cheap check happens before a token-heavy one:

```bash
make verify-offline          # tests, no network
make verify-live-ping        # key works, models reachable
make verify-live-smoke       # one 7x7, arm a3, cheap model, writes results/live-smoke.jsonl
make verify-live-pair        # a2 vs a3 on one 7x7, one seed
make verify-live-ablation    # a2 vs a3 on the four 7x7s, repair model
```

## Stack

- **`crossword/client.py`** — [Nebius Token
  Factory](https://api.tokenfactory.nebius.com/v1/) over its OpenAI-compatible
  `chat/completions` endpoint, with retry, jittered backoff, and `Retry-After`
  handling. Alongside it: recording, replay, scripted and oracle clients, which
  is what lets the entire agent be tested with no network.
- **`crossword/agent/`** — the loop. `candidates.py` batches clues and parses
  responses, `constraints.py` propagates crossing letters, `search.py` picks
  the best consistent grid (dynamic MRV, degree, LCV, nogoods), `solver.py`
  runs the rounds.
- **`crossword/eval/`** — metrics, statistics, the arm matrix, and the report
  generator.
- **`crossword/gen/`** — the puzzle generator that produces the committed
  corpus.
- **`corpus/`** — 12 generated puzzles (7×7, 9×9, 11×11) and the word/clue bank
  they are built from, both fully public domain.

---

## How the agent works

```
round 0   ask the model for candidates for every clue, cold
          → drop candidates contradicting known cells
          → soft AC-3 across every intersection
          → weighted search (MRV + degree + LCV, nogood cache)
          → lock cells where a confident across and a confident down agree

round 1+  pick the slots still unresolved or in conflict; ask again, now
          supplying the pattern the grid has pinned down, the crossing
          answers that pinned it, and everything already ruled out

endgame   fill anything still blank from the per-cell letter marginals
```

Four decisions shape the design. Three of them earn their keep; the fourth is
reported as a negative result rather than quietly dropped.

**Soft AC-3, not textbook AC-3.** Arc consistency assumes the domains contain
every legal value. Here they contain whatever the model happened to say, so a
correct answer often has no supporting candidate on the crossing slot — and
strict propagation deletes it. Instead, a prune that would empty a domain is
skipped and the crossing is recorded as a *conflict site*. Conflict sites are
then exactly the list of places worth spending another model call on.

**A wildcard value — which the ablation shows does not pay off.** Every slot can
be left blank at a fixed probability cost (`unknown_mass`, default 0.15), so the
search can decline instead of committing to a wrong answer that corrupts four
crossing slots. Arm `a6` disables it, and measures the idea as worth roughly
nothing: the two arms land within 0.02 WCR of each other, with `a6` *ahead* at
one noise level. The endgame fill is why — a declined slot does not stay blank,
it gets filled from letter marginals, which is a weaker inference than the
search would have made. Kept in the codebase and reported as a negative result;
see [REPORT.md §6](REPORT.md).

**Batching by locality.** Clues are grouped into requests by walking the
intersection graph, not by clue number, so each request holds clues that
actually cross. The model can then check its own answers against each other.

**Locking requires two directions to agree.** A cell is fixed only when a
confident across answer and a confident down answer produce the same letter.
One model can be confidently wrong; two entries agreeing on a letter is a much
stronger claim. Locks are never released, which is also what guarantees the
loop terminates.

**Repair is a local star, then a global stitch.** A clash is the two entries
plus everything they touch. The agent first enumerates the candidate lists it
already has for that star. Only if they cannot mesh does it re-query the model
about *those* clues, with the shared cells spelled out. If that still fails it
keeps the higher-confidence hub and leaves the other blank, then runs a global
search so neighbouring stars stay consistent.

**Candidates come from the model, not the word bank.** The bank is a
construction tool for the generated corpus. The solver never searches it by
clue — that would be looking up the answer key. Each clue is sent to the LLM
with its length; the model returns a short candidate list. If a proposed word
happens to have a dictionary sense on file, that sense can *boost* the guess,
but it cannot introduce a word the model did not offer. Verify then checks
length, blanks, crossings, and that every complete entry is a real word
(abbreviations and compounds included). A fill that spells LFA down from
LINE across is rejected even though each across is itself a word.

### Model routing

| Stage | Default model | Why |
|---|---|---|
| Round-0 bulk candidates | `Qwen/Qwen3-30B-A3B-Instruct-2507` | High volume, cheap MoE |
| Repair rounds | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Few tokens, highest value per token |
| Ensemble partner (arm A4) | `meta-llama/Llama-3.3-70B-Instruct` | Different family ⇒ decorrelated errors |

Round 0 is most of the tokens and least of the difficulty; the repair rounds are
few tokens on the genuinely hard slots. Spending the expensive model only there
is the main cost lever, and arm A5 measures what using it everywhere buys.

### Structured output

Requests use Nebius's guided JSON (`response_format` with a JSON schema).
Support varies across the 60+ models Token Factory serves, and a model that
rejects a schema returns HTTP 400 — so rather than lose the request, the client
walks a **degradation ladder**: strict schema → unconstrained schema → schema
without `strict` → `json_object` → free text with the schema in the prompt. The
rung each model settled on is recorded and reported, because *which models
honour strict schemas* is itself a useful finding.

Model output is then parsed leniently: reasoning blocks (`<think>…</think>`,
which Qwen3-thinking and gpt-oss emit), code fences, leading prose and trailing
commas are all stripped before parsing, and every candidate is validated against
the slot's length and known pattern before it is allowed into a domain.

---

## Evaluation

Full methodology, results and caveats: **[REPORT.md](REPORT.md)**.

### Metrics

| Metric | Meaning |
|---|---|
| **WCR** | Word Coverage Rate — correct slots / total slots. The primary metric. |
| **LCR** | Letter Coverage Rate — correct letters / fillable cells. |
| **ICR** | Intersection Consistency Rate — crossings where across and down agree. **Needs no answer key**, so it doubles as the solver's own stopping signal. |
| **Exact** | The whole grid correct. Comparable to published solver results. |
| **Precision / recall** | Of the cells it filled, how many were right — versus of all cells, how many it got right. Separates *declining* from *guessing wrong*. |
| **ECE / Brier** | Calibration of self-reported confidence. Matters here because the search consumes those confidences as probabilities: a confidently-wrong model actively damages it. |

WCR/LCR/ICR follow [CrossWordBench](https://arxiv.org/abs/2504.00043); exact
puzzle accuracy follows the [Berkeley Crossword
Solver](https://arxiv.org/abs/2205.09665), which reports 82% exact and 99.9%
letter accuracy on NYT puzzles.

### Ablation arms

Each adjacent pair isolates one mechanism.

| Arm | What it is |
|---|---|
| `a0` | One prompt, whole grid, whole clue list. The naive baseline. |
| `a1` | Per-clue queries, top answer each, crossings ignored. |
| `a2` | `a1` + constraint propagation and search. **No re-query.** |
| `a3` | `a2` + repair rounds. **The full agent.** |
| `a4` | `a3` + a second model family in round 0. |
| `a5` | `a3` with the reasoning model everywhere — the cost ceiling. |
| `a6` | `a3` with the wildcard disabled, so it must guess. |

Every arm runs the identical puzzles at the identical seeds, so all comparisons
are **paired** — which is what makes them meaningful at sample sizes an LLM eval
can afford. Between-puzzle variance (some grids are just harder) is removed
rather than averaged over.

```bash
python3 -m crossword eval --suite mini --arms a0,a1,a2,a3 --seeds 3
python3 -m crossword eval --suite mini --arms a0,a1,a2,a3 --backend oracle  # free
```

### The corpus

**Generated (committed, 12 puzzles).** Built by `crossword/gen/` from a word
bank derived by intersecting [dwyl/english-words](https://github.com/dwyl/english-words)
(public domain) with [Webster's 1913](https://github.com/matthewreagan/WebstersEnglishDictionary)
(public domain, via Project Gutenberg). Commonness is estimated from the
dictionary itself — how many definitions use a given word — which separates real
crossword vocabulary from Webster's deep obscurities without depending on a
restrictively licensed frequency list.

These puzzles have **never been published**, so no model can have memorized
them. That makes them the contamination-free slice of the evaluation. Their
clues are condensed dictionary definitions rather than NYT-style clues, so this
suite is a *control*, not a difficulty proxy — see REPORT.md.

**Real (fetched, not committed).** `python3 scripts/fetch_xd.py` pulls 6,000+
pre-1965 NYT puzzles from [xd.saul.pw](https://xd.saul.pw/), the era now in the
public domain. Those puzzles are famous and long-published, so treat them as
contaminated and report them as a separate stratum.

**One modern Friday (local, not committed).** `python3 scripts/write_nyt_2021_05_28.py`
writes `corpus/nyt/nyt-2021-05-28.xd` — Andrew J. Ries, Friday May 28 2021,
copyright the New York Times. This is the real-clue 15×15 in the evaluation
(`--suite nyt`). Do not redistribute the filled grid.

---

## Why no dependencies

`crossword/client.py` speaks the Nebius REST API directly over `urllib` rather
than through the `openai` SDK. Two reasons:

1. `git clone && make test && make demo` works with no install step, no virtual
   environment, and no network. For a project whose main claim is about
   *evaluation*, being able to reproduce the offline results in one command
   matters more than SDK ergonomics.
2. The needed surface is one `POST /v1/chat/completions`. The client is ~250
   lines including retry, backoff and usage accounting.

The trade-off is real: we reimplement retries and lose streaming (which
structured output does not benefit from anyway). Nebius documents the
`openai` SDK with `base_url="https://api.tokenfactory.nebius.com/v1/"`, and
`pyproject.toml` carries it as an optional `[sdk]` extra for anyone who prefers
it — the `ModelClient` protocol is the seam.

## Commands

```bash
python3 -m crossword solve PUZZLE.xd [--live] [--arm a3] [--prefill 0.25]
python3 -m crossword eval --suite mini --arms a0,a1,a2,a3 [--seeds 3]
python3 -m crossword eval --suite nyt --arms a3   # after scripts/write_nyt_2021_05_28.py
python3 -m crossword report results/run-.../
python3 -m crossword generate --size 9 --seed 3
python3 -m crossword models ping
```

Useful flags: `--backend oracle` runs everything offline against synthetic
candidates; `--backend replay --replay trace.jsonl` replays a recorded run
deterministically (handy for a demo that must not depend on sampling);
`--record trace.jsonl` captures a live run for later replay.

## Repository layout

```
crossword/
  client.py        Nebius client + oracle/replay/recording clients
  schemas.py       guided-JSON schemas, degradation ladder, lenient parser
  model.py         Grid, Slot, Puzzle, Solution
  xd.py            .xd reader/writer
  normalize.py     answer normalization, clue-type heuristics
  agent/           prompts, candidates, constraints, search, solver, trace
  eval/            metrics, stats, harness, report
  gen/             bank, grid templates, filler
  ui/live.py       the animated terminal view
corpus/            generated puzzles, grid templates, word bank
scripts/           build_bank.py, make_corpus.py, oracle_sweep.py, fetch_xd.py,
                   write_nyt_2021_05_28.py
tests/             stdlib unittest, no network
```

## Known limitations

- **No 15×15 generation.** A full-size grid needs a constructor-grade word list;
  our public-domain bank fills up to 11×11. Real 15×15 puzzles come from the
  fetched public-domain corpus and the local NYT Friday fixture (`--suite nyt`).
- **No rebus solving.** The data model handles rebus squares (and parses the
  1955 NYT rebus puzzle correctly), but the agent does not propose them.
- **No theme detection.** Themed puzzles have interdependent long answers; the
  agent treats every slot independently.
- **Clue-type labels are heuristic.** Regex rules, not human labels. Their
  agreement rate should be spot-checked before leaning on the per-type table.

Deliberately out of scope: `.puz` binary parsing, embedding-based retrieval over
the 6M-clue database (a strong lever, but it measures retrieval rather than
reasoning), and fine-tuning.
