# Staged live eval tournament

Date: 2026-09-01
Status: approved (chat design, 2026-09-01)

A live evaluation you can actually read: one metric row per
(puzzle × model × arm), produced in three pause-gated stages so we do not
pay for a 7-model × 7-arm × 3-size cube.

This extends the existing `crossword eval` harness. It does not replace it.

## Decisions already made

- Stages, not a full factorial. Kill strategies, then models, then a 3×3×3 grid.
- Rank by **WCR**. Cost, tokens, turns, and time are reported on every row
  and do not decide who advances.
- Pause after each stage. Print the grid. The next recipe reads `winners.json`
  (editable). Nothing auto-starts the next stage.
- Strategy screen includes **a0–a6** (a4 ensemble and a5 “big model
  everywhere” are in).
- Screens run on **one 7×7**. The final grid is one 7×7, one 9×9, one 11×11.
- Visible grid columns: size, puzzle, model, arm, WCR, LCR, ICR, exact,
  tokens (sum), USD, turns, calls, sec. Prompt/completion stay split in JSON.
- `--recipe` sugar over flags. `--from` loads winners. jsonl resume.
- Existing `crossword eval` / `make eval` is unchanged when the new flags
  are omitted: one `--model`, arms × suite, no winners file required.

## Out of scope

Reliability diagrams, xd/NYT 15×15, auto-advance, parallelizing *across*
cells (intra-solve concurrency stays as it is), fetching live prices on
every run, a new `crossword tournament` command, web UI for the grid.

The existing arm-ablation report (paired bootstrap, McNemar, clue-type
slices) stays. The new grid is an extra section, not a replacement.

---

## 1. What already exists

`crossword eval` already runs arm × puzzle × seed, scores WCR/LCR/ICR,
and records calls, prompt tokens, completion tokens, rounds, and seconds.
`summary.md` aggregates those by arm.

Gaps this spec fills:

- USD cost (tokens are recorded; there is no rate table).
- Model as a matrix axis (`--model` is one value for the whole run).
- One row per (puzzle, model, arm), not a mean-by-arm table.
- Named recipes and a winners file so a human can cut the matrix between
  stages.
- Crash-safe resume.

Puzzle IDs for the three sizes, from `corpus/manifest.json`:

| size | id |
|---|---|
| 7×7 | `mini-07-00-0` |
| 9×9 | `mini-09-00-0` |
| 11×11 | `mini-11-04-0` |

---

## 2. Cell, cost, and ranking

### Cell key

One completed solve is a **cell**:

```
(puzzle_id, model, arm, seed, prefill)
```

`model` is the primary model for that cell (the `--models` value). For arm
`a5` it is also the repair model. For arm `a4` it is the round-0 model; the
ensemble model stays `ENSEMBLE_MODEL` unless `--ensemble-model` overrides it.

### Cost

`Usage.by_model` today stores a single token total per model. Prompt and
completion have different rates, so that is not enough.

Change `Usage.by_model` to:

```python
by_model: dict[str, dict[str, int]]
# {model: {"prompt": int, "completion": int}}
```

`crossword/eval/pricing.py` holds a frozen table of `$ / 1M tokens` for
every id in `KNOWN_MODELS`. Copied from public Nebius Token Factory rates
observed 2026-08-28 (InferenceBench). Update the file when rates move; do
not fetch on every eval.

| model | prompt $/1M | completion $/1M |
|---|---|---|
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 0.10 | 0.30 |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | 0.20 | 0.60 |
| `meta-llama/Llama-3.3-70B-Instruct` | 0.13 | 0.40 |
| `openai/gpt-oss-120b` | 0.15 | 0.60 |
| `deepseek-ai/DeepSeek-V4-Pro` | 1.75 | 3.50 |
| `moonshotai/Kimi-K2.6` | 0.95 | 4.00 |
| `zai-org/GLM-5.2` | 1.40 | 4.40 |

```
cost_usd(usage) = sum over models of
    prompt * prompt_rate / 1e6 + completion * completion_rate / 1e6
```

Unknown model id → `cost_usd` is `None`, shown as `?` in the grid. Never
invent a rate.

`SolveResult.as_dict()` gains `cost_usd` (float or null) and already has
`rounds` (this is **turns**), `calls`, `prompt_tokens`, `completion_tokens`,
`seconds`.

### Ranking

Inside a stage, group cells that share the thing being ranked (arm in
stage 1, model in stage 2). Mean WCR across that group, descending.

Take the top 3. Ties break by lower `cost_usd` (None sorts last), then by
name. Write the ordered list to `winners.json`. Cost did not decide the
rank order except as a tie-break.

Failed cells (exception, empty assignment) stay in the grid with
`scores` null and `error` set. They do not enter the mean. A rank group
with zero successful cells is omitted from winners and printed as a
failure row.

---

## 3. Results on disk

A run directory `results/<run-id>/` contains:

| file | role |
|---|---|
| `cells.jsonl` | One JSON object per finished cell, appended **before** the next cell starts. Resume reads this. |
| `results.json` | Full payload, rewritten after the run (and by `crossword report`). |
| `summary.md` | Human report, including the new **Results grid** and **Leaderboard**. |
| `winners.json` | Ranked arms and/or models for `--from`. |

`cells.jsonl` line (success):

```json
{
  "puzzle_id": "mini-07-00-0",
  "size": "7x7",
  "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
  "arm": "a3",
  "seed": 0,
  "prefill": 0.0,
  "scores": {"wcr": 0.812, "lcr": 0.84, "icr": 1.0, "exact": false},
  "solve": {
    "rounds": 3,
    "calls": 8,
    "prompt_tokens": 40000,
    "completion_tokens": 8000,
    "cost_usd": 0.072,
    "seconds": 41.2,
    "error": null
  }
}
```

On failure, `scores` is null and `solve.error` is a short string. The
line is still written so resume will skip the failed cell unless
`--retry-errors` is set.

### Resume

`Harness.run` builds the planned cell list, loads keys already in
`cells.jsonl` for this `run-id`, and skips them. Re-invoking the same
`--run-id` continues. `crossword report <dir>` rebuilds `results.json`,
`summary.md`, and `winners.json` from `cells.jsonl` even if the process
died before the final write.

---

## 4. Grid and leaderboard in `summary.md`

New section **Results grid**, one markdown row per cell, columns in this
order:

size | puzzle | model | arm | WCR | LCR | ICR | exact | tokens | USD | turns | calls | sec

- `tokens` is prompt + completion, integer.
- `turns` is `rounds`.
- `USD` is `0.072` or `?`.
- Failed cells: scores shown as `err`, last column may carry the error
  snippet.

New section **Leaderboard**, the ranking used for `winners.json`. After
`screen-arms` it is by arm; after `screen-models` it is by model; after
`final-grid` it is by (model, arm) mean WCR across the three sizes.

Existing arm-mean tables, paired comparisons, and calibration stay below.

---

## 5. Recipes and CLI

`build_parser` on `eval` gains:

```
--models          comma list; default is --model (one value). Each value
                  is a primary-model axis entry.
--puzzles         comma list of puzzle ids (or paths). Intersects the suite.
--recipe          screen-arms | screen-models | final-grid
--from            path to a prior run directory (reads winners.json)
--retry-errors    re-run cells whose jsonl line has error set
```

`--recipe` expands flags **unless the user already passed that flag**:

| recipe | models | arms | puzzles |
|---|---|---|---|
| `screen-arms` | `DEFAULT_MODEL` (Qwen 30B) | `a0,a1,a2,a3,a4,a5,a6` | `mini-07-00-0` |
| `screen-models` | all `KNOWN_MODELS` | **single best arm** from `--from` (else `a3`) | `mini-07-00-0` |
| `final-grid` | top 3 models from `--from` | top 3 arms from `--from` | `mini-07-00-0`, `mini-09-00-0`, `mini-11-04-0` |

`--from` is required for `screen-models` and `final-grid` unless `--arms`
and/or `--models` fully specify the axis that would have come from
winners. Missing winners with no override → exit 2:
`no winners.json; pass --from or --arms/--models`.

### How `--from` chains

`screen-arms` writes `winners.json` with ranked `arms` and the one `model`
that ran.

`screen-models --from results/<arms-run>` uses `arms[0]` (best WCR) as the
sole arm, ranks models, and writes `winners.json` with:

- `arms`: copied from the stage-1 file (all three, not just the one used
  to screen models)
- `models`: top 3 from this stage

`final-grid --from results/<models-run>` uses those two lists. You can
`--from` the stage-1 run and pass `--models` by hand if you skip a clean
stage 2.

Editing `winners.json` between stages is the pause-gate. The file is JSON
the user is expected to touch:

```json
{
  "stage": "screen-arms",
  "ranking": "wcr",
  "arms": ["a3", "a5", "a4"],
  "models": ["Qwen/Qwen3-30B-A3B-Instruct-2507"]
}
```

### Arm × model wiring

For each `(model, arm)` cell, `build_arms(model, repair_model, ensemble_model)`
runs as today, then:

- Arms other than `a5`: `config.model = model`; `repair_model` stays the
  CLI `--repair-model` (default 235B).
- Arm `a5`: `config.model = config.repair_model = model`.
- Arm `a4`: `ensemble_model` stays `--ensemble-model`.

That is why a4/a5 belong in the strategy screen with a fixed cheap
primary model, and why the model screen should use a single non-a5 arm
when possible. If the best stage-1 arm *is* a5, the model screen varies
the model used everywhere — that is correct for a5.

### Makefile

```
make screen-arms
make screen-models FROM=results/<run>
make final-grid FROM=results/<run>
```

Each is the corresponding `crossword eval --recipe ...`. Live, needs
`NEBIUS_API_KEY`. Oracle still works for plumbing tests:
`--backend oracle --recipe screen-arms`.

---

## 6. Harness changes

`Harness.run` today loops `puzzle × arm × seed × prefill`. Add an outer
(or inner) `models` list. `client_factory` already receives `(puzzle, arm,
seed)`; pass `model` too so the Nebius client can label traces.

Write a `cells.jsonl` line as soon as `score_solution` (or the except
block) returns, then call `progress`.

`RunRecord` gains `model: str` and `error: str | None`. `as_dict` includes
them. `results.json` `arms` metadata stays; add a top-level `models` list.

Do not change the oracle client’s scoring path. Oracle cells still get
`cost_usd: null` (no real rates applied to fake tokens), shown as `?`.

---

## 7. Tests (all offline)

No live tokens in `make test`.

| test | assert |
|---|---|
| `test_cost_known_model` | 1M prompt + 1M completion of Qwen 30B equals the table’s two rates summed. |
| `test_cost_unknown_model` | returns `None`. |
| `test_cost_two_models` | a4-style usage (primary + ensemble) sums both. |
| `test_usage_by_model_splits_prompt_completion` | `record()` stores both counters. Update `tests/test_client.py` which currently asserts `by_model` as `{name: total_int}`. |
| `test_recipe_screen_arms_flags` | parser expansion matches the table in §5. |
| `test_recipe_screen_models_requires_from` | exit 2 with no `--from` / `--arms`. |
| `test_winners_top3_by_wcr` | three arms, ranking order, tie-break on cost. |
| `test_grid_columns` | `summarize()` contains the header row in §4. |
| `test_resume_skips_jsonl_keys` | second `Harness.run` with the same run-id does not re-solve a completed key (oracle, 2×2 matrix, spy client). |
| `test_failed_cell_written` | a raising client still appends a jsonl line with `error`. |
| `test_retry_errors_re_runs` | `--retry-errors` re-solves the failed key only. |

Existing `test_matrix_runs_and_writes_results` must still pass. If
`RunRecord` / `results.json` shape grows, update the minimal payload in
`test_summarize_handles_a_minimal_payload` so old keys still render.

---

## 8. Files

| File | Role |
|---|---|
| `crossword/eval/pricing.py` | Rate table + `cost_usd(usage) -> float \| None`. |
| `crossword/client.py` | `Usage.by_model` becomes per-model prompt/completion. |
| `crossword/agent/solver.py` | `SolveResult.as_dict` includes `cost_usd`. |
| `crossword/eval/harness.py` | Model axis, jsonl append, resume skip, error cells. |
| `crossword/eval/report.py` | Results grid + leaderboard; `write_winners`. |
| `crossword/eval/recipes.py` | Recipe → flag expansion; `--from` merge. Keep it out of `cli.py` so tests do not parse argparse. |
| `crossword/cli.py` | New flags, `--recipe` / `--from` / `--retry-errors`, pass models into the harness. |
| `Makefile` | `screen-arms`, `screen-models`, `final-grid`. |
| `tests/test_pricing.py` | Cost arithmetic. |
| `tests/test_cli.py` | Recipe flags, summarize grid, winners. |
| `tests/test_harness.py` | Resume, failed cell, retry (new file; harness tests today live in `test_cli.py`). |
| `README.md` | Short “Live tournament” paragraph pointing at the three make targets. |

Do not rewrite `REPORT.md`’s §5 (“what a live run should report”) beyond
one sentence that the tournament recipes are how that matrix is produced
in stages.

---

## 9. Expected spend shape (not a test)

Rough, live, one seed, current default timeouts:

- Stage 1: 7 arms × 1 7×7. a0 is one fat prompt; a5 is the expensive cell;
  a4 is two model families in round 0.
- Stage 2: 7 models × 1 arm × 1 7×7.
- Stage 3: 3 models × 3 arms × 3 sizes = 27 cells. The 11×11 × a5-like
  cells dominate.

If a stage-1 7×7 is a lottery ticket, that is accepted: screens eliminate
obvious losers; the 27-cell grid is what you quote. Do not add extra
puzzles to stages 1–2 in this pass.
