# Staged Eval Tournament Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pause-gated live eval: per-cell WCR/LCR/ICR + tokens/USD/turns/calls/time, three named recipes, jsonl resume.

**Architecture:** Keep `Harness.run` as the only solver loop. Add pricing, a model axis, `cells.jsonl` resume, recipe expansion, and a results-grid section in `summary.md`. Do not change `client_factory(puzzle, arm, seed)` arity — stamp the cell's model onto `Arm.config` instead.

**Tech Stack:** Python 3 stdlib, existing `crossword eval` harness, unittest.

**Spec:** `docs/superpowers/specs/2026-09-01-eval-tournament-design.md`

## Global Constraints

- No new dependencies. `make test` stays offline (oracle only).
- Rank by WCR; cost is a reported column and a tie-break only.
- Existing `crossword eval` with no new flags behaves as today.
- Work only in the `feat/eval-tournament` worktree. Do not edit `web-corpus-ui`.
- Oracle cells have `cost_usd: null` (shown as `?`). Never invent a rate for unknown models.

## File map

- Create: `crossword/eval/pricing.py` — rates + `cost_usd`
- Create: `crossword/eval/recipes.py` — recipe expansion, ranking, winners IO
- Create: `tests/test_pricing.py`
- Create: `tests/test_recipes.py`
- Create: `tests/test_harness.py`
- Modify: `crossword/client.py` — `Usage.by_model` prompt/completion split
- Modify: `crossword/agent/solver.py` — `cost_usd` on `SolveResult.as_dict`
- Modify: `crossword/eval/harness.py` — model axis, jsonl, resume, errors
- Modify: `crossword/eval/report.py` — grid, leaderboard, `write_winners`
- Modify: `crossword/cli.py` — new flags, recipes, puzzle filter
- Modify: `Makefile`, `README.md`, `tests/test_client.py`, `tests/test_cli.py`

---

### Task 1: Pricing and per-model prompt/completion usage

**Files:**
- Create: `crossword/eval/pricing.py`
- Create: `tests/test_pricing.py`
- Modify: `crossword/client.py`
- Modify: `tests/test_client.py`
- Modify: `crossword/agent/solver.py` (`SolveResult.as_dict`)

**Interfaces:**
- Consumes: `Usage` with `by_model: dict[str, dict[str, int]]` (`prompt`, `completion`)
- Produces: `RATES: dict[str, tuple[float, float]]`, `cost_usd(usage) -> float | None`

- [ ] **Step 1: Write failing tests** in `tests/test_pricing.py` and update `tests/test_client.py::test_accumulates_per_model` to expect `{"a": {"prompt": 11, "completion": 6}, "b": {"prompt": 2, "completion": 2}}`.

```python
import unittest
from crossword.client import Completion, Usage
from crossword.eval.pricing import cost_usd, RATES

QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class TestCostUsd(unittest.TestCase):
    def test_known_model_one_million_each(self):
        usage = Usage()
        usage.record(Completion(text="", model=QWEN, prompt_tokens=1_000_000, completion_tokens=1_000_000))
        prompt_rate, completion_rate = RATES[QWEN]
        self.assertAlmostEqual(cost_usd(usage), prompt_rate + completion_rate)

    def test_unknown_model_is_none(self):
        usage = Usage()
        usage.record(Completion(text="", model="no-such/model", prompt_tokens=10, completion_tokens=10))
        self.assertIsNone(cost_usd(usage))

    def test_two_models_sum(self):
        usage = Usage()
        usage.record(Completion(text="", model=QWEN, prompt_tokens=1_000_000, completion_tokens=0))
        llama = "meta-llama/Llama-3.3-70B-Instruct"
        usage.record(Completion(text="", model=llama, prompt_tokens=0, completion_tokens=1_000_000))
        self.assertAlmostEqual(cost_usd(usage), RATES[QWEN][0] + RATES[llama][1])
```

- [ ] **Step 2: Run tests, confirm they fail** because `pricing` is missing and `by_model` is still an int.

Run: `python3 -m unittest tests.test_pricing tests.TestUsage.test_accumulates_per_model -v`

- [ ] **Step 3: Implement** `pricing.py` with the spec rate table (2026-08-28). Change `Usage.record` to accumulate prompt/completion per model. `cost_usd` returns `None` if any model is missing from `RATES`. Add `cost_usd` to `SolveResult.as_dict` via `cost_usd(self.usage)`.

- [ ] **Step 4: Re-run tests, confirm pass.** Then `python3 -m unittest discover -s tests -t . -q`

---

### Task 2: Recipe expansion and WCR ranking

**Files:**
- Create: `crossword/eval/recipes.py`
- Create: `tests/test_recipes.py`

**Interfaces:**
- Produces:
  - `RecipeError(ValueError)`
  - `EvalSpec(models, arms, puzzle_ids, stage)`
  - `SCREEN_PUZZLE = "mini-07-00-0"`
  - `FINAL_PUZZLES = ("mini-07-00-0", "mini-09-00-0", "mini-11-04-0")`
  - `expand_recipe(name, *, models=None, arms=None, puzzle_ids=None, winners=None) -> EvalSpec`
  - `rank_keys(records, key) -> list[str]` mean WCR desc, cost_usd asc (None last), name asc; skip failed scores
  - `winners_payload(stage, arms, models) -> dict`
  - `load_winners(path) -> dict`
  - `write_winners(directory, payload)`

User-supplied models/arms/puzzle_ids win over recipe defaults. `screen-models` / `final-grid` raise `RecipeError` if the axis that should come from winners is empty and not overridden.

- [ ] **Step 1: Write failing tests** covering screen-arms defaults, screen-models requiring winners or `--arms`, final-grid using top-3 lists, user override, and rank order + cost tie-break.

- [ ] **Step 2: Run, confirm fail.** `python3 -m unittest tests.test_recipes -v`

- [ ] **Step 3: Implement `recipes.py`.** `screen-models` uses `winners["arms"][0]` as the sole arm unless `arms` was passed. It still copies all `winners["arms"]` into the *output* winners file (that is `write_winners`'s job in Task 3; `expand_recipe` only chooses what to *run*).

- [ ] **Step 4: Tests pass.**

---

### Task 3: Results grid, leaderboard, winners.json

**Files:**
- Modify: `crossword/eval/report.py`
- Modify: `tests/test_cli.py` (`test_summarize_handles_a_minimal_payload`)

**Interfaces:**
- Produces: `summarize` includes `## Results grid` with header `size | puzzle | model | arm | WCR | LCR | ICR | exact | tokens | USD | turns | calls | sec` and `## Leaderboard`. `write_summary` also writes `winners.json`. `write_winners(directory, payload)`. Ranking group: `payload.get("rank_by", "arm")`.

- [ ] **Step 1: Extend the minimal payload test** so a record with `model` and `solve.cost_usd` / `rounds` produces those two section headers and `0.400` USD for a known-rate cell. Assert `winners.json` after `write_summary`.

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement grid + leaderboard + winners write.** Failed cells: `err` in score columns. `USD` is `?` when `cost_usd` is null. Short model names in the table are the id after `/` if present, full id in JSON.

- [ ] **Step 4: Tests pass.** Existing summarize tests still pass.

---

### Task 4: Harness model axis, jsonl resume, error cells

**Files:**
- Modify: `crossword/eval/harness.py`
- Create: `tests/test_harness.py`

**Interfaces:**
- `RunRecord` gains `model: str` and `error: str | None`
- `Harness.run(..., models=None, retry_errors=False)`
- `models` default: unique models already on the arm configs, else `[DEFAULT_MODEL]`
- For each cell, `replace` the arm config: non-a5 `model=cell_model`; a5 `model=repair_model=cell_model`. Factory arity stays `(puzzle, arm, seed)`.
- Append `cells.jsonl` immediately after each cell. Skip keys already present unless `retry_errors` and the line has `error`.
- Exceptions during solve become a record with `error`, empty solution, scores still computed (likely zeros) — spec says scores null. Prefer `scores=None` on the record and `as_dict` emitting `"scores": null`.
- Rebuild `results.json` from in-memory records plus any jsonl-loaded prior cells.

- [ ] **Step 1: Failing tests** with a counting fake client: 2 models × 1 arm × 1 puzzle, second run same `run-id` does not increment the counter; a raising client writes a jsonl error line; `retry_errors=True` increments the counter for that key only.

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement.** Wrap solve in try/except. Load jsonl at start. Keep progress callback.

- [ ] **Step 4: Tests pass.** Also `test_matrix_runs_and_writes_results`.

---

### Task 5: CLI, Makefile, README

**Files:**
- Modify: `crossword/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `REPORT.md` (one sentence in §5)

**Interfaces:**
- Eval flags: `--models`, `--puzzles`, `--recipe`, `--from`, `--retry-errors`
- `cmd_eval` calls `expand_recipe` when `--recipe` is set; filters puzzles by id; passes `models` into `Harness.run`; `write_summary` writes winners.
- `--from` required for screen-models/final-grid unless overrides exist; `RecipeError` → exit 2 with the spec message.
- Progress line includes model.
- Makefile: `screen-arms`, `screen-models FROM=...`, `final-grid FROM=...`

- [ ] **Step 1: Failing CLI tests** for recipe flag expansion (parse + a dry helper, or invoke `expand_recipe` from the same path `cmd_eval` uses), screen-models without `--from` returns 2, `--puzzles mini-07-00-0` with oracle `--limit` not needed because puzzle filter wins, `--recipe screen-arms --backend oracle` writes winners.

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement CLI + make targets + README “Live tournament” paragraph.**

- [ ] **Step 4: Full `python3 -m unittest discover -s tests -t . -q` green.

---

## Notes for the executor

- Worktree: `/Users/carterprice/repos/crossword-solver-eval-tournament` on `feat/eval-tournament`, branched from `main` @ 735b213. The Cursor workspace on `web-corpus-ui` is a different agent; do not write there.
- TDD: no production code without a failing test first.
- Do not run live Nebius evals as part of this plan.
