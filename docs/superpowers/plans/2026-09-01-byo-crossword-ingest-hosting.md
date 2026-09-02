# BYO Crossword Ingest + Nebius Hosting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a visitor submit a grid screenshot plus Across/Down clues (or paste `.xd`), auto-solve when the mask matches, and ship a Docker Compose recipe for a small Nebius CPU VM in front of Token Factory.

**Architecture:** Pure functions in `crossword/ingest.py` turn rows + clue text into a `Puzzle` (no gold) and optional prefill. FastAPI stores drafts in memory, calls a vision model through the existing `NebiusClient.complete` + degradation ladder, and reuses `JobStore` / SSE. The SPA adds a **Your puzzle** tab. Hosting is Caddy + uvicorn on one VM; inference stays on Token Factory.

**Tech Stack:** Python 3.11+, FastAPI, existing `Grid`/`Puzzle`/`parse_xd`, Vite/React, Docker Compose, Caddy.

**Spec:** `docs/superpowers/specs/2026-09-01-byo-crossword-ingest-hosting-design.md`

## Global Constraints

- Python ≥ 3.11; no new required dependencies; web extra stays FastAPI + uvicorn + httpx.
- Max image 4 MiB png/jpeg; max grid 15×15; 5 ingest-or-solve starts per IP per hour; `DAILY_SOLVE_CAP` default 40.
- Vision default `Qwen/Qwen2.5-VL-72B-Instruct`, override `NEBIUS_VISION_MODEL`.
- Tab label **Your puzzle**. 503 copy: `This host has no Token Factory key.` 429 copy: `This host is rate-limited. Try again later.`
- `make test` stays offline. Do not bake `NEBIUS_API_KEY` into images.
- Letters in the vision mask are prefill, never `Slot.gold`, unless `.xd` included answers.
- BYO auto-starts a Nebius solve on match. Oracle on a no-gold puzzle is HTTP 400.

## File map

- Create: `crossword/ingest.py` — rows, clues, match/mismatch, no HTTP.
- Create: `crossword/api/drafts.py` — in-memory `upload-<id>` store.
- Create: `crossword/api/limits.py` — IP hour cap + daily cap.
- Create: `web/src/components/Ingest.tsx` — form + block editor.
- Create: `deploy/Dockerfile`, `deploy/docker-compose.yml`, `deploy/Caddyfile`.
- Create: `docs/hosting.md`.
- Create: `tests/test_ingest.py`.
- Modify: `crossword/schemas.py` — `grid_schema`, generalize `response_format_for`.
- Modify: `crossword/client.py` — `DEFAULT_VISION_MODEL`.
- Modify: `crossword/agent/candidates.py` — keep ladder; ingest reuses `request_with_ladder` with a schema factory or a small `request_with_ladder` generalization.
- Modify: `crossword/run.py` — draft-aware `find_puzzle`.
- Modify: `crossword/api/app.py` — ingest endpoints, health, limits, prefill on solve.
- Modify: `web/src/*` — types, api, App, Picker, CSS.
- Modify: `tests/test_api.py`, `tests/test_schemas.py`, `README.md`.

---

### Task 1: Ingest core (rows + clues → Puzzle)

**Files:**
- Create: `crossword/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `crossword.model.Grid`, `Puzzle`, `Slot`, `PuzzleError`; `crossword.xd.parse_xd`
- Produces:
  - `MAX_SIZE = 15`
  - `IngestError(ValueError)`
  - `ClueLine(number: int | None, direction: str | None, text: str)`
  - `parse_clue_list(text: str, *, direction: str) -> list[tuple[str, str]]`  
    returns `(slot_id, clue)` in order, slot_id like `A1`
  - `rows_to_grid(rows: list[str]) -> tuple[Grid, dict[tuple[int,int], str]]`  
    second value is prefill letters
  - `assemble(rows, across_text, down_text, *, puzzle_id, title="") -> IngestDraft`
  - `IngestDraft` dataclass: `status: str` (`ready`|`needs_edit`), `puzzle: Puzzle | None`, `prefill: dict`, `rows: list[str]`, `across_slots: int`, `down_slots: int`, `across_clues: int`, `down_clues: int`, `unknown_numbers: list[str]`, `message: str`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
from crossword.ingest import assemble, parse_clue_list, rows_to_grid
from crossword.xd import parse_xd

MINI = [
    "C.T",
    ".#.",
    "T.G",
]


class TestParseClues(unittest.TestCase):
    def test_numbered_across(self):
        pairs = parse_clue_list("1. Feline\n3. Canine", direction="A")
        self.assertEqual(pairs, [("A1", "Feline"), ("A3", "Canine")])

    def test_prefixed_direction(self):
        pairs = parse_clue_list("D1. Vertical\nD2. Other", direction="A")
        self.assertEqual(pairs, [("D1", "Vertical"), ("D2", "Other")])

    def test_sequential_when_unnumbered(self):
        pairs = parse_clue_list("Feline\nCanine", direction="A")
        self.assertEqual([p[1] for p in pairs], ["Feline", "Canine"])
        self.assertEqual([p[0] for p in pairs], ["A1", "A2"])


class TestRows(unittest.TestCase):
    def test_letters_are_prefill_not_gold(self):
        grid, prefill = rows_to_grid(MINI)
        self.assertEqual(grid.height, 3)
        self.assertEqual(grid.width, 3)
        self.assertIn((1, 1), grid.blocks)
        self.assertEqual(prefill[(0, 0)], "C")
        self.assertNotIn((0, 1), prefill)

    def test_rejects_non_rectangular(self):
        with self.assertRaises(Exception):
            rows_to_grid(["..", "..."])


class TestAssemble(unittest.TestCase):
    def test_ready_when_counts_match(self):
        across = "1. C to T\n3. T to G"
        down = "1. C to T\n2. T to G"
        draft = assemble(MINI, across, down, puzzle_id="upload-ab")
        self.assertEqual(draft.status, "ready")
        self.assertIsNotNone(draft.puzzle)
        self.assertFalse(draft.puzzle.has_gold())
        self.assertEqual(draft.puzzle.slot("A1").clue, "C to T")
        self.assertEqual(draft.prefill[(0, 0)], "C")

    def test_needs_edit_on_count_mismatch(self):
        draft = assemble(MINI, "1. only one", "1. a\n2. b", puzzle_id="x")
        self.assertEqual(draft.status, "needs_edit")
        self.assertIsNone(draft.puzzle)

    def test_unknown_number(self):
        draft = assemble(
            MINI, "1. a\n9. missing", "1. a\n2. b", puzzle_id="x"
        )
        self.assertEqual(draft.status, "needs_edit")
        self.assertIn("A9", draft.unknown_numbers)

    def test_xd_without_gold(self):
        text = "Title: X\n\n\n...\n.#.\n...\n\n\nA1. One\nA3. Two\nD1. Three\nD2. Four\n"
        puzzle = parse_xd(text, puzzle_id="xd-1")
        self.assertFalse(puzzle.has_gold())
```

Adjust the mini grid if `grid.slots()` numbering on that mask is not 1/3 across and 1/2 down — print `grid.slots()` once and lock the fixture to whatever numbering it actually produces.

- [ ] **Step 2: Run tests, expect fail** (module missing)

Run: `python3 -m unittest tests.test_ingest -v`

- [ ] **Step 3: Implement `crossword/ingest.py`**

`rows_to_grid`: strip, uppercase, `#` → block, `.` → empty, A–Z → prefill. Reject empty, non-rect, size outside 2–15, chars other than `#.A-Z`. Isolated cells → `IngestError` or `needs_edit` in `assemble`.

`parse_clue_list`: compile `r"^\s*([ADad])?\s*(\d+)[.)]\s+(.*)$"`. If any line matches, numbered mode (blank lines skipped; unmatched non-blank → `IngestError`). Sequential mode assigns `A1`, `A2`, … in list order (slot ids are rewritten in `assemble` when sequential: zip against actual slot numbers).

Sequential rewrite in `assemble`: if parse returned placeholder ids `A1..An`, rebind to real across slot ids in numbering order.

- [ ] **Step 4: Run tests until pass**

- [ ] **Step 5: Commit** (only if the user asked for commits)

---

### Task 2: Grid schema + ladder

**Files:**
- Modify: `crossword/schemas.py`
- Modify: `crossword/agent/candidates.py` (`request_with_ladder` accepts `schema_for=candidates_schema` default)
- Test: `tests/test_schemas.py`

**Produces:**
- `grid_schema(*, strict=True, constrained=True) -> dict`
- `response_format_for(rung, schema_fn=candidates_schema)`
- `parse_grid_rows(text: str) -> list[str]` using `extract_json`, expecting `{"rows": [...]}`

`request_with_ladder` signature add `schema_fn=candidates_schema`.

- [ ] Failing test: `grid_schema()["json_schema"]["schema"]["properties"]["rows"]["type"] == "array"` and `parse_grid_rows` round-trips `{"rows": ["#.#", ".#.", "#.#"]}`.
- [ ] Implement and pass `python3 -m unittest tests.test_schemas tests.test_ingest -v`

---

### Task 3: Draft store, limits, health, ingest API (scripted vision)

**Files:**
- Create: `crossword/api/drafts.py`, `crossword/api/limits.py`
- Modify: `crossword/api/app.py`, `crossword/run.py` (`find_puzzle(puzzle_id, drafts=None)`)
- Test: `tests/test_api.py`, `tests/test_ingest.py` (xd path via API)

**Produces:**
- `DraftStore.put(draft) -> id` (`upload-` + 12 hex), `.get(id)`, `.update_rows(id, rows)`
- `RateLimiter(hourly=5, daily=40)` with `check(ip, now=None)` / `hit(ip, now=None)` raising `LimitError`
- `POST /api/ingest`, `POST /api/ingest/{id}/grid`, `GET /api/health`
- App-level injectable `vision_complete` callable for tests: `(image_bytes, mime) -> list[str]` of rows. Production uses ladder + NebiusClient.
- On `ready`: `store.begin`, thread `_run_job` with `prefill=draft.prefill`, backend forced `nebius`.
- No key → 503. Oracle + no gold → 400.

Test pattern: `create_app(vision=lambda *_: MINI_ROWS)` so ingest never networks.

- [ ] Write API tests for ready (returns `job_id`), needs_edit, grid fix, xd paste, 503 without key (patch env), 429 after 5 hits, finished gold empty.
- [ ] Implement until `python3 -m unittest tests.test_api tests.test_ingest -v` passes.
- For 503: temporarily clear `NEBIUS_API_KEY` in the test, or pass `has_key=False` into `create_app`.

**Vision production path** (same task): `crossword/ingest.py` `decode_image(image: str) -> tuple[bytes, str]` (data URL or raw base64; sniff png/jpeg magic; size ≤ 4 MiB). `vision_rows(client, image_bytes, mime, model)` builds messages and calls `request_with_ladder(..., schema_fn=grid_schema)` then `parse_grid_rows`.

---

### Task 4: Web UI — Your puzzle tab

**Files:**
- Create: `web/src/components/Ingest.tsx`
- Modify: `web/src/types.ts`, `web/src/api.ts`, `web/src/App.tsx`, `web/src/components/Picker.tsx`, `web/src/index.css`

**Produces:**
- `ingestPuzzle({ image?, across, down, title?, xd? })` and `fixIngestGrid(id, rows)`
- Picker grows a local tab `yours` that does not call `fetchPuzzles`. When selected, render `Ingest` instead of chips.
- On `ready`: set puzzle from payload, set selected to `puzzle.id`, attach EventSource to `job_id` (extract stream handler from `onSolve` so ingest can reuse it).
- On `needs_edit`: click cells to toggle `#`/`.`, keep letters, Submit grid.
- Hide gold board and Scorecard when `!puzzle.has_gold`. Hide HIT/MISS columns in CandidateDebug when no gold.
- Disable Oracle while on Your puzzle (or ignore it and always send nebius).
- File input accept `image/png,image/jpeg`. Read as data URL.

- [ ] Typecheck: `cd web && npx tsc --noEmit`
- [ ] Manual: `make serve-dev` + Vite; paste a tiny `.xd` without gold; confirm solve streams (oracle blocked; use nebius or skip live). Offline check: `.xd` with gold on oracle still works from Mini tab.

---

### Task 5: Docker + hosting doc

**Files:**
- Create: `deploy/Dockerfile`, `deploy/docker-compose.yml`, `deploy/Caddyfile`, `docs/hosting.md`
- Modify: `README.md` (Your puzzle + hosting pointer), `.dockerignore`

Dockerfile: stage `node:22-alpine` `npm ci && npm run build` in `/web`; stage `python:3.12-slim` `pip install -e '.[web]'`, copy `crossword/`, `corpus/`, `web/dist`, `CMD uvicorn crossword.api.app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 75`.

Compose: `app` (env_file `/etc/crossword.env` documented as bind `./crossword.env`), `caddy:2` with ports 80/443, volume for certs. Caddyfile reverse_proxy `app:8000`, optional `{$CROSSWORD_HOST}`.

`.dockerignore`: `.venv`, `node_modules`, `.env`, `corpus/nyt`.

`docs/hosting.md`: cpu-e2 2vcpu-8gb, ports, docker install, clone, `NEBIUS_API_KEY`, compose up, DNS.

- [ ] `docker build -f deploy/Dockerfile .` locally if Docker is available; otherwise document and skip.

---

### Task 6: README + Makefile health

- Mention Your puzzle, ingest limits, `docs/hosting.md`.
- `GET /api/health` in hosting doc.
- Do not increase the “no dependencies” claim; Docker is optional ops.

---

## Execution

Implement tasks in order. User asked to roll immediately — run inline in this session (executing-plans style): TDD per task, no commits unless asked.
