# BYO crossword ingest and Nebius hosting

Date: 2026-09-01
Status: approved (chat design, 2026-09-01)

A visitor can bring a crossword the corpus does not contain, watch the existing
agent solve it, and we can put that page on a small Nebius CPU VM in front of
Token Factory.

## Decisions already made

- Ingest: screenshot of the **grid** plus typed Across and Down clues. `.xd`
  paste is a small escape hatch.
- After vision parse: **solve immediately** when across/down slot counts match
  the clue lists. Show a click-to-toggle block editor only on mismatch.
- Hosting: public demo on one Nebius `cpu-e2` VM, Docker Compose (app + Caddy),
  rate-limited. Inference stays on Token Factory. No Kubernetes, no GPU VM.
- Oracle cannot score a puzzle with no gold. BYO solves are Nebius-only.

## Out of scope

Clue OCR from the photo, rebus, `.puz`, a free-draw grid from scratch, baking
`NEBIUS_API_KEY` into an image, creating the Nebius VM from CI, and a live
vision test in `make test`.

---

## 1. Ingest

The solver, tracer, and SSE loop stay as they are. New work is
photo + clues → `Puzzle` with no gold (plus optional prefill letters), then
the current solve path.

### Happy path

1. New picker tab **Your puzzle** next to Mini / NYT.
2. Drop or paste a grid screenshot (png/jpeg, max 4 MiB). Two textareas:
   Across and Down. Optional title. Optional `.xd` paste that skips vision.
3. `POST /api/ingest` with JSON `{ image, across, down, title? }` or `{ xd }`.
   `image` is a data URL or raw base64. The bytes stay in memory for that
   request only.
4. A Token Factory vision model returns guided JSON: `rows` of equal length,
   `#` = block, `.` = empty, `A–Z` = letter already in the photo.
5. Throw away any numbering the model invents. `Grid` + `grid.slots()` number
   the puzzle the same way corpus files do.
6. Clue lists: numbered (`1. …`, `A1. …`, `D14. …`) when that shape is present,
   otherwise one clue per line in slot order. Match by number onto `A1` / `D1`.
7. If across count matches across slots **and** down matches down, and every
   numbered clue exists on the grid: store the puzzle in an in-memory draft
   store (`upload-<12 hex>`), treat photo letters as solver **prefill** (locked
   cells, not gold), and **start a Nebius solve immediately**. Same EventSource
   UI as corpus solves.
8. If counts disagree or a clue number is unknown: HTTP 200 with
   `status: "needs_edit"`, the parsed mask, slot counts, clue counts, and
   unknown numbers. The page shows a block editor. `POST /api/ingest/{id}/grid`
   retries the match. Solve starts only when it matches.

### `.xd` hatch

`parse_xd` already exists. Gold present → scoring works. Gold absent → same
no-gold path as a screenshot.

### What the UI hides on no-gold puzzles

No answer-key board, no WCR/LCR, no HIT/MISS. ICR, live fill, and the event
rail still show. Debug candidates may list offers without gold/hit columns.

### Files

| File | Role |
|---|---|
| `crossword/ingest.py` | Rows → `Grid`/`Puzzle`, clue parse, match vs mismatch. No HTTP. |
| `crossword/schemas.py` | `grid_schema()` for vision JSON. |
| `crossword/client.py` | `DEFAULT_VISION_MODEL`. `complete()` already accepts multimodal messages. |
| `crossword/api/drafts.py` | In-memory draft store (puzzle + prefill + rows). |
| `crossword/api/limits.py` | Per-IP hour cap and process-wide daily cap. |
| `crossword/api/app.py` | `POST /api/ingest`, `POST /api/ingest/{id}/grid`, draft lookup on solve. |
| `crossword/run.py` | `find_puzzle` also looks in the draft store; `run_solve` already takes `prefill`. |
| `web/src/components/Ingest.tsx` | Form + mismatch editor. |
| `web/src/components/Picker.tsx` | **Your puzzle** tab. |
| `tests/test_ingest.py`, `tests/test_api.py` | Offline coverage. |

### Vision

- Default model: `Qwen/Qwen2.5-VL-72B-Instruct`, override with
  `NEBIUS_VISION_MODEL`.
- Message content: text prompt + `image_url` data URL.
- Schema: `{ "rows": ["#..#", "..."] }` with `minItems`/`maxItems` 2–15 and
  equal-length rows validated in Python, not only in the schema.
- Same degradation ladder as candidates (`request_with_ladder` or a thin
  sibling that takes `grid_schema` instead of `candidates_schema`).
- A mask that still will not parse after the ladder is HTTP 400, not a hung
  spinner.

### Clue parse rules

1. Split on newlines; drop blank lines.
2. If any line matches `^\s*([AD])?\s*(\d+)[.)]\s+(.*)$` (case-insensitive),
   treat the list as numbered. Direction defaults to the textarea (Across → A,
   Down → D) when omitted. Duplicate numbers in one list are an error →
   `needs_edit`.
3. Otherwise sequential: line `i` maps to across/down slot `i` in numbering
   order.
4. Match succeeds only when `len(across clues) == across slot count` and
   `len(down clues) == down slot count`, and every numbered id exists.

### Prefill vs gold

Letters in the vision mask become `prefill: dict[Cell, str]` passed into
`Solver.solve(..., prefill=...)`. They are **not** written onto `Slot.gold`.
`puzzle.has_gold()` is false unless `.xd` included answers.

### Caps

- Image: png or jpeg, ≤ 4 MiB decoded.
- Grid: 2–15 inclusive on both axes, rectangular.
- Isolated open cells (`grid.isolated_cells()`) → `needs_edit` (broken mask).

---

## 2. API

### `POST /api/ingest`

Body:

```json
{
  "image": "data:image/png;base64,...",
  "across": "1. ...\n5. ...",
  "down": "1. ...",
  "title": "optional"
}
```

or `{ "xd": "<full .xd text>" }`. Image and xd are mutually exclusive.
Across/down required with image, ignored with xd.

Responses:

- `200 { "status": "ready", "puzzle": <serialize_puzzle>, "job_id": "...", "backend": "nebius", ... }`
  when the match succeeds. A solve job has already started. The client
  opens `/api/solves/{job_id}/events` exactly as today.
- `200 { "status": "needs_edit", "draft_id": "...", "rows": [...], "height", "width", "across_slots", "down_slots", "across_clues", "down_clues", "unknown_numbers": [], "message": "..." }`
- `400` bad image / bad xd / non-rectangular / >15
- `503` no `NEBIUS_API_KEY` (vision and BYO live solve both need it)
- `409` a solve is already running
- `429` rate limit

### `POST /api/ingest/{draft_id}/grid`

Body: `{ "rows": ["#..", "..."] }` plus optional across/down to replace clues.
Rebuilds the grid, retries the match, same response shapes as ingest
(including auto-start on `ready`).

### `GET /api/puzzles/{id}`

Unchanged for corpus ids. Draft ids (`upload-…`) resolve from the draft store.

### `POST /api/solves`

Unchanged shape. `find_puzzle` checks drafts. If the puzzle has no gold and
`backend` is `oracle` → `400` with a message that oracle needs an answer key.

### `GET /api/health`

`{ "ok": true, "has_key": bool, "busy": bool }`. No auth.

### Rate limits

- Existing: one active solve (`JobStore`).
- 5 ingest-or-solve **starts** per client IP per hour.
- Env `DAILY_SOLVE_CAP` (default 40): process-wide ingest+solve starts per UTC
  day, then 429 until midnight UTC.
- Client IP: first `X-Forwarded-For` hop if present (Caddy), else
  `request.client.host`.
- 429 includes `Retry-After` when the limiter can compute it.

Ingest that returns `needs_edit` still counts as a start (vision already spent
tokens). Grid-fix retries that do not call vision do **not** count against the
hourly cap; they still require the store not to be busy if they auto-start a
solve.

---

## 3. Hosting

Inference stays on Token Factory. The web app is a CPU box in the same Nebius
project.

```
browser → Caddy :443 → uvicorn :8000 (FastAPI + web/dist) → Token Factory
```

- Machine: standalone VM, platform `cpu-e2`, preset `2vcpu-8gb`, Ubuntu,
  public IPv4, ports 22 / 80 / 443. ~20 GiB disk. Region of the existing
  project (typically `eu-north1`).
- Not Container VM (secret + TLS sidecar). Not mk8s.
- Multi-stage `Dockerfile`: Node builds `web/dist`, Python installs
  `.[web]`, copies dist, runs
  `uvicorn crossword.api.app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 75`.
- `deploy/docker-compose.yml`: `app` + `caddy`. Caddy uses
  `CROSSWORD_HOST` for Let’s Encrypt. First bring-up may be HTTP on the raw
  IP if there is no domain yet.
- `NEBIUS_API_KEY` only as runtime env (`/etc/crossword.env`, mode 600).
  Never in the image or git.
- Optional later: GHCR push. v1 is `docker compose build` on the VM from git.
- Caddy and uvicorn idle timeouts high enough for a 15×15 live solve
  (minutes). Existing SSE keepalive stays.
- CORS: localhost Vite origins remain for local dev. Hosted SPA is
  same-origin through Caddy.

`docs/hosting.md` is the operator checklist: create VM, open ports, install
Docker, clone, env file, compose up, DNS A record.

---

## 4. Errors the user sees

| Failure | Status | UI |
|---|---|---|
| No Token Factory key | 503 | “This host has no Token Factory key.” |
| Image too big / not png|jpeg / bad mask | 400 | Stay on the form. |
| Slot/clue count mismatch or unknown numbers | 200 `needs_edit` | Block editor; Solve disabled until match. |
| `.xd` parse error | 400 | Form error. |
| Oracle on a no-gold upload | 400 | Force Nebius. |
| Solve already running | 409 | Existing busy copy. |
| Hour or daily cap | 429 | “Try later”. |
| Vision or solve throws mid-job | SSE `error` | Same red log as today. |

---

## 5. Tests

`make test` stays offline. No live Token Factory in CI.

- `tests/test_ingest.py`: numbered vs sequential clues; rows → numbering;
  letters in the mask become prefill not gold; mismatch vs match; unknown
  numbers; isolated cells; `.xd` with and without gold.
- `tests/test_api.py`: ingest with a **scripted** vision client (injected
  fixture rows); `needs_edit` then grid-fix → solve start; `.xd` paste; 400 /
  503 / 429; BYO `finished` has empty `gold` and no scores; health endpoint.
- Rate-limit tests use an injected clock, no `sleep`.
- Optional later: `make verify-live-ingest` is **not** part of this spec.

---

## 6. Copy and constraints

- Python ≥ 3.11, no new required dependencies. Web extra stays FastAPI +
  uvicorn + httpx.
- Vision model id is an env override, not a UI dropdown in this slice.
- Tab label: **Your puzzle**.
- 429 message: “This host is rate-limited. Try again later.”
- 503 message: “This host has no Token Factory key.”
- Max image 4 MiB, max grid 15×15, 5 starts/IP/hour, `DAILY_SOLVE_CAP=40`.
