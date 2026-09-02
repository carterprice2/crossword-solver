"""FastAPI app: corpus listing, BYO ingest, and a streaming solve.

Imported only by the serve command and by tests that skip when FastAPI is
missing. The rest of the package does not import this module.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agent.candidates import request_with_ladder
from ..agent.trace import CANDIDATES, Tracer
from ..client import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_MODEL,
    KNOWN_MODELS,
    NebiusClient,
)
from ..eval.harness import build_arms
from ..ingest import (
    IngestDraft,
    IngestError,
    assemble,
    decode_image,
    vision_messages,
    vision_model,
)
from ..model import ACROSS, DOWN, Puzzle
from ..run import (
    ENSEMBLE_MODEL,
    RunError,
    annotate_candidate_event,
    cell_correctness,
    find_puzzle,
    gold_cells,
    list_puzzles,
    list_suites,
    make_client,
    run_solve,
    serialize_puzzle,
    solver_config,
)
from ..schemas import grid_schema, parse_grid_rows
from ..xd import parse_xd
from .drafts import DraftStore, StoredDraft
from .jobs import BusyError, Job, JobStore
from .limits import LimitError, RateLimiter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NO_KEY = "This host has no Token Factory key."
ORACLE_NEEDS_GOLD = "Oracle needs a puzzle with an answer key. Use Nebius for uploads."
DEFAULT_BACKEND = "nebius"
WEB_ARMS = ("a2", "a3", "a4", "a5", "a6")


class SolveRequest(BaseModel):
    puzzle_id: str
    backend: str = DEFAULT_BACKEND
    arm: str = "a3"
    model: str | None = None
    ensemble_model: str | None = None
    seed: int = 7
    debug: bool = False
    oracle_recall: float = Field(default=0.8, ge=0.0, le=1.0)
    oracle_top1_error: float = Field(default=0.35, ge=0.0, le=1.0)


class IngestRequest(BaseModel):
    image: str | None = None
    across: str = ""
    down: str = ""
    title: str = ""
    xd: str | None = None
    arm: str = "a3"
    model: str | None = None
    ensemble_model: str | None = None
    seed: int = 7
    debug: bool = False


class GridFixRequest(BaseModel):
    rows: list[str]
    across: str | None = None
    down: str | None = None
    arm: str = "a3"
    model: str | None = None
    ensemble_model: str | None = None
    seed: int = 7
    debug: bool = False


def create_app(
    *,
    vision: Callable[[bytes, str], list[str]] | None = None,
    client_factory: Callable[..., Any] | None = None,
    limiter: RateLimiter | None = None,
    drafts: DraftStore | None = None,
    require_key: bool | None = None,
) -> FastAPI:
    app = FastAPI(title="Crossword Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = JobStore()
    draft_store = drafts or DraftStore()
    rate = limiter or RateLimiter(
        hourly=int(os.environ.get("HOURLY_SOLVE_CAP", "5") or "5"),
        daily=int(os.environ.get("DAILY_SOLVE_CAP", "40") or "40"),
    )
    app.state.store = store
    app.state.drafts = draft_store
    app.state.limiter = rate
    app.state.vision = vision
    app.state.client_factory = client_factory or make_client
    app.state.require_key = require_key

    def has_key() -> bool:
        if app.state.require_key is False:
            return True
        return bool(os.environ.get("NEBIUS_API_KEY"))

    def lookup_puzzle(puzzle_id: str) -> tuple[Puzzle, dict | None]:
        stored = draft_store.get(puzzle_id)
        if stored and stored.ingest.puzzle is not None:
            return stored.ingest.puzzle, stored.ingest.prefill
        return find_puzzle(puzzle_id), None

    def charge(request: Request) -> None:
        ip = _client_ip(request)
        if _is_loopback(ip):
            return
        try:
            rate.hit(ip)
        except LimitError as exc:
            headers = {}
            if exc.retry_after is not None:
                headers["Retry-After"] = str(exc.retry_after)
            raise HTTPException(status_code=429, detail=str(exc), headers=headers) from exc

    @app.get("/api/health")
    def health():
        busy = store.active_id is not None
        current = store.jobs.get(store.active_id) if store.active_id else None
        if current and current.status not in ("queued", "running"):
            busy = False
        return {
            "ok": True,
            "has_key": bool(os.environ.get("NEBIUS_API_KEY")),
            "busy": busy,
        }

    @app.get("/api/suites")
    def suites():
        return list_suites()

    @app.get("/api/defaults")
    def defaults():
        arms = build_arms()
        return {
            "backend": DEFAULT_BACKEND,
            "arm": "a3",
            "model": DEFAULT_MODEL,
            "models": list(KNOWN_MODELS),
            "repair_model": DEFAULT_REASONING_MODEL,
            "ensemble_model": ENSEMBLE_MODEL,
            "has_key": bool(os.environ.get("NEBIUS_API_KEY")),
            "arms": [
                {"id": name, "label": arm.label, "description": arm.description}
                for name, arm in arms.items()
                if name in WEB_ARMS
            ],
        }

    @app.get("/api/puzzles")
    def puzzles(suite: str = "mini"):
        try:
            return list_puzzles(suite)
        except RunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/puzzles/{puzzle_id}")
    def puzzle(puzzle_id: str):
        try:
            found, _ = lookup_puzzle(puzzle_id)
            return serialize_puzzle(found)
        except RunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/ingest")
    def ingest(req: IngestRequest, request: Request):
        if req.xd and req.image:
            raise HTTPException(status_code=400, detail="send image or xd, not both")
        if not req.xd and not req.image:
            raise HTTPException(status_code=400, detail="image or xd is required")
        if not has_key() and not req.xd:
            raise HTTPException(status_code=503, detail=NO_KEY)
        charge(request)
        if req.xd:
            try:
                puzzle = parse_xd(req.xd)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            wrapped = IngestDraft(
                status="ready",
                puzzle=puzzle,
                prefill={},
                rows=puzzle.grid.render(blank="."),
                across_slots=sum(1 for s in puzzle.slots if s.direction == ACROSS),
                down_slots=sum(1 for s in puzzle.slots if s.direction == DOWN),
                across_clues=sum(1 for s in puzzle.slots if s.direction == ACROSS),
                down_clues=sum(1 for s in puzzle.slots if s.direction == DOWN),
            )
            stored = draft_store.put(
                wrapped,
                across="",
                down="",
                title=req.title or puzzle.metadata.get("Title", ""),
            )
            if stored.ingest.puzzle:
                stored.ingest.puzzle.metadata["id"] = stored.id
            return _ready_or_edit(app, store, stored, req)

        try:
            image_bytes, mime = decode_image(req.image or "")
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            rows = _read_rows(app, image_bytes, mime)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ingest_draft = assemble(
            rows,
            req.across,
            req.down,
            puzzle_id="upload-pending",
            title=req.title,
        )
        stored = draft_store.put(
            ingest_draft, across=req.across, down=req.down, title=req.title
        )
        if stored.ingest.puzzle is not None:
            stored.ingest.puzzle.metadata["id"] = stored.id
        return _ready_or_edit(app, store, stored, req)

    @app.post("/api/ingest/{draft_id}/grid")
    def ingest_grid(draft_id: str, req: GridFixRequest, request: Request):
        stored = draft_store.get(draft_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="no such draft")
        across = req.across if req.across is not None else stored.across
        down = req.down if req.down is not None else stored.down
        ingest_draft = assemble(
            req.rows, across, down, puzzle_id=draft_id, title=stored.title
        )
        updated = draft_store.put(
            ingest_draft,
            across=across,
            down=down,
            title=stored.title,
            draft_id=draft_id,
        )
        if updated.ingest.puzzle is not None:
            updated.ingest.puzzle.metadata["id"] = draft_id
        if updated.ingest.status == "ready":
            charge(request)
        return _ready_or_edit(app, store, updated, req)

    @app.post("/api/solves")
    def start_solve(req: SolveRequest, request: Request):
        try:
            puzzle, prefill = lookup_puzzle(req.puzzle_id)
        except RunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        backend = req.backend or DEFAULT_BACKEND
        if backend == "oracle" and not puzzle.has_gold():
            raise HTTPException(status_code=400, detail=ORACLE_NEEDS_GOLD)
        if backend == DEFAULT_BACKEND and not has_key() and app.state.client_factory is make_client:
            raise HTTPException(status_code=503, detail=NO_KEY)
        charge(request)
        return _launch(app, store, puzzle, req, backend, prefill=prefill)

    @app.get("/api/solves/{job_id}")
    def solve_status(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job.snapshot()

    @app.get("/api/solves/{job_id}/events")
    def solve_events(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return StreamingResponse(
            _sse(job),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    _mount_frontend(app)
    return app


def _read_rows(app: FastAPI, image_bytes: bytes, mime: str) -> list[str]:
    if app.state.vision is not None:
        return app.state.vision(image_bytes, mime)
    if not os.environ.get("NEBIUS_API_KEY"):
        raise HTTPException(status_code=503, detail=NO_KEY)
    client = NebiusClient()

    def build_messages(schema_in_prompt: bool = False):
        return vision_messages(image_bytes, mime, schema_in_prompt=schema_in_prompt)

    completion, _ = request_with_ladder(
        client,
        model=vision_model(),
        build_messages=build_messages,
        temperature=0.0,
        max_tokens=1024,
        seed=None,
        schema_fn=grid_schema,
    )
    return parse_grid_rows(completion.text)


def _ready_or_edit(
    app, store: JobStore, stored: StoredDraft, req: IngestRequest | GridFixRequest
) -> dict:
    ingest = stored.ingest
    if ingest.status != "ready" or ingest.puzzle is None:
        return {
            "status": "needs_edit",
            "draft_id": stored.id,
            "rows": ingest.rows,
            "height": len(ingest.rows),
            "width": len(ingest.rows[0]) if ingest.rows else 0,
            "across_slots": ingest.across_slots,
            "down_slots": ingest.down_slots,
            "across_clues": ingest.across_clues,
            "down_clues": ingest.down_clues,
            "unknown_numbers": ingest.unknown_numbers,
            "message": ingest.message,
            "puzzle": None,
        }
    solve_req = SolveRequest(
        puzzle_id=stored.id,
        arm=req.arm,
        model=req.model,
        ensemble_model=req.ensemble_model,
        seed=req.seed,
        debug=req.debug,
    )
    launched = _launch(
        app, store, ingest.puzzle, solve_req, DEFAULT_BACKEND, prefill=ingest.prefill
    )
    return {
        "status": "ready",
        "draft_id": stored.id,
        "puzzle": serialize_puzzle(ingest.puzzle),
        **launched,
    }


def _launch(app, store: JobStore, puzzle: Puzzle, req: SolveRequest, backend: str, prefill=None) -> dict:
    try:
        job = store.begin(puzzle.id)
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    thread = threading.Thread(
        target=_run_job,
        args=(store, job, puzzle, req, backend, prefill, app.state.client_factory),
        daemon=True,
    )
    thread.start()
    return {
        "job_id": job.id,
        "puzzle_id": puzzle.id,
        "backend": backend,
        "arm": req.arm,
        "model": req.model or DEFAULT_MODEL,
        "debug": req.debug,
    }


def _run_job(
    store: JobStore,
    job: Job,
    puzzle,
    req: SolveRequest,
    backend: str,
    prefill=None,
    client_factory=None,
) -> None:
    candidate_batches: list[dict[str, Any]] = []
    factory = client_factory or make_client

    def listener(event) -> None:
        payload = _jsonable(event.as_dict())
        if event.kind == CANDIDATES:
            if req.debug:
                payload = annotate_candidate_event(payload, puzzle)
            slots = (payload.get("data") or {}).get("slots") or []
            candidate_batches.append({"round": payload.get("round", 0), "slots": slots})
        job.push(payload)

    tracer = Tracer(listeners=[listener])
    try:
        client = factory(
            puzzle,
            backend=backend,
            seed=req.seed,
            oracle_recall=req.oracle_recall,
            oracle_top1_error=req.oracle_top1_error,
        )
        arm, config = solver_config(
            arm=req.arm,
            seed=req.seed,
            model=req.model or DEFAULT_MODEL,
            ensemble_model=req.ensemble_model or ENSEMBLE_MODEL,
        )
        result, scores = run_solve(
            puzzle,
            client=client,
            config=config,
            one_shot=arm.one_shot,
            tracer=tracer,
            prefill=prefill,
        )
        locked = [[r, c] for r, c in sorted(result.locked)]
        finished = {
            "kind": "finished",
            "round": result.rounds,
            "message": "gold check" if scores else "solved",
            "data": {},
            "scores": scores.as_dict() if scores else None,
            "cells": cell_correctness(puzzle, result.solution),
            "gold": gold_cells(puzzle),
            "locked": locked,
            "solve": result.as_dict(),
            "assignment": result.assignment,
            "candidate_batches": candidate_batches if req.debug else [],
        }
        job.complete(finished, {
            "status": "done",
            "scores": finished["scores"],
            "cells": finished["cells"],
            "locked": locked,
            "solve": result.as_dict(),
        })
    except Exception as exc:
        job.fail(str(exc))
    finally:
        store.release(job.id)


def _is_loopback(ip: str) -> bool:
    return ip in {"127.0.0.1", "::1", "localhost"}


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    if request.client:
        return request.client.host
    return "unknown"


def _sse(job: Job):
    index = 0
    while True:
        batch = job.wait_from(index)
        if not batch:
            if job.status in ("done", "error") and index >= len(job.events):
                return
            yield ": keepalive\n\n"
            continue
        for event in batch:
            yield f"data: {json.dumps(event, default=str)}\n\n"
            index += 1
            if event.get("kind") in ("finished", "error"):
                return


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _mount_frontend(app: FastAPI) -> None:
    dist = os.path.join(ROOT, "web", "dist")
    if not os.path.isdir(dist):
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")


app = create_app()
