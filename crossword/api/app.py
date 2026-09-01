"""FastAPI app: corpus listing and a streaming solve.

Imported only by the serve command and by tests that skip when FastAPI is
missing. The rest of the package does not import this module.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agent.trace import CANDIDATES, Tracer
from ..client import DEFAULT_MODEL, DEFAULT_REASONING_MODEL, KNOWN_MODELS
from ..eval.harness import build_arms
from ..run import (
    RunError,
    annotate_candidate_event,
    cell_correctness,
    default_backend,
    find_puzzle,
    gold_cells,
    list_puzzles,
    list_suites,
    make_client,
    run_solve,
    serialize_puzzle,
    solver_config,
)
from .jobs import BusyError, Job, JobStore

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SolveRequest(BaseModel):
    puzzle_id: str
    backend: str | None = None
    arm: str = "a3"
    model: str | None = None
    seed: int = 7
    debug: bool = False
    oracle_recall: float = Field(default=0.8, ge=0.0, le=1.0)
    oracle_top1_error: float = Field(default=0.35, ge=0.0, le=1.0)


def create_app() -> FastAPI:
    app = FastAPI(title="Crossword Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = JobStore()
    app.state.store = store

    @app.get("/api/suites")
    def suites():
        return list_suites()

    @app.get("/api/defaults")
    def defaults():
        arms = build_arms()
        return {
            "backend": default_backend(),
            "arm": "a3",
            "model": DEFAULT_MODEL,
            "models": list(KNOWN_MODELS),
            "repair_model": DEFAULT_REASONING_MODEL,
            "has_key": bool(os.environ.get("NEBIUS_API_KEY")),
            "arms": [
                {"id": name, "label": arm.label, "description": arm.description}
                for name, arm in arms.items()
                if name in ("a2", "a3")
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
            return serialize_puzzle(find_puzzle(puzzle_id))
        except RunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/solves")
    def start_solve(req: SolveRequest):
        try:
            puzzle = find_puzzle(req.puzzle_id)
        except RunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        backend = req.backend or default_backend()
        try:
            job = store.begin(req.puzzle_id)
        except BusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        thread = threading.Thread(
            target=_run_job,
            args=(store, job, puzzle, req, backend),
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


def _run_job(store: JobStore, job: Job, puzzle, req: SolveRequest, backend: str) -> None:
    candidate_batches: list[dict[str, Any]] = []

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
        client = make_client(
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
        )
        result, scores = run_solve(
            puzzle,
            client=client,
            config=config,
            one_shot=arm.one_shot,
            tracer=tracer,
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
