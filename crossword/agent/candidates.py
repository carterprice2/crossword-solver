"""Getting candidate answers out of a model and into validated form.

Two things here are not obvious.

**Batching by locality.** Clues are grouped into batches by walking the
intersection graph, not by index order. A batch then contains clues that
actually cross each other, so the model can check its own answers against one
another as it writes them, and any theme material stays together. Chunking the
clue list in order scatters each neighbourhood across several requests.

**The degradation ladder.** Guided-JSON support varies across the models Nebius
serves, and a rejected schema returns HTTP 400. Rather than lose the request,
retry with a progressively weaker constraint and record which rung worked --
that record is itself an evaluation result.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..client import Completion, ModelClient, SchemaRejected
from ..model import Puzzle
from ..progress import log as progress_log
from ..schemas import (
    FREE_TEXT,
    LADDER,
    Candidate,
    candidates_schema,
    looks_like_candidates,
    parse_candidates,
    response_format_for,
)
from .constraints import SlotGraph
from .trace import BATCH_DONE, BATCH_SENT, CANDIDATES, Tracer

DEFAULT_BATCH_SIZE = 14
REPAIR_BATCH_SIZE = 8


@dataclass
class BatchResult:
    slot_ids: list[str]
    candidates: list[Candidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    completion: Completion | None = None
    rung: str = LADDER[0]
    error: str = ""


def batch_by_locality(
    graph: SlotGraph, slot_ids: list[str], size: int = DEFAULT_BATCH_SIZE
) -> list[list[str]]:
    """Group slots into batches of crossing neighbours via breadth-first walk."""
    if size <= 0:
        return [list(slot_ids)]
    wanted = [s for s in slot_ids if s in graph.by_slot]
    remaining = set(wanted)
    order = {sid: i for i, sid in enumerate(wanted)}
    batches: list[list[str]] = []

    while remaining:
        # Start from the earliest remaining slot so batching is deterministic.
        seed = min(remaining, key=lambda s: order[s])
        batch = [seed]
        remaining.discard(seed)
        frontier = [seed]
        while frontier and len(batch) < size:
            current = frontier.pop(0)
            for neighbor in sorted(graph.neighbors(current), key=lambda s: order.get(s, 1 << 30)):
                if len(batch) >= size:
                    break
                if neighbor in remaining:
                    remaining.discard(neighbor)
                    batch.append(neighbor)
                    frontier.append(neighbor)
        batches.append(batch)
    return batches


def _truncated(completion: Completion, budget: int) -> bool:
    return completion.completion_tokens >= max(1, int(budget * 0.95))


def request_with_ladder(
    client: ModelClient,
    *,
    model: str,
    build_messages,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    start_rung: str = LADDER[0],
    schema_fn=candidates_schema,
) -> tuple[Completion, str]:
    """Send a request, weakening the schema constraint until one is usable.

    Token Factory often *accepts* ``json_schema`` (HTTP 200) while a thinking
    model burns the whole ``max_tokens`` budget on chain-of-thought and never
    emits JSON. Treat that as a failed rung: bump the budget once, then step
    down the ladder. Only pin a rung when ``looks_like_candidates`` is true.
    """
    began = LADDER.index(start_rung) if start_rung in LADDER else 0
    last: Exception | None = None
    last_completion: Completion | None = None
    last_rung = start_rung
    bumped = max(8192, max_tokens)
    for rung in LADDER[began:]:
        messages = build_messages(schema_in_prompt=rung in (FREE_TEXT, "json_object"))
        budget = max_tokens
        while True:
            try:
                completion = client.complete(
                    model=model,
                    messages=messages,
                    response_format=response_format_for(rung, schema_fn=schema_fn),
                    temperature=temperature,
                    max_tokens=budget,
                    seed=seed,
                )
            except SchemaRejected as exc:
                last = exc
                progress_log(f"ladder {model} {rung} rejected schema; next rung")
                break
            last_completion = completion
            last_rung = rung
            if looks_like_candidates(completion.text):
                if rung != start_rung or budget != max_tokens:
                    progress_log(
                        f"ladder {model} usable at {rung} max_tokens={budget}"
                    )
                return completion, rung
            if _truncated(completion, budget) and budget < bumped:
                progress_log(
                    f"ladder {model} {rung} truncated at {budget} tokens; "
                    f"retry {bumped}"
                )
                budget = bumped
                continue
            progress_log(
                f"ladder {model} {rung} no JSON ({completion.completion_tokens} tok); "
                "next rung"
            )
            break
    if last_completion is not None:
        return last_completion, last_rung
    raise last or RuntimeError("no rung of the schema ladder succeeded")


class CandidateGenerator:
    """Runs candidate requests, in parallel, with validation and tracing."""

    def __init__(
        self,
        client: ModelClient,
        puzzle: Puzzle,
        graph: SlotGraph,
        *,
        tracer: Tracer | None = None,
        max_workers: int = 8,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        seed: int | None = 7,
    ):
        self.client = client
        self.puzzle = puzzle
        self.graph = graph
        self.tracer = tracer
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        #: Remembered so a model that rejects strict schemas is not re-probed
        #: on every batch.
        self.rung_by_model: dict[str, str] = {}

    def _run_batch(
        self, model: str, slot_ids: list[str], build_messages, patterns: dict[str, str],
        round_index: int,
    ) -> BatchResult:
        expected = {sid: self.graph.length[sid] for sid in slot_ids}
        if self.tracer:
            self.tracer.emit(
                BATCH_SENT,
                f"batch {','.join(slot_ids)} -> {model}",
                round=round_index,
                slots=slot_ids,
                model=model,
            )
        try:
            completion, rung = request_with_ladder(
                self.client,
                model=model,
                build_messages=build_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                start_rung=self.rung_by_model.get(model, LADDER[0]),
            )
        except Exception as exc:
            if self.tracer:
                self.tracer.emit(
                    BATCH_DONE,
                    f"batch of {len(slot_ids)} failed: {exc}",
                    round=round_index,
                    slots=slot_ids,
                    error=str(exc),
                )
            return BatchResult(slot_ids=slot_ids, error=str(exc))

        self.rung_by_model[model] = rung
        candidates, warnings = parse_candidates(
            completion.text, expected=expected, patterns=patterns
        )
        if self.tracer:
            by_slot: dict[str, list[dict]] = {sid: [] for sid in slot_ids}
            for candidate in candidates:
                by_slot.setdefault(candidate.slot_id, []).append(
                    {
                        "answer": candidate.answer,
                        "confidence": round(candidate.confidence, 3),
                    }
                )
            self.tracer.emit(
                CANDIDATES,
                f"{len(candidates)} candidates across {len(slot_ids)} slots",
                round=round_index,
                slots=[
                    {
                        "id": sid,
                        "pattern": patterns.get(sid, ""),
                        "candidates": by_slot.get(sid, []),
                    }
                    for sid in slot_ids
                ],
            )
            self.tracer.emit(
                BATCH_DONE,
                f"{len(slot_ids)} clues -> {len(candidates)} candidates",
                round=round_index,
                slots=slot_ids,
                model=completion.model,
                rung=rung,
                tokens=completion.total_tokens,
                latency_s=round(completion.latency_s, 3),
                warnings=warnings[:4],
            )
        return BatchResult(
            slot_ids=slot_ids,
            candidates=candidates,
            warnings=warnings,
            completion=completion,
            rung=rung,
        )

    def run(
        self, model: str, jobs: list[tuple[list[str], object, dict[str, str]]],
        round_index: int = 0,
    ) -> list[BatchResult]:
        """Run prepared batches concurrently.

        Threads (not asyncio) because urllib is blocking and the work is purely
        I/O-bound; the pool size caps concurrency against the API's rate limit.
        """
        if not jobs:
            return []
        if len(jobs) == 1 or self.max_workers <= 1:
            return [
                self._run_batch(model, slots, build, patterns, round_index)
                for slots, build, patterns in jobs
            ]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(jobs))) as pool:
            futures = [
                pool.submit(self._run_batch, model, slots, build, patterns, round_index)
                for slots, build, patterns in jobs
            ]
            return [f.result() for f in futures]
