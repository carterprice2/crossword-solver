"""One cheap live call per model: can we parse candidates at all?

Full evals are too expensive to discover that a model accepted json_schema,
burned its token budget on thinking, and returned no object. This asks two
clues from a 3x3, walks the schema ladder, and reports rung + parse.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent.candidates import request_with_ladder
from ..agent.constraints import SlotGraph
from ..agent.prompts import first_pass_messages
from ..client import KNOWN_MODELS, ModelClient
from ..progress import log as progress_log
from ..schemas import parse_candidates
from ..xd import parse_xd

SMOKE_XD = """\
Title: Parse smoke


CAT
ARE
BEE


A1. Feline ~ CAT
A4. Exist ~ ARE
A5. Buzzer ~ BEE

D1. Taxi ~ CAB
D2. Exist ~ ARE
D3. Golf peg ~ TEE
"""

SMOKE_SLOTS = ("A1", "A4")


@dataclass
class SmokeResult:
    model: str
    ok: bool
    rung: str = ""
    n_candidates: int = 0
    slots: tuple[str, ...] = ()
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    skipped: bool = False


def smoke_puzzle():
    return parse_xd(SMOKE_XD)


def smoke_one(client: ModelClient, model: str, *, max_tokens: int = 2048) -> SmokeResult:
    """One first-pass batch on ``SMOKE_SLOTS`` for ``model``."""
    import time

    puzzle = smoke_puzzle()
    graph = SlotGraph(puzzle)
    patterns = {sid: graph.pattern(sid, {}) for sid in SMOKE_SLOTS}
    expected = {sid: graph.length[sid] for sid in SMOKE_SLOTS}

    def build_messages(*, schema_in_prompt: bool = False):
        return first_pass_messages(
            puzzle, graph, list(SMOKE_SLOTS), patterns,
            schema_in_prompt=schema_in_prompt,
        )

    started = time.monotonic()
    try:
        completion, rung = request_with_ladder(
            client,
            model=model,
            build_messages=build_messages,
            temperature=0.3,
            max_tokens=max_tokens,
            seed=0,
        )
    except Exception as exc:
        return SmokeResult(
            model=model,
            ok=False,
            seconds=time.monotonic() - started,
            error=str(exc),
        )
    elapsed = time.monotonic() - started
    candidates, warnings = parse_candidates(
        completion.text, expected=expected, patterns=patterns
    )
    slots = tuple(sorted({c.slot_id for c in candidates}))
    error = ""
    if not candidates:
        error = (warnings[0] if warnings else "no candidates")[:200]
    return SmokeResult(
        model=model,
        ok=bool(candidates),
        rung=rung,
        n_candidates=len(candidates),
        slots=slots,
        seconds=elapsed,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        error=error,
    )


def smoke_catalog(
    client: ModelClient,
    models: list[str] | None = None,
    *,
    available: list[str] | None = None,
    max_tokens: int = 2048,
) -> list[SmokeResult]:
    names = list(models or KNOWN_MODELS)
    reachable = set(available) if available is not None else None
    results: list[SmokeResult] = []
    for name in names:
        if reachable is not None and name not in reachable:
            results.append(
                SmokeResult(model=name, ok=False, skipped=True, error="not reachable")
            )
            progress_log(f"smoke skip {name} (not on this account)")
            continue
        progress_log(f"smoke {name}")
        results.append(smoke_one(client, name, max_tokens=max_tokens))
    return results


def format_smoke_table(results: list[SmokeResult]) -> str:
    headers = ["status", "model", "rung", "cands", "slots", "tok", "sec", "note"]
    rows = []
    for row in results:
        if row.skipped:
            status = "SKIP"
        elif row.ok:
            status = "OK"
        else:
            status = "FAIL"
        tok = row.prompt_tokens + row.completion_tokens
        rows.append(
            [
                status,
                row.model,
                row.rung or "-",
                str(row.n_candidates),
                ",".join(row.slots) or "-",
                str(tok),
                f"{row.seconds:.1f}",
                row.error[:40] if row.error else "",
            ]
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    ]
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        lines.append(
            "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"
        )
    ok = sum(1 for r in results if r.ok)
    fail = sum(1 for r in results if not r.ok and not r.skipped)
    skip = sum(1 for r in results if r.skipped)
    lines.append("")
    lines.append(f"{ok} ok, {fail} failed, {skip} skipped / {len(results)} models")
    return "\n".join(lines) + "\n"
