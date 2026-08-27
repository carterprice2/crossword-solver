"""Event stream from a solve: drives the live view and the recorded trace."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

#: Event kinds, in roughly the order they occur.
ROUND_START = "round_start"
BATCH_SENT = "batch_sent"
BATCH_DONE = "batch_done"
CANDIDATES = "candidates"
CONSTRAINTS = "constraints"
SEARCH = "search"
LOCKED = "locked"
REPAIR = "repair"
ROUND_END = "round_end"
VERIFY = "verify"
SOLVED = "solved"
WARNING = "warning"


@dataclass
class SolveEvent:
    kind: str
    round: int = 0
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return asdict(self)


Listener = Callable[[SolveEvent], None]


class Tracer:
    """Fans events out to listeners and optionally to a JSONL file."""

    def __init__(self, path: str | None = None, listeners: list[Listener] | None = None):
        self.path = path
        self.listeners = list(listeners or [])
        self.events: list[SolveEvent] = []
        self._lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            open(path, "w").close()

    def emit(self, kind: str, message: str = "", *, round: int = 0, **data) -> SolveEvent:
        event = SolveEvent(kind=kind, round=round, message=message, data=data)
        with self._lock:
            self.events.append(event)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event.as_dict(), default=str) + "\n")
        # Listeners run outside the lock: a slow renderer must not block the
        # solver, and a broken one must not kill the run.
        for listener in self.listeners:
            try:
                listener(event)
            except Exception:  # pragma: no cover - a UI bug is not a solve bug
                pass
        return event

    def add_listener(self, listener: Listener) -> None:
        self.listeners.append(listener)

    def of_kind(self, kind: str) -> list[SolveEvent]:
        return [e for e in self.events if e.kind == kind]
