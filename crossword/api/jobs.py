"""In-memory solve jobs. One active solve at a time."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    id: str
    puzzle_id: str
    status: str = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    _cv: threading.Condition = field(default_factory=threading.Condition)

    def push(self, event: dict[str, Any]) -> None:
        with self._cv:
            self.events.append(event)
            self._cv.notify_all()

    def wait_from(self, index: int, timeout: float = 0.5) -> list[dict[str, Any]]:
        with self._cv:
            self._cv.wait_for(
                lambda: len(self.events) > index or self.status in ("done", "error"),
                timeout=timeout,
            )
            return list(self.events[index:])

    def complete(self, event: dict[str, Any], result: dict[str, Any]) -> None:
        with self._cv:
            self.result = result
            self.status = "done"
            self.events.append(event)
            self._cv.notify_all()

    def fail(self, message: str) -> None:
        with self._cv:
            self.status = "error"
            self.error = message
            self.events.append({"kind": "error", "round": 0, "message": message, "data": {}})
            self._cv.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._cv:
            payload = {
                "job_id": self.id,
                "puzzle_id": self.puzzle_id,
                "status": self.status,
            }
            if self.result:
                payload.update(self.result)
            if self.error:
                payload["error"] = self.error
            return payload


class JobStore:
    """At most one running solve. Finished jobs linger so the UI can poll."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.active_id: str | None = None

    def occupy(self, job_id: str) -> None:
        """Mark the store busy without starting a worker. Used by tests."""
        with self._lock:
            self.active_id = job_id
            self.jobs[job_id] = Job(id=job_id, puzzle_id="", status="running")

    def begin(self, puzzle_id: str) -> Job:
        with self._lock:
            if self.active_id is not None:
                current = self.jobs.get(self.active_id)
                if current and current.status in ("queued", "running"):
                    raise BusyError("busy: a solve is already running")
            job = Job(id=uuid.uuid4().hex[:12], puzzle_id=puzzle_id, status="running")
            self.jobs[job.id] = job
            self.active_id = job.id
            return job

    def release(self, job_id: str) -> None:
        with self._lock:
            if self.active_id == job_id:
                self.active_id = None

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)


class BusyError(Exception):
    pass
