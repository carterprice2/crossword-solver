"""In-memory BYO drafts. Survive until the process restarts."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from ..ingest import IngestDraft


@dataclass
class StoredDraft:
    id: str
    ingest: IngestDraft
    across: str
    down: str
    title: str = ""


class DraftStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.items: dict[str, StoredDraft] = {}

    def put(
        self,
        ingest: IngestDraft,
        *,
        across: str,
        down: str,
        title: str = "",
        draft_id: str | None = None,
    ) -> StoredDraft:
        with self._lock:
            ident = draft_id or f"upload-{uuid.uuid4().hex[:12]}"
            stored = StoredDraft(id=ident, ingest=ingest, across=across, down=down, title=title)
            self.items[ident] = stored
            return stored

    def get(self, draft_id: str) -> StoredDraft | None:
        return self.items.get(draft_id)
