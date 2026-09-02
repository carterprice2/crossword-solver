"""Public-demo rate limits: per-IP hourly and process-wide daily."""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class LimitError(Exception):
    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    def __init__(self, hourly: int = 5, daily: int = 40, clock=time.time):
        self.hourly = hourly
        self.daily = daily
        self.clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._day = ""
        self._day_count = 0

    def hit(self, ip: str, now: float | None = None) -> None:
        """Record a start. Raises LimitError if the caller is over quota."""
        ts = self.clock() if now is None else now
        day = time.strftime("%Y-%m-%d", time.gmtime(ts))
        with self._lock:
            if day != self._day:
                self._day = day
                self._day_count = 0
            if self._day_count >= self.daily:
                tomorrow = time.mktime(time.strptime(day, "%Y-%m-%d")) + 86400
                raise LimitError(
                    "This host is rate-limited. Try again later.",
                    retry_after=max(1, int(tomorrow - ts)),
                )
            window = ts - 3600
            recent = [t for t in self._hits[ip] if t > window]
            self._hits[ip] = recent
            if len(recent) >= self.hourly:
                retry = int(recent[0] + 3600 - ts) + 1
                raise LimitError(
                    "This host is rate-limited. Try again later.",
                    retry_after=max(1, retry),
                )
            recent.append(ts)
            self._hits[ip] = recent
            self._day_count += 1
