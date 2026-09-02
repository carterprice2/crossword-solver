"""Stderr progress for long live evals.

Off by default so unit tests stay quiet. ``crossword eval`` turns it on
unless ``--quiet`` is passed. Every line is flushed so a blocking HTTP
call cannot look like a freeze.
"""

from __future__ import annotations

import sys
import threading
import time

_enabled = False
_t0 = 0.0


def enable() -> None:
    global _enabled, _t0
    _enabled = True
    _t0 = time.monotonic()


def disable() -> None:
    global _enabled
    _enabled = False


def enabled() -> bool:
    return _enabled


def log(message: str) -> None:
    if not _enabled:
        return
    elapsed = time.monotonic() - _t0
    thread = threading.current_thread().name
    if thread.startswith("Main"):
        thread = "main"
    print(f"  {elapsed:7.1f}s [{thread}] {message}", file=sys.stderr, flush=True)
