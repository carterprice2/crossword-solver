"""In-place terminal view of a solve in progress.

Redraws the grid alongside a status panel as the solver emits events. Colour
carries the state that matters while watching: dim = unknown, yellow =
tentative, green = locked by two agreeing confident entries. The point of the
view is that the repair round is *visible* -- you can watch slots that round 0
left blank get filled once their crossings pin the pattern down.

The final frame is a separate *gold check*: every letter is graded against the
answer key (green = correct, red = wrong). The repair log and lock colours do
not survive into that frame, so "the agent locked this" is not confused with
"this is the right letter".
"""

from __future__ import annotations

import os
import shutil
import sys
import time

from ..agent.trace import (
    BATCH_DONE,
    CONSTRAINTS,
    LOCKED,
    REPAIR,
    ROUND_START,
    SEARCH,
    SOLVED,
    SolveEvent,
)
from ..model import Puzzle

RESET = "\033[0m"
DIM = "\033[2;37m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
CYAN = "\033[36m"
BLOCK_COLOR = "\033[90m"

HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


def supports_color(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


class LiveView:
    """Renders solve events. Degrades to plain line-by-line logging when the
    output is not a terminal, so piping to a file still produces something
    readable."""

    def __init__(
        self,
        puzzle: Puzzle,
        *,
        stream=None,
        color: bool | None = None,
        min_interval: float = 0.08,
    ):
        self.puzzle = puzzle
        self.stream = stream or sys.stdout
        self.color = supports_color(self.stream) if color is None else color
        self.interactive = self.color and getattr(self.stream, "isatty", lambda: False)()
        self.min_interval = min_interval
        self.started = time.monotonic()
        self.lines_drawn = 0
        self.last_draw = 0.0

        self.cells: dict[tuple[int, int], str] = {}
        self.locked: set[tuple[int, int]] = set()
        self.round = 0
        self.status = "starting"
        self.log: list[str] = []
        self.calls = 0
        self.tokens = 0
        self.filled = 0
        self.icr = 1.0
        self.model = ""
        self._gold_scores = None

    # -- painting ---------------------------------------------------------

    def _paint(self, text: str, color: str) -> str:
        return f"{color}{text}{RESET}" if self.color else text

    def _grid_lines(self, *, scorecard: bool = False) -> list[str]:
        gold = self.puzzle.gold_solution() if (scorecard and self.puzzle.has_gold()) else {}
        out = []
        for r in range(self.puzzle.grid.height):
            row = []
            for c in range(self.puzzle.grid.width):
                cell = (r, c)
                if self.puzzle.grid.is_block(cell):
                    row.append(self._paint("##", BLOCK_COLOR))
                    continue
                letter = self.cells.get(cell)
                if not letter:
                    row.append(self._paint(" .", DIM))
                    continue
                glyph = f" {letter[:1]}"
                if scorecard and gold:
                    want = gold.get(cell, "")
                    if letter.upper() == want.upper():
                        row.append(self._paint(glyph, GREEN))
                    else:
                        row.append(self._paint(glyph, RED))
                elif cell in self.locked:
                    row.append(self._paint(glyph, GREEN))
                else:
                    row.append(self._paint(glyph, YELLOW))
            out.append("".join(row))
        return out

    def _panel_lines(self, *, scorecard: bool = False) -> list[str]:
        if scorecard:
            return self._scorecard_panel()
        elapsed = time.monotonic() - self.started
        total = len(self.puzzle.slots)
        panel = [
            self._paint(f"round {self.round}", BOLD)
            + f"   slots {self.filled}/{total}   ICR {self.icr:.2f}",
            f"calls {self.calls}   tokens {self.tokens:,}   {elapsed:5.1f}s",
        ]
        if self.model:
            panel.append(self._paint(self.model, CYAN))
        panel.append("")
        panel.extend(self.log[-8:])
        return panel

    def _scorecard_panel(self) -> list[str]:
        scores = self._gold_scores
        elapsed = time.monotonic() - self.started
        panel = [self._paint("gold check", BOLD) + "  vs the answer key"]
        if scores is None:
            panel.append("no answer key on this puzzle")
            return panel
        verdict = "SOLVED" if scores.exact else "PARTIAL"
        color = GREEN if scores.exact else YELLOW
        panel.append(self._paint(verdict, color))
        panel.append(
            f"WCR {scores.wcr:.3f}  LCR {scores.lcr:.3f}  ICR {scores.icr:.3f}"
        )
        wrong = scores.cells_filled - scores.cells_correct
        blank = scores.cells_total - scores.cells_filled
        panel.append(
            f"{scores.cells_correct} correct  {wrong} wrong  {blank} blank"
        )
        panel.append(self._paint("green = correct", GREEN))
        panel.append(self._paint("red = wrong", RED))
        panel.append(f"{self.calls} calls  {elapsed:5.1f}s")
        return panel

    def render(self, *, scorecard: bool = False, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_draw < self.min_interval:
            return
        self.last_draw = now

        grid = self._grid_lines(scorecard=scorecard)
        panel = self._panel_lines(scorecard=scorecard)
        width = shutil.get_terminal_size((100, 30)).columns
        gutter = "   "
        body = []
        for index in range(max(len(grid), len(panel))):
            left = grid[index] if index < len(grid) else " " * (self.puzzle.grid.width * 2)
            right = panel[index] if index < len(panel) else ""
            body.append((left + gutter + right)[: width + 64].rstrip())

        kind = "gold check" if scorecard else f"{self.puzzle.grid.height}x{self.puzzle.grid.width}"
        header = self._paint(
            f"Reno Crossword Agent  ·  {self.puzzle.id}  ·  {kind}",
            BOLD,
        )
        lines = [header, ""] + body

        if self.interactive:
            if self.lines_drawn:
                self.stream.write(f"\033[{self.lines_drawn}A")
            for line in lines:
                self.stream.write("\033[2K" + line + "\n")
            self.lines_drawn = len(lines)
        else:
            for line in lines:
                self.stream.write(line + "\n")
            self.stream.write("\n")
        self.stream.flush()

    # -- event handling ---------------------------------------------------

    def __call__(self, event: SolveEvent) -> None:
        self.handle(event)

    def handle(self, event: SolveEvent) -> None:
        data = event.data
        self.round = event.round
        force = False

        if event.kind == ROUND_START:
            self.log.append(self._paint(f"R{event.round} {event.message}", BOLD))
            force = True
        elif event.kind == REPAIR:
            slots = ", ".join(data.get("slots", [])[:6])
            self.log.append(
                self._paint(f"R{event.round} repair", YELLOW) + f" {slots}"
            )
            self.model = data.get("model", self.model)
            force = True
        elif event.kind == BATCH_DONE:
            self.calls += 1
            self.tokens += int(data.get("tokens", 0) or 0)
            if data.get("model"):
                self.model = data["model"]
            if data.get("error"):
                self.log.append(self._paint(f"  ! {data['error'][:60]}", RED))
        elif event.kind == CONSTRAINTS:
            conflicts = data.get("conflicts") or []
            if conflicts:
                joined = ", ".join("x".join(c) for c in conflicts[:4])
                self.log.append(f"  conflicts: {joined}")
        elif event.kind == SEARCH:
            self.filled = int(data.get("filled", self.filled))
            self.icr = float(data.get("icr", self.icr))
            rows = data.get("grid")
            if rows:
                self._absorb(rows)
            force = True
        elif event.kind == LOCKED:
            self.log.append(self._paint(f"  {event.message}", GREEN))
        elif event.kind == SOLVED:
            rows = data.get("grid")
            if rows:
                self._absorb(rows)
            self.status = event.message
            force = True

        self.render(force=force)

    def _absorb(self, rows: list[str]) -> None:
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch in ("#", ".", " "):
                    continue
                self.cells[(r, c)] = ch

    def mark_locked(self, cells) -> None:
        self.locked = set(cells)

    def finish(self, result, scores=None) -> None:
        """Replace the working view with a gold-check scorecard.

        Working-grid colours mean lock state. Scorecard colours mean correctness.
        The repair log is dropped so the two are not on screen at the same time.
        """
        self.cells.update({c: v for c, v in result.solution.items()})
        self._gold_scores = scores
        self.log = []
        self.render(scorecard=True, force=True)
        if scores is None:
            return
        verdict = (
            self._paint("SOLVED", GREEN)
            if scores.exact
            else self._paint("PARTIAL", YELLOW)
        )
        self.stream.write(
            f"\n{verdict}  WCR {scores.wcr:.3f}  LCR {scores.lcr:.3f}  "
            f"ICR {scores.icr:.3f}  "
            f"{result.usage.total_tokens:,} tok  "
            f"{result.usage.calls} calls  {result.seconds:.1f}s\n"
        )
        self.stream.flush()

    def __enter__(self):
        if self.interactive:
            self.stream.write(HIDE_CURSOR)
        return self

    def __exit__(self, *exc):
        if self.interactive:
            self.stream.write(SHOW_CURSOR)
            self.stream.flush()
        return False
