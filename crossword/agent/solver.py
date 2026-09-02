"""The agent loop: propose, propagate, search, repair.

    round 0   ask the model for candidates for every clue, cold
              -> filter by locked cells -> soft AC-3 -> search
              -> lock the cells the search is confident about
    round 1+  treat each clash as a star (the two entries plus everything
              they touch). Enumerate existing candidates first. If they
              cannot mesh, re-query that star. If they still cannot, keep
              the higher-confidence hub and leave the other blank.
              Then stitch with a global search.
    endgame   promote a declined slot only when crossings already spell
              a real word

Termination is guaranteed by the lock set, which only ever grows, plus a round
cap and a token budget.

Why rounds at all: a model asked "Big galoot (5)" cold may answer OAFISH
(wrong length) or LUMMOX or GALOOT. Asked again as "?E??R with A17=LEDGE
crossing", the question has usually become trivial. Measuring that gap is the
point of the ablation (A3 - A2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..client import ModelClient, Usage
from ..eval.pricing import cost_usd
from ..model import Cell, Puzzle, Solution
from ..schemas import Candidate, merge_candidates
from . import prompts
from .candidates import (
    DEFAULT_BATCH_SIZE,
    REPAIR_BATCH_SIZE,
    CandidateGenerator,
    batch_by_locality,
)
from .constraints import (
    DEFAULT_UNKNOWN_MASS,
    WILDCARD,
    SlotGraph,
    build_domains,
    cells_from_assignment,
    intersection_consistency,
    merge_domains,
    pattern_filter,
    soft_ac3,
)
from .search import endgame_fill, solve as search_solve
from .star import (
    apply_star,
    collect_stars,
    fallback_star,
    solve_star,
    star_complete,
)
from .trace import (
    CONSTRAINTS,
    LOCKED,
    REPAIR,
    ROUND_END,
    ROUND_START,
    SEARCH,
    SOLVED,
    VERIFY,
    Tracer,
)
from .verify import implied_spellings, rescore_with_dictionary, verify_fill


@dataclass
class SolverConfig:
    """Everything that distinguishes one ablation arm from another."""

    model: str = ""
    repair_model: str = ""
    #: Second model queried in round 0; its disagreements are the signal.
    ensemble_model: str = ""
    max_rounds: int = 5
    batch_size: int = DEFAULT_BATCH_SIZE
    repair_batch_size: int = REPAIR_BATCH_SIZE
    unknown_mass: float = DEFAULT_UNKNOWN_MASS
    #: Confidence at which both crossing slots must agree before a cell locks.
    lock_confidence: float = 0.85
    #: Below this, a slot is a repair candidate even if it was assigned.
    repair_confidence: float = 0.6
    temperature: float = 0.3
    max_tokens: int = 2048
    seed: int | None = 7
    max_workers: int = 8
    search_nodes: int = 100_000
    search_restarts: int = 5
    #: Arm switches.
    use_constraints: bool = True
    use_repair: bool = True
    use_endgame: bool = True
    #: After the model proposes words, look up each word's dictionary sense
    #: (never the other way around) and boost ones that match the clue.
    check_definitions: bool = True
    #: Every complete slot must be a real word, abbreviation, or a candidate
    #: the model proposed for that slot. Implied downs like LFA fail this.
    require_real_words: bool = True
    max_calls: int = 60

    @property
    def repair_model_or_default(self) -> str:
        return self.repair_model or self.model


@dataclass
class SolveResult:
    puzzle: Puzzle
    solution: Solution
    assignment: dict[str, str] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    locked: dict[Cell, str] = field(default_factory=dict)
    rounds: int = 0
    usage: Usage = field(default_factory=Usage)
    icr: float = 1.0
    open_slots: list[str] = field(default_factory=list)
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    #: (confidence, was_correct) pairs, when gold is known -- feeds calibration.
    calibration_pairs: list[tuple[float, bool]] = field(default_factory=list)
    rungs: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        usd = cost_usd(self.usage)
        return {
            "rounds": self.rounds,
            "icr": round(self.icr, 6),
            "open_slots": self.open_slots,
            "seconds": round(self.seconds, 3),
            "calls": self.usage.calls,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "cost_usd": None if usd is None else round(usd, 6),
            "rungs": self.rungs,
            "warnings": self.warnings[:10],
        }


class Solver:
    def __init__(
        self,
        client: ModelClient,
        config: SolverConfig,
        *,
        tracer: Tracer | None = None,
    ):
        self.client = client
        self.config = config
        self.tracer = tracer or Tracer()
        self._definitions: dict[str, str] | None = None
        self._definitions_loaded = False

    def _load_definitions(self) -> dict[str, str] | None:
        """Word → dictionary sense. Used only to score words the model proposed."""
        if self._definitions_loaded:
            return self._definitions
        self._definitions_loaded = True
        if not self.config.check_definitions:
            return None
        try:
            from ..gen.bank import load_bank

            bank = load_bank()
        except (OSError, ValueError):
            self._definitions = None
            return None
        self._definitions = {entry.word: entry.clue for entry in bank.entries}
        return self._definitions

    # -- helpers -----------------------------------------------------------

    def _lock_cells(
        self,
        assignment: dict[str, str],
        confidence: dict[str, float],
        graph: SlotGraph,
        locked: dict[Cell, str],
    ) -> dict[Cell, str]:
        """Cells where a confident across and a confident down agree.

        Requiring *both* directions is the whole safeguard: one confident
        answer can be confidently wrong, but two independent entries agreeing
        on a letter is a much stronger claim -- and locks are never released.
        """
        new: dict[Cell, str] = {}
        threshold = self.config.lock_confidence
        for crossing in graph.crossings:
            a_answer = assignment.get(crossing.a)
            b_answer = assignment.get(crossing.b)
            if not a_answer or not b_answer:
                continue
            if a_answer == WILDCARD or b_answer == WILDCARD:
                continue
            if crossing.ai >= len(a_answer) or crossing.bi >= len(b_answer):
                continue
            letter = a_answer[crossing.ai]
            if letter != b_answer[crossing.bi]:
                continue
            if (
                confidence.get(crossing.a, 0.0) >= threshold
                and confidence.get(crossing.b, 0.0) >= threshold
                and crossing.cell not in locked
            ):
                new[crossing.cell] = letter
        return new

    def _repair_targets(
        self,
        assignment: dict[str, str],
        confidence: dict[str, float],
        graph: SlotGraph,
        conflicts,
        locked: dict[Cell, str],
        failed: list[str] | None = None,
    ) -> list[str]:
        """Slots worth spending another call on, best expected gain first."""
        scored: dict[str, float] = {}

        for slot_id, answer in assignment.items():
            if answer == WILDCARD:
                # Unfilled: nothing to lose, and the pattern may now pin it.
                scored[slot_id] = 3.0

        for slot_id in failed or []:
            scored[slot_id] = max(scored.get(slot_id, 0.0), 2.8)

        for site in conflicts:
            for slot_id in site.slots:
                scored[slot_id] = max(scored.get(slot_id, 0.0), 2.5)

        for slot_id, answer in assignment.items():
            if answer == WILDCARD:
                continue
            conf = confidence.get(slot_id, 0.0)
            if conf >= self.config.repair_confidence:
                continue
            cells = graph.cells.get(slot_id, ())
            if not cells:
                continue
            known = sum(1 for cell in cells if cell in locked)
            share = known / len(cells)
            # A shaky answer whose crossings are now mostly pinned is the best
            # possible re-query: the pattern does the work the clue could not.
            scored[slot_id] = max(scored.get(slot_id, 0.0), 1.0 + share)

        return sorted(scored, key=lambda s: (-scored[s], s))

    def _likely_pattern(
        self,
        slot_id: str,
        assignment: dict[str, str],
        graph: SlotGraph,
        locked: dict[Cell, str],
    ) -> str:
        """The pattern implied by *other* slots' current answers.

        The slot's own answer is excluded deliberately. Including it produced
        prompts that showed a slot's pattern as the very answer listed one
        field below as already rejected -- which tells the model nothing and
        invites it to repeat the mistake.
        """
        cells: dict[Cell, str] = {}
        for other, answer in assignment.items():
            if other == slot_id or answer == WILDCARD:
                continue
            for index, cell in enumerate(graph.cells.get(other, ())):
                if index < len(answer):
                    cells.setdefault(cell, answer[index])
        cells.update(locked)
        return graph.pattern(slot_id, cells)

    def _crossing_context(
        self,
        slot_id: str,
        assignment: dict[str, str],
        graph: SlotGraph,
        puzzle: Puzzle,
    ) -> list[tuple[str, str, str]]:
        out = []
        for crossing in graph.by_slot.get(slot_id, []):
            other = crossing.b
            answer = assignment.get(other)
            if not answer or answer == WILDCARD:
                continue
            out.append((other, answer, puzzle.slot(other).clue))
        return out

    # -- main loop ---------------------------------------------------------

    def solve(
        self, puzzle: Puzzle, *, prefill: dict[Cell, str] | None = None
    ) -> SolveResult:
        started = time.monotonic()
        config = self.config
        graph = SlotGraph(puzzle)
        generator = CandidateGenerator(
            self.client,
            puzzle,
            graph,
            tracer=self.tracer,
            max_workers=config.max_workers,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        usage = Usage()
        warnings: list[str] = []
        locked: dict[Cell, str] = dict(prefill or {})
        rejected: dict[str, set[str]] = {}
        proposed: dict[str, set[str]] = {}
        domains: dict = {}
        assignment: dict[str, str] = {}
        confidence: dict[str, float] = {}
        result_search = None
        rounds_run = 0
        issues_by_slot: dict[str, list[str]] = {}
        conflicts = []

        targets = list(graph.slot_ids)
        for round_index in range(config.max_rounds):
            if not targets:
                break
            if usage.calls >= config.max_calls:
                warnings.append(f"stopped at the {config.max_calls}-call budget")
                break
            rounds_run = round_index + 1
            self.tracer.emit(
                ROUND_START,
                f"round {round_index}: {len(targets)} slot(s)",
                round=round_index,
                slots=len(targets),
                locked=len(locked),
            )

            patterns = {sid: graph.pattern(sid, locked) for sid in targets}

            model = config.model
            jobs: list = []
            pending_stars = []
            if round_index == 0:
                batches = batch_by_locality(graph, targets, config.batch_size)
                jobs = [
                    (
                        batch,
                        _first_pass_builder(puzzle, graph, batch, patterns),
                        patterns,
                    )
                    for batch in batches
                ]
                model = config.model
            else:
                stars = collect_stars(graph, conflicts, targets)
                need_query = []
                for star in stars:
                    local = solve_star(
                        domains,
                        graph,
                        star,
                        assignment,
                        locked,
                        seed=(config.seed or 0) + round_index,
                        require_words=config.require_real_words,
                    )
                    if star_complete(local, star):
                        apply_star(local, star, assignment, confidence, domains)
                        self.tracer.emit(
                            REPAIR,
                            f"star {star.hubs[0]}x{star.hubs[1]} solved from "
                            f"existing candidates ({len(star.slots)} slots)",
                            round=round_index,
                            slots=list(star.slots)[:12],
                        )
                    else:
                        need_query.append(star)
                new_locks = self._lock_cells(assignment, confidence, graph, locked)
                locked.update(new_locks)
                if need_query:
                    likely = {
                        sid: self._likely_pattern(sid, assignment, graph, locked)
                        for star in need_query
                        for sid in star.slots
                    }
                    for star in need_query:
                        star_patterns = {
                            sid: graph.pattern(sid, locked) for sid in star.slots
                        }
                        jobs.append(
                            (
                                list(star.slots),
                                _star_builder(
                                    puzzle,
                                    graph,
                                    star,
                                    star_patterns,
                                    {s: sorted(rejected.get(s, set())) for s in star.slots},
                                    {
                                        s: [
                                            c.answer
                                            for c in (
                                                domains.get(s).candidates
                                                if domains.get(s)
                                                else []
                                            )[:6]
                                        ]
                                        for s in star.slots
                                    },
                                    likely,
                                    {s: issues_by_slot.get(s, []) for s in star.slots},
                                ),
                                star_patterns,
                            )
                        )
                    model = config.repair_model_or_default
                    self.tracer.emit(
                        REPAIR,
                        f"re-querying {len(need_query)} star(s): "
                        + ", ".join(f"{s.hubs[0]}x{s.hubs[1]}" for s in need_query[:6]),
                        round=round_index,
                        slots=[h for star in need_query for h in star.hubs][:12],
                        model=model,
                    )
                    pending_stars = need_query

            results = generator.run(model, jobs, round_index) if jobs else []
            fresh: list[Candidate] = []
            for item in results:
                fresh.extend(item.candidates)
                warnings.extend(item.warnings[:2])
                if item.completion is not None:
                    usage.record(item.completion)
                if item.error:
                    warnings.append(item.error)

            if round_index == 0 and config.ensemble_model:
                # A second family's errors are decorrelated from the first's,
                # so agreement between them is worth more than either alone.
                other = generator.run(config.ensemble_model, jobs, round_index)
                second: list[Candidate] = []
                for item in other:
                    second.extend(item.candidates)
                    if item.completion is not None:
                        usage.record(item.completion)
                fresh = merge_candidates([fresh, second])

            definitions = self._load_definitions()
            if definitions and fresh:
                fresh = rescore_with_dictionary(fresh, puzzle, definitions)

            fresh = [
                c
                for c in fresh
                if c.answer not in rejected.get(c.slot_id, ())
            ]
            for candidate in fresh:
                proposed.setdefault(candidate.slot_id, set()).add(candidate.answer)
                rejected.setdefault(candidate.slot_id, set())

            incoming = build_domains(graph, fresh, unknown_mass=config.unknown_mass)
            domains = merge_domains(domains, incoming) if domains else incoming
            for slot_id, banned in rejected.items():
                domain = domains.get(slot_id)
                if domain is not None and banned:
                    domain.candidates = [
                        c for c in domain.candidates if c.answer not in banned
                    ]

            conflicts = []
            if config.use_constraints:
                domains = pattern_filter(domains, graph, locked)
                domains, conflicts = soft_ac3(domains, graph)
                self.tracer.emit(
                    CONSTRAINTS,
                    f"{sum(len(d.candidates) for d in domains.values())} candidates, "
                    f"{len(conflicts)} conflict site(s)",
                    round=round_index,
                    conflicts=[list(c.slots) for c in conflicts][:8],
                )

            if round_index > 0 and pending_stars:
                still = []
                for star in pending_stars:
                    local = solve_star(
                        domains,
                        graph,
                        star,
                        assignment,
                        locked,
                        seed=(config.seed or 0) + round_index + 17,
                        require_words=config.require_real_words,
                    )
                    if star_complete(local, star):
                        apply_star(local, star, assignment, confidence, domains)
                        self.tracer.emit(
                            REPAIR,
                            f"star {star.hubs[0]}x{star.hubs[1]} solved after re-query",
                            round=round_index,
                            slots=list(star.slots)[:12],
                        )
                    else:
                        still.append(star)
                for star in still:
                    reason = fallback_star(
                        star, assignment, confidence, domains, rejected
                    )
                    self.tracer.emit(
                        REPAIR,
                        f"star {star.hubs[0]}x{star.hubs[1]} exhausted; {reason}",
                        round=round_index,
                        slots=list(star.hubs),
                    )
                new_locks = self._lock_cells(assignment, confidence, graph, locked)
                if new_locks:
                    locked.update(new_locks)
                    domains = pattern_filter(domains, graph, locked)

            if config.use_constraints:
                result_search = search_solve(
                    domains,
                    graph,
                    seed=(config.seed or 0) + round_index,
                    max_nodes=config.search_nodes,
                    restarts=config.search_restarts,
                    require_words=config.require_real_words,
                )
                assignment = dict(result_search.assignment)
                confidence = dict(result_search.confidence)
            else:
                # Arm A1: take each slot's top candidate, ignore crossings.
                assignment = {}
                confidence = {}
                for slot_id, domain in domains.items():
                    best = domain.best()
                    assignment[slot_id] = best.answer if best else WILDCARD
                    confidence[slot_id] = best.confidence if best else 0.0

            filled = {s: a for s, a in assignment.items() if a != WILDCARD}
            agree, total = intersection_consistency(filled, graph)
            self.tracer.emit(
                SEARCH,
                f"{len(filled)}/{len(graph.slot_ids)} slots, ICR "
                f"{agree}/{total or 0}",
                round=round_index,
                filled=len(filled),
                icr=(agree / total) if total else 1.0,
                nodes=getattr(result_search, "nodes", 0),
                nogoods=getattr(result_search, "nogoods", 0),
                grid=puzzle.grid.render(cells_from_assignment(assignment, graph)),
            )

            # Only answers that contradict locked cells go on the rejected
            # list. A conflict site means the *crossing* may be wrong, so
            # banning this slot's current fill can delete the gold answer.
            for slot_id, answer in assignment.items():
                if answer == WILDCARD:
                    continue
                confirmed = graph.pattern(slot_id, locked)
                breaks_pattern = any(
                    want not in ("?", got) for got, want in zip(answer, confirmed)
                )
                if breaks_pattern:
                    rejected.setdefault(slot_id, set()).add(answer)

            new_locks = self._lock_cells(assignment, confidence, graph, locked)
            if new_locks:
                locked.update(new_locks)
                self.tracer.emit(
                    LOCKED,
                    f"locked {len(new_locks)} cell(s), {len(locked)} total",
                    round=round_index,
                    cells=len(new_locks),
                )

            # Definition checks are only useful when the clue *is* a dictionary
            # sense (the generated corpus). NYT wordplay would false-fail them.
            report = verify_fill(
                puzzle,
                graph,
                assignment,
                definitions=(
                    definitions
                    if puzzle.metadata.get("Source") == "generated"
                    else None
                ),
                proposed=proposed,
                require_words=config.require_real_words,
            )
            if config.require_real_words:
                spelled = implied_spellings(assignment, graph)
                for issue in report.issues:
                    if issue.kind != "word":
                        continue
                    fake = spelled.get(issue.slot_id)
                    if fake:
                        rejected.setdefault(issue.slot_id, set()).add(fake)
                    for crossing in graph.by_slot.get(issue.slot_id, []):
                        other = assignment.get(crossing.b)
                        if other and other != WILDCARD:
                            rejected.setdefault(crossing.b, set()).add(other)
            issues_by_slot = {
                sid: [i.detail for i in report.for_slot(sid)]
                for sid in report.slot_ids()
            }
            self.tracer.emit(
                VERIFY,
                f"{len(report.issues)} issue(s), complete={report.complete}",
                round=round_index,
                issues=[(i.slot_id, i.kind) for i in report.issues[:12]],
            )

            solved = report.ok or (
                len(filled) == len(graph.slot_ids)
                and (total == 0 or agree == total)
                and min(confidence.values(), default=0.0) >= config.repair_confidence
                and not report.issues
            )
            self.tracer.emit(
                ROUND_END,
                f"round {round_index} done",
                round=round_index,
                filled=len(filled),
                locked=len(locked),
                calls=usage.calls,
                tokens=usage.total_tokens,
            )
            if solved or not config.use_repair:
                break

            next_targets = self._repair_targets(
                assignment,
                confidence,
                graph,
                conflicts,
                locked,
                failed=report.slot_ids(),
            )
            if not next_targets:
                break
            if (
                round_index > 0
                and set(next_targets) == set(targets)
                and not report.issues
            ):
                break
            targets = next_targets

        # -- endgame ------------------------------------------------------
        cells = cells_from_assignment(assignment, graph)
        cells.update(locked)
        if config.use_endgame:
            cells = endgame_fill(assignment, graph) | cells

        filled = {s: a for s, a in assignment.items() if a != WILDCARD}
        agree, total = intersection_consistency(filled, graph)

        calibration_pairs: list[tuple[float, bool]] = []
        if puzzle.has_gold():
            for slot_id, answer in filled.items():
                gold = (puzzle.slot(slot_id).gold or "").upper()
                calibration_pairs.append((confidence.get(slot_id, 0.0), answer == gold))

        self.tracer.emit(
            SOLVED,
            f"{len(filled)}/{len(graph.slot_ids)} slots in {rounds_run} round(s)",
            round=rounds_run,
            grid=puzzle.grid.render(cells),
            calls=usage.calls,
            tokens=usage.total_tokens,
        )

        return SolveResult(
            puzzle=puzzle,
            solution=cells,
            assignment=assignment,
            confidence=confidence,
            locked=locked,
            rounds=rounds_run,
            usage=usage,
            icr=(agree / total) if total else 1.0,
            open_slots=sorted(s for s, a in assignment.items() if a == WILDCARD),
            seconds=time.monotonic() - started,
            warnings=warnings,
            calibration_pairs=calibration_pairs,
            rungs=dict(generator.rung_by_model),
        )


def _first_pass_builder(puzzle, graph, batch, patterns):
    def build(*, schema_in_prompt: bool = False):
        return prompts.first_pass_messages(
            puzzle, graph, batch, patterns, schema_in_prompt=schema_in_prompt
        )

    return build


def _repair_builder(puzzle, graph, batch, patterns, rejected, context, likely, issues=None):
    def build(*, schema_in_prompt: bool = False):
        return prompts.repair_messages(
            puzzle,
            graph,
            batch,
            patterns,
            rejected,
            context,
            likely=likely,
            issues=issues,
            schema_in_prompt=schema_in_prompt,
        )

    return build


def _star_builder(puzzle, graph, star, patterns, rejected, current, likely, issues=None):
    def build(*, schema_in_prompt: bool = False):
        return prompts.star_repair_messages(
            puzzle,
            graph,
            list(star.slots),
            star.hubs,
            patterns,
            rejected,
            current,
            likely=likely,
            issues=issues,
            schema_in_prompt=schema_in_prompt,
        )

    return build
