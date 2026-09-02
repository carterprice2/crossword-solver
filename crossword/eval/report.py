"""Turn results.json into a readable summary.

Every number in the write-up is generated from here rather than transcribed by
hand, so the report cannot drift from the run that produced it.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from .recipes import rank_keys, winners_payload, write_winners
from .stats import mcnemar, paired_bootstrap, required_n, wilson

METRIC_COLUMNS = (
    ("wcr", "WCR"),
    ("lcr", "LCR"),
    ("icr", "ICR"),
    ("exact", "Exact"),
    ("cell_precision", "Prec"),
    ("cell_recall", "Rec"),
)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return "\n".join(out)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ok(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("scores")]


def _short_model(name: str) -> str:
    if not name:
        return "?"
    return name.split("/")[-1]


def _fmt_usd(value) -> str:
    if value is None:
        return "?"
    return f"{float(value):.3f}"


def _cell_row(record: dict) -> list[str]:
    scores = record.get("scores")
    solve = record.get("solve") or {}
    size = (record.get("strata") or {}).get("size", "?")
    model = _short_model(record.get("model") or "")
    tokens = int(
        (solve.get("prompt_tokens") or 0) + (solve.get("completion_tokens") or 0)
    )
    turns = str(solve.get("rounds") or 0)
    calls = str(solve.get("calls") or 0)
    sec = f"{float(solve.get('seconds') or 0):.1f}"
    usd = _fmt_usd(solve.get("cost_usd"))
    base = [size, record.get("puzzle_id", "?"), model, record.get("arm", "?")]
    if not scores:
        return base + ["err", "err", "err", "err", str(tokens), usd, turns, calls, sec]
    return base + [
        f"{scores['wcr']:.3f}",
        f"{scores['lcr']:.3f}",
        f"{scores['icr']:.3f}",
        "1" if scores.get("exact") else "0",
        str(tokens),
        usd,
        turns,
        calls,
        sec,
    ]


def _leaderboard_rows(records: list[dict], rank_by: str) -> list[list[str]]:
    if rank_by == "pair":
        keyed = []
        for record in records:
            row = dict(record)
            row["_pair"] = f"{_short_model(record.get('model') or '')} {record.get('arm')}"
            keyed.append(row)
        names = rank_keys(keyed, "_pair")
        groups = defaultdict(list)
        for record in keyed:
            if record.get("scores"):
                groups[record["_pair"]].append(record)
        label = "model arm"
    else:
        names = rank_keys(records, rank_by)
        groups = defaultdict(list)
        for record in _ok(records):
            groups[record[rank_by]].append(record)
        label = rank_by
    rows = []
    for index, name in enumerate(names, start=1):
        group = groups[name]
        wcr = _mean([r["scores"]["wcr"] for r in group])
        costs = [r.get("solve", {}).get("cost_usd") for r in group]
        usd = (
            _fmt_usd(sum(costs) / len(costs))
            if costs and all(c is not None for c in costs)
            else "?"
        )
        turns = _mean([r.get("solve", {}).get("rounds") or 0 for r in group])
        sec = _mean([r.get("solve", {}).get("seconds") or 0 for r in group])
        rows.append(
            [str(index), name, f"{wcr:.3f}", usd, f"{turns:.1f}", f"{sec:.1f}"]
        )
    return label, rows


def winners_from_payload(payload: dict) -> dict:
    records = payload.get("records") or []
    stage = payload.get("stage") or "eval"
    if stage == "screen-models":
        models = rank_keys(records, "model")
        arms = list(payload.get("carry_arms") or []) or rank_keys(records, "arm")
        return winners_payload(stage, arms=arms, models=models)
    if stage == "screen-arms":
        arms = rank_keys(records, "arm")
        models = []
        seen: set[str] = set()
        for record in records:
            model = record.get("model")
            if model and model not in seen:
                seen.add(model)
                models.append(model)
        return winners_payload(stage, arms=arms, models=models)
    return winners_payload(
        stage,
        arms=rank_keys(records, "arm"),
        models=rank_keys(records, "model"),
    )


def _json_misses(records: list[dict]) -> int:
    return sum(
        1
        for record in records
        for warning in (record.get("solve") or {}).get("warnings") or []
        if "no JSON object" in str(warning)
    )


def _pick_lines(
    records: list[dict], arms: dict, rank_by: str
) -> list[str]:
    ranked = rank_keys(records, rank_by)
    if not ranked:
        return []
    pick = ranked[0]
    label = (arms.get(pick) or {}).get("label") or pick
    lines = [
        "## Pick",
        "",
        f"**Use `{pick}`.** It ranked first by WCR ({label}).",
        "",
    ]
    by_name = defaultdict(list)
    for record in _ok(records):
        by_name[record[rank_by]].append(record)
    for name in ranked:
        group = by_name.get(name) or []
        wcr = _mean([r["scores"]["wcr"] for r in group])
        open_slots = _mean(
            [len((r.get("solve") or {}).get("open_slots") or []) for r in group]
        )
        mark = " ← use this" if name == pick else ""
        lines.append(
            f"- `{name}` WCR {wcr:.3f}, open slots {open_slots:.0f}{mark}"
        )
    lines.append("")
    misses = _json_misses(records)
    if misses:
        lines.append(
            f"{misses} warning(s) were `no JSON object found in response`. "
            "When the model returns no candidates, forcing a guess (a6) cannot "
            "beat declining (a5) — there is nothing to guess from."
        )
        lines.append("")
    return lines


def summarize(payload: dict) -> str:
    records = payload["records"]
    arms = payload["arms"]
    arm_names = list(arms)
    by_arm = defaultdict(list)
    for record in records:
        by_arm[record["arm"]].append(record)

    lines: list[str] = []
    lines.append(f"# Evaluation summary -- {payload['run_id']}")
    lines.append("")
    lines.append(f"Generated {payload.get('generated_at', '')}")
    lines.append("")
    puzzles = payload.get("puzzles", [])
    sizes = defaultdict(int)
    for puzzle in puzzles:
        sizes[puzzle.get("size", "?")] += 1
    lines.append(
        f"{len(puzzles)} puzzle(s) "
        f"({', '.join(f'{n} x {s}' for s, n in sorted(sizes.items()))}), "
        f"seeds {payload.get('seeds')}, prefill {payload.get('prefill_ratios')}."
    )
    lines.append("")

    # -- per-cell grid ----------------------------------------------------
    if records:
        lines.append("## Results grid")
        lines.append("")
        grid_headers = [
            "size", "puzzle", "model", "arm", "WCR", "LCR", "ICR",
            "exact", "tokens", "USD", "turns", "calls", "sec",
        ]
        lines.append(_table(grid_headers, [_cell_row(r) for r in records]))
        lines.append("")
        rank_by = payload.get("rank_by") or "arm"
        label, board = _leaderboard_rows(records, rank_by)
        if board:
            lines.append("## Leaderboard")
            lines.append("")
            lines.append(
                _table(["rank", label, "WCR", "USD", "turns", "sec"], board)
            )
            lines.append("")
            lines.append("Ranked by mean WCR. Cost is a tie-break only.")
            lines.append("")
            lines.extend(_pick_lines(records, arms, rank_by))

    # -- headline table ---------------------------------------------------
    lines.append("## Arms")
    lines.append("")
    headers = ["Arm", "Description"] + [label for _, label in METRIC_COLUMNS]
    headers += ["Open", "Calls", "Tokens", "Sec"]
    rows = []
    for name in arm_names:
        group = _ok(by_arm.get(name, []))
        if not group:
            continue
        row = [name, arms[name]["label"]]
        for key, _ in METRIC_COLUMNS:
            if key == "exact":
                value = _mean([1.0 if r["scores"]["exact"] else 0.0 for r in group])
            else:
                value = _mean([r["scores"][key] for r in group])
            row.append(f"{value:.3f}")
        row.append(f"{_mean([len(r['solve']['open_slots']) for r in group]):.1f}")
        row.append(f"{_mean([r['solve']['calls'] for r in group]):.1f}")
        row.append(
            f"{_mean([r['solve']['prompt_tokens'] + r['solve']['completion_tokens'] for r in group]):.0f}"
        )
        row.append(f"{_mean([r['solve']['seconds'] for r in group]):.1f}")
        rows.append(row)
    lines.append(_table(headers, rows))
    lines.append("")
    lines.append(
        "`Open` is the mean number of slots the solver declined to answer. "
        "Read it with `Prec`: declining is not the same failure as guessing "
        "wrong, and only the pair distinguishes them."
    )
    lines.append("")

    # -- paired comparisons ----------------------------------------------
    ladder = [(arm_names[i], arm_names[i - 1]) for i in range(1, len(arm_names))]
    if ladder:
        lines.append("## Paired comparisons (WCR)")
        lines.append("")
        lines.append(
            "Every arm ran the same puzzles at the same seeds, so these are "
            "paired differences -- between-puzzle variance is removed rather "
            "than averaged over."
        )
        lines.append("")
        comparison_rows = []
        for high, low in ladder:
            a = [r["scores"]["wcr"] for r in _ok(by_arm.get(high, []))]
            b = [r["scores"]["wcr"] for r in _ok(by_arm.get(low, []))]
            if len(a) != len(b) or not a:
                continue
            result = paired_bootstrap(a, b, resamples=10_000, seed=1)
            exact_a = [r["scores"]["exact"] for r in _ok(by_arm[high])]
            exact_b = [r["scores"]["exact"] for r in _ok(by_arm[low])]
            test = mcnemar(exact_a, exact_b)
            comparison_rows.append(
                [
                    f"{high} - {low}",
                    f"{result.delta.point:+.3f}",
                    f"[{result.delta.low:+.3f}, {result.delta.high:+.3f}]",
                    f"{result.p_better:.3f}",
                    "yes" if result.significant else "no",
                    f"{test['p_value']:.3f}",
                ]
            )
        lines.append(
            _table(
                ["Contrast", "dWCR", "95% CI", "P(>0)", "Sig", "McNemar p (exact)"],
                comparison_rows,
            )
        )
        lines.append("")

    # -- exact-solve power note -------------------------------------------
    lines.append("## Exact-solve, with its uncertainty")
    lines.append("")
    exact_rows = []
    for name in arm_names:
        group = _ok(by_arm.get(name, []))
        if not group:
            continue
        successes = sum(1 for r in group if r["scores"]["exact"])
        interval = wilson(successes, len(group))
        exact_rows.append(
            [
                name,
                f"{successes}/{len(group)}",
                f"{interval.point:.3f}",
                f"[{interval.low:.3f}, {interval.high:.3f}]",
                f"{interval.high - interval.low:.3f}",
            ]
        )
    lines.append(_table(["Arm", "Solved", "Rate", "95% Wilson CI", "Width"], exact_rows))
    lines.append("")
    lines.append(
        f"Exact-solve is one Bernoulli trial per puzzle, so its interval is "
        f"wide at these sample sizes -- telling 80% from 70% apart at 80% power "
        f"needs about {required_n(0.8, 0.7)} puzzles. Treat the rates above as "
        f"descriptive and read the paired McNemar column for arm-vs-arm claims."
    )
    lines.append("")

    # -- per-stratum -------------------------------------------------------
    slot_records = payload.get("slot_records", [])
    if slot_records:
        for field, title in (
            ("clue_type", "clue type"),
            ("length_bucket", "answer length"),
        ):
            lines.append(f"## WCR by {title}")
            lines.append("")
            buckets = sorted({row[field] for row in slot_records})
            headers = ["Arm"] + buckets
            rows = []
            for name in arm_names:
                row = [name]
                for bucket in buckets:
                    subset = [
                        r
                        for r in slot_records
                        if r["arm"] == name and r[field] == bucket
                    ]
                    if subset:
                        rate = sum(1 for r in subset if r["correct"]) / len(subset)
                        row.append(f"{rate:.3f} ({len(subset)})")
                    else:
                        row.append("-")
                rows.append(row)
            lines.append(_table(headers, rows))
            lines.append("")

    # -- by puzzle size ----------------------------------------------------
    lines.append("## WCR by grid size")
    lines.append("")
    sizes_seen = sorted({r["strata"].get("size", "?") for r in records})
    headers = ["Arm"] + sizes_seen
    rows = []
    for name in arm_names:
        row = [name]
        for size in sizes_seen:
            subset = _ok(
                [r for r in by_arm.get(name, []) if r["strata"].get("size") == size]
            )
            row.append(f"{_mean([r['scores']['wcr'] for r in subset]):.3f}" if subset else "-")
        rows.append(row)
    lines.append(_table(headers, rows))
    lines.append("")

    # -- provenance (contamination) ---------------------------------------
    provenances = sorted({r["strata"].get("provenance", "unknown") for r in records})
    if len(provenances) > 1:
        lines.append("## Generated vs published (contamination check)")
        lines.append("")
        lines.append(
            "Generated puzzles have never been published, so no model can have "
            "memorized them. A large positive gap on published puzzles is a "
            "memorization signal, not a capability signal."
        )
        lines.append("")
        rows = []
        for name in arm_names:
            row = [name]
            for provenance in provenances:
                subset = _ok(
                    [
                        r
                        for r in by_arm.get(name, [])
                        if r["strata"].get("provenance") == provenance
                    ]
                )
                row.append(
                    f"{_mean([r['scores']['wcr'] for r in subset]):.3f}" if subset else "-"
                )
            rows.append(row)
        lines.append(_table(["Arm"] + provenances, rows))
        lines.append("")

    # -- calibration -------------------------------------------------------
    calibrations = payload.get("calibration", {})
    if any(c.get("n") for c in calibrations.values()):
        lines.append("## Confidence calibration")
        lines.append("")
        lines.append(
            "The constraint layer consumes self-reported confidence as a "
            "probability, so a confidently-wrong model actively damages the "
            "search. Lower ECE and Brier are better."
        )
        lines.append("")
        rows = []
        for name in arm_names:
            data = calibrations.get(name) or {}
            if not data.get("n"):
                continue
            rows.append(
                [name, f"{data['ece']:.3f}", f"{data['brier']:.3f}", str(data["n"])]
            )
        if rows:
            lines.append(_table(["Arm", "ECE", "Brier", "n"], rows))
            lines.append("")

    # -- structured output support ----------------------------------------
    rungs: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for model, rung in (record["solve"].get("rungs") or {}).items():
            rungs[model].add(rung)
    if rungs:
        lines.append("## Structured-output support")
        lines.append("")
        lines.append(
            "Which rung of the schema ladder each model accepted. A model that "
            "falls back to free text needs the lenient parser to be usable at all."
        )
        lines.append("")
        lines.append(
            _table(
                ["Model", "Rung(s) used"],
                [[model, ", ".join(sorted(v))] for model, v in sorted(rungs.items())],
            )
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_summary(directory: str) -> str:
    with open(os.path.join(directory, "results.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    text = summarize(payload)
    path = os.path.join(directory, "summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    write_winners(directory, winners_from_payload(payload))
    return path
