"""Turn results.json into a readable summary.

Every number in the write-up is generated from here rather than transcribed by
hand, so the report cannot drift from the run that produced it.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

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

    # -- headline table ---------------------------------------------------
    lines.append("## Arms")
    lines.append("")
    headers = ["Arm", "Description"] + [label for _, label in METRIC_COLUMNS]
    headers += ["Open", "Calls", "Tokens", "Sec"]
    rows = []
    for name in arm_names:
        group = by_arm.get(name, [])
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
            a = [r["scores"]["wcr"] for r in by_arm.get(high, [])]
            b = [r["scores"]["wcr"] for r in by_arm.get(low, [])]
            if len(a) != len(b) or not a:
                continue
            result = paired_bootstrap(a, b, resamples=10_000, seed=1)
            exact_a = [r["scores"]["exact"] for r in by_arm[high]]
            exact_b = [r["scores"]["exact"] for r in by_arm[low]]
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
        group = by_arm.get(name, [])
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
            subset = [
                r for r in by_arm.get(name, []) if r["strata"].get("size") == size
            ]
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
                subset = [
                    r
                    for r in by_arm.get(name, [])
                    if r["strata"].get("provenance") == provenance
                ]
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
    return path
