#!/usr/bin/env python3
"""Regenerate REPORT.md result tables from committed result files.

The tables live between BEGIN/END markers and are rewritten in place, so no
number in the report is ever transcribed by hand and the report cannot drift
from the run that produced it.

    python3 scripts/update_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ARMS = ["a0", "a1", "a2", "a3", "a6"]
LABELS = {
    "a0": "a0 one-shot",
    "a1": "a1 per-clue",
    "a2": "a2 +constraints",
    "a3": "a3 full agent",
    "a6": "a6 no wildcard",
}


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return "\n".join(out)


def sweep_table(data: dict) -> str:
    headers = ["candidate recall", "top-1 error"] + [LABELS[a] for a in ARMS]
    rows = []
    for entry in data["sweep"]:
        row = [f"{entry['recall']:.2f}", f"{entry['top1_error']:.2f}"]
        row += [f"{entry['arms'][a]['wcr']:.3f}" for a in ARMS]
        rows.append(row)
    note = (
        "\n\nMean WCR over "
        f"{data['puzzles']} puzzles x {len(data['seeds'])} seeds. "
        "`recall` is the probability the correct answer appears among the "
        "candidates at all; `top-1 error` the probability it is not ranked "
        "first. Lower rows are weaker models."
    )
    return table(headers, rows) + note


def repair_table(data: dict) -> str:
    headers = [
        "candidate recall", "a2 WCR", "a3 WCR", "repair gain", "95% CI", "calls a2 -> a3"
    ]
    rows = []
    for entry in data["sweep"]:
        delta = entry["deltas"]["a3_minus_a2"]["delta"]
        arms = entry["arms"]
        rows.append(
            [
                f"{entry['recall']:.2f}",
                f"{arms['a2']['wcr']:.3f}",
                f"{arms['a3']['wcr']:.3f}",
                f"{delta['point']:+.3f}",
                f"[{delta['low']:+.3f}, {delta['high']:+.3f}]",
                f"{arms['a2']['calls']:.1f} -> {arms['a3']['calls']:.1f}",
            ]
        )
    note = (
        "\n\nPaired bootstrap over puzzles on the difference, 4,000 resamples. "
        "Every interval excludes zero."
    )
    return table(headers, rows) + note


def load_cells(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "puzzle_id" in payload:
                rows.append(payload)
    return rows


def _short(model: str) -> str:
    return model.split("/")[-1] if model else "?"


def _usd(value) -> str:
    if value is None:
        return "?"
    return f"{float(value):.3f}"


def live_model_table(cells: list[dict]) -> str:
    def sort_key(record: dict) -> tuple:
        scores = record.get("scores") or {}
        wcr = scores.get("wcr")
        if wcr is None:
            return (1, 0.0, record.get("model") or "")
        usd = (record.get("solve") or {}).get("cost_usd")
        cost = float(usd) if usd is not None else float("inf")
        return (0, -wcr, cost, record.get("model") or "")

    ordered = sorted(cells, key=sort_key)
    headers = [
        "model", "WCR", "LCR", "exact", "tokens", "USD", "turns", "calls", "sec", "rung",
    ]
    rows = []
    for record in ordered:
        scores = record.get("scores") or {}
        solve = record.get("solve") or {}
        tokens = int((solve.get("prompt_tokens") or 0) + (solve.get("completion_tokens") or 0))
        rungs = solve.get("rungs") or {}
        rung = ", ".join(sorted(set(rungs.values()))) if rungs else "?"
        rows.append(
            [
                _short(record.get("model") or ""),
                f"{scores['wcr']:.3f}",
                f"{scores['lcr']:.3f}",
                "1" if scores.get("exact") else "0",
                str(tokens),
                _usd(solve.get("cost_usd")),
                str(solve.get("rounds") or 0),
                str(solve.get("calls") or 0),
                f"{float(solve.get('seconds') or 0):.1f}",
                rung,
            ]
        )
    note = (
        "\n\nArm `a5` (the screened model at every stage), puzzle "
        "`mini-11-04-0` (11×11, 42 slots), seed 0, prefill 0. "
        "Ranked by WCR, then lower USD. Raw cells: "
        "`results/live-screen-models-11/cells.jsonl`."
    )
    return table(headers, rows) + note


def live_qwen_arms_table(cells: list[dict]) -> str:
    headers = ["arm", "WCR", "LCR", "Prec", "Rec", "open", "USD", "JSON misses"]
    rows = []
    for record in sorted(cells, key=lambda r: r.get("arm") or ""):
        scores = record.get("scores") or {}
        solve = record.get("solve") or {}
        misses = sum(
            1
            for warning in (solve.get("warnings") or [])
            if "no JSON object" in str(warning)
        )
        rows.append(
            [
                record.get("arm") or "?",
                f"{scores['wcr']:.3f}",
                f"{scores['lcr']:.3f}",
                f"{scores['cell_precision']:.3f}",
                f"{scores['cell_recall']:.3f}",
                str(len(solve.get("open_slots") or [])),
                _usd(solve.get("cost_usd")),
                str(misses),
            ]
        )
    note = (
        "\n\n`Qwen/Qwen3.5-397B-A17B` on `mini-11-04-0`, seed 0. "
        "`JSON misses` counts `no JSON object found in response` warnings. "
        "Raw cells: `results/live-max-correct-arms-11/cells.jsonl`."
    )
    return table(headers, rows) + note


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(
        rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(f"no BEGIN/END {name} markers in the report")
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=os.path.join(here, "results", "synthetic-sweep.json"))
    ap.add_argument(
        "--live-models",
        default=os.path.join(here, "results", "live-screen-models-11", "cells.jsonl"),
    )
    ap.add_argument(
        "--live-arms",
        default=os.path.join(here, "results", "live-max-correct-arms-11", "cells.jsonl"),
    )
    ap.add_argument("--report", default=os.path.join(here, "REPORT.md"))
    args = ap.parse_args()

    with open(args.sweep, encoding="utf-8") as fh:
        data = json.load(fh)
    with open(args.report, encoding="utf-8") as fh:
        text = fh.read()

    text = replace_block(text, "SWEEP_TABLE", sweep_table(data))
    text = replace_block(text, "REPAIR_DELTA", repair_table(data))
    text = replace_block(text, "LIVE_MODEL_GRID", live_model_table(load_cells(args.live_models)))
    text = replace_block(text, "LIVE_QWEN397_ARMS", live_qwen_arms_table(load_cells(args.live_arms)))
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"updated {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
