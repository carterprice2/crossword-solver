# 60-second demo: shot list and script

The whole demo runs offline (`--backend oracle`), so it needs no API key, costs
nothing, and cannot fail on camera because of a rate limit. Use
`--backend replay --replay <trace>` if you want a *live-model* recording that is
still bit-for-bit reproducible.

## Setup

```bash
# From the repo root. ~100x30 terminal, large font. Check it fits before recording:
FORCE_COLOR=1 python3 -m crossword solve corpus/mini/mini-11-04-0.xd \
    --backend oracle --oracle-recall 0.6 --oracle-top1-error 0.5 --live
```

`--oracle-recall 0.6` means 40% of the correct answers are missing from the
first pass. That is the point: the repair rounds visibly recover them, so the
mechanism is on screen rather than merely described.

## Shot list

| Time | Shot | Say |
|---|---|---|
| 0:00–0:08 | Terminal, type `make demo` | "A crossword isn't trivia — it's a constraint problem where the answers come from a language model." |
| 0:08–0:20 | Round 0 sweeps the grid, cells turn yellow | "Round one asks the model for every clue cold. It batches clues that actually cross each other, so the model can check itself." |
| 0:20–0:35 | **Repair lines appear; conflicts listed; cells turn green** | "Then the crossings vote. Where two confident answers agree, the letter locks — green. Where they conflict, the agent goes *back* to the model with the letters it now knows and the answers it already ruled out." |
| 0:35–0:45 | Frame flips to **gold check**: every letter green, `SOLVED WCR 1.000` | "Forty percent of the correct answers were missing from that first pass. The loop recovered them. Green now means *correct*, not just locked." |
| 0:45–0:60 | Cut to `results/synthetic-sweep.json` table or REPORT.md | "And that's the measurement: the worse the model, the more the loop is worth — plus two-tenths of a point of WCR at the low end." |

## The one number to say out loud

> The repair loop adds **+0.02 WCR when candidates are good and +0.29 when
> they're bad.** Its value grows exactly as the base model gets weaker.

That is the finding worth a follow-up conversation: it says when the extra
tokens are justified, and when a single pass is enough.

## Recording notes

- `FORCE_COLOR=1` is already set by `make demo`; without a TTY the view falls
  back to plain scrolling output, which reads badly on video.
- The in-place redraw needs a terminal at least as tall as the grid plus 12
  lines. An 11×11 wants ~30 rows.
- If a take runs long, use `mini-09-00-0.xd` (9×9, ~2s) instead of the 11×11.
- For a live-model take: `--record trace.jsonl` once, then
  `--backend replay --replay trace.jsonl --live` for every subsequent take.
