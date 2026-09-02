# Evaluation summary -- live-screen-models-11

Generated 2026-09-02T06:58:57Z

1 puzzle(s) (1 x 11x11), seeds [0], prefill [0.0].

## Results grid

| size  | puzzle       | model      | arm | WCR   | LCR   | ICR   | exact | tokens | USD   | turns | calls | sec  |
|-------|--------------|------------|-----|-------|-------|-------|-------|--------|-------|-------|-------|------|
| 11x11 | mini-11-04-0 | MiniMax-M3 | a5  | 1.000 | 1.000 | 1.000 | 1     | 23251  | 0.011 | 5     | 15    | 18.6 |

## Leaderboard

| rank | model                | WCR   | USD   | turns | sec  |
|------|----------------------|-------|-------|-------|------|
| 1    | MiniMaxAI/MiniMax-M3 | 1.000 | 0.011 | 5.0   | 18.6 |

Ranked by mean WCR. Cost is a tie-break only.

## Pick

**Use `MiniMaxAI/MiniMax-M3`.** It ranked first by WCR (MiniMaxAI/MiniMax-M3).

- `MiniMaxAI/MiniMax-M3` WCR 1.000, open slots 0 ← use this

## Arms

| Arm | Description                | WCR   | LCR   | ICR   | Exact | Prec  | Rec   | Open | Calls | Tokens | Sec  |
|-----|----------------------------|-------|-------|-------|-------|-------|-------|------|-------|--------|------|
| a5  | reasoning model throughout | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0  | 15.0  | 23251  | 18.6 |

`Open` is the mean number of slots the solver declined to answer. Read it with `Prec`: declining is not the same failure as guessing wrong, and only the pair distinguishes them.

## Exact-solve, with its uncertainty

| Arm | Solved | Rate  | 95% Wilson CI  | Width |
|-----|--------|-------|----------------|-------|
| a5  | 1/1    | 1.000 | [0.207, 1.000] | 0.793 |

Exact-solve is one Bernoulli trial per puzzle, so its interval is wide at these sample sizes -- telling 80% from 70% apart at 80% power needs about 294 puzzles. Treat the rates above as descriptive and read the paired McNemar column for arm-vs-arm claims.

## WCR by clue type

| Arm | definition | proper     |
|-----|------------|------------|
| a5  | 1.000 (31) | 1.000 (11) |

## WCR by answer length

| Arm | 3          | 4-5        | 6-8       |
|-----|------------|------------|-----------|
| a5  | 1.000 (10) | 1.000 (28) | 1.000 (4) |

## WCR by grid size

| Arm | 11x11 |
|-----|-------|
| a5  | 1.000 |

## Confidence calibration

The constraint layer consumes self-reported confidence as a probability, so a confidently-wrong model actively damages the search. Lower ECE and Brier are better.

| Arm | ECE   | Brier | n  |
|-----|-------|-------|----|
| a5  | 0.064 | 0.007 | 42 |

## Structured-output support

Which rung of the schema ladder each model accepted. A model that falls back to free text needs the lenient parser to be usable at all.

| Model                | Rung(s) used  |
|----------------------|---------------|
| MiniMaxAI/MiniMax-M3 | strict_schema |

