# Evaluation summary -- live-max-correct-arms-11

Generated 2026-09-02T05:53:14Z

1 puzzle(s) (1 x 11x11), seeds [0], prefill [0.0].

## Results grid

| size  | puzzle       | model             | arm | WCR   | LCR   | ICR   | exact | tokens | USD   | turns | calls | sec  |
|-------|--------------|-------------------|-----|-------|-------|-------|-------|--------|-------|-------|-------|------|
| 11x11 | mini-11-04-0 | Qwen3.5-397B-A17B | a5  | 0.095 | 0.151 | 1.000 | 0     | 64638  | 0.165 | 5     | 21    | 93.0 |
| 11x11 | mini-11-04-0 | Qwen3.5-397B-A17B | a6  | 0.000 | 0.000 | 1.000 | 0     | 65659  | 0.168 | 5     | 21    | 92.8 |

## Leaderboard

| rank | arm | WCR   | USD   | turns | sec  |
|------|-----|-------|-------|-------|------|
| 1    | a5  | 0.095 | 0.165 | 5.0   | 93.0 |
| 2    | a6  | 0.000 | 0.168 | 5.0   | 92.8 |

Ranked by mean WCR. Cost is a tie-break only.

## Pick

**Use `a5`.** It ranked first by WCR (reasoning model throughout).

- `a5` WCR 0.095, open slots 38 ← use this
- `a6` WCR 0.000, open slots 42

18 warning(s) were `no JSON object found in response`. When the model returns no candidates, forcing a guess (a6) cannot beat declining (a5) — there is nothing to guess from.

## Arms

| Arm | Description                | WCR   | LCR   | ICR   | Exact | Prec  | Rec   | Open | Calls | Tokens | Sec  |
|-----|----------------------------|-------|-------|-------|-------|-------|-------|------|-------|--------|------|
| a5  | reasoning model throughout | 0.095 | 0.151 | 1.000 | 0.000 | 1.000 | 0.151 | 38.0 | 21.0  | 64638  | 93.0 |
| a6  | no wildcard (must guess)   | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 42.0 | 21.0  | 65659  | 92.8 |

`Open` is the mean number of slots the solver declined to answer. Read it with `Prec`: declining is not the same failure as guessing wrong, and only the pair distinguishes them.

## Paired comparisons (WCR)

Every arm ran the same puzzles at the same seeds, so these are paired differences -- between-puzzle variance is removed rather than averaged over.

| Contrast | dWCR   | 95% CI           | P(>0) | Sig | McNemar p (exact) |
|----------|--------|------------------|-------|-----|-------------------|
| a6 - a5  | -0.095 | [-0.095, -0.095] | 0.000 | yes | 1.000             |

## Exact-solve, with its uncertainty

| Arm | Solved | Rate  | 95% Wilson CI  | Width |
|-----|--------|-------|----------------|-------|
| a5  | 0/1    | 0.000 | [0.000, 0.793] | 0.793 |
| a6  | 0/1    | 0.000 | [0.000, 0.793] | 0.793 |

Exact-solve is one Bernoulli trial per puzzle, so its interval is wide at these sample sizes -- telling 80% from 70% apart at 80% power needs about 294 puzzles. Treat the rates above as descriptive and read the paired McNemar column for arm-vs-arm claims.

## WCR by clue type

| Arm | definition | proper     |
|-----|------------|------------|
| a5  | 0.129 (31) | 0.000 (11) |
| a6  | 0.000 (31) | 0.000 (11) |

## WCR by answer length

| Arm | 3          | 4-5        | 6-8       |
|-----|------------|------------|-----------|
| a5  | 0.100 (10) | 0.107 (28) | 0.000 (4) |
| a6  | 0.000 (10) | 0.000 (28) | 0.000 (4) |

## WCR by grid size

| Arm | 11x11 |
|-----|-------|
| a5  | 0.095 |
| a6  | 0.000 |

## Structured-output support

Which rung of the schema ladder each model accepted. A model that falls back to free text needs the lenient parser to be usable at all.

| Model                  | Rung(s) used  |
|------------------------|---------------|
| Qwen/Qwen3.5-397B-A17B | strict_schema |

