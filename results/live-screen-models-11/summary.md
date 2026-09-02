# Evaluation summary -- live-screen-models-11

Generated 2026-09-02T08:11:49Z

1 puzzle(s) (1 x 11x11), seeds [0], prefill [0.0].

## Results grid

| size  | puzzle       | model                         | arm | WCR   | LCR   | ICR   | exact | tokens | USD   | turns | calls | sec   |
|-------|--------------|-------------------------------|-----|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| 11x11 | mini-11-04-0 | Qwen3-30B-A3B-Instruct-2507   | a5  | 0.524 | 0.763 | 1.000 | 0     | 51891  | 0.011 | 5     | 21    | 300.9 |
| 11x11 | mini-11-04-0 | Qwen3-235B-A22B-Instruct-2507 | a5  | 0.833 | 0.946 | 1.000 | 0     | 28044  | 0.010 | 5     | 14    | 122.7 |
| 11x11 | mini-11-04-0 | Qwen3.5-397B-A17B             | a5  | 0.952 | 0.989 | 1.000 | 0     | 48364  | 0.143 | 3     | 10    | 554.6 |
| 11x11 | mini-11-04-0 | Llama-3.3-70B-Instruct        | a5  | 0.619 | 0.806 | 1.000 | 0     | 28645  | 0.006 | 5     | 19    | 146.2 |
| 11x11 | mini-11-04-0 | gpt-oss-120b                  | a5  | 0.881 | 0.957 | 1.000 | 0     | 51634  | 0.026 | 3     | 11    | 171.8 |
| 11x11 | mini-11-04-0 | DeepSeek-V4-Pro               | a5  | 1.000 | 1.000 | 1.000 | 1     | 14012  | 0.034 | 3     | 9     | 24.3  |
| 11x11 | mini-11-04-0 | MiniMax-M3                    | a5  | 1.000 | 1.000 | 1.000 | 1     | 23251  | 0.011 | 5     | 15    | 18.6  |

## Leaderboard

| rank | model                       | WCR   | USD   | turns | sec   |
|------|-----------------------------|-------|-------|-------|-------|
| 1    | MiniMaxAI/MiniMax-M3        | 1.000 | 0.011 | 5.0   | 18.6  |
| 2    | deepseek-ai/DeepSeek-V4-Pro | 1.000 | 0.034 | 3.0   | 24.3  |
| 3    | Qwen/Qwen3.5-397B-A17B      | 0.952 | 0.143 | 3.0   | 554.6 |

Ranked by mean WCR. Cost is a tie-break only.

## Pick

**Use `MiniMaxAI/MiniMax-M3`.** It ranked first by WCR (MiniMaxAI/MiniMax-M3).

- `MiniMaxAI/MiniMax-M3` WCR 1.000, open slots 0 ← use this
- `deepseek-ai/DeepSeek-V4-Pro` WCR 1.000, open slots 0
- `Qwen/Qwen3.5-397B-A17B` WCR 0.952, open slots 0

## Arms

| Arm | Description                | WCR   | LCR   | ICR   | Exact | Prec  | Rec   | Open | Calls | Tokens | Sec   |
|-----|----------------------------|-------|-------|-------|-------|-------|-------|------|-------|--------|-------|
| a5  | reasoning model throughout | 0.830 | 0.923 | 1.000 | 0.286 | 0.957 | 0.923 | 3.6  | 14.1  | 35120  | 191.3 |

`Open` is the mean number of slots the solver declined to answer. Read it with `Prec`: declining is not the same failure as guessing wrong, and only the pair distinguishes them.

## Exact-solve, with its uncertainty

| Arm | Solved | Rate  | 95% Wilson CI  | Width |
|-----|--------|-------|----------------|-------|
| a5  | 2/7    | 0.286 | [0.082, 0.641] | 0.559 |

Exact-solve is one Bernoulli trial per puzzle, so its interval is wide at these sample sizes -- telling 80% from 70% apart at 80% power needs about 294 puzzles. Treat the rates above as descriptive and read the paired McNemar column for arm-vs-arm claims.

## WCR by clue type

| Arm | definition  | proper     |
|-----|-------------|------------|
| a5  | 0.820 (217) | 0.857 (77) |

## WCR by answer length

| Arm | 3          | 4-5         | 6-8        |
|-----|------------|-------------|------------|
| a5  | 0.871 (70) | 0.827 (196) | 0.750 (28) |

## WCR by grid size

| Arm | 11x11 |
|-----|-------|
| a5  | 0.830 |

## Structured-output support

Which rung of the schema ladder each model accepted. A model that falls back to free text needs the lenient parser to be usable at all.

| Model                              | Rung(s) used  |
|------------------------------------|---------------|
| MiniMaxAI/MiniMax-M3               | strict_schema |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | strict_schema |
| Qwen/Qwen3-30B-A3B-Instruct-2507   | strict_schema |
| Qwen/Qwen3.5-397B-A17B             | strict_schema |
| deepseek-ai/DeepSeek-V4-Pro        | strict_schema |
| meta-llama/Llama-3.3-70B-Instruct  | strict_schema |
| openai/gpt-oss-120b                | json_object   |

