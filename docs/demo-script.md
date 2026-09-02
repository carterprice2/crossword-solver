# 60-second demo: shot list and script

The assignment cap is one minute: show the agent working, and say how it
works. Record the **website** against Token Factory. Do not use the oracle.

Pick a 7×7 from Mini (`mini-07-00-0`) and arm `a3`.
Do one clean live take first; if a later take needs to be bit-for-bit the
same, record that solve (`--record`) and replay it in the CLI, but the
submitted video should still *look* like the website.

## Shot list

| Time | Shot | Say |
|---|---|---|
| 0:00–0:05 | Full-screen card, four boxes: **CANDIDATES → CROSSINGS → SEARCH → REPAIR** | "A crossword isn’t trivia. It’s a constraint problem. Answers come from a language model, then the grid checks them." |
| 0:05–0:12 | Website: Mini, 7×7, arm `a3`, click **Solve** | "Round one asks the model for every clue cold. Clues that actually cross get batched together, so it can check itself." |
| 0:12–0:35 | Grid filling. Yellow, then green locks. Rail: round, conflicts, repair. | "Then the crossings vote. Where two confident answers agree, the letter locks. Where they conflict, the agent goes back to the model with the letters it now knows and the answers it already ruled out." |
| 0:35–0:46 | Gold check: Agent vs answer key. **Solved**, WCR. | "The first pass is a guess. The loop is the solver." |
| 0:46–0:60 | Overlay or cut: repair-gain row from REPORT.md | "That’s the measurement: the repair loop adds two hundredths of WCR when candidates are good, and almost three tenths when they’re bad. Spend the extra tokens only when they earn it." |

## The one number to say out loud

> The repair loop adds **+0.02 WCR when candidates are good and +0.29 when
> they’re bad.** Its value grows as the base model gets weaker.

Leave the model name out of the voiceover until the final pick is locked.
If the live 7×7 overruns, speed the middle of the fill in the edit — keep
round 0, the first repair, and the gold check at full speed.

## Recording notes

- Site: `make serve PY=.venv/bin/python` → http://127.0.0.1:8000
- Needs `NEBIUS_API_KEY`. Solve is live; a take can fail. Budget two backups.
- Browser window ~1280×800, no bookmarks bar, Debug off.
- Do not open the NYT 15×15. Do not use Your puzzle. Do not show the arm list
  beyond `a3`.
- Word count is ~140. Do not add a second architecture slide.
