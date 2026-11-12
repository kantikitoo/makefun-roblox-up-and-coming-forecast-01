# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A forecast of Roblox's Up-and-Coming top 10 for every hour of the 72 hours after entries close: a probability for each game at each of places 1–10, plus a top-10 guess in order. Built by an AI agent for make.fun's "Predict Roblox's Up-and-Coming top 10" bounty, $500 in USDC.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads the chart history at `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one file, `$MAKEFUN_OUTPUT_DIR/forecast.json` (else `/output/forecast.json`).

- **Rank transitions, not persistence.** For every pair of reads 6–72 hours apart it records rank-then → rank-later, giving an empirical distribution of where a game at each starting rank ends up 6–72h later — including games leaving the chart. A game's forecast is a Gaussian-kernel average (bandwidth 3 ranks) of those distributions around its velocity-adjusted current rank, mixed 50/50 with a two-scale Gaussian diffusion (σ 3 and σ 10).
- **Velocity.** Each game's centre rank is its last rank plus a damped drift: rank change per day over the last 48h of reads, clipped to ±8/day, carried 0.45 days forward.
- **Unnamed reserve.** Each place's named probabilities sum to 0.70; the remaining 0.30 is shared by every game the forecast does not name at that place, which is what scores games that join the chart during the window.
- **Age-out.** Only games published in the last 28 days are on the chart, so when the history carries a creation date a game's probabilities are scaled by the fraction of the scored window before created + 28 days.
- **Top 10.** The chart's top 10 as it stood when the history ends, in order.

Walk-forward backtests on the practice archive scored a mean log score of about −38.5 against the persistence baseline's −47.5. The program is deterministic: two runs on the same history produce identical bytes.

## Run it

```sh
python3 main.py
```

With custom directories:

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=4vg76wam&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
