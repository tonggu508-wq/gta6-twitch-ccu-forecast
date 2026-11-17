# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tonggu508-wq/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tonggu508-wq/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC). The program reads everything recorded up to the end of hour 1 and writes five quantiles (5%, 25%, 50%, 75%, 95%) for each of hours 2–24, scored by pinball loss on the log scale.

The model: each hour's forecast is the hour-1 count times a multiplier. The multipliers' median path and spread were fitted on past console launches with the same shape as GTA VI (rolling local-midnight unlock, no early access) and calibrated by leave-one-out cross-validation; a "second wave" tilt lifts the later hours when the first regional wave has already peaked and faded before the UTC day begins. Pure Python standard library, fully offline, runs in milliseconds.

## Run it

```sh
python3 main.py
```

The competition sandbox runs `main.py` at the repository root: it reads the case file at `/input` and writes the forecast file to `/output`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=0zrpbbm3&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
