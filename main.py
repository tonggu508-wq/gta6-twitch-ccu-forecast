"""Forecast a big game launch's release-day Twitch viewers, hour by hour (gta6-twitch-ccu-forecast-01).

The model, in one paragraph: an hour's viewers are forecast as a multiple of hour 1's print. The
multiplier's median path over hours 2..24 (in log space) and the spread around it were fitted at build
time on the thirteen past launches in the kit's comparables that match the target's shape -- console
games unlocked at rolling local midnight, no early access (ELDEN RING, Resident Evil Requiem, Ghost of
Yotei, Monster Hunter Wilds, the four Call of Duty launches, Tears of the Kingdom, FF VII Rebirth, The
Last of Us Part II, Ghost of Tsushima, Stellar Blade) -- and are baked in below as constants. On top of
the median path sits one case-specific adjustment, the "second wave" tilt: when a launch's first,
pre-UTC-day wave (New Zealand / Asia / pre-launch events) had already peaked and the count was falling
into hour 1, hour 1 sits in a trough and the day's later hours (Europe's evening, the US unlock and its
prime time) rebound far above the pooled median path. The tilt measures that from the case's own last
24 pre-release hours and lifts the late hours accordingly. Everything else about the case is ignored.

How the constants were fitted (offline, from the kit's practice comparables, SullyGnome hourly data):
for each training launch, r_i(h) = ln(viewers_h / viewers_1) on its release day and its own tilt_i(h)
as defined below; CENTER[h] is the median of r_i(h) - tilt_i(h) over the thirteen launches, smoothed
over neighbouring hours (3-hour moving average); SIGMA is the standard deviation of all residuals
r_i(h) - tilt_i(h) - CENTER[h]. Leave-one-out over the thirteen launches (refit without the held-out
launch, forecast it) gives a mean log pinball loss of 0.219 with 90% coverage 0.886, against 0.349 for
the house persistence baseline on the same folds and 0.263 for the same median path with no tilt.
The band is mildly asymmetric (ASYM): in the folds the upper tail of the residuals ran heavier than
the lower, so the 95% quantile's sigma is scaled by 1.15 and the lower quantiles' by 0.9.

Runs as `python3 /bot/main.py` (Python 3.13, standard library only, no network): reads every *.in in
$MAKEFUN_INPUT_DIR (else /input), writes <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output).
Deterministic: the same case always yields the same bytes. Any unexpected input falls back to the
house persistence shape, so a case always produces a valid forecast file.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Per-quantile spread scaling (see the docstring): lower band slightly narrower, top wider.
ASYM = [0.9, 0.9, 1.0, 1.0, 1.15]

# CENTER[h - 2] for h = 2..24: the fitted median of ln(viewers_h / viewers_1), tilt-adjusted,
# 3-hour smoothed, over the thirteen training launches named in the docstring.
CENTER = [
    -0.2337, -0.2807, -0.2928, -0.1861, -0.0775, -0.0377, -0.0639, -0.0871,
    -0.1221, -0.1327, -0.0873, 0.0022, 0.1165, 0.2217, 0.2839, 0.3115,
    0.3474, 0.4210, 0.4815, 0.5132, 0.5159, 0.4632, 0.4265,
]
# Standard deviation of the fitted residuals, in log units.
SIGMA = 0.4259

# The second-wave tilt: tilt(h) = min(TILT_BETA * drop * gate * ramp(h), TILT_CAP), where
# drop = ln(peak viewers in the 24 h before the day / hour 1's print),
# gate = clip(fall / 0.3, 0, 1) with fall = ln(viewers in the day's last pre-hour / hour 1's print),
# ramp(h) = clip((h - 6) / 10, 0, 1): no lift before hour 6, full lift from hour 16 on.
TILT_BETA = 0.8
TILT_CAP = 1.1


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp from a case file; None when it does not parse."""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def round2(x):
    """Round to 2 decimals, half up (as the starter does), whole numbers as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def case_features(case, p1):
    """(drop, gate) from the case's own pre-release hours; (0, 0) when they are unusable.

    Uses history_hourly's whole-hour means in the 24 hours before day_start. Needs at least 12 of
    those hours and the last pre-hour (day_start - 1h); otherwise the tilt stays off, which is the
    safe default (the pooled median path with no case adjustment).
    """
    day_start = parse_time(case.get("day_start", ""))
    if day_start is None:
        return 0.0, 0.0
    window_start = day_start - timedelta(hours=24)
    last_hour_start = day_start - timedelta(hours=1)
    peak = 0.0
    last = None
    latest_at = None
    latest = None
    seen = 0
    for row in case.get("history_hourly") or []:
        at = parse_time(row.get("from", ""))
        viewers = row.get("viewers")
        if at is None or not isinstance(viewers, (int, float)):
            continue
        if window_start <= at < day_start:
            seen += 1
            peak = max(peak, float(viewers))
            if at == last_hour_start:
                last = float(viewers)
            if latest_at is None or at > latest_at:
                latest_at, latest = at, float(viewers)
    if last is None:
        # No row starts exactly one hour before the day (an offset hourly grid): take the
        # window's latest row instead, provided it starts within the day's last two hours.
        if latest_at is not None and latest_at >= day_start - timedelta(hours=2):
            last = latest
    if seen < 12 or last is None or peak <= 0 or last <= 0:
        return 0.0, 0.0
    drop = math.log(peak / p1)
    fall = math.log(last / p1)
    gate = min(1.0, max(0.0, fall / 0.3))
    return drop, gate


def tilt(drop, gate, h):
    """The second-wave lift of hour h's log forecast, never negative, capped."""
    ramp = min(1.0, max(0.0, (h - 6) / 10))
    return max(0.0, min(TILT_BETA * drop * gate * ramp, TILT_CAP))


def hour_quantiles(p1, drop, gate, h):
    center = CENTER[h - 2] if 2 <= h <= 24 else 0.0
    m = center + tilt(drop, gate, h)
    qs = []
    prev = 0.0
    for z, a in zip(Z, ASYM):
        q = p1 * math.exp(m + z * SIGMA * a)
        q = min(max(q, prev), 1_000_000_000)
        qs.append(round2(q))
        prev = q
    return qs


def baseline_quantiles(p1, h):
    """The house persistence shape, used only when a case cannot be read as expected."""
    qs = []
    prev = 0.0
    for z in Z:
        q = p1 * math.exp(z * 0.15 * math.sqrt(max(0, h - 1)))
        q = min(max(q, prev), 1_000_000_000)
        qs.append(round2(q))
        prev = q
    return qs


def hour1_print(case):
    """Hour 1's print from `known`; falls back to the last history point, then to 1."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and isinstance(row.get("viewers"), (int, float)):
            return max(1.0, float(row["viewers"]))
    history = case.get("history") or []
    if history and isinstance(history[-1].get("viewers"), (int, float)):
        return max(1.0, float(history[-1]["viewers"]))
    return 1.0


def forecast_case(case):
    p1 = hour1_print(case)
    try:
        drop, gate = case_features(case, p1)
        rows = [
            {"hour": row["hour"], "q": hour_quantiles(p1, drop, gate, row["hour"])}
            for row in case["forecast"]
        ]
    except Exception:
        rows = [
            {"hour": row["hour"], "q": baseline_quantiles(p1, row["hour"])}
            for row in case.get("forecast") or [{"hour": h} for h in range(2, 25)]
        ]
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
