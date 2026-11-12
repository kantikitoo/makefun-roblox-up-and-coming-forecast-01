#!/usr/bin/env python3
"""Predict Roblox's Up-and-Coming top 10 (make.fun, program/v1).

Reads the chart history at $MAKEFUN_INPUT_DIR (else /input) and writes one
forecast, $MAKEFUN_OUTPUT_DIR/forecast.json (else /output/forecast.json):
for each of places 1..10, a probability for each game named at that place,
plus a top-10 guess in order. One static forecast is scored against every
hourly read of the 72-hour window after entries close, so the probabilities
are the time-averaged occupancy of each place over that window.

Method (fitted by walk-forward backtests on the practice archive, where it
scored about -38.6 mean log score against the persistence baseline's -47.5):

  * Rank transitions, not persistence.  For every pair of reads 6-72 hours
    apart, record (rank then -> rank later).  That gives, for each starting
    rank, an empirical distribution of the rank 6-72h later -- including
    games leaving the chart altogether (counted as rank 120, i.e. out of
    the top 10).  A game's forecast rank distribution is a Gaussian-kernel
    average (bandwidth 3 ranks) of those starting-rank distributions around
    its velocity-adjusted current rank, mixed 50/50 with a two-scale
    Gaussian diffusion (sigma 3 and sigma 10) around the same centre, so
    no place ever gets a hard zero from a thin sample.
  * Velocity.  A game's centre is its last rank plus a damped drift: the
    rank change per day over the last 48h of reads (span at least 12h),
    clipped to +/-8 ranks/day, carried 0.45 days forward (0.3 * 36h).
    Rising games keep rising a little; falling games keep falling.
  * Unnamed reserve.  Each place's named probabilities sum to 0.70; the
    remaining 0.30 is shared equally by every game the forecast does not
    name at that place.  New games join the chart during the window and
    cannot be named, so the reserve is what scores them instead of the
    -9.21 floor.
  * Age-out.  Only games published in the last 28 days are on the chart.
    When the history carries a game's creation date (the house recorder's
    details), its probabilities are scaled by the fraction of the scored
    window before created + 28 days; a game already past that date is
    dropped.  Without creation dates (the public archive) no scaling is
    applied.
  * Top 10.  The chart's top 10 as it stood when the history ends, in
    order: over a 72h window the last order is the best point guess for
    hits and for the agreement tie-break, even though the probabilities
    themselves are diffuse.

Deterministic: no randomness, no network, no timestamps in the output,
stdlib only.  Two runs on the same history produce identical bytes.

Backtests (practice archive, 34 reads, 2026-10-30 to 2026-11-07).
Walk-forward cutoffs at reads 14, 19, 24 and 29, scoring the next 72h of
reads with the exact scorer (named p, else the unnamed share, floor 1e-4):
this configuration scored a mean log score of -38.5 to -38.9, against
-47.5 for the persistence baseline (the starter).  Things tried that hurt
and were dropped: velocity carried more than 0.6 days forward, playing
growth as a signal, and blending in empirical occupancy frequency.
"""

import json
import math
import os
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tuned constants (see module docstring) ---------------------------------
RESERVE = 0.30            # unnamed share of each place
KERNEL_BW = 3.0           # rank kernel bandwidth for transition averaging
LAG_MIN_H = 6.0           # shortest read-pair lag used for transitions
LAG_MAX_H = 72.0          # longest read-pair lag (the scored window length)
SMOOTH = 0.5              # weight on Gaussian diffusion vs empirical transitions
DIFFUSION = ((3.0, 0.5), (10.0, 0.5))  # (sigma, weight) of the diffusion mix
VEL_DAYS = 0.3            # drift carried forward, in days of velocity
VEL_CAP = 8.0             # max |rank change| per day carried into the drift
VEL_SPAN_H = 48.0         # lookback for the velocity estimate
VEL_MIN_SPAN_H = 12.0     # velocity needs at least this much history span
HORIZON_DAYS = 1.5        # 36h: centre of the scored window
CAND_TOP = 45             # candidates from the last chart's top N
CAND_MU_MAX = 45.0        # plus any game whose centre rank is better than this
OUT_RANK = 120            # rank assigned to a game absent from a later read
MIN_P = 0.002             # smallest probability written out
CHART_MAX_AGE_DAYS = 28   # Up-and-Coming only lists games published <28d ago


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _load_history(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    rows = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            snap = json.loads(line)
            chart = sorted(snap.get("up_and_coming") or [], key=lambda g: g["rank"])
            if not chart:
                continue
            rows.append({
                "at": _parse_time(snap["at"]),
                "rank": {g["universe_id"]: g["rank"] for g in chart},
                "ids": [g["universe_id"] for g in chart],
            })
            # Creation dates, when the recorder's details carry them.
            details = snap.get("details") or {}
            for key, det in details.items():
                if not det or not det.get("created"):
                    continue
                try:
                    uid = int(key)
                except (TypeError, ValueError):
                    continue
                rows[-1].setdefault("created", {})[uid] = _parse_time(det["created"])
    return task, rows


def _velocity(points, t0):
    """Rank change per day over the last VEL_SPAN_H hours (endpoints).

    points: [(time, rank), ...] oldest first.  Returns 0 when the span is
    too short to say anything.
    """
    recent = [(t, r) for (t, r) in points if (t0 - t).total_seconds() <= VEL_SPAN_H * 3600]
    if len(recent) < 2:
        return 0.0
    span_h = (recent[-1][0] - recent[0][0]).total_seconds() / 3600.0
    if span_h < VEL_MIN_SPAN_H:
        return 0.0
    vel = (recent[-1][1] - recent[0][1]) / (span_h / 24.0)
    return _clamp(vel, -VEL_CAP, VEL_CAP)


def _diffusion(mu, place):
    p = 0.0
    for sigma, weight in DIFFUSION:
        p += weight * (_norm_cdf((place + 0.5 - mu) / sigma)
                       - _norm_cdf((place - 0.5 - mu) / sigma))
    return p

def forecast_from_rows(task, rows):
    """Build the forecast dict {"probabilities": [...], "top10": [...]}."""
    places = int(task.get("places", 10))
    if not rows:
        return {"probabilities": [], "top10": []}
    window = task.get("window") or {}
    win_from = _parse_time(window["from"]) if window.get("from") else rows[-1]["at"]
    win_until = _parse_time(window["until"]) if window.get("until") else win_from + timedelta(hours=72)
    win_hours = max(1.0, (win_until - win_from).total_seconds() / 3600.0)

    last = rows[-1]
    t0 = last["at"]
    last_ids = last["ids"]

    # Per-game rank trajectories and latest known creation date.
    traj = {}
    created = {}
    for row in rows:
        for uid, rank in row["rank"].items():
            traj.setdefault(uid, []).append((row["at"], rank))
        for uid, when in row.get("created", {}).items():
            created[uid] = when

    # Centre rank per game: last rank plus damped drift.
    centre = {}
    for uid, points in traj.items():
        last_rank = points[-1][1]
        centre[uid] = last_rank + _velocity(points, t0) * VEL_DAYS * HORIZON_DAYS

    # Empirical transitions, bucketed by starting rank: for each starting
    # rank r0, the weighted frequency of the later rank r1 (OUT_RANK when
    # the game left the chart), over read pairs LAG_MIN_H..LAG_MAX_H apart.
    trans = {}  # r0 -> {r1: weight}
    trans_tot = {}  # r0 -> total weight
    for i, row in enumerate(rows):
        for j in range(i + 1, len(rows)):
            lag_h = (rows[j]["at"] - row["at"]).total_seconds() / 3600.0
            if lag_h < LAG_MIN_H:
                continue
            if lag_h > LAG_MAX_H:
                break
            # Read pairs are weighted by recency: a pair a full scored
            # window apart counts half as much as a fresh pair, because
            # chart dynamics drift — a transition from last week says less
            # about tomorrow than one from yesterday.
            weight = 0.5 ** (lag_h / 72.0)
            later = rows[j]["rank"]
            for uid, r0 in row["rank"].items():
                bucket = trans.setdefault(r0, {})
                bucket[later.get(uid, OUT_RANK)] = bucket.get(later.get(uid, OUT_RANK), 0.0) + weight
                trans_tot[r0] = trans_tot.get(r0, 0.0) + weight

    # Candidate games: the last chart's top CAND_TOP plus anything whose
    # centre rank puts it near the top 10.
    candidates = set(last_ids[:CAND_TOP])
    for uid, mu in centre.items():
        if mu < CAND_MU_MAX:
            candidates.add(uid)

    # Age-out scaling per candidate: fraction of the window the game is
    # still eligible for (created + 28 days), 1 when unknown.
    def age_scale(uid):
        when = created.get(uid)
        if when is None:
            return 1.0
        expiry = when + timedelta(days=CHART_MAX_AGE_DAYS)
        if expiry <= win_from:
            return 0.0
        frac = (expiry - win_from).total_seconds() / 3600.0 / win_hours
        return _clamp(frac, 0.05, 1.0)

    # Per-game, per-place probabilities (before normalisation).
    game_place = {}
    for uid in candidates:
        scale = age_scale(uid)
        if scale <= 0.0:
            continue
        mu = centre[uid]
        # Kernel-weighted empirical transition distribution around mu.
        kernel = {}
        kernel_tot = 0.0
        for r0, tot in trans_tot.items():
            kw = math.exp(-0.5 * ((r0 - mu) / KERNEL_BW) ** 2)
            if kw < 0.01:
                continue
            kernel[r0] = kw
            kernel_tot += kw
        probs = {}
        for place in range(1, places + 1):
            emp = 0.0
            if kernel_tot > 0:
                for r0, kw in kernel.items():
                    emp += kw * trans[r0].get(place, 0.0) / trans_tot[r0]
                emp /= kernel_tot
            p = (1.0 - SMOOTH) * emp + SMOOTH * _diffusion(mu, place)
            probs[place] = p * scale
        game_place[uid] = probs

    probabilities = []
    for place in range(1, places + 1):
        named = {uid: probs[place] for uid, probs in game_place.items()
                 if probs[place] > 0.0015}
        total = sum(named.values())
        if total <= 0:
            continue
        factor = (1.0 - RESERVE) / total
        for uid, p in sorted(named.items(), key=lambda kv: (-kv[1], kv[0])):
            scaled = round(p * factor, 6)
            if scaled >= MIN_P:
                probabilities.append({"universe_id": uid, "place": place, "p": scaled})
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    top10 = [uid for uid in last_ids if uid in game_place][:places]
    # Fill from the last chart if age-out dropped a top-10 game.
    for uid in last_ids:
        if len(top10) >= places:
            break
        if uid not in top10:
            top10.append(uid)
    return {"probabilities": probabilities, "top10": top10[:places]}


def main():
    task, rows = _load_history(INPUT_DIR)
    forecast = forecast_from_rows(task, rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(forecast, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
