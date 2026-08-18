#!/usr/bin/env python3
"""
Template-based generator for the structured/schedule source (pilot corpus).

Produces persona-conditioned 7-day household activity schedules — the same
shape as the Replica energy-simulation use case (occupancy, meals, showers,
appliances, EV charging, trips) — rendered in XML, JSON, and key-value
formats. Free, infinite, deterministic (seeded per document), and perfectly
on-format: exactly what a tiny model needs to learn structure emission.

Output: parquet with columns [text, format], ready to reference as a source
in corpus.yaml with the `structured` profile (normalize_mode: none).

Example:
    python bin/generate_structured.py --output data/raw/schedules.parquet \\
        --num-docs 50000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import polars as pl
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Missing dependency: {e}. Install tiny_corpus_prep first.", flush=True)
    sys.exit(1)

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAYS = set(DAYS[:5])

# occupation -> (works_weekdays, away_window or None)
OCCUPATIONS = {
    "office_worker": (True, ("08:00", "17:45")),
    "teacher": (True, ("07:30", "16:00")),
    "shopkeeper": (True, ("08:30", "19:15")),
    "student": (True, ("08:15", "17:00")),
    "school_child": (True, ("08:00", "16:15")),
    "remote_worker": (False, None),
    "retired": (False, None),
}

HEATING_TYPES = ["gas_boiler", "heat_pump", "district_heating", "electric_radiators"]
ENERGY_LABELS = ["A", "B", "C", "D", "E", "F"]
SEASONS = {
    "winter": (2, 6), "spring": (10, 6), "summer": (24, 6), "autumn": (13, 6),
}
CLOUD = ["clear", "partly_cloudy", "overcast"]
TRIP_MODES = ["car", "bike", "bus", "walk"]


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _hhmm(minutes: int) -> str:
    minutes = max(0, min(minutes, 23 * 60 + 59))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _jitter(rng: random.Random, hhmm: str, spread_min: int) -> str:
    return _hhmm(_minutes(hhmm) + rng.randint(-spread_min, spread_min))


def make_persona(rng: random.Random) -> Dict:
    n_adults = rng.choice([1, 1, 2, 2, 2, 3])
    n_children = rng.choice([0, 0, 0, 1, 1, 2])
    occupants = []
    for _ in range(n_adults):
        age = rng.randint(23, 65)
        occupation = rng.choice(
            ["office_worker", "office_worker", "teacher", "remote_worker",
             "shopkeeper", "retired" if age > 60 else "office_worker"]
        )
        occupants.append({"age": age, "occupation": occupation})
    for _ in range(n_children):
        age = rng.randint(4, 19)
        occupants.append({
            "age": age,
            "occupation": "school_child" if age < 15 else "student",
        })
    season = rng.choice(list(SEASONS))
    base_temp, spread = SEASONS[season]
    weather = []
    for _ in DAYS:
        temp = round(base_temp + rng.uniform(-spread, spread), 1)
        cloud = rng.choice(CLOUD)
        precip = round(rng.uniform(0.5, 12.0), 1) if rng.random() < (
            0.45 if cloud == "overcast" else 0.1
        ) else 0.0
        weather.append({"temp_c": temp, "cloud": cloud, "precip_mm": precip})
    return {
        "occupants": occupants,
        "surface_m2": rng.choice([45, 60, 75, 90, 110, 140, 180]),
        "energy_label": rng.choice(ENERGY_LABELS),
        "heating": rng.choice(HEATING_TYPES),
        "has_pv": rng.random() < 0.25,
        "has_ev": rng.random() < 0.18,
        "season": season,
        "weather": weather,
    }


def make_week(rng: random.Random, persona: Dict) -> List[Dict]:
    """Simulate 7 days of household activity from the persona + weather."""
    occupants = persona["occupants"]
    n = len(occupants)
    wash_days = set(rng.sample(DAYS, rng.choice([2, 3])))
    oven_days = set(rng.sample(DAYS, rng.choice([2, 3])))
    shop_day = rng.choice(DAYS)
    week = []
    for day, wx in zip(DAYS, persona["weather"]):
        rainy = wx["precip_mm"] > 2.0
        away: List[Tuple[Dict, str, str]] = []  # (occupant, depart, return)
        if day in WEEKDAYS:
            for occ in occupants:
                works, window = OCCUPATIONS[occ["occupation"]]
                if works and window:
                    away.append((
                        occ,
                        _jitter(rng, window[0], 15),
                        _jitter(rng, window[1], 20),
                    ))
        # Occupancy slots from the earliest departure / latest return
        if away:
            first_out = min(_minutes(d) for _, d, _ in away)
            last_back = max(_minutes(r) for _, _, r in away)
            occupancy = [
                {"from": "00:00", "to": _hhmm(first_out), "people": n},
                {"from": _hhmm(first_out), "to": _hhmm(last_back), "people": n - len(away)},
                {"from": _hhmm(last_back), "to": "24:00", "people": n},
            ]
        else:
            occupancy = [{"from": "00:00", "to": "24:00", "people": n}]

        home_at_lunch = n - sum(
            1 for _, d, r in away if _minutes(d) < 13 * 60 < _minutes(r)
        )
        meals = [{"type": "breakfast", "time": _jitter(rng, "07:15", 25), "people": n}]
        if home_at_lunch > 0:
            meals.append({
                "type": "lunch", "time": _jitter(rng, "13:00", 20),
                "people": home_at_lunch,
            })
        meals.append({"type": "dinner", "time": _jitter(rng, "19:30", 25), "people": n})

        showers = []
        for i, occ in enumerate(occupants):
            evening = rng.random() < 0.35
            base = "20:45" if evening else "06:50"
            showers.append({
                "time": _hhmm(_minutes(_jitter(rng, base, 15)) + i * 12),
                "duration_min": rng.randint(5, 12),
            })

        appliances = []
        if day in wash_days:
            appliances.append({
                "name": "washing_machine",
                "start": _jitter(rng, "18:15", 45),
                "duration_min": rng.choice([60, 90, 120]),
            })
        if persona["surface_m2"] >= 75 or rng.random() < 0.5:
            appliances.append({
                "name": "dishwasher",
                "start": _jitter(rng, "20:30", 20),
                "duration_min": rng.choice([50, 70]),
            })
        if day in oven_days:
            appliances.append({
                "name": "oven",
                "start": _jitter(rng, "18:45", 20),
                "duration_min": rng.randint(30, 75),
            })
        if wx["temp_c"] >= 27.0:
            appliances.append({
                "name": "air_conditioner",
                "start": "13:30",
                "duration_min": rng.randint(180, 360),
            })
        if wx["temp_c"] <= 7.0:
            appliances.append({
                "name": "heating",
                "start": "06:00",
                "duration_min": rng.randint(600, 900),
            })

        ev_sessions = []
        if persona["has_ev"] and (away or rng.random() < 0.3):
            ev_sessions.append({
                "start": _jitter(rng, "19:00", 60),
                "duration_min": rng.choice([120, 180, 240]),
            })

        trips = [
            {
                "purpose": "commute", "depart": d, "return": r,
                "mode": "car" if persona["has_ev"] else rng.choice(TRIP_MODES),
            }
            for _, d, r in away
        ]
        if day == shop_day and home_at_lunch > 0:
            trips.append({
                "purpose": "shopping",
                "depart": _jitter(rng, "10:30", 45),
                "return": _jitter(rng, "11:45", 45),
                "mode": rng.choice(TRIP_MODES),
            })
        if day not in WEEKDAYS and not rainy and rng.random() < 0.7:
            trips.append({
                "purpose": "leisure",
                "depart": _jitter(rng, "14:30", 60),
                "return": _jitter(rng, "17:30", 60),
                "mode": rng.choice(TRIP_MODES),
            })

        week.append({
            "day": day, "weather": wx, "occupancy": occupancy, "meals": meals,
            "showers": showers, "appliances": appliances,
            "ev_charging": ev_sessions, "trips": trips,
        })
    return week


# ── renderers ────────────────────────────────────────────────────


def render_xml(hh_id: str, persona: Dict, week: List[Dict]) -> str:
    out = [f'<household id="{hh_id}">']
    p = persona
    out.append(
        f'  <profile occupants="{len(p["occupants"])}" surface_m2="{p["surface_m2"]}" '
        f'energy_label="{p["energy_label"]}" heating="{p["heating"]}" '
        f'pv="{str(p["has_pv"]).lower()}" ev="{str(p["has_ev"]).lower()}">'
    )
    for occ in p["occupants"]:
        out.append(f'    <occupant age="{occ["age"]}" occupation="{occ["occupation"]}"/>')
    out.append("  </profile>")
    out.append("  <week>")
    for d in week:
        wx = d["weather"]
        out.append(
            f'    <day name="{d["day"]}" temp_c="{wx["temp_c"]}" '
            f'cloud="{wx["cloud"]}" precip_mm="{wx["precip_mm"]}">'
        )
        out.append("      <occupancy>")
        for s in d["occupancy"]:
            out.append(f'        <slot from="{s["from"]}" to="{s["to"]}" people="{s["people"]}"/>')
        out.append("      </occupancy>")
        out.append("      <meals>")
        for m in d["meals"]:
            out.append(f'        <meal type="{m["type"]}" time="{m["time"]}" people="{m["people"]}"/>')
        out.append("      </meals>")
        out.append("      <showers>")
        for s in d["showers"]:
            out.append(f'        <shower time="{s["time"]}" duration_min="{s["duration_min"]}"/>')
        out.append("      </showers>")
        out.append("      <appliances>")
        for a in d["appliances"]:
            out.append(
                f'        <appliance name="{a["name"]}" start="{a["start"]}" '
                f'duration_min="{a["duration_min"]}"/>'
            )
        out.append("      </appliances>")
        out.append("      <ev_charging>")
        for s in d["ev_charging"]:
            out.append(f'        <session start="{s["start"]}" duration_min="{s["duration_min"]}"/>')
        out.append("      </ev_charging>")
        out.append("      <trips>")
        for t in d["trips"]:
            out.append(
                f'        <trip purpose="{t["purpose"]}" depart="{t["depart"]}" '
                f'return="{t["return"]}" mode="{t["mode"]}"/>'
            )
        out.append("      </trips>")
        out.append("    </day>")
    out.append("  </week>")
    out.append("</household>")
    return "\n".join(out)


def render_json(hh_id: str, persona: Dict, week: List[Dict]) -> str:
    doc = {
        "household_id": hh_id,
        "profile": {
            "occupants": persona["occupants"],
            "surface_m2": persona["surface_m2"],
            "energy_label": persona["energy_label"],
            "heating": persona["heating"],
            "pv": persona["has_pv"],
            "ev": persona["has_ev"],
        },
        "week": week,
    }
    return json.dumps(doc, indent=1)


def render_kv(hh_id: str, persona: Dict, week: List[Dict]) -> str:
    out = [f"household: {hh_id}"]
    p = persona
    out.append(
        f"profile: occupants={len(p['occupants'])} surface_m2={p['surface_m2']} "
        f"energy_label={p['energy_label']} heating={p['heating']} "
        f"pv={str(p['has_pv']).lower()} ev={str(p['has_ev']).lower()}"
    )
    for occ in p["occupants"]:
        out.append(f"occupant: age={occ['age']} occupation={occ['occupation']}")
    for d in week:
        wx = d["weather"]
        out.append(
            f"day: {d['day']} | temp_c={wx['temp_c']} cloud={wx['cloud']} "
            f"precip_mm={wx['precip_mm']}"
        )
        for s in d["occupancy"]:
            out.append(f"  occupancy: {s['from']}-{s['to']} people={s['people']}")
        for m in d["meals"]:
            out.append(f"  meal: {m['type']} time={m['time']} people={m['people']}")
        for s in d["showers"]:
            out.append(f"  shower: time={s['time']} duration_min={s['duration_min']}")
        for a in d["appliances"]:
            out.append(f"  appliance: {a['name']} start={a['start']} duration_min={a['duration_min']}")
        for s in d["ev_charging"]:
            out.append(f"  ev_charging: start={s['start']} duration_min={s['duration_min']}")
        for t in d["trips"]:
            out.append(
                f"  trip: {t['purpose']} depart={t['depart']} "
                f"return={t['return']} mode={t['mode']}"
            )
    return "\n".join(out)


RENDERERS = {"xml": render_xml, "json": render_json, "kv": render_kv}


def make_summary(rng: random.Random, persona: Dict, week: List[Dict]) -> str:
    """Short natural-language summary — grounds structure in prose."""
    n = len(persona["occupants"])
    rainy = [d["day"] for d in week if d["weather"]["precip_mm"] > 2.0]
    hot = [d["day"] for d in week if d["weather"]["temp_c"] >= 27.0]
    parts = [
        f"This home has {n} {'person' if n == 1 else 'people'} and "
        f"{persona['surface_m2']} square meters with {persona['heating'].replace('_', ' ')} heating."
    ]
    if rainy:
        parts.append(f"It rained on {rainy[0]}, so the family spent more time indoors.")
    elif hot:
        parts.append(f"It was hot on {hot[0]}, so the air conditioner ran in the afternoon.")
    else:
        parts.append(f"The weather was mild all {persona['season']} week.")
    if persona["has_ev"]:
        parts.append("The electric car charged in the evening after the commute.")
    return " ".join(parts)


def generate_doc(seed: int, index: int, fmt: str, with_summary: bool) -> str:
    rng = random.Random(f"{seed}:{index}")
    persona = make_persona(rng)
    week = make_week(rng, persona)
    hh_id = f"HH{index:06d}"
    text = RENDERERS[fmt](hh_id, persona, week)
    if with_summary:
        text = text + "\n\n" + make_summary(rng, persona, week)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate persona-conditioned household schedules (structured source)"
    )
    ap.add_argument("--output", required=True, help="Output parquet path")
    ap.add_argument("--num-docs", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--formats", default="xml,json,kv",
        help="Comma-separated subset of xml,json,kv (default: all three, round-robin)",
    )
    ap.add_argument(
        "--no-summary", action="store_true",
        help="Skip the trailing natural-language summary sentence",
    )
    ap.add_argument("--chunk-size", type=int, default=2000, help="Docs per write chunk")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = set(formats) - set(RENDERERS)
    if unknown or not formats:
        logger.error("Unknown formats %s; choose from %s", sorted(unknown), sorted(RENDERERS))
        return 1

    from pathlib import Path

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    total_chars = 0
    try:
        for start in range(0, args.num_docs, args.chunk_size):
            end = min(start + args.chunk_size, args.num_docs)
            texts, fmts = [], []
            for i in range(start, end):
                fmt = formats[i % len(formats)]
                texts.append(generate_doc(args.seed, i, fmt, not args.no_summary))
                fmts.append(fmt)
            total_chars += sum(len(t) for t in texts)
            table = pl.DataFrame({"text": texts, "format": fmts}).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            logger.info("Generated %d / %d docs...", end, args.num_docs)
    finally:
        if writer is not None:
            writer.close()

    logger.info(
        "Done: %d docs, %.1f MB of text (~%d tokens at 4 chars/token) → %s",
        args.num_docs, total_chars / 1e6, total_chars // 4, out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
