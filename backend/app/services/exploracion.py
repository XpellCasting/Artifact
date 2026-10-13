"""Descriptive exploration only: no labels, interventions or detector execution."""
import json
from collections import Counter
from datetime import datetime, timedelta
from statistics import median

from app.database.review import iso_time
from app.database.review import ReviewError

EXPLORATION_VERSION = 1


def overview(store, dataset_id):
    dataset = store.dataset(dataset_id)
    with store.connect() as connection:
        cached = connection.execute("SELECT payload FROM exploration_cache WHERE dataset_id=? AND version=?",
                                    (dataset_id, EXPLORATION_VERSION)).fetchone()
        if cached:
            return json.loads(cached[0])
        entities = []
        for entity in dataset["entities"]:
            counts, daily = Counter(), {}
            total = valid = transitions = 0
            first = last = previous = previous_state = None
            largest_gap = 0
            gap_start = gap_end = None
            for row in connection.execute(
                "SELECT ts, state, is_valid FROM events WHERE dataset_id=? AND entity_id=? ORDER BY ts, source_row",
                (dataset_id, entity["entity_id"])):
                ts, state, is_valid = row
                total += 1
                first = ts if first is None else first
                last = ts
                day = iso_time(ts)[:10]
                activity = daily.setdefault(day, {"date": day, "valid": 0, "invalid": 0})
                activity["valid" if is_valid else "invalid"] += 1
                if previous is not None and ts - previous > largest_gap:
                    largest_gap, gap_start, gap_end = ts - previous, previous, ts
                previous = ts
                if is_valid:
                    valid += 1
                    counts[state] += 1
                    if previous_state is not None and previous_state != state:
                        transitions += 1
                    previous_state = state
                else:
                    # Do not infer a transition across an explicitly invalid report.
                    previous_state = None
            entities.append({**entity, "rows": total, "valid_rows": valid, "invalid_rows": total-valid,
                             "start": iso_time(first) if first is not None else None,
                             "end": iso_time(last) if last is not None else None,
                             "distinct_states": len(counts),
                             "transitions": transitions if entity["kind"] != "continuous" else None,
                             "max_report_gap_seconds": largest_gap / 1e6 if total > 1 else None,
                             "gap_start": iso_time(gap_start) if gap_start is not None else None,
                             "gap_end": iso_time(gap_end) if gap_end is not None else None,
                             "daily": list(daily.values())})
        result = {"dataset_id": dataset_id, "sha256": dataset["sha256"],
                  "partition": dataset["partition"], "start": dataset["start"], "end": dataset["end"],
                  "source_rows": dataset["rows"], "indexed_rows": dataset["indexed_rows"],
                  "rejected_rows": dataset["rejected_rows"], "entities": entities,
                  "version": EXPLORATION_VERSION}
        connection.execute("INSERT OR REPLACE INTO exploration_cache VALUES (?, ?, ?)",
                           (dataset_id, EXPLORATION_VERSION, json.dumps(result, ensure_ascii=False)))
    return result


def entity_detail(store, dataset_id, entity_id, start, end):
    dataset, ids, lower, upper, _, _ = store._window(dataset_id, [entity_id], start, end, 0)
    summary = overview(store, dataset_id)
    entity = next(item for item in summary["entities"] if item["entity_id"] == entity_id)
    states, numbers = Counter(), []
    total = invalid = 0
    with store.connect() as connection:
        for row in connection.execute(
            "SELECT state, numeric_value, is_valid FROM events WHERE dataset_id=? AND entity_id=? AND ts BETWEEN ? AND ?",
            (dataset_id, entity_id, lower, upper)):
            total += 1
            if not row["is_valid"]:
                invalid += 1
            else:
                states[row["state"]] += 1
                if entity["kind"] == "continuous" and row["numeric_value"] is not None:
                    numbers.append(row["numeric_value"])
    day = datetime.fromisoformat(summary["start"][:10])
    last_day = datetime.fromisoformat(summary["end"][:10])
    # Bound the response for accidental files spanning thousands of years.
    if (last_day-day).days > 36600:
        raise ReviewError("La actividad diaria admite históricos de hasta 100 años.")
    observed = {item["date"]: item for item in entity["daily"]}
    daily = []
    while day <= last_day:
        key = day.date().isoformat()
        daily.append(observed.get(key, {"date": key, "valid": 0, "invalid": 0}))
        day += timedelta(days=1)
    valid = total-invalid
    return {"dataset_id": dataset_id, "partition": dataset["partition"], "entity_id": entity_id,
            "start": iso_time(lower), "end": iso_time(upper), "rows": total,
            "valid_rows": valid, "invalid_rows": invalid, "daily": daily,
            "frequencies": [{"state": state, "rows": count, "percent": count*100/valid if valid else 0}
                            for state, count in sorted(states.items(), key=lambda item: (-item[1], item[0]))]
                           if entity["kind"] != "continuous" else [],
            "numeric": {"count": len(numbers), "min": min(numbers) if numbers else None,
                        "median": median(numbers) if numbers else None, "max": max(numbers) if numbers else None},
            "fragment": store.fragment(dataset_id, ids, start, end, padding_minutes=0, max_points=600)}
