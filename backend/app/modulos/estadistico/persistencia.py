"""Elapsed state persistence compared with historical state and UTC hour."""
from collections import defaultdict
import math

import numpy as np
import pandas as pd

from app.modulos.estadistico.cobertura import reporting_policy


def _valid(row):
    value = row.get("value")
    return bool(row.get("is_valid_value", True)) and value is not None and pd.notna(value) and math.isfinite(float(value))


def _advance(state, row, policy):
    at = row["timestamp_dt"]
    if not _valid(row):
        state.clear()
        return
    value = float(row["value"])
    gap = (at - state["last_at"]).total_seconds() if state else 0.
    limit = policy.get("max_report_gap_seconds")
    broken = policy["mode"] == "periodic" and (limit is None or gap > limit)
    if not state or value != state["value"] or broken:
        state.update(value=value, started_at=at)
    state["last_at"] = at


def prepare_persistence(frame):
    policy = reporting_policy(frame)
    state, runs = {}, []
    rows = frame.sort_values("timestamp_dt", kind="stable").to_dict("records")
    for row in rows:
        before = dict(state)
        _advance(state, row, policy)
        if before and (not state or state["started_at"] != before["started_at"]):
            end = row["timestamp_dt"]
            limit = policy.get("max_report_gap_seconds")
            if policy["mode"] == "periodic":
                end = min(end, before["last_at"] + pd.Timedelta(seconds=limit or 0))
            if end > before["started_at"]:
                runs.append((before["value"], before["started_at"], end))
    # Right-censored final runs are not evidence of an unusually short duration.
    grouped = defaultdict(list)
    for value, start, end in runs:
        duration = (end - start).total_seconds()
        offset = start.minute * 60 + start.second + start.microsecond / 1e6
        count = min(24, int((duration + offset) // 3600) + 1)
        hours = {(start.hour + index) % 24 for index in range(count)}
        for hour in hours:
            grouped[(str(value), hour)].append(duration)
            grouped[("*", hour)].append(duration)
    thresholds = {f"{value}:{hour}": float(np.quantile(durations, .99)) * 1.2
                  for (value, hour), durations in grouped.items() if len(durations) >= 3}
    return {"policy": policy, "thresholds": thresholds, "completed_runs": len(runs),
            "threshold_policy": "state_and_utc_hour_q99_times_1.2_min_3_runs",
            "runtime": state, "evaluated_on": "received_reports_only"}


def evaluate_persistence(reference, row):
    state = reference["runtime"]
    _advance(state, row, reference["policy"])
    if not state:
        return {"evaluable": False, "reason": "invalid_state", "detected": False}
    key = f"{state['value']}:{row['timestamp_dt'].hour}"
    threshold = reference["thresholds"].get(key)
    basis = "state_and_hour"
    if threshold is None:
        threshold = reference["thresholds"].get(f"*:{row['timestamp_dt'].hour}")
        basis = "hour_fallback"
    duration = max(0., (row["timestamp_dt"] - state["started_at"]).total_seconds())
    evaluable = threshold is not None and reference["policy"]["mode"] != "unknown"
    if reference["policy"]["mode"] == "periodic" and reference["policy"]["max_report_gap_seconds"] is None:
        evaluable = False
    return {"evaluable": evaluable, "reason": None if evaluable else "insufficient_persistence_reference",
            "detected": bool(evaluable and duration > threshold), "duration_seconds": duration,
            "threshold_seconds": threshold, "reference_basis": basis,
            "started_at": state["started_at"].isoformat(), "last_observation_at": state["last_at"].isoformat(),
            "connectivity_verified": False}
