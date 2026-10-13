"""Reporting semantics and elapsed coverage, without inventing observations."""
import math

import pandas as pd

MODES = {"on_change", "periodic", "unknown"}


def reporting_policy(frame):
    modes = set(frame.get("reporting_mode", pd.Series(dtype=str)).dropna().astype(str)) - {""}
    if not modes <= MODES or len(modes) > 1:
        raise ValueError("reporting_mode debe ser on_change, periodic o unknown por entidad")
    mode = next(iter(modes), "unknown")
    limits = pd.to_numeric(frame.get("max_report_gap_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(limits) and (not all(math.isfinite(float(x)) and x > 0 for x in limits) or limits.nunique() > 1):
        raise ValueError("max_report_gap_seconds debe ser positivo y único por entidad")
    return {"mode": mode, "max_report_gap_seconds": float(limits.iloc[0]) if len(limits) else None,
            "availability_basis": "explicit_invalid_states_only",
            "silence_proves_disconnection": False}


def coverage(frame, start, end, policy):
    """Bounded calendar exposure. Invalid reports break held state coverage.

    On-change exposure assumes a complete event log; it is not a heartbeat or
    evidence of connectivity. Periodic exposure requires a declared maximum gap.
    Unknown sources expose their observed span, explicitly marked uncertain.
    """
    work = frame.sort_values("timestamp_dt", kind="stable")
    rows = list(work.to_dict("records"))
    seconds = 0.0
    for left, right in zip(rows, rows[1:]):
        if not bool(left.get("is_valid_value", True)):
            continue
        value = left.get("value")
        if left.get("sensor_kind") == "continuous" and (value is None or pd.isna(value) or not math.isfinite(float(value))):
            continue
        a, b = max(start, left["timestamp_dt"]), min(end, right["timestamp_dt"])
        if b <= a:
            continue
        limit = policy["max_report_gap_seconds"]
        if policy["mode"] == "periodic":
            if limit is None:
                continue
            b = min(b, left["timestamp_dt"] + pd.Timedelta(seconds=limit))
        seconds += max(0., (b - a).total_seconds())
    calendar = max(0., (end - start).total_seconds())
    seconds = min(seconds, calendar)
    return {"calendar_days": calendar / 86400, "exposure_days": seconds / 86400,
            "uncovered_days": max(0., calendar - seconds) / 86400,
            "coverage_basis": "held_state_assuming_complete_change_log" if policy["mode"] == "on_change" else
                              "declared_periodic_gap" if policy["mode"] == "periodic" else "observed_span_connectivity_unknown",
            "connectivity_verified": False}


def daily_partitions(timestamps):
    """24-hour blocks from the first sample, retaining the final partial block."""
    first, last = timestamps[0], timestamps[-1]
    start = first
    blocks = max(0, int((last - start).total_seconds() // 86400))
    end = start + pd.Timedelta(days=blocks)
    holdout = max(7, int(math.ceil(blocks * .15)))
    threshold_start = end - pd.Timedelta(days=holdout)
    normalization_start = threshold_start - pd.Timedelta(days=holdout)
    split = sum(t < normalization_start for t in timestamps)
    midpoint = sum(t < threshold_start for t in timestamps)
    stop = len(timestamps)
    return split, midpoint, stop, {"policy": "elapsed_24_hour_blocks_v1", "block_hours": 24,
        "complete_blocks": blocks, "predictor_start": first.isoformat(),
        "normalization_start": normalization_start.isoformat(),
        "threshold_start": threshold_start.isoformat(), "threshold_end": last.isoformat(),
        "last_observation_at": last.isoformat(),
        "final_partial_block_hours": (last - end).total_seconds() / 3600,
        "predictor_fit": split, "normalization": midpoint - split, "threshold": stop - midpoint,
        "normalization_calendar_days": holdout,
        "threshold_calendar_days": (last - threshold_start).total_seconds() / 86400,
        "enough_calendar_blocks": blocks >= 2 * holdout + 1}
