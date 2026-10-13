"""Causal per-entity segment graphs; no dependency on F1 or F2."""
from collections import Counter
from datetime import timedelta
from statistics import mean, pstdev
from zoneinfo import ZoneInfo
import hashlib
import json
import math

from app.modulos.clasificador.config import parse_utc
from app.modulos.clasificador.features import _finite_number, dtw_distance, entropy, histogram_id, state_id
from app.services.estados_validos import valid_state

ARCHITECTURE = "gat_lstm_v3_segments_multiscale"
SCHEMA = "gat-lstm-segments-v1"
MAX_SEGMENTS = 12
MAX_POINTS = 24
SCALES = ((3600, 60, (5, 10, 15)), (86400, 900, (8, 12)), (259200, 3600, (6, 12, 24)))
DEFAULT_SEGMENT_STEPS = (10, 8, 12)
BASE_NAMES = ["value", "mean", "min", "max", "std", "changes", "records", "age_seconds",
              "known_fraction", "invalid_fraction", "entropy", "transition_ratio",
              "dominant_fraction", "diversity", "hour_sin", "hour_cos", "hourly_record_rate", "hourly_invalid_rate", "profile_available"]
FEATURE_NAMES = BASE_NAMES + [f"numeric_{i}" for i in range(MAX_POINTS)] + [f"valid_{i}" for i in range(MAX_POINTS)] + [f"category_{i}" for i in range(MAX_POINTS)] + [f"similarity_{i}" for i in range(MAX_SEGMENTS)]


def _validate_segment_steps(horizon, resolution, steps):
    """Reject incompatible saved specs before writing into fixed-size tensors."""
    if type(steps) is not int or not 1 <= steps <= MAX_POINTS:
        raise ValueError("Longitud de segmento inválida para el tensor de entrada")
    count = (horizon // resolution) // steps
    if not 2 <= count <= MAX_SEGMENTS:
        raise ValueError(
            f"La escala de {horizon} segundos genera {count} segmentos; "
            f"se requieren entre 2 y {MAX_SEGMENTS}. Prepara nuevamente el dataset.")


def categorical_dtw(left, right, fraction):
    n, m = len(left), len(right)
    band = math.ceil(fraction * max(n, m))
    previous = {}
    for i in range(n):
        current = {}
        for j in range(max(0, i-band), min(m, i+band+1)):
            choices = ([(0., 0)] if i == j == 0 else [])
            choices += [v for v in (previous.get(j-1), previous.get(j), current.get(j-1)) if v is not None]
            if choices:
                cost, length = min(choices)
                current[j] = (cost + float(left[i] != right[j]), length + 1)
        previous = current
    result = previous.get(m-1)
    return result[0] / result[1] if result else None


def intervals(rows, start, end, resolution, info):
    """Time-weighted summaries; a carry-forward value is not a new reading."""
    rows = sorted((parse_utc(r["timestamp"]), int(r.get("source_row") or 0), str(r["state"]))
                  for r in rows if parse_utc(r["timestamp"]) <= end)
    index, latest = 0, None
    while index < len(rows) and rows[index][0] <= start:
        latest = rows[index]
        index += 1
    ticks = []
    count = int((end-start).total_seconds() // resolution)
    for step in range(count):
        left, right = start + timedelta(seconds=step*resolution), start + timedelta(seconds=(step+1)*resolution)
        cursor, known, invalid, durations, numbers, records, changes = left, 0., 0., Counter(), [], 0, 0
        def accumulate(until):
            nonlocal cursor, known, invalid
            seconds = (until-cursor).total_seconds()
            if latest is not None:
                known += seconds
                if valid_state(latest[2]) and (info["kind"] != "continuous" or _finite_number(latest[2])):
                    durations[latest[2]] += seconds
                    if info["kind"] == "continuous":
                        numbers.append((float(latest[2]), seconds))
                else:
                    invalid += seconds
            cursor = until
        while index < len(rows) and rows[index][0] <= right:
            row = rows[index]
            accumulate(row[0])
            changes += int(latest is not None and row[2] != latest[2])
            latest = row
            records += 1
            index += 1
        accumulate(right)
        valid = latest is not None and valid_state(latest[2]) and (info["kind"] != "continuous" or _finite_number(latest[2]))
        numeric = (float(latest[2])-info["median"])/info["scale"] if valid and info["kind"] == "continuous" else 0.
        scaled = [((v-info["median"])/info["scale"], d) for v, d in numbers]
        ticks.append({"state": latest[2] if latest else None, "numeric": numeric, "valid": valid,
                      "category": state_id(latest[2] if latest else None, info), "known": known/resolution,
                      "invalid": invalid/resolution, "records": records, "changes": changes,
                      "age": (right-latest[0]).total_seconds() if latest else 0.,
                      "durations": durations, "weighted_mean": sum(v*d for v,d in scaled)/max(1., sum(d for _,d in scaled)),
                      "min": min((v for v,_ in scaled), default=0.), "max": max((v for v,_ in scaled), default=0.),
                      "at": right})
    return ticks


def fit_spec(events, preprocessor, config, fit_bounds):
    """Fit segmentation and hourly profiles on train_fit only."""
    start, end = fit_bounds
    zone = ZoneInfo(config.local_timezone)
    grouped = {key: [] for key in preprocessor["entity_ids"]}
    for row in events:
        if row["entity_id"] in grouped and start <= parse_utc(row["timestamp"]) < end:
            grouped[row["entity_id"]].append(row)
    for entity, rows in grouped.items():
        info = preprocessor["entities"][entity]
        known_start = max(start, min((parse_utc(r["timestamp"]) for r in rows), default=end))
        days = int((end-known_start).total_seconds() // 86400)
        counts = Counter(parse_utc(r["timestamp"]).astimezone(zone).hour for r in rows)
        invalid_counts = Counter(parse_utc(r["timestamp"]).astimezone(zone).hour for r in rows if not valid_state(r["state"]))
        info["hourly_profile"] = {"available": days >= 14 and bool(rows),
                                  "record_rate": [counts[h]/max(1, days) for h in range(24)],
                                  "invalid_rate": [invalid_counts[h]/max(1, days) for h in range(24)],
                                  "fit_start": start.isoformat(), "fit_end": end.isoformat()}
    lengths = []
    for scale, (horizon, resolution, candidates) in enumerate(SCALES):
        for candidate in candidates:
            _validate_segment_steps(horizon, resolution, candidate)
        default = DEFAULT_SEGMENT_STEPS[scale]
        if not config.use_entropy:
            lengths.append(default)
            continue
        scores = {w: [] for w in candidates}
        for entity, rows in grouped.items():
            if not rows:
                continue
            info = preprocessor["entities"][entity]
            # Deterministic bounded fit: at most 14 days spread through training.
            duration_days = max(1, int((end-start).total_seconds()//86400))
            chosen = sorted({round(i*(duration_days-1)/max(1, min(14,duration_days)-1)) for i in range(min(14,duration_days))})
            entity_scores = {w: [] for w in candidates}
            for day in chosen:
                a = start + timedelta(days=day)
                b = min(a+timedelta(days=1), end)
                ticks = intervals(rows, a, b, resolution, info)
                for w in candidates:
                    for offset in range(0, len(ticks)-w+1, w):
                        values = [tick["state"] for tick in ticks[offset:offset+w] if tick["valid"]]
                        if len(values) < 3:
                            continue
                        bins = ([histogram_id(float(v), info["histogram_edges"]) for v in values]
                                if info["kind"] == "continuous" else values)
                        counts = Counter(bins)
                        h = -sum((c/len(bins))*math.log(c/len(bins)) for c in counts.values())
                        entity_scores[w].append(h/math.log(w))
            for w in candidates:
                if entity_scores[w]:
                    scores[w].append(mean(entity_scores[w]))
        lengths.append(max(candidates, key=lambda w: (mean(scores[w]) if scores[w] else -1., w))
                       if any(scores.values()) else default)
    nodes = len(preprocessor["entity_ids"])
    result = {"method": "within_entity_segment_dtw", "entity_ids": preprocessor["entity_ids"],
              "segment_steps": lengths, "threshold": config.segment_threshold,
              "scales": [list(s[:2]) for s in SCALES], "local_timezone": config.local_timezone,
              "spatial_evidence": False, "adjacency": [[i == j for j in range(nodes)] for i in range(nodes)],
              "fit_start": start.isoformat(), "fit_end": end.isoformat(), "schema_version": SCHEMA}
    result["graph_version"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    preprocessor["feature_names"] = FEATURE_NAMES
    return result


def build_sample(provider, target, preprocessor, graph, config, labels=None):
    lengths = graph["segment_steps"]
    if len(lengths) != len(SCALES):
        raise ValueError("Se requiere una longitud de segmento para cada escala")
    for (horizon, resolution, _), steps in zip(SCALES, lengths):
        _validate_segment_steps(horizon, resolution, steps)
    target = parse_utc(target) if isinstance(target, str) else target
    start = target-timedelta(seconds=SCALES[-1][0])
    entities = preprocessor["entity_ids"]
    rows = provider(start, target, entities)
    grouped = {key: [] for key in entities}
    for row in rows:
        if row["entity_id"] in grouped and parse_utc(row["timestamp"]) <= target:
            grouped[row["entity_id"]].append(row)
    x = [[[0.]*len(FEATURE_NAMES) for _ in entities] for _ in range(3*MAX_SEGMENTS)]
    states = [[0 for _ in entities] for _ in x]
    present = [[False for _ in entities] for _ in x]
    coverage = {}
    zone = ZoneInfo(config.local_timezone)
    for node, entity in enumerate(entities):
        info = preprocessor["entities"][entity]
        coverage[entity] = []
        if not grouped[entity]:
            coverage[entity] = [{"horizon_seconds": horizon, "known_fraction": 0.,
                                 "segments": (horizon // resolution) // graph["segment_steps"][scale],
                                 "usable": False} for scale, (horizon, resolution, _) in enumerate(SCALES)]
            continue
        for scale, (horizon, resolution, _) in enumerate(SCALES):
            ticks = intervals(grouped[entity], target-timedelta(seconds=horizon), target, resolution, info)
            w = graph["segment_steps"][scale]
            chunks = [ticks[i:i+w] for i in range(0, len(ticks), w) if len(ticks[i:i+w]) == w]
            usable = config.active_scales[scale] and len(chunks) >= 2 and mean(t["known"] for t in ticks) >= .5
            coverage[entity].append({"horizon_seconds": horizon, "known_fraction": mean(t["known"] for t in ticks),
                                     "segments": len(chunks), "usable": usable})
            sequences = []
            for i, chunk in enumerate(chunks):
                valid = [t for t in chunk if t["valid"]]
                numerical = info["kind"] == "continuous"
                series = [t["numeric"] if numerical else t["state"] for t in chunk]
                sequences.append(series if all(t["valid"] for t in chunk) else None)
                nums = [t["numeric"] for t in valid] if numerical else []
                frequencies = Counter()
                for t in chunk:
                    frequencies.update(t["durations"])
                bins = ([histogram_id(float(t["state"]), info["histogram_edges"]) for t in valid] if numerical else [t["state"] for t in valid])
                ent, _ = entropy(bins, max(2, config.entropy_bins if numerical else len(info.get("vocab", {}))))
                hour = chunk[-1]["at"].astimezone(zone).hour
                profile = info.get("hourly_profile", {})
                profile_ok = profile.get("available", False)
                features = [chunk[-1]["numeric"], mean(t["weighted_mean"] for t in chunk) if numerical else 0.,
                            min((t["min"] for t in chunk), default=0.), max((t["max"] for t in chunk), default=0.),
                            pstdev(nums) if len(nums)>1 else 0., sum(t["changes"] for t in chunk), sum(t["records"] for t in chunk),
                            chunk[-1]["age"]/horizon, mean(t["known"] for t in chunk), mean(t["invalid"] for t in chunk),
                            ent if config.use_entropy else 0., sum(t["changes"] for t in chunk)/max(1,w),
                            max(frequencies.values(), default=0.)/max(1.,sum(frequencies.values())), len(frequencies)/max(1,w),
                            math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24),
                            math.log1p(profile["record_rate"][hour]) if profile_ok else 0.,
                            math.log1p(profile.get("invalid_rate", [0.]*24)[hour]) if profile_ok else 0., float(profile_ok)]
                features += [t["numeric"] for t in chunk]+[0.]*(MAX_POINTS-w)
                features += [float(t["valid"]) for t in chunk]+[0.]*(MAX_POINTS-w)
                features += [float(t["category"]) if not numerical else 0. for t in chunk]+[0.]*(MAX_POINTS-w)
                features += [0.]*MAX_SEGMENTS
                slot = scale*MAX_SEGMENTS+i
                x[slot][node] = features
                states[slot][node] = chunk[-1]["category"] if not numerical else 0
                present[slot][node] = usable
            for i in range(len(chunks)):
                for j in range(len(chunks)):
                    similarity = float(i == j)
                    if i != j and config.use_dtw and sequences[i] is not None and sequences[j] is not None:
                        distance = (dtw_distance(sequences[i], sequences[j], config.dtw_band_fraction) if info["kind"] == "continuous"
                                    else categorical_dtw(sequences[i], sequences[j], config.dtw_band_fraction))
                        similarity = 1/(1+distance) if distance is not None else 0.
                    x[scale*MAX_SEGMENTS+i][node][-MAX_SEGMENTS+j] = similarity
    y = [(labels or {}).get(entity, -1) for entity in entities]
    return {"X": x, "state_ids": states, "node_present": present, "adjacency": graph["adjacency"],
            "y": y, "label_mask": [label >= 0 and any(s["usable"] for s in coverage[entity]) for entity, label in zip(entities, y)],
            "metadata": {"target_time": target.isoformat(), "context_start": start.isoformat(), "entity_ids": entities,
                         "architecture_version": ARCHITECTURE, "scale_coverage": coverage,
                         "input_provenance": sorted({str(r.get("provenance", "")) for r in rows})}}
