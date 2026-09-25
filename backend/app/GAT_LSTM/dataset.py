"""Índice de objetivos supervisados con etiquetas parciales y grupos estables."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from . import LABELS
from .config import actual_partition, parse_utc, required_context_seconds


@dataclass(frozen=True)
class Target:
    entity_id: str
    timestamp: str
    label: int
    annotation_id: str
    partition: str
    episode_id: str
    source_row: int | None


def index_labelled_targets(annotations: list[dict], event_provider, bounds: dict,
                           config) -> tuple[list[Target], list[dict]]:
    """Los intervalos etiquetan únicamente eventos reales de la entidad objetivo.

    Los puntos conservan su instante exacto. Los intervalos usan extremos
    inclusivos, conforme al exportador de ReviewStore.
    """
    targets = {}
    conflicts = set()
    excluded = []
    group_parts = defaultdict(set)
    context = timedelta(seconds=required_context_seconds(config))
    for ann in annotations:
        if ann.get("status") not in {"reviewed", "verified"} or ann.get("label") not in LABELS:
            continue
        start, end = parse_utc(ann["start"]), parse_utc(ann["end"])
        episode = ann.get("parent_episode_id") or ann["annotation_id"]
        begin_part = actual_partition(start, bounds)
        end_part = actual_partition(end, bounds)
        group_parts[episode].update((begin_part, end_part))
        if begin_part != end_part or begin_part is None:
            excluded.append({"annotation_id": ann["annotation_id"], "reason": "partition_crossing"})
            continue
        lo, hi = bounds[begin_part]
        if start - context < lo or (hi is not None and end >= hi):
            excluded.append({"annotation_id": ann["annotation_id"], "reason": "context_crossing"})
            continue
        if ann.get("source_row") is not None or start == end:
            moments = [(start, ann.get("source_row"))]
        else:
            rows = event_provider(start, end, [ann["entity_id"]])
            moments = [(parse_utc(row["timestamp"]), row.get("source_row")) for row in rows
                       if row.get("entity_id") == ann["entity_id"]
                       and start <= parse_utc(row["timestamp"]) <= end]
            # Deterministic thinning without moving the interval endpoint.
            stride = timedelta(seconds=config.labelled_stride_seconds)
            last = None
            kept = []
            for moment in sorted(moments):
                if last is None or moment[0] - last >= stride:
                    kept.append(moment); last = moment[0]
            moments = kept
        for moment, source_row in moments:
            key = (ann["entity_id"], moment.isoformat())
            row = Target(ann["entity_id"], moment.isoformat(), LABELS.index(ann["label"]),
                         ann["annotation_id"], begin_part, episode, source_row)
            if key in targets and targets[key].label != row.label:
                targets.pop(key)
                conflicts.add(key)
                excluded.append({"annotation_id": ann["annotation_id"], "reason": "label_conflict",
                                 "timestamp": moment.isoformat()})
            elif key not in targets and key not in conflicts:
                targets[key] = row
    shared = {group for group, parts in group_parts.items() if len(parts) > 1}
    if shared:
        targets = {key: value for key, value in targets.items() if value.episode_id not in shared}
        excluded.extend({"episode_id": group, "reason": "shared_episode"} for group in sorted(shared))
    return sorted(targets.values(), key=lambda item: (item.timestamp, item.entity_id)), excluded
