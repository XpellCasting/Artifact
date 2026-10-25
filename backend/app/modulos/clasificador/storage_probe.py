"""Muestreo reproducible de instantes; nunca recorta el preprocesamiento temporal."""
from collections import defaultdict
from hashlib import sha256
import json

import numpy as np

from app.modulos.clasificador.compressed_storage import SHARED, LABELS_DTYPES


def sample_digest(sample):
    digest = sha256()
    for name, dtype in {**SHARED, **LABELS_DTYPES}.items():
        values = np.ascontiguousarray(sample[name], dtype=dtype)
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    digest.update(json.dumps(sample["metadata"], sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()


def select_probe_targets(targets, limit):
    if limit < 1:
        raise ValueError("La muestra debe contener al menos un instante")
    instants = defaultdict(list)
    for target in targets:
        instants[target.timestamp].append(target)
    strata = defaultdict(list)
    for stamp, rows in instants.items():
        key = (stamp[:7], rows[0].partition, tuple(sorted({row.label for row in rows})))
        strata[key].append(stamp)
    if limit < len(strata):
        raise ValueError(f"La muestra requiere al menos {len(strata)} instantes para cubrir meses, particiones y clases")
    limit = min(limit, len(instants))
    counts = {key: 1 for key in strata}
    while sum(counts.values()) < limit:
        key = max((key for key in strata if counts[key] < len(strata[key])),
                  key=lambda key: len(strata[key]) / (counts[key] + 1))
        counts[key] += 1
    chosen, summary = set(), []
    for key, stamps in sorted(strata.items()):
        picked = sorted(stamps, key=lambda stamp: sha256(stamp.encode()).digest())[:counts[key]]
        chosen.update(picked)
        summary.append({"month": key[0], "partition": key[1], "labels": key[2],
                        "full_instants": len(stamps), "probe_instants": len(picked)})
    rows = [target for target in targets if target.timestamp in chosen]
    return rows, {"full_samples": len(targets), "full_unique_windows": len(instants),
                  "probe_samples": len(rows), "probe_unique_windows": len(chosen), "strata": summary}
