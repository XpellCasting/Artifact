"""Topología dirigida y congelada derivada de un mapa contextual F2."""
from __future__ import annotations

import hashlib
import json


def graph_from_f2_map(f2_map: dict, entity_ids: list[str], top_k: int = 4) -> dict:
    """F2 source se evalúa y target aporta contexto: A[receptor][emisor]."""
    entities = sorted(entity_ids)
    index = {entity: i for i, entity in enumerate(entities)}
    candidates = {entity: [] for entity in entities}
    for edge in f2_map.get("edges", []):
        source, target = edge.get("source"), edge.get("target")
        if edge.get("accepted") and source in index and target in index and source != target:
            strength = edge.get("strength", edge.get("association_signed", 0.0))
            candidates[source].append((abs(float(strength or 0.0)), target, edge))
    adjacency = [[i == j for j in range(len(entities))] for i in range(len(entities))]
    selected = []
    for source in entities:
        for _, target, edge in sorted(candidates[source], key=lambda row: (-row[0], row[1]))[:top_k]:
            adjacency[index[source]][index[target]] = True
            selected.append({"source": source, "target": target,
                             "strength": edge.get("strength", edge.get("association_signed")),
                             "association_signed": edge.get("association_signed"),
                             "lag": edge.get("lag_seconds"), "relation_family": edge.get("relation_family")})
    state = {"entity_ids": entities, "adjacency": adjacency, "edges": selected,
             "spatial_evidence": bool(selected),
             "f2_source_fingerprint": f2_map.get("source_fingerprint")}
    state["graph_version"] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    return state


def build_graph(train_fit_events, entity_catalog, config):
    """Invoca F2 en memoria sobre el rango recibido; nunca reemplaza el mapa operativo."""
    from app.modulos.contextual.contextual import FiltroContextualEspacioTemporal
    import pandas as pd
    events = pd.DataFrame(train_fit_events)
    catalog = pd.DataFrame(entity_catalog)
    f2_map = FiltroContextualEspacioTemporal().construir_mapa(events, catalog, trigger="gat_lstm_train_fit")
    return graph_from_f2_map(f2_map, sorted(catalog["entity_id"].tolist()), config.graph_top_k)
