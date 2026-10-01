from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd


DEFAULT_QUIET_SECONDS = 1800
EPISODE_POLICY_VERSION = "statistical-episodes-v1"


def agrupar_sospechas(
    resultados: list[dict[str, Any]],
    quiet_seconds: int = DEFAULT_QUIET_SECONDS,
    anteriores: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Emite el primer evento; registra repeticiones hasta un lapso de silencio.

    Usa tiempo del evento, nunca tiempo de procesamiento. Una pausa de UI no
    cierra episodios. Opera despues de la supresion interna de F1 y conserva
    todos los scores recibidos para auditoria; el
    estado acotado tiene solo el ultimo episodio por entidad/patron. Los datos
    de entrada y el estado anterior no se modifican (reintentos del llamador).
    """
    if isinstance(quiet_seconds, bool) or not isinstance(quiet_seconds, int) or not 0 <= quiet_seconds <= 86400:
        raise ValueError("El lapso entre episodios debe ser de 0 a 86400 segundos.")
    state = {key: dict(value) for key, value in (anteriores or {}).items()} if quiet_seconds else {}
    ordered = []
    for position, item in enumerate(resultados):
        stamp = pd.to_datetime(item.get("timestamp"), errors="coerce", utc=True)
        ordered.append((pd.Timestamp.max.tz_localize("UTC") if pd.isna(stamp) else stamp, position, item))
    output: list[dict[str, Any] | None] = [None] * len(resultados)
    for timestamp, position, item in sorted(ordered, key=lambda row: (row[0], row[1])):
        enriched = dict(item)
        if not (item.get("is_suspicious") or item.get("passed_to_contextual_filter")):
            output[position] = enriched
            continue
        entity = str(item.get("entity_id") or "")
        pattern = str(item.get("suspected_pattern") or item.get("pattern") or "")
        known = entity and pattern and timestamp != pd.Timestamp.max.tz_localize("UTC")
        key = (entity, pattern)
        previous = state.get(key) if known and quiet_seconds else None
        gap = (timestamp - pd.Timestamp(previous["last_seen"])).total_seconds() if previous else None
        repeated = previous is not None and 0 <= gap < quiet_seconds
        # Desorden temporal inesperado: emitir con diagnostico, no ocultar ni
        # sobrescribir un estado mas reciente con un evento antiguo.
        late = gap is not None and gap < 0
        score = float(item.get("suspicion_score") or 0.0)
        score = score if math.isfinite(score) else 0.0
        stamp_text = str(item.get("timestamp")) if not known else timestamp.isoformat()
        event_id = item.get("event_id") or f"{entity}@{stamp_text}"
        episode = dict(previous) if repeated else {
            "episode_id": hashlib.sha256(f"{entity}|{pattern}|{stamp_text}".encode()).hexdigest()[:24],
            "entity_id": entity,
            "pattern": pattern,
            "first_seen": stamp_text,
            "representative_event_id": event_id,
            "occurrences": 0,
            "max_score": score,
        }
        episode.update(
            last_seen=stamp_text,
            occurrences=episode["occurrences"] + 1,
            max_score=max(score, episode["max_score"]),
            quiet_seconds=quiet_seconds,
            policy_version=EPISODE_POLICY_VERSION,
        )
        episode["duration_seconds"] = (
            (timestamp - pd.Timestamp(episode["first_seen"])).total_seconds() if known else None
        )
        episode["suppressed_count"] = episode["occurrences"] - 1
        if known and quiet_seconds and not late:
            state[key] = episode
        enriched.update(
            event_id=event_id,
            episode={**episode, "decision": "repeat" if repeated else "forward",
                     "reason": "same_episode" if repeated else "out_of_order" if late else
                               "disabled" if not quiet_seconds else "missing_identity_or_time" if not known else
                               "quiet_period_elapsed" if previous else "first_suspicion"},
            suppressed_by_episode=repeated,
            passed_to_contextual_filter=not repeated,
        )
        output[position] = enriched
    return output, state


def enviar_a_contextual(item: dict[str, Any]) -> bool:
    return bool(item.get("is_suspicious") or item.get("passed_to_contextual_filter")) and not item.get("suppressed_by_episode", False)


def resumen_episodios(resultados: list[dict[str, Any]]) -> dict[str, int]:
    detected = sum(bool(item.get("is_suspicious") or item.get("passed_to_contextual_filter")) for item in resultados)
    forwarded = sum(enviar_a_contextual(item) for item in resultados)
    return {"detected": detected, "forwarded": forwarded, "suppressed": detected - forwarded}
