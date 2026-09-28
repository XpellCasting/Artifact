"""Oráculo escalar previo a la optimización, solo para pruebas."""
from typing import Any
import pandas as pd
from app.Filtro_Contextual.Contextual import FiltroContextualEspacioTemporal

class ReferenceContext(FiltroContextualEspacioTemporal):
    def _muestras_modelo_temporal(self, edge, observations, bounds: dict[str, Any]) -> dict[str, pd.DataFrame]:
        source, target = str(edge["source"]), str(edge["target"])
        source_events, target_events = observations.get(source), observations.get(target)
        rows = {name: [] for name in ("level", "innovation", "trend")}
        if source_events is None or target_events is None or source_events.empty or target_events.empty:
            return {name: pd.DataFrame(columns=["target", "context", "timestamp"]) for name in rows}
        start, end = (pd.to_datetime(bounds.get(key), errors="coerce", utc=True) for key in ("start", "end"))
        if pd.isna(start) or pd.isna(end):
            return {name: pd.DataFrame(columns=["target", "context", "timestamp"]) for name in rows}
        candidates = source_events[(source_events.index > start) & (source_events.index <= end)]
        # Solo la ultima lectura estable de un timestamp puede ser x(t).
        candidates = candidates[~candidates.index.duplicated(keep="last")]
        for timestamp, row in candidates.iterrows():
            if not bool(row.get("is_valid_value")):
                continue
            actual = self._numeric_or_none(row.get("value", row.get("state")))
            if actual is None:
                continue
            source_features = self._caracteristicas_continuas(
                source_events, timestamp, source, window_seconds=int(edge["window_seconds"]),
                max_age_seconds=edge.get("max_age_seconds"), current_value=actual, partition_start=start,
            )
            target_features = self._caracteristicas_continuas(
                target_events, timestamp - pd.Timedelta(seconds=int(edge.get("lag_seconds", 0))), target,
                window_seconds=int(edge["window_seconds"]), max_age_seconds=edge.get("max_age_seconds"), partition_start=start,
            )
            for name in rows:
                response, predictor = source_features.get(name), target_features.get(name)
                if response is not None and predictor is not None:
                    rows[name].append({"timestamp": timestamp, "target": response, "context": predictor})
        return {name: pd.DataFrame(values, columns=["timestamp", "target", "context"]) for name, values in rows.items()}

    def _calibrar_decisiones_temporales(self, edges, observations, partitions, progress_callback=None) -> dict[str, Any]:
        by_source: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            if edge.get("accepted") and edge.get("relation_family") == "continuous_continuous":
                by_source.setdefault(str(edge["source"]), []).append(edge)
        results: dict[str, Any] = {}
        for source_index, (source, source_edges) in enumerate(by_source.items()):
            if progress_callback is not None:
                progress_callback("context_decisions", 75.0 + 20.0 * source_index / max(1, len(by_source)),
                                  f"Calibrando decisiones: {source} ({source_index}/{len(by_source)} sensores completados)")
            events = observations.get(source)
            bounds = partitions["decision"]
            start, end = (pd.to_datetime(bounds.get(key), errors="coerce", utc=True) for key in ("start", "end"))
            groups: dict[str, list[dict[str, Any]]] = {}
            if events is not None and not pd.isna(start) and not pd.isna(end):
                candidates = events[(events.index > start) & (events.index <= end)]
                candidates = candidates[~candidates.index.duplicated(keep="last")]
                for row_index, (timestamp, row) in enumerate(candidates.iterrows()):
                    if progress_callback is not None and row_index % 100 == 0:
                        progress_callback("context_decisions", 75.0 + 20.0 * (source_index + row_index / max(1, len(candidates))) / max(1, len(by_source)),
                                          f"Calibrando decisiones: {source} — {row_index}/{len(candidates)} eventos; {source_index}/{len(by_source)} sensores completados")
                    actual = self._numeric_or_none(row.get("value", row.get("state"))) if bool(row.get("is_valid_value")) else None
                    scores = self._puntajes_familiares(source, timestamp, actual, source_edges, observations, partition_start=start)
                    families = scores["available_families"]
                    if families:
                        key = "+".join(families)
                        groups.setdefault(key, []).append({"score": scores["decision_score"], "neighbors": scores["evaluable_neighbors"]})
            calibrated = {}
            for key, values in groups.items():
                if len(values) < self.config.temporal_decision_min_observations:
                    calibrated[key] = {"status": "insufficient_calibration", "samples": len(values)}
                    continue
                series = pd.Series([item["score"] for item in values])
                calibrated[key] = {"status": "available", "samples": len(values),
                                   "threshold": float(series.quantile(self.config.residual_quantile, interpolation="higher")),
                                   "neighbor_counts": sorted({int(item["neighbors"]) for item in values}),
                                   "partition": "decision", "quantile": self.config.residual_quantile}
            results[source] = {"partition": bounds, "combinations": calibrated}
        return results


    def _calibrar_modelos_temporales(self, edge, observations, partitions):
        fit = self._muestras_modelo_temporal(edge, observations, partitions['fit'])
        tolerance = self._muestras_modelo_temporal(edge, observations, partitions['tolerance'])
        models = {name: self._ajustar_modelo_temporal(name, fit[name], tolerance[name])
                  for name in ('level', 'innovation', 'trend')}
        return {'partitioning': partitions, 'models': models,
                'sample_counts': {name: {'fit': len(fit[name]), 'tolerance': len(tolerance[name])} for name in models}}
