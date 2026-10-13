"""Entrada causal para el modelo único de segmentos multiescala."""
from __future__ import annotations


class CausalInputError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def build_causal_sample(event_provider, target_time, preprocessor: dict, graph: dict, config,
                        labels: dict[str, int] | None = None) -> dict:
    """Construye las tres escalas temporales del modelo SegmentGATLSTM."""
    from app.modulos.clasificador.segments import build_sample
    return build_sample(event_provider, target_time, preprocessor, graph, config, labels)
