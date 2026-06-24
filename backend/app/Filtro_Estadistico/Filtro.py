from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from .Continuos import FiltroContinuoEstadistico
from .Estados_Binarios import FiltroEstadoBinarioEstadistico


class FiltroEstadistico:
    """Orquesta los filtros estadisticos por tipo de entidad.

    En esta etapa se ejecutan entidades continuas y de estado binario. El filtro
    multivalorico queda fuera del alcance del paso actual.
    """

    def __init__(self) -> None:
        self.continuos = FiltroContinuoEstadistico()
        self.binarios = FiltroEstadoBinarioEstadistico()

    def analizar_historial(
        self,
        eventos: pd.DataFrame,
        entidades: pd.DataFrame,
        suspicion_threshold: float = 0.6,
    ) -> dict[str, Any]:
        inicio_total = perf_counter()
        inicio_continuos = perf_counter()
        resultados_continuos = self.continuos.analizar(
            eventos=eventos,
            entidades=entidades,
            suspicion_threshold=suspicion_threshold,
        )
        tiempo_continuos_ms = (perf_counter() - inicio_continuos) * 1000.0
        inicio_binarios = perf_counter()
        resultados_binarios = self.binarios.analizar(
            eventos=eventos,
            entidades=entidades,
            suspicion_threshold=suspicion_threshold,
        )
        tiempo_binarios_ms = (perf_counter() - inicio_binarios) * 1000.0
        elapsed_ms = (perf_counter() - inicio_total) * 1000.0

        resultados = sorted(
            resultados_continuos + resultados_binarios,
            key=lambda item: item["suspicion_score"],
            reverse=True,
        )
        top_results = resultados[:1000]
        resumen = {
            "results": len(resultados),
            "suspicious": sum(1 for resultado in resultados if resultado.get("is_suspicious")),
            "by_kind": {
                "continuous": len(resultados_continuos),
                "binary": len(resultados_binarios),
            },
            "top_results": top_results,
            "elapsed_ms": round(elapsed_ms, 3),
            "elapsed_seconds": round(elapsed_ms / 1000.0, 6),
            "timings_by_kind_ms": {
                "continuous": round(tiempo_continuos_ms, 3),
                "binary": round(tiempo_binarios_ms, 3),
            },
            "first_detection_delay_seconds": None,
            "implemented_kinds": ["continuous", "binary"],
            "pending_kinds": ["multistate"],
        }
        return {
            **resumen,
            "result_count": len(resultados),
            "results": top_results,
            "top_results": top_results,
            "all_results": resultados,
            "summary": resumen,
        }
