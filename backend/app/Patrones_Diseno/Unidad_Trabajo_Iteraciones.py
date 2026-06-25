from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ResultadoUnidadTrabajo:
    solicitadas: int
    completadas: int


class UnidadTrabajoIteraciones:
    """Agrupa muchos ticks en una sola unidad antes de reanalizar.

    Evita el costo de reejecutar el filtro y reconstruir la respuesta completa
    por cada iteracion individual.
    """

    def __init__(self, total: int) -> None:
        self.total = max(1, int(total))
        self.completadas = 0

    def ejecutar(self, accion_tick: Callable[[], None]) -> ResultadoUnidadTrabajo:
        for _ in range(self.total):
            accion_tick()
            self.completadas += 1
        return ResultadoUnidadTrabajo(solicitadas=self.total, completadas=self.completadas)

