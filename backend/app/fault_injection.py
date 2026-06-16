from __future__ import annotations

from abc import ABC
from random import Random
from typing import Any


class FaultInjector(ABC):
    """Clase base para inyectar fallas sinteticas."""

    def apply_series(
        self,
        entity: dict[str, Any],
        values: list[Any],
        index: int,
        start: int,
        end: int,
        fault: dict[str, Any],
        rng: Random,
    ) -> None:
        values[index] = self.apply_value(entity, values[index], fault, index, start, end, rng)

    def apply_tick(
        self,
        entity: dict[str, Any],
        clean_state: Any,
        fault: dict[str, Any],
        tick_index: int,
        rng: Random,
    ) -> tuple[Any, bool]:
        return self.apply_value(entity, clean_state, fault, tick_index, 0, max(tick_index, 1), rng), False

    def apply_value(
        self,
        entity: dict[str, Any],
        value: Any,
        fault: dict[str, Any],
        index: int,
        start: int,
        end: int,
        rng: Random,
    ) -> Any:
        return value

    def sensor_kind(self, entity: dict[str, Any]) -> str:
        kind = str(entity.get("sensor_kind", ""))
        if kind:
            return kind
        tipo = str(entity.get("tipo_interno", ""))
        if tipo == "continua":
            return "continuous"
        if tipo == "estado_binario":
            return "binary"
        return "multistate"


class MissingFaultInjector(FaultInjector):
    def apply_series(
        self,
        entity: dict[str, Any],
        values: list[Any],
        index: int,
        start: int,
        end: int,
        fault: dict[str, Any],
        rng: Random,
    ) -> None:
        values[index] = None

    def apply_tick(
        self,
        entity: dict[str, Any],
        clean_state: Any,
        fault: dict[str, Any],
        tick_index: int,
        rng: Random,
    ) -> tuple[Any, bool]:
        return clean_state, True


class RandomSpikeFaultInjector(FaultInjector):
    def apply_value(
        self,
        entity: dict[str, Any],
        value: Any,
        fault: dict[str, Any],
        index: int,
        start: int,
        end: int,
        rng: Random,
    ) -> Any:
        kind = self.sensor_kind(entity)
        if kind == "continuous":
            base = float(value)
            if start == 0:
                return round(base + rng.choice([-1, 1]) * rng.uniform(6.0, 12.0), 2)
            return round(base + (8 if index in {start, (start + end) // 2} else 0), 2)
        if kind == "binary":
            return "on" if (index - start) % 3 != 0 else "off"
        return rng.choice(["emergency", "vacation", "guest", "unknown_mode"])


class MalfunctionFaultInjector(FaultInjector):
    def apply_value(
        self,
        entity: dict[str, Any],
        value: Any,
        fault: dict[str, Any],
        index: int,
        start: int,
        end: int,
        rng: Random,
    ) -> Any:
        kind = self.sensor_kind(entity)
        if kind == "continuous":
            return round(float(value) + rng.gauss(0, 6.0), 2)
        if kind == "binary":
            return "on" if index % 2 == 0 else "off"
        options = ["home", "away", "vacation", "guest", "emergency"]
        return options[index % len(options)]


class BiasFaultInjector(FaultInjector):
    def apply_value(
        self,
        entity: dict[str, Any],
        value: Any,
        fault: dict[str, Any],
        index: int,
        start: int,
        end: int,
        rng: Random,
    ) -> Any:
        kind = self.sensor_kind(entity)
        if kind == "continuous":
            return round(float(value) + 5.0, 2)
        if kind == "binary":
            return str(fault.get("stuck_value") if fault.get("stuck_value") is not None else "on")
        return "away"


class DriftFaultInjector(FaultInjector):
    def apply_value(
        self,
        entity: dict[str, Any],
        value: Any,
        fault: dict[str, Any],
        index: int,
        start: int,
        end: int,
        rng: Random,
    ) -> Any:
        kind = self.sensor_kind(entity)
        progress = (index - start) / max(end - start, 1)
        if kind == "continuous":
            if start == 0:
                return round(float(value) + min(10.0, 0.18 * index), 2)
            return round(float(value) - 10.0 * progress, 2)
        if kind == "binary":
            return "on" if index > start + 5 else value
        return "away" if index > start + 5 else value


class StuckAtFaultInjector(FaultInjector):
    def apply_series(
        self,
        entity: dict[str, Any],
        values: list[Any],
        index: int,
        start: int,
        end: int,
        fault: dict[str, Any],
        rng: Random,
    ) -> None:
        values[index] = fault.get("stuck_value") if fault.get("stuck_value") is not None else values[start]

    def apply_value(
        self,
        entity: dict[str, Any],
        value: Any,
        fault: dict[str, Any],
        index: int,
        start: int,
        end: int,
        rng: Random,
    ) -> Any:
        if fault.get("stuck_value") is not None:
            return fault["stuck_value"]
        kind = self.sensor_kind(entity)
        if kind == "continuous":
            return default_stuck_value(entity)
        if kind == "binary":
            return "on"
        return value


class NoopFaultInjector(FaultInjector):
    pass


class FaultInjectorFactory:
    """Factory Method para crear inyectores segun el tipo canonico de falla."""

    def create(self, fault_type: str) -> FaultInjector:
        fault_type = str(fault_type).strip().lower()
        if fault_type == "missing":
            return MissingFaultInjector()
        if fault_type in {"random", "random/spike", "spike"}:
            return RandomSpikeFaultInjector()
        if fault_type == "malfunction":
            return MalfunctionFaultInjector()
        if fault_type == "bias":
            return BiasFaultInjector()
        if fault_type == "drift":
            return DriftFaultInjector()
        if fault_type == "stuck-at":
            return StuckAtFaultInjector()
        return NoopFaultInjector()


def default_stuck_value(entity: dict[str, Any]) -> Any:
    entity_id = str(entity.get("entity_id", ""))
    if entity_id.endswith("temperature"):
        return 20.0
    if entity_id.endswith("humidity"):
        return 50.0
    return "on"
