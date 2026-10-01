from __future__ import annotations
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from app.services.application_common import serializable_json
from app.config import database_path, CACHE_DIR

class DatabaseReplayMixin:
    def _ruta_almacen_resultados_validacion(self) -> Path:
        return database_path("validation_replay_results.sqlite3", self.analysis_cache_path.parent if self.analysis_cache_path.parent.resolve() != CACHE_DIR.resolve() else None)
    def _reiniciar_almacen_resultados_validacion(self) -> None:
        path = self._ruta_almacen_resultados_validacion()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=30.0)) as connection:
            with connection:
                connection.execute("DROP TABLE IF EXISTS replay_results")
                connection.execute(
                    """
                    CREATE TABLE replay_results (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind TEXT NOT NULL,
                        payload TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX replay_results_kind_sequence "
                    "ON replay_results(kind, sequence DESC)"
                )
        self.validation_results_store_ready = True
    def _guardar_resultados_validacion(
        self,
        estadisticos: list[dict[str, Any]],
        contextuales: list[dict[str, Any]],
        salidas: dict[str, Any],
    ) -> None:
        if not self.validation_results_store_ready:
            self._reiniciar_almacen_resultados_validacion()
        por_tipo = {
            "statistical": estadisticos,
            "contextual": contextuales,
            "environmental": salidas.get("environmental_results", []),
            "pending": salidas.get("pending_classification_results", []),
            "classifier": salidas.get("classifier_results", []),
        }
        rows = [
            (
                kind,
                json.dumps(
                    serializable_json(item),
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
            )
            for kind, items in por_tipo.items()
            for item in items
        ]
        if not rows:
            return
        with closing(
            sqlite3.connect(
                self._ruta_almacen_resultados_validacion(), timeout=30.0
            )
        ) as connection:
            with connection:
                connection.executemany(
                    "INSERT INTO replay_results(kind, payload) VALUES (?, ?)",
                    rows,
                )
    def _leer_resultados_validacion(
        self,
        kind: str,
        offset: int,
        limit: int,
    ) -> tuple[int, list[dict[str, Any]]]:
        with closing(
            sqlite3.connect(
                self._ruta_almacen_resultados_validacion(), timeout=30.0
            )
        ) as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM replay_results WHERE kind = ?",
                    (kind,),
                ).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT payload FROM replay_results WHERE kind = ? "
                "ORDER BY sequence DESC LIMIT ? OFFSET ?",
                (kind, limit, offset),
            ).fetchall()
        return total, [json.loads(row[0]) for row in rows]
