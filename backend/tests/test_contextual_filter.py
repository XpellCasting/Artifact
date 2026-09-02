from __future__ import annotations

import unittest

import pandas as pd

from app.Filtro_Contextual import (
    ConfiguracionContextual,
    FiltroContextualEspacioTemporal,
)


def construir_historial(relacion_inestable: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    inicio = pd.Timestamp("2026-01-01T00:00:00Z")
    filas = []
    for indice in range(120):
        timestamp = inicio + pd.Timedelta(hours=indice)
        base = 1.0 if indice % 2 else -1.0
        humidity = base * 2.0
        if relacion_inestable and indice >= 60:
            humidity *= -1.0
        valores = {
            "sensor.temperature": ("continuous", base, str(base)),
            "sensor.humidity": ("continuous", humidity, str(humidity)),
            "binary_sensor.climate": (
                "binary",
                1.0 if base > 0 else 0.0,
                "on" if base > 0 else "off",
            ),
        }
        for entity_id, (kind, value, state) in valores.items():
            filas.append(
                {
                    "entity_id": entity_id,
                    "timestamp": timestamp.isoformat(),
                    "timestamp_dt": timestamp,
                    "state": state,
                    "value": value,
                    "sensor_kind": kind,
                    "is_valid_value": True,
                }
            )
    entidades = pd.DataFrame(
        [
            {
                "entity_id": "sensor.temperature",
                "sensor_kind": "continuous",
                "device_class": "temperature",
                "domain": "sensor",
                "unit_of_measurement": "C",
            },
            {
                "entity_id": "sensor.humidity",
                "sensor_kind": "continuous",
                "device_class": "humidity",
                "domain": "sensor",
                "unit_of_measurement": "%",
            },
            {
                "entity_id": "binary_sensor.climate",
                "sensor_kind": "binary",
                "device_class": "running",
                "domain": "binary_sensor",
                "unit_of_measurement": "",
            },
        ]
    )
    return pd.DataFrame(filas), entidades


class ContextualFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ConfiguracionContextual(
            top_k=2,
            window_seconds=3600,
            max_lag_windows=1,
            association_threshold=0.35,
            min_observations=30,
            min_pair_coverage=0.60,
            stability_slices=4,
            min_stable_slices=3,
            stability_association_threshold=0.20,
            min_sign_consistency=0.75,
        )
        self.filtro = FiltroContextualEspacioTemporal(self.config)

    def test_mapa_expone_regla_reproducible_covarianza_y_relacion_mixta(self):
        eventos, entidades = construir_historial()

        mapa = self.filtro.construir_mapa(eventos, entidades)

        self.assertEqual(
            mapa["implementation_sha256"],
            self.filtro.implementation_sha256(),
        )
        self.assertEqual(mapa["config"]["top_k"], 2)
        self.assertEqual(mapa["config"]["window_seconds"], 3600)
        self.assertEqual(mapa["config"]["min_observations"], 30)
        self.assertIn("acceptance_rule", mapa["config"])
        self.assertEqual(
            len(mapa["matrices"]["covariance"]),
            len(mapa["matrices"]["entities"]),
        )
        temp_index = mapa["matrices"]["entities"].index("sensor.temperature")
        humidity_index = mapa["matrices"]["entities"].index("sensor.humidity")
        climate_index = mapa["matrices"]["entities"].index("binary_sensor.climate")
        self.assertIsNotNone(mapa["matrices"]["covariance"][temp_index][humidity_index])
        self.assertIsNone(mapa["matrices"]["covariance"][temp_index][climate_index])
        mixed = next(
            edge
            for edge in mapa["edges"]
            if edge["source"] == "sensor.temperature"
            and edge["target"] == "binary_sensor.climate"
        )
        self.assertEqual(mixed["relation_family"], "mixed")
        self.assertEqual(mixed["metric"], "point_biserial")
        self.assertTrue(mixed["accepted"])
        self.assertGreaterEqual(mixed["stable_slices"], 3)

    def test_relacion_temporalmente_inestable_no_se_acepta(self):
        eventos, entidades = construir_historial(relacion_inestable=True)

        mapa = self.filtro.construir_mapa(eventos, entidades)

        edge = next(
            item
            for item in mapa["edges"]
            if item["source"] == "sensor.temperature"
            and item["target"] == "sensor.humidity"
        )
        self.assertFalse(edge["accepted"])
        self.assertTrue(
            {"weak_association", "temporal_instability", "sign_instability"}
            & set(edge["rejection_reasons"])
        )

    def test_cj_distingue_respaldo_contradiccion_y_cobertura(self):
        historico, entidades = construir_historial()
        mapa = self.filtro.construir_mapa(historico, entidades)
        timestamp = pd.Timestamp("2026-01-06T00:00:00Z")
        sospecha = {
            "entity_id": "sensor.temperature",
            "timestamp": timestamp.isoformat(),
            "sensor_kind": "continuous",
            "value": 8.0,
            "suspicion_score": 0.8,
            "is_suspicious": True,
            "passed_to_contextual_filter": True,
            "pattern": "abrupt_change",
        }

        supported_events = self._agregar_ventana(
            historico,
            timestamp,
            temperature=8.0,
            humidity=16.0,
            climate_state="on",
            climate_valid=True,
        )
        supported = self.filtro.analizar(
            [sospecha], supported_events, entidades, mapa
        )["results"][0]
        self.assertEqual(supported["context_outcome"], "supported")
        self.assertGreater(supported["support_score"], 0)
        self.assertTrue(
            any(item["c_j_normalized"] > 0 for item in supported["supports"])
        )

        contradicted_events = self._agregar_ventana(
            historico,
            timestamp,
            temperature=8.0,
            humidity=-16.0,
            climate_state="off",
            climate_valid=True,
        )
        contradicted = self.filtro.analizar(
            [sospecha], contradicted_events, entidades, mapa
        )["results"][0]
        self.assertEqual(contradicted["context_outcome"], "contradicted")
        self.assertGreater(contradicted["contradiction_score"], 0)

        missing_events = self._agregar_ventana(
            historico,
            timestamp,
            temperature=8.0,
            humidity=None,
            climate_state="unavailable",
            climate_valid=False,
        )
        missing = self.filtro.analizar(
            [sospecha], missing_events, entidades, mapa
        )["results"][0]
        self.assertEqual(missing["context_outcome"], "insufficient_coverage")
        self.assertEqual(missing["suspicion_effect"], "maintain")
        self.assertEqual(missing["contradiction_score"], 0.0)

    def test_patron_multientidad_marca_causa_comun_no_identificable(self):
        eventos, entidades = construir_historial()
        mapa = self.filtro.construir_mapa(eventos, entidades)
        timestamp = pd.Timestamp("2026-01-05T23:00:00Z").isoformat()
        sospechas = [
            {
                "entity_id": entity_id,
                "timestamp": timestamp,
                "sensor_kind": kind,
                "value": 1.0,
                "suspicion_score": 0.8,
                "is_suspicious": True,
            }
            for entity_id, kind in [
                ("sensor.temperature", "continuous"),
                ("sensor.humidity", "continuous"),
                ("binary_sensor.climate", "binary"),
            ]
        ]

        resultados = self.filtro.analizar(sospechas, eventos, entidades, mapa)["results"]

        self.assertTrue(all(item["common_cause"]["candidate"] for item in resultados))
        self.assertTrue(
            all(
                item["common_cause"]["status"]
                == "unidentifiable_without_independent_evidence"
                for item in resultados
            )
        )
        self.assertTrue(
            all(not item["common_cause"]["independent_evidence_available"] for item in resultados)
        )
        self.assertTrue(
            all(item["adjustment_blocked_by_common_cause"] for item in resultados)
        )
        self.assertTrue(
            all(item["suspicion_effect"] == "maintain" for item in resultados)
        )
        self.assertTrue(
            all(
                item["adjusted_suspicion_score"] == item["statistical_score"]
                for item in resultados
            )
        )

    def test_contexto_causal_ignora_eventos_posteriores_del_mismo_lote(self):
        historico, entidades = construir_historial()
        mapa = self.filtro.construir_mapa(historico, entidades)
        timestamp = pd.Timestamp("2026-01-06T00:05:00Z")
        sospecha = {
            "entity_id": "sensor.temperature",
            "timestamp": timestamp.isoformat(),
            "sensor_kind": "continuous",
            "value": 8.0,
            "suspicion_score": 0.8,
            "is_suspicious": True,
            "passed_to_contextual_filter": True,
            "pattern": "abrupt_change",
        }
        disponibles = self._agregar_ventana(
            historico,
            timestamp,
            temperature=8.0,
            humidity=-16.0,
            climate_state="off",
            climate_valid=True,
        )
        futuro = self._agregar_ventana(
            historico.iloc[0:0],
            timestamp + pd.Timedelta(minutes=20),
            temperature=8.0,
            humidity=16.0,
            climate_state="on",
            climate_valid=True,
        )
        lote_con_futuro = pd.concat(
            [disponibles, futuro], ignore_index=True
        )

        sin_futuro = self.filtro.analizar_causal(
            [sospecha], disponibles, entidades, mapa
        )["results"][0]
        con_futuro = self.filtro.analizar_causal(
            [sospecha], lote_con_futuro, entidades, mapa
        )["results"][0]

        self.assertEqual(
            con_futuro["context_score"], sin_futuro["context_score"]
        )
        self.assertEqual(con_futuro["evidence"], sin_futuro["evidence"])
        self.assertTrue(con_futuro["causal_runtime"]["causal"])
        self.assertFalse(con_futuro["causal_runtime"]["uses_future_rows"])
        self.assertEqual(
            con_futuro["causal_runtime"]["available_through"],
            timestamp.isoformat(),
        )

    def test_politica_de_actualizacion_detecta_catalogo_y_antiguedad(self):
        eventos, entidades = construir_historial()
        mapa = self.filtro.construir_mapa(eventos, entidades)
        mapa["built_at"] = "2026-01-01T00:00:00Z"
        ampliadas = pd.concat(
            [
                entidades,
                pd.DataFrame(
                    [
                        {
                            "entity_id": "sensor.new_temperature",
                            "sensor_kind": "continuous",
                            "device_class": "temperature",
                            "domain": "sensor",
                            "unit_of_measurement": "C",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        reasons = self.filtro.razones_actualizacion(mapa, eventos, ampliadas)

        self.assertIn("entity_catalog_changed", reasons)
        self.assertIn("max_age_exceeded", reasons)

    def _agregar_ventana(
        self,
        historico: pd.DataFrame,
        timestamp: pd.Timestamp,
        *,
        temperature: float,
        humidity: float | None,
        climate_state: str,
        climate_valid: bool,
    ) -> pd.DataFrame:
        filas = [
            {
                "entity_id": "sensor.temperature",
                "timestamp": timestamp.isoformat(),
                "timestamp_dt": timestamp,
                "state": str(temperature),
                "value": temperature,
                "sensor_kind": "continuous",
                "is_valid_value": True,
            },
            {
                "entity_id": "binary_sensor.climate",
                "timestamp": timestamp.isoformat(),
                "timestamp_dt": timestamp,
                "state": climate_state,
                "value": 1.0 if climate_state == "on" else None,
                "sensor_kind": "binary",
                "is_valid_value": climate_valid,
            },
        ]
        if humidity is not None:
            filas.append(
                {
                    "entity_id": "sensor.humidity",
                    "timestamp": timestamp.isoformat(),
                    "timestamp_dt": timestamp,
                    "state": str(humidity),
                    "value": humidity,
                    "sensor_kind": "continuous",
                    "is_valid_value": True,
                }
            )
        return pd.concat([historico, pd.DataFrame(filas)], ignore_index=True)


if __name__ == "__main__":
    unittest.main()
