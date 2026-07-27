from pathlib import Path
import json
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.Cargar_Historial.Cargar_Historial import CargadorHistorial  # noqa: E402
from app.Filtro_Estadistico.Estados_Binarios import FiltroEstadoBinarioEstadistico, PerfilBinario  # noqa: E402
from app.Filtro_Estadistico.Filtro import FiltroEstadistico  # noqa: E402
from app.main import EstadoAplicacion  # noqa: E402


class CargaHistorialTest(unittest.TestCase):
    def test_carga_historial_real_y_clasifica_entidades(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "history.csv"
            pd.DataFrame(
                [
                    {
                        "entity_id": "sensor.bathroom_temperature",
                        "state": "18.5",
                        "last_changed": "2026-05-01T18:00:00.000Z",
                    },
                    {
                        "entity_id": "sensor.bathroom_humidity",
                        "state": "70.2",
                        "last_changed": "2026-05-01T18:01:00.000Z",
                    },
                    {
                        "entity_id": "binary_sensor.bathroom_motion",
                        "state": "on",
                        "last_changed": "2026-05-01T18:02:00.000Z",
                    },
                    {
                        "entity_id": "select.home_mode",
                        "state": "home",
                        "last_changed": "2026-05-01T18:03:00.000Z",
                    },
                ]
            ).to_csv(ruta, index=False)

            resultado = CargadorHistorial().cargar_real(ruta)
            tipos = {
                fila["entity_id"]: fila["sensor_kind"]
                for fila in resultado.entidades.to_dict("records")
            }

            self.assertEqual(resultado.metadatos["rows"], 4)
            self.assertEqual(tipos["sensor.bathroom_temperature"], "continuous")
            self.assertEqual(tipos["binary_sensor.bathroom_motion"], "binary")
            self.assertEqual(tipos["select.home_mode"], "multistate")

    def test_crea_historial_canonico_minimo(self):
        with tempfile.TemporaryDirectory() as tmp:
            directorio = Path(tmp) / "static_history"
            resultado = CargadorHistorial().crear_canonico_minimo(directorio, dias=60)

            self.assertTrue((directorio / "events.csv").exists())
            self.assertTrue((directorio / "entities.csv").exists())
            self.assertGreaterEqual(resultado.metadatos["rows"], 60 * 24)
            self.assertIn("sensor_kind", resultado.entidades.columns)

    def test_status_carga_historial_real_por_defecto_sin_analizar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            pd.DataFrame(
                [
                    {
                        "entity_id": "sensor.default_temperature",
                        "state": "18.5",
                        "last_changed": "2026-05-01T18:00:00.000Z",
                    }
                ]
            ).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            status = estado.status()

            self.assertTrue(status["exists"])
            self.assertEqual(status["source_type"], "real")
            self.assertTrue(status["normality_history"]["exists"])
            self.assertEqual(status["normality_history"]["mode"], "persisted")
            self.assertTrue(Path(status["normality_history"]["path"]).exists())
            with Path(status["normality_history"]["path"]).open("r", encoding="utf-8") as archivo:
                normalidad = json.load(archivo)
            entidad_normal = normalidad["entities"]["sensor.default_temperature"]
            self.assertEqual(entidad_normal["sensor_kind"], "continuous")
            self.assertIn("median", entidad_normal)
            self.assertIn("scaled_mad", entidad_normal)
            self.assertEqual(estado.resultados, [])

            primera_normalidad = status["normality_history"]["path"]
            nuevo = b"entity_id,state,last_changed\nsensor.default_temperature,20.5,2026-05-02T18:00:00.000Z\n"
            status_subido = estado.subir_real(nuevo)

            self.assertEqual(status_subido["rows"], 1)
            self.assertNotEqual(primera_normalidad, status_subido["normality_history"]["path"])
            self.assertTrue(estado.real_history_path.exists())

    def test_filtro_continuo_detecta_pico_y_usa_indicadores_de_tesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = []
            for indice in range(120):
                valor = 18.0 + (0.1 if indice % 2 else 0.0)
                if indice == 60:
                    valor = -6.9
                filas.append(
                    {
                        "entity_id": "sensor.bathroom_temperature",
                        "state": str(valor),
                        "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                    }
                )
            pd.DataFrame(filas).to_csv(ruta, index=False)

            historial = CargadorHistorial().cargar_real(ruta)
            resultado = FiltroEstadistico().analizar_historial(
                historial.eventos,
                historial.entidades,
                suspicion_threshold=0.6,
            )
            indicadores_validos = {
                "hampel",
                "robust_rate_change",
                "cusum_residual",
                "constant_persistence",
                "reading_gap",
                "dspot_extreme",
            }

            self.assertGreaterEqual(resultado["summary"]["suspicious"], 1)
            self.assertEqual(resultado["summary"]["temporal_activation_threshold"], 0.35)
            self.assertTrue(any(item["suspected_pattern"] == "isolated_spike" for item in resultado["results"]))
            self.assertTrue(any(item["value"] == -6.9 for item in resultado["results"]))
            self.assertTrue(any("dspot_extreme" in item["triggered_indicators"] for item in resultado["results"]))
            for item in resultado["results"]:
                self.assertIn("temporal_context", item)
                self.assertIn(item["time_window"], {"local", "hour_of_day", "month_repetition"})
                self.assertIn("evidence_score", item)
                self.assertGreaterEqual(item["evidence_score"], 0.0)
                self.assertLessEqual(item["evidence_score"], 1.0)
                self.assertGreaterEqual(item["suspicion_score"], 0.0)
                self.assertLessEqual(item["suspicion_score"], 1.0)
                self.assertTrue(set(item["triggered_indicators"]).issubset(indicadores_validos))
            self.assertTrue(any(item["raw_score"] > 1.0 for item in resultado["results"]))

    def test_repeticion_temporal_solo_agrega_contexto_continuo(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = []
            for dia in range(3):
                for minuto in range(120):
                    valor = 18.0
                    if 60 <= minuto <= 62:
                        valor = 30.0
                    filas.append(
                        {
                            "entity_id": "sensor.office_temperature",
                            "state": str(valor),
                            "last_changed": (inicio + pd.Timedelta(days=dia, minutes=minuto)).isoformat(),
                        }
                    )
            pd.DataFrame(filas).to_csv(ruta, index=False)

            historial = CargadorHistorial().cargar_real(ruta)
            resultado = FiltroEstadistico().analizar_historial(
                historial.eventos,
                historial.entidades,
                suspicion_threshold=0.5,
            )

            self.assertTrue(any(item["time_window"] == "month_repetition" for item in resultado["results"]))
            self.assertTrue(any(item["temporal_context"].get("repeated_pattern") for item in resultado["results"]))
            self.assertTrue(any(item["temporal_context"].get("temporal_consistency") == "alta" for item in resultado["results"]))
            self.assertTrue(all(item["suspicion_score"] >= 0.5 for item in resultado["results"]))

    def test_racha_extrema_mismo_dia_no_se_reduce_como_patron_mensual(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = []
            for minuto in range(240):
                valor = 18.0
                if 120 <= minuto <= 130:
                    valor = -7.0
                filas.append(
                    {
                        "entity_id": "sensor.bathroom_temperature",
                        "state": str(valor),
                        "last_changed": (inicio + pd.Timedelta(minutes=minuto)).isoformat(),
                    }
                )
            pd.DataFrame(filas).to_csv(ruta, index=False)

            historial = CargadorHistorial().cargar_real(ruta)
            resultado = FiltroEstadistico().analizar_historial(
                historial.eventos,
                historial.entidades,
                suspicion_threshold=0.6,
            )

            extremos = [item for item in resultado["results"] if item["value"] <= -6.9]
            self.assertTrue(extremos)
            self.assertTrue(all(not item["temporal_context"].get("repeated_pattern") for item in extremos))

    def test_filtro_binario_detecta_cambios_repetidos_en_ventana(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = []
            estado = "off"
            for indice in range(180):
                if 90 <= indice < 120:
                    estado = "on" if indice % 2 == 0 else "off"
                else:
                    estado = "on" if indice in {30, 31, 150, 151} else "off"
                filas.append(
                    {
                        "entity_id": "binary_sensor.living_motion",
                        "state": estado,
                        "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                    }
                )
            pd.DataFrame(filas).to_csv(ruta, index=False)

            historial = CargadorHistorial().cargar_real(ruta)
            resultado = FiltroEstadistico().analizar_historial(
                historial.eventos,
                historial.entidades,
                suspicion_threshold=0.6,
            )
            binarios = [item for item in resultado["results"] if item["sensor_kind"] == "binary"]
            indicadores_validos = {
                "activation_count",
                "state_change_count",
                "active_time_ratio",
                "time_since_last_activation",
                "report_gap",
                "inactive_duration",
                "time_since_last_change",
                "z_duration_score",
            }

            self.assertTrue(binarios)
            self.assertTrue(
                any(
                    {"activation_count", "state_change_count"}.intersection(item["triggered_indicators"])
                    for item in binarios
                )
            )
            for item in binarios:
                self.assertIn("evidence_score", item)
                self.assertGreaterEqual(item["evidence_score"], 0.0)
                self.assertLessEqual(item["evidence_score"], 1.0)
                self.assertGreaterEqual(item["suspicion_score"], 0.0)
                self.assertLessEqual(item["suspicion_score"], 1.0)
                self.assertIn("inactive_seconds", item["reading"])
                self.assertIn("time_since_last_change_seconds", item["reading"])
                self.assertTrue(set(item["triggered_indicators"]).issubset(indicadores_validos))

    def test_filtro_binario_report_gap_no_usa_p99_como_umbral_fuerte(self):
        filtro = FiltroEstadoBinarioEstadistico()
        perfil = PerfilBinario(
            window_seconds=60 * 60,
            min_report_gap_seconds=60 * 60,
            min_absence_gap_seconds=60 * 60,
            ratio_floor=0.05,
        )
        features = pd.DataFrame(
            {
                "activations": [0] * 31,
                "state_changes": [0] * 31,
                "active_ratio": [0.0] * 31,
                "active_seconds": [0.0] * 31,
                "inactive_seconds": [3600.0] * 31,
                "gap_seconds": [60.0] * 31,
                "time_since_last_change_seconds": [60.0] * 31,
                "report_gap_seconds": [60.0] * 30 + [7 * 24 * 60 * 60.0],
            }
        )

        base = filtro._construir_base_normalidad(features, perfil)

        self.assertEqual(base["report_gap_alert_seconds"], 60 * 60)

    def test_filtro_binario_usa_base_reciente_para_cambio_de_rutina(self):
        filtro = FiltroEstadoBinarioEstadistico()
        perfil = PerfilBinario(
            window_seconds=60 * 60,
            min_report_gap_seconds=60 * 60,
            min_absence_gap_seconds=60 * 60,
            ratio_floor=0.05,
        )
        inicio = pd.Timestamp("2026-05-01T00:00:00Z")
        timestamps = [inicio + pd.Timedelta(days=indice) for indice in range(30)]
        timestamps += [inicio + pd.Timedelta(days=30, hours=indice) for indice in range(24)]
        features = pd.DataFrame(
            {
                "timestamp_dt": timestamps,
                "activations": [0] * 54,
                "state_changes": [0] * 54,
                "active_ratio": [0.0] * 54,
                "active_seconds": [0.0] * 54,
                "inactive_seconds": [3600.0] * 54,
                "gap_seconds": [60.0] * 54,
                "time_since_last_change_seconds": [60.0] * 54,
                "report_gap_seconds": [60.0] * 30 + [3600.0] * 24,
            }
        )
        base = filtro._construir_base_normalidad(features, perfil)
        fila = features.iloc[-1].to_dict()

        _, componentes, evidencia = filtro._puntuar_doble_horizonte(fila, base)

        self.assertTrue(base["recent_horizon_usable"])
        self.assertEqual(componentes["s_M"], 0.0)
        self.assertEqual(evidencia, 0.0)

    def test_filtro_binario_omite_sensores_con_poco_historial(self):
        inicio = pd.Timestamp("2026-05-01T00:00:00Z")
        eventos = pd.DataFrame(
            [
                {
                    "entity_id": "binary_sensor.new_motion",
                    "sensor_kind": "binary",
                    "state": "on" if indice % 2 else "off",
                    "value": float(indice % 2),
                    "timestamp": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                    "timestamp_dt": inicio + pd.Timedelta(minutes=indice),
                }
                for indice in range(8)
            ]
        )
        entidades = pd.DataFrame(
            [
                {
                    "entity_id": "binary_sensor.new_motion",
                    "device_class": "motion",
                    "sensor_kind": "binary",
                }
            ]
        )

        resultados = FiltroEstadoBinarioEstadistico().analizar(eventos, entidades, suspicion_threshold=0.1)

        self.assertEqual(resultados, [])

    def test_analisis_real_reutiliza_cache_persistente(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = []
            for indice in range(80):
                filas.append(
                    {
                        "entity_id": "sensor.cache_temperature",
                        "state": str(18.0 if indice != 40 else -8.0),
                        "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                    }
                )
            pd.DataFrame(filas).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            estado.cargar_real()
            primero = estado.analizar_historial(0.6)
            estado.resultados = []
            segundo = estado.analizar_historial(0.6)

            self.assertTrue(estado.analysis_cache_path.exists())
            self.assertFalse(primero["summary"]["cache_hit"])
            self.assertTrue(segundo["summary"]["cache_hit"])
            self.assertEqual(primero["results"], segundo["results"])

    def test_tick_realtime_aplica_falla_y_reanaliza(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = [
                {
                    "entity_id": "sensor.realtime_temperature",
                    "state": str(20.0 + (0.05 if indice % 2 else 0.0)),
                    "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                }
                for indice in range(120)
            ]
            pd.DataFrame(filas).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            estado.cargar_real()
            payload = estado.tick(
                request=type(
                    "Request",
                    (),
                    {
                        "selected_entities": ["sensor.realtime_temperature"],
                        "tick_seconds": 60,
                        "faults": [
                            {
                                "entity_id": "sensor.realtime_temperature",
                                "fault_type": "random/spike",
                                "enabled": True,
                            }
                        ],
                        "suspicion_threshold": 0.6,
                    },
                )()
            )

            self.assertEqual(payload["events"], 1)
            self.assertTrue(payload["faults"])
            self.assertTrue(payload["fault_ticks"])
            self.assertEqual(payload["ground_truth_rows"], 1)
            self.assertFalse(payload["fault_ticks"][0]["is_reliable"])
            self.assertGreaterEqual(payload["result_count"], 1)
            self.assertGreaterEqual(payload["classification_result_count"], 1)
            self.assertTrue(payload["historical_classification_results"])
            self.assertTrue(
                any(
                    item["entity_id"] == "sensor.realtime_temperature"
                    and item["timestamp"] >= estado.realtime_current_timestamp[:19]
                    for item in payload["results"]
                )
            )
            evaluation = estado.evaluacion_clasificacion()
            self.assertTrue(evaluation["available"])
            self.assertEqual(
                evaluation["dataset"]["source"],
                "controlled_realtime_faults",
            )
            self.assertEqual(evaluation["funnel"]["all_labeled"], 1)
            self.assertIn(
                "fn_before_classifier",
                evaluation["end_to_end"]["by_class"]["sensor_fault"],
            )

    def test_tick_realtime_sin_falla_usa_normalidad_de_entidad(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = [
                {
                    "entity_id": "sensor.normalized_temperature",
                    "state": str(20.0 + (0.05 if indice % 2 else 0.0)),
                    "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                }
                for indice in range(119)
            ]
            filas.append(
                {
                    "entity_id": "sensor.normalized_temperature",
                    "state": "80.0",
                    "last_changed": (inicio + pd.Timedelta(minutes=119)).isoformat(),
                }
            )
            pd.DataFrame(filas).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            estado.cargar_real()
            payload = estado.tick(
                request=type(
                    "Request",
                    (),
                    {
                        "selected_entities": ["sensor.normalized_temperature"],
                        "tick_seconds": 60,
                        "faults": [],
                        "suspicion_threshold": 0.6,
                    },
                )()
            )

            valor_tick = float(estado.realtime_events.iloc[-1]["value"])
            self.assertLess(valor_tick, 21.0)
            self.assertEqual(payload["fault_ticks"], [])

    def test_ticks_binarios_sin_falla_respetan_ratio_temporal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = [
                ("off", 0),
                ("on", 10),
                ("off", 11),
                ("on", 20),
                ("off", 21),
            ]
            pd.DataFrame(
                [
                    {
                        "entity_id": "binary_sensor.normalized_motion",
                        "state": estado,
                        "last_changed": (inicio + pd.Timedelta(minutes=minuto)).isoformat(),
                    }
                    for estado, minuto in filas
                ]
            ).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )
            request = type(
                "Request",
                (),
                {
                    "selected_entities": ["binary_sensor.normalized_motion"],
                    "tick_seconds": 60,
                    "faults": [],
                    "suspicion_threshold": 0.6,
                },
            )()

            estado.cargar_real()
            for _ in range(20):
                estado._insertar_tick_realtime(request)

            activos = int((estado.realtime_events["value"] >= 0.5).sum())
            self.assertLessEqual(activos, 2)

    def test_manual_tick_inserta_muestra_continua(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = [
                {
                    "entity_id": "sensor.manual_temperature",
                    "state": "20.0",
                    "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                }
                for indice in range(80)
            ]
            pd.DataFrame(filas).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            estado.cargar_real()
            payload = estado.manual_tick(
                request=type(
                    "Request",
                    (),
                    {
                        "selected_entities": ["sensor.manual_temperature"],
                        "sensor_states": {"sensor.manual_temperature": 24.5},
                        "tick_seconds": 60,
                        "suspicion_threshold": 0.6,
                    },
                )()
            )

            self.assertEqual(payload["summary"]["manual_inserted"], 1)
            self.assertEqual(float(estado.realtime_events.iloc[-1]["value"]), 24.5)

    def test_manual_tick_inserta_muestra_binaria(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = [
                {
                    "entity_id": "binary_sensor.manual_motion",
                    "state": "off",
                    "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                }
                for indice in range(80)
            ]
            pd.DataFrame(filas).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            estado.cargar_real()
            payload = estado.manual_tick(
                request=type(
                    "Request",
                    (),
                    {
                        "selected_entities": ["binary_sensor.manual_motion"],
                        "sensor_states": {"binary_sensor.manual_motion": True},
                        "tick_seconds": 60,
                        "suspicion_threshold": 0.6,
                    },
                )()
            )

            self.assertEqual(payload["summary"]["manual_inserted"], 1)
            self.assertEqual(estado.realtime_events.iloc[-1]["state"], "on")
            self.assertEqual(float(estado.realtime_events.iloc[-1]["value"]), 1.0)

    def test_manual_tick_respeta_cadencia_historica(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = [
                {
                    "entity_id": "sensor.slow_temperature",
                    "state": "20.0",
                    "last_changed": (inicio + pd.Timedelta(hours=indice)).isoformat(),
                }
                for indice in range(24)
            ]
            pd.DataFrame(filas).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            estado.cargar_real()
            payload = estado.manual_tick(
                request=type(
                    "Request",
                    (),
                    {
                        "selected_entities": ["sensor.slow_temperature"],
                        "sensor_states": {"sensor.slow_temperature": 21.0},
                        "tick_seconds": 60,
                        "suspicion_threshold": 0.6,
                    },
                )()
            )

            self.assertEqual(payload["summary"]["manual_inserted"], 0)
            self.assertEqual(payload["summary"]["manual_skipped_by_cadence"], 1)
            self.assertTrue(payload["summary"]["skipped_analysis"])
            self.assertTrue(estado.realtime_events.empty)

    def test_manual_tick_valor_continuo_fuera_de_normalidad_genera_sospecha(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = [
                {
                    "entity_id": "sensor.manual_alarm_temperature",
                    "state": str(20.0 + (0.05 if indice % 2 else 0.0)),
                    "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                }
                for indice in range(120)
            ]
            pd.DataFrame(filas).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            estado.cargar_real()
            payload = estado.manual_tick(
                request=type(
                    "Request",
                    (),
                    {
                        "selected_entities": ["sensor.manual_alarm_temperature"],
                        "sensor_states": {"sensor.manual_alarm_temperature": 80.0},
                        "tick_seconds": 60,
                        "suspicion_threshold": 0.6,
                    },
                )()
            )

            self.assertEqual(payload["summary"]["manual_inserted"], 1)
            self.assertTrue(payload["summary"]["incremental"])
            self.assertTrue(
                any(
                    item["entity_id"] == "sensor.manual_alarm_temperature"
                    and item["is_suspicious"]
                    for item in payload["results"]
                )
            )

    def test_iteraciones_realtime_usan_unidad_de_trabajo(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = [
                {
                    "entity_id": "sensor.batch_temperature",
                    "state": str(21.0 + (0.05 if indice % 2 else 0.0)),
                    "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                }
                for indice in range(120)
            ]
            pd.DataFrame(filas).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            estado.cargar_real()
            payload = estado.iterations(
                request=type(
                    "Request",
                    (),
                    {
                        "selected_entities": ["sensor.batch_temperature"],
                        "tick_seconds": 60,
                        "iteration_count": 5,
                        "faults": [
                            {
                                "entity_id": "sensor.batch_temperature",
                                "fault_type": "drift",
                                "enabled": True,
                            }
                        ],
                        "suspicion_threshold": 0.6,
                    },
                )()
            )

            self.assertEqual(payload["events"], 5)
            self.assertEqual(payload["summary"]["iterations_done"], 5)
            self.assertTrue(payload["faults"])
            self.assertGreaterEqual(payload["result_count"], 1)


if __name__ == "__main__":
    unittest.main()
