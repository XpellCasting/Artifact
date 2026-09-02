from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.Cargar_Historial.Cargar_Historial import CargadorHistorial  # noqa: E402
from app.Filtro_Estadistico.Estados_Binarios import FiltroEstadoBinarioEstadistico, PerfilBinario  # noqa: E402
from app.Filtro_Estadistico.Filtro import FiltroEstadistico  # noqa: E402
from app.main import EstadoAplicacion  # noqa: E402


class CargaHistorialTest(unittest.TestCase):
    def test_carga_asincrona_reporta_progreso_y_resultado(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            pd.DataFrame(
                [
                    {
                        "entity_id": "sensor.progress_temperature",
                        "state": str(18 + index * 0.1),
                        "last_changed": f"2026-05-01T18:{index:02d}:00.000Z",
                    }
                    for index in range(10)
                ]
            ).to_csv(ruta, index=False)
            filtro = FiltroEstadistico()
            filtro.continuos.conf_path = base / "conf.txt"
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=filtro,
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            job = estado.iniciar_carga_real()
            limite = time.monotonic() + 10
            while job["status"] in {"queued", "running"} and time.monotonic() < limite:
                time.sleep(0.01)
                job = estado.estado_carga_real(job["job_id"])

            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["phase"], "ready")
            self.assertEqual(job["progress"], 100.0)
            self.assertEqual(job["history"]["rows"], 10)
            self.assertEqual(job["eta_seconds"], 0.0)

    def test_status_sin_autoload_informa_archivo_disponible(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            ruta.write_text(
                "entity_id,state,last_changed\n"
                "sensor.available_temperature,20,2026-05-01T18:00:00.000Z\n",
                encoding="utf-8",
            )
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )

            status = estado.status(autoload=False)

            self.assertFalse(status["exists"])
            self.assertTrue(status["real_history_available"])
            self.assertGreater(status["real_history_size_bytes"], 0)
            self.assertIsNone(estado.historial)

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

    def test_timeline_historial_usa_solo_las_filas_cargadas(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            pd.DataFrame(
                [
                    {
                        "entity_id": "sensor.real_temperature",
                        "state": "18.5",
                        "last_changed": "2026-05-01T18:00:00.000Z",
                    },
                    {
                        "entity_id": "sensor.real_temperature",
                        "state": "19.0",
                        "last_changed": "2026-05-01T19:00:00.000Z",
                    },
                ]
            ).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )
            estado.historial = estado.cargador.cargar_real(ruta)

            payload = estado.timeline_historial(start=None, end=None, max_points=100)
            timestamps = [
                point["timestamp"]
                for entity in payload["entities"]
                for point in entity["points"]
            ]

            self.assertEqual(len(timestamps), 2)
            self.assertNotIn("faults", payload)

    def test_timeline_multistate_codifica_solo_estados_observados(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ruta = base / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            pd.DataFrame(
                [
                    {
                        "entity_id": "sensor.review_status",
                        "state": "ALERT" if indice % 3 == 0 else "DETECTION",
                        "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                    }
                    for indice in range(30)
                ]
            ).to_csv(ruta, index=False)
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=ruta,
                analysis_cache_path=base / "cache" / "analysis.json",
            )
            estado.historial = estado.cargador.cargar_real(ruta)

            timeline = estado.timeline_historial(
                start=None,
                end=None,
                max_points=100,
            )["entities"][0]

            self.assertEqual(timeline["sensor_kind"], "multistate")
            self.assertEqual(timeline["chart_kind"], "categorical_state")
            self.assertEqual(set(timeline["categories"]), {"ALERT", "DETECTION"})
            self.assertEqual(timeline["analysis_status"], "eligible")
            self.assertTrue(
                all(
                    point["value"] in timeline["categories"].values()
                    for point in timeline["points"]
                )
            )

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
                self.assertLess(item["suspicion_score"], 1.0)
                self.assertEqual(item["evidence_score"], item["suspicion_score"])
                self.assertEqual(
                    item["score_version"],
                    "statistical-continuous-score-v2.0.0",
                )
                self.assertEqual(
                    item["score_kind"],
                    "statistical_evidence_strength_not_probability",
                )
                self.assertLess(
                    item["temporal_context"]["hour_of_day_score"],
                    1.0,
                )
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

    def test_filtro_multistate_analiza_transiciones_y_excluye_alta_cardinalidad(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "history.csv"
            inicio = pd.Timestamp("2026-05-01T00:00:00Z")
            filas = []
            for indice in range(480):
                estado = "DETECTION"
                if indice in {60, 61, 180, 181, 360, 361}:
                    estado = "ALERT"
                if 240 <= indice < 270:
                    estado = "ALERT" if indice % 2 else "DETECTION"
                filas.append(
                    {
                        "entity_id": "sensor.sittingroom_review_status",
                        "state": estado,
                        "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                    }
                )
            filas.extend(
                {
                    "entity_id": "sensor.weatherstation_last_rain_2",
                    "state": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                    "last_changed": (inicio + pd.Timedelta(minutes=indice)).isoformat(),
                }
                for indice in range(60)
            )
            pd.DataFrame(filas).to_csv(ruta, index=False)

            historial = CargadorHistorial().cargar_real(ruta)
            resultado = FiltroEstadistico().analizar_historial(
                historial.eventos,
                historial.entidades,
                suspicion_threshold=0.6,
            )
            multistate = [
                item
                for item in resultado["results"]
                if item["sensor_kind"] == "multistate"
            ]
            elegibilidad = resultado["summary"]["multistate_eligibility"]

            self.assertTrue(multistate)
            self.assertTrue(
                any(item["suspected_pattern"] == "transition_burst" for item in multistate)
            )
            self.assertTrue(
                all(item["score_version"] == "statistical-multistate-score-v1.0.0" for item in multistate)
            )
            self.assertEqual(elegibilidad["eligible_entities"], 1)
            self.assertEqual(elegibilidad["by_status"]["high_cardinality"], 1)
            self.assertEqual(
                resultado["summary"]["implemented_kinds"],
                ["continuous", "binary", "multistate"],
            )
            self.assertEqual(resultado["summary"]["pending_kinds"], [])

    def test_reproduccion_validacion_es_causal_y_mide_retraso_por_lote(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            desarrollo_path = base / "desarrollo.csv"
            validacion_path = base / "validacion.csv"
            inicio_desarrollo = pd.Timestamp("2026-02-01T00:00:00Z")
            pd.DataFrame(
                [
                    {
                        "entity_id": "sensor.replay_temperature",
                        "state": str(20.0 + (indice % 2) * 0.1),
                        "last_changed": (
                            inicio_desarrollo + pd.Timedelta(minutes=indice)
                        ).isoformat(),
                    }
                    for indice in range(120)
                ]
            ).to_csv(desarrollo_path, index=False)
            inicio_validacion = pd.Timestamp("2026-05-21T00:00:00Z")
            pd.DataFrame(
                [
                    {
                        "entity_id": "sensor.replay_temperature",
                        "state": str(
                            50.0 if indice == 10 else 20.0 + (indice % 2) * 0.1
                        ),
                        "last_changed": (
                            inicio_validacion + pd.Timedelta(minutes=indice)
                        ).isoformat(),
                    }
                    for indice in range(30)
                ]
            ).to_csv(validacion_path, index=False)
            filtro = FiltroEstadistico()
            replay_conf_path = base / "replay-conf.txt"
            filtro.continuos.conf_path = replay_conf_path
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=filtro,
                static_history_dir=base / "static_history",
                real_history_path=desarrollo_path,
                analysis_cache_path=base / "cache" / "analysis.json",
                validation_history_path=base / "uploaded_validation.csv",
            )
            estado.cargar_real()
            estado_preparado_en = estado.future_evaluation_prepared_at
            estado._preparar_continuidad_historica = lambda: self.fail(
                "Subir Evaluacion no debe volver a preparar el estado historico"
            )

            preparado = estado.subir_validacion(validacion_path.read_bytes())
            self.assertIsNone(estado.validation_baseline)
            self.assertIsNone(estado.validation_context_map)
            self.assertIsNone(estado.validation_context_events)
            self.assertEqual(preparado["detector_state"], "pending_until_advance")
            antes_del_pico = estado.avanzar_reproduccion_validacion(10, 0.6)
            self.assertIsNotNone(estado.validation_baseline)
            self.assertEqual(antes_del_pico["detector_state"], "active")
            con_el_pico = estado.avanzar_reproduccion_validacion(3, 0.6)

            self.assertTrue(preparado["loaded"])
            self.assertEqual(preparado["evaluation_upload_mode"], "load_only")
            self.assertEqual(preparado["state_source"], "historical_template")
            self.assertEqual(
                preparado["historical_state_prepared_at"], estado_preparado_en
            )
            self.assertEqual(estado.future_evaluation_prepared_at, estado_preparado_en)
            self.assertFalse(replay_conf_path.exists())
            self.assertEqual(preparado["baseline_summary"]["continuous"], 1)
            self.assertFalse(
                any(item.get("value") == 50.0 for item in antes_del_pico["results"])
            )
            pico = next(
                item for item in con_el_pico["new_results"] if item.get("value") == 50.0
            )
            self.assertTrue(pico["replay"]["causal"])
            self.assertEqual(pico["replay"]["detection_delay_seconds"], 120.0)
            self.assertEqual(
                len(con_el_pico["new_contextual_results"]),
                len(con_el_pico["new_results"]),
            )
            self.assertEqual(
                len(con_el_pico["new_classification_results"]),
                len(con_el_pico["new_results"]),
            )
            diagnostico = next(
                item
                for item in con_el_pico["new_classification_results"]
                if item["entity_id"] == "sensor.replay_temperature"
            )
            self.assertEqual(diagnostico["source"], "validation_replay")
            self.assertTrue(diagnostico["replay"]["causal"])
            self.assertEqual(
                diagnostico["pipeline"]["contextual"]["outcome"],
                "insufficient_coverage",
            )
            self.assertEqual(
                con_el_pico["pipeline_summary"]["statistical"],
                con_el_pico["pipeline_summary"]["classifier"],
            )
            self.assertEqual(
                len(estado.contexto_actual()["results"]),
                con_el_pico["contextual_count"],
            )
            self.assertEqual(
                len(estado.clasificacion_actual()["results"]),
                con_el_pico["classification_count"],
            )
            self.assertTrue(con_el_pico["last_batch"]["keeps_up_with_event_time"])
            self.assertEqual(con_el_pico["processed_rows"], 13)

            reiniciado = estado.reiniciar_reproduccion_validacion()
            self.assertEqual(reiniciado["processed_rows"], 0)
            self.assertEqual(reiniciado["detector_state"], "pending_until_advance")
            self.assertIsNone(estado.validation_baseline)
            self.assertEqual(reiniciado["detection_count"], 0)
            self.assertEqual(reiniciado["contextual_count"], 0)
            self.assertEqual(reiniciado["classification_count"], 0)

    def test_reproduccion_validacion_multistate_detecta_estado_no_visto(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            desarrollo_path = base / "desarrollo.csv"
            inicio = pd.Timestamp("2026-02-01T00:00:00Z")
            pd.DataFrame(
                [
                    {
                        "entity_id": "sensor.review_status",
                        "state": "ALERT" if indice % 2 else "DETECTION",
                        "last_changed": (
                            inicio + pd.Timedelta(minutes=indice)
                        ).isoformat(),
                    }
                    for indice in range(60)
                ]
            ).to_csv(desarrollo_path, index=False)
            validacion = pd.DataFrame(
                [
                    {
                        "entity_id": "sensor.review_status",
                        "state": "ERROR",
                        "last_changed": "2026-05-21T00:00:00Z",
                    }
                ]
            )
            estado = EstadoAplicacion(
                cargador=CargadorHistorial(),
                filtro_estadistico=FiltroEstadistico(),
                static_history_dir=base / "static_history",
                real_history_path=desarrollo_path,
                analysis_cache_path=base / "cache" / "analysis.json",
                validation_history_path=base / "uploaded_validation.csv",
            )
            estado.cargar_real()
            estado.subir_validacion(validacion.to_csv(index=False).encode("utf-8"))

            avance = estado.avanzar_reproduccion_validacion(1, 0.6)
            resultado = avance["new_results"][0]

            self.assertEqual(resultado["sensor_kind"], "multistate")
            self.assertEqual(resultado["observed_state"], "ERROR")
            self.assertIn("state_probability", resultado["triggered_indicators"])
            self.assertTrue(resultado["replay"]["causal"])

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
            with estado.analysis_cache_path.open("r", encoding="utf-8") as archivo:
                cache = json.load(archivo)
            self.assertEqual(cache["identity"]["cache_version"], 17)
            self.assertEqual(
                cache["identity"]["continuous_score_specification"]["version"],
                "statistical-continuous-score-v2.0.0",
            )
            self.assertTrue(cache["identity"]["continuous_conf_sha256"])
            self.assertEqual(
                cache["identity"]["multistate_score_specification"]["version"],
                "statistical-multistate-score-v1.0.0",
            )
            self.assertEqual(
                cache["identity"]["contextual_filter_version"],
                "contextual-spatiotemporal-v1",
            )
            self.assertTrue(
                cache["identity"]["contextual_implementation_sha256"]
            )
            self.assertTrue(cache["identity"]["contextual_map_sha256"])
            self.assertFalse(cache["identity"]["contextual_map_expired"])
            config_contextual_original = estado.filtro_contextual.config
            estado.filtro_contextual.config = replace(
                config_contextual_original,
                decision_threshold=0.36,
            )
            self.assertIsNone(estado._leer_cache_analisis_real(0.6))
            estado.filtro_contextual.config = config_contextual_original

            mapa_path = estado._ruta_cache_contextual()
            with mapa_path.open("r", encoding="utf-8") as archivo:
                mapa = json.load(archivo)
            mapa["built_at"] = "2000-01-01T00:00:00+00:00"
            with mapa_path.open("w", encoding="utf-8") as archivo:
                json.dump(mapa, archivo)
            identidad_vencida = estado._identidad_historial_real(0.6)
            self.assertTrue(identidad_vencida["contextual_map_expired"])
            self.assertIsNone(estado._leer_cache_analisis_real(0.6))


if __name__ == "__main__":
    unittest.main()
