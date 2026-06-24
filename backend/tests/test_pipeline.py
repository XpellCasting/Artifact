from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.Cargar_Historial.Cargar_Historial import CargadorHistorial  # noqa: E402
from app.Filtro_Estadistico.Filtro import FiltroEstadistico  # noqa: E402


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
            }

            self.assertGreaterEqual(resultado["summary"]["suspicious"], 1)
            self.assertTrue(any(item["suspected_pattern"] == "isolated_spike" for item in resultado["results"]))
            for item in resultado["results"]:
                self.assertIn("evidence_score", item)
                self.assertGreaterEqual(item["evidence_score"], 0.0)
                self.assertLessEqual(item["evidence_score"], 1.0)
                self.assertGreaterEqual(item["suspicion_score"], 0.0)
                self.assertLessEqual(item["suspicion_score"], 1.0)
                self.assertTrue(set(item["triggered_indicators"]).issubset(indicadores_validos))
            self.assertTrue(any(item["raw_score"] > 1.0 for item in resultado["results"]))

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
                self.assertTrue(set(item["triggered_indicators"]).issubset(indicadores_validos))


if __name__ == "__main__":
    unittest.main()
