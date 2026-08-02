import math
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.Filtro_Estadistico.Continuos import (  # noqa: E402
    FiltroContinuoEstadistico,
    PerfilContinuo,
)


class PuntajeEstadisticoContinuoV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.filtro = FiltroContinuoEstadistico()

    def test_zona_muerta_y_entradas_no_finitas_producen_cero(self):
        for value in [float("nan"), float("inf"), float("-inf"), 2.49, 2.5]:
            with self.subTest(value=value):
                self.assertEqual(self.filtro._score_desde_z(value), 0.0)

    def test_curva_respeta_ancla_y_valores_de_regresion(self):
        esperados = {
            3.1: 0.6,
            3.5: 0.6797495640482295,
            6.5: 0.8150853101805504,
            27.0: 0.8898662837677342,
            150.0: 0.9226071492878178,
            4231.0: 0.9504328472179218,
        }
        self.assertAlmostEqual(
            self.filtro.lambda_log,
            (2.0 / 3.0) * math.log(2.0),
            places=15,
        )
        for value, esperado in esperados.items():
            with self.subTest(value=value):
                self.assertAlmostEqual(
                    self.filtro._score_desde_z(value),
                    esperado,
                    places=12,
                )

    def test_curva_es_estrictamente_creciente_y_no_satura(self):
        values = [2.500001, 2.6, 3.1, 3.5, 6.5, 27.0, 150.0, 4231.0, 1e9]
        scores = [self.filtro._score_desde_z(value) for value in values]

        self.assertTrue(all(0.0 < score < 1.0 for score in scores))
        self.assertTrue(
            all(left < right for left, right in zip(scores, scores[1:]))
        )

    def test_ratio_unitario_activa_con_evidencia_cero_coma_seis(self):
        self.assertAlmostEqual(
            self.filtro._score_desde_ratio(1.0),
            0.6,
            places=12,
        )
        self.assertLess(self.filtro._score_desde_ratio(1e300), 1.0)

    def test_limite_operacional_usa_curva_suave(self):
        perfil = PerfilContinuo(
            value_floor=0.1,
            rate_floor_per_minute=0.1,
            const_epsilon_floor=0.1,
            min_constant_seconds=60.0,
            min_gap_seconds=60.0,
            physical_low=0.0,
            physical_high=100.0,
            extreme_low=10.0,
            extreme_high=90.0,
        )

        raw, score = self.filtro._score_fisico(95.0, perfil)
        self.assertGreater(raw, self.filtro.ancla_activacion_z)
        self.assertGreater(score, 0.6)
        self.assertLess(score, 1.0)
        self.assertEqual(self.filtro._score_fisico(50.0, perfil), (0.0, 0.0))

    def test_cusum_incremental_continua_el_mismo_estado_del_batch(self):
        perfil = PerfilContinuo(
            value_floor=0.1,
            rate_floor_per_minute=0.1,
            const_epsilon_floor=0.1,
            min_constant_seconds=60.0,
            min_gap_seconds=60.0,
        )
        base = {
            "hourly_baseline": {},
            "normal_lower": -1.0,
            "normal_upper": 1.0,
            "residual_scale": 1.0,
            "cusum_k": 0.5,
            "cusum_theta": 8.0,
            "cusum_reset_gap_seconds": 3600.0,
            "perfil": perfil,
        }
        start = pd.Timestamp("2026-05-01T00:00:00Z")
        history = pd.DataFrame(
            [
                {
                    "timestamp": (start + pd.Timedelta(minutes=index)).isoformat(),
                    "timestamp_dt": start + pd.Timedelta(minutes=index),
                    "value": 0.0,
                }
                for index in range(20)
            ]
        )
        new_row = {
            "timestamp": (start + pd.Timedelta(minutes=20)).isoformat(),
            "timestamp_dt": start + pd.Timedelta(minutes=20),
            "value": 10.0,
        }

        previous_state = self.filtro.construir_estado_cusum(history, base)
        incremental_state, raw, score, _ = self.filtro._actualizar_cusum(
            valor=10.0,
            gap_seconds=60.0,
            base=base,
            dentro_banda=False,
            score_limite_operacional=0.0,
            estado=previous_state,
        )
        batch_state = self.filtro.construir_estado_cusum(
            pd.concat([history, pd.DataFrame([new_row])], ignore_index=True),
            base,
        )

        self.assertAlmostEqual(raw, float(batch_state["last_raw"]), places=12)
        self.assertAlmostEqual(
            score,
            float(batch_state["last_score"]),
            places=12,
        )
        self.assertAlmostEqual(
            float(incremental_state["ewma"]),
            float(batch_state["ewma"]),
            places=12,
        )
        self.assertGreater(raw, base["cusum_theta"])
        self.assertGreater(score, 0.6)

    def test_score_final_es_evidencia_y_brecha_aislada_no_avanza(self):
        componentes = {
            "s_H": 0.0,
            "s_R": 0.0,
            "s_C": 0.0,
            "s_D_const": 0.0,
            "s_M": 0.9,
            "s_DSPOT": 0.0,
        }
        indicadores = self.filtro._indicadores_activados(componentes, 0.35)
        evidencia = self.filtro._evidencia_continua(componentes, indicadores)

        self.assertEqual(indicadores, ["reading_gap"])
        self.assertEqual(evidencia, 0.5)
        self.assertEqual(self.filtro._score_sospecha(evidencia), evidencia)
        self.assertEqual(self.filtro._umbral_evidencia(0.6), 0.6)

    def test_especificacion_declara_version_y_no_usa_simulacion(self):
        specification = self.filtro.especificacion_puntaje()

        self.assertEqual(
            specification["version"],
            "statistical-continuous-score-v2.0.0",
        )
        self.assertEqual(
            specification["kind"],
            "statistical_evidence_strength_not_probability",
        )
        self.assertFalse(specification["uses_simulated_dataset"])


if __name__ == "__main__":
    unittest.main()
