from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.Filtro_Clasificacion import (  # noqa: E402
    ClasificadorDiagnostico,
    EvaluadorExtremoAExtremo,
)


def statistical_result(
    timestamp: str = "2026-05-10T10:00:00Z",
    indicators: list[str] | None = None,
    components: dict[str, float] | None = None,
    ground_truth_label: str | None = None,
) -> dict:
    return {
        "entity_id": "sensor.office_temperature",
        "timestamp": timestamp,
        "window_start": timestamp,
        "window_end": timestamp,
        "sensor_kind": "continuous",
        "suspicion_score": 0.82,
        "evidence_score": 0.82,
        "passed_to_contextual_filter": True,
        "suspected_pattern": "persistent_shift",
        "triggered_indicators": indicators or ["cusum_residual"],
        "reading": {"components": components or {"s_C": 0.82}},
        "ground_truth_label": ground_truth_label,
    }


def contextual_result(
    timestamp: str = "2026-05-10T10:00:00Z",
    outcome: str = "unexplained",
    adjusted: float = 0.82,
    support: float = 0.0,
    contradiction: float = 0.0,
    coverage: float = 0.90,
    common_cause: bool = False,
    independent_evidence: bool = False,
    stale: bool = False,
) -> dict:
    return {
        "event_id": f"sensor.office_temperature@{timestamp}",
        "entity_id": "sensor.office_temperature",
        "timestamp": timestamp,
        "sensor_kind": "continuous",
        "statistical_score": 0.82,
        "adjusted_suspicion_score": adjusted,
        "context_score": support - contradiction,
        "coherence_score": (support - contradiction + 1.0) / 2.0,
        "support_score": support,
        "contradiction_score": contradiction,
        "context_outcome": outcome,
        "coverage": {
            "eligible_neighbors": 3,
            "observed_neighbors": 3,
            "ratio": coverage,
            "status": "sufficient" if coverage >= 0.50 else "insufficient",
        },
        "common_cause": {
            "candidate": common_cause,
            "status": (
                "unidentifiable_without_independent_evidence"
                if common_cause and not independent_evidence
                else "not_indicated"
            ),
            "independent_evidence_available": independent_evidence,
        },
        "map_quality": {
            "map_version": "contextual-spatiotemporal-v1",
            "stale": stale,
            "stale_reasons": ["age"] if stale else [],
        },
    }


class ClasificadorDiagnosticoTest(unittest.TestCase):
    def setUp(self):
        self.classifier = ClasificadorDiagnostico()

    def classify(self, stat=None, context=None):
        payload = self.classifier.analizar(
            [stat or statistical_result()],
            [context or contextual_result()],
        )
        return payload["results"][0]

    def test_especificacion_es_definitiva_reproducible_y_sin_ground_truth(self):
        specification = self.classifier.especificacion()

        self.assertEqual(
            specification["model_version"],
            "diagnostic-rule-list-v1.0.0",
        )
        self.assertTrue(specification["definitive_for_prototype"])
        self.assertEqual(
            specification["selection_status"],
            "pre_registered_pending_real_validation",
        )
        self.assertFalse(
            specification["validation_protocol"]["uses_simulated_dataset"]
        )
        self.assertFalse(
            specification["ground_truth_feature_policy"]["allowed_as_input"]
        )
        self.assertEqual(
            [row["precedence"] for row in specification["decision_table"]],
            list(range(9)),
        )

    def test_respaldo_contextual_fuerte_es_anomalia_ambiental(self):
        result = self.classify(
            context=contextual_result(
                outcome="supported",
                support=0.80,
                adjusted=0.59,
            )
        )

        self.assertEqual(result["diagnostic_class"], "environmental_anomaly")
        self.assertEqual(
            result["decision_rule_id"],
            "R04_STRONG_CONTEXTUAL_SUPPORT",
        )
        self.assertIsNone(result["subtype"])

    def test_respaldo_debil_que_reduce_sospecha_puede_rechazar_candidato(self):
        result = self.classify(
            context=contextual_result(
                outcome="supported",
                support=0.40,
                adjusted=0.55,
            )
        )

        self.assertEqual(result["diagnostic_class"], "normal")
        self.assertEqual(
            result["decision_rule_id"],
            "R05_CONTEXT_REJECTS_SUSPICION",
        )
        self.assertEqual(result["suggested_action"], "accept_candidate_as_normal")

    def test_contradiccion_es_posible_falla_y_sugiere_patron_no_causa(self):
        result = self.classify(
            context=contextual_result(
                outcome="contradicted",
                contradiction=0.75,
            )
        )

        self.assertEqual(result["diagnostic_class"], "possible_sensor_fault")
        self.assertEqual(
            result["decision_rule_id"],
            "R06_CONTEXTUAL_CONTRADICTION",
        )
        self.assertEqual(result["subtype"], "persistent_shift")
        self.assertIn(
            "fault_subtype_is_suggested_not_confirmed_root_cause",
            result["limitations"],
        )
        self.assertEqual(
            result["confidence_kind"],
            "evidence_strength_not_calibrated_probability",
        )

    def test_no_explicado_solo_es_falla_con_patron_especifico_fuerte(self):
        strong = self.classify(
            stat=statistical_result(components={"s_H": 0.91}),
            context=contextual_result(outcome="unexplained"),
        )
        weak = self.classify(
            stat=statistical_result(components={"s_H": 0.65}),
            context=contextual_result(outcome="unexplained"),
        )

        self.assertEqual(strong["diagnostic_class"], "possible_sensor_fault")
        self.assertEqual(strong["subtype"], "persistent_shift")
        self.assertEqual(weak["diagnostic_class"], "indeterminate")
        self.assertIsNone(weak["confidence"])

    def test_causa_comun_no_identificable_tiene_precedencia(self):
        result = self.classify(
            context=contextual_result(
                outcome="supported",
                support=0.95,
                common_cause=True,
                independent_evidence=False,
            )
        )

        self.assertEqual(result["diagnostic_class"], "indeterminate")
        self.assertEqual(
            result["decision_rule_id"],
            "R01_AMBIGUOUS_COMMON_CAUSE",
        )
        self.assertIn("common_cause_unidentifiable", result["limitations"])

    def test_mapa_obsoleto_y_cobertura_insuficiente_producen_abstencion(self):
        stale = self.classify(
            context=contextual_result(
                outcome="contradicted",
                contradiction=0.90,
                stale=True,
            )
        )
        uncovered = self.classify(
            context=contextual_result(
                outcome="contradicted",
                contradiction=0.90,
                coverage=0.30,
            )
        )

        self.assertEqual(stale["decision_rule_id"], "R02_STALE_CONTEXT_MAP")
        self.assertEqual(
            uncovered["decision_rule_id"],
            "R03_INSUFFICIENT_CONTEXT_COVERAGE",
        )
        self.assertTrue(stale["abstained"])
        self.assertTrue(uncovered["abstained"])

    def test_ground_truth_del_input_no_cambia_la_decision_ni_se_propaga(self):
        without_label = self.classify(
            stat=statistical_result(ground_truth_label=None),
            context=contextual_result(
                outcome="contradicted",
                contradiction=0.80,
            ),
        )
        malicious_label = self.classify(
            stat=statistical_result(ground_truth_label="normal"),
            context=contextual_result(
                outcome="contradicted",
                contradiction=0.80,
            ),
        )

        self.assertEqual(
            without_label["diagnostic_class"],
            malicious_label["diagnostic_class"],
        )
        self.assertEqual(
            without_label["diagnosis_id"],
            malicious_label["diagnosis_id"],
        )
        self.assertNotIn("ground_truth_label", malicious_label)
        self.assertFalse(
            malicious_label["input_provenance"]["ground_truth_used_as_feature"]
        )


class EvaluacionExtremoAExtremoTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = EvaluadorExtremoAExtremo()

    def test_falla_descartada_por_filtro_estadistico_es_fn_e2e(self):
        ground_truth = pd.DataFrame(
            [
                {
                    "timestamp": "2026-05-10T10:00:00Z",
                    "entity_id": "sensor.office_temperature",
                    "label": "sensor_fault",
                    "root_cause": "bias-a",
                },
                {
                    "timestamp": "2026-05-10T10:01:00Z",
                    "entity_id": "sensor.office_temperature",
                    "label": "sensor_fault",
                    "root_cause": "bias-a",
                },
                {
                    "timestamp": "2026-05-10T12:00:00Z",
                    "entity_id": "sensor.office_temperature",
                    "label": "sensor_fault",
                    "root_cause": "drift-b",
                },
            ]
        )
        stat = statistical_result(timestamp="2026-05-10T10:01:00Z")
        context = contextual_result(
            timestamp="2026-05-10T10:01:00Z",
            outcome="contradicted",
            contradiction=0.80,
        )
        diagnosis = ClasificadorDiagnostico().analizar(
            [stat],
            [context],
        )["results"]

        evaluation = self.evaluator.evaluar(
            ground_truth,
            [stat],
            [context],
            diagnosis,
            partition="all",
            dataset_id="real-controlled-test",
        )

        self.assertTrue(evaluation["available"])
        self.assertEqual(evaluation["dataset"]["positive_episodes"], 2)
        self.assertEqual(evaluation["funnel"]["all_labeled"], 2)
        self.assertEqual(evaluation["funnel"]["passed_statistical"], 1)
        self.assertEqual(
            evaluation["end_to_end"]["by_class"]["sensor_fault"][
                "fn_before_classifier"
            ],
            1,
        )
        self.assertEqual(
            evaluation["end_to_end"]["by_class"]["sensor_fault"]["tp"],
            1,
        )
        self.assertEqual(
            evaluation["end_to_end"]["by_class"]["sensor_fault"]["recall"],
            0.5,
        )
        matrix = evaluation["confusion_matrix_e2e"]
        fault_row = matrix["values"][
            matrix["truth_labels"].index("sensor_fault")
        ]
        self.assertEqual(
            fault_row[matrix["predicted_labels"].index("possible_sensor_fault")],
            1,
        )
        self.assertEqual(
            fault_row[matrix["predicted_labels"].index("not_reached")],
            1,
        )
        self.assertIsNone(evaluation["end_to_end"]["macro_f1"])

    def test_indeterminado_es_abstencion_y_fn_posterior_estricto(self):
        ground_truth = pd.DataFrame(
            [
                {
                    "start": "2026-05-10T10:00:00Z",
                    "end": "2026-05-10T10:05:00Z",
                    "entity_id": "sensor.office_temperature",
                    "label": "sensor_fault",
                    "fault_type": "unknown",
                }
            ]
        )
        stat = statistical_result(timestamp="2026-05-10T10:01:00Z")
        context = contextual_result(
            timestamp="2026-05-10T10:01:00Z",
            outcome="insufficient_coverage",
            coverage=0.20,
        )
        diagnosis = ClasificadorDiagnostico().analizar(
            [stat],
            [context],
        )["results"]

        evaluation = self.evaluator.evaluar(
            ground_truth,
            [stat],
            [context],
            diagnosis,
            partition="all",
        )
        fault = evaluation["end_to_end"]["by_class"]["sensor_fault"]

        self.assertEqual(fault["fn_before_classifier"], 0)
        self.assertEqual(fault["fn_after_classifier"], 1)
        self.assertEqual(fault["recall"], 0.0)
        self.assertEqual(evaluation["end_to_end"]["abstention_rate"], 1.0)
        self.assertEqual(
            evaluation["metric_policy"]["indeterminate_policy"],
            "abstention_and_non_true_positive",
        )

    def test_particion_sin_casos_no_fabrica_metricas_cero(self):
        ground_truth = pd.DataFrame(
            [
                {
                    "timestamp": "2026-05-01T00:00:00Z",
                    "entity_id": "sensor.a",
                    "label": "sensor_fault",
                },
                {
                    "timestamp": "2026-05-02T00:00:00Z",
                    "entity_id": "sensor.a",
                    "label": "sensor_fault",
                },
            ]
        )

        evaluation = self.evaluator.evaluar(
            ground_truth,
            [],
            [],
            [],
            partition="validation",
        )

        self.assertFalse(evaluation["available"])
        self.assertEqual(evaluation["status"], "not_evaluated")
        self.assertNotIn("end_to_end", evaluation)


if __name__ == "__main__":
    unittest.main()
