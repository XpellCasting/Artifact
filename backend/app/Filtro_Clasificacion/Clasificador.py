from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd


CLASE_NORMAL = "normal"
CLASE_AMBIENTAL = "environmental_anomaly"
CLASE_FALLA = "possible_sensor_fault"
CLASE_INDETERMINADA = "indeterminate"


@dataclass(frozen=True)
class ConfiguracionClasificador:
    """Parametros congelados de la lista de decision del tercer filtro."""

    model_version: str = "diagnostic-rule-list-v1.0.0"
    feature_schema_version: str = "diagnostic-features-v1"
    statistical_threshold: float = 0.60
    adjusted_normal_threshold: float = 0.60
    minimum_context_coverage: float = 0.50
    environmental_support_threshold: float = 0.50
    contradiction_threshold: float = 0.35
    fault_pattern_threshold: float = 0.70
    low_confidence_threshold: float = 0.70
    alert_confidence_threshold: float = 0.75
    reject_stale_map: bool = True
    require_independent_evidence_for_common_cause: bool = True


class ClasificadorDiagnostico:
    """Lista de decision determinista que integra los dos filtros previos.

    El clasificador no aprende de ``ground_truth`` ni lo acepta como variable.
    Su cuarta salida, ``indeterminate``, es una abstencion diagnostica y no una
    causa fisica adicional.
    """

    SCHEMA_VERSION = "classifier-results-v1"
    MODEL_TYPE = "versioned_deterministic_decision_list"
    CONFIDENCE_KIND = "evidence_strength_not_calibrated_probability"

    def __init__(self, config: ConfiguracionClasificador | None = None) -> None:
        self.config = config or ConfiguracionClasificador()

    def especificacion(self) -> dict[str, Any]:
        artifact_hashes = self._artifact_hashes()
        return {
            "schema_version": "classifier-specification-v1",
            "model_version": self.config.model_version,
            "model_type": self.MODEL_TYPE,
            "feature_schema_version": self.config.feature_schema_version,
            "confidence_kind": self.CONFIDENCE_KIND,
            "definitive_for_prototype": True,
            "selection_status": "pre_registered_pending_real_validation",
            "selection_rationale": (
                "Se fija una lista de decision por su trazabilidad, abstencion "
                "explicita y ausencia de etiquetas reales suficientes para "
                "estimar sin fuga un arbol o un modelo supervisado."
            ),
            "classes": [
                CLASE_NORMAL,
                CLASE_AMBIENTAL,
                CLASE_FALLA,
                CLASE_INDETERMINADA,
            ],
            "config": asdict(self.config),
            "artifacts": artifact_hashes,
            "features": self._features_specification(),
            "decision_table": self._decision_table(),
            "subtype_policy": {
                "scope": CLASE_FALLA,
                "meaning": (
                    "Patron de lectura sugerido; no identifica por si solo la "
                    "causa raiz del dispositivo."
                ),
                "minimum_component_score": self.config.fault_pattern_threshold,
                "supported": [
                    "communication_gap",
                    "stuck_value",
                    "stuck_state_or_abnormal_duration",
                    "repeated_activation_or_interference",
                    "omission_or_expected_inactivity",
                    "persistent_shift",
                    "spike_or_extreme",
                    "unspecified_sensor_pattern",
                ],
            },
            "validation_protocol": {
                "status": "pre_registered_pending_real_validation",
                "data_policy": "real_history_only",
                "uses_simulated_dataset": False,
                "planned_history_months": 5,
                "current_history_months": 1,
                "temporal_split": {
                    "development": 0.60,
                    "validation": 0.20,
                    "test_locked": 0.20,
                },
                "fixed_thresholds": True,
                "selected_before_test": False,
                "primary_selection_metric": "fault_f2_end_to_end",
                "eligibility_gates": {
                    "fault_recall_end_to_end_min": 0.80,
                    "environmental_recall_end_to_end_min": 0.70,
                    "determinate_coverage_min": 0.70,
                    "ambiguous_common_cause_as_environmental_max": 0.05,
                    "legitimate_normal_hold_rate_max": 0.05,
                },
                "tie_breaking": [
                    "macro_f1_end_to_end",
                    "determinate_coverage",
                    "latency_p95",
                ],
                "parsimony_rule": (
                    "Conservar la lista de decision si queda a no mas de 0.02 "
                    "de macro-F1 E2E del mejor candidato elegible."
                ),
                "freeze_artifacts": [
                    "code_hash",
                    "configuration_hash",
                    "feature_schema_version",
                    "split_boundaries",
                    "context_map_version",
                ],
                "current_limitation": (
                    "El mes real permite operacion provisional, pero no una "
                    "estimacion final ni el desbloqueo de la particion de prueba."
                ),
            },
            "ground_truth_feature_policy": {
                "allowed_as_input": False,
                "usage": "evaluation_only",
            },
        }

    def analizar(
        self,
        resultados_estadisticos: list[dict[str, Any]],
        resultados_contextuales: list[dict[str, Any]],
        source: str = "historical",
    ) -> dict[str, Any]:
        inicio = perf_counter()
        context_lookup = self._context_lookup(resultados_contextuales)
        resultados: list[dict[str, Any]] = []
        for statistical in resultados_estadisticos:
            event_id = self._event_id(statistical)
            contextual = context_lookup.get(event_id)
            if contextual is None:
                contextual = context_lookup.get(
                    self._normalized_event_key(
                        statistical.get("entity_id"),
                        statistical.get("timestamp"),
                    )
                )
            resultados.append(
                self._clasificar(
                    statistical=statistical,
                    contextual=contextual,
                    event_id=event_id,
                    source=source,
                )
            )

        elapsed_ms = (perf_counter() - inicio) * 1000.0
        summary = self._summary(resultados, elapsed_ms)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "model": {
                "version": self.config.model_version,
                "type": self.MODEL_TYPE,
                "feature_schema_version": self.config.feature_schema_version,
                "confidence_kind": self.CONFIDENCE_KIND,
                "selection_status": "pre_registered_pending_real_validation",
                "trained_at": None,
                **self._artifact_hashes(),
            },
            "summary": summary,
            "results": resultados,
        }

    def _clasificar(
        self,
        statistical: dict[str, Any],
        contextual: dict[str, Any] | None,
        event_id: str,
        source: str,
    ) -> dict[str, Any]:
        started = perf_counter()
        s_stat = self._number(statistical.get("suspicion_score"), None)
        adjusted = self._number(
            None if contextual is None else contextual.get("adjusted_suspicion_score"),
            s_stat,
        )
        outcome = str(
            "missing_context"
            if contextual is None
            else contextual.get("context_outcome", "unexplained")
        )
        support = self._number(
            None if contextual is None else contextual.get("support_score"),
            0.0,
        )
        contradiction = self._number(
            None if contextual is None else contextual.get("contradiction_score"),
            0.0,
        )
        context_score = self._number(
            None if contextual is None else contextual.get("context_score"),
            0.0,
        )
        coverage = dict((contextual or {}).get("coverage") or {})
        coverage_ratio = self._number(coverage.get("ratio"), 0.0)
        map_quality = dict((contextual or {}).get("map_quality") or {})
        common_cause = dict((contextual or {}).get("common_cause") or {})
        common_candidate = bool(common_cause.get("candidate"))
        independent_evidence = bool(
            common_cause.get("independent_evidence_available")
        )
        map_stale = bool(map_quality.get("stale"))
        pattern_score, pattern_component = self._pattern_strength(statistical)
        invalid_reasons = self._validate_input(statistical, contextual)

        diagnostic_class: str
        rule_id: str
        reasons: list[str]

        if invalid_reasons:
            diagnostic_class = CLASE_INDETERMINADA
            rule_id = "R00_INVALID_OR_MISSING_INPUT"
            reasons = invalid_reasons
        elif (
            self.config.require_independent_evidence_for_common_cause
            and common_candidate
            and not independent_evidence
        ):
            diagnostic_class = CLASE_INDETERMINADA
            rule_id = "R01_AMBIGUOUS_COMMON_CAUSE"
            reasons = ["possible_common_cause_without_independent_evidence"]
        elif self.config.reject_stale_map and map_stale:
            diagnostic_class = CLASE_INDETERMINADA
            rule_id = "R02_STALE_CONTEXT_MAP"
            reasons = ["stale_context_map"]
        elif (
            coverage_ratio < self.config.minimum_context_coverage
            or str(coverage.get("status", "")).lower() != "sufficient"
            or outcome == "insufficient_coverage"
        ):
            diagnostic_class = CLASE_INDETERMINADA
            rule_id = "R03_INSUFFICIENT_CONTEXT_COVERAGE"
            reasons = ["insufficient_context_coverage"]
        elif (
            outcome == "supported"
            and support >= self.config.environmental_support_threshold
        ):
            diagnostic_class = CLASE_AMBIENTAL
            rule_id = "R04_STRONG_CONTEXTUAL_SUPPORT"
            reasons = ["strong_contextual_support"]
        elif (
            adjusted is not None
            and adjusted < self.config.adjusted_normal_threshold
        ):
            diagnostic_class = CLASE_NORMAL
            rule_id = "R05_CONTEXT_REJECTS_SUSPICION"
            reasons = ["adjusted_suspicion_below_classification_threshold"]
        elif (
            outcome == "contradicted"
            and contradiction >= self.config.contradiction_threshold
        ):
            diagnostic_class = CLASE_FALLA
            rule_id = "R06_CONTEXTUAL_CONTRADICTION"
            reasons = ["contextual_contradiction"]
        elif (
            outcome == "unexplained"
            and pattern_score >= self.config.fault_pattern_threshold
        ):
            diagnostic_class = CLASE_FALLA
            rule_id = "R07_UNEXPLAINED_SPECIFIC_PATTERN"
            reasons = [
                "unexplained_specific_sensor_pattern",
                f"dominant_component:{pattern_component}",
            ]
        else:
            diagnostic_class = CLASE_INDETERMINADA
            rule_id = "R08_UNRESOLVED_EVIDENCE"
            reasons = ["unexplained_or_conflicting_evidence"]

        confidence = self._confidence(
            diagnostic_class=diagnostic_class,
            rule_id=rule_id,
            s_stat=s_stat or 0.0,
            adjusted=adjusted or 0.0,
            support=support,
            contradiction=contradiction,
            coverage=coverage_ratio,
            pattern_strength=pattern_score,
        )
        subtype = (
            self._suggested_subtype(statistical)
            if diagnostic_class == CLASE_FALLA
            else None
        )
        limitations = self._limitations(
            diagnostic_class,
            common_candidate,
            independent_evidence,
            map_stale,
            coverage_ratio,
            contextual is None,
        )
        action = self._action(diagnostic_class, confidence)
        features = self._feature_trace(
            statistical_score=s_stat,
            adjusted_suspicion_score=adjusted,
            context_outcome=outcome,
            context_score=context_score,
            support_score=support,
            contradiction_score=contradiction,
            coverage_ratio=coverage_ratio,
            context_map_stale=map_stale,
            common_cause_candidate=common_candidate,
            independent_evidence_available=independent_evidence,
            indicator_component_max=pattern_score,
            dominant_component=pattern_component,
            diagnostic_class=diagnostic_class,
        )
        elapsed_ms = (perf_counter() - started) * 1000.0
        timestamp = statistical.get("timestamp")
        diagnosis_id = hashlib.sha256(
            f"{self.config.model_version}|{event_id}".encode("utf-8")
        ).hexdigest()[:20]
        explanation = self._explanation(
            diagnostic_class,
            rule_id,
            reasons,
            subtype,
        )
        return {
            "diagnosis_id": f"diag-{diagnosis_id}",
            "event_id": event_id,
            "entity_id": statistical.get("entity_id"),
            "timestamp": timestamp,
            "window_start": statistical.get("window_start", timestamp),
            "window_end": statistical.get("window_end", timestamp),
            "sensor_kind": statistical.get("sensor_kind"),
            "source": source,
            "diagnostic_class": diagnostic_class,
            "subtype": subtype,
            "confidence": confidence,
            "confidence_kind": self.CONFIDENCE_KIND,
            "abstained": diagnostic_class == CLASE_INDETERMINADA,
            "decision_rule_id": rule_id,
            "reason_codes": reasons,
            "suggested_action": action,
            "features_used": features,
            "evidence": features,
            "limitations": limitations,
            "explanation": explanation,
            "model_version": self.config.model_version,
            "feature_schema_version": self.config.feature_schema_version,
            "input_provenance": {
                "statistical_event_id": event_id,
                "contextual_event_id": (contextual or {}).get("event_id"),
                "ground_truth_used_as_feature": False,
            },
            "pipeline": {
                "statistical": {
                    "score": s_stat,
                    "evidence_score": statistical.get("evidence_score"),
                    "score_version": statistical.get("score_version"),
                    "score_kind": statistical.get("score_kind"),
                    "threshold": self.config.statistical_threshold,
                    "passed": bool(
                        statistical.get("passed_to_contextual_filter", True)
                    ),
                    "window_start": statistical.get("window_start", timestamp),
                    "window_end": statistical.get("window_end", timestamp),
                    "pattern": statistical.get(
                        "suspected_pattern",
                        statistical.get("pattern"),
                    ),
                    "indicators": list(
                        statistical.get("triggered_indicators") or []
                    ),
                },
                "contextual": {
                    "outcome": outcome,
                    "context_score": context_score,
                    "coherence_score": (contextual or {}).get("coherence_score"),
                    "support_score": support,
                    "contradiction_score": contradiction,
                    "adjusted_suspicion_score": adjusted,
                    "coverage": coverage,
                    "common_cause": common_cause,
                    "map": map_quality,
                },
                "classifier": {
                    "model_version": self.config.model_version,
                    "decision_rule_id": rule_id,
                    "elapsed_ms": round(elapsed_ms, 6),
                    "reason_codes": reasons,
                    "explanation": explanation,
                },
            },
        }

    def _features_specification(self) -> list[dict[str, Any]]:
        return [
            self._feature_spec("statistical_score", "statistical", "float [0,1]"),
            self._feature_spec("adjusted_suspicion_score", "contextual", "float [0,1]"),
            self._feature_spec(
                "context_outcome",
                "contextual",
                "supported|contradicted|unexplained|insufficient_coverage",
            ),
            self._feature_spec("context_score", "contextual", "float [-1,1]"),
            self._feature_spec("support_score", "contextual", "float [0,1]"),
            self._feature_spec("contradiction_score", "contextual", "float [0,1]"),
            self._feature_spec("coverage_ratio", "contextual", "float [0,1]"),
            self._feature_spec("coverage_status", "contextual", "categorical"),
            self._feature_spec("common_cause_candidate", "contextual", "boolean"),
            self._feature_spec(
                "independent_evidence_available",
                "independent telemetry metadata",
                "boolean",
            ),
            self._feature_spec("context_map_stale", "contextual map", "boolean"),
            self._feature_spec("sensor_kind", "statistical", "categorical"),
            self._feature_spec("triggered_indicators", "statistical", "list[string]"),
            self._feature_spec("indicator_component_max", "statistical", "float [0,1]"),
        ]

    def _feature_spec(self, name: str, source: str, domain: str) -> dict[str, Any]:
        return {
            "name": name,
            "source": source,
            "type": domain,
            "required": True,
            "normalization": domain,
            "ground_truth": False,
        }

    def _decision_table(self) -> list[dict[str, Any]]:
        c = self.config
        return [
            {
                "rule_id": "R00_INVALID_OR_MISSING_INPUT",
                "precedence": 0,
                "condition": "required input invalid OR contextual result missing",
                "output": CLASE_INDETERMINADA,
            },
            {
                "rule_id": "R01_AMBIGUOUS_COMMON_CAUSE",
                "precedence": 1,
                "condition": (
                    "common_cause.candidate AND NOT "
                    "common_cause.independent_evidence_available"
                ),
                "output": CLASE_INDETERMINADA,
            },
            {
                "rule_id": "R02_STALE_CONTEXT_MAP",
                "precedence": 2,
                "condition": "map_quality.stale = true",
                "output": CLASE_INDETERMINADA,
            },
            {
                "rule_id": "R03_INSUFFICIENT_CONTEXT_COVERAGE",
                "precedence": 3,
                "condition": f"coverage.ratio < {c.minimum_context_coverage}",
                "output": CLASE_INDETERMINADA,
            },
            {
                "rule_id": "R04_STRONG_CONTEXTUAL_SUPPORT",
                "precedence": 4,
                "condition": (
                    "context_outcome = supported AND "
                    f"support_score >= {c.environmental_support_threshold}"
                ),
                "output": CLASE_AMBIENTAL,
            },
            {
                "rule_id": "R05_CONTEXT_REJECTS_SUSPICION",
                "precedence": 5,
                "condition": (
                    "adjusted_suspicion_score < "
                    f"{c.adjusted_normal_threshold}"
                ),
                "output": CLASE_NORMAL,
            },
            {
                "rule_id": "R06_CONTEXTUAL_CONTRADICTION",
                "precedence": 6,
                "condition": (
                    "context_outcome = contradicted AND "
                    f"contradiction_score >= {c.contradiction_threshold}"
                ),
                "output": CLASE_FALLA,
            },
            {
                "rule_id": "R07_UNEXPLAINED_SPECIFIC_PATTERN",
                "precedence": 7,
                "condition": (
                    "context_outcome = unexplained AND "
                    f"indicator_component_max >= {c.fault_pattern_threshold}"
                ),
                "output": CLASE_FALLA,
            },
            {
                "rule_id": "R08_UNRESOLVED_EVIDENCE",
                "precedence": 8,
                "condition": "fallback",
                "output": CLASE_INDETERMINADA,
            },
        ]

    def _feature_trace(self, **values: Any) -> list[dict[str, Any]]:
        source_by_name = {
            "statistical_score": "statistical",
            "adjusted_suspicion_score": "contextual",
            "context_outcome": "contextual",
            "context_score": "contextual",
            "support_score": "contextual",
            "contradiction_score": "contextual",
            "coverage_ratio": "contextual",
            "context_map_stale": "contextual_map",
            "common_cause_candidate": "contextual",
            "independent_evidence_available": "independent_telemetry",
            "indicator_component_max": "statistical",
            "dominant_component": "statistical",
        }
        direction = values.pop("diagnostic_class", None)
        rows = []
        for name, value in values.items():
            rows.append(
                {
                    "name": name,
                    "value": value,
                    "source": source_by_name.get(name, "derived"),
                    "available": value is not None,
                    "contribution": None,
                    "direction": direction,
                }
            )
        return rows

    def _artifact_hashes(self) -> dict[str, str]:
        configuration = json.dumps(
            asdict(self.config),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        implementation = Path(__file__).read_bytes()
        return {
            "configuration_sha256": hashlib.sha256(configuration).hexdigest(),
            "implementation_sha256": hashlib.sha256(implementation).hexdigest(),
        }

    def _validate_input(
        self,
        statistical: dict[str, Any],
        contextual: dict[str, Any] | None,
    ) -> list[str]:
        reasons: list[str] = []
        for field_name in ["entity_id", "timestamp", "sensor_kind"]:
            if not statistical.get(field_name):
                reasons.append(f"missing_statistical_field:{field_name}")
        if self._number(statistical.get("suspicion_score"), None) is None:
            reasons.append("missing_statistical_field:suspicion_score")
        elif not 0.0 <= float(statistical["suspicion_score"]) <= 1.0:
            reasons.append("out_of_range:suspicion_score")
        reading = statistical.get("reading")
        if not isinstance(reading, dict) or not isinstance(
            reading.get("components"),
            dict,
        ):
            reasons.append("missing_statistical_field:reading.components")

        if contextual is None:
            reasons.append("missing_contextual_result")
            return reasons
        for field_name in [
            "event_id",
            "adjusted_suspicion_score",
            "context_score",
            "support_score",
            "contradiction_score",
            "context_outcome",
        ]:
            if field_name not in contextual or contextual.get(field_name) is None:
                reasons.append(f"missing_contextual_field:{field_name}")
        for field_name in [
            "adjusted_suspicion_score",
            "support_score",
            "contradiction_score",
        ]:
            value = self._number(contextual.get(field_name), None)
            if value is not None and not 0.0 <= value <= 1.0:
                reasons.append(f"out_of_range:{field_name}")
        context_score = self._number(contextual.get("context_score"), None)
        if context_score is not None and not -1.0 <= context_score <= 1.0:
            reasons.append("out_of_range:context_score")
        if str(contextual.get("context_outcome")) not in {
            "supported",
            "contradicted",
            "unexplained",
            "insufficient_coverage",
        }:
            reasons.append("invalid_value:context_outcome")

        coverage = contextual.get("coverage")
        if not isinstance(coverage, dict):
            reasons.append("missing_contextual_field:coverage")
        else:
            ratio = self._number(coverage.get("ratio"), None)
            if ratio is None:
                reasons.append("missing_contextual_field:coverage.ratio")
            elif not 0.0 <= ratio <= 1.0:
                reasons.append("out_of_range:coverage.ratio")
            if "status" not in coverage:
                reasons.append("missing_contextual_field:coverage.status")

        common_cause = contextual.get("common_cause")
        if not isinstance(common_cause, dict):
            reasons.append("missing_contextual_field:common_cause")
        else:
            for field_name in [
                "candidate",
                "independent_evidence_available",
            ]:
                if field_name not in common_cause:
                    reasons.append(
                        f"missing_contextual_field:common_cause.{field_name}"
                    )
        map_quality = contextual.get("map_quality")
        if not isinstance(map_quality, dict) or "stale" not in map_quality:
            reasons.append("missing_contextual_field:map_quality.stale")
        return reasons

    def _pattern_strength(self, statistical: dict[str, Any]) -> tuple[float, str]:
        components = dict(
            (statistical.get("reading") or {}).get("components") or {}
        )
        candidates = [
            (str(name), self._number(value, 0.0) or 0.0)
            for name, value in components.items()
        ]
        if candidates:
            name, score = max(candidates, key=lambda item: (item[1], item[0]))
            return max(0.0, min(1.0, score)), name
        score = self._number(statistical.get("evidence_score"), 0.0) or 0.0
        indicators = sorted(
            str(item) for item in statistical.get("triggered_indicators") or []
        )
        return max(0.0, min(1.0, score)), (
            indicators[0] if indicators else "unavailable"
        )

    def _suggested_subtype(self, statistical: dict[str, Any]) -> str:
        indicators = set(
            str(item) for item in statistical.get("triggered_indicators") or []
        )
        ordered = [
            ({"reading_gap", "report_gap"}, "communication_gap"),
            ({"constant_persistence"}, "stuck_value"),
            (
                {
                    "active_time_ratio",
                    "time_since_last_change",
                    "z_duration_score",
                },
                "stuck_state_or_abnormal_duration",
            ),
            (
                {"activation_count", "state_change_count"},
                "repeated_activation_or_interference",
            ),
            (
                {"time_since_last_activation", "inactive_duration"},
                "omission_or_expected_inactivity",
            ),
            ({"cusum_residual"}, "persistent_shift"),
            (
                {"hampel", "robust_rate_change", "dspot_extreme"},
                "spike_or_extreme",
            ),
        ]
        for names, subtype in ordered:
            if indicators.intersection(names):
                return subtype
        pattern = str(
            statistical.get("suspected_pattern")
            or statistical.get("pattern")
            or ""
        )
        return pattern if pattern else "unspecified_sensor_pattern"

    def _confidence(
        self,
        diagnostic_class: str,
        rule_id: str,
        s_stat: float,
        adjusted: float,
        support: float,
        contradiction: float,
        coverage: float,
        pattern_strength: float,
    ) -> float | None:
        if diagnostic_class == CLASE_INDETERMINADA:
            return None
        if diagnostic_class == CLASE_AMBIENTAL:
            value = 0.40 * s_stat + 0.35 * support + 0.25 * coverage
        elif diagnostic_class == CLASE_NORMAL:
            value = (
                0.45 * (1.0 - adjusted)
                + 0.35 * support
                + 0.20 * coverage
            )
        elif rule_id == "R06_CONTEXTUAL_CONTRADICTION":
            value = (
                0.45 * s_stat
                + 0.35 * contradiction
                + 0.20 * coverage
            )
        else:
            value = (
                0.45 * s_stat
                + 0.35 * pattern_strength
                + 0.20 * coverage
            )
        return round(max(0.0, min(1.0, value)), 6)

    def _action(self, diagnostic_class: str, confidence: float | None) -> str:
        if diagnostic_class == CLASE_NORMAL:
            return "accept_candidate_as_normal"
        if diagnostic_class == CLASE_AMBIENTAL:
            return "observe_environmental_event"
        if diagnostic_class == CLASE_FALLA:
            if (
                confidence is not None
                and confidence >= self.config.alert_confidence_threshold
            ):
                return "alert_and_hold_sensitive_automation_recommended"
            return "inspect_sensor_before_automation"
        return "collect_context_or_independent_evidence"

    def _limitations(
        self,
        diagnostic_class: str,
        common_candidate: bool,
        independent_evidence: bool,
        map_stale: bool,
        coverage: float,
        missing_context: bool,
    ) -> list[str]:
        limitations = ["associative_context_does_not_prove_causality"]
        if missing_context:
            limitations.append("contextual_result_missing")
        if common_candidate and not independent_evidence:
            limitations.append("common_cause_unidentifiable")
        if map_stale:
            limitations.append("context_map_stale")
        if coverage < self.config.minimum_context_coverage:
            limitations.append("insufficient_context_coverage")
        if diagnostic_class == CLASE_FALLA:
            limitations.append("fault_subtype_is_suggested_not_confirmed_root_cause")
        return limitations

    def _explanation(
        self,
        diagnostic_class: str,
        rule_id: str,
        reasons: list[str],
        subtype: str | None,
    ) -> str:
        subtype_text = f" Patron sugerido: {subtype}." if subtype else ""
        return (
            f"La primera regla satisfecha fue {rule_id}; la salida es "
            f"{diagnostic_class}. Evidencia: {', '.join(reasons)}."
            f"{subtype_text}"
        )

    def _summary(
        self,
        resultados: list[dict[str, Any]],
        elapsed_ms: float,
    ) -> dict[str, Any]:
        by_class: dict[str, int] = {}
        confidences = []
        for item in resultados:
            key = str(item.get("diagnostic_class", CLASE_INDETERMINADA))
            by_class[key] = by_class.get(key, 0) + 1
            confidence = item.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))
        return {
            "input_candidates": len(resultados),
            "evaluated": len(resultados),
            "classified": len(resultados) - by_class.get(CLASE_INDETERMINADA, 0),
            "abstained": by_class.get(CLASE_INDETERMINADA, 0),
            "by_class": by_class,
            "low_confidence": sum(
                confidence < self.config.low_confidence_threshold
                for confidence in confidences
            ),
            "confidence_mean": (
                round(sum(confidences) / len(confidences), 6)
                if confidences
                else None
            ),
            "elapsed_ms": round(float(elapsed_ms), 3),
        }

    def _context_lookup(
        self,
        contextual_results: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for item in contextual_results:
            event_id = str(item.get("event_id") or "")
            if event_id:
                lookup[event_id] = item
            lookup[
                self._normalized_event_key(
                    item.get("entity_id"),
                    item.get("timestamp"),
                )
            ] = item
        return lookup

    def _event_id(self, item: dict[str, Any]) -> str:
        return str(
            item.get("event_id")
            or f"{item.get('entity_id')}@{item.get('timestamp')}"
        )

    def _normalized_event_key(self, entity_id: Any, timestamp: Any) -> str:
        value = pd.to_datetime(timestamp, errors="coerce", utc=True)
        normalized = str(timestamp) if pd.isna(value) else value.isoformat()
        return f"{entity_id}@{normalized}"

    def _number(self, value: Any, default: float | None) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if pd.isna(number):
            return default
        return number
