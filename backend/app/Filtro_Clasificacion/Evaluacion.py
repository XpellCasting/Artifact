from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import pandas as pd


TRUTH_LABELS = ["normal", "environmental_anomaly", "sensor_fault"]
PREDICTION_LABELS = [
    "normal",
    "environmental_anomaly",
    "possible_sensor_fault",
    "indeterminate",
    "not_reached",
]


@dataclass(frozen=True)
class ConfiguracionEvaluacion:
    """Politica congelada para evaluar el flujo completo por episodios."""

    development_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    normal_window_seconds: int = 3600
    minimum_episode_gap_seconds: int = 120
    maximum_episode_gap_seconds: int = 3600
    detection_grace_seconds: int = 0
    minimum_planned_history_days: int = 150
    missed_case_sample_size: int = 25


class EvaluadorExtremoAExtremo:
    """Evalua cada episodio etiquetado, incluso si un gate lo descarta."""

    SCHEMA_VERSION = "classifier-evaluation-v1"

    def __init__(self, config: ConfiguracionEvaluacion | None = None) -> None:
        self.config = config or ConfiguracionEvaluacion()

    def evaluar(
        self,
        ground_truth: pd.DataFrame,
        resultados_estadisticos: list[dict[str, Any]],
        resultados_contextuales: list[dict[str, Any]],
        resultados_clasificacion: list[dict[str, Any]],
        partition: str = "test",
        dataset_id: str = "real-history-ground-truth",
        timeline_start: Any | None = None,
        timeline_end: Any | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        prepared, uncertain_excluded = self._prepare_ground_truth(ground_truth)
        selected, boundaries = self._select_partition(
            prepared,
            partition,
            timeline_start=timeline_start,
            timeline_end=timeline_end,
        )
        episodes = self._build_episodes(selected)
        if episodes.empty:
            return self._unavailable(
                dataset_id=dataset_id,
                partition=partition,
                uncertain_excluded=uncertain_excluded,
                boundaries=boundaries,
                warning=(
                    "La particion no contiene episodios etiquetados evaluables; "
                    "no se calculan metricas iguales a cero."
                ),
            )

        records = [
            self._evaluate_episode(
                episode,
                resultados_estadisticos,
                resultados_contextuales,
                resultados_clasificacion,
            )
            for episode in episodes.to_dict("records")
        ]
        matrix_e2e = self._confusion(
            records,
            truth_labels=TRUTH_LABELS,
            predicted_labels=PREDICTION_LABELS,
        )
        reached_records = [
            item for item in records if item["reached_classifier"]
        ]
        conditional_predictions = [
            "normal",
            "environmental_anomaly",
            "possible_sensor_fault",
            "indeterminate",
        ]
        conditional_matrix = self._confusion(
            reached_records,
            truth_labels=TRUTH_LABELS,
            predicted_labels=conditional_predictions,
        )
        end_to_end = self._metrics(records, scope="end_to_end")
        classifier_conditional = self._metrics(
            reached_records,
            scope="only_reached_classifier",
        )
        classifier_conditional["confusion_matrix"] = conditional_matrix
        class_support = {
            label: sum(item["truth_class"] == label for item in records)
            for label in TRUTH_LABELS
        }
        missing_classes = [
            label for label, support in class_support.items() if support == 0
        ]
        selected_start = episodes["_start"].min()
        selected_end = episodes["_end"].max()
        duration_days = max(
            0.0,
            float((selected_end - selected_start).total_seconds() / 86400.0),
        )
        warnings: list[str] = []
        if missing_classes:
            warnings.append(
                "Faltan clases reales en la particion: "
                + ", ".join(missing_classes)
                + ". Las metricas agregadas finales permanecen no disponibles."
            )
        if duration_days < self.config.minimum_planned_history_days:
            warnings.append(
                "La cobertura etiquetada es menor que los cinco meses "
                "planificados; los resultados son provisionales."
            )
        if partition.lower() == "test":
            warnings.append(
                "La particion de prueba solo debe interpretarse despues de "
                "congelar modelo, umbrales, mapa y limites temporales."
            )

        valid_for_final_metrics = (
            not missing_classes
            and duration_days >= self.config.minimum_planned_history_days
        )
        if not valid_for_final_metrics:
            for aggregate in [
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "balanced_accuracy",
            ]:
                end_to_end[aggregate] = None
                classifier_conditional[aggregate] = None

        losses = self._preclassifier_losses(records)
        funnel = {
            "all_labeled": len(records),
            "passed_statistical": sum(
                item["reached_statistical"] for item in records
            ),
            "rejected_statistical": sum(
                not item["reached_statistical"] for item in records
            ),
            "contextual_evaluated": sum(
                item["reached_contextual"] for item in records
            ),
            "reached_classifier": len(reached_records),
            "classified": sum(
                item["reached_classifier"]
                and item["predicted_class"] != "indeterminate"
                for item in records
            ),
            "indeterminate": sum(
                item["predicted_class"] == "indeterminate" for item in records
            ),
        }
        missed = [
            self._missed_case(item)
            for item in records
            if item["truth_class"] != "normal" and not item["correct"]
        ][: self.config.missed_case_sample_size]
        evaluation_hash = self._evaluation_hash(
            selected,
            partition,
            dataset_id,
        )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "available": True,
            "status": (
                "valid_for_final_metrics"
                if valid_for_final_metrics
                else "provisional_partial_ground_truth"
            ),
            "evaluation_id": f"eval-{evaluation_hash[:20]}",
            "dataset": {
                "id": dataset_id,
                "hash": evaluation_hash,
                "source_policy": "real_or_controlled_ground_truth_only",
                "uses_simulated_dataset": False,
                "split": partition.lower(),
                "labeled_rows": int(len(selected)),
                "labeled_cases": len(records),
                "positive_episodes": int(
                    sum(item["truth_class"] != "normal" for item in records)
                ),
                "uncertain_excluded": int(uncertain_excluded),
                "start": selected_start.isoformat(),
                "end": selected_end.isoformat(),
                "duration_days": round(duration_days, 3),
                "months_available": round(duration_days / 30.4375, 3),
                "months_required": 5,
                "partition_boundaries": boundaries,
                "unit_of_analysis": "episode",
            },
            "metric_policy": {
                "truth_labels": TRUTH_LABELS,
                "prediction_labels": PREDICTION_LABELS,
                "gate_rejection_policy": (
                    "not_reached_in_audit_matrix; implicit_normal_for_metrics"
                ),
                "indeterminate_policy": "abstention_and_non_true_positive",
                "normal_not_reached_policy": "implicit_normal_true_positive",
                "positive_not_reached_policy": "strict_false_negative",
                "fixed_thresholds": True,
                "matching": (
                    "same entity and temporal interval overlap; earliest "
                    "classifier decision wins"
                ),
                "configuration": asdict(self.config),
            },
            "funnel": funnel,
            "preclassifier_losses": losses,
            "confusion_matrix_e2e": matrix_e2e,
            "end_to_end": end_to_end,
            "classifier_conditional": classifier_conditional,
            "valid_for_final_metrics": valid_for_final_metrics,
            "warnings": warnings,
            "missed_cases_sample": missed,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }

    def _prepare_ground_truth(
        self,
        ground_truth: pd.DataFrame,
    ) -> tuple[pd.DataFrame, int]:
        if ground_truth is None or ground_truth.empty:
            return pd.DataFrame(), 0
        frame = ground_truth.copy()
        if "entity_id" not in frame or "label" not in frame:
            raise ValueError(
                "La base de verdad requiere las columnas entity_id y label."
            )
        start_source = (
            frame["start"]
            if "start" in frame
            else frame["timestamp"]
            if "timestamp" in frame
            else None
        )
        if start_source is None:
            raise ValueError(
                "La base de verdad requiere timestamp o el intervalo start/end."
            )
        end_source = frame["end"] if "end" in frame else start_source
        frame["_start"] = pd.to_datetime(start_source, errors="coerce", utc=True)
        frame["_end"] = pd.to_datetime(end_source, errors="coerce", utc=True)
        frame["_end"] = frame["_end"].where(
            frame["_end"].ge(frame["_start"]),
            frame["_start"],
        )
        frame["_truth"] = frame["label"].map(self._truth_label)
        frame["_root"] = self._root_cause_series(frame)
        uncertain = frame["_truth"].isna()
        uncertain_excluded = int(uncertain.sum())
        frame = frame[
            frame["_start"].notna()
            & frame["_end"].notna()
            & frame["entity_id"].astype(str).ne("")
            & ~uncertain
        ].copy()
        frame["entity_id"] = frame["entity_id"].astype(str)
        return frame.sort_values(["_start", "entity_id"]).reset_index(drop=True), uncertain_excluded

    def _select_partition(
        self,
        frame: pd.DataFrame,
        partition: str,
        timeline_start: Any | None = None,
        timeline_end: Any | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        key = str(partition or "test").lower()
        if frame.empty:
            return frame, {}
        requested_minimum = pd.to_datetime(
            timeline_start,
            errors="coerce",
            utc=True,
        )
        requested_maximum = pd.to_datetime(
            timeline_end,
            errors="coerce",
            utc=True,
        )
        minimum = (
            requested_minimum
            if pd.notna(requested_minimum)
            else frame["_start"].min()
        )
        maximum = (
            requested_maximum
            if pd.notna(requested_maximum)
            else frame["_end"].max()
        )
        if maximum < minimum:
            raise ValueError("El eje temporal de evaluacion tiene limites invalidos.")
        span = maximum - minimum
        development_end = minimum + span * self.config.development_fraction
        validation_end = minimum + span * (
            self.config.development_fraction
            + self.config.validation_fraction
        )
        boundaries = {
            "start": minimum.isoformat(),
            "development_end": development_end.isoformat(),
            "validation_end": validation_end.isoformat(),
            "end": maximum.isoformat(),
        }
        if key in {"all", "full"}:
            return frame.copy(), boundaries
        if key in {"development", "train"}:
            mask = frame["_start"].lt(development_end)
        elif key in {"validation", "val"}:
            mask = frame["_start"].ge(development_end) & frame["_start"].lt(
                validation_end
            )
        elif key == "test":
            mask = frame["_start"].ge(validation_end)
        else:
            raise ValueError(
                "partition debe ser development, validation, test o all."
            )
        return frame[mask].copy(), boundaries

    def _build_episodes(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        positives = frame[frame["_truth"].ne("normal")].copy()
        normals = frame[frame["_truth"].eq("normal")].copy()
        episodes: list[dict[str, Any]] = []
        group_columns = ["entity_id", "_truth", "_root"]
        for keys, group in positives.groupby(
            group_columns,
            sort=True,
            dropna=False,
        ):
            rows = group.sort_values("_start").to_dict("records")
            gap_seconds = self._adaptive_gap_seconds(group)
            current: dict[str, Any] | None = None
            for row in rows:
                if current is None:
                    current = self._new_episode(keys, row)
                    continue
                allowed_start = current["_end"] + pd.Timedelta(
                    seconds=gap_seconds
                )
                if row["_start"] <= allowed_start:
                    current["_end"] = max(current["_end"], row["_end"])
                    current["row_count"] += 1
                else:
                    episodes.append(self._finish_episode(current))
                    current = self._new_episode(keys, row)
            if current is not None:
                episodes.append(self._finish_episode(current))

        if not normals.empty:
            normals["_bucket"] = normals["_start"].dt.floor(
                f"{self.config.normal_window_seconds}s"
            )
            for (entity_id, bucket), group in normals.groupby(
                ["entity_id", "_bucket"],
                sort=True,
            ):
                row = {
                    "entity_id": entity_id,
                    "_truth": "normal",
                    "_root": "normal",
                    "_start": group["_start"].min(),
                    "_end": group["_end"].max(),
                }
                episode = self._new_episode(
                    (entity_id, "normal", "normal"),
                    row,
                )
                episode["row_count"] = int(len(group))
                episode["_start"] = bucket
                bucket_end = bucket + pd.Timedelta(
                    seconds=self.config.normal_window_seconds
                )
                episode["_end"] = min(max(episode["_end"], bucket), bucket_end)
                episodes.append(self._finish_episode(episode))
        return pd.DataFrame(episodes).sort_values(
            ["_start", "entity_id"]
        ).reset_index(drop=True)

    def _new_episode(
        self,
        keys: tuple[Any, ...],
        row: dict[str, Any],
    ) -> dict[str, Any]:
        entity_id, truth, root = keys
        return {
            "entity_id": str(entity_id),
            "truth_class": str(truth),
            "root_cause": str(root),
            "_start": row["_start"],
            "_end": row["_end"],
            "row_count": 1,
        }

    def _finish_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        raw = (
            f"{episode['entity_id']}|{episode['truth_class']}|"
            f"{episode['root_cause']}|{episode['_start'].isoformat()}|"
            f"{episode['_end'].isoformat()}"
        )
        episode["episode_id"] = (
            "gt-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
        )
        return episode

    def _adaptive_gap_seconds(self, group: pd.DataFrame) -> float:
        differences = (
            group.sort_values("_start")["_start"].diff().dt.total_seconds()
        )
        median = differences[differences.gt(0)].median()
        if pd.isna(median):
            return float(self.config.minimum_episode_gap_seconds)
        return float(
            max(
                self.config.minimum_episode_gap_seconds,
                min(
                    self.config.maximum_episode_gap_seconds,
                    float(median) * 1.5,
                ),
            )
        )

    def _evaluate_episode(
        self,
        episode: dict[str, Any],
        statistical: list[dict[str, Any]],
        contextual: list[dict[str, Any]],
        classifications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stat_matches = self._matching_results(episode, statistical)
        context_matches = self._matching_results(episode, contextual)
        classifier_matches = self._matching_results(episode, classifications)
        classifier_matches.sort(key=self._decision_sort_key)
        reached_statistical = bool(stat_matches)
        reached_contextual = bool(context_matches)
        reached_classifier = bool(classifier_matches)
        if reached_classifier:
            selected = classifier_matches[0]
            predicted = self._prediction_label(
                selected.get("diagnostic_class")
            )
            diagnosis_id = selected.get("diagnosis_id")
            decision_timestamp = self._result_interval(selected)[0]
        elif reached_statistical:
            selected = None
            predicted = "indeterminate"
            diagnosis_id = None
            decision_timestamp = None
        else:
            selected = None
            predicted = "not_reached"
            diagnosis_id = None
            decision_timestamp = None
        truth = str(episode["truth_class"])
        expected = self._expected_prediction(truth)
        metric_prediction = (
            "normal" if predicted == "not_reached" else predicted
        )
        correct = metric_prediction == expected
        return {
            "episode_id": episode["episode_id"],
            "entity_id": episode["entity_id"],
            "truth_class": truth,
            "root_cause": episode["root_cause"],
            "start": episode["_start"],
            "end": episode["_end"],
            "row_count": int(episode["row_count"]),
            "reached_statistical": reached_statistical,
            "reached_contextual": reached_contextual,
            "reached_classifier": reached_classifier,
            "predicted_class": predicted,
            "metric_prediction": metric_prediction,
            "expected_prediction": expected,
            "correct": correct,
            "diagnosis_id": diagnosis_id,
            "decision_timestamp": decision_timestamp,
            "statistical_matches": len(stat_matches),
            "contextual_matches": len(context_matches),
            "classifier_matches": len(classifier_matches),
        }

    def _matching_results(
        self,
        episode: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        matches = []
        grace = pd.Timedelta(seconds=self.config.detection_grace_seconds)
        for item in results:
            if str(item.get("entity_id")) != str(episode["entity_id"]):
                continue
            start, end = self._result_interval(item)
            if start is None or end is None:
                continue
            if end >= episode["_start"] and start <= episode["_end"] + grace:
                matches.append(item)
        return matches

    def _result_interval(
        self,
        item: dict[str, Any],
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        pipeline_stat = dict((item.get("pipeline") or {}).get("statistical") or {})
        start_value = (
            item.get("window_start")
            or pipeline_stat.get("window_start")
            or item.get("timestamp")
        )
        end_value = (
            item.get("window_end")
            or pipeline_stat.get("window_end")
            or item.get("timestamp")
        )
        start = pd.to_datetime(start_value, errors="coerce", utc=True)
        end = pd.to_datetime(end_value, errors="coerce", utc=True)
        if pd.isna(start):
            return None, None
        if pd.isna(end) or end < start:
            end = start
        return start, end

    def _decision_sort_key(self, item: dict[str, Any]) -> tuple[Any, str]:
        start, _ = self._result_interval(item)
        return (
            start if start is not None else pd.Timestamp.max.tz_localize("UTC"),
            str(item.get("diagnosis_id") or item.get("event_id") or ""),
        )

    def _confusion(
        self,
        records: list[dict[str, Any]],
        truth_labels: list[str],
        predicted_labels: list[str],
    ) -> dict[str, Any]:
        values = [
            [
                sum(
                    item["truth_class"] == truth
                    and item["predicted_class"] == predicted
                    for item in records
                )
                for predicted in predicted_labels
            ]
            for truth in truth_labels
        ]
        return {
            "truth_labels": truth_labels,
            "predicted_labels": predicted_labels,
            "values": values,
        }

    def _metrics(
        self,
        records: list[dict[str, Any]],
        scope: str,
    ) -> dict[str, Any]:
        by_class: dict[str, Any] = {}
        f1_values = []
        recalls = []
        weighted_f1_sum = 0.0
        support_total = len(records)
        for truth_label in TRUTH_LABELS:
            expected = self._expected_prediction(truth_label)
            support = sum(
                item["truth_class"] == truth_label for item in records
            )
            tp = sum(
                item["truth_class"] == truth_label
                and item["metric_prediction"] == expected
                for item in records
            )
            fp = sum(
                item["truth_class"] != truth_label
                and item["metric_prediction"] == expected
                for item in records
            )
            fn = support - tp
            fn_before = sum(
                item["truth_class"] == truth_label
                and not item["reached_statistical"]
                and truth_label != "normal"
                for item in records
            )
            fn_after = fn - fn_before if truth_label != "normal" else fn
            precision = self._ratio(tp, tp + fp)
            recall = self._ratio(tp, support)
            f1 = self._f_score(precision, recall, beta=1.0)
            f2 = self._f_score(precision, recall, beta=2.0)
            by_class[truth_label] = {
                "support_all": support,
                "tp": tp,
                "fp": fp,
                "fn_e2e": fn,
                "fn_before_classifier": fn_before,
                "fn_after_classifier": fn_after,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
            }
            if support > 0 and f1 is not None:
                f1_values.append(f1)
                weighted_f1_sum += f1 * support
            if support > 0 and recall is not None:
                recalls.append(recall)
        correct = sum(item["correct"] for item in records)
        abstained = sum(
            item["predicted_class"] == "indeterminate" for item in records
        )
        determined = sum(
            item["predicted_class"]
            not in {"indeterminate", "not_reached"}
            for item in records
        )
        return {
            "scope": scope,
            "support": support_total,
            "accuracy": self._ratio(correct, support_total),
            "macro_f1": (
                round(sum(f1_values) / len(f1_values), 6)
                if f1_values
                else None
            ),
            "weighted_f1": self._ratio(weighted_f1_sum, support_total),
            "balanced_accuracy": (
                round(sum(recalls) / len(recalls), 6) if recalls else None
            ),
            "abstention_rate": self._ratio(abstained, support_total),
            "determinate_coverage": self._ratio(determined, support_total),
            "fault_recall_e2e": by_class["sensor_fault"]["recall"],
            "fault_f2_e2e": by_class["sensor_fault"]["f2"],
            "fn_before_classifier": by_class["sensor_fault"][
                "fn_before_classifier"
            ],
            "fn_after_classifier": by_class["sensor_fault"][
                "fn_after_classifier"
            ],
            "by_class": by_class,
        }

    def _preclassifier_losses(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = []
        for truth in ["environmental_anomaly", "sensor_fault"]:
            support_all = sum(item["truth_class"] == truth for item in records)
            rejected_statistical = sum(
                item["truth_class"] == truth
                and not item["reached_statistical"]
                for item in records
            )
            passed_statistical = support_all - rejected_statistical
            dropped_after_statistical = sum(
                item["truth_class"] == truth
                and item["reached_statistical"]
                and not item["reached_classifier"]
                for item in records
            )
            rows.extend(
                [
                    {
                        "gate": "statistical",
                        "truth_class": truth,
                        "support_before_gate": support_all,
                        "rejected": rejected_statistical,
                        "false_negatives": rejected_statistical,
                        "reason": "below_threshold_or_no_statistical_candidate",
                    },
                    {
                        "gate": "contextual_or_classifier_delivery",
                        "truth_class": truth,
                        "support_before_gate": passed_statistical,
                        "rejected": dropped_after_statistical,
                        "false_negatives": dropped_after_statistical,
                        "reason": "candidate_did_not_reach_classifier",
                    },
                ]
            )
        return rows

    def _missed_case(self, item: dict[str, Any]) -> dict[str, Any]:
        if not item["reached_statistical"]:
            last_stage = "statistical"
            reason = "not_reached"
        elif not item["reached_contextual"]:
            last_stage = "contextual"
            reason = "pipeline_drop"
        elif not item["reached_classifier"]:
            last_stage = "classifier_delivery"
            reason = "pipeline_drop"
        else:
            last_stage = "classifier"
            reason = (
                "abstained"
                if item["predicted_class"] == "indeterminate"
                else "wrong_class"
            )
        return {
            "episode_id": item["episode_id"],
            "entity_id": item["entity_id"],
            "truth_class": item["truth_class"],
            "root_cause": item["root_cause"],
            "start": item["start"].isoformat(),
            "end": item["end"].isoformat(),
            "last_stage": last_stage,
            "reason": reason,
            "predicted_class": item["predicted_class"],
            "implicit_operational_prediction": (
                "normal"
                if item["predicted_class"] == "not_reached"
                else item["predicted_class"]
            ),
        }

    def _unavailable(
        self,
        dataset_id: str,
        partition: str,
        uncertain_excluded: int,
        boundaries: dict[str, Any],
        warning: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "available": False,
            "status": "not_evaluated",
            "dataset": {
                "id": dataset_id,
                "split": str(partition).lower(),
                "labeled_cases": 0,
                "uncertain_excluded": uncertain_excluded,
                "partition_boundaries": boundaries,
                "uses_simulated_dataset": False,
            },
            "metric_policy": {
                "gate_rejection_policy": "not_reached",
                "indeterminate_policy": "abstention_and_non_true_positive",
                "fixed_thresholds": True,
            },
            "warnings": [warning],
        }

    def _evaluation_hash(
        self,
        selected: pd.DataFrame,
        partition: str,
        dataset_id: str,
    ) -> str:
        columns = [
            column
            for column in ["entity_id", "_start", "_end", "_truth", "_root"]
            if column in selected
        ]
        hashed = pd.util.hash_pandas_object(selected[columns], index=False)
        digest = hashlib.sha256()
        digest.update(hashed.values.tobytes())
        digest.update(str(partition).lower().encode("utf-8"))
        digest.update(dataset_id.encode("utf-8"))
        digest.update(str(asdict(self.config)).encode("utf-8"))
        return digest.hexdigest()

    def _root_cause_series(self, frame: pd.DataFrame) -> pd.Series:
        result = pd.Series("", index=frame.index, dtype="object")
        for column in ["root_cause", "fault_type", "anomaly_type"]:
            if column in frame:
                values = frame[column].fillna("").astype(str)
                result = result.mask(result.eq("") & values.ne(""), values)
        return result.mask(result.eq(""), frame["label"].astype(str))

    def _truth_label(self, value: Any) -> str | None:
        key = str(value or "").strip().lower()
        aliases = {
            "normal": "normal",
            "environmental_anomaly": "environmental_anomaly",
            "environmental": "environmental_anomaly",
            "sensor_fault": "sensor_fault",
            "possible_sensor_fault": "sensor_fault",
            "possible_fault": "sensor_fault",
        }
        return aliases.get(key)

    def _prediction_label(self, value: Any) -> str:
        key = str(value or "").strip().lower()
        aliases = {
            "normal": "normal",
            "environmental_anomaly": "environmental_anomaly",
            "environmental": "environmental_anomaly",
            "possible_sensor_fault": "possible_sensor_fault",
            "possible_fault": "possible_sensor_fault",
            "sensor_fault": "possible_sensor_fault",
            "indeterminate": "indeterminate",
        }
        return aliases.get(key, "indeterminate")

    def _expected_prediction(self, truth_label: str) -> str:
        return (
            "possible_sensor_fault"
            if truth_label == "sensor_fault"
            else truth_label
        )

    def _ratio(self, numerator: float, denominator: float) -> float | None:
        if denominator <= 0:
            return None
        return round(float(numerator) / float(denominator), 6)

    def _f_score(
        self,
        precision: float | None,
        recall: float | None,
        beta: float,
    ) -> float | None:
        if precision is None or recall is None:
            return None
        beta_sq = beta * beta
        denominator = beta_sq * precision + recall
        if denominator <= 0:
            return 0.0
        return round(
            (1.0 + beta_sq) * precision * recall / denominator,
            6,
        )
