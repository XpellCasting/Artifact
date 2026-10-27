"""Replay an arbitrary graph against future observations, with separate ground truth."""
from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from threading import Lock, Thread
from time import perf_counter
from uuid import uuid4

import pandas as pd

from app.flow.runner import FlowRunner
from app.flow.definitions import _time
from app.flow.definitions import observation_valid
from app.services.reproduccion import MotorReproduccionValidacion
from app.modulos.clasificador.disponibilidad import CLASSIFIER_MODEL_REQUIRED

FLOW_PROGRESS_INTERVAL_SECONDS = 0.5


class FlowResultSpool:
    """Per-job JSONL spool so completed node outputs do not accumulate in RAM."""
    def __init__(self, job_id):
        fd, name = tempfile.mkstemp(prefix=f"flow-evaluation-{job_id}-", suffix=".jsonl")
        self.path = Path(name)
        self.file = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        self.sequences = Counter()
        self.unflushed_inputs = 0
        self.closed = False

    def write_block(self, results):
        for node_id, items in results.items():
            for item in items:
                record = {"node_id": node_id, "sequence": self.sequences[node_id],
                          "observation_id": str(item["observation_id"]), "payload": item}
                self.file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str))
                self.file.write("\n")
                self.sequences[node_id] += 1
        self.unflushed_inputs += len(results.get("input", []))
        if self.unflushed_inputs >= 500:
            self.file.flush()
            self.unflushed_inputs = 0

    def finish(self):
        if not self.closed:
            self.file.flush()
            self.file.close()
            self.closed = True

    def records(self):
        self.finish()
        with self.path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)

    def cleanup(self):
        self.finish()
        self.path.unlink(missing_ok=True)


def labelled_report(results, truth, *, truth_index=None):
    expected = truth_index if truth_index is not None else {int(row["raw_index"]): row for row in truth}
    groups = {}
    for item in results:
        label = expected.get(int(item["source_raw_index"]))
        if label is None:
            continue
        decision = item.get("final_decision")
        detected = decision in {"fault_compatible", "suspicion", "environmental"}
        unassessed = decision in {"indeterminate", "not_routed"}
        diagnoses = {entry.get("data", {}).get("fault_type") for entry in item.get("evidence", []) if entry.get("status") == "compatible"}
        correct_type = bool(label.get("fault_type") and decision == "fault_compatible" and diagnoses == {label["fault_type"]})
        outcome = "sin_etiqueta" if label["tipo"] == "NULL" else "falso_positivo" if label["tipo"] == "normal" and detected else "normal_correcto" if label["tipo"] == "normal" and decision in {"normal", "no_alert"} else "no_evaluable" if label["tipo"] == "normal" else "detectado" if detected else "no_detectado"
        if label["tipo"] in {"Falla", "anomalia"} and unassessed:
            outcome = "no_evaluable"
        # This annotation is attached only AFTER all filters finish.
        item["evaluation"] = {"expected": label["tipo"], "fault_type": label.get("fault_type"), "synthetic": label["synthetic"] == "true", "outcome": outcome, "type_identified": correct_type}
        key = (label["entity"], label["tipo"], label["synthetic"], label.get("fault_type"))
        group = groups.setdefault(key, {"entity_id": key[0], "expected": key[1], "synthetic": key[2] == "true", "fault_type": key[3], "total": 0, "detected": 0, "missed": 0, "missed_evaluable": 0, "non_evaluable": 0, "false_positives": 0, "type_identified": 0, "indeterminate": 0})
        group["total"] += 1
        group["detected"] += int(detected and label["tipo"] in {"Falla", "anomalia"})
        group["missed"] += int(not detected and label["tipo"] in {"Falla", "anomalia"})
        group["missed_evaluable"] += int(not detected and not unassessed and label["tipo"] in {"Falla", "anomalia"})
        group["non_evaluable"] += int(unassessed)
        group["false_positives"] += int(outcome == "falso_positivo")
        group["type_identified"] += int(correct_type)
        group["indeterminate"] += int(decision in {"indeterminate", "not_routed"})
    return {"version": "separate_non_evaluable_v2", "labelled": bool(truth), "groups": list(groups.values()), "matched_rows": sum(g["total"] for g in groups.values())}


def statistical_labelled_report(results, truth, node_id, *, truth_index=None):
    expected = truth_index if truth_index is not None else {int(row["raw_index"]): row for row in truth}
    groups = {}
    seen = set()
    for item in results:
        raw_index = item.get("source_raw_index")
        if raw_index is None or int(raw_index) in seen:
            continue
        seen.add(int(raw_index))
        label = expected.get(int(raw_index))
        if label is None:
            continue
        entry = next((e for e in reversed(item.get("evidence", [])) if e.get("node_id") == node_id and "statistical_observation" in e.get("data", {})), None)
        if entry is None:
            continue
        observation = entry["data"]["statistical_observation"]
        key = (label["entity"], label["tipo"], label["synthetic"], label.get("fault_type"))
        group = groups.setdefault(key, {"entity_id": key[0], "expected": key[1], "synthetic": key[2] == "true", "fault_type": key[3], "total": 0, "processed": 0, "evaluable": 0, "non_evaluable": 0, "detected": 0, "missed_evaluable": 0, "false_positives": 0, "false_positive_alerts": 0, "suppressed_false_positives": 0, "alerted_rows": 0, "suppressed_rows": 0})
        evaluable = bool(observation["evaluable"])
        detected = bool(observation["detected_before_suppression"])
        alerted = bool(observation["alert_emitted"])
        abnormal = label["tipo"] in {"Falla", "anomalia"}
        group["total"] += 1
        group["processed"] += 1
        group["evaluable"] += int(evaluable)
        group["non_evaluable"] += int(not evaluable)
        group["detected"] += int(abnormal and detected)
        group["missed_evaluable"] += int(abnormal and evaluable and not detected)
        group["false_positives"] += int(label["tipo"] == "normal" and detected)
        group["false_positive_alerts"] += int(label["tipo"] == "normal" and alerted)
        group["suppressed_false_positives"] += int(label["tipo"] == "normal" and detected and not alerted)
        group["alerted_rows"] += int(alerted)
        group["suppressed_rows"] += int(detected and not alerted)
    return {"labelled": bool(truth), "scope": "received_by_node", "groups": list(groups.values()), "matched_rows": sum(g["total"] for g in groups.values())}


def _merge_report_groups(target, report):
    descriptors = {"entity_id", "expected", "synthetic", "fault_type"}
    for item in report.get("groups", []):
        key = (item["entity_id"], item["expected"], item["synthetic"], item.get("fault_type"))
        current = target.get(key)
        if current is None:
            target[key] = dict(item)
            continue
        for field, value in item.items():
            if field not in descriptors:
                current[field] = current.get(field, 0) + value


class CausalFlowRunner(FlowRunner):
    def evaluate(self, flow, development, future, entities, context_map, baseline, truth=(), progress=None,
                 result_sink=None, batch_source=None, source_identity=None):
        total_started = perf_counter()
        compiled = self.compile(flow)
        if (future is None or future.empty) and batch_source is None:
            raise ValueError("Carga un conjunto de eventos futuros antes de evaluar el flujo.")
        if not future.empty and future["timestamp_dt"].min() < development["timestamp_dt"].max():
            raise ValueError("Los eventos de evaluación deben ser posteriores al histórico de Desarrollo.")
        self.execution_id = uuid4().hex
        self.context_states = {}
        self.diagnostic_sessions = {}
        if progress:
            progress(0, len(future), "preparation", "Preparando modelos y estados causales")
        performance = {"version": "flow-performance-v2", "phases_ms": {}, "counts": {"future_rows": len(future)}, "included_in": {}}
        phase_started = perf_counter()
        # Labels stay in reporting; reuse this run's lookup across groups and nodes.
        truth_index = {int(row["raw_index"]): row for row in truth}
        diagnostic_snapshots = {}
        if hasattr(self.diagnostic, "_run_snapshot"):
            for node in flow["nodes"]:
                if node["type"] == "classifier" and node.get("config", {}).get("enabled") is not False:
                    model_id = node.get("config", {}).get("model_id")
                    snapshot_fn = (self.diagnostic._run_snapshot_for_model
                                   if hasattr(self.diagnostic, "_run_snapshot_for_model") else self.diagnostic._run_snapshot)
                    args = ("flow_evaluation", f"{self.execution_id}:{node['id']}")
                    diagnostic_snapshots[node["id"]] = (snapshot_fn(*args, model_id=model_id)
                                                         if model_id and hasattr(self.diagnostic, "_run_snapshot_for_model")
                                                         else snapshot_fn(*args))
                    if not diagnostic_snapshots[node["id"]].get("model"):
                        raise ValueError(CLASSIFIER_MODEL_REQUIRED)
        statistical_nodes = [n for n in flow["nodes"] if n["type"] in {"statistical_focus"} and n.get("config", {}).get("enabled") is not False]
        motor = MotorReproduccionValidacion(self.statistical)
        # Models may be shared internally, but each node receives an independent runtime.
        prepared = {}
        for node in statistical_nodes:
            method = "residual_focus_v1"
            if method not in prepared:
                prepared[method] = baseline if baseline.get("continuous_method") == method else motor.construir_baseline(development, entities, continuous_method=method)
        def independent_runtime(source):
            copied = deepcopy(source)
            # sklearn estimators are immutable during prediction; retain their identity,
            # but never retain a history, accumulator, episode registry, or suppression state.
            for entity_id, entry in copied.get("continuous", {}).items():
                original = source.get("continuous", {}).get(entity_id, {})
                if "predictor" in original:
                    entry["predictor"] = original["predictor"]
            return copied
        self.baselines = {n["id"]: independent_runtime(prepared["residual_focus_v1"]) for n in statistical_nodes}
        self.statistical_results = {}
        performance["phases_ms"]["baseline_preparation"] = round((perf_counter() - phase_started) * 1000, 3)
        # Whitelist inputs: labels and expected outcomes never reach any detector.
        columns = [c for c in future.columns if c not in {"es_sintetica", "tipo", "tipo_falla", "fault_type", "label", "evaluation"}]
        ordered = future[columns].sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable").copy()
        identity_columns = [c for c in ("entity_id", "timestamp_dt", "state", "_raw_index") if c in ordered]
        source_id = source_identity or hashlib.sha256(pd.util.hash_pandas_object(ordered[identity_columns], index=False).values.tobytes()).hexdigest()
        contextual_nodes = [node for node in flow["nodes"]
                            if node["type"] == "contextual" and node.get("config", {}).get("enabled") is not False]
        active_classifier_ids = {node["id"] for node in flow["nodes"]
                                 if node["type"] == "classifier" and node.get("config", {}).get("enabled") is not False}
        config = getattr(self.contextual, "config", None)
        context_window = ((getattr(config, "window_seconds", 300) *
                           (getattr(config, "max_lag_windows", 0) + 2))
                          if contextual_nodes else 0)
        diagnostic_windows = []
        for snapshot in diagnostic_snapshots.values():
            model = snapshot.get("model") or {}
            model_config = model.get("config") or {}
            if model_config.get("architecture_version") == "gat_lstm_v3_segments_multiscale":
                diagnostic_windows.append(259200)
                continue
            lookback = model_config.get("lookback_seconds")
            if lookback is None:
                lookback = max(0, model_config.get("sequence_steps", 1) - 1) * model_config.get("grid_seconds", 0)
            diagnostic_windows.append(model.get("window_seconds") or
                                      lookback + model_config.get("feature_window_seconds", 0))
        diagnostic_window = max([3600 if active_classifier_ids else 0, *diagnostic_windows])
        horizon = max(context_window, diagnostic_window)

        def retain(frame, at):
            if not horizon:
                return frame
            cutoff = at - pd.Timedelta(seconds=horizon)
            older = frame[frame["timestamp_dt"] < cutoff].sort_values("timestamp_dt", kind="stable").groupby("entity_id", sort=False).tail(1)
            return pd.concat([older, frame[frame["timestamp_dt"] >= cutoff]], ignore_index=True)

        history_initialization_started = perf_counter()
        needs_history = bool(contextual_nodes or active_classifier_ids)
        if needs_history:
            available = development[[c for c in development.columns if c in columns]].copy()
            available["_raw_index"] = -1 - pd.RangeIndex(len(available))
            available = retain(available, ordered["timestamp_dt"].min() if not ordered.empty else development["timestamp_dt"].max())
        else:
            available = None
        if hasattr(self.diagnostic, "create_run_session"):
            for node in flow["nodes"]:
                if (available is not None and node["type"] == "classifier"
                        and node.get("config", {}).get("enabled") is not False):
                    model_id = node.get("config", {}).get("model_id")
                    session = (self.diagnostic.create_run_session("flow_evaluation", f"{self.execution_id}:{node['id']}", model_id=model_id)
                              if model_id else self.diagnostic.create_run_session("flow_evaluation", f"{self.execution_id}:{node['id']}"))
                    if ((session.snapshot.get("model") or {}).get("method") == "gat_lstm"
                            and session.gat_snapshot is None):
                        raise ValueError("No se pudo cargar el modelo del filtro clasificador. Comprueba sus archivos y dependencias, selecciona otro modelo o desactiva ese filtro.")
                    session.add_events(available)
                    self.diagnostic_sessions[node["id"]] = session
        if contextual_nodes:
            prepared_context = self.contextual.preparar_eventos_causales(entities, available)
            if hasattr(prepared_context, "bind_map"):
                prepared_context.bind_map(context_map)
            self.context_sessions = {node["id"]: prepared_context for node in contextual_nodes}
        else:
            prepared_context = None
            self.context_sessions = {}
        history_initialization_ms = (perf_counter() - history_initialization_started) * 1000
        needs_legacy_history_frame = bool(
            active_classifier_ids - set(self.diagnostic_sessions)
            or (contextual_nodes and prepared_context is None)
        )
        if not needs_legacy_history_frame:
            available = None
        outputs = defaultdict(list) if result_sink is None else None
        streamed_node_counts = Counter()
        streamed_decisions = Counter()
        streamed_report_groups = {}
        streamed_statistical_groups = {node_id: {} for node_id in self.baselines}
        timings = Counter()
        processed = 0
        sample_start = sample_end = None
        expected_total = len(ordered) if batch_source is None else 0
        groups_completed = 0
        classifier_inputs = Counter()
        checkpoints = []
        next_checkpoint = 5000
        history_maintenance_ms = 0.0
        contextual_index_update_ms = 0.0
        statistical_preparation_ms = 0.0
        graph_execution_ms = 0.0
        started = perf_counter()
        last_progress_at = None
        # Equal timestamps are simultaneous; no node can see a later timestamp.
        try:
            for _, batch in (ordered.groupby("timestamp_dt", sort=True) if batch_source is None else batch_source):
                if batch.empty:
                    continue
                sample_start = sample_start if sample_start is not None else batch["timestamp_dt"].min()
                sample_end = batch["timestamp_dt"].max()
                batch_started = perf_counter()
                if progress and (last_progress_at is None or
                                 batch_started - last_progress_at >= FLOW_PROGRESS_INTERVAL_SECONDS):
                    progress(processed, expected_total, "evaluation", "Procesando grupo temporal",
                             current_timestamp=_time(batch["timestamp_dt"].iloc[0]),
                             current_batch_rows=len(batch), groups_completed=groups_completed,
                             group_started_at=pd.Timestamp.now(tz="UTC").isoformat())
                    last_progress_at = batch_started
                if available is not None:
                    available = retain(pd.concat([available, batch], ignore_index=True), batch["timestamp_dt"].max())
                for session in self.diagnostic_sessions.values():
                    session.add_events(batch, events_sorted=True)
                if prepared_context is not None:
                    context_started = perf_counter()
                    advance = getattr(prepared_context, "advance_normalized", prepared_context.advance)
                    advance(
                        batch,
                        batch["timestamp_dt"].max() - pd.Timedelta(seconds=horizon),
                    )
                    contextual_index_update_ms += (perf_counter() - context_started) * 1000
                history_maintenance_ms += (perf_counter() - batch_started) * 1000
                statistical_started = perf_counter()
                for node in flow["nodes"]:
                    if node["id"] in self.baselines:
                        method = "residual_focus_v1"
                        self.statistical_results[node["id"]] = motor.analizar_lote(batch, self.baselines[node["id"]], node.get("config", {}).get("suspicion_threshold", .6), continuous_method=method)
                statistical_preparation_ms += (perf_counter() - statistical_started) * 1000
                graph_started = perf_counter()
                run_events = available if available is not None else batch
                summary, batch_outputs, _ = self.run(flow, run_events, entities, context_map, candidate_events=batch, source_id=source_id, compiled=compiled, events_sorted=True, include_snapshot=False)
                graph_execution_ms += (perf_counter() - graph_started) * 1000
                if result_sink is None:
                    for node_id, items in batch_outputs.items():
                        outputs[node_id].extend(items)
                else:
                    block_report = labelled_report(batch_outputs.get("output", []), truth, truth_index=truth_index)
                    _merge_report_groups(streamed_report_groups, block_report)
                    for node_id in streamed_statistical_groups:
                        block_stats = statistical_labelled_report(batch_outputs.get(node_id, []), truth, node_id, truth_index=truth_index)
                        _merge_report_groups(streamed_statistical_groups[node_id], block_stats)
                    result_sink(batch_outputs)
                    streamed_node_counts.update({key: len(value) for key, value in batch_outputs.items()})
                    streamed_decisions.update(item.get("final_decision") for item in batch_outputs.get("output", []))
                timings.update(summary["node_timings_ms"])
                for node in flow["nodes"]:
                    if node["type"] == "classifier":
                        classifier_inputs[node["id"]] += summary["node_counts"].get(node["id"], 0)
                processed += len(batch)
                groups_completed += 1
                completed_at = perf_counter()
                while processed >= next_checkpoint:
                    checkpoints.append({
                        "processed": processed,
                        "elapsed_ms": round((completed_at - started) * 1000, 3),
                        "groups_completed": groups_completed,
                        "classifier_inputs": dict(classifier_inputs),
                        "history_rows": (len(available) if available is not None else
                                         sum(len(frame) for frame in prepared_context.observations.values())
                                         if prepared_context is not None else
                                         max((session.stats.get("buffered_event_rows", 0)
                                              for session in self.diagnostic_sessions.values()), default=0)),
                        "diagnostic_buffers": {
                            node_id: session.stats.get("buffered_event_rows", 0)
                            for node_id, session in self.diagnostic_sessions.items()
                        },
                    })
                    next_checkpoint += 5000
                # Always publish the final count, even inside the half-second interval.
                if progress and (batch_source is not None or processed == len(ordered) or
                                 completed_at - last_progress_at >= FLOW_PROGRESS_INTERVAL_SECONDS):
                    progress(processed, expected_total, "evaluation", "Evaluando eventos entrantes" if batch_source is not None else "Reproduciendo grupos temporales",
                             current_timestamp=_time(batch["timestamp_dt"].iloc[0]),
                             current_batch_rows=len(batch), groups_completed=groups_completed,
                             last_batch_ms=round((completed_at - batch_started) * 1000, 3),
                             group_started_at=None)
                    last_progress_at = completed_at
            if progress and groups_completed == 0:
                progress(0, expected_total, "evaluation", "Reproduciendo grupos temporales")
        finally:
            for session in self.diagnostic_sessions.values():
                session.flush()
        report_started = perf_counter()
        if result_sink is None:
            report = labelled_report(outputs["output"], truth, truth_index=truth_index)
            statistical_evaluation = {node_id: statistical_labelled_report(outputs.get(node_id, []), truth, node_id, truth_index=truth_index) for node_id in self.baselines}
            result_count = len(outputs["output"])
            node_counts = {key: len(value) for key, value in outputs.items()}
            final_decisions = dict(Counter(item["final_decision"] for item in outputs["output"]))
        else:
            report = {"version": "separate_non_evaluable_v2", "labelled": bool(truth), "groups": list(streamed_report_groups.values()),
                      "matched_rows": sum(group["total"] for group in streamed_report_groups.values())}
            statistical_evaluation = {
                node_id: {"labelled": bool(truth), "scope": "received_by_node",
                          "groups": list(groups.values()),
                          "matched_rows": sum(group["total"] for group in groups.values())}
                for node_id, groups in streamed_statistical_groups.items()
            }
            result_count = streamed_node_counts.get("output", 0)
            node_counts = dict(streamed_node_counts)
            final_decisions = dict(streamed_decisions)
        report_finalization_ms = (perf_counter() - report_started) * 1000
        performance["phases_ms"]["causal_evaluation"] = round((perf_counter() - started) * 1000, 3)
        node_types = {node["id"]: node["type"] for node in flow["nodes"]}
        performance["phases_ms"].update({
            "statistical_execution": round(sum(value for node_id, value in timings.items() if node_types.get(node_id) in {"statistical_focus"}), 3),
            "contextual_execution": round(sum(value for node_id, value in timings.items() if node_types.get(node_id) == "contextual"), 3),
            "classifier_execution": round(sum(value for node_id, value in timings.items() if node_types.get(node_id) == "classifier"), 3),
            "history_maintenance": round(history_maintenance_ms, 3),
            "contextual_index_update": round(contextual_index_update_ms, 3),
            "history_initialization": round(history_initialization_ms, 3),
            "report_finalization": round(report_finalization_ms, 3),
            "statistical_preparation": round(statistical_preparation_ms, 3),
            "graph_execution": round(graph_execution_ms, 3),
            "classifier_sample_construction": round(sum(session.stats.get("gat_sample_ms", 0.0) for session in self.diagnostic_sessions.values()), 3),
            "classifier_model_inference": round(sum(session.stats.get("gat_inference_ms", 0.0) for session in self.diagnostic_sessions.values()), 3),
            "classifier_sample_cache_hits": sum(session.stats.get("gat_sample_cache_hits", 0) for session in self.diagnostic_sessions.values()),
            "classifier_inference_cache_hits": sum(session.stats.get("gat_inference_cache_hits", 0) for session in self.diagnostic_sessions.values()),
            "diagnostic_writes": round(sum(session.stats["write_ms"] for session in self.diagnostic_sessions.values()), 3),
        })
        performance["phases_ms"]["total"] = round((perf_counter() - total_started) * 1000, 3)
        performance["counts"].update(future_rows=processed, groups=groups_completed, statistical_nodes=len(statistical_nodes),
                                      classifier_inputs=dict(classifier_inputs), checkpoints=checkpoints)
        performance["included_in"] = {
            "baseline_preparation": "total", "causal_evaluation": "total",
            "history_initialization": "total",
            "report_finalization": "causal_evaluation",
            "node_timings_ms": "graph_execution", "history_maintenance": "causal_evaluation",
            "contextual_index_update": "causal_evaluation",
            "statistical_preparation": "causal_evaluation", "graph_execution": "causal_evaluation",
            "classifier_sample_construction": "classifier_execution and causal_evaluation",
            "classifier_model_inference": "classifier_execution and causal_evaluation",
            "diagnostic_writes": "classifier_execution and causal_evaluation",
            "classifier_execution": "graph_execution; includes diagnostic inference and writes",
        }
        summary = {"candidate_count": processed, "output_count": result_count, "node_counts": node_counts, "node_timings_ms": dict(timings), "elapsed_ms": round((perf_counter() - started) * 1000, 2), "by_final_decision": final_decisions, "evaluation": report, "statistical_evaluation": statistical_evaluation, "statistical_evaluation_version": "pre_suppression_v1", "performance": performance}
        provenance = deepcopy(baseline.get("provenance"))
        continuous_models = {node_id: deepcopy(value.get("continuous_model_manifest")) for node_id, value in self.baselines.items()}
        identity_payload = {"base": (provenance or {}).get("baseline_id"), "continuous_methods": {node_id: value.get("continuous_method") for node_id, value in self.baselines.items()}, "continuous_models": continuous_models}
        run_baseline_id = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        performance["counts"]["diagnostic_sessions"] = {node_id: session.stats for node_id, session in self.diagnostic_sessions.items()}
        snapshot = {"flow": deepcopy(flow), "execution_mode": "causal_live" if batch_source is not None else "causal_future", "statistical_evaluation_version": "pre_suppression_v1", "source": {"rows": processed, "fingerprint": source_id, "sample_start": _time(sample_start) if sample_start is not None else None, "sample_end": _time(sample_end) if sample_end is not None else None}, "development_end": _time(development["timestamp_dt"].max()), "context_map_version": context_map.get("map_version"), "baseline_version": baseline.get("version"), "baseline_id": run_baseline_id, "baseline_provenance": {**(provenance or {}), "continuous_models": continuous_models}, "diagnostic_snapshots": diagnostic_snapshots}
        return summary, dict(outputs) if outputs is not None else None, snapshot

    def _statistical(self, node_id, incoming, events, entities, config):
        statistical = self.statistical_results[node_id]
        detections = statistical.get("detections", [])
        alerts = statistical.get("results", [])
        detected_by_raw = {str(r["source_raw_index"]): r for r in detections if r.get("source_raw_index") is not None}
        alerted_by_raw = {str(r["source_raw_index"]): r for r in alerts if r.get("source_raw_index") is not None}
        observations_by_raw = self.baselines[node_id].get("last_observations", {})
        detected_by_point, alerted_by_point = defaultdict(list), defaultdict(list)
        for collection, index in ((detections, detected_by_point), (alerts, alerted_by_point)):
            for row in collection:
                if row.get("source_raw_index") is None:
                    index[(str(row.get("entity_id")), _time(row.get("timestamp")))].append(row)
        points = Counter((item["entity_id"], item["timestamp"]) for item in incoming)
        output = []
        for item in incoming:
            point = (item["entity_id"], item["timestamp"])
            detection = detected_by_raw.get(str(item["source_raw_index"]))
            alert = alerted_by_raw.get(str(item["source_raw_index"]))
            if points[point] == 1:
                if detection is None and len(detected_by_point[point]) == 1:
                    detection = detected_by_point[point][0]
                if alert is None and len(alerted_by_point[point]) == 1:
                    alert = alerted_by_point[point][0]
            eligible = item["entity_id"] in self.baselines[node_id].get(item.get("sensor_kind"), {})
            valid = observation_valid(item)
            reason = "invalid_state" if not valid else "missing_baseline" if not eligible else None
            calibration = self.baselines[node_id].get("continuous_model_manifest", {}).get(item["entity_id"], {})
            if reason == "missing_baseline" and calibration.get("reason"):
                reason = calibration["reason"]
            evaluable = reason is None
            if detection is not None and detection.get("suspected_pattern") == "unavailable":
                evaluable, reason = True, None
            detected = detection is not None
            emitted = alert is not None
            temporal_observation = observations_by_raw.get(str(item.get("source_raw_index")), {})
            if temporal_observation:
                evaluable = bool(temporal_observation.get("evaluable"))
                reason = temporal_observation.get("reason")
                detected = bool(temporal_observation.get("detected_before_suppression"))
            observation = {"evaluable": evaluable, "reason": reason, "detected_before_suppression": detected if evaluable else False, "alert_emitted": emitted if evaluable else False, "suppressed": bool(evaluable and detected and not emitted), **{key: value for key, value in temporal_observation.items() if key not in {"evaluable", "reason", "detected_before_suppression"}}}
            policy = config.get("signal_policy", "detection_before_suppression")
            signal = "positive" if (detected if policy == "detection_before_suppression" else emitted) else "negative" if evaluable else "unknown"
            status = "suspicion" if emitted else "detected_suppressed" if detected and evaluable else "no_suspicion" if evaluable else "not_evaluable"
            data = deepcopy(alert or detection or {})
            if temporal_observation:
                data.update({key: value for key, value in temporal_observation.items() if key not in {"evaluable", "reason", "detected_before_suppression"}})
            data["statistical_observation"] = observation
            if calibration and not evaluable:
                data["calibration"] = deepcopy(calibration)
            output.append(self._with_evidence(item, node_id, signal, status, data))
        return output


class FlowEvaluationJobs:
    """Run outside HTTP requests; completed results persist in FlowStore."""
    def __init__(self, store):
        self.store = store
        self.jobs = {}
        self.sources = {}
        self.lock = Lock()

    def start(self, flow_id, evaluate, live_source=None, flow=None):
        with self.lock:
            if any(job["status"] == "running" for job in self.jobs.values()):
                raise ValueError("Ya hay una evaluación de flujos en curso. Espera a que termine.")
            job_id = uuid4().hex
            self.store.begin_alert_execution(job_id)
            self.jobs[job_id] = {"job_id": job_id, "flow_id": flow_id, "flow": deepcopy(flow), "status": "running", "processed": 0, "total": 0, "phase": "preparation", "mode": "home_assistant" if live_source is not None else "csv"}
            if live_source is not None:
                self.sources[job_id] = live_source
        recent_completions = deque()
        last_completed = {"processed": 0, "at": None, "wall": None}
        job_started = perf_counter()
        def progress(processed, total, phase="evaluation", detail=None, **metadata):
            now = perf_counter()
            wall_now = pd.Timestamp.now(tz="UTC").isoformat()
            if processed > last_completed["processed"]:
                recent_completions.append((now, int(processed)))
                last_completed.update(processed=int(processed), at=now, wall=wall_now)
            cutoff = now - 30.0
            while len(recent_completions) > 1 and recent_completions[1][0] <= cutoff:
                recent_completions.popleft()
            rate = None
            if len(recent_completions) > 1:
                elapsed = recent_completions[-1][0] - recent_completions[0][0]
                if elapsed > 0:
                    rate = round((recent_completions[-1][1] - recent_completions[0][1]) / elapsed, 2)
            if last_completed["at"] is None or now - last_completed["at"] > 30.0:
                rate = None
            with self.lock:
                self.jobs[job_id].update(processed=processed, total=total, phase=phase, detail=detail,
                                         updated_at=wall_now,
                                         elapsed_seconds=round(now - job_started, 1),
                                         rate_window_seconds=30,
                                         recent_events_per_second=rate,
                                         last_completed_at=last_completed["wall"],
                                         **metadata)
        spool = FlowResultSpool(job_id)
        def sink(results):
            spool.write_block(results)
            self.store.record_alerts(job_id, flow_id, results.get("output", []))
            if live_source is not None:
                with self.lock:
                    job = self.jobs[job_id]
                    job["recent_results"] = (job.get("recent_results", []) + results.get("output", []))[-100:]
        def work():
            try:
                summary, _, snapshot = evaluate(progress, sink)
                progress(summary.get("candidate_count", 0), summary.get("candidate_count", 0),
                         "saving", "Guardando resultados por nodo")
                spool.finish()
                run_id, persistence_ms = self.store.save_run_stream(flow_id, snapshot, summary, spool.records(), execution_id=job_id)
                with self.lock:
                    self.jobs[job_id].update(status="completed", persistence_ms=persistence_ms,
                                             warning=snapshot.get("source", {}).get("connection_error"),
                                             run={"run_id": run_id, "summary": summary, "snapshot": snapshot})
            except Exception as exc:
                logging.getLogger(__name__).exception("Falló la evaluación del flujo %s", flow_id)
                with self.lock:
                    self.jobs[job_id].update(status="failed", error=str(exc))
            finally:
                if live_source is not None:
                    live_source.stop()
                    with self.lock:
                        connection = live_source.status()
                        if connection["status"] == "stopping":
                            connection["status"] = "stopped"
                        self.jobs[job_id]["connection"] = connection
                        self.sources.pop(job_id, None)
                spool.cleanup()
        Thread(target=work, daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    def get(self, job_id):
        with self.lock:
            return self._snapshot(job_id)

    def latest(self, include_run=True):
        with self.lock:
            job_id = next(reversed(self.jobs), None)
            return self._snapshot(job_id, include_run=include_run) if job_id is not None else None

    def _snapshot(self, job_id, include_run=True):
        job = self.jobs[job_id]
        payload = deepcopy({key: value for key, value in job.items() if include_run or key != "run"})
        if not include_run and "run" in job:
            payload["run_id"] = job["run"]["run_id"]
        source = self.sources.get(job_id)
        if source is not None:
            payload["connection"] = source.status()
        return payload

    def stop(self, job_id):
        with self.lock:
            source = self.sources.get(job_id)
            if source is None:
                if job_id not in self.jobs:
                    raise KeyError(job_id)
            else:
                self.jobs[job_id]["phase"] = "stopping"
                source.stop()
        return self.get(job_id)
