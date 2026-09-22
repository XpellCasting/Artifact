"""Replay an arbitrary graph against future observations, with separate ground truth."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from threading import Lock, Thread
from time import perf_counter
from uuid import uuid4

import pandas as pd

from .Flujos import FlowRunner, _time, observation_valid, validate_flow
from .Reproduccion_Validacion import MotorReproduccionValidacion


def labelled_report(results, truth):
    expected = {int(row["raw_index"]): row for row in truth}
    groups = {}
    for item in results:
        label = expected.get(int(item["source_raw_index"]))
        if label is None:
            continue
        decision = item.get("final_decision")
        detected = decision in {"fault_compatible", "suspicion", "environmental"}
        diagnoses = {entry.get("data", {}).get("fault_type") for entry in item.get("evidence", []) if entry.get("status") == "compatible"}
        correct_type = bool(label.get("fault_type") and decision == "fault_compatible" and diagnoses == {label["fault_type"]})
        outcome = "sin_etiqueta" if label["tipo"] == "NULL" else "falso_positivo" if label["tipo"] == "normal" and detected else "normal_correcto" if label["tipo"] == "normal" and decision in {"normal", "no_alert"} else "no_evaluable" if label["tipo"] == "normal" else "detectado" if detected else "no_detectado"
        # This annotation is attached only AFTER all filters finish.
        item["evaluation"] = {"expected": label["tipo"], "fault_type": label.get("fault_type"), "synthetic": label["synthetic"] == "true", "outcome": outcome, "type_identified": correct_type}
        key = (label["entity"], label["tipo"], label["synthetic"], label.get("fault_type"))
        group = groups.setdefault(key, {"entity_id": key[0], "expected": key[1], "synthetic": key[2] == "true", "fault_type": key[3], "total": 0, "detected": 0, "missed": 0, "false_positives": 0, "type_identified": 0, "indeterminate": 0})
        group["total"] += 1
        group["detected"] += int(detected and label["tipo"] in {"Falla", "anomalia"})
        group["missed"] += int(not detected and label["tipo"] in {"Falla", "anomalia"})
        group["false_positives"] += int(outcome == "falso_positivo")
        group["type_identified"] += int(correct_type)
        group["indeterminate"] += int(decision in {"indeterminate", "not_routed"})
    return {"labelled": bool(truth), "groups": list(groups.values()), "matched_rows": sum(g["total"] for g in groups.values())}


def statistical_labelled_report(results, truth, node_id):
    expected = {int(row["raw_index"]): row for row in truth}
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
        group = groups.setdefault(key, {"entity_id": key[0], "expected": key[1], "synthetic": key[2] == "true", "fault_type": key[3], "total": 0, "processed": 0, "evaluable": 0, "non_evaluable": 0, "detected": 0, "missed_evaluable": 0, "false_positives": 0, "alerted_rows": 0, "suppressed_rows": 0})
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
        group["alerted_rows"] += int(alerted)
        group["suppressed_rows"] += int(detected and not alerted)
    return {"labelled": bool(truth), "scope": "received_by_node", "groups": list(groups.values()), "matched_rows": sum(g["total"] for g in groups.values())}


class CausalFlowRunner(FlowRunner):
    def evaluate(self, flow, development, future, entities, context_map, baseline, truth=(), progress=None):
        total_started = perf_counter()
        compiled = self.compile(flow)
        if future is None or future.empty:
            raise ValueError("Carga un conjunto de eventos futuros antes de evaluar el flujo.")
        if future["timestamp_dt"].min() < development["timestamp_dt"].max():
            raise ValueError("Los eventos de evaluación deben ser posteriores al histórico de Desarrollo.")
        self.execution_id = uuid4().hex
        self.context_states = {}
        self.diagnostic_sessions = {}
        if progress:
            progress(0, len(future), "preparation", "Preparando modelos y estados causales")
        performance = {"version": "flow-performance-v1", "phases_ms": {}, "counts": {"future_rows": len(future)}, "included_in": {}}
        phase_started = perf_counter()
        diagnostic_snapshots = {}
        if hasattr(self.diagnostic, "_run_snapshot"):
            for node in flow["nodes"]:
                if node["type"] == "classifier" and node.get("config", {}).get("enabled") is not False:
                    diagnostic_snapshots[node["id"]] = self.diagnostic._run_snapshot("flow_evaluation", f"{self.execution_id}:{node['id']}")
        statistical_nodes = [n for n in flow["nodes"] if n["type"] in {"statistical", "statistical_focus"} and n.get("config", {}).get("enabled") is not False]
        motor = MotorReproduccionValidacion(self.statistical)
        # Models may be shared internally, but each node receives an independent runtime.
        prepared = {}
        for node in statistical_nodes:
            method = "residual_focus_v1" if node["type"] == "statistical_focus" else node.get("config", {}).get("continuous_method", "temporal_quantile_v1")
            if method not in prepared:
                prepared[method] = baseline if method == "temporal_quantile_v1" and baseline.get("continuous_method") == method else motor.construir_baseline(development, entities, continuous_method=method)
        # EWMA and FOCuS use exactly the same immutable central predictor.
        if "residual_ewma_v1" in prepared and "residual_focus_v1" in prepared:
            for entity_id, focus_entry in prepared["residual_focus_v1"].get("continuous", {}).items():
                ewma_entry = prepared["residual_ewma_v1"].get("continuous", {}).get(entity_id)
                if ewma_entry and focus_entry.get("eligible") and ewma_entry.get("eligible"):
                    focus_entry["predictor"] = ewma_entry["predictor"]
        def independent_runtime(source):
            copied = deepcopy(source)
            # sklearn estimators are immutable during prediction; retain their identity,
            # but never retain a history, accumulator, episode registry, or suppression state.
            for entity_id, entry in copied.get("continuous", {}).items():
                original = source.get("continuous", {}).get(entity_id, {})
                if "predictor" in original:
                    entry["predictor"] = original["predictor"]
            return copied
        self.baselines = {n["id"]: independent_runtime(prepared["residual_focus_v1" if n["type"] == "statistical_focus" else n.get("config", {}).get("continuous_method", "temporal_quantile_v1")]) for n in statistical_nodes}
        self.statistical_results = {}
        performance["phases_ms"]["baseline_preparation"] = round((perf_counter() - phase_started) * 1000, 3)
        # Whitelist inputs: labels and expected outcomes never reach any detector.
        columns = [c for c in future.columns if c not in {"es_sintetica", "tipo", "tipo_falla", "fault_type", "label", "evaluation"}]
        ordered = future[columns].sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable").copy()
        identity_columns = [c for c in ("entity_id", "timestamp_dt", "state", "_raw_index") if c in ordered]
        source_id = hashlib.sha256(pd.util.hash_pandas_object(ordered[identity_columns], index=False).values.tobytes()).hexdigest()
        available = development[[c for c in development.columns if c in columns]].copy()
        available["_raw_index"] = -1 - pd.RangeIndex(len(available))
        if hasattr(self.diagnostic, "create_run_session"):
            for node in flow["nodes"]:
                if node["type"] == "classifier" and node.get("config", {}).get("enabled") is not False:
                    session = self.diagnostic.create_run_session("flow_evaluation", f"{self.execution_id}:{node['id']}")
                    session.add_events(available)
                    self.diagnostic_sessions[node["id"]] = session
        config = getattr(self.contextual, "config", None)
        context_window = getattr(config, "window_seconds", 300) * (getattr(config, "max_lag_windows", 0) + 2)
        diagnostic_window = max([3600, *[(s.get("model") or {}).get("window_seconds", 3600) for s in diagnostic_snapshots.values()]])
        horizon = max(context_window, diagnostic_window)
        def retain(frame, at):
            cutoff = at - pd.Timedelta(seconds=horizon)
            older = frame[frame["timestamp_dt"] < cutoff].sort_values("timestamp_dt", kind="stable").groupby("entity_id", sort=False).tail(1)
            return pd.concat([older, frame[frame["timestamp_dt"] >= cutoff]], ignore_index=True)
        available = retain(available, ordered["timestamp_dt"].min())
        outputs = defaultdict(list)
        timings = Counter()
        processed = 0
        started = perf_counter()
        if progress:
            progress(0, len(ordered), "evaluation", "Reproduciendo grupos temporales")
        # Equal timestamps are simultaneous; no node can see a later timestamp.
        try:
            for _, batch in ordered.groupby("timestamp_dt", sort=True):
                available = retain(pd.concat([available, batch], ignore_index=True), batch["timestamp_dt"].max())
                for session in self.diagnostic_sessions.values():
                    session.add_events(batch)
                for node in flow["nodes"]:
                    if node["id"] in self.baselines:
                        method = "residual_focus_v1" if node["type"] == "statistical_focus" else node.get("config", {}).get("continuous_method", "temporal_quantile_v1")
                        self.statistical_results[node["id"]] = motor.analizar_lote(batch, self.baselines[node["id"]], node.get("config", {}).get("suspicion_threshold", .6), continuous_method=method)
                summary, batch_outputs, _ = self.run(flow, available, entities, context_map, candidate_events=batch, source_id=source_id, compiled=compiled, events_sorted=True, include_snapshot=False)
                for node_id, items in batch_outputs.items():
                    outputs[node_id].extend(items)
                timings.update(summary["node_timings_ms"])
                processed += len(batch)
                if progress:
                    progress(processed, len(ordered), "evaluation", "Reproduciendo grupos temporales")
        finally:
            for session in self.diagnostic_sessions.values():
                session.flush()
        report = labelled_report(outputs["output"], truth)
        statistical_evaluation = {node_id: statistical_labelled_report(outputs.get(node_id, []), truth, node_id) for node_id in self.baselines}
        performance["phases_ms"]["causal_evaluation"] = round((perf_counter() - started) * 1000, 3)
        node_types = {node["id"]: node["type"] for node in flow["nodes"]}
        performance["phases_ms"].update({
            "statistical_execution": round(sum(value for node_id, value in timings.items() if node_types.get(node_id) in {"statistical", "statistical_focus"}), 3),
            "contextual_execution": round(sum(value for node_id, value in timings.items() if node_types.get(node_id) == "contextual"), 3),
            "classifier_execution": round(sum(value for node_id, value in timings.items() if node_types.get(node_id) == "classifier"), 3),
            "diagnostic_writes": round(sum(session.stats["write_ms"] for session in self.diagnostic_sessions.values()), 3),
        })
        performance["phases_ms"]["total"] = round((perf_counter() - total_started) * 1000, 3)
        performance["counts"].update(groups=int(ordered["timestamp_dt"].nunique()), statistical_nodes=len(statistical_nodes))
        performance["included_in"] = {"baseline_preparation": "total", "causal_evaluation": "total", "node_timings_ms": "causal_evaluation"}
        summary = {"candidate_count": len(ordered), "output_count": len(outputs["output"]), "node_counts": {key: len(value) for key, value in outputs.items()}, "node_timings_ms": dict(timings), "elapsed_ms": round((perf_counter() - started) * 1000, 2), "by_final_decision": dict(Counter(item["final_decision"] for item in outputs["output"])), "evaluation": report, "statistical_evaluation": statistical_evaluation, "statistical_evaluation_version": "pre_suppression_v1", "performance": performance}
        provenance = deepcopy(baseline.get("provenance"))
        continuous_models = {node_id: deepcopy(value.get("continuous_model_manifest")) for node_id, value in self.baselines.items()}
        identity_payload = {"base": (provenance or {}).get("baseline_id"), "continuous_methods": {node_id: value.get("continuous_method") for node_id, value in self.baselines.items()}, "continuous_models": continuous_models}
        run_baseline_id = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        performance["counts"]["diagnostic_sessions"] = {node_id: session.stats for node_id, session in self.diagnostic_sessions.items()}
        snapshot = {"flow": deepcopy(flow), "execution_mode": "causal_future", "statistical_evaluation_version": "pre_suppression_v1", "source": {"rows": len(ordered), "fingerprint": source_id, "sample_start": _time(ordered["timestamp_dt"].min()), "sample_end": _time(ordered["timestamp_dt"].max())}, "development_end": _time(development["timestamp_dt"].max()), "context_map_version": context_map.get("map_version"), "baseline_version": baseline.get("version"), "baseline_id": run_baseline_id, "baseline_provenance": {**(provenance or {}), "continuous_models": continuous_models}, "diagnostic_snapshots": diagnostic_snapshots}
        return summary, dict(outputs), snapshot

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
            evaluable = reason is None
            detected = detection is not None
            emitted = alert is not None
            temporal_observation = observations_by_raw.get(str(item.get("source_raw_index")), {})
            if temporal_observation:
                evaluable = bool(temporal_observation.get("evaluable"))
                reason = temporal_observation.get("reason")
                detected = bool(temporal_observation.get("detected_before_suppression"))
            observation = {"evaluable": evaluable, "reason": reason, "detected_before_suppression": detected if evaluable else False, "alert_emitted": emitted if evaluable else False, "suppressed": bool(evaluable and detected and not emitted), **{key: value for key, value in temporal_observation.items() if key not in {"evaluable", "reason", "detected_before_suppression"}}}
            policy = config.get("signal_policy", "legacy" if observation.get("continuous_method") == "temporal_quantile_v1" else "detection_before_suppression")
            signal = "positive" if (detected if policy == "detection_before_suppression" else emitted) else "negative" if evaluable else "unknown"
            status = "suspicion" if emitted else "detected_suppressed" if detected and evaluable else "no_suspicion" if evaluable else "not_evaluable"
            data = deepcopy(alert or detection or {})
            if temporal_observation:
                data.update({key: value for key, value in temporal_observation.items() if key not in {"evaluable", "reason", "detected_before_suppression"}})
            data["statistical_observation"] = observation
            output.append(self._with_evidence(item, node_id, signal, status, data))
        return output


class FlowEvaluationJobs:
    """Run outside HTTP requests; completed results persist in FlowStore."""
    def __init__(self, store):
        self.store = store
        self.jobs = {}
        self.lock = Lock()

    def start(self, flow_id, evaluate):
        with self.lock:
            if any(job["status"] == "running" for job in self.jobs.values()):
                raise ValueError("Ya hay una evaluación de flujos en curso. Espera a que termine.")
            job_id = uuid4().hex
            self.jobs[job_id] = {"job_id": job_id, "flow_id": flow_id, "status": "running", "processed": 0, "total": 0, "phase": "preparation"}
        def progress(processed, total, phase="evaluation", detail=None):
            with self.lock:
                self.jobs[job_id].update(processed=processed, total=total, phase=phase, detail=detail)
        def work():
            try:
                summary, results, snapshot = evaluate(progress)
                progress(len(results.get("output", [])), len(results.get("output", [])), "saving", "Guardando resultados por nodo")
                run_id = self.store.save_run(flow_id, snapshot, summary, results)
                with self.lock:
                    self.jobs[job_id].update(status="completed", run={"run_id": run_id, "summary": summary, "snapshot": snapshot})
            except Exception as exc:
                with self.lock:
                    self.jobs[job_id].update(status="failed", error=str(exc))
        Thread(target=work, daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    def get(self, job_id):
        with self.lock:
            return deepcopy(self.jobs[job_id])
