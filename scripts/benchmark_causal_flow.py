"""Isolated causal-flow benchmark; reads repo fixtures and reference SQLite read-only."""
from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import tempfile
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.historial.cargar_historial import CargadorHistorial
from app.services.diagnostico import DiagnosticWorkbench
from app.flow.evaluacion import CausalFlowRunner
from app.flow.evaluacion import FlowResultSpool
from app.flow.result_spool import archive_for_run, ResultArchive, iter_run_results
from app.modulos.contextual.contextual import FiltroContextualEspacioTemporal, ConfiguracionContextual
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.database.flows import FlowStore
from app.services.reproduccion import MotorReproduccionValidacion


ROOT = Path(__file__).resolve().parents[1]
RUN_IDENTITY_KEYS = {"run_id", "diagnosis_id", "execution_id"}


class _WorkingSet:
    class _Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    def __init__(self):
        self.value = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    @classmethod
    def current(cls):
        if sys.platform != "win32":
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
            raise RuntimeError("No se pudo medir RSS en este sistema")
        counters = cls._Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        get_info = psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(cls._Counters), ctypes.c_ulong]
        get_info.restype = ctypes.c_int
        if not get_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)

    def _sample(self):
        while not self.stop.is_set():
            self.value = max(self.value, self.current())
            self.stop.wait(.1)

    def __enter__(self):
        self.value = self.current()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.value = max(self.value, self.current())


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_connection(path):
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _clean(value):
    """Normalize operational IDs, validating every retained evidence reference."""
    identifiers, executions = {}, set()
    for entry in value.get("evidence", []):
        identifier = entry.get("evidence_id")
        if identifier:
            canonical = f"{entry.get('observation_id')}:{entry['node_id']}"
            if identifier in identifiers:
                raise AssertionError(f"Evidencia duplicada: {identifier}")
            identifiers[identifier] = canonical
        if entry.get("execution_id"):
            executions.add(entry["execution_id"])

    def normalize(item, field=None):
        if isinstance(item, dict):
            return {key: normalize(content, key) for key, content in item.items()
                    if key not in RUN_IDENTITY_KEYS}
        if isinstance(item, list):
            return [normalize(content, field) for content in item]
        if field in {"evidence_id", "used_evidence_ids", "unused_evidence_ids"}:
            if item not in identifiers:
                raise AssertionError(f"Referencia de evidencia ausente: {item}")
            return identifiers[item]
        if isinstance(item, str):
            for execution in executions:
                item = item.replace(execution, "<execution>")
        return item
    return normalize(value)


def chronological_prefix(future, limit):
    ordered = future.eventos.sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable")
    if limit is None or limit >= len(ordered):
        return ordered.copy()
    if limit < 1:
        raise ValueError("El prefijo debe contener al menos una lectura")
    boundary = ordered["timestamp_dt"].iloc[limit - 1]
    return ordered.iloc[:ordered["timestamp_dt"].searchsorted(boundary, side="right")].copy()


def _assert_equal(actual, expected, path="$"):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: keys differ: {actual.keys() ^ expected.keys()}")
        for key in actual:
            _assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: lengths differ: {len(actual)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_equal(left, right, f"{path}[{index}]")
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if abs(float(actual) - float(expected)) > 1e-8 + 1e-6 * abs(float(expected)):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def _load_inputs(root):
    loader = CargadorHistorial()
    development = loader.cargar_real(root / "data/history/cargados/history.csv")
    future = loader.cargar_real(root / "data/history/cargados/historico_validacion.csv")
    context_map = _read_json(root / "storage/cache/contextual_map.json")
    return development, future, context_map


def _load_truth(root, reference_data=None):
    if reference_data is not None:
        return reference_data["truth"]
    with _reference_connection(root / "storage/database/validation_replay_results.sqlite3") as db:
        return [dict(zip(("raw_index", "entity", "tipo", "synthetic", "fault_type"), row))
                for row in db.execute("SELECT raw_index,entity,tipo,synthetic,fault_type "
                                      "FROM evaluation_truth ORDER BY replay_index")]


def _load_reference_flow(root, run_id, reference_data=None):
    if reference_data is not None:
        row = reference_data["runs"][run_id]
        return row["flow_id"], row["snapshot"]["flow"], row["summary"], row["snapshot"]
    with _reference_connection(root / "storage/database/flow_lab.sqlite3") as db:
        row = db.execute("SELECT flow_id,snapshot,summary FROM flow_runs WHERE run_id=?",
                         (run_id,)).fetchone()
        if row is None:
            raise RuntimeError("No se encontró la ejecución de referencia local.")
        flow = json.loads(row[1])["flow"]
        summary = json.loads(row[2])
        snapshot = json.loads(row[1])
        archive = archive_for_run(db, root / "storage/database/flow_lab.sqlite3", run_id)
        if archive is not None and "diagnostic_snapshots" in snapshot:
            snapshot["diagnostic_snapshots"] = ResultArchive(archive).diagnostic_snapshots(snapshot["diagnostic_snapshots"])
        return row[0], flow, summary, snapshot


def _isolated_workbench(root, temporary_root, model_ids):
    source = root / "models/gat_lstm"
    target = temporary_root / "review/diagnostics"
    for folder in ("models", "weights", "preprocessors", "graphs"):
        (target / folder).mkdir(parents=True, exist_ok=True)
        for model_id in model_ids:
            name = f"{model_id}.{'pt' if folder == 'weights' else 'json'}"
            shutil.copy2(source / folder / name, target / folder / name)
    (target / "active.json").write_text(json.dumps({"model_id": sorted(model_ids)[0]}), encoding="utf-8")
    return DiagnosticWorkbench(temporary_root / "review")


def _verify_source(future, expected_fingerprint):
    ordered = future.eventos.sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable")
    identity = ordered[[column for column in ("entity_id", "timestamp_dt", "state", "_raw_index")
                        if column in ordered]]
    actual = hashlib.sha256(pd.util.hash_pandas_object(identity, index=False).values.tobytes()).hexdigest()
    if actual != expected_fingerprint:
        raise RuntimeError(f"El archivo cargado no coincide con la referencia: {actual}")
    return actual


def _prepared_baseline(root, development, entities):
    # Only the application's internal cache is accepted, never an uploaded pickle.
    # Its general key also covers classifier code. For this benchmark validate
    # the narrower statistical provenance, which must be unchanged by the Proxy.
    from app.services.cache_causal import load
    from app.services.procedencia_normalidad import build_identity, effective_parameters, sha256_canonical
    path = root / "storage/cache/causal_preparation.pkl"
    with path.open("rb") as source:
        header = json.loads(source.readline(4096))
    cached = load(path, header["key"])
    baseline = (cached or {}).get("baseline", {})
    provenance = baseline.get("provenance", {})
    if provenance.get("identity") != build_identity(development.eventos, entities, {}):
        raise ValueError("La preparación guardada no coincide con Desarrollo, parámetros, código estadístico o dependencias")
    if provenance.get("effective_parameters_sha256") != sha256_canonical(effective_parameters(baseline)):
        raise ValueError("La preparación guardada no conserva sus parámetros certificados")
    return baseline


def run_one(root, flow_id, flow, summary, development, future, entities, context_map, truth,
            limit, reference_db, compare, run_number, reference_run, sample_mode, baseline):
    run_started = time.perf_counter()
    cpu_started = time.process_time()
    from app.modulos.clasificador.runtime import ClassifierProxy, load_snapshot
    selected = copy.copy(future)
    selected.eventos = chronological_prefix(future, limit)
    selected_rows = set(selected.eventos._raw_index.astype(int))
    active_truth = [item for item in truth if item["raw_index"] in selected_rows]
    if len(active_truth) != len(selected.eventos):
        raise ValueError("La referencia no contiene una etiqueta por fila del prefijo")
    model_ids = {node["config"]["model_id"] for node in flow["nodes"]
                 if node["type"] == "classifier" and node.get("config", {}).get("enabled") is not False}

    class FullSampleProxy(ClassifierProxy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.targets = None

    with tempfile.TemporaryDirectory(prefix=f"causal-flow-{run_number}-") as temporary:
        temp = Path(temporary)
        workbench = _isolated_workbench(root, temp, model_ids) if model_ids else None
        config = {key: value for key, value in context_map.get("config", {}).items()
                  if key in ConfiguracionContextual.__dataclass_fields__}
        runner = CausalFlowRunner(FiltroEstadistico(), FiltroContextualEspacioTemporal(ConfiguracionContextual(**config)), workbench)
        spool = FlowResultSpool(f"benchmark-{run_number}")
        reference_cursor = None
        if compare:
            reference_cursor = iter_run_results(reference_db, root / "storage/database/flow_lab.sqlite3", reference_run, "output")
        compared = 0
        digests = {}
        comparison_ms = 0.

        def sink(block):
            nonlocal compared, comparison_ms
            spool.write_block(block)
            comparison_started = time.perf_counter()
            for node, items in block.items():
                digest = digests.setdefault(node, hashlib.sha256())
                for item in items:
                    digest.update(json.dumps(_clean(item), sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":"), default=str).encode())
                    digest.update(b"\n")
            if reference_cursor is None:
                comparison_ms += (time.perf_counter() - comparison_started) * 1000
                return
            for item in block.get("output", []):
                reference_row = next(reference_cursor, None)
                if reference_row is None:
                    raise AssertionError("La referencia tiene menos resultados que la ejecución nueva.")
                expected = _clean(reference_row["payload"])
                _assert_equal(_clean(item), expected, f"output[{compared}]")
                compared += 1
            comparison_ms += (time.perf_counter() - comparison_started) * 1000

        last_progress = 0.
        def progress(processed, total, phase="evaluation", detail=None, **metadata):
            nonlocal last_progress
            now = time.perf_counter()
            if now - last_progress >= 30 or processed == total:
                print(json.dumps({"progress": {"run": run_number, "sample_mode": sample_mode,
                      "processed": processed, "total": total, "phase": phase}}), file=sys.stderr, flush=True)
                last_progress = now

        rss_start = _WorkingSet.current()
        started = time.perf_counter()
        try:
            with _WorkingSet() as memory, patch("app.modulos.clasificador.runtime.ClassifierProxy",
                                               FullSampleProxy if sample_mode == "full" else ClassifierProxy):
                result, _, snapshot = runner.evaluate(flow, development.eventos, selected.eventos,
                    entities, context_map, baseline, truth=active_truth, progress=progress, result_sink=sink)
                evaluation_wall_ms = (time.perf_counter() - started) * 1000
                spool.finish(snapshot=snapshot)
                spool_bytes = spool.path.stat().st_size
                local_store = FlowStore(temp / "flow-lab.sqlite3")
                persist_started = time.perf_counter()
                _, persist_ms = local_store.save_run_stream(flow_id, snapshot, result, spool.records())
                persistence_wall_ms = (time.perf_counter() - persist_started) * 1000
                database_bytes = sum(path.stat().st_size for path in temp.glob("flow-lab.sqlite3*"))
                database_bytes += sum(path.stat().st_size for path in (temp / "flow-lab.results").rglob("*") if path.is_file())
        finally:
            spool.cleanup()
            load_snapshot.cache_clear()
        if reference_cursor is not None and limit is None and next(reference_cursor, None) is not None:
            raise AssertionError("La ejecución nueva produjo menos resultados que la referencia.")
        phases = result.get("performance", {}).get("phases_ms", {})
        report = {"run": run_number, "sample_mode": sample_mode, "requested_rows": limit,
                  "rows": len(selected.eventos), "evaluation_wall_ms": round(evaluation_wall_ms, 3),
                  "end_to_end_ms": round((time.perf_counter() - run_started) * 1000, 3),
                  "main_process_cpu_ms": round((time.process_time() - cpu_started) * 1000, 3),
                  "persistence_wall_ms": round(persistence_wall_ms, 3), "comparison_ms": round(comparison_ms, 3),
                  "spool_bytes": spool_bytes, "database_bytes": database_bytes,
                  "phases_ms": phases, "persist_ms": persist_ms, "peak_rss_mib": round(memory.value / 1048576, 2),
                  "rss_start_mib": round(rss_start / 1048576, 2), "matched_outputs": compared,
                  "node_digests": {node: digest.hexdigest() for node, digest in digests.items()},
                  "evaluation": result.get("evaluation"), "statistical_evaluation": result.get("statistical_evaluation"),
                  "node_counts": result["node_counts"], "by_final_decision": result["by_final_decision"],
                  "checkpoints": phases and result["performance"]["counts"].get("checkpoints", []),
                  "session_stats": result["performance"]["counts"].get("diagnostic_sessions", {})}
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, help="ejecución cuya huella debe coincidir con el CSV local")
    parser.add_argument("--flow-run", help="usar el flujo congelado de otra ejecución como experimento nuevo")
    parser.add_argument("--reference-file", type=Path, help="referencia JSON exportada para medir con los datos montados en solo lectura")
    parser.add_argument("--export-reference", type=Path, help="exportar solo las instantáneas seleccionadas y etiquetas; no ejecutar el flujo")
    parser.add_argument("--use-prepared-cache", action="store_true", help="usar la preparación interna tras verificar su procedencia estadística")
    parser.add_argument("--sizes", nargs="+", type=int, default=[5000, 20000],
                        help="prefijos cronológicos del archivo (por ejemplo 20000 40000 80000)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sample-modes", nargs="+", choices=["full", "requested"], default=["full", "requested"])
    parser.add_argument("--compare-reference", action="store_true",
                        help="compara todos los resultados con la ejecución original almacenada")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = ROOT
    load_started = time.perf_counter()
    reference_data = _read_json(args.reference_file) if args.reference_file else None
    flow_id, flow, baseline_summary, reference = _load_reference_flow(root, args.reference_run, reference_data)
    if args.flow_run:
        if args.compare_reference:
            parser.error("Un flujo diferente se compara como experimento nuevo, no con las salidas históricas")
        flow_id, flow, flow_summary, flow_snapshot = _load_reference_flow(root, args.flow_run, reference_data)
    else:
        flow_snapshot = reference
        flow_summary = baseline_summary
    if args.export_reference:
        references = {args.reference_run: {"flow_id": reference["flow"]["flow_id"],
                      "snapshot": reference, "summary": baseline_summary},
                      args.flow_run or args.reference_run: {"flow_id": flow_id,
                      "snapshot": flow_snapshot, "summary": flow_summary}}
        args.export_reference.parent.mkdir(parents=True, exist_ok=True)
        args.export_reference.write_text(json.dumps({"runs": references, "truth": _load_truth(root, reference_data)},
                                                    ensure_ascii=False), encoding="utf-8")
        return
    if args.reference_file and args.compare_reference:
        parser.error("La referencia JSON no contiene las salidas históricas; compara los hashes de ambas variantes")
    flow = copy.deepcopy(flow)
    for node in flow["nodes"]:
        if node["type"] == "classifier" and node.get("config", {}).get("enabled") is not False:
            model_id = flow_snapshot.get("diagnostic_snapshots", {}).get(node["id"], {}).get("model", {}).get("model_id")
            if not model_id:
                raise ValueError(f"La referencia no identifica el modelo del nodo {node['id']}")
            node.setdefault("config", {})["model_id"] = model_id
    development, future, context_map = _load_inputs(root)
    source_fingerprint = _verify_source(future, reference["source"]["fingerprint"])
    entities, truth = development.entidades, _load_truth(root, reference_data)
    input_load_ms = (time.perf_counter() - load_started) * 1000
    # The API supplies a prepared baseline. Measure that one-time cold cost
    # separately, then give each replay its own runtime via evaluate's deep copy.
    print(json.dumps({"stage": "baseline_preparation"}), file=sys.stderr, flush=True)
    preparation_started = time.perf_counter()
    baseline = (_prepared_baseline(root, development, entities) if args.use_prepared_cache else
                MotorReproduccionValidacion(FiltroEstadistico()).construir_baseline(development.eventos, entities)) if any(
                    node["type"] == "statistical_focus" and node.get("config", {}).get("enabled") is not False
                    for node in flow["nodes"]) else {}
    baseline_preparation_ms = (time.perf_counter() - preparation_started) * 1000
    metadata = {"reference_run": args.reference_run, "flow_run": args.flow_run or args.reference_run,
                "source_fingerprint": source_fingerprint, "flow": flow, "input_load_ms": round(input_load_ms, 3),
                "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                "dependencies": {name: importlib.metadata.version(name) for name in ("pandas", "torch", "torch-geometric", "scikit-learn")},
                "context_map_sha256": hashlib.sha256((root/"storage/cache/contextual_map.json").read_bytes()).hexdigest(),
                "development_sha256": hashlib.sha256((root/"data/history/cargados/history.csv").read_bytes()).hexdigest(),
                "truth_sha256": hashlib.sha256(json.dumps(truth, sort_keys=True).encode()).hexdigest(),
                "baseline_preparation_ms": round(baseline_preparation_ms, 3),
                "baseline_mode": "verified_internal_cache" if args.use_prepared_cache else "cold",
                "baseline_id": baseline.get("provenance", {}).get("baseline_id"),
                "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "HISTORY_TRAINING_WORKERS")},
                "code_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((ROOT/"backend/app").rglob("*.py"))},
                "measurement_scope": "input_load_ms and baseline_preparation_ms are separate one-time costs; per-run end_to_end_ms includes prefix selection, isolated setup, replay, validation, spool and persistence including commit; RSS and CPU cover the main process"}
    print(json.dumps({"metadata": metadata}, ensure_ascii=False), flush=True)
    reference_db = _reference_connection(root / "storage/database/flow_lab.sqlite3") if args.compare_reference else None
    reports = []
    try:
        for size in args.sizes:
            if size < 1 or size > len(future.eventos):
                raise ValueError(f"Tamaño fuera de rango: {size}")
            limit = None if size == len(future.eventos) else size
            for run_number in range(1, args.runs + 1):
                modes = args.sample_modes if run_number % 2 else list(reversed(args.sample_modes))
                for mode in modes:
                    report = run_one(root, flow_id, flow, baseline_summary, development, future,
                                     entities, context_map, truth, limit, reference_db,
                                     args.compare_reference, run_number, args.reference_run, mode, baseline)
                    prior = next((item for item in reports if item["rows"] == report["rows"]), None)
                    if prior:
                        for key in ("node_digests", "evaluation", "statistical_evaluation", "node_counts", "by_final_decision"):
                            _assert_equal(report[key], prior[key], key)
                    reports.append(report)
                    gc.collect()
                    print(json.dumps(report, ensure_ascii=False, default=str), flush=True)
                    if args.output:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps({"metadata": metadata, "reports": reports}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    finally:
        if reference_db is not None:
            reference_db.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"metadata": metadata, "reports": reports}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
