"""Isolated causal-flow benchmark; reads repo fixtures and reference SQLite read-only."""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import shutil
import sqlite3
import tempfile
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.Cargar_Historial.Cargar_Historial import CargadorHistorial
from app.Diagnostico_Trabajo import DiagnosticWorkbench
from app.Evaluacion_Flujos import CausalFlowRunner, FlowResultSpool
from app.Filtro_Contextual.Contextual import FiltroContextualEspacioTemporal
from app.Filtro_Estadistico.Filtro import FiltroEstadistico
from app.Flujos import FlowStore, default_flow


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_RUN = "c78e3eb7db0a4568af785432e07002f0"
MODEL_ID = "669501ff239c48eaaeafce499bdb8d43"
RUN_IDENTITY_KEYS = {"run_id", "diagnosis_id", "execution_id", "observation_id", "event_id"}


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
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if key not in RUN_IDENTITY_KEYS}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


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
    development = loader.cargar_real(root / "HistoricosEnUso/cargados/history.csv")
    future = loader.cargar_real(root / "HistoricosEnUso/cargados/historico_validacion.csv")
    context_map = _read_json(root / "data/cache/contextual_map.json")
    return development, future, context_map


def _load_truth(root):
    with _reference_connection(root / "data/cache/validation_replay_results.sqlite3") as db:
        return [dict(zip(("raw_index", "entity", "tipo", "synthetic", "fault_type"), row))
                for row in db.execute("SELECT raw_index,entity,tipo,synthetic,fault_type "
                                      "FROM evaluation_truth ORDER BY replay_index")]


def _load_reference_flow(root):
    with _reference_connection(root / "data/cache/flow_lab.sqlite3") as db:
        row = db.execute("SELECT flow_id,snapshot,summary FROM flow_runs WHERE run_id=?",
                         (REFERENCE_RUN,)).fetchone()
        if row is None:
            raise RuntimeError("No se encontró la ejecución de referencia local.")
        flow = json.loads(row[1])["flow"]
        summary = json.loads(row[2])
        return row[0], flow, summary


def _isolated_workbench(root, temporary_root):
    source = root / "data/review/diagnostics"
    target = temporary_root / "review/diagnostics"
    for folder in ("models", "weights", "preprocessors", "graphs"):
        (target / folder).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / folder / f"{MODEL_ID}.{'pt' if folder == 'weights' else 'json'}",
                     target / folder / f"{MODEL_ID}.{'pt' if folder == 'weights' else 'json'}")
    shutil.copy2(source / "active.json", target / "active.json")
    return DiagnosticWorkbench(temporary_root / "review")


def _verify_source(future, expected_fingerprint):
    ordered = future.eventos.sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable")
    identity = ordered[[column for column in ("entity_id", "timestamp_dt", "state", "_raw_index")
                        if column in ordered]]
    actual = hashlib.sha256(pd.util.hash_pandas_object(identity, index=False).values.tobytes()).hexdigest()
    if actual != expected_fingerprint:
        raise RuntimeError(f"El archivo cargado no coincide con la referencia: {actual}")


def run_one(root, flow_id, flow, summary, development, future, entities, context_map, truth,
            limit, reference_db, compare, run_number):
    selected = future
    if limit is not None:
        ordered = future.eventos.sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable")
        if limit < len(ordered):
            selected = copy.copy(future)
            selected.eventos = ordered.head(limit).copy()
    active_truth = truth if limit is None else [item for item in truth
                                                 if item["raw_index"] in set(selected.eventos._raw_index.astype(int))]
    with tempfile.TemporaryDirectory(prefix=f"causal-flow-{run_number}-") as temporary:
        temp = Path(temporary)
        workbench = _isolated_workbench(root, temp)
        runner = CausalFlowRunner(FiltroEstadistico(), FiltroContextualEspacioTemporal(), workbench)
        spool = FlowResultSpool(f"benchmark-{run_number}")
        reference_cursor = None
        if compare:
            reference_cursor = reference_db.execute(
                "SELECT payload FROM flow_results WHERE run_id=? AND node_id='output' ORDER BY sequence",
                (REFERENCE_RUN,))
        compared = 0

        def sink(block):
            nonlocal compared
            spool.write_block(block)
            if reference_cursor is None:
                return
            for item in block.get("output", []):
                reference_row = reference_cursor.fetchone()
                if reference_row is None:
                    raise AssertionError("La referencia tiene menos resultados que la ejecución nueva.")
                expected = _clean(json.loads(reference_row[0]))
                _assert_equal(_clean(item), expected, f"output[{compared}]")
                compared += 1

        def progress(processed, total, phase="evaluation", detail=None, **metadata):
            return None

        rss_start = _WorkingSet.current()
        started = time.perf_counter()
        with _WorkingSet() as memory:
            result, _, snapshot = runner.evaluate(flow, development.eventos, selected.eventos,
                entities, context_map, {}, truth=active_truth, progress=progress, result_sink=sink)
        evaluation_wall_ms = (time.perf_counter() - started) * 1000
        spool.finish()
        local_store = FlowStore(temp / "flow-lab.sqlite3")
        _, persist_ms = local_store.save_run_stream(flow_id, snapshot, result, spool.records())
        spool.cleanup()
        if reference_cursor is not None and limit is None and reference_cursor.fetchone() is not None:
            raise AssertionError("La ejecución nueva produjo menos resultados que la referencia.")
        phases = result.get("performance", {}).get("phases_ms", {})
        report = {"run": run_number, "rows": len(selected.eventos), "evaluation_wall_ms": round(evaluation_wall_ms, 3),
                  "phases_ms": phases, "persist_ms": persist_ms, "peak_rss_mib": round(memory.value / 1048576, 2),
                  "rss_start_mib": round(rss_start / 1048576, 2), "matched_outputs": compared,
                  "checkpoints": phases and result["performance"]["counts"].get("checkpoints", []),
                  "session_stats": result["performance"]["counts"].get("diagnostic_sessions", {})}
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[86059],
                        help="prefijos cronológicos del archivo (por ejemplo 20000 40000 80000)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--compare-reference", action="store_true",
                        help="compara todos los resultados con la ejecución original almacenada")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = ROOT
    flow_id, flow, baseline_summary = _load_reference_flow(root)
    development, future, context_map = _load_inputs(root)
    _verify_source(future, baseline_summary["performance"]["counts"].get("source_fingerprint",
        "4450793b45c51878b6db638759a142fe7dc87a3ac906c87b29ad4ddc8c6aa1b4"))
    entities, truth = development.entidades, _load_truth(root)
    reference_db = _reference_connection(root / "data/cache/flow_lab.sqlite3")
    reports = []
    try:
        for size in args.sizes:
            if size < 1 or size > len(future.eventos):
                raise ValueError(f"Tamaño fuera de rango: {size}")
            limit = None if size == len(future.eventos) else size
            for run_number in range(1, args.runs + 1):
                report = run_one(root, flow_id, flow, baseline_summary, development, future,
                                 entities, context_map, truth, limit, reference_db,
                                 args.compare_reference, run_number)
                reports.append(report)
                print(json.dumps(report, ensure_ascii=False, default=str), flush=True)
    finally:
        reference_db.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
