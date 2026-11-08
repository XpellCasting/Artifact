"""Compare JSONL/SQLite and Flyweight archives using already evaluated results.

No detectors run and no production database is modified. The fixture is a JSONL
stream of complete timestamp groups, with node_id/sequence/observation_id/payload.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter, process_time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.database.flows import FlowStore
from app.flow.definitions import _json
from app.flow.evaluacion import LegacyFlowResultSpool, labelled_report, statistical_labelled_report, _merge_report_groups
from app.flow.result_spool import FlowResultSpool


def blocks(path, size):
    block, at, inputs = {}, None, 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            timestamp = record["payload"]["timestamp"]
            if at is not None and at != timestamp:
                yield block
                if inputs >= size:
                    return
                block = {}
            at = timestamp
            block.setdefault(record["node_id"], []).append(record["payload"])
            inputs += record["node_id"] == "input"
        if block:
            yield block
    if inputs < size:
        raise ValueError(f"La referencia solo contiene {inputs} entradas")


def digest_item(digest, item):
    digest.update(json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
    digest.update(b"\n")


def run_one(args, metadata, size, mode, repetition):
    with tempfile.TemporaryDirectory(prefix="flow-storage-", dir=args.work_dir) as temporary:
        root = Path(temporary)
        store = FlowStore(root / "flows.sqlite3")
        spool_class = FlowResultSpool if mode == "flyweight" else LegacyFlowResultSpool
        spool = spool_class("benchmark", directory=root)
        expected, counts, decisions, groups, statistical_groups = {}, Counter(), Counter(), {}, {}
        statistics = [node["id"] for node in metadata["snapshot"]["flow"]["nodes"] if node["type"] == "statistical_focus"]
        started, cpu_started = perf_counter(), process_time()
        validation_ms = 0.
        try:
            for block in blocks(args.fixture, size):
                validation_started = perf_counter()
                for node, items in block.items():
                    digest = expected.setdefault(node, hashlib.sha256())
                    for item in items:
                        digest_item(digest, item)
                    counts[node] += len(items)
                output = block.get("output", [])
                decisions.update(item.get("final_decision") for item in output)
                truth = [{"raw_index": item["source_raw_index"], "entity": item["entity_id"],
                          "tipo": item["evaluation"]["expected"], "fault_type": item["evaluation"].get("fault_type"),
                          "synthetic": "true" if item["evaluation"].get("synthetic") else "false"} for item in output]
                _merge_report_groups(groups, labelled_report(deepcopy(output), truth))
                for node in statistics:
                    _merge_report_groups(statistical_groups.setdefault(node, {}), statistical_labelled_report(block.get(node, []), truth, node))
                validation_ms += (perf_counter() - validation_started) * 1000
                spool.write_block(block)
            write_ms = (perf_counter() - started) * 1000
            finalization_started = perf_counter()
            if mode == "flyweight":
                spool.finish(snapshot=metadata["snapshot"])
            else:
                spool.finish()
            finalization_ms = (perf_counter() - finalization_started) * 1000
            temporary_bytes = spool.path.stat().st_size
            summary = {"candidate_count": counts["input"], "output_count": counts["output"],
                       "node_counts": dict(counts), "by_final_decision": dict(decisions),
                       "evaluation": {"labelled": True, "groups": list(groups.values())},
                       "statistical_evaluation": {node: {"groups": list(values.values())} for node, values in statistical_groups.items()}}
            saving_started = perf_counter()
            run, _ = store.save_run_stream(metadata["flow_id"], deepcopy(metadata["snapshot"]), summary, spool.records())
            saving_ms = (perf_counter() - saving_started) * 1000
            total_ms = (perf_counter() - started) * 1000
            cpu_ms = (process_time() - cpu_started) * 1000
            persistent_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and path != spool.path)
            if mode == "flyweight":
                persistent_bytes += spool.path.stat().st_size
            actual = {}
            verification_started = perf_counter()
            for node, count in counts.items():
                digest = hashlib.sha256()
                for offset in range(0, count, 500):
                    page = store.results(run, node, offset, 500)
                    if page["total"] != count:
                        raise AssertionError(f"Conteo diferente en {node}")
                    for item in page["items"]:
                        digest_item(digest, item)
                actual[node] = digest.hexdigest()
                if actual[node] != expected[node].hexdigest():
                    raise AssertionError(f"Datos diferentes después de guardar: {node}")
            saved = store.list_runs()[0]["summary"]
            for key in ("candidate_count", "output_count", "node_counts", "by_final_decision", "evaluation", "statistical_evaluation"):
                if saved[key] != summary[key]:
                    raise AssertionError(f"Métricas diferentes: {key}")
            return {"mode": mode, "repetition": repetition, "requested_rows": size,
                    "input_rows": counts["input"], "records": sum(counts.values()), "node_counts": dict(counts),
                    "write_ms": round(write_ms, 3), "finalize_ms": round(finalization_ms, 3),
                    "saving_ms": round(saving_ms, 3), "end_to_end_ms": round(total_ms, 3),
                    "cpu_ms": round(cpu_ms, 3), "validation_inside_write_ms": round(validation_ms, 3),
                    "read_verification_ms": round((perf_counter() - verification_started) * 1000, 3),
                    "temporary_bytes": temporary_bytes, "persistent_bytes": persistent_bytes,
                    "phases_ms": saved.get("performance", {}).get("phases_ms"),
                    "storage": saved.get("performance", {}).get("counts", {}).get("result_storage"),
                    "node_digests": actual}
        finally:
            spool.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="same filesystem as the production data for representative I/O")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 20000])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=["legacy", "flyweight"], default=["legacy", "flyweight"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    with args.fixture.open("rb") as source:
        fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
    code_root = Path(__file__).resolve().parents[1]
    report = {"reference_run": metadata["run_id"], "reference_prefix_counts": metadata["prefix_counts"],
              "platform": platform.platform(), "python": sys.version,
              "fixture_sha256": fingerprint,
              "code_sha256": {relative: hashlib.sha256((code_root/relative).read_bytes()).hexdigest() for relative in (
                  "backend/app/flow/result_spool.py", "backend/app/database/flows.py", "backend/app/flow/evaluacion.py",
                  "scripts/benchmark_flow_storage.py")},
              "scope": "storage of already evaluated results; input parsing and digest/metric checks are included identically; final read verification is excluded from end_to_end_ms; fresh isolated databases",
              "reports": []}
    for size in args.sizes:
        for repetition in range(1, args.runs + 1):
            modes = args.modes if repetition % 2 else list(reversed(args.modes))
            for mode in modes:
                print(json.dumps({"stage": "start", "mode": mode, "size": size, "repetition": repetition}), flush=True)
                result = run_one(args, metadata, size, mode, repetition)
                prior = next((row for row in report["reports"] if row["input_rows"] == result["input_rows"]), None)
                if prior and prior["node_digests"] != result["node_digests"]:
                    raise AssertionError("Las variantes no contienen los mismos resultados")
                report["reports"].append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
