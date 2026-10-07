"""Profile a contextual-only causal replay without altering labels or runs."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
from time import perf_counter

import pandas as pd

from app.flow.evaluacion import CausalFlowRunner
from app.modulos.contextual.contextual import FiltroContextualEspacioTemporal, ConfiguracionContextual
from app.services.historial.cargar_historial import CargadorHistorial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loader = CargadorHistorial()
    development = loader.cargar_real(Path("data/history/cargados/history.csv"))
    raw = pd.read_csv("data/history/cargados/historico_validacion.csv", nrows=args.rows,
                      dtype=str, keep_default_na=False).rename(columns={"last_changed": "timestamp"})
    future = loader._normalizar(raw, origen="real", ruta=Path("data/history/cargados/historico_validacion.csv"))
    context_map = json.loads(Path("storage/cache/contextual_map.json").read_text())
    config = {key: value for key, value in context_map.get("config", {}).items()
              if key in ConfiguracionContextual.__dataclass_fields__}
    filtro = FiltroContextualEspacioTemporal(ConfiguracionContextual(**config))
    flow = {"name": "Contextual performance profile", "nodes": [
        {"id": name, "type": kind, "position": {}, "config": {"evidence_policy": "independent"}}
        for name, kind in (("input", "input"), ("contextual", "contextual"), ("output", "output"))],
        "edges": [{"id": "input-contextual", "source": "input", "target": "contextual"},
                  {"id": "contextual-output", "source": "contextual", "target": "output"}]}
    print("Profiling contextual replay", flush=True)
    profile = cProfile.Profile()
    started = perf_counter()
    profile.enable()
    summary, _, _ = CausalFlowRunner(None, filtro, None).evaluate(
        flow, development.eventos, future.eventos, development.entidades, context_map, {},
        result_sink=lambda block: None)
    profile.disable()
    report = {"rows": len(future.eventos), "elapsed_seconds": perf_counter() - started,
              "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(25)
    args.output.with_suffix(".profile.txt").write_text(stream.getvalue())
    print(json.dumps({"rows": report["rows"], "elapsed_seconds": report["elapsed_seconds"],
                      "phases_ms": summary.get("performance", {}).get("phases_ms")}), flush=True)
    print(stream.getvalue(), flush=True)


if __name__ == "__main__":
    main()
