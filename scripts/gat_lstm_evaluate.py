"""Evalúa un flujo con GAT-LSTM sobre los eventos futuros etiquetados de la API.

Carga Desarrollo y el CSV etiquetado en la aplicación, guarda un flujo con el
clasificador habilitado y selecciona su modelo antes de ejecutar este comando.
El resultado JSON contiene el identificador, las métricas y la configuración
de la ejecución persistida en Evaluación de flujos.
"""
import argparse
import json
import sys
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def request_json(base, path, payload=None, *, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"La API rechazó {path} ({exc.code}): {detail}") from exc


def evaluate_flow(base, flow_id, wait_seconds):
    flow_path = f"/flows/{quote(flow_id, safe='')}"
    flow = request_json(base, flow_path)
    if not any(node["type"] == "classifier" and node.get("config", {}).get("enabled") is not False
               for node in flow["nodes"]):
        raise ValueError("Selecciona un flujo con al menos un clasificador GAT-LSTM habilitado.")

    job = request_json(base, flow_path + "/runs", {})
    job_id = job["job_id"]
    status_path = f"/flow-evaluations/{quote(job_id, safe='')}"
    print(f"Evaluación iniciada. job_id={job_id}. Estado en {base}{status_path}",
          file=sys.stderr, flush=True)
    deadline = time.monotonic() + wait_seconds
    while job["status"] in {"queued", "running"}:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Se agotó la espera del trabajo {job_id}. Consulta {base}{status_path} "
                "para conocer su estado; la evaluación continúa en el servidor.")
        job = request_json(base, status_path, timeout=min(30, remaining))
        if job["status"] in {"queued", "running"}:
            time.sleep(min(1, max(0, deadline - time.monotonic())))

    if job["status"] != "completed":
        raise RuntimeError(f"La evaluación {job_id} terminó con estado {job['status']}: "
                           f"{job.get('error', 'sin detalle')}")
    return job["run"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "El comando usa el flujo y el CSV ya cargados en el servidor. "
        "Sustituye los antiguos model_id/dataset_id/--partition/--review-dir por --flow-id."))
    parser.add_argument("--flow-id", required=True, help="Identificador del flujo guardado en la aplicación")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="URL base de la API de Artifact")
    parser.add_argument("--wait-seconds", type=int, default=3600, help="Tiempo máximo de espera del resultado")
    args = parser.parse_args(argv)
    if args.wait_seconds <= 0:
        parser.error("--wait-seconds debe ser mayor que cero")
    result = evaluate_flow(args.base.rstrip("/"), args.flow_id, args.wait_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Espera interrumpida. Consulta el job_id mostrado para conocer el estado de la evaluación.",
              file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
