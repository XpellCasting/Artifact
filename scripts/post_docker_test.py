from __future__ import annotations

import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
WEB_BASE_URL = os.getenv("WEB_BASE_URL", "http://localhost:5173").rstrip("/")
FAULT_CASES = {
    "continuous": ["random", "malfunction", "bias", "drift", "missing"],
    "binary": ["random", "malfunction", "bias", "missing"],
}


def request(method: str, url: str, payload: dict | None = None) -> dict | str:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=20) as response:
        body = response.read().decode("utf-8")
        content_type = response.headers.get("content-type", "")
        return json.loads(body) if "application/json" in content_type else body


def api(method: str, path: str, payload: dict | None = None) -> dict:
    result = request(method, f"{API_BASE_URL}{path}", payload)
    if not isinstance(result, dict):
        raise AssertionError(f"{path} no devolvio JSON")
    return result


def wait_for_api(seconds: int = 60) -> None:
    deadline = time.time() + seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            health = api("GET", "/health")
            if health.get("status") == "ok":
                return
        except (HTTPError, URLError, TimeoutError, AssertionError) as exc:
            last_error = exc
        time.sleep(2)
    raise AssertionError(f"API no disponible en {API_BASE_URL}: {last_error}")


def first_entity(status: dict, sensor_kind: str) -> str:
    for entity in status.get("entities", []):
        if entity.get("sensor_kind") == sensor_kind:
            return str(entity["entity_id"])
    raise AssertionError(f"No hay entidad {sensor_kind} en el historico canonico")


def assert_fault_case(entity_id: str, fault_type: str) -> dict:
    tick_seconds = 7200 if fault_type == "missing" else 3600
    api("POST", "/realtime/reset")
    realtime = api(
        "POST",
        "/realtime/iterations",
        {
            "selected_entities": [entity_id],
            "tick_seconds": tick_seconds,
            "iteration_count": 2,
            "faults": [{"entity_id": entity_id, "fault_type": fault_type, "enabled": True}],
            "suspicion_threshold": 0.1,
        },
    )
    fault_ticks = realtime.get("fault_ticks") or []
    matched = [
        tick
        for tick in fault_ticks
        if tick.get("entity_id") == entity_id and tick.get("fault_type") == fault_type
    ]
    assert realtime.get("events") == 2, f"{fault_type}: events inesperados"
    assert realtime.get("summary", {}).get("iterations_done") == 2, f"{fault_type}: iteraciones incompletas"
    assert matched, f"{fault_type}: no aparece en fault_ticks"
    for tick in matched:
        assert tick.get("clean_state") is not None
        assert tick.get("observed_state") is not None
        assert tick.get("metadata", {}).get("modelo") == "SENSORNETS_2016_fault_injection"
    return realtime


def suspicion_summary(realtime: dict, entity_id: str) -> str:
    results = [item for item in realtime.get("results", []) if item.get("entity_id") == entity_id]
    if not results:
        return "sospecha=no"
    top = max(results, key=lambda item: float(item.get("suspicion_score") or 0.0))
    indicators = ",".join(top.get("triggered_indicators") or [])
    return (
        "sospecha=si "
        f"count={len(results)} "
        f"pattern={top.get('suspected_pattern') or top.get('pattern')} "
        f"score={top.get('suspicion_score')} "
        f"indicadores={indicators or '-'}"
    )


def main() -> int:
    wait_for_api()

    static_status = api("POST", "/history/static/ensure")
    assert static_status.get("exists") is True
    assert int(static_status.get("rows", 0)) > 0
    assert int(static_status.get("entity_count", 0)) > 0

    status = api("GET", "/history/static/status")
    continuous_entity = first_entity(status, "continuous")
    binary_entity = first_entity(status, "binary")

    analysis = api("POST", "/history/analyze", {"suspicion_threshold": 0.6})
    summary = analysis.get("summary", {})
    assert "timings_by_kind_ms" in summary
    assert "binary" in summary.get("implemented_kinds", [])

    checked_faults: list[str] = []
    reports: list[str] = []
    latest_binary = {}
    for sensor_kind, fault_types in FAULT_CASES.items():
        entity_id = continuous_entity if sensor_kind == "continuous" else binary_entity
        for fault_type in fault_types:
            realtime = assert_fault_case(entity_id, fault_type)
            checked_faults.append(f"{sensor_kind}:{fault_type}")
            reports.append(
                f"{sensor_kind}:{fault_type} inyectada=si ticks={len(realtime.get('fault_ticks') or [])} "
                f"{suspicion_summary(realtime, entity_id)}"
            )
            if sensor_kind == "binary":
                latest_binary = realtime

    binary_results = [
        item
        for item in latest_binary.get("historical_results", [])
        if item.get("sensor_kind") == "binary" and item.get("baseline_summary")
    ]
    if binary_results:
        baseline = binary_results[0]["baseline_summary"]
        assert baseline.get("horizon_model") == "long_and_recent"
        assert "recent_horizon_usable" in baseline

    html = request("GET", WEB_BASE_URL)
    assert isinstance(html, str) and "<html" in html.lower()

    print("post-docker smoke ok")
    print(f"api={API_BASE_URL} web={WEB_BASE_URL} continuous_entity={continuous_entity} binary_entity={binary_entity}")
    print(f"fault_checks={', '.join(checked_faults)}")
    print("detecciones:")
    for report in reports:
        print(f"- {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"post-docker smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
