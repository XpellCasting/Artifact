from __future__ import annotations

import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
WEB_BASE_URL = os.getenv("WEB_BASE_URL", "http://localhost:5173").rstrip("/")


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


def wait_for_history(job: dict, seconds: int = 300) -> dict:
    deadline = time.time() + seconds
    current = job
    while current.get("status") in {"queued", "running"} and time.time() < deadline:
        time.sleep(0.5)
        current = api("GET", f"/history/real/load/jobs/{current['job_id']}")
    if current.get("status") != "completed":
        raise AssertionError(
            f"La carga real no finalizo correctamente: {current.get('error') or current}"
        )
    return current.get("history") or {}


def main() -> int:
    wait_for_api()

    availability = api("GET", "/history/static/status?autoload=false")
    if availability.get("real_history_available"):
        status = wait_for_history(api("POST", "/history/real/load/start"))
    else:
        status = api("POST", "/history/static/ensure")
    assert status.get("exists") is True
    assert int(status.get("rows", 0)) > 0
    assert int(status.get("entity_count", 0)) > 0

    timeline = api("GET", "/history/timeline?max_points=100")
    assert timeline.get("entities"), "La linea temporal historica esta vacia"
    assert "faults" not in timeline, "La API historica no debe exponer fallas sinteticas"

    replay = api("GET", "/validation/replay/status")
    assert "loaded" in replay
    assert "progress" in replay

    html = request("GET", WEB_BASE_URL)
    assert isinstance(html, str) and "<html" in html.lower()

    print("post-docker smoke ok")
    print(
        f"api={API_BASE_URL} web={WEB_BASE_URL} "
        f"rows={status.get('rows')} entities={status.get('entity_count')}"
    )
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"post-docker smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
