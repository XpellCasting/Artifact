"""Exercise the HACS adapter against HTTP and Home Assistant's actual classes."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from threading import Event
import sys

import pytest
import pytest_asyncio
from aiohttp import ClientResponseError, ClientSession, web
from homeassistant.config_entries import ConfigEntryNotReady, ConfigEntryState
from homeassistant.core import HomeAssistant

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from custom_components.artifact import async_setup_entry, async_unload_entry
from custom_components.artifact import api, config_flow, coordinator
from custom_components.artifact.api import ArtifactClient, InvalidArtifactResponse, normalize_url
from custom_components.artifact.binary_sensor import ArtifactConnection
from custom_components.artifact.config_flow import ArtifactConfigFlow
from custom_components.artifact.bridge import ArtifactBridge
from custom_components.artifact.confidence import ConfidenceHub
from custom_components.artifact.confidence_dashboard import confidence_panel_path, async_setup_confidence_dashboard, async_remove_confidence_dashboard
from custom_components.artifact.sensor import ConfidenceSensor, async_setup_entry as setup_confidence_sensors
from app.database.confidence_entities import ConfidenceEntityStore
from app.services.home_assistant_bridge import HomeAssistantBridge
from custom_components.artifact.panel import alerts_panel_path, app_panel_path, async_remove_panels, async_setup_panels
from homeassistant.components.frontend import DATA_PANELS, DATA_EXTRA_MODULE_URL, UrlManager, async_register_built_in_panel


@pytest_asyncio.fixture
async def endpoint(tmp_path) -> AsyncIterator[tuple[str, dict]]:
    """Serve the real health contract without loading Paranoid's ML runtime."""
    state = {
        "payload": {"status": "ok", "static_history_dir": "/private/history", "real_history_path": "/private/history.csv"},
        "status": 200,
        "delay": 0,
        "paths": [],
        "bridge_support": False,
        "bridge": HomeAssistantBridge(tmp_path / "bridge.json"),
    }

    async def health(request):
        state["paths"].append(request.path)
        if state["delay"]:
            await asyncio.sleep(state["delay"])
        if "text" in state:
            return web.Response(text=state["text"], status=state["status"])
        return web.json_response(state["payload"], status=state["status"])

    app = web.Application()
    async def register_bridge(request):
        if not state["bridge_support"]:
            raise web.HTTPNotFound()
        payload = await request.json()
        try:
            state["bridge"].register(payload["bridge_id"], request.headers.get("Authorization", "")[7:])
        except PermissionError:
            raise web.HTTPConflict() from None
        return web.json_response({"registered": True})

    async def bridge_socket(request):
        if not state["bridge"].authorized(request.headers.get("Authorization", "")[7:]):
            raise web.HTTPUnauthorized()
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        connection = await state["bridge"].attach(
            socket, supports_test_confidence=request.headers.get("X-Paranoid-Confidence-Test") == "1")
        try:
            async for message in socket:
                if message.type == web.WSMsgType.TEXT:
                    connection.receive(message.json())
        finally:
            state["bridge"].detach(connection)
        return socket

    app.router.add_post("/home-assistant/bridge/register", register_bridge)
    app.router.add_get("/home-assistant/bridge", bridge_socket)
    app.router.add_get("/health", health)
    app.router.add_get("/artifact/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        await state["bridge"].close()
        await runner.cleanup()


@pytest_asyncio.fixture
async def session():
    async with ClientSession() as client:
        yield client


@pytest_asyncio.fixture
async def hass(tmp_path):
    instance = HomeAssistant(str(tmp_path))
    instance.http = SimpleNamespace(async_register_static_paths=AsyncMock())
    instance.data["lovelace"] = {"dashboards": {}}
    instance.data[DATA_EXTRA_MODULE_URL] = UrlManager(lambda *_: None, [])
    instance.config_entries = SimpleNamespace(
        async_forward_entry_setups=AsyncMock(),
        async_unload_platforms=AsyncMock(return_value=True),
        async_update_entry=Mock(),
    )
    yield instance
    await instance.async_block_till_done()


def make_entry(base_url):
    return SimpleNamespace(
        entry_id="artifact-test",
        title="Paranoid",
        data={"url": base_url, "web_url": "http://artifact.example:5173"},
        state=ConfigEntryState.SETUP_IN_PROGRESS,
        async_on_unload=Mock(),
        pref_disable_polling=False,
        runtime_data=None,
        async_create_background_task=lambda current_hass, coroutine, name: current_hass.async_create_background_task(coroutine, name),
    )


def make_flow(hass, session, monkeypatch, entries=()):
    flow = ArtifactConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}
    monkeypatch.setattr(flow, "_async_current_entries", lambda: entries)
    monkeypatch.setattr(config_flow, "async_get_clientsession", lambda _: session)
    return flow


@pytest.mark.parametrize("value", ["ftp://host", "host:8000", "http://", "http://host:99999", "http://user:pass@host", "http://host?token=secret", "http://host#fragment", "http://bad host", "http://host/\nhealth"])
def test_reject_invalid_addresses(value):
    with pytest.raises(ValueError):
        normalize_url(value)


def test_canonical_url_preserves_proxy_subpath():
    assert normalize_url(" HTTPS://ARTIFACT.EXAMPLE:443/api/ ") == "https://artifact.example/api"


@pytest.mark.asyncio
async def test_health_uses_subpath_and_omits_private_paths(endpoint, session):
    base, state = endpoint
    client = ArtifactClient(session, base + "/artifact/")
    assert await client.async_health() == {"status": "ok"}
    assert state["paths"] == ["/artifact/health"]
    assert not session.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"status": "ok"}, {"status": "error"}, [], None])
async def test_reject_unrelated_or_unhealthy_services(endpoint, session, payload):
    base, state = endpoint
    state["payload"] = payload
    with pytest.raises(InvalidArtifactResponse):
        await ArtifactClient(session, base).async_health()


@pytest.mark.asyncio
async def test_reject_html_page(endpoint, session):
    base, state = endpoint
    state["text"] = "<html>Login page</html>"
    with pytest.raises(InvalidArtifactResponse):
        await ArtifactClient(session, base).async_health()


@pytest.mark.asyncio
async def test_http_errors_and_timeouts(endpoint, session, monkeypatch):
    base, state = endpoint
    state["status"] = 503
    with pytest.raises(ClientResponseError):
        await ArtifactClient(session, base).async_health()
    state["status"] = 200
    state["delay"] = 0.05
    monkeypatch.setattr(api, "REQUEST_TIMEOUT", 0.01)
    with pytest.raises(TimeoutError):
        await ArtifactClient(session, base).async_health()


@pytest.mark.asyncio
async def test_setup_form_and_success(hass, endpoint, session, monkeypatch):
    base, _ = endpoint
    flow = make_flow(hass, session, monkeypatch)
    assert (await flow.async_step_user())["type"] == "form"
    result = await flow.async_step_user({"url": base + "/", "web_url": "http://artifact.example:5173/"})
    assert result["type"] == "create_entry"
    assert result["title"] == "Paranoid"
    assert result["data"] == {"url": base, "web_url": "http://artifact.example:5173"}


@pytest.mark.asyncio
async def test_duplicate_server_aborts(hass, endpoint, session, monkeypatch):
    base, _ = endpoint
    flow = make_flow(hass, session, monkeypatch, [make_entry(base)])
    result = await flow.async_step_user({"url": base, "web_url": "http://artifact.example"})
    assert result["type"] == "abort"
    assert result["reason"] == "already_configured"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,error", [("invalid", "invalid_url"), ("wrong_service", "invalid_response"), ("http_error", "cannot_connect")])
async def test_flow_reports_connection_errors(hass, endpoint, session, monkeypatch, kind, error):
    base, state = endpoint
    if kind == "wrong_service":
        state["payload"] = {"status": "ok"}
    elif kind == "http_error":
        state["status"] = 503
    flow = make_flow(hass, session, monkeypatch)
    result = await flow.async_step_user({"url": "file:///tmp" if kind == "invalid" else base, "web_url": "http://artifact.example"})
    assert result["type"] == "form"
    assert result["errors"] == {"base": error}


@pytest.mark.asyncio
async def test_reconfigure_updates_same_entry(hass, endpoint, session, monkeypatch):
    base, _ = endpoint
    entry = make_entry(base)
    flow = make_flow(hass, session, monkeypatch, [entry])
    monkeypatch.setattr(flow, "_get_reconfigure_entry", lambda: entry)
    update = Mock(return_value={"type": "abort", "reason": "reconfigure_successful"})
    monkeypatch.setattr(flow, "async_update_reload_and_abort", update)
    assert (await flow.async_step_reconfigure())["step_id"] == "reconfigure"
    result = await flow.async_step_reconfigure({"url": base, "web_url": "http://new.example:5173"})
    assert result["reason"] == "reconfigure_successful"
    update.assert_called_once_with(entry, data_updates={"url": base, "web_url": "http://new.example:5173"})


@pytest.mark.asyncio
@pytest.mark.parametrize("title", ["Artifact", "Paranoid", "Sensores de mi casa"])
async def test_setup_link_outage_recovery_and_unload(hass, endpoint, session, monkeypatch, title):
    base, state = endpoint
    entry = make_entry(base)
    entry.title = title
    def update_entry(current, **changes):
        for key, value in changes.items():
            setattr(current, key, value)
    hass.config_entries.async_update_entry.side_effect = update_entry
    monkeypatch.setattr(coordinator, "async_get_clientsession", lambda _: session)
    assert await async_setup_entry(hass, entry)
    expected_title = "Paranoid" if title == "Artifact" else title
    assert entry.title == expected_title
    title_updates = [call.kwargs["title"] for call in hass.config_entries.async_update_entry.call_args_list if "title" in call.kwargs]
    assert title_updates == (["Paranoid"] if title == "Artifact" else [])
    assert hass.config_entries.async_forward_entry_setups.await_count == 1
    app_panel = hass.data[DATA_PANELS][app_panel_path(entry)]
    assert app_panel.sidebar_title == "Paranoid"
    assert app_panel.config["url"] == "http://artifact.example:5173"
    panel = hass.data[DATA_PANELS][alerts_panel_path(entry)]
    assert panel.sidebar_title == "Paranoid-Alertas"
    assert panel.config["url"] == "http://artifact.example:5173/alertas"
    assert hass.data[DATA_PANELS][confidence_panel_path(entry)].sidebar_title == "Paranoid-Confianza"
    assert any("dashboard-theme.js" in url for url in hass.data[DATA_EXTRA_MODULE_URL].urls)
    sensor = ArtifactConnection(entry.runtime_data, entry)
    assert sensor.available and sensor.is_on
    assert sensor.device_info["name"] == expected_title
    assert sensor.device_info["model"] == "Paranoid"
    assert sensor.device_info["configuration_url"] == entry.data["web_url"]
    assert sensor.unique_id == "artifact-test_connection"
    state["status"] = 503
    await entry.runtime_data.async_refresh()
    assert not sensor.available
    state["status"] = 200
    await entry.runtime_data.async_refresh()
    assert sensor.available and sensor.is_on
    assert await async_unload_entry(hass, entry)
    assert app_panel_path(entry) not in hass.data[DATA_PANELS]
    assert alerts_panel_path(entry) not in hass.data[DATA_PANELS]
    assert confidence_panel_path(entry) not in hass.data[DATA_PANELS]
    assert not hass.data[DATA_EXTRA_MODULE_URL].urls
    await entry.runtime_data.async_shutdown()
    assert not session.closed


@pytest.mark.asyncio
async def test_panels_reload_without_removing_other_entries_or_user_dashboards(hass):
    first, second = make_entry("http://first.example"), make_entry("http://second.example")
    second.entry_id = "artifact-second"
    second.data["web_url"] = "http://second.example:5173/app/"
    async_register_built_in_panel(hass, "lovelace", "Artifact", frontend_url_path="dashboard-artifact")
    await async_setup_panels(hass, first)
    await async_setup_panels(hass, second)
    panels = hass.data[DATA_PANELS]
    assert len(panels) == 5
    assert panels[app_panel_path(second)].config["url"] == "http://second.example:5173/app"
    assert panels[alerts_panel_path(second)].config["url"] == "http://second.example:5173/app/alertas"
    async_remove_panels(hass, first)
    assert set(panels) == {app_panel_path(second), alerts_panel_path(second), "dashboard-artifact"}
    first.data["web_url"] = "http://changed.example:5173"
    await async_setup_panels(hass, first)
    assert panels[app_panel_path(first)].config["url"] == "http://changed.example:5173"
    assert panels[alerts_panel_path(first)].config["url"] == "http://changed.example:5173/alertas"
    assert hass.http.async_register_static_paths.await_count == 1


@pytest.mark.asyncio
async def test_first_refresh_retries_without_forwarding_entities(hass, endpoint, session, monkeypatch):
    base, state = endpoint
    state["status"] = 503
    entry = make_entry(base)
    monkeypatch.setattr(coordinator, "async_get_clientsession", lambda _: session)
    with pytest.raises(ConfigEntryNotReady):
        await async_setup_entry(hass, entry)
    hass.config_entries.async_forward_entry_setups.assert_not_awaited()


async def connected_bridge(hass, endpoint, session):
    base, state = endpoint
    state["bridge_support"] = True
    entry = make_entry(base)
    integration = ArtifactBridge(hass, entry, ArtifactClient(session, base))
    await integration.async_start()
    for _ in range(100):
        if state["bridge"].status()["bridge_connected"]:
            return integration, state["bridge"]
        await asyncio.sleep(.01)
    await integration.async_close()
    pytest.fail("The automatic bridge did not connect")


@pytest.mark.asyncio
async def test_bridge_pairs_automatically_without_creating_a_home_assistant_token(hass, endpoint, session):
    integration, bridge = await connected_bridge(hass, endpoint, session)
    try:
        assert bridge.status() == {"bridge_configured": True, "bridge_connected": True}
        assert len(integration.token) >= 32
        assert integration.token not in bridge.path.read_text()
        stored = hass.config_entries.async_update_entry.call_args.kwargs["data"]
        assert stored["bridge_token"] == integration.token
        assert "access_token" not in stored
    finally:
        await integration.async_close()


@pytest.mark.asyncio
async def test_bridge_reads_states_using_hass_and_preserves_rpc_chunk_order(hass, endpoint, session):
    hass.states.async_set("sensor.temperature", "20")
    hass.states.async_set("light.private", "on")
    integration, bridge = await connected_bridge(hass, endpoint, session)
    try:
        states = await asyncio.to_thread(bridge.request, "states", {})
        assert [row["entity_id"] for row in states] == ["sensor.temperature"]
        assert states[0]["state"] == "20"
    finally:
        await integration.async_close()


@pytest.mark.asyncio
async def test_catalog_resolves_entity_override_device_inheritance_and_missing_registrations(hass, endpoint, session, monkeypatch):
    from homeassistant.helpers import area_registry, device_registry, entity_registry
    for key in ("direct", "inherited", "override", "unassigned", "unregistered", "missing_device"):
        hass.states.async_set(f"sensor.{key}", "20")
    hass.states.async_set("light.unsupported", "on")
    entities = {
        "sensor.direct": SimpleNamespace(area_id="kitchen", device_id=None),
        "sensor.inherited": SimpleNamespace(area_id=None, device_id="device"),
        "sensor.override": SimpleNamespace(area_id="kitchen", device_id="device"),
        "sensor.unassigned": SimpleNamespace(area_id=None, device_id=None),
        "sensor.missing_device": SimpleNamespace(area_id=None, device_id="missing"),
    }
    getters = []
    for module, name, index in (
        (entity_registry, "entities", entities),
        (device_registry, "devices", {"device": SimpleNamespace(area_id="living")}),
        (area_registry, "areas", {"kitchen": SimpleNamespace(name="Cocina"), "living": SimpleNamespace(name="Salón")}),
    ):
        getter = Mock(return_value=SimpleNamespace(**{name: index}))
        monkeypatch.setattr(module, "async_get", getter)
        getters.append(getter)
    integration, bridge = await connected_bridge(hass, endpoint, session)
    try:
        catalog = await asyncio.to_thread(bridge.entity_catalog)
        assert catalog["areas_available"] is True
        rows = {row["entity_id"]: row for row in catalog["entities"]}
        assert len(rows) == 6
        assert rows["sensor.direct"]["area_name"] == "Cocina"
        assert rows["sensor.inherited"]["area_id"] == "living"
        assert rows["sensor.override"]["area_id"] == "kitchen"
        assert all(rows[f"sensor.{key}"]["area_id"] is None for key in ("unassigned", "unregistered", "missing_device"))
        for getter in getters:
            getter.assert_called_once_with(hass)
    finally:
        await integration.async_close()


@pytest.mark.asyncio
async def test_catalog_reads_real_empty_registries_and_keeps_no_areas_distinct_from_unavailable(hass, endpoint, session, monkeypatch):
    from homeassistant.helpers import area_registry, device_registry, entity_registry
    await area_registry.async_load(hass)
    await device_registry.async_load(hass)
    await entity_registry.async_load(hass)
    hass.states.async_set("sensor.unregistered", "20")
    integration, bridge = await connected_bridge(hass, endpoint, session)
    try:
        catalog = await asyncio.to_thread(bridge.entity_catalog)
        assert catalog["areas_available"] is True
        assert catalog["entities"][0]["area_id"] is None
        assert catalog["entities"][0]["area_name"] is None
        monkeypatch.setattr(area_registry, "async_get", Mock(side_effect=RuntimeError("unavailable")))
        catalog = await asyncio.to_thread(bridge.entity_catalog)
        assert catalog["areas_available"] is False
        assert "area_id" not in catalog["entities"][0]
    finally:
        await integration.async_close()


@pytest.mark.asyncio
async def test_catalog_chunks_5000_entities_without_history_queries(hass, monkeypatch):
    from homeassistant.helpers import area_registry, device_registry, entity_registry
    for module, attribute in ((area_registry, "areas"), (device_registry, "devices"), (entity_registry, "entities")):
        monkeypatch.setattr(module, "async_get", Mock(return_value=SimpleNamespace(**{attribute: {}})))
    for index in range(5000):
        hass.states.async_set(f"sensor.entity_{index}", str(index))
    integration = ArtifactBridge(hass, make_entry("http://artifact.local"), None)
    queue = asyncio.Queue()
    await integration._command({"request_id": "catalog", "operation": "entity_catalog"}, queue, None)
    chunks = []
    while not queue.empty():
        chunks.append(queue.get_nowait())
    assert [len(chunk["items"]) for chunk in chunks[:-1]] == [500] * 10
    assert chunks[-1] == {"type": "result", "request_id": "catalog", "success": True, "data": {"areas_available": True}}
    assert chunks[-2]["items"][-1]["entity_id"] == "sensor.entity_4999"


@pytest.mark.asyncio
async def test_bridge_queries_recorder_in_executor_without_a_rest_token(hass, endpoint, session, monkeypatch):
    from homeassistant.components import recorder
    from homeassistant.components.recorder import history
    hass.config.components.add("recorder")
    rows = [{"entity_id": "sensor.temperature", "state": str(index), "last_changed": "2026-01-01T01:00:00Z"} for index in range(601)]
    query = Mock(return_value={"sensor.temperature": rows})
    monkeypatch.setattr(history, "get_significant_states", query)
    executor = AsyncMock(side_effect=lambda operation: operation())
    monkeypatch.setattr(recorder, "get_instance", lambda _: SimpleNamespace(async_add_executor_job=executor))
    integration, bridge = await connected_bridge(hass, endpoint, session)
    try:
        result = await asyncio.to_thread(bridge.request, "history", {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z", "entity_ids": ["sensor.temperature"]})
        assert result == [rows]
        executor.assert_awaited_once()
        assert query.call_args.kwargs["include_start_time_state"] is False
        assert query.call_args.kwargs["minimal_response"] is True
    finally:
        await integration.async_close()


@pytest.mark.asyncio
async def test_bridge_forwards_selected_state_events_and_removes_listener_on_stop(hass, endpoint, session):
    hass.states.async_set("binary_sensor.door", "off")
    integration, bridge = await connected_bridge(hass, endpoint, session)
    stopped = Event()
    received = asyncio.Event()
    events = []
    def accept(message):
        events.append(message)
        hass.loop.call_soon_threadsafe(received.set)
    listener = asyncio.create_task(asyncio.to_thread(bridge.listen, ["binary_sensor.door"], accept, stopped))
    try:
        for _ in range(100):
            if integration.unsubscribers:
                break
            await asyncio.sleep(.01)
        assert integration.unsubscribers
        hass.states.async_set("binary_sensor.door", "on")
        await asyncio.wait_for(received.wait(), timeout=3)
        assert events[0]["event"]["data"]["new_state"]["state"] == "on"
        stopped.set()
        await asyncio.wait_for(listener, timeout=3)
        assert not integration.unsubscribers
    finally:
        stopped.set()
        await integration.async_close()
        await asyncio.gather(listener, return_exceptions=True)


@pytest.mark.asyncio
async def test_bridge_refuses_service_calls_and_unbounded_history(hass, endpoint, session):
    integration, bridge = await connected_bridge(hass, endpoint, session)
    try:
        with pytest.raises(ValueError, match="no está permitida"):
            await asyncio.to_thread(bridge.request, "call_service", {})
        with pytest.raises(ValueError, match="intervalo"):
            await asyncio.to_thread(bridge.request, "history", {"start": "2026-01-01T00:00:00Z", "end": "2026-01-05T00:00:00Z", "entity_ids": ["sensor.temperature"]})
    finally:
        await integration.async_close()


def confidence_payload(value=90, revision=1, source="sensor.temperature", **changes):
    item = {"entity_id": f"confianza.{source.replace('.', '_')}", "source_entity_id": source,
            "value": value, "confidence": value / 100 if type(value) in (int, float) else None,
            "store_id": "store-one", "revision": revision, "available": True,
            "confidence_kind": "operational_index", "confidence_calibrated": False,
            "evaluated_at": "2026-11-18T01:00:00+00:00", "flow_id": "flow-one"}
    item.update(changes)
    return {"schema_version": "confidence-entities-v1", "store_id": item["store_id"],
            "revision": revision, "entities": [item], "reset": True, "complete": True}


@pytest.mark.asyncio
async def test_native_confidence_sensor_percentage_unknown_disconnect_and_stable_identity(hass):
    hub = ConfidenceHub()
    hub.set_connected(True)
    hub.apply(confidence_payload(), sync=True)
    entry = make_entry("http://artifact.example")
    sensor = ConfidenceSensor(hub, entry, "confianza.sensor_temperature")
    sensor.hass = hass
    sensor.async_write_ha_state = Mock()
    await sensor.async_added_to_hass()
    assert sensor.entity_id == "sensor.confianza_sensor_temperature"
    assert sensor.native_unit_of_measurement == "%"
    assert sensor.state_class == "measurement"
    assert sensor.native_value == 90
    assert sensor.available
    identity = sensor.unique_id
    hub.apply(confidence_payload(0, 2, flow_id="flow-two"))
    assert sensor.native_value == 0
    assert sensor.unique_id == identity
    assert sensor.extra_state_attributes["flow_id"] == "flow-two"
    assert sensor.extra_state_attributes["confidence_entity_id"] == "confianza.sensor_temperature"
    assert sensor.extra_state_attributes["paranoid_confidence"] is True
    hub.apply(confidence_payload(None, 3))
    assert sensor.native_value is None and sensor.available
    hub.set_connected(False)
    assert not sensor.available
    hub.set_connected(True)
    assert not sensor.available  # Wait for the new authoritative snapshot.
    hub.apply(confidence_payload(100, 4), sync=True)
    assert sensor.available and sensor.native_value == 100
    assert sensor.async_write_ha_state.call_count >= 4


@pytest.mark.asyncio
async def test_dynamic_sensor_discovery_only_adds_each_source_once(hass):
    hub = ConfidenceHub()
    entry = make_entry("http://artifact.example")
    entry.runtime_data = SimpleNamespace(bridge=SimpleNamespace(confidence=hub))
    add = Mock()
    await setup_confidence_sensors(hass, entry, add)
    add.assert_not_called()
    hub.apply(confidence_payload(), sync=True)
    assert add.call_count == 1
    hub.apply(confidence_payload(75, 2))
    assert add.call_count == 1
    hub.apply(confidence_payload(80, 3, source="binary_sensor.door"))
    assert add.call_count == 2
    assert add.call_args.args[0][0].entity_id == "sensor.confianza_binary_sensor_door"


@pytest.mark.asyncio
async def test_confidence_hub_rejects_invalid_states_and_ignores_older_publications(hass):
    hub = ConfidenceHub()
    hub.apply(confidence_payload(80, 5), sync=True)
    hub.apply(confidence_payload(10, 4))
    assert hub.entities["confianza.sensor_temperature"]["value"] == 80
    for invalid in (-1, 101, float("nan"), float("inf"), True, "90"):
        with pytest.raises(ValueError):
            hub.apply(confidence_payload(invalid, 6))
    with pytest.raises(ValueError):
        hub.apply(confidence_payload(source="sensor.confianza_sensor_temperature"))
    hub.apply(confidence_payload(60, 1, store_id="replacement"), sync=True)
    assert hub.entities["confianza.sensor_temperature"]["value"] == 60


@pytest.mark.asyncio
async def test_confidence_dashboard_uses_native_graph_bounds_and_preserves_user_dashboards(hass):
    entry = make_entry("http://artifact.example")
    hub = ConfidenceHub()
    user_dashboard = object()
    path = confidence_panel_path(entry)
    hass.data["lovelace"]["dashboards"][path] = user_dashboard
    async_setup_confidence_dashboard(hass, entry, hub)
    dashboard = hass.data["artifact"]["confidence_dashboards"][entry.entry_id]
    assert dashboard.url_path != path
    for index in range(10):
        hub.register_entity(f"confianza.sensor_s{index}", f"sensor.confianza_sensor_s{index}")
    hub.apply(confidence_payload(source_mode="test", entity_id="confianza.test_sensor_x", source="sensor.x"))
    hub.register_entity("confianza.test_sensor_x", "sensor.confianza_test_sensor_x")
    config = await dashboard.async_load(False)
    graphs = [card for card in config["views"][0]["cards"] if card["type"] == "history-graph"]
    assert [len(card["entities"]) for card in graphs] == [8, 2, 1]
    assert graphs[-1]["entities"] == ["sensor.confianza_test_sensor_x"]
    assert graphs[-1]["title"].endswith("Test")
    assert all(card["min_y_axis"] == 0 and card["max_y_axis"] == 100 and card["fit_y_data"] is False for card in graphs)
    assert dashboard.mode == "yaml"
    assert await dashboard.async_json(False)
    async_remove_confidence_dashboard(hass, entry)
    assert hass.data["lovelace"]["dashboards"][path] is user_dashboard
    assert dashboard.url_path not in hass.data[DATA_PANELS]


@pytest.mark.asyncio
async def test_bridge_synchronizes_persistent_confidence_and_excludes_feedback(hass, endpoint, session, tmp_path):
    store = ConfidenceEntityStore(tmp_path / "confidence.sqlite3")
    store.start("live", "flow", ["sensor.temperature"])
    endpoint[1]["bridge"].set_confidence_provider(store.snapshot)
    integration, bridge = await connected_bridge(hass, endpoint, session)
    try:
        for _ in range(100):
            if integration.confidence.synchronized:
                break
            await asyncio.sleep(.01)
        assert integration.confidence.synchronized
        item = {"entity_id": "sensor.temperature", "observation_id": "one", "execution_id": "execution",
                "timestamp": "2026-11-18T01:00:00+00:00", "confidence": .93,
                "confidence_fusion": {"active_filters": 2, "available_filters": 2, "method": "uniform_v1"}}
        await asyncio.to_thread(bridge.publish_confidence, store.record("live", [item]))
        assert integration.confidence.entities["confianza.sensor_temperature"]["value"] == 93
        entry = integration.entry
        entry.runtime_data = SimpleNamespace(bridge=integration)
        add = Mock()
        await setup_confidence_sensors(hass, entry, add)
        live_sensor = add.call_args.args[0][0]
        await asyncio.to_thread(bridge.publish_confidence, store.start("test", "flow", ["sensor.temperature"], test=True))
        test_sensor = add.call_args.args[0][0]
        assert test_sensor.entity_id == "sensor.confianza_test_sensor_temperature"
        assert test_sensor.unique_id != live_sensor.unique_id
        assert test_sensor.extra_state_attributes["source_mode"] == "test"
        assert test_sensor.native_value is None
        values = []
        integration.confidence.listen(lambda: values.append(test_sensor.native_value))
        readings = [dict(item, observation_id="two", confidence=.55), dict(item, observation_id="three", confidence=.71)]
        await asyncio.to_thread(bridge.publish_confidence, store.record("test", readings))
        assert values == [55, 71]
        assert live_sensor.native_value == 93
        await asyncio.to_thread(bridge.publish_confidence, store.stop("test"))
        assert not test_sensor.available
        assert live_sensor.available
        hass.states.async_set("sensor.temperature", 24.5)
        hass.states.async_set("sensor.confianza_sensor_temperature", 93, {"paranoid_confidence": True})
        hass.states.async_set("sensor.confianza_test_sensor_temperature", 71, {"paranoid_confidence": True})
        states = await asyncio.to_thread(bridge.request, "states", {})
        assert {row["entity_id"] for row in states} == {"sensor.temperature"}
        await asyncio.to_thread(bridge.publish_confidence, store.stop("live"))
        assert not live_sensor.available
    finally:
        await integration.async_close()
