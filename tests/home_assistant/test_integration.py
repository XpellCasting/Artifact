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
from app.services.home_assistant_bridge import HomeAssistantBridge
from custom_components.artifact.panel import alerts_panel_path, async_setup_alerts_panel, async_remove_alerts_panel
from homeassistant.components.frontend import DATA_PANELS


@pytest_asyncio.fixture
async def endpoint(tmp_path) -> AsyncIterator[tuple[str, dict]]:
    """Serve the real health contract without loading Artifact's ML runtime."""
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
        connection = await state["bridge"].attach(socket)
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
        title="Artifact",
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
async def test_setup_link_outage_recovery_and_unload(hass, endpoint, session, monkeypatch):
    base, state = endpoint
    entry = make_entry(base)
    monkeypatch.setattr(coordinator, "async_get_clientsession", lambda _: session)
    assert await async_setup_entry(hass, entry)
    assert hass.config_entries.async_forward_entry_setups.await_count == 1
    panel = hass.data[DATA_PANELS][alerts_panel_path(entry)]
    assert panel.sidebar_title == "Artifact-Alertas"
    assert panel.config["url"] == "http://artifact.example:5173/alertas"
    sensor = ArtifactConnection(entry.runtime_data, entry)
    assert sensor.available and sensor.is_on
    assert sensor.device_info["configuration_url"] == entry.data["web_url"]
    assert sensor.unique_id == "artifact-test_connection"
    state["status"] = 503
    await entry.runtime_data.async_refresh()
    assert not sensor.available
    state["status"] = 200
    await entry.runtime_data.async_refresh()
    assert sensor.available and sensor.is_on
    assert await async_unload_entry(hass, entry)
    assert alerts_panel_path(entry) not in hass.data[DATA_PANELS]
    await entry.runtime_data.async_shutdown()
    assert not session.closed


@pytest.mark.asyncio
async def test_alerts_panels_use_distinct_paths_and_can_reload_without_registering_static_routes_twice(hass):
    first, second = make_entry("http://first.example"), make_entry("http://second.example")
    second.entry_id = "artifact-second"
    second.data["web_url"] = "http://second.example:5173/"
    await async_setup_alerts_panel(hass, first)
    await async_setup_alerts_panel(hass, second)
    assert len(hass.data[DATA_PANELS]) == 2
    assert hass.data[DATA_PANELS][alerts_panel_path(second)].config["url"] == "http://second.example:5173/alertas"
    async_remove_alerts_panel(hass, first)
    assert alerts_panel_path(second) in hass.data[DATA_PANELS]
    first.data["web_url"] = "http://changed.example:5173"
    await async_setup_alerts_panel(hass, first)
    assert hass.data[DATA_PANELS][alerts_panel_path(first)].config["url"] == "http://changed.example:5173/alertas"
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
