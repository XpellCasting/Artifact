"""Serve history and selected state changes through the integration's hass object."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import timedelta
from functools import partial
import json
import logging
import secrets

from aiohttp import ClientError, WSMsgType
from homeassistant.core import callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.json import json_dumps
from homeassistant.util import dt as dt_util

from .api import InvalidArtifactResponse
from .const import CONF_BRIDGE_TOKEN

_LOGGER = logging.getLogger(__name__)
DOMAINS = {"sensor", "binary_sensor", "switch", "input_boolean", "select", "input_select"}


class ArtifactBridge:
    def __init__(self, hass, entry, client):
        self.hass, self.entry, self.client = hass, entry, client
        self.task = None
        self.token = entry.data.get(CONF_BRIDGE_TOKEN) or secrets.token_urlsafe(32)
        self.unsubscribers = {}

    async def async_start(self):
        """Provision a private, scoped link without creating a Home Assistant token."""
        if CONF_BRIDGE_TOKEN not in self.entry.data:
            self.hass.config_entries.async_update_entry(self.entry, data={**self.entry.data, CONF_BRIDGE_TOKEN: self.token})
        if not await self.client.async_register_bridge(self.entry.entry_id, self.token):
            return
        self.task = self.entry.async_create_background_task(self.hass, self._run(), "Artifact bridge")

    async def async_close(self):
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None

    def _clear_subscriptions(self):
        for cancel in self.unsubscribers.values():
            cancel()
        self.unsubscribers.clear()

    async def _run(self):
        delay = 1
        while True:
            tasks = set()
            writer = None
            try:
                if not await self.client.async_register_bridge(self.entry.entry_id, self.token):
                    raise InvalidArtifactResponse("The Artifact server does not support the bridge")
                async with self.client.bridge_socket(self.token) as socket:
                    queue = asyncio.Queue(maxsize=5000)
                    delay = 1

                    async def send():
                        while True:
                            await socket.send_str(json_dumps(await queue.get()))
                    writer = asyncio.create_task(send())
                    # Closing the socket also wakes the reader if writing fails.
                    def writing_done(task):
                        if not task.cancelled() and task.exception() is not None:
                            tasks.add(asyncio.create_task(socket.close()))
                    writer.add_done_callback(writing_done)
                    async for message in socket:
                        if message.type == WSMsgType.TEXT:
                            payload = json.loads(message.data)
                            if payload.get("type") == "request":
                                task = asyncio.create_task(self._command(payload, queue, socket))
                                tasks.add(task)
                                task.add_done_callback(tasks.discard)
                        elif message.type in {WSMsgType.CLOSED, WSMsgType.ERROR}:
                            break
            except asyncio.CancelledError:
                raise
            except (ClientError, TimeoutError, ValueError, InvalidArtifactResponse):
                _LOGGER.debug("Artifact bridge disconnected; retrying")
            finally:
                self._clear_subscriptions()
                if writer is not None:
                    writer.cancel()
                    tasks.add(writer)
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(delay)
            delay = min(30, delay * 2)

    async def _command(self, message, queue, socket):
        identifier = message.get("request_id")
        operation = message.get("operation")
        arguments = message.get("arguments") or {}
        try:
            result = []
            if operation == "states":
                rows = [state.as_dict() for state in self.hass.states.async_all() if state.domain in DOMAINS]
                for index in range(0, len(rows), 500):
                    await queue.put({"type": "result_chunk", "request_id": identifier, "items": rows[index:index + 500]})
            elif operation == "entity_catalog":
                rows, areas_available = self._entity_catalog()
                for index in range(0, len(rows), 500):
                    await queue.put({"type": "result_chunk", "request_id": identifier, "items": rows[index:index + 500]})
                result = {"areas_available": areas_available}
            elif operation == "history":
                from homeassistant.components.recorder import get_instance
                from homeassistant.components.recorder.history import get_significant_states
                start, end = dt_util.parse_datetime(arguments.get("start", "")), dt_util.parse_datetime(arguments.get("end", ""))
                entities = self._entities(arguments)
                if (start is None or end is None or start.tzinfo is None or end.tzinfo is None
                        or not start < end <= start + timedelta(days=1, seconds=1)):
                    raise ValueError("El intervalo solicitado no es válido.")
                if "recorder" not in self.hass.config.components:
                    raise ValueError("Activa Recorder en Home Assistant para consultar su historial.")
                query = partial(get_significant_states, self.hass, start_time=dt_util.as_utc(start),
                                end_time=dt_util.as_utc(end), entity_ids=entities,
                                include_start_time_state=False, significant_changes_only=False, minimal_response=True)
                history = await get_instance(self.hass).async_add_executor_job(query)
                for entity_id, rows in history.items():
                    for index in range(0, len(rows), 500):
                        await queue.put({"type": "result_chunk", "request_id": identifier,
                                         "entity_id": entity_id, "items": rows[index:index + 500]})
            elif operation == "subscribe":
                entities = self._entities(arguments)
                subscription = arguments["subscription_id"]
                if subscription in self.unsubscribers:
                    self.unsubscribers.pop(subscription)()
                @callback
                def forward(event):
                    try:
                        # Serialize HA State objects with Home Assistant's own encoder.
                        queue.put_nowait({"type": "event", "subscription_id": subscription, "event": event.as_dict()})
                    except asyncio.QueueFull:
                        self.hass.async_create_task(socket.close())
                self.unsubscribers[subscription] = async_track_state_change_event(self.hass, entities, forward)
            elif operation == "unsubscribe":
                cancel = self.unsubscribers.pop(arguments.get("subscription_id"), None)
                if cancel is not None:
                    cancel()
            else:
                raise ValueError("La operación solicitada no está permitida.")
            await queue.put({"type": "result", "request_id": identifier, "success": True, "data": result})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = str(exc) if isinstance(exc, ValueError) else "Home Assistant no pudo completar la consulta."
            await queue.put({"type": "result", "request_id": identifier, "success": False, "error": error})

    def _entity_catalog(self):
        """Resolve entity overrides and device inheritance once, without per-sensor I/O."""
        rows = [dict(state.as_dict()) for state in self.hass.states.async_all() if state.domain in DOMAINS]
        try:
            from homeassistant.helpers import area_registry, device_registry, entity_registry
            entities = entity_registry.async_get(self.hass).entities
            devices = device_registry.async_get(self.hass).devices
            areas = area_registry.async_get(self.hass).areas
            assignments = []
            for row in rows:
                entity = entities.get(row["entity_id"])
                device = devices.get(entity.device_id) if entity and entity.device_id else None
                area_id = (entity.area_id if entity else None) or (device.area_id if device else None)
                # An unresolved reference is unknown metadata, not an unassigned entity.
                area_name = areas[area_id].name if area_id else None
                assignments.append((area_id, area_name))
        except Exception:
            _LOGGER.debug("Area metadata is unavailable for the Artifact catalog", exc_info=True)
            return rows, False
        for row, (area_id, area_name) in zip(rows, assignments):
            row.update(area_id=area_id, area_name=area_name)
        return rows, True

    @staticmethod
    def _entities(arguments):
        entities = arguments.get("entity_ids")
        if (not isinstance(entities, list) or not 1 <= len(entities) <= 1000
                or any(not isinstance(entity, str) or entity.split(".", 1)[0] not in DOMAINS or "." not in entity for entity in entities)):
            raise ValueError("Selecciona entidades compatibles de Home Assistant.")
        return list(dict.fromkeys(entities))
