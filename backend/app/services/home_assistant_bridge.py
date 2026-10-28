"""A scoped, outbound Home Assistant connection for history and state events."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from concurrent.futures import TimeoutError as FutureTimeout
import hashlib
import hmac
import json
from pathlib import Path
from queue import Empty, Full, Queue
import re
from threading import Lock
from time import monotonic
from uuid import uuid4


class BridgeUnavailable(ValueError):
    pass


class BridgeOperationUnsupported(BridgeUnavailable):
    pass


class BridgeConnection:
    def __init__(self, socket):
        self.socket = socket
        self.loop = asyncio.get_running_loop()
        self.pending = {}
        self.subscriptions = {}
        self.send_lock = asyncio.Lock()
        self.connected = True

    async def request(self, operation, arguments):
        if not self.connected:
            raise BridgeUnavailable("La integración de Home Assistant está desconectada.")
        identifier = uuid4().hex
        future = self.loop.create_future()
        self.pending[identifier] = {"future": future, "items": [], "groups": {}, "operation": operation}
        try:
            async with self.send_lock:
                await self.socket.send_json({"type": "request", "request_id": identifier,
                                             "operation": operation, "arguments": arguments})
            return await asyncio.wait_for(future, timeout=120)
        finally:
            self.pending.pop(identifier, None)

    def receive(self, message):
        kind = message.get("type")
        if kind == "event":
            queue = self.subscriptions.get(message.get("subscription_id"))
            if queue is not None:
                try:
                    queue.put_nowait(message.get("event"))
                except Full:
                    # A full receiver must fail explicitly rather than lose readings.
                    self.disconnect("La recepción de Home Assistant superó la capacidad de evaluación.")
            return
        pending = self.pending.get(message.get("request_id"))
        if pending is None or pending["future"].done():
            return
        if kind == "result_chunk":
            if message.get("entity_id"):
                pending["groups"].setdefault(message["entity_id"], []).extend(message.get("items", []))
            else:
                pending["items"].extend(message.get("items", []))
        elif kind == "result":
            if not message.get("success"):
                error = message.get("error") or "Home Assistant no pudo completar la consulta."
                # Older integrations do not advertise capabilities; recognize only their
                # explicit unsupported-operation reply, never a timeout or disconnect.
                error_type = BridgeOperationUnsupported if error == "La operación solicitada no está permitida." else BridgeUnavailable
                pending["future"].set_exception(error_type(error))
            else:
                data = ({**(message.get("data") or {}), "entities": pending["items"]}
                        if pending["operation"] == "entity_catalog" else
                        list(pending["groups"].values()) if pending["groups"] else
                        pending["items"] if pending["items"] else message.get("data"))
                pending["future"].set_result(data)

    def disconnect(self, reason="Se perdió la conexión con la integración de Home Assistant."):
        self.connected = False
        for pending in self.pending.values():
            if not pending["future"].done():
                pending["future"].set_exception(BridgeUnavailable(reason))
        for queue in self.subscriptions.values():
            # Make space for the failure notification if this subscriber overflowed.
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(BridgeUnavailable(reason))

    async def subscribe(self, identifier, entity_ids, queue):
        self.subscriptions[identifier] = queue
        try:
            await self.request("subscribe", {"subscription_id": identifier, "entity_ids": entity_ids})
        except BaseException:
            self.subscriptions.pop(identifier, None)
            raise

    async def unsubscribe(self, identifier):
        self.subscriptions.pop(identifier, None)
        if self.connected:
            await self.request("unsubscribe", {"subscription_id": identifier})


class HomeAssistantBridge:
    def __init__(self, registration_path: Path):
        self.path = registration_path
        self.lock = Lock()
        self.connection = None
        self.registration = None
        if self.path.exists():
            # A damaged registration must not reopen automatic pairing.
            self.registration = json.loads(self.path.read_text(encoding="utf-8"))

    def register(self, bridge_id, token):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", bridge_id) or not 32 <= len(token) <= 128:
            raise ValueError("La identificación del vínculo no es válida.")
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.lock:
            if self.registration is not None:
                if (self.registration.get("bridge_id") != bridge_id
                        or not hmac.compare_digest(self.registration.get("token_hash", ""), digest)):
                    raise PermissionError("Artifact ya está vinculado a otra integración de Home Assistant.")
                return
            registration = {"bridge_id": bridge_id, "token_hash": digest}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(registration), encoding="utf-8")
            temporary.replace(self.path)
            self.registration = registration

    def authorized(self, token):
        with self.lock:
            return bool(self.registration and hmac.compare_digest(
                self.registration.get("token_hash", ""), hashlib.sha256(token.encode()).hexdigest()))

    def status(self):
        with self.lock:
            return {"bridge_configured": self.registration is not None,
                    "bridge_connected": bool(self.connection and self.connection.connected)}

    async def attach(self, socket):
        connection = BridgeConnection(socket)
        with self.lock:
            previous, self.connection = self.connection, connection
        if previous is not None:
            previous.disconnect()
            with suppress(RuntimeError):
                await previous.socket.close(code=1001)
        return connection

    def detach(self, connection):
        connection.disconnect()
        with self.lock:
            if self.connection is connection:
                self.connection = None

    def _connection(self):
        with self.lock:
            connection = self.connection
        if connection is None or not connection.connected:
            if not self.status()["bridge_configured"]:
                raise BridgeUnavailable("Integración no configurada. Añade Artifact en las integraciones de Home Assistant.")
            raise BridgeUnavailable("La integración de Home Assistant está desconectada. Comprueba que esté cargada y que Artifact sea accesible.")
        return connection

    def require_connected(self):
        self._connection()

    @staticmethod
    def _wait(connection, coroutine, timeout=125, stop_event=None):
        future = asyncio.run_coroutine_threadsafe(coroutine, connection.loop)
        try:
            if stop_event is None:
                return future.result(timeout)
            deadline = monotonic() + timeout
            while not stop_event.is_set():
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise FutureTimeout()
                try:
                    return future.result(min(.25, remaining))
                except FutureTimeout:
                    if future.done():
                        raise
            future.cancel()
            raise BridgeUnavailable("La recepción se detuvo por solicitud del usuario.")
        except FutureTimeout:
            future.cancel()
            raise BridgeUnavailable("La integración de Home Assistant no respondió a tiempo.") from None

    def request(self, operation, arguments):
        connection = self._connection()
        return self._wait(connection, connection.request(operation, arguments))

    def entity_catalog(self):
        try:
            return self.request("entity_catalog", {})
        except BridgeOperationUnsupported:
            return {"entities": self.request("states", {}), "areas_available": False}

    def history_transport(self, path):
        from urllib.parse import parse_qs, unquote, urlsplit
        if path == "/states":
            return self.request("states", {})
        parsed = urlsplit(path)
        query = parse_qs(parsed.query)
        return self.request("history", {"start": unquote(parsed.path.rsplit("/", 1)[-1]),
                                        "end": query["end_time"][0],
                                        "entity_ids": query["filter_entity_id"][0].split(",")})

    def listen(self, entity_ids, accept, stop_event, on_connected=None):
        connection = self._connection()
        entity_ids = list(entity_ids)
        identifiers, queue = [], Queue(maxsize=5000)
        try:
            states = self._wait(connection, connection.request("states", {}), stop_event=stop_event)
            if not isinstance(states, list):
                raise ValueError("Home Assistant no devolvió una lista de entidades válida.")
            available = {state.get("entity_id") for state in states if isinstance(state, dict)}
            # The caller supplies the CSV selection; HA only determines which of those it can publish.
            entity_ids = [entity_id for entity_id in entity_ids if entity_id in available]
            if not entity_ids:
                raise ValueError("No hay entidades seleccionadas del historial disponibles en Home Assistant.")
            # Keep compatibility with integrations limiting each subscription to 1000 IDs.
            for index in range(0, len(entity_ids), 1000):
                if stop_event.is_set():
                    break
                identifier = uuid4().hex
                identifiers.append(identifier)
                self._wait(connection, connection.subscribe(identifier, entity_ids[index:index + 1000], queue), stop_event=stop_event)
            if on_connected is not None and not stop_event.is_set():
                on_connected()
            while not stop_event.is_set():
                try:
                    event = queue.get(timeout=.25)
                except Empty:
                    continue
                if isinstance(event, Exception):
                    raise event
                accept({"type": "event", "event": event})
        finally:
            async def unsubscribe_all():
                await asyncio.gather(*(connection.unsubscribe(identifier) for identifier in identifiers), return_exceptions=True)
            try:
                self._wait(connection, unsubscribe_all(), timeout=5)
            except (BridgeUnavailable, RuntimeError):
                pass

    async def close(self):
        with self.lock:
            connection = self.connection
        if connection:
            self.detach(connection)
            with suppress(RuntimeError):
                await connection.socket.close(code=1001)
