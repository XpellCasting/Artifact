"""Receive state changes, keeping a bounded queue and chronological batches."""
from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic

import pandas as pd

from app.services.estados_validos import valid_state
from app.services.historial.cargar_historial import ESTADOS_ACTIVOS


class HomeAssistantEventSource:
    def __init__(self, entities, development_end, entity_ids=(), *, bridge):
        bridge.require_connected()
        self.bridge = bridge
        self.entities = {str(row["entity_id"]): dict(row) for row in entities.to_dict("records")
                         if row.get("sensor_kind") in {"continuous", "binary", "multistate"}}
        selected = set(entity_ids) or set(self.entities)
        if not selected or selected - self.entities.keys():
            raise ValueError("Selecciona entidades que estén presentes en el historial de referencia.")
        self.selected = selected
        self.development_end = pd.Timestamp(development_end)
        self.queue = Queue(maxsize=5000)
        self.stop_event = Event()
        self.lock = Lock()
        self.error = None
        self.stats = {"status": "connecting", "received": 0, "ignored": 0, "late": 0, "last_event_at": None,
                      "interruptions": 0, "reconnections": 0, "last_error": None, "retry_seconds": None}
        self._has_connected = False
        self.last_seen = {}
        self.sequence = 0

    def status(self):
        with self.lock:
            return {**self.stats, "error": self.error, "buffered": self.queue.qsize()}

    def stop(self):
        with self.lock:
            self.stop_event.set()
            if self.stats["status"] not in {"failed", "stopped"}:
                self.stats["status"] = "stopping"

    def _connected(self):
        with self.lock:
            if not self.stop_event.is_set():
                if self._has_connected:
                    self.stats["reconnections"] += 1
                self._has_connected = True
                self.stats.update(status="connected", retry_seconds=None)

    def accept(self, message):
        event = message.get("event") or {}
        data = event.get("data") or {}
        entity_id = data.get("entity_id")
        state = data.get("new_state")
        old_state = data.get("old_state") or {}
        if (message.get("type") != "event" or event.get("event_type") != "state_changed"
                or entity_id not in self.selected or not isinstance(state, dict)
                or old_state.get("state") == state.get("state")):
            with self.lock:
                self.stats["ignored"] += 1
            return
        at = pd.to_datetime(state.get("last_changed"), utc=True, errors="coerce")
        if pd.isna(at) or at <= self.development_end or at <= self.last_seen.get(entity_id, self.development_end):
            with self.lock:
                self.stats["ignored"] += 1
            return
        metadata = self.entities[entity_id]
        text = str(state.get("state", "")).strip()
        kind = metadata["sensor_kind"]
        value = (pd.to_numeric(text, errors="coerce") if kind == "continuous" else
                 float(text.lower() in ESTADOS_ACTIVOS) if kind == "binary" and valid_state(text) else float("nan"))
        row = {**metadata, "entity_id": entity_id, "state": text, "value": value,
               "timestamp_dt": at, "timestamp": at.isoformat(), "_raw_index": self.sequence,
               "is_valid_value": valid_state(text), "reporting_mode": "on_change"}
        with self.lock:
            if self.stop_event.is_set():
                return
            try:
                self.queue.put_nowait(row)
            except Full:
                raise ValueError("La evaluación no alcanza el ritmo de entrada. Se detuvo la recepción para evitar perder eventos silenciosamente.") from None
            self.sequence += 1
            self.last_seen[entity_id] = at
            self.stats.update(received=self.sequence, last_event_at=at.isoformat())

    def _receive(self):
        delay = 1
        try:
            while not self.stop_event.is_set():
                try:
                    self.bridge.listen(self.selected, self.accept, self.stop_event, on_connected=self._connected)
                    if not self.stop_event.is_set():
                        raise ConnectionError("Home Assistant closed the reception")
                except Exception as exc:
                    if self.stop_event.is_set():
                        break
                    # Keep the same evaluation and its causal state while reconnecting.
                    # Library errors may contain private handshake details.
                    detail = str(exc) if isinstance(exc, ValueError) else "Se perdió la conexión con Home Assistant."
                    with self.lock:
                        if self.stats["status"] == "connected":
                            delay = 1
                            self.stats["interruptions"] += 1
                        self.stats.update(status="reconnecting", last_error=detail, retry_seconds=delay)
                    if self.stop_event.wait(delay):
                        break
                    delay = min(30, delay * 2)
        finally:
            with self.lock:
                self.stats.update(status="stopped", retry_seconds=None)

    def batches(self):
        Thread(target=self._receive, daemon=True, name="home-assistant-events").start()
        last_at = self.development_end
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                first = self.queue.get(timeout=.25)
            except Empty:
                continue
            rows = [first]
            until = monotonic() + .25
            while len(rows) < 5000:
                try:
                    rows.append(self.queue.get(timeout=max(0, until - monotonic()) if not self.stop_event.is_set() else 0))
                except Empty:
                    break
            frame = pd.DataFrame(rows).sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable")
            late = frame["timestamp_dt"] <= last_at
            with self.lock:
                self.stats["late"] += int(late.sum())
            frame = frame[~late]
            for at, batch in frame.groupby("timestamp_dt", sort=True):
                yield at, batch
                last_at = at
