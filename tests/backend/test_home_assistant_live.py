import asyncio
from copy import deepcopy
import json
from pathlib import Path
from queue import Queue
import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import patch

import pandas as pd

from app.database.flows import FlowStore
from app.flow.evaluacion import CausalFlowRunner, FlowEvaluationJobs
from app.services.home_assistant_live import HomeAssistantEventSource
from app.services.home_assistant_history import resolve_connection


def flow():
    return {"name": "Live", "nodes": [{"id": "input", "type": "input", "config": {}, "position": {}},
              {"id": "output", "type": "output", "config": {}, "position": {}}],
            "edges": [{"id": "in-out", "source": "input", "target": "output"}]}


class LiveEventsTests(unittest.TestCase):
    def setUp(self):
        self.entities = pd.DataFrame([{"entity_id": "binary_sensor.door", "sensor_kind": "binary"},
                                      {"entity_id": "sensor.temperature", "sensor_kind": "continuous"}])
        self.start = pd.Timestamp("2026-01-01T00:00:00Z")

    def source(self, **changes):
        return HomeAssistantEventSource("http://ha.local:8123", "secret", self.entities, self.start, **changes)

    def message(self, at, state="on", entity="binary_sensor.door", old="off"):
        return {"type": "event", "event": {"event_type": "state_changed", "data": {
            "entity_id": entity, "old_state": {"state": old},
            "new_state": {"state": state, "last_changed": at}}}}

    def test_normalizes_known_entities_and_rejects_duplicates_historical_and_attribute_only_events(self):
        source = self.source()
        source.accept(self.message("2026-01-01T00:00:01Z"))
        row = source.queue.get_nowait()
        self.assertEqual(row["value"], 1)
        self.assertEqual(row["_raw_index"], 0)
        source.accept(self.message("2026-01-01T00:00:01Z"))
        source.accept(self.message("2025-12-31T23:00:00Z"))
        source.accept(self.message("2026-01-01T00:00:02Z", old="on"))
        source.accept(self.message("2026-01-01T00:00:02Z", entity="sensor.unknown"))
        source.accept(self.message("2026-01-01T00:00:02Z", "20.5", "sensor.temperature"))
        self.assertEqual(source.queue.get_nowait()["value"], 20.5)
        self.assertEqual(source.status()["received"], 2)
        self.assertEqual(source.status()["ignored"], 4)
        self.assertNotIn("secret", repr(source))

    def test_unknown_entity_selection_and_buffer_overflow_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "historial de referencia"):
            self.source(entity_ids=["sensor.unknown"])
        source = self.source()
        source.queue = Queue(maxsize=1)
        source.accept(self.message("2026-01-01T00:00:01Z"))
        with self.assertRaisesRegex(ValueError, "ritmo de entrada"):
            source.accept(self.message("2026-01-01T00:00:02Z", state="off", old="on"))

    def test_batches_sort_events_and_group_equal_timestamps(self):
        source = self.source()
        def receive():
            source.accept(self.message("2026-01-01T00:00:02Z"))
            source.accept(self.message("2026-01-01T00:00:01Z", "20", "sensor.temperature"))
            source.accept(self.message("2026-01-01T00:00:02Z", "21", "sensor.temperature"))
            source.stop()
        with patch.object(source, "_receive", side_effect=receive):
            groups = list(source.batches())
        self.assertEqual([len(batch) for _, batch in groups], [1, 2])
        self.assertEqual(groups[0][0], pd.Timestamp("2026-01-01T00:00:01Z"))

    def test_streamed_causal_runner_matches_csv_results_without_reinitializing_each_event(self):
        development = pd.DataFrame([{"entity_id": "binary_sensor.door", "timestamp_dt": self.start,
                                     "state": "off", "sensor_kind": "binary", "_raw_index": 0}])
        future = pd.DataFrame([{"entity_id": "binary_sensor.door", "timestamp_dt": self.start + pd.Timedelta(seconds=i),
                               "state": "on", "sensor_kind": "binary", "_raw_index": i} for i in (1, 2, 3)])
        regular, outputs, _ = CausalFlowRunner(None, None, None).evaluate(flow(), development, future, self.entities, {}, {})
        blocks = []
        def sink(items):
            blocks.append(deepcopy(items))
        summary, _, snapshot = CausalFlowRunner(None, None, None).evaluate(
            flow(), development, future.iloc[:0], self.entities, {}, {}, result_sink=sink,
            batch_source=future.groupby("timestamp_dt", sort=True), source_identity="live-session")
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["by_final_decision"], regular["by_final_decision"])
        self.assertEqual([item["state"] for block in blocks for item in block["output"]], [item["state"] for item in outputs["output"]])
        self.assertEqual(snapshot["execution_mode"], "causal_live")
        self.assertEqual(snapshot["source"]["rows"], 3)
        self.assertEqual(snapshot["source"]["fingerprint"], "live-session")

    def test_websocket_authenticates_and_retries_without_discarding_received_events(self):
        source = self.source()
        messages = iter([{"type": "auth_required"}, {"type": "auth_ok"}, {"type": "result", "id": 1, "success": True},
                         self.message("2026-01-01T00:00:01Z")])
        sent = []
        attempts = []
        class Socket:
            async def recv(self):
                try:
                    return json.dumps(next(messages))
                except StopIteration:
                    raise ConnectionError("private handshake data") from None
            async def send(self, value):
                sent.append(json.loads(value))
        class Connection:
            def __init__(self, uri, **kwargs):
                self.uri = uri
            async def __aenter__(self):
                attempts.append(True)
                if len(attempts) == 2:
                    source.stop()
                return Socket()
            async def __aexit__(self, *args):
                return False
        with patch("websockets.asyncio.client.connect", Connection):
            source._receive()
        self.assertEqual(sent[0], {"type": "auth", "access_token": "secret"})
        self.assertEqual(sent[1]["event_type"], "state_changed")
        self.assertEqual(source.queue.qsize(), 1)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(source.status()["interruptions"], 1)
        self.assertEqual(source.status()["status"], "stopped")
        self.assertNotIn("private", source.status()["last_error"])
        self.assertEqual(source._token, "")

    def test_server_token_cannot_be_reused_for_another_url(self):
        with patch.dict("os.environ", {"HOME_ASSISTANT_URL": "http://ha.local", "HOME_ASSISTANT_TOKEN": "secret"}):
            self.assertEqual(resolve_connection("", ""), ("http://ha.local", "secret"))
            with self.assertRaises(ValueError):
                resolve_connection("http://another.host", "")

    def test_missing_classifier_model_blocks_start_with_an_actionable_notice(self):
        from app.services.diagnostico import DiagnosticWorkbench
        with tempfile.TemporaryDirectory() as folder:
            workbench = DiagnosticWorkbench(Path(folder))
            with self.assertRaisesRegex(ValueError, "clasificador.*modelo disponible"):
                workbench.require_classifier_model()
            with self.assertRaisesRegex(ValueError, "Selecciona otro modelo o desactiva"):
                workbench.require_classifier_model("a" * 32)
            history = pd.DataFrame([{"entity_id": "binary_sensor.door", "timestamp_dt": self.start,
                                     "state": "off", "sensor_kind": "binary", "_raw_index": 0}])
            selected = flow()
            selected["nodes"].insert(1, {"id": "classifier", "type": "classifier", "config": {}, "position": {}})
            selected["edges"] = [{"id": "in-classifier", "source": "input", "target": "classifier"},
                                 {"id": "classifier-out", "source": "classifier", "target": "output"}]
            with self.assertRaisesRegex(ValueError, "Selecciona un modelo válido"):
                CausalFlowRunner(None, None, workbench).evaluate(selected, history, history.iloc[:0], self.entities, {}, {}, batch_source=iter(()))

    def test_stopping_live_job_drains_and_saves_received_results(self):
        with tempfile.TemporaryDirectory() as folder:
            store = FlowStore(Path(folder) / "flows.sqlite")
            saved = store.create(flow())
            jobs = FlowEvaluationJobs(store)
            source = self.source()
            finished = Event()
            received = Event()
            development = pd.DataFrame([{"entity_id": "binary_sensor.door", "timestamp_dt": self.start,
                                         "state": "off", "sensor_kind": "binary", "_raw_index": 0}])
            def receive():
                source.accept(self.message("2026-01-01T00:00:01Z"))
                received.set()
                source.stop_event.wait(5)
            def evaluate(progress, sink):
                result = CausalFlowRunner(None, None, None).evaluate(saved, development, development.iloc[:0], self.entities, {}, {},
                    progress=progress, result_sink=sink, batch_source=source.batches(), source_identity="session")
                finished.set()
                return result
            with patch.object(source, "_receive", side_effect=receive):
                job = jobs.start(saved["flow_id"], evaluate, live_source=source)
                self.assertEqual(jobs.get(job["job_id"])["mode"], "home_assistant")
                self.assertTrue(received.wait(5))
                jobs.stop(job["job_id"])
                self.assertTrue(finished.wait(5))
                # Wait for persistence, not a fixed sleep.
                for _ in range(100):
                    status = jobs.get(job["job_id"])
                    if status["status"] != "running":
                        break
                    Event().wait(.01)
            self.assertEqual(status["status"], "completed", status)
            self.assertEqual(status["run"]["summary"]["candidate_count"], 1)
            self.assertEqual(store.results(status["run"]["run_id"])["total"], 1)
            self.assertNotIn(job["job_id"], jobs.sources)

    def test_idle_live_job_with_disabled_classifier_waits_until_the_user_stops(self):
        from app.services.diagnostico import DiagnosticWorkbench
        connected = Event()
        class Bridge:
            def listen(self, entities, accept, stop_event, on_connected):
                on_connected()
                connected.set()
                stop_event.wait()
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            store = FlowStore(directory / "flows.sqlite")
            selected = flow()
            selected["nodes"].insert(1, {"id": "classifier", "type": "classifier", "config": {"enabled": False}, "position": {}})
            selected["edges"] = [{"id": "in-classifier", "source": "input", "target": "classifier"},
                                 {"id": "classifier-out", "source": "classifier", "target": "output"}]
            saved = store.create(selected)
            jobs = FlowEvaluationJobs(store)
            source = self.source(bridge=Bridge())
            history = pd.DataFrame([{"entity_id": "binary_sensor.door", "timestamp_dt": self.start,
                                     "state": "off", "sensor_kind": "binary", "_raw_index": 0}])
            workbench = DiagnosticWorkbench(directory / "diagnostics")
            def evaluate(progress, sink):
                return CausalFlowRunner(None, None, workbench).evaluate(saved, history, history.iloc[:0], self.entities, {}, {},
                    progress=progress, result_sink=sink, batch_source=source.batches())
            job = jobs.start(saved["flow_id"], evaluate, live_source=source)
            try:
                self.assertTrue(connected.wait(5), jobs.get(job["job_id"]))
                # Cross several empty batch reads; no event is needed to stay active.
                Event().wait(.8)
                status = jobs.get(job["job_id"])
                self.assertEqual(status["status"], "running", status)
                self.assertEqual(status["connection"]["status"], "connected")
                self.assertEqual(status["processed"], 0)
            finally:
                jobs.stop(job["job_id"])
            for _ in range(200):
                status = jobs.get(job["job_id"])
                if status["status"] != "running":
                    break
                Event().wait(.01)
            self.assertEqual(status["status"], "completed", status)
            self.assertEqual(status["run"]["summary"]["candidate_count"], 0)

    def test_native_bridge_reconnects_in_the_same_session_and_stop_interrupts_backoff(self):
        from app.services.home_assistant_bridge import BridgeUnavailable
        first, retried = Event(), Event()
        attempts = []
        message = self.message("2026-01-01T00:00:01Z")
        class Bridge:
            def listen(self, entities, accept, stop_event, on_connected):
                attempts.append(True)
                on_connected()
                if len(attempts) == 1:
                    accept(message)
                    first.set()
                    raise BridgeUnavailable("Se cortó el vínculo.")
                retried.set()
                stop_event.wait()
        source = self.source(bridge=Bridge())
        receiver = Thread(target=source._receive)
        receiver.start()
        try:
            self.assertTrue(first.wait(2))
            self.assertTrue(retried.wait(3))
            self.assertEqual(source.status()["status"], "connected")
            self.assertEqual(source.status()["reconnections"], 1)
            self.assertEqual(source.status()["received"], 1)
            self.assertFalse(source.stop_event.is_set())
        finally:
            source.stop()
            receiver.join(2)
        self.assertFalse(receiver.is_alive())
        self.assertEqual(source._token, "")
        disconnected = self.source()
        retrying = Event()
        def fail():
            retrying.set()
            raise ConnectionError("private transport details")
        with patch.object(disconnected, "_listen", side_effect=fail):
            receiver = Thread(target=disconnected._receive)
            receiver.start()
            self.assertTrue(retrying.wait(2))
            disconnected.stop()
            receiver.join(2)
            self.assertFalse(receiver.is_alive())


class LocalWebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_websocket_client_reconnects_and_receives_changes_until_stopped(self):
        from websockets.asyncio.server import serve
        entities = pd.DataFrame([{"entity_id": "binary_sensor.door", "sensor_kind": "binary"}])
        attempts = []
        async def server(socket):
            attempts.append(True)
            index = len(attempts)
            await socket.send(json.dumps({"type": "auth_required"}))
            self.assertEqual(json.loads(await socket.recv()), {"type": "auth", "access_token": "local-test-token"})
            await socket.send(json.dumps({"type": "auth_ok"}))
            self.assertEqual(json.loads(await socket.recv())["event_type"], "state_changed")
            await socket.send(json.dumps({"type": "result", "id": 1, "success": True}))
            await socket.send(json.dumps({"type": "event", "event": {"event_type": "state_changed", "data": {
                "entity_id": "binary_sensor.door", "old_state": {"state": "off" if index == 1 else "on"},
                "new_state": {"state": "on" if index == 1 else "off", "last_changed": f"2026-01-01T00:00:0{index}Z"}}}}))
            if index == 1:
                await socket.close()
            else:
                await socket.wait_closed()
        async with serve(server, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            source = HomeAssistantEventSource(f"http://127.0.0.1:{port}", "local-test-token", entities,
                                               pd.Timestamp("2026-01-01T00:00:00Z"))
            def consume():
                groups = []
                for item in source.batches():
                    groups.append(item)
                    if sum(len(batch) for _, batch in groups) >= 2:
                        source.stop()
                return groups
            try:
                batches = await asyncio.wait_for(asyncio.to_thread(consume), timeout=8)
            finally:
                source.stop()
        self.assertEqual(sum(len(batch) for _, batch in batches), 2)
        self.assertEqual(batches[0][1].iloc[0]["state"], "on")
        self.assertEqual(source.status()["reconnections"], 1)
        self.assertEqual(source.status()["status"], "stopped")
        self.assertEqual(source._token, "")

    async def test_authentication_failure_remains_visible_and_waits_for_manual_stop(self):
        from websockets.asyncio.server import serve
        async def server(socket):
            await socket.send(json.dumps({"type": "auth_required"}))
            await socket.recv()
            await socket.send(json.dumps({"type": "auth_invalid"}))
        async with serve(server, "127.0.0.1", 0) as listener:
            source = HomeAssistantEventSource(f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}", "invalid-test-token",
                pd.DataFrame([{"entity_id": "binary_sensor.door", "sensor_kind": "binary"}]), pd.Timestamp("2026-01-01T00:00:00Z"))
            reception = asyncio.create_task(asyncio.to_thread(lambda: list(source.batches())))
            try:
                for _ in range(200):
                    if source.status()["last_error"]:
                        break
                    await asyncio.sleep(.01)
                self.assertFalse(reception.done())
                self.assertEqual(source.status()["status"], "reconnecting")
                self.assertIn("rechazó el token", source.status()["last_error"])
            finally:
                source.stop()
            batches = await asyncio.wait_for(reception, timeout=3)
        self.assertEqual(batches, [])
        self.assertIsNone(source.error)
        self.assertNotIn("invalid-test-token", json.dumps(source.status()))


if __name__ == "__main__":
    unittest.main()
