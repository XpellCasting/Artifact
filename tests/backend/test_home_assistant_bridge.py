import asyncio
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from app.services.home_assistant_bridge import HomeAssistantBridge, BridgeConnection, BridgeUnavailable, BridgeOperationUnsupported


class BridgeRegistrationTests(unittest.TestCase):
    def test_pairing_survives_restart_stores_only_hash_and_rejects_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bridge.json"
            bridge = HomeAssistantBridge(path)
            token = "private-bridge-token-" + "x" * 32
            bridge.register("integration-1", token)
            self.assertNotIn(token, path.read_text())
            restored = HomeAssistantBridge(path)
            self.assertTrue(restored.authorized(token))
            self.assertFalse(restored.authorized("wrong-token"))
            restored.register("integration-1", token)
            with self.assertRaises(PermissionError):
                restored.register("another-integration", "y" * 32)
            with self.assertRaises(PermissionError):
                restored.register("integration-1", "y" * 32)
            self.assertFalse(restored.status()["bridge_connected"])

    def test_disconnected_bridge_fails_without_falling_back_to_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge = HomeAssistantBridge(Path(folder) / "bridge.json")
            with self.assertRaises(BridgeUnavailable):
                bridge.request("states", {})

    def test_history_transport_preserves_entity_filters_and_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge = HomeAssistantBridge(Path(folder) / "bridge.json")
            with patch.object(bridge, "request", return_value=[]) as request:
                bridge.history_transport("/history/period/2026-01-01T00%3A00%3A00%2B00%3A00?end_time=2026-01-02T00%3A00%3A00%2B00%3A00&filter_entity_id=sensor.a%2Csensor.b")
                self.assertEqual(request.call_args.args, ("history", {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-02T00:00:00+00:00", "entity_ids": ["sensor.a", "sensor.b"]}))

    def test_old_integration_falls_back_to_states_only_for_unsupported_operation(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge = HomeAssistantBridge(Path(folder) / "bridge.json")
            rows = [{"entity_id": "sensor.a"}]
            with patch.object(bridge, "request", side_effect=[BridgeOperationUnsupported("old integration"), rows]) as request:
                self.assertEqual(bridge.entity_catalog(), {"entities": rows, "areas_available": False})
                self.assertEqual([call.args[0] for call in request.call_args_list], ["entity_catalog", "states"])
            with patch.object(bridge, "request", side_effect=BridgeUnavailable("disconnected")) as request:
                with self.assertRaises(BridgeUnavailable):
                    bridge.entity_catalog()
                self.assertEqual(request.call_count, 1)

    def test_missing_and_disconnected_link_have_distinct_notices(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge = HomeAssistantBridge(Path(folder) / "bridge.json")
            with self.assertRaisesRegex(BridgeUnavailable, "no configurada"):
                bridge.require_connected()
            bridge.register("integration", "x" * 32)
            with self.assertRaisesRegex(BridgeUnavailable, "desconectada"):
                bridge.require_connected()


class BridgeProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunked_catalog_preserves_metadata_and_legacy_errors_are_identified(self):
        class Socket:
            async def send_json(self, message):
                identifier = message["request_id"]
                if message["operation"] == "entity_catalog":
                    for index in range(10):
                        connection.receive({"type": "result_chunk", "request_id": identifier, "items": [
                            {"entity_id": f"sensor.entity_{index * 500 + offset}", "area_id": None, "area_name": None}
                            for offset in range(500)]})
                    connection.receive({"type": "result", "request_id": identifier, "success": True,
                                        "data": {"areas_available": True}})
                else:
                    connection.receive({"type": "result", "request_id": identifier, "success": False,
                                        "error": "La operación solicitada no está permitida."})
        connection = BridgeConnection(Socket())
        result = await connection.request("entity_catalog", {})
        self.assertTrue(result["areas_available"])
        self.assertEqual(len(result["entities"]), 5000)
        self.assertEqual(result["entities"][-1]["entity_id"], "sensor.entity_4999")
        with self.assertRaises(BridgeOperationUnsupported):
            await connection.request("unknown", {})

    async def test_subscription_uses_only_reference_entities_published_by_home_assistant(self):
        selection = ["sensor.temperature", "light.csv_only", "media_player.csv_only"]
        requests = []
        stopped = Event()
        class Socket:
            async def send_json(self, message):
                requests.append(message)
                data = ([{"entity_id": "sensor.temperature"}, {"entity_id": "sensor.outside_reference"}]
                        if message["operation"] == "states" else [])
                self.connection.receive({"type": "result", "request_id": message["request_id"],
                                         "success": True, "data": data})
            async def close(self, **kwargs):
                pass
        with tempfile.TemporaryDirectory() as folder:
            bridge = HomeAssistantBridge(Path(folder) / "bridge.json")
            socket = Socket()
            socket.connection = await bridge.attach(socket)
            try:
                await asyncio.wait_for(asyncio.to_thread(bridge.listen, selection, lambda event: None,
                                                        stopped, stopped.set), timeout=3)
                self.assertEqual([request["operation"] for request in requests], ["states", "subscribe", "unsubscribe"])
                self.assertEqual(requests[1]["arguments"]["entity_ids"], ["sensor.temperature"])
                self.assertEqual(selection, ["sensor.temperature", "light.csv_only", "media_player.csv_only"])
                self.assertFalse(socket.connection.subscriptions)
            finally:
                stopped.set()
                await bridge.close()

    async def test_no_matching_reference_entities_does_not_subscribe_to_unselected_entities(self):
        requests = []
        class Socket:
            async def send_json(self, message):
                requests.append(message["operation"])
                self.connection.receive({"type": "result", "request_id": message["request_id"],
                                         "success": True, "data": [{"entity_id": "sensor.outside_reference"}]})
            async def close(self, **kwargs):
                pass
        with tempfile.TemporaryDirectory() as folder:
            bridge = HomeAssistantBridge(Path(folder) / "bridge.json")
            socket = Socket()
            socket.connection = await bridge.attach(socket)
            try:
                with self.assertRaisesRegex(ValueError, "No hay entidades seleccionadas del historial"):
                    await asyncio.wait_for(asyncio.to_thread(bridge.listen, ["sensor.csv_only"],
                                                            lambda event: None, Event()), timeout=3)
                self.assertEqual(requests, ["states"])
            finally:
                await bridge.close()

    async def test_5000_entities_use_bounded_subscriptions_and_all_are_removed_on_stop(self):
        connected = Event()
        stopped = Event()
        subscriptions, unsubscribed, received = [], [], []
        class Socket:
            async def send_json(inner, message):
                if message["operation"] == "states":
                    inner.connection.receive({"type": "result", "request_id": message["request_id"],
                                              "success": True, "data": [{"entity_id": entity_id} for entity_id in ids]})
                    return
                if message["operation"] == "subscribe":
                    subscribed_ids = message["arguments"]["entity_ids"]
                    self.assertLessEqual(len(subscribed_ids), 1000)
                    subscriptions.extend(subscribed_ids)
                else:
                    unsubscribed.append(message["arguments"]["subscription_id"])
                inner.connection.receive({"type": "result", "request_id": message["request_id"], "success": True, "data": []})
            async def close(self, **kwargs):
                pass
        with tempfile.TemporaryDirectory() as folder:
            bridge = HomeAssistantBridge(Path(folder) / "bridge.json")
            socket = Socket()
            socket.connection = await bridge.attach(socket)
            ids = [f"sensor.entity_{index}" for index in range(5000)]
            listener = asyncio.create_task(asyncio.to_thread(bridge.listen, ids, received.append, stopped, connected.set))
            try:
                self.assertTrue(await asyncio.to_thread(connected.wait, 3))
                self.assertEqual(subscriptions, ids)
                self.assertEqual(len(socket.connection.subscriptions), 5)
                socket.connection.receive({"type": "event", "subscription_id": list(socket.connection.subscriptions)[-1],
                    "event": {"data": {"entity_id": "sensor.entity_4999"}}})
                for _ in range(100):
                    if received:
                        break
                    await asyncio.sleep(.01)
                self.assertEqual(received[0]["event"]["data"]["entity_id"], "sensor.entity_4999")
            finally:
                stopped.set()
                await asyncio.wait_for(listener, 3)
                await bridge.close()
            self.assertEqual(len(unsubscribed), 5)
            self.assertFalse(socket.connection.subscriptions)

    async def test_live_receiver_resubscribes_after_the_native_bridge_is_replaced(self):
        import pandas as pd
        from app.services.home_assistant_live import HomeAssistantEventSource
        class Socket:
            async def send_json(self, message):
                data = [{"entity_id": "binary_sensor.door"}] if message["operation"] == "states" else []
                self.connection.receive({"type": "result", "request_id": message["request_id"], "success": True, "data": data})
            async def close(self, **kwargs):
                pass
        async def until(condition):
            for _ in range(400):
                if condition():
                    return
                await asyncio.sleep(.01)
            self.fail("The native reception did not reach the expected state")
        with tempfile.TemporaryDirectory() as folder:
            bridge = HomeAssistantBridge(Path(folder) / "bridge.json")
            first_socket = Socket()
            first = first_socket.connection = await bridge.attach(first_socket)
            source = HomeAssistantEventSource(pd.DataFrame([
                {"entity_id": "binary_sensor.door", "sensor_kind": "binary"}]),
                pd.Timestamp("2026-01-01T00:00:00Z"), bridge=bridge)
            reception = asyncio.create_task(asyncio.to_thread(lambda: list(source.batches())))
            def deliver(connection, second, state, old):
                connection.receive({"type": "event", "subscription_id": next(iter(connection.subscriptions)),
                    "event": {"event_type": "state_changed", "data": {"entity_id": "binary_sensor.door",
                        "old_state": {"state": old}, "new_state": {"state": state, "last_changed": f"2026-01-01T00:00:0{second}Z"}}}})
            try:
                await until(lambda: source.status()["status"] == "connected")
                deliver(first, 1, "on", "off")
                await until(lambda: source.status()["received"] == 1)
                bridge.detach(first)
                second_socket = Socket()
                second = second_socket.connection = await bridge.attach(second_socket)
                await until(lambda: source.status()["reconnections"] == 1)
                deliver(second, 2, "off", "on")
                await until(lambda: source.status()["received"] == 2)
                self.assertFalse(reception.done())
                self.assertIsNone(source.error)
                self.assertEqual(source.status()["interruptions"], 1)
            finally:
                source.stop()
                batches = await asyncio.wait_for(reception, timeout=3)
                await bridge.close()
            self.assertEqual(sum(len(batch) for _, batch in batches), 2)

    async def test_manual_stop_cancels_a_pending_native_subscription(self):
        sent = asyncio.Event()
        class Socket:
            async def send_json(self, message):
                if message["operation"] == "subscribe":
                    sent.set()
                else:
                    data = [{"entity_id": "sensor.a"}] if message["operation"] == "states" else []
                    self.connection.receive({"type": "result", "request_id": message["request_id"], "success": True, "data": data})
            async def close(self, **kwargs):
                pass
        with tempfile.TemporaryDirectory() as folder:
            bridge = HomeAssistantBridge(Path(folder) / "bridge.json")
            socket = Socket()
            socket.connection = await bridge.attach(socket)
            stopped = Event()
            listener = asyncio.create_task(asyncio.to_thread(bridge.listen, ["sensor.a"], lambda event: None, stopped))
            try:
                await asyncio.wait_for(sent.wait(), timeout=2)
                stopped.set()
                with self.assertRaisesRegex(BridgeUnavailable, "solicitud del usuario"):
                    await asyncio.wait_for(listener, timeout=2)
                await asyncio.sleep(0)
                self.assertEqual(socket.connection.subscriptions, {})
            finally:
                stopped.set()
                await bridge.close()

    async def test_chunked_history_reassembles_before_completing_request(self):
        sent = asyncio.Queue()
        class Socket:
            async def send_json(self, message):
                sent.put_nowait(message)
        connection = BridgeConnection(Socket())
        request = asyncio.create_task(connection.request("history", {}))
        identifier = (await sent.get())["request_id"]
        for index in range(3):
            connection.receive({"type": "result_chunk", "request_id": identifier, "entity_id": "sensor.a", "items": [{"state": str(index)}]})
        self.assertFalse(request.done())
        connection.receive({"type": "result", "request_id": identifier, "success": True, "data": []})
        self.assertEqual(await request, [[{"state": "0"}, {"state": "1"}, {"state": "2"}]])

    async def test_disconnect_releases_pending_queries_and_subscribers(self):
        from queue import Queue
        sent = asyncio.Event()
        class Socket:
            async def send_json(self, message):
                sent.set()
        connection = BridgeConnection(Socket())
        queue = Queue()
        connection.subscriptions["one"] = queue
        request = asyncio.create_task(connection.request("states", {}))
        await sent.wait()
        connection.disconnect()
        with self.assertRaises(BridgeUnavailable):
            await request
        self.assertIsInstance(queue.get_nowait(), BridgeUnavailable)


if __name__ == "__main__":
    unittest.main()
