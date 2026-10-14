import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.services.home_assistant_bridge import HomeAssistantBridge, BridgeConnection, BridgeUnavailable


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


class BridgeProtocolTests(unittest.IsolatedAsyncioTestCase):
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
