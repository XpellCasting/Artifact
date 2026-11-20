import asyncio
from contextlib import closing
from functools import partial
import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.confidence import create_confidence_router
from app.database.confidence_entities import ConfidenceEntityStore, confidence_entity_id
from app.database.flows import FlowStore
from app.flow.evaluacion import FlowEvaluationJobs
from app.flow.result_spool import FlowResultSpool
from app.services.home_assistant_bridge import HomeAssistantBridge


def reading(identity="one", confidence=.93, source="sensor.temperature"):
    return {"entity_id": source, "observation_id": identity, "execution_id": "execution",
            "timestamp": "2026-11-18T01:00:00+00:00", "confidence": confidence,
            "value": 24.5, "state": "24.5", "final_decision": "normal", "evidence": [],
            "confidence_kind": "operational_index", "confidence_calibrated": False, "evidence_coverage": 1.,
            "confidence_fusion": {"method": "uniform_v1", "active_filters": 2, "available_filters": 2}}


class ConfidenceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "flow.sqlite3"
        self.store = ConfidenceEntityStore(self.path)

    def test_stable_names_and_percentage_values_include_unknown_zero_and_one_hundred(self):
        self.assertEqual(confidence_entity_id("sensor.temperature"), "confianza.sensor_temperature")
        self.assertEqual(confidence_entity_id("binary_sensor.door"), "confianza.binary_sensor_door")
        initial = self.store.start("live", "flow", ["sensor.temperature"])[0]
        self.assertIsNone(initial["value"])
        self.assertTrue(initial["available"])
        for index, (confidence, expected) in enumerate(((0, 0), (.93, 93), (1, 100), (None, None))):
            row = self.store.record("live", [reading(str(index), confidence)])[0]
            self.assertEqual(row["value"], expected)
            self.assertEqual(row["entity_id"], initial["entity_id"])
            self.assertEqual(row["source_timestamp"], "2026-11-18T01:00:00+00:00")
        self.assertFalse(self.store.snapshot()["entities"][0]["confidence_calibrated"])

    def test_identical_confidence_keeps_last_change_but_tracks_each_reading(self):
        self.store.start("live", "flow", ["sensor.temperature"])
        first = self.store.record("live", [reading("one")])[0]
        with patch("app.database.confidence_entities._now", return_value="2026-11-18T01:02:00+00:00"):
            second = self.store.record("live", [reading("two")])[0]
        self.assertEqual(first["last_changed"], second["last_changed"])
        self.assertNotEqual(first["evaluated_at"], second["evaluated_at"])
        self.assertGreater(second["revision"], first["revision"])
        self.assertEqual(self.store.record("live", [reading("two")]), [])

    def test_stop_restart_and_flow_reassignment_never_restore_stale_availability(self):
        self.store.start("first", "flow-one", ["sensor.temperature"])
        self.store.record("first", [reading()])
        before = self.store.snapshot()
        restarted = ConfidenceEntityStore(self.path)
        after = restarted.snapshot()
        self.assertEqual(before["store_id"], after["store_id"])
        self.assertFalse(after["entities"][0]["available"])
        self.assertEqual(restarted.record("first", [reading("late")]), [])
        restarted.start("second", "flow-two", ["sensor.temperature"])
        self.assertIsNone(restarted.snapshot()["entities"][0]["confidence"])
        row = restarted.record("second", [reading()])[0]
        self.assertEqual(row["flow_id"], "flow-two")
        self.assertEqual(row["confidence_change_detail"]["kind"], "initial")
        self.assertEqual(row["entity_id"], before["entities"][0]["entity_id"])
        self.assertFalse(restarted.stop("second")[0]["available"])

    def test_an_unavailable_reading_breaks_the_numeric_comparison(self):
        self.store.start("live", "flow", ["sensor.temperature"])
        initial, unavailable, resumed = self.store.record("live", [reading(), reading("gap", None), reading("back", .4)])
        self.assertEqual([row["confidence_change_detail"]["kind"] for row in (initial, unavailable, resumed)],
                         ["initial", "unavailable", "resumed"])
        self.assertIsNone(resumed["confidence_change_detail"]["delta_pp"])

    def test_invalid_confidence_and_derived_sources_are_rejected_atomically(self):
        self.store.start("live", "flow", ["sensor.temperature"])
        before = self.store.snapshot()
        for confidence in (-1, 1.1, float("nan"), float("inf"), True, "0.9"):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                self.store.record("live", [reading(confidence=confidence)])
            self.assertEqual(self.store.snapshot(), before)
        with self.assertRaises(ValueError):
            self.store.start("loop", "flow", ["sensor.confianza_sensor_temperature"])
        self.assertEqual(self.store.snapshot(), before)
        self.assertEqual(self.store.record("live", [reading(source="sensor.other")]), [])

    def test_csv_jobs_publish_test_entities_and_keep_the_live_catalog_separate(self):
        store = FlowStore(self.path)
        publish = Mock()
        jobs = FlowEvaluationJobs(store, confidence_store=self.store, confidence_publish=publish)
        self.store.start("live", "live-flow", ["sensor.temperature"])
        live_state = self.store.record("live", [reading(confidence=.46)])[0]
        def evaluate(progress, sink):
            sink({"output": [reading(), reading("invalid", .77, "without_a_domain")]})
            return {"candidate_count": 2}, None, {"source": {}}
        with patch("app.flow.evaluacion.Thread") as worker:
            csv = jobs.start("csv", evaluate, test_sources=["sensor.temperature", "without_a_domain"])
            worker.call_args.kwargs["target"]()
        self.assertEqual(jobs.get(csv["job_id"])["status"], "completed")
        waiting, reading_state, stopped = [call.args[0][0] for call in publish.call_args_list]
        self.assertEqual(waiting["entity_id"], "confianza.test_sensor_temperature")
        self.assertEqual(waiting["source_mode"], "test")
        self.assertIsNone(waiting["value"])
        self.assertEqual(reading_state["value"], 93)
        self.assertFalse(stopped["available"])
        self.assertEqual(self.store.snapshot()["entities"], [live_state, stopped])

    def test_disconnected_integration_does_not_fail_csv_evaluation(self):
        store = FlowStore(self.path)
        publish = Mock(side_effect=ValueError("bridge disconnected"))
        jobs = FlowEvaluationJobs(store, confidence_store=self.store, confidence_publish=publish)
        def evaluate(progress, sink):
            sink({"output": [reading()]})
            return {"candidate_count": 1}, None, {"source": {}}
        with patch("app.flow.evaluacion.Thread") as worker:
            csv = jobs.start("csv", evaluate, test_sources=["sensor.temperature"])
            worker.call_args.kwargs["target"]()
        result = jobs.get(csv["job_id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["confidence_publication"]["status"], "unavailable")
        self.assertEqual(publish.call_count, 3)
        self.assertFalse(self.store.snapshot()["entities"][0]["available"])
        self.assertEqual(self.store.snapshot()["entities"][0]["value"], 93)

    def test_migrates_the_live_only_catalog_in_place(self):
        self.store.start("live", "flow", ["sensor.temperature"])
        original = self.store.record("live", [reading()])[0]
        # Recreate the 0.5.0 catalog and preserve its existing revision clock.
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE legacy_catalog (
                source_entity_id TEXT PRIMARY KEY,session_id TEXT,revision INTEGER NOT NULL,payload TEXT NOT NULL
            )""")
            db.execute("""INSERT INTO legacy_catalog
                SELECT source_entity_id,session_id,revision,json_remove(payload,'$.source_mode') FROM confidence_catalog
            """)
            db.execute("DROP TABLE confidence_catalog")
            db.execute("ALTER TABLE legacy_catalog RENAME TO confidence_catalog")
        migrated = ConfidenceEntityStore(self.path)
        test = migrated.start("test", "flow", ["sensor.temperature"], test=True)[0]
        live, actual_test = migrated.snapshot()["entities"]
        self.assertEqual(actual_test, test)
        self.assertEqual(live["entity_id"], original["entity_id"])
        self.assertEqual(live["value"], 93)
        self.assertEqual(live["store_id"], original["store_id"])
        self.assertGreater(live["revision"], original["revision"])
        self.assertFalse(live["available"])
        self.assertEqual(migrated.snapshot(compact=True)["entities"][0]["source_mode"], "live")


class ConfidenceHistoryTests(unittest.TestCase):
    def test_indexed_history_is_paged_per_sensor_without_unpacking_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = FlowStore(root / "flow.sqlite3")
            spool = FlowResultSpool("series", root / "pending")
            rows = [reading("zero", 0), reading("other", 1, "sensor.other"), reading("unknown", None), reading("last", .93)]
            try:
                spool.write_block({"output": rows})
                run_id, _ = store.save_run_stream("flow", {}, {}, spool.records())
                with patch("app.database.flows.iter_run_results", side_effect=AssertionError("must use scalar index")):
                    page = store.confidence_series(run_id, "sensor.temperature", limit=2)
                    self.assertEqual(page["entity_id"], "confianza.sensor_temperature")
                    self.assertEqual([point["value"] for point in page["points"]], [0, None])
                    self.assertEqual(page["total"], 3)
                    self.assertEqual(page["next_offset"], 2)
                    last = store.confidence_series(run_id, "sensor.temperature", offset=2)
                    self.assertEqual([point["value"] for point in last["points"]], [93])
                    self.assertIsNone(last["next_offset"])
            finally:
                spool.cleanup()

    def test_older_archives_and_runs_without_confidence_remain_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = FlowStore(root / "flow.sqlite3")
            spool = FlowResultSpool("old", root / "pending")
            try:
                spool.write_block({"output": [reading()]})
                spool.db.execute("DROP TABLE confidence_series")
                run_id, _ = store.save_run_stream("flow", {}, {}, spool.records())
                self.assertEqual(store.confidence_series(run_id)["points"][0]["value"], 93)
            finally:
                spool.cleanup()
            old = reading()
            del old["confidence"]
            run_id = store.save_run("old", {}, {}, {"output": [old]})
            self.assertEqual(store.confidence_series(run_id)["entities"], [])
            self.assertEqual(store.confidence_series(run_id)["points"], [])
            with self.assertRaises(KeyError):
                store.confidence_series("missing")


class ConfidenceBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_chunks_reconnect_and_live_push_use_paired_transport(self):
        requests = []
        class Socket:
            async def send_json(self, message):
                requests.append(message)
                self.connection.receive({"type": "result", "request_id": message["request_id"], "success": True, "data": []})
            async def close(self, **kwargs):
                pass
        with tempfile.TemporaryDirectory() as folder:
            bridge = HomeAssistantBridge(Path(folder) / "bridge.json")
            store = ConfidenceEntityStore(Path(folder) / "confidence.sqlite3")
            store.start("live", "flow", [f"sensor.s{index}" for index in range(501)])
            test_rows = store.start("test", "flow", ["sensor.s0"], test=True)
            bridge.set_confidence_provider(store.snapshot)
            socket = Socket()
            socket.connection = await bridge.attach(socket)
            await asyncio.wait_for(socket.connection.confidence_sync_task, 3)
            sync = [item["arguments"] for item in requests]
            self.assertEqual([len(item["entities"]) for item in sync], [500, 1])
            self.assertTrue(sync[0]["reset"])
            self.assertFalse(sync[0]["complete"])
            self.assertTrue(sync[-1]["complete"])
            self.assertFalse(any(row.get("source_mode") == "test" for batch in sync for row in batch["entities"]))
            with self.assertRaisesRegex(ValueError, "Actualiza la integración"):
                await asyncio.to_thread(bridge.publish_confidence, test_rows)
            rows = store.record("live", [reading(source="sensor.s0")])
            await asyncio.to_thread(bridge.publish_confidence, rows)
            self.assertEqual(requests[-1]["operation"], "confidence_update")
            self.assertEqual(requests[-1]["arguments"]["entities"][0]["value"], 93)
            bridge.detach(socket.connection)
            requests.clear()
            socket.connection = await bridge.attach(socket, supports_test_confidence=True)
            await asyncio.wait_for(socket.connection.confidence_sync_task, 3)
            self.assertEqual(requests[0]["arguments"]["entities"][0]["source_entity_id"], "sensor.s0")
            self.assertEqual(requests[0]["arguments"]["entities"][0]["value"], 93)
            self.assertEqual(requests[0]["arguments"]["entities"][1]["entity_id"], "confianza.test_sensor_s0")
            await bridge.close()

    async def test_deletion_is_replayed_on_reconnect_only_to_updated_integrations(self):
        requests = []

        class Socket:
            async def send_json(self, message):
                requests.append(message)
                self.connection.receive({"type": "result", "request_id": message["request_id"], "success": True})

            async def close(self, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bridge = HomeAssistantBridge(root / "bridge.json")
            store = ConfidenceEntityStore(root / "confidence.sqlite3")
            store.start("live", "flow", ["sensor.temperature"])
            stopped = store.stop("live")[0]
            removed = store.delete({("sensor.temperature", "live"): stopped["revision"]})
            store = ConfidenceEntityStore(store.path)
            bridge.set_confidence_provider(partial(store.snapshot, include_deleted=True))
            for supported in (False, True):
                requests.clear()
                socket = Socket()
                socket.connection = await bridge.attach(socket, supports_confidence_removal=supported)
                await socket.connection.confidence_sync_task
                self.assertEqual(requests[0]["arguments"]["entities"], removed if supported else [])
                if supported:
                    await asyncio.to_thread(bridge.publish_confidence, removed)
                    self.assertEqual(requests[-1]["operation"], "confidence_update")
                else:
                    with self.assertRaisesRegex(ValueError, "Actualiza la integración"):
                        await asyncio.to_thread(bridge.publish_confidence, removed)
                bridge.detach(socket.connection)


class ConfidenceApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        path = Path(self.folder.name) / "flow.sqlite3"
        self.flows = FlowStore(path)
        self.catalog = ConfidenceEntityStore(path)
        runtime = ModuleType("app.services.runtime")
        runtime.state, runtime.flow_store = Mock(), self.flows
        runtime.flow_evaluation_jobs, runtime.home_assistant_bridge = Mock(), Mock()
        spec = importlib.util.spec_from_file_location("isolated_confidence_flows_api",
            Path(__file__).parents[2] / "backend/app/api/flows.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"app.services.runtime": runtime}):
            spec.loader.exec_module(module)
        app = FastAPI()
        app.include_router(module.router)
        self.bridge = Mock()
        app.include_router(create_confidence_router(self.catalog, self.bridge))
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_catalog_and_saved_history_are_separate_and_paginate_per_source(self):
        self.catalog.start("live", "flow", ["sensor.temperature"])
        self.catalog.record("live", [reading()])
        before = self.client.get("/confidence/entities").json()
        self.assertEqual(before["entities"][0]["value"], 93)
        run = self.flows.save_run("test", {}, {}, {"output": [reading("zero", 0), reading("last", 1)]})
        response = self.client.get(f"/flow-runs/{run}/confidence?entity_id=sensor.temperature&limit=1")
        self.assertEqual(response.status_code, 200, response.text)
        first = response.json()
        self.assertEqual(first["points"][0]["value"], 0)
        self.assertEqual(first["next_offset"], 1)
        second = self.client.get(f"/flow-runs/{run}/confidence?entity_id=sensor.temperature&offset=1&limit=1").json()
        self.assertEqual(second["points"][0]["value"], 100)
        self.assertIsNone(second["next_offset"])
        missing_source = self.client.get(f"/flow-runs/{run}/confidence?entity_id=sensor.other").json()
        self.assertEqual(missing_source["total"], 0)
        self.assertEqual(missing_source["points"], [])
        self.assertEqual(self.client.get("/confidence/entities").json(), before)

    def test_history_rejects_invalid_pages_and_handles_missing_or_legacy_runs(self):
        run = self.flows.save_run("old", {}, {}, {"output": []})
        self.assertEqual(self.client.get(f"/flow-runs/{run}/confidence").json()["entities"], [])
        self.assertEqual(self.client.get("/flow-runs/missing/confidence").status_code, 404)
        for query in ("offset=-1", "limit=0", "limit=5001", "limit=invalid"):
            self.assertEqual(self.client.get(f"/flow-runs/{run}/confidence?{query}").status_code, 422)

    def test_delete_is_atomic_and_keeps_saved_history_and_other_modes(self):
        run = self.flows.save_run("flow", {}, {}, {"output": [reading()]})
        self.catalog.start("old", "flow", ["sensor.temperature"])
        live = self.catalog.stop("old")[0]
        test = self.catalog.start("test", "flow", ["sensor.temperature"], test=True)[0]
        compact = self.client.get("/confidence/entities?compact=true").json()["entities"]
        self.assertEqual(len(compact), 2)
        self.assertNotIn("value", compact[0])

        def selection(item):
            return {key: item[key] for key in ("source_entity_id", "source_mode", "revision")}

        response = self.client.request("DELETE", "/confidence/entities", json={"entities": [selection(live), selection(test)]})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.catalog.snapshot()["entities"], [live, test])
        self.bridge.publish_confidence.assert_not_called()

        body = {"entities": [selection(live)]}
        for _ in range(2):
            response = self.client.request("DELETE", "/confidence/entities", json=body)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json(), {"removed": 1, "synchronization": "completed"})
            self.assertEqual(self.catalog.snapshot()["entities"], [test])
        tombstone = self.catalog.snapshot(include_deleted=True)["entities"][0]
        self.assertTrue(tombstone["deleted"])
        self.assertGreater(tombstone["revision"], live["revision"])
        self.assertEqual(self.bridge.publish_confidence.call_args.args[0], [tombstone])
        self.assertEqual(self.client.get(f"/flow-runs/{run}/confidence").json()["points"][0]["value"], 93)
        self.catalog.start("new", "flow", ["sensor.temperature"])
        self.catalog.stop("new")
        response = self.client.request("DELETE", "/confidence/entities", json=body)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(len(self.catalog.snapshot()["entities"]), 2)


if __name__ == "__main__":
    unittest.main()
