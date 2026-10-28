from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from threading import Barrier
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database.flows import FlowStore


def flow(name="Prueba guardada", label="Original"):
    return {"name": name, "nodes": [
        {"id": "input", "type": "input", "config": {"label": label}},
        {"id": "output", "type": "output", "config": {}},
    ], "edges": [{"id": "route", "source": "input", "target": "output"}]}


class FlowNameTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "flows.sqlite"
        self.store = FlowStore(self.path)

    def test_repeated_name_loads_the_original_without_creating_or_overwriting_it(self):
        saved = self.store.create(flow("  Prueba   de diagnóstico  "))
        original = self.store.get(saved["flow_id"])
        for name in ("Prueba de diagnóstico", "PRUEBA DE DIAGNÓSTICO", " prueba de diagno\u0301stico "):
            loaded = self.store.create(flow(name, "New unsaved design"))
            self.assertTrue(loaded["loaded_existing"])
            self.assertEqual({key: value for key, value in loaded.items() if key != "loaded_existing"}, original)
        self.assertEqual(self.store.get(saved["flow_id"]), original)
        self.assertEqual(len(self.store.list()), 1)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM flows").fetchone()[0], 1)

    def test_two_devices_saving_the_same_name_simultaneously_share_one_record(self):
        barrier = Barrier(2)
        def save(index):
            store = FlowStore(self.path)
            barrier.wait(timeout=5)
            return store.create(flow("Concurrent save", f"Device {index}"))
        with ThreadPoolExecutor(max_workers=2) as pool:
            saved = list(pool.map(save, (1, 2)))
        self.assertEqual(saved[0]["flow_id"], saved[1]["flow_id"])
        self.assertEqual(sum(bool(item.get("loaded_existing")) for item in saved), 1)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM flows").fetchone()[0], 1)

    def test_renaming_to_an_existing_name_loads_it_and_editing_the_selected_record_still_updates_it(self):
        first = self.store.create(flow("First"))
        second = self.store.create(flow("Second"))
        original_first, original_second = self.store.get(first["flow_id"]), self.store.get(second["flow_id"])
        loaded = self.store.update(second["flow_id"], flow(" first ", "Discarded change"))
        self.assertEqual(loaded["flow_id"], first["flow_id"])
        self.assertTrue(loaded["loaded_existing"])
        self.assertEqual(self.store.get(first["flow_id"]), original_first)
        self.assertEqual(self.store.get(second["flow_id"]), original_second)
        changed = self.store.update(first["flow_id"], flow("First", "Updated original"))
        self.assertFalse(changed.get("loaded_existing", False))
        self.assertEqual(self.store.get(first["flow_id"])["nodes"][0]["config"]["label"], "Updated original")
        self.assertEqual(len(self.store.list()), 2)

    def test_legacy_duplicates_list_the_most_recent_and_old_execution_references_remain_readable(self):
        old, recent = {**flow("Existing", "Old"), "flow_id": "old"}, {**flow(" EXISTING ", "Recent"), "flow_id": "recent"}
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executemany("INSERT INTO flows VALUES (?,?,?,?)", [
                ("old", old["name"], json.dumps(old), "2026-01-01"),
                ("recent", recent["name"], json.dumps(recent), "2026-01-02"),
            ])
        self.store.save_run("old", {"flow": deepcopy(old)}, {}, {})
        self.assertEqual([item["flow_id"] for item in self.store.list()], ["recent"])
        self.assertEqual(self.store.create(flow("existing"))["flow_id"], "recent")
        restarted = FlowStore(self.path)
        self.assertEqual([item["flow_id"] for item in restarted.list()], ["recent"])
        self.assertEqual(restarted.get("old")["nodes"], old["nodes"])
        self.assertEqual(restarted.list_runs()[0]["snapshot"]["flow"], old)

    def test_api_returns_the_saved_flow_and_marks_name_collisions(self):
        runtime = ModuleType("app.services.runtime")
        runtime.state, runtime.flow_store = Mock(), self.store
        runtime.flow_evaluation_jobs, runtime.home_assistant_bridge = Mock(), Mock()
        spec = importlib.util.spec_from_file_location("isolated_flow_names_api", Path(__file__).parents[2] / "backend/app/api/flows.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"app.services.runtime": runtime}):
            spec.loader.exec_module(module)
        api = FastAPI()
        api.include_router(module.router)
        with TestClient(api) as client:
            original = client.post("/flows", json=flow()).json()
            loaded = client.post("/flows", json=flow(label="Different design"))
            self.assertEqual(loaded.status_code, 200)
            self.assertTrue(loaded.json()["loaded_existing"])
            self.assertEqual(loaded.json()["flow_id"], original["flow_id"])
            self.assertEqual(client.get("/flows").json()["items"][0]["nodes"], original["nodes"])
            self.assertEqual(len(client.get("/flows").json()["items"]), 1)


if __name__ == "__main__":
    unittest.main()
