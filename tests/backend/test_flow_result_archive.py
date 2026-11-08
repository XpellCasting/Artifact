from copy import deepcopy
import errno
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.database.flows import FlowStore
from app.flow.definitions import _json
from app.flow.evidence import append_evidence
from app.flow.result_spool import FlowResultSpool, ResultArchive, SpoolRecords, decode_observation


class ResultArchiveTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.store = FlowStore(self.root / "flows.sqlite3")

    def spool(self):
        spool = FlowResultSpool("test", self.root / "staging")
        self.addCleanup(spool.cleanup)
        return spool

    def block(self, identity="one"):
        base = {"observation_id": identity, "entity_id": "sensor.a", "timestamp": "2026-01-01T00:00:00Z",
                "state": "20", "value": 20., "source_raw_index": 1, "signal": "unknown", "evidence": []}
        statistical = append_evidence(base, "stat", "statistical_focus", "positive", "suspicion",
                                      {"details": [1., 2., 3.], "statistical_observation": {"evaluable": True}}, "independent", "run")
        classifier = append_evidence(statistical, "classifier", "classifier", "positive", "compatible",
                                     {"class_scores": {"normal": .1, "sensor_fault": .9}, "fault_type": "spike"}, "combine_v1", "run")
        output = append_evidence(classifier, "output", "output", "positive", "merge_output", {}, "combine_v1", "run")
        output["final_decision"] = "fault_compatible"
        output["evaluation"] = {"expected": "Falla", "outcome": "detectado", "type_identified": True}
        return {"input": [base], "stat": [statistical], "classifier": [classifier], "output": [output]}

    def test_shared_evidence_round_trips_every_field_and_reference_after_restart(self):
        block = self.block()
        original = deepcopy(block)
        spool = self.spool()
        spool.write_block(block)
        records = spool.records()
        with patch.object(SpoolRecords, "__iter__", side_effect=AssertionError("must publish the prepared file")):
            summary = {"candidate_count": 1, "evaluation": {"labelled": True, "groups": [{"detected": 1, "total": 1}]}}
            run, _ = self.store.save_run_stream("flow", {}, summary, records)
        self.assertEqual(block, original)
        self.assertEqual(spool.stats["unique_evidence"], 3)
        self.assertEqual(spool.stats["evidence_references"], 6)
        spool.cleanup()
        self.assertTrue(spool.path.exists())
        restarted = FlowStore(self.store.path)
        for node, items in block.items():
            self.assertEqual(restarted.results(run, node)["items"], items)
        self.assertEqual(restarted.list_runs()[0]["summary"], summary)
        self.assertEqual(restarted.alerts()["items"][0]["result"], block["output"][0])
        with self.store._database() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM flow_results WHERE run_id=?", (run,)).fetchone()[0], 0)

    def test_per_node_order_pagination_and_iterator_order_survive_grouping(self):
        first, second = self.block("first"), self.block("second")
        block = {"input": [first["input"][0], second["input"][0]],
                 "output": [second["output"][0], first["output"][0]]}
        spool = self.spool()
        spool.write_block(block)
        records = list(spool.records())
        self.assertEqual([(r["node_id"], r["observation_id"]) for r in records],
                         [("input", "first"), ("input", "second"), ("output", "second"), ("output", "first")])
        run, _ = self.store.save_run_stream("flow", {}, {}, spool.records())
        page = self.store.results(run, "output", limit=1)
        self.assertEqual(page["items"], block["output"][:1])
        self.assertEqual(page["next_offset"], 1)
        self.assertEqual(self.store.results(run, "output", offset=1)["items"], block["output"][1:])
        self.assertEqual(self.store.results(run, "absent")["total"], 0)

    def test_missing_fields_legacy_entries_and_metadata_variants_are_preserved(self):
        item = self.block()["input"][0]
        legacy = {"node_id": "old", "signal": "positive", "data": {"a": [1, False]}}
        left = {**item, "evidence": [legacy, {"evidence_id": "same", "data": {"flag": False}}]}
        right = {key: value for key, value in left.items() if key != "value"}
        right["evidence"] = [{**legacy, "module_type": "unknown"}, {"evidence_id": "same", "data": {"flag": 0}}]
        spool = self.spool()
        spool.write_block({"a": [left], "b": [right]})
        restored = list(spool.records())
        self.assertEqual(restored[0]["payload"], left)
        self.assertEqual(restored[1]["payload"], right)
        self.assertIs(type(restored[0]["payload"]["evidence"][1]["data"]["flag"]), bool)
        self.assertIs(type(restored[1]["payload"]["evidence"][1]["data"]["flag"]), int)

    def test_json_normalization_matches_legacy_and_reads_do_not_share_mutations(self):
        block = self.block()
        block["input"][0]["value"] = float("nan")
        block["input"][0]["extra"] = (1, 2)
        block["stat"][0]["value"] = None
        block["output"][0]["evidence"] = "legacy opaque evidence"
        spool = self.spool()
        spool.write_block(block)
        run, _ = self.store.save_run_stream("flow", {}, {}, spool.records())
        for node, items in block.items():
            self.assertEqual(self.store.results(run, node)["items"], json.loads(_json(items)))
        changed = self.store.results(run, "stat")["items"][0]
        changed["evidence"][0]["data"]["details"].append(999)
        self.assertEqual(self.store.results(run, "stat")["items"], block["stat"])

    def test_duplicate_observation_across_blocks_cannot_be_published(self):
        spool = self.spool()
        spool.write_block(self.block())
        with self.assertRaises(sqlite3.IntegrityError):
            spool.write_block(self.block())
        with self.assertRaisesRegex(ValueError, "falló"):
            spool.records()
        self.assertEqual(self.store.list_runs(), [])

    def test_failed_metadata_save_restores_the_scratch_file_and_rolls_back_alerts(self):
        spool = self.spool()
        spool.write_block(self.block())
        records = spool.records()
        original_path = spool.path

        class FailingConnection(sqlite3.Connection):
            def execute(self, statement, *args):
                if statement.startswith("INSERT INTO flow_runs"):
                    raise sqlite3.OperationalError("metadata fixture failed")
                return super().execute(statement, *args)

        with patch.object(self.store, "_connect", side_effect=lambda: sqlite3.connect(self.store.path, factory=FailingConnection)):
            with self.assertRaisesRegex(sqlite3.OperationalError, "metadata fixture"):
                self.store.save_run_stream("flow", {}, {}, records)
        self.assertTrue(original_path.is_file())
        self.assertEqual(self.store.list_runs(), [])
        self.assertEqual(self.store.alerts()["items"], [])
        self.assertEqual(len(list(records)), 4)
        run, _ = self.store.save_run_stream("flow", {}, {}, records)
        self.assertEqual(self.store.results(run)["items"], self.block()["output"])

    def test_cross_filesystem_publication_copies_then_preserves_the_completed_file(self):
        spool = self.spool()
        spool.write_block(self.block())
        records = spool.records()
        replace = __import__("os").replace
        def cross_device(source, target):
            if Path(source) == spool.path:
                raise OSError(errno.EXDEV, "cross device fixture")
            return replace(source, target)
        original = spool.path
        with patch("app.flow.result_spool.os.replace", side_effect=cross_device):
            run, _ = self.store.save_run_stream("flow", {}, {}, records)
        self.assertFalse(original.exists(), "source scratch must be retired after a copied publication")
        spool.cleanup()
        self.assertTrue(spool.path.exists())
        self.assertEqual(self.store.results(run)["items"], self.block()["output"])

    def test_an_empty_archive_and_legacy_runs_are_both_readable(self):
        legacy = self.store.save_run("old", {}, {}, self.block())
        spool = self.spool()
        run, _ = self.store.save_run_stream("new", {}, {}, spool.records())
        self.assertEqual(self.store.results(run)["total"], 0)
        self.assertEqual(self.store.results(legacy)["items"], self.block()["output"])

    def test_full_audit_snapshot_is_compressed_once_and_reconstructed_for_queries(self):
        for diagnostic in ({"classifier": {"run_id": "frozen-run", "model": {
                "model_id": "frozen-model", "config": {"steps": 60}, "training_ids": ["audit" * 1000] * 200}}}, None):
            snapshot = {"flow": {"name": "audit"}, "diagnostic_snapshots": diagnostic}
            original = deepcopy(snapshot)
            spool = self.spool()
            spool.write_block(self.block())
            spool.finish(snapshot=snapshot)
            run, _ = self.store.save_run_stream("flow", snapshot, {}, spool.records())
            self.assertEqual(snapshot, original)
            self.assertEqual(next(item for item in self.store.list_runs() if item["run_id"] == run)["snapshot"], original)
            self.assertNotIn("diagnostic_snapshots", next(item for item in self.store.list_runs(compact=True) if item["run_id"] == run)["snapshot"])
            if diagnostic is not None:
                with self.store._database() as db:
                    raw = db.execute("SELECT snapshot FROM flow_runs WHERE run_id=?", (run,)).fetchone()[0]
                self.assertLess(len(raw), len(_json(original)) // 100)
                self.assertEqual(json.loads(raw)["diagnostic_snapshots"]["classifier"]["model"]["model_id"], "frozen-model")

    def test_sealing_failure_never_creates_a_completed_run(self):
        spool = self.spool()
        spool.write_block(self.block())
        with patch("app.flow.result_spool.os.fsync", side_effect=OSError("fsync fixture")):
            with self.assertRaisesRegex(OSError, "fsync fixture"):
                spool.records()
        self.assertEqual(self.store.list_runs(), [])


if __name__ == "__main__":
    unittest.main()
