"""Históricos cortos conservan discretos y no usan el detector continuo retirado."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.services.historial.cargar_historial import CargadorHistorial
from app.modulos.estadistico.filtro import FiltroEstadistico


def fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        history = root / "history.csv"
        rows = ["entity_id,state,last_changed"]
        from datetime import datetime, timedelta, timezone
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for step in range(576):
            at = (start + timedelta(minutes=5 * step)).isoformat()
            temperature = 21 + math.sin(step / 12) + (30 if step == 450 else 0)
            rows.extend([f"sensor.temperature,{temperature},{at}",
                         f"binary_sensor.motion,{'on' if step % 8 < 2 else 'off'},{at}",
                         f"sensor.alarm_mode,{['idle', 'armed', 'alert'][step // 15 % 3]},{at}"])
        history.write_text("\n".join(rows), encoding="utf-8")
        config = root / "conf.txt"
        config.write_text("", encoding="utf-8")
        with patch.dict(os.environ, {"CONTINUOUS_CONF_PATH": str(config)}):
            loaded = CargadorHistorial().cargar_real(history)
            output = FiltroEstadistico().analizar_historial(loaded.eventos, loaded.entidades)
        values = output["all_results"]
        payload = json.dumps(values, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
        return {"sha256": hashlib.sha256(payload.encode()).hexdigest(), "results": len(values),
                "by_kind": output["summary"]["by_kind"]}


class FilterRegressionTests(unittest.TestCase):
    def test_short_history_has_no_uncalibrated_continuous_alerts(self):
        result = fingerprint()
        self.assertEqual(result['results'], 0)
        self.assertEqual(result['by_kind'], {'continuous': 0, 'binary': 0, 'multistate': 0})
