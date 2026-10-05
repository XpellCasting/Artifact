"""FOCuS Bernoulli on observed binary reports (not time occupancy)."""
from dataclasses import dataclass, field
import math
from typing import Any

import pandas as pd

METHOD = "binary_focus_bernoulli_v1"


def detector():
    from focus_cpt import Detector
    return Detector(type="univariate")


@dataclass
class BernoulliRuntime:
    values: list[float] = field(default_factory=list)
    timestamps: list[pd.Timestamp] = field(default_factory=list)
    focus: Any = None

    def __getstate__(self):
        # Native detector cannot be pickled. Reconstruct exactly, never reset
        # a partially evaluated stream when deepcopy/cache loading is used.
        return {"values": self.values.copy(), "timestamps": self.timestamps.copy(), "focus": None}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update(self, value: float, timestamp: pd.Timestamp, p0: float):
        if value not in (0., 1.):
            raise ValueError("bernoulli_requires_zero_or_one")
        if self.timestamps and timestamp < self.timestamps[-1]:
            raise ValueError("bernoulli_requires_chronological_reports")
        if self.focus is None:
            self.focus = detector()
            for previous in self.values:
                self.focus.update(previous)
        self.focus.update(value)
        self.values.append(value)
        self.timestamps.append(timestamp)
        return self.focus.get_statistics(family="bernoulli", theta0=p0)


class FiltroBernoulli:
    min_history_events = 60

    def clean(self, frame):
        work = frame.copy()
        work["timestamp_dt"] = pd.to_datetime(work["timestamp_dt"], errors="coerce", utc=True)
        work["value"] = pd.to_numeric(work["value"], errors="coerce")
        if "is_valid_value" in work:
            work = work[work["is_valid_value"].fillna(False).astype(bool)]
        # Do not clip non-binary data or turn unavailable readings into zero.
        return work[work.timestamp_dt.notna() & work.value.isin([0., 1.])].sort_values(
            "timestamp_dt", kind="stable")

    def prepare(self, frame):
        work = self.clean(frame)
        if len(work) < self.min_history_events:
            return None
        split = int(len(work)*.6)
        while split > 0 and work.iloc[split-1].timestamp_dt == work.iloc[split].timestamp_dt:
            split -= 1
        if split < 24 or len(work)-split < 24:
            return None
        fit, calibration = work.iloc[:split], work.iloc[split:]
        # Jeffreys smoothing keeps p0 strictly inside (0,1), including constant
        # training histories, where the native log likelihood has boundaries.
        p0 = (float(fit.value.sum()) + .5)/(len(fit)+1)
        runtime = BernoulliRuntime()
        stats = [float(runtime.update(float(row.value), row.timestamp_dt, p0)['stat'])
                 for row in calibration.itertuples()]
        threshold = max(20., 1.1*max(stats))
        if not math.isfinite(threshold):
            raise RuntimeError("bernoulli_nonfinite_calibration")
        return {"method": METHOD, "p0": p0, "threshold": threshold,
                "fit_rows": len(fit), "calibration_rows": len(calibration),
                "fit_end": fit.iloc[-1].timestamp_dt.isoformat(),
                "calibration_start": calibration.iloc[0].timestamp_dt.isoformat(),
                "development_end": work.iloc[-1].timestamp_dt.isoformat(),
                "sampling_semantics": "observed_reports_not_time_occupancy",
                "threshold_policy": "max_20_or_1.1_times_calibration_max",
                "p0_policy": "jeffreys_smoothed_fit_frequency",
                "runtime": BernoulliRuntime()}

    def evaluate(self, frame, cached, entity_id):
        if cached.get("method") != METHOD:
            raise ValueError("binary_baseline_requires_bernoulli_rebuild")
        work = self.clean(frame)
        boundary = pd.Timestamp(cached["development_end"])
        if not work.empty and work.timestamp_dt.min() <= boundary:
            raise ValueError("bernoulli_evaluation_must_follow_development")
        results = []
        runtime = cached['runtime']
        for row in work.to_dict('records'):
            at = row['timestamp_dt']
            point = runtime.update(float(row['value']), at, cached['p0'])
            stat = float(point['stat'])
            if not math.isfinite(stat):
                raise RuntimeError("bernoulli_nonfinite_statistic")
            if stat <= cached['threshold']:
                continue
            cp = point.get('changepoint')
            index = int(cp) if cp is not None and math.isfinite(float(cp)) else None
            # Library changepoint is the number of preceding observations.
            start = runtime.timestamps[index] if index is not None and 0 <= index < len(runtime.timestamps) else runtime.timestamps[0]
            ratio = stat/cached['threshold']
            # Bounded display strength, explicitly NOT an anomaly probability.
            score = ratio/(1+ratio)
            summary = {key: value for key, value in cached.items() if key != 'runtime'}
            results.append({"entity_id": str(entity_id), "source_raw_index": row.get('_raw_index'),
                "timestamp": at.isoformat(), "window_start": start.isoformat(), "window_end": at.isoformat(),
                "sensor_kind": "binary", "tipo_interno": "estado_binario", "value": float(row['value']),
                "observed_state": str(row.get('state', row['value'])), "binary_method": METHOD,
                "raw_score": stat, "evidence_score": ratio, "suspicion_score": score,
                "score_kind": "calibrated_statistic_strength_not_probability", "is_suspicious": True,
                "pattern": "activation_burst", "suspected_pattern": "activation_burst",
                "pattern_detail": "Cambio en probabilidad de activacion (FOCuS Bernoulli)",
                "filter": METHOD, "passed_to_contextual_filter": True,
                "triggered_indicators": ['bernoulli_probability_change'],
                "reading": {"focus_statistic": stat, "focus_threshold": cached['threshold'],
                    "baseline_probability": cached['p0'], "focus_changepoint_index": index,
                    "reports_evaluated": len(runtime.values)},
                "baseline_summary": summary, "ground_truth_label": None,
                "severity": "alta" if ratio >= 2 else "media"})
        return results

    def analizar(self, eventos, entidades, suspicion_threshold=.6):
        if eventos.empty or 'sensor_kind' not in eventos:
            return []
        results = []
        for entity_id, group in eventos[eventos.sensor_kind.eq('binary')].groupby('entity_id', sort=True):
            work = self.clean(group)
            if len(work) < 2:
                continue
            split = max(1, int(len(work)*.8))
            boundary = work.iloc[split].timestamp_dt
            cached = self.prepare(work[work.timestamp_dt < boundary])
            if cached:
                results.extend(self.evaluate(work[work.timestamp_dt >= boundary], cached, entity_id))
        return results
