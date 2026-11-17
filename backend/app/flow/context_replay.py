from threading import Lock
from time import monotonic

import pandas as pd


class ContextReplay:
    """A CSV reception clock and a separate causal cursor into its received events."""

    def __init__(self, events, candidates, development_end, filename, *, clock=monotonic):
        if candidates.empty:
            raise ValueError("El conjunto etiquetado no contiene lecturas para evaluar.")
        self.filename = filename
        self._groups = pd.DatetimeIndex(candidates["timestamp_dt"]).unique().sort_values()
        self._end = self._groups[-1]
        in_range = events["timestamp_dt"].between(development_end, self._end)
        eligible = events.loc[in_range]
        keys = ["entity_id", "timestamp_dt"]
        overlap = pd.MultiIndex.from_frame(eligible[keys]).isin(pd.MultiIndex.from_frame(candidates[keys]))
        self._events = eligible.loc[~overlap].sort_values("timestamp_dt", kind="stable")
        if self._events.empty:
            raise ValueError("El CSV de contexto no contiene eventos adicionales entre el fin de Desarrollo y el fin de la evaluación.")
        # The reception clock can land between two microseconds even when the
        # input CSV has only microsecond timestamps.
        self._timestamps = pd.DatetimeIndex(self._events["timestamp_dt"]).as_unit("ns")
        self._start = min(self._groups[0], self._timestamps[0])
        self._clock = clock
        self._lock = Lock()
        self._anchor_at = None
        self._completed = 0
        self._seconds_per_group = 0.05
        self._received_through = None
        self._applied = 0

    def _advance_clock(self, now):
        if self._anchor_at is None:
            return
        anchor_group = max(0, self._completed - 1)
        position = min(len(self._groups) - 1, anchor_group + (now - self._anchor_at) / self._seconds_per_group)
        index = int(position)
        stamp = self._groups[index]
        if index + 1 < len(self._groups):
            step = self._groups[index + 1].value - stamp.value
            stamp += pd.Timedelta(int(step * (position - index)), unit="ns")
        self._received_through = max(stamp, self._received_through)

    def take_through(self, timestamp):
        with self._lock:
            if self._anchor_at is None:
                self._anchor_at = self._clock()
                self._received_through = timestamp
            else:
                self._received_through = max(timestamp, self._received_through)
            end = int(self._timestamps.searchsorted(timestamp, side="right"))
            block = self._events.iloc[self._applied:end]
            self._applied = end
            return block

    def complete_group(self):
        with self._lock:
            now = self._clock()
            self._advance_clock(now)
            # Adapt to the evaluation cadence. Bound each update so a single slow
            # group does not immediately slow reception to the same rate.
            observed = min(now - self._anchor_at, 2 * self._seconds_per_group)
            self._seconds_per_group = max(0.001, 0.8 * self._seconds_per_group + 0.2 * observed)
            self._anchor_at = now
            self._completed += 1

    def _percent(self, timestamp):
        if timestamp is None:
            return 0.0
        span = self._end.value - self._start.value
        return 100.0 * (timestamp.value - self._start.value) / span if span else 100.0

    def status(self, *, advance=True):
        with self._lock:
            if advance:
                self._advance_clock(self._clock())
            through = self._received_through
            received = int(self._timestamps.searchsorted(through, side="right")) if through is not None else 0
            evaluated = self._groups[self._completed - 1] if self._completed else None
            return {
                "filename": self.filename, "total": len(self._events), "received": received,
                "applied": self._applied, "buffered": received - self._applied,
                "start": self._start.isoformat(), "end": self._end.isoformat(),
                "received_through": through.isoformat() if through is not None else None,
                "evaluated_through": evaluated.isoformat() if evaluated is not None else None,
                "reception_percent": self._percent(through), "evaluation_percent": self._percent(evaluated),
            }
