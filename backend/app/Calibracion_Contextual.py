"""Características de calibración por bloques y procesos con memoria acotada.

Se usa únicamente con observaciones históricas inmutables. La inferencia en
vivo conserva la función escalar y su evidencia detallada.
"""
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from hashlib import sha256
from multiprocessing import get_context
import os
from queue import Empty, Full

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


FAMILIES = ('level', 'innovation', 'trend')
BLOCK_SIZE = 2048


class PreparedFeatures:
    def __init__(self, observations, config, cache_bytes=64 * 1024 * 1024):
        self.observations = observations
        self.config = config
        self.arrays = {}
        self.cache = OrderedDict()
        self.cache_bytes = cache_bytes
        self.used_bytes = 0

    def _arrays(self, entity):
        if entity not in self.arrays:
            events = self.observations[entity]
            times = events.index.as_unit('ns').asi8
            values = pd.to_numeric(events.get('value', events.get('state')), errors='coerce').to_numpy(dtype=float, na_value=np.nan)
            valid = events['is_valid_value'].to_numpy(dtype=bool)
            invalid_prefix = np.r_[0, np.cumsum(~valid)]
            self.arrays[entity] = times, values, valid, invalid_prefix
        return self.arrays[entity]

    def features(self, entity, references, window, max_age, partition_start, current=None, report=None):
        # current solo contiene lecturas históricas ya presentes en observations.
        # La inserción de lecturas futuras sigue en la implementación escalar.
        references = pd.DatetimeIndex(references).as_unit('ns')
        queries = references.asi8
        current = None if current is None else np.asarray(current, dtype=float)
        ttl = self.config.antiguedad_maxima(entity) if max_age is None else float(max_age)
        start = None if partition_start is None else pd.Timestamp(partition_start).value
        key = (entity, window, ttl, start, sha256(queries.tobytes()).digest(),
               None if current is None else sha256(current.tobytes()).digest())
        if key in self.cache:
            self.cache.move_to_end(key)
            if report:
                report(len(queries), len(queries))
            return self.cache[key]
        result = np.full((len(queries), 3), np.nan)
        events = self.observations.get(entity)
        if events is not None and not events.empty:
            times, values, valid, invalid_prefix = self._arrays(entity)
            width = int(window) * 1_000_000_000
            for offset in range(0, len(queries), BLOCK_SIZE):
                q = queries[offset:offset + BLOCK_SIZE]
                latest = np.searchsorted(times, q, side='right') - 1
                pos = np.maximum(latest, 0)
                actual = values[pos].copy()
                usable = (latest >= 0) & valid[pos] & np.isfinite(actual)
                explicit = np.zeros(len(q), dtype=bool)
                if current is not None:
                    supplied = current[offset:offset + BLOCK_SIZE]
                    explicit = np.isfinite(supplied)
                    actual[explicit] = supplied[explicit]
                    usable |= explicit
                usable &= explicit | ((q - times[pos]) / 1e9 <= ttl)
                usable &= q != pd.NaT.value
                actual[~usable] = np.nan

                def median(left_at, right_at, left_open, right_closed):
                    left = np.searchsorted(times, left_at, side='right' if left_open else 'left')
                    right = np.searchsorted(times, right_at, side='right' if right_closed else 'left')
                    good = usable & (right > left) & (invalid_prefix[right] == invalid_prefix[left])
                    if start is not None:
                        good &= left_at >= start
                    medians = np.full(len(q), np.nan)
                    # Consultas con los mismos límites comparten una sola mediana.
                    indices = np.flatnonzero(good)
                    if len(indices):
                        pairs, inverse = np.unique(np.column_stack((left[indices], right[indices])), axis=0, return_inverse=True)
                        computed = np.full(len(pairs), np.nan)
                        for i, (lo, hi) in enumerate(pairs):
                            numeric = values[lo:hi]
                            numeric = numeric[np.isfinite(numeric)]
                            if len(numeric) >= self.config.temporal_feature_min_observations:
                                computed[i] = np.median(numeric)
                        medians[indices] = computed[inverse]
                    return medians

                previous = median(q - width, q, False, False)
                recent = median(q - width, q, True, True)
                earlier = median(q - 2 * width, q - width, True, True)
                result[offset:offset + len(q)] = np.column_stack((actual, actual - previous, recent - earlier))
                if report:
                    report(offset + len(q), len(queries))
        if result.nbytes <= self.cache_bytes:
            while self.cache and self.used_bytes + result.nbytes > self.cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.used_bytes -= old.nbytes
            self.cache[key] = result
            self.used_bytes += result.nbytes
        return result

    def samples(self, edge, bounds, report=None):
        source, target = str(edge['source']), str(edge['target'])
        source_events, target_events = self.observations.get(source), self.observations.get(target)
        empty = lambda: {name: pd.DataFrame(columns=['target', 'context', 'timestamp']) for name in FAMILIES}
        if source_events is None or target_events is None or source_events.empty or target_events.empty:
            return empty()
        start, end = (pd.to_datetime(bounds.get(key), errors='coerce', utc=True) for key in ('start', 'end'))
        if pd.isna(start) or pd.isna(end):
            return empty()
        candidates = source_events[(source_events.index > start) & (source_events.index <= end)]
        candidates = candidates[~candidates.index.duplicated(keep='last')]
        values = pd.to_numeric(candidates.get('value', candidates.get('state')), errors='coerce').to_numpy(dtype=float, na_value=np.nan)
        mask = candidates['is_valid_value'].to_numpy(dtype=bool) & np.isfinite(values)
        timestamps, current = candidates.index[mask], values[mask]
        source_features = self.features(source, timestamps, int(edge['window_seconds']), edge.get('max_age_seconds'), start, current,
            report=(lambda done, total: report(.5 * done / max(1, total), f'características de {source}: {done}/{total} eventos')) if report else None)
        target_features = self.features(target, timestamps - pd.Timedelta(seconds=int(edge.get('lag_seconds', 0))),
            int(edge['window_seconds']), edge.get('max_age_seconds'), start,
            report=(lambda done, total: report(.5 + .5 * done / max(1, total), f'características de {target}: {done}/{total} eventos')) if report else None)
        rows = {}
        for index, family in enumerate(FAMILIES):
            usable = np.isfinite(source_features[:, index]) & np.isfinite(target_features[:, index])
            rows[family] = pd.DataFrame({'timestamp': timestamps[usable], 'target': source_features[usable, index], 'context': target_features[usable, index]})
        return rows


def calibrate_relation(filtro, edge, partitions, prepared, report=None):
    fit = prepared.samples(edge, partitions['fit'],
        (lambda fraction, message: report(.45 * fraction, 'ajuste — ' + message)) if report else None)
    tolerance = prepared.samples(edge, partitions['tolerance'],
        (lambda fraction, message: report(.45 + .45 * fraction, 'tolerancia — ' + message)) if report else None)
    if report:
        report(.95, 'ajustando modelos y umbrales')
    models = {name: filtro._ajustar_modelo_temporal(name, fit[name], tolerance[name]) for name in FAMILIES}
    return {'partitioning': partitions, 'models': models,
            'sample_counts': {name: {'fit': len(fit[name]), 'tolerance': len(tolerance[name])} for name in models}}


_progress_queue = None
_native_limits = None


def _initialize_worker(queue):
    global _progress_queue, _native_limits
    _progress_queue = queue
    queue.cancel_join_thread()  # Los avisos son prescindibles; nunca retener la salida del hijo.
    _native_limits = threadpool_limits(limits=1)


def _calibrate_source(config, source, edges, observations, partitions, callback=None):
    from .Filtro_Contextual.Contextual import FiltroContextualEspacioTemporal
    filtro = FiltroContextualEspacioTemporal(config)
    prepared = PreparedFeatures(observations, config)
    results = []
    for index, edge in enumerate(edges):
        def report(fraction, stage):
            message = f"{edge['source']} → {edge['target']}: {stage}"
            completed = index + fraction
            if callback is not None:
                callback(source, completed, message)
            elif _progress_queue is not None:
                try:
                    _progress_queue.put_nowait((source, completed, message))
                except Full:
                    pass  # La finalización siempre se recibe mediante el Future.
        report(0, 'preparando características')
        try:
            results.append(calibrate_relation(filtro, edge, partitions, prepared, report))
        except Exception as exc:
            raise RuntimeError(f"Error calibrando {edge['source']} → {edge['target']}: {exc}") from exc
        report(1, 'relación completada')
    return results


def calibrate_relations(config, edges, observations, partitions, report):
    try:
        requested = int(os.environ.get('CONTEXT_CALIBRATION_WORKERS', '1'))
    except ValueError as exc:
        raise ValueError('CONTEXT_CALIBRATION_WORKERS debe ser un entero positivo') from exc
    if requested < 1:
        raise ValueError('CONTEXT_CALIBRATION_WORKERS debe ser un entero positivo')
    groups = {}
    for edge in edges:
        groups.setdefault(str(edge['source']), []).append(edge)
    total = len(edges)
    if not total:
        report('context_models', 75., 'Calibración contextual: 0/0 relaciones completadas')
        return
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    workers = min(requested, len(groups), cpu_count)
    progress = dict.fromkeys(groups, 0.)
    def update(source, done, message):
        if done < progress[source]:
            return
        progress[source] = done
        completed = sum(int(value) for value in progress.values())
        report('context_models', 55. + 20. * sum(progress.values()) / total,
               f'Calibrando ({completed}/{total} relaciones completadas; {workers} procesos): {message}')
    def subset(source_edges):
        ids = {str(e[key]) for e in source_edges for key in ('source', 'target')}
        return {entity: observations[entity][[column for column in ('value', 'state', 'is_valid_value') if column in observations[entity]]]
                for entity in ids if entity in observations}
    if workers == 1:
        for source, source_edges in groups.items():
            results = _calibrate_source(config, source, source_edges, subset(source_edges), partitions, update)
            for edge, result in zip(source_edges, results):
                edge['temporal_models'] = result
        return
    context = get_context('spawn')
    queue = context.Queue(maxsize=64)
    pending = {}
    iterator = iter(groups.items())
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_initialize_worker, initargs=(queue,)) as pool:
            def fill():
                while len(pending) < workers:
                    try:
                        source, source_edges = next(iterator)
                    except StopIteration:
                        break
                    future = pool.submit(_calibrate_source, config, source, source_edges, subset(source_edges), partitions)
                    pending[future] = (source, source_edges)
                    update(source, 0., f'{source}: preparando relaciones')
            try:
                fill()
                while pending:
                    finished, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                    for _ in range(64):
                        try:
                            update(*queue.get_nowait())
                        except Empty:
                            break
                    for future in finished:
                        source, source_edges = pending.pop(future)
                        results = future.result()
                        for edge, result in zip(source_edges, results):
                            edge['temporal_models'] = result
                        update(source, float(len(source_edges)), f'{source}: relaciones completadas')
                    fill()
            finally:
                for future in pending:
                    future.cancel()
    finally:
        queue.close()
        queue.join_thread()
