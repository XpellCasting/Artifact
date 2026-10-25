"""Preparación por procesos con índices compartidos, orden estable y DTW en lotes."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
from time import monotonic

import numpy as np

from app.modulos.clasificador.batched_dtw import BatchDTW
from app.modulos.clasificador.event_index import IndexedEventProvider
from app.modulos.clasificador.segments import build_sample

_worker_provider = _worker_preprocessor = _worker_graph = _worker_config = None
_worker_limits = None


def _initialize(index_path, preprocessor, graph, config):
    global _worker_provider, _worker_preprocessor, _worker_graph, _worker_config, _worker_limits
    from threadpoolctl import threadpool_limits
    _worker_limits = threadpool_limits(limits=1)
    _worker_provider = IndexedEventProvider.load(index_path)
    _worker_preprocessor, _worker_graph, _worker_config = preprocessor, graph, config


def _build(stamps, provider, preprocessor, graph, config):
    result = []
    for stamp in stamps:
        sample = build_sample(provider, stamp, preprocessor, graph, config, defer_dtw=True)
        # Compact IPC; cast once, exactly as PartitionWriter does for storage.
        for field, dtype in (("X", np.float32), ("state_ids", np.int64), ("node_present", np.bool_)):
            sample[field] = np.asarray(sample[field], dtype=dtype)
        result.append(sample)
    return result


def _worker_build(stamps):
    return _build(stamps, _worker_provider, _worker_preprocessor, _worker_graph, _worker_config)


def options():
    try:
        workers = int(os.environ.get("GAT_PREPARE_WORKERS", "1"))
        batch_size = int(os.environ.get("GAT_PREPARE_BATCH_SIZE", "16"))
    except ValueError as exc:
        raise ValueError("GAT_PREPARE_WORKERS y GAT_PREPARE_BATCH_SIZE deben ser enteros positivos") from exc
    if workers < 1 or batch_size < 1:
        raise ValueError("Procesos y lote de preparación deben ser positivos")
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return min(workers, available), batch_size, os.environ.get("GAT_PREPARE_DEVICE", "cpu")


def prepared_targets(targets, event_index, preprocessor, graph, config, directory, stats):
    """Yields one labelled sample per target, reusing inputs at equal timestamps.

    Only the parent writes samples and owns the optional GPU. Workers mmap the
    compact event index, never receive the full Python history per task, and
    keep bounded private interval caches. At most 2 * workers tasks are queued.
    """
    workers, batch_size, device = options()
    dtw = BatchDTW(config.dtw_band_fraction, device=device)
    grouped = [(stamp, list(group)) for stamp, group in groupby(targets, key=lambda t: t.timestamp)]
    stats.update(workers=workers, device=str(dtw.device), batch_size=batch_size,
                 labelled_targets=len(targets), unique_timestamps=len(grouped))
    provider = IndexedEventProvider.from_events(event_index)
    index_path = Path(directory) / ".preparation_index"
    pool = None
    started = monotonic()
    try:
        if workers > 1 and grouped:
            provider.save(index_path)
            provider.close()
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                initializer=_initialize, initargs=(str(index_path), preprocessor, graph, config))
        work_size = min(4, batch_size)
        batches = iter([grouped[i:i + work_size] for i in range(0, len(grouped), work_size)])
        pending = deque()
        def submit():
            try:
                group = next(batches)
            except StopIteration:
                return False
            stamps = [stamp for stamp, _ in group]
            task = pool.submit(_worker_build, stamps) if pool else None
            pending.append((group, task))
            return True
        for _ in range(2 * workers if pool else 1):
            if not submit():
                break
        ready_groups, ready_samples = [], []
        while pending:
            group, task = pending.popleft()
            samples = task.result() if task else _build([stamp for stamp, _ in group], provider, preprocessor, graph, config)
            submit()
            ready_groups.extend(group); ready_samples.extend(samples)
            if len(ready_samples) < batch_size and pending:
                continue
            dtw.apply(ready_samples)
            for (_, same_time), base in zip(ready_groups, ready_samples):
                coverage = base["metadata"]["scale_coverage"]
                entities = base["metadata"]["entity_ids"]
                for target in same_time:
                    y = [target.label if entity == target.entity_id else -1 for entity in entities]
                    yield target, {**base, "y": y,
                        "label_mask": [label >= 0 and any(scale["usable"] for scale in coverage[entity])
                                       for entity, label in zip(entities, y)],
                        "metadata": dict(base["metadata"])}
            ready_groups, ready_samples = [], []
        stats.update(dtw_cache_hits=dtw.hits, dtw_pairs_computed=dtw.misses,
                     duration_seconds=monotonic() - started)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        provider.close()
        if index_path.exists():
            shutil.rmtree(index_path)
