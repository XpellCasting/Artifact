"""Entrenamiento por sensor en procesos independientes, con cola acotada."""
from app.config import setting
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
import os

from threadpoolctl import threadpool_limits

from app.modulos.estadistico.residuos_secuenciales import RESIDUAL_METHODS
from app.modulos.estadistico.residuos_secuenciales import prepare_sensor


_thread_limits = None


def _initialize_worker():
    # Solo afecta al proceso hijo: evita cuatro procesos con cuatro hilos cada uno.
    global _thread_limits
    _thread_limits = threadpool_limits(limits=1)


def _train_sensor(temporal, entity_id, frame, method):
    try:
        if method not in RESIDUAL_METHODS:
            raise ValueError(f"Método continuo retirado: {method}; usa residual_focus_v1")
        return prepare_sensor(temporal, frame, entity_id, method), None
    except Exception as exc:
        raise RuntimeError(f"Error al entrenar el sensor {entity_id}: {exc}") from exc


def train_sensors(temporal, grouped, method, report):
    """Devuelve resultados a medida que terminan; solo envía un grupo por proceso.

    El callback corre en el padre. spawn evita heredar locks y pools numéricos
    de los hilos del API. HISTORY_TRAINING_WORKERS=1 conserva ejecución local.
    """
    try:
        requested = int(setting('HISTORY_TRAINING_WORKERS'))
    except ValueError as exc:
        raise ValueError("HISTORY_TRAINING_WORKERS debe ser un entero positivo") from exc
    if requested < 1:
        raise ValueError("HISTORY_TRAINING_WORKERS debe ser un entero positivo")
    total = len(grouped)
    if not total:
        report(100.0, "Entrenamiento temporal: 0/0 sensores completados")
        return
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    workers = min(requested, total, cpu_count)
    if workers == 1:
        for index, (entity_id, frame) in enumerate(grouped):
            report(100.0 * index / total,
                   f"Entrenando modelo temporal: {entity_id} ({index}/{total} sensores completados)")
            learned, runtime = _train_sensor(temporal, str(entity_id), frame, method)
            yield str(entity_id), learned, runtime
        report(100.0, f"Entrenamiento temporal: {total}/{total} sensores completados")
        return

    iterator = iter(grouped)
    completed = 0
    pending = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=_initialize_worker) as pool:
        def fill():
            while len(pending) < workers:
                try:
                    entity_id, frame = next(iterator)
                except StopIteration:
                    break
                future = pool.submit(_train_sensor, temporal, str(entity_id), frame, method)
                pending[future] = str(entity_id)

        def progress():
            active = ", ".join(pending.values()) or "ninguno"
            report(100.0 * completed / total,
                   f"Entrenamiento temporal: {completed}/{total} sensores completados; "
                   f"{workers} procesos; en curso: {active}")

        try:
            fill()
            progress()
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    entity_id = pending.pop(future)
                    learned, runtime = future.result()
                    completed += 1
                    yield entity_id, learned, runtime
                fill()
                progress()
        finally:
            for future in pending:
                future.cancel()
