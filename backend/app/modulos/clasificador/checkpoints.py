"""Checkpoints por época y procedencia de un entrenamiento independiente."""
from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch

from app.modulos.clasificador.digests import sha256_file


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def provenance():
    directory = Path(__file__).resolve().parent
    sources = {path.name: hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
               for path in sorted(directory.glob("*.py"))}
    versions = {}
    for name in ("torch", "torch-geometric", "numpy", "tzdata"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    root = directory.parents[3]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {"code_sha256": sources, "git_revision": revision,
            "library_versions": versions, "python": platform.python_version(),
            "platform": platform.platform(), "cuda_runtime": torch.version.cuda}


def dataset_identity(dataset):
    """La ubicación puede cambiar entre equipos; el contenido debe coincidir."""
    directory = Path(dataset["sequence_dir"])
    manifest = {key: value for key, value in dataset.items() if key != "sequence_dir"}
    inventory = {path.relative_to(directory).as_posix(): sha256_file(path)
                 for path in sorted(directory.rglob("*")) if path.is_file()}
    if not inventory:
        raise ValueError("El dataset no contiene secuencias para entrenar")
    return {"dataset_id": dataset["dataset_id"],
            "manifest_sha256": hashlib.sha256(canonical(manifest).encode()).hexdigest(),
            "files_sha256": inventory}


def sample_identity(train, dev):
    """Identidad para llamadas directas al entrenador sin un manifiesto persistido."""
    digest = hashlib.sha256()
    for samples in (train, dev):
        digest.update(str(len(samples)).encode())
        for row in samples:
            for field in ("X", "state_ids", "node_present", "adjacency", "y", "label_mask"):
                array = np.asarray(row[field])
                digest.update(canonical([field, str(array.dtype), list(array.shape)]).encode())
                digest.update(array.tobytes())
            digest.update(canonical(row["metadata"]).encode())
    return {"samples_sha256": digest.hexdigest()}


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(item) for item in value)
    return value


def random_state(device):
    numpy_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_random_state(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gaussian, cached))
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


@contextmanager
def directory_lock(directory):
    """El SO libera el bloqueo incluso si se mata el proceso."""
    with (directory / ".lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Otro proceso está usando el directorio de checkpoints") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class TrainingRun:
    def __init__(self, directory, config, identity, resume, environment):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.lock = directory_lock(self.directory)
        self.lock.__enter__()
        try:
            parameters = {key: value for key, value in config.as_dict().items()
                          if key not in {"device", "max_epochs"}}
            libraries = {key: (value.split("+")[0] if value else None)
                         for key, value in environment["library_versions"].items()}
            contract = {"version": 1, "config": parameters, "dataset": identity,
                        "code_sha256": environment["code_sha256"], "libraries": libraries}
            self.fingerprint = hashlib.sha256(canonical(contract).encode()).hexdigest()
            path = self.directory / "run.json"
            if resume:
                if not path.is_file():
                    raise ValueError("No hay checkpoints para reanudar en este directorio")
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("fingerprint") != self.fingerprint:
                    raise ValueError("Checkpoint incompatible: cambió el dataset, la configuración, el código o las dependencias")
            else:
                if path.exists() or list(self.directory.glob("trial-*.pt")):
                    raise ValueError("El directorio ya contiene un entrenamiento; usa --resume o un directorio nuevo")
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({**contract, "fingerprint": self.fingerprint,
                    "provenance": environment}, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, path)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.lock.__exit__(None, None, None)

    def load(self, trial):
        path = self.directory / f"trial-{trial:02d}.pt"
        if not self.resume or not path.is_file():
            return None
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("fingerprint") != self.fingerprint or checkpoint.get("trial") != trial:
            raise ValueError("Checkpoint incompatible con este entrenamiento")
        return checkpoint

    def save(self, trial, state):
        path = self.directory / f"trial-{trial:02d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        try:
            torch.save({**cpu_state(state), "version": 1, "trial": trial,
                        "fingerprint": self.fingerprint}, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
