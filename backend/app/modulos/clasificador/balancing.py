"""Equal class/episode contribution, fitted exclusively on train_fit labels."""
from collections import Counter
import numpy as np

from app.modulos.clasificador import LABELS

VERSION = "equal_class_episode_v1"


def episode_identity(metadata):
    episode = metadata.get("parent_episode_id") or metadata.get("sample_id")
    if not episode:
        raise ValueError("Cada muestra necesita parent_episode_id o sample_id para equilibrar episodios")
    source = (metadata.get("source_id") or metadata.get("source_dataset_id")
              or metadata.get("world_id") or metadata.get("source_sha256") or "unspecified")
    return str(source), str(episode)


class EpisodeBalance:
    def __init__(self, summaries, labels=LABELS, *, require_all=True):
        self.labels=tuple(labels)
        self.target_counts = Counter()
        self.rows = 0
        by_class = [set() for _ in self.labels]
        self.window_support = [0] * len(self.labels)
        for row in summaries:
            self.rows += 1
            identity = episode_identity(row["metadata"])
            for label, active in zip(row["y"], row["label_mask"]):
                if active:
                    label = int(label)
                    if label not in range(len(self.labels)):
                        raise ValueError("Etiqueta supervisada inválida")
                    by_class[label].add(identity)
                    self.target_counts[label, identity] += 1
                    self.window_support[label] += 1
        self.episode_support = [len(groups) for groups in by_class]
        if not any(self.episode_support) or (require_all and min(self.episode_support) == 0):
            raise ValueError(f"insufficient_labels: episodios por clase {self.episode_support}")
        total = sum(self.episode_support)
        self.supported=sum(count>0 for count in self.episode_support)
        self.class_weights = [total / (self.supported * count) if count else 0. for count in self.episode_support]
        # Fixed global denominator: batch-local weight sums would cancel the
        # rare-class correction in single-class batches (especially batch_size=1).
        self.normalizer = total / self.rows

    def target_weights(self, samples):
        if isinstance(samples, dict):
            labels = np.asarray(samples["y"])
            masks = np.asarray(samples["label_mask"], dtype=bool)
            metadata = samples["metadata"]
        else:
            labels = np.asarray([s["y"] for s in samples])
            masks = np.asarray([s["label_mask"] for s in samples], dtype=bool)
            metadata = [s["metadata"] for s in samples]
        result = np.zeros(labels.shape, dtype=np.float32)
        for row, meta in enumerate(metadata):
            identity = episode_identity(meta)
            for node in np.flatnonzero(masks[row]):
                key = int(labels[row, node]), identity
                if key not in self.target_counts:
                    raise ValueError("El lote contiene un episodio/clase ajeno a train_fit")
                result[row, node] = 1 / self.target_counts[key]
        return result

    def report(self):
        return {"version": VERSION, "fit_partition": "train_fit",
                "normalization": "fixed_train_fit_episode_mass_per_row",
                "class_weights": dict(zip(self.labels, self.class_weights)),
                "window_support": dict(zip(self.labels, self.window_support)),
                "episode_support": dict(zip(self.labels, self.episode_support)),
                "expected_class_mass_fraction": {label:1/self.supported if count else 0.
                    for label,count in zip(self.labels,self.episode_support)}}


def fault_supervision(samples):
    from app.modulos.clasificador.model_constants import FAULT_TYPES
    if isinstance(samples,dict):
        labels=np.asarray(samples["y"])
        masks=np.asarray(samples["label_mask"],dtype=bool)
        metadata=samples["metadata"]
    else:
        labels=np.asarray([row["y"] for row in samples])
        masks=np.asarray([row["label_mask"] for row in samples],dtype=bool)
        metadata=[row["metadata"] for row in samples]
    targets=np.full(labels.shape,-1,dtype=np.int64)
    active=np.zeros(masks.shape,dtype=bool)
    for index,meta in enumerate(metadata):
        fault=meta.get("fault_subtype")
        if meta.get("target_label") not in (2,"sensor_fault") or fault not in FAULT_TYPES:
            continue
        nodes=np.flatnonzero(masks[index])
        preferred=meta.get("next_entity") or meta.get("target_entity_id")
        entity_ids=meta.get("entity_ids",[])
        node=entity_ids.index(preferred) if preferred in entity_ids else int(nodes[0]) if len(nodes) else None
        if node is not None and masks[index,node]:
            targets[index,node]=FAULT_TYPES.index(fault);active[index,node]=True
    return {"y":targets,"label_mask":active,"metadata":metadata}
