"""DTW en lotes con la misma distancia absoluta, banda y desempate del modelo."""
from collections import OrderedDict, defaultdict
import math

import numpy as np

from app.modulos.clasificador.segments import MAX_SEGMENTS


def distances(left, right, band_fraction, *, numeric=True, device="cpu"):
    """One independent pair per row; FP64 preserves the scalar reference semantics."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or len(left) != len(right) or not left.shape[1] or not right.shape[1]:
        raise ValueError("DTW requiere dos matrices de secuencias no vacías con igual número de pares")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("DTW requiere valores finitos")
    pairs, n = left.shape
    m = right.shape[1]
    band = math.ceil(band_fraction * max(n, m))
    if abs(n - m) > band:
        return [None] * pairs
    if str(device) == "cpu":
        xp = np
        cost = np.full((pairs, n + 1, m + 1), np.inf, dtype=np.float64)
        length = np.zeros((pairs, n + 1, m + 1), dtype=np.int64)
    else:
        import torch
        xp = torch
        left = torch.as_tensor(left, dtype=torch.float64, device=device)
        right = torch.as_tensor(right, dtype=torch.float64, device=device)
        cost = torch.full((pairs, n + 1, m + 1), float("inf"), dtype=torch.float64, device=device)
        length = torch.zeros((pairs, n + 1, m + 1), dtype=torch.long, device=device)
    cost[:, 0, 0] = 0.
    # Cells on the same anti-diagonal are independent: parallelize both pairs
    # and cells, rather than launching a GPU kernel per scalar recurrence.
    for diagonal in range(2, n + m + 1):
        ii = np.arange(max(1, diagonal - m), min(n, diagonal - 1) + 1)
        jj = diagonal - ii
        allowed = np.abs(ii - jj) <= band
        ii, jj = ii[allowed], jj[allowed]
        if not len(ii):
            continue
        if xp is not np:
            ii = xp.as_tensor(ii, dtype=xp.long, device=device)
            jj = xp.as_tensor(jj, dtype=xp.long, device=device)
        best_cost, best_length = cost[:, ii - 1, jj - 1], length[:, ii - 1, jj - 1]
        for a, b in ((ii - 1, jj), (ii, jj - 1)):
            other_cost, other_length = cost[:, a, b], length[:, a, b]
            take = (other_cost < best_cost) | ((other_cost == best_cost) & (other_length < best_length))
            best_cost = xp.where(take, other_cost, best_cost)
            best_length = xp.where(take, other_length, best_length)
        delta = abs(left[:, ii - 1] - right[:, jj - 1]) if numeric else (left[:, ii - 1] != right[:, jj - 1])
        cost[:, ii, jj] = best_cost + delta
        length[:, ii, jj] = best_length + 1
    result = cost[:, n, m] / length[:, n, m]
    return result.tolist() if xp is np else result.cpu().tolist()


class BatchDTW:
    def __init__(self, band_fraction, device="cpu", cache_size=8192, pair_batch_size=None):
        self.band_fraction, self.device = band_fraction, device
        # With <=24 points per segment, 8192 FP64 pairs keep the GPU working
        # while bounding its DP matrices to roughly 80 MiB plus temporaries.
        self.cache_size = cache_size
        self.pair_batch_size = pair_batch_size or (2048 if device == "cpu" else 8192)
        self.cache = OrderedDict()
        self.hits = self.misses = 0
        if device != "cpu":
            from app.modulos.clasificador.devices import training_device
            self.device = str(training_device(device))

    def apply(self, samples):
        locations = defaultdict(list)
        for sample in samples:
            for scale, node, numeric, sequences in sample.pop("_dtw_jobs", []):
                for i, left in enumerate(sequences):
                    for j, right in enumerate(sequences):
                        if i == j or left is None or right is None:
                            continue
                        # Raw symbolic keys preserve distinct unseen categories.
                        key_left, key_right = tuple(left), tuple(right)
                        locations[(numeric, key_left, key_right)].append((sample, scale, node, i, j))
        missing = defaultdict(list)
        values = {}
        for key, places in locations.items():
            if key[0] and not all(math.isfinite(value) for seq in key[1:] for value in seq):
                # The scalar numeric reference rejects nonfinite normalized
                # series, including overflow from otherwise finite readings.
                values[key] = None
            elif key in self.cache:
                self.hits += len(places)
                values[key] = self.cache[key]
                self.cache.move_to_end(key)
            else:
                self.misses += 1
                self.hits += len(places) - 1
                missing[(key[0], len(key[1]), len(key[2]))].append(key)
        for (numeric, _, _), keys in missing.items():
            for offset in range(0, len(keys), self.pair_batch_size):
                chunk = keys[offset:offset + self.pair_batch_size]
                if numeric:
                    left, right = [k[1] for k in chunk], [k[2] for k in chunk]
                else:
                    # A fresh codebook per independent pair is sufficient.
                    left, right = [], []
                    for _, a, b in chunk:
                        codes = {v: i for i, v in enumerate(dict.fromkeys((*a, *b)))}
                        left.append([codes[v] for v in a]); right.append([codes[v] for v in b])
                result = distances(left, right, self.band_fraction, numeric=numeric, device=self.device)
                for key, value in zip(chunk, result):
                    values[key] = value
                    self.cache[key] = value
        for key, places in locations.items():
            distance = values[key]
            similarity = 1 / (1 + distance) if distance is not None else 0.
            for sample, scale, node, i, j in places:
                sample["X"][scale * MAX_SEGMENTS + i][node][-MAX_SEGMENTS + j] = similarity
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return samples
