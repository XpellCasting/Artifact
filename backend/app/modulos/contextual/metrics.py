"""Documented association measures and bounded numeric DTW for F2."""
import math
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats.contingency import association
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from dtaidistance import dtw


def library_versions():
    return {name: version(name) for name in
            ("scipy", "scikit-learn", "dtaidistance", "networkx", "threadpoolctl")}


def signed_association(pair, method="pearson"):
    values = pair[["source", "target"]].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2 or any(values[key].nunique() < 2 for key in values):
        return None
    function = spearmanr if method == "spearman" else pearsonr
    result = float(function(values.source.to_numpy(), values.target.to_numpy()).statistic)
    return result if math.isfinite(result) else None


def nominal_association(source, target):
    pair = pd.concat([source.rename("source"), target.rename("target")], axis=1).dropna()
    if pair.empty or pair.source.nunique() < 2 or pair.target.nunique() < 2:
        return {"cramers_v": None, "mutual_information_nats": None, "observations": len(pair)}
    # Factor IDs represent equality only; neither metric uses their magnitude.
    left = pd.factorize(pair.source.astype(str))[0]
    right = pd.factorize(pair.target.astype(str))[0]
    table = pd.crosstab(left, right).to_numpy()
    return {"cramers_v": float(association(table, method="cramer")),
            "mutual_information_nats": float(mutual_info_score(left, right)),
            "observations": len(pair)}


def numeric_dtw(pair, direction=1., max_points=256, band_fraction=.1):
    """Compare aligned, valid windows; never close missing time gaps."""
    values = pair[["source", "target"]].to_numpy(dtype=np.double)
    if len(values) < 2 or not np.isfinite(values).all():
        return None
    # Missing windows split trajectories. Choose the longest contiguous run.
    if isinstance(pair.index, pd.DatetimeIndex) and len(pair) > 2:
        differences = np.diff(pair.index.asi8)
        cadence = np.min(differences[differences > 0]) if np.any(differences > 0) else 0
        if cadence:
            cuts = np.flatnonzero(differences > cadence) + 1
            values = max(np.split(values, cuts), key=len)
    if len(values) < 2:
        return None
    original_points = len(values)
    if len(values) > max_points:
        values = values[np.linspace(0, len(values)-1, max_points, dtype=int)]
    centre = np.median(values, axis=0)
    scale = np.std(values, axis=0)
    if np.any(scale <= np.finfo(float).eps):
        return None
    values = (values-centre)/scale
    left = np.ascontiguousarray(values[:, 0], dtype=np.double)
    right = np.ascontiguousarray(values[:, 1] * (1. if direction >= 0 else -1.), dtype=np.double)
    window = max(1, int(math.ceil(len(left)*band_fraction)))
    distance = float(dtw.distance_fast(left, right, window=window, use_pruning=False))
    distance /= math.sqrt(len(left))
    return {"distance": distance, "similarity": 1./(1.+distance),
            "points": len(left), "original_points": original_points,
            "window_points": window, "backend": "dtaidistance-c",
            "definition": "squared_euclidean_sqrt_div_sqrt_points",
            "normalization": "median_std_per_series", "direction_aligned": True}


def mixed_information(source, target, source_kind):
    pair = pd.concat([source.rename("source"), target.rename("target")], axis=1).dropna()
    numeric_key, category_key = ("source", "target") if source_kind == "continuous" else ("target", "source")
    numeric = pd.to_numeric(pair[numeric_key], errors="coerce")
    pair = pair[np.isfinite(numeric)]
    numeric = numeric.loc[pair.index]
    if len(pair) < 5 or numeric.nunique() < 2 or pair[category_key].nunique() < 2:
        return {"mutual_information_nats": None, "observations": len(pair)}
    category = pd.factorize(pair[category_key].astype(str))[0]
    if not np.any(np.bincount(category) > 1):
        return {"mutual_information_nats": None, "observations": len(pair)}
    score = mutual_info_regression(category.reshape(-1, 1), numeric.to_numpy(),
                                   discrete_features=True, n_neighbors=3, random_state=42)[0]
    return {"mutual_information_nats": float(score), "observations": len(pair)}
