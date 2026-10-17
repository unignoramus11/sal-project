"""Speaker-cluster bootstrap.

Utterances of one speaker are correlated (same voice, microphone, room, often the same picture prompt), so
resampling utterances understates uncertainty. All intervals here resample speakers with replacement and
take all of a drawn speaker's utterances.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _groups(clusters: np.ndarray) -> list[np.ndarray]:
    _, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean,
                      n: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Point estimate and percentile CI of stat(values) under resampling of clusters."""
    values = np.asarray(values)
    groups = _groups(np.asarray(clusters))
    rng = np.random.default_rng(seed)
    boots = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = stat(values[np.concatenate([groups[i] for i in pick])])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(stat(values)), "lo": float(lo), "hi": float(hi), "n_clusters": len(groups)}


def paired_cluster_bootstrap(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, n: int = 1000, seed: int = 0,
                             alpha: float = 0.05) -> dict:
    """Difference of means mean(a) - mean(b) for two systems scored on the same items, resampling clusters.

    Returns the estimate, CI and a two-sided p-value (share of bootstrap differences on the other side of 0,
    doubled)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    groups = _groups(np.asarray(clusters))
    rng = np.random.default_rng(seed)
    boots = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        boots[k] = d[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(min(p, 1.0)),
            "n_clusters": len(groups)}
