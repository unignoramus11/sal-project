"""Scoring for the cross-validated benchmark (scripts/make_benchmark.py, scripts/benchmark_eval.py).

A probe is run once per fold of a track (run name ..._<track>_f<k>_bench_s<seed>). Each fold's test predictions
are averaged over seeds, and the folds are pooled, so every utterance of the track gets exactly one out-of-fold
prediction. Scores are computed on the pooled predictions; CIs resample speaker groups.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sal.probes import geo
from sal.stats import cluster_bootstrap, paired_cluster_bootstrap

RUN = re.compile(r"^(?P<cfg>.+)_(?P<track>all|shared)_f(?P<fold>\d+)_bench(?P<tag>.*)_s(?P<seed>\d+)$")


def find_runs(runs: Path, tag: str = "") -> dict[tuple[str, str], dict[int, list[Path]]]:
    """{(config, track): {fold: [run dirs]}} for finished benchmark runs whose name carries `tag` after _bench."""
    out: dict = {}
    for d in sorted(runs.glob("*_bench*_s*")):
        m = RUN.match(d.name)
        if m and m["tag"] == tag and (d / "predictions.parquet").exists():
            out.setdefault((m["cfg"], m["track"]), {}).setdefault(int(m["fold"]), []).append(d)
    return out


def pool_folds(fold_runs: dict[int, list[Path]]) -> pd.DataFrame:
    """One out-of-fold prediction per utterance: unit-normalised test predictions averaged over seeds within each
    fold, then concatenated over folds. Columns metadata_idx, x, y, z, fold, seeds."""
    parts = []
    for k, dirs in sorted(fold_runs.items()):
        p = pd.concat([pd.read_parquet(d / "predictions.parquet").query("split == 'test'") for d in dirs])
        v = p[["x", "y", "z"]].to_numpy()
        p[["x", "y", "z"]] = v / np.linalg.norm(v, axis=1, keepdims=True)
        e = p.groupby("metadata_idx")[["x", "y", "z"]].mean().reset_index()
        parts.append(e.assign(fold=k, seeds=len(dirs)))
    out = pd.concat(parts, ignore_index=True)
    if not out["metadata_idx"].is_unique:
        raise ValueError("an utterance is tested in more than one fold")
    return out


def speaker_macro(hit: np.ndarray, groups: np.ndarray) -> float:
    """Mean over speaker groups of each group's mean: every speaker counts once, however much they recorded."""
    return float(pd.Series(np.asarray(hit, float)).groupby(np.asarray(groups)).mean().mean())


class Scorer:
    """Scores pooled predictions of one track against its utterances (frame `u`: metadata_idx, latitude, longitude,
    state, group; one row per utterance of the track) with the track's pincodes as Recall@1 candidates."""

    def __init__(self, u: pd.DataFrame, reference: dict[str, np.ndarray] | None = None):
        self.u = u.reset_index(drop=True)
        self.t = torch.tensor(np.radians(self.u[["latitude", "longitude"]].to_numpy(np.float64)))
        loc = self.u.groupby(["latitude", "longitude"], sort=True)["state"].first().reset_index()
        self.cand = torch.tensor(np.radians(loc[["latitude", "longitude"]].to_numpy(np.float64)))
        self.cand_state = loc["state"].to_numpy()
        self.cxyz = geo.latlon_to_xyz(self.cand[:, 0], self.cand[:, 1]).numpy()
        self.groups = self.u["group"].to_numpy()
        self.reference = reference or {}            # name -> per-utterance hit of a baseline, for lifts

    def align(self, pred: pd.DataFrame) -> np.ndarray:
        p = pred.set_index("metadata_idx").reindex(self.u["metadata_idx"])
        if p[["x", "y", "z"]].isna().any().any():
            raise ValueError(f"{int(p['x'].isna().sum())} utterances of the track have no prediction")
        return p[["x", "y", "z"]].to_numpy(np.float64)

    def hits(self, pred_xyz: np.ndarray) -> dict[str, np.ndarray]:
        p = torch.tensor(pred_xyz, dtype=torch.float64)
        state = self.cand_state[np.argmax(pred_xyz @ self.cxyz.T, axis=1)] == self.u["state"].to_numpy()
        return {"recall1": 100 * geo.recall_at_k(p, self.t, 1, self.cand).numpy().astype(float),
                "state": 100 * state.astype(float), "km": geo.error_km(p, self.t).numpy()}

    def score(self, pred_xyz: np.ndarray, n: int = 1000) -> dict:
        h = self.hits(pred_xyz)
        r = {}
        for k in ("recall1", "state", "km"):
            ci = cluster_bootstrap(h[k], self.groups, n=n)
            r.update({k: ci["estimate"], f"{k}_lo": ci["lo"], f"{k}_hi": ci["hi"]})
        r["recall1_macro"] = speaker_macro(h["recall1"], self.groups)
        for name, ref in self.reference.items():
            for k in ("recall1", "state", "km"):
                d = paired_cluster_bootstrap(h[k], ref[k], self.groups, n=n)
                sign = -1 if k == "km" else 1                 # km: gain = reference error - probe error
                r.update({f"{k}_vs_{name}": sign * d["diff"], f"{k}_vs_{name}_lo": sign * (d["hi"] if sign < 0 else d["lo"]),
                          f"{k}_vs_{name}_hi": sign * (d["lo"] if sign < 0 else d["hi"])})
        r.update({"utterances": len(self.u), "groups": int(pd.Series(self.groups).nunique())})
        return r

    def paired(self, a_xyz: np.ndarray, b_xyz: np.ndarray, n: int = 1000) -> dict:
        ha, hb = self.hits(a_xyz), self.hits(b_xyz)
        r = {}
        for k in ("recall1", "state", "km"):
            d = paired_cluster_bootstrap(ha[k], hb[k], self.groups, n=n)
            r.update({f"{k}_diff": d["diff"], f"{k}_lo": d["lo"], f"{k}_hi": d["hi"], f"{k}_p": d["p"]})
        return r
