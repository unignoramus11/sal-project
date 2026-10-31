"""Aggregate results/runs/* into the study's tables.

A run directory is named <config>_s<seed>. For each config:
  - mean and sd over seeds of test Recall@1, mean km, median km (from metrics.json);
  - the seed ensemble: predicted directions averaged over seeds per test utterance, scored once, with a 95%
    speaker-cluster bootstrap CI for Recall@1 and mean km;
  - the same ensemble scored only on test utterances about GENERIC (shared) pictures, where the topic of
    the description cannot reveal the district.
Every score is paired with the majority baseline on the same test utterances (most frequent training
location of the same protocol and prompt subset): the shared-picture subset has a different location prior
(its majority location covers ~20% of test utterances, against ~10% overall), so raw Recall@1 values on it
are not comparable with the full test set. lift_* = probe minus majority, with speaker-bootstrap CI;
km_gain = best-constant km minus probe km (positive = better than any fixed guess).
Paired comparisons (same test utterances): difference in per-utterance hit rate and km error between two
configs' seed ensembles, speaker-cluster bootstrap CI and p-value.

Output: results/tables/configs.csv and results/tables/comparisons.csv; prints both.
Run:    uv run python scripts/summarise.py [config_a config_b ...]   (extra args: pairs to compare)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sal.data.prompts import prompt_type
from sal.paths import DATA, RESULTS
from sal.probes import geo
from sal.stats import cluster_bootstrap, paired_cluster_bootstrap

runs = RESULTS / "runs"
coords = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet",
                         columns=["metadata_idx", "latitude", "longitude"]).set_index("metadata_idx")
splits = pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet").set_index("metadata_idx")
splits["generic"] = prompt_type(splits.index) == "GENERIC"


def majority_scores(e: pd.DataFrame, cfg: str) -> pd.DataFrame:
    """Hit and km of the majority baseline (train split of cfg's protocol and prompt subset) on e's utterances,
    and km of the best constant location (lowest mean km over the same training utterances)."""
    proto = "speaker_split" if "_speaker_split" in cfg else "prism"
    tr = splits[splits[proto] == "train"]
    if cfg.endswith("_generic"):
        tr = tr[tr["generic"]]
    pin = tr["pincode"].mode().iloc[0]
    lat, lon = coords.loc[tr.index[tr["pincode"] == pin][0]]
    xyz = geo.latlon_to_xyz(torch.tensor([np.radians(lat)]), torch.tensor([np.radians(lon)]))
    t = torch.tensor(np.radians(coords.loc[e["metadata_idx"], ["latitude", "longitude"]].to_numpy()))
    pred = xyz.expand(len(e), 3).double()
    const = torch.tensor(geo.best_constant(np.radians(coords.loc[tr.index, ["latitude", "longitude"]].to_numpy())))
    return pd.DataFrame({"hit": geo.recall_at_k(pred, t, 1).numpy(), "km": geo.error_km(pred, t).numpy(),
                         "const_km": geo.error_km(const.expand(len(e), 3), t).numpy()}, index=e.index)


def config_of(name: str) -> tuple[str, int]:
    m = re.match(r"(.+)_s(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def ensemble(config: str, split: str = "test") -> pd.DataFrame:
    """Per-utterance seed-averaged prediction with hit and km error."""
    parts = []
    for d in sorted(runs.glob(f"{config}_s*")):
        p = pd.read_parquet(d / "predictions.parquet")
        p = p[p["split"] == split]
        v = p[["x", "y", "z"]].to_numpy()
        p[["x", "y", "z"]] = v / np.linalg.norm(v, axis=1, keepdims=True)
        parts.append(p)
    df = pd.concat(parts).groupby("metadata_idx").agg(
        x=("x", "mean"), y=("y", "mean"), z=("z", "mean"), speaker=("speaker", "first"),
        state=("state", "first"), n_seeds=("x", "size")).reset_index()
    t = np.radians(coords.loc[df["metadata_idx"], ["latitude", "longitude"]].to_numpy())
    pred = torch.tensor(df[["x", "y", "z"]].to_numpy())
    tt = torch.tensor(t)
    df["hit"] = geo.recall_at_k(pred, tt, 1).numpy()
    df["km"] = geo.error_km(pred, tt).numpy()
    return df


def default_pairs(have: set[str]) -> list[str]:
    """The study's standard comparisons (a, b), kept where both configs exist."""
    out = []
    for proto in ("speaker_split", "prism"):
        for m in ("w2v2", "zipa"):
            tp = f"tp_{m}_rebuild"
            out += [(f"{tp}_phone_bigram_{proto}", f"{tp}_phone_unigram_{proto}"),
                    (f"{tp}_char_bigram_{proto}", f"{tp}_char_unigram_{proto}"),
                    (f"{tp}_phone_bigram_{proto}", f"{tp}_char_bigram_{proto}"),
                    (f"{tp}_phone_gru_{proto}", f"{tp}_phone_shuffle_{proto}"),
                    (f"{tp}_phone_gru_{proto}", f"{tp}_char_gru_{proto}"),
                    (f"{tp}_phone_gru_{proto}", f"{tp}_phone_bigram_{proto}"),
                    (f"{tp}_phone_bigram_{proto}", f"dense_ecapa_rebuild_{proto}")]
        for k in ("phone_bigram", "char_bigram", "phone_gru", "char_gru", "phone_shuffle"):
            out.append((f"tp_zipa_rebuild_{k}_{proto}", f"tp_w2v2_rebuild_{k}_{proto}"))
    return [c for a, b in out if a in have and b in have for c in (a, b)]


def main():
    rows = {}
    for d in sorted(runs.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        if "_official_" in d.name or "_bench" in d.name:   # other data/splits: official_eval.py, benchmark_eval.py
            continue
        cfg, seed = config_of(d.name)
        m = json.loads((d / "metrics.json").read_text())["metrics"]["test"]
        rows.setdefault(cfg, []).append(m)
    table = []
    for cfg, ms in rows.items():
        r1 = np.array([m["recall1"] for m in ms])
        km = np.array([m["mean_km"] for m in ms])
        med = np.array([m["median_km"] for m in ms])
        e = ensemble(cfg)
        ci_r = cluster_bootstrap(100 * e["hit"].to_numpy(), e["speaker"].to_numpy(), n=1000)
        ci_k = cluster_bootstrap(e["km"].to_numpy(), e["speaker"].to_numpy(), n=1000)
        is_g = prompt_type(e["metadata_idx"]) == "GENERIC"
        g = e[is_g]                                              # shared pictures only: no topic cue at test
        ci_g = cluster_bootstrap(100 * g["hit"].to_numpy(), g["speaker"].to_numpy(), n=1000)
        mj = majority_scores(e, cfg)
        spk = e["speaker"].to_numpy()
        lift = paired_cluster_bootstrap(100 * e["hit"], 100 * mj["hit"], spk, n=1000)
        gain = paired_cluster_bootstrap(mj["const_km"], e["km"], spk, n=1000)
        glift = paired_cluster_bootstrap(100 * g["hit"], 100 * mj.loc[is_g, "hit"], g["speaker"].to_numpy(), n=1000)
        ggain = paired_cluster_bootstrap(mj.loc[is_g, "const_km"], g["km"], g["speaker"].to_numpy(), n=1000)
        table.append({"config": cfg, "seeds": len(ms), "recall1_mean": r1.mean(), "recall1_sd": r1.std(ddof=1) if len(ms) > 1 else 0,
                      "km_mean": km.mean(), "km_sd": km.std(ddof=1) if len(ms) > 1 else 0, "median_km": med.mean(),
                      "ens_recall1": ci_r["estimate"], "ens_recall1_lo": ci_r["lo"], "ens_recall1_hi": ci_r["hi"],
                      "ens_km": ci_k["estimate"], "ens_km_lo": ci_k["lo"], "ens_km_hi": ci_k["hi"],
                      "generic_recall1": ci_g["estimate"], "generic_recall1_lo": ci_g["lo"],
                      "generic_recall1_hi": ci_g["hi"], "generic_km": float(g["km"].mean()), "generic_n": len(g),
                      "maj_recall1": 100 * mj["hit"].mean(), "const_km": mj["const_km"].mean(),
                      "lift_recall1": lift["diff"], "lift_recall1_lo": lift["lo"], "lift_recall1_hi": lift["hi"],
                      "km_gain": gain["diff"], "km_gain_lo": gain["lo"], "km_gain_hi": gain["hi"],
                      "generic_maj_recall1": 100 * mj.loc[is_g, "hit"].mean(), "generic_const_km": mj.loc[is_g, "const_km"].mean(),
                      "generic_lift_recall1": glift["diff"], "generic_lift_lo": glift["lo"], "generic_lift_hi": glift["hi"],
                      "generic_km_gain": ggain["diff"], "generic_km_gain_lo": ggain["lo"], "generic_km_gain_hi": ggain["hi"],
                      "test_speakers": ci_r["n_clusters"]})
    out = RESULTS / "tables"
    out.mkdir(exist_ok=True)
    t = pd.DataFrame(table).sort_values("config")
    t.to_csv(out / "configs.csv", index=False)
    print(t.round(1).to_string(index=False))

    pairs = sys.argv[1:] or default_pairs(set(t["config"]))
    comps = []
    for a, b in zip(pairs[0::2], pairs[1::2]):
        ea, eb = ensemble(a).set_index("metadata_idx"), ensemble(b).set_index("metadata_idx")
        common = ea.index.intersection(eb.index)
        ea, eb = ea.loc[common], eb.loc[common]
        rh = paired_cluster_bootstrap(100 * ea["hit"], 100 * eb["hit"], ea["speaker"].to_numpy(), n=2000)
        rk = paired_cluster_bootstrap(ea["km"], eb["km"], ea["speaker"].to_numpy(), n=2000)
        comps.append({"a": a, "b": b, "n": len(common), "recall1_diff": rh["diff"], "recall1_lo": rh["lo"],
                      "recall1_hi": rh["hi"], "recall1_p": rh["p"], "km_diff": rk["diff"], "km_lo": rk["lo"],
                      "km_hi": rk["hi"], "km_p": rk["p"]})
    if comps:
        c = pd.DataFrame(comps)
        c.to_csv(out / "comparisons.csv", index=False)
        print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
