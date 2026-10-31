"""Checks of the prosody BiGRU and its late fusion with the transcript BiGRU (RQ3), speaker-disjoint split.

For every seed with both runs (prosody_gru_speaker_split_s<seed> and tp_zipa_rebuild_phone_gru_speaker_split_s<seed>),
the fused prediction uses the weight that scripts/fuse.py chose on validation (results/tables/fusion_*.csv). Paired
on the same test utterances, bootstrap over speakers:
  fusion vs transcript BiGRU      Recall@1 and mean km, on all test utterances and on the strict subset without
                                  look-alike speaker codes in training (scripts/make_benchmark.py groups)
  prosody BiGRU vs references     Recall@1 lift over the majority pincode, km gain over the best constant location,
                                  on all, look-alike and strict test utterances
Output: results/tables/prosody_checks.csv
Run:    uv run python scripts/prosody_checks.py
"""
import numpy as np
import pandas as pd
import torch

from sal.paths import DATA, RESULTS
from sal.probes import geo
from sal.stats import paired_cluster_bootstrap

TP, PR = "tp_zipa_rebuild_phone_gru_speaker_split", "prosody_gru_speaker_split"
runs = RESULTS / "runs"
meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet",
                       columns=["metadata_idx", "latitude", "longitude"]).set_index("metadata_idx")
sp = pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet").set_index("metadata_idx")
group = sp["speaker"].map(pd.read_csv(RESULTS / "benchmark" / "speaker_groups.csv", index_col=0)["group"])
tr = sp[sp["speaker_split"] == "train"]
fus = pd.read_csv(RESULTS / "tables" / f"fusion_{TP}__{PR}.csv").set_index("seed")


def unit(cfg, seed, idx):
    p = pd.read_parquet(runs / f"{cfg}_s{seed}" / "predictions.parquet").query("split == 'test'").set_index("metadata_idx")
    v = p.loc[idx, ["x", "y", "z"]].to_numpy(np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def xyz(lat, lon):
    return geo.latlon_to_xyz(torch.tensor(np.radians([lat])), torch.tensor(np.radians([lon]))).numpy()[0]


rows = []
for seed, w in fus["w"].items():
    idx = pd.read_parquet(runs / f"{TP}_s{seed}" / "predictions.parquet").query("split == 'test'")["metadata_idx"].to_numpy()
    t = torch.tensor(np.radians(meta.loc[idx, ["latitude", "longitude"]].to_numpy(np.float64)))
    cand = torch.unique(t, dim=0)
    a, b = unit(TP, seed, idx), unit(PR, seed, idx)
    maj = tr["pincode"].mode().iloc[0]
    m = np.tile(xyz(*meta.loc[tr.index[tr["pincode"] == maj][0]]), (len(idx), 1))
    c = np.tile(geo.best_constant(np.radians(meta.loc[tr.index, ["latitude", "longitude"]].to_numpy(np.float64))), (len(idx), 1))

    def hits(p):
        p = torch.tensor(p)
        return 100 * geo.recall_at_k(p, t, 1, cand).numpy().astype(float), geo.error_km(p, t).numpy()
    (ha, ka), (hb, kb), (hf, kf), (hm, _), (_, kc) = hits(a), hits(b), hits(w * a + (1 - w) * b), hits(m), hits(c)
    spk = sp.loc[idx, "speaker"].to_numpy()
    look = group.loc[idx].isin(set(group.loc[tr.index])).to_numpy()
    for part, k in (("all", np.ones(len(idx), bool)), ("lookalike", look), ("strict", ~look)):
        for what, x, y in (("fusion - transcript, Recall@1", hf, ha), ("fusion - transcript, km", kf, ka),
                           ("prosody - majority, Recall@1", hb, hm), ("best constant - prosody, km", kc, kb)):
            r = paired_cluster_bootstrap(x[k], y[k], spk[k], n=2000)
            rows.append({"seed": seed, "w": w, "subset": part, "comparison": what, "diff": r["diff"], "lo": r["lo"],
                         "hi": r["hi"], "p": r["p"], "utterances": int(k.sum()), "speakers": int(len(set(spk[k])))})
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "tables" / "prosody_checks.csv", index=False)
print(out.round(2).to_string(index=False))
