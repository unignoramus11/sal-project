"""How much non-dialect information each data version and split shares between training and test.

For every combination of data version (PRiSM's release, the corrected release, our rebuild on all pictures, on
shared pictures only, with control labels, and the planned benchmark) and split (PRiSM's random split, new speakers),
on the test utterances:
  same_audio     % whose byte-identical audio is in training
  same_code      % whose speaker code is in training (release: pseudo-speakers from voice clusters)
  same_group     % whose speaker group is in training; a group joins the look-alike codes of one district
                 (scripts/make_benchmark.py), so this also counts one person recorded under two codes
  est_speaker    % of test speakers estimated to be in training from calibrated voice embeddings (release only,
                 results/official/speaker_calibration.txt)
  lookup, majority, topic_lift
                 Recall@1 of the picture-ID lookup and of the majority location on the same test utterances and
                 labels, and their difference: how much the picture alone gives away. The release has no picture IDs.
  dialect        whether the location labels still follow the speakers' real locations (no for control labels)
The benchmark pools its five folds, so every utterance is tested once.

Output: results/tables/shared_cues.csv
Run:    uv run python scripts/shared_cues.py
"""
import hashlib
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sal.data.control import control_targets
from sal.paths import DATA, RESULTS

src = DATA / "vaani_hi_rebuild"
d = (pd.read_parquet(src / "splits.parquet")
       .merge(pd.read_parquet(src / "prompts.parquet")[["metadata_idx", "prompt_type", "image"]], on="metadata_idx"))
d["speaker"] = d["speaker"].astype(object)
d["group"] = d["speaker"].map(pd.read_csv(RESULTS / "benchmark" / "speaker_groups.csv", index_col=0)["group"])
au = pq.read_table(src / "vaani_hi_rebuild.parquet", columns=["metadata_idx", "audio"]).to_pandas()
au["h"] = [hashlib.blake2b(x["bytes"] if isinstance(x, dict) else x, digest_size=12).hexdigest() for x in au.pop("audio")]
d = d.merge(au, on="metadata_idx")
assert d["group"].notna().all() and len(d) == len(au)

# control labels: the pincode each speaker is reassigned to, built as in sal.data.control.control_targets
meta = pd.read_parquet(src / "vaani_hi_rebuild.parquet", columns=["metadata_idx", "speaker", "pincode", "latitude", "longitude"])
home = meta.groupby("speaker")["pincode"].agg(lambda s: s.mode().iloc[0])
assigned = pd.Series(np.random.default_rng(0).permutation(home.to_numpy()), index=home.index)
meta["control"] = meta["speaker"].map(assigned)
loc = meta.groupby("pincode")[["latitude", "longitude"]].first()
assert np.allclose(np.radians(loc.loc[meta["control"]].to_numpy(np.float64)), control_targets(meta))
d["control"] = d["metadata_idx"].map(meta.set_index("metadata_idx")["control"])

bench = pd.read_parquet(RESULTS / "benchmark" / "splits.parquet")
bench = bench.merge(d[["metadata_idx", "h"]], on="metadata_idx")


def score(tr, te, label="pincode"):
    """Shared-cue levels for one train/test pair of the rebuild (percent of test utterances)."""
    m = tr[label].mode().iloc[0]
    lk = tr.groupby("image")[label].agg(lambda s: s.mode().iloc[0])
    lookup = 100 * (te["image"].map(lk).fillna(m).to_numpy() == te[label].to_numpy()).mean()
    majority = 100 * (te[label].to_numpy() == m).mean()
    return {"n_test": len(te), "same_audio": 100 * te["h"].isin(set(tr["h"])).mean(),
            "same_code": 100 * te["speaker"].isin(set(tr["speaker"])).mean(),
            "same_group": 100 * te["group"].isin(set(tr["group"])).mean(),
            "lookup": lookup, "majority": majority, "topic_lift": lookup - majority}


def pooled(df, cols):
    """Score each fold, then pool over folds weighted by test size."""
    parts = [score(df[df[c] == "train"], df[df[c] == "test"]) for c in cols]
    n = np.array([p["n_test"] for p in parts])
    return {k: (int(n.sum()) if k == "n_test" else float(np.average([p[k] for p in parts], weights=n))) for k in parts[0]}


rows = []
# PRiSM's release and the corrected release
cal = (RESULTS / "official" / "speaker_calibration.txt").read_text()
est_released = float(re.search(r"estimated speaker leakage ([\d.]+)%", cal)[1])
est_pseudo = float(re.search(r"rebuild true test leakage ([\d.]+)%", cal)[1])
ah = pd.read_parquet(DATA / "vaani_hi_official" / "audio_hash.parquet", columns=["metadata_idx", "h"])
for data, path, splits in (("PRiSM's release", "splits_released.parquet", ("prism", "speaker_split")),
                           ("Corrected release", "splits_clean.parquet", ("prism", "speaker_split"))):
    o = pd.read_parquet(RESULTS / "official" / path).merge(ah, on="metadata_idx")
    for sp in splits:
        r = {"data": data, "split": sp, "dialect": "yes"}
        if sp not in o:
            r["built"] = False
            rows.append(r)
            continue
        tr, te = o[o[sp] == "train"], o[o[sp] == "test"]
        r.update(built=True, n_test=len(te), same_audio=100 * te["h"].isin(set(tr["h"])).mean(),
                 same_code=100 * te["speaker"].isin(set(tr["speaker"])).mean(),
                 est_speaker=est_released if sp == "prism" else est_pseudo)
        rows.append(r)

# our rebuild: all pictures, shared pictures, control labels
shared = d[d["prompt_type"] == "GENERIC"]
for data, df, label, dialect in (("Rebuild", d, "pincode", "yes"), ("Rebuild, shared pictures", shared, "pincode", "yes"),
                                 ("Rebuild, control labels", d, "control", "no")):
    for sp in ("prism", "speaker_split"):
        r = score(df[df[sp] == "train"], df[df[sp] == "test"], label)
        rows.append({"data": data, "split": sp, "dialect": dialect, "built": True, **r})

# the benchmark: five folds over speaker groups, all pictures and shared pictures
for data, track in (("Benchmark", "all"), ("Benchmark, shared pictures", "shared")):
    rows.append({"data": data, "split": "folds", "dialect": "yes", "built": True,
                 **pooled(bench, [f"{track}_f{k}" for k in range(5)])})

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "tables" / "shared_cues.csv", index=False)
print(out.round(1).to_string(index=False))
