"""Probe inputs for the official Vaani-Hi, from data/vaani_hi_official/matched.parquet (scripts/official_check.py).

  meta.parquet    metadata_idx, latitude, longitude, pincode, state, district (what the probe runners read)
  splits.parquet  metadata_idx, speaker, pincode, state, district,
                  prism          the official release's own split (PRiSM's), as train / val / test
                  speaker_split  our speaker-disjoint split (sal.data.splits, pincode-stratified, seed 42), built only
                                 over clips whose speaker was recovered; clips without a recovered speaker get
                                 "unmatched" and are left out of that protocol
Unmatched clips get a unique placeholder speaker so that speaker-cluster statistics stay conservative.
Run: uv run python scripts/official_prepare.py
"""
import pandas as pd

from sal.data.splits import leakage, speaker_disjoint_splits
from sal.paths import DATA

D = DATA / "vaani_hi_official"
m = pd.read_parquet(D / "matched.parquet")
assert m["metadata_idx"].is_unique, "metadata_idx is not unique in the official release"
split_map = {"train": "train", "validation": "val", "val": "val", "dev": "val", "test": "test"}
m["prism"] = m["split"].map(split_map)
assert m["prism"].notna().all(), f"unexpected split values {set(m['split'])}"
matched = m["speaker"].notna()
m.loc[~matched, "speaker"] = "unmatched_" + m.loc[~matched, "metadata_idx"].astype(str)
m["speaker_split"] = "unmatched"
m.loc[matched, "speaker_split"] = speaker_disjoint_splits(m[matched], speaker_col="speaker", group_col="pincode", seed=42)
m[["metadata_idx", "latitude", "longitude", "pincode", "state", "district"]].to_parquet(D / "meta.parquet", index=False)
m[["metadata_idx", "speaker", "pincode", "state", "district", "prism", "speaker_split"]].to_parquet(
    D / "splits.parquet", index=False)
print(f"{len(m)} rows, {matched.mean():.1%} with a recovered speaker")
print("official split sizes:", m["prism"].value_counts().to_dict())
print("leakage of the official split (matched clips):", leakage(m[matched], "prism", speaker_col="speaker"))
print("speaker-disjoint split sizes:", m["speaker_split"].value_counts().to_dict())
