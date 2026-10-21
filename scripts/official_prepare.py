"""Probe inputs for the official Vaani-Hi: the release as published, and a corrected version.

released  every row with the released labels and split (protocol "prism"); "speaker" = ECAPA pseudo-speaker
          (scripts/speaker_calibration.py), used only for cluster bootstrap CIs
clean     one row per distinct recording; recordings whose copies carry different pincodes are dropped (their true
          label is unknown); pincode 171202 (Shimla) moved to the median of its offices with valid coordinates
          (results/official/coordinates_audit.csv); protocols "prism" (the released split, restricted to these
          rows; a recording with copies in several splits keeps the split of its train copy if it has one, else
          its first copy) and "speaker_split" (pincode-stratified split over pseudo-speakers, which the
          rebuild calibration puts at 2.1% true test leakage)
Output (small, versioned for Ada): results/official/{meta,splits}_{released,clean}.parquet
Run:   uv run python scripts/official_prepare.py
"""
import pandas as pd

from sal.data.splits import leakage, speaker_disjoint_splits
from sal.paths import DATA, RESULTS

D, OUT = DATA / "vaani_hi_official", RESULTS / "official"
h = pd.read_parquet(D / "audio_hash.parquet")[["metadata_idx", "h", "split", "pincode", "state", "district",
                                               "latitude", "longitude"]]
ps = pd.read_parquet(D / "pseudo_speakers.parquet")
d = h.merge(ps, on="metadata_idx")
assert len(d) == len(h) and d["metadata_idx"].is_unique
d["speaker"] = d["pseudo_speaker"]
d["prism"] = d["split"]

meta_cols = ["metadata_idx", "latitude", "longitude", "pincode", "state", "district"]
d[meta_cols].to_parquet(OUT / "meta_released.parquet", index=False)
d[["metadata_idx", "speaker", "pincode", "state", "district", "prism"]].to_parquet(OUT / "splits_released.parquet", index=False)

# clean version
consistent = d.groupby("h")["pincode"].transform("nunique") == 1
c = d[consistent].copy()
order = {"train": 0, "val": 1, "test": 2}
c = c.sort_values(["h", "prism"], key=lambda s: s.map(order) if s.name == "prism" else s).drop_duplicates("h", keep="first")
coord = pd.read_csv(OUT / "coordinates_audit.csv").set_index("pincode")
fix = coord[coord["km_from_valid_median"] > 100]
for p, r in fix.iterrows():
    c.loc[c["pincode"] == p, ["latitude", "longitude"]] = [r["valid_median_lat"], r["valid_median_lon"]]
c["speaker_split"] = speaker_disjoint_splits(c, speaker_col="speaker", group_col="pincode", seed=42)
c[meta_cols].to_parquet(OUT / "meta_clean.parquet", index=False)
c[["metadata_idx", "speaker", "pincode", "state", "district", "prism", "speaker_split"]].to_parquet(
    OUT / "splits_clean.parquet", index=False)

print(f"released: {len(d)} rows, split {d['prism'].value_counts().to_dict()}")
print(f"clean: {len(c)} rows ({len(d) - len(c)} removed: {int((~consistent).sum())} rows of recordings with conflicting "
      f"pincodes, the rest duplicate copies); pincodes moved: {list(fix.index)}")
print("clean prism split:", c["prism"].value_counts().to_dict(), "| pseudo-speaker leakage", leakage(c, "prism"))
print("clean speaker_split:", c["speaker_split"].value_counts().to_dict(), "| pseudo-speaker leakage",
      leakage(c, "speaker_split"))
