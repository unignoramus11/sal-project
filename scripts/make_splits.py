"""Fix the evaluation splits and write the dataset report.

Writes next to the dataset:
  splits.parquet   metadata_idx, speaker, pincode, state, district, prism (PRiSM random split), speaker (disjoint)
  report.json      counts, hours, speakers per pincode, utterances per speaker, leakage per protocol
  split_report.csv per-pincode utterance shares and speaker counts under each protocol

Run: uv run python scripts/make_splits.py data/vaani_hi_rebuild/vaani_hi_rebuild.parquet
"""
import json
import sys
from pathlib import Path

import pandas as pd

from sal.data.splits import leakage, speaker_disjoint_splits, split_report

data = Path(sys.argv[1])
cols = ["metadata_idx", "speaker", "pincode", "state", "district", "split", "gender", "language",
        "audio_seconds", "latitude", "longitude", "transcript_available"]
df = pd.read_parquet(data, columns=cols).rename(columns={"split": "prism"})
df["speaker_split"] = speaker_disjoint_splits(df, speaker_col="speaker", group_col="pincode", seed=42)

per_pin = df.groupby("pincode").agg(utts=("speaker", "size"), speakers=("speaker", "nunique"),
                                    top_share=("speaker", lambda s: s.value_counts().iloc[0] / len(s)))
per_spk = df.groupby("speaker").size()


def q(s):
    return {k: float(v) for k, v in s.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).items()}


report = {
    "dataset": str(data),
    "utterances": len(df),
    "hours": float(df["audio_seconds"].sum() / 3600),
    "mean_seconds": float(df["audio_seconds"].mean()),
    "pincodes": int(df["pincode"].nunique()),
    "districts": int(df.groupby(["state", "district"]).ngroups),
    "states": df["state"].value_counts().to_dict(),
    "speakers": int(df["speaker"].nunique()),
    "speakers_per_pincode": q(per_pin["speakers"]),
    "top_speaker_share_per_pincode": q(per_pin["top_share"]),
    "utterances_per_speaker": q(per_spk),
    "share_from_speakers_with_20plus": float(per_spk[per_spk >= 20].sum() / len(df)),
    "gender": df["gender"].value_counts(normalize=True).to_dict(),
    "language": df["language"].value_counts().to_dict(),
    "transcribed_share": float(df["transcript_available"].mean()),
    "protocols": {},
}
for proto in ("prism", "speaker_split"):
    report["protocols"][proto] = {
        "sizes": df[proto].value_counts().to_dict(),
        "leakage": leakage(df, proto, speaker_col="speaker"),
        "test_speakers": int(df.loc[df[proto] == "test", "speaker"].nunique()),
        "pincodes_in_all_splits": int((df.groupby("pincode")[proto].nunique() == 3).sum()),
    }

out = data.parent
df[["metadata_idx", "speaker", "pincode", "state", "district", "prism", "speaker_split"]].to_parquet(
    out / "splits.parquet", index=False)
reps = {p: split_report(df, p) for p in ("prism", "speaker_split")}
pd.concat(reps, axis=1).to_csv(out / "split_report.csv")
(out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({k: report[k] for k in ("utterances", "hours", "pincodes", "speakers", "protocols")}, indent=2))
