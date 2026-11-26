"""Write the public test set: the speaker-disjoint test split of our Vaani-Hi rebuild, with audio and labels.

The released checkpoints (checkpoints/) were trained on the training part of this split, so no test speaker was
heard in training. The folder this script writes is what we share on Google Drive:

  testset.parquet            one row per utterance: metadata_idx, audio (16 kHz mono WAV bytes), pincode, latitude,
                             longitude, state, district, speaker, gender, audio_seconds, picture (the image prompt)
                             and picture_type (GENERIC = a shared picture, otherwise tied to the speaker's district)
  transcripts_w2v2.parquet   the W2V2 phone transcripts of these utterances, as written by scripts/transcribe.py
  transcripts_zipa.parquet   the same for ZIPA
  README.txt                 source, licence and how to use the files
  SHA256SUMS                 checksums of the files above

Run:  uv run python scripts/export_testset.py ~/Downloads/sal_testset
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from sal import testset
from sal.paths import DATA, OUTPUTS

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--split", default="speaker_split")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

rebuild = DATA / "vaani_hi_rebuild"
splits = pd.read_parquet(rebuild / "splits.parquet")
test_ids = set(splits.loc[splits[args.split] == "test", "metadata_idx"])
cols = ["metadata_idx", "audio", "pincode", "latitude", "longitude", "state", "district", "speaker", "gender",
        "audio_seconds"]
parts = []
for batch in pq.ParquetFile(rebuild / "vaani_hi_rebuild.parquet").iter_batches(batch_size=2000, columns=cols):
    b = batch.to_pandas()
    parts.append(b[b["metadata_idx"].isin(test_ids)])
test = pd.concat(parts, ignore_index=True)
prompts = pd.read_parquet(rebuild / "prompts.parquet").rename(columns={"image": "picture",
                                                                       "prompt_type": "picture_type"})
test = test.merge(prompts[["metadata_idx", "picture", "picture_type"]], on="metadata_idx", how="left")
test = test.sort_values("metadata_idx").reset_index(drop=True)
assert len(test) == len(test_ids), (len(test), len(test_ids))
test.to_parquet(args.out / "testset.parquet", index=False)

for m in ("w2v2", "zipa"):
    tr = pd.read_parquet(OUTPUTS / "transcripts" / f"{m}_rebuild.parquet",
                         columns=["metadata_idx", "processed_transcript", "tokens"])
    tr = tr[tr["metadata_idx"].isin(test_ids)].sort_values("metadata_idx").reset_index(drop=True)
    assert len(tr) == len(test_ids), (m, len(tr))
    tr.to_parquet(args.out / f"transcripts_{m}.parquet", index=False)

hours = test["audio_seconds"].sum() / 3600
(args.out / "README.txt").write_text(f"""Test set for "Order, Duration, or Melody? Probing what phone recognizers use to geolocate
Haryanvi and Hindi-belt speech" (Speech Analysis and Linguistics, Monsoon 2026, IIIT Hyderabad).
Code: https://github.com/unignoramus11/sal-project

What it is
  The test part of the speaker-disjoint split of our rebuild of Vaani-Hi (the data of PRiSM's geolocation task):
  {len(test):,} utterances, {hours:.1f} hours, {test['speaker'].nunique():,} speakers, {test['pincode'].nunique()} pincodes.
  No speaker in this set appears in the training data of the released checkpoints.

Files
  testset.parquet           metadata_idx, audio (16 kHz mono WAV bytes), pincode, latitude, longitude, state,
                            district, speaker, gender, audio_seconds, picture, picture_type
                            (picture_type GENERIC = a shared picture; any other value = a picture of the
                            speaker's district, which can reveal the location by itself)
  transcripts_w2v2.parquet  W2V2 phone transcripts (metadata_idx, processed_transcript, tokens)
  transcripts_zipa.parquet  ZIPA phone transcripts (same columns)

Use
  Probe only (no GPU needed), from the repository root:
    uv run python scripts/infer.py --checkpoint checkpoints/<name>.pt \\
        --transcripts transcripts_zipa.parquet --labels testset.parquet --out predictions.csv
  From the audio (Linux with an NVIDIA GPU, after bash tools/setup_env.sh):
    envs/prism/bin/python scripts/transcribe.py --model zipa --data testset.parquet --out zipa_test.parquet
    uv run python scripts/infer.py --checkpoint checkpoints/<name>.pt --transcripts zipa_test.parquet \\
        --labels testset.parquet --out predictions.csv

Source and licence
  Audio and labels come from the Vaani dataset (ARTPARK and IISc, https://huggingface.co/datasets/ARTPARK-IISc/Vaani),
  released under CC BY 4.0. The selection follows PRiSM (Bharadwaj et al., ACL 2026). Latitude and longitude are
  PRiSM's per-pincode coordinates, the labels the checkpoints were trained on. This subset is shared under the same
  licence; please cite Vaani and PRiSM when using it.
""")

sums = [f"{hashlib.sha256((args.out / f).read_bytes()).hexdigest()}  {f}"
        for f in ("testset.parquet", "transcripts_w2v2.parquet", "transcripts_zipa.parquet", "README.txt")]
(args.out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
print(f"wrote {args.out}: {len(test):,} utterances, {hours:.1f} h")
changed = testset.failures(args.out)
if changed:
    print(f"NOTE: {changed} differ from the published test set (sal.testset.SHA256). Publishing these files means "
          "replacing them on Drive and updating SHA256 in src/sal/testset.py in the same commit; otherwise "
          "scripts/infer.py refuses the download.")
else:
    print("identical to the published test set")
