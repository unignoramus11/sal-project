"""Check that a rebuilt Vaani-Hi holds the same audio as ours, row by row.

The dataset file's sha256 can differ between machines for reasons that do not touch the data (parquet writer
version, row-group sizes, column metadata). What the probes see is each row's audio bytes and labels, so this
compares those: for every metadata_idx, a BLAKE2b hash of the audio bytes, and the pincode, speaker and coordinates.

  --write   write the reference for this checkout's dataset to results/rebuild_checks/audio_hashes.csv
  (none)    compare this checkout's dataset with that reference and report every differing row

Run: python scripts/audio_hashes.py [--data data/vaani_hi_rebuild/vaani_hi_rebuild.parquet] [--write]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results" / "rebuild_checks" / "audio_hashes.csv"
COLS = ["metadata_idx", "pincode", "speaker", "latitude", "longitude"]


def hashes(path: Path) -> pd.DataFrame:
    f = pq.ParquetFile(path)
    parts = []
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=COLS + ["audio"]).to_pandas()
        t["audio_hash"] = [hashlib.blake2b(a["bytes"] if isinstance(a, dict) else a, digest_size=12).hexdigest()
                           for a in t.pop("audio")]
        parts.append(t)
    d = pd.concat(parts).sort_values("metadata_idx").reset_index(drop=True)
    d["speaker"] = d["speaker"].astype(str)
    for c in ("latitude", "longitude"):                  # float32 in the parquet: compare at 5 decimals (~1 m)
        d[c] = d[c].map(lambda x: f"{x:.5f}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    mine = hashes(args.data)
    if args.write:
        REF.parent.mkdir(parents=True, exist_ok=True)
        mine.to_csv(REF, index=False)
        print(f"wrote {REF} ({len(mine)} rows)")
        return
    ref = pd.read_csv(REF, dtype={"speaker": str, "latitude": str, "longitude": str})
    print(f"rows: {len(mine)} here, {len(ref)} in the reference")
    m = ref.merge(mine, on="metadata_idx", how="outer", suffixes=("_ref", ""), indicator=True)
    only_ref, only_here = (m["_merge"] == "left_only").sum(), (m["_merge"] == "right_only").sum()
    both = m[m["_merge"] == "both"]
    bad = 0
    for c in ["audio_hash", "pincode", "speaker", "latitude", "longitude"]:
        n = int((both[c] != both[f"{c}_ref"]).sum())
        bad += n
        print(f"{c:12s} {n} rows differ")
        if n:
            print(both.loc[both[c] != both[f"{c}_ref"], ["metadata_idx", f"{c}_ref", c]].head(5).to_string(index=False))
    print(f"rows only in the reference: {only_ref}; rows only here: {only_here}")
    ok = bad == 0 and only_ref == 0 and only_here == 0
    print("SAME DATA" if ok else "DIFFERENT DATA")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
