"""Recover the Vaani source clip of every official Vaani-Hi row (speaker, true pincode and gender).

The release has no file names, and the Vaani revision PRiSM sampled from (December 2025) was re-sharded and its
files purged, so the source cannot be read back by shard and row. Instead, for the 20 official districts this
reads only the small metadata columns of every shard at the current revision (file name, duration, pincode,
gender, picture), with no audio bytes, and matches each official clip to the source clips of its district by
duration (Vaani's duration equals the clip length to within 0.1 ms; the release clips at 20 s). A match is
accepted only when exactly one source clip of the district has that duration. The speaker comes from the file
name (sal.data.rebuild.speaker_from_path).

Validation: official clips matched sample-for-sample to our cached Vaani shards (scripts/official_check.py) must
get the same source file here.

Output: data/vaani_hi_official/sources_parts/*.parquet (resumable), data/vaani_hi_official/sources.parquet
        (official metadata_idx -> source file, speaker, source pincode, gender, picture, match status)
Run:    uv run python scripts/official_sources.py --workers 8
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from sal.data.rebuild import speaker_from_path
from sal.paths import DATA

REPO, REV = "ARTPARK-IISc/Vaani", "9e6ed4f94f7060e633cbc500480623ae7521522c"
D = DATA / "vaani_hi_official"
PARTS = D / "sources_parts"
COLS = ["audio.path", "duration", "pincode", "gender", "referenceImage"]
TOL = 1e-4                       # seconds


def fetch(path: str) -> int:
    out = PARTS / (path.replace("audio/", "").replace("/", "__"))
    if out.exists():
        return pq.ParquetFile(out).metadata.num_rows
    fs = HfFileSystem()
    for attempt in range(4):
        try:
            with fs.open(f"datasets/{REPO}@{REV}/{path}", "rb", block_size=2 ** 20) as f:
                t = pq.ParquetFile(f).read(columns=COLS).to_pandas()
            break
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
    t["source_path"] = t.pop("audio").map(lambda a: a["path"])
    t["source_shard"] = path
    t["pincode"] = pd.to_numeric(t["pincode"], errors="coerce")
    tmp = out.with_suffix(".tmp")
    t.to_parquet(tmp, index=False)
    tmp.rename(out)
    return len(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    off = pd.read_parquet(D / "matched.parquet", columns=["metadata_idx", "state", "district", "pincode", "gender",
                                                          "seconds", "split", "source_path"])
    api = HfApi()
    shards = []
    for st, di in sorted(set(zip(off["state"], off["district"], strict=True))):
        shards += sorted(f.path for f in api.list_repo_tree(REPO, path_in_repo=f"audio/{st}/{di}", repo_type="dataset",
                                                            revision=REV) if f.path.endswith(".parquet"))
    print(f"{len(shards)} shards in {off['district'].nunique()} districts", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, s) for s in shards]
        for k, f in enumerate(as_completed(futs)):
            f.result()
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(shards)} shards", flush=True)

    src = pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("*.parquet"))], ignore_index=True)
    src["district"] = src["source_shard"].str.split("/").str[2]
    src["dur_c"] = np.minimum(src["duration"], 20.0)
    rows = []
    for di, o in off.groupby("district"):
        s = src[src["district"] == di].sort_values("dur_c")
        dv = s["dur_c"].to_numpy()
        for r in o.itertuples(index=False):
            lo, hi = np.searchsorted(dv, r.seconds - TOL), np.searchsorted(dv, r.seconds + TOL, side="right")
            n = hi - lo
            rec = {"metadata_idx": r.metadata_idx, "candidates": int(n)}
            if n == 1:
                c = s.iloc[lo]
                rec.update({"source_path": c["source_path"], "source_shard": c["source_shard"],
                            "source_pincode": c["pincode"], "source_gender": c["gender"], "picture": c["referenceImage"]})
            rows.append(rec)
    m = off.drop(columns=["source_path"]).merge(pd.DataFrame(rows), on="metadata_idx")
    m["speaker"] = m["source_path"].map(lambda p: speaker_from_path(p) if isinstance(p, str) else None)
    m.to_parquet(D / "sources.parquet", index=False)

    exact = off[off["source_path"].notna()][["metadata_idx", "source_path"]].merge(
        m[["metadata_idx", "source_path"]], on="metadata_idx", suffixes=("_audio", "_duration"))
    print(f"source clips scanned {len(src)}; official rows {len(m)}: unique match {(m['candidates'] == 1).mean():.1%}, "
          f"ambiguous {(m['candidates'] > 1).mean():.1%}, none {(m['candidates'] == 0).mean():.1%}")
    print(f"validation on {len(exact)} audio-matched rows: same source file "
          f"{(exact['source_path_audio'] == exact['source_path_duration']).mean():.1%}")


if __name__ == "__main__":
    main()
