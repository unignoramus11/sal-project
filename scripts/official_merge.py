"""One parquet of the official Vaani-Hi for the feature extractor and the probe runners:
data/vaani_hi_official/vaani_hi_official.parquet with metadata_idx, audio (WAV bytes), audio_seconds, latitude,
longitude, pincode, state, district, split; streamed shard by shard.
Needs the release in data/vaani_hi_official/data/ (download: see docs/ada.md, official-data step).
Run: python scripts/official_merge.py
"""
import glob
import io

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from sal.paths import DATA

out = DATA / "vaani_hi_official" / "vaani_hi_official.parquet"
writer, n = None, 0
for f in sorted(glob.glob(str(DATA / "vaani_hi_official" / "data" / "train-*.parquet"))):
    t = pq.read_table(f)
    secs = [sf.info(io.BytesIO(a)).duration for a in t.column("audio").to_pylist()]
    t = t.select(["metadata_idx", "audio", "latitude", "longitude", "pincode", "state", "district", "split"])
    t = t.append_column("audio_seconds", pa.array(secs, pa.float64()))
    writer = writer or pq.ParquetWriter(out, t.schema)
    writer.write_table(t)
    n += t.num_rows
writer.close()
print(f"wrote {out}: {n} rows")
