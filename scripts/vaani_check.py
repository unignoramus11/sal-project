"""Read-only metadata check of Vaani-Hi (changelinglab/vaani-hi-geo).

Reads parquet footers, the small label columns, and one WAV header. It does not download the
full audio, but the first row group's audio column is read once (up to a few hundred MB).

Needs: Vaani-Hi terms accepted on Hugging Face and `hf auth login` done.
Run:   uv run --with huggingface_hub --with pyarrow python scripts/vaani_check.py
"""
import collections
import struct

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/changelinglab/vaani-hi-geo"

fs = HfFileSystem()
files = [f for f in fs.find(REPO) if f.endswith(".parquet")]
print("parquet files:", len(files))

rows = 0
audio_bytes = 0
first_header = None
names = []
states, langs, split_c = collections.Counter(), collections.Counter(), collections.Counter()
hr_dist, hr_lang = collections.Counter(), collections.Counter()

for f in files:
    with fs.open(f, "rb") as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        rows += md.num_rows
        names = pf.schema_arrow.names
        audio_col = names.index("audio")
        for rg in range(md.num_row_groups):
            audio_bytes += md.row_group(rg).column(audio_col).total_uncompressed_size
        cols = [c for c in ["state", "district", "language", "split"] if c in names]
        t = pf.read(columns=cols).to_pydict()
        for i in range(len(t[cols[0]])):
            st = t["state"][i]
            states[st] += 1
            langs[t["language"][i]] += 1
            split_c[t["split"][i]] += 1
            if st == "Haryana":
                hr_dist[t["district"][i]] += 1
                hr_lang[t["language"][i]] += 1
        if first_header is None:
            b = pf.read_row_group(0, columns=["audio"]).column(0)[0].as_py()
            first_header = b[:64]

print("schema:", names)
print("rows:", rows, "splits:", dict(split_c))
fmt = struct.unpack("<H", first_header[20:22])[0]
ch, sr, _, _, bps = struct.unpack("<HIIHH", first_header[22:36])
print(f"wav format tag {fmt} (1=PCM int, 3=float), channels {ch}, sr {sr}, bits {bps}")
hours = audio_bytes / (sr * ch * bps / 8) / 3600
print(f"audio column ~{audio_bytes/1e9:.2f} GB -> ~{hours:.1f} h, mean {hours*3600/rows:.2f} s/utt")
print("states:", dict(states))
print("languages:", dict(langs.most_common(20)))
print("Haryana districts:", dict(hr_dist))
print("Haryana languages:", dict(hr_lang))
