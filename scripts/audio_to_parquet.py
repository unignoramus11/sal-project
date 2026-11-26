"""Pack a folder of audio files into the parquet format scripts/transcribe.py reads.

Each file becomes one row: `audio_path` (relative to the folder) and `audio` (16 kHz mono 16-bit WAV bytes, cut to
20 s as in Vaani-Hi). WAV, FLAC and OGG files are resampled and down-mixed if needed.

Run:  uv run python scripts/audio_to_parquet.py my_audio/ my_audio.parquet
      envs/prism/bin/python scripts/transcribe.py --model zipa --data my_audio.parquet --id-col audio_path \
          --out outputs/transcripts/zipa_my_audio.parquet
"""
import argparse
import io
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--max-seconds", type=float, default=20.0)
args = ap.parse_args()

rows = []
for f in sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".ogg"}):
    wav, sr = sf.read(f, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)                                    # down-mix to mono
    if sr != 16000:
        g = gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    wav = wav[:int(args.max_seconds * 16000)]
    buf = io.BytesIO()
    sf.write(buf, wav, 16000, format="WAV", subtype="PCM_16")
    rows.append({"audio_path": str(f.relative_to(args.folder)), "audio": buf.getvalue()})
if not rows:
    raise SystemExit(f"no audio files under {args.folder}")
pd.DataFrame(rows).to_parquet(args.out, index=False)
print(f"wrote {args.out} ({len(rows)} files)")
