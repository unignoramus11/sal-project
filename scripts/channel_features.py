"""Recording-channel features: what the recording setup sounds like, with as little of the speech as possible.

Speakers from one place may share recording conditions (phones, rooms, background noise, gain), so any probe that
hears audio could locate new speakers by the channel rather than by dialect. These features describe the channel
only, and a probe trained on them measures how much location the channel alone carries:
  noise spectrum   mean log mel energy (40 bands, 0-8 kHz) over the 10% quietest 25 ms frames of the utterance,
                   which hold background noise and the codec's band limit rather than speech
  levels           noise floor (mean dB of those frames), speech level (95th percentile frame dB), and their
                   difference (a signal-to-noise proxy)
  clipping, DC     share of samples within 0.1% of full scale, and the mean sample value
Quiet frames can still hold breaths or soft speech, so the features are channel-dominated, not channel-pure.

Output: outputs/features/channel_rebuild/{emb.npy (N, 45) float32, index.parquet, meta.json}
Run:    uv run python scripts/channel_features.py
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sal.channel import NAMES, features, read_wav
from sal.paths import DATA, OUTPUTS

SRC = DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet"
OUT = OUTPUTS / "features" / "channel_rebuild"


def row_group(i: int):
    t = pq.ParquetFile(SRC).read_row_group(i, columns=["metadata_idx", "audio"]).to_pydict()
    return t["metadata_idx"], np.stack([features(read_wav(a)) for a in t["audio"]])


def main():
    t0 = time.time()
    n = pq.ParquetFile(SRC).num_row_groups
    with ProcessPoolExecutor(max_workers=8) as ex:
        parts = list(ex.map(row_group, range(n)))
    idx = np.concatenate([p[0] for p in parts])
    x = np.concatenate([p[1] for p in parts]).astype(np.float32)
    assert np.isfinite(x).all()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "emb.npy", x)
    pd.DataFrame({"metadata_idx": idx}).to_parquet(OUT / "index.parquet", index=False)
    names = NAMES
    (OUT / "meta.json").write_text(json.dumps({"features": names, "n": len(x), "dim": x.shape[1],
                                               "seconds": time.time() - t0}, indent=1))
    print(f"{len(x)} utterances, {x.shape[1]} features, {time.time() - t0:.0f} s")
    print(pd.DataFrame(x[:, -5:], columns=names[-5:]).describe().round(3).to_string())


if __name__ == "__main__":
    main()
