"""Frame-level prosodic contours for every Vaani-Hi utterance (RQ3), with Parselmouth (Praat).

Per 10 ms frame: [F0 in semitones relative to the utterance median of voiced frames (0 if unvoiced),
voicing flag, intensity in dB z-scored within the utterance, F0 change from the previous voiced frame].
Normalising within the utterance removes speaker pitch range and recording level, so the probe sees the
shape of the melody, not the voice.

Output: outputs/features/prosody_rebuild/{frames_L0.npy (total_frames, 4) float16, index.parquet, meta.json,
pooled.npy (N, 1, 12) summary statistics}
Run:    uv run python scripts/prosody_features.py --workers 6
"""
import argparse
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import parselmouth
import pyarrow.parquet as pq
import soundfile as sf

from sal.paths import DATA, OUTPUTS

STEP = 0.01


def contour(wav_bytes: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(wav_bytes), dtype="float64")
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=STEP, pitch_floor=75, pitch_ceiling=500)
    inten = snd.to_intensity(minimum_pitch=75, time_step=STEP)
    n = max(1, int(snd.duration / STEP))
    t = (np.arange(n) + 0.5) * STEP
    f0 = np.array([pitch.get_value_at_time(tt) for tt in t])
    db = np.array([inten.get_value(tt) for tt in t])
    voiced = np.isfinite(f0) & (f0 > 0)
    st = np.zeros(n)
    if voiced.any():
        st[voiced] = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
    db = np.where(np.isfinite(db), db, np.nanmin(db) if np.isfinite(db).any() else 0.0)
    dbz = (db - db.mean()) / (db.std() + 1e-6)
    d = np.zeros(n)
    last = None
    for i in range(n):
        if voiced[i]:
            if last is not None:
                d[i] = st[i] - st[last]
            last = i
    return np.stack([st, voiced.astype(float), dbz, d], 1).astype(np.float32)


def _onset_gap(v: np.ndarray) -> float:
    """Mean time between onsets of voiced stretches; the voiced duration if there is only one stretch."""
    onsets = np.flatnonzero(np.diff(np.r_[0, v.astype(int)]) == 1)
    return float(np.mean(np.diff(onsets)) * STEP) if len(onsets) > 1 else float(v.sum() * STEP)


def summary(c: np.ndarray) -> np.ndarray:
    v = c[:, 1] > 0
    st = c[v, 0] if v.any() else np.zeros(1)
    return np.array([st.mean(), st.std(), np.percentile(st, 10), np.percentile(st, 90), v.mean(),
                     c[:, 2].std(), np.abs(c[v, 3]).mean() if v.any() else 0.0, len(c) * STEP,
                     _onset_gap(v), st.max() - st.min(), np.median(st), (c[:, 2] > 1).mean()],
                    dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = OUTPUTS / "features" / "prosody_rebuild"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ids, conts = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for batch in pq.ParquetFile(args.data).iter_batches(batch_size=512, columns=["metadata_idx", "audio"]):
            b = batch.to_pydict()
            conts += list(ex.map(contour, b["audio"], chunksize=16))
            ids += b["metadata_idx"]
            print(f"{len(ids)} utterances, {len(ids) / (time.time() - t0):.1f} utt/s", flush=True)
    lengths = np.array([len(c) for c in conts])
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    np.save(out / "frames_L0.npy", np.concatenate(conts).astype(np.float16))
    np.save(out / "pooled.npy", np.stack([summary(c) for c in conts])[:, None, :])
    pd.DataFrame({"metadata_idx": ids, "n_frames": lengths, "offset": offsets}).to_parquet(out / "index.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({
        "model": "prosody", "layers": ["contour"], "widths": [4], "frame_rate": 100, "frame_layers": [0],
        "features": ["f0_semitones_re_median", "voiced", "intensity_z", "delta_f0"], "n_utterances": len(ids),
        "total_frames": int(lengths.sum()), "seconds": time.time() - t0}, indent=2))
    print("done", len(ids))


if __name__ == "__main__":
    main()
