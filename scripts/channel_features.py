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
import io
import json
import time
import wave
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sal.paths import DATA, OUTPUTS

SRC = DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet"
OUT = OUTPUTS / "features" / "channel_rebuild"
SR, WIN, HOP, NFFT, NMEL, QUIET = 16000, 400, 160, 512, 40, 0.10


def mel_filters():
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(SR / 2), NMEL + 2))
    bins = np.floor((NFFT + 1) * pts / SR).astype(int)
    fb = np.zeros((NMEL, NFFT // 2 + 1))
    for m in range(1, NMEL + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[m - 1, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    return fb


FB = mel_filters()


def read_wav(b: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(b)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, (w.getframerate(), w.getnchannels())
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dt = {2: np.int16, 4: np.int32}[width]
    return np.frombuffer(raw, dt).astype(np.float64) / np.iinfo(dt).max


def features(x: np.ndarray) -> np.ndarray:
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = 1 + (len(x) - WIN) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP][:n] * np.hanning(WIN)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    k = max(5, int(QUIET * n))
    quiet = np.argsort(db)[:k]
    spec = np.abs(np.fft.rfft(frames[quiet], NFFT)) ** 2
    mel = 10 * np.log10(spec @ FB.T + 1e-12).mean(axis=0)
    floor, speech = db[quiet].mean(), np.percentile(db, 95)
    clip = np.mean(np.abs(x) >= 0.999)
    return np.concatenate([mel, [floor, speech, speech - floor, clip, x.mean()]])


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
    names = [f"noise_mel_{b}" for b in range(NMEL)] + ["noise_floor_db", "speech_db", "snr_db", "clip_share", "dc"]
    (OUT / "meta.json").write_text(json.dumps({"features": names, "n": len(x), "dim": x.shape[1],
                                               "seconds": time.time() - t0}, indent=1))
    print(f"{len(x)} utterances, {x.shape[1]} features, {time.time() - t0:.0f} s")
    print(pd.DataFrame(x[:, -5:], columns=names[-5:]).describe().round(3).to_string())


if __name__ == "__main__":
    main()
