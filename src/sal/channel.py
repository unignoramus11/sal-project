"""Recording-channel features of an utterance: background noise spectrum, levels, clipping (scripts/channel_features.py).

The 10% quietest 25 ms frames carry background noise and the codec's band limit rather than speech; their mean log
mel spectrum (40 bands, 0-8 kHz) and level, the speech level (95th percentile frame dB), the difference (a
signal-to-noise proxy), the share of clipped samples and the DC offset describe the recording, not the talker.
"""
import io
import wave

import numpy as np

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


NAMES = [f"noise_mel_{b}" for b in range(NMEL)] + ["noise_floor_db", "speech_db", "snr_db", "clip_share", "dc"]
