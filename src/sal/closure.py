"""Acoustic measurement of intervocalic voiceless stops: closure, VOT, following vowel.

Within a search window around the aligned stop:
  closure start   last voiced frame of the preceding vowel before the stop's aligned centre
  burst           the largest frame-to-frame rise of 2-8 kHz band energy after the closure start
  voice onset     first voiced frame after the burst
  vowel end       last voiced frame of the following voiced stretch
closure = burst - closure start; VOT = voice onset - burst; vowel = vowel end - voice onset.

Voicing comes from Praat's pitch tracker (Parselmouth), energy from a 5 ms-hop STFT. A token is rejected,
with a reason, if any landmark is missing or implausible: closure outside 20-400 ms, an intensity drop of
less than 10 dB from the preceding vowel into the closure, or a burst rise under 6 dB.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth

HOP = 0.005


@dataclass
class Measure:
    closure_ms: float = np.nan
    vot_ms: float = np.nan
    vowel_ms: float = np.nan
    closure_start: float = np.nan
    burst: float = np.nan
    voice_onset: float = np.nan
    intensity_drop_db: float = np.nan
    burst_rise_db: float = np.nan
    reject: str = ""


def _frames(x: np.ndarray, sr: int, t0: float, t1: float):
    """Times, voicing (bool), overall and 2-8 kHz energy (dB) on a 5 ms grid over [t0, t1]."""
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    part = snd.extract_part(from_time=max(0.0, t0 - 0.05), to_time=min(snd.duration, t1 + 0.05),
                            preserve_times=True)
    pitch = part.to_pitch(time_step=HOP, pitch_floor=75, pitch_ceiling=500)
    times = np.arange(t0, t1, HOP)
    f0 = np.array([pitch.get_value_at_time(t) for t in times])
    voiced = np.isfinite(f0) & (f0 > 0)
    win = int(0.010 * sr)
    hop = int(HOP * sr)
    spec_freqs = np.fft.rfftfreq(win, 1 / sr)
    band = (spec_freqs >= 2000) & (spec_freqs <= 8000)
    total, hf = [], []
    w = np.hanning(win)
    for t in times:
        c = int(t * sr)
        seg = x[max(0, c - win // 2): c - win // 2 + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        p = np.abs(np.fft.rfft(seg * w)) ** 2
        total.append(10 * np.log10(p.sum() + 1e-12))
        hf.append(10 * np.log10(p[band].sum() + 1e-12))
    return times, voiced, np.array(total), np.array(hf), hop


def measure(x: np.ndarray, sr: int, stop_centre: float, t0: float, t1: float) -> Measure:
    m = Measure()
    times, voiced, total, hf, _ = _frames(x, sr, t0, t1)
    if len(times) < 10:
        m.reject = "window too short"
        return m
    c = int(np.argmin(np.abs(times - stop_centre)))
    # closure start: walk back from the centre to the last voiced frame
    before = np.flatnonzero(voiced[:c])
    if len(before) == 0:
        m.reject = "no voiced frame before the stop"
        return m
    cs = before[-1] + 1
    # burst: largest rise in high-band energy after the closure start (look up to 400 ms ahead)
    lim = min(len(times) - 1, cs + int(0.4 / HOP))
    if lim - cs < 4:
        m.reject = "no room for a burst"
        return m
    rises = np.diff(hf[cs: lim + 1])
    b = cs + 1 + int(np.argmax(rises))
    m.burst_rise_db = float(rises.max())
    after = np.flatnonzero(voiced[b:])
    if len(after) == 0:
        m.reject = "no voicing after the burst"
        return m
    vo = b + after[0]
    vend = vo
    while vend + 1 < len(voiced) and voiced[vend + 1]:
        vend += 1
    m.closure_start, m.burst, m.voice_onset = float(times[cs]), float(times[b]), float(times[vo])
    m.closure_ms = 1000 * (m.burst - m.closure_start)
    m.vot_ms = 1000 * (m.voice_onset - m.burst)
    m.vowel_ms = 1000 * (times[vend] + HOP - m.voice_onset)
    vowel_level = np.median(total[max(0, cs - 8): cs]) if cs > 0 else np.nan
    closure_level = np.median(total[cs: b]) if b > cs else np.nan
    m.intensity_drop_db = float(vowel_level - closure_level)
    if not 20 <= m.closure_ms <= 400:
        m.reject = "closure outside 20-400 ms"
    elif not m.intensity_drop_db >= 10:
        m.reject = "intensity drop under 10 dB"
    elif not m.burst_rise_db >= 6:
        m.reject = "burst rise under 6 dB"
    return m
