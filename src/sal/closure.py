"""Acoustic measurement of intervocalic voiceless stops: voiceless interval, closure, VOT, following vowel.

Within a search window around the aligned stop:
  voiceless stretch  the run of unvoiced frames (Praat pitch tracker) that contains the stop's aligned centre,
                     or the nearest one within 60 ms of it
  closure start      first frame of that stretch (voicing offset of the preceding vowel)
  voice onset        first voiced frame after it (onset of the following vowel)
  burst              the largest frame-to-frame rise of 2-8 kHz band energy inside the stretch, at least 10 ms
                     after the closure start
  vowel end          last voiced frame of the following voiced stretch
voiceless interval = voice onset - closure start (= closure + VOT; needs voicing only, so it can be measured on
nearly every token); closure = burst - closure start; VOT = voice onset - burst; vowel = vowel end - voice onset.

Rejected only when the stretch cannot be measured: no voiceless stretch near the stop, a stretch that reaches
the edge of the window (its start or end is not visible), or a voiceless interval outside 20-400 ms. Burst rise
and the intensity drop from the vowel into the closure are kept as quality covariates (clear_burst: rise >= 6 dB
and drop >= 10 dB) instead of filters, because hard thresholds rejected most tokens of these phone recordings
and did so unequally across regions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth

HOP = 0.005


@dataclass
class Measure:
    voi_ms: float = np.nan
    closure_ms: float = np.nan
    vot_ms: float = np.nan
    vowel_ms: float = np.nan
    closure_start: float = np.nan
    burst: float = np.nan
    voice_onset: float = np.nan
    intensity_drop_db: float = np.nan
    burst_rise_db: float = np.nan
    clear_burst: bool = False
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


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of runs of True, end exclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def measure(x: np.ndarray, sr: int, stop_centre: float, t0: float, t1: float) -> Measure:
    m = Measure()
    times, voiced, total, hf, _ = _frames(x, sr, t0, t1)
    if len(times) < 10:
        m.reject = "window too short"
        return m
    c = int(np.argmin(np.abs(times - stop_centre)))
    runs = _runs(~voiced)
    near = int(0.06 / HOP)
    dist = [0 if s <= c < e else min(abs(c - s), abs(c - (e - 1))) for s, e in runs]
    if not runs or min(dist) > near:
        m.reject = "no voiceless stretch near the stop"
        return m
    cs, vo = runs[int(np.argmin(dist))]
    if cs == 0 or vo == len(times):
        m.reject = "voiceless stretch reaches the window edge"
        return m
    m.voi_ms = 1000 * (vo - cs) * HOP
    vend = vo
    while vend + 1 < len(voiced) and voiced[vend + 1]:
        vend += 1
    m.closure_start, m.voice_onset = float(times[cs]), float(times[vo])
    m.vowel_ms = 1000 * (times[vend] + HOP - m.voice_onset)
    if not 20 <= m.voi_ms <= 400:
        m.reject = "voiceless interval outside 20-400 ms"
        return m
    if vo - cs >= 3:                                   # burst at least 2 frames (10 ms) after voicing offset
        rises = np.diff(hf[cs: vo + 1])[1:]
        b = cs + 2 + int(np.argmax(rises))
        m.burst_rise_db = float(rises.max())
        m.burst = float(times[b])
        m.closure_ms = 1000 * (b - cs) * HOP
        m.vot_ms = 1000 * (vo - b) * HOP
        vowel_level = np.median(total[max(0, cs - 8): cs])
        m.intensity_drop_db = float(vowel_level - np.median(total[cs: b]))
        m.clear_burst = bool(m.burst_rise_db >= 6 and m.intensity_drop_db >= 10)
    return m
