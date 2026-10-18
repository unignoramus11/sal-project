import numpy as np
import pytest

from sal.closure import measure

SR = 16000


def vowel(dur, f0=140.0, rng=None):
    t = np.arange(int(dur * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 25))   # harmonic-rich voiced sound
    return 0.3 * x / np.abs(x).max()


def synthetic_token(closure, vot, v2, seed=0):
    rng = np.random.default_rng(seed)
    parts = [vowel(0.12), 0.002 * rng.normal(size=int(closure * SR)),
             0.4 * rng.normal(size=int(0.008 * SR)),                    # release burst
             0.05 * rng.normal(size=int(vot * SR)),                     # aspiration
             vowel(v2), 0.002 * rng.normal(size=int(0.1 * SR))]
    return np.concatenate(parts).astype(np.float32)


@pytest.mark.parametrize("closure,vot", [(0.08, 0.02), (0.16, 0.06), (0.11, 0.04)])
def test_measures_known_durations(closure, vot):
    x = synthetic_token(closure, vot, 0.15)
    centre = 0.12 + closure / 2
    m = measure(x, SR, centre, 0.02, len(x) / SR - 0.02)
    assert m.reject == ""
    assert m.closure_ms == pytest.approx(1000 * closure, abs=15)
    assert m.vot_ms == pytest.approx(1000 * (vot + 0.008), abs=20)
    assert m.vowel_ms == pytest.approx(150, abs=25)
    assert m.intensity_drop_db > 20


def test_longer_closure_measures_longer():
    short = measure(synthetic_token(0.07, 0.02, 0.15), SR, 0.155, 0.02, 0.5)
    long = measure(synthetic_token(0.15, 0.02, 0.15, seed=1), SR, 0.195, 0.02, 0.5)
    assert long.closure_ms - short.closure_ms == pytest.approx(80, abs=15)


def test_rejects_when_no_closure():
    x = np.concatenate([vowel(0.3)]).astype(np.float32)   # all voiced: no stop
    m = measure(x, SR, 0.15, 0.02, 0.28)
    assert m.reject != ""


@pytest.mark.parametrize("closure,vot", [(0.08, 0.02), (0.15, 0.05)])
def test_voiceless_interval(closure, vot):
    x = synthetic_token(closure, vot, 0.15)
    m = measure(x, SR, 0.12 + closure / 2, 0.02, len(x) / SR - 0.02)
    assert m.voi_ms == pytest.approx(1000 * (closure + 0.008 + vot), abs=15)
    assert m.clear_burst


def test_aligned_centre_slightly_off_still_finds_stretch():
    x = synthetic_token(0.10, 0.02, 0.15)
    m = measure(x, SR, 0.12 - 0.04, 0.02, len(x) / SR - 0.02)     # centre 40 ms early, inside the vowel
    assert m.reject == "" and m.voi_ms == pytest.approx(128, abs=15)
