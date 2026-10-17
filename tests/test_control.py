import numpy as np
import pandas as pd

from sal.data.control import control_targets


def test_control_constant_within_speaker_and_keeps_label_distribution():
    rng = np.random.default_rng(1)
    pins = {p: (20 + p, 75 + p) for p in range(5)}
    rows = []
    for s in range(40):
        p = s % 5
        for _ in range(rng.integers(1, 6)):
            rows.append({"speaker": f"s{s}", "pincode": p, "latitude": pins[p][0], "longitude": pins[p][1]})
    df = pd.DataFrame(rows)
    t = control_targets(df, seed=0)
    assert t.shape == (len(df), 2)
    per_spk = pd.DataFrame(t, columns=["a", "b"]).groupby(df["speaker"]).nunique()
    assert (per_spk <= 1).all().all()
    spk_loc = pd.DataFrame(t).groupby(df["speaker"]).first().round(6)
    true_loc = np.radians(df.groupby("speaker")[["latitude", "longitude"]].first()).round(6)
    assert sorted(map(tuple, spk_loc.to_numpy())) == sorted(map(tuple, true_loc.to_numpy()))
    assert not np.allclose(spk_loc.to_numpy(), true_loc.to_numpy())
