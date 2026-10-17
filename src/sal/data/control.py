"""Control task (Hewitt and Liang, 2019) adapted to speaker leakage.

Every speaker is given the location of a randomly chosen other speaker's pincode (the speakers' home
pincodes are permuted among speakers), and all of that speaker's utterances get that location as target.
The labels then carry no dialect information but stay constant within a speaker. Under PRiSM's random split
a probe can score above the majority baseline on control labels only by recognising test speakers it saw in
training, so its control Recall@1 measures how much speaker identity it can read from its input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def control_targets(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """(N, 2) radians targets; df needs speaker, pincode, latitude, longitude (one row per utterance)."""
    home = df.groupby("speaker")["pincode"].agg(lambda s: s.mode().iloc[0])
    rng = np.random.default_rng(seed)
    assigned = pd.Series(rng.permutation(home.to_numpy()), index=home.index)
    loc = df.groupby("pincode")[["latitude", "longitude"]].first()
    return np.radians(loc.loc[assigned.loc[df["speaker"]].to_numpy()].to_numpy(np.float64))
