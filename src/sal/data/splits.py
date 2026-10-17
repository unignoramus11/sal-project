"""Speaker-disjoint train/val/test splits, stratified by location.

PRiSM splits utterances at random within each pincode, so a speaker's utterances fall on both sides.
Here speakers are assigned to splits. Within each pincode, speakers are taken from the largest to
the smallest (ties broken by a seeded shuffle) and each goes to the split that minimises the squared
relative deviation from the target utterance shares after adding them. Heavy speakers therefore land
in train rather than overshooting a small split. A speaker who recorded in more than one pincode is assigned in the
pincode where they have most utterances, and that split applies to all their utterances.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.75, "val": 0.10, "test": 0.15}


def speaker_disjoint_splits(
    df: pd.DataFrame,
    speaker_col: str = "speaker",
    group_col: str = "pincode",
    fractions: dict[str, float] = DEFAULT_FRACTIONS,
    seed: int = 42,
) -> pd.Series:
    """Return a split label per row of `df` (index aligned)."""
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)

    counts = df.groupby([speaker_col, group_col]).size().rename("n").reset_index()
    # each speaker belongs to the group where they have most utterances (ties: smallest group id)
    home = (counts.sort_values(["n", group_col], ascending=[False, True])
                  .drop_duplicates(speaker_col).set_index(speaker_col)[group_col])
    per_speaker = df.groupby(speaker_col).size()

    assignment: dict[str, str] = {}
    for g in sorted(home.unique()):
        spk = home.index[home == g].to_numpy()
        spk = spk[rng.permutation(len(spk))]               # seeded tie-breaking
        n = per_speaker.loc[spk].to_numpy()
        order = np.argsort(-n, kind="stable")
        total = n.sum()
        filled = dict.fromkeys(SPLITS, 0)
        target = {s: fractions[s] * total for s in SPLITS}
        for k, i in enumerate(order):
            remaining = len(order) - k
            empty = [s for s in SPLITS if filled[s] == 0 and fractions[s] > 0]
            if empty and remaining <= len(empty):
                # make sure every split gets at least one speaker when possible
                choice = max(empty, key=lambda s: target[s] - filled[s])
            else:
                def cost(s, n_i=n[i]):
                    return sum((filled[t] + (n_i if t == s else 0) - target[t]) ** 2 / max(target[t], 1e-9)
                               for t in SPLITS)
                choice = min(SPLITS, key=cost)
            assignment[spk[i]] = choice
            filled[choice] += n[i]
    return df[speaker_col].map(assignment)


def leakage(df: pd.DataFrame, split_col: str, speaker_col: str = "speaker") -> dict[str, float]:
    """Share of val/test utterances whose speaker also appears in train."""
    train_spk = set(df.loc[df[split_col] == "train", speaker_col])
    out = {}
    for s in ("val", "test"):
        part = df[df[split_col] == s]
        out[s] = float(part[speaker_col].isin(train_spk).mean()) if len(part) else float("nan")
    return out


def split_report(df: pd.DataFrame, split_col: str, speaker_col: str = "speaker",
                 group_col: str = "pincode") -> pd.DataFrame:
    """Per-group utterance shares and speaker counts per split."""
    t = df.groupby([group_col, split_col]).agg(utts=(speaker_col, "size"), speakers=(speaker_col, "nunique"))
    t = t.unstack(split_col, fill_value=0)
    tot = t["utts"].sum(axis=1)
    for s in SPLITS:
        t[("share", s)] = t["utts"].get(s, 0) / tot
    return t
