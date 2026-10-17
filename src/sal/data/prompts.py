"""Picture prompts. Vaani speakers describe pictures; district-specific pictures carry location through topic.

`restrict` keeps only utterances about GENERIC (shared) pictures in every split, removing topic as a cue.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sal.paths import DATA

PROMPTS = DATA / "vaani_hi_rebuild" / "prompts.parquet"


def prompt_type(metadata_idx: pd.Series | np.ndarray, path: Path = PROMPTS) -> np.ndarray:
    p = pd.read_parquet(path).set_index("metadata_idx")["prompt_type"]
    return p.loc[np.asarray(metadata_idx)].to_numpy()


def restrict(splits: dict[str, np.ndarray], metadata_idx: np.ndarray, kind: str) -> dict[str, np.ndarray]:
    """kind 'all' returns splits unchanged; 'generic' keeps only GENERIC-prompt examples in every split."""
    if kind == "all":
        return splits
    if kind != "generic":
        raise ValueError(kind)
    keep = prompt_type(metadata_idx) == "GENERIC"
    return {s: idx[keep[idx]] for s, idx in splits.items()}
