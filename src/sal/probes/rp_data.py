"""Batches of cached frame features for representation probes.

Features come from scripts/extract_features.py: for each layer l, `frames_L{l}.npy` (float16, all utterances
concatenated) and `index.parquet` (metadata_idx, n_frames, offset).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sal.probes.models import shuffle_chunks


class FeatureStore:
    def __init__(self, feat_dir: Path, layers: list[int], in_memory: bool = False):
        self.dir = Path(feat_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        n_layers = len(self.meta["layers"])
        self.layers = sorted({l % n_layers for l in layers})
        missing = [l for l in self.layers if not (self.dir / f"frames_L{l}.npy").exists()]
        if missing:
            raise FileNotFoundError(f"no cached frames for layers {missing} in {self.dir}")
        self.index = pd.read_parquet(self.dir / "index.parquet").set_index("metadata_idx")
        mode = None if in_memory else "r"
        self.frames = {l: np.load(self.dir / f"frames_L{l}.npy", mmap_mode=mode) for l in self.layers}
        self.dim = max(self.frames[l].shape[1] for l in self.layers)
        self.rate = self.meta["frame_rate"]

    def get(self, metadata_idx: int) -> np.ndarray:
        """(L, T, Dmax) float16 for one utterance; narrower layers are zero-padded (ZIPA stacks)."""
        off, n = self.index.at[metadata_idx, "offset"], self.index.at[metadata_idx, "n_frames"]
        out = np.zeros((len(self.layers), n, self.dim), dtype=np.float16)
        for k, l in enumerate(self.layers):
            f = self.frames[l]
            out[k, :, : f.shape[1]] = f[off: off + n]
        return out


class FrameBatcher:
    """Pads to the longest utterance in the batch. Returns (B, T, D) for one layer, (B, L, T, D) for several."""

    def __init__(self, store: FeatureStore, metadata_idx: np.ndarray, targets: np.ndarray,
                 chunk_shuffle_ms: int | None = None, seed: int = 0):
        self.store, self.mi = store, metadata_idx
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.chunk = None if not chunk_shuffle_ms else max(1, round(chunk_shuffle_ms * store.rate / 1000))
        self.gen = torch.Generator().manual_seed(seed)
        self.eval_gen_seed = seed + 1

    def __call__(self, idx: np.ndarray, train: bool):
        arrs = [self.store.get(int(self.mi[i])) for i in idx]
        lengths = torch.tensor([a.shape[1] for a in arrs])
        L, D = arrs[0].shape[0], arrs[0].shape[2]
        x = torch.zeros(len(arrs), L, int(lengths.max()), D, dtype=torch.float32)
        for b, a in enumerate(arrs):
            x[b, :, : a.shape[1]] = torch.from_numpy(a.astype(np.float32))
        if self.chunk:
            if train:
                x = shuffle_chunks(x, lengths, self.chunk, self.gen)
            else:  # a fixed permutation per utterance, independent of batch composition
                for b, i in enumerate(idx):
                    g = torch.Generator().manual_seed(self.eval_gen_seed * 1_000_003 + int(i))
                    x[b: b + 1] = shuffle_chunks(x[b: b + 1], lengths[b: b + 1], self.chunk, g)
        if L == 1:
            x = x[:, 0]
        return x, lengths, self.targets[idx]
