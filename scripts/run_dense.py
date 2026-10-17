"""Geolocation probe on one fixed-size vector per utterance (e.g. ECAPA speaker embeddings).

MLP: LayerNorm -> Linear(D, 256) -> GELU -> Dropout(0.1) -> Linear(256, 3), angular loss, Adam 1e-3,
batch 256, 10-200 epochs, early stopping patience 20 validation checks (4 per epoch).

Run: uv run python scripts/run_dense.py --features outputs/features/ecapa_rebuild --protocol prism --seeds 42 43 44 45 46
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sal.data.prompts import restrict
from sal.paths import DATA, RESULTS
from sal.probes.models import VectorMLP
from sal.probes.tp_data import DenseBatcher
from sal.probes.train import TrainConfig, save_run, train_probe

ap = argparse.ArgumentParser()
ap.add_argument("--features", type=Path, required=True)
ap.add_argument("--file", default="emb.npy")
ap.add_argument("--protocol", choices=["prism", "speaker_split"], default="prism")
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--prompts", choices=["all", "generic"], default="all",
                help="generic: train and test only on utterances about shared pictures (no topic cue)")
ap.add_argument("--device", default="cpu")
ap.add_argument("--zscore", action="store_true", help="standardise each feature with train-split mean and sd")
args = ap.parse_args()

meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet", columns=["metadata_idx", "latitude", "longitude"])
meta = meta.merge(pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet"), on="metadata_idx")
index = pd.read_parquet(args.features / "index.parquet")
x_all = np.load(args.features / args.file).astype(np.float32)
x_all = x_all.reshape(len(x_all), -1)                  # e.g. pooled.npy stores (N, 1, D)
row = {mi: k for k, mi in enumerate(index["metadata_idx"])}
meta = meta.sort_values("metadata_idx").reset_index(drop=True)
x = x_all[[row[mi] for mi in meta["metadata_idx"]]]
targets = np.radians(meta[["latitude", "longitude"]].to_numpy(np.float64))
splits = {s: np.flatnonzero(meta[args.protocol].to_numpy() == s) for s in ("train", "val", "test")}
splits = restrict(splits, meta["metadata_idx"].to_numpy(), args.prompts)
if args.zscore:                                         # heterogeneous summary statistics: train mean and sd
    mu, sd = x[splits["train"]].mean(0), x[splits["train"]].std(0) + 1e-6
    x = (x - mu) / sd
d = x.shape[1]

for seed in args.seeds:
    cfg = TrainConfig(lr=1e-3, batch_size=256, min_epochs=10, max_epochs=200, patience=20, val_per_epoch=4,
                      seed=seed, device=args.device, extra={"features": str(args.features), "dim": d})
    make = lambda: VectorMLP(d)  # noqa: E731
    res = train_probe(make, DenseBatcher(x, targets), targets, splits, cfg, log=lambda s: None)
    name = f"dense_{args.features.name}_{args.protocol}{'_generic' if args.prompts == 'generic' else ''}_s{seed}"
    save_run(res, RESULTS / "runs" / name, meta=meta[["metadata_idx", "speaker", "pincode", "state", "district"]])
    print(name, json.dumps({k: round(v, 2) for k, v in res["metrics"]["test"].items()}), flush=True)
