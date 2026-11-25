"""Train the prosody probe (RQ3) on contours from scripts/prosody_features.py.

Settings as the transcript probe (Adam 1e-3, batch 64, 10-200 epochs, patience 100 checks, 4 per epoch), so
TP and prosody scores are directly comparable and can be fused at the prediction level.
Run: python scripts/run_prosody.py --protocol speaker_split --seeds 42 43 44 45 46 --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sal.data.prompts import restrict
from sal.paths import DATA, OUTPUTS, RESULTS
from sal.probes.models import ProsodyGRU
from sal.probes.rp_data import FeatureStore, FrameBatcher
from sal.probes.train import TrainConfig, checkpoint_for, save_run, train_probe

ap = argparse.ArgumentParser()
ap.add_argument("--features", type=Path, default=OUTPUTS / "features" / "prosody_rebuild")
ap.add_argument("--protocol", choices=["prism", "speaker_split"], default="speaker_split")
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--prompts", choices=["all", "generic"], default="all",
                help="generic: train and test only on utterances about shared pictures (no topic cue)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--patience", type=int, default=100)
args = ap.parse_args()

meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet", columns=["metadata_idx", "latitude", "longitude"])
meta = meta.merge(pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet"), on="metadata_idx")
meta = meta.sort_values("metadata_idx").reset_index(drop=True)
store = FeatureStore(args.features, [0], in_memory=True)
targets = np.radians(meta[["latitude", "longitude"]].to_numpy(np.float64))
splits = {s: np.flatnonzero(meta[args.protocol].to_numpy() == s) for s in ("train", "val", "test")}
splits = restrict(splits, meta["metadata_idx"].to_numpy(), args.prompts)
for seed in args.seeds:
    cfg = TrainConfig(lr=1e-3, batch_size=64, min_epochs=10, max_epochs=200, patience=args.patience, val_per_epoch=4,
                      seed=seed, device=args.device, eval_batch_size=128, extra={"features": str(args.features)})
    name = f"prosody_gru_{args.protocol}{'_generic' if args.prompts == 'generic' else ''}_s{seed}"
    res = train_probe(lambda: ProsodyGRU(), FrameBatcher(store, meta["metadata_idx"].to_numpy(), targets, seed=seed),
                      targets, splits, cfg, checkpoint=checkpoint_for(name))
    save_run(res, RESULTS / "runs" / name, meta=meta[["metadata_idx", "speaker", "pincode", "state", "district"]])
    print(name, json.dumps(res["metrics"]["test"]), flush=True)
