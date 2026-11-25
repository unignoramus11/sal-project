"""Train representation probes on cached frame features.

Defaults are PRiSM's RP settings (docs/prism_reproduction.md): Adam 2e-4, batch 48, 10-100 epochs, early
stopping patience 5 validation checks, 4 checks per epoch. Order-aware probes are sized to within 10% of
RP-Attn's trainable parameters (sal.probes.models.match_size).

Probes:  attn          RP-Attn (PRiSM head)                       order-free
         attn_pe       RP-Attn with sinusoidal positions added     weak order
         gru           projection + biGRU + attention head         order-aware
         transformer   projection + PE + Transformer + attn head   order-aware
         chunkshuffle  RP-GRU on 200 ms chunks permuted in time    local order only

Layers: one index (e.g. -1 for the last layer) or several, combined by a learned softmax-weighted sum.

Run: python scripts/run_rp.py --features outputs/features/w2v2_rebuild --layers -1 --probe attn \
        --protocol speaker_split --seeds 42 43 44 --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sal.data.control import control_targets
from sal.data.prompts import restrict
from sal.paths import DATA, RESULTS
from sal.probes import models as M
from sal.probes.rp_data import FeatureStore, FrameBatcher
from sal.probes.train import TrainConfig, save_run, train_probe

ap = argparse.ArgumentParser()
ap.add_argument("--features", type=Path, required=True)
ap.add_argument("--layers", type=int, nargs="+", default=[-1])
ap.add_argument("--probe", choices=["attn", "attn_pe", "gru", "transformer", "chunkshuffle"], default="attn")
ap.add_argument("--protocol", default="prism",
                help="split column of --splits: prism, speaker_split, or a benchmark fold such as all_f0")
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--dataset", type=Path, default=DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet")
ap.add_argument("--splits", type=Path, default=DATA / "vaani_hi_rebuild" / "splits.parquet")
ap.add_argument("--in-memory", action="store_true", help="load frames into RAM (needs ~15 GB per layer)")
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--max-epochs", type=int, default=100)
ap.add_argument("--batch-size", type=int, default=48)
ap.add_argument("--prompts", choices=["all", "generic"], default="all",
                help="generic: train and test only on utterances about shared pictures (no topic cue)")
ap.add_argument("--control", action="store_true",
                help="control task: each speaker gets another speaker's location (sal.data.control)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out-root", type=Path, default=RESULTS / "runs")
ap.add_argument("--tag", default="")
ap.add_argument("--overwrite", action="store_true", help="rerun seeds that already have results")
args = ap.parse_args()

meta = pd.read_parquet(args.dataset, columns=["metadata_idx", "latitude", "longitude"])
meta = meta.merge(pd.read_parquet(args.splits), on="metadata_idx").sort_values("metadata_idx").reset_index(drop=True)
store = FeatureStore(args.features, args.layers, in_memory=args.in_memory)
missing = set(meta["metadata_idx"]) - set(store.index.index)
assert not missing, f"{len(missing)} utterances have no features"

targets = np.radians(meta[["latitude", "longitude"]].to_numpy(np.float64))
if args.control:
    targets = control_targets(meta)
    args.tag += "_control"
splits = {s: np.flatnonzero(meta[args.protocol].to_numpy() == s) for s in ("train", "val", "test")}
splits = restrict(splits, meta["metadata_idx"].to_numpy(), args.prompts)
n_layers = len(store.layers)
size = M.match_size(args.probe, store.dim, n_layers)
feat_tag = args.features.name
layer_tag = "L" + "-".join(map(str, store.layers))
info = {"features": str(args.features), "layers": store.layers, "layer_names": [store.meta["layers"][i] for i in store.layers],
        "probe": args.probe, "protocol": args.protocol, "dim": store.dim, "size": size}

for seed in args.seeds:
    cfg = TrainConfig(lr=2e-4, batch_size=args.batch_size, min_epochs=10, max_epochs=args.max_epochs,
                      patience=args.patience, val_per_epoch=4, seed=seed, device=args.device, eval_batch_size=64,
                      extra=info)
    batcher = FrameBatcher(store, meta["metadata_idx"].to_numpy(), targets,
                           chunk_shuffle_ms=200 if args.probe == "chunkshuffle" else None, seed=seed)
    make = lambda: M.build_rp(args.probe, store.dim, n_layers, size)  # noqa: E731
    name = f"rp_{feat_tag}_{layer_tag}_{args.probe}_{args.protocol}{args.tag}{'_generic' if args.prompts == 'generic' else ''}_s{seed}"
    if (args.out_root / name / "metrics.json").exists() and not args.overwrite:
        print(name, "already done, skipped", flush=True)
        continue
    res = train_probe(make, batcher, targets, splits, cfg)
    save_run(res, args.out_root / name, meta=meta[["metadata_idx", "speaker", "pincode", "state", "district"]])
    print(name, json.dumps(res["metrics"]["test"]), flush=True)
