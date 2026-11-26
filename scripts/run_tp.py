"""Train transcript probes on Vaani-Hi transcripts.

Defaults are PRiSM's (docs/prism_reproduction.md): Adam 1e-3, batch 64, 10-200 epochs, early stopping
patience 100 validation checks, 4 checks per epoch.

Probes:  gru      TP-GRU (PRiSM architecture)
         shuffle  TP-GRU on tokens permuted within each utterance
         unigram  MLP on token relative frequencies (inventory only)
         bigram   MLP on unigram + bigram relative frequencies (local order)

Run:  uv run python scripts/run_tp.py --transcripts outputs/transcripts/w2v2_rebuild.parquet \
          --tokens char --probe gru --protocol prism --seeds 42 43 44 45 46
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
from sal.probes import tp_data as T
from sal.probes.train import TrainConfig, checkpoint_for, save_run, train_probe

ap = argparse.ArgumentParser()
ap.add_argument("--transcripts", type=Path, required=True)
ap.add_argument("--dataset", type=Path, default=DATA / "vaani_hi_rebuild" / "vaani_hi_rebuild.parquet")
ap.add_argument("--splits", type=Path, default=DATA / "vaani_hi_rebuild" / "splits.parquet")
ap.add_argument("--tokens", choices=["char", "phone"], default="char")
ap.add_argument("--probe", choices=["gru", "shuffle", "unigram", "bigram"], default="gru")
ap.add_argument("--protocol", default="prism",
                help="split column of --splits: prism, speaker_split, or a benchmark fold such as all_f0")
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--patience", type=int, default=100)
ap.add_argument("--max-epochs", type=int, default=200)
ap.add_argument("--control", action="store_true",
                help="control task: each speaker gets another speaker's location (sal.data.control)")
ap.add_argument("--prompts", choices=["all", "generic"], default="all",
                help="generic: train and test only on utterances about shared pictures (no topic cue)")
ap.add_argument("--device", default="cpu")
ap.add_argument("--tag", default="")
ap.add_argument("--save-model", type=Path, default=None,
                help="directory for a self-contained checkpoint per seed (weights, vocabulary, bigram list, "
                     "training locations), read by scripts/infer.py")
args = ap.parse_args()

meta = pd.read_parquet(args.dataset, columns=["metadata_idx", "latitude", "longitude"])
meta = meta.merge(pd.read_parquet(args.splits), on="metadata_idx").sort_values("metadata_idx")
tr = pd.read_parquet(args.transcripts, columns=["metadata_idx", "processed_transcript", "tokens"])
df = meta.merge(tr, on="metadata_idx", how="inner").reset_index(drop=True)
assert len(df) == len(meta), f"{len(meta) - len(df)} utterances without a transcript"

targets = np.radians(df[["latitude", "longitude"]].to_numpy(np.float64))
if args.control:
    targets = control_targets(df)
    args.tag += "_control"
splits = {s: np.flatnonzero(df[args.protocol].to_numpy() == s) for s in ("train", "val", "test")}
splits = restrict(splits, df["metadata_idx"].to_numpy(), args.prompts)
unit_seqs = [T.units(t, list(k), args.tokens) for t, k in zip(df["processed_transcript"], df["tokens"], strict=True)]
vocab = T.build_vocab([unit_seqs[i] for i in splits["train"]])
seqs = [T.encode(s, vocab) for s in unit_seqs]
model_tag = Path(args.transcripts).stem
info = {"vocab_size": len(vocab), "tokens": args.tokens, "probe": args.probe, "protocol": args.protocol,
        "transcripts": str(args.transcripts), "mean_len": float(np.mean([len(s) for s in seqs]))}

bigram_list = None
if args.probe in ("unigram", "bigram"):
    bigram_list = T.bigram_vocab(seqs, splits["train"]) if args.probe == "bigram" else None
    x = T.bag_matrix(seqs, len(vocab), bigram_list)
    info.update({"unigrams": len(vocab), "bigrams": len(bigram_list) if bigram_list is not None else 0})

for seed in args.seeds:
    cfg = TrainConfig(lr=1e-3, batch_size=64, min_epochs=10, max_epochs=args.max_epochs, patience=args.patience,
                      val_per_epoch=4, seed=seed, device=args.device, extra=info)
    if args.probe in ("gru", "shuffle"):
        batcher = T.SequenceBatcher(seqs, targets, shuffle=args.probe == "shuffle", seed=seed)
        make = lambda: M.TPGRU(len(vocab))  # noqa: E731
    else:
        batcher = T.DenseBatcher(x, targets)
        make = lambda: M.BagMLP(x.shape[1])  # noqa: E731
    name = f"tp_{model_tag}_{args.tokens}_{args.probe}_{args.protocol}{args.tag}{'_generic' if args.prompts == 'generic' else ''}_s{seed}"
    res = train_probe(make, batcher, targets, splits, cfg, checkpoint=checkpoint_for(name))
    if args.save_model is None:
        save_run(res, RESULTS / "runs" / name, meta=df[["metadata_idx", "speaker", "pincode", "state", "district"]])
    else:   # a checkpoint run: results/runs stays untouched, the test metrics go into the checkpoint
        import torch
        places = (df.iloc[splits["train"]].groupby("pincode")[["latitude", "longitude"]].first()
                  .reset_index().to_dict("list"))
        args.save_model.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "transcript_probe", "name": name, "recognizer": model_tag, "tokens": args.tokens,
                    "probe": args.probe, "vocab": vocab, "bigrams": bigram_list,
                    "input_dim": int(x.shape[1]) if args.probe in ("unigram", "bigram") else None,
                    "state_dict": res["model_state"], "train_places": places, "protocol": args.protocol,
                    "prompts": args.prompts, "seed": seed, "test_metrics": res["metrics"]["test"]},
                   args.save_model / f"{name}.pt")
    print(name, json.dumps(res["metrics"]["test"]))
