"""Predict locations with a saved transcript probe, and score them if the true locations are known.

With only --checkpoint, the script scores the checkpoint on the public test set with that recognizer's transcripts.
The test set is downloaded on first use (about 700 MB, into data/testset/) and every file is checked against fixed
SHA-256 digests before anything runs (sal.testset). Only checkpoints trained on the training part of the
speaker-disjoint split of our rebuild are accepted there, since any other training data contains test speakers:
    uv run python scripts/infer.py --checkpoint checkpoints/tp_zipa_rebuild_phone_bigram_speaker_split_s42.pt

Inference has two steps, because the recognizers run in PRiSM's environment (envs/prism) and the probes in ours:

  1. Transcribe the audio with the probe's recognizer (W2V2 or ZIPA), using PRiSM's own model code:
       envs/prism/bin/python scripts/transcribe.py --model zipa --data testset.parquet \
           --out outputs/transcripts/zipa_testset.parquet --device cuda
     (`--data` is a parquet file with an `audio` column of 16 kHz mono WAV bytes and an id column;
      scripts/audio_to_parquet.py builds one from a folder of audio files.)
  2. Run the probe on the transcripts:
       uv run python scripts/infer.py --checkpoint checkpoints/<name>.pt \
           --transcripts outputs/transcripts/zipa_testset.parquet --labels testset.parquet \
           --out results/inference/<name>_testset.csv

The output has one row per utterance: the predicted latitude and longitude, and the training pincode nearest to the
prediction. With --labels (a parquet with the id column, latitude and longitude), the script also prints Recall@1
and mean error in km, computed exactly as in training (sal.probes.geo; Recall@1 ranks the distinct locations of the
scored utterances). A shuffled-order checkpoint (probe "shuffle") draws its own fixed permutations, so its predictions
are not identical to those of the training run; every other probe reproduces them exactly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sal import testset
from sal.paths import OUTPUTS
from sal.probes import geo
from sal.probes import models as M
from sal.probes import tp_data as T

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", type=Path, required=True)
ap.add_argument("--transcripts", type=Path, default=None,
                help="output of scripts/transcribe.py (default: the public test set, downloaded if needed)")
ap.add_argument("--labels", type=Path, default=None, help="parquet with the id column, latitude and longitude")
ap.add_argument("--id-col", default="metadata_idx")
ap.add_argument("--out", type=Path, default=None, help="default: outputs/inference/<checkpoint name>.csv")
ap.add_argument("--device", default="cpu")
ap.add_argument("--batch-size", type=int, default=256)
args = ap.parse_args()

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
if ck.get("kind") != "transcript_probe":
    raise SystemExit(f"{args.checkpoint} is not a transcript-probe checkpoint written by scripts/run_tp.py")
if args.transcripts is None:
    recognizer, _, data = ck["recognizer"].partition("_")
    if data != "rebuild" or ck["protocol"] != "speaker_split":
        raise SystemExit(f"{ck['name']} was trained on {ck['recognizer']} with protocol {ck['protocol']}, whose "
                         "training data includes speakers of the public test set; pass --transcripts to run it on "
                         "other data")
    try:
        args.transcripts = testset.transcripts(recognizer)
    except testset.TestSetError as e:
        raise SystemExit(str(e)) from e
    if args.id_col != "metadata_idx":
        raise SystemExit("the test set's id column is metadata_idx")
    args.labels = args.labels or args.transcripts.parent / "testset.parquet"
args.out = args.out or OUTPUTS / "inference" / f"{ck['name']}.csv"
cols = [args.id_col, "processed_transcript", "tokens"]
try:
    tr = pd.read_parquet(args.transcripts, columns=cols)
except (KeyError, ValueError, OSError) as e:
    raise SystemExit(f"{args.transcripts} needs the columns {cols} (scripts/transcribe.py writes them): {e}") from e
if tr.empty:
    raise SystemExit(f"{args.transcripts} has no utterances")
if tr[args.id_col].duplicated().any():
    raise SystemExit(f"{args.transcripts} repeats ids in {args.id_col}")
vocab = ck["vocab"]
seqs = [T.encode(T.units(t, list(k), ck["tokens"]), vocab)
        for t, k in zip(tr["processed_transcript"], tr["tokens"], strict=True)]

if ck["probe"] in ("gru", "shuffle"):
    model = M.TPGRU(len(vocab))
    batcher = T.SequenceBatcher(seqs, np.zeros((len(seqs), 2)), shuffle=ck["probe"] == "shuffle", seed=ck["seed"])
else:
    x = T.bag_matrix(seqs, len(vocab), [tuple(b) for b in ck["bigrams"]] if ck["bigrams"] is not None else None)
    assert x.shape[1] == ck["input_dim"], (x.shape, ck["input_dim"])
    model = M.BagMLP(x.shape[1])
    batcher = T.DenseBatcher(x, np.zeros((len(seqs), 2)))
model.load_state_dict(ck["state_dict"])
model.to(args.device).eval()

out = []
with torch.no_grad():
    for s in range(0, len(seqs), args.batch_size):
        xb, lengths, _ = batcher(np.arange(s, min(s + args.batch_size, len(seqs))), train=False)
        out.append(model(xb.to(args.device), lengths.to(args.device)).float().cpu())
pred = torch.cat(out)
lat, lon = geo.xyz_to_latlon(pred)

places = pd.DataFrame(ck["train_places"])
place_xyz = geo.latlon_to_xyz(torch.tensor(np.radians(places["latitude"].to_numpy())),
                              torch.tensor(np.radians(places["longitude"].to_numpy()))).float()
nearest = (torch.nn.functional.normalize(pred, dim=-1) @ place_xyz.T).argmax(1).numpy()

res = pd.DataFrame({args.id_col: tr[args.id_col], "pred_latitude": np.degrees(lat.numpy()),
                    "pred_longitude": np.degrees(lon.numpy()),
                    "nearest_train_pincode": places["pincode"].to_numpy()[nearest]})
args.out.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(args.out, index=False)
print(f"wrote {args.out} ({len(res)} utterances) with {ck['name']}")

if args.labels is not None:
    try:
        lab = pd.read_parquet(args.labels, columns=[args.id_col, "latitude", "longitude"])
    except (KeyError, ValueError, OSError) as e:
        raise SystemExit(f"{args.labels} needs the columns {args.id_col}, latitude, longitude: {e}") from e
    lab = lab.drop_duplicates(args.id_col)
    lab = res[[args.id_col]].merge(lab, on=args.id_col, how="left")
    if lab["latitude"].isna().any():
        raise SystemExit(f"{lab['latitude'].isna().sum()} transcribed utterances have no label in {args.labels}")
    target = np.radians(lab[["latitude", "longitude"]].to_numpy(np.float64))
    print(json.dumps(geo.summarise(pred.numpy(), target)))
