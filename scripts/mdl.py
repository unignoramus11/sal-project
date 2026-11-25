"""Minimum description length of location given a transcript, by online coding (Voita and Titov, 2020).

Recall@1 and km tell how well a probe does after training on everything; online codelength also counts how much
data the probe needs, so it separates information that is easy to extract from information that a flexible
probe can only memorise. Location is coded as the pincode class (the same candidate set as Recall@1).

The training split is cut into blocks at 1/64, 1/32, ..., 1/2, 1 of its examples. The first block is sent with
a uniform code (log2 K bits per label); every later block is sent with the cross-entropy of a probe trained on
all earlier blocks. Under the speaker-disjoint protocol the order is by speaker (each block adds new speakers),
so the code measures transfer to unseen speakers; under PRiSM's protocol the order is by utterance.
Probes: bag MLPs (unigram / bigram over chars or phones), and TP-GRU / shuffled TP-GRU (needs a GPU), each with a
K-way softmax head instead of the 3-d direction. Early stopping on the protocol's validation split.

Each block model is temperature-calibrated on the validation split. The reference code sends each block with the
label frequencies of the earlier blocks (a probe that learns only the location prior); gain_over_prior is the
number of bits the transcript saves beyond that.
Output: results/mdl/<config>_s<seed>.json and results/mdl/summary.csv (codelengths in kbits, compression =
uniform / online codelength, gain over the prior code; mean and sd over seeds).
Run:    uv run python scripts/mdl.py --transcripts outputs/transcripts/w2v2_rebuild.parquet --tokens phone \
            --probe bigram --protocol speaker_split --seeds 0 1 2
        uv run python scripts/mdl.py --summary      (rebuild summary.csv from every per-seed file, e.g. after a pull)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sal.paths import DATA, RESULTS
from sal.probes import models as M
from sal.probes import tp_data as T
from sal.probes.train import seed_everything

FRACTIONS = [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]


def make_model(probe, n_in, k):
    if probe in ("gru", "shuffle"):
        m = M.TPGRU(n_in)
        m.out = torch.nn.Linear(m.out.in_features, k)
    else:
        m = M.BagMLP(n_in)
        m.f[-1] = torch.nn.Linear(m.f[-1].in_features, k)
    return m


def logits_of(model, batcher, idx, device, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            x, lengths, _ = batcher(idx[s:s + bs], train=False)
            out.append(model(x.to(device), lengths.to(device)).float().cpu())
    return torch.cat(out)


def nll_bits(model, batcher, idx, y, device, temperature=1.0):
    z = logits_of(model, batcher, idx, device) / temperature
    return F.cross_entropy(z, torch.as_tensor(y[idx]), reduction="sum").item() / math.log(2)


def calibrate(model, batcher, val_idx, y, device):
    """Temperature that minimises validation cross-entropy (grid in log space): unseen speakers make the probe
    confidently wrong, and a code must not pay for overconfidence the receiver could remove."""
    z, t = logits_of(model, batcher, val_idx, device), torch.as_tensor(y[val_idx])
    grid = np.exp(np.linspace(np.log(0.5), np.log(50), 60))
    return float(grid[int(np.argmin([F.cross_entropy(z / g, t).item() for g in grid]))])


def prior_bits(y_seen, y_next, y_val, k):
    """Bits for the next block under the label frequencies of the blocks already sent (add-one smoothing),
    flattened by the temperature that minimises validation cross-entropy, as the probes are."""
    logp = torch.log(torch.as_tensor((np.bincount(y_seen, minlength=k) + 1) / (len(y_seen) + k)))
    grid = np.exp(np.linspace(np.log(0.5), np.log(50), 60))
    yv = torch.as_tensor(y_val)
    t = grid[int(np.argmin([F.cross_entropy(logp.expand(len(yv), k) / g, yv).item() for g in grid]))]
    yn = torch.as_tensor(y_next)
    return F.cross_entropy(logp.expand(len(yn), k) / t, yn, reduction="sum").item() / math.log(2)


def fit(probe, n_in, k, batcher, y, train_idx, val_idx, seed, device, max_epochs, patience):
    seed_everything(seed)
    model = make_model(probe, n_in, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    best, best_state, bad = float("inf"), None, 0
    bs = 64
    for _ in range(max_epochs):
        model.train()
        order = train_idx[rng.permutation(len(train_idx))]
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            x, lengths, _ = batcher(idx, train=True)
            loss = F.cross_entropy(model(x.to(device), lengths.to(device)).float(),
                                   torch.as_tensor(y[idx], device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = nll_bits(model, batcher, val_idx, y, device)
        if v < best:
            best, best_state, bad = v, {k_: t.detach().clone() for k_, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", type=Path)
    ap.add_argument("--summary", action="store_true", help="only rebuild results/mdl/summary.csv from the per-seed files")
    ap.add_argument("--tokens", choices=["char", "phone"], default="phone")
    ap.add_argument("--probe", choices=["unigram", "bigram", "gru", "shuffle"], default="bigram")
    ap.add_argument("--protocol", choices=["prism", "speaker_split"], default="speaker_split")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()
    if args.summary:
        summarise(RESULTS / "mdl")
        return
    if args.transcripts is None:
        ap.error("--transcripts is required unless --summary is given")

    meta = pd.read_parquet(DATA / "vaani_hi_rebuild" / "splits.parquet").sort_values("metadata_idx")
    tr = pd.read_parquet(args.transcripts, columns=["metadata_idx", "processed_transcript", "tokens"])
    df = meta.merge(tr, on="metadata_idx", how="inner").reset_index(drop=True)
    classes = sorted(df["pincode"].unique())
    y = df["pincode"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)
    k = len(classes)
    train_all = np.flatnonzero(df[args.protocol].to_numpy() == "train")
    val_idx = np.flatnonzero(df[args.protocol].to_numpy() == "val")
    unit_seqs = [T.units(t, list(kk), args.tokens) for t, kk in zip(df["processed_transcript"], df["tokens"], strict=True)]
    tag = args.transcripts.stem
    config = f"mdl_{tag}_{args.tokens}_{args.probe}_{args.protocol}"
    out = RESULTS / "mdl"
    out.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        if args.protocol == "speaker_split":            # blocks add whole speakers
            spk = df.loc[train_all, "speaker"].to_numpy()
            order_spk = rng.permutation(np.unique(spk))
            rank = {s: r for r, s in enumerate(order_spk)}
            order = train_all[np.lexsort((rng.random(len(train_all)), [rank[s] for s in spk]))]
        else:
            order = train_all[rng.permutation(len(train_all))]
        ends = [max(1, round(f * len(order))) for f in FRACTIONS]
        # inputs (transcripts) are known to sender and receiver; only labels are coded, so the vocabulary and
        # bag statistics may come from the whole train split
        vocab = T.build_vocab([unit_seqs[i] for i in train_all])
        seqs = [T.encode(s, vocab) for s in unit_seqs]
        if args.probe in ("unigram", "bigram"):
            x, _ = T.bag_features(seqs, len(vocab), train_all, bigrams=args.probe == "bigram")
            batcher, n_in = T.DenseBatcher(x, np.zeros((len(df), 2))), x.shape[1]
        else:
            batcher = T.SequenceBatcher(seqs, np.zeros((len(df), 2)), shuffle=args.probe == "shuffle", seed=seed)
            n_in = len(vocab)
        blocks = [{"end": ends[0], "bits": ends[0] * math.log2(k), "uniform": True}]
        for a, b in zip(ends[:-1], ends[1:], strict=False):
            model = fit(args.probe, n_in, k, batcher, y, order[:a], val_idx, seed, args.device,
                        args.max_epochs, args.patience)
            temp = calibrate(model, batcher, val_idx, y, args.device)
            bits = nll_bits(model, batcher, order[a:b], y, args.device, temp)
            pbits = prior_bits(y[order[:a]], y[order[a:b]], y[val_idx], k)
            blocks.append({"end": b, "bits": bits, "prior_bits": pbits, "temperature": temp,
                           "bits_per_label": bits / (b - a), "prior_bits_per_label": pbits / (b - a)})
            print(f"{config} s{seed}: block {a}->{b}: {bits / (b - a):.3f} bits/label (prior "
                  f"{pbits / (b - a):.3f}, T={temp:.2f})", flush=True)
        online = sum(bl["bits"] for bl in blocks)
        prior = blocks[0]["bits"] + sum(bl["prior_bits"] for bl in blocks[1:])
        uniform = len(order) * math.log2(k)
        res = {"config": config, "seed": seed, "n": len(order), "classes": k, "online_kbits": online / 1000,
               "prior_kbits": prior / 1000, "uniform_kbits": uniform / 1000, "compression": uniform / online,
               "gain_over_prior_kbits": (prior - online) / 1000, "blocks": blocks}
        (out / f"{config}_s{seed}.json").write_text(json.dumps(res, indent=1))
        print(f"{config} s{seed}: online {online / 1000:.1f} kbits, uniform {uniform / 1000:.1f}, "
              f"compression {uniform / online:.3f}", flush=True)

    summarise(out)


def summarise(out):
    """summary.csv: mean and sd over seeds of every config's codelengths, from the per-seed files in `out`."""
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("mdl_*_s*.json"))]
    s = pd.DataFrame(rows).groupby("config").agg(seeds=("seed", "size"), online_kbits=("online_kbits", "mean"),
                                                 online_sd=("online_kbits", "std"), prior_kbits=("prior_kbits", "mean"),
                                                 gain_over_prior=("gain_over_prior_kbits", "mean"),
                                                 gain_sd=("gain_over_prior_kbits", "std"), uniform_kbits=("uniform_kbits", "first"),
                                                 compression=("compression", "mean"), compression_sd=("compression", "std"))
    s.to_csv(out / "summary.csv")
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
