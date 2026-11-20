# Cross-validated benchmark (final evaluation)

A speaker-disjoint, topic-controlled evaluation built from our rebuilt Vaani-Hi, in which every speaker is tested
exactly once. Built by `scripts/make_benchmark.py` (tables in `results/benchmark/`, versioned), scored by
`scripts/benchmark_eval.py`. The splits and the code are ready and tested; the probe runs are scheduled for
November, after the mid evaluation.

## What it fixes

| Problem in the current evaluation | Evidence | Fix |
|---|---|---|
| Speaker leakage under PRiSM's random split | 97.9% of test utterances have their speaker in train; a speaker-ID lookup scores 98.0 | folds over speakers |
| Look-alike speaker codes | 49 of 223 test speakers of our speaker-disjoint split have a code in train whose voice is as close as two halves of one speaker's recordings; ECAPA's lift on new speakers comes only from them (`results/rebuild_checks/lookalike.csv`) | codes merged into speaker groups (ECAPA centroid cosine >= 0.757, the 5th percentile of same-speaker split-half cosines; average linkage within district): 1,570 codes -> 1,327 groups |
| Small test set | 223 test speakers (119 on shared pictures), wide CIs | 5-fold cross-validation: every group tested once (1,327 groups; 650 on shared pictures) |
| Topic | the picture ID alone scores 61.8 R@1 (random split) and 27.5 on new speakers (`results/rebuild_checks/picture_lookup.csv`) | track `shared`: train and test on shared-picture utterances only, where the picture lookup is below the majority |
| Heavy speakers | median 5, max 257 utterances per speaker | speaker-macro Recall@1 next to the utterance-level score |
| Misplaced coordinates | Dhar 454552 266 km and Ranchi 834008 61 km from their post offices (mean of offices, as in PRiSM) | every pincode at the densest cluster of its offices (`sal.data.rebuild.robust_pincode_coordinates`) |
| Duplicates | 3 recordings appear twice | one copy kept (22,548 rows) |

## Design

- Folds: speaker groups are assigned to 5 folds within each pincode with the greedy balancing of
  `sal.data.splits` (`speaker_folds`). In fold k of a track, test = the groups of fold k, val = 12.5% of the
  other groups (pincode-stratified), train = the rest. Both tracks use the same fold of each group.
- Tracks: `all` (22,548 utterances, 1,327 groups, 26 pincodes) and `shared` (7,157 utterances, 650 groups, 24
  pincodes; Jaisalmer 345001 and Bhopal 462001 have fewer than 4 groups on shared pictures and are left out, so a
  tested pincode always has training speakers).
- Columns of `results/benchmark/splits.parquet`: `all_f0..all_f4`, `shared_f0..shared_f4` with train/val/test
  (missing = not in the track); `meta.parquet` holds the corrected coordinates. Any probe script takes them with
  `--dataset results/benchmark/meta.parquet --splits results/benchmark/splits.parquet --protocol <track>_f<k>
  --tag _bench`.
- Scoring (`sal.benchmark`): each fold's test predictions are averaged over seeds and the folds are pooled, so every
  utterance has one out-of-fold prediction. Recall@1 over the track's pincodes, speaker-macro Recall@1, state
  accuracy, mean km; lifts over the per-fold majority and km gain over the per-fold best constant, paired per
  utterance; 95% CIs by bootstrap over speaker groups. Paired comparisons: BiGRU vs shuffled BiGRU, bigram vs
  unigram, phones vs characters, ZIPA vs W2V2.

## Baselines (no training; `scripts/benchmark_eval.py`)

| Track | Majority | Picture lookup | 1-NN transcript (W2V2 / ZIPA) |
|---|---|---|---|
| all | 10.6 | 29.0 (+18.4 [15.1, 22.1]) | 15.7 / 18.1 |
| shared | 20.3 | 10.1 (-10.2) | 17.6 / 18.7 |

## Runs (November, all on Ada)
| What | Jobs | Script |
|---|---|---|
| bag probes {W2V2, ZIPA} x {phone, char} x {unigram, bigram}, ECAPA, prosody statistics; 2 tracks x 5 folds x 3 seeds (300 runs, CPU) | 10 | `slurm/bench_light.sbatch` |
| transcript BiGRUs {W2V2, ZIPA} x {char GRU, phone GRU, shuffled phone GRU}; 2 tracks x 5 folds x 3 seeds (180 runs, GPU) | 8 | `slurm/bench_gru.sbatch` |
| RP-Attn, 3 models x 2 tracks x 5 folds x 3 seeds (90 runs, GPU) | 12 | `slurm/rp_benchmark.sh` |

The first two are submitted by `slurm/november.sh` (docs/ada.md step 13), the third is docs/ada.md step 12.
`tools/bench_mac.sh` and `tools/pc_bench.sh` run the same probes locally and are kept as a fallback.
Then `uv run python scripts/benchmark_eval.py` -> `results/benchmark/{scores,paired,folds}.csv`.

## Tests

`tests/test_splits.py` (folds), `tests/test_rebuild.py` (coordinates), `tests/test_benchmark.py` (run names, fold
pooling, speaker-macro, lifts). An end-to-end smoke run (bag, BiGRU and ECAPA probes on every fold for 1-2 epochs,
scored by `benchmark_eval.py --tag _smoke`) passed on 24 Oct 2026 and was deleted.
