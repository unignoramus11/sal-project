# Nidhi: consolidated existing-results report

> These are previously saved team results. Running this script does not reproduce training, annotation, model inference, or acoustic measurement.

## Existing evidence inventory

- Valid saved model runs: **391**; family counts: `{'dense': 55, 'tp': 336}`.
- Invalid model runs: **0**.
- See `source_manifest.json` for hashes and missing prerequisites.

## RQ0: speaker-only ECAPA control

| Split / config | Seeds | Recall@1 (%) | SD (pp) | Mean km |
| --- | --- | --- | --- | --- |
| dense_ecapa_rebuild_prism | 5 | 41.57 | 1.62 | 196.90 |
| dense_ecapa_rebuild_speaker_split | 5 | 10.14 | 0.62 | 449.97 |

ECAPA results were produced previously; they diagnose speaker leakage in the random split.

## RQ1: order effects with speaker controls

Scores below are taken directly from existing `results/tables/configs.csv`. Do not interpret the random speaker-overlapping split as unseen-speaker generalization.

| Config | Seeds | Mean Recall@1 (%) | SD (pp) | Mean km |
| --- | --- | --- | --- | --- |
| tp_w2v2_rebuild_phone_bigram_prism | 5 | 27.87 | 0.77 | 377.00 |
| tp_w2v2_rebuild_phone_bigram_speaker_split | 5 | 17.83 | 1.68 | 453.07 |
| tp_w2v2_rebuild_phone_gru_prism | 2 | 21.37 | 1.27 | 404.00 |
| tp_w2v2_rebuild_phone_gru_speaker_split | 2 | 11.78 | 1.26 | 455.12 |
| tp_w2v2_rebuild_phone_shuffle_prism | 2 | 18.30 | 0.00 | 446.01 |
| tp_w2v2_rebuild_phone_shuffle_speaker_split | 2 | 15.65 | 0.49 | 475.76 |
| tp_w2v2_rebuild_phone_unigram_prism | 5 | 15.52 | 1.10 | 453.15 |
| tp_w2v2_rebuild_phone_unigram_speaker_split | 5 | 12.64 | 0.74 | 477.94 |
| tp_zipa_rebuild_phone_bigram_prism | 5 | 28.63 | 0.79 | 362.05 |
| tp_zipa_rebuild_phone_bigram_speaker_split | 5 | 21.27 | 1.00 | 418.94 |
| tp_zipa_rebuild_phone_gru_prism | 2 | 21.94 | 0.54 | 387.70 |
| tp_zipa_rebuild_phone_gru_speaker_split | 2 | 16.00 | 2.13 | 440.89 |
| tp_zipa_rebuild_phone_shuffle_prism | 2 | 18.64 | 0.27 | 449.62 |
| tp_zipa_rebuild_phone_shuffle_speaker_split | 2 | 15.58 | 2.49 | 474.47 |
| tp_zipa_rebuild_phone_unigram_prism | 5 | 15.17 | 0.52 | 458.02 |
| tp_zipa_rebuild_phone_unigram_speaker_split | 5 | 12.67 | 1.04 | 480.66 |

### Existing paired phone GRU vs shuffle contrasts

| A vs B | N test rows | Recall@1 A−B (pp) | 95% CI (pp) | p |
| --- | --- | --- | --- | --- |
| w2v2_rebuild_phone_gru_prism vs w2v2_rebuild_phone_shuffle_prism | 3404 | 3.94 | [1.91, 5.92] | 0.000 |
| w2v2_rebuild_phone_gru_speaker_split vs w2v2_rebuild_phone_shuffle_speaker_split | 3581 | -3.49 | [-6.48, -1.04] | 0.005 |
| zipa_rebuild_phone_gru_prism vs zipa_rebuild_phone_shuffle_prism | 3404 | 4.29 | [2.53, 6.02] | 0.000 |
| zipa_rebuild_phone_gru_speaker_split vs zipa_rebuild_phone_shuffle_speaker_split | 3581 | 0.45 | [-1.91, 2.90] | 0.738 |

## RQ2: recognizer doubled-phone rate (previous experiment)

| Recognizer | Haryana (%) | Other states (%) | H−other (pp) | 95% CI (pp) | Bootstrap p |
| --- | --- | --- | --- | --- | --- |
| w2v2 | 1.00 | 1.17 | -0.17 | [-0.32, -0.04] | 0.012 |
| zipa | 1.63 | 1.82 | -0.19 | [-0.53, 0.18] | 0.298 |

DPR measures output symbol repetition/length marks; it is **not** acoustic consonant duration.

## RQ2: consonant duration (previous experiment)

Unadjusted descriptions only; see `results/gemination/mixed_model.txt` and `robustness.txt` for adjusted models and sensitivity analysis.

| Region | Tokens | Speakers | Mean voiceless interval (ms) | Mean closure (ms) |
| --- | --- | --- | --- | --- |
| Delhi | 4198.0 | 716.0 | 77.64 | 41.53 |
| Haryana | 21835.0 | 2287.0 | 86.71 | 48.97 |

### Existing robustness analysis

```text
tokens 26033, speakers 3003, words 62
district CharkhiDadri vs NewDelhi: x1.132 (95% CI x1.090 to x1.176), p = 1.3e-10, tokens 6885
district Jhajjar vs NewDelhi: x1.153 (95% CI x1.109 to x1.197), p = 2.9e-13, tokens 7221
district Rohtak vs NewDelhi: x1.140 (95% CI x1.099 to x1.183), p = 3.6e-12, tokens 7729
by word: Haryana longer in 45 of 57 words (sign test p = 1.3e-05); median ratio x1.095
speaker level: 2287 Haryana vs 716 New Delhi speakers, word-centred log voiceless interval difference x1.140 (95% bootstrap CI x1.105 to x1.177)
random-effects meta-analysis over 53 words (speaker means, >= 10 speakers per region): x1.110 (95% CI x1.081 to x1.139); between-word tau = 0.049 on the log scale, I2 = 31%
```

## RQ3: pooled prosody baseline (previous experiment)

| Config | Seeds | Recall@1 (%) | SD (pp) | Mean km |
| --- | --- | --- | --- | --- |
| dense_prosody_rebuild_prism | 5 | 5.53 | 0.66 | 572.36 |
| dense_prosody_rebuild_speaker_split | 5 | 4.79 | 0.41 | 598.15 |

These are utterance-level *pooled features*, not a frame-contour BiGRU or TP/prosody fusion.

## Missing inputs for further experiments

- rebuild_metadata: MISSING
- split_table: MISSING
- w2v2_transcripts: MISSING
- zipa_transcripts: MISSING
- prosody_frames: MISSING
- prosody_index: MISSING
- prosody_meta: MISSING
- prosody_pooled: MISSING
- w2v2_frame_cache: MISSING
- district_closures: MISSING
- district_clips: MISSING
- blinded_audit_key: MISSING

## Remaining experimental work

- **50-token blinded Praat audit:** needs original wave clips and annotated TextGrids. Not completed simply by having acoustic summaries.
- **Syllabification:** needs actual recognizer IPA phone strings. DPR summary parquet does not retain those strings. Run `scripts/nidhi_syllables.py` if transcripts become available.
- **Prosody contour BiGRU/fusion:** needs original precomputed contour features, split and geolocation metadata (or audio). This repository ZIP contains none of these.
- **Last-layer RP-Attn vs RP-GRU:** needs full hidden-state frame caches and aligned metadata; not inferable from TP or pooled prosody results.
- **Layer-wise geminate representation/attribution:** missing frame-level model features/audio.

## Submission limitation

For the current ADA account, home storage is capped at 50 GB, gateway-only `/share1` is not mounted on compute nodes, scratch is temporary, and an available GPU has 11 GB of VRAM. We retained verified saved analyses and did not independently reproduce their training or acoustic extraction on ADA. Full multi-layer RP runs, comprehensive IG, and any new analyses without raw inputs must be labelled not performed. The existing TP order results do not by themselves settle the RP representation/ordering hypothesis.
