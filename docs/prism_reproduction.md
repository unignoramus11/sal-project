# PRiSM GEO-v: what the code actually does

Our probes are reimplemented from PRiSM's code (commit `2e7b754`, 9 Apr 2026), not from the paper
alone. This file records every setting we copy, where it comes from, and where code and paper differ.
"Code" paths are relative to `external/prism/`.

## Data (Vaani-Hi)

| Item | Value | Source |
|---|---|---|
| Target | (latitude, longitude) of the pincode, **in radians** | `src/data/vaani/geolocation.py:72` |
| Audio | 16 kHz, clipped to 20 s; released as 32-bit float WAV (`torchaudio.save` of a float tensor) | `vaanihindi_upload.py` |
| Splits | 75/10/15 within pincode, numpy seed 42, `sample(frac=1, random_state=42)` | `vaanihindi_upload.py:create_splits` |
| Pincodes | kept if > 450 utterances (strictly) | `vaanihindi_sample.py` |
| Coordinates | mean over all post offices of the pincode, from a local `pincode_metadata.csv` (India Post directory; no placeholder filtering) | `vaanihindi_upload.py` |
| Size | released card: 26,385 utterances; paper Table 5: 19,780 / 2,668 / 3,985 = 26,433 | HF card vs. paper |

## Transcription (input to the transcript probe)

- W2V2P: `Wav2Vec2Processor` (waveform normalised to zero mean, unit variance), `Wav2Vec2ForCTC`, argmax per
  frame, CTC collapse (drop blanks, merge repeats), tokens **joined with no separator**, `<unk>` and `<pad>`
  removed. One utterance per forward pass, no padding. `src/model/wav2vec2phoneme/*.py`
- The joined string loses phone boundaries: a doubled consonant appears as `kk`, a long one as `kː`.

## Transcript probe (TP)

| Item | Value | Source |
|---|---|---|
| Tokens | **characters** of the joined string (so `kː` is two tokens); vocab from train split; ids 0-3 = `<pad>`,`<unk>`,`<s>`,`</s>` | `src/core/tokenizer/character_tokenizer.py`, `src/data/json_dataset.py` |
| Truncation | first 512 characters; empty transcript replaced by `?` | `json_dataset.py` |
| Embedding | 128-d, padding_idx 0, dropout 0.1 after embedding | `src/model/common/ipa_embedding.py`, `configs/model/net/ipa_embedding.yaml` |
| RNN | 2-layer bidirectional GRU, hidden 256, inter-layer dropout 0.1, packed sequences | `src/model/heads/rnn_head.py`, `configs/model/head/rnn.yaml` |
| Pooling | masked mean over time, then LayerNorm(512), dropout 0.1, Linear(512 -> 3) | `rnn_head.py` |
| Batch | 64 | `configs/experiment/cascade/rnn_geolocation.yaml` |
| LR | Adam 1e-3, weight decay 0 | same, `configs/model/classification.yaml` |
| Epochs | min 10, **max 200** | `rnn_geolocation.yaml` |
| Early stopping | on `val/err_km`, **patience 100** validation checks | `rnn_geolocation.yaml` (paper App. C says patience 5) |

## Representation probe (RP)

| Item | Value | Source |
|---|---|---|
| Features | `hidden_states[-1]` of the frozen encoder (last layer); encoder run online every step | `wav2vec2phoneme_model.py:encode`, `freeze_encoder: true` |
| Pooling | L2-normalise frames; single-head `nn.MultiheadAttention` with one **learned query** (L2-normalised), key padding mask; L2-normalise the pooled vector | `src/model/heads/attnmlp_head.py` |
| MLP | Linear(D -> D/2), tanh, Linear(D/2 -> 3); dropout 0 | `attnmlp_head.py`, `configs/model/head/attnmlp.yaml` |
| Batch | 48 | `configs/experiment/probing/geolocation_vaani_*.yaml` |
| LR | Adam 2e-4, weight decay 0 | same |
| Epochs | min 10, max 100 | same |
| Early stopping | on `val/err_km`, patience 5 validation checks | same (matches paper) |

## Shared training details

- Loss: mean great-circle angle between the predicted direction (xyz output, any norm) and the target
  (`src/recipe/common/geolocation_loss.py`).
- Scheduler: `LinearLR(total_iters=200)` stepped **per batch**, default start factor 1/3, so the LR ramps
  from lr/3 to lr over the first 200 steps (`configs/model/classification.yaml`, `classification_module.py:423`).
- Gradient clipping 1.0; validation 4 times per epoch (`val_check_interval: 0.25`). Patience counts
  validation checks, so RP patience 5 is 1.25 epochs.
- Checkpoint: best `val/err_km`; test metrics come from that checkpoint.
- Seed 42 for the published single runs; Table 4 reports mean ± sd over 5 seeds.

## Metrics

- `err_km`: mean great-circle error with Earth radius 6378.1 km (`src/metrics/geolocation.py`).
- Recall@1 = 100 - `miss_rate_top1`: candidates are the **distinct target locations present in the
  evaluated split**; a hit means the prediction's nearest candidate by cosine is the true one.
- The paper reports Recall@1 on test.

## Differences between code and paper

| Item | Paper | Code | We follow |
|---|---|---|---|
| TP early stopping patience | 5 | 100 (max 200 epochs) | code for O1 (it produced the published numbers); we also report patience 5 |
| RP head | "attention pool + 2-layer MLP" | learned-query MHA over normalised frames | code |
| Vaani-Hi size | 26,433 | 26,385 on the released card | report both |

## Reproduction status (18 Oct 2026)

Official Vaani-Hi access is still pending (403 on 18 Oct), so every number so far is on the rebuilt copy.

| | PRiSM paper (official data) | Ours (rebuilt data, random split) |
|---|---|---|
| Utterances | 26,385 (card) | 22,551 |
| Mean duration | ~4.5 s (card byte count / float32 16 kHz) | 6.4 s |
| Naive (centroid) Recall@1 | 3.3 | 3.4 |
| TP-GRU, W2V2 chars, Recall@1 | 45.7 | 22.0 (s42), 20.3 (s46) |
| TP-GRU, mean km | 146 | 395-405 |

The probe code matches PRiSM's (architecture, tokenisation, optimiser, schedule, early stopping; table
above), and the transcripts match PRiSM's runner output exactly. On our data the char TP-GRU fits less well
than a bag-of-bigrams MLP (train Recall@1 65 vs 82, test 22.0 vs 26.3), and test transcripts almost never
repeat a training transcript (0.03%).

Leads, not yet tested:
- The two samples differ in size (26,385 vs 22,551) and utterance length (~4.5 vs 6.4 s), so PRiSM's
  sampling selects different utterances, possibly more per speaker. On our data a voice-only embedding
  (ECAPA) reaches 41.6 Recall@1 and 197 km under the random split, close to PRiSM's TP; a TP that could
  recognise speakers through their transcripts would land in the same place. Our control task shows that
  on our data trained transcript bags barely recognise speakers (control lift +2.1 vs ECAPA +19.1), but a
  1-nearest-neighbour transcript matcher reaches 39.6 Recall@1 under the random split (18.2 speaker-disjoint)
  and finds a same-speaker neighbour for 25% of test utterances (chance 0.3%). So the transcripts do allow
  speaker memorisation; a TP that fits training data more tightly than ours (e.g., more utterances per
  speaker in PRiSM's sample) would approach the matcher.
- Once access arrives: rerun the TP-GRU on the official data (same code), measure speaker leakage there with
  the file-name speaker codes if the official rows can be matched to Vaani files (metadata_idx), and compare
  utterances per speaker.
