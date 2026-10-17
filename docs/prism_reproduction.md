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
