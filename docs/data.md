# Data and models

Facts checked on 9 Oct 2026, with their sources.

## Vaani-Hi (GEO-v task)
- Hugging Face: `changelinglab/vaani-hi-geo` (gated with manual approval by the authors, CC BY 4.0, 6.61 GB parquet).
  The PRiSM README and `configs/data/vaanigeo.yaml` still point to `shikhar7ssu/vaani-hi-geo`; override `data.hf_repo`.
- Columns: audio (WAV bytes), pincode, latitude, longitude, split, metadata_idx, lang_sym, language, gender,
  state, district. No speaker ID, no transcript (`vaanihindi_upload.py` Features dict).
- Construction: 12 states, up to 4 districts per state, 4 shards per district, 600 utterances per shard;
  pincodes kept only if > 450 utterances; splits 75/10/15 random within pincode; 16 kHz, clipped to 20 s
  (PRiSM App. E, `src/recipe/geolocation/local/vaanihindi_sample.py`).
- Splits: 19,780 / 2,668 / 3,985 utterances (PRiSM Table 5).
- Audio bit depth unknown (float32 gives ~29 h, int16 ~57 h). `scripts/vaani_check.py` settles this.

## Vaani (gemination check)
- Hugging Face: `ARTPARK-IISc/Vaani` (gated, CC BY 4.0, 7.21 TB, ~2,122 h transcribed).
- Audio is stored as `audio/{State}/{District}/train-xxxxx-of-*.parquet`; HF configs are per language.
- Districts we need (audio h / transcribed h): Rohtak 199.5 / 16.6, Jhajjar 199.2 / 15.1,
  Charkhi Dadri 194.8 / 13.9, New Delhi 204.5 / 10.4.
- Each file carries language, duration, speakerID, gender, pincode, isTranscriptionAvailable and transcript
  (`scripts/vaani_districts.py`, run 9 Oct 2026). Transcribed rows are mixed with the rest, so the whole
  folders must be downloaded (92.3 GB for the four districts).

| District | Files | Audio h | Transcribed h (utts) | Labelled Haryanvi | Speakers | Rows with speaker ID | Pincodes |
|---|---|---|---|---|---|---|---|
| Haryana/Rohtak | 55 | 200.5 | 16.6 (11,607) | 7.1% (13.6 h) | 537 | 23% | 15 |
| Haryana/Jhajjar | 62 | 200.3 | 15.1 (10,461) | 9.0% (18.7 h) | 536 | 22% | 20 |
| Haryana/CharkhiDadri | 44 | 195.7 | 13.9 (9,716) | 3.5% (6.0 h) | 528 | 21% | 9 |
| Delhi/NewDelhi | 40 | 205.1 | 10.4 (7,359) | 0 | 524 | 31% | 20 |

Only 165 transcribed utterances across the three Haryana districts are labelled Haryanvi; nearly all
Haryana speech is labelled Hindi. Compare districts, not language labels.

### Candidate words for the closure check (`scripts/stop_words.py`)
Transcribed rows only (39,143 utterances; 461k Haryana and 116k Delhi word tokens). A candidate has a single
voiceless stop between vowels in the spoken Devanagari form.
- 110 words occur >= 10 times in both regions: 34,880 Haryana / 7,567 Delhi tokens; ~1,050 / ~380 speakers with ID.
- Top: दिखाई, रखी, फोटो, रखा, काफी, दुकान, रखे, लिखा, सकते, पीछे, नीचे (image-prompted speech favours these).
- Transcribers write standard spelling: Haryana रखा 1,728 vs रक्खा 3; रखी 1,905 vs रक्खी 1; कूलर 73 vs कुल्लर 4.
  So the text does not mark the dialect doubling; selection by spelling is independent of it.

## Models
| Model | Checkpoint | Notes |
|---|---|---|
| W2V2P-LV60 | `facebook/wav2vec2-lv-60-espeak-cv-ft` | Apache-2.0; 24 layers, 1024-d, stride 320 (50 Hz); vocab 392 with kː lː tː ... tokens |
| ZIPA-CTC-NS | `anyspeech/zipa-large-crctc-ns-800k` | MIT; ~300M; ships .pth and ONNX; tokens.txt has 127 tokens incl. standalone ː (id 111) |
| Whisper-small | `openai/whisper-small` | Apache-2.0; 244M; 13 hidden-state outputs, 768-d |
| POWSM (fallback) | `espnet/powsm` | supported by the PRiSM toolkit |

PRiSM ships GEO-v recipes for all of these (`configs/experiment/probing/geolocation_vaani_*.yaml`)
and its net configs (`configs/model/net/{w2v2ph,zipactc,whisper}.yaml`) list the exact checkpoints.

## PRiSM probe settings (App. C)
- TP: 2-layer biGRU, hidden 256, input 128, dropout 0.1, mean pooling, lr 1e-3.
- RP: last layer, attention pooling, 2-layer MLP, lr 2e-4, angular loss.
- Early stopping on validation km (patience 5, minimum 10 epochs).
- Metric: Recall@1 and mean great-circle error. TP <= 15 min/run, RP <= 3 h/run on a 40 GB GPU.

## Published GEO-v numbers (Recall@1, PRiSM Tables 4 and 7)
| Model | TP | RP last | RP weighted |
|---|---|---|---|
| MultiIPA | 44.5 | 24.5 | 34.1 |
| W2V2P-LV60 | 45.7 | 24.6 | 34.9 |
| W2V2P-XLSR53 | 44.5 | 21.5 | 35.5 |
| ZIPA-CTC | 44.1 | 23.0 | 34.3 |
| ZIPA-CTC-NS | 44.7 | 24.0 | 32.8 |
| POWSM | 43.7 | 21.5 | 32.0 |
| POWSM-CTC | 42.9 | 14.1 | 24.4 |
| WavLM | -- | 23.5 | 30.8 |
| Whisper | -- | 35.0 | 24.0 |
| Mean (7 PR) | 44.3 | 21.9 | 32.6 |

Mean error: TP 146 km vs RP 253 km; region spans 1478 km E-W and 1703 km N-S (PRiSM §6.3).

## Sources
- PRiSM paper: arXiv 2601.14046; code https://github.com/changelinglab/prism (no LICENSE file)
- https://huggingface.co/collections/changelinglab/prism
- https://huggingface.co/datasets/changelinglab/vaani-hi-geo
- https://huggingface.co/datasets/ARTPARK-IISc/Vaani
- https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
- https://huggingface.co/anyspeech/zipa-large-crctc-ns-800k, https://github.com/lingjzhu/zipa
- https://huggingface.co/openai/whisper-small
