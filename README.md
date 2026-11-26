# Order, Duration, or Melody?

Probing what phone recognizers use to geolocate Haryanvi and Hindi-belt speech. Course project for Speech Analysis
and Linguistics (Monsoon 2026), IIIT Hyderabad, by Mohit Kumar Singh and Nidhi Vaidya.

PRiSM (Bharadwaj et al., ACL 2026) geolocates Vaani-Hi utterances from phone recognizer output and credits phone
order for its transcript probe's lead. We test whether the probes use speakers, topic, phone order, consonant
duration or intonation. The report is `paper/final.pdf`.

- **Test set:** [Google Drive](https://drive.google.com/drive/folders/16y49znRs_ojTTsIt91AJ0aLzDm8geLSi), the
  speaker-disjoint test split (3,581 utterances, 6.3 h, 223 speakers) with audio, labels and phone transcripts.
- **Checkpoints:** `checkpoints/`, trained on the training part of that split.

| Checkpoint | Recognizer | Test Recall@1 | Mean error |
| --- | --- | --- | --- |
| `tp_zipa_rebuild_phone_bigram_speaker_split_s42.pt` | ZIPA | 21.9 | 417 km |
| `tp_w2v2_rebuild_phone_bigram_speaker_split_s42.pt` | W2V2 | 18.2 | 449 km |

Both are MLPs over phone unigram and bigram frequencies, our best transcript probes on new speakers.

## Setup

Main environment (Mac or Linux, Python 3.10 to 3.12), for everything except transcribing audio:

```sh
uv sync                                   # or: pip install -r requirements.txt && pip install -e .
```

Transcription environment (Linux with an NVIDIA GPU). The probes read phone strings, not audio. Turning audio into
phone strings runs the W2V2 and ZIPA recognizers with PRiSM's code, which needs torch 2.6 and k2, so it has its own
environment. You need it only for new audio: the test set already includes the transcripts. The script fetches
PRiSM's code at the commit we used and installs the exact package versions we ran:

```sh
bash tools/setup_env.sh                   # creates envs/prism; ends with "environment OK"
```

## Inference

On the test set (no GPU needed):

```sh
uv run python scripts/infer.py --checkpoint checkpoints/tp_zipa_rebuild_phone_bigram_speaker_split_s42.pt
```

On first use this downloads the test set from Drive into `data/testset/` (about 700 MB) and checks every data file
against SHA-256 digests fixed in `src/sal/testset.py` before running. It writes the predicted latitude, longitude
and nearest pincode per utterance to `outputs/inference/`, and prints Recall@1 and mean error (the numbers in the
table above).

On your own audio, transcribe with the checkpoint's recognizer first:

```sh
uv run python scripts/audio_to_parquet.py my_audio/ my_audio.parquet
envs/prism/bin/python scripts/transcribe.py --model zipa --data my_audio.parquet --id-col audio_path --out zipa.parquet
uv run python scripts/infer.py --checkpoint checkpoints/tp_zipa_rebuild_phone_bigram_speaker_split_s42.pt \
    --transcripts zipa.parquet --id-col audio_path --out predictions.csv
```

## Training

The data is our rebuild of Vaani-Hi (accept the Vaani terms on Hugging Face first):

```sh
uv run python scripts/build_vaani_hi.py                                              # audio and labels
uv run python scripts/make_splits.py data/vaani_hi_rebuild/vaani_hi_rebuild.parquet  # PRiSM's and speaker-disjoint splits
uv run python scripts/make_benchmark.py                                              # 5-fold speaker-group benchmark
envs/prism/bin/python scripts/transcribe.py --model zipa --data data/vaani_hi_rebuild/vaani_hi_rebuild.parquet \
    --out outputs/transcripts/zipa_rebuild.parquet                                  # and --model w2v2
```

Then train a probe. `--probe` is `gru`, `shuffle`, `unigram` or `bigram`; `--protocol` is `prism`, `speaker_split` or
a benchmark fold such as `all_f0`. Results go to `results/runs/<name>/`, and `--save-model checkpoints` writes a
checkpoint instead:

```sh
uv run python scripts/run_tp.py --transcripts outputs/transcripts/zipa_rebuild.parquet --tokens phone \
    --probe bigram --protocol speaker_split --seeds 42
```

The other probes have their own scripts: `run_rp.py` (hidden states, after `extract_features.py`), `run_dense.py`
(voice and channel features), `run_prosody.py` (pitch and intensity) and `baselines.py` (majority and lookups).

## Evaluation

```sh
make analysis     # scores, confidence intervals and null tests from results/runs
make final        # tables, figures and the report, with a check of every number quoted in it
uv run pytest
```

Recall@1 and mean error in km follow PRiSM's definitions (`src/sal/probes/geo.py`).

## Layout

`src/sal/` library code · `scripts/` one script per step · `results/` every run and table · `checkpoints/` released
probes · `paper/` report · `docs/` design notes and data audits · `slurm/`, `tools/` cluster and GPU job scripts ·
`tests/` unit tests.
