# Order, Duration, or Melody?

Probing what phone recognizers use to geolocate Haryanvi and Hindi-belt speech.

Course project for Speech Analysis and Linguistics (Monsoon 2026), IIIT Hyderabad.
Mohit Kumar Singh and Nidhi Vaidya.

PRiSM (Bharadwaj et al., ACL 2026) finds that a transcript probe over predicted phones geolocates Vaani-Hi
speakers better than a representation probe over hidden states (146 km vs 253 km mean error), and suggests
phone order as the reason without testing it. We test what the phone string and the hidden states actually
carry:

- **RQ0 Speakers.** PRiSM splits utterances at random, so 97.9% of test utterances have their speaker in
  training. How much of each probe's score is speaker recognition (and topic) rather than dialect?
  Speaker-disjoint split, speaker-only baselines, control task, nearest-neighbour transcript matcher.
- **RQ1 Order.** {transcript, representation} x {order-aware, order-free} probes at matched size, with layer
  choice held fixed; unigram, bigram, BiGRU and shuffled BiGRU over characters and phones; MDL.
- **RQ2 Duration.** Consonant gemination in Haryanvi: doubled phones in recognizer output, Praat closure
  durations in Haryana vs New Delhi audio, and encoding across layers.
- **RQ3 Melody.** Whether F0 and intensity contours add to the transcript probe.

Current results, with confidence intervals: [`docs/findings.md`](docs/findings.md). Status of the PRiSM
reproduction: [`docs/prism_reproduction.md`](docs/prism_reproduction.md).

## Layout

| Path        | Contents                                                                                                                                                                                |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/sal/`  | Library: dataset rebuild and splits, phone segmentation, probes (PRiSM-faithful TP and RP, order-aware variants), training, statistics (speaker-cluster bootstrap), closure measurement |
| `scripts/`  | One script per step: build data, transcribe, extract features, run probes, baselines, analyses, figures                                                                                 |
| `slurm/`    | Ada (SLURM) jobs; how to run them: [`docs/ada.md`](docs/ada.md)                                                                                                                         |
| `tools/`    | Environment setup for the PRiSM stack, PC sync, job queues for the PC and the Mac                                                                                                       |
| `results/`  | Every number in the write-up: per-run metrics and predictions (`runs/`), tables, figures, DPR, MDL, gemination                                                                          |
| `docs/`     | Design decisions, plan, data notes, findings, reproduction notes, Ada guide                                                                                                             |
| `slides/`   | Evaluation decks, generated from `results/`                                                                                                                                             |
| `proposal/` | ACL-format proposal (LaTeX and PDF)                                                                                                                                                     |
| `tests/`    | Unit tests (`uv run pytest`)                                                                                                                                                            |
| `checkpoints/` | Released probe checkpoints (see Checkpoints below) |

Datasets, features and transcripts are not tracked (see `.gitignore`); the scripts rebuild them.

## Documents

| File                         | What it is                                                                                    |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| `docs/design.md`             | Every methodological decision and why, where we depart from PRiSM, pre-registered predictions |
| `docs/plan.md`               | Timeline, task split, risks                                                                   |
| `docs/findings.md`           | Results so far; numbers only from `results/`                                                  |
| `docs/prism_reproduction.md` | What PRiSM's code does (settings we copy) and the reproduction status                         |
| `docs/data.md`               | Datasets and models, with sources and licences                                                |
| `docs/compute.md`            | Hardware and storage estimates                                                                |
| `docs/ada.md`                | Step-by-step guide to every Ada job (Nidhi)                                                   |
| `docs/official_data.md`      | Audit of PRiSM's released GEO-v data and a corrected version                                  |
| `docs/benchmark.md`          | Cross-validated, speaker-disjoint, topic-controlled benchmark for the final evaluation        |

## Where things run

| Machine                          | Jobs                                                                                                                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mac (Apple silicon, `uv`)        | dataset rebuild, baselines, bag probes, MDL, control task, prosody features, forced alignment and closure measurement, all analyses, figures, slides |
| PC (RTX 4050, WSL, `envs/prism`) | phone transcription (W2V2, ZIPA), TP-GRU runs, district transcription                                                                                |
| Ada (SLURM, `envs/prism`)        | feature extraction, layer sweep, representation probes, TP-GRU seeds 44-46, topic and control checks                                                 |

## Setup

The project uses two Python environments.

**1. The main environment** (Mac or Linux, Python 3.10 to 3.12) runs data preparation, probe training, inference from
transcripts, statistics, tables and figures:

```sh
uv sync                                    # reads pyproject.toml and uv.lock
# or, without uv:
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

**2. The recognizer environment** (`envs/prism`, Linux with an NVIDIA GPU) runs the phone recognizers (W2V2, ZIPA)
with PRiSM's own model code, to transcribe audio and extract hidden states. PRiSM's code is not part of this
repository. The setup script clones it at the commit we used (`2e7b754`) into `external/prism`, creates a Python 3.10
environment with torch 2.6 (CUDA 12.6) and k2, and installs the exact package versions we ran with
(`tools/prism-requirements.lock`):

```sh
bash tools/setup_env.sh                    # ends with "environment OK"
envs/prism/bin/hf auth login               # Hugging Face token; the Vaani data is gated
```

The recognizer checkpoints are public and download on first use:
[`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) (W2V2) and
[`anyspeech/zipa-large-crctc-ns-800k`](https://huggingface.co/anyspeech/zipa-large-crctc-ns-800k) (ZIPA).

## Test set

The speaker-disjoint test split (3,581 utterances, 6.3 hours, 223 speakers, 26 pincodes), with audio, labels and both
recognizers' transcripts: [Google Drive folder](https://drive.google.com/drive/folders/16y49znRs_ojTTsIt91AJ0aLzDm8geLSi) (public). No test speaker is in the training data of the
released checkpoints. `scripts/export_testset.py` writes the same files from a rebuilt dataset; `SHA256SUMS` in the
folder lists their checksums.

## Checkpoints

`checkpoints/` holds the transcript probes our results rest on, trained on the speaker-disjoint split (seed 42).
Each file contains the weights, the token vocabulary, the bigram list (bag probes) and the training locations, so it
needs nothing else from training.

| Checkpoint | Recognizer | Probe | Test Recall@1 | Mean error |
| --- | --- | --- | --- | --- |
| `tp_zipa_rebuild_phone_bigram_speaker_split_s42.pt` | ZIPA | MLP over phone unigram and bigram frequencies | 21.9 | 417 km |
| `tp_w2v2_rebuild_phone_bigram_speaker_split_s42.pt` | W2V2 | MLP over phone unigram and bigram frequencies | 18.2 | 449 km |

## Inference

From transcripts (main environment, no GPU needed), using the test set files:

```sh
uv run python scripts/infer.py --checkpoint checkpoints/tp_zipa_rebuild_phone_bigram_speaker_split_s42.pt \
    --transcripts sal_testset/transcripts_zipa.parquet --labels sal_testset/testset.parquet \
    --out predictions.csv
```

This writes one row per utterance (predicted latitude and longitude, nearest training pincode) and, with `--labels`,
prints Recall@1 and mean error in km. On the test set it reproduces the table above.

From audio, first transcribe with the probe's recognizer (recognizer environment), then run the probe:

```sh
uv run python scripts/audio_to_parquet.py my_audio/ my_audio.parquet      # any folder of WAV/FLAC/OGG files
envs/prism/bin/python scripts/transcribe.py --model zipa --data my_audio.parquet --id-col audio_path \
    --out outputs/transcripts/zipa_my_audio.parquet
uv run python scripts/infer.py --checkpoint checkpoints/tp_zipa_rebuild_phone_bigram_speaker_split_s42.pt \
    --transcripts outputs/transcripts/zipa_my_audio.parquet --id-col audio_path --out predictions.csv
```

`--data sal_testset/testset.parquet` transcribes the test set the same way.

## Training

Build the data (main environment; needs the Vaani terms accepted on Hugging Face):

```sh
uv run python scripts/build_vaani_hi.py --workers 8     # our rebuild of Vaani-Hi; checksums in slurm/rebuild.sbatch
uv run python scripts/make_splits.py data/vaani_hi_rebuild/vaani_hi_rebuild.parquet   # PRiSM's and speaker-disjoint splits
uv run python scripts/make_benchmark.py                 # 5-fold speaker-group benchmark (docs/benchmark.md)
```

Transcribe and extract features (recognizer environment, GPU):

```sh
envs/prism/bin/python scripts/transcribe.py --model zipa --data data/vaani_hi_rebuild/vaani_hi_rebuild.parquet \
    --out outputs/transcripts/zipa_rebuild.parquet          # same with --model w2v2
envs/prism/bin/python scripts/extract_features.py --model w2v2 --data data/vaani_hi_rebuild/vaani_hi_rebuild.parquet \
    --out-dir outputs/features/w2v2_rebuild --frame-layers -1
uv run python scripts/speaker_embed.py                  # ECAPA voice embeddings
uv run python scripts/prosody_features.py --workers 6   # F0, voicing and intensity contours
uv run python scripts/channel_features.py               # recording-channel features
```

Train probes. `--protocol` is `prism` (PRiSM's split), `speaker_split` (new speakers) or a benchmark fold
(`all_f0` ... `all_f4`, `shared_f0` ... `shared_f4`); every run writes `results/runs/<name>/`:

```sh
uv run python scripts/run_tp.py --transcripts outputs/transcripts/zipa_rebuild.parquet --tokens phone \
    --probe bigram --protocol speaker_split --seeds 42 43 44 45 46     # --probe gru | shuffle | unigram | bigram
uv run python scripts/run_rp.py --features outputs/features/w2v2_rebuild --layers -1 --probe attn \
    --protocol speaker_split --seeds 42 43 44 --device cuda             # representation probes
uv run python scripts/run_dense.py --features outputs/features/ecapa_rebuild --protocol speaker_split --seeds 42 43 44 45 46
uv run python scripts/run_prosody.py --protocol speaker_split --seeds 42 43 44 45 46 --device cuda
uv run python scripts/baselines.py                      # majority, best constant, speaker and picture lookups
```

Add `--save-model checkpoints` to `run_tp.py` to write a checkpoint for `scripts/infer.py` (results/runs is then left
untouched). `slurm/` and `tools/` hold the batch jobs we ran on the Ada cluster and on a local GPU; `docs/ada.md`
describes them.

## Evaluation

Every table and statistic is computed from `results/runs` by scripts:

```sh
make analysis   # seed ensembles, paired comparisons with speaker-bootstrap intervals, benchmark scores, nulls, MDL, fusion
make final      # tables, figures, the report PDF, and a check of every number quoted in it
uv run pytest   # unit tests
```

Recall@1 and mean error follow PRiSM's definitions (`src/sal/probes/geo.py`).

## Build the proposal

```sh
make proposal   # latexmk, writes proposal/proposal.pdf
make plot       # regenerates proposal/figs/geov_gap.* from PRiSM Tables 4 and 7
```

## Dates

Proposal 9 Oct 2026. Mid evaluation 3 Nov 2026. End evaluation 4 Dec 2026.
