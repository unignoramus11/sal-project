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

Datasets, features, transcripts and checkpoints are not tracked (see `.gitignore`); the scripts rebuild them.

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

## Where things run

| Machine                          | Jobs                                                                                                                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mac (Apple silicon, `uv`)        | dataset rebuild, baselines, bag probes, MDL, control task, prosody features, forced alignment and closure measurement, all analyses, figures, slides |
| PC (RTX 4050, WSL, `envs/prism`) | phone transcription (W2V2, ZIPA), TP-GRU runs, district transcription                                                                                |
| Ada (SLURM, `envs/prism`)        | feature extraction, layer sweep, representation probes, TP-GRU seeds 44-46, topic and control checks                                                 |

## Reproduce

```sh
uv sync                                            # Mac/Linux, Python 3.10-3.12
uv run python scripts/build_vaani_hi.py --workers 8  # rebuilt Vaani-Hi (needs the Vaani terms accepted on HF)
uv run python scripts/make_splits.py data/vaani_hi_rebuild/vaani_hi_rebuild.parquet
uv run python scripts/baselines.py                 # majority, best constant, speaker lookup
bash tools/setup_env.sh                            # PRiSM inference stack (Linux + NVIDIA) for transcription
uv run python scripts/summarise.py                 # tables from results/runs (seed ensembles, CIs, paired tests)
uv run python scripts/make_figures.py
uv run --with python-pptx python slides/mid/build_deck.py --pdf
```

The dataset checksums are in `slurm/rebuild.sbatch`; a rebuild must match them bit for bit.

## Build the proposal

```sh
make proposal   # latexmk, writes proposal/proposal.pdf
make plot       # regenerates proposal/figs/geov_gap.* from PRiSM Tables 4 and 7
```

## Dates

Proposal 9 Oct 2026. Mid evaluation 3 Nov 2026. End evaluation 4 Dec 2026.
