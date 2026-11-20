# Ada guide (Nidhi)

This is the single reference for everything that runs on Ada. It lists every job we can run there now, in
order, with what each produces, how to check it, and roughly what it costs. Everything is one command per
step; the scripts submit the SLURM jobs themselves. You are welcome to do more than this list (ideas at the
end), but everything below is needed for the study.

Why Ada: the representation probes (RPs) need hidden-state frames (~15 GB per layer per model) and ~150-300
GPU-hours in total, which Mohit's laptop GPU (6 GB, power-capped) cannot do in time. From November every probe
runs on Ada, the transcript probes and the benchmark included, so all final numbers come from one cluster and one
environment; the Mac only runs the analyses (tables, figures, report). Results meet in this repo under `results/`.

**Storage.** `/home2` has a 50 GB quota and `/share1` is not mounted on compute nodes, so frames are never kept
between jobs. Every RP job extracts the layers it needs into the node's local scratch (`/ssd_scratch`, else
`/scratch`; ~2 TB per node, wiped after about 7 days), trains, writes its small results to `/home2` and deletes its
scratch folder (`slurm/frames.sh`). This adds 1-2 h of extraction to each RP job and needs no storage planning. The
job log prints `df -h` of both scratch folders and stops early if less than ~20 GB per layer is free.

## At a glance

**Now (final evaluation, 4 Dec).** The mid-evaluation steps (2, 3, 5, 6) were not run on Ada before 3 Nov; they
are folded into the order below. Step 6 is replaced by step 13.

| Step | What | Jobs | Time (wall) | Priority |
|---|---|---|---|---|
| 0, 1 | Accounts and environment | login node | 30 min | first |
| 2 | Rebuild the dataset, checksum check | 1 CPU | 1-2 h | first |
| 13 | Transcript, prosody and benchmark runs (moved from the Mac and the PC) | 18 CPU/GPU benchmark + 32 GPU | 1-10 h each | 1 (needs step 2 only) |
| 3 | Time-averaged features (W2V2, Whisper, ZIPA) | 3 GPU | 1-2 h each | 1 |
| 5 | RP baselines (last layer, PRiSM's probe) | 6 GPU | 5-7 h each | 2 (needs step 2) |
| 12 | RP-Attn on the cross-validated benchmark | 12 GPU | 4-10 h each | 2 |
| 4 | Layer sweep | 3 GPU | < 1 h each | 3 (needs step 3) |
| 7 | RP sweep (order x layers), W2V2 first | 19 GPU per model | 5-7 h each | 3 (RQ1, the core result) |
| 8 | RP topic check and control task | 12 GPU | 4-7 h each | 4 (RQ0) |
| 11 | PRiSM's release: download, RP-Attn as released and corrected | 6 GPU | 5-8 h each | 4 |
| 10 | Geminate probe by layer, all three models | 3 GPU | 2-6 h each | 5 (RQ2) |
| 9 | Send results back | – | minutes | after every batch |

Order: 0 → 1 → 2, then submit 13, 3 and 5 together; then 12 and 4; then 7 (W2V2, then ZIPA, then Whisper), 8, 11,
10. Every script submits the speaker-disjoint jobs first, and every job skips runs that already have results, so
resubmitting after a failure is always safe. If time runs short, step 7 for Whisper and step 10 are the ones to
drop.

## 0. Before starting

- Hugging Face: log in, open https://huggingface.co/datasets/ARTPARK-IISc/Vaani and accept the terms
  (approved automatically). Create a read token at https://huggingface.co/settings/tokens. For step 11 only
  (November), also request https://huggingface.co/datasets/changelinglab/vaani-hi-geo (manual approval).
- Storage: everything persistent fits in the 50 GB of `/home2`: the environment, the 4.6 GB dataset (delete the
  12 GB of raw shards in `data/vaani_hi_rebuild/raw/` once step 2's checksums match), the model checkpoints,
  time-averaged features (at most 2.3 GB per model), transcripts and results. Frames live only in scratch.
- `uv` must be on your PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh` if `uv --version` fails).
- GitHub: you need push access to `unignoramus11/sal-project` (ask Mohit to add you as a collaborator). If
  that is not possible, send results as described in step 9.

## 1. Code and environment (once)

```bash
cd ~                                 # /home2
git clone https://github.com/unignoramus11/sal-project.git && cd sal-project
bash tools/setup_env.sh              # needs internet; run on the login node or an interactive job
envs/prism/bin/hf auth login         # paste the Hugging Face token
cp slurm/ada.env.example slurm/ada.env && nano slurm/ada.env   # account, partition, GPU type
```

- `setup_env.sh` builds `envs/prism` (Python 3.10, torch 2.6, k2, PRiSM's code at a pinned commit) and ends
  with `environment OK` plus the torch, k2 and transformers versions. On the login node `cuda False` is
  expected; on a GPU node it must say `cuda True`.
- In `slurm/ada.env` keep the lines `CPUS=${CPUS:-10}` and `MEM=${MEM:-60G}` in that form: step 7 asks for
  more memory for some jobs and that only works if the file does not hard-code the value.
- Every later command is run from the repo root on the login node; `slurm/submit.sh` adds account,
  partition, GPU and memory flags from `slurm/ada.env`. Logs go to `outputs/slurm/<job>_<jobid>.out`.

Please report once (so the scripts fit the cluster):
1. `sinfo -o "%P %G %l"`: partitions, GPU types, time limits.
2. `nvidia-smi` on a GPU node: GPU model and driver (needs CUDA 12.6 support, driver 560 or newer).
3. Whether compute nodes reach huggingface.co (`curl -sI https://huggingface.co | head -1` inside a job).
4. Disk quota of the directory you use.

## 2. Rebuild the dataset and check it (1 CPU job, 1-2 h)

```bash
GPU=0 bash slurm/submit.sh slurm/rebuild.sbatch
tail -4 outputs/slurm/rebuild_<jobid>.out
```

Rebuilds Vaani-Hi from the public Vaani corpus with PRiSM's recipe (22,551 utterances, 26 pincodes) and our
speaker-disjoint split. The `dataset`/`expected` lines and the `splits`/`expected` lines must be identical: then
the dataset is bit-for-bit the one used on the PC and the Mac, which is our reproducibility check. If they
differ, stop and send the whole log. If compute nodes have no internet, run the two `python` lines of
`slurm/rebuild.sbatch` on the login node instead.

## 3. Extract features (3 GPU jobs, 1-2 h each)

```bash
for M in w2v2 whisper zipa; do bash slurm/submit.sh slurm/extract.sbatch $M; done
```

Each writes `outputs/features/<model>_rebuild/`: `pooled.npy` (every layer, time-averaged; at most 2.3 GB),
`index.parquet`, `meta.json`. No frames are written here: the RP jobs extract their own into scratch. The log ends with a JSON summary; `n_utterances` must
be 22551. Models: W2V2 phone recognizer (wav2vec2-lv-60-espeak-cv-ft, 25 layers, 1024-d), Whisper-small
encoder (13 layers, 768-d), ZIPA-CR-CTC-NS large.

## 4. Layer sweep (final, November; 3 GPU jobs, under 1 h each)

```bash
for M in w2v2 whisper zipa; do bash slurm/submit.sh slurm/sweep.sbatch $M; done
```

A small MLP per layer on the time-averaged vectors, scored on **validation only** (test is never used to choose
layers). Writes `results/layer_sweep/<model>_rebuild_{speaker_split,prism}.csv` and prints the ranking. Push
these (step 9). Step 7 reads them to pick the top 3 layers per model.

## 5. RP baselines (6 GPU jobs, 5-7 h each)

```bash
bash slurm/rp_baselines.sh
```

PRiSM's representation probe (RP-Attn: learned-query attention over L2-normalised frames, then an MLP) on the
last layer: 3 models x 2 split protocols x 5 seeds. Each run writes
`results/runs/rp_<model>_rebuild_L<l>_attn_<protocol>_s<seed>/` with `metrics.json`, `history.csv`,
`predictions.parquet`. This is the RP number compared with the transcript probe. Each job first extracts the last
layer's frames into scratch (1-2 h), then trains its 5 seeds.

## 6. Light jobs (replaced by step 13; only the transcription check is still useful)

```bash
bash slurm/light_jobs.sh
```

Submits:
- **2 transcription jobs** (W2V2 and ZIPA phone recognizers, greedy CTC with PRiSM's wrappers). Each log ends
  with a `transcripts` line and an `expected` line; `MATCH` means Ada reproduces the PC's transcripts exactly
  (a second reproducibility check). `DIFFERENT` is not fatal (GPU kernels can flip rare ties); send the log.
  Check with `tail -2 outputs/slurm/transcribe_*.out`.
- **1 CPU job**: prosody contours (F0, voicing, intensity per 10 ms, Praat via Parselmouth).
- When the transcripts exist: **12 TP-GRU jobs for seeds 44-46** of the main transcript-probe grid (2
  recognizers x {char GRU, phone GRU, shuffled phone GRU} x 2 splits; the PC runs seeds 42-43), and **12
  topic-check jobs** (the same probes trained and tested on shared-picture utterances only, 5 seeds).
- When the prosody contours exist: **2 prosody-probe jobs** (BiGRU on the contours, both splits, 5 seeds).

Results land in `results/runs/tp_*_s44..46/`, `results/runs/tp_*_generic_s*/`, `results/runs/prosody_gru_*/`.

## 7. RP sweep: order x layers (final, November; per model, after steps 3 and 4)

```bash
DRY=1 bash slurm/rp_sweep.sh w2v2     # prints what it would submit; check the layer list
bash slurm/rp_sweep.sh w2v2
bash slurm/rp_sweep.sh zipa
bash slurm/rp_sweep.sh whisper
```

The core experiment for RQ1: does the RP improve when it can see order, at the same parameter count, with the
layer choice held fixed? Per model the script
1. reads the top 3 layers by validation km (speaker-disjoint split) from the step-4 CSV,
2. submits 19 probe jobs, each of which extracts its layers' frames into scratch first: probes `attn` (PRiSM), `attn_pe` (with positions), `gru`,
   `transformer` (order-aware, sized within ±10% of RP-Attn's parameters), `chunkshuffle` (frames permuted in
   200 ms chunks: local order only) x layers {last, learned weighted sum of the top 3} x 2 splits, 5 seeds per
   job. Last-layer `attn` is skipped because step 5 already ran it. 3-layer jobs request 100 GB RAM
   (`MEM_WEIGHTED=...` to change).

Total: 57 jobs, 285 runs, roughly 150-300 GPU-hours; the speaker-disjoint jobs are submitted first. Results in
`results/runs/rp_<model>_rebuild_L<layers>_<probe>_<protocol>_s<seed>/`.

## 8. RP topic check and control task (final, November; 12 GPU jobs)

```bash
bash slurm/rp_extra.sh
```

The same two checks we ran for the transcript probes, so that TP and RP are compared on equal terms:
- **Topic check**: RP-Attn trained and tested on shared-picture utterances only (two thirds of Vaani pictures
  are named after the speaker's own district, so topic can give away the location).
- **Control task**: every speaker gets another speaker's location as the label. Under the random split a score
  above the control majority means the probe recognises speakers rather than dialects. On our data the
  speaker-embedding probe scores 33.2 here and the transcript bag probes stay at the control majority (14.0),
  so this measures how much of the RP's random-split score is speaker recognition.

## 9. Sending results back (after every step)

```bash
git pull --rebase
git add results/
git commit -m "Add <what> results"
git push
```

Only `results/` is versioned (CSV, JSON and small prediction parquet files). Data, features and transcripts
stay on Ada; they are gitignored and too large. Without push access: `tar czf ada_results_<date>.tgz results/`
and send the file to Mohit.

## 10. Where the encoders represent consonant length (final, November; 3 GPU jobs)

The token table `results/gemination/layer_probe_tokens.parquet` is in the repo (29,004 tokens).

```bash
for M in w2v2 zipa whisper; do bash slurm/submit.sh slurm/geminate_layers.sbatch $M; done
```

For the accepted single voiceless stops between vowels (roughly 25,000; exact count in the token table) and
lexical geminates (roughly 2,600: अच्छा, बच्चे, मिट्टी, पत्थर, ...) from Haryana and New Delhi Vaani audio, each job averages every layer's frames over each token's
measured closure and burst, then trains a geminate-vs-single linear probe per layer (speaker-grouped
cross-validation). It reports the AUC by layer, the AUC within place of articulation, and whether Haryana single
stops look more geminate-like than New Delhi ones. The first job also fetches the district audio (~12 GB
streamed, ~3.5 GB kept); if compute nodes have no internet, run `python scripts/district_audio_files.py
--workers 4` on the login node first. Results: `results/gemination/geminate_layers_<model>.csv` and
`results/figures/fig_geminate_layers_<model>.*`.

## 11. Official Vaani-Hi (final, November)

The official GEO-v release has problems that change how its scores should be read (docs/official_data.md):
duplicated audio with conflicting labels, 28% of test rows with their exact audio in training, and one pincode
placed in Afghanistan. We run PRiSM's probes on the release as published and on a corrected version.

```bash
# 1. access: request https://huggingface.co/datasets/changelinglab/vaani-hi-geo with your HF account (or ask Mohit)
envs/prism/bin/hf download changelinglab/vaani-hi-geo --repo-type dataset \
  --revision 264fb053bd1e8f060cd09b9677950d5be96fe77a --local-dir data/vaani_hi_official
python scripts/official_merge.py                     # -> data/vaani_hi_official/vaani_hi_official.parquet
rm -rf data/vaani_hi_official/data                    # the raw download, once the merged parquet exists
# 2. RP-Attn, last layer, 5 seeds: the release as published, and the corrected version (deduplicated, consistent
#    labels, fixed coordinates) under the released split and a pseudo-speaker-disjoint split (6 GPU jobs; each
#    extracts the release's frames into scratch first)
git pull                                              # brings results/official/{meta,splits}_{released,clean}.parquet
bash slurm/rp_official.sh
```

The split tables are in the repo. Results land in `results/runs/rp_<model>_official_*`.

## 12. RP-Attn on the cross-validated benchmark (final, November; 30 GPU jobs, 2-3 h each)

The benchmark (`docs/benchmark.md`, tables in `results/benchmark/`) tests every speaker exactly once: 5 folds over
speaker groups (look-alike speaker codes merged), a track with all utterances and one with shared pictures only,
and corrected pincode coordinates. The transcript probes run in step 13; this step runs PRiSM's
representation probe on the same folds so the two are compared on identical test speakers.

```bash
git pull                                              # brings results/benchmark/{meta,splits}.parquet
bash slurm/rp_benchmark.sh                            # 3 models x {all, shared} x 5 folds, seeds 42-44; 12 jobs
```

Each job extracts the last layer's frames into scratch once and runs two or three folds. Results land in `results/runs/rp_<model>_rebuild_L-1_attn_<track>_f<fold>_bench_s<seed>/`;
`scripts/benchmark_eval.py` (Mac) pools the folds.

## 13. Transcript, prosody and benchmark runs (moved from the Mac and the PC)

Everything that used to wait for the Mac and the PC, so that all final probes run in one place. Needs step 2 (the
dataset, for coordinates) and four files from Mohit's bundle, copied into the repo:
`outputs/transcripts/{w2v2,zipa}_rebuild.parquet` (you have these), `outputs/transcripts/{w2v2,zipa}_official.parquet`,
`outputs/features/ecapa_rebuild/` and `outputs/features/prosody_rebuild/` (you have this one).

```bash
git pull
bash slurm/november.sh
```

Submits 50 jobs:
- **Benchmark, CPU** (10 jobs, `slurm/bench_light.sbatch`): per track and fold, the 8 bag probes, the ECAPA voice
  embedding and pooled prosody statistics, seeds 42-44, run in parallel on 16 CPUs (`BENCH_CPUS=...` to change).
- **Benchmark, GPU** (8 jobs, `slurm/bench_gru.sbatch`): PRiSM's character BiGRU, the phone BiGRU and the shuffled
  phone BiGRU for W2V2 and ZIPA, seeds 42-44.
- **Seeds** (24 jobs, `slurm/tp.sbatch`): the rebuild's BiGRUs up to 5 seeds under both splits, and the same probes
  on shared pictures only (the topic check, which so far exists only for the bag probes).
- **Prosody** (2 jobs): the prosody BiGRU up to 5 seeds under both splits.
- **PRiSM's release** (2 jobs, `slurm/tp_official.sbatch`): the three BiGRUs for W2V2 and ZIPA on the release as
  published and corrected, up to 5 seeds (ZIPA's BiGRUs on the release have never been run).
- **Description length** (4 jobs, `slurm/mdl.sbatch`): the ZIPA phone BiGRU and shuffled BiGRU, both splits.

Every job skips runs that already have results (the repo has 1,177 run files), so only what is missing runs.
Results land in `results/runs/` and `results/mdl/`; push them (step 9). Mohit pools them on the Mac
(`scripts/benchmark_eval.py`, `scripts/summarise.py`, `scripts/official_eval.py`, `scripts/fuse.py`).

## Troubleshooting

- `CUDA error: no kernel image is available`: the GPU is too old for the torch build; report the GPU model.
- `Access denied` / 401 / 403 from Hugging Face: the Vaani terms were not accepted for the logged-in account,
  or `hf auth login` was not run in this environment.
- Out of GPU memory in an RP job (11 GB cards, long utterances): resubmit that job with
  `RP_ARGS="--batch-size 24" bash slurm/submit.sh ...` (all RP jobs read `RP_ARGS`). Mention it when you push,
  because it changes a setting.
- Out of RAM (`oom-kill` in the log): resubmit with more memory, e.g. `MEM=120G bash slurm/submit.sh ...`.
- `no writable scratch` or `only N GB free`: the node's scratch is full or missing; resubmit (another node), or set
  `SCRATCH_ROOTS="/scratch"` to skip `/ssd_scratch`.
- A job hit the time limit: resubmit it with fewer seeds (`"45 46"` instead of `"42 43 44 45 46"`); finished
  seeds are already saved in `results/runs/`.
- Re-running is safe: transcript, prosody, MDL and benchmark jobs skip finished runs; an RP job that is resubmitted
  reruns its seeds with the same configuration.

## If you want to do more

- Run the RP baselines and checks with **3-layer weighted** inputs too (`rp_check.sbatch` with `--layers` set to
  the top layers) for a fuller topic and control picture.
- Look at the RP attention weights on Haryana test utterances: which frames does the learned query attend to?
- Anything else that strengthens the study: write it down in `docs/` and push it, or tell Mohit.
