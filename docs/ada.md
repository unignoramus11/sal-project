# Ada guide (Nidhi)

This is the single reference for everything that runs on Ada. It lists every job we can run there now, in
order, with what each produces, how to check it, and roughly what it costs. Everything is one command per
step; the scripts submit the SLURM jobs themselves. You are welcome to do more than this list (ideas at the
end), but everything below is needed for the study.

Why Ada: the representation probes (RPs) need cached hidden-state frames (~15 GB per layer per model) and
~150-300 GPU-hours in total, which Mohit's laptop GPU (6 GB, power-capped) cannot do in time. The PC runs the
transcript probes; the Mac runs the analyses. Results from all three meet in this repo under `results/`.

## At a glance

| Step | What | Jobs | Time (wall) | Needed for |
|---|---|---|---|---|
| 0 | Accounts, storage | – | 15 min | everything |
| 1 | Code and environment | login node | 15-30 min | everything |
| 2 | Rebuild the dataset, checksum check | 1 CPU | 1-2 h | everything |
| 3 | Extract features (W2V2, Whisper, ZIPA) | 3 GPU | 1-2 h each | 4, 5, 6, 7 |
| 4 | Layer sweep | 3 GPU | < 1 h each | 7 |
| 5 | RP baselines (last layer, PRiSM's probe) | 6 GPU | 3-5 h each | **mid evaluation (3 Nov)** |
| 6 | Light jobs: transcription check, TP-GRU seeds 44-46, topic check, prosody | 2 GPU + 1 CPU, then 26 GPU | 1-6 h each | mid evaluation |
| 7 | RP sweep (order x layers), per model | 1 extract + 19 GPU per model | 3-5 h each | final (RQ1, the core result) |
| 8 | RP topic check and control task | 12 GPU | 2-5 h each | final (RQ0) |
| 9 | Send results back | – | minutes | after every step |
| 10 | Geminate probe by layer (when Mohit has pushed the token table) | 3 GPU | 2-6 h each | final (RQ2) |

Priority if the queue is slow: 2 → 3 → 5 and 6 → 4 → 7 (speaker-disjoint jobs first; the scripts already
submit those first) → 8. Steps 3 and 6 can run at the same time; 4, 5 and 8 need step 3.

## 0. Before starting

- Hugging Face: log in, open https://huggingface.co/datasets/ARTPARK-IISc/Vaani and accept the terms
  (approved automatically). Create a read token at https://huggingface.co/settings/tokens. Also request
  https://huggingface.co/datasets/changelinglab/vaani-hi-geo (manual approval; Mohit's request is still
  pending, so a second request helps; not needed to start).
- Storage: about **200 GB** in a directory that persists between jobs (not node-local `/scratch`): 12 GB raw
  Vaani shards, 4.6 GB dataset, ~15 GB per cached frame layer per model (last layer + 3 top layers for 3
  models), and a few GB of transcripts and prosody features.
- `uv` must be on your PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh` if `uv --version` fails).
- GitHub: you need push access to `unignoramus11/sal-project` (ask Mohit to add you as a collaborator). If
  that is not possible, send results as described in step 9.

## 1. Code and environment (once)

```bash
cd <your 200 GB directory>
git clone https://github.com/unignoramus11/sal-project.git && cd sal-project
bash tools/setup_env.sh              # needs internet; run on the login node or an interactive job
envs/prism/bin/hf auth login         # paste the Hugging Face token
cp slurm/ada.env.example slurm/ada.env && nano slurm/ada.env   # account, partition, GPU type, WORK dir
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

Each writes `outputs/features/<model>_rebuild/`: `pooled.npy` (every layer, time-averaged), `frames_L<l>.npy`
(the last layer's frames), `index.parquet`, `meta.json`. The log ends with a JSON summary; `n_utterances` must
be 22551. Models: W2V2 phone recognizer (wav2vec2-lv-60-espeak-cv-ft, 25 layers, 1024-d), Whisper-small
encoder (13 layers, 768-d), ZIPA-CR-CTC-NS large.

## 4. Layer sweep (3 GPU jobs, under 1 h each)

```bash
for M in w2v2 whisper zipa; do bash slurm/submit.sh slurm/sweep.sbatch $M; done
```

A small MLP per layer on the time-averaged vectors, scored on **validation only** (test is never used to choose
layers). Writes `results/layer_sweep/<model>_rebuild_{speaker_split,prism}.csv` and prints the ranking. Push
these (step 9). Step 7 reads them to pick the top 3 layers per model.

## 5. RP baselines for the mid evaluation (6 GPU jobs, 3-5 h each)

```bash
bash slurm/rp_baselines.sh
```

PRiSM's representation probe (RP-Attn: learned-query attention over L2-normalised frames, then an MLP) on the
last layer: 3 models x 2 split protocols x 5 seeds. Each run writes
`results/runs/rp_<model>_rebuild_L<l>_attn_<protocol>_s<seed>/` with `metrics.json`, `history.csv`,
`predictions.parquet`. This is the RP number the mid-evaluation slides compare with the transcript probe.

## 6. Light jobs (any time after step 2; 3 jobs, then 26 that wait for them)

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

## 7. RP sweep: order x layers (per model; after steps 3 and 4)

```bash
DRY=1 bash slurm/rp_sweep.sh w2v2     # prints what it would submit; check the layer list
bash slurm/rp_sweep.sh w2v2
bash slurm/rp_sweep.sh zipa
bash slurm/rp_sweep.sh whisper
```

The core experiment for RQ1: does the RP improve when it can see order, at the same parameter count, with the
layer choice held fixed? Per model the script
1. reads the top 3 layers by validation km (speaker-disjoint split) from the step-4 CSV,
2. submits one extract job that caches those layers' frames (skipped if cached),
3. submits 19 probe jobs that wait for the cache: probes `attn` (PRiSM), `attn_pe` (with positions), `gru`,
   `transformer` (order-aware, sized within ±10% of RP-Attn's parameters), `chunkshuffle` (frames permuted in
   200 ms chunks: local order only) x layers {last, learned weighted sum of the top 3} x 2 splits, 5 seeds per
   job. Last-layer `attn` is skipped because step 5 already ran it. 3-layer jobs request 100 GB RAM
   (`MEM_WEIGHTED=...` to change).

Total: 57 jobs, 285 runs, roughly 150-300 GPU-hours; the speaker-disjoint jobs are submitted first. Results in
`results/runs/rp_<model>_rebuild_L<layers>_<probe>_<protocol>_s<seed>/`.

## 8. RP topic check and control task (12 GPU jobs; after step 3)

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

## 10. Where the encoders represent consonant length (3 GPU jobs; when the token table is in the repo)

Start this once `results/gemination/layer_probe_tokens.parquet` exists after a `git pull` (Mohit pushes it when
the closure measurements on the Mac are done; you will see it mentioned in the commit log).

```bash
for M in w2v2 zipa whisper; do bash slurm/submit.sh slurm/geminate_layers.sbatch $M; done
```

For the accepted single voiceless stops between vowels (roughly 25,000; exact count in the token table) and
lexical geminates (roughly 2,600) (अच्छा, बच्चे, मिट्टी,
पत्थर, ...) from Haryana and New Delhi Vaani audio, each job averages every layer's frames over each token's
measured closure and burst, then trains a geminate-vs-single linear probe per layer (speaker-grouped
cross-validation). It reports the AUC by layer, the AUC within place of articulation, and whether Haryana single
stops look more geminate-like than New Delhi ones. The first job also fetches the district audio (~12 GB
streamed, ~3.5 GB kept); if compute nodes have no internet, run `python scripts/district_audio_files.py
--workers 4` on the login node first. Results: `results/gemination/geminate_layers_<model>.csv` and
`results/figures/fig_geminate_layers_<model>.*`.

## Troubleshooting

- `CUDA error: no kernel image is available`: the GPU is too old for the torch build; report the GPU model.
- `Access denied` / 401 / 403 from Hugging Face: the Vaani terms were not accepted for the logged-in account,
  or `hf auth login` was not run in this environment.
- Out of GPU memory in an RP job (11 GB cards, long utterances): add `--batch-size 24` to the `python` line in
  `slurm/rp.sbatch` (or `slurm/rp_check.sbatch`) and resubmit that job. Mention it when you push, because it
  changes a setting.
- Out of RAM (`oom-kill` in the log): resubmit with more memory, e.g. `MEM=120G bash slurm/submit.sh ...`,
  or remove `--in-memory` from the `python` line (slower, reads frames from disk).
- A job hit the time limit: resubmit it with fewer seeds (`"45 46"` instead of `"42 43 44 45 46"`); finished
  seeds are already saved in `results/runs/`.
- Re-running is safe: a finished run is just overwritten with the same configuration.

## Coming later (not ready yet)

- **Official Vaani-Hi reproduction**: if either access request is approved, rerun the TP-GRU and RP-Attn on
  the official data (scripts will accept `--dataset`); this settles the gap between PRiSM's 45.7 Recall@1 and
  our ~21 on the rebuilt data.

## If you want to do more

- Run the RP baselines and checks with **3-layer weighted** inputs too (`rp_check.sbatch` with `--layers` set to
  the top layers) for a fuller topic and control picture.
- Look at the RP attention weights on Haryana test utterances: which frames does the learned query attend to?
- Anything else that strengthens the study: write it down in `docs/` and push it, or tell Mohit.
